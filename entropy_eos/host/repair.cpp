#include "entropy_eos/host/repair.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "entropy_eos/core/bspline_eval.hpp"
#include "entropy_eos/host/bspline_fit.hpp"
#include "entropy_eos/host/units.hpp"

namespace eeos {

void repair_column(std::vector<double> &col, double min_slope) {
  const size_t n = col.size();
  if (n == 0) {
    return;
  }

  // L2 isotonic regression via the stack-based pool-adjacent-violators
  // algorithm: pools carry (mean, count). Appending a new element that would
  // create a decrease merges it into the trailing pool(s) until pool means
  // are non-decreasing again ("out of order" is a strict >, so pools already
  // equal are left separate -- harmless, since expansion gives the same
  // values either way, and it keeps a non-decreasing input an exact no-op).
  std::vector<double> pool_mean;
  std::vector<size_t> pool_count;
  pool_mean.reserve(n);
  pool_count.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    double mean = col[i];
    size_t count = 1;
    while (!pool_mean.empty() && pool_mean.back() > mean) {
      const size_t merged_count = pool_count.back() + count;
      mean = (pool_mean.back() * static_cast<double>(pool_count.back()) +
              mean * static_cast<double>(count)) /
             static_cast<double>(merged_count);
      count = merged_count;
      pool_mean.pop_back();
      pool_count.pop_back();
    }
    pool_mean.push_back(mean);
    pool_count.push_back(count);
  }

  size_t idx = 0;
  for (size_t p = 0; p < pool_mean.size(); ++p) {
    for (size_t r = 0; r < pool_count[p]; ++r) {
      col[idx++] = pool_mean[p];
    }
  }

  // Strictification forward pass. Written as an explicit comparison (not
  // col[j] = std::max(col[j], lower)) so a column that already satisfies the
  // minimum slope is a true no-op: no arithmetic touches col[j], so its bits
  // cannot drift even though the "equivalent" computed value would compare
  // equal.
  for (size_t j = 1; j < n; ++j) {
    const double lower = col[j - 1] + min_slope;
    if (col[j] < lower) {
      col[j] = lower;
    }
  }
}

namespace {

double min_slope_for(const std::string &field, const RepairOptions &options) {
  if (field == "entropy") {
    return options.min_slope_entropy;
  }
  if (field == "logenergy") {
    return options.min_slope_logenergy;
  }
  throw std::invalid_argument("repair_table: no known minimum slope for field '" + field +
                               "' (only \"entropy\" and \"logenergy\" are supported)");
}

// --- M2c-prime spline-safe smoothing loop (repair.hpp step 1) --------------

// Fits `col` at unit spacing (x0=0, h=1) and samples S' at every audit point
// (repair.hpp's repair_table() doc comment, step 1b): cell j = 0..n-2 at j +
// m/refine for m = 0..refine-1, plus the column's last node. Sets
// offending[j] whenever any sample inside cell j has S' <= floor; returns
// true iff at least one cell was marked.
bool spline_audit_column(const std::vector<double> &col, int refine, double slope_floor,
                          std::vector<char> &offending) {
  const int n = static_cast<int>(col.size());
  offending.assign(static_cast<size_t>(n > 0 ? n - 1 : 0), 0);
  if (n < 2) {
    return false;
  }

  const std::vector<double> coeffs = fit_bspline_1d(col);
  const BsplineView1 view{coeffs.data(), n, 0.0, 1.0};

  bool any = false;
  const int refine_clamped = std::max(refine, 1);
  for (int j = 0; j + 1 < n; ++j) {
    for (int m = 0; m < refine_clamped; ++m) {
      const double x =
          static_cast<double>(j) + static_cast<double>(m) / static_cast<double>(refine_clamped);
      const BsplineEval1 e = bspline_eval1(view, x);
      if (e.fx <= slope_floor) {
        offending[static_cast<size_t>(j)] = 1;
        any = true;
      }
    }
  }
  // The last node (x = n-1) is never hit by the j+m/refine sweep above (its
  // m only reaches (refine-1)/refine within the last cell) -- sample it
  // explicitly, attributed to the last cell.
  {
    const BsplineEval1 e = bspline_eval1(view, static_cast<double>(n - 1));
    if (e.fx <= slope_floor) {
      offending[static_cast<size_t>(n - 2)] = 1;
      any = true;
    }
  }
  return any;
}

// One Jacobi diffusion step (repair.hpp step 1d), applied in place to every
// *interior* index marked in `marked` (column endpoints never move, even if
// marked): v_j <- v_j + alpha*(v_{j-1} - 2*v_j + v_{j+1}), using the
// pre-step snapshot on the right-hand side so the result does not depend on
// the order marked indices are visited in.
void jacobi_diffuse_step(std::vector<double> &col, const std::vector<char> &marked, double alpha) {
  const size_t n = col.size();
  if (n < 3) {
    return;
  }
  const std::vector<double> before = col;
  for (size_t j = 1; j + 1 < n; ++j) {
    if (!marked[j]) {
      continue;
    }
    col[j] = before[j] + alpha * (before[j - 1] - 2.0 * before[j] + before[j + 1]);
  }
}

// Per-column outcome of the spline-safe loop, folded into
// RepairResult::FieldSummary by repair_table().
struct SplineSafeStats {
  int rounds_used = 0;        // smoothing rounds actually run on this column
  bool needed_smoothing = false;
  bool still_violating = false; // cap reached with an audit violation remaining
};

// Runs the spline-safe smoothing loop on `col` in place (repair.hpp's
// repair_table() doc comment, step 1); `col` must already have passed
// through repair_column(col, min_slope) once (the base pass, step 0). No-op
// (returns default-constructed stats) if fit_bspline_1d() cannot fit the
// column (n < 4) -- too short to safely audit/smooth.
SplineSafeStats spline_safe_column(std::vector<double> &col, double min_slope,
                                    const RepairOptions &options) {
  SplineSafeStats stats;
  const int n = static_cast<int>(col.size());
  if (n < 4) {
    return stats;
  }

  // spline_rounds_max+1 audits bracket spline_rounds_max diffusion rounds,
  // so the *last* diffusion round's effect is itself audited before giving
  // up -- otherwise "still violating" could be wrong for a column that
  // happens to become clean on exactly the last permitted round.
  std::vector<char> offending;
  for (int round = 0; round <= options.spline_rounds_max; ++round) {
    const bool any = spline_audit_column(col, options.spline_refine, options.spline_slope_floor, offending);
    if (!any) {
      return stats; // clean: done (rounds_used/needed_smoothing reflect prior rounds, if any)
    }
    if (round == options.spline_rounds_max) {
      break; // cap reached; the loop above already confirmed a violation remains
    }

    stats.needed_smoothing = true;
    stats.rounds_used = round + 1;

    std::vector<char> marked(static_cast<size_t>(n), 0);
    for (int j = 0; j + 1 < n; ++j) {
      if (!offending[static_cast<size_t>(j)]) {
        continue;
      }
      const int lo = std::max(j - options.diffuse_window, 0);
      const int hi = std::min(j + 1 + options.diffuse_window, n - 1);
      for (int i = lo; i <= hi; ++i) {
        marked[static_cast<size_t>(i)] = 1;
      }
    }
    marked[0] = 0;                              // column endpoints never move
    marked[static_cast<size_t>(n - 1)] = 0;

    jacobi_diffuse_step(col, marked, options.diffuse_alpha);
    repair_column(col, min_slope); // restores monotonicity the diffusion may have nudged
  }

  stats.still_violating = true;
  return stats;
}

// --- M2d-1 spline-safe-3d smoothing loop (repair.hpp step 2-3) -------------

// Per-axis refined sample positions for the 3D audit (repair.hpp's
// repair_table() doc comment, step 2b): the union of every data node and
// (refine-1) points interior to each cell, i.e. i/refine for i =
// 0..(n-1)*refine -- (n-1)*refine+1 positions, unit-spacing-aligned (every
// node sits exactly at a multiple of refine). Used for all three axes
// (refine3d_u for u, refine3d_xy for x and y).
std::vector<double> refined_axis_positions(int n, int refine) {
  const int refine_c = std::max(refine, 1);
  const int count = (n - 1) * refine_c + 1;
  std::vector<double> pos(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    pos[static_cast<size_t>(i)] = static_cast<double>(i) / static_cast<double>(refine_c);
  }
  return pos;
}

// Marks the data-index box [i-w,i+1+w] x [j-w,j+1+w] x [k-w,k+1+w] (clamped
// to the grid, repair.hpp step 2d) in `marked` (sized nrho*ntemp*nye,
// RawTable::index() layout: irho fastest, then jT, then kYe).
void mark_box3d(std::vector<char> &marked, int nrho, int ntemp, int nye, int i, int j, int k, int w) {
  const int ilo = std::max(i - w, 0), ihi = std::min(i + 1 + w, nrho - 1);
  const int jlo = std::max(j - w, 0), jhi = std::min(j + 1 + w, ntemp - 1);
  const int klo = std::max(k - w, 0), khi = std::min(k + 1 + w, nye - 1);
  for (int kk = klo; kk <= khi; ++kk) {
    for (int jj = jlo; jj <= jhi; ++jj) {
      const int base = nrho * (jj + ntemp * kk);
      for (int ii = ilo; ii <= ihi; ++ii) {
        marked[static_cast<size_t>(base + ii)] = 1;
      }
    }
  }
}

// Fits `data` (a whole field, RawTable layout) as a tensor-product B-spline
// at unit spacing and audits its u-derivative on the 3D grid described by
// repair.hpp's repair_table() doc comment (step 2b), at (refine_xy,
// refine_u, refine_xy) resolution. Every violating sample (fu <=
// slope_floor) marks the data-index box around its owning cell (`window` on
// each side, clamped -- step 2d) into `marked` (must already be sized
// nrho*ntemp*nye; only ever OR'd into, never cleared, by this function) and
// is counted; returns the violation *sample* count.
//
// Performance: this is the hot path the M2d-1 work order's guardrail is
// about (~1e8 samples per main-loop round on an SRO-sized table, ~5e8 for a
// (4,4,4) verification pass), so it deliberately does *not* call
// core::bspline_eval3() (which also computes fx/fy/fxx/fxu/fuu this audit
// never needs) and hoists the u/y axes' cell/basis computation out of the
// innermost (rho) loop, which OpenMP parallelizes over -- each rho sample
// computes its own cell/basis once and then reuses every precomputed u/y
// pair, an O(count_u+count_y) cost instead of O(count_x*count_u*count_y).
// The returned count and the marked set are both independent of thread
// count/scheduling (a plain sum over independent samples and an idempotent
// OR, respectively), so the `#pragma omp critical` below -- entered only on
// an actual violation -- does not introduce any nondeterminism.
size_t audit3d_and_mark(int nrho, int ntemp, int nye, const std::vector<double> &data, int refine_xy,
                         int refine_u, double slope_floor, int window, std::vector<char> &marked) {
  const Bspline3 fit = fit_bspline_3d(nrho, ntemp, nye, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, data);
  const BsplineView3 view = fit.view();

  const std::vector<double> xs = refined_axis_positions(nrho, refine_xy);
  const std::vector<double> us = refined_axis_positions(ntemp, refine_u);
  const std::vector<double> ys = refined_axis_positions(nye, refine_xy);
  const int count_x = static_cast<int>(xs.size());
  const int count_u = static_cast<int>(us.size());
  const int count_y = static_cast<int>(ys.size());

  const int refine_xy_c = std::max(refine_xy, 1);
  const int refine_u_c = std::max(refine_u, 1);

  // Precompute the u and y axes' cell index / basis values once each (reused
  // by every rho sample below).
  std::vector<int> cu_i(static_cast<size_t>(count_u));
  std::vector<std::array<double, 4>> cu_du(static_cast<size_t>(count_u));
  for (int iu = 0; iu < count_u; ++iu) {
    const detail::BsplineCell c = detail::bspline_cell(us[static_cast<size_t>(iu)], view.u0, view.hu, view.nu);
    const detail::Basis4 d = detail::bspline_dbasis(c.t);
    cu_i[static_cast<size_t>(iu)] = c.i;
    cu_du[static_cast<size_t>(iu)] = {d.b0, d.b1, d.b2, d.b3};
  }
  std::vector<int> cy_i(static_cast<size_t>(count_y));
  std::vector<std::array<double, 4>> cy_b(static_cast<size_t>(count_y));
  for (int iy = 0; iy < count_y; ++iy) {
    const detail::BsplineCell c = detail::bspline_cell(ys[static_cast<size_t>(iy)], view.y0, view.hy, view.ny);
    const detail::Basis4 b = detail::bspline_basis(c.t);
    cy_i[static_cast<size_t>(iy)] = c.i;
    cy_b[static_cast<size_t>(iy)] = {b.b0, b.b1, b.b2, b.b3};
  }

  const int nxp2 = view.nx + 2;
  const int nup2 = view.nu + 2;

  size_t violations = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : violations) schedule(static)
#endif
  for (int ix = 0; ix < count_x; ++ix) {
    const detail::BsplineCell cx = detail::bspline_cell(xs[static_cast<size_t>(ix)], view.x0, view.hx, view.nx);
    const detail::Basis4 Bx = detail::bspline_basis(cx.t);
    const double bx[4] = {Bx.b0, Bx.b1, Bx.b2, Bx.b3};
    const int i = std::min(ix / refine_xy_c, nrho - 2);

    for (int iu = 0; iu < count_u; ++iu) {
      const int j = std::min(iu / refine_u_c, ntemp - 2);
      const int iuc = cu_i[static_cast<size_t>(iu)];
      const std::array<double, 4> &du = cu_du[static_cast<size_t>(iu)];

      for (int iy = 0; iy < count_y; ++iy) {
        const int iyc = cy_i[static_cast<size_t>(iy)];
        const std::array<double, 4> &by = cy_b[static_cast<size_t>(iy)];

        double fu = 0.0;
        for (int r = 0; r < 4; ++r) {
          const int iyy = iyc + r;
          for (int q = 0; q < 4; ++q) {
            const int iuu = iuc + q;
            const int base = nxp2 * (iuu + nup2 * iyy);
            double sf = 0.0;
            for (int p = 0; p < 4; ++p) {
              sf += bx[p] * view.c[base + cx.i + p];
            }
            fu += du[static_cast<size_t>(q)] * by[static_cast<size_t>(r)] * sf;
          }
        }
        fu /= view.hu;

        if (fu <= slope_floor) {
          ++violations;
          const int k = std::min(iy / refine_xy_c, nye - 2);
#ifdef _OPENMP
#pragma omp critical
#endif
          { mark_box3d(marked, nrho, ntemp, nye, i, j, k, window); }
        }
      }
    }
  }
  return violations;
}

// One 3D Jacobi diffusion step (repair.hpp's repair_table() doc comment,
// step 2d), applied in place to every point of `data` marked in `marked`
// that is interior in *all three* axes (points on any axis's boundary never
// move, even if marked): v_ijk <- v_ijk + alpha*(sum of the 6 axis-neighbors
// - 6*v_ijk)/6, using the pre-step snapshot on the right-hand side (so the
// result does not depend on the order marked points are visited in; the
// loop below is serial, but is written so it would give the same answer
// under any parallelization). Every point actually modified is marked in
// `ever_diffused` (sized like `data`; only ever set, never cleared, across a
// field's whole 3D stage) and, the first time, counted into
// `newly_diffused`.
void jacobi_diffuse3d_step(std::vector<double> &data, int nrho, int ntemp, int nye,
                            const std::vector<char> &marked, double alpha, std::vector<char> &ever_diffused,
                            size_t &newly_diffused) {
  if (nrho < 3 || ntemp < 3 || nye < 3) {
    return; // no point can be interior in all three axes
  }
  const std::vector<double> before = data;
  const int stride_k = nrho * ntemp;
  for (int kYe = 1; kYe + 1 < nye; ++kYe) {
    for (int jT = 1; jT + 1 < ntemp; ++jT) {
      const int row_base = nrho * (jT + ntemp * kYe);
      for (int irho = 1; irho + 1 < nrho; ++irho) {
        const int idx = row_base + irho;
        if (!marked[static_cast<size_t>(idx)]) {
          continue;
        }
        const double v = before[static_cast<size_t>(idx)];
        const double sum6 = before[static_cast<size_t>(idx - 1)] + before[static_cast<size_t>(idx + 1)] +
                             before[static_cast<size_t>(idx - nrho)] + before[static_cast<size_t>(idx + nrho)] +
                             before[static_cast<size_t>(idx - stride_k)] +
                             before[static_cast<size_t>(idx + stride_k)];
        data[static_cast<size_t>(idx)] = v + alpha * (sum6 - 6.0 * v) / 6.0;
        if (!ever_diffused[static_cast<size_t>(idx)]) {
          ever_diffused[static_cast<size_t>(idx)] = 1;
          ++newly_diffused;
        }
      }
    }
  }
}

// Re-runs the per-column pipeline (repair_column(), then -- if
// options.spline_safe -- spline_safe_column(), repair.hpp's repair_table()
// doc comment step 2d) on every (irho, kYe) column that intersects `marked`
// (has any jT marked), in place on `data`. OpenMP over the affected columns
// (each touches only its own data, so this is deterministic regardless of
// scheduling).
void rerepair_affected_columns(std::vector<double> &data, int nrho, int ntemp, int nye,
                                const std::vector<char> &marked, double min_slope,
                                const RepairOptions &options) {
  std::vector<char> affected(static_cast<size_t>(nrho) * static_cast<size_t>(nye), 0);
  for (int kYe = 0; kYe < nye; ++kYe) {
    for (int jT = 0; jT < ntemp; ++jT) {
      const int base = nrho * (jT + ntemp * kYe);
      for (int irho = 0; irho < nrho; ++irho) {
        if (marked[static_cast<size_t>(base + irho)]) {
          affected[static_cast<size_t>(irho + nrho * kYe)] = 1;
        }
      }
    }
  }

  std::vector<std::pair<int, int>> cols; // (irho, kYe), in a fixed (kYe, irho) order
  cols.reserve(affected.size());
  for (int kYe = 0; kYe < nye; ++kYe) {
    for (int irho = 0; irho < nrho; ++irho) {
      if (affected[static_cast<size_t>(irho + nrho * kYe)]) {
        cols.emplace_back(irho, kYe);
      }
    }
  }

  const long long ncols = static_cast<long long>(cols.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long long ci = 0; ci < ncols; ++ci) {
    const int irho = cols[static_cast<size_t>(ci)].first;
    const int kYe = cols[static_cast<size_t>(ci)].second;

    std::vector<double> col(static_cast<size_t>(ntemp));
    for (int jT = 0; jT < ntemp; ++jT) {
      col[static_cast<size_t>(jT)] = data[static_cast<size_t>(irho + nrho * (jT + ntemp * kYe))];
    }

    repair_column(col, min_slope);
    if (options.spline_safe) {
      spline_safe_column(col, min_slope, options);
    }

    for (int jT = 0; jT < ntemp; ++jT) {
      data[static_cast<size_t>(irho + nrho * (jT + ntemp * kYe))] = col[static_cast<size_t>(jT)];
    }
  }
}

// Per-field outcome of the 3D spline-safe stage, folded into
// RepairResult::FieldSummary by repair_table().
struct ThreeDStats {
  int rounds_used = 0;
  size_t points_diffused = 0;
  size_t violations_remaining = 0;
  std::vector<size_t> violation_history;
};

// Runs the M2d-1 tensor-product 3D spline-safe stage on `data` (a whole
// field, RawTable layout) in place -- repair.hpp's repair_table() doc
// comment, steps 2-3. `data` must already have passed through the
// per-column stage (steps 0-1) for every column. No-op (default-constructed
// stats) if any axis has fewer than 4 points -- fit_bspline_3d()'s own
// minimum, matching the per-column stage's n < 4 no-op.
//
// Empirical finding driving the "best-tracking" logic below (repair.hpp's
// doc comment references this): plain repeated audit-diffuse-rerepair is
// *not* always monotonically improving. Two concrete cases found while
// implementing this stage: (a) a field with a wide, physically near-flat
// region (e.g. "logenergy" at low T, where eps << energy_shift makes L_u
// naturally tiny almost everywhere -- exactly SRO's real "near-flat
// plateau" pathology CODE.md already describes) has essentially no safety
// margin, so a diffusion step aimed at one violation can tip a
// barely-positive *neighboring* sample negative, and since the next round
// marks and diffuses that too, the violation count can run away outward
// instead of shrinking; (b) conversely, some genuinely-converging
// trajectories (e.g. a sharp block-edge defect) dip *up* for a few rounds
// before trending back down, so bailing out on the very first uptick would
// leave easy wins on the table. Both are handled by tracking the
// best (lowest-violation) state seen so far and only giving up after
// several consecutive rounds fail to beat it (kPatience), then reverting to
// that best state -- and, as a final backstop, never returning a field
// worse off (at the authoritative (4,4,4) resolution) than doing nothing at
// all.
ThreeDStats spline_safe_3d_field(std::vector<double> &data, size_t nrho_sz, size_t ntemp_sz, size_t nye_sz,
                                  double min_slope, const RepairOptions &options) {
  ThreeDStats stats;
  const int nrho = static_cast<int>(nrho_sz);
  const int ntemp = static_cast<int>(ntemp_sz);
  const int nye = static_cast<int>(nye_sz);
  if (nrho < 4 || ntemp < 4 || nye < 4) {
    return stats;
  }

  // The state this whole stage started from -- the final backstop (below)
  // never leaves a field worse off, at (4,4,4), than this.
  const std::vector<double> original = data;

  std::vector<char> ever_diffused(data.size(), 0);
  std::vector<char> marked(data.size(), 0);

  // audit_pass: re-fits and re-audits `data` at (refine_xy, refine_u,
  // refine_xy), overwriting `marked` with the freshly found violating boxes
  // (cleared first, so a caller always sees exactly this pass's marks).
  auto audit_pass = [&](int refine_xy, int refine_u) -> size_t {
    std::fill(marked.begin(), marked.end(), 0);
    return audit3d_and_mark(nrho, ntemp, nye, data, refine_xy, refine_u, options.spline_slope_floor,
                             options.diffuse_window, marked);
  };
  // apply_fix: diffuses the points `marked` last flagged, then re-repairs
  // every column that intersects them (repair.hpp step 2d).
  auto apply_fix = [&]() {
    size_t newly = 0;
    jacobi_diffuse3d_step(data, nrho, ntemp, nye, marked, options.diffuse_alpha, ever_diffused, newly);
    stats.points_diffused += newly;
    rerepair_affected_columns(data, nrho, ntemp, nye, marked, min_slope, options);
  };

  // Step 2: main loop at the (cheaper) loop-audit resolution, tracking the
  // best (lowest-violation) state seen and giving up early -- reverting to
  // that best state -- after kPatience consecutive rounds fail to beat it.
  constexpr int kPatience = 4;
  size_t best_violations = std::numeric_limits<size_t>::max(); // sentinel: "not measured yet"
  std::vector<double> best_data;
  std::vector<char> best_ever_diffused;
  size_t best_points_diffused = 0;
  int best_rounds_used = 0;
  int patience = 0;
  int rounds_applied = 0;

  for (int round = 0; round < options.rounds3d_max; ++round) {
    const size_t violations = audit_pass(options.refine3d_xy, options.refine3d_u);
    stats.violation_history.push_back(violations);
    if (violations == 0) {
      best_violations = 0;
      best_data = data;
      best_ever_diffused = ever_diffused;
      best_points_diffused = stats.points_diffused;
      best_rounds_used = rounds_applied;
      break;
    }
    if (best_violations == std::numeric_limits<size_t>::max() || violations < best_violations) {
      best_violations = violations;
      best_data = data;
      best_ever_diffused = ever_diffused;
      best_points_diffused = stats.points_diffused;
      best_rounds_used = rounds_applied;
      patience = 0;
    } else if (++patience >= kPatience) {
      break; // stuck (or worsening): stop trying, revert to the best state below
    }
    apply_fix();
    ++rounds_applied;
  }
  if (best_violations != std::numeric_limits<size_t>::max()) { // at least one round ran
    data = best_data;
    ever_diffused = std::move(best_ever_diffused);
    stats.points_diffused = best_points_diffused;
    stats.rounds_used = best_rounds_used;
  }

  // Step 3: final verification, always at (4,4,4) regardless of the main
  // loop's refine settings, plus up to 3 extra fix rounds if it finds
  // anything -- again tracking the best state and stopping (reverting) the
  // instant a round fails to improve on it, since there is no round budget
  // left to recover from a bad one here.
  size_t violations = audit_pass(4, 4);
  stats.violation_history.push_back(violations);
  size_t best_verify_violations = violations;
  std::vector<double> best_verify_data = data;
  std::vector<char> best_verify_ever_diffused = ever_diffused;
  size_t best_verify_points_diffused = stats.points_diffused;
  int best_verify_rounds_used = stats.rounds_used;
  for (int extra = 0; extra < 3 && violations > 0; ++extra) {
    apply_fix();
    violations = audit_pass(4, 4);
    stats.violation_history.push_back(violations);
    if (violations < best_verify_violations) {
      best_verify_violations = violations;
      best_verify_data = data;
      best_verify_ever_diffused = ever_diffused;
      best_verify_points_diffused = stats.points_diffused;
      best_verify_rounds_used = stats.rounds_used + extra + 1;
    } else {
      break;
    }
  }
  data = std::move(best_verify_data);
  ever_diffused = std::move(best_verify_ever_diffused);
  stats.points_diffused = best_verify_points_diffused;
  stats.rounds_used = best_verify_rounds_used;
  size_t final_violations = best_verify_violations;

  // Final backstop: never report (or leave in `data`) a worse (4,4,4)
  // outcome than skipping this stage entirely. Only costs an extra (4,4,4)
  // audit of `original` when there is still something to check against
  // (final_violations == 0 is already optimal).
  if (final_violations > 0) {
    std::vector<char> baseline_marked(data.size(), 0);
    const size_t baseline_violations = audit3d_and_mark(
        nrho, ntemp, nye, original, 4, 4, options.spline_slope_floor, options.diffuse_window, baseline_marked);
    if (final_violations > baseline_violations) {
      data = original;
      final_violations = baseline_violations;
      stats.points_diffused = 0;
      stats.rounds_used = 0;
    }
  }
  stats.violations_remaining = final_violations;

  return stats;
}

// --- M3f causal-cap stage (repair.hpp's repair_table() doc comment, steps
// 5-10; eos-causality-repair.md) --------------------------------------------

// ln(10). The table stores its rho and T axes as log10, while the causality
// identities of eos-causality-repair.md S3 live in x = ln rho; every
// conversion below is one factor of this per order of x-differentiation.
constexpr double kLn10 = 2.30258509299404568402;

// c^2 in cgs. Together with the table's own energy_shift this is the *only*
// physical input the audit needs: c_s^2 is invariant under the adapter's
// kappa rescaling (which multiplies h by 1/kappa, a constant) and under the
// table's m_B convention, so auditing the raw-variable fit is exactly
// auditing the production adapter's interior (repair.hpp, step 5).
constexpr double kCLightSq = c_light_cm_s * c_light_cm_s;

// The six derivatives of one fitted field at one sample, with respect to the
// table's stored log10 axes (X = log10 rho, U = log10 T). fy is deliberately
// absent: Ye is a spectator of the causality chain rule (adiabats are taken
// at fixed Ye), so it is never differentiated here.
struct CausalDerivs {
  double f = 0.0, fx = 0.0, fu = 0.0, fxx = 0.0, fxu = 0.0, fuu = 0.0;
};

// c_s^2 at one sample, plus the two quantities it was assembled from (the
// projection reuses `h` along the traced adiabat; `eps` is carried for
// symmetry and for debugging, since the projection takes the node's own
// exact stored value rather than this spline-evaluated one).
struct CausalSample {
  // False when the chain rule has no answer at all here -- sigma_u <= 0 (no
  // adiabat through this point), h <= 0, or a non-finite intermediate. Such
  // a sample is neither a violation nor a repair target; the audit counts it
  // as `indeterminate` and moves on.
  bool ok = false;
  double cs2 = 0.0;
  double h = 0.0;   // 1 + eps + deps/dx|_s, dimensionless
  double eps = 0.0; // dimensionless specific internal energy (eps_cgs / c^2)
};

// eos-causality-repair.md S3, written out in the table's stored variables.
// With the adiabat's implicit slope U' = -sigma_X/sigma_U and curvature
// U'' = -(sigma_XX + 2 sigma_XU U' + sigma_UU U'^2)/sigma_U (the
// implicit-function derivatives of eos-adapter-F-to-U.md S3.1, in log10
// axes), and E = 10^L = eps_cgs + energy_shift:
//
//   dE/dX|_s   = E_X + E_U U'
//   d2E/dX2|_s = E_XX + 2 E_XU U' + E_UU U'^2 + E_U U''
//   eps = (E - shift)/c^2,  eps_x = (dE/dX|_s)/(ln10 c^2),
//   eps_xx = (d2E/dX2|_s)/(ln10^2 c^2)      [x = ln rho]
//   h = 1 + eps + eps_x,   c_s^2 = (eps_x + eps_xx)/h
//
// The last line is eos-adapter-F-to-U.md S3.2's h c_s^2 = 2 rho U_rho + rho^2
// U_rhorho rewritten in logs (rho U_rho = eps_x, rho^2 U_rhorho = eps_xx -
// eps_x), i.e. exactly c_s^2 = d ln h / dx|_s.
CausalSample causal_sample(const CausalDerivs &sg, const CausalDerivs &lg, double energy_shift) {
  CausalSample out;
  if (!(sg.fu > 0.0) || !std::isfinite(sg.fx)) {
    return out;
  }
  const double up = -sg.fx / sg.fu;
  const double upp = -(sg.fxx + 2.0 * sg.fxu * up + sg.fuu * up * up) / sg.fu;

  const double e_val = std::pow(10.0, lg.f); // eps_cgs + energy_shift, erg/g
  const double e_x = kLn10 * e_val * lg.fx;
  const double e_u = kLn10 * e_val * lg.fu;
  const double e_xx = kLn10 * e_val * (lg.fxx + kLn10 * lg.fx * lg.fx);
  const double e_xu = kLn10 * e_val * (lg.fxu + kLn10 * lg.fx * lg.fu);
  const double e_uu = kLn10 * e_val * (lg.fuu + kLn10 * lg.fu * lg.fu);

  const double d1 = e_x + e_u * up;
  const double d2 = e_xx + 2.0 * e_xu * up + e_uu * up * up + e_u * upp;

  const double eps = (e_val - energy_shift) / kCLightSq;
  const double eps_x = d1 / (kLn10 * kCLightSq);
  const double eps_xx = d2 / (kLn10 * kLn10 * kCLightSq);
  const double h = 1.0 + eps + eps_x;
  if (!(h > 0.0) || !std::isfinite(h)) {
    return out;
  }
  const double cs2 = (eps_x + eps_xx) / h;
  if (!std::isfinite(cs2)) {
    return out;
  }
  out.ok = true;
  out.cs2 = cs2;
  out.h = h;
  out.eps = eps;
  return out;
}

// What one causal audit pass measured. Sample counts, not node counts (the
// audit lives on the refined grid, exactly like the 3D stage's).
struct CausalAuditCounts {
  size_t violations = 0;         // c_s^2 >= cs2_max
  size_t interior_untouched = 0; // violating samples in non-edge-anchored runs
  size_t nonpositive = 0;        // c_s^2 <= 0
  size_t indeterminate = 0;      // CausalSample::ok == false
  size_t near_cap = 0;           // c_s^2 >= cs2_cap
  size_t treated_runs = 0;       // edge-anchored runs, i.e. runs in scope
  double cs2_max_seen = 0.0;
};

// Audits c_s^2 of the fitted `sv` (entropy) / `lv` (logenergy) splines on the
// refined grid (repair.hpp step 5), scopes the violations into runs along x
// (step 6), and -- when `node_start` is non-null (sized ntemp*nye, each entry
// pre-set to nrho) -- records, per (jT, kYe) data column, the lowest irho any
// treated (edge-anchored) run asks the projection to start from.
//
// Performance note, same spirit as audit3d_and_mark() above: this is a hot
// path (~2.6e8 samples for an SRO-sized (4,4,4) pass, each needing SIX
// derivatives of TWO fields), so it does not call bspline_eval3() per sample.
// Instead, for each refined (u, y) row it contracts the coefficient block
// along u and y once into three arrays indexed by the x coefficient index --
// value, d/du, d2/du2 -- after which every x sample on that row costs six
// length-4 dot products instead of a full 4x4x4 contraction. Rows are
// independent (-> OpenMP over the y axis); the counters are plain sums and
// the node_start updates are minima, so the result does not depend on thread
// count or scheduling.
CausalAuditCounts causal_audit(const BsplineView3 &sv, const BsplineView3 &lv, int nrho, int ntemp,
                                int nye, double energy_shift, int refine_xy, int refine_u,
                                double cs2_max, double cs2_cap, std::vector<int> *node_start) {
  const int refine_x = std::max(refine_xy, 1);
  const int refine_uc = std::max(refine_u, 1);
  const int count_x = (nrho - 1) * refine_x + 1;
  const int count_u = (ntemp - 1) * refine_uc + 1;
  const int count_y = (nye - 1) * refine_x + 1;

  // Per-x-sample cell index and the three basis quadruples, hoisted out of
  // every row (they depend only on ix).
  std::vector<int> cx_i(static_cast<size_t>(count_x));
  std::vector<std::array<double, 4>> cx_b(static_cast<size_t>(count_x)),
      cx_d(static_cast<size_t>(count_x)), cx_h(static_cast<size_t>(count_x));
  for (int ix = 0; ix < count_x; ++ix) {
    const double x = sv.x0 + static_cast<double>(ix) * sv.hx / static_cast<double>(refine_x);
    const detail::BsplineCell c = detail::bspline_cell(x, sv.x0, sv.hx, sv.nx);
    const detail::Basis4 b = detail::bspline_basis(c.t);
    const detail::Basis4 d = detail::bspline_dbasis(c.t);
    const detail::Basis4 h = detail::bspline_d2basis(c.t);
    cx_i[static_cast<size_t>(ix)] = c.i;
    cx_b[static_cast<size_t>(ix)] = {b.b0, b.b1, b.b2, b.b3};
    cx_d[static_cast<size_t>(ix)] = {d.b0, d.b1, d.b2, d.b3};
    cx_h[static_cast<size_t>(ix)] = {h.b0, h.b1, h.b2, h.b3};
  }

  const int nxp2 = sv.nx + 2;
  const int nup2 = sv.nu + 2;
  const double inv_hx = 1.0 / sv.hx;
  const double inv_hu = 1.0 / sv.hu;

  size_t violations = 0, interior = 0, nonpositive = 0, indeterminate = 0, near_cap = 0,
         treated_runs = 0;
  double cs2_max_seen = 0.0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) reduction(+ : violations, interior, nonpositive,           \
                                                          indeterminate, near_cap, treated_runs)      \
    reduction(max : cs2_max_seen)
#endif
  for (int iy = 0; iy < count_y; ++iy) {
    const double y = sv.y0 + static_cast<double>(iy) * sv.hy / static_cast<double>(refine_x);
    const detail::BsplineCell cy = detail::bspline_cell(y, sv.y0, sv.hy, sv.ny);
    const detail::Basis4 By = detail::bspline_basis(cy.t);
    const double by[4] = {By.b0, By.b1, By.b2, By.b3};
    const int ky = std::min(iy / refine_x, nye - 2);

    // Per-row workspace: the (u, y)-contracted coefficient lines, one triple
    // per field.
    std::vector<double> sb(static_cast<size_t>(nxp2)), sbu(static_cast<size_t>(nxp2)),
        sbuu(static_cast<size_t>(nxp2));
    std::vector<double> lb(static_cast<size_t>(nxp2)), lbu(static_cast<size_t>(nxp2)),
        lbuu(static_cast<size_t>(nxp2));

    for (int iu = 0; iu < count_u; ++iu) {
      const double u = sv.u0 + static_cast<double>(iu) * sv.hu / static_cast<double>(refine_uc);
      const detail::BsplineCell cu = detail::bspline_cell(u, sv.u0, sv.hu, sv.nu);
      const detail::Basis4 Bu = detail::bspline_basis(cu.t);
      const detail::Basis4 Du = detail::bspline_dbasis(cu.t);
      const detail::Basis4 Hu = detail::bspline_d2basis(cu.t);
      const double bu[4] = {Bu.b0, Bu.b1, Bu.b2, Bu.b3};
      const double du[4] = {Du.b0, Du.b1, Du.b2, Du.b3};
      const double hu4[4] = {Hu.b0, Hu.b1, Hu.b2, Hu.b3};
      const int ju = std::min(iu / refine_uc, ntemp - 2);

      std::fill(sb.begin(), sb.end(), 0.0);
      std::fill(sbu.begin(), sbu.end(), 0.0);
      std::fill(sbuu.begin(), sbuu.end(), 0.0);
      std::fill(lb.begin(), lb.end(), 0.0);
      std::fill(lbu.begin(), lbu.end(), 0.0);
      std::fill(lbuu.begin(), lbuu.end(), 0.0);
      for (int r = 0; r < 4; ++r) {
        for (int q = 0; q < 4; ++q) {
          const int base = nxp2 * ((cu.i + q) + nup2 * (cy.i + r));
          const double w = bu[q] * by[r];
          const double wu = du[q] * by[r];
          const double wuu = hu4[q] * by[r];
          for (int m = 0; m < nxp2; ++m) {
            const double cs = sv.c[base + m];
            sb[static_cast<size_t>(m)] += w * cs;
            sbu[static_cast<size_t>(m)] += wu * cs;
            sbuu[static_cast<size_t>(m)] += wuu * cs;
            const double cl = lv.c[base + m];
            lb[static_cast<size_t>(m)] += w * cl;
            lbu[static_cast<size_t>(m)] += wu * cl;
            lbuu[static_cast<size_t>(m)] += wuu * cl;
          }
        }
      }

      int run_start = -1; // first ix of the current maximal violation run
      for (int ix = 0; ix < count_x; ++ix) {
        const int ci = cx_i[static_cast<size_t>(ix)];
        const std::array<double, 4> &bx = cx_b[static_cast<size_t>(ix)];
        const std::array<double, 4> &dx = cx_d[static_cast<size_t>(ix)];
        const std::array<double, 4> &hx4 = cx_h[static_cast<size_t>(ix)];

        CausalDerivs sg, lg;
        for (int p = 0; p < 4; ++p) {
          const size_t m = static_cast<size_t>(ci + p);
          sg.f += bx[static_cast<size_t>(p)] * sb[m];
          sg.fx += dx[static_cast<size_t>(p)] * sb[m];
          sg.fxx += hx4[static_cast<size_t>(p)] * sb[m];
          sg.fu += bx[static_cast<size_t>(p)] * sbu[m];
          sg.fxu += dx[static_cast<size_t>(p)] * sbu[m];
          sg.fuu += bx[static_cast<size_t>(p)] * sbuu[m];
          lg.f += bx[static_cast<size_t>(p)] * lb[m];
          lg.fx += dx[static_cast<size_t>(p)] * lb[m];
          lg.fxx += hx4[static_cast<size_t>(p)] * lb[m];
          lg.fu += bx[static_cast<size_t>(p)] * lbu[m];
          lg.fxu += dx[static_cast<size_t>(p)] * lbu[m];
          lg.fuu += bx[static_cast<size_t>(p)] * lbuu[m];
        }
        sg.fx *= inv_hx;
        sg.fxx *= inv_hx * inv_hx;
        sg.fu *= inv_hu;
        sg.fxu *= inv_hx * inv_hu;
        sg.fuu *= inv_hu * inv_hu;
        lg.fx *= inv_hx;
        lg.fxx *= inv_hx * inv_hx;
        lg.fu *= inv_hu;
        lg.fxu *= inv_hx * inv_hu;
        lg.fuu *= inv_hu * inv_hu;

        const CausalSample cs = causal_sample(sg, lg, energy_shift);
        bool violating = false;
        if (!cs.ok) {
          ++indeterminate;
        } else {
          if (cs.cs2 > cs2_max_seen) {
            cs2_max_seen = cs.cs2;
          }
          if (cs.cs2 <= 0.0) {
            ++nonpositive;
          }
          if (cs.cs2 >= cs2_cap) {
            ++near_cap;
          }
          if (cs.cs2 >= cs2_max) {
            violating = true;
            ++violations;
          }
        }

        if (violating) {
          if (run_start < 0) {
            run_start = ix;
          }
        } else if (run_start >= 0) {
          // A run that ends before the x_hi edge is interior: reported, never
          // edited (repair.hpp step 6).
          interior += static_cast<size_t>(ix - run_start);
          run_start = -1;
        }
      }

      if (run_start >= 0) {
        ++treated_runs;
      }
      if (run_start >= 0 && node_start != nullptr) {
        // Edge-anchored: every node at or above the run's first refined x
        // position, on the 2x2 (jT, kYe) node corners of this row's owning
        // cell, is a projection target.
        const int i_start = std::min((run_start + refine_x - 1) / refine_x, nrho - 1);
        const int j_hi = std::min(ju + 1, ntemp - 1);
        const int k_hi = std::min(ky + 1, nye - 1);
#ifdef _OPENMP
#pragma omp critical(eeos_causal_node_start)
#endif
        {
          for (int k = ky; k <= k_hi; ++k) {
            for (int j = ju; j <= j_hi; ++j) {
              int &slot = (*node_start)[static_cast<size_t>(j + ntemp * k)];
              if (i_start < slot) {
                slot = i_start;
              }
            }
          }
        }
      }
    }
  }

  CausalAuditCounts out;
  out.violations = violations;
  out.interior_untouched = interior;
  out.nonpositive = nonpositive;
  out.indeterminate = indeterminate;
  out.near_cap = near_cap;
  out.treated_runs = treated_runs;
  out.cs2_max_seen = cs2_max_seen;
  return out;
}

// sigma and its u-derivative only: the inner loop of the adiabat trace's
// 1D T-solve. Deliberately cheaper than bspline_eval3() (which would also
// compute five derivatives the solve never looks at).
struct SigmaVal {
  double f = 0.0, fu = 0.0;
};

SigmaVal eval_sigma_fu(const BsplineView3 &v, double x, double u, double y) {
  const detail::BsplineCell cx = detail::bspline_cell(x, v.x0, v.hx, v.nx);
  const detail::BsplineCell cu = detail::bspline_cell(u, v.u0, v.hu, v.nu);
  const detail::BsplineCell cy = detail::bspline_cell(y, v.y0, v.hy, v.ny);
  const detail::Basis4 Bx = detail::bspline_basis(cx.t);
  const detail::Basis4 Bu = detail::bspline_basis(cu.t);
  const detail::Basis4 Du = detail::bspline_dbasis(cu.t);
  const detail::Basis4 By = detail::bspline_basis(cy.t);
  const double bx[4] = {Bx.b0, Bx.b1, Bx.b2, Bx.b3};
  const double bu[4] = {Bu.b0, Bu.b1, Bu.b2, Bu.b3};
  const double du[4] = {Du.b0, Du.b1, Du.b2, Du.b3};
  const double by[4] = {By.b0, By.b1, By.b2, By.b3};

  const int nxp2 = v.nx + 2;
  const int nup2 = v.nu + 2;
  double f = 0.0, fu = 0.0;
  for (int r = 0; r < 4; ++r) {
    for (int q = 0; q < 4; ++q) {
      const int base = nxp2 * ((cu.i + q) + nup2 * (cy.i + r));
      double sf = 0.0;
      for (int p = 0; p < 4; ++p) {
        sf += bx[p] * v.c[base + cx.i + p];
      }
      f += bu[q] * by[r] * sf;
      fu += du[q] * by[r] * sf;
    }
  }
  SigmaVal out;
  out.f = f;
  out.fu = fu / v.hu;
  return out;
}

// Solves sigma(x, u, y) = s for u in [u_lo, u_hi] (safeguarded Newton on a
// maintained bracket, warm-started from u_guess). When s lies outside
// [sigma(u_lo), sigma(u_hi)] the adiabat has left the box through that edge:
// the solve returns the edge value and sets *on_edge (repair.hpp step 7a --
// the u_min case is the fully degenerate regime where following the edge
// approximates the adiabat excellently, eos-causality-repair.md S6).
double solve_u_on_adiabat(const BsplineView3 &v, double x, double y, double s, double u_lo,
                           double u_hi, double u_guess, bool *on_edge) {
  *on_edge = false;
  const SigmaVal lo = eval_sigma_fu(v, x, u_lo, y);
  if (!(lo.f < s)) {
    *on_edge = true;
    return u_lo;
  }
  const SigmaVal hi = eval_sigma_fu(v, x, u_hi, y);
  if (!(hi.f > s)) {
    *on_edge = true;
    return u_hi;
  }

  double a = u_lo, b = u_hi;
  double u = std::min(std::max(u_guess, u_lo), u_hi);
  const double span = u_hi - u_lo;
  for (int it = 0; it < 80; ++it) {
    const SigmaVal e = eval_sigma_fu(v, x, u, y);
    const double r = e.f - s;
    if (r > 0.0) {
      b = u;
    } else if (r < 0.0) {
      a = u;
    } else {
      return u;
    }
    if (b - a <= 1e-15 * span) {
      break;
    }
    double next = (e.fu > 0.0) ? (u - r / e.fu) : (0.5 * (a + b));
    if (!(next > a) || !(next < b)) {
      next = 0.5 * (a + b);
    }
    if (std::fabs(next - u) <= 1e-15 * (1.0 + std::fabs(u))) {
      return next;
    }
    u = next;
  }
  return 0.5 * (a + b);
}

// The integrating-factor step factor (e^{c d} - e^{-d}) / (1 + c), written as
// e^{-d} * d * expm1(z)/z with z = (1+c) d so that the c -> -1 limit (where
// the closed form is 0/0) is handled by the same expression rather than a
// special case, and so that small z keeps full relative accuracy.
double ode_step_factor(double one_plus_c, double delta) {
  const double z = one_plus_c * delta;
  const double ratio = (std::fabs(z) < 1e-8) ? (1.0 + 0.5 * z) : (std::expm1(z) / z);
  return std::exp(-delta) * delta * ratio;
}

// Outcome of one node's projection (repair.hpp step 7).
enum class ProjectResult { unchanged, changed, gave_up };

// Projects the single data node (irho, jT, kYe) -- s and the node's own
// stored logenergy passed in -- onto the causal envelope along its adiabat.
// `h_orig` / `h_env` are caller-owned scratch buffers, reused across the
// nodes of a column so the per-node cost carries no allocation.
ProjectResult project_node(const BsplineView3 &sv, const BsplineView3 &lv, int irho, int jT,
                            int ntemp, double y, double s, double logenergy_node,
                            double energy_shift, const RepairOptions &options,
                            std::vector<double> &h_orig, double *new_logenergy) {
  const int refine_x = std::max(options.refine3d_xy, 1);
  const double dx_log10 = sv.hx / static_cast<double>(refine_x);
  const double delta = dx_log10 * kLn10; // one refined step in x = ln rho, > 0
  const double u_lo = sv.u0;
  const double u_hi = sv.u0 + static_cast<double>(ntemp - 1) * sv.hu;
  const int ix_node = irho * refine_x; // refined x index of the node
  const int max_steps = std::min(options.trace_depth_max * refine_x, ix_node);
  const int pad = std::max(options.anchor_pad, 1);

  h_orig.clear();
  double u_prev = sv.u0 + static_cast<double>(jT) * sv.hu;
  int anchor = -1;
  int causal_run = 0;
  for (int m = 0; m <= max_steps; ++m) {
    const double x = sv.x0 + static_cast<double>(ix_node - m) * dx_log10;
    double u = u_prev;
    if (m > 0) {
      bool on_edge = false;
      u = solve_u_on_adiabat(sv, x, y, s, u_lo, u_hi, u_prev, &on_edge);
    }
    u_prev = u;

    const BsplineEval3 se = bspline_eval3(sv, x, u, y);
    const BsplineEval3 le = bspline_eval3(lv, x, u, y);
    const CausalDerivs sg{se.f, se.fx, se.fu, se.fxx, se.fxu, se.fuu};
    const CausalDerivs lg{le.f, le.fx, le.fu, le.fxx, le.fxu, le.fuu};
    const CausalSample cs = causal_sample(sg, lg, energy_shift);
    if (!cs.ok) {
      return ProjectResult::gave_up; // no adiabat here: report, do not edit
    }
    h_orig.push_back(cs.h);

    if (cs.cs2 <= options.cs2_cap) {
      if (++causal_run >= pad) {
        anchor = m;
        break;
      }
    } else {
      causal_run = 0;
    }
  }
  if (anchor < 0) {
    return ProjectResult::gave_up; // trace_depth_max, or the low-rho edge
  }
  if (anchor == 0) {
    return ProjectResult::unchanged; // the node itself anchors: nothing above it
  }

  // March back up: the envelope + energy-consistency ODE (repair.hpp step
  // 7c), on the traced stretch [node .. anchor].
  const double e_orig = std::pow(10.0, logenergy_node);
  const double eps_orig = (e_orig - energy_shift) / kCLightSq;
  h_orig.resize(static_cast<size_t>(anchor) + 1);
  const CausalEnvelope env = causal_envelope(h_orig, eps_orig, options.cs2_cap, delta);
  if (!env.ok) {
    return ProjectResult::gave_up;
  }
  if (!env.bound) {
    return ProjectResult::unchanged;
  }
  const double eps_new = env.eps_node;
  if (!std::isfinite(eps_new) || !(eps_new < eps_orig)) {
    return ProjectResult::unchanged;
  }
  const double e_new = eps_new * kCLightSq + energy_shift;
  if (!(e_new > 0.0) || !std::isfinite(e_new)) {
    return ProjectResult::unchanged; // would break eps + shift > 0: leave it alone
  }
  const double l_new = std::log10(e_new);
  if (!std::isfinite(l_new) || !(l_new < logenergy_node)) {
    return ProjectResult::unchanged;
  }
  *new_logenergy = l_new;
  return ProjectResult::changed;
}

struct CausalProjectOutcome {
  size_t nodes_written = 0;
  size_t gave_up = 0;
};

// Runs project_node() over every node the audit scoped in (repair.hpp step
// 7), writing the results into `data` and marking every changed index in
// `touched` (for the per-column re-repair of step 8). Columns are
// independent and every trace reads only the pre-round fits plus its own
// node's stored value, so the outcome does not depend on scheduling.
CausalProjectOutcome causal_project(const BsplineView3 &sv, const BsplineView3 &lv, int nrho,
                                     int ntemp, int nye, const std::vector<double> &sigma,
                                     std::vector<double> &data, const std::vector<int> &node_start,
                                     double energy_shift, const RepairOptions &options,
                                     std::vector<char> &touched) {
  std::vector<std::pair<int, int>> cols; // (jT, kYe), fixed (kYe, jT) order
  for (int k = 0; k < nye; ++k) {
    for (int j = 0; j < ntemp; ++j) {
      if (node_start[static_cast<size_t>(j + ntemp * k)] < nrho) {
        cols.emplace_back(j, k);
      }
    }
  }

  size_t written = 0, gave_up = 0;
  const long long ncols = static_cast<long long>(cols.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) reduction(+ : written, gave_up)
#endif
  for (long long ci = 0; ci < ncols; ++ci) {
    const int jT = cols[static_cast<size_t>(ci)].first;
    const int kYe = cols[static_cast<size_t>(ci)].second;
    const double y = sv.y0 + static_cast<double>(kYe) * sv.hy;
    const int i0 = node_start[static_cast<size_t>(jT + ntemp * kYe)];

    std::vector<double> h_orig;
    h_orig.reserve(static_cast<size_t>(options.trace_depth_max) *
                   static_cast<size_t>(std::max(options.refine3d_xy, 1)) + 2);

    for (int irho = i0; irho < nrho; ++irho) {
      const size_t idx = static_cast<size_t>(irho) +
                         static_cast<size_t>(nrho) *
                             (static_cast<size_t>(jT) + static_cast<size_t>(ntemp) *
                                                             static_cast<size_t>(kYe));
      double l_new = 0.0;
      const ProjectResult r =
          project_node(sv, lv, irho, jT, ntemp, y, sigma[idx], data[idx], energy_shift, options,
                       h_orig, &l_new);
      if (r == ProjectResult::gave_up) {
        ++gave_up;
      } else if (r == ProjectResult::changed) {
        data[idx] = l_new;
        touched[idx] = 1;
        ++written;
      }
    }
  }

  CausalProjectOutcome out;
  out.nodes_written = written;
  out.gave_up = gave_up;
  return out;
}

// The whole causal-cap stage (repair.hpp steps 5-10), run once per
// repair_table() call after every field has been through both monotonicity
// stages. Edits only `logenergy_data`; `sigma_data` is read-only, so node
// adiabat labels are stable across rounds and the T-solve's sigma_u > 0
// guarantee is untouched.
//
// `pre_mono_entropy` / `pre_mono_logenergy`, when non-null, supply the
// pre-stage (4,4,4) fu-monotonicity counts the backstop compares against;
// repair_table() passes the 3D stage's own violations3d_remaining there when
// that stage ran, so the common path costs no extra audit. When null, the
// counts are measured here.
void causal_cap_stage(const RawTable &table, const std::vector<double> &sigma_data,
                       std::vector<double> &logenergy_data, const RepairOptions &options,
                       const size_t *pre_mono_entropy, const size_t *pre_mono_logenergy,
                       RepairResult::CausalCapSummary &out) {
  const int nrho = static_cast<int>(table.nrho());
  const int ntemp = static_cast<int>(table.ntemp());
  const int nye = static_cast<int>(table.nye());
  if (nrho < 4 || ntemp < 4 || nye < 4) {
    return;
  }
  if (!table.has_attribute("energy_shift")) {
    return;
  }
  const double energy_shift = table.attribute("energy_shift");
  if (!std::isfinite(energy_shift)) {
    return;
  }

  // Axis geometry, exactly as adapter_build.cpp derives it: the fit is
  // uniform-knot, so only the first value and the average spacing enter
  // (uniformity itself is the adapter build's precondition, CODE.md "M2
  // design notes"; a non-uniform axis is already outside this library's
  // contract).
  const double x0 = table.logrho().front();
  const double hx = (table.logrho().back() - x0) / static_cast<double>(nrho - 1);
  const double u0 = table.logtemp().front();
  const double hu = (table.logtemp().back() - u0) / static_cast<double>(ntemp - 1);
  const double y0 = table.ye().front();
  const double hy = (table.ye().back() - y0) / static_cast<double>(nye - 1);
  if (!(hx > 0.0) || !(hu > 0.0) || !(hy > 0.0)) {
    return;
  }

  out.ran = true;

  // sigma is never edited by this stage, so its fit is built once and reused
  // by every round's audit and every node's trace.
  const Bspline3 sfit = fit_bspline_3d(nrho, ntemp, nye, x0, hx, u0, hu, y0, hy, sigma_data);
  const BsplineView3 sv = sfit.view();

  const std::vector<double> pre_stage = logenergy_data;

  // Step 9's main loop: M2d-1's best-state tracking, run *while the stage
  // keeps improving* rather than for a fixed number of rounds (M3j).
  //
  // Why the patience is 2 here where spline_safe_3d_field() uses 4: the two
  // loops have measurably different trajectories, and 4 buys that one
  // something it does not buy this one. The 3D stage diffuses, which is a
  // noisy operation -- see its own doc comment: some of its converging
  // trajectories dip *up* for a few rounds before trending back down (SRO's
  // "entropy" field walks ..., 1010, 976, 976, 995, 1039, 997), so a short
  // patience there risks stopping before a recovery it would have found. The
  // causal-cap loop instead re-solves the same envelope from progressively
  // deeper anchors, and no measured trajectory ever turns back up while it is
  // still descending: every one falls monotonically and then flattens (LS220
  // 1035875, 2452, 2176, 2061, 2008, 1967, 1961, 1956, then a plateau; SFHo
  // 333862, 113887, 60713, ..., 11446, then a plateau; SRO wobbles by a
  // single sample, 6050 -> 6051, inside its own). Two non-improving rounds
  // here therefore already ARE the plateau, and rounds 3 and 4 would each pay
  // a full fit + audit + project to confirm it. Measured to settle it: 2 and
  // 4 give bit-identical output on all four real tables and on both synthetic
  // presets -- see CODE.md "DD2 / SFHo empirical findings".
  constexpr int kPatience = 2;
  size_t best_violations = std::numeric_limits<size_t>::max();
  std::vector<double> best_data;
  size_t best_gave_up = 0;
  int best_rounds = 0;
  int patience = 0;
  int rounds_applied = 0;
  size_t gave_up_total = 0;
  std::vector<int> node_start(static_cast<size_t>(ntemp) * static_cast<size_t>(nye), nrho);

  // options.causal_rounds_max is the runaway backstop, not the working
  // budget (repair.hpp): every exit below is a property of the data, so a
  // loop that leaves through this `for`'s own condition is the one case
  // where the stage can still have work left for a second run.
  for (int round = 0; round < options.causal_rounds_max; ++round) {
    const Bspline3 lfit =
        fit_bspline_3d(nrho, ntemp, nye, x0, hx, u0, hu, y0, hy, logenergy_data);
    std::fill(node_start.begin(), node_start.end(), nrho);
    const CausalAuditCounts counts =
        causal_audit(sv, lfit.view(), nrho, ntemp, nye, energy_shift, options.refine3d_xy,
                     options.refine3d_u, options.cs2_max, options.cs2_cap, &node_start);
    out.rounds_violation_history.push_back(counts.violations);

    if (best_violations == std::numeric_limits<size_t>::max() ||
        counts.violations < best_violations) {
      best_violations = counts.violations;
      best_data = logenergy_data;
      best_gave_up = gave_up_total;
      best_rounds = rounds_applied;
      patience = 0;
    } else if (++patience >= kPatience) {
      break; // plateaued or worsening: revert to the best state below
    }
    if (counts.violations == 0 || counts.treated_runs == 0) {
      // Clean, or nothing left in scope: every remaining violation sits in
      // an interior run (repair.hpp step 6), which this stage never edits.
      break;
    }

    std::vector<char> touched(logenergy_data.size(), 0);
    const CausalProjectOutcome po =
        causal_project(sv, lfit.view(), nrho, ntemp, nye, sigma_data, logenergy_data, node_start,
                       energy_shift, options, touched);
    gave_up_total += po.gave_up;
    if (po.nodes_written == 0) {
      break; // nothing left this stage can act on; further rounds would repeat
    }
    rerepair_affected_columns(logenergy_data, nrho, ntemp, nye, touched,
                              options.min_slope_logenergy, options);
    ++rounds_applied;
  }
  if (best_violations != std::numeric_limits<size_t>::max()) {
    logenergy_data = best_data;
    out.rounds_used = best_rounds;
    out.trace_giveups = best_gave_up;
  } else {
    out.trace_giveups = gave_up_total;
  }

  // Step 10: verification at (4,4,4) plus the lexicographic backstop.
  const bool unchanged = (logenergy_data == pre_stage);
  const Bspline3 lfit_after =
      fit_bspline_3d(nrho, ntemp, nye, x0, hx, u0, hu, y0, hy, logenergy_data);
  const CausalAuditCounts after = causal_audit(sv, lfit_after.view(), nrho, ntemp, nye,
                                                energy_shift, 4, 4, options.cs2_max,
                                                options.cs2_cap, nullptr);
  out.rounds_violation_history.push_back(after.violations);

  CausalAuditCounts before = after;
  if (!unchanged) {
    const Bspline3 lfit_pre = fit_bspline_3d(nrho, ntemp, nye, x0, hx, u0, hu, y0, hy, pre_stage);
    before = causal_audit(sv, lfit_pre.view(), nrho, ntemp, nye, energy_shift, 4, 4,
                          options.cs2_max, options.cs2_cap, nullptr);
  }

  std::vector<char> scratch(logenergy_data.size(), 0);
  out.mono_entropy = pre_mono_entropy != nullptr
                          ? *pre_mono_entropy
                          : audit3d_and_mark(nrho, ntemp, nye, sigma_data, 4, 4,
                                             options.spline_slope_floor, options.diffuse_window,
                                             scratch);
  out.mono_logenergy_before =
      pre_mono_logenergy != nullptr
          ? *pre_mono_logenergy
          : audit3d_and_mark(nrho, ntemp, nye, pre_stage, 4, 4, options.spline_slope_floor,
                             options.diffuse_window, scratch);
  out.mono_logenergy_after =
      unchanged ? out.mono_logenergy_before
                : audit3d_and_mark(nrho, ntemp, nye, logenergy_data, 4, 4,
                                   options.spline_slope_floor, options.diffuse_window, scratch);

  out.projected_violations = after.violations;
  out.projected_mono_logenergy = out.mono_logenergy_after;

  const bool worse_monotonicity = out.mono_logenergy_after > out.mono_logenergy_before;
  const bool worse_causality = after.violations > before.violations;
  const CausalAuditCounts *kept = &after;
  if (!unchanged && (worse_monotonicity || worse_causality)) {
    logenergy_data = pre_stage;
    out.reverted = true;
    out.rounds_used = 0;
    out.mono_logenergy_after = out.mono_logenergy_before;
    kept = &before;
  }

  out.violations_before = before.violations;
  out.violations_after = kept->violations;
  out.interior_untouched = kept->interior_untouched;
  out.cs2_nonpositive = kept->nonpositive;
  out.cs2_indeterminate = kept->indeterminate;
  out.cs2_near_cap = kept->near_cap;
  out.cs2_max_seen = kept->cs2_max_seen;

  size_t capped = 0;
  for (size_t i = 0; i < logenergy_data.size(); ++i) {
    if (logenergy_data[i] != pre_stage[i]) {
      ++capped;
    }
  }
  out.nodes_capped = capped;
}

} // namespace

CausalEnvelope causal_envelope(const std::vector<double> &h, double eps_node, double cs2_cap,
                                double delta) {
  CausalEnvelope out;
  out.eps_node = eps_node;
  if (h.size() < 2 || !(delta > 0.0) || !std::isfinite(delta)) {
    out.ok = h.size() < 2; // nothing above the anchor is a well-posed no-op
    return out;
  }

  const int anchor = static_cast<int>(h.size()) - 1;
  const double decay = std::exp(-delta);
  const double cap_growth = std::exp(cs2_cap * delta);

  // Marching up from the anchor, carrying only the previous step's envelope
  // value (the envelope is a one-term recurrence) and the running deficit
  // D = eps_orig - eps_env, which starts at exactly 0 at the anchor. On a
  // step where the envelope does not bind, h_env == h_orig *bitwise*, so the
  // two integrating-factor terms cancel exactly and D is only decayed --
  // hence D stays exactly 0 while the original stays causal, and a causal
  // node is left bit-identical (this is what makes the stage idempotent).
  double h_env_prev = h[static_cast<size_t>(anchor)];
  double deficit = 0.0;
  for (int m = anchor - 1; m >= 0; --m) {
    const double h_prev_orig = h[static_cast<size_t>(m + 1)];
    const double h_here_orig = h[static_cast<size_t>(m)];
    if (!(h_env_prev > 0.0) || !(h_prev_orig > 0.0) || !(h_here_orig > 0.0) ||
        !std::isfinite(h_here_orig)) {
      return out; // ok stays false: an ill-posed step
    }
    const double h_capped = h_env_prev * cap_growth;
    double h_here_env = h_here_orig;
    if (h_capped < h_here_env) {
      h_here_env = h_capped;
      out.bound = true;
    }

    const double a_term =
        h_prev_orig * ode_step_factor(1.0 + std::log(h_here_orig / h_prev_orig) / delta, delta);
    const double b_term =
        h_env_prev * ode_step_factor(1.0 + std::log(h_here_env / h_env_prev) / delta, delta);
    deficit = decay * deficit + (a_term - b_term);
    h_env_prev = h_here_env;
  }

  if (!std::isfinite(deficit)) {
    return out;
  }
  out.ok = true;
  if (out.bound) {
    out.eps_node = eps_node - deficit;
  }
  return out;
}

RepairResult repair_table(RawTable &table, const RepairOptions &options) {
  // Validate every listed field before touching any of them, so a throw
  // leaves `table` completely untouched (CODE.md: structural problems mean
  // a broken file and must not be partially "fixed" on the way to being
  // reported).
  for (const std::string &field : options.fields) {
    if (!table.has_field(field)) {
      throw std::runtime_error("repair_table: table has no field '" + field + "'");
    }
    min_slope_for(field, options); // throws std::invalid_argument if unknown
    for (double v : table.field(field)) {
      if (!std::isfinite(v)) {
        throw std::runtime_error("repair_table: field '" + field + "' has a non-finite value");
      }
    }
  }

  // The causal-cap stage's own precondition: the cap must be a physical sound
  // speed strictly below the audit threshold, or its hysteresis (the whole
  // mechanism that converges the refit ringing) is gone. Reported the same way
  // an unknown field name is -- a caller error, not table noise.
  if (options.causal_cap && (!(options.cs2_cap > 0.0) || !(options.cs2_cap < options.cs2_max))) {
    throw std::invalid_argument("repair_table: causal_cap requires 0 < cs2_cap < cs2_max");
  }

  RepairResult result;

  const size_t nrho = table.nrho();
  const size_t ntemp = table.ntemp();
  const size_t nye = table.nye();
  const size_t ncol = nrho * nye;

  // Snapshots taken before ANY stage runs, so the final write-back below can
  // diff against the true original regardless of how many stages/rounds
  // touched a given index (repair.hpp's repair_table() doc comment, "Final
  // write-back"). Held for every listed field at once because the causal-cap
  // stage runs *after* all of them and can still change "logenergy".
  std::vector<std::vector<double>> originals;
  originals.reserve(options.fields.size());
  std::vector<std::vector<SplineSafeStats>> per_column_spline_by_field;
  per_column_spline_by_field.reserve(options.fields.size());
  std::vector<ThreeDStats> stats3d_by_field(options.fields.size());

  for (size_t fi = 0; fi < options.fields.size(); ++fi) {
    const std::string &field = options.fields[fi];
    const double min_slope = min_slope_for(field, options);
    std::vector<double> &data = table.field(field);

    originals.push_back(data);
    per_column_spline_by_field.emplace_back(ncol);
    std::vector<SplineSafeStats> &per_column_spline = per_column_spline_by_field.back();

    // Per-column stage (repair.hpp steps 0-1), independently on every
    // (irho, kYe) column.
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t icol = 0; icol < ncol; ++icol) {
      const size_t kYe = icol / nrho;
      const size_t irho = icol % nrho;

      std::vector<double> column(ntemp);
      for (size_t jT = 0; jT < ntemp; ++jT) {
        column[jT] = data[table.index(irho, jT, kYe)];
      }

      // Step 0: base pass (PAVA + strictify).
      repair_column(column, min_slope);

      // Step 1: spline-safe smoothing loop (repair.hpp doc comment).
      if (options.spline_safe) {
        per_column_spline[icol] = spline_safe_column(column, min_slope, options);
      }

      for (size_t jT = 0; jT < ntemp; ++jT) {
        data[table.index(irho, jT, kYe)] = column[jT];
      }
    }

    // Field-wide 3D stage (M2d-1, repair.hpp steps 2-3), once per field.
    if (options.spline_safe_3d) {
      stats3d_by_field[fi] = spline_safe_3d_field(data, nrho, ntemp, nye, min_slope, options);
    }
  }

  // Causal-cap stage (M3f, repair.hpp steps 5-10): once, over BOTH fields,
  // after every field has been through the two monotonicity stages. It reads
  // "entropy" and edits only "logenergy", so it must run before the
  // write-back below.
  if (options.causal_cap) {
    const auto field_index = [&](const std::string &name) -> size_t {
      for (size_t fi = 0; fi < options.fields.size(); ++fi) {
        if (options.fields[fi] == name) {
          return fi;
        }
      }
      return options.fields.size();
    };
    const size_t si = field_index("entropy");
    const size_t li = field_index("logenergy");
    if (si < options.fields.size() && li < options.fields.size()) {
      // The 3D stage's own final (4,4,4) count IS the pre-causal-stage
      // monotonicity count for that field (it measured exactly the state it
      // left behind, and nothing has touched either field since), so reuse
      // it rather than paying for two more (4,4,4) audits.
      const bool have_s = !stats3d_by_field[si].violation_history.empty();
      const bool have_l = !stats3d_by_field[li].violation_history.empty();
      const size_t mono_s = stats3d_by_field[si].violations_remaining;
      const size_t mono_l = stats3d_by_field[li].violations_remaining;
      causal_cap_stage(table, table.field("entropy"), table.field("logenergy"), options,
                       have_s ? &mono_s : nullptr, have_l ? &mono_l : nullptr, result.causal_cap);
    }
  }

  // Final write-back, per field: diff the FINAL data (after every stage
  // above) against the pre-repair original, in the fixed (field, kYe, irho,
  // jT) order RepairResult::entries documents -- independent of how any
  // stage above scheduled its OpenMP work.
  for (size_t fi = 0; fi < options.fields.size(); ++fi) {
    const std::string &field = options.fields[fi];
    const std::vector<double> &data = table.field(field);
    const std::vector<double> &original_field = originals[fi];
    const std::vector<SplineSafeStats> &per_column_spline = per_column_spline_by_field[fi];

    RepairResult::FieldSummary summary;
    summary.field = field;
    double sum_sq = 0.0;
    for (size_t kYe = 0; kYe < nye; ++kYe) {
      for (size_t irho = 0; irho < nrho; ++irho) {
        for (size_t jT = 0; jT < ntemp; ++jT) {
          const size_t idx = table.index(irho, jT, kYe);
          if (data[idx] != original_field[idx]) {
            result.entries.push_back(RepairEntry{field, irho, jT, kYe, original_field[idx], data[idx]});
            ++summary.modified;
            const double abs_change = std::fabs(data[idx] - original_field[idx]);
            summary.max_abs_change = std::max(summary.max_abs_change, abs_change);
            sum_sq += abs_change * abs_change;
          }
        }
      }
    }
    summary.rms_change =
        summary.modified > 0 ? std::sqrt(sum_sq / static_cast<double>(summary.modified)) : 0.0;

    if (options.spline_safe) {
      summary.spline_rounds_histogram.assign(static_cast<size_t>(options.spline_rounds_max) + 1, 0);
    }
    for (size_t icol = 0; icol < ncol; ++icol) {
      const SplineSafeStats &s = per_column_spline[icol];
      summary.spline_rounds_used_max = std::max(summary.spline_rounds_used_max, s.rounds_used);
      if (s.needed_smoothing) {
        ++summary.spline_columns_smoothed;
      }
      if (s.still_violating) {
        ++summary.spline_columns_still_violating;
      }
      if (options.spline_safe) {
        ++summary.spline_rounds_histogram[static_cast<size_t>(s.rounds_used)];
      }
    }

    summary.rounds3d_used = stats3d_by_field[fi].rounds_used;
    summary.points_diffused_3d = stats3d_by_field[fi].points_diffused;
    summary.violations3d_remaining = stats3d_by_field[fi].violations_remaining;
    summary.rounds3d_violation_history = std::move(stats3d_by_field[fi].violation_history);

    result.summaries.push_back(std::move(summary));
  }

  result.status = result.entries.empty() ? Status::ok : Status::repaired;
  return result;
}

void RepairResult::print(std::ostream &os) const {
  os << "repair_table: " << (status == Status::ok ? "no changes (already monotone)" : "repaired")
     << "\n";
  for (const FieldSummary &s : summaries) {
    os << "  " << s.field << ": " << s.modified << " value(s) changed";
    if (s.modified > 0) {
      os << ", max |change| = " << s.max_abs_change << ", rms change = " << s.rms_change;
    }
    os << "\n";
    os << "    spline-safe: rounds_used(max)=" << s.spline_rounds_used_max
       << " columns_smoothed=" << s.spline_columns_smoothed
       << " columns_still_violating=" << s.spline_columns_still_violating << "\n";
    if (!s.spline_rounds_histogram.empty()) {
      os << "    spline-safe rounds histogram:";
      for (size_t r = 0; r < s.spline_rounds_histogram.size(); ++r) {
        if (s.spline_rounds_histogram[r] > 0) {
          os << " rounds=" << r << ":" << s.spline_rounds_histogram[r];
        }
      }
      os << "\n";
    }
    os << "    spline-safe-3d: rounds_used=" << s.rounds3d_used
       << " points_diffused=" << s.points_diffused_3d
       << " violations_remaining=" << s.violations3d_remaining << "\n";
  }

  const CausalCapSummary &c = causal_cap;
  if (c.ran) {
    os << "  causal-cap: rounds_used=" << c.rounds_used << " nodes_capped=" << c.nodes_capped
       << " cs2_violations " << c.violations_before << " -> " << c.violations_after << " (4,4,4)"
       << (c.reverted ? " [REVERTED]" : "") << "\n";
    os << "    causal-cap scope: interior_untouched=" << c.interior_untouched
       << " cs2_nonpositive=" << c.cs2_nonpositive << " cs2_indeterminate=" << c.cs2_indeterminate
       << " cs2_ge_cap=" << c.cs2_near_cap << " cs2_max=" << c.cs2_max_seen
       << " trace_giveups=" << c.trace_giveups << "\n";
    os << "    causal-cap monotonicity (4,4,4): entropy=" << c.mono_entropy
       << " logenergy " << c.mono_logenergy_before << " -> " << c.mono_logenergy_after << "\n";
    if (c.reverted) {
      os << "    causal-cap rejected state (4,4,4): cs2_violations=" << c.projected_violations
         << " logenergy_monotonicity=" << c.projected_mono_logenergy << "\n";
    }
  }
}

} // namespace eeos
