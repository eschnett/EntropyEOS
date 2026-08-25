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

} // namespace

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

  RepairResult result;

  const size_t nrho = table.nrho();
  const size_t ntemp = table.ntemp();
  const size_t nye = table.nye();
  const size_t ncol = nrho * nye;

  for (const std::string &field : options.fields) {
    const double min_slope = min_slope_for(field, options);
    std::vector<double> &data = table.field(field);

    // Snapshot before ANY stage runs, so the final write-back below can
    // diff against the true original regardless of how many stages/rounds
    // touched a given index (repair.hpp's repair_table() doc comment,
    // "Final write-back").
    const std::vector<double> original_field = data;

    std::vector<SplineSafeStats> per_column_spline(ncol);

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
    ThreeDStats stats3d;
    if (options.spline_safe_3d) {
      stats3d = spline_safe_3d_field(data, nrho, ntemp, nye, min_slope, options);
    }

    // Final write-back: diff the FINAL data (after every stage above)
    // against the pre-repair original, in the fixed (field, kYe, irho, jT)
    // order RepairResult::entries documents -- independent of how either
    // stage above scheduled its OpenMP work.
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

    summary.rounds3d_used = stats3d.rounds_used;
    summary.points_diffused_3d = stats3d.points_diffused;
    summary.violations3d_remaining = stats3d.violations_remaining;
    summary.rounds3d_violation_history = std::move(stats3d.violation_history);

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
}

} // namespace eeos
