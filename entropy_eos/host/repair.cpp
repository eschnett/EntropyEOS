#include "entropy_eos/host/repair.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

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

    // Per-column entries, indexed by icol = kYe*nrho + irho so that
    // concatenating in icol order visits kYe outermost and irho next --
    // the order required of RepairResult::entries -- regardless of how an
    // OpenMP schedule interleaves the parallel loop below.
    std::vector<std::vector<RepairEntry>> per_column(ncol);
    std::vector<SplineSafeStats> per_column_spline(ncol);

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
      const std::vector<double> original = column;

      // Step 0: base pass (PAVA + strictify).
      repair_column(column, min_slope);

      // Step 1: spline-safe smoothing loop (repair.hpp doc comment).
      if (options.spline_safe) {
        per_column_spline[icol] = spline_safe_column(column, min_slope, options);
      }

      // Step 2: diff the FINAL column against the ORIGINAL input, not
      // incrementally against the step-0 output, so the log stays
      // meaningful across smoothing rounds (repair.hpp step 2).
      std::vector<RepairEntry> &entries = per_column[icol];
      for (size_t jT = 0; jT < ntemp; ++jT) {
        if (column[jT] != original[jT]) {
          entries.push_back(RepairEntry{field, irho, jT, kYe, original[jT], column[jT]});
        }
      }
      // Write back every value, not just the changed ones: entries that
      // neither pass touched keep the exact same bits, so this cannot
      // perturb them.
      for (size_t jT = 0; jT < ntemp; ++jT) {
        data[table.index(irho, jT, kYe)] = column[jT];
      }
    }

    RepairResult::FieldSummary summary;
    summary.field = field;
    double sum_sq = 0.0;
    if (options.spline_safe) {
      summary.spline_rounds_histogram.assign(static_cast<size_t>(options.spline_rounds_max) + 1, 0);
    }
    for (size_t icol = 0; icol < ncol; ++icol) {
      for (const RepairEntry &e : per_column[icol]) {
        result.entries.push_back(e);
        ++summary.modified;
        const double abs_change = std::fabs(e.new_value - e.old_value);
        summary.max_abs_change = std::max(summary.max_abs_change, abs_change);
        sum_sq += abs_change * abs_change;
      }
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
    summary.rms_change =
        summary.modified > 0 ? std::sqrt(sum_sq / static_cast<double>(summary.modified)) : 0.0;
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
  }
}

} // namespace eeos
