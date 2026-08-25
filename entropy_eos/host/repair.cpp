#include "entropy_eos/host/repair.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

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

      repair_column(column, min_slope);

      std::vector<RepairEntry> &entries = per_column[icol];
      for (size_t jT = 0; jT < ntemp; ++jT) {
        if (column[jT] != original[jT]) {
          entries.push_back(RepairEntry{field, irho, jT, kYe, original[jT], column[jT]});
        }
      }
      // Write back every value, not just the changed ones: entries that
      // repair_column left untouched keep the exact same bits, so this
      // cannot perturb them.
      for (size_t jT = 0; jT < ntemp; ++jT) {
        data[table.index(irho, jT, kYe)] = column[jT];
      }
    }

    RepairResult::FieldSummary summary;
    summary.field = field;
    double sum_sq = 0.0;
    for (size_t icol = 0; icol < ncol; ++icol) {
      for (const RepairEntry &e : per_column[icol]) {
        result.entries.push_back(e);
        ++summary.modified;
        const double abs_change = std::fabs(e.new_value - e.old_value);
        summary.max_abs_change = std::max(summary.max_abs_change, abs_change);
        sum_sq += abs_change * abs_change;
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
  }
}

} // namespace eeos
