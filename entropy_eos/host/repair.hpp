// entropy_eos/host/repair.hpp
//
// Table repair: L2 isotonic regression (PAVA) followed by strict-minimum-
// slope enforcement, applied per (irho, kYe) column along T to the fields
// that the adapter's inner T-solve requires to be strictly monotone in T
// (see eos-adapter-F-to-U.md §8: sigma_T = ds/dT > 0 and e_T = deps/dT > 0
// after repair). Repairs act on the stored variables directly ("entropy",
// "logenergy") -- monotonicity of logenergy in T is equivalent to
// monotonicity of eps since log10 is monotone and energy_shift is a
// constant -- so no unit round trip is needed (see CODE.md "Repair
// harness").
//
// Host-only: STL containers, may throw. Structural problems -- a listed
// field missing from the table, or a non-finite value in a listed field --
// mean a broken file, not physics noise, and are reported by throwing
// rather than silently "fixed" (CODE.md).

#pragma once

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "entropy_eos/core/defs.hpp"
#include "entropy_eos/host/table.hpp"

namespace eeos {

struct RepairOptions {
  // Strict minimum increase per T grid step, imposed after PAVA. Placeholder
  // defaults; CODE.md marks tuning on real LS220 violations as open.
  double min_slope_entropy = 1e-8;    // kB/baryon per step (absolute)
  double min_slope_logenergy = 1e-10; // log10(erg/g) per step (absolute)

  // Repaired fields, in this order; this is also the order
  // RepairResult::entries and ::summaries are grouped by. Each name must be
  // "entropy" or "logenergy" -- the only two with a known minimum slope
  // above -- or repair_table() throws std::invalid_argument.
  std::vector<std::string> fields = {"entropy", "logenergy"};
};

// One value changed by repair_table(), identified by field and grid index.
struct RepairEntry {
  std::string field;
  size_t irho = 0, jT = 0, kYe = 0;
  double old_value = 0.0, new_value = 0.0;
};

struct RepairResult {
  // Every changed value, in a deterministic order: by field (in
  // RepairOptions::fields order), then kYe, then irho, then jT. This order
  // does not depend on how repair_table() schedules the per-column work
  // (e.g. under OpenMP).
  std::vector<RepairEntry> entries;

  // ok iff entries is empty, repaired otherwise. repair_table() never
  // returns Status::fatal: structural problems throw instead (see above).
  Status status = Status::ok;

  struct FieldSummary {
    std::string field;
    size_t modified = 0;
    double max_abs_change = 0.0, rms_change = 0.0;
  };
  std::vector<FieldSummary> summaries;

  // Human-readable summary (status, then one line per field).
  void print(std::ostream &os) const;
};

// Repairs `table` in place. For each field in options.fields, and for every
// (irho, kYe) column along T, this:
//   1. Applies L2 isotonic regression (PAVA, uniform weights) to make the
//      column non-decreasing.
//   2. Runs a strictification forward pass enforcing the field's minimum
//      slope: v[j] = max(v[j], v[j-1] + min_slope) for j = 1..n-1.
//   3. Writes the column back, recording a RepairEntry for every index whose
//      value changed (exact bitwise != against the original). Indices that
//      did not change are left bit-identical -- no arithmetic touches them.
//
// Columns are independent of each other and may be repaired in parallel
// (OpenMP), but entries are always concatenated in the fixed order described
// on RepairResult::entries, so the result does not depend on thread count.
// Repairing an already-repaired table is a no-op (RepairResult::entries
// empty, Status::ok, no bits changed): this follows from PAVA and
// strictification both being true no-ops on a column that already satisfies
// the minimum slope everywhere.
//
// Throws std::runtime_error if a listed field is missing from `table` or
// contains a non-finite value (checked for every listed field before any
// field is modified, so a throw leaves `table` untouched), and
// std::invalid_argument if a listed field name has no known minimum slope
// (only "entropy" and "logenergy" do).
RepairResult repair_table(RawTable &table, const RepairOptions &options = RepairOptions());

// The per-column algorithm, exposed for unit testing: L2 isotonic
// regression (pool-adjacent-violators, uniform weights) to make `col`
// non-decreasing, followed by the strictification forward pass col[j] =
// max(col[j], col[j-1] + min_slope) for j = 1..n-1.
//
// With min_slope == 0, the strictification step reduces to col[j] =
// max(col[j], col[j-1]), which cannot change the PAVA output since that
// output is already non-decreasing -- so repair_column(col, 0.0) isolates
// the pure PAVA result for testing.
void repair_column(std::vector<double> &col, double min_slope);

} // namespace eeos
