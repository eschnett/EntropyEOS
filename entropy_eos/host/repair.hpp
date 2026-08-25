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
// M2c-prime "spline-safe" repair (eos-adapter-F-to-U.md §4): plain PAVA +
// strictification can leave a near-plateau immediately adjacent to a steep
// recovery -- data-monotone, but the *fitted* C^2 cubic B-spline (the same
// not-a-knot fit adapter_build.cpp uses) can still ring between nodes there,
// producing pockets where S' <= 0 even though every node-to-node secant
// slope is positive. Since the adapter differentiates the fitted spline, not
// the raw data, that ringing is a real sigma_u/e_u <= 0 hazard. The
// spline-safe loop (RepairOptions::spline_safe, on by default) audits the
// fitted 1D spline of each repaired column on a refined grid and, wherever
// it finds a violation, nudges the data with one small local diffusion step
// before re-repairing and re-auditing -- see repair_table()'s doc comment
// for the exact algorithm.
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

  // --- M2c-prime spline-safe smoothing (see the module comment above) -----

  // Run the audit-driven smoothing loop after the base PAVA + strictify
  // pass, per column. On by default: a strictly-monotone-in-data column can
  // still fit a spline that rings non-monotone between nodes (the
  // motivating case: a long near-plateau immediately followed by a steep
  // recovery), and the adapter differentiates the fitted spline.
  bool spline_safe = true;

  // Cap on smoothing rounds per column (see repair_table()'s doc comment,
  // step 1). A column still violating after this many rounds is recorded in
  // RepairResult::FieldSummary rather than looped on forever.
  int spline_rounds_max = 20;

  // Per-cell oversampling used to audit the fitted 1D spline: cell j (0 ..
  // n-2) is sampled at j + m/spline_refine for m = 0 .. spline_refine-1,
  // plus the column's last node.
  int spline_refine = 4;

  // Minimum acceptable S' at an audit sample, in per-index units (the fit
  // uses unit spacing x0=0, h=1 -- see repair_table() -- so this floor is
  // grid-independent; only its sign matters for the physical sigma_u/e_u >
  // 0 requirement, but a small positive value can be used to build in
  // margin).
  double spline_slope_floor = 0.0;

  // One Jacobi diffusion step's weight, applied to indices around an
  // offending cell (repair_table() step 1d): v_j <- v_j +
  // diffuse_alpha*(v_{j-1} - 2*v_j + v_{j+1}), using pre-step values on the
  // right-hand side.
  double diffuse_alpha = 0.25;

  // How many extra cells of padding, on each side of an offending cell, are
  // marked for the diffusion step (repair_table() step 1d): cell j marks
  // indices [j - diffuse_window, j + 1 + diffuse_window], clamped to the
  // column.
  int diffuse_window = 2;
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

    // --- M2c-prime spline-safe stats (all 0 if options.spline_safe was
    // false, or if no column ever needed a smoothing round) ---------------

    // Largest number of smoothing rounds any single column of this field
    // needed (0 if none needed any).
    int spline_rounds_used_max = 0;
    // How many columns needed at least one smoothing round.
    size_t spline_columns_smoothed = 0;
    // How many columns still had a spline audit violation after
    // options.spline_rounds_max rounds (see repair_table()'s doc comment,
    // step 2).
    size_t spline_columns_still_violating = 0;
    // Histogram of rounds_used across this field's columns: index i (0 ..
    // options.spline_rounds_max) counts the columns that used exactly i
    // smoothing rounds (index 0 is columns that needed none, including
    // every column when options.spline_safe is false). Sized
    // options.spline_rounds_max+1 whenever spline_safe ran, empty
    // otherwise.
    std::vector<size_t> spline_rounds_histogram;
  };
  std::vector<FieldSummary> summaries;

  // Human-readable summary (status, then one line per field).
  void print(std::ostream &os) const;
};

// Repairs `table` in place. For each field in options.fields, and for every
// (irho, kYe) column along T, this:
//   0. Base pass: applies L2 isotonic regression (PAVA, uniform weights) to
//      make the column non-decreasing, then a strictification forward pass
//      enforcing the field's minimum slope: v[j] = max(v[j], v[j-1] +
//      min_slope) for j = 1..n-1 (repair_column(), unchanged from earlier
//      milestones).
//   1. If options.spline_safe (default on), a spline-safe smoothing loop,
//      up to options.spline_rounds_max rounds:
//        a. Fit the column with fit_bspline_1d() at unit spacing (x0=0,
//           h=1 -- only the sign of the derivative matters, and per-index
//           units make spline_slope_floor grid-independent).
//        b. Audit: sample S'(j + m/spline_refine) for j = 0..n-2, m =
//           0..spline_refine-1, plus the last node, via bspline_eval1().
//           Collect the set of cells (0..n-2) containing any sample with S'
//           <= options.spline_slope_floor.
//        c. If that set is empty, the column is done.
//        d. Otherwise, for every offending cell j, mark indices [j -
//           diffuse_window, j + 1 + diffuse_window] (clamped to the
//           column). Apply one Jacobi diffusion step to the union of marked
//           *interior* indices (column endpoints never move): v_j <- v_j +
//           diffuse_alpha*(v_{j-1} - 2*v_j + v_{j+1}), using the pre-step
//           values on the right-hand side (so the step does not depend on
//           the order marked indices are visited in). Then re-run
//           repair_column() (PAVA + strictify) on the whole column -- the
//           diffusion step can nudge the data out of monotonicity, and this
//           restores it -- and go back to (a).
//      If the cap is reached with violations remaining, the column is
//      recorded (FieldSummary::spline_columns_still_violating) rather than
//      repaired further; this never throws.
//   2. Writes the column back, recording a RepairEntry for every index
//      whose *final* value (after both the base pass and, if run, the
//      spline-safe loop) differs from the original input value (exact
//      bitwise != -- this is always computed against the original, not
//      incrementally against the base-pass output, so the log stays
//      meaningful across smoothing rounds). Indices that never changed are
//      left bit-identical -- no arithmetic touches them.
//
// Columns are independent of each other and may be repaired in parallel
// (OpenMP), but entries are always concatenated in the fixed order described
// on RepairResult::entries, so the result does not depend on thread count;
// the spline-safe loop's audit (step 1a-b) is itself deterministic (same
// fit, same samples) for a given column, so this holds with spline_safe on
// or off.
//
// Repairing an already-repaired table is a no-op (RepairResult::entries
// empty, Status::ok, no bits changed): with spline_safe off, this follows
// from PAVA and strictification both being true no-ops on a column that
// already satisfies the minimum slope everywhere; with spline_safe on, it
// additionally requires the spline audit (step 1b) to find no offending
// cell on its first pass over already-repaired data -- guaranteed by the
// audit's own determinism, since a first repair_table() run only stops
// smoothing a column once that same audit reports it clean.
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
