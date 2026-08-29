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
// M2d-1 "spline-safe-3d" repair (eos-adapter-F-to-U.md §4, CODE.md open
// decision 4): the per-column loop above only ever looks at S'(u) along a
// single (irho, kYe) column, so it provably cannot see or fix a violation
// that only shows up in the *tensor-product* 3D fit adapter_build.cpp
// actually evaluates -- e.g. two neighboring columns that are each
// individually monotone but differ enough in local steepness that the
// smooth blend the spline performs *between* them dips non-monotone in u at
// an off-node (rho, Ye) position no per-column audit ever samples. The
// spline-safe-3d stage (RepairOptions::spline_safe_3d, on by default) runs
// after the per-column stage, per field: it fits the *whole field* as one
// tensor-product not-a-knot cubic B-spline (the same fit_bspline_3d()
// adapter_build.cpp uses), audits fu (= S_u) on a refined 3D grid, and
// wherever it finds a violation, nudges the data with one small local 3D
// diffusion step, then re-runs the per-column pipeline on every column the
// nudge touched (to restore the 1D monotonicity/spline-safety the 3D
// diffusion may have perturbed) -- see repair_table()'s doc comment for the
// exact algorithm.
//
// M3f "causal-cap" repair (eos-causality-repair.md): the two stages above
// both enforce *monotonicity* of the fitted splines; neither can see the
// other hard requirement of eos-adapter-F-to-U.md S8, causality
// (0 < c_s^2 < 1) of the potential U the adapter builds out of those same
// splines. On the real tables c_s^2 >= 1 over a systematic high-rho corner
// (LS220-2009: 4.2% of nodes, a contiguous suffix in rho at every (T, Ye);
// SRO: 4.3%) -- genuine table physics, confirmed by those tables' own
// stored cs2 columns, but fatal to the con2prim Newton's conditioning
// (its diagonal entry is z_w = z(1-c_s^2)tanh w). The causal-cap stage
// (RepairOptions::causal_cap, on by default) runs last, audits c_s^2 of the
// fitted "entropy"/"logenergy" splines on a refined grid via the analytic
// chain rule, and where the audit finds edge-anchored violation runs
// projects "logenergy" onto the causal Lipschitz envelope along adiabats --
// see repair_table()'s doc comment (causal-cap stage) for the algorithm and
// eos-causality-repair.md S3 for the mathematics. Only "logenergy" is ever
// edited; "entropy" never gains a RepairEntry from this stage.
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

  // --- M2d-1 spline-safe-3d smoothing (see the module comment above) ------

  // Run the tensor-product 3D audit-driven diffusion stage, per field,
  // after the per-column stage above. On by default: fixes cross-column
  // x/y ringing that no per-column repair can reach (CODE.md open decision
  // 4). A field whose grid has fewer than 4 points along rho, T, or Ye is
  // silently skipped (fit_bspline_3d's own minimum -- a tensor-product fit
  // is not meaningfully defined below that), matching the per-column
  // stage's own n < 4 no-op.
  bool spline_safe_3d = true;

  // Cap on rounds of the main 3D loop (repair_table()'s doc comment, 3D
  // stage step a-d) per field. Unlike spline_rounds_max (per column), this
  // is per *field*: the whole tensor-product fit/audit/diffuse cycle is one
  // round.
  int rounds3d_max = 10;

  // Per-cell oversampling used by the main 3D loop's audit, along u (T) and
  // along x/y (rho/Ye) respectively -- see repair_table()'s doc comment for
  // the exact sample grid. The final verification pass always audits at
  // (4,4,4) regardless of these (see below); the main loop's cheaper
  // resolution exists purely so most of the convergence work happens at a
  // fraction of the final pass's cost (repair.hpp's module comment;
  // performance guardrail in the M2d-1 work order).
  int refine3d_u = 4;
  int refine3d_xy = 2;

  // diffuse_alpha and diffuse_window (above) are reused by the 3D stage's
  // Jacobi diffusion step and box-marking, exactly as documented for the
  // per-column stage but applied in 3D (repair_table()'s doc comment).

  // --- M3f causal-cap stage (eos-causality-repair.md S5) ------------------

  // Run the causal-cap stage after both monotonicity stages above. On by
  // default (decided 2026-08-27, eos-causality-repair.md S5): eos_repair's
  // contract is "make the table satisfy the hard requirements of
  // eos-adapter-F-to-U.md S8", and 0 < c_s^2 < 1 is on that list; like the
  // monotonicity repair it is a logged physics edit, and it acts only where
  // the table is already unphysical by its own stored cs2 column.
  //
  // Silently skipped (a no-op, RepairResult::CausalCapSummary::ran = false)
  // unless BOTH "entropy" and "logenergy" are in `fields` (c_s^2 is a joint
  // property of the two splines), the table carries an "energy_shift"
  // attribute, and every axis has >= 4 points and is non-degenerate.
  bool causal_cap = true;

  // Audit threshold: a refined sample whose c_s^2 is >= this is a violation.
  // 1.0 is the physical causality bound.
  double cs2_max = 1.0;

  // Target slope c_bar of the Lipschitz envelope the projection imposes on
  // ln h along adiabats (eos-causality-repair.md S3). Strictly below
  // cs2_max on purpose: the hysteresis absorbs the ringing a refit puts
  // back near the anchor seam, and 1 - c_s^2 is exactly the con2prim
  // Newton's diagonal conditioning headroom. 0 < cs2_cap < cs2_max is a
  // precondition, not a preference -- repair_table() throws
  // std::invalid_argument otherwise (see its doc comment), since without the
  // gap the loop has no mechanism to converge.
  double cs2_cap = 0.99;

  // Absolute ceiling on the stage's audit/project/re-repair loop -- a pure
  // runaway backstop, NOT the working budget (M3j semantic change; it was a
  // hard budget of 8 through M3i). The loop normally ends on its own: the
  // audit comes back clean, nothing is left in scope, a round writes no
  // node, or -- the usual case on a real table -- two consecutive rounds
  // fail to improve on the best state seen (see repair_table()'s doc
  // comment, causal-cap step 9). This bound exists only so a pathological
  // table cannot spin forever, so it is set well above any measured need:
  // SFHo, the deepest real table measured, converges in ~30 rounds
  // (CODE.md "DD2 / SFHo empirical findings"); LS220/SRO/DD2 use <= 10.
  // Lowering it below the natural stopping point re-introduces exactly the
  // M3i behavior it replaced -- a converging loop cut off mid-descent,
  // whose remaining work only a second eos_repair run could finish.
  int causal_rounds_max = 64;

  // How many rho *cells* a node's adiabat trace may descend looking for a
  // causal anchor before the node is given up on (reported via
  // CausalCapSummary::trace_giveups, never edited). Measured anchor depths:
  // <= 11 cells (LS220-2009), <= 18 (SRO).
  int trace_depth_max = 64;

  // Consecutive causal refined steps required before the trace anchors
  // (hysteresis: anchoring on the very first causal step would put the
  // anchor exactly where the refit is most likely to ring back above the
  // cap).
  int anchor_pad = 2;

  // The stage's refined audit grid reuses refine3d_xy / refine3d_u above:
  // the main loop audits at (refine3d_xy, refine3d_u, refine3d_xy) and the
  // final verification always at (4,4,4), mirroring the 3D stage. The
  // adiabat trace steps by one refined rho step, hrho/refine3d_xy.
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

    // --- M2d-1 spline-safe-3d stats (all 0 / empty if options.spline_safe_3d
    // was false, or if the field's grid was too small to fit -- see
    // RepairOptions::spline_safe_3d) ----------------------------------------

    // Number of 3D diffusion rounds *kept* in this field's final state:
    // every main-loop round, and every verification extra round (up to 3),
    // that was still part of the best (lowest (4,4,4)-equivalent
    // violation-count) state repair_table() settled on -- see
    // repair_table()'s doc comment (steps 2-4) and spline_safe_3d_field()'s
    // doc comment in repair.cpp for why "kept" and "attempted" can differ
    // (a round whose fix made things worse is discarded, not counted here).
    // A round whose audit finds nothing (including the final verification's
    // own first audit, when it is already clean) does *not* count, so a
    // field with no 3D-stage defect at all reports 0, same as a field with
    // the stage turned off; a field where the backstop (step 4) reverted
    // everything back to the pre-3D-stage state also reports 0.
    int rounds3d_used = 0;

    // How many DISTINCT data points are different from the pre-3D-stage
    // state because of a 3D Jacobi diffusion step, in this field's final
    // (kept) state -- counted once each no matter how many rounds touched
    // them, and 0 whenever the backstop (step 4) reverted everything.
    size_t points_diffused_3d = 0;

    // Violation *sample* count of this field's final state at (4,4,4)
    // resolution when repair_table() returned -- either the final
    // verification's own last audit (repair_table()'s doc comment, step 3),
    // or, if the backstop (step 4) reverted to the pre-3D-stage state, that
    // state's own (4,4,4) count. 0 means the field is fully
    // tensor-fit-monotone in u at (4,4,4) resolution.
    size_t violations3d_remaining = 0;

    // Every 3D-stage audit's violation *sample* count, in chronological
    // order: the main loop's rounds (at refine (refine3d_xy, refine3d_u,
    // refine3d_xy)), one entry per round including the one that finds zero
    // and ends the loop, followed by the final verification's audits (at
    // (4,4,4)), again one entry per audit including its last (clean or
    // not). Empty iff options.spline_safe_3d was false or the field's grid
    // was too small to fit. repair_table() itself prints nothing (CODE.md
    // "Environment": library code stays quiet); tools/eos_repair prints
    // these as per-round progress lines.
    std::vector<size_t> rounds3d_violation_history;
  };
  std::vector<FieldSummary> summaries;

  // --- M3f causal-cap stage (eos-causality-repair.md) ----------------------
  //
  // One summary for the whole stage, not one per field: c_s^2 is a joint
  // property of the "entropy" and "logenergy" splines, and the stage runs
  // once over both (editing only "logenergy").
  struct CausalCapSummary {
    // False (and everything below at its default) whenever the stage did
    // not run: options.causal_cap off, a missing field/attribute, or a grid
    // too small to fit (see RepairOptions::causal_cap).
    bool ran = false;

    // True iff the stage's lexicographic backstop (repair_table()'s doc
    // comment, causal-cap step 6) rejected the projected state and reverted
    // "logenergy" to exactly its pre-stage bytes -- either because the
    // (4,4,4) L_u monotonicity count got worse (never trade the T-solve's
    // hard requirement for causality) or because the (4,4,4) c_s^2 count
    // did. When true, nodes_capped and rounds_used are 0 and
    // violations_after is the pre-stage count.
    bool reverted = false;

    // Rounds of the main audit/project/re-repair loop whose effect is part
    // of the kept state (a round whose projection made the count worse is
    // discarded and not counted, exactly like FieldSummary::rounds3d_used).
    int rounds_used = 0;

    // DISTINCT "logenergy" nodes whose value differs from the pre-stage
    // state in the kept state (counted once each, no matter how many rounds
    // touched them; 0 when reverted). "entropy" is never edited by this
    // stage, so it contributes nothing here or to RepairResult::entries.
    size_t nodes_capped = 0;

    // Nodes whose adiabat trace descended options.trace_depth_max cells (or
    // ran off the low-rho edge of the table) without finding a causal
    // anchor: reported, never edited (eos-causality-repair.md S4 step 3).
    size_t trace_giveups = 0;

    // c_s^2 violation *sample* counts at (4,4,4) of the pre-stage state and
    // of the state repair_table() actually returned. violations_after ==
    // violations_before exactly when the stage reverted (or changed
    // nothing).
    size_t violations_before = 0;
    size_t violations_after = 0;

    // Diagnostics of the returned state's own (4,4,4) audit, all
    // report-only (eos-causality-repair.md S2, S4 step 2, S6):
    //   interior_untouched   violating samples in runs that do NOT reach the
    //                        rho_max edge -- the sigma_T-pocket spikes and
    //                        SRO's Ye = 0.645 sliver. Out of scope by
    //                        design: capping eps cannot fix a defect that
    //                        lives in sigma's flatness.
    //   cs2_nonpositive      samples with c_s^2 <= 0 (fit noise in near-flat
    //                        regions, and the same sigma pockets). A
    //                        different defect class; z_w stays positive
    //                        there, so con2prim conditioning is unharmed.
    //   cs2_indeterminate    samples where the chain rule has no answer at
    //                        all (sigma_u <= 0, h <= 0, or a non-finite
    //                        intermediate); neither counted as a violation
    //                        nor edited.
    //   cs2_near_cap         samples with c_s^2 >= options.cs2_cap, i.e. how
    //                        much of the hysteresis band is in use.
    //   cs2_max_seen         the largest finite c_s^2 the audit saw.
    size_t interior_untouched = 0;
    size_t cs2_nonpositive = 0;
    size_t cs2_indeterminate = 0;
    size_t cs2_near_cap = 0;
    double cs2_max_seen = 0.0;

    // (4,4,4) fu-monotonicity violation counts the backstop compares, for
    // "entropy" (which this stage never edits, so the value is the same
    // before and after and is reported once) and for "logenergy" before and
    // after the stage.
    size_t mono_entropy = 0;
    size_t mono_logenergy_before = 0;
    size_t mono_logenergy_after = 0;

    // The same two (4,4,4) counts for the state the stage actually produced,
    // whether or not the backstop kept it. Identical to violations_after /
    // mono_logenergy_after unless `reverted` is set, in which case these are
    // the numbers that caused the rejection -- the only place the reason is
    // visible, since the "after" fields describe the state that was returned
    // (i.e. the pre-stage one).
    size_t projected_violations = 0;
    size_t projected_mono_logenergy = 0;

    // Every main-loop round's c_s^2 violation sample count, in chronological
    // order (at refine (refine3d_xy, refine3d_u, refine3d_xy)), one entry
    // per round including the one that finds zero and ends the loop, then
    // the final verification's (4,4,4) count as the last entry. Empty iff
    // the stage did not run. repair_table() itself prints nothing (CODE.md
    // "Environment"); tools/eos_repair prints these as per-round progress
    // lines, exactly as it does for the 3D stage.
    std::vector<size_t> rounds_violation_history;
  };
  CausalCapSummary causal_cap;

  // Human-readable summary (status, then one line per field, then the
  // causal-cap stage's block if it ran).
  void print(std::ostream &os) const;
};

// Repairs `table` in place. For each field in options.fields:
//
// Per-column stage, applied independently to every (irho, kYe) column along
// T (columns are independent -> OpenMP):
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
//
// Field-wide 3D stage (M2d-1), applied once per field after every column has
// gone through the per-column stage above, if options.spline_safe_3d
// (default on) and every axis has >= 4 points (else a silent no-op for that
// field):
//   2. Main loop, up to options.rounds3d_max rounds:
//        a. Fit the *whole field* with fit_bspline_3d() at unit spacing
//           (x0=u0=y0=0, hx=hu=hy=1 -- again only derivative signs matter).
//        b. Audit fu (= S_u) of bspline_eval3() on the grid {every rho node
//           plus (refine3d_xy-1) interior points per rho cell} x {every T
//           node plus (refine3d_u-1) interior points per T cell} x {same as
//           rho, for Ye}, i.e. per axis the union of every data node and
//           (refine-1) points interior to each cell. Collect every sample
//           with fu <= options.spline_slope_floor (OpenMP over the rho
//           axis; the violation count and the marked set below are sums/an
//           idempotent OR over independent samples, so both are the same
//           regardless of thread count or scheduling).
//        c. If no sample violated, the field is done with this stage.
//        d. Otherwise, for every violating sample's owning cell (i,j,k) (rho,
//           T, Ye cell indices), mark the data-index box [i-w, i+1+w] x
//           [j-w, j+1+w] x [k-w, k+1+w] (w = options.diffuse_window, clamped
//           to the grid). Apply one 3D Jacobi diffusion step to every marked
//           point that is interior in *all three* axes (points on any
//           axis's boundary never move, even if marked): v_ijk <- v_ijk +
//           diffuse_alpha*(sum of the 6 axis-neighbors - 6*v_ijk)/6, using
//           the pre-step snapshot on the right-hand side. Then re-run the
//           per-column stage (steps 0-1 above) on every (irho, kYe) column
//           that intersects a marked box (i.e. has any marked jT) -- the 3D
//           diffusion can perturb a column's own 1D monotonicity or
//           spline-safety, and this restores it. Go back to (a).
//      This loop is *not* assumed to improve monotonically round over round
//      (see spline_safe_3d_field()'s doc comment in repair.cpp for the
//      empirical reason: fields with a very thin natural safety margin --
//      e.g. "logenergy" where eps << energy_shift makes S_u tiny almost
//      everywhere at low T -- can have a round's diffusion tip a
//      neighboring, previously-fine sample negative, so the violation count
//      can rise before it falls), so it tracks the best (lowest-violation)
//      state seen and gives up early -- reverting to that best state --
//      after 4 consecutive rounds fail to beat it. Reaching options.
//      rounds3d_max rounds without giving up early behaves the same way
//      (revert to the best state seen); this step never throws.
//   3. Final verification, run unconditionally (even if step 2 exited via
//      (c) on round 1): one audit pass exactly like 2b but always at refine
//      (4,4,4), regardless of options.refine3d_u/refine3d_xy. If it finds
//      nothing, the field's 3D stage is done. Otherwise, up to 3 more
//      rounds of "fix like 2d, then re-audit at (4,4,4)", stopping (and
//      reverting to the best of these (4,4,4)-audited states) the instant a
//      round fails to improve on it -- there is no rounds3d_max-sized
//      budget left here to recover from a bad round, unlike step 2.
//   4. Backstop, only when step 3 ends with a nonzero count: one more
//      (4,4,4) audit, of the state `data` held *before* this whole 3D stage
//      started (i.e. after the per-column stage but before step 2's first
//      round) -- never returning (or leaving in `data`) a field worse off,
//      at this authoritative resolution, than skipping the 3D stage
//      entirely. If step 3's result is worse, every field byte and
//      FieldSummary::rounds3d_used/points_diffused_3d revert to exactly
//      that pre-3D-stage state (0 rounds, 0 points diffused).
//      FieldSummary::violations3d_remaining is this step's final count.
//   Every 3D audit's violation count actually run (main loop rounds, then
//   the verification's own audits -- *not* including the backstop's own
//   audit in step 4, which is bookkeeping, not a "round") is appended, in
//   order, to FieldSummary::rounds3d_violation_history.
//
// Causal-cap stage (M3f, eos-causality-repair.md), run ONCE after every
// field has been through both stages above, if options.causal_cap (default
// on) and the preconditions on RepairOptions::causal_cap hold (both fields
// listed, "energy_shift" present, every axis >= 4 points). It edits only
// "logenergy"; "entropy" is left bit-identical, so node adiabat labels
// s = sigma(node) are stable across rounds and the T-solve's sigma_u > 0
// guarantee is untouched.
//   5. Audit. Fit both fields with fit_bspline_3d() at the table's own
//      log10 axes (x = log10 rho, u = log10 T, y = Ye; uniform spacing
//      assumed, as everywhere in M2) and sample
//        c_s^2 = (eps_x + eps_xx) / h,   h = 1 + eps + eps_x
//      on the same refined grid step 2b uses, at (refine3d_xy, refine3d_u,
//      refine3d_xy) in the main loop and at (4,4,4) for verification. Here
//      x = ln rho and the derivatives are taken *along the adiabat* through
//      the sample, via the implicit-function chain rule on the sigma spline
//      (eos-adapter-F-to-U.md S3.1, in raw table variables): with
//      U'= -sigma_x/sigma_u and U''= -(sigma_xx + 2 sigma_xu U' + sigma_uu
//      U'^2)/sigma_u, eps and its two x-derivatives follow from the
//      "logenergy" spline analytically. Only c^2 and energy_shift enter --
//      c_s^2 is invariant under the adapter's kappa rescaling and under the
//      table's m_B convention, so auditing the raw-variable fit is exactly
//      auditing the production adapter's interior. A sample violates iff
//      c_s^2 >= options.cs2_max; samples with c_s^2 <= 0, samples the chain
//      rule cannot resolve, and samples merely above options.cs2_cap are
//      counted for reporting only.
//   6. Scope. Group the violating samples of each refined (u, y) row into
//      maximal runs along x. Only runs that reach the x_hi (rho_max) edge
//      are treated; interior runs are counted into
//      CausalCapSummary::interior_untouched and never edited (their defect
//      lives in sigma_T's flatness, not in eps's stiffness, and chasing
//      them is what burns round budgets -- the M2d-1 lesson). A treated run
//      starting at refined index a marks, for the 2x2 data-node corners
//      (jT, kYe) of its owning (u, y) cell, every node with irho >=
//      ceil(a / refine3d_xy).
//   7. Project, per marked node (i, j, k), independently (-> OpenMP; each
//      trace reads only the pre-round splines, so the result does not
//      depend on scheduling):
//        a. Trace the node's adiabat s = entropy(i,j,k) downward in x in
//           refined steps hrho/refine3d_xy, solving sigma(x', u, y) = s for
//           u at each step (safeguarded Newton on the sigma spline, warm
//           started from the previous step). Where the adiabat leaves the
//           box through u_min, the trace follows the u_min edge (the fully
//           degenerate regime, where T-dependence is negligible --
//           eos-causality-repair.md S6).
//        b. Anchor where c_s^2 <= options.cs2_cap has held for
//           options.anchor_pad consecutive steps. If that has not happened
//           within options.trace_depth_max cells, or the trace reaches the
//           low-rho edge of the table first, give up on the node
//           (CausalCapSummary::trace_giveups) and leave it bit-identical.
//        c. March back up to the node imposing the one-sided Lipschitz
//           envelope h_env(x+d) = min(h_orig(x+d), h_env(x)*exp(cs2_cap*d))
//           and integrating the energy-consistency ODE
//           d(eps)/dx = h - 1 - eps with h = h_env, from eps_orig(anchor),
//           by exact integrating-factor steps (exact whenever h is
//           exponential on a step, which is how each step is modelled).
//           Implementation detail: the ODE is integrated for the *deficit*
//           D = eps_orig - eps_env, which obeys the same linear ODE with
//           source h_orig - h_env and D(anchor) = 0, so D stays exactly 0
//           on stretches where the envelope does not bind and the node's
//           own exact stored eps carries the value (algebraically identical
//           to integrating eps itself, but with no quadrature drift on the
//           already-causal part -- which is what makes the stage idempotent
//           and keeps causal nodes bit-identical).
//        d. Write log10(eps_env*c^2 + energy_shift) into "logenergy" iff
//           the envelope actually bound somewhere on the traced stretch AND
//           eps_env < eps_orig AND the result is finite and above the
//           storage shift. Otherwise the node keeps its bits exactly.
//   8. Restore monotonicity: re-run the per-column pipeline (steps 0-1) on
//      every (irho, kYe) column the projection touched, exactly as step 2d
//      does for the 3D stage.
//   9. Loop: re-audit and repeat *while the stage keeps improving*, tracking
//      the best (lowest c_s^2 violation count) state seen and giving up --
//      reverting to that best state -- after 2 consecutive rounds fail to
//      beat it, the same rule step 2 uses with a shorter patience (M3j; the
//      2-vs-4 choice is measured, see repair.cpp's causal_cap_stage()).
//      options.causal_rounds_max caps the round count as a pure runaway
//      backstop, not as the working budget: a table that hits it was still
//      converging, and its remaining work is exactly what a second
//      repair_table() call would pick up (this is what M3j fixed -- see
//      CODE.md "DD2 / SFHo empirical findings").
//  10. Verification + lexicographic backstop: one final (4,4,4) audit of
//      c_s^2, plus (4,4,4) fu-monotonicity counts of both fields. The kept
//      state must satisfy, in order, (a) monotonicity counts no worse than
//      the pre-stage state's -- never trade the T-solve's hard requirement
//      for causality -- and (b) a c_s^2 count no worse than the pre-stage
//      state's. If either fails, "logenergy" reverts to exactly its
//      pre-stage bytes and CausalCapSummary::reverted is set.
//
// Final write-back, per field: a RepairEntry for every index whose value
// after *all* stages above (per-column, then 3D if it ran, then causal-cap
// if it ran) differs from the original input value (exact bitwise != -- always computed against the
// pre-repair original, not incrementally against an intermediate stage's
// output, so the log stays meaningful across every round of every stage),
// in the fixed order documented on RepairResult::entries. Indices that never
// changed are left bit-identical -- no arithmetic touches them.
//
// The per-column stage is embarrassingly parallel (OpenMP over columns) and
// the 3D stage's own audit/diffusion/re-repair are each parallelized
// internally (over the rho axis for the audit, over affected columns for
// the re-repair); every stage's *result* is independent of thread count and
// scheduling (see 2b's parenthetical and the per-column stage's existing
// determinism argument), so the final write-back above is deterministic
// regardless of how many threads ran.
//
// Repairing an already-repaired table is a no-op (RepairResult::entries
// empty, Status::ok, no bits changed) *whenever the previous repair_table()
// run left every processed field genuinely clean*: with spline_safe (and
// spline_safe_3d) off, this follows from PAVA and strictification both
// being true no-ops on a column that already satisfies the minimum slope
// everywhere; with spline_safe on, it additionally requires that stage's
// own audit to find no violation on its first pass over already-repaired
// data -- guaranteed by the audit's own determinism, since a first
// repair_table() run only stops that stage once that same audit reports it
// clean. With spline_safe_3d on and a field left with
// FieldSummary::violations3d_remaining == 0, the same argument applies
// whenever options.refine3d_xy divides the verification's fixed refine of 4
// (true for the documented default refine3d_xy=2): every main-loop audit
// sample is then also one of the final verification's samples, so a field
// left clean by the verification pass is guaranteed clean on the next run's
// very first (cheaper) main-loop audit too. A field left with
// violations3d_remaining > 0, however, is *not* guaranteed idempotent: step
// 2's doc comment above (and spline_safe_3d_field()'s doc comment in
// repair.cpp) describe an empirically-observed defect shape (a wide,
// sharp-edged cross-column discontinuity) that the tensor-product stage
// cannot fully resolve within its round budget -- repairing such a table a
// second time can still find (and repair) more of the same residual, since
// each run explores independently from wherever the data currently stands.
// This is expected, not a bug: it mirrors the parent design's own framing
// (eos-adapter-F-to-U.md S4, CODE.md "M2 empirical findings") of stubborn
// residual pathologies as something to measure and report, not something
// this stage promises to eliminate in one pass.
//
// The causal-cap stage is idempotent under the same conditions plus its own
// write condition: whenever a first run left CausalCapSummary::
// violations_after == 0, the next run's very first (cheaper) main-loop
// audit is guaranteed clean too (refine3d_xy divides the verification's
// fixed refine of 4 for the documented default refine3d_xy = 2, so every
// main-loop sample is also a verification sample), so the loop exits before
// projecting anything. Even where the audit does find something, a node
// whose traced stretch is already causal is left bit-identical: the deficit
// formulation of step 7c makes the projection an exact no-op there rather
// than a roundoff-level rewrite.
//
// Since M3j this stage is idempotent in the *general* case too -- a residual
// it leaves behind does not make a second run find more work in it (the
// stages before it still carry their own conditions, above) -- and the
// argument is worth spelling out, because it is exactly what the old fixed
// round budget broke. Every round's projection is a deterministic function
// of the current data alone, so the loop's trajectory from a given state is
// fixed.
// Step 9 stops on a property OF THE DATA (clean, nothing in scope, nothing
// written, or two consecutive rounds that fail to beat the best state seen)
// and returns the best state; a second run therefore starts at that best
// state, replays the very same rounds that followed it the first time --
// each of which was, by construction, no better -- and reverts to it again,
// bit for bit, reporting zero entries. A *budget* cutoff has no such
// property: it stops on how many rounds this call has already spent, which
// a fresh call resets, so a run cut off mid-descent leaves work a second run
// picks up (measured on SFHo before M3j: four eos_repair runs to converge,
// CODE.md "DD2 / SFHo empirical findings"). The one case where that can
// still happen is a run that hits options.causal_rounds_max, which is why
// that bound is set as a runaway backstop far above any measured need
// rather than as a working budget.
//
// Throws std::runtime_error if a listed field is missing from `table` or
// contains a non-finite value (checked for every listed field before any
// field is modified, so a throw leaves `table` untouched), and
// std::invalid_argument if a listed field name has no known minimum slope
// (only "entropy" and "logenergy" do) or if options.causal_cap is on without
// 0 < options.cs2_cap < options.cs2_max (the cap must be a physical sound
// speed strictly below the audit threshold, or the hysteresis that converges
// the refit ringing does not exist). Both are checked before any field is
// modified.
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

// Result of causal_envelope() below.
struct CausalEnvelope {
  // The projected dimensionless eps at the node. Bitwise equal to the
  // caller's eps_node whenever `bound` is false.
  double eps_node = 0.0;
  // Whether the Lipschitz envelope actually bound anywhere on the stretch
  // (i.e. whether the original h exceeded the cap's growth on some step).
  // The causal-cap stage writes a node only when this is true.
  bool bound = false;
  // False if a step was ill-posed (a non-positive or non-finite h);
  // eps_node is then meaningless and the stage gives up on the node.
  bool ok = false;
};

// The causal-cap stage's edit primitive on one already-traced stretch of an
// adiabat, exposed for unit testing (eos-causality-repair.md S3).
//
// `h` holds the ORIGINAL specific enthalpy h = 1 + eps + deps/dx|_s sampled
// along the adiabat at uniform steps `delta` > 0 in x = ln rho, ordered
// DOWNWARD from the node: h[0] is the node itself, h.back() the causal
// anchor. `eps_node` is the node's own (exact, stored) dimensionless eps.
//
// The one-sided Lipschitz envelope h_env(x+delta) = min(h_orig(x+delta),
// h_env(x) e^{cs2_cap*delta}) is imposed marching up from the anchor, and
// the energy-consistency ODE deps/dx = h - 1 - eps is integrated alongside
// it -- in the deficit form D = eps_orig - eps_env, D(anchor) = 0, dD/dx =
// (h_orig - h_env) - D -- with exact integrating-factor steps that treat h
// as exponential on each step. On a stretch where the envelope binds
// throughout with a purely exponential h_orig, the result is *exactly* the
// closed form of eos-causality-repair.md S3,
//
//   eps(x) = (eps_a + 1) e^{-dx} - 1 + h_a (e^{cs2_cap*dx} - e^{-dx})
//                                          / (1 + cs2_cap),
//
// to roundoff (the per-step recursion telescopes algebraically). On a
// stretch where it never binds the result is eps_node unchanged, bitwise.
//
// A stretch of length 0 or 1 (nothing above the anchor) returns eps_node
// with bound == false. Throws nothing.
CausalEnvelope causal_envelope(const std::vector<double> &h, double eps_node, double cs2_cap,
                                double delta);

} // namespace eeos
