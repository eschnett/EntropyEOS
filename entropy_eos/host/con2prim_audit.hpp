// entropy_eos/host/con2prim_audit.hpp
//
// M3b con2prim-level round-trip audit harness: `check_con2prim()` runs
// random-state prim2con -> con2prim round trips per con2prim-entropy-
// rapidity.md deliverable 2 ("round trip con2prim . prim2con = id to near
// machine precision over table samples x w in [0,6] x magnetization
// B^2/(rho*h) in [0,1e4] x arbitrary angle(S,B)"), exercised through
// `tools/eos_test --level con2prim` (CODE.md "Test harness"). Like
// check_table()/check_adapter(), this is a library function first: pure (no
// I/O side effects), so an HPC consumer can validate a solver in-process
// right after building its adapter. Reuses host/adapter_audit.hpp's
// CheckClassResult/QuantileStats so all three levels' reports share one
// shape (worst-offender lists, count/max/rms, print()-ability, quantile
// summaries), and the same deterministic splitmix64+xorshift64* per-sample
// PRNG / per-thread-accumulator-then-merge pattern as check_adapter()'s
// physicality soak (see adapter_audit.cpp's SampleRng/TopK/Accum -- this
// module keeps its own small copies of the same shape, per that file's own
// "each host/*.cpp keeps its own" convention). Host-only: STL throughout,
// may throw; OpenMP where loops are independent (guarded by `_OPENMP`,
// serial otherwise -- see CODE.md "Environment").
//
// `check_con2prim()` needs only the built `EntropyEOS`: every sampled state
// lives inside the adapter's own physical box (view.x_lo/x_hi, y_lo/y_hi,
// srange()), so -- unlike check_adapter()'s node round trip -- no RawTable
// is required.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <limits>
#include <vector>

#include "entropy_eos/core/con2prim.hpp"
#include "entropy_eos/core/defs.hpp"
#include "entropy_eos/core/prim2con.hpp"
#include "entropy_eos/core/state_policy.hpp"
#include "entropy_eos/host/adapter_audit.hpp"
#include "entropy_eos/host/adapter_build.hpp"

namespace eeos {

// Tunables for check_con2prim().
struct Con2PrimCheckOptions {
  size_t n_states = 20000; // total sampled states (warm pass); ~10% also get a cold pass
  unsigned seed = 987654u; // deterministic per-sample PRNG seed (see adapter_audit.cpp's SampleRng)

  double w_max_sample = 6.0; // sampled rapidity range [0, w_max_sample] (deliverable 2's w in [0,6])

  // Magnetization sigma = B^2/(rho*h) sampled log-uniform in [1e-6,
  // sigma_max] (deliverable 2's magnetization range [0,1e4]); B2 is then
  // formed as sigma*rho*h from a preliminary evaluate() at the sampled
  // (rho,s,ye). 10% of states (a per-sample coin flip, independent of the
  // sigma draw itself -- see the .cpp) get B2 = 0 exactly instead.
  double sigma_max = 1e4;

  // Warm-start perturbation magnitude, evolution-like: the warm pass's
  // s_guess/w_guess/u_guess are the truth primitive's own (s, w, u_solved)
  // perturbed by about this fraction (see the .cpp's sample_state() for the
  // exact per-quantity perturbation form) -- standing in for "last
  // timestep's converged state", which a real evolution's guess resembles
  // closely but not exactly.
  double warm_rel = 1e-3;

  // Relative threshold on the round-trip metric (max of the D/tau/S
  // conservative-space relative errors, see Con2PrimReport::rt_D/rt_tau/
  // rt_S) above which a state enters the "c2p_roundtrip" violation class.
  double tol_roundtrip = 1e-8;

  size_t worst_n = 10; // how many worst locations to keep per class

  Con2PrimOptions solver; // passed through to every con2prim() call unchanged

  // --- M3e policy section (core/state_policy.hpp) --------------------------
  // The policy pass re-runs the SAME warm state set through con2prim_safe()
  // and counts interventions (which must be zero on a state the solver
  // handled well -- see Con2PrimReport::policy_n_valid_touched), then runs a
  // small fixed broken-state battery covering the S11 taxonomy.
  bool policy = true;

  // rho_atm / w_cap for that pass. NaN means "derive it", and the derivation
  // is deliberately tied to THIS AUDIT'S OWN SAMPLED RANGE rather than to a
  // production default, because a threshold that sits inside the sampled
  // range makes the false-positive measurement vacuous rather than
  // informative:
  //   rho_atm: half the smallest D the sampled set contains, so no valid
  //     sampled state can trip the atmosphere trigger. A caller-supplied
  //     value is LOWERED to that if it is larger (and the value actually
  //     used is reported as Con2PrimReport::policy_rho_atm_used).
  //   w_cap: max(default_policy()'s acosh(100), w_max_sample + 0.5), so no
  //     valid sampled state can trip the rapidity cap -- and, because
  //     PolicyOptions::D_max/tau_max are DERIVED from w_cap (see
  //     policy_derive_bounds()), no valid sampled state can trip those
  //     either: cosh(w)^2 <= cosh(w_cap)^2 is exactly what makes tau_max an
  //     upper bound. Capped at 0.9 * solver.w_max so the cap stays able to
  //     fire at all. A caller-supplied value is RAISED to that if smaller.
  double policy_rho_atm = std::numeric_limits<double>::quiet_NaN();
  double policy_w_cap = std::numeric_limits<double>::quiet_NaN();

  // Round-trip threshold for the battery's "is the repaired state exactly
  // solvable?" check (a WARM re-solve of the returned conservatives, seeded
  // with the returned primitives -- see check_con2prim()'s doc comment for
  // why warm and not cold).
  double policy_tol_resolve = 1e-8;
};

// Result of check_con2prim(). `status` is `fatal` only if some con2prim()
// or the round-trip prim2con() call in the audit produced a non-finite
// recovered primitive/EOSPoint (see con2prim_needs_attention() for the
// "should a caller care?" summary of everything else).
struct Con2PrimReport {
  Status status = Status::ok;
  size_t n_states = 0;

  // Warm-start pass (every sampled state): C2PResult histogram.
  size_t n_newton = 0, n_fallback = 0, n_failed_no_bracket = 0, n_failed_max_iter = 0;

  // Cold-start pass (every 10th sampled state, all guesses NaN): C2PResult
  // histogram (M3c: failed_no_bracket/failed_max_iter split, matching the
  // warm pass's n_failed_no_bracket/n_failed_max_iter -- the cold pass is
  // where the S9 fallback's own bracket-detection reliability is actually
  // measured, since it always exercises the fallback path; see
  // con2prim.hpp's bracket-scan doc comment) -- no dedicated worst-offender
  // list for the cold subset (see check_con2prim()'s doc comment).
  size_t cold_n_newton = 0, cold_n_fallback = 0;
  size_t cold_n_failed_no_bracket = 0, cold_n_failed_max_iter = 0;

  // Total-iteration (iters_newton + iters_fallback) histograms, index =
  // min(total_iters, 63) -- same convention as AdapterReport::iters_hist.
  size_t iters_hist_warm[64] = {};
  size_t iters_hist_cold[64] = {};

  // Conservative-space round-trip relative errors, warm pass only (see
  // check_con2prim()'s doc comment for the exact recomputation): D and tau
  // each compared to their own truth value; S is, per state, the MAX of the
  // S_par/S_perp relative errors (a single scalar so it can carry one
  // QuantileStats like D/tau).
  QuantileStats rt_D, rt_tau, rt_S;

  // Prim-space errors, warm pass only (diagnostics -- never drive
  // con2prim_needs_attention() or the tool's exit code): |drho/rho|,
  // |dw|/(1+w), |ds|/max(|s|,1).
  QuantileStats prim_rho, prim_w, prim_s;

  // One CheckClassResult per audit below, in this fixed order:
  //   "c2p_failed" (violation: a warm-pass state whose C2PResult is
  //     failed_no_bracket or failed_max_iter; CheckClassResult::Loc has no
  //     dedicated rapidity field, so each worst entry carries the state's
  //     input rho/T/ye in Loc::rho/temp/ye and its sampled w in Loc::value
  //     -- see check_con2prim()'s doc comment)
  //   "c2p_roundtrip" (violation: warm-pass max(rt_D, rt_tau, rt_S) at a
  //     state > opts.tol_roundtrip; Loc::rho/temp/ye is the state's input
  //     location, Loc::value is that max)
  std::vector<CheckClassResult> classes;

  // --- M3e policy section (Con2PrimCheckOptions::policy) -------------------
  // All counters default to 0 / "clean", so a hand-built Con2PrimReport is
  // still clean under con2prim_needs_attention() (see
  // tests/test_con2prim_audit.cpp's truth table).
  bool policy_ran = false;
  double policy_rho_atm_used = 0.0, policy_w_cap_used = 0.0; // after the derivation above

  // Warm-set pass: con2prim_safe() over the same states as the warm pass.
  size_t policy_n_interventions = 0; // states with policy_flags != 0 (for ANY reason)

  // Per-flag counts, index = bit - 8, i.e. in flag order
  // {atmosphere, ceiling, s_floored, s_ceiled, w_capped, rho_clamped,
  //  ye_clamped, nonfinite} (core/defs.hpp).
  size_t policy_flag_counts[8] = {};

  // THE false-positive counter, and the one that drives
  // con2prim_needs_attention(): an intervention on a state whose sampled
  // (truth) primitives were policy-valid AND whose plain warm con2prim()
  // converged AND whose recovered primitives were themselves policy-valid.
  // On such a state the layer had nothing to repair, so a nonzero count
  // means a threshold is wrong (a too-tight tau_max, an atmosphere trigger
  // inside the sampled range, a sign slip) -- not that the table is hard.
  // Interventions on the M3 solver's documented residual failure tail are
  // deliberately NOT counted here: absorbing those is this layer's job.
  size_t policy_n_valid_touched = 0;

  // policy_flags == 0 but `cons` was not the input bit-identically. Must be
  // 0: the no-touch path is contractually a verbatim copy.
  size_t policy_n_cons_mismatch = 0;

  // Returned state not policy-valid, or non-finite, or D <= 0. Must be 0 --
  // this is the "never fails" contract itself.
  size_t policy_n_invalid_out = 0;

  // Broken-state battery (a fixed, deterministic set covering the S11
  // taxonomy: vacuum/sub-atmosphere, collapse under both
  // collapse_to_atmosphere settings, tau alone runaway, planted NaN/Inf in
  // every conservative, tau below the cold floor, a rapidity above the cap,
  // out-of-range Ye, a negative B^2).
  size_t policy_battery_n = 0;         // cases run
  size_t policy_battery_n_invalid = 0; // cases whose output failed a validity check (must be 0)
  size_t policy_battery_n_no_flag = 0; // cases where the policy did NOT fire at all (must be 0)
  size_t policy_battery_n_cold_missed = 0; // diagnostic only: cases a COLD re-solve misses

  double solves_per_sec_warm = 0.0, solves_per_sec_cold = 0.0, solves_per_sec_policy = 0.0;

  // Human-readable summary: status, state/result counts (warm and cold),
  // both iteration histograms, the round-trip and prim-space quantiles,
  // each class's count/max/rms and worst offenders, the M3e policy section,
  // and solve throughput. Self-contained (no cross-referencing needed to
  // read the policy lines).
  void print(std::ostream &os) const;
};

// Runs random-state prim2con -> con2prim round trips against `adapter`
// (con2prim-entropy-rapidity.md deliverable 2 / S12; CODE.md "Test harness
// --level con2prim"). For each of opts.n_states deterministically sampled
// states (rho, Ye, s, w, B2, cos_vB -- see the .cpp's sample_state() for the
// exact sampling ranges):
//   1. prim2con() builds the truth conservative state (D, tau, D_Y, S_par,
//      S_perp, B2).
//   2. A WARM con2prim() call, with s_guess/w_guess/u_guess perturbed from
//      the truth primitive state by about opts.warm_rel (evolution-like:
//      the previous timestep's converged state is a close, not exact,
//      guess).
//   3. On every 10th state, ADDITIONALLY a COLD con2prim() call (every
//      guess NaN).
// Each call's recovered primitive state is round-tripped back through
// prim2con() -- the RECOVERED rho/s/Ye/w, the SAME B2, and cos_vB
// reconstructed sign-correctly as v_par/V from the recovered v_par/v_perp
// (V = sqrt(v_par^2+v_perp^2); falls back to the ORIGINAL sampled cos_vB
// when V is degenerately small, matching prim2con.hpp's own note that
// cos_vB is irrelevant whenever B2==0 or w==0, i.e. exactly when V->0) --
// and compared against the truth conservative state. This is deliberately
// con2prim's own forward map applied to its own answer, not a comparison to
// the sampled (s,w,cos_vB): the solver's residuals only constrain
// V=tanh(w) and the energy balance, not that v_par/V reproduces the
// original cos_vB, so if a state lands on an alternate root of the (s,w)
// system (con2prim.hpp S9's caveat that the outer solve's uniqueness is not
// proven) comparing to the ORIGINAL direction would show a spurious
// mismatch even though the solver's own residuals converged tightly -- see
// tests/test_con2prim.cpp's run_one() for the same rationale applied to a
// hand-unrolled version of this recomputation.
//
// M3e POLICY SECTION (opts.policy, on by default). After the warm/cold
// passes, the SAME warm state set is run once more through
// con2prim_safe() (core/state_policy.hpp) with a PolicyOptions built by
// default_policy() plus the two audit-specific derivations documented on
// Con2PrimCheckOptions::policy_rho_atm/_w_cap. Two things are measured:
//
//   1. FALSE POSITIVES. Every sampled state is valid by construction, so the
//      layer must be transparent on it. The counter that matters is
//      Con2PrimReport::policy_n_valid_touched -- an intervention on a state
//      whose truth primitives were policy-valid, whose plain warm solve
//      converged, and whose recovered primitives were policy-valid too.
//      (The raw intervention count is reported as well, but on a real table
//      it legitimately picks up the M3 solver's documented residual failure
//      tail -- absorbing that is the layer's purpose, so it must not be read
//      as a false positive.) Also checked, on every state: the no-touch path
//      returns the input conservatives bit-identically, and every returned
//      state is policy-valid and finite.
//   2. A BROKEN-STATE BATTERY: a fixed, deterministic set of ~28 states
//      covering the S11 taxonomy (see Con2PrimReport::policy_battery_n).
//      Each output must be finite, policy-valid, actually flagged, and
//      exactly solvable -- the last checked by a WARM re-solve of the
//      returned conservatives seeded with the returned primitives. Warm and
//      not cold on purpose: the excision targets (the atmosphere, and the
//      hot/dense ceiling corner) are precisely where the M3d cold seed and
//      the S9 bracket scan have their documented real-table failure tail, so
//      a cold re-solve would measure that tail rather than this layer. The
//      cold outcome is still recorded, as policy_battery_n_cold_missed.
//
// Timing: the warm, cold and policy con2prim() passes are timed separately
// (solves_per_sec_warm/_cold/_policy), excluding state sampling/
// prim2con(truth) and the round-trip bookkeeping that follows -- same "point
// generation is untimed, only the solve itself is" discipline as
// check_adapter()'s physicality soak. The policy pass runs a full extra
// con2prim() per state, so it roughly doubles the audit's solve cost; pass
// opts.policy = false to skip it.
//
// Never throws on its own account: a non-finite recovered primitive/EOSPoint
// or round-trip prim2con() output at any state (warm or cold) is recorded
// (status becomes Status::fatal) and that state is excluded from the
// round-trip/prim-space statistics, but the audit continues over the
// remaining states.
Con2PrimReport check_con2prim(const EntropyEOS &adapter,
                               const Con2PrimCheckOptions &opts = Con2PrimCheckOptions());

// True iff the report indicates something a caller should look at: any
// failed_no_bracket/failed_max_iter state, warm or cold pass, the
// "c2p_roundtrip" class has count > 0, or (M3e) the policy section found a
// false positive (policy_n_valid_touched), a non-verbatim no-touch path
// (policy_n_cons_mismatch), an invalid output (policy_n_invalid_out), or a
// broken-state battery case that came out invalid or unflagged
// (policy_battery_n_invalid / policy_battery_n_no_flag). The raw
// policy_n_interventions count is deliberately NOT consulted: on a real
// table it legitimately absorbs the M3 solver's documented residual failure
// tail. (status == Status::fatal is a
// separate, more severe signal callers should check on its own -- see
// tools/eos_test.cpp. "c2p_failed"'s own class count is not consulted here
// -- it is redundant with n_failed_no_bracket/n_failed_max_iter by
// construction in check_con2prim(), but a caller building a Con2PrimReport
// by hand need not populate `classes` consistently with those scalars, so
// this function -- like adapter_needs_attention()'s maxiter_count -- reads
// the scalar counters directly.)
bool con2prim_needs_attention(const Con2PrimReport &report);

} // namespace eeos
