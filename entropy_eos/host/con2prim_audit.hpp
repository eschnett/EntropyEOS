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
#include <vector>

#include "entropy_eos/core/con2prim.hpp"
#include "entropy_eos/core/defs.hpp"
#include "entropy_eos/core/prim2con.hpp"
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
  // histogram, failed_no_bracket/failed_max_iter combined into one bucket
  // (no dedicated worst-offender list for the cold subset -- see
  // check_con2prim()'s doc comment).
  size_t cold_n_newton = 0, cold_n_fallback = 0, cold_n_failed = 0;

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

  double solves_per_sec_warm = 0.0, solves_per_sec_cold = 0.0;

  // Human-readable summary: status, state/result counts (warm and cold),
  // both iteration histograms, the round-trip and prim-space quantiles,
  // each class's count/max/rms and worst offenders, and solve throughput.
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
// Timing: the warm and cold con2prim() passes are timed separately
// (solves_per_sec_warm/_cold), excluding state sampling/prim2con(truth) and
// the round-trip bookkeeping that follows -- same "point generation is
// untimed, only the solve itself is" discipline as check_adapter()'s
// physicality soak.
//
// Never throws on its own account: a non-finite recovered primitive/EOSPoint
// or round-trip prim2con() output at any state (warm or cold) is recorded
// (status becomes Status::fatal) and that state is excluded from the
// round-trip/prim-space statistics, but the audit continues over the
// remaining states.
Con2PrimReport check_con2prim(const EntropyEOS &adapter,
                               const Con2PrimCheckOptions &opts = Con2PrimCheckOptions());

// True iff the report indicates something a caller should look at: any
// failed_no_bracket/failed_max_iter (warm pass) or failed (cold pass) state,
// or the "c2p_roundtrip" class has count > 0. (status == Status::fatal is a
// separate, more severe signal callers should check on its own -- see
// tools/eos_test.cpp. "c2p_failed"'s own class count is not consulted here
// -- it is redundant with n_failed_no_bracket/n_failed_max_iter by
// construction in check_con2prim(), but a caller building a Con2PrimReport
// by hand need not populate `classes` consistently with those scalars, so
// this function -- like adapter_needs_attention()'s maxiter_count -- reads
// the scalar counters directly.)
bool con2prim_needs_attention(const Con2PrimReport &report);

} // namespace eeos
