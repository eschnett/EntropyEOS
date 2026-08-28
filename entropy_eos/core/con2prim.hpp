// entropy_eos/core/con2prim.hpp
//
// M3a entropy/rapidity con2prim (con2prim-entropy-rapidity.md SS6-9):
// recovers the primitive state (rho, s, Ye, w) from the scalar conservative
// state (D, tau, D_Y, S_par, S_perp, B^2) produced by prim2con.hpp. Header-
// only, device-ready (see CODE.md "Layout"): no STL containers, no
// exceptions, no allocation, every function EEOS_HOST_DEVICE.
//
// Algorithm (design doc SS6-9, restated with this file's implementation
// decisions):
//   1. Ye = D_Y/D (exact, S4). Trial (s,w): caller-supplied guesses if
//      finite, else cold-start seeds (S6) -- see con2prim()'s doc comment;
//      these are documented as *just seeds*, the safeguards below do the
//      real work.
//   2. Per iterate (s,w): rho = D/cosh(w); one EOS call (warm-started by
//      threading u_solved between iterations); z, v_par, v_perp, V; the
//      two residuals f1 (S7 momentum) and f2 (S7 cancellation-free energy,
//      normalized by max(tau, tau_floor_rel*D)) -- see detail::c2p_eval().
//   3. Jacobian assembled by the chain rule from the S8 pieces (z_w, z_s,
//      p_w, p_s, dv_par/d*, dv_perp/d*) -- see detail::c2p_eval()'s second
//      half; the V*dV/d* product form sidesteps the V->0 special case (S8),
//      dividing by V only where the numerator is already O(V^2).
//   4. Damped 2x2 Newton (S7-8): solve, limit the step (|dw| <= 1,
//      |ds| <= 0.25*extended-span), clamp w >= 0 -- see con2prim(), including
//      why an initial backtracking-on-non-decrease scheme was measured to be
//      net harmful and replaced by unconditional clamped steps. The loop also
//      remembers its best-so-far iterate (M3h item 7) and hands THAT, not its
//      last, to step 5.
//   5. S9 nested fallback when Newton fails to converge within
//      max_iter_newton (including a non-finite/singular step): a bracketed
//      safeguarded 1D Newton/bisection inner solve of f1(w;s)=0 (S9's
//      strict-monotonicity proof), nested inside a bracketed safeguarded
//      1D secant/bisection outer solve of g(s) = f2(s, w*(s)) -- see
//      detail::c2p_inner_solve_w() and con2prim()'s fallback section, plus
//      the bracket-collapse/stagnation/precision-polish safeguards for the
//      double-precision floor the outer bisection runs into for large |s|.
//      failed_no_bracket when the scan below finds no sign change anywhere
//      (S11: the caller's invalid-state policy); failed_max_iter when the
//      outer solve exhausts its budget without meeting both tolerances.
//
//      TOLERANCE SEMANTICS (M3h; see item 7 and Con2PrimOptions::tol).
//      "Both tolerances", here and at every convergence test in this file,
//      means the two NORMALIZED residuals:
//        |f1|/cosh(w) = |tanh(w) - V| <= tol     and     |f2| <= tol.
//      f2 is normalized already (by max(tau, tau_floor_rel*D)); f1 is
//      O(cosh w) by construction, so it is the comparison -- never f1 itself,
//      and never the Jacobian rows -- that carries the 1/cosh(w). The Newton
//      and secant steps are invariant under row scaling, so nothing else in
//      the algorithm changes.
//
//      M3c bracket scan (measured on real tables -- LS220/SRO, --states
//      4000: warm n_failed_no_bracket ~0.5%, but the seed-independent COLD
//      pass -- which always exercises this fallback -- saw ~37% failed;
//      diagnosis below). The old bracket was the union of srange_extended
//      at rho=D (w=0) and rho=D/cosh(w_max) (w=w_max), checked ONLY at its
//      two endpoints. Measured root cause (confirmed, not assumed -- e.g. a
//      reproduced LS220 state at rho=5784 g/cc, T=232 MeV, w=5.74): that
//      union can span 10+ orders of magnitude in s (entropy per baryon
//      blows up at the low-density end), while g's sign-changing region can
//      occupy a sub-percent-wide window inside it (measured: bracket
//      [0.3, 4.9e13], root's sign flip confined to [1.10e12, 1.24e12] --
//      about a millionth of the bracket) -- exactly the task spec's
//      "non-monotone g / hidden root between same-sign endpoints"
//      hypothesis, verified. Fix (detail::c2p_bracket_scan()): scan
//      Con2PrimOptions::bracket_scan candidate s-values -- the two extended
//      and two physical srange endpoints (a global safety net), plus
//      interior points split between a GLOBAL half (uniform in s across
//      the physical range, which alone finds any single monotone root
//      regardless of how good the incoming iterate is -- needed because a
//      pure-fallback call's "incoming iterate" is just the crude cold seed,
//      not a near-converged Newton position) and a LOCAL half (a dense
//      window centered on the incoming s, sized to its own local srange --
//      the part that actually resolves the sub-percent-wide real-table
//      features above, when the incoming iterate IS a good one, e.g. a
//      warm-started Newton that stalled on a near-singular Jacobian rather
//      than a bad guess -- see the "S7-8 damped 2x2 Newton" comment below).
//      Verified on the reproduced state above: this scan finds a sign
//      change (a genuine root of f1=f2=0, though not necessarily the
//      SAME root that generated the test conservative state -- S9's
//      uniqueness is not proven, and con2prim_audit.hpp's own round-trip
//      logic already accounts for landing on an alternate root of the same
//      conservative state). See c2p_bracket_scan()'s own doc comment for
//      the earlier (reverted) single-percentile-track design and why it
//      regressed the cold/forced-fallback case.
//
//   6. M3d B^2-AWARE COLD SEED (detail::c2p_cold_seed(), used by con2prim()
//      whenever a guess is absent). The M3c crude seed (tanh w0 =
//      |S|/(tau+D+p0)) is B^2-BLIND: S_perp = (z+B^2)*v_perp (S5) inflates
//      |S| independently of the actual velocity, so for a strongly
//      magnetized state w0 comes out far too large, rho0 = D/cosh(w0) far
//      too small, and s0 = mid-srange(rho0) lands in the wrong basin --
//      after which even the M3c scan's LOCAL half is anchored in the wrong
//      place. Measured cost of that blindness (--level con2prim, --states
//      4000): ~1/3 of COLD calls failed on both real tables (LS220 143/400,
//      SRO 131/400), concentrated in the magnetized states, while WARM calls
//      (98.8% Newton) were unaffected. The M3d seed replaces it with a
//      fixed-cost block iteration in which every step is an EXACT S4/S5/S7
//      relation and the only lagged quantity is the pressure -- see
//      c2p_cold_seed()'s own doc comment for the four ingredients (the
//      solved magnetic energy relation, the exact momentum projection, the
//      tau-identity energy estimate, and the provably-cannot-fail monotone
//      s-recovery), the measurements that set its two tunables, and the two
//      alternative closures that were implemented and measured worse.
//      Measured result (--states 4000, i.e. 400 cold states): cold
//      failed_total 143 -> 0 (LS220) and 131 -> 0 (SRO), against WARM
//      failed_totals of 7 and 13 that are BIT-IDENTICAL to pre-M3d (same
//      counts, same round-trip and prim-space quantiles to every digit --
//      a fully warm call never touches the seed), and cold throughput up
//      ~9x (8.3e3 -> 8.0e4 solves/s LS220, 7.8e3 -> 6.2e4 SRO): the seed's
//      ~20 EOS evaluations are far cheaper than the 17-point bracket scan
//      (17 full inner w-solves) it now usually avoids entirely. The cold
//      iteration histogram moves with it -- LS220 median total iterations
//      7 -> 4, and the >30-iteration tail (85 states, 27 of them at the
//      63-iteration histogram cap) disappears.
//
//      At 10x statistics (--states 40000: 40000 warm, 4000 cold) the
//      residual failure RATES are cold 0.45% vs warm 0.22% (LS220) and
//      cold 0.80% vs warm 0.41% (SRO) -- within 2x on both tables, and the
//      remaining states are the hot-edge/acausal-corner tail that BOTH
//      passes share (open item (ii)), not a seed deficiency. Pre-M3d the
//      same ratio was ~200x.
//
//   7. M3h CLASS-B SOLVER HARDENING (CODE.md "M3h empirical findings"), three
//      changes that between them take the residual failure tail from
//      31 warm + 8 cold (LS220) and 82 + 15 (SRO) per 40k warm / 4k cold
//      audit states to 1 + 0 and 0 + 1:
//        (a) RELATIVE f1 CONVERGENCE TESTS -- detail::c2p_f1_converged(),
//            used by c2p_inner_solve_w()'s loop, con2prim()'s Newton test and
//            its final acceptance; detail::c2p_scaled_norm() for the
//            precision polish's improvement test and the best-iterate
//            tracking. This is the one that moves the numbers: the dominant
//            failure class was states with |f2| ~ 1e-14 and |f1| ~ 2e-12 at
//            w ~ 6, i.e. |tanh w - V| ~ 1e-14 -- converged, and rejected only
//            by a mis-scaled comparison. See c2p_f1_converged()'s own
//            comment.
//        (b) GEOMETRIC RELATIVE LOCAL SCAN WINDOW -- c2p_bracket_scan()'s
//            LOCAL half is now a ladder of relative offsets from the anchor
//            rather than a linear window sized to a physical srange span that
//            can be 5e4 times the anchor itself. MEASURED NEUTRAL on both
//            real tables once (a) and (c) are in place (the states that
//            depended on the scan are gone), and measurably better on the
//            geometry that motivated it (tests/test_con2prim.cpp test 10).
//        (c) BEST-ITERATE HANDOFF -- con2prim()'s Newton loop tracks its
//            minimum scaled-norm iterate and uses it as the fallback's scan
//            anchor and warm start, and every failure exit reports a state no
//            worse than it. Exit-state bookkeeping only: one Residuals copy,
//            no extra EOS evaluations, the unconditional-clamped-step design
//            of item 4 untouched.
//      What did NOT change: the defaults (tol 1e-12, max_iter_newton 30,
//      max_iter_1d 60, bracket_scan 17), the M3d cold seed, the residuals,
//      the Jacobian, and the steps. Warm Newton solves whose trajectory never
//      touches a changed comparison are bit-identical.

#pragma once

#include <cmath>

#include "entropy_eos/core/adapter_eval.hpp"
#include "entropy_eos/core/defs.hpp"
#include "entropy_eos/core/prim2con.hpp"

namespace eeos {

struct Con2PrimIn {
  real D, tau, D_Y, S_par, S_perp, B2;
};

struct Con2PrimOptions {
  // Convergence tolerance on the NORMALIZED residual pair (M3h):
  //   |tanh(w) - V| = |f1|/cosh(w) <= tol   and   |f2| <= tol
  // (f2 is already normalized, by max(tau, tau_floor_rel*D), and is
  // unchanged from M3a). The f1 normalization is what makes tol mean the
  // same thing at every rapidity: the raw momentum residual
  // f1 = sinh(w) - cosh(w)*V is O(cosh w), so an absolute bound on it would
  // demand cosh(w) times more accuracy at w = 6 than at w = 0 and more than
  // double precision can deliver at all past w ~ 9 -- see
  // detail::c2p_f1_converged() for the measurement that motivated this.
  //
  // 1e-12 is where the failure tail is empty and the bulk is comfortable, and
  // the two trade against each other here (measured, LS220 40k warm + 4k
  // cold): at 1e-13 the recovered primitives are ~30x tighter (prim_rho p99
  // 2.3e-9 -> 7.2e-11) but 46 states fail instead of 1, and at 1e-14, 1204 --
  // past ~1e-13 the request outruns what the EOS-derived residuals can
  // deliver. Tighten it only with that trade in view.
  real tol = real(1e-12);
  int max_iter_newton = 30;
  int max_iter_1d = 60;
  real w_max = real(12.0);
  real tau_floor_rel = real(1e-16);

  // M3c: number of points in the S9 outer bracket scan (see con2prim()'s
  // fallback section doc comment for the diagnosis this replaces the old
  // endpoints-only check). Clamped to [4, detail::kC2PBracketScanMax] at
  // use, so any positive value is safe to pass; 4 is the minimum that still
  // provides the two extended + two physical safety-net points the scan's
  // design relies on.
  int bracket_scan = 17;

  // M3d cold-start seed (detail::c2p_cold_seed()): number of outer
  // (p -> z -> w -> eps -> s -> p) passes, and the iteration cap of the
  // inner monotone s-recovery solve. Clamped to [1,8] / [2,40] at use, so
  // any value is safe to pass.
  //
  // Both defaults are measured on the LS220/SRO cold subsets of
  // `eos_test --level con2prim --states 4000` (400 cold states each),
  // reported as (cold failed_total, seeds within 20% of truth in BOTH s
  // and w):
  //   passes:  1 -> (1, 322/400) LS220, (4, 321/400) SRO
  //            2 -> (3, 398/400),       (6, 398/400)
  //            3 -> (0, 399/400),       (0, 399/400)   <-- default
  //            4 -> (1, 400/400),       (1, 400/400)
  //            6 -> (0, 400/400),       (0, 400/400)
  // The seed's own error falls ~10x per pass in s and ~3x in w, but
  // reliability saturates at 3: from there the residual 0-1 failures are
  // the documented hot-edge/acausal-corner tail that the WARM pass shows
  // too (7 LS220 / 13 SRO on the same sample), not a seed deficiency, so
  // passes 4-6 buy prettier seeds and no fewer failures. Pass 1 alone
  // already removes the whole magnetized failure class (143/131 -> 1/4),
  // which is the measurement that identifies the B^2-blind z as the root
  // cause rather than anything downstream of it.
  //
  // seed_s_iters is a CAP on a safeguarded Newton that exits early, not a
  // fixed trip count: 6, 8, 12, 16 and 24 all produce BIT-IDENTICAL results
  // on both real tables (the s-solve converges in <= 6 steps once
  // warm-started along the pass chain), while 4 is measurably worse
  // (LS220 3 failed, 397/400). 16 is kept as free headroom on the CPU; a
  // GPU port running a fixed trip count can safely use 8.
  int seed_passes = 3;
  int seed_s_iters = 16;
};

enum class C2PResult { converged_newton, converged_fallback, failed_no_bracket, failed_max_iter };

struct Con2PrimOut {
  real rho, s, ye, w, W, v_par, v_perp;
  EOSPoint eos;
  C2PResult result;
  int iters_newton, iters_fallback;
  unsigned flags; // OR of EOSPoint flags seen at the solution (the final evaluate() call's pt.flags)
};

namespace detail {

// True iff v is neither NaN nor +-Inf -- device-portable (no std::isfinite
// dependency, mirroring aeval_is_nan()'s self-compare trick, see
// adapter_eval.hpp): for finite v, v - v == 0 exactly; NaN - NaN and
// Inf - Inf are both NaN, so the comparison fails for either.
EEOS_HOST_DEVICE inline bool c2p_is_finite(real v) { return (v - v) == real(0); }

// "Reflect w<0 to tiny positive" (con2prim-entropy-rapidity.md S9 task
// spec): a Newton step that overshoots past w=0 lands here instead of
// exactly on the w=0 boundary, so the next Jacobian still has a meaningful
// (nonzero) w to work with.
constexpr real kC2PTinyW = real(1e-10);

// One residual/Jacobian evaluation at a trial (s,w), echoing the inputs
// (r.s, r.w) for callers that thread the result through warm starts.
// `coshw` is cached (M3h) purely so the residual SCALES are available to the
// convergence tests below without a second std::cosh() call -- f1 is
// O(cosh w) by construction (see c2p_f1_converged()).
struct Residuals {
  real s, w;
  real rho;
  real coshw;
  EOSPoint pt;
  real z, v_par, v_perp, V;
  real f1, f2;
  real df1_ds, df1_dw, df2_ds, df2_dw;
};

// M3h: the momentum residual's convergence test, in the NORMALIZED residual
// f1/cosh(w) = tanh(w) - V rather than in f1 itself.
//
// f1 = sinh(w) - cosh(w)*V is a difference of two O(cosh w) quantities, so
// an ABSOLUTE |f1| <= tol test is a test on tanh(w) - V scaled by cosh(w) --
// i.e. it silently demands cosh(w) times more accuracy at high rapidity than
// at low. That is not merely strict, it is unattainable: V is derived from
// the EOS (V = |S|-projections / z, z = D*h*cosh w), so its relative error is
// a few ulp and |f1| has a double-precision floor of order eps*cosh(w) times
// the EOS chain's own error amplification. MEASURED on the two real tables
// (40k warm + 4k cold audit states, post-M3g): the dominant residual failure
// class returned |f1| = 1.2e-12 ... 4.2e-12 with |f2| = 1e-14 ... 1e-15 at
// w = 5.5 ... 6.0, where cosh w ~ 180-200 -- i.e. |tanh w - V| ~ 1e-14, fully
// converged, rejected only by the mis-scaled test. At trial w >~ 9 (which
// inner solves reach while bracketing) eps*cosh(w) EXCEEDS tol = 1e-12
// outright, so the old test could never be satisfied there and the inner
// solve burned its whole budget at every such scan point.
//
// Written as |f1| <= tol*cosh(w) rather than |f1|/cosh(w) <= tol so the hot
// path does one multiply instead of one divide; cosh(w) >= 1 always, so this
// is never tighter than the old test and never divides by zero.
EEOS_HOST_DEVICE inline bool c2p_f1_converged(const Residuals &r, real tol) {
  return std::fabs(r.f1) <= tol * r.coshw;
}

// M3h: the residual pair's norm with f1 normalized to f2's O(1) scale
// (max-norm of the two normalized residuals). Used wherever the two
// residuals are COMPARED or minimized together -- the precision polish's
// improvement test and the Newton loop's best-iterate tracking -- where a
// plain ||f||_2 would be dominated by the O(cosh w) component and would
// happily trade a real f2 improvement for a cosmetic f1 one.
EEOS_HOST_DEVICE inline real c2p_scaled_norm(const Residuals &r) {
  const real n1 = std::fabs(r.f1) / r.coshw;
  const real n2 = std::fabs(r.f2);
  return n1 > n2 ? n1 : n2;
}

// Evaluates (f1, f2) and their Jacobian at one trial (s, w) (design doc
// SS6-8). D, tau, ye, S_par, S_perp, B2 are the fixed conservative data;
// u_prev threads the EOS T-solve's warm start between calls.
EEOS_HOST_DEVICE inline Residuals c2p_eval(const EntropyEOSView &eos, real D, real tau, real ye,
                                            real S_par, real S_perp, real B2, real s, real w, real u_prev,
                                            real tau_floor_rel) {
  Residuals r;
  r.s = s;
  r.w = w;

  const real coshw = std::cosh(w);
  const real sinhw = std::sinh(w);
  const real tanhw = std::tanh(w);
  const real half_sinh = std::sinh(real(0.5) * w);

  r.coshw = coshw;
  r.rho = D / coshw;
  r.pt = eos.evaluate(r.rho, s, ye, u_prev);

  const real h = r.pt.h;
  const real U = r.pt.U;
  const real p = r.pt.p;
  const real cs2 = r.pt.cs2;
  const real U_rho = r.pt.U_rho;
  const real U_s = r.pt.U_s; // == r.pt.That (EOSPoint doc: "That = U_s")
  const real U_rhos = r.pt.U_rhos;

  // S6: z = D*h*cosh(w) (== rho*h*W^2 via rho = D/cosh(w)).
  r.z = D * h * coshw;
  const real zpB2 = r.z + B2;
  r.v_par = S_par / r.z;
  r.v_perp = S_perp / zpB2;
  const real Vsq = r.v_par * r.v_par + r.v_perp * r.v_perp;
  r.V = std::sqrt(Vsq);

  // S7 momentum residual.
  r.f1 = sinhw - coshw * r.V;

  // S7 cancellation-free energy residual (identical construction to
  // prim2con.hpp's tau, evaluated at the trial point instead of at the
  // truth -- see that file's doc comment for the E-D algebra).
  const real tau_model = real(2) * D * half_sinh * half_sinh + r.rho * U * coshw * coshw +
                          p * sinhw * sinhw + real(0.5) * B2 * (real(1) + Vsq) -
                          real(0.5) * B2 * r.v_par * r.v_par;
  const real denom = tau > tau_floor_rel * D ? tau : tau_floor_rel * D;
  r.f2 = (tau_model - tau) / denom;

  // S8 Jacobian pieces.
  const real rho_w = -r.rho * tanhw;
  const real z_w = r.z * (real(1) - cs2) * tanhw;
  const real z_s = D * coshw * (U_s + r.rho * U_rhos);
  const real p_w = -r.rho * h * cs2 * tanhw;
  const real p_s = r.rho * r.rho * U_rhos;

  const real dvpar_dw = -(r.v_par / r.z) * z_w;
  const real dvpar_ds = -(r.v_par / r.z) * z_s;
  const real dvperp_dw = -(r.v_perp / zpB2) * z_w;
  const real dvperp_ds = -(r.v_perp / zpB2) * z_s;

  // V*dV/d* (not dV/d* itself): finite and correct in the V -> 0 limit
  // without a special case, since the numerator is O(V^2) there (S8 task
  // spec) -- v_par, v_perp -> 0 linearly with V, and dvpar_d*/dvperp_d* are
  // themselves proportional to v_par/v_perp, so each product term is
  // O(V^2).
  const real VdV_dw = r.v_par * dvpar_dw + r.v_perp * dvperp_dw;
  const real VdV_ds = r.v_par * dvpar_ds + r.v_perp * dvperp_ds;
  const real Vfloor = r.V > real(1e-300) ? r.V : real(1e-300);
  const real dV_dw = VdV_dw / Vfloor;
  const real dV_ds = VdV_ds / Vfloor;

  r.df1_dw = coshw - sinhw * r.V - coshw * dV_dw;
  r.df1_ds = -coshw * dV_ds;

  // d(tau_model)/dw|_s and d(tau_model)/ds|_w, term by term (rho depends on
  // w only, not s, since rho = D/cosh(w)):
  //   term1 = 2*D*sinh(w/2)^2:            d/dw = D*sinh(w);            d/ds = 0
  //   term2 = rho*U*cosh(w)^2:            d/dw = rho_w*(U+rho*U_rho)*cosh(w)^2
  //                                                + 2*rho*U*cosh(w)*sinh(w)
  //                                        d/ds = rho*U_s*cosh(w)^2
  //     (d(rho*U)/dw|_s = rho_w*U + rho*(U_rho*rho_w) = rho_w*(U+rho*U_rho);
  //      d(rho*U)/ds|_w = rho*U_s since rho does not depend on s)
  //   term3 = p*sinh(w)^2:                d/dw = p_w*sinh(w)^2 + 2*p*sinh(w)*cosh(w)
  //                                        d/ds = p_s*sinh(w)^2
  //   term4 = 0.5*B2*(1+V^2):             d/dw = B2*(V*dV/dw); d/ds = B2*(V*dV/ds)
  //   term5 = -0.5*B2*v_par^2:            d/dw = -B2*v_par*dvpar_dw
  //                                        d/ds = -B2*v_par*dvpar_ds
  const real dtau_dw = D * sinhw + rho_w * (U + r.rho * U_rho) * coshw * coshw +
                        real(2) * r.rho * U * coshw * sinhw + p_w * sinhw * sinhw +
                        real(2) * p * sinhw * coshw + B2 * VdV_dw - B2 * r.v_par * dvpar_dw;
  const real dtau_ds =
      r.rho * U_s * coshw * coshw + p_s * sinhw * sinhw + B2 * VdV_ds - B2 * r.v_par * dvpar_ds;

  r.df2_dw = dtau_dw / denom;
  r.df2_ds = dtau_ds / denom;

  return r;
}

// S9 inner solve: the unique root of f1(w; s) = 0 on [0, w_max] (strictly
// monotone increasing in w by the design doc's proof), via bracketed
// safeguarded Newton/bisection -- same style as adapter_eval.hpp's T-solve.
// Returns the converged (or best-effort, if max_iter_1d is exhausted)
// Residuals; iters_out receives the iteration count.
//
// M3h: `tol` is a tolerance on the NORMALIZED residual tanh(w) - V (see
// c2p_f1_converged()), not on f1 itself. This matters most here: the
// bisection half of this solve walks w over the whole [0, w_max] bracket, so
// it routinely evaluates f1 at trial w near w_max = 12, where cosh(w) ~ 8e4
// puts f1's own double-precision floor an order of magnitude ABOVE the old
// absolute tol -- every such solve ran its full max_iter_1d budget without
// ever being able to pass.
EEOS_HOST_DEVICE inline Residuals c2p_inner_solve_w(const EntropyEOSView &eos, real D, real tau, real ye,
                                                     real S_par, real S_perp, real B2, real s, real w_max,
                                                     real tau_floor_rel, real w_start, real u_prev,
                                                     int max_iter_1d, real tol, int &iters_out) {
  real lo = real(0), hi = w_max;
  real w = aeval_clamp(w_start, lo, hi);
  Residuals r = c2p_eval(eos, D, tau, ye, S_par, S_perp, B2, s, w, u_prev, tau_floor_rel);

  int iters = 0;
  while (!c2p_f1_converged(r, tol) && iters < max_iter_1d) {
    ++iters;
    // f1 increasing in w: f1<0 means the root is further right (raise lo);
    // f1>0 means it is further left (lower hi).
    if (r.f1 < real(0)) {
      lo = w;
    } else {
      hi = w;
    }

    real w_next;
    bool newton_ok = false;
    if (r.df1_dw != real(0) && c2p_is_finite(r.df1_dw)) {
      w_next = w - r.f1 / r.df1_dw;
      newton_ok = c2p_is_finite(w_next) && w_next > lo && w_next < hi;
    }
    if (!newton_ok) w_next = real(0.5) * (lo + hi);

    w = w_next;
    r = c2p_eval(eos, D, tau, ye, S_par, S_perp, B2, s, w, r.pt.u_solved, tau_floor_rel);

    // Bracket-collapse safeguard (mirrors the outer solve's, see
    // con2prim()): once lo/hi can no longer be distinguished as distinct
    // doubles, no further iteration can improve on floating-point grounds.
    if (!(lo < hi)) break;
  }

  iters_out = iters;
  return r;
}

// M3c: fixed compile-time cap for the bracket scan's arrays (no allocation,
// per this file's device-ready discipline) -- Con2PrimOptions::bracket_scan
// is clamped into [4, kC2PBracketScanMax] before use.
constexpr int kC2PBracketScanMax = 33;

// M3h: the LOCAL half of the bracket scan places its points at
// s_in*(1 +- delta), delta log-spaced over [kC2PScanDeltaMin,
// kC2PScanDeltaMax] -- see c2p_bracket_scan()'s LOCAL-half comment for the
// measurement that replaced the pre-M3h srange-width-sized linear window.
//
// The two ends are set by what the anchor can be trusted to mean, not tuned
// to a particular table. UPPER (0.6): the pre-M3h window's own coefficient,
// reused -- past +-60% of the anchor, "local" stops saying anything the
// GLOBAL half does not already cover. LOWER (1e-5): just below the tightest
// stalled-Newton anchor measured on the M3g failure set (relative distances
// to the root of 3e-5 ... 1e-2). Anything closer to the anchor than this is
// caught anyway by the innermost PAIR, which straddles s_in, so a smaller
// value would only help against MULTIPLE roots inside +-1e-5 -- and it would
// spend a rung out of the six the default n_scan = 17 leaves for this half.
// Measured (LS220/SRO, 40k states, plus a forced-fallback stress at
// max_iter_newton = 0): 1e-5, 1e-4 and 1e-3 are indistinguishable on both
// real tables, so this is a reasoned choice, not a fitted one.
constexpr real kC2PScanDeltaMin = real(1e-5);
constexpr real kC2PScanDeltaMax = real(0.6);

// Ascending insertion sort of up to kC2PBracketScanMax (s,g) pairs by s,
// keeping g paired with its s. O(n^2) worst case, but n <= 33 here -- and
// each entry already cost one inner w-solve to produce -- so this is
// negligible; a hand-rolled sort (rather than std::sort) keeps this file's
// no-STL-containers, EEOS_HOST_DEVICE discipline (module header "Layout").
EEOS_HOST_DEVICE inline void c2p_sort_scan(real *s, real *g, int n) {
  for (int i = 1; i < n; ++i) {
    const real si = s[i];
    const real gi = g[i];
    int j = i - 1;
    while (j >= 0 && s[j] > si) {
      s[j + 1] = s[j];
      g[j + 1] = g[j];
      --j;
    }
    s[j + 1] = si;
    g[j + 1] = gi;
  }
}

// Outcome of c2p_bracket_scan(): either a sign-changing interval [s_lo,
// s_hi] ready for the existing Illinois/stagnation/polish machinery, or (if
// no scan interval showed a sign change) the single scanned point closest
// to a root, for the failed_no_bracket report.
struct BracketScanResult {
  bool bracketed;
  real s_lo, s_hi;
  real s_best;
};

// M3c multi-point outer bracket scan (see con2prim()'s fallback section doc
// comment for the measured diagnosis and design rationale). Builds
// `n_scan` candidate s-values --
//   - the two EXTENDED srange endpoints (rho=D, rho=D/cosh(w_max)),
//   - the two PHYSICAL srange endpoints (same two rho's),
//   - n_scan-4 interior points, HALF uniform in s across the physical
//     range [s_phys_lo, s_phys_hi] and HALF a geometric ladder of RELATIVE
//     offsets from `s_in` (M3h; through M3g this half was a linear window
//     of +-0.6 of s_in's own local physical srange width -- see the local
//     half's own comment for the measurement that replaced it) --
// -- evaluates g=f2 at each with a FULL-precision inner w-solve (see the
// evaluation loop's own comment below for why a loosened scan-only
// tolerance was tried and reverted), sorts the (s,g) pairs by s (the
// generation order above is not itself monotone in s), then picks the
// FIRST adjacent sign-changing pair whose midpoint is closest to `s_in`
// (Newton's last s, or the seed for a pure-fallback call -- task spec:
// "keep it simpler and deterministic"). If no pair changes sign, reports
// the single closest-to-zero scanned point instead so the caller's
// failed_no_bracket result still carries N_scan-point evidence.
//
// Design note (measured, not the first thing tried): interior points were
// originally ALL placed by tracking s_in's fractional position within its
// own local srange across every sampled w (one "percentile track" through
// density space) -- this is exactly right when s_in is already close to a
// root (a warm-started Newton that stalled on a near-singular Jacobian, not
// a bad guess -- see the "S7-8 damped 2x2 Newton" comment), but measurably
// WRONG when s_in is a crude, untargeted cold seed (e.g. every state in a
// forced max_iter_newton=0 call): the wrong fraction gets propagated to
// EVERY interior point, and the scan can miss even a single globally
// monotone root entirely (regressed tests/test_con2prim.cpp's forced-
// fallback and cold-slow-precision cases in exactly this way during
// development). The half-global/half-local split is robust to both: the
// global half guarantees the single-root synthetic-table case is found
// regardless of how good s_in is, while the local half still concentrates
// resolution near a likely-good s_in for the narrow real-table features
// that motivated this scan in the first place.
EEOS_HOST_DEVICE inline BracketScanResult c2p_bracket_scan(const EntropyEOSView &eos, real D, real tau,
                                                             real ye, real S_par, real S_perp, real B2,
                                                             real w_max, real tau_floor_rel, int n_scan_opt,
                                                             real s_in, real w_in, real w_seed0, real u_seed0,
                                                             int max_iter_1d, real tol) {
  int n = n_scan_opt < 4 ? 4 : n_scan_opt;
  if (n > kC2PBracketScanMax) n = kC2PBracketScanMax;

  const real rho_a = D;                  // w = 0
  const real rho_b = D / std::cosh(w_max); // w = w_max

  const SRange ext_a = eos.srange_extended(rho_a, ye);
  const SRange ext_b = eos.srange_extended(rho_b, ye);
  const real s_ext_lo = ext_a.s_min < ext_b.s_min ? ext_a.s_min : ext_b.s_min;
  const real s_ext_hi = ext_a.s_max > ext_b.s_max ? ext_a.s_max : ext_b.s_max;

  const SRange phys_a = eos.srange(rho_a, ye);
  const SRange phys_b = eos.srange(rho_b, ye);
  const real s_phys_lo = phys_a.s_min < phys_b.s_min ? phys_a.s_min : phys_b.s_min;
  const real s_phys_hi = phys_a.s_max > phys_b.s_max ? phys_a.s_max : phys_b.s_max;

  real s_cand[kC2PBracketScanMax];
  int nc = 0;
  s_cand[nc++] = s_ext_lo;
  s_cand[nc++] = s_phys_lo;

  const int n_interior = n - 4;
  const int n_global = (n_interior + 1) / 2; // ceil half: GLOBAL safety net
  const int n_local = n_interior - n_global; // LOCAL targeted window

  // GLOBAL half: uniform in s across the full physical range -- finds any
  // single monotone root (the synthetic-table case) regardless of how good
  // s_in is.
  for (int i = 0; i < n_global; ++i) {
    const real frac = n_global > 1 ? static_cast<real>(i) / static_cast<real>(n_global - 1) : real(0.5);
    s_cand[nc++] = s_phys_lo + frac * (s_phys_hi - s_phys_lo);
  }

  // LOCAL half (M3h): a GEOMETRIC ladder of RELATIVE offsets from s_in --
  // points at s_in*(1 +- delta_i), delta_i log-spaced over [kC2PScanDeltaMin,
  // kC2PScanDeltaMax] with the sides ALTERNATING, clamped into the extended
  // bracket.
  //
  // Through M3g this half was linear and sized to s_in's own local PHYSICAL
  // srange width (+-0.6 of it). MEASURED failure of that sizing (LS220/SRO,
  // 40k warm + 4k cold audit states): at the radiation-dominated low-density
  // states where the residual failed_no_bracket class lived, the physical
  // srange spans 13-14.5 decades -- a span of ~5e13 around an s_in of ~1e9 --
  // so consecutive LOCAL points sat ~5e12 apart while g(s)'s sign-changing
  // window around the true root was 1e-7 ... 1e-3 RELATIVE to s_in. The
  // "local" window was thus five orders of magnitude coarser than the feature
  // it existed to resolve, and its points were, in relative terms, no more
  // local than the global half's. A relative ladder is the natural sizing:
  // s_in is a good anchor exactly when Newton stalled near the root, and
  // "near" is a relative statement (measured on the M3g failure set: the
  // stalled iterates sat 3e-5 ... 1e-2 relative to their root).
  //
  // The ladder is emitted in PAIRS (+delta, -delta), so BOTH sides span the
  // full [kC2PScanDeltaMin, kC2PScanDeltaMax] and the innermost pair
  // straddles s_in. A one-sided ladder with alternating signs (which halves
  // the ladder step, at the price of halving each side's REACH) was tried
  // first and measured worse: SRO audit state k = 17457 has its root at
  // +0.157 relative to the anchor, inside the pair ladder's outermost gap but
  // beyond the alternating ladder's positive reach of 0.066, and only the
  // pair form recovers it. Coverage beats resolution here because the outer
  // Illinois solve resolves a wide bracket cheaply, whereas a missed root is
  // a failed_no_bracket outright.
  //
  // s_in <= 0 or non-finite falls back to the old span-based window: s can
  // legitimately be at or below 0 (cold states on tables whose entropy axis
  // reaches 0), where a multiplicative offset degenerates.
  {
    const bool relative_ok = s_in > real(0) && c2p_is_finite(s_in);
    if (relative_ok) {
      const int n_pair = (n_local + 1) / 2; // distinct |delta| values
      // One std::pow for the ladder ratio, then a running multiply -- this is
      // the fallback path (it already pays n_scan full inner w-solves), but
      // there is no reason to spend a transcendental per point.
      const real ratio = n_pair > 1 ? std::pow(kC2PScanDeltaMax / kC2PScanDeltaMin,
                                                real(1) / static_cast<real>(n_pair - 1))
                                    : real(1);
      real delta = kC2PScanDeltaMin;
      for (int i = 0, emitted = 0; i < n_pair && emitted < n_local; ++i) {
        s_cand[nc++] = aeval_clamp(s_in * (real(1) + delta), s_ext_lo, s_ext_hi);
        ++emitted;
        if (emitted < n_local) { // odd n_local spends its last point on the + side
          s_cand[nc++] = aeval_clamp(s_in * (real(1) - delta), s_ext_lo, s_ext_hi);
          ++emitted;
        }
        delta *= ratio;
      }
    } else {
      // Pre-M3h sizing, kept verbatim for the degenerate anchor: a dense
      // window centered on s_in, spanning +-0.6 of s_in's own local physical
      // srange width at w_in, clamped into the extended bracket.
      const real w_in_c = aeval_clamp(w_in, real(0), w_max);
      const real rho_in = D / std::cosh(w_in_c);
      const SRange sr_in = eos.srange(rho_in, ye);
      const real half_width = real(0.6) * (sr_in.s_max - sr_in.s_min);
      real lo = s_in - half_width;
      real hi = s_in + half_width;
      lo = aeval_clamp(lo, s_ext_lo, s_ext_hi);
      hi = aeval_clamp(hi, s_ext_lo, s_ext_hi);
      for (int i = 0; i < n_local; ++i) {
        const real frac = n_local > 1 ? static_cast<real>(i) / static_cast<real>(n_local - 1) : real(0.5);
        s_cand[nc++] = lo + frac * (hi - lo);
      }
    }
  }

  s_cand[nc++] = s_phys_hi;
  s_cand[nc++] = s_ext_hi;

  // Inner solves at FULL precision (opts.tol / opts.max_iter_1d), not a
  // loosened scan-only tolerance -- TRIED, MEASURED, REVERTED (task's "a
  // coarser tol is fine for sign determination; document" suggestion). A
  // loosened tolerance (1e-6, capped at 20 inner iterations) measurably
  // produced WRONG signs exactly where it matters most: near a genuine
  // near-degenerate g (|g| ~ 1e-9, i.e. close to a root), an under-
  // converged w-solve's residual f1 error propagates into f2 and can flip
  // its sign -- reproduced on the synthetic table (a cold state whose true
  // root sits at g~-3e-10: the coarse solve reported +1.6e-9 one scan point
  // over, manufacturing a false bracket that masked the real one and
  // regressed tests/test_con2prim.cpp test 3's cold subset from 0 to 1
  // failed_no_bracket). Each inner solve here is the same bisection-
  // safeguarded-Newton c2p_inner_solve_w() the endpoint solves already use
  // at full precision (S9's strict f1-monotonicity proof means it is
  // well-conditioned), so scanning `n` points at full precision costs `n`
  // ordinary inner solves -- more than the old 2-endpoint check, but this
  // only runs on the already-slow fallback path (never on the warm-Newton
  // path that dominates throughput).
  real g_cand[kC2PBracketScanMax];
  real w_seed = w_seed0, u_seed = u_seed0;
  for (int i = 0; i < nc; ++i) {
    int it = 0;
    const Residuals r = c2p_inner_solve_w(eos, D, tau, ye, S_par, S_perp, B2, s_cand[i], w_max,
                                           tau_floor_rel, w_seed, u_seed, max_iter_1d, tol, it);
    (void)it;
    g_cand[i] = r.f2;
    w_seed = r.w;
    u_seed = r.pt.u_solved;
  }

  c2p_sort_scan(s_cand, g_cand, nc);

  BracketScanResult out;
  out.bracketed = false;
  out.s_lo = out.s_hi = out.s_best = s_cand[0];

  int pick = -1;
  real best_dist = real(0);
  for (int i = 0; i + 1 < nc; ++i) {
    const real glo = g_cand[i];
    const real ghi = g_cand[i + 1];
    const bool sign_change = (glo <= real(0) && ghi >= real(0)) || (glo >= real(0) && ghi <= real(0));
    if (!sign_change) continue;
    const real mid = real(0.5) * (s_cand[i] + s_cand[i + 1]);
    const real dist = std::fabs(mid - s_in);
    if (pick < 0 || dist < best_dist) {
      pick = i;
      best_dist = dist;
    }
  }

  if (pick >= 0) {
    out.bracketed = true;
    out.s_lo = s_cand[pick];
    out.s_hi = s_cand[pick + 1];
  } else {
    int best_i = 0;
    real best_g = std::fabs(g_cand[0]);
    for (int i = 1; i < nc; ++i) {
      const real ag = std::fabs(g_cand[i]);
      if (ag < best_g) {
        best_g = ag;
        best_i = i;
      }
    }
    out.s_best = s_cand[best_i];
  }
  return out;
}

// ===========================================================================
// M3d cold-start seed
// ===========================================================================

// Iteration cap for the seed's inner EOS-FREE scalar solve
// (c2p_seed_z_solve()). Generous because that solve touches no EOS at all --
// a handful of multiplies per step on a bracketed monotone residual -- so
// its cost is invisible next to the s-recovery's evaluate() calls, and
// running it to convergence is exactly what the seed's accuracy rests on.
constexpr int kC2PSeedScalarIters = 40;

// What c2p_cold_seed() hands back: the (s, w) the Newton starts from, the
// EOS warm start u that came with them, plus the seed's own rho/z (exposed
// for the seed-quality unit test in tests/test_con2prim.cpp and for scratch
// probes -- POD, no cost).
struct ColdSeed {
  real s, w, u;
  real rho, z;
};

// Recovers s from U(rho, s, ye) = eps by a bisection-safeguarded Newton on
// the srange_extended() bracket. This CANNOT fail, which is the reason the
// seed is built around it: U_s = That > 0 everywhere the adapter is defined,
// including both designed extension tails (adapter_eval.hpp's monotonicity
// guard on the u-direction primary track of sigma and L is exactly what
// makes That > 0 there too), so U is strictly monotone in s on
// [s_min_ext, s_max_ext] and the bracket is a genuine invariant --
// eps outside [U(s_min_ext), U(s_max_ext)] simply drives the iteration onto
// the corresponding endpoint instead of diverging. Same shape as
// adapter_eval.hpp's own T-solve: Newton where the derivative is usable and
// the step stays inside the bracket, plain bisection otherwise.
//
// n_iter is a CAP, not a fixed trip count: the |g| test below exits early on
// the CPU (typically 4-7 evaluates once warm-started along the seed's own
// pass chain). A GPU port that wants a branch-free fixed cost can simply
// drop the two breaks and always run n_iter times -- nothing else in the
// loop is data-dependent.
EEOS_HOST_DEVICE inline real c2p_seed_s_solve(const EntropyEOSView &eos, real rho, real ye, real eps,
                                               real u_prev, int n_iter, EOSPoint &pt_out) {
  const SRange ext = eos.srange_extended(rho, ye);
  real lo = ext.s_min, hi = ext.s_max;
  if (!(lo < hi)) { // degenerate bracket (never seen on a real table; guarded)
    pt_out = eos.evaluate(rho, lo, ye, u_prev);
    return lo;
  }

  const real eps_scale = eps > real(1e-300) ? eps : real(1e-300);
  real s = real(0.5) * (lo + hi);
  EOSPoint pt = eos.evaluate(rho, s, ye, u_prev);

  for (int i = 0; i < n_iter; ++i) {
    const real g = pt.U - eps;
    // U increasing in s: g<0 means the root is further right (raise lo).
    if (g < real(0)) {
      lo = s;
    } else {
      hi = s;
    }
    if (std::fabs(g) <= real(1e-12) * eps_scale) break;

    real s_next;
    bool newton_ok = false;
    if (pt.U_s > real(0)) {
      s_next = s - g / pt.U_s;
      newton_ok = c2p_is_finite(s_next) && s_next > lo && s_next < hi;
    }
    if (!newton_ok) s_next = real(0.5) * (lo + hi);
    if (s_next == s || !(lo < hi)) break; // bracket collapsed to adjacent doubles

    s = s_next;
    pt = eos.evaluate(rho, s, ye, pt.u_solved);
  }

  pt_out = pt;
  return s;
}

// The seed's inner ENERGY solve. Given the pressure p, the S5 energy
// relation and the S5 perpendicular momentum projection close on each other
// with NO reference to the rapidity at all:
//
//   z = E + p - B^2/2 * (1 + v_perp^2),      v_perp = S_perp/(z + B^2)
//
// (the two other magnetic terms of S5's E cancel into that single
// (1 + v_perp^2), see prim2con.hpp's algebra note). In the variable
// q = z + B^2 this is the cubic
//
//   H(q) = q - A + (B^2 * S_perp^2)/(2 q^2) = 0,   A = E + p + B^2/2,
//
// solved here by bisection-safeguarded Newton. H has a single minimum at
// q_branch = cbrt(B^2 S_perp^2) and is strictly increasing above it; the
// physical root always lies on that increasing branch, since
// q_branch/q = (B^2 v_perp^2/q)^(1/3) < 1. So the bracket is
//   q_lo = max(D + B^2, q_branch)   (z >= D exactly: z = D*h*cosh w, h >= 1)
//   q_hi = max(q_lo, A)             (H(A) = B^2 S_perp^2/(2A^2) >= 0)
// -- both endpoints exact physical bounds, no tuned constants. For B^2 = 0
// this degenerates to the linear H(q) = q - (E + p), i.e. the design doc's
// exact hydro z = E + p, recovered rather than special-cased.
//
// EOS-free (pure arithmetic), so the iteration cap is generous.
EEOS_HOST_DEVICE inline real c2p_seed_z_solve(real D, real E, real p, real S_perp, real B2, int n_iter) {
  const real A = E + p + real(0.5) * B2;
  const real half_num = real(0.5) * B2 * S_perp * S_perp; // H's numerator over q^2

  // q_branch = cbrt(B^2 * S_perp^2), formed factor-wise so an intermediate
  // product cannot overflow for the ~1e19-scale magnitudes real tables reach.
  const real q_branch = std::cbrt(B2) * std::cbrt(S_perp) * std::cbrt(S_perp);
  real lo = D + B2;
  if (q_branch > lo) lo = q_branch;
  real hi = A > lo ? A : lo;

  // Start AT q_hi = A rather than at the bracket midpoint. Two reasons, both
  // load-bearing: (a) for B^2 = 0 the residual is exactly H(q) = q - A, so
  // this first point IS the root and the |H| test below exits with the exact
  // hydro z = E + p -- starting from the midpoint instead would reject every
  // Newton step (it lands precisely on the open bracket's endpoint) and leave
  // the answer to 40 bisections of an O(tau)-wide interval, i.e. ~1e-12
  // relative, which is not enough for a residual the solver then measures
  // against a 1e-12 tolerance; (b) for B^2 > 0, A over-estimates the root
  // (H(A) >= 0), which is the side Newton descends from monotonically.
  real q = hi;
  if (hi > lo) {
    // H(lo) > 0 would mean no root on the increasing branch (an inconsistent
    // lagged p); the bisection below then simply converges to `lo`, the
    // physical floor, which is the right answer to report.
    for (int i = 0; i < n_iter; ++i) {
      const real H = q - A + half_num / (q * q);
      if (H < real(0)) {
        lo = q;
      } else {
        hi = q;
      }
      if (std::fabs(H) <= real(1e-14) * A) break;
      const real dH = real(1) - real(2) * half_num / (q * q * q);
      real q_next;
      bool newton_ok = false;
      if (dH > real(0)) {
        q_next = q - H / dH;
        newton_ok = c2p_is_finite(q_next) && q_next > lo && q_next < hi;
      }
      if (!newton_ok) q_next = real(0.5) * (lo + hi);
      if (q_next == q || !(lo < hi)) break;
      q = q_next;
    }
  }

  const real z = q - B2;
  return z > D ? z : D; // z = D*h*cosh(w) >= D exactly
}

// M3d B^2-aware cold-start seed (module header item 6 for the diagnosis this
// replaces). A fixed-cost block iteration built so that EVERY step is an
// exact design-doc relation and the ONLY lagged quantity is the pressure p.
// Per pass, given p (0 on the first pass):
//
//   1. ENERGY -> z, exactly, via c2p_seed_z_solve() above. This is the step
//      that cures the B^2-blindness, and it has to be solved rather than
//      estimated: the magnetic contribution to E is B^2/2*(1+v_perp^2),
//      known only to a factor of two a priori, and for a magnetically
//      dominated state that factor of two is many times z itself. (Measured:
//      substituting the midpoint z ~ E - 0.75*B^2 for this solve puts z 13%
//      off on a sigma = 2.7e3 LS220 state, which lands w at 1.6 instead of
//      4.9 -- see item 2's sensitivity note. The whole seed is only as good
//      as this solve.)
//
//   2. MOMENTUM -> w, exactly, from S5's two projections:
//        tanh^2(w) = (S_par/z)^2 + (S_perp/(z+B^2))^2.
//      Given the exact z this is exact, and its sensitivity to the ONE
//      lagged input is bounded and small: dw = tanh(w) * dp/(rho*h) at
//      leading order, and p/(rho*h) < 1/4 even for a radiation-dominated
//      state, so even the p = 0 first pass lands within ~0.25 in w
//      ABSOLUTE, with later passes second order. (This bound is why the
//      seed iterates on p and not on the enthalpy: closing the loop through
//      h instead gives sinh(w) = |S|/(D*h) for B^2 = 0, i.e.
//      dw = d(log h), and an h-lagged iteration started from the exact bound
//      h >= 1 was MEASURED to be useless -- on radiation-dominated LS220
//      states h ~ 10^4 puts the first w at w_max, where the kinetic term
//      2*D*sinh^2(w/2) alone exceeds tau, so eps floors at 0, s floors at
//      s_min_ext, h comes back as 1 and the iteration sits in that spurious
//      fixed point: 198/400 cold failures, worse than no seed at all.)
//
//   3. ENERGY -> eps (S7, exact given p). Solving the cancellation-free tau
//      identity for the one term that carries eps,
//        rho*eps*W^2 = tau - 2*D*sinh^2(w/2) - p*sinh^2(w)
//                      - B^2/2*(1+V^2) + B^2/2*v_par^2,
//      and dividing by rho*W^2 = D*cosh(w). A negative right-hand side means
//      the lagged p overshot, i.e. the state is colder than this iterate can
//      express; eps is floored at 0, which step 4 turns into "the coldest s
//      this density can represent" -- the correct direction to be wrong in,
//      and a floor the adapter's U >= 0 over the whole extended box makes
//      meaningful rather than arbitrary.
//
//   4. ENTROPY -> s (exact inversion of the table), by the monotone bracketed
//      solve of c2p_seed_s_solve(). Load-bearing: it is the only step that
//      CANNOT fail, and it is what replaces the old "midpoint of srange"
//      guess that put s in the wrong basin.
//
//   5. p <- EOSPoint::p for the next pass.
//
// The returned (s, w) is self-consistent by construction -- s was solved at
// rho = D/cosh(w) of the SAME pass -- which matters because con2prim()'s
// Newton and the S9 bracket scan both consume the pair, not either half
// alone. No allocation, no recursion; per pass, two EOS-free scalar solves
// plus one srange_extended() and a handful of evaluate() calls.
EEOS_HOST_DEVICE inline ColdSeed c2p_cold_seed(const EntropyEOSView &eos, real D, real tau, real ye,
                                                real S_par, real S_perp, real B2, real w_max, int n_pass,
                                                int n_s_iter) {
  const real E = tau + D;
  const real S_perp_pos = S_perp > real(0) ? S_perp : real(0);
  const real V_max = std::tanh(w_max); // the only "cap", and it is opts.w_max itself

  const int npass = n_pass < 1 ? 1 : (n_pass > 8 ? 8 : n_pass);
  const int nsit = n_s_iter < 2 ? 2 : (n_s_iter > 40 ? 40 : n_s_iter);

  real p = real(0);        // the one lagged quantity: 0 on the first pass
  real u_prev = p2c_nan(); // EOS T-solve warm start, threaded across passes

  ColdSeed out;
  out.s = real(0);
  out.w = real(0);
  out.u = p2c_nan();
  out.rho = D;
  out.z = D;

  for (int pass = 0; pass < npass; ++pass) {
    // --- 1. energy -> z ----------------------------------------------------
    const real z = c2p_seed_z_solve(D, E, p, S_perp_pos, B2, kC2PSeedScalarIters);
    const real q = z + B2;

    // --- 2. momentum -> w --------------------------------------------------
    real v_par = S_par / z;
    const real v_perp = S_perp_pos / q;
    real V = std::sqrt(v_par * v_par + v_perp * v_perp);
    if (!c2p_is_finite(V)) V = real(0);
    // V >= V_max means the lagged p is not yet consistent with a state of
    // rapidity <= w_max; report w_max and let the next pass's p fix it.
    const real w = V < V_max ? aeval_clamp(std::atanh(V), real(0), w_max) : w_max;

    const real coshw = std::cosh(w);
    const real sinhw = std::sinh(w);
    const real half_sinh = std::sinh(real(0.5) * w);
    const real rho = D / coshw;

    // Re-express the velocity at the rapidity actually adopted. A no-op on
    // the ordinary path (V < V_max makes Vc = tanh(atanh(V)) = V), but it
    // keeps step 3's magnetic terms consistent with w on the capped branch,
    // where V itself may exceed 1. Only v_par needs rescaling: step 3 uses
    // the split solely through Vc (= |v|) and v_par, the (B.v)^2 term.
    const real Vc = std::tanh(w);
    if (V > real(0)) v_par *= Vc / V;

    // --- 3. energy -> eps --------------------------------------------------
    const real rhoW2 = D * coshw; // rho*W^2, with rho = D/cosh(w)
    real eps = (tau - real(2) * D * half_sinh * half_sinh - p * sinhw * sinhw -
                real(0.5) * B2 * (real(1) + Vc * Vc) + real(0.5) * B2 * v_par * v_par) /
               rhoW2;
    if (!(eps > real(0))) eps = real(0); // also catches a non-finite quotient

    // --- 4. entropy -> s ---------------------------------------------------
    EOSPoint pt;
    const real s = c2p_seed_s_solve(eos, rho, ye, eps, u_prev, nsit, pt);
    u_prev = pt.u_solved;

    out.s = s;
    out.w = w;
    out.u = pt.u_solved;
    out.rho = rho;
    out.z = z;

    // --- 5. p for the next pass --------------------------------------------
    p = (c2p_is_finite(pt.p) && pt.p > real(0)) ? pt.p : real(0);
  }

  return out;
}

// Fills the shared Con2PrimOut fields from a converged/best-effort
// Residuals; `out.ye` and `out.result` are set separately by the caller.
EEOS_HOST_DEVICE inline void c2p_fill(Con2PrimOut &out, const Residuals &r, int iters_newton,
                                       int iters_fallback) {
  out.rho = r.rho;
  out.s = r.s;
  out.w = r.w;
  out.W = std::cosh(r.w);
  out.v_par = r.v_par;
  out.v_perp = r.v_perp;
  out.eos = r.pt;
  out.iters_newton = iters_newton;
  out.iters_fallback = iters_fallback;
  out.flags = r.pt.flags;
}

} // namespace detail

// Recovers primitives from a scalar conservative state (design doc SS6-9).
// s_guess/w_guess/u_guess default to NaN ("no guess"); when finite they
// warm-start the corresponding piece, otherwise the M3d B^2-aware cold seed
// of detail::c2p_cold_seed() supplies it (see that function's doc comment
// for the construction, and the module header's item 6 for the measured
// diagnosis it was built from).
//
// Historical note, kept because it is the reason the M3d seed is built the
// way it is: through M3a-M3c this was a crude B^2-blind estimate
// (tanh w0 = |S|/(tau+D+0.3*tau), s0 = midpoint of srange(D/cosh w0)),
// documented as "just a seed" on the theory that only iteration count
// depended on its quality. That theory was measurably false for strongly
// magnetized states (module header item 6). M3c then tried a partial fix --
// eps0 from the tau identity plus s0 chosen from 4 quarter-points of
// srange(rho0) by matching U to eps0 -- and REVERTED it: a 4-point grid's
// best candidate is still ~12% of the way across a physical srange that can
// span many orders of magnitude, so on the coldest-row edge case
// (tests/test_con2prim.cpp test 6) it landed far from the near-s_min truth
// in absolute terms, and the bracket scan's local window then inherited that
// wrong position. Both failures are addressed here by construction rather
// than by tuning: the s recovery is an exact monotone solve on the whole
// extended bracket (not a coarse grid search), so it is as accurate at
// s_min as anywhere else, and the seed's rho is B^2-aware, so the window it
// anchors is centred on the right basin.
EEOS_HOST_DEVICE inline Con2PrimOut con2prim(const EntropyEOSView &eos, const Con2PrimIn &in,
                                              const Con2PrimOptions &opts,
                                              real s_guess = detail::p2c_nan(),
                                              real w_guess = detail::p2c_nan(),
                                              real u_guess = detail::p2c_nan()) {
  using detail::Residuals;

  Con2PrimOut out{};
  out.ye = in.D_Y / in.D;

  // --- M3d cold-start seed -------------------------------------------------
  // Computed only when at least one of (s, w) is missing; a fully warm call
  // never pays for it and is bit-identical to the pre-M3d solver.
  const bool cold_s = detail::aeval_is_nan(s_guess);
  const bool cold_w = detail::aeval_is_nan(w_guess);

  real s = s_guess;
  real w = cold_w ? real(0) : detail::aeval_clamp(w_guess, real(0), opts.w_max);
  real u_start = u_guess;

  if (cold_s || cold_w) {
    const detail::ColdSeed seed =
        detail::c2p_cold_seed(eos, in.D, in.tau, out.ye, in.S_par, in.S_perp, in.B2, opts.w_max,
                               opts.seed_passes, opts.seed_s_iters);
    if (cold_s) s = seed.s;
    if (cold_w) w = detail::aeval_clamp(seed.w, real(0), opts.w_max);
    // The seed's converged EOS T-solve position is also the best available
    // warm start for the Newton's first evaluate(), so pass it on unless the
    // caller supplied one of its own.
    if (detail::aeval_is_nan(u_start)) u_start = seed.u;
  }

  // --- S7-8 damped 2x2 Newton ----------------------------------------------
  //
  // "Damped" here means clamped, not backtracked: an earlier version of this
  // loop additionally backtracked (halved the step up to 4x, as the task
  // spec's literal wording suggests) whenever ||f||_2 did not decrease, and
  // fell back to the full step only when no halving improved either.
  // Measured on the round-trip suite (tests/test_con2prim.cpp test 3) this
  // was actively harmful: near a locally ill-conditioned Jacobian (the two
  // residuals' gradients nearly parallel -- det ~ 0 relative to the
  // individual products |a*d|, |b*c|; this can happen at points the
  // iteration passes through even though the true root is regular) a halved
  // step barely moves, so the iteration stalls for many iterations in the
  // bad region, while the FULL clamped step -- though it may transiently
  // increase ||f||_2 -- actually crosses the ill-conditioned region and
  // recovers quadratic convergence on the far side. Concretely, this raised
  // the warm-start round-trip's converged_newton fraction from 86% to 99%
  // (test 3a's acceptance bar) with no change in accuracy for the states
  // that already converged either way. So this loop only clamps the step
  // (|dw| <= 1, |ds| <= 0.25*extended-span, w >= 0) and takes it
  // unconditionally; the SS9 fallback (guaranteed globally convergent) is
  // the safety net for the states that still do not converge within
  // max_iter_newton, exactly as the design doc intends.
  //
  // M3h adds exit-state bookkeeping only (no extra EOS evaluation, no change
  // to the step): the loop remembers the iterate with the smallest SCALED
  // residual norm max(|f1|/cosh w, |f2|) it ever visited, and hands THAT to
  // the S9 fallback below instead of the last one. Unconditional clamped
  // steps are exactly what makes this necessary -- an iterate oscillating at
  // the precision floor (or one that crossed an ill-conditioned region and
  // is mid-excursion when the budget runs out) leaves `r` far from the best
  // position the trajectory reached, and that stale position was then used
  // both to anchor the bracket scan and as the reported failure state
  // (measured pre-M3g on LS220 k=36066: f2 = 1.1e-3 at return, against
  // <= 1e-10 along the trajectory).
  Residuals r = detail::c2p_eval(eos, in.D, in.tau, out.ye, in.S_par, in.S_perp, in.B2, s, w, u_start,
                                  opts.tau_floor_rel);
  int iters = 0;
  bool converged = detail::c2p_f1_converged(r, opts.tol) && std::fabs(r.f2) <= opts.tol;

  // best_norm may start out NaN (a non-finite first residual); the update
  // below is written as !(n >= best_norm) precisely so that case is replaced
  // by the first finite iterate rather than latched forever.
  Residuals r_best = r;
  real best_norm = detail::c2p_scaled_norm(r);

  while (!converged && iters < opts.max_iter_newton) {
    ++iters;

    // Solve [[df1_ds, df1_dw], [df2_ds, df2_dw]] * [ds, dw]^T = -[f1, f2]^T
    // by Cramer's rule.
    const real a = r.df1_ds, b = r.df1_dw, c = r.df2_ds, d = r.df2_dw;
    const real det = a * d - b * c;
    if (det == real(0) || !detail::c2p_is_finite(det)) break; // singular Jacobian -> S9 fallback below

    real ds = (b * r.f2 - d * r.f1) / det;
    real dw = (c * r.f1 - a * r.f2) / det;

    // Step limits: |dw| <= 1; |ds| <= 0.25*(extended s-span) at the current rho.
    const SRange sr_ext = eos.srange_extended(r.rho, out.ye);
    const real ds_max = real(0.25) * (sr_ext.s_max - sr_ext.s_min);
    ds = detail::aeval_clamp(ds, -ds_max, ds_max);
    dw = detail::aeval_clamp(dw, real(-1), real(1));

    const real w_raw = w + dw;
    const real w_next = w_raw < real(0) ? detail::kC2PTinyW : detail::aeval_clamp(w_raw, real(0), opts.w_max);
    const real s_next = s + ds;
    const Residuals r_next = detail::c2p_eval(eos, in.D, in.tau, out.ye, in.S_par, in.S_perp, in.B2, s_next,
                                               w_next, r.pt.u_solved, opts.tau_floor_rel);

    if (!detail::c2p_is_finite(r_next.f1) || !detail::c2p_is_finite(r_next.f2)) break; // -> S9 fallback

    s = s_next;
    w = w_next;
    r = r_next;
    converged = detail::c2p_f1_converged(r, opts.tol) && std::fabs(r.f2) <= opts.tol;

    const real n_now = detail::c2p_scaled_norm(r);
    if (detail::c2p_is_finite(n_now) && !(n_now >= best_norm)) {
      r_best = r;
      best_norm = n_now;
    }
  }

  if (converged) {
    out.result = C2PResult::converged_newton;
    detail::c2p_fill(out, r, iters, 0);
    return out;
  }

  // --- S9 nested 1D fallback -------------------------------------------------
  // M3c bracket scan (see this function's fallback-section doc comment
  // above for the measured diagnosis and design): the anchor below is the
  // incoming iterate -- Newton's BEST visited position (M3h; the last one
  // through M3g), or the untouched seed if max_iter_newton==0 -- used both
  // to anchor the scan's interior points and to break ties among multiple
  // sign-changing intervals.
  //
  // M3h: `r_best` is bit-identical to `r` whenever the Newton trajectory
  // improved monotonically in the scaled norm (the common case), so for
  // those states this whole section is unchanged; it differs exactly for the
  // oscillating/excursion trajectories the fallback exists to rescue, where
  // the last iterate is a poor anchor for a scan whose local half is now
  // sized RELATIVE to it (see c2p_bracket_scan()).
  //
  // M3d asked whether a COLD call should instead anchor the scan's LOCAL
  // half on the seed itself, on the theory that a cold Newton which ran out
  // of iterations has not "stalled near a root" (the case the local window
  // was designed for) but simply wandered. MEASURED and NOT adopted: over
  // 4000 cold states per table the two anchors are indistinguishable --
  // LS220 18 vs 20 failed, SRO 32 vs 31 -- so the extra state is not worth
  // carrying. The reason they tie is itself a property of the M3d seed:
  // with a seed accurate to ~1e-4 in s, a cold Newton either converges (it
  // does for 99% of cold states, see the module header) or ends up near
  // where it started, so "Newton's last iterate" and "the seed" are the
  // same neighbourhood -- and for max_iter_newton == 0 they are literally
  // the same value.
  const real s_anchor = r_best.s;
  const real w_anchor = r_best.w;
  const real u_anchor = r_best.pt.u_solved;

  const detail::BracketScanResult scan =
      detail::c2p_bracket_scan(eos, in.D, in.tau, out.ye, in.S_par, in.S_perp, in.B2, opts.w_max,
                                opts.tau_floor_rel, opts.bracket_scan, s_anchor, w_anchor, w_anchor,
                                u_anchor, opts.max_iter_1d, opts.tol);

  if (!scan.bracketed) {
    int best_it = 0;
    const Residuals best = detail::c2p_inner_solve_w(eos, in.D, in.tau, out.ye, in.S_par, in.S_perp, in.B2,
                                                       scan.s_best, opts.w_max, opts.tau_floor_rel, w_anchor,
                                                       u_anchor, opts.max_iter_1d, opts.tol, best_it);
    (void)best_it;
    out.result = C2PResult::failed_no_bracket;
    // M3h: report whichever of the scan's best point and the Newton's best
    // iterate sits closer to a root in the scaled norm. The scan point is a
    // converged f1 root by construction but can sit far from the true s,
    // while a Newton that stalled near the root is often the better answer
    // for a caller that inspects the failed state (and for the M3e policy
    // layer, which repairs from it).
    const Residuals &rep = detail::c2p_scaled_norm(r_best) < detail::c2p_scaled_norm(best) ? r_best : best;
    detail::c2p_fill(out, rep, iters, 0);
    return out;
  }

  // Re-solve at the scan-chosen interval's endpoints, seeded from the
  // ORIGINAL incoming (w, u_solved) rather than whatever the scan's own
  // internal warm-start chain last held (c2p_bracket_scan() only returns
  // the winning s-values, not its Residuals, to keep its stack footprint
  // small -- see that function's doc comment) -- both solves are already
  // full precision (c2p_bracket_scan() no longer loosens tolerance; see its
  // evaluation loop's own comment for why), so this reproduces the scan's
  // own (s_lo,g_lo)/(s_hi,g_hi) values, just from a fresh warm start, before
  // handing off to the existing Illinois/stagnation/polish machinery,
  // unchanged below.
  int ia = 0, ib = 0;
  Residuals r_lo = detail::c2p_inner_solve_w(eos, in.D, in.tau, out.ye, in.S_par, in.S_perp, in.B2,
                                              scan.s_lo, opts.w_max, opts.tau_floor_rel, w_anchor, u_anchor,
                                              opts.max_iter_1d, opts.tol, ia);
  Residuals r_hi = detail::c2p_inner_solve_w(eos, in.D, in.tau, out.ye, in.S_par, in.S_perp, in.B2,
                                              scan.s_hi, opts.w_max, opts.tau_floor_rel, r_lo.w,
                                              r_lo.pt.u_solved, opts.max_iter_1d, opts.tol, ib);
  (void)ia;
  (void)ib;

  real slo = scan.s_lo, shi = scan.s_hi, glo = r_lo.f2, ghi = r_hi.f2;
  const bool bracketed = (glo <= real(0) && ghi >= real(0)) || (glo >= real(0) && ghi <= real(0));

  if (!bracketed) {
    // Extremely rare (both solves are full precision, so this would mean
    // the fresh warm start above landed on a genuinely different g(s) --
    // not observed with the default budgets, but reachable with a crippled
    // max_iter_1d, and guarded defensively): same best-of-two-endpoints
    // report as the ordinary no-bracket path, and (M3h) the same
    // best-Newton-iterate comparison, so EVERY failure exit reports a state
    // at least as good as the trajectory's best.
    out.result = C2PResult::failed_no_bracket;
    const Residuals &best = std::fabs(glo) <= std::fabs(ghi) ? r_lo : r_hi;
    const Residuals &rep = detail::c2p_scaled_norm(r_best) < detail::c2p_scaled_norm(best) ? r_best : best;
    detail::c2p_fill(out, rep, iters, 0);
    return out;
  }

  Residuals r_cur = std::fabs(glo) <= std::fabs(ghi) ? r_lo : r_hi;
  bool outer_converged = std::fabs(glo) <= opts.tol || std::fabs(ghi) <= opts.tol;
  Residuals r_bracket_lo = r_lo, r_bracket_hi = r_hi; // Residuals matching (slo,glo)/(shi,ghi)

  real w_seed = r_hi.w, u_seed = r_hi.pt.u_solved;
  int outer_iters = 0;

  // Illinois-modified regula falsi (safeguarded secant/bisection, S9): plain
  // regula falsi -- always interpolating between the two current BRACKET
  // endpoints -- stalls badly whenever g is strongly convex/concave across
  // the bracket, since one endpoint then never gets replaced and the
  // interval barely shrinks each step (observed empirically: >>60
  // iterations to converge on some states). The Illinois fix: whenever the
  // same side is retained twice in a row, halve the OTHER (stale) side's
  // function value before the next secant step -- this preserves the sign
  // (so the bracket and the convergence check are unaffected) while pulling
  // the secant estimate toward the stale side, restoring superlinear
  // convergence. Falls back to plain bisection whenever the secant point
  // would land outside the current bracket.
  int stale_side = 0; // 0 = neither side just repeated, 1 = lo just replaced twice, 2 = hi just replaced twice
  real prev_s_next = detail::p2c_nan();
  while (!outer_converged && outer_iters < opts.max_iter_1d) {
    ++outer_iters;

    real s_next;
    bool secant_ok = false;
    if (ghi != glo) {
      s_next = shi - ghi * (shi - slo) / (ghi - glo);
      const real smin = slo < shi ? slo : shi;
      const real smax = slo < shi ? shi : slo;
      secant_ok = detail::c2p_is_finite(s_next) && s_next > smin && s_next < smax;
    }
    if (!secant_ok) s_next = real(0.5) * (slo + shi);

    // Stagnation safeguard: the Illinois deflation above can shrink one
    // side's function value so far (relative to the other, real, side) that
    // the secant division becomes numerically degenerate and reproduces the
    // exact same s_next every iteration thereafter -- WITHOUT slo and shi
    // themselves having collapsed to adjacent doubles (they can still be
    // far apart; it is the two residual VALUES feeding the division that
    // have become comparably tiny). r_cur already holds the residual at
    // this exact point from the previous iteration -- the best achievable
    // here -- so accept it instead of burning the remaining iteration
    // budget re-deriving the identical point.
    if (s_next == prev_s_next) {
      outer_converged = true;
      break;
    }
    prev_s_next = s_next;

    int in_it = 0;
    Residuals r_next = detail::c2p_inner_solve_w(eos, in.D, in.tau, out.ye, in.S_par, in.S_perp, in.B2,
                                                  s_next, opts.w_max, opts.tau_floor_rel, w_seed, u_seed,
                                                  opts.max_iter_1d, opts.tol, in_it);
    (void)in_it;
    w_seed = r_next.w;
    u_seed = r_next.pt.u_solved;
    r_cur = r_next;
    const real g_next = r_next.f2;

    if (!detail::c2p_is_finite(g_next)) break; // safety net; leaves outer_converged false
    if (std::fabs(g_next) <= opts.tol) {
      outer_converged = true;
      break;
    }

    if ((g_next < real(0) && glo < real(0)) || (g_next > real(0) && glo > real(0))) {
      if (stale_side == 1) ghi *= real(0.5); // lo replaced twice running -> deflate the stale hi value
      slo = s_next;
      glo = g_next;
      r_bracket_lo = r_next;
      stale_side = 1;
    } else {
      if (stale_side == 2) glo *= real(0.5);
      shi = s_next;
      ghi = g_next;
      r_bracket_hi = r_next;
      stale_side = 2;
    }

    // Bracket-collapse safeguard: s can be O(10-100) in physical units, so
    // double precision only resolves it to ~1e-15 relative -- and since
    // df2/ds can be large in exactly the regions the design doc flags as
    // s-sensitive (cold/slow states), a single ULP step in s can move f2 by
    // more than opts.tol. Once slo/shi can no longer be distinguished as
    // distinct doubles, no further bisection/secant iteration can shrink
    // the bracket further, so accept the closer-to-zero endpoint as the
    // best achievable answer instead of burning the rest of max_iter_1d
    // re-evaluating the same point.
    if (!(slo < shi)) {
      r_cur = std::fabs(glo) <= std::fabs(ghi) ? r_bracket_lo : r_bracket_hi;
      outer_converged = true;
      break;
    }
  }

  // Precision polish: the outer bisection/secant above is fundamentally
  // limited to REPRESENTABLE-DOUBLE resolution in s (see the bracket-
  // collapse and stagnation safeguards) -- s can be O(10-100) so that floor
  // is only ~1e-15 relative, and where df2/ds is large (the same cold/
  // s-sensitive regions the design doc flags) that is not always enough to
  // reach opts.tol. A coupled 2x2 Newton correction from this point is not
  // similarly limited: it computes a continuous real-valued step rather
  // than a bisection-bracket midpoint, so it can land on a strictly better
  // representable double than bisection found -- iterated a handful of
  // times (same quadratic-convergence machinery as the main Newton loop,
  // just warm-started from the fallback's own best point), stopping as soon
  // as a step fails to improve ||f||_2 further. A handful of extra EOS
  // evaluations only, never worse than skipping this step entirely.
  for (int polish = 0; polish < 5; ++polish) {
    const real a = r_cur.df1_ds, b = r_cur.df1_dw, c = r_cur.df2_ds, d = r_cur.df2_dw;
    const real det = a * d - b * c;
    if (det == real(0) || !detail::c2p_is_finite(det)) break;

    const real ds = (b * r_cur.f2 - d * r_cur.f1) / det;
    const real dw = (c * r_cur.f1 - a * r_cur.f2) / det;
    const real s_try = r_cur.s + ds;
    const real w_raw = r_cur.w + dw;
    const real w_try = w_raw < real(0) ? detail::kC2PTinyW : detail::aeval_clamp(w_raw, real(0), opts.w_max);
    const Residuals r_try = detail::c2p_eval(eos, in.D, in.tau, out.ye, in.S_par, in.S_perp, in.B2, s_try,
                                              w_try, r_cur.pt.u_solved, opts.tau_floor_rel);
    // M3h: the improvement test is the SCALED norm max(|f1|/cosh w, |f2|).
    // The old plain ||f||_2 mixed f1 (O(cosh w)) with f2 (O(1)), so at high
    // rapidity it was effectively a test on f1 alone and would accept a step
    // that shaved the mis-scaled momentum residual while degrading the
    // energy residual -- or, worse, stop early because f1's floor dominated
    // the norm and no step could lower it.
    const real n_cur = detail::c2p_scaled_norm(r_cur);
    const real n_try = detail::c2p_scaled_norm(r_try);
    if (!detail::c2p_is_finite(n_try) || !(n_try < n_cur)) break;
    r_cur = r_try;
  }

  // M3h: as on the no-bracket path above, a Newton iterate that got closer
  // to a root than the fallback's own best point is the better thing to
  // report -- but only when the fallback did NOT converge. Substituting into
  // a converged fallback would risk swapping in a point near a DIFFERENT
  // root of the same conservative state (the S9 outer solve's uniqueness is
  // not proven), which is exactly the wrong-root class the audit tracks.
  bool accepted = outer_converged && detail::c2p_f1_converged(r_cur, opts.tol);
  if (!accepted && detail::c2p_scaled_norm(r_best) < detail::c2p_scaled_norm(r_cur)) {
    r_cur = r_best;
    // Acceptance is then judged on the substituted state's OWN residuals
    // rather than on the outer solve's bookkeeping. A Newton iterate meeting
    // both tolerances would already have exited the loop as
    // converged_newton, so in practice this only re-reports a FAILURE with a
    // better state; it is written out rather than assumed so the two stay
    // consistent if the loop's exit conditions ever change.
    accepted = detail::c2p_f1_converged(r_cur, opts.tol) && std::fabs(r_cur.f2) <= opts.tol;
  }
  out.result = accepted ? C2PResult::converged_fallback : C2PResult::failed_max_iter;
  detail::c2p_fill(out, r_cur, iters, outer_iters);
  return out;
}

} // namespace eeos
