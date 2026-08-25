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
//      net harmful and replaced by unconditional clamped steps.
//   5. S9 nested fallback when Newton fails to converge within
//      max_iter_newton (including a non-finite/singular step): a bracketed
//      safeguarded 1D Newton/bisection inner solve of f1(w;s)=0 (S9's
//      strict-monotonicity proof), nested inside a bracketed safeguarded
//      1D secant/bisection outer solve of g(s) = f2(s, w*(s)) over
//      [s_min_ext, s_max_ext] (the union of srange_extended at rho = D and
//      rho = D/cosh(w_max), the two ends of rho's possible range) -- see
//      detail::c2p_inner_solve_w() and con2prim()'s fallback section, plus
//      the bracket-collapse/stagnation/precision-polish safeguards for the
//      double-precision floor the outer bisection runs into for large |s|.
//      failed_no_bracket when the outer endpoints do not bracket a root
//      (S11: the caller's invalid-state policy); failed_max_iter when the
//      outer solve exhausts its budget without meeting both tolerances.

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
  real tol = real(1e-12);
  int max_iter_newton = 30;
  int max_iter_1d = 60;
  real w_max = real(12.0);
  real tau_floor_rel = real(1e-16);
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
struct Residuals {
  real s, w;
  real rho;
  EOSPoint pt;
  real z, v_par, v_perp, V;
  real f1, f2;
  real df1_ds, df1_dw, df2_ds, df2_dw;
};

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
EEOS_HOST_DEVICE inline Residuals c2p_inner_solve_w(const EntropyEOSView &eos, real D, real tau, real ye,
                                                     real S_par, real S_perp, real B2, real s, real w_max,
                                                     real tau_floor_rel, real w_start, real u_prev,
                                                     int max_iter_1d, real tol, int &iters_out) {
  real lo = real(0), hi = w_max;
  real w = aeval_clamp(w_start, lo, hi);
  Residuals r = c2p_eval(eos, D, tau, ye, S_par, S_perp, B2, s, w, u_prev, tau_floor_rel);

  int iters = 0;
  while (std::fabs(r.f1) > tol && iters < max_iter_1d) {
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
// s_guess/w_guess/u_guess default to NaN ("no guess"); when finite they warm
// -start the corresponding piece, otherwise a cold-start seed is used (S6):
//   w0 = atanh(min(|S|/(tau+D+p0), 0.999)), p0 = 0.3*tau (a crude bounded
//        estimate -- any bounded seed works, the safeguards do the work);
//   s0 = midpoint of srange(D/cosh(w0), ye).
// These are documented as *just seeds*: correctness does not depend on
// their quality, only iteration count does.
EEOS_HOST_DEVICE inline Con2PrimOut con2prim(const EntropyEOSView &eos, const Con2PrimIn &in,
                                              const Con2PrimOptions &opts,
                                              real s_guess = detail::p2c_nan(),
                                              real w_guess = detail::p2c_nan(),
                                              real u_guess = detail::p2c_nan()) {
  using detail::Residuals;

  Con2PrimOut out{};
  out.ye = in.D_Y / in.D;

  // --- S6 cold-start seeds -------------------------------------------------
  real w;
  if (!detail::aeval_is_nan(w_guess)) {
    w = detail::aeval_clamp(w_guess, real(0), opts.w_max);
  } else {
    const real Smag = std::sqrt(in.S_par * in.S_par + in.S_perp * in.S_perp);
    const real p0 = real(0.3) * in.tau;
    const real vden_raw = in.tau + in.D + p0;
    const real vden = vden_raw > real(1e-300) ? vden_raw : real(1e-300);
    const real vfrac = Smag / vden;
    const real vclamped = vfrac < real(0.999) ? vfrac : real(0.999);
    w = detail::aeval_clamp(std::atanh(vclamped), real(0), opts.w_max);
  }

  real s;
  if (!detail::aeval_is_nan(s_guess)) {
    s = s_guess;
  } else {
    const real rho0 = in.D / std::cosh(w);
    const SRange sr0 = eos.srange(rho0, out.ye);
    s = real(0.5) * (sr0.s_min + sr0.s_max);
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
  Residuals r = detail::c2p_eval(eos, in.D, in.tau, out.ye, in.S_par, in.S_perp, in.B2, s, w, u_guess,
                                  opts.tau_floor_rel);
  int iters = 0;
  bool converged = std::fabs(r.f1) <= opts.tol && std::fabs(r.f2) <= opts.tol;

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
    converged = std::fabs(r.f1) <= opts.tol && std::fabs(r.f2) <= opts.tol;
  }

  if (converged) {
    out.result = C2PResult::converged_newton;
    detail::c2p_fill(out, r, iters, 0);
    return out;
  }

  // --- S9 nested 1D fallback -------------------------------------------------
  // Outer bracket [s_min_ext, s_max_ext]: rho = D/cosh(w) ranges over the
  // bounded interval [D/cosh(w_max), D] as w ranges over its own domain
  // [0, w_max], so rather than trusting a single anchor rho (e.g. the last
  // Newton iterate's w, which -- precisely because Newton failed to
  // converge -- may be far from the true root's w), take the UNION of
  // srange_extended at both ends of that interval. Each g(s) evaluation
  // below still calls the EOS at the s-appropriate solved rho via the inner
  // w-solve, so "rho evaluated per iterate" is honored where it drives the
  // residual; this union only widens the search bracket robustly.
  const SRange sr_w0 = eos.srange_extended(in.D, out.ye);
  const SRange sr_wmax = eos.srange_extended(in.D / std::cosh(opts.w_max), out.ye);
  const real s_a0 = sr_w0.s_min < sr_wmax.s_min ? sr_w0.s_min : sr_wmax.s_min;
  const real s_b0 = sr_w0.s_max > sr_wmax.s_max ? sr_w0.s_max : sr_wmax.s_max;

  int ia = 0, ib = 0;
  Residuals r_lo = detail::c2p_inner_solve_w(eos, in.D, in.tau, out.ye, in.S_par, in.S_perp, in.B2, s_a0,
                                              opts.w_max, opts.tau_floor_rel, w, r.pt.u_solved,
                                              opts.max_iter_1d, opts.tol, ia);
  Residuals r_hi = detail::c2p_inner_solve_w(eos, in.D, in.tau, out.ye, in.S_par, in.S_perp, in.B2, s_b0,
                                              opts.w_max, opts.tau_floor_rel, r_lo.w, r_lo.pt.u_solved,
                                              opts.max_iter_1d, opts.tol, ib);
  (void)ia;
  (void)ib;

  real slo = s_a0, shi = s_b0, glo = r_lo.f2, ghi = r_hi.f2;
  const bool bracketed = (glo <= real(0) && ghi >= real(0)) || (glo >= real(0) && ghi <= real(0));

  if (!bracketed) {
    out.result = C2PResult::failed_no_bracket;
    const Residuals &best = std::fabs(glo) <= std::fabs(ghi) ? r_lo : r_hi;
    detail::c2p_fill(out, best, iters, 0);
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
    const real n_cur = std::sqrt(r_cur.f1 * r_cur.f1 + r_cur.f2 * r_cur.f2);
    const real n_try = std::sqrt(r_try.f1 * r_try.f1 + r_try.f2 * r_try.f2);
    if (!detail::c2p_is_finite(n_try) || !(n_try < n_cur)) break;
    r_cur = r_try;
  }

  const bool f1_ok = std::fabs(r_cur.f1) <= opts.tol;
  out.result = (outer_converged && f1_ok) ? C2PResult::converged_fallback : C2PResult::failed_max_iter;
  detail::c2p_fill(out, r_cur, iters, outer_iters);
  return out;
}

} // namespace eeos
