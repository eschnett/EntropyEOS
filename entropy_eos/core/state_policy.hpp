// entropy_eos/core/state_policy.hpp
//
// M3e invalid-state policy layer (con2prim-entropy-rapidity.md S11, and S13
// open question 9's "policy values"). Header-only, device-ready (see CODE.md
// "Layout"): no STL containers, no exceptions, no allocation, every function
// EEOS_HOST_DEVICE.
//
// === THE CONTRACT ==========================================================
//
// con2prim_safe() NEVER fails. For ANY input -- non-finite garbage, a
// collapse state whose D and tau grow without bound (typically inside an
// event horizon), a superluminal momentum demand, a vacuum region, a state
// whose tau sits below the coldest value the table can express -- it returns
// a valid, EXACTLY SOLVABLE state plus flags saying what it did.
//
// Physics fidelity is explicitly NOT a goal in the excision regime: the
// point is to keep the evolution going by keeping the state valid, no matter
// how bad the physics. Replacing a hopeless point by atmosphere is "hydro
// excision", and the caller performs it by adopting Con2PrimSafeOut::cons as
// its new conserved state.
//
// === THE UNIVERSAL PROJECTION MECHANISM ====================================
//
// Every repair goes through PRIMITIVES. Clamp (rho, s, Ye, w) per policy,
// evaluate the EOS, then prim2con() to regenerate the conservatives. tau and
// S_i are NEVER hand-edited: the returned conservatives always come out of
// prim2con() applied to a policy-valid primitive state, so
//
//     con2prim(returned cons) == returned prims
//
// holds by construction -- the output is exactly solvable, not merely
// "hopefully solvable". (Verified as a property, not assumed: see
// tests/test_state_policy.cpp, which re-solves every repaired state with the
// plain solver, and host/con2prim_audit.cpp's broken-state battery.)
//
// "Exactly solvable" is a statement about the STATE, not about the cold-start
// machinery. Measured on LS220: two of the extreme repairs -- the ceiling
// projection at the table's hot high-density corner, and a w-capped state
// whose recovered entropy lands in LS220's radiation-dominated pocket -- come
// back to 1.5e-16 when con2prim() is warm-started from the returned
// primitives (which a caller always has), but are missed by a fully COLD
// re-solve, because those are exactly the corners of the M3d seed / S9
// bracket-scan failure tail documented in CODE.md M3 open item (i). That is a
// property of the cold start, not of the state; the audits report both.
//
// One consequence worth stating plainly: an excised point's D, tau and S_i
// all change. That is the whole point -- the incoming conservatives were not
// realizable, so something had to give. The one thing the layer does try to
// preserve, where it can, is D (see con2prim_safe() step 3's w-cap branch):
// the rest mass density is the best-conditioned conservative and the only one
// recoverable in closed form (D = rho*cosh w).
//
// === UNITS =================================================================
//
// Adapter units throughout, identical to prim2con.hpp/con2prim.hpp (see
// CODE.md "M2 design notes"): rho = rho* (kappa-rescaled g/cc), s in
// kB/baryon, w the rapidity (W = cosh w, v = tanh w), D/tau/S_par/S_perp/B^2
// all g/cc. rho_atm and rho_ceiling are therefore kappa-rescaled g/cc, NOT
// raw table densities.
//
// === WHAT LIVES WHERE ======================================================
//
//   PolicyOptions        the thresholds; default_policy() derives all but
//                        rho_atm from the view.
//   PrimState            the scalar primitive core (rho, s, Ye, w). Direction
//                        vectors stay the caller's business, exactly as in
//                        prim2con.hpp's scalar form.
//   check_prim_state()   diagnose a primitive state (no mutation, no solves).
//   project_prim_state() clamp a primitive state in place (one srange() call,
//                        no solves).
//   check_con_state()    cheap diagnosis of a CONSERVATIVE state; the
//                        expensive endpoint-sign part is opt-in.
//   con2prim_safe()      the never-fails entry point (decision tree below).

#pragma once

#include <cmath>

#include "entropy_eos/core/adapter_eval.hpp"
#include "entropy_eos/core/con2prim.hpp"
#include "entropy_eos/core/defs.hpp"
#include "entropy_eos/core/prim2con.hpp"

namespace eeos {

// Thresholds of the invalid-state policy. Use default_policy() to fill every
// derived field from a built view; only rho_atm is genuinely a caller choice.
//
// A zero/non-finite D_max or tau_max means "no such bound" -- the
// corresponding step-1 test is skipped (step 4 still catches everything the
// solver cannot do), so a hand-built PolicyOptions that only sets rho_atm and
// w_cap degrades gracefully rather than excising everything.
struct PolicyOptions {
  // --- Atmosphere: the excision target -------------------------------------
  // rho_atm is in adapter units (kappa-rescaled g/cc).
  real rho_atm = real(0);

  // s_atm: NaN means "the midpoint of the PHYSICAL srange(rho_atm, Ye) at
  // call time". It cannot be resolved at build time because it depends on the
  // Ye the atmosphere adopts, which by default is the INCOMING Ye (below) --
  // and PolicyOptions is deliberately stateless (no cached view, no cached
  // srange), so that one PolicyOptions value is safely shared by every thread
  // and every point of a grid. A finite s_atm is clamped into the physical
  // srange at the atmosphere point, so that the atmosphere state is itself
  // policy-valid whatever the caller passes.
  real s_atm = detail::p2c_nan();

  // ye_atm: NaN means "preserve the incoming Ye" (clamped to the table's
  // [y_lo, y_hi]). Composition is the one primitive that survives excision
  // intact whenever it can, because Ye = D_Y/D is exact and outside the
  // iteration (design doc S6) -- there is nothing to gain by discarding it.
  // When the incoming Ye is itself unusable (non-finite, or D <= 0 so it
  // cannot be formed) the midpoint of [y_lo, y_hi] is used.
  real ye_atm = detail::p2c_nan();

  // The atmosphere's velocity is always v = 0 (w = 0) and its B^2 always
  // passes through (finite and >= 0, else 0) -- neither is a policy knob.

  // Conservative-space atmosphere trigger: D < atm_trigger * rho_atm ->
  // atmosphere. This is a D-space test on purpose: W >= 1 so rho <= D, hence
  // any state whose density is at or below the floor is caught, and the 1%
  // headroom excises a point sitting exactly ON the floor rather than asking
  // the solver to resolve it. (The PRIM-space test in project_prim_state() is
  // the unscaled rho < rho_atm -- see that function's doc comment for why the
  // trigger factor must not appear there.)
  real atm_trigger = real(1.01);

  // --- Ceilings: collapse / hydro excision ---------------------------------
  real rho_ceiling = real(0); // default_policy(): 10^x_hi, the physical table maximum
  real w_cap = real(0);       // default_policy(): acosh(100), i.e. W <= 100.
                              // MUST be < Con2PrimOptions::w_max (12 by default): the cap
                              // only ever fires on a solved w, and the solver cannot
                              // return w > w_max, so w_cap >= w_max would silently
                              // disable it.
  real D_max = real(0);       // default_policy(): rho_ceiling * cosh(w_cap) -- exact, since
                              // D = rho*W <= rho_ceiling * cosh(w_cap)
  real tau_max = real(0);     // default_policy(): see policy_derive_bounds()

  // Owner's default: project a collapse state onto the ceiling primitives
  // (rho = rho_ceiling, s = the physical s_max there, w from the momentum
  // against the ceiling inertia), keeping a heavy, hot, moving cell. true
  // instead excises it to atmosphere ("hydro excision" in the strict sense:
  // the point is replaced by vacuum-like data).
  bool collapse_to_atmosphere = false;
};

// The scalar primitive core the policy acts on. Direction vectors are the
// caller's (prim2con.hpp S1: the general-metric reduction of S_i/B^i to
// S_par/S_perp is the caller's job); con2prim_safe() recovers the one
// direction datum it needs -- the angle between v and B -- from the incoming
// momentum split itself.
struct PrimState {
  real rho, s, ye, w;
};

// What con2prim_safe() returns.
struct Con2PrimSafeOut {
  // The solve -- or, on a repaired/excised point, the returned primitive
  // state's own EOS evaluation packaged in the same shape. `base.result` and
  // `base.iters_*` report the SOLVER's outcome and cost (converged_newton
  // with zero iterations when no solve was attempted at all, i.e. steps 0-1
  // of the decision tree); they are NOT a validity signal -- the returned
  // (base, cons) pair is always valid and exactly consistent, even when
  // base.result is failed_max_iter. `policy_flags` is the signal that
  // something was repaired.
  Con2PrimOut base;

  // Conservatives CONSISTENT with base's primitives: bit-identical to the
  // input when policy_flags == 0, and the prim2con() of the repaired
  // primitive state otherwise. The caller updates its conserved variables
  // from this -- that IS the excision.
  Prim2ConOut cons;

  // OR of the flag_pol_* bits (core/defs.hpp). 0 = the input was already
  // policy-valid and nothing was touched.
  unsigned policy_flags;

  // Whether con2prim() was actually attempted (false on the step-0/step-1
  // short circuits, where the input never reached the solver).
  bool solved;
};

namespace detail {

// --- Option sanitizers ----------------------------------------------------
// The policy layer must not fail on a badly filled PolicyOptions either, so
// every threshold is read through one of these.

EEOS_HOST_DEVICE inline real pol_rho_ceiling(const EntropyEOSView &eos, const PolicyOptions &pol) {
  if (c2p_is_finite(pol.rho_ceiling) && pol.rho_ceiling > real(0)) return pol.rho_ceiling;
  return std::pow(real(10), eos.x_hi);
}

// The atmosphere density actually used: pol.rho_atm when usable, else the
// table's own physical density floor; never above the ceiling.
EEOS_HOST_DEVICE inline real pol_rho_atm(const EntropyEOSView &eos, const PolicyOptions &pol) {
  real lo = (c2p_is_finite(pol.rho_atm) && pol.rho_atm > real(0)) ? pol.rho_atm
                                                                  : std::pow(real(10), eos.x_lo);
  const real hi = pol_rho_ceiling(eos, pol);
  if (lo > hi) lo = hi;
  return lo;
}

EEOS_HOST_DEVICE inline real pol_w_cap(const PolicyOptions &pol) {
  if (c2p_is_finite(pol.w_cap) && pol.w_cap > real(0)) return pol.w_cap;
  return std::acosh(real(100));
}

EEOS_HOST_DEVICE inline real pol_atm_trigger(const PolicyOptions &pol) {
  return (c2p_is_finite(pol.atm_trigger) && pol.atm_trigger > real(0)) ? pol.atm_trigger : real(1);
}

// The Ye the atmosphere/ceiling adopts (PolicyOptions::ye_atm's doc comment).
EEOS_HOST_DEVICE inline real pol_ye(const EntropyEOSView &eos, const PolicyOptions &pol, real ye_in) {
  real y = pol.ye_atm;
  if (!c2p_is_finite(y)) y = ye_in;
  if (!c2p_is_finite(y)) y = real(0.5) * (eos.y_lo + eos.y_hi);
  return aeval_clamp(y, eos.y_lo, eos.y_hi);
}

} // namespace detail

// Fills PolicyOptions::D_max and ::tau_max from the options' current
// rho_ceiling and w_cap. default_policy() calls this; a caller that
// afterwards overrides either rho_ceiling or w_cap should call it again (the
// two bounds are derived quantities, not independent knobs).
//
//   D_max = rho_ceiling * cosh(w_cap)
// is exact: D = rho*W, and the policy admits neither rho > rho_ceiling nor
// W > cosh(w_cap).
//
//   tau_max = cosh(w_cap)^2 * max over a table-box scan of (rho * h)
//             at s = the PHYSICAL srange(rho,Ye).s_max
// bounds the FLUID part of tau, and it does so as an inequality rather than
// an estimate. From prim2con.hpp's cancellation-free form,
//   tau_fluid = 2*D*sinh^2(w/2) + rho*eps*W^2 + p*sinh^2(w)
//             = rho*W*(W-1) + rho*eps*W^2 + p*(W^2-1)
//             <= W^2 * (rho + rho*eps + p) = W^2 * rho * h,
// term by term (each factor is nonnegative for eps, p >= 0, which the
// adapter guarantees over its whole extended box -- CODE.md "M2"). So the
// largest fluid tau the policy admits is cosh(w_cap)^2 times the largest
// rho*h the table box contains, and rho*h is maximal at the hottest row
// (s = s_max) -- hence the scan below over (x, Ye) at s = s_max. The scan is
// a small fixed grid (5 densities x 3 Ye) rather than the single hot corner
// because rho*h is not guaranteed monotone in either coordinate on a real
// table (at low density and high T the radiation term rho*eps ~ a*T^4 is
// density-independent, so the corner is not automatically the maximum); 15
// evaluate() calls at setup is free.
//
// The magnetic part of tau is NOT included -- it cannot be, since B^2 is
// per-point caller data. con2prim_safe() therefore compares tau - B^2
// against tau_max: the magnetic contribution to tau is
// B^2/2*(1 + v_perp^2) in [B^2/2, B^2] (prim2con.hpp S7), so subtracting the
// full B^2 is the conservative choice -- it can never turn a legitimate
// magnetized state into a false collapse detection, at the price of letting
// a genuinely runaway tau hide behind a runaway B^2 (which then simply falls
// through to step 4 of the decision tree, where a failed solve is excised
// anyway).
EEOS_HOST_DEVICE inline void policy_derive_bounds(const EntropyEOSView &eos, PolicyOptions &pol) {
  const real w_cap = detail::pol_w_cap(pol);
  const real W_cap = std::cosh(w_cap);
  const real rho_ceiling = detail::pol_rho_ceiling(eos, pol);

  pol.D_max = rho_ceiling * W_cap;

  const int nx = 5, ny = 3;
  real max_rho_h = real(0);
  for (int i = 0; i < nx; ++i) {
    const real fx = static_cast<real>(i) / static_cast<real>(nx - 1);
    const real rho = std::pow(real(10), eos.x_lo + fx * (eos.x_hi - eos.x_lo));
    for (int j = 0; j < ny; ++j) {
      const real fy = static_cast<real>(j) / static_cast<real>(ny - 1);
      const real ye = eos.y_lo + fy * (eos.y_hi - eos.y_lo);
      const SRange sr = eos.srange(rho, ye);
      const EOSPoint pt = eos.evaluate(rho, sr.s_max, ye, detail::p2c_nan());
      const real rh = rho * pt.h;
      if (detail::c2p_is_finite(rh) && rh > max_rho_h) max_rho_h = rh;
    }
  }
  pol.tau_max = max_rho_h * W_cap * W_cap;
}

// The shipped default policy: everything derived from the view, except
// rho_atm which is the caller's (it is an evolution choice, not a table
// property). See PolicyOptions' per-field comments and
// policy_derive_bounds() for each default's derivation.
EEOS_HOST_DEVICE inline PolicyOptions default_policy(const EntropyEOSView &eos, real rho_atm) {
  PolicyOptions pol;
  pol.rho_atm = rho_atm;
  pol.s_atm = detail::p2c_nan();
  pol.ye_atm = detail::p2c_nan();
  pol.atm_trigger = real(1.01);
  pol.rho_ceiling = std::pow(real(10), eos.x_hi);
  pol.w_cap = std::acosh(real(100)); // W <= 100; well below Con2PrimOptions::w_max = 12
  pol.collapse_to_atmosphere = false;
  policy_derive_bounds(eos, pol);
  return pol;
}

// The atmosphere primitive state at this policy, for an incoming Ye of
// `ye_in` (pass NaN when there is none). Exactly one srange() call.
EEOS_HOST_DEVICE inline PrimState policy_atmosphere(const EntropyEOSView &eos, const PolicyOptions &pol,
                                                     real ye_in) {
  PrimState ps;
  ps.rho = detail::pol_rho_atm(eos, pol);
  ps.ye = detail::pol_ye(eos, pol, ye_in);
  const SRange sr = eos.srange(ps.rho, ps.ye);
  ps.s = detail::c2p_is_finite(pol.s_atm) ? detail::aeval_clamp(pol.s_atm, sr.s_min, sr.s_max)
                                          : real(0.5) * (sr.s_min + sr.s_max);
  ps.w = real(0);
  return ps;
}

namespace detail {

// Shared core of check_prim_state()/project_prim_state(): computes the
// policy-valid state `out` corresponding to `in`, and returns the flags
// describing what had to change (0 = `in` was already policy-valid, and then
// `out == in` field by field).
//
// Order matters and is fixed: (1) finiteness, (2) rho -- either a full
// atmosphere reset or a clamp, (3) Ye, (4) s at the FINAL (rho, Ye), (5) w.
// s comes after rho/Ye because the physical srange is a function of both, so
// clamping s against a stale srange could leave it out of range again; w is
// independent of everything else and comes last. Cost: exactly one srange()
// call on every path, no solves, no evaluate().
//
// DOCUMENTED DEVIATION from the M3e sketch: the prim-space atmosphere test is
// the UNSCALED rho < rho_atm, not rho < atm_trigger*rho_atm. With the trigger
// factor here the atmosphere state itself (rho = rho_atm < 1.01*rho_atm)
// would be reported as a violation, so check_prim_state() on an atmosphere
// point would return flag_pol_atmosphere instead of 0 and
// project_prim_state() would not be idempotent -- both properties this layer
// relies on (and tests). atm_trigger keeps its full meaning where it belongs:
// the conservative-space D test of con2prim_safe() step 1, whose job is to
// decide whether to hand a low-density point to the solver at all.
EEOS_HOST_DEVICE inline unsigned pol_project_core(const EntropyEOSView &eos, const PolicyOptions &pol,
                                                   const PrimState &in, PrimState &out) {
  // (1) Non-finite anything -> full atmosphere. Must come first: every
  // comparison below is false for a NaN, so a NaN would otherwise slip
  // through untouched.
  if (!c2p_is_finite(in.rho) || !c2p_is_finite(in.s) || !c2p_is_finite(in.ye) || !c2p_is_finite(in.w)) {
    out = policy_atmosphere(eos, pol, in.ye);
    return flag_pol_nonfinite | flag_pol_atmosphere;
  }

  const real rho_hi = pol_rho_ceiling(eos, pol);
  const real rho_lo = pol_rho_atm(eos, pol);

  // (2) rho. Below the floor (this includes rho <= 0) the state carries no
  // usable information, so the WHOLE state is replaced -- there is no
  // meaningful (s, Ye, w) to keep at a density the table cannot represent.
  if (in.rho < rho_lo) {
    out = policy_atmosphere(eos, pol, in.ye);
    return flag_pol_atmosphere;
  }

  unsigned flags = 0u;
  out = in;

  if (out.rho > rho_hi) {
    out.rho = rho_hi;
    flags |= flag_pol_rho_clamped;
  }

  // (3) Ye.
  if (out.ye < eos.y_lo) {
    out.ye = eos.y_lo;
    flags |= flag_pol_ye_clamped;
  } else if (out.ye > eos.y_hi) {
    out.ye = eos.y_hi;
    flags |= flag_pol_ye_clamped;
  }

  // (4) s against the PHYSICAL srange (not srange_extended): the extension
  // exists so the ITERATION stays finite and smooth (adapter_eval.hpp's tail
  // mathematics), not so a converged state may live there -- a state in the
  // extension zone is outside the table's validity, which is exactly what
  // this layer is for.
  const SRange sr = eos.srange(out.rho, out.ye);
  if (out.s < sr.s_min) {
    out.s = sr.s_min;
    flags |= flag_pol_s_floored;
  } else if (out.s > sr.s_max) {
    out.s = sr.s_max;
    flags |= flag_pol_s_ceiled;
  }

  // (5) w. Both ends report flag_pol_w_capped (there is no separate
  // "negative rapidity" bit, and w < 0 is meaningless rather than merely
  // out of range: the design doc's w in [0, inf) carries all direction
  // information in S_i/B^i algebraically, S9).
  const real w_cap = pol_w_cap(pol);
  if (out.w < real(0)) {
    out.w = real(0);
    flags |= flag_pol_w_capped;
  } else if (out.w > w_cap) {
    out.w = w_cap;
    flags |= flag_pol_w_capped;
  }

  return flags;
}

// Builds the (Con2PrimOut, Prim2ConOut) pair for a policy-chosen primitive
// state: one evaluate() for the EOSPoint, then prim2con() for the
// conservatives -- the universal projection mechanism of this file's module
// header, in one place, used by every repair path below.
//
// The one direction datum prim2con() needs is cos(angle(v,B)). The policy
// preserves the incoming momentum's DIRECTION and replaces only its
// MAGNITUDE (which is fixed by ps.w through v = tanh w): cos_vB is read off
// the design doc's S5 projections v_par = S_par/z, v_perp = S_perp/(z+B^2)
// evaluated at the NEW state's own inertia z = rho*h*W^2. Any z would give
// the same answer up to the relative weighting of the two components, and
// using the state's own z is the self-consistent choice. When there is no
// momentum to take a direction from (V = 0, or a degenerate z) cos_vB = 0,
// which prim2con.hpp documents as irrelevant in exactly that case.
//
// `base.result`/`base.iters_*` are NOT set here -- the caller owns them.
EEOS_HOST_DEVICE inline void pol_package(const EntropyEOSView &eos, const PrimState &ps, real S_par,
                                          real S_perp, real B2, real u_guess, Con2PrimOut &base,
                                          Prim2ConOut &cons) {
  const EOSPoint pt = eos.evaluate(ps.rho, ps.s, ps.ye, u_guess);

  const real W = std::cosh(ps.w);
  const real v = std::tanh(ps.w);
  const real z = ps.rho * pt.h * W * W;

  const real sp = c2p_is_finite(S_par) ? S_par : real(0);
  const real spp = (c2p_is_finite(S_perp) && S_perp > real(0)) ? S_perp : real(0);
  real cos_vB = real(0);
  if (z > real(0) && c2p_is_finite(z)) {
    const real vp = sp / z;
    const real vq = spp / (z + B2);
    const real Vr = std::sqrt(vp * vp + vq * vq);
    if (c2p_is_finite(Vr) && Vr > real(0)) cos_vB = aeval_clamp(vp / Vr, real(-1), real(1));
  }

  cons = prim2con(eos, ps.rho, ps.s, ps.ye, ps.w, B2, cos_vB, pt.u_solved);

  const real sin_vB2 = real(1) - cos_vB * cos_vB;
  const real sin_vB = std::sqrt(sin_vB2 > real(0) ? sin_vB2 : real(0));

  base.rho = ps.rho;
  base.s = ps.s;
  base.ye = ps.ye;
  base.w = ps.w;
  base.W = W;
  base.v_par = v * cos_vB;
  base.v_perp = v * sin_vB;
  base.eos = pt;
  base.flags = pt.flags;
}

// The collapse / hydro-excision branch, shared by con2prim_safe() steps 1
// and 4. Returns the policy flags it set.
//
// collapse_to_atmosphere = true: excise to atmosphere, and report BOTH
// flag_pol_ceiling (a ceiling was what triggered it) and
// flag_pol_atmosphere (what was done about it).
//
// Otherwise: project onto the ceiling primitives -- rho = rho_ceiling, s =
// the PHYSICAL s_max there, and a rapidity estimate. The estimate is the
// design doc's exact S5 momentum projection with z frozen at the LARGEST
// inertia the ceiling state can carry, z = rho_ceiling*h*cosh(w_cap)^2:
//   tanh^2(w) = (S_par/z)^2 + (S_perp/(z+B^2))^2.
// Freezing z at its maximum makes V minimal, i.e. the estimate is a LOWER
// bound on the rapidity -- so a momentum-poor collapse state gets a small w
// instead of being slammed to the cap, while a genuinely superluminal
// momentum demand saturates at w_cap (and reports flag_pol_w_capped). This is
// the M3d cold seed's step 2 without its pressure-lagged outer iteration:
// one exact relation, no iteration, no tuned constant -- appropriate for a
// branch whose declared purpose is validity rather than fidelity.
EEOS_HOST_DEVICE inline unsigned pol_collapse(const EntropyEOSView &eos, const PolicyOptions &pol,
                                               const Con2PrimIn &in, real ye_in, real B2, Con2PrimOut &base,
                                               Prim2ConOut &cons) {
  if (pol.collapse_to_atmosphere) {
    const PrimState ps = policy_atmosphere(eos, pol, ye_in);
    pol_package(eos, ps, in.S_par, in.S_perp, B2, p2c_nan(), base, cons);
    return flag_pol_ceiling | flag_pol_atmosphere;
  }

  const real w_cap = pol_w_cap(pol);
  PrimState ps;
  ps.rho = pol_rho_ceiling(eos, pol);
  ps.ye = pol_ye(eos, pol, ye_in);
  const SRange sr = eos.srange(ps.rho, ps.ye);
  ps.s = sr.s_max;
  ps.w = real(0);

  const EOSPoint pt = eos.evaluate(ps.rho, ps.s, ps.ye, p2c_nan());
  const real Wc = std::cosh(w_cap);
  const real z = ps.rho * pt.h * Wc * Wc;

  unsigned flags = flag_pol_ceiling;
  real V = real(0);
  if (z > real(0) && c2p_is_finite(z)) {
    const real sp = c2p_is_finite(in.S_par) ? in.S_par : real(0);
    const real spp = (c2p_is_finite(in.S_perp) && in.S_perp > real(0)) ? in.S_perp : real(0);
    const real vp = sp / z;
    const real vq = spp / (z + B2);
    V = std::sqrt(vp * vp + vq * vq);
  }
  if (!c2p_is_finite(V)) V = real(0);
  const real V_max = std::tanh(w_cap);
  if (V < V_max) {
    ps.w = aeval_clamp(std::atanh(V), real(0), w_cap);
  } else {
    ps.w = w_cap;
    flags |= flag_pol_w_capped;
  }

  pol_package(eos, ps, in.S_par, in.S_perp, B2, pt.u_solved, base, cons);
  return flags;
}

} // namespace detail

// Diagnoses a primitive state against the policy WITHOUT modifying it.
// Returns the exact flags project_prim_state() would set (0 = valid), so the
// two are usable interchangeably as predicate and repair. Cost: one srange()
// call, no solves. See detail::pol_project_core() for the ordering and the
// one documented deviation from the M3e sketch.
EEOS_HOST_DEVICE inline unsigned check_prim_state(const EntropyEOSView &eos, const PrimState &ps,
                                                   const PolicyOptions &pol) {
  PrimState tmp;
  return detail::pol_project_core(eos, pol, ps, tmp);
}

// Clamps a primitive state into policy validity in place and returns what it
// changed (0 = nothing). Idempotent: project_prim_state() on its own output
// returns 0. Cost: one srange() call, no solves.
//
//   rho  -> [rho_atm, rho_ceiling], with rho < rho_atm replaced by the FULL
//           atmosphere state (flag_pol_atmosphere)
//   Ye   -> the table's [y_lo, y_hi]                    (flag_pol_ye_clamped)
//   s    -> the PHYSICAL srange(rho, Ye) at the already-clamped (rho, Ye)
//                              (flag_pol_s_floored / flag_pol_s_ceiled)
//   w    -> [0, w_cap]                                   (flag_pol_w_capped)
//   non-finite anything -> the full atmosphere state
//                         (flag_pol_nonfinite | flag_pol_atmosphere)
EEOS_HOST_DEVICE inline unsigned project_prim_state(const EntropyEOSView &eos, PrimState &ps,
                                                     const PolicyOptions &pol) {
  PrimState out;
  const unsigned flags = detail::pol_project_core(eos, pol, ps, out);
  ps = out;
  return flags;
}

// Cheap diagnosis of a CONSERVATIVE state, i.e. of what con2prim_safe()'s
// steps 0-1 would decide before any solve happens:
//
//   flag_pol_nonfinite | flag_pol_atmosphere   non-finite input, D <= 0, or B^2 < 0
//   flag_pol_atmosphere                        D < atm_trigger * rho_atm
//   flag_pol_ceiling                           D > D_max, or tau - B^2 > tau_max
//
// COST: with check_endpoints == false (the default) this is pure arithmetic
// -- no evaluate(), no srange(), no solve -- so it is cheap enough to run on
// every point of every timestep.
//
// With check_endpoints == true it additionally runs the S9 outer-function
// endpoint diagnosis of con2prim_safe() step 4: g(s) = f2(s, w*(s)) is
// evaluated at both ends of the union of the physical sranges over the
// admissible density range (see the code), adding flag_pol_s_floored when
// g(s_lo) > 0 (tau below the coldest state the table can express here) and
// flag_pol_ceiling when g(s_hi) < 0 (tau above the hottest).
//
// COST of that: TWO full inner w-solves (each a bisection-safeguarded Newton
// over the EOS, i.e. tens of evaluate() calls) -- comparable to a whole
// con2prim() call. Hence opt-in, and meant as a DIAGNOSTIC, not a per-point
// gate: unlike con2prim_safe() step 4, which brackets at
// rho = D/cosh(w*) using the failed solve's own rapidity, this function has
// no w to work with and must bracket over the whole admissible density range
// instead, which makes it conservative but not exact. Measured on the
// synthetic gas (100 magnetized states, w in [0.5, 3]): 0 false reports on
// valid states, 100/100 correct on tau/10 (s_floored) and on tau*1e6
// (ceiling). `solver` supplies the tolerances/iteration caps/w_max for those
// solves and is unused when check_endpoints is false.
EEOS_HOST_DEVICE inline unsigned check_con_state(const EntropyEOSView &eos, const Con2PrimIn &in,
                                                  const PolicyOptions &pol, bool check_endpoints = false,
                                                  const Con2PrimOptions &solver = Con2PrimOptions()) {
  using detail::c2p_is_finite;

  const bool fin = c2p_is_finite(in.D) && c2p_is_finite(in.tau) && c2p_is_finite(in.D_Y) &&
                   c2p_is_finite(in.S_par) && c2p_is_finite(in.S_perp) && c2p_is_finite(in.B2);
  if (!fin || !(in.D > real(0)) || in.B2 < real(0)) return flag_pol_nonfinite | flag_pol_atmosphere;

  const real rho_atm = detail::pol_rho_atm(eos, pol);
  if (in.D < detail::pol_atm_trigger(pol) * rho_atm) return flag_pol_atmosphere;

  unsigned flags = 0u;
  const real B2 = in.B2 > real(0) ? in.B2 : real(0);
  const bool D_over = c2p_is_finite(pol.D_max) && pol.D_max > real(0) && in.D > pol.D_max;
  const bool tau_over =
      c2p_is_finite(pol.tau_max) && pol.tau_max > real(0) && (in.tau - B2) > pol.tau_max;
  if (D_over || tau_over) flags |= flag_pol_ceiling;
  if (flags != 0u || !check_endpoints) return flags;

  const real ye = detail::aeval_clamp(in.D_Y / in.D, eos.y_lo, eos.y_hi);

  // The UNION of the physical sranges at the two extreme densities the state
  // could have -- rho = D (w = 0) and rho = D/cosh(w_max) (the fastest the
  // solver admits) -- exactly as con2prim.hpp's own bracket scan builds its
  // outer safety net. The union is what makes the two implications below
  // sound without knowing w: g(min over rho of s_min) > 0 means tau is below
  // the coldest state at ANY admissible density, and symmetrically for the
  // hot end. (Bracketing at rho = D alone was measured to misreport 7 of 100
  // valid magnetized synthetic states with w in [0.5, 3]; with the union,
  // 0 of 100. It is still a diagnostic and not a proof -- see this
  // function's doc comment.)
  const SRange sr_a = eos.srange(in.D, ye);
  const SRange sr_b = eos.srange(in.D / std::cosh(solver.w_max), ye);
  const real s_lo = sr_a.s_min < sr_b.s_min ? sr_a.s_min : sr_b.s_min;
  const real s_hi = sr_a.s_max > sr_b.s_max ? sr_a.s_max : sr_b.s_max;

  int it = 0;
  const detail::Residuals r_lo =
      detail::c2p_inner_solve_w(eos, in.D, in.tau, ye, in.S_par, in.S_perp, B2, s_lo, solver.w_max,
                                 solver.tau_floor_rel, real(0), detail::p2c_nan(), solver.max_iter_1d,
                                 solver.tol, it);
  if (c2p_is_finite(r_lo.f2) && r_lo.f2 > real(0)) return flags | flag_pol_s_floored;

  const detail::Residuals r_hi =
      detail::c2p_inner_solve_w(eos, in.D, in.tau, ye, in.S_par, in.S_perp, B2, s_hi, solver.w_max,
                                 solver.tau_floor_rel, r_lo.w, r_lo.pt.u_solved, solver.max_iter_1d,
                                 solver.tol, it);
  if (c2p_is_finite(r_hi.f2) && r_hi.f2 < real(0)) return flags | flag_pol_ceiling;

  return flags;
}

// The never-fails con2prim. See this file's module header for the contract
// and the universal projection mechanism.
//
// DECISION TREE (order matters; every branch ends in a prim2con() of a
// policy-valid primitive state, so the returned pair is exactly solvable):
//
//  0. Any non-finite input (D, tau, D_Y, S_par, S_perp, B^2), D <= 0, or
//     B^2 < 0: atmosphere. flag_pol_nonfinite | flag_pol_atmosphere. All
//     three share the one bit because all three are BROKEN INPUT rather than
//     an extreme state: D = rho*cosh(w) > 0 identically, and
//     B^2 = gamma_ij B^i B^j is a squared norm. In particular a negative B^2
//     cannot be repaired HERE -- B^i passes through this layer untouched (it
//     is the caller's, prim2con.hpp S1) -- so the point is excised and
//     flagged instead of being quietly solved with a nonsense magnetic
//     inertia. (A caller whose reconstruction can emit roundoff-level
//     negative B^2 should clamp it at zero before calling; the layer is then
//     transparent again.) In the excised state B^2 passes through when it is
//     usable (finite and > 0) and is zeroed otherwise, which is the only
//     choice that keeps prim2con()'s magnetic terms meaningful.
//     No solve is attempted (`solved == false`).
//
//  1. Bounds, on the raw conservatives:
//       D < atm_trigger * rho_atm  -> atmosphere (vacuum / below-floor;
//                                     flag_pol_atmosphere)
//       D > D_max  or  tau - B^2 > tau_max
//                                  -> COLLAPSE branch (event-horizon-like
//                                     runaway): atmosphere if
//                                     collapse_to_atmosphere, else the
//                                     ceiling projection -- see
//                                     detail::pol_collapse().
//     Both are pure arithmetic, so the overwhelmingly common valid-state case
//     pays almost nothing for them. No solve is attempted.
//
//  2. Otherwise run the existing solver, con2prim(), with `opts` and the
//     caller's guesses passed through unchanged.
//
//  3. On success, run project_prim_state() on the SOLVED primitives. That one
//     call is the whole step-3 test: it subsumes the solver's own
//     EXT_S_LOW/EXT_S_HIGH (s outside the physical srange), rho-extension and
//     Ye-clamp flags by construction -- they say exactly what
//     project_prim_state() measures directly -- and additionally catches a
//     non-finite solved state. The one thing it cannot see is the rapidity
//     cap, which is applied first:
//       w > w_cap  ->  w = w_cap AND rho = D/cosh(w_cap).
//     Recomputing rho (rather than leaving the solved rho and letting D
//     change) PRESERVES D exactly, so a superluminal-momentum repair costs
//     the caller its momentum and energy but not its rest mass. This is a
//     documented refinement of the M3e sketch's "rho handled by its own
//     clamps": the clamps still run afterwards, on the recomputed rho -- and
//     they win when they fire. Capping w RAISES rho by the factor
//     cosh(w_solved)/cosh(w_cap), so a state that was already near the
//     density ceiling can be pushed past it; the rho clamp then takes
//     precedence over D preservation and reports flag_pol_rho_clamped
//     alongside flag_pol_w_capped (measured on the synthetic gas: 5 of 100
//     w = 5 states under a w_cap = acosh(10)).
//     If nothing fired (project_prim_state() returned 0), `cons` is the INPUT
//     conservatives copied field by field -- bit-identical, no round trip
//     through prim2con() -- and policy_flags is 0.
//
//  4. On failure (failed_no_bracket / failed_max_iter), diagnose with the S9
//     outer function's own endpoints, evaluated with con2prim.hpp's own
//     detail::c2p_inner_solve_w() so the diagnosis is by construction
//     consistent with what the solver saw: bracket the PHYSICAL srange at
//     rho_ref = D/cosh(w_ref) (w_ref = the failed solve's own best-effort
//     rapidity) and evaluate g(s) = f2(s, w*(s)) at both ends.
//       g(s_min) > 0: even the coldest state this density can express has
//                     more energy than tau -- tau is below the cold floor.
//                     -> s-FLOOR projection: keep the inner solve's w (a
//                     genuine root of the momentum residual at s_min, so the
//                     momentum stays consistent), set rho = D/cosh(w) and
//                     s = srange(rho, Ye).s_min at THAT rho (so the result is
//                     exactly on the floor, not merely near it), then
//                     project + prim2con. flag_pol_s_floored.
//       g(s_max) < 0: even the hottest state has less energy than tau -- tau
//                     is above the hot ceiling. -> COLLAPSE branch, as in
//                     step 1.
//       anything else (a sign change the solver's own multi-point scan could
//                     not exploit, or max_iter): atmosphere -- excision.
//                     flag_pol_atmosphere.
//     g(s_min) is evaluated first and short-circuits, so the common cold
//     failure costs one inner solve rather than two.
//
// The guesses default to NaN ("no guess"), exactly as in con2prim().
EEOS_HOST_DEVICE inline Con2PrimSafeOut con2prim_safe(const EntropyEOSView &eos, const Con2PrimIn &in,
                                                       const Con2PrimOptions &opts, const PolicyOptions &pol,
                                                       real s_guess = detail::p2c_nan(),
                                                       real w_guess = detail::p2c_nan(),
                                                       real u_guess = detail::p2c_nan()) {
  using detail::c2p_is_finite;

  Con2PrimSafeOut out{};
  out.policy_flags = 0u;
  out.solved = false;
  out.base.result = C2PResult::converged_newton;
  out.base.iters_newton = 0;
  out.base.iters_fallback = 0;

  const real B2 = (c2p_is_finite(in.B2) && in.B2 > real(0)) ? in.B2 : real(0);

  // --- 0. non-finite / D <= 0 ---------------------------------------------
  const bool fin = c2p_is_finite(in.D) && c2p_is_finite(in.tau) && c2p_is_finite(in.D_Y) &&
                   c2p_is_finite(in.S_par) && c2p_is_finite(in.S_perp) && c2p_is_finite(in.B2);
  if (!fin || !(in.D > real(0)) || in.B2 < real(0)) {
    const real ye_in = (c2p_is_finite(in.D) && in.D > real(0) && c2p_is_finite(in.D_Y))
                           ? in.D_Y / in.D
                           : detail::p2c_nan();
    const PrimState ps = policy_atmosphere(eos, pol, ye_in);
    detail::pol_package(eos, ps, in.S_par, in.S_perp, B2, detail::p2c_nan(), out.base, out.cons);
    out.policy_flags = flag_pol_nonfinite | flag_pol_atmosphere;
    return out;
  }

  const real ye_in = in.D_Y / in.D;

  // --- 1. bounds -----------------------------------------------------------
  const real rho_atm = detail::pol_rho_atm(eos, pol);
  if (in.D < detail::pol_atm_trigger(pol) * rho_atm) {
    const PrimState ps = policy_atmosphere(eos, pol, ye_in);
    detail::pol_package(eos, ps, in.S_par, in.S_perp, B2, detail::p2c_nan(), out.base, out.cons);
    out.policy_flags = flag_pol_atmosphere;
    return out;
  }

  const bool D_over = c2p_is_finite(pol.D_max) && pol.D_max > real(0) && in.D > pol.D_max;
  const bool tau_over =
      c2p_is_finite(pol.tau_max) && pol.tau_max > real(0) && (in.tau - B2) > pol.tau_max;
  if (D_over || tau_over) {
    out.policy_flags = detail::pol_collapse(eos, pol, in, ye_in, B2, out.base, out.cons);
    return out;
  }

  // --- 2. the solve --------------------------------------------------------
  out.base = con2prim(eos, in, opts, s_guess, w_guess, u_guess);
  out.solved = true;
  const C2PResult res = out.base.result;
  const int it_n = out.base.iters_newton, it_f = out.base.iters_fallback;
  const real u_warm = out.base.eos.u_solved;

  if (res == C2PResult::converged_newton || res == C2PResult::converged_fallback) {
    // --- 3. success: is the solved state policy-valid? --------------------
    const real w_cap = detail::pol_w_cap(pol);
    PrimState ps{out.base.rho, out.base.s, out.base.ye, out.base.w};
    unsigned pre = 0u;
    if (c2p_is_finite(out.base.w) && out.base.w > w_cap) {
      pre = flag_pol_w_capped;
      ps.w = w_cap;
      ps.rho = in.D / std::cosh(w_cap); // D-preserving, see the doc comment
    }
    const unsigned pf = pre | project_prim_state(eos, ps, pol);

    if (pf == 0u) {
      // The no-touch path: the input conservatives verbatim, bit-identical
      // (deliberately NOT a prim2con() round trip).
      out.cons.D = in.D;
      out.cons.tau = in.tau;
      out.cons.D_Y = in.D_Y;
      out.cons.S_par = in.S_par;
      out.cons.S_perp = in.S_perp;
      out.cons.B2 = in.B2;
      out.policy_flags = 0u;
      return out;
    }

    detail::pol_package(eos, ps, in.S_par, in.S_perp, B2, u_warm, out.base, out.cons);
    out.base.result = res;
    out.base.iters_newton = it_n;
    out.base.iters_fallback = it_f;
    out.policy_flags = pf;
    return out;
  }

  // --- 4. failure: diagnose with the S9 outer function's endpoints --------
  const real w_ref = c2p_is_finite(out.base.w) ? detail::aeval_clamp(out.base.w, real(0), opts.w_max)
                                                : real(0);
  const real ye_c = detail::aeval_clamp(ye_in, eos.y_lo, eos.y_hi);
  const real rho_ref = in.D / std::cosh(w_ref);
  const SRange sr_ref = eos.srange(rho_ref, ye_c);

  int it = 0;
  const detail::Residuals r_lo =
      detail::c2p_inner_solve_w(eos, in.D, in.tau, ye_in, in.S_par, in.S_perp, B2, sr_ref.s_min, opts.w_max,
                                 opts.tau_floor_rel, w_ref, u_warm, opts.max_iter_1d, opts.tol, it);

  if (c2p_is_finite(r_lo.f2) && r_lo.f2 > real(0)) {
    // tau below the coldest compatible value -> s-floor projection.
    const real w_cap = detail::pol_w_cap(pol);
    const real w1 = c2p_is_finite(r_lo.w) ? detail::aeval_clamp(r_lo.w, real(0), w_cap) : real(0);
    const real rho1 = in.D / std::cosh(w1);
    const real rho1c = detail::aeval_clamp(rho1, detail::pol_rho_atm(eos, pol),
                                            detail::pol_rho_ceiling(eos, pol));
    const SRange sr1 = eos.srange(rho1c, ye_c);
    PrimState ps{rho1c, sr1.s_min, ye_in, w1};
    const unsigned pf = flag_pol_s_floored | project_prim_state(eos, ps, pol);
    detail::pol_package(eos, ps, in.S_par, in.S_perp, B2, r_lo.pt.u_solved, out.base, out.cons);
    out.base.result = res;
    out.base.iters_newton = it_n;
    out.base.iters_fallback = it_f;
    out.policy_flags = pf;
    return out;
  }

  const detail::Residuals r_hi =
      detail::c2p_inner_solve_w(eos, in.D, in.tau, ye_in, in.S_par, in.S_perp, B2, sr_ref.s_max, opts.w_max,
                                 opts.tau_floor_rel, r_lo.w, r_lo.pt.u_solved, opts.max_iter_1d, opts.tol,
                                 it);

  if (c2p_is_finite(r_hi.f2) && r_hi.f2 < real(0)) {
    // tau above the hottest compatible value -> collapse.
    out.policy_flags = detail::pol_collapse(eos, pol, in, ye_in, B2, out.base, out.cons);
    out.base.result = res;
    out.base.iters_newton = it_n;
    out.base.iters_fallback = it_f;
    return out;
  }

  // Anything else: excise.
  const PrimState ps = policy_atmosphere(eos, pol, ye_in);
  detail::pol_package(eos, ps, in.S_par, in.S_perp, B2, r_hi.pt.u_solved, out.base, out.cons);
  out.base.result = res;
  out.base.iters_newton = it_n;
  out.base.iters_fallback = it_f;
  out.policy_flags = flag_pol_atmosphere;
  return out;
}

} // namespace eeos
