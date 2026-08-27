// entropy_eos/core/adapter_eval.hpp
//
// M2b adapter core (M2d-2 domain extensions): EntropyEOSView::evaluate()
// implements the F(rho,T,Ye) -> U(rho,s,Ye) construction of
// eos-adapter-F-to-U.md (S2-S5, S11): a warm-started safeguarded Newton
// T-solve on the fitted entropy spline, followed by the S3.1 chain rule
// composing the two fitted splines (Sigma-hat = entropy, L-hat =
// log10(eps_cgs + energy_shift_cgs)) into U and its rho/s derivatives.
// Header-only, device-ready (see CODE.md "Layout"): no STL containers, no
// exceptions, no allocation, every function EEOS_HOST_DEVICE, operating on
// the POD BsplineView3 views bspline_eval.hpp already defines.
//
// Internal spline coordinates (see CODE.md "M2 design notes"): x =
// log10(rho* [g/cc]) evaluated against the kappa-shifted grid origin
// already baked into the view's BsplineView3 (see host/adapter_build.hpp
// step 3: "Store the shifted grid origin x0* = logrho[0] + log10(kappa)"),
// u = log10(T [MeV]), y = Ye, lambda = ln(10). Adapter-facing quantities:
// rho* in g/cc (kappa-rescaled), U = eps/c^2 dimensionless, p returned as
// p/c^2 in g/cc, s in kB/baryon, That = U_s dimensionless, T_F in MeV.
//
// === M2d-2 TAIL MATHEMATICS (eos-adapter-F-to-U.md S7 domain extensions) ==
//
// During con2prim iteration the adapter must return finite, smooth,
// monotone values for arguments outside the table -- extension plus flag,
// never a hard clamp (a clamp zeroes derivatives and stalls Newton). Every
// extension below is a 1D operator acting on one axis (x or u) at a time,
// evaluated on top of a boundary BsplineEval3 sample; the physical box
// still defines the flags (evaluate() step 1-2), and validity is judged by
// the caller on the converged state (unchanged from M2b/M2c).
//
// Generic C2 "curvature-ramp" tail. Let d = coord - seam be the signed
// offset from the boundary (d<0 below x_lo/u_lo, d>0 above x_hi/u_hi), and
// w > 0 the blend width -- one grid cell of that axis (hx or hu). Given the
// boundary track (f_b, f'_b, f''_b) (value/1st/2nd derivative *along the
// tail axis*, sampled at the seam):
//
//   phase 1 (|d| <= w): with sgn = sign(d),
//     f(d)  = f_b + f'_b*d + f''_b*(d^2/2 - |d|^3/(6w))
//     f'(d) = f'_b + f''_b*(d - sgn*d^2/(2w))
//     f''(d)= f''_b*(1 - |d|/w)
//   phase 2 (|d| > w): linear continuation from the phase-1 endpoint,
//     m = f'_b + sgn*f''_b*w/2,  f(d) = f(sgn*w) + m*(d - sgn*w),
//     f'(d) = m,  f''(d) = 0.
//
// (f'' ramps linearly from f''_b at the seam to 0 at |d|=w, so f is C2 at
// both the seam and the |d|=w blend edge; expanding sgn=-1/+1 recovers the
// low/high-side formulas of the task spec verbatim -- see
// detail::aeval_ramp_track().) This is "the generic tail" below.
//
// Monotonicity guard (applied ONLY to the u-direction primary track of
// *both* fitted fields -- sigma_u and L_u must stay positive for the
// T-solve/U_s to make sense): the tail slope m must satisfy m >= m_floor >
// 0 (BuildOptions::ext_slope_floor_sigma / _L). If the raw (f'_b, f''_b)
// would violate that, first raise the effective f'_b to max(f'_b,
// m_floor), then clamp the effective f''_b so m stays >= m_floor (see
// detail::aeval_generic_track()). Because phase 1's f'(d) interpolates
// monotonically between f'_b and m (a fact used, not just asserted -- see
// that function's doc comment), this bounds f'(d) >= m_floor for every d in
// the blend cell, not just at its endpoints. Clamping costs C2 (drops to
// C1/C0 at the seam) only inside flagged territory.
//
// Slope-to-zero variant (used ONLY for L's primary track on the x-low
// seam; design: eps becomes rho-independent, ideal-gas-like as rho -> 0):
// overrides the curvature so f'(d) ramps from f'_b to *exactly* 0 at
// d = -w (then phase 2 is the constant f'=0 continuation), instead of using
// the raw sampled f''_b. From the generic tail's own low-side asymptotic
// slope m = f'_b - f''_b*w/2, forcing m=0 gives f''_eff = 2*f'_b/w (see
// detail::aeval_slope_zero_track()). C1 (not C2) at the seam unless f''_b
// already happened to equal 2*f'_b/w -- acceptable, flagged territory only.
// sigma's x-low tail and both fields' x-high tail use the plain generic
// tail instead (x-high keeps the hard flag_oob_rho_high regardless -- a
// converged state there is invalid per eos-adapter-F-to-U.md S7, the tail
// only exists so the *iteration* stays finite and can converge to a
// reportable point). Neither x-direction tail uses the monotonicity guard
// (x is clamped, never iterated -- only u is solved for).
//
// Mixed-derivative composition. BsplineEval3 carries 7 fields (f, fx, fu,
// fy, fxx, fxu, fuu); a single-axis tail acts on them track-wise, each
// track being one (value, 1st-deriv, 2nd-deriv) triple fed through the
// tail construction above:
//
//   x-tail: track f  = (f_b, fx_b, fxx_b)  -> f, fx, fxx
//           track fu = (fu_b, fxu_b, 0)    -> fu, fxu = d/dd of that track
//           track fuu = (fuu_b, -, -)      -> frozen (no fxuu available)
//           track fy  = (fy_b, -, -)       -> frozen
//   u-tail: track f  = (f_b, fu_b, fuu_b)  -> f, fu, fuu
//           track fx = (fx_b, fxu_b, 0)    -> fx, fxu = d/dd of that track
//           track fxx = (fxx_b, -, -)      -> frozen
//           track fy  = (fy_b, -, -)       -> frozen
//
// Only the primary "f" track (whose 1st derivative is the tail-axis
// derivative: fu for a u-tail, fx for an x-tail) ever gets the
// monotonicity guard or the slope-to-zero override; the secondary track
// always uses the plain, unguarded generic tail. See
// detail::aeval_apply_tail().
//
// Corner composition order: a query point outside the box in *both* x and
// u first samples the raw spline at the seam clamped in x (i.e. at
// (x_lo|x_hi, u_lo|u_hi)), applies the u-tail, then applies the x-tail to
// the resulting (already u-extended) tracks. See detail::aeval_extended(),
// which is the single implementation of everything above, shared by
// evaluate(), srange_extended(), and sigma_extended() (and reused by
// host/adapter_build.cpp's extended kappa/eps-floor scan) -- and which is
// exactly transparent (falls through to a single plain bspline_eval3()
// call) for any interior point, so it never perturbs an in-box evaluation.
//
// === M3g CAUSAL TAILS (eos-causal-tail.md) ================================
//
// The S7 guiding principle above is missing one word: the values an
// iterating solver sees must also be **causal**. They were not: continuing
// sigma LINEARLY in u past the hot seam forces T (and with it eps) to climb
// far too fast along adiabats, and the resulting c_s^2 crosses 1 about one
// grid cell past u_hi and saturates ~3.8 -- which breaks the con2prim inner
// solve's monotonicity proof (z_w = z(1-c_s^2)tanh w flips sign, f1(w;s)
// stops being monotone) and produced ~95% of M3's con2prim failure tail
// (CODE.md "M3 failure-tail root cause"). Two changes, u-HIGH side only:
//
// 1. **Log-sigma u-high tail** (sigma only). Physical entropy grows
//    exponentially in u (s ~ T^3/rho), so the tail is built on g = ln sigma
//    instead of sigma: transform the whole boundary sample
//      g = ln f, g_a = f_a/f, g_ab = f_ab/f - (f_a/f)(f_b/f)
//    (detail::aeval_log_sample()), run the *unchanged* curvature-ramp
//    machinery above on it -- primary track (g, g_u, g_uu), secondary
//    (g_x, g_xu, 0), frozen g_xx/g_y, exactly the same table -- and map back
//      f = e^g, f_a = f g_a, f_ab = f (g_ab + g_a g_b)
//    (detail::aeval_exp_sample()). The corner composition order is
//    untouched: the log transform lives entirely inside the u-high tail
//    operator, so the x-tail still acts afterwards on the mapped-back
//    tracks. sigma_b > 0 at the u_hi seam by construction (it is the
//    column's largest entropy); should a pathological table violate that,
//    the tail falls back to the linear construction (the guard is explicit
//    in aeval_extended()).
//    The monotonicity guard transfers exactly: flooring the LOG slope at
//    m_floor/sigma_b keeps sigma_u = sigma*g' >= sigma_b*g' >= m_floor,
//    since sigma >= sigma_b throughout a growing tail.
//    Why it works: with ln sigma and ln eps both asymptotically linear in u
//    (rates alpha = dln(sigma)/du and b = dln(eps)/du) and 1/rho seam
//    scaling, the far-tail fixed-s slope q = dlnW/dx|_s is *constant*, so
//    c_s^2(tail) = b/alpha - 1 exactly; radiation slopes (b = 4 ln10,
//    alpha = 3 ln10) give 1/3 -- the physically correct hot-gas asymptote --
//    and p > 0 <=> b > alpha.
//
// 2. **Causal slope clamp on L's u-high tail.** Nothing guarantees
//    b/alpha - 1 <= 1 at every seam point, so L's phase-2 slope is capped:
//    with b_eff = ln10 * m_L * (eps_b + Delta)/eps_b and alpha_eff the
//    sigma tail's log slope, enforce b_eff <= (1 + cs2_ext_cap)*alpha_eff
//    (EntropyEOSView::cs2_ext_cap, BuildOptions::cs2_ext_cap, default 0.99)
//    by lowering the effective (f1, f2) -- the exact mirror image of the
//    monotonicity guard (detail::aeval_cap_slope()), C2 dropping to C1 only
//    inside flagged territory. Guard priority is lexicographic as
//    everywhere else: the monotonicity floor ext_slope_floor_L wins if the
//    two ever conflict (host/adapter_audit.cpp's extension-band map reports
//    such points; none occur on the real tables).
//
// Unchanged by M3g: both u-LOW tails (sigma must run linearly to -infinity
// so every s < s_min maps to a finite T -- the escape hatch stands), both
// x-tails, all flag semantics, srange()/physical-box logic, and every
// in-box evaluation (aeval_extended() still falls straight through; the
// tail operator is exactly transparent inside the box -- asserted
// bit-for-bit in tests/test_adapter_tail.cpp).
//
// Consequence to expect, not fear: srange_extended().s_max grows (an
// 8-cell log tail reaches ~10^0.8 ~ 6x s_max where the linear tail reached
// ~1.8x). The bracket scan is log-spaced and the T-solve is a safeguarded
// Newton on a still strictly monotone sigma, so no solver mechanics change.

#pragma once

#include <cmath>

#include "entropy_eos/core/bspline_eval.hpp"
#include "entropy_eos/core/defs.hpp"

namespace eeos {

// One evaluate() result: U and its rho/s derivatives (what the con2prim
// Newton consumes), derived quantities, the solved table temperature, the
// optional composition potential, warm-start state, iteration count, and
// flags. POD, trivially copyable (device-compatible).
struct EOSPoint {
  real U, U_rho, U_s, U_rhorho, U_rhos; // what the con2prim Newton consumes
  real That, p, h, cs2;                 // derived (That = U_s)
  real T_F_MeV;                         // solved table temperature (cross-table physics)
  real mu_tilde;                        // U_Ye, per baryon in units of m_B* c^2
  real u_solved;                        // log10(T_F_MeV), warm start for the next call
  int iters;                            // T-solve iterations performed (0 if an s-extension short-circuited it)
  unsigned flags; // flag_clamp_ye | flag_ext_s_low | flag_ext_s_high | flag_ext_rho_low
                  // | flag_oob_rho_high | flag_maxiter (see core/defs.hpp)
                  //
                  // M2d-2: flag_ext_s_low/flag_ext_s_high are now set by
                  // comparing the *solved* u against the physical [u_lo,
                  // u_hi] box (not by pre-clamping s -- see evaluate()):
                  // with the extension in place the T-solve almost always
                  // lands on a genuine (extended) root rather than pinning
                  // at the box edge, so these flags now mark "the answer
                  // is in the designed extension zone", still finite and
                  // smooth, rather than "the input was out of range and got
                  // clamped".
};

// Pointwise physical entropy range at fixed (rho*, Ye): [s(T_min), s(T_max)]
// (solver bracketing; eos-adapter-F-to-U.md S7).
struct SRange {
  real s_min, s_max;
};

namespace detail {

// ln(10): converts the fitted splines' log10-axis derivatives to physical
// (rho*, That_t) ones (S3.1's "log-axis conversions", generalized to the
// tensor-product case per this file's chain-rule stages below).
constexpr real kLn10 = real(2.302585092994045684017991454684364207601101488628772976033);

EEOS_HOST_DEVICE inline real aeval_clamp(real v, real lo, real hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Self-compare NaN test: avoids relying on std::isnan's device availability
// (core/ must stay portable to a CUDA compile per CODE.md "Layout"), and
// works for any IEEE-754 real.
EEOS_HOST_DEVICE inline bool aeval_is_nan(real v) { return v != v; }

// --- M2d-2 domain extensions: see the module header "TAIL MATHEMATICS" ---

// One 1D "track" the mixed-derivative composition reduces a BsplineEval3
// field to: value + 1st + 2nd derivative *along the tail axis*, at the
// seam (offset d=0) on input, at the query offset d on output (same shape
// serves both, mirroring bspline_eval.hpp's Basis4-style reuse).
struct Track1D {
  real f0, f1, f2;
};

// The two-phase generic curvature-ramp construction (module header),
// unified across the low (d<0) / high (d>0) sides via sgn = sign(d): phase
// 1 (|d|<=w) blends f'' linearly from t.f2 (at d=0) to 0 (at |d|=w); phase
// 2 (|d|>w) is the linear continuation from the phase-1 endpoint (f''=0
// identically). No guard, no shape override -- the raw building block the
// two variants below configure by choosing what (t.f1, t.f2) to pass in.
//
// Monotonicity fact used by aeval_generic_track(): within phase 1, f'(d)
// is an affine, monotonic function of the "progress" phi(|d|) = |d| -
// d^2/(2w) (itself monotone increasing on [0,w]) between its two endpoints
// f'(0) = t.f1 and f'(sgn*w) = m -- so f'(d) never overshoots outside
// [min(t.f1,m), max(t.f1,m)] anywhere in the blend cell.
EEOS_HOST_DEVICE inline Track1D aeval_ramp_track(Track1D t, real d, real w) {
  const real ad = d < real(0) ? -d : d;
  const real sgn = d < real(0) ? real(-1) : real(1);
  Track1D o;
  if (ad <= w) {
    o.f0 = t.f0 + t.f1 * d + t.f2 * (real(0.5) * d * d - ad * ad * ad / (real(6) * w));
    o.f1 = t.f1 + t.f2 * (d - sgn * d * d / (real(2) * w));
    o.f2 = t.f2 * (real(1) - ad / w);
  } else {
    const real m = t.f1 + sgn * t.f2 * w * real(0.5);
    const real f_edge = t.f0 + t.f1 * sgn * w + t.f2 * w * w / real(3);
    o.f0 = f_edge + m * (d - sgn * w);
    o.f1 = m;
    o.f2 = real(0);
  }
  return o;
}

// The phase-2 (|d| > w) asymptotic slope aeval_ramp_track() will produce
// for a track already carrying its *effective* (f1, f2): m = f1 + sgn*f2*w/2.
EEOS_HOST_DEVICE inline real aeval_phase2_slope(const Track1D &t, real sgn, real w) {
  return t.f1 + sgn * t.f2 * w * real(0.5);
}

// The monotonicity guard's effective-track rewrite (module header), factored
// out of aeval_generic_track() so the M3g causal clamp and the audit hook
// (EntropyEOSView::u_high_tail_info()) reuse the identical arithmetic:
// m_floor <= 0 is a no-op (every track except the u-direction primary track
// of sigma/L); m_floor > 0 raises the effective f1 to at least m_floor, then
// caps the effective f2 so the asymptotic slope m = f1 +- f2*w/2 (sign per
// side) does not fall back below m_floor.
EEOS_HOST_DEVICE inline Track1D aeval_floor_slope(Track1D t, real sgn, real w, real m_floor) {
  if (m_floor > real(0)) {
    const real f1_eff = t.f1 < m_floor ? m_floor : t.f1;
    const real cap = real(2) * (f1_eff - m_floor) / w; // >= 0
    real f2_eff = t.f2;
    if (sgn < real(0)) {
      if (f2_eff > cap) f2_eff = cap; // low side: m = f1_eff - f2*w/2
    } else {
      if (f2_eff < -cap) f2_eff = -cap; // high side: m = f1_eff + f2*w/2
    }
    t.f1 = f1_eff;
    t.f2 = f2_eff;
  }
  return t;
}

// M3g causal slope cap -- the counterpart of aeval_floor_slope() on the HIGH
// side, and the only guard that ever lowers a slope: if the phase-2 slope
// m = f1 + f2*w/2 exceeds m_cap, lower the effective CURVATURE (f2 alone) to
// 2*(m_cap - f1)/w so it equals m_cap exactly.
//
// f1 is deliberately left alone, unlike the monotonicity guard: the seam
// value AND seam slope must keep matching the boundary spline sample, so the
// tail stays C1 there (only f'' jumps) and U/U_s are continuous across the
// seam -- check_adapter()'s class D measures exactly that. The price is that
// f'(d) inside the blend cell rides down from f1 to m_cap and can therefore
// transiently exceed the cap over that one cell, which eos-causal-tail.md S4
// anticipates and hands to the class E extension-band map to adjudicate.
// Since aeval_capped_track() only ever passes m_cap >= m_floor, and phase 1's
// f'(d) stays between f1 and m_cap, the monotonicity guarantee survives.
// Never called with m_cap <= 0.
EEOS_HOST_DEVICE inline Track1D aeval_cap_slope(Track1D t, real w, real m_cap) {
  if (aeval_phase2_slope(t, real(1), w) > m_cap) {
    t.f2 = real(2) * (m_cap - t.f1) / w;
  }
  return t;
}

// Generic tail with the optional monotonicity guard (module header):
// m_floor <= 0 skips the guard entirely; otherwise aeval_floor_slope()
// rewrites the track before aeval_ramp_track() evaluates it.
EEOS_HOST_DEVICE inline Track1D aeval_generic_track(Track1D t, real d, real w, real m_floor) {
  const real sgn = d < real(0) ? real(-1) : real(1);
  return aeval_ramp_track(aeval_floor_slope(t, sgn, w, m_floor), d, w);
}

// M3g: the generic tail with BOTH u-HIGH guards, used only for L's u-high
// primary track (module header point 2). The monotonicity floor is applied
// first and wins lexicographically: a causal cap that fell below the floor
// is raised back to it (the audit reports such points; they do not occur on
// the real tables). Only ever called with d > 0 and m_cap > 0.
EEOS_HOST_DEVICE inline Track1D aeval_capped_track(Track1D t, real d, real w, real m_floor, real m_cap) {
  t = aeval_floor_slope(t, real(1), w, m_floor);
  const real hi = (m_floor > real(0) && m_cap < m_floor) ? m_floor : m_cap;
  return aeval_ramp_track(aeval_cap_slope(t, w, hi), d, w);
}

// M3g: the log-space image g = ln(f) of one BsplineEval3 sample (module
// header point 1). Requires f > 0 -- aeval_extended() checks that before
// calling. Plain chain rule, field by field:
//   g = ln f, g_a = f_a/f, g_ab = f_ab/f - (f_a/f)(f_b/f).
EEOS_HOST_DEVICE inline BsplineEval3 aeval_log_sample(const BsplineEval3 &b) {
  const real inv = real(1) / b.f;
  BsplineEval3 g;
  g.f = std::log(b.f);
  g.fx = b.fx * inv;
  g.fu = b.fu * inv;
  g.fy = b.fy * inv;
  g.fxx = b.fxx * inv - g.fx * g.fx;
  g.fxu = b.fxu * inv - g.fx * g.fu;
  g.fuu = b.fuu * inv - g.fu * g.fu;
  return g;
}

// M3g: the exact inverse of aeval_log_sample() -- f = e^g, f_a = f g_a,
// f_ab = f (g_ab + g_a g_b). Applied to the tail-evolved log sample, so the
// secondary (fx) and frozen (fxx, fy) tracks are mapped back with the
// tail-evolved f, exactly as the design's composition table requires.
EEOS_HOST_DEVICE inline BsplineEval3 aeval_exp_sample(const BsplineEval3 &g) {
  const real f = std::exp(g.f);
  BsplineEval3 b;
  b.f = f;
  b.fx = f * g.fx;
  b.fu = f * g.fu;
  b.fy = f * g.fy;
  b.fxx = f * (g.fxx + g.fx * g.fx);
  b.fxu = f * (g.fxu + g.fx * g.fu);
  b.fuu = f * (g.fuu + g.fu * g.fu);
  return b;
}

// M3g: convert a cap on b = dln(eps_hat)/du into a cap on L's own phase-2
// slope at a u-high seam whose L value is L_b. With E = 10^L * inv_c2 and
// eps_hat = E - shift_hat, b = ln10 * L_u * E / eps_hat, so
// m_L_cap = b_cap * eps_hat / (ln10 * E). Returns 0 ("no cap") when
// eps_hat <= 0: there is no causal statement to make there, and the
// monotonicity floor is then the only meaningful guard.
EEOS_HOST_DEVICE inline real aeval_L_slope_cap(real L_b, real b_cap, real shift_hat, real inv_c2) {
  const real E = std::pow(real(10), L_b) * inv_c2;
  const real eps = E - shift_hat;
  if (!(eps > real(0)) || !(E > real(0))) return real(0);
  return b_cap * eps / (kLn10 * E);
}

// Slope-to-zero variant (module header; L's x-low primary track only):
// overrides the curvature to f2_eff = 2*f1/w, which forces the low-side
// asymptotic slope m = f1 - f2_eff*w/2 to exactly 0, then evaluates
// aeval_ramp_track(). Only ever called with d<0 (the x-low seam).
EEOS_HOST_DEVICE inline Track1D aeval_slope_zero_track(Track1D t, real d, real w) {
  t.f2 = real(2) * t.f1 / w;
  return aeval_ramp_track(t, d, w);
}

enum class TailAxis { x, u };

// One axis's tail parameters for aeval_apply_tail(): w = that axis's grid
// cell (BsplineView3::hx or hu), m_floor = the primary track's slope floor
// (<=0 to skip the guard -- every tail except a u-direction one),
// slope_zero = use aeval_slope_zero_track for the primary track instead of
// the (possibly guarded) generic one (true only for L's x-low tail),
// m_cap = the M3g causal cap on the primary track's phase-2 slope (<=0 to
// skip it -- every tail except L's u-HIGH one).
struct TailSpec {
  real w;
  real m_floor;
  bool slope_zero;
  real m_cap;
};

// Applies one axis's designed tail to a full BsplineEval3-shaped sample by
// splitting it into the four tracks the mixed-derivative composition needs
// (module header) and reassembling the result. `b` may itself already be
// the output of a previous aeval_apply_tail() call -- the corner case, see
// aeval_extended() below.
EEOS_HOST_DEVICE inline BsplineEval3 aeval_apply_tail(const BsplineEval3 &b, real d, TailAxis axis,
                                                       const TailSpec &spec) {
  BsplineEval3 out = b;
  if (axis == TailAxis::x) {
    const Track1D f_in{b.f, b.fx, b.fxx};
    const Track1D f_out = spec.slope_zero ? aeval_slope_zero_track(f_in, d, spec.w)
                                           : aeval_generic_track(f_in, d, spec.w, spec.m_floor);
    const Track1D fu_in{b.fu, b.fxu, real(0)};
    const Track1D fu_out = aeval_generic_track(fu_in, d, spec.w, real(0));
    // (m_cap is not consulted on an x-tail: causality is a u-direction
    // statement in this construction -- see the module header's M3g note.)
    out.f = f_out.f0;
    out.fx = f_out.f1;
    out.fxx = f_out.f2;
    out.fu = fu_out.f0;
    out.fxu = fu_out.f1;
    // fuu, fy: frozen (no fxuu available -- module header).
  } else {
    const Track1D f_in{b.f, b.fu, b.fuu};
    // M3g: the causal cap only ever applies to L's u-HIGH primary track;
    // every other track (and every d < 0) takes the pre-M3g path
    // bit-for-bit.
    const Track1D f_out = (spec.m_cap > real(0) && d > real(0))
                              ? aeval_capped_track(f_in, d, spec.w, spec.m_floor, spec.m_cap)
                              : aeval_generic_track(f_in, d, spec.w, spec.m_floor);
    const Track1D fx_in{b.fx, b.fxu, real(0)};
    const Track1D fx_out = aeval_generic_track(fx_in, d, spec.w, real(0));
    out.f = f_out.f0;
    out.fu = f_out.f1;
    out.fuu = f_out.f2;
    out.fx = fx_out.f0;
    out.fxu = fx_out.f1;
    // fxx, fy: frozen.
  }
  return out;
}

// Per-field extension parameters for aeval_extended(), assembled by
// EntropyEOSView::sigma_ext_spec() / L_ext_spec() (and, at build time,
// by host/adapter_build.cpp's extended eps-floor scan):
//   x_lo..u_hi          the *physical* box (the extended box is the caller's
//                       business -- see aeval_extended()'s doc comment)
//   u_m_floor           this field's u-direction monotonicity-guard floor
//                       (ext_slope_floor_sigma or _L)
//   x_low_slope_zero    slope-to-zero variant for the x-low primary track
//                       (true only for L)
//   u_high_log          M3g: build the u-HIGH tail in log space (true only
//                       for sigma)
//   u_high_b_cap        M3g: cap on b = dln(eps_hat)/du at the u-high seam,
//                       i.e. (1 + cs2_ext_cap)*alpha (>0 only for L, and
//                       only where sigma's log tail supplied an alpha)
//   shift_hat, inv_c2   energy zero point, which turns u_high_b_cap into a
//                       cap on L's own slope (aeval_L_slope_cap())
struct ExtSpec {
  real x_lo, x_hi, u_lo, u_hi;
  real u_m_floor;
  bool x_low_slope_zero;
  bool u_high_log;
  real u_high_b_cap;
  real shift_hat, inv_c2;
};

// Full designed-extension evaluation of one fitted spline (`field` is
// `sigma` or `L`) at a query point (x,u,y). x and u are assumed already
// clamped into the *extended* box by the caller (evaluate()'s x_use / the
// T-solve's u bracket / srange_extended() / sigma_extended() all do this --
// see their call sites); spec.x_lo/x_hi/u_lo/u_hi are the *physical* box.
// Interior points (x in [x_lo,x_hi] and u in [u_lo,u_hi]) fall through to a
// single plain bspline_eval3() call with no tail applied at all, so this
// function is exactly transparent to every existing interior evaluation.
//
// Corner composition order (module header): the raw spline sample is
// always taken at the seam clamped to the *physical* box in both
// directions, the u-tail is applied first, then the x-tail is applied to
// the (possibly already u-tailed) result. M3g's log-space u-high tail sits
// entirely inside the u-tail step, so that order is unchanged.
EEOS_HOST_DEVICE inline BsplineEval3 aeval_extended(const BsplineView3 &field, real x, real u, real y,
                                                      const ExtSpec &spec) {
  const bool u_below = u < spec.u_lo;
  const bool u_above = u > spec.u_hi;
  const bool x_below = x < spec.x_lo;
  const bool x_above = x > spec.x_hi;

  const real u_seam = u_below ? spec.u_lo : (u_above ? spec.u_hi : u);
  const real x_seam = x_below ? spec.x_lo : (x_above ? spec.x_hi : x);

  BsplineEval3 b = bspline_eval3(field, x_seam, u_seam, y);

  if (u_below || u_above) {
    const real seam = u_below ? spec.u_lo : spec.u_hi;
    const real d = u - seam;
    if (u_above && spec.u_high_log && b.f > real(0)) {
      // M3g log-sigma tail: the same machinery, run on g = ln(sigma). The
      // monotonicity floor transfers as m_floor/sigma_b (module header).
      const real m_floor_g = spec.u_m_floor > real(0) ? spec.u_m_floor / b.f : real(0);
      const BsplineEval3 g = aeval_apply_tail(aeval_log_sample(b), d, TailAxis::u,
                                               TailSpec{field.hu, m_floor_g, false, real(0)});
      b = aeval_exp_sample(g);
    } else {
      // M3g causal clamp (L's u-high tail only; 0 everywhere else, which
      // reproduces the pre-M3g construction bit-for-bit).
      const real m_cap = (u_above && spec.u_high_b_cap > real(0))
                             ? aeval_L_slope_cap(b.f, spec.u_high_b_cap, spec.shift_hat, spec.inv_c2)
                             : real(0);
      b = aeval_apply_tail(b, d, TailAxis::u, TailSpec{field.hu, spec.u_m_floor, false, m_cap});
    }
  }
  if (x_below || x_above) {
    const real seam = x_below ? spec.x_lo : spec.x_hi;
    b = aeval_apply_tail(b, x - seam, TailAxis::x,
                          TailSpec{field.hx, real(0), x_below && spec.x_low_slope_zero, real(0)});
  }
  return b;
}

// M3g: sigma's u-HIGH log-space asymptotic (phase-2) growth rate
// alpha = dln(sigma)/du -- exactly the slope the log tail continues with,
// and the quantity L's causal clamp is measured against
// (c_s^2(tail) = b/alpha - 1). Returns 0 when the log tail is inactive
// (sigma_b <= 0 at the seam), which disables the clamp. `x_seam` must
// already be clamped to the physical [x_lo, x_hi].
EEOS_HOST_DEVICE inline real aeval_sigma_u_high_alpha(const BsplineView3 &sigma, real x_seam, real u_hi,
                                                        real y, real m_floor) {
  const BsplineEval3 b = bspline_eval3(sigma, x_seam, u_hi, y);
  if (!(b.f > real(0))) return real(0);
  const BsplineEval3 g = aeval_log_sample(b);
  const real m_floor_g = m_floor > real(0) ? m_floor / b.f : real(0);
  const Track1D t = aeval_floor_slope(Track1D{g.f, g.fu, g.fuu}, real(1), sigma.hu, m_floor_g);
  return aeval_phase2_slope(t, real(1), sigma.hu);
}

} // namespace detail

// M3g audit hook: everything the u-HIGH causal slope clamp is built from at
// one seam point, so host/adapter_audit.cpp's extension-band map can report
// the (never expected, but lexicographically resolved) monotonicity-floor /
// causal-cap conflicts without duplicating the arithmetic. See
// EntropyEOSView::u_high_tail_info().
struct UHighTailInfo {
  real alpha;      // sigma's log-tail growth rate dln(sigma)/du (0: log tail inactive)
  real b_raw;      // L's asymptotic b = dln(eps_hat)/du after the monotonicity floor, before the cap
  real b_cap;      // the causal cap on b, (1 + cs2_ext_cap)*alpha (0: inactive)
  real m_L_raw;    // L's phase-2 slope after the monotonicity floor, before the cap
  real m_L_cap;    // b_cap expressed in L-slope units (0: inactive)
  bool clamped;    // the cap actually lowers the effective slope at this point
  bool floor_wins; // the cap fell below ext_slope_floor_L -- the floor wins (lexicographic)
};

// Device-portable POD view of a built EntropyEOS (see
// host/adapter_build.hpp): the two fitted splines (sigma = entropy,
// L = log10(eps_cgs + energy_shift_cgs)) plus the scalar parameters and
// physical box the run-time construction needs. sigma/L's BsplineView3::x0
// already carries the kappa density shift, so evaluate() needs no further
// kappa bookkeeping beyond the Stage C division below.
struct EntropyEOSView {
  BsplineView3 sigma, L;

  real kappa;     // <= 1; m_B* = kappa * m_B_table (eos-adapter-F-to-U.md S5)
  real shift_hat; // energy_shift_cgs / c^2
  real conv_t;    // MeV_to_erg / (m_B_table_g * c^2); That_t = conv_t * 10^u
  real inv_c2;    // 1 / c_light_cm_s^2; E = 10^L * inv_c2 = eps_hat + shift_hat

  real x_lo, x_hi; // physical box in x* = log10(rho* [g/cc])
  real u_lo, u_hi; // physical box in u = log10(T [MeV])
  real y_lo, y_hi; // physical box in y = Ye

  // M2d-2 domain extensions (eos-adapter-F-to-U.md S7 / this file's TAIL
  // MATHEMATICS): the extended box -- BuildOptions::ext_cells grid cells
  // beyond the physical box on every side of x and u -- that evaluate()
  // tails into instead of hard-clamping, plus the per-field u-direction
  // monotonicity-guard slope floors (BuildOptions::ext_slope_floor_sigma /
  // _L). Ye has no extension (never iterated -- still a hard clamp+flag).
  real x_ext_lo, x_ext_hi;
  real u_ext_lo, u_ext_hi;
  real ext_slope_floor_sigma, ext_slope_floor_L;

  // M3g (eos-causal-tail.md / this file's "CAUSAL TAILS"): the causality
  // bound the u-HIGH tail's asymptotic slopes are held to,
  // b_eff <= (1 + cs2_ext_cap)*alpha_eff, i.e. c_s^2(tail) <= cs2_ext_cap.
  // BuildOptions::cs2_ext_cap, default 0.99 (matching the M3f data-side
  // cs2_cap).
  real cs2_ext_cap;

  int max_iter; // T-solve iteration cap (default 50, set at build)

  EEOS_HOST_DEVICE EOSPoint evaluate(real rho_star, real s, real ye, real u_guess) const;

  // The per-field extension parameters aeval_extended() takes (M3g; see
  // detail::ExtSpec). `b_cap` for L comes from u_high_b_cap() below, and is
  // 0 whenever the query does not sit above the u_hi seam.
  EEOS_HOST_DEVICE detail::ExtSpec sigma_ext_spec() const {
    return detail::ExtSpec{x_lo,
                           x_hi,
                           u_lo,
                           u_hi,
                           ext_slope_floor_sigma,
                           /*x_low_slope_zero=*/false,
                           /*u_high_log=*/true,
                           /*u_high_b_cap=*/real(0),
                           shift_hat,
                           inv_c2};
  }
  EEOS_HOST_DEVICE detail::ExtSpec L_ext_spec(real b_cap) const {
    return detail::ExtSpec{x_lo,
                           x_hi,
                           u_lo,
                           u_hi,
                           ext_slope_floor_L,
                           /*x_low_slope_zero=*/true,
                           /*u_high_log=*/false,
                           b_cap,
                           shift_hat,
                           inv_c2};
  }

  // M3g: the causal cap on b = dln(eps_hat)/du for an L evaluation whose u
  // sits above the u_hi seam, at the x already clamped into the extended
  // box (x_use in evaluate()). One extra sigma spline sample, taken only on
  // the u-high tail path -- in-box evaluations never call this.
  EEOS_HOST_DEVICE real u_high_b_cap(real x_use, real y) const {
    const real x_seam = detail::aeval_clamp(x_use, x_lo, x_hi);
    const real alpha =
        detail::aeval_sigma_u_high_alpha(sigma, x_seam, u_hi, y, ext_slope_floor_sigma);
    return alpha > real(0) ? (real(1) + cs2_ext_cap) * alpha : real(0);
  }

  // M3g audit hook (see UHighTailInfo): the u-high seam's clamp arithmetic
  // at (rho*, Ye), for host/adapter_audit.cpp's extension-band map.
  EEOS_HOST_DEVICE UHighTailInfo u_high_tail_info(real rho_star, real ye) const;

  // The chain-rule half of evaluate() (its steps 4-6), at an already-located
  // point: `x_use` is x = log10(rho*) clamped into the EXTENDED box, `u` the
  // solved log10(T_MeV), `y` the clamped Ye, and `flags`/`iters` are copied
  // into the returned EOSPoint verbatim. evaluate() is exactly "locate u,
  // then call this", so the two never drift apart.
  //
  // Exposed because M3g's extension-band map (host/adapter_audit.cpp class E)
  // walks a band in NATIVE (x, u, y) coordinates, where running the T-solve
  // would only re-derive the u it was handed -- 3 spline samples per point
  // instead of ~12, over tens of millions of points.
  EEOS_HOST_DEVICE EOSPoint eval_at(real x_use, real u, real y, unsigned flags, int iters) const;

  // Pointwise PHYSICAL entropy range [sigma(u_lo), sigma(u_hi)] (unchanged
  // from M2b/M2c -- see eos-adapter-F-to-U.md S7's srange definition).
  EEOS_HOST_DEVICE SRange srange(real rho_star, real ye) const;

  // M2d-2: the same pointwise range evaluated at the EXTENDED box edges
  // (x clamped to [x_ext_lo,x_ext_hi], u = u_ext_lo/u_ext_hi), i.e. the
  // widest s an evaluate() call at this (rho*, Ye) can resolve before
  // hitting the hard clamp at the extended edge. Solver bracketing (M3)
  // wants both this and the physical srange().
  EEOS_HOST_DEVICE SRange srange_extended(real rho_star, real ye) const;

  // M2d-2 audit/testing hook: the extended entropy spline's *value* at an
  // arbitrary (rho*, u=log10(T_MeV), Ye) -- u need not lie in the physical
  // [u_lo,u_hi] box, only the extended one (clamped here if it does not).
  // Lets a caller pick a point in the designed extension zone by its
  // native (x,u,y) coordinates and recover the s that evaluate() would
  // need to land back on it (see tests/test_adapter.cpp's extended-T-solve
  // round trip and host/adapter_audit.cpp's extension-seam-jump audit).
  EEOS_HOST_DEVICE real sigma_extended(real rho_star, real u, real ye) const;
};

EEOS_HOST_DEVICE inline SRange EntropyEOSView::srange(real rho_star, real ye) const {
  const real x = detail::aeval_clamp(std::log10(rho_star), x_lo, x_hi);
  const real y = detail::aeval_clamp(ye, y_lo, y_hi);
  const BsplineEval3 slo = bspline_eval3(sigma, x, u_lo, y);
  const BsplineEval3 shi = bspline_eval3(sigma, x, u_hi, y);
  return SRange{slo.f, shi.f};
}

EEOS_HOST_DEVICE inline SRange EntropyEOSView::srange_extended(real rho_star, real ye) const {
  const real x = detail::aeval_clamp(std::log10(rho_star), x_ext_lo, x_ext_hi);
  const real y = detail::aeval_clamp(ye, y_lo, y_hi);
  const detail::ExtSpec spec = sigma_ext_spec();
  const BsplineEval3 slo = detail::aeval_extended(sigma, x, u_ext_lo, y, spec);
  const BsplineEval3 shi = detail::aeval_extended(sigma, x, u_ext_hi, y, spec);
  return SRange{slo.f, shi.f};
}

EEOS_HOST_DEVICE inline real EntropyEOSView::sigma_extended(real rho_star, real u, real ye) const {
  const real x = detail::aeval_clamp(std::log10(rho_star), x_ext_lo, x_ext_hi);
  const real y = detail::aeval_clamp(ye, y_lo, y_hi);
  const real uu = detail::aeval_clamp(u, u_ext_lo, u_ext_hi);
  return detail::aeval_extended(sigma, x, uu, y, sigma_ext_spec()).f;
}

// M3g: recompute the u-high seam's clamp arithmetic at one (rho*, Ye) --
// exactly the quantities aeval_extended() derives internally on the L
// u-high path, exposed for the extension-band audit.
EEOS_HOST_DEVICE inline UHighTailInfo EntropyEOSView::u_high_tail_info(real rho_star, real ye) const {
  const real x_seam = detail::aeval_clamp(std::log10(rho_star), x_lo, x_hi);
  const real y = detail::aeval_clamp(ye, y_lo, y_hi);

  UHighTailInfo info;
  info.alpha = detail::aeval_sigma_u_high_alpha(sigma, x_seam, u_hi, y, ext_slope_floor_sigma);
  info.b_cap = info.alpha > real(0) ? (real(1) + cs2_ext_cap) * info.alpha : real(0);

  const BsplineEval3 Lb = bspline_eval3(L, x_seam, u_hi, y);
  const detail::Track1D t =
      detail::aeval_floor_slope(detail::Track1D{Lb.f, Lb.fu, Lb.fuu}, real(1), L.hu, ext_slope_floor_L);
  info.m_L_raw = detail::aeval_phase2_slope(t, real(1), L.hu);
  info.m_L_cap =
      info.b_cap > real(0) ? detail::aeval_L_slope_cap(Lb.f, info.b_cap, shift_hat, inv_c2) : real(0);

  const real E = std::pow(real(10), Lb.f) * inv_c2;
  const real eps = E - shift_hat;
  info.b_raw = eps > real(0) ? detail::kLn10 * info.m_L_raw * E / eps : real(0);

  info.floor_wins = info.m_L_cap > real(0) && ext_slope_floor_L > real(0) &&
                     info.m_L_cap < ext_slope_floor_L;
  const real hi = info.floor_wins ? ext_slope_floor_L : info.m_L_cap;
  info.clamped = info.m_L_cap > real(0) && info.m_L_raw > hi;
  return info;
}

EEOS_HOST_DEVICE inline EOSPoint EntropyEOSView::evaluate(real rho_star, real s, real ye,
                                                           real u_guess) const {
  unsigned flags = 0u;

  // --- 1-2: clamp Ye; locate rho* against the physical/extended box
  // (S3 steps 1-2 / S7, M2d-2) ----------------------------------------------
  const real x = std::log10(rho_star);
  real x_use; // the x actually evaluated at: x itself if in range, softly
              // extended (still tailed, not clamped) up to x_ext_lo/hi,
              // hard-clamped only beyond that.
  if (x < x_lo) {
    flags |= flag_ext_rho_low;
    x_use = x < x_ext_lo ? x_ext_lo : x;
  } else if (x > x_hi) {
    flags |= flag_oob_rho_high;
    x_use = x > x_ext_hi ? x_ext_hi : x;
  } else {
    x_use = x;
  }

  real y = ye;
  if (y < y_lo) {
    y = y_lo;
    flags |= flag_clamp_ye;
  } else if (y > y_hi) {
    y = y_hi;
    flags |= flag_clamp_ye;
  }

  // --- 3: T-solve, sigma_ext(x_use,u,y) = s for u, on the EXTENDED u
  // bracket (M2d-2). sigma_ext is monotone in u by the u-tail's slope-floor
  // guard, so the bracket is globally invertible exactly as the physical
  // sigma was before; the safeguarded Newton/bisection loop itself is
  // unchanged apart from its bracket and its point evaluator (S3 point 2's
  // safeguard still applies verbatim). -------------------------------------
  const detail::ExtSpec sig_spec = sigma_ext_spec();
  const BsplineEval3 sig_lo = detail::aeval_extended(sigma, x_use, u_ext_lo, y, sig_spec);
  const BsplineEval3 sig_hi = detail::aeval_extended(sigma, x_use, u_ext_hi, y, sig_spec);

  real u;
  int iters = 0;
  if (s <= sig_lo.f) {
    // Below even the extended bracket: hard clamp to the extended edge,
    // still flagged (S7: "the extension makes this genuinely rare").
    u = u_ext_lo;
  } else if (s >= sig_hi.f) {
    u = u_ext_hi;
  } else {
    real lo = u_ext_lo, hi = u_ext_hi;

    real u0;
    if (!detail::aeval_is_nan(u_guess)) {
      u0 = detail::aeval_clamp(u_guess, u_ext_lo, u_ext_hi);
    } else {
      // Secant estimate from the endpoint values -- better than a plain
      // midpoint start (S3 point 2).
      u0 = lo + (hi - lo) * (s - sig_lo.f) / (sig_hi.f - sig_lo.f);
      u0 = detail::aeval_clamp(u0, lo, hi);
    }

    const real s_scale = std::fabs(s) > real(1) ? std::fabs(s) : real(1);
    real uu = u0;
    bool converged = false;

    while (iters < max_iter) {
      ++iters;
      const BsplineEval3 e = detail::aeval_extended(sigma, x_use, uu, y, sig_spec);
      const real g = e.f - s;

      // Maintain the bracket by sign (sigma_ext strictly increasing in u).
      if (g < real(0)) {
        lo = uu;
      } else {
        hi = uu;
      }

      if (std::fabs(g) <= real(1e-12) * s_scale) {
        converged = true;
        break;
      }

      // Newton step, accepted only if it stays inside the bracket and
      // sigma_ext_u > 0 at the iterate; bisect otherwise (S3 point 2's
      // safeguard).
      real u_next;
      bool newton_ok = false;
      if (e.fu > real(0)) {
        u_next = uu - g / e.fu;
        newton_ok = (u_next > lo) && (u_next < hi);
      }
      if (!newton_ok) {
        u_next = real(0.5) * (lo + hi);
      }

      const real du = u_next - uu;
      uu = u_next;

      const real u_scale = std::fabs(uu) > real(1) ? std::fabs(uu) : real(1);
      if (std::fabs(du) <= real(1e-13) * u_scale) {
        converged = true;
        break;
      }
    }

    if (!converged) {
      flags |= flag_maxiter;
    }
    u = uu;
  }

  // M2d-2: flag_ext_s_low/high are set by comparing the SOLVED u against
  // the physical box, not by pre-clamping s (module header / EOSPoint's
  // doc comment).
  if (u < u_lo) {
    flags |= flag_ext_s_low;
  } else if (u > u_hi) {
    flags |= flag_ext_s_high;
  }

  // --- 4-6: chain rule and derived quantities at the located point -------
  return eval_at(x_use, u, y, flags, iters);
}

EEOS_HOST_DEVICE inline EOSPoint EntropyEOSView::eval_at(real x_use, real u, real y, unsigned flags,
                                                          int iters) const {
  // --- 4: spline evaluations at the located point (extended) -------------
  const BsplineEval3 sig = detail::aeval_extended(sigma, x_use, u, y, sigma_ext_spec());
  // M3g: above the u_hi seam, L's tail slope is held to the causal bound
  // b <= (1 + cs2_ext_cap)*alpha, alpha being sigma's own log-tail growth
  // rate at the same seam (one extra sigma sample, on this path only).
  const real b_cap = u > u_hi ? u_high_b_cap(x_use, y) : real(0);
  const BsplineEval3 Lv = detail::aeval_extended(L, x_use, u, y, L_ext_spec(b_cap));

  // --- 5: chain rule -------------------------------------------------------
  const real lambda = detail::kLn10;
  const real rho_eff = std::pow(real(10), x_use); // rho* at the (possibly extended) solved point
  const real That_t = conv_t * std::pow(real(10), u);

  // Stage A part 1: undo the log10-energy fit analytically.
  // E = 10^L / c^2 = eps_hat + shift_hat.
  const real E = std::pow(real(10), Lv.f) * inv_c2;
  const real eps_hat = E - shift_hat;

  const real eps_x = lambda * E * Lv.fx;
  const real eps_u = lambda * E * Lv.fu;
  const real eps_y = lambda * E * Lv.fy;
  const real eps_xx = lambda * E * (Lv.fxx + lambda * Lv.fx * Lv.fx);
  const real eps_uu = lambda * E * (Lv.fuu + lambda * Lv.fu * Lv.fu);
  const real eps_xu = lambda * E * (Lv.fxu + lambda * Lv.fx * Lv.fu);

  // Stage A part 2: (x,u) spline partials -> physical (rho*, That_t)
  // partials, for sigma and eps alike.
  const real inv_lrho = real(1) / (lambda * rho_eff);
  const real inv_lrho2 = inv_lrho * inv_lrho;
  const real inv_lT = real(1) / (lambda * That_t);
  const real inv_lT2 = inv_lT * inv_lT;

  const real sig_rho = sig.fx * inv_lrho;
  const real sig_rhorho = (sig.fxx - lambda * sig.fx) * inv_lrho2;
  const real sig_T = sig.fu * inv_lT;
  const real sig_TT = (sig.fuu - lambda * sig.fu) * inv_lT2;
  const real sig_rhoT = sig.fxu / (lambda * lambda * rho_eff * That_t);
  const real sig_y = sig.fy;

  const real eps_rho = eps_x * inv_lrho;
  const real eps_rhorho = (eps_xx - lambda * eps_x) * inv_lrho2;
  const real eps_T = eps_u * inv_lT;
  const real eps_TT = (eps_uu - lambda * eps_u) * inv_lT2;
  const real eps_rhoT = eps_xu / (lambda * lambda * rho_eff * That_t);

  // Stage B: implicit function theorem (eos-adapter-F-to-U.md S3.1), with
  // "T" meaning That_t and "e"/"sigma" the physical partials above.
  const real T_s = real(1) / sig_T;
  const real T_rho = -sig_rho / sig_T;
  const real T_rhos = -(sig_rhoT + sig_TT * T_rho) / (sig_T * sig_T);
  const real T_rhorho = -(sig_rhorho + real(2) * sig_rhoT * T_rho + sig_TT * T_rho * T_rho) / sig_T;

  const real Uh = eps_hat;
  const real Uh_s = eps_T * T_s;
  const real Uh_rho = eps_rho + eps_T * T_rho;
  const real Uh_rhos = eps_rhoT * T_s + eps_TT * T_rho * T_s + eps_T * T_rhos;
  const real Uh_rhorho = eps_rhorho + real(2) * eps_rhoT * T_rho + eps_TT * T_rho * T_rho + eps_T * T_rhorho;
  const real mu_h = eps_y + eps_T * (-sig_y / sig_T);

  // Stage C: kappa re-zeroing transform (eos-adapter-F-to-U.md S5).
  const real U = (real(1) + Uh) / kappa - real(1);
  const real U_rho = Uh_rho / kappa;
  const real U_s = Uh_s / kappa;
  const real U_rhorho = Uh_rhorho / kappa;
  const real U_rhos = Uh_rhos / kappa;
  const real mu_tilde = mu_h / kappa;

  // --- 6: derived quantities -----------------------------------------------
  const real p = rho_eff * rho_eff * U_rho;
  const real h = real(1) + U + p / rho_eff;
  const real cs2 = (real(2) * rho_eff * U_rho + rho_eff * rho_eff * U_rhorho) / h;

  EOSPoint pt;
  pt.U = U;
  pt.U_rho = U_rho;
  pt.U_s = U_s;
  pt.U_rhorho = U_rhorho;
  pt.U_rhos = U_rhos;
  pt.That = U_s;
  pt.p = p;
  pt.h = h;
  pt.cs2 = cs2;
  pt.T_F_MeV = std::pow(real(10), u);
  pt.mu_tilde = mu_tilde;
  pt.u_solved = u;
  pt.iters = iters;
  pt.flags = flags;
  return pt;
}

} // namespace eeos
