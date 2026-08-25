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

// Generic tail with the optional monotonicity guard (module header):
// m_floor <= 0 skips the guard entirely (every track except the
// u-direction primary track of sigma/L); m_floor > 0 raises the effective
// f1 to at least m_floor, then caps the effective f2 so the asymptotic
// slope m = f1 +- f2*w/2 (sign per side) does not fall back below m_floor,
// before evaluating aeval_ramp_track().
EEOS_HOST_DEVICE inline Track1D aeval_generic_track(Track1D t, real d, real w, real m_floor) {
  if (m_floor > real(0)) {
    const real f1_eff = t.f1 < m_floor ? m_floor : t.f1;
    const real sgn = d < real(0) ? real(-1) : real(1);
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
  return aeval_ramp_track(t, d, w);
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
// the (possibly guarded) generic one (true only for L's x-low tail).
struct TailSpec {
  real w;
  real m_floor;
  bool slope_zero;
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
    out.f = f_out.f0;
    out.fx = f_out.f1;
    out.fxx = f_out.f2;
    out.fu = fu_out.f0;
    out.fxu = fu_out.f1;
    // fuu, fy: frozen (no fxuu available -- module header).
  } else {
    const Track1D f_in{b.f, b.fu, b.fuu};
    const Track1D f_out = aeval_generic_track(f_in, d, spec.w, spec.m_floor);
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

// Full designed-extension evaluation of one fitted spline (`field` is
// `sigma` or `L`) at a query point (x,u,y). x and u are assumed already
// clamped into the *extended* box by the caller (evaluate()'s x_use / the
// T-solve's u bracket / srange_extended() / sigma_extended() all do this --
// see their call sites); x_lo/x_hi/u_lo/u_hi are the *physical* box.
// Interior points (x in [x_lo,x_hi] and u in [u_lo,u_hi]) fall through to a
// single plain bspline_eval3() call with no tail applied at all, so this
// function is exactly transparent to every existing interior evaluation.
//
// Corner composition order (module header): the raw spline sample is
// always taken at the seam clamped to the *physical* box in both
// directions, the u-tail is applied first, then the x-tail is applied to
// the (possibly already u-tailed) result.
//
// `u_m_floor` is the u-direction monotonicity-guard floor for this field
// (ext_slope_floor_sigma or _L); `x_low_slope_zero` selects the slope-to-
// zero variant for this field's x-low primary track (true only for L).
EEOS_HOST_DEVICE inline BsplineEval3 aeval_extended(const BsplineView3 &field, real x, real u, real y,
                                                      real x_lo, real x_hi, real u_lo, real u_hi,
                                                      real u_m_floor, bool x_low_slope_zero) {
  const bool u_below = u < u_lo;
  const bool u_above = u > u_hi;
  const bool x_below = x < x_lo;
  const bool x_above = x > x_hi;

  const real u_seam = u_below ? u_lo : (u_above ? u_hi : u);
  const real x_seam = x_below ? x_lo : (x_above ? x_hi : x);

  BsplineEval3 b = bspline_eval3(field, x_seam, u_seam, y);

  if (u_below || u_above) {
    const real seam = u_below ? u_lo : u_hi;
    b = aeval_apply_tail(b, u - seam, TailAxis::u, TailSpec{field.hu, u_m_floor, false});
  }
  if (x_below || x_above) {
    const real seam = x_below ? x_lo : x_hi;
    b = aeval_apply_tail(b, x - seam, TailAxis::x,
                          TailSpec{field.hx, real(0), x_below && x_low_slope_zero});
  }
  return b;
}

} // namespace detail

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

  int max_iter; // T-solve iteration cap (default 50, set at build)

  EEOS_HOST_DEVICE EOSPoint evaluate(real rho_star, real s, real ye, real u_guess) const;

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
  const BsplineEval3 slo =
      detail::aeval_extended(sigma, x, u_ext_lo, y, x_lo, x_hi, u_lo, u_hi, ext_slope_floor_sigma, false);
  const BsplineEval3 shi =
      detail::aeval_extended(sigma, x, u_ext_hi, y, x_lo, x_hi, u_lo, u_hi, ext_slope_floor_sigma, false);
  return SRange{slo.f, shi.f};
}

EEOS_HOST_DEVICE inline real EntropyEOSView::sigma_extended(real rho_star, real u, real ye) const {
  const real x = detail::aeval_clamp(std::log10(rho_star), x_ext_lo, x_ext_hi);
  const real y = detail::aeval_clamp(ye, y_lo, y_hi);
  const real uu = detail::aeval_clamp(u, u_ext_lo, u_ext_hi);
  return detail::aeval_extended(sigma, x, uu, y, x_lo, x_hi, u_lo, u_hi, ext_slope_floor_sigma, false).f;
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
  const BsplineEval3 sig_lo = detail::aeval_extended(sigma, x_use, u_ext_lo, y, x_lo, x_hi, u_lo, u_hi,
                                                       ext_slope_floor_sigma, false);
  const BsplineEval3 sig_hi = detail::aeval_extended(sigma, x_use, u_ext_hi, y, x_lo, x_hi, u_lo, u_hi,
                                                       ext_slope_floor_sigma, false);

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
      const BsplineEval3 e =
          detail::aeval_extended(sigma, x_use, uu, y, x_lo, x_hi, u_lo, u_hi, ext_slope_floor_sigma, false);
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

  // --- 4: final spline evaluations at the solved point (extended) --------
  const BsplineEval3 sig =
      detail::aeval_extended(sigma, x_use, u, y, x_lo, x_hi, u_lo, u_hi, ext_slope_floor_sigma, false);
  const BsplineEval3 Lv =
      detail::aeval_extended(L, x_use, u, y, x_lo, x_hi, u_lo, u_hi, ext_slope_floor_L, true);

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
