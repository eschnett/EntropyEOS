// entropy_eos/core/prim2con.hpp
//
// M3a rapidity-form prim2con (con2prim-entropy-rapidity.md SS4-5,7): maps a
// primitive state (rho, s, Ye, w, B^2, cos_vB) through one EOS evaluation to
// the scalar conservative state (D, tau, D_Y, S_par, S_perp, B^2) the con2prim
// solver's SS6-9 residuals are built from. Header-only, device-ready (see
// CODE.md "Layout"): no STL containers, no exceptions, no allocation, every
// function EEOS_HOST_DEVICE.
//
// Units (adapter units throughout, geometric c=1 -- see CODE.md "M2 design
// notes" and adapter_eval.hpp's module header): rho = rho* (kappa-rescaled
// g/cc), s in kB/baryon, U = eps/c^2 dimensionless, p = EOSPoint::p (g/cc),
// h = EOSPoint::h dimensionless, z = rho*h*W^2 (g/cc, "total enthalpy
// density"). v = tanh(w), W = cosh(w) dimensionless. B^2 is B_iB^i with 4*pi
// absorbed, in g/cc (so magnetic pressure = B^2/2 in the same units as p).
// Conservatives: D = rho*W, D_Y = D*Ye, tau = E - D, S_par/S_perp (g/cc).
//
// Scalar reduction (con2prim-entropy-rapidity.md S1): forming S_par =
// (S_iB^i)/|B| and S_perp = sqrt(S^2 - S_par^2) from the GR metric algebra
// (gamma_ij) is the CALLER's job; this header's scalar prim2con() takes
// those projections directly. A flat-metric Cartesian convenience overload
// is provided below for callers that have not already done that reduction.

#pragma once

#include <cmath>
#include <limits>

#include "entropy_eos/core/adapter_eval.hpp"
#include "entropy_eos/core/defs.hpp"

namespace eeos {

namespace detail {

// A quiet NaN of type `real`, usable as a default-argument sentinel for "no
// warm-start guess" -- the receiving end is EntropyEOSView::evaluate(),
// whose own u_guess NaN test is detail::aeval_is_nan() (adapter_eval.hpp).
EEOS_HOST_DEVICE inline real p2c_nan() { return std::numeric_limits<real>::quiet_NaN(); }

} // namespace detail

// Output of prim2con(): the scalar conservative state the con2prim solver's
// residuals are built from.
struct Prim2ConOut {
  real D, tau, D_Y, S_par, S_perp, B2;
};

// Rapidity-form prim2con, scalar/projected form (con2prim-entropy-rapidity.md
// S4 hydro ansatz + S5 magnetic projection + S7 cancellation-free tau).
// cos_vB in [-1,1] is the cosine of the angle between v and B; its value is
// irrelevant (never affects the result) when B2 == 0 or w == 0, since v_par
// and v_perp both then reduce to values independent of direction.
EEOS_HOST_DEVICE inline Prim2ConOut prim2con(const EntropyEOSView &eos, real rho, real s, real ye, real w,
                                              real B2, real cos_vB, real u_guess = detail::p2c_nan()) {
  const EOSPoint pt = eos.evaluate(rho, s, ye, u_guess);

  const real W = std::cosh(w);
  const real v = std::tanh(w);
  const real sin_vB2 = real(1) - cos_vB * cos_vB;
  const real sin_vB = std::sqrt(sin_vB2 > real(0) ? sin_vB2 : real(0));
  const real v_par = v * cos_vB;
  const real v_perp = v * sin_vB;

  // S4: z = rho*h*W^2, the "total enthalpy density".
  const real z = rho * pt.h * W * W;

  Prim2ConOut out;
  out.D = rho * W;
  out.D_Y = out.D * ye;
  out.B2 = B2;

  // S5: the momentum projections. The +B^2 inertia term cancels exactly in
  // the parallel direction and only survives perpendicular to B -- this is
  // the design doc's S_par = z*v_par, S_perp = (z+B^2)*v_perp verbatim, not
  // a simplification of a more general common form.
  out.S_par = z * v_par;
  out.S_perp = (z + B2) * v_perp;

  // S7 cancellation-free tau = E - D: every term below is individually
  // small and (for eps,p >= 0) manifestly nonnegative, so no digits are
  // lost even as w -> 0 in a cold flow (unlike naively forming E - D from
  // E = z - p + B^2/2*(1+v^2) - (B.v)^2/2, which cancels two O(D) terms
  // down to an O(tau) result).
  //
  // Algebraic check that this equals E - D exactly (used by test 1, and
  // worth re-deriving here since the whole point of S7 is this identity):
  // with z = rho*h*W^2 = (rho + rho*U + p)*W^2 (from h = 1 + U + p/rho),
  //
  //   E - D = (z - p + B^2/2*(1+v^2) - B^2/2*v_par^2) - D
  //         = rho*W^2 + rho*U*W^2 + p*W^2 - p + [B^2 terms] - rho*W
  //         = rho*W*(W-1) + rho*U*W^2 + p*(W^2-1) + [B^2 terms]
  //
  // and rho*W*(W-1) = D*(W-1) = 2*D*sinh(w/2)^2 (half-angle identity
  // cosh(w)-1 = 2*sinh(w/2)^2, D = rho*W), p*(W^2-1) = p*sinh(w)^2
  // (cosh^2-sinh^2=1), and (B.v)^2 = B^2*v_par^2 because the
  // B-perpendicular component of v does not contribute to B.v. Substituting
  // recovers exactly the tau_model below.
  const real half_sinh = std::sinh(real(0.5) * w);
  const real sinh_w = std::sinh(w);
  out.tau = real(2) * out.D * half_sinh * half_sinh + rho * pt.U * W * W + pt.p * sinh_w * sinh_w +
            real(0.5) * B2 * (real(1) + v * v) - real(0.5) * B2 * v_par * v_par;

  return out;
}

// Flat-metric 3-vector convenience overload (con2prim-entropy-rapidity.md
// S1: the general-metric scalar reduction of S_i, B^i to S_par/S_perp is the
// caller's job when gamma_ij is involved; this is the flat-metric special
// case done for a caller that has Cartesian vectors instead). `v_dir` must
// be a unit vector giving the velocity's direction -- its magnitude is
// carried by `w` (v = tanh w), not by |v_dir|. Built directly on the scalar
// prim2con() above: derives (B2, cos_vB) from v_dir/B, calls the scalar
// form, then reconstructs the Cartesian S[3] from (S_par, S_perp) via the
// same (b, perpendicular-direction) basis used internally -- trivial
// algebra, no new physics.
EEOS_HOST_DEVICE inline Prim2ConOut prim2con(const EntropyEOSView &eos, real rho, real s, real ye, real w,
                                              const real v_dir[3], const real B[3], real S_out[3],
                                              real u_guess = detail::p2c_nan()) {
  const real B2 = B[0] * B[0] + B[1] * B[1] + B[2] * B[2];
  const real Bmag = std::sqrt(B2);

  real b[3];
  real cos_vB;
  if (Bmag > real(0)) {
    const real inv_Bmag = real(1) / Bmag;
    b[0] = B[0] * inv_Bmag;
    b[1] = B[1] * inv_Bmag;
    b[2] = B[2] * inv_Bmag;
    cos_vB = v_dir[0] * b[0] + v_dir[1] * b[1] + v_dir[2] * b[2];
  } else {
    // No field to project against: the whole of v is "perpendicular" (see
    // the S_out assembly below, where b == 0 makes S_par's contribution
    // vanish and t == v_dir carries all of S_perp).
    b[0] = b[1] = b[2] = real(0);
    cos_vB = real(0);
  }

  const Prim2ConOut out = prim2con(eos, rho, s, ye, w, B2, cos_vB, u_guess);

  // Perpendicular unit direction: Gram-Schmidt v_dir against b. Left as
  // v_dir itself when the projection is degenerate (B2 == 0, b == 0 above,
  // so v_dir already has no B-parallel component to remove) or when
  // sin(angle) ~ 0 (v_dir parallel/antiparallel to b) -- in the latter case
  // out.S_perp is itself ~0 (v_perp = v*sin_vB), so whatever direction `t`
  // takes contributes nothing to S_out.
  const real sin_vB2_raw = real(1) - cos_vB * cos_vB;
  const real sin_vB2 = sin_vB2_raw > real(0) ? sin_vB2_raw : real(0);
  real t[3] = {v_dir[0], v_dir[1], v_dir[2]};
  if (Bmag > real(0) && sin_vB2 > real(1e-300)) {
    const real inv_sin = real(1) / std::sqrt(sin_vB2);
    t[0] = (v_dir[0] - cos_vB * b[0]) * inv_sin;
    t[1] = (v_dir[1] - cos_vB * b[1]) * inv_sin;
    t[2] = (v_dir[2] - cos_vB * b[2]) * inv_sin;
  }

  S_out[0] = out.S_par * b[0] + out.S_perp * t[0];
  S_out[1] = out.S_par * b[1] + out.S_perp * t[1];
  S_out[2] = out.S_par * b[2] + out.S_perp * t[2];

  return out;
}

} // namespace eeos
