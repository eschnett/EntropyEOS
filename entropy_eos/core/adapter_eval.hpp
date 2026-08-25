// entropy_eos/core/adapter_eval.hpp
//
// M2b adapter core: EntropyEOSView::evaluate() implements the
// F(rho,T,Ye) -> U(rho,s,Ye) construction of eos-adapter-F-to-U.md (S2-S5,
// S11): a warm-started safeguarded Newton T-solve on the fitted entropy
// spline, followed by the S3.1 chain rule composing the two fitted splines
// (Sigma-hat = entropy, L-hat = log10(eps_cgs + energy_shift_cgs)) into U
// and its rho/s derivatives. Header-only, device-ready (see CODE.md
// "Layout"): no STL containers, no exceptions, no allocation, every
// function EEOS_HOST_DEVICE, operating on the POD BsplineView3 views
// bspline_eval.hpp already defines.
//
// Internal spline coordinates (see CODE.md "M2 design notes"): x =
// log10(rho* [g/cc]) evaluated against the kappa-shifted grid origin
// already baked into the view's BsplineView3 (see host/adapter_build.hpp
// step 3: "Store the shifted grid origin x0* = logrho[0] + log10(kappa)"),
// u = log10(T [MeV]), y = Ye, lambda = ln(10). Adapter-facing quantities:
// rho* in g/cc (kappa-rescaled), U = eps/c^2 dimensionless, p returned as
// p/c^2 in g/cc, s in kB/baryon, That = U_s dimensionless, T_F in MeV.

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

  int max_iter; // T-solve iteration cap (default 50, set at build)

  EEOS_HOST_DEVICE EOSPoint evaluate(real rho_star, real s, real ye, real u_guess) const;
  EEOS_HOST_DEVICE SRange srange(real rho_star, real ye) const;
};

EEOS_HOST_DEVICE inline SRange EntropyEOSView::srange(real rho_star, real ye) const {
  const real x = detail::aeval_clamp(std::log10(rho_star), x_lo, x_hi);
  const real y = detail::aeval_clamp(ye, y_lo, y_hi);
  const BsplineEval3 slo = bspline_eval3(sigma, x, u_lo, y);
  const BsplineEval3 shi = bspline_eval3(sigma, x, u_hi, y);
  return SRange{slo.f, shi.f};
}

EEOS_HOST_DEVICE inline EOSPoint EntropyEOSView::evaluate(real rho_star, real s, real ye,
                                                           real u_guess) const {
  unsigned flags = 0u;

  // --- 1-2: clamp rho*, Ye (S3 steps 1-2 / S7) ----------------------------
  real x = std::log10(rho_star);
  if (x < x_lo) {
    x = x_lo;
    flags |= flag_ext_rho_low;
  } else if (x > x_hi) {
    x = x_hi;
    flags |= flag_oob_rho_high;
  }

  real y = ye;
  if (y < y_lo) {
    y = y_lo;
    flags |= flag_clamp_ye;
  } else if (y > y_hi) {
    y = y_hi;
    flags |= flag_clamp_ye;
  }

  // --- 3: T-solve, sigma(x,u,y) = s for u ---------------------------------
  const BsplineEval3 sig_lo = bspline_eval3(sigma, x, u_lo, y);
  const BsplineEval3 sig_hi = bspline_eval3(sigma, x, u_hi, y);

  real u;
  int iters = 0;
  if (s <= sig_lo.f) {
    u = u_lo;
    flags |= flag_ext_s_low;
  } else if (s >= sig_hi.f) {
    u = u_hi;
    flags |= flag_ext_s_high;
  } else {
    real lo = u_lo, hi = u_hi;

    real u0;
    if (!detail::aeval_is_nan(u_guess)) {
      u0 = detail::aeval_clamp(u_guess, u_lo, u_hi);
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
      const BsplineEval3 e = bspline_eval3(sigma, x, uu, y);
      const real g = e.f - s;

      // Maintain the bracket by sign (sigma strictly increasing in u).
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
      // sigma_u > 0 at the iterate; bisect otherwise (S3 point 2's
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

  // --- 4: final spline evaluations at the solved point --------------------
  const BsplineEval3 sig = bspline_eval3(sigma, x, u, y);
  const BsplineEval3 Lv = bspline_eval3(L, x, u, y);

  // --- 5: chain rule -------------------------------------------------------
  const real lambda = detail::kLn10;
  const real rho_eff = std::pow(real(10), x); // rho* at the (possibly clamped) solved point
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
