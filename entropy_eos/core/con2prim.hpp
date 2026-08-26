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
//      1D secant/bisection outer solve of g(s) = f2(s, w*(s)) -- see
//      detail::c2p_inner_solve_w() and con2prim()'s fallback section, plus
//      the bracket-collapse/stagnation/precision-polish safeguards for the
//      double-precision floor the outer bisection runs into for large |s|.
//      failed_no_bracket when the scan below finds no sign change anywhere
//      (S11: the caller's invalid-state policy); failed_max_iter when the
//      outer solve exhausts its budget without meeting both tolerances.
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

  // M3c: number of points in the S9 outer bracket scan (see con2prim()'s
  // fallback section doc comment for the diagnosis this replaces the old
  // endpoints-only check). Clamped to [4, detail::kC2PBracketScanMax] at
  // use, so any positive value is safe to pass; 4 is the minimum that still
  // provides the two extended + two physical safety-net points the scan's
  // design relies on.
  int bracket_scan = 17;
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

// M3c: fixed compile-time cap for the bracket scan's arrays (no allocation,
// per this file's device-ready discipline) -- Con2PrimOptions::bracket_scan
// is clamped into [4, kC2PBracketScanMax] before use.
constexpr int kC2PBracketScanMax = 33;

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
//     range [s_phys_lo, s_phys_hi] and HALF a dense window centered on
//     `s_in` spanning its own local physical srange width at `w_in` --
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

  // Incoming iterate's local physical srange (design comment above): used
  // to build the LOCAL half of the interior scan below.
  const real w_in_c = aeval_clamp(w_in, real(0), w_max);
  const real rho_in = D / std::cosh(w_in_c);
  const SRange sr_in = eos.srange(rho_in, ye);
  const real span_in = sr_in.s_max - sr_in.s_min;

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

  // LOCAL half: dense window centered on s_in, spanning +-0.6 of s_in's own
  // local physical srange width (generous enough to catch a narrow real-
  // table feature near a good s_in without being as wide as the global
  // range), clamped into the extended bracket for safety.
  {
    const real half_width = real(0.6) * span_in;
    real lo = s_in - half_width;
    real hi = s_in + half_width;
    lo = aeval_clamp(lo, s_ext_lo, s_ext_hi);
    hi = aeval_clamp(hi, s_ext_lo, s_ext_hi);
    for (int i = 0; i < n_local; ++i) {
      const real frac = n_local > 1 ? static_cast<real>(i) / static_cast<real>(n_local - 1) : real(0.5);
      s_cand[nc++] = lo + frac * (hi - lo);
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
// their quality, only iteration count does. (M3c tried the design doc S4
// exact-at-B=0 recipe here -- eps0 from the cancellation-free tau identity,
// s0 from a 4-point quarter-of-range U match -- and measured a regression
// on the coldest-row edge case; see the seed's own comment below for the
// full account of why it was reverted.)
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
    // M3c side investigation: raising this 0.999 cap (atanh(0.999) ~ 3.8,
    // well under real-table failure states' w in [3.4, 5.9]) was tried and
    // MEASURED to change nothing on LS220/SRO -- the actual cold-pass
    // bottleneck (con2prim()'s fallback-section doc comment) turned out to
    // be vfrac itself landing far from 1 for strongly magnetized states
    // (B2 comparable to or exceeding tau+D, where |S| is inflated by the
    // (z+B2) inertia factor of S7's S_perp = (z+B2)*v_perp rather than by v
    // itself -- this crude estimate, unlike the exact design doc S4
    // relations, has no B2 term at all), not by this cap. Left unchanged
    // (spec: "w0 ... unchanged") since deviating from it measured no
    // benefit.
    const real vclamped = vfrac < real(0.999) ? vfrac : real(0.999);
    w = detail::aeval_clamp(std::atanh(vclamped), real(0), opts.w_max);
  }

  real s;
  if (!detail::aeval_is_nan(s_guess)) {
    s = s_guess;
  } else {
    // M3c informed cold seed -- TRIED, MEASURED, REVERTED (task's "keep it
    // only if it helps" instruction). Implemented design doc S4's
    // exact-at-B=0 recipe: rho0 = D/cosh(w) (w from the S6 estimate above),
    // eps0 estimated from the same cancellation-free tau identity
    // c2p_eval() uses for f2 (dropping the unknown-before-an-EOS-call
    // pressure term and the bounded (B.v)^2 term), then s0 chosen from 4
    // trial quarter-points of srange(rho0,ye) by matching U to eps0 (no
    // eps->s map to invert directly). Measured effect: tests/test_con2prim.cpp
    // test 6 (w=1e-10, coldest s row -- s within 1e-6 of s_min) regressed
    // from 0 failures to 3 failed_no_bracket plus a tau round-trip tolerance
    // breach. Root cause: the quarter-point grid's closest candidate to a
    // near-s_min truth is still ~12% of the way across a physical srange
    // that can itself span many orders of magnitude (module header /
    // con2prim()'s fallback doc comment), so the seed lands far from s_min
    // in absolute terms; worse, the M3c bracket scan's frac_in tracking
    // (c2p_bracket_scan()) then propagates that same wrong fractional
    // position to every interior scan point, systematically missing the
    // true near-s_min root instead of just converging slowly to it. Kept
    // the original crude seed instead, unchanged from M3a: correctness
    // never depended on this seed's quality (only iteration count did, per
    // this function's own doc comment), and the coldest-row states are
    // exactly the regime prim2con.hpp's cancellation-free tau form exists
    // to handle correctly, so a regression there is not an acceptable trade
    // for faster convergence elsewhere.
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
  // M3c bracket scan (see this function's fallback-section doc comment
  // above for the measured diagnosis and design): `s`/`w` here are the
  // incoming iterate -- Newton's last (possibly failed) position, or the
  // untouched seed if max_iter_newton==0 -- used both to anchor the scan's
  // interior points and to break ties among multiple sign-changing
  // intervals.
  const detail::BracketScanResult scan =
      detail::c2p_bracket_scan(eos, in.D, in.tau, out.ye, in.S_par, in.S_perp, in.B2, opts.w_max,
                                opts.tau_floor_rel, opts.bracket_scan, s, w, w, r.pt.u_solved,
                                opts.max_iter_1d, opts.tol);

  if (!scan.bracketed) {
    int best_it = 0;
    const Residuals best = detail::c2p_inner_solve_w(eos, in.D, in.tau, out.ye, in.S_par, in.S_perp, in.B2,
                                                       scan.s_best, opts.w_max, opts.tau_floor_rel, w,
                                                       r.pt.u_solved, opts.max_iter_1d, opts.tol, best_it);
    (void)best_it;
    out.result = C2PResult::failed_no_bracket;
    detail::c2p_fill(out, best, iters, 0);
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
                                              scan.s_lo, opts.w_max, opts.tau_floor_rel, w, r.pt.u_solved,
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
    // not observed, but guarded defensively): same best-of-two-endpoints
    // report as the ordinary no-bracket path.
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
