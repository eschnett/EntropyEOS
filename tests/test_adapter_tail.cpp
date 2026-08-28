// tests/test_adapter_tail.cpp — unit tests for the M3g/M3i causal extension
// tails (eos-causal-tail.md; entropy_eos/core/adapter_eval.hpp's "CAUSAL
// TAILS" and "M3i X-LOW CAUSAL TAILS" module sections): the log-space u-high
// sigma tail, the causal slope clamp on L's u-high tail, the log-space x-low
// sigma tail with its u-low corner guard, L's plain (no longer
// slope-to-zero) x-low tail, and the two properties the whole design rests
// on -- that the tail operator stays EXACTLY transparent inside the
// physical box, and that both extension bands are causal.
//
// Tests 1-8 are M3g (u-high side), 9-13 are M3i (x-low side).
//
// Split out of tests/test_adapter.cpp (which owns the M2b/M2d-2 adapter
// core) so the tail mathematics has one place to grow; the real-table
// checks are guarded exactly as that file's are.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "entropy_eos/core/adapter_eval.hpp"
#include "entropy_eos/host/adapter_build.hpp"
#include "entropy_eos/host/bspline_fit.hpp"
#include "entropy_eos/host/io_stellarcollapse.hpp"
#include "entropy_eos/host/synthetic.hpp"
#include "entropy_eos/host/table.hpp"
#include "entropy_eos/host/units.hpp"
#include "test_scale.hpp"

using eeos::BsplineEval3;
using eeos::BuildOptions;
using eeos::EntropyEOS;
using eeos::EntropyEOSView;
using eeos::EOSPoint;
using eeos::RawTable;
using eeos::SyntheticOptions;
using eeos::UHighTailInfo;
using eeos::detail::Track1D;

namespace {

constexpr double kLn10 = 2.302585092994045684017991454684364207601101488628772976033;

double rel_err(double got, double ref, double floor = 1e-300) {
  return std::fabs(got - ref) / std::max(std::fabs(ref), floor);
}

bool bit_equal(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

bool sample_bit_equal(const BsplineEval3 &a, const BsplineEval3 &b) {
  return bit_equal(a.f, b.f) && bit_equal(a.fx, b.fx) && bit_equal(a.fu, b.fu) &&
         bit_equal(a.fy, b.fy) && bit_equal(a.fxx, b.fxx) && bit_equal(a.fxu, b.fxu) &&
         bit_equal(a.fuu, b.fuu);
}

EntropyEOS build_synthetic(const SyntheticOptions &opts) {
  RawTable table = eeos::make_synthetic_table(opts);
  return eeos::build_entropy_eos(table);
}

const std::string kLS220Path = "tables/LS220_234r_136t_50y_analmu_20091212_SVNr26.h5";
const std::string kSROPath = "tables/LS220_3335_rho391_temp163_ye66.h5";
const std::string kDD2Path = "tables/Hempel_DD2EOS_rho234_temp180_ye60_version_1.1_20120817.h5";
const std::string kSFHoPath = "tables/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5";

bool table_exists(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

} // namespace

// ==========================================================================
// 1. The log-space sample transforms are exact inverses
// ==========================================================================
//
// aeval_log_sample()/aeval_exp_sample() are the whole mechanism by which the
// M3g tail reuses the pre-existing curvature-ramp machinery: run the tail on
// g = ln(sigma), map back. If they are not inverse, every derivative the
// chain rule consumes is wrong in the tail -- so this checks the round trip
// on a hand-built sample with deliberately unrelated derivative values.

TEST_CASE("M3g log/exp sample transforms round-trip to roundoff") {
  const BsplineEval3 b{7.25, -1.5, 3.75, 0.125, 2.5, -0.875, 11.0};
  const BsplineEval3 g = eeos::detail::aeval_log_sample(b);

  // The transform itself, term by term (the derivations in the module
  // header), so a sign slip cannot hide inside a self-consistent round trip.
  CHECK(rel_err(g.f, std::log(b.f)) <= 1e-15);
  CHECK(rel_err(g.fx, b.fx / b.f) <= 1e-15);
  CHECK(rel_err(g.fu, b.fu / b.f) <= 1e-15);
  CHECK(rel_err(g.fy, b.fy / b.f) <= 1e-15);
  CHECK(rel_err(g.fxx, b.fxx / b.f - (b.fx / b.f) * (b.fx / b.f)) <= 1e-14);
  CHECK(rel_err(g.fxu, b.fxu / b.f - (b.fx / b.f) * (b.fu / b.f)) <= 1e-14);
  CHECK(rel_err(g.fuu, b.fuu / b.f - (b.fu / b.f) * (b.fu / b.f)) <= 1e-14);

  const BsplineEval3 r = eeos::detail::aeval_exp_sample(g);
  CHECK(rel_err(r.f, b.f) <= 1e-15);
  CHECK(rel_err(r.fx, b.fx) <= 1e-14);
  CHECK(rel_err(r.fu, b.fu) <= 1e-14);
  CHECK(rel_err(r.fy, b.fy) <= 1e-14);
  CHECK(rel_err(r.fxx, b.fxx) <= 1e-13);
  CHECK(rel_err(r.fxu, b.fxu) <= 1e-13);
  CHECK(rel_err(r.fuu, b.fuu) <= 1e-13);
}

// ==========================================================================
// 2. Causal slope clamp arithmetic (hand-built tracks, known slopes)
// ==========================================================================

TEST_CASE("M3g causal slope clamp: exact asymptotic slope, C1 at the seam, floor wins") {
  const double w = 0.25;

  SUBCASE("no-op when the raw asymptotic slope already obeys the cap") {
    const Track1D t{3.0, 2.0, 1.0}; // m = 2 + 1*0.25/2 = 2.125
    const Track1D o = eeos::detail::aeval_cap_slope(t, w, 5.0);
    CHECK(bit_equal(o.f0, t.f0));
    CHECK(bit_equal(o.f1, t.f1));
    CHECK(bit_equal(o.f2, t.f2));
  }

  SUBCASE("binding cap sets the asymptotic slope exactly, leaving f0/f1 alone") {
    const Track1D t{3.0, 2.0, 8.0}; // m = 2 + 8*0.125 = 3.0
    const double m_cap = 1.25;
    const Track1D o = eeos::detail::aeval_cap_slope(t, w, m_cap);
    CHECK(bit_equal(o.f0, t.f0));
    CHECK(bit_equal(o.f1, t.f1)); // C1 at the seam is the point (see the module header)
    CHECK(rel_err(eeos::detail::aeval_phase2_slope(o, 1.0, w), m_cap) <= 1e-15);

    // ...and the ramp really does continue with that slope: f(d) is affine
    // beyond the blend cell.
    const Track1D a = eeos::detail::aeval_ramp_track(o, 2.0 * w, w);
    const Track1D b = eeos::detail::aeval_ramp_track(o, 3.0 * w, w);
    CHECK(rel_err(a.f1, m_cap) <= 1e-15);
    CHECK(rel_err(b.f1, m_cap) <= 1e-15);
    CHECK(rel_err((b.f0 - a.f0) / w, m_cap) <= 1e-14);
    CHECK(b.f2 == 0.0);
  }

  SUBCASE("aeval_capped_track: the monotonicity floor wins a conflict") {
    const double m_floor = 0.75;
    const Track1D t{3.0, 2.0, 8.0};
    // A cap far below the floor must NOT drag the asymptotic slope under it.
    const Track1D o = eeos::detail::aeval_capped_track(t, 2.0 * w, w, m_floor, 0.01);
    CHECK(o.f1 >= m_floor);
    // Beyond the blend cell f' IS the asymptotic slope; it must be the floor.
    CHECK(rel_err(o.f1, m_floor) <= 1e-15);
  }

  SUBCASE("aeval_capped_track: cap above the floor binds normally") {
    const double m_floor = 0.75;
    const Track1D t{3.0, 2.0, 8.0};
    const Track1D o = eeos::detail::aeval_capped_track(t, 2.0 * w, w, m_floor, 1.25);
    CHECK(rel_err(o.f1, 1.25) <= 1e-15);
  }
}

// ==========================================================================
// 3. b <-> L-slope conversion (aeval_L_slope_cap)
// ==========================================================================
//
// The clamp is stated on b = dln(eps_hat)/du but applied to L's own slope;
// the conversion must be exactly invertible, or the enforced bound is not
// the one the design specifies.

TEST_CASE("M3g aeval_L_slope_cap inverts b = ln10 * m_L * E / eps_hat") {
  const double inv_c2 = 1.0 / (eeos::c_light_cm_s * eeos::c_light_cm_s);
  const double L_b = 20.5; // log10(eps + shift), cgs
  const double E = std::pow(10.0, L_b) * inv_c2;
  const double shift_hat = 0.3 * E; // a non-negligible energy shift
  const double eps = E - shift_hat;

  const double b_cap = 5.5;
  const double m_cap = eeos::detail::aeval_L_slope_cap(L_b, b_cap, shift_hat, inv_c2);
  CHECK(m_cap > 0.0);
  CHECK(rel_err(kLn10 * m_cap * E / eps, b_cap) <= 1e-14);

  // eps_hat <= 0 disables the cap rather than producing a nonsense bound.
  CHECK(eeos::detail::aeval_L_slope_cap(L_b, b_cap, E, inv_c2) == 0.0);
  CHECK(eeos::detail::aeval_L_slope_cap(L_b, b_cap, 2.0 * E, inv_c2) == 0.0);
}

// ==========================================================================
// 4. In-box bit-identity: the tail operator is exactly transparent
// ==========================================================================
//
// eos-causal-tail.md S3's non-negotiable: "every in-box evaluation (the tail
// operator is bypassed identically)". Checked bit-for-bit -- not to a
// tolerance -- on both fields, all 7 spline outputs, at random interior
// points, INCLUDING the sigma configuration that carries M3g's log-space
// u-high tail. (evaluate() itself is bit-identical for an in-box warm solve
// by the same argument; a cold solve's secant seed legitimately moves,
// because srange_extended().s_max does -- see the srange growth tests at the
// end of this file.)

TEST_CASE("M3g: aeval_extended is bit-identical to a plain bspline_eval3 inside the box") {
  SyntheticOptions opts;
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();

  std::mt19937 rng(0x3603A11u);
  std::uniform_real_distribution<double> xq(view.x_lo, view.x_hi);
  std::uniform_real_distribution<double> uq(view.u_lo, view.u_hi);
  std::uniform_real_distribution<double> yq(view.y_lo, view.y_hi);

  const eeos::detail::ExtSpec sig_spec = view.sigma_ext_spec();
  const eeos::detail::ExtSpec L_spec = view.L_ext_spec(0.0);
  CHECK(sig_spec.u_high_log); // the M3g log tail really is enabled on this path
  CHECK(sig_spec.x_low_log);  // ...and the M3i one
  CHECK(!L_spec.x_low_log);   // (L's x-low tail is the plain generic one)

  const int npts = static_cast<int>(eeos_n(20000, 2000));
  for (int k = 0; k < npts; ++k) {
    const double x = xq(rng), u = uq(rng), y = yq(rng);
    CHECK(sample_bit_equal(eeos::detail::aeval_extended(view.sigma, x, u, y, sig_spec),
                           bspline_eval3(view.sigma, x, u, y)));
    CHECK(sample_bit_equal(eeos::detail::aeval_extended(view.L, x, u, y, L_spec),
                           bspline_eval3(view.L, x, u, y)));
  }

  // The box corners themselves are in-box (the tail starts strictly outside).
  const double corners_x[2] = {view.x_lo, view.x_hi};
  const double corners_u[2] = {view.u_lo, view.u_hi};
  for (double x : corners_x) {
    for (double u : corners_u) {
      const double y = 0.5 * (view.y_lo + view.y_hi);
      CHECK(sample_bit_equal(eeos::detail::aeval_extended(view.sigma, x, u, y, sig_spec),
                             bspline_eval3(view.sigma, x, u, y)));
      CHECK(sample_bit_equal(eeos::detail::aeval_extended(view.L, x, u, y, L_spec),
                             bspline_eval3(view.L, x, u, y)));
    }
  }
}

// ==========================================================================
// 5. The log-sigma tail: C2 at the seam, exponential in phase 2, monotone
// ==========================================================================

TEST_CASE("M3g log-sigma u-high tail: C2 at the seam and exactly exponential beyond the blend cell") {
  SyntheticOptions opts;
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();
  const eeos::detail::ExtSpec spec = view.sigma_ext_spec();
  const double hu = view.sigma.hu;

  std::mt19937 rng(0x109514u);
  std::uniform_real_distribution<double> xq(view.x_lo, view.x_hi);
  std::uniform_real_distribution<double> yq(view.y_lo, view.y_hi);

  for (int k = 0; k < 25; ++k) {
    const double x = xq(rng), y = yq(rng);

    // C2 at the seam: the tail evaluated AT d = 0 must reproduce the
    // boundary sample exactly -- value, both first derivatives, and both
    // second derivatives -- i.e. the log transform is undone to roundoff.
    // (d is passed as 0 rather than a tiny positive offset on purpose: a
    // finite offset would measure sigma's genuine slope, not the tail's
    // exactness.)
    const BsplineEval3 in = bspline_eval3(view.sigma, x, view.u_hi, y);
    const double m_floor_g = view.ext_slope_floor_sigma / in.f;
    const BsplineEval3 out = eeos::detail::aeval_exp_sample(eeos::detail::aeval_apply_tail(
        eeos::detail::aeval_log_sample(in), 0.0, eeos::detail::TailAxis::u,
        eeos::detail::TailSpec{hu, m_floor_g, 0.0}));

    CHECK(rel_err(out.f, in.f) <= 1e-15);
    CHECK(rel_err(out.fu, in.fu) <= 1e-14);
    CHECK(rel_err(out.fx, in.fx) <= 1e-14);
    // The second derivatives come back through sigma*(g'' + g'^2), and
    // g'' = sigma_uu/sigma - g'^2 subtracts exactly what is added back. On a
    // table where sigma is near-linear in u (the synthetic ideal gas is
    // exactly linear, so sigma_uu ~ 0) that is a catastrophic cancellation
    // whose ABSOLUTE error scale is ulp(sigma_u^2/sigma), not ulp(sigma_uu).
    // A mixed abs/rel bound is therefore the honest check -- and the scale it
    // uses is the physically meaningful one (cf. tests/test_bspline.cpp's
    // 3D separable exactness check, which needs the same treatment).
    const double uu_scale = std::fabs(in.fuu) + std::fabs(in.fu * in.fu / in.f);
    const double xu_scale = std::fabs(in.fxu) + std::fabs(in.fx * in.fu / in.f);
    CHECK(std::fabs(out.fuu - in.fuu) <= 1e-13 * uu_scale);
    CHECK(std::fabs(out.fxu - in.fxu) <= 1e-13 * xu_scale);

    // Phase 2 (d > hu): ln(sigma) is affine in u with slope alpha, and
    // alpha is exactly what aeval_sigma_u_high_alpha() reports (which is
    // what L's causal clamp is measured against).
    const double alpha =
        eeos::detail::aeval_sigma_u_high_alpha(view.sigma, x, view.u_hi, y, view.ext_slope_floor_sigma);
    CHECK(alpha > 0.0);
    const double u1 = view.u_hi + 2.0 * hu;
    const double u2 = view.u_hi + 5.0 * hu;
    const BsplineEval3 s1 = eeos::detail::aeval_extended(view.sigma, x, u1, y, spec);
    const BsplineEval3 s2 = eeos::detail::aeval_extended(view.sigma, x, u2, y, spec);
    CHECK(s1.f > 0.0);
    CHECK(rel_err((std::log(s2.f) - std::log(s1.f)) / (u2 - u1), alpha) <= 1e-12);
    // sigma_u = sigma * alpha and sigma_uu = sigma * alpha^2 in phase 2.
    CHECK(rel_err(s1.fu, s1.f * alpha) <= 1e-12);
    CHECK(rel_err(s1.fuu, s1.f * alpha * alpha) <= 1e-11);
  }

  // Strict monotonicity with the slope floor honored across the whole band.
  const int nsteps = static_cast<int>(eeos_n(400, 80));
  for (int line = 0; line < 10; ++line) {
    const double x = xq(rng), y = yq(rng);
    double prev = eeos::detail::aeval_extended(view.sigma, x, view.u_hi, y, spec).f;
    for (int i = 1; i <= nsteps; ++i) {
      const double u = view.u_hi + (view.u_ext_hi - view.u_hi) * static_cast<double>(i) /
                                        static_cast<double>(nsteps);
      const BsplineEval3 s = eeos::detail::aeval_extended(view.sigma, x, u, y, spec);
      CHECK(s.f > prev);
      CHECK(s.fu >= view.ext_slope_floor_sigma);
      prev = s.f;
    }
  }
}

// ==========================================================================
// 6. The u-high extension band is causal (synthetic gas)
// ==========================================================================
//
// The miniature of check_adapter()'s class E acceptance: every point of the
// u-high band must satisfy c_s^2 < 1 (and p > 0, U_s > 0). On the synthetic
// IDEAL gas the causal clamp binds at every seam point -- entropy is linear,
// not exponential, in u there, so b/alpha - 1 would be O(10) -- which makes
// this table the one that actually exercises the clamp; the real tables
// (test 7) are radiation-like at their hot seam and never trip it.

TEST_CASE("M3g: the whole u-high extension band is causal on the synthetic gas") {
  SyntheticOptions opts;
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();

  size_t n = 0, n_clamped = 0, n_floor_wins = 0, n_cs2_nonpos = 0;
  double cs2_max = -1e300, cs2_min = 1e300;

  const int na = static_cast<int>(eeos_n(60, 20));
  const int nb = static_cast<int>(eeos_n(12, 6));
  const int nu = static_cast<int>(eeos_n(48, 16));
  for (int ia = 0; ia < na; ++ia) {
    const double x = view.x_lo + (view.x_hi - view.x_lo) * (ia + 0.5) / na;
    for (int ib = 0; ib < nb; ++ib) {
      const double y = view.y_lo + (view.y_hi - view.y_lo) * (ib + 0.5) / nb;

      const UHighTailInfo info = view.u_high_tail_info(std::pow(10.0, x), y);
      CHECK(info.alpha > 0.0);
      if (info.clamped) ++n_clamped;
      if (info.floor_wins) ++n_floor_wins;

      for (int j = 1; j <= nu; ++j) {
        const double u = view.u_hi + (view.u_ext_hi - view.u_hi) * j / nu;
        const EOSPoint pt = view.eval_at(x, u, y, 0u, 0);
        REQUIRE(std::isfinite(pt.cs2));
        CHECK(pt.cs2 < 1.0);
        CHECK(pt.p > 0.0);
        CHECK(pt.U_s > 0.0);
        cs2_max = std::max(cs2_max, pt.cs2);
        cs2_min = std::min(cs2_min, pt.cs2);
        if (pt.cs2 <= 0.0) ++n_cs2_nonpos;
        ++n;
      }
    }
  }
  // c_s^2 <= 0 is REPORTED, not asserted: physicality in the extension is not
  // guaranteed by design (eos-adapter-F-to-U.md S7 -- the tails are an escape
  // hatch for the solver, not a claim of physical validity far from the
  // table), and the only property the con2prim S9 inner-solve proof needs is
  // c_s^2 < 1, which IS asserted above. On this table the clamp's strongly
  // negative L_uu inside the single blend cell does drive c_s^2 slightly
  // negative in places; on the real tables the clamp never fires and this
  // count is 0. Same convention as tests/test_adapter_audit.cpp test 6.
  MESSAGE("M3g synthetic u-high band: n=" << n << " cs2 in [" << cs2_min << ", " << cs2_max
                                           << "] cs2<=0: " << n_cs2_nonpos << " clamped seam points="
                                           << n_clamped << " floor_wins=" << n_floor_wins);
  CHECK(n_clamped > 0);      // the ideal gas is exactly the case that needs the clamp
  CHECK(n_floor_wins == 0);  // ...and the monotonicity floor never has to override it
}

// ==========================================================================
// 7. Real tables (guarded): the far u-high tail is the radiation asymptote
// ==========================================================================
//
// eos-causal-tail.md S3's "why it works": at a radiation-dominated hot seam
// the fitted slopes are b = 4*ln10 and alpha = 3*ln10 per decade of T, so
// the far tail's fixed-s slope is constant and c_s^2 -> b/alpha - 1 = 1/3 --
// not merely causal but the physically correct hot-gas asymptote. The clamp
// (cs2_ext_cap = 0.99) is far above that and must stay dormant.

namespace {

// `strict_radiation_window` selects which of the two claim tiers is asserted
// (see the block comment right above the two windowed CHECKs at the bottom of
// this function).
void check_real_table_far_tail(const std::string &path, const char *label, double m_B_g,
                               bool strict_radiation_window = true) {
  const std::string use_path = eeos_san_table(path);
  if (!table_exists(use_path)) {
    MESSAGE("skipping " << label << " far-tail test: " << use_path << " not present");
    return;
  }
  RawTable table = eeos::read_stellarcollapse(use_path);
  BuildOptions bopts;
  bopts.m_B_table_g = m_B_g;
  EntropyEOS adapter = eeos::build_entropy_eos(table, bopts);
  const EntropyEOSView view = adapter.view();

  CHECK(view.cs2_ext_cap == doctest::Approx(0.99));

  size_t n = 0, n_clamped = 0, n_floor_wins = 0;
  double cs2_min = 1e300, cs2_max = -1e300;

  const int na = static_cast<int>(eeos_n(24, 8));
  const int nb = static_cast<int>(eeos_n(8, 4));
  for (int ia = 0; ia < na; ++ia) {
    // Stay off the top rho decade: both real tables keep a small residual
    // acausal pocket at the joint rho_max/T_max corner (CODE.md's M3f
    // findings), which the tail faithfully inherits from the data -- that is
    // measured by check_adapter()'s class E, not asserted here.
    const double x = view.x_lo + (view.x_hi - 1.0 - view.x_lo) * (ia + 0.5) / na;
    for (int ib = 0; ib < nb; ++ib) {
      const double y = view.y_lo + (view.y_hi - view.y_lo) * (ib + 0.5) / nb;
      const UHighTailInfo info = view.u_high_tail_info(std::pow(10.0, x), y);
      CHECK(info.alpha > 0.0);
      if (info.clamped) ++n_clamped;
      if (info.floor_wins) ++n_floor_wins;

      const EOSPoint pt = view.eval_at(x, view.u_ext_hi, y, 0u, 0);
      REQUIRE(std::isfinite(pt.cs2));
      cs2_min = std::min(cs2_min, pt.cs2);
      cs2_max = std::max(cs2_max, pt.cs2);
      ++n;
    }
  }
  MESSAGE(std::string(label) << " far u-high tail (u = u_ext_hi): n=" << n << " cs2 in [" << cs2_min << ", "
                << cs2_max << "] clamped=" << n_clamped << " floor_wins=" << n_floor_wins);

  // Tier 1 -- construction invariants, asserted for EVERY table: the tail is
  // finite (REQUIRE above), sigma's log tail is live (alpha > 0, above), and
  // the causal clamp/monotonicity floor stay dormant because b/alpha - 1 is
  // nowhere near cs2_ext_cap = 0.99. These are properties of the M3g
  // construction itself and hold table-independently.
  CHECK(n_clamped == 0);
  CHECK(n_floor_wins == 0);

  // Tier 2 -- the radiation-asymptote WINDOW, with generous room for the
  // table's own slope scatter (the pre-M3g linear-sigma tail saturated at
  // c_s^2 ~ 3.8 here, so this is a 10x-margin discriminator, not a tight
  // fit). Unlike tier 1 this is a claim about the DATA at the hot seam, not
  // about the construction: it presupposes the seam is radiation-dominated,
  // which is what makes b = 4*ln10, alpha = 3*ln10 and hence c_s^2 -> 1/3.
  //
  // LS220 (cs2 in [0.339, 0.637]) and SRO ([0.340, 0.622]) satisfy it. DD2
  // and SFHo, measured 2026-08-28, do NOT, at two identified rho columns each
  // -- and in both cases the far tail is the messenger, not the cause:
  //
  //  (a) SFHo rho = 1.2e7 g/cc (and DD2 rho = 6.7e6): c_s^2 is ALREADY
  //      -2.98 (SFHo) / +0.894 (DD2) at the seam and inside the box, at every
  //      Ye alike, in the hot radiation-dominated low-density corner both
  //      Hempel tabulations share (the same corner check_table flags as
  //      delta_p ~ 20 and check_adapter as p_nonpositive/cs2_nonpositive near
  //      rho ~ 1.4e7, T ~ 157 MeV). The tail reproduces the seam value it is
  //      built from, as designed.
  //  (b) DD2 rho = 8.9e14 g/cc: the seam is fine (c_s^2 = 0.518) but the
  //      seam is NOT radiation-dominated there -- alpha = 4.56 against the
  //      radiation 3*ln10 = 6.91, because these tables stop at T_max = 158
  //      MeV, which at supra-nuclear density is still matter-dominated. The
  //      tail's own asymptote (b/alpha - 1 = 0.394) is causal and sane, but
  //      the approach to it dips: c_s^2 falls 0.518 -> 0.379 -> 0.133 ->
  //      -0.033 across the band, and p turns negative just past u_ext_hi.
  //      This is a genuine narrowing of the M3g design envelope on a
  //      non-radiation hot seam, reported by check_adapter()'s class E
  //      (ext_u_high_cs2_nonpositive 22848, ext_u_high_p_nonpositive 13804 on
  //      DD2 against LS220's 6883/226) -- accept-and-guard for now, tracked in
  //      CODE.md's M3g/M3i findings, not silently tolerated here.
  //
  // So the window is asserted where it is a valid claim and MEASURED (tier 1
  // still fully asserted) where the table's hot seam does not meet its
  // premise. Do not switch a table to non-strict to make a red run green:
  // the premise, not the bound, is what has to fail first.
  if (strict_radiation_window) {
    CHECK(cs2_min > 0.15);
    CHECK(cs2_max < 0.75);
  } else {
    MESSAGE(std::string(label) << " far u-high tail: radiation-window check MEASURED, not asserted"
                  << " (non-radiation hot seam; see this function's tier-2 comment) -- cs2_min="
                  << cs2_min << " cs2_max=" << cs2_max);
  }
}

} // namespace

TEST_CASE("M3g far u-high tail: LS220 real table (guarded)") {
  check_real_table_far_tail(kLS220Path, "LS220", eeos::m_amu_g);
}

TEST_CASE("M3g far u-high tail: SRO real table (guarded)") {
  if (eeos_skip_big_table("test_adapter_tail (SRO): real table")) return;
  check_real_table_far_tail(kSROPath, "SRO", eeos::m_neutron_g);
}

// DD2/SFHo take m_amu_g, not m_neutron_g: their convention was measured the
// same way SRO's was and came out amu (tests/integration.sh Part B).
TEST_CASE("M3g far u-high tail: DD2 real table (guarded)") {
  if (eeos_skip_big_table("test_adapter_tail (DD2): real table")) return;
  check_real_table_far_tail(kDD2Path, "DD2", eeos::m_amu_g, /*strict_radiation_window=*/false);
}

TEST_CASE("M3g far u-high tail: SFHo real table (guarded)") {
  if (eeos_skip_big_table("test_adapter_tail (SFHo): real table")) return;
  check_real_table_far_tail(kSFHoPath, "SFHo", eeos::m_amu_g, /*strict_radiation_window=*/false);
}

// ==========================================================================
// 8. srange_extended().s_max legitimately grows (documented consequence)
// ==========================================================================
//
// eos-causal-tail.md S4: on a real (radiation-like) table the 8-cell log
// tail reaches ~10^0.8 ~ 6x s_max where the linear tail reached ~1.8x. The
// growth factor is exp(alpha * (u_ext_hi - u_hi)) with alpha = dln(sigma)/du,
// so it is entirely table-dependent: on the synthetic IDEAL gas, whose
// entropy is linear (not exponential) in u, alpha is small and the log tail
// is barely distinguishable from the old linear one (~1.07x). Nothing
// downstream is sized to that number (the bracket scan is log-spaced and the
// T-solve is a safeguarded Newton on a still strictly monotone sigma), but it
// IS a visible change, so it is pinned here deliberately rather than
// discovered by surprise.

namespace {

void check_srange_growth(const EntropyEOSView &view, const char *label, double min_ratio,
                          double max_ratio) {
  std::mt19937 rng(0x5A2Eu);
  std::uniform_real_distribution<double> xq(view.x_lo, view.x_hi);
  std::uniform_real_distribution<double> yq(view.y_lo, view.y_hi);

  double ratio_min = 1e300, ratio_max = -1e300;
  for (int k = 0; k < 20; ++k) {
    const double rho = std::pow(10.0, xq(rng));
    const double ye = yq(rng);
    const eeos::SRange phys = view.srange(rho, ye);
    const eeos::SRange ext = view.srange_extended(rho, ye);

    CHECK(ext.s_max > phys.s_max);
    CHECK(ext.s_min < phys.s_min);
    const double ratio = ext.s_max / phys.s_max;
    ratio_min = std::min(ratio_min, ratio);
    ratio_max = std::max(ratio_max, ratio);

    // ...and the T-solve still resolves a point deep in the extension.
    const double u_far = view.u_hi + 0.75 * (view.u_ext_hi - view.u_hi);
    const double s_far = view.sigma_extended(rho, u_far, ye);
    const EOSPoint pt = view.evaluate(rho, s_far, ye, std::numeric_limits<double>::quiet_NaN());
    CHECK(rel_err(pt.u_solved, u_far) <= 1e-10);
    CHECK((pt.flags & eeos::flag_maxiter) == 0);
    CHECK((pt.flags & eeos::flag_ext_s_high) != 0);
    CHECK(pt.U_s > 0.0);
  }
  MESSAGE(std::string(label) << " srange_extended().s_max / srange().s_max in [" << ratio_min << ", " << ratio_max
                << "]");
  CHECK(ratio_min > min_ratio);
  CHECK(ratio_max < max_ratio);
}

} // namespace

TEST_CASE("M3g: srange_extended().s_max grows with the log tail, and stays solvable") {
  SyntheticOptions opts;
  EntropyEOS adapter = build_synthetic(opts);
  // Synthetic ideal gas: sigma is LINEAR in u, so the log tail's growth rate
  // alpha = sigma_u/sigma is small and the extended s_max barely moves.
  check_srange_growth(adapter.view(), "synthetic", 1.0, 1.5);
}

TEST_CASE("M3g: srange_extended().s_max on the LS220 real table (guarded)") {
  const std::string path = eeos_san_table(kLS220Path);
  if (!table_exists(path)) {
    MESSAGE("skipping LS220 srange growth test: " << path << " not present");
    return;
  }
  RawTable table = eeos::read_stellarcollapse(path);
  EntropyEOS adapter = eeos::build_entropy_eos(table);
  // Radiation-like hot seam: alpha ~ 3*ln10 per decade of T and the 8-cell
  // extension spans ~0.26 dex, so s_max grows by ~10^0.78 ~ 6x -- the
  // eos-causal-tail.md S4 number. Measured 3.27-6.01x here, against
  // 2.26-2.99x for the pre-M3g linear tail on the same 20 sampled points.
  check_srange_growth(adapter.view(), "LS220", 3.0, 12.0);
}

// ==========================================================================
// 9. M3i: the log-space x-low sigma tail (transforms, phase 2, L's tail)
// ==========================================================================
//
// The mirror of test 5, one axis over: the log transforms must be undone
// exactly at the seam (d = 0), and beyond the blend cell ln(sigma) must be
// affine in x -- i.e. sigma is a power law in rho, which is the whole point
// (radiation has sigma ~ 1/rho). L is checked in the same loop for the
// complementary M3i change: its x-low tail is now the PLAIN generic tail, so
// L itself continues linearly (eps a power law in rho, slope from the seam)
// instead of being frozen at the seam value by the old slope-to-zero
// override.

TEST_CASE("M3i log-sigma x-low tail: exact at the seam, exactly exponential beyond the blend cell") {
  SyntheticOptions opts;
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();
  const eeos::detail::ExtSpec sig_spec = view.sigma_ext_spec();
  const eeos::detail::ExtSpec L_spec = view.L_ext_spec(0.0);
  const double hx = view.sigma.hx;

  std::mt19937 rng(0x3110Bu);
  std::uniform_real_distribution<double> uq(view.u_lo, view.u_hi);
  std::uniform_real_distribution<double> yq(view.y_lo, view.y_hi);

  for (int k = 0; k < 25; ++k) {
    const double u = uq(rng), y = yq(rng);

    const BsplineEval3 in = bspline_eval3(view.sigma, view.x_lo, u, y);
    REQUIRE(in.f > 0.0);

    // d = 0: value, both first derivatives and both second derivatives come
    // back to roundoff (the mixed abs/rel bound on the second derivatives is
    // the same catastrophic-cancellation argument as test 5's).
    const BsplineEval3 out = eeos::detail::aeval_exp_sample(
        eeos::detail::aeval_apply_tail(eeos::detail::aeval_log_sample(in), 0.0,
                                       eeos::detail::TailAxis::x, eeos::detail::TailSpec{hx, 0.0, 0.0}));
    CHECK(rel_err(out.f, in.f) <= 1e-15);
    CHECK(rel_err(out.fx, in.fx) <= 1e-14);
    CHECK(rel_err(out.fu, in.fu) <= 1e-14);
    const double xx_scale = std::fabs(in.fxx) + std::fabs(in.fx * in.fx / in.f);
    const double xu_scale = std::fabs(in.fxu) + std::fabs(in.fx * in.fu / in.f);
    CHECK(std::fabs(out.fxx - in.fxx) <= 1e-13 * xx_scale);
    CHECK(std::fabs(out.fxu - in.fxu) <= 1e-13 * xu_scale);

    // Phase 2 (d < -hx): ln(sigma) is affine in x with the log track's own
    // phase-2 slope m_g, so sigma_x = sigma*m_g and sigma_xx = sigma*m_g^2.
    const BsplineEval3 g = eeos::detail::aeval_log_sample(in);
    const double m_g = eeos::detail::aeval_phase2_slope(Track1D{g.f, g.fx, g.fxx}, -1.0, hx);
    const double x1 = view.x_lo - 2.0 * hx;
    const double x2 = view.x_lo - 5.0 * hx;
    const BsplineEval3 s1 = eeos::detail::aeval_extended(view.sigma, x1, u, y, sig_spec);
    const BsplineEval3 s2 = eeos::detail::aeval_extended(view.sigma, x2, u, y, sig_spec);
    CHECK(s1.f > 0.0);
    CHECK(s2.f > 0.0);
    CHECK(rel_err((std::log(s2.f) - std::log(s1.f)) / (x2 - x1), m_g) <= 1e-11);
    CHECK(rel_err(s1.fx, s1.f * m_g) <= 1e-12);
    CHECK(rel_err(s1.fxx, s1.f * m_g * m_g) <= 1e-11);

    // L: the plain generic tail, i.e. L itself affine in x beyond the blend
    // cell with the seam-derived phase-2 slope. (Before M3i this slope was
    // overridden to exactly 0 -- "eps becomes rho-independent" -- everywhere,
    // including at radiation seams where it is -1; see the module header.)
    const BsplineEval3 Lin = bspline_eval3(view.L, view.x_lo, u, y);
    const double m_L = eeos::detail::aeval_phase2_slope(Track1D{Lin.f, Lin.fx, Lin.fxx}, -1.0, view.L.hx);
    const BsplineEval3 L1 = eeos::detail::aeval_extended(view.L, x1, u, y, L_spec);
    const BsplineEval3 L2 = eeos::detail::aeval_extended(view.L, x2, u, y, L_spec);
    CHECK(std::fabs((L2.f - L1.f) / (x2 - x1) - m_L) <= 1e-11 * (std::fabs(m_L) + 1.0));
    CHECK(std::fabs(L1.fx - m_L) <= 1e-12 * (std::fabs(m_L) + 1.0));
    CHECK(L1.fxx == 0.0); // phase 2 of the generic tail is exactly linear
  }
}

// ==========================================================================
// 10. M3i: the u-LOW x-low corner guard
// ==========================================================================
//
// sigma at the x_lo seam is NOT positive by construction (unlike at the u_hi
// seam): the u-LOW tail runs sigma linearly toward -infinity by design, so a
// deep-cold column can hand the x-tail a sigma_b <= 0 (ln undefined) or a
// sigma_b so small that g_x = sigma_x/sigma_b explodes and e^{g_x d}
// overflows across the band. aeval_xlow_log_ok() is the guard; when it says
// no, aeval_extended() must fall back to the pre-M3i linear x-tail *exactly*.
//
// The guard never fires on the tables we have (measured min sigma at the
// x_lo seam over the whole extended u range: 1.64 LS220, 0.80 SRO, 37.2
// synthetic; largest band excursion 1.03 against the bound of 40), so both
// of its branches are provoked deliberately here.

TEST_CASE("M3i x-low log-tail guard: non-positive or exploding seam samples fall back to the linear tail") {
  const double w = 0.05;
  const double depth = 8.0 * w;

  SUBCASE("aeval_xlow_log_ok bounds the band's total log excursion") {
    // `g` is a LOG-space sample: g.fx = dln(sigma)/dx, g.fxx its curvature.
    BsplineEval3 g{0.0, -kLn10, 0.0, 0.0, 0.0, 0.0, 0.0};
    CHECK(eeos::detail::aeval_xlow_log_ok(g, w, depth)); // radiation: excursion 0.92

    g.fx = -200.0; // seam slope alone: excursion 80 > 40
    CHECK(!eeos::detail::aeval_xlow_log_ok(g, w, depth));

    // The PHASE-2 slope m = f1 - f2*w/2 is bounded too, not just the seam
    // slope: a zero seam slope with a huge curvature is just as unbounded.
    g.fx = 0.0;
    g.fxx = 2.0 * 200.0 / w; // m = -200
    CHECK(!eeos::detail::aeval_xlow_log_ok(g, w, depth));

    // ...and conversely a huge seam slope that the curvature cancels back to
    // m = 0 is still rejected, because phase 1 rides the seam slope.
    g.fx = -200.0;
    g.fxx = -2.0 * 200.0 / w; // m = -200 + 200 = 0
    CHECK(!eeos::detail::aeval_xlow_log_ok(g, w, depth));

    // A NaN anywhere in the track fails the comparison and so falls back.
    g.fx = std::numeric_limits<double>::quiet_NaN();
    g.fxx = 0.0;
    CHECK(!eeos::detail::aeval_xlow_log_ok(g, w, depth));

    // The bound is on the BAND, not on one cell: the same slope is fine over
    // a shallow band and rejected over a deep one, which is what keeps the
    // guard grid-resolution independent.
    g.fx = -100.0;
    g.fxx = 0.0;
    CHECK(eeos::detail::aeval_xlow_log_ok(g, w, 0.2));
    CHECK(!eeos::detail::aeval_xlow_log_ok(g, w, 0.6));
  }

  SUBCASE("an exploding log slope reproduces the pre-M3i linear tail bit-for-bit") {
    SyntheticOptions opts;
    EntropyEOS adapter = build_synthetic(opts);
    const EntropyEOSView view = adapter.view();

    eeos::detail::ExtSpec ok = view.sigma_ext_spec();
    eeos::detail::ExtSpec linear = ok;
    linear.x_low_log = false; // the pre-M3i construction
    eeos::detail::ExtSpec tripped = ok;
    // Same field, same seam, absurdly deep band: the excursion bound is
    // exceeded, so the guard must send this back to `linear`.
    tripped.x_ext_lo = ok.x_lo - 1.0e4;

    const double u = 0.5 * (view.u_lo + view.u_hi);
    const double y = 0.5 * (view.y_lo + view.y_hi);
    for (int i = 1; i <= 8; ++i) {
      const double x = view.x_lo - static_cast<double>(i) * view.sigma.hx * 0.9;
      const BsplineEval3 a = eeos::detail::aeval_extended(view.sigma, x, u, y, tripped);
      const BsplineEval3 b = eeos::detail::aeval_extended(view.sigma, x, u, y, linear);
      CHECK(sample_bit_equal(a, b));
      // ...and the guard is not vacuous: with the real band depth the log
      // tail IS taken, and differs from the linear one.
      const BsplineEval3 c = eeos::detail::aeval_extended(view.sigma, x, u, y, ok);
      CHECK(!bit_equal(c.f, b.f));
    }
  }

  SUBCASE("a non-positive seam sigma reproduces the pre-M3i linear tail bit-for-bit") {
    // A deliberately NEGATIVE fitted field, standing in for what sigma's
    // u-low tail can produce at a deep-cold column: ln is undefined, so the
    // log branch must be skipped before it is ever attempted.
    const int nx = 6, nu = 5, ny = 4;
    std::vector<double> data(static_cast<size_t>(nx * nu * ny));
    for (int ky = 0; ky < ny; ++ky)
      for (int ju = 0; ju < nu; ++ju)
        for (int ix = 0; ix < nx; ++ix)
          data[static_cast<size_t>((ky * nu + ju) * nx + ix)] =
              -(1.0 + 0.30 * ix + 0.17 * ju + 0.09 * ky);
    const eeos::Bspline3 fit = eeos::fit_bspline_3d(nx, nu, ny, 1.0, 0.25, -1.0, 0.2, 0.1, 0.05, data);
    const eeos::BsplineView3 field = fit.view();

    eeos::detail::ExtSpec logspec{/*x_lo=*/1.0,
                                  /*x_hi=*/1.0 + 5 * 0.25,
                                  /*u_lo=*/-1.0,
                                  /*u_hi=*/-1.0 + 4 * 0.2,
                                  /*x_ext_lo=*/1.0 - 8 * 0.25,
                                  /*u_m_floor=*/0.0,
                                  /*x_low_log=*/true,
                                  /*u_high_log=*/false,
                                  /*u_high_b_cap=*/0.0,
                                  /*shift_hat=*/0.0,
                                  /*inv_c2=*/1.0};
    eeos::detail::ExtSpec linear = logspec;
    linear.x_low_log = false;

    // The seam sample really is negative -- that is what the guard sees.
    REQUIRE(bspline_eval3(field, 1.0, -0.6, 0.2).f < 0.0);
    for (int i = 1; i <= 8; ++i) {
      const double x = 1.0 - static_cast<double>(i) * 0.25 * 0.9;
      const BsplineEval3 a = eeos::detail::aeval_extended(field, x, -0.6, 0.2, logspec);
      const BsplineEval3 b = eeos::detail::aeval_extended(field, x, -0.6, 0.2, linear);
      CHECK(sample_bit_equal(a, b));
    }
  }
}

// ==========================================================================
// 11. M3i: the x-low x u-low corner, end to end through evaluate()
// ==========================================================================
//
// The corner the guard exists for, exercised through the public API: rho
// below the box AND s below the (extended) entropy range at the same time,
// which is where the u-low tail's linear descent and the x-low log tail
// meet. Everything must stay finite, correctly flagged and monotone
// (U_s > 0), and an s taken FROM a band point must solve back to it.

namespace {

void check_finite_eospoint(const EOSPoint &pt) {
  REQUIRE(std::isfinite(pt.U));
  REQUIRE(std::isfinite(pt.U_rho));
  REQUIRE(std::isfinite(pt.U_s));
  REQUIRE(std::isfinite(pt.U_rhorho));
  REQUIRE(std::isfinite(pt.U_rhos));
  REQUIRE(std::isfinite(pt.That));
  REQUIRE(std::isfinite(pt.p));
  REQUIRE(std::isfinite(pt.h));
  REQUIRE(std::isfinite(pt.cs2));
  REQUIRE(std::isfinite(pt.T_F_MeV));
  REQUIRE(std::isfinite(pt.mu_tilde));
  REQUIRE(std::isfinite(pt.u_solved));
}

void check_xlow_corner(const EntropyEOSView &view, const char *label) {
  CAPTURE(label);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const int nd = 8, ny = 5;
  for (int i = 1; i <= nd; ++i) {
    const double x = view.x_lo - (view.x_lo - view.x_ext_lo) * i / nd;
    const double rho = std::pow(10.0, x);
    for (int k = 0; k < ny; ++k) {
      const double y = view.y_lo + (view.y_hi - view.y_lo) * (k + 0.5) / ny;

      // (a) s taken from a point in the u-LOW band at this band density:
      // the T-solve must land back on it, through both tails at once.
      const double u_low = view.u_lo - 0.5 * (view.u_lo - view.u_ext_lo);
      const double s_low = view.sigma_extended(rho, u_low, y);
      const EOSPoint back = view.evaluate(rho, s_low, y, nan);
      check_finite_eospoint(back);
      CHECK(rel_err(back.u_solved, u_low) <= 1e-9);
      CHECK((back.flags & eeos::flag_ext_rho_low) != 0);
      CHECK((back.flags & eeos::flag_ext_s_low) != 0);
      CHECK((back.flags & eeos::flag_maxiter) == 0);
      CHECK(back.U_s > 0.0);

      // (b) s far BELOW even the extended bracket: the documented hard clamp
      // at u_ext_lo, still finite and still flagged.
      const eeos::SRange ext = view.srange_extended(rho, y);
      const EOSPoint under = view.evaluate(rho, ext.s_min - 1.0e3, y, nan);
      check_finite_eospoint(under);
      CHECK(under.u_solved == view.u_ext_lo);
      CHECK((under.flags & eeos::flag_ext_rho_low) != 0);
      CHECK((under.flags & eeos::flag_ext_s_low) != 0);

      // (c) the opposite corner of the same band: s above the physical range
      // (the u-HIGH log tail composed with the x-low log tail).
      const double u_high = view.u_hi + 0.5 * (view.u_ext_hi - view.u_hi);
      const double s_high = view.sigma_extended(rho, u_high, y);
      const EOSPoint hot = view.evaluate(rho, s_high, y, nan);
      check_finite_eospoint(hot);
      CHECK(rel_err(hot.u_solved, u_high) <= 1e-9);
      CHECK((hot.flags & eeos::flag_ext_rho_low) != 0);
      CHECK((hot.flags & eeos::flag_ext_s_high) != 0);
      CHECK(hot.U_s > 0.0);
    }
  }
}

} // namespace

TEST_CASE("M3i: the x-low x u-low corner stays finite, flagged and solvable (synthetic)") {
  SyntheticOptions opts;
  EntropyEOS adapter = build_synthetic(opts);
  check_xlow_corner(adapter.view(), "synthetic");
}

TEST_CASE("M3i: the x-low x u-low corner on the LS220 real table (guarded)") {
  const std::string path = eeos_san_table(kLS220Path);
  if (!table_exists(path)) {
    MESSAGE("skipping LS220 x-low corner test: " << path << " not present");
    return;
  }
  RawTable table = eeos::read_stellarcollapse(path);
  EntropyEOS adapter = eeos::build_entropy_eos(table);
  check_xlow_corner(adapter.view(), "LS220");
}

// ==========================================================================
// 12. M3i: the whole x-low extension band is causal (synthetic gas)
// ==========================================================================
//
// The miniature of check_adapter()'s class E acceptance for the band M3i
// repairs -- the x-low counterpart of test 6. Causality is asserted; the
// asymptotic VALUE is only reported, because it is table-dependent: the
// synthetic table is an ideal gas whose entropy is linear (not exponential)
// in u and whose eps is rho-independent at fixed T for its matter part, so
// its x-low band has no single clean asymptote to pin the way a
// radiation-dominated real seam does (test 13 pins that one).

TEST_CASE("M3i: the whole x-low extension band is causal on the synthetic gas") {
  SyntheticOptions opts;
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();

  size_t n = 0, n_cs2_nonpos = 0;
  double cs2_max = -1e300, cs2_min = 1e300;

  const int na = static_cast<int>(eeos_n(48, 16)); // u samples
  const int nb = static_cast<int>(eeos_n(12, 6));  // Ye samples
  const int nd = static_cast<int>(eeos_n(48, 16)); // depth samples
  for (int ia = 0; ia < na; ++ia) {
    const double u = view.u_lo + (view.u_hi - view.u_lo) * (ia + 0.5) / na;
    for (int ib = 0; ib < nb; ++ib) {
      const double y = view.y_lo + (view.y_hi - view.y_lo) * (ib + 0.5) / nb;
      for (int j = 1; j <= nd; ++j) {
        const double x = view.x_lo - (view.x_lo - view.x_ext_lo) * j / nd;
        const EOSPoint pt = view.eval_at(x, u, y, 0u, 0);
        REQUIRE(std::isfinite(pt.cs2));
        CHECK(pt.cs2 < 1.0);
        CHECK(pt.p > 0.0);
        CHECK(pt.U_s > 0.0);
        cs2_max = std::max(cs2_max, pt.cs2);
        cs2_min = std::min(cs2_min, pt.cs2);
        if (pt.cs2 <= 0.0) ++n_cs2_nonpos;
        ++n;
      }
    }
  }
  // c_s^2 <= 0 is REPORTED, not asserted -- same convention as test 6.
  MESSAGE("M3i synthetic x-low band: n=" << n << " cs2 in [" << cs2_min << ", " << cs2_max
                                          << "] cs2<=0: " << n_cs2_nonpos);
}

// ==========================================================================
// 13. Real tables (guarded): the x-low band's far tail is 1/3 too
// ==========================================================================
//
// The x-low counterpart of test 7, and the sharp form of the M3i claim: at a
// radiation-dominated seam the fitted slopes are a_x = b_x = -ln10 per
// decade of rho with a_u = 3*ln10 and b_u = 4*ln10 per decade of T, so the
// fixed-s slope is q = -1 + 4/3 = 1/3 and c_s^2 = (q + q^2 + q')/(1 + q)
// = 1/3 -- exactly the hot-gas asymptote the u-high tail already reaches
// from the other side. Before M3i, L's slope-to-zero override forced
// b_x = 0, hence q = 4/3 and c_s^2 = 4/3: measured 1.35 all across this
// band, on both tables.

namespace {

void check_real_table_xlow_tail(const std::string &path, const char *label, double m_B_g) {
  const std::string use_path = eeos_san_table(path);
  if (!table_exists(use_path)) {
    MESSAGE("skipping " << label << " x-low tail test: " << use_path << " not present");
    return;
  }
  RawTable table = eeos::read_stellarcollapse(use_path);
  BuildOptions bopts;
  bopts.m_B_table_g = m_B_g;
  EntropyEOS adapter = eeos::build_entropy_eos(table, bopts);
  const EntropyEOSView view = adapter.view();

  size_t n = 0;
  double cs2_min = 1e300, cs2_max = -1e300;

  // The radiation-dominated part of the band: T >~ 1 MeV (the cold third of
  // the u axis is matter-dominated, was always causal, and has its own -- much
  // smaller -- asymptote; class E covers the whole band, this pins the value).
  const double u_rad_lo = 0.0;
  const int na = static_cast<int>(eeos_n(24, 8));
  const int nb = static_cast<int>(eeos_n(8, 4));
  for (int ia = 0; ia < na; ++ia) {
    const double u = u_rad_lo + (view.u_hi - u_rad_lo) * (ia + 0.5) / na;
    for (int ib = 0; ib < nb; ++ib) {
      const double y = view.y_lo + (view.y_hi - view.y_lo) * (ib + 0.5) / nb;
      // Deep in the band (past the blend cell), where the tail is the pure
      // power law and q is constant.
      const EOSPoint pt = view.eval_at(view.x_ext_lo, u, y, 0u, 0);
      REQUIRE(std::isfinite(pt.cs2));
      CHECK(pt.cs2 < 1.0);
      CHECK(pt.p > 0.0);
      cs2_min = std::min(cs2_min, pt.cs2);
      cs2_max = std::max(cs2_max, pt.cs2);
      ++n;
    }
  }
  MESSAGE(std::string(label) << " far x-low tail (x = x_ext_lo, T >= 1 MeV): n=" << n << " cs2 in ["
                             << cs2_min << ", " << cs2_max << "]");
  // Same 10x-margin discriminator as test 7: the pre-M3i tail sat at 1.35
  // here, so anything in this window is unambiguously the 1/3 asymptote.
  CHECK(cs2_min > 0.15);
  CHECK(cs2_max < 0.75);
}

} // namespace

TEST_CASE("M3i far x-low tail: LS220 real table (guarded)") {
  check_real_table_xlow_tail(kLS220Path, "LS220", eeos::m_amu_g);
}

TEST_CASE("M3i far x-low tail: SRO real table (guarded)") {
  if (eeos_skip_big_table("test_adapter_tail (SRO): real table")) return;
  check_real_table_xlow_tail(kSROPath, "SRO", eeos::m_neutron_g);
}

TEST_CASE("M3i far x-low tail: DD2 real table (guarded)") {
  if (eeos_skip_big_table("test_adapter_tail (DD2): real table")) return;
  check_real_table_xlow_tail(kDD2Path, "DD2", eeos::m_amu_g);
}

TEST_CASE("M3i far x-low tail: SFHo real table (guarded)") {
  if (eeos_skip_big_table("test_adapter_tail (SFHo): real table")) return;
  check_real_table_xlow_tail(kSFHoPath, "SFHo", eeos::m_amu_g);
}
