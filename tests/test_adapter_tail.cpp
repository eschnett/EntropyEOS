// tests/test_adapter_tail.cpp — unit tests for the M3g causal extension
// tails (eos-causal-tail.md; entropy_eos/core/adapter_eval.hpp's "CAUSAL
// TAILS" module section): the log-space u-high sigma tail, the causal slope
// clamp on L's u-high tail, and the two properties the whole design rests
// on -- that the tail operator stays EXACTLY transparent inside the
// physical box, and that the far u-high tail is causal.
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
  CHECK(sig_spec.u_high_log); // the log tail really is enabled on this path

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
        eeos::detail::TailSpec{hu, m_floor_g, false, 0.0}));

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

void check_real_table_far_tail(const std::string &path, const char *label, double m_B_g) {
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
  // The radiation asymptote, with generous room for the table's own slope
  // scatter -- the pre-M3g linear-sigma tail saturated at c_s^2 ~ 3.8 here,
  // so this is a 10x-margin discriminator, not a tight fit.
  CHECK(cs2_min > 0.15);
  CHECK(cs2_max < 0.75);
  CHECK(n_clamped == 0);     // b/alpha - 1 ~ 1/3 is nowhere near cs2_ext_cap
  CHECK(n_floor_wins == 0);
}

} // namespace

TEST_CASE("M3g far u-high tail: LS220 real table (guarded)") {
  check_real_table_far_tail(kLS220Path, "LS220", eeos::m_amu_g);
}

TEST_CASE("M3g far u-high tail: SRO real table (guarded)") {
  if (eeos_skip_big_table("test_adapter_tail (SRO): real table")) return;
  check_real_table_far_tail(kSROPath, "SRO", eeos::m_neutron_g);
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
