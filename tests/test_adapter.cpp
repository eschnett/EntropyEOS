// tests/test_adapter.cpp — unit tests for the M2b adapter core:
// entropy_eos/core/adapter_eval.hpp (EntropyEOSView::evaluate()/srange())
// and entropy_eos/host/adapter_build.{hpp,cpp} (build_entropy_eos()),
// exercised against the synthetic ground-truth gas (entropy_eos/host/
// synthetic.hpp) and, when present locally, the real LS220/SRO tables (see
// CODE.md "Test harness" / eos-adapter-F-to-U.md S10-S11).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "entropy_eos/core/adapter_eval.hpp"
#include "entropy_eos/host/adapter_build.hpp"
#include "entropy_eos/host/io_stellarcollapse.hpp"
#include "entropy_eos/host/repair.hpp"
#include "entropy_eos/host/synthetic.hpp"
#include "entropy_eos/host/table.hpp"
#include "entropy_eos/host/units.hpp"

using eeos::AdapterAudit;
using eeos::BuildOptions;
using eeos::EntropyEOS;
using eeos::EntropyEOSView;
using eeos::EOSPoint;
using eeos::RawTable;
using eeos::SRange;
using eeos::SyntheticOptions;

namespace {

double rel_err(double got, double ref, double floor = 1e-300) {
  return std::fabs(got - ref) / std::max(std::fabs(ref), floor);
}

double nan_guess() { return std::numeric_limits<double>::quiet_NaN(); }

// --- synthetic closed-form ground truth (task spec / CODE.md) --------------
//
// With g = 1+Ye and conv_t = MeV_to_erg/(m_B* c^2) (== MeV_to_erg/(m_amu_g
// c^2) for the synthetic table, since kappa == 1 exactly -- see test 1):
//
//   T(rho,s,Ye) [MeV] = T0 * exp((2/3)*(s/g - s0 + ln(rho/rho0)))
//   U = 1.5*g*conv_t*T_MeV
//   U_rho = (2/3)*U/rho,        U_s = (2/(3g))*U
//   U_rhorho = -(2/9)*U/rho^2,  U_rhos = (4/(9g))*U/rho
//   mu_tilde = (U/g)*(1 - (2/3)*s/g)
//   That = conv_t*T_MeV, T_F = T
//
// This is the analytic inverse of eeos::synthetic_s()/synthetic_eps()
// (host/synthetic.hpp), independent of anything under test here.

double g_of_ye(double ye) { return 1.0 + ye; }

double closed_T_MeV(double rho, double s, double ye, const SyntheticOptions &opts) {
  const double g = g_of_ye(ye);
  return opts.T0_MeV * std::exp((2.0 / 3.0) * (s / g - opts.s0 + std::log(rho / opts.rho0_gcc)));
}

struct ClosedForm {
  double U, U_rho, U_s, U_rhorho, U_rhos, mu_tilde, cs2, T_F_MeV;
};

ClosedForm closed_form(double rho, double s, double ye, const SyntheticOptions &opts, double conv_t) {
  const double g = g_of_ye(ye);
  const double T_MeV = closed_T_MeV(rho, s, ye, opts);
  const double U = 1.5 * g * conv_t * T_MeV;

  ClosedForm c;
  c.U = U;
  c.U_rho = (2.0 / 3.0) * U / rho;
  c.U_s = (2.0 / (3.0 * g)) * U;
  c.U_rhorho = -(2.0 / 9.0) * U / (rho * rho);
  c.U_rhos = (4.0 / (9.0 * g)) * U / rho;
  c.mu_tilde = (U / g) * (1.0 - (2.0 / 3.0) * s / g);
  const double h = 1.0 + (5.0 / 3.0) * U;
  c.cs2 = (10.0 / 9.0) * U / h;
  c.T_F_MeV = T_MeV;
  return c;
}

EntropyEOS build_synthetic(const SyntheticOptions &opts) {
  RawTable table = eeos::make_synthetic_table(opts);
  return eeos::build_entropy_eos(table);
}

// Interior sampling helpers: keep random (x=log10 rho, y=Ye) points a small
// margin away from the box edges, and random s a margin inside
// [srange.s_min, srange.s_max], so a sampled point (and, for test 4, its FD
// stencil neighbors) never crosses into the S7 clamp-and-flag extensions --
// those are a deliberate discontinuity in the derivative and would corrupt
// both the closed-form comparison (spline boundary conditions bias the fit
// right at the edge) and any finite-difference check.
struct InteriorSampler {
  std::mt19937 rng;
  std::uniform_real_distribution<double> xq, yq, frac;

  InteriorSampler(const SyntheticOptions &opts, unsigned seed, double margin_frac = 0.02)
      : rng(seed), frac(0.0, 1.0) {
    const double xlo = std::log10(opts.rho_min_gcc), xhi = std::log10(opts.rho_max_gcc);
    const double xspan = xhi - xlo;
    xq = std::uniform_real_distribution<double>(xlo + margin_frac * xspan, xhi - margin_frac * xspan);
    const double yspan = opts.ye_max - opts.ye_min;
    yq = std::uniform_real_distribution<double>(opts.ye_min + margin_frac * yspan,
                                                 opts.ye_max - margin_frac * yspan);
  }

  double rho() { return std::pow(10.0, xq(rng)); }
  double ye() { return yq(rng); }

  // s uniform in [srange.s_min, srange.s_max], shrunk by `s_margin_frac` on
  // each side.
  double s_in(const SRange &sr, double s_margin_frac) {
    const double span = sr.s_max - sr.s_min;
    const double m = s_margin_frac * span;
    return sr.s_min + m + frac(rng) * (span - 2.0 * m);
  }
};

double max_abs(const std::vector<double> &v) {
  double m = 0.0;
  for (double x : v) m = std::max(m, std::fabs(x));
  return std::max(m, 1e-300);
}

void print_iters_histogram(const std::string &label, const std::vector<int> &iters) {
  std::map<int, int> hist;
  for (int it : iters) ++hist[it];
  std::cout << label << " (" << iters.size() << " samples):\n";
  for (const auto &kv : hist) {
    std::cout << "    iters=" << kv.first << ": " << kv.second << "\n";
  }
}

} // namespace

// ==========================================================================
// 1. kappa near 1.0 on the synthetic table (eps > 0 everywhere physically)
// ==========================================================================

TEST_CASE("build_entropy_eos: synthetic table gives kappa close to 1.0 (eps > 0 everywhere physically)") {
  SyntheticOptions opts; // default grid, no defects: eps > 0 everywhere in the physical box
  EntropyEOS adapter = build_synthetic(opts);

  // M2d-2: the eps-floor scan now also samples the S7 extension zones
  // (host/adapter_build.hpp's build_entropy_eos() step 3 doc comment), and
  // the L u-low tail's designed decay can dip eps_hat *there* below the
  // table's own (physical) minimum by a bounded amount even on an
  // everywhere-positive table -- so kappa is no longer exactly 1.0 in
  // general, only close to it (see test 3 below: "U >= 0 over the extended
  // box" is the real invariant this scan protects, not kappa==1 on a clean
  // table).
  CHECK(adapter.kappa() > 0.0);
  CHECK(adapter.kappa() <= 1.0);
  CHECK(adapter.kappa() > 1.0 - 1e-2);
  CHECK(std::isfinite(adapter.m_B_star_g()));
  CHECK(rel_err(adapter.m_B_star_g(), eeos::m_amu_g) <= 1e-2);
}

// ==========================================================================
// 2. Node round trip: splines interpolate exactly, so the T-solve lands
//    back on the node
// ==========================================================================

TEST_CASE("EntropyEOSView::evaluate: node round trip at every 5th table node") {
  SyntheticOptions opts; // default 40x30x10
  RawTable table = eeos::make_synthetic_table(opts);
  EntropyEOS adapter = eeos::build_entropy_eos(table);
  EntropyEOSView view = adapter.view();

  const std::vector<double> &entropy = table.field("entropy");
  const std::vector<double> &logenergy = table.field("logenergy");
  const double shift_cgs = table.energy_shift();
  const double inv_c2 = 1.0 / (eeos::c_light_cm_s * eeos::c_light_cm_s);
  const double kappa = adapter.kappa();

  int n_checked = 0;
  for (size_t k = 0; k < table.nye(); k += 5) {
    for (size_t j = 0; j < table.ntemp(); j += 5) {
      for (size_t i = 0; i < table.nrho(); i += 5) {
        const double rho = table.rho(i);
        const double T_j = table.temp(j);
        const double ye = table.yev(k);
        const size_t idx = table.index(i, j, k);
        const double s = entropy[idx];
        // U at a node equals eps_hat exactly, transformed by the S5
        // kappa re-zeroing (eos-adapter-F-to-U.md S5): eps_hat =
        // (10^logenergy - energy_shift_cgs) / c^2, U = (1+eps_hat)/kappa-1.
        const double eps_hat_node = (std::pow(10.0, logenergy[idx]) - shift_cgs) * inv_c2;
        const double U_expected = (1.0 + eps_hat_node) / kappa - 1.0;

        // The adapter's evaluate() takes rho* = kappa*rho (S5's "from here
        // on, all solver-facing formulas say rho and mean rho*"), not the
        // table's raw physical rho -- see adapter_audit.cpp's audit_nodes()
        // for the same convention.
        const EOSPoint pt = view.evaluate(kappa * rho, s, ye, nan_guess());

        CHECK(rel_err(pt.T_F_MeV, T_j) <= 1e-10);
        CHECK(rel_err(pt.U, U_expected) <= 1e-12);
        ++n_checked;
      }
    }
  }
  CHECK(n_checked > 0);
}

// ==========================================================================
// 3. Closed-form comparison at 500 random interior points, default vs. fine
//    grid, with grid convergence
// ==========================================================================

namespace {

struct SampleStats {
  double max_rel_U = 0, max_rel_U_rho = 0, max_rel_U_s = 0, max_rel_U_rhorho = 0, max_rel_U_rhos = 0,
         max_rel_mu = 0, max_rel_cs2 = 0, max_rel_TF = 0;
  std::vector<int> iters_cold;
  bool any_maxiter = false;
};

SampleStats run_closed_form_samples(const EntropyEOSView &view, const SyntheticOptions &opts,
                                     double conv_t, double kappa, int npts, unsigned seed) {
  InteriorSampler sampler(opts, seed);
  SampleStats st;
  st.iters_cold.reserve(static_cast<size_t>(npts));

  for (int k = 0; k < npts; ++k) {
    const double rho = sampler.rho();
    const double rho_star = kappa * rho; // evaluate()/srange() take rho* (S5), not physical rho
    const double ye = sampler.ye();
    const SRange sr = view.srange(rho_star, ye);
    const double s = sampler.s_in(sr, /*s_margin_frac=*/0.05);

    const EOSPoint pt = view.evaluate(rho_star, s, ye, nan_guess());
    st.iters_cold.push_back(pt.iters);
    if (pt.flags & eeos::flag_maxiter) st.any_maxiter = true;

    // closed_form()'s fields are the *pre-kappa* (Uh-like) quantities (see
    // its doc comment); apply the same S5 Stage C transform evaluate()
    // itself applies before comparing against pt's fields. T_F is
    // unaffected by kappa (it depends only on the solved u, and rho*'s
    // x-label shift by log10(kappa) exactly cancels against the spline's
    // own x0-shift by the same amount -- see build_entropy_eos()).
    const ClosedForm cf = closed_form(rho, s, ye, opts, conv_t);
    const double U_exp = (1.0 + cf.U) / kappa - 1.0;
    const double Urho_exp = cf.U_rho / kappa;
    const double Us_exp = cf.U_s / kappa;
    const double Urhorho_exp = cf.U_rhorho / kappa;
    const double Urhos_exp = cf.U_rhos / kappa;
    const double mu_exp = cf.mu_tilde / kappa;
    const double h_exp = 1.0 + U_exp + rho_star * Urho_exp;
    const double cs2_exp = (2.0 * rho_star * Urho_exp + rho_star * rho_star * Urhorho_exp) / h_exp;

    st.max_rel_U = std::max(st.max_rel_U, rel_err(pt.U, U_exp));
    st.max_rel_U_rho = std::max(st.max_rel_U_rho, rel_err(pt.U_rho, Urho_exp));
    st.max_rel_U_s = std::max(st.max_rel_U_s, rel_err(pt.U_s, Us_exp));
    st.max_rel_U_rhorho = std::max(st.max_rel_U_rhorho, rel_err(pt.U_rhorho, Urhorho_exp));
    st.max_rel_U_rhos = std::max(st.max_rel_U_rhos, rel_err(pt.U_rhos, Urhos_exp));
    st.max_rel_mu = std::max(st.max_rel_mu, rel_err(pt.mu_tilde, mu_exp));
    st.max_rel_cs2 = std::max(st.max_rel_cs2, rel_err(pt.cs2, cs2_exp));
    st.max_rel_TF = std::max(st.max_rel_TF, rel_err(pt.T_F_MeV, cf.T_F_MeV));
  }
  return st;
}

void print_sample_stats(const std::string &label, const SampleStats &st) {
  std::cout << label << ":\n"
            << "    U="        << st.max_rel_U
            << " U_rho="       << st.max_rel_U_rho
            << " U_s="         << st.max_rel_U_s
            << " U_rhorho="    << st.max_rel_U_rhorho
            << " U_rhos="      << st.max_rel_U_rhos
            << " mu_tilde="    << st.max_rel_mu
            << " cs2="         << st.max_rel_cs2
            << " T_F="         << st.max_rel_TF << "\n";
}

} // namespace

TEST_CASE("EntropyEOSView::evaluate: closed-form comparison and grid convergence "
          "(default 40x30x10 vs. fine 120x120x12)") {
  SyntheticOptions opts_default; // 40x30x10
  SyntheticOptions opts_fine = opts_default;
  opts_fine.nrho = 120;
  opts_fine.ntemp = 120;
  opts_fine.nye = 12;

  EntropyEOS adapter_default = build_synthetic(opts_default);
  EntropyEOS adapter_fine = build_synthetic(opts_fine);
  const EntropyEOSView view_default = adapter_default.view();
  const EntropyEOSView view_fine = adapter_fine.view();

  const int npts = 500;
  const SampleStats st_default = run_closed_form_samples(
      view_default, opts_default, adapter_default.conv_t(), adapter_default.kappa(), npts, 20260825u);
  const SampleStats st_fine = run_closed_form_samples(view_fine, opts_fine, adapter_fine.conv_t(),
                                                        adapter_fine.kappa(), npts, 20260826u);

  print_sample_stats("test_adapter 3: default-grid (40x30x10) max relative errors", st_default);
  print_sample_stats("test_adapter 3: fine-grid (120x120x12) max relative errors", st_fine);
  print_iters_histogram("test_adapter 3: cold iteration histogram (default grid)", st_default.iters_cold);

  // Default-grid tolerances.
  CHECK(st_default.max_rel_U <= 1e-3);
  CHECK(st_default.max_rel_U_rho <= 1e-3);
  CHECK(st_default.max_rel_U_s <= 1e-3);
  CHECK(st_default.max_rel_TF <= 1e-3);
  CHECK(st_default.max_rel_U_rhorho <= 3e-2);
  CHECK(st_default.max_rel_U_rhos <= 3e-2);
  CHECK(st_default.max_rel_mu <= 3e-2);
  CHECK(st_default.max_rel_cs2 <= 3e-2);

  // Fine-grid tolerances.
  CHECK(st_fine.max_rel_U <= 1e-5);
  CHECK(st_fine.max_rel_U_rho <= 1e-5);
  CHECK(st_fine.max_rel_U_s <= 1e-5);
  CHECK(st_fine.max_rel_TF <= 1e-5);
  CHECK(st_fine.max_rel_U_rhorho <= 1e-3);
  CHECK(st_fine.max_rel_U_rhos <= 1e-3);
  CHECK(st_fine.max_rel_mu <= 1e-3);
  CHECK(st_fine.max_rel_cs2 <= 1e-3);

  // Grid convergence: every quantity's fine-grid error is at least 5x
  // smaller than its default-grid error -- except T_F, see below.
  auto conv_factor = [](double coarse, double fine) { return coarse / std::max(fine, 1e-300); };
  CHECK(conv_factor(st_default.max_rel_U, st_fine.max_rel_U) >= 5.0);
  CHECK(conv_factor(st_default.max_rel_U_rho, st_fine.max_rel_U_rho) >= 5.0);
  CHECK(conv_factor(st_default.max_rel_U_s, st_fine.max_rel_U_s) >= 5.0);
  CHECK(conv_factor(st_default.max_rel_U_rhorho, st_fine.max_rel_U_rhorho) >= 5.0);
  CHECK(conv_factor(st_default.max_rel_U_rhos, st_fine.max_rel_U_rhos) >= 5.0);
  CHECK(conv_factor(st_default.max_rel_mu, st_fine.max_rel_mu) >= 5.0);
  CHECK(conv_factor(st_default.max_rel_cs2, st_fine.max_rel_cs2) >= 5.0);

  // T_F is exempt from the 5x-shrink form of the convergence check: the
  // synthetic entropy s(x,u,y) = (1+Ye)*(s0 + 1.5*ln(10)*u - ln(10)*x + ...)
  // is *affine* in the fit's native (x,u,y) coordinates (ln T and ln rho
  // enter s linearly), and a not-a-knot cubic spline reproduces any
  // polynomial of degree <= 3 -- hence degree <= 1 -- exactly, at any
  // resolution (see host/bspline_fit.hpp's doc comment and
  // tests/test_bspline.cpp test 2). So Sigma-hat already equals sigma to
  // roundoff on the *coarse* grid, the T-solve (and thus T_F) has no
  // spline truncation error left to shrink, and both grids' errors sit at
  // the T-solve's own 1e-12/1e-13 convergence-tolerance floor (observed:
  // ~1e-14 on both grids -- see the printed table above). logenergy, in
  // contrast, is log(affine-in-10^u), genuinely nonlinear in u, so U/U_rho/
  // etc. (all derived from the L spline) show real truncation error that
  // does shrink with resolution, as checked above. What's asserted here
  // instead is the stronger fact that both grids already clear the *fine*-
  // grid ceiling by several orders of magnitude.
  CHECK(st_default.max_rel_TF <= 1e-9);
  CHECK(st_fine.max_rel_TF <= 1e-9);

  // No sample anywhere hit the iteration cap.
  CHECK_FALSE(st_default.any_maxiter);
  CHECK_FALSE(st_fine.any_maxiter);
  for (int it : st_default.iters_cold) CHECK(it <= adapter_default.max_iter());
}

// ==========================================================================
// 4. Derivative self-consistency: Richardson-extrapolated central FD of
//    evaluate()'s own U, independent of the closed form
// ==========================================================================

TEST_CASE("EntropyEOSView::evaluate: derivatives match Richardson-extrapolated FD of U "
          "(fine grid, 100 points)") {
  SyntheticOptions opts;
  opts.nrho = 120;
  opts.ntemp = 120;
  opts.nye = 12;
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();

  auto Uof = [&](double rho, double s, double ye) { return view.evaluate(rho, s, ye, nan_guess()).U; };
  auto Urho_of = [&](double rho, double s, double ye) {
    return view.evaluate(rho, s, ye, nan_guess()).U_rho;
  };
  auto Us_of = [&](double rho, double s, double ye) {
    return view.evaluate(rho, s, ye, nan_guess()).U_s;
  };

  // Richardson-extrapolates a pair of 2nd-order-accurate central-difference
  // estimates (step h and 2h) to 4th order, cancelling the leading
  // truncation term (same technique as tests/test_bspline.cpp test 8).
  auto richardson = [](double d_h, double d_2h) { return (4.0 * d_h - d_2h) / 3.0; };

  InteriorSampler sampler(opts, 70070707u);

  const int npts = 100;
  std::vector<double> ana_Urho(npts), ana_Us(npts), ana_Urhorho(npts), ana_Urhos(npts);
  std::vector<double> fd_Urho(npts), fd_Us(npts), fd_Urhorho(npts);
  std::vector<double> fd_Urhos_a(npts), fd_Urhos_b(npts); // U_rho along s, U_s along rho

  for (int k = 0; k < npts; ++k) {
    const double rho = sampler.rho();
    const double ye = sampler.ye();
    const SRange sr = view.srange(rho, ye);
    const double s = sampler.s_in(sr, /*s_margin_frac=*/0.1);
    const double s_range = sr.s_max - sr.s_min;

    const EOSPoint pt = view.evaluate(rho, s, ye, nan_guess());
    ana_Urho[static_cast<size_t>(k)] = pt.U_rho;
    ana_Us[static_cast<size_t>(k)] = pt.U_s;
    ana_Urhorho[static_cast<size_t>(k)] = pt.U_rhorho;
    ana_Urhos[static_cast<size_t>(k)] = pt.U_rhos;

    // Relative steps: h_rho proportional to rho (rho spans ~10 decades),
    // h_s proportional to the local entropy bracket width. Both are tiny
    // compared to the 10% interior margin used above, so the FD stencils
    // (up to 2h) never approach an S7 extension boundary.
    const double h_rho = 1e-4 * rho;
    const double h_s = 1e-3 * s_range;

    auto d1_rho = [&](double h) { return (Uof(rho + h, s, ye) - Uof(rho - h, s, ye)) / (2.0 * h); };
    fd_Urho[static_cast<size_t>(k)] = richardson(d1_rho(h_rho), d1_rho(2.0 * h_rho));

    auto d1_s = [&](double h) { return (Uof(rho, s + h, ye) - Uof(rho, s - h, ye)) / (2.0 * h); };
    fd_Us[static_cast<size_t>(k)] = richardson(d1_s(h_s), d1_s(2.0 * h_s));

    auto d2_rho = [&](double h) {
      return (Urho_of(rho + h, s, ye) - Urho_of(rho - h, s, ye)) / (2.0 * h);
    };
    fd_Urhorho[static_cast<size_t>(k)] = richardson(d2_rho(h_rho), d2_rho(2.0 * h_rho));

    auto dmix_a = [&](double h) {
      return (Urho_of(rho, s + h, ye) - Urho_of(rho, s - h, ye)) / (2.0 * h);
    };
    fd_Urhos_a[static_cast<size_t>(k)] = richardson(dmix_a(h_s), dmix_a(2.0 * h_s));

    auto dmix_b = [&](double h) { return (Us_of(rho + h, s, ye) - Us_of(rho - h, s, ye)) / (2.0 * h); };
    fd_Urhos_b[static_cast<size_t>(k)] = richardson(dmix_b(h_rho), dmix_b(2.0 * h_rho));
  }

  const double scale_Urho = max_abs(ana_Urho);
  const double scale_Us = max_abs(ana_Us);
  const double scale_Urhorho = max_abs(ana_Urhorho);
  const double scale_Urhos = max_abs(ana_Urhos);

  for (int k = 0; k < npts; ++k) {
    const size_t i = static_cast<size_t>(k);
    CHECK(std::fabs(ana_Urho[i] - fd_Urho[i]) <= 1e-7 * scale_Urho);
    CHECK(std::fabs(ana_Us[i] - fd_Us[i]) <= 1e-7 * scale_Us);
    CHECK(std::fabs(ana_Urhorho[i] - fd_Urhorho[i]) <= 1e-5 * scale_Urhorho);
    // Mixed-partial symmetry, checked both ways end to end.
    CHECK(std::fabs(ana_Urhos[i] - fd_Urhos_a[i]) <= 1e-5 * scale_Urhos);
    CHECK(std::fabs(ana_Urhos[i] - fd_Urhos_b[i]) <= 1e-5 * scale_Urhos);
  }
}

// ==========================================================================
// 5. Flags: s/rho/Ye out of range still return finite EOSPoints
// ==========================================================================

TEST_CASE("EntropyEOSView::evaluate: out-of-range flags, always finite") {
  SyntheticOptions opts;
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();

  auto check_finite = [](const EOSPoint &pt) {
    CHECK(std::isfinite(pt.U));
    CHECK(std::isfinite(pt.U_rho));
    CHECK(std::isfinite(pt.U_s));
    CHECK(std::isfinite(pt.U_rhorho));
    CHECK(std::isfinite(pt.U_rhos));
    CHECK(std::isfinite(pt.That));
    CHECK(std::isfinite(pt.p));
    CHECK(std::isfinite(pt.h));
    CHECK(std::isfinite(pt.cs2));
    CHECK(std::isfinite(pt.T_F_MeV));
    CHECK(std::isfinite(pt.mu_tilde));
  };

  const double rho_mid = std::sqrt(std::pow(10.0, view.x_lo) * std::pow(10.0, view.x_hi));
  const double ye_mid = 0.5 * (view.y_lo + view.y_hi);
  const SRange sr_mid = view.srange(rho_mid, ye_mid);
  const double s_mid = 0.5 * (sr_mid.s_min + sr_mid.s_max);

  // M2d-2: a moderately out-of-range s now lands the T-solve *inside* the
  // designed extension zone (a genuine root of sigma_ext, not pinned at the
  // physical edge) -- see tests/test_adapter.cpp test 9 below for the full
  // extended-T-solve round trip. Only an s beyond even the extended bracket
  // hard-clamps at the extended edge.
  SUBCASE("s below range -> flag_ext_s_low, u_solved in the extension zone") {
    const EOSPoint pt = view.evaluate(rho_mid, sr_mid.s_min - 1.0, ye_mid, nan_guess());
    CHECK((pt.flags & eeos::flag_ext_s_low) != 0);
    CHECK(pt.u_solved < view.u_lo);
    CHECK(pt.u_solved >= view.u_ext_lo);
    check_finite(pt);
  }

  SUBCASE("s far below the extended bracket -> hard clamp at u_ext_lo") {
    const EOSPoint pt = view.evaluate(rho_mid, sr_mid.s_min - 1.0e6, ye_mid, nan_guess());
    CHECK((pt.flags & eeos::flag_ext_s_low) != 0);
    CHECK(pt.u_solved == view.u_ext_lo);
    check_finite(pt);
  }

  SUBCASE("s above range -> flag_ext_s_high, u_solved in the extension zone") {
    const EOSPoint pt = view.evaluate(rho_mid, sr_mid.s_max + 1.0, ye_mid, nan_guess());
    CHECK((pt.flags & eeos::flag_ext_s_high) != 0);
    CHECK(pt.u_solved > view.u_hi);
    CHECK(pt.u_solved <= view.u_ext_hi);
    check_finite(pt);
  }

  SUBCASE("s far above the extended bracket -> hard clamp at u_ext_hi") {
    const EOSPoint pt = view.evaluate(rho_mid, sr_mid.s_max + 1.0e6, ye_mid, nan_guess());
    CHECK((pt.flags & eeos::flag_ext_s_high) != 0);
    CHECK(pt.u_solved == view.u_ext_hi);
    check_finite(pt);
  }

  SUBCASE("rho below grid -> flag_ext_rho_low") {
    const double rho_below = std::pow(10.0, view.x_lo) * 1e-2;
    const EOSPoint pt = view.evaluate(rho_below, s_mid, ye_mid, nan_guess());
    CHECK((pt.flags & eeos::flag_ext_rho_low) != 0);
    check_finite(pt);
  }

  SUBCASE("rho above grid -> flag_oob_rho_high") {
    const double rho_above = std::pow(10.0, view.x_hi) * 1e2;
    const EOSPoint pt = view.evaluate(rho_above, s_mid, ye_mid, nan_guess());
    CHECK((pt.flags & eeos::flag_oob_rho_high) != 0);
    check_finite(pt);
  }

  SUBCASE("Ye below range -> flag_clamp_ye") {
    const EOSPoint pt = view.evaluate(rho_mid, s_mid, view.y_lo - 0.1, nan_guess());
    CHECK((pt.flags & eeos::flag_clamp_ye) != 0);
    check_finite(pt);
  }

  SUBCASE("Ye above range -> flag_clamp_ye") {
    const EOSPoint pt = view.evaluate(rho_mid, s_mid, view.y_hi + 0.1, nan_guess());
    CHECK((pt.flags & eeos::flag_clamp_ye) != 0);
    check_finite(pt);
  }

  // M2d-2 corner case: both rho *and* s outside the physical box at once
  // (the "apply the u-tail first at the x-clamped seam, then the x-tail on
  // the resulting tracks" composition order -- core/adapter_eval.hpp's
  // aeval_extended()) still returns a finite, correctly-flagged, monotone
  // (U_s > 0) point.
  SUBCASE("corner: rho below grid AND s below range -> both flags, finite") {
    const double rho_below = std::pow(10.0, view.x_lo) * 1e-2; // 2 decades below x_lo
    // A generously large s offset: the x-low tail's entropy grows without
    // bound as rho drops (S7's "sigma_x -> const < 0", entropy increases
    // as density decreases), so sr_mid's *physical-x* s_min is not a tight
    // local bound this far into the x-tail -- 100 kB/baryon comfortably
    // clears any such shift for this synthetic table's O(10) entropy scale.
    const EOSPoint pt = view.evaluate(rho_below, sr_mid.s_min - 100.0, ye_mid, nan_guess());
    CHECK((pt.flags & eeos::flag_ext_rho_low) != 0);
    CHECK((pt.flags & eeos::flag_ext_s_low) != 0);
    CHECK(pt.U_s > 0.0);
    check_finite(pt);
  }

  SUBCASE("corner: rho above grid AND s above range -> both flags, finite") {
    const double rho_above = std::pow(10.0, view.x_hi) * 1e2; // 2 decades above x_hi
    const EOSPoint pt = view.evaluate(rho_above, sr_mid.s_max + 100.0, ye_mid, nan_guess());
    CHECK((pt.flags & eeos::flag_oob_rho_high) != 0);
    CHECK((pt.flags & eeos::flag_ext_s_high) != 0);
    CHECK(pt.U_s > 0.0);
    check_finite(pt);
  }
}

// ==========================================================================
// 6. Warm starts converge in a handful of iterations
// ==========================================================================

TEST_CASE("EntropyEOSView::evaluate: warm start from a nearby converged point converges fast") {
  SyntheticOptions opts; // default grid
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();

  InteriorSampler sampler(opts, 600060006u);

  const int npts = 50;
  for (int k = 0; k < npts; ++k) {
    const double rho = sampler.rho();
    const double ye = sampler.ye();
    const SRange sr = view.srange(rho, ye);
    const double s = sampler.s_in(sr, /*s_margin_frac=*/0.1);

    const EOSPoint cold = view.evaluate(rho, s, ye, nan_guess());
    CHECK(cold.iters <= adapter.max_iter());
    CHECK((cold.flags & eeos::flag_maxiter) == 0);

    const double rho2 = rho * (1.0 + 1e-4);
    const double s2 = s * (1.0 + 1e-4);
    const EOSPoint warm = view.evaluate(rho2, s2, ye, cold.u_solved);
    CHECK(warm.iters <= 4);
  }
}

// ==========================================================================
// 7. srange matches the closed-form s(rho, T_min/T_max, Ye) at nodes
// ==========================================================================

TEST_CASE("EntropyEOSView::srange: matches closed-form s(rho, T_min/T_max, Ye) at every node") {
  SyntheticOptions opts; // default grid
  RawTable table = eeos::make_synthetic_table(opts);
  EntropyEOS adapter = eeos::build_entropy_eos(table);
  const EntropyEOSView view = adapter.view();

  const double kappa = adapter.kappa();
  for (size_t i = 0; i < table.nrho(); ++i) {
    const double rho = table.rho(i);
    for (size_t k = 0; k < table.nye(); ++k) {
      const double ye = table.yev(k);
      // srange() takes rho* = kappa*rho (S5); sigma itself (entropy) is not
      // transformed by kappa -- only the x-label is shifted -- so the
      // closed-form s reference stays a function of the physical rho.
      const SRange sr = view.srange(kappa * rho, ye);

      const double s_lo_ref = eeos::synthetic_s(rho, opts.temp_min_MeV, ye, opts);
      const double s_hi_ref = eeos::synthetic_s(rho, opts.temp_max_MeV, ye, opts);

      CHECK(rel_err(sr.s_min, s_lo_ref) <= 1e-10);
      CHECK(rel_err(sr.s_max, s_hi_ref) <= 1e-10);
    }
  }
}

// ==========================================================================
// 8. Real tables (guarded: skipped cleanly if tables/ is absent, like
//    tests/test_io_stellarcollapse.cpp)
// ==========================================================================

namespace {

const std::string kLS220Path = "tables/LS220_234r_136t_50y_analmu_20091212_SVNr26.h5";
const std::string kSROPath = "tables/LS220_3335_rho391_temp163_ye66.h5";

bool table_exists(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

void run_real_table_adapter_test(const std::string &path, const char *label, unsigned seed) {
  if (!table_exists(path)) {
    WARN_MESSAGE(false, label << " table not found at '" << path << "' -- skipped ('skipped')");
    return;
  }

  RawTable table = eeos::read_stellarcollapse(path);
  const eeos::RepairResult repair_result = eeos::repair_table(table);
  std::cout << "test_adapter 8 (" << label << "): repair changed " << repair_result.entries.size()
            << " value(s)\n";

  const auto build_t0 = std::chrono::steady_clock::now();
  EntropyEOS adapter = eeos::build_entropy_eos(table);
  const auto build_t1 = std::chrono::steady_clock::now();
  const double build_seconds = std::chrono::duration<double>(build_t1 - build_t0).count();

  const AdapterAudit &audit = adapter.audit();
  std::cout << "test_adapter 8 (" << label << "): kappa=" << adapter.kappa()
            << " m_B_star_g=" << adapter.m_B_star_g() << " build_wall_time_s=" << build_seconds << "\n"
            << "  audit sigma_u: min=" << audit.sigma_u.min_value
            << " violations=" << audit.sigma_u.violation_count << "\n"
            << "  audit L_u:     min=" << audit.L_u.min_value
            << " violations=" << audit.L_u.violation_count << "\n";

  CHECK(std::isfinite(adapter.kappa()));
  CHECK(adapter.kappa() > 0.0);
  CHECK(adapter.kappa() <= 1.0);
  CHECK(std::isfinite(adapter.m_B_star_g()));

  const EntropyEOSView view = adapter.view();

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> xq(view.x_lo, view.x_hi);
  std::uniform_real_distribution<double> yq(view.y_lo, view.y_hi);
  std::uniform_real_distribution<double> frac(0.0, 1.0);

  const int npts = 20000;
  std::vector<int> iters_cold;
  iters_cold.reserve(static_cast<size_t>(npts));
  int maxiter_count = 0;

  const auto soak_t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < npts; ++k) {
    const double rho = std::pow(10.0, xq(rng));
    const double ye = yq(rng);
    const SRange sr = view.srange(rho, ye);
    const double s = sr.s_min + frac(rng) * (sr.s_max - sr.s_min);

    const EOSPoint pt = view.evaluate(rho, s, ye, nan_guess());

    CHECK(std::isfinite(pt.U));
    CHECK(std::isfinite(pt.U_rho));
    CHECK(std::isfinite(pt.U_s));
    CHECK(std::isfinite(pt.U_rhorho));
    CHECK(std::isfinite(pt.U_rhos));
    CHECK(std::isfinite(pt.That));
    CHECK(std::isfinite(pt.p));
    CHECK(std::isfinite(pt.h));
    CHECK(std::isfinite(pt.cs2));
    CHECK(std::isfinite(pt.T_F_MeV));
    CHECK(std::isfinite(pt.mu_tilde));

    if (pt.flags & eeos::flag_maxiter) ++maxiter_count;
    iters_cold.push_back(pt.iters);
  }
  const auto soak_t1 = std::chrono::steady_clock::now();
  const double soak_seconds = std::chrono::duration<double>(soak_t1 - soak_t0).count();
  const double evals_per_sec = static_cast<double>(npts) / std::max(soak_seconds, 1e-9);

  CHECK(maxiter_count == 0);

  std::cout << "test_adapter 8 (" << label << "): soak " << npts << " points, " << evals_per_sec
            << " evals/sec, maxiter_count=" << maxiter_count << "\n";
  print_iters_histogram(std::string("test_adapter 8 (") + label + "): cold iteration histogram",
                         iters_cold);
}

} // namespace

TEST_CASE("build_entropy_eos + evaluate: LS220 real table (guarded)") {
  run_real_table_adapter_test(kLS220Path, "LS220", 0x15220001u);
}

TEST_CASE("build_entropy_eos + evaluate: SRO real table (guarded)") {
  run_real_table_adapter_test(kSROPath, "SRO", 0x5502001u);
}

// ==========================================================================
// 9. M2d-2: seam continuity -- FD across each of u_lo, u_hi, x_lo at
//    offsets +-1e-7 in the seam coordinate, both grids where cheap.
// ==========================================================================
//
// The designed tails match value/1st/2nd derivative to the boundary spline
// sample exactly at the seam (d=0) by construction (core/adapter_eval.hpp's
// aeval_ramp_track()); they are not built to make U(seam+eps)-U(seam-eps)
// vanish, since U (and U_s/U_rho) generically has a nonzero derivative
// right at the seam -- a symmetric offset of eps=1e-7 straddling it moves U
// by O(eps * dU/d(seam coord)), which is *not* zero for a physically
// sensible EOS (verified empirically on the default grid: relative jumps
// of a few e-7 for U, U_s, and U_rho alike at all three seams -- consistent
// with the ~O(1) log-axis sensitivities of this table, not a defect). What
// a broken (non-C2, e.g. clamp-and-flag or a corner-composition-order bug)
// implementation would instead show is a jump *independent* of eps (an
// O(1) discontinuity) or several orders of magnitude larger than this
// natural first-derivative-driven scale; kTol below is set with generous
// (~30-300x) headroom above the observed natural scale so it stays
// meaningful without being flaky.
namespace {

constexpr double kSeamOffset = 1e-7;
constexpr double kSeamTol = 1e-5;

void check_seam_continuity(const EntropyEOSView &view, const char *label) {
  std::mt19937 rng(0x5EA30001u ^ static_cast<unsigned>(std::hash<std::string>{}(label)));
  const double x_margin = 0.1 * (view.x_hi - view.x_lo);
  std::uniform_real_distribution<double> xq(view.x_lo + x_margin, view.x_hi - x_margin);
  std::uniform_real_distribution<double> yq(view.y_lo, view.y_hi);

  auto check_pair = [](const EOSPoint &pin, const EOSPoint &pout, const char *seam) {
    CAPTURE(seam);
    CHECK(std::isfinite(pin.U));
    CHECK(std::isfinite(pout.U));
    CHECK(rel_err(pout.U, pin.U) <= kSeamTol);
    CHECK(rel_err(pout.U_s, pin.U_s) <= kSeamTol);
    CHECK(rel_err(pout.U_rho, pin.U_rho) <= kSeamTol);
  };

  const int npts = 20;
  for (int k = 0; k < npts; ++k) {
    const double x = xq(rng);
    const double ye = yq(rng);
    const double rho_star = std::pow(10.0, x);
    const double u_mid = 0.5 * (view.u_lo + view.u_hi);

    // u_lo seam: inside = u_lo + eps (physical side), outside = u_lo - eps.
    {
      const double s_in = view.sigma_extended(rho_star, view.u_lo + kSeamOffset, ye);
      const double s_out = view.sigma_extended(rho_star, view.u_lo - kSeamOffset, ye);
      const EOSPoint pin = view.evaluate(rho_star, s_in, ye, nan_guess());
      const EOSPoint pout = view.evaluate(rho_star, s_out, ye, nan_guess());
      CHECK((pout.flags & eeos::flag_ext_s_low) != 0);
      check_pair(pin, pout, "u_lo");
    }
    // u_hi seam: inside = u_hi - eps, outside = u_hi + eps.
    {
      const double s_in = view.sigma_extended(rho_star, view.u_hi - kSeamOffset, ye);
      const double s_out = view.sigma_extended(rho_star, view.u_hi + kSeamOffset, ye);
      const EOSPoint pin = view.evaluate(rho_star, s_in, ye, nan_guess());
      const EOSPoint pout = view.evaluate(rho_star, s_out, ye, nan_guess());
      CHECK((pout.flags & eeos::flag_ext_s_high) != 0);
      check_pair(pin, pout, "u_hi");
    }
    // x_lo seam: inside = x_lo + eps, outside = x_lo - eps, at a fixed u
    // well inside the physical range (only x crosses a seam here).
    {
      const double rho_in = std::pow(10.0, view.x_lo + kSeamOffset);
      const double rho_out = std::pow(10.0, view.x_lo - kSeamOffset);
      const double s = view.sigma_extended(rho_in, u_mid, ye);
      const EOSPoint pin = view.evaluate(rho_in, s, ye, nan_guess());
      const EOSPoint pout = view.evaluate(rho_out, s, ye, nan_guess());
      CHECK((pout.flags & eeos::flag_ext_rho_low) != 0);
      check_pair(pin, pout, "x_lo");
    }

    // Deep in the tails: finite and flagged correctly (u_lo tail, halfway
    // to the extended edge).
    {
      const double u_deep = view.u_lo - 0.5 * (view.u_lo - view.u_ext_lo);
      const double s_deep = view.sigma_extended(rho_star, u_deep, ye);
      const EOSPoint pt = view.evaluate(rho_star, s_deep, ye, nan_guess());
      CHECK(std::isfinite(pt.U));
      CHECK(std::isfinite(pt.U_s));
      CHECK(pt.U_s > 0.0);
      CHECK((pt.flags & eeos::flag_ext_s_low) != 0);
    }
  }
}

} // namespace

TEST_CASE("EntropyEOSView::evaluate: seam continuity across u_lo/u_hi/x_lo (M2d-2 C2 tails)") {
  SyntheticOptions opts_default; // 40x30x10
  SyntheticOptions opts_fine = opts_default;
  opts_fine.nrho = 120;
  opts_fine.ntemp = 120;
  opts_fine.nye = 12;

  EntropyEOS adapter_default = build_synthetic(opts_default);
  EntropyEOS adapter_fine = build_synthetic(opts_fine);

  check_seam_continuity(adapter_default.view(), "default");
  check_seam_continuity(adapter_fine.view(), "fine");
}

// ==========================================================================
// 10. M2d-2: extended T-solve -- s beyond the physical range still solves
//     to a finite, correctly-flagged point, and evaluate() round-trips a
//     point chosen inside the extension zone via sigma_extended().
// ==========================================================================

TEST_CASE("EntropyEOSView::evaluate: extended T-solve (s beyond range; round trip in the extension)") {
  SyntheticOptions opts; // default grid
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();
  const double kappa = adapter.kappa();

  InteriorSampler sampler(opts, 0x31415926u);
  const int npts = 30;
  for (int k = 0; k < npts; ++k) {
    const double rho = sampler.rho();
    const double ye = sampler.ye();
    const double rho_star = kappa * rho;
    const SRange sr = view.srange(rho_star, ye);

    // s_min(rho,Ye) - 2 kB / s_max + 2 kB.
    {
      const double s = sr.s_min - 2.0;
      const EOSPoint pt = view.evaluate(rho_star, s, ye, nan_guess());
      CHECK((pt.flags & eeos::flag_ext_s_low) != 0);
      CHECK(std::isfinite(pt.U));
      CHECK(std::isfinite(pt.U_s));
      CHECK(pt.U_s > 0.0);
      CHECK(pt.T_F_MeV < opts.temp_min_MeV);
      CHECK(pt.u_solved < view.u_lo);
    }
    {
      const double s = sr.s_max + 2.0;
      const EOSPoint pt = view.evaluate(rho_star, s, ye, nan_guess());
      CHECK((pt.flags & eeos::flag_ext_s_high) != 0);
      CHECK(std::isfinite(pt.U));
      CHECK(std::isfinite(pt.U_s));
      CHECK(pt.U_s > 0.0);
      CHECK(pt.T_F_MeV > opts.temp_max_MeV);
      CHECK(pt.u_solved > view.u_hi);
    }

    // Round trip within the extension: pick u_ext strictly inside the
    // extension zone on each side, get s_ext via sigma_extended() (the
    // audit/testing hook -- eos-adapter-F-to-U.md S7 / core/adapter_eval.hpp),
    // then evaluate(rho_star, s_ext, ye) must return u_solved == u_ext to
    // 1e-10.
    const double u_ext_below = view.u_lo - 0.5 * (view.u_lo - view.u_ext_lo);
    const double s_ext_below = view.sigma_extended(rho_star, u_ext_below, ye);
    const EOSPoint pt_below = view.evaluate(rho_star, s_ext_below, ye, nan_guess());
    CHECK(rel_err(pt_below.u_solved, u_ext_below) <= 1e-10);
    CHECK((pt_below.flags & eeos::flag_ext_s_low) != 0);

    const double u_ext_above = view.u_hi + 0.5 * (view.u_ext_hi - view.u_hi);
    const double s_ext_above = view.sigma_extended(rho_star, u_ext_above, ye);
    const EOSPoint pt_above = view.evaluate(rho_star, s_ext_above, ye, nan_guess());
    CHECK(rel_err(pt_above.u_solved, u_ext_above) <= 1e-10);
    CHECK((pt_above.flags & eeos::flag_ext_s_high) != 0);
  }
}

// ==========================================================================
// 11. M2d-2: U >= 0 over the extended box (the kappa scan now covers the
//     extensions too -- host/adapter_build.cpp's scan_extended_eps_floor()).
// ==========================================================================

TEST_CASE("EntropyEOSView::evaluate: U >= 0 over the extended box (synthetic table)") {
  SyntheticOptions opts; // default grid
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();

  std::mt19937 rng(0xE1E5F100u);
  std::uniform_real_distribution<double> xq(view.x_ext_lo, view.x_ext_hi);
  std::uniform_real_distribution<double> uq(view.u_ext_lo, view.u_ext_hi);
  std::uniform_real_distribution<double> yq(view.y_lo, view.y_hi);

  const int npts = 20000;
  int n_checked = 0;
  for (int k = 0; k < npts; ++k) {
    const double x = xq(rng);
    const double u = uq(rng);
    const double ye = yq(rng);
    const double rho_star = std::pow(10.0, x);
    const double s = view.sigma_extended(rho_star, u, ye);

    const EOSPoint pt = view.evaluate(rho_star, s, ye, nan_guess());
    CHECK(std::isfinite(pt.U));
    CHECK(pt.U >= 0.0);
    ++n_checked;
  }
  CHECK(n_checked == npts);
}

// ==========================================================================
// 12. M2d-2: sigma_extended() is strictly monotone increasing in u across
//     the whole extended range (finite differences, random (rho,Ye) lines).
// ==========================================================================

TEST_CASE("EntropyEOSView::sigma_extended: strictly monotone increasing in u across the extended range") {
  SyntheticOptions opts; // default grid
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();

  std::mt19937 rng(0x11235813u);
  std::uniform_real_distribution<double> xq(view.x_lo, view.x_hi);
  std::uniform_real_distribution<double> yq(view.y_lo, view.y_hi);

  const int nlines = 20;
  const int nsteps = 400;
  for (int line = 0; line < nlines; ++line) {
    const double x = xq(rng);
    const double ye = yq(rng);
    const double rho_star = std::pow(10.0, x);

    double prev = view.sigma_extended(rho_star, view.u_ext_lo, ye);
    for (int i = 1; i <= nsteps; ++i) {
      const double u =
          view.u_ext_lo + (view.u_ext_hi - view.u_ext_lo) * static_cast<double>(i) / static_cast<double>(nsteps);
      const double cur = view.sigma_extended(rho_star, u, ye);
      CHECK(cur > prev);
      prev = cur;
    }
  }
}
