// tests/test_synthetic.cpp — unit tests for entropy_eos::make_synthetic_table
// and the underlying analytic ideal-gas model (entropy_eos/host/synthetic.hpp).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cmath>
#include <random>

#include "entropy_eos/host/synthetic.hpp"
#include "entropy_eos/host/units.hpp"

using eeos::RawTable;
using eeos::SeededViolation;
using eeos::SyntheticOptions;
using eeos::synthetic_eps;
using eeos::synthetic_p;
using eeos::synthetic_s;

namespace {
constexpr double kLn10 = 2.302585092994045684017991454684364207601101488628772976033;
}

// --- (a) table values equal the analytic functions exactly -----------------

TEST_CASE("synthetic table: stored fields equal the analytic functions exactly") {
  SyntheticOptions opts; // defaults per CODE.md
  RawTable table = eeos::make_synthetic_table(opts);

  std::mt19937 rng(12345);
  std::uniform_int_distribution<size_t> pick_rho(0, table.nrho() - 1);
  std::uniform_int_distribution<size_t> pick_temp(0, table.ntemp() - 1);
  std::uniform_int_distribution<size_t> pick_ye(0, table.nye() - 1);

  const std::vector<double> &logenergy = table.field("logenergy");
  const std::vector<double> &entropy = table.field("entropy");
  const std::vector<double> &logpress = table.field("logpress");

  for (int sample = 0; sample < 200; ++sample) {
    const size_t irho = pick_rho(rng);
    const size_t jT = pick_temp(rng);
    const size_t kYe = pick_ye(rng);
    const double rho = table.rho(irho);
    const double temp = table.temp(jT);
    const double ye = table.yev(kYe);
    const size_t idx = table.index(irho, jT, kYe);

    const double eps = synthetic_eps(rho, temp, ye, opts);
    const double p = synthetic_p(rho, temp, ye, opts);
    const double s = synthetic_s(rho, temp, ye, opts);

    CHECK(logenergy[idx] == std::log10(eps + opts.energy_shift));
    CHECK(entropy[idx] == s);
    CHECK(logpress[idx] == std::log10(p));
  }
}

// --- (b) positivity and strict monotonicity in T ----------------------------

TEST_CASE("synthetic table: entropy positive; entropy and logenergy strictly increasing in T") {
  SyntheticOptions opts;
  RawTable table = eeos::make_synthetic_table(opts);

  const std::vector<double> &logenergy = table.field("logenergy");
  const std::vector<double> &entropy = table.field("entropy");

  for (size_t kYe = 0; kYe < table.nye(); ++kYe) {
    for (size_t irho = 0; irho < table.nrho(); ++irho) {
      double prev_s = 0.0;
      double prev_loge = 0.0;
      for (size_t jT = 0; jT < table.ntemp(); ++jT) {
        const size_t idx = table.index(irho, jT, kYe);
        CHECK(entropy[idx] > 0.0);
        if (jT > 0) {
          CHECK(entropy[idx] > prev_s);
          CHECK(logenergy[idx] > prev_loge);
        }
        prev_s = entropy[idx];
        prev_loge = logenergy[idx];
      }
    }
  }
}

// --- (c) finite-difference thermodynamic consistency on a finer grid -------

namespace {

// Central difference in the (uniform) log10 storage coordinate, converted to
// a physical d/dT or d/drho via the chain rule given in CODE.md's test spec:
// d/dT = 1/(T ln10) d/dlog10T, and d/drho analogously.
double d_dlog10(const std::vector<double> &f, size_t idx_minus, size_t idx_plus, double dlog10) {
  return (f[idx_plus] - f[idx_minus]) / (2.0 * dlog10);
}

double d_dlog10(double f_minus, double f_plus, double dlog10) {
  return (f_plus - f_minus) / (2.0 * dlog10);
}

} // namespace

TEST_CASE("synthetic table: FD Maxwell/consistency identities on a finer grid") {
  SyntheticOptions opts;
  opts.ntemp = 120;
  opts.nrho = 120;
  RawTable table = eeos::make_synthetic_table(opts);

  const std::vector<double> &logenergy = table.field("logenergy");
  const std::vector<double> &entropy = table.field("entropy");
  const std::vector<double> &logpress = table.field("logpress");

  const double dlog10T =
      (table.logtemp().back() - table.logtemp().front()) / static_cast<double>(table.ntemp() - 1);
  const double dlog10Rho =
      (table.logrho().back() - table.logrho().front()) / static_cast<double>(table.nrho() - 1);

  const double energy_shift = opts.energy_shift;
  const double rel_tol = 1e-3;

  // Sample a grid of interior points (avoid the boundary rows/columns, which
  // central differences cannot straddle) rather than every node, to keep the
  // test fast; the model has no structure that would hide a bug between
  // sampled points.
  for (size_t kYe = 0; kYe < table.nye(); kYe += 3) {
    for (size_t jT = 5; jT + 5 < table.ntemp(); jT += 11) {
      const double temp_MeV = table.temp(jT);
      for (size_t irho = 5; irho + 5 < table.nrho(); irho += 11) {
        const double rho_gcc = table.rho(irho);

        const size_t idx = table.index(irho, jT, kYe);
        const size_t idx_Tm = table.index(irho, jT - 1, kYe);
        const size_t idx_Tp = table.index(irho, jT + 1, kYe);
        const size_t idx_Rm = table.index(irho - 1, jT, kYe);
        const size_t idx_Rp = table.index(irho + 1, jT, kYe);

        // Stored representation -> physical values at each node first
        // ("exercise the stored representation, not the analytic
        // shortcuts"), then finite-difference the resulting physical values.
        auto eps_of = [&](size_t i) { return std::pow(10.0, logenergy[i]) - energy_shift; };
        auto p_of = [&](size_t i) { return std::pow(10.0, logpress[i]); };

        const double p = p_of(idx);

        const double dEps_dlog10T = d_dlog10(eps_of(idx_Tm), eps_of(idx_Tp), dlog10T);
        const double dEps_dT = dEps_dlog10T / (temp_MeV * kLn10);

        const double dS_dlog10T = d_dlog10(entropy, idx_Tm, idx_Tp, dlog10T);
        const double dS_dT = dS_dlog10T / (temp_MeV * kLn10);

        const double dP_dlog10T = d_dlog10(p_of(idx_Tm), p_of(idx_Tp), dlog10T);
        const double dP_dT = dP_dlog10T / (temp_MeV * kLn10);

        const double dEps_dlog10Rho = d_dlog10(eps_of(idx_Rm), eps_of(idx_Rp), dlog10Rho);
        const double dEps_dRho = dEps_dlog10Rho / (rho_gcc * kLn10);

        const double dS_dlog10Rho = d_dlog10(entropy, idx_Rm, idx_Rp, dlog10Rho);
        const double dS_dRho = dS_dlog10Rho / (rho_gcc * kLn10);

        // 1) dEps/dT ~= (kT_erg/m_B) * dS/dT
        const double kT_erg = temp_MeV * eeos::MeV_to_erg;
        const double rhs1 = (kT_erg / eeos::m_amu_g) * dS_dT;
        CHECK(dEps_dT == doctest::Approx(rhs1).epsilon(rel_tol));

        // 2) |rho^2 dEps/dRho - (p - T dP/dT)| <= 1e-3 p
        const double lhs2 = rho_gcc * rho_gcc * dEps_dRho;
        const double rhs2 = p - temp_MeV * dP_dT;
        CHECK(std::abs(lhs2 - rhs2) <= rel_tol * p);

        // 3) dS/dRho ~= -(dP/dT) * m_B / (rho^2 * MeV_to_erg)
        const double rhs3 = -(dP_dT)*eeos::m_amu_g / (rho_gcc * rho_gcc * eeos::MeV_to_erg);
        CHECK(dS_dRho == doctest::Approx(rhs3).epsilon(rel_tol));
      }
    }
  }
}

// --- (d) seeded violations land exactly where requested, and only there ----

TEST_CASE("synthetic table: seeded violation appears exactly at the requested node, nowhere else") {
  SyntheticOptions opts;
  opts.nrho = 12;
  opts.ntemp = 10;
  opts.nye = 4;

  RawTable clean = eeos::make_synthetic_table(opts);

  const size_t irho = 5, jT = 3, kYe = 2;
  const double delta = 7.5;
  opts.seed = {SeededViolation{"entropy", irho, jT, kYe, delta}};
  RawTable seeded = eeos::make_synthetic_table(opts);

  const std::vector<double> &clean_s = clean.field("entropy");
  const std::vector<double> &seeded_s = seeded.field("entropy");
  const size_t hit = seeded.index(irho, jT, kYe);

  REQUIRE(clean_s.size() == seeded_s.size());
  for (size_t i = 0; i < clean_s.size(); ++i) {
    if (i == hit) {
      CHECK(seeded_s[i] == clean_s[i] + delta);
    } else {
      CHECK(seeded_s[i] == clean_s[i]);
    }
  }

  // Untouched fields must be completely unaffected.
  CHECK(clean.field("logenergy") == seeded.field("logenergy"));
  CHECK(clean.field("logpress") == seeded.field("logpress"));
}

TEST_CASE("synthetic table: multiple seeded violations are independent and localized") {
  SyntheticOptions opts;
  opts.nrho = 10;
  opts.ntemp = 8;
  opts.nye = 3;

  RawTable clean = eeos::make_synthetic_table(opts);

  opts.seed = {
      SeededViolation{"entropy", 1, 1, 0, 3.0},
      SeededViolation{"logenergy", 4, 5, 2, -0.25},
  };
  RawTable seeded = eeos::make_synthetic_table(opts);

  const size_t hit_s = seeded.index(1, 1, 0);
  const size_t hit_e = seeded.index(4, 5, 2);

  const std::vector<double> &clean_s = clean.field("entropy");
  const std::vector<double> &seeded_s = seeded.field("entropy");
  for (size_t i = 0; i < clean_s.size(); ++i) {
    CHECK(seeded_s[i] == clean_s[i] + (i == hit_s ? 3.0 : 0.0));
  }

  const std::vector<double> &clean_e = clean.field("logenergy");
  const std::vector<double> &seeded_e = seeded.field("logenergy");
  for (size_t i = 0; i < clean_e.size(); ++i) {
    CHECK(seeded_e[i] == clean_e[i] + (i == hit_e ? -0.25 : 0.0));
  }

  CHECK(clean.field("logpress") == seeded.field("logpress"));
}
