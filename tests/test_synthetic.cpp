// tests/test_synthetic.cpp — unit tests for entropy_eos::make_synthetic_table
// and the underlying analytic ideal-gas model (entropy_eos/host/synthetic.hpp).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cmath>
#include <random>

#include "entropy_eos/host/synthetic.hpp"
#include "entropy_eos/host/units.hpp"
#include "test_scale.hpp"

using eeos::dirty_synthetic_options;
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

  const int n_samples = static_cast<int>(eeos_n(200, 40));
  for (int sample = 0; sample < n_samples; ++sample) {
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

// --- (e) default options are unaffected by the new defect machinery --------

TEST_CASE("synthetic table: default options (all defect vectors empty) are a no-op through the "
          "defect-application code paths") {
  SyntheticOptions defaults; // with_aux_fields=false, flatten/wiggle/offset/seed/setvalue all empty
  RawTable a = eeos::make_synthetic_table(defaults);

  // Same grid, but every defect vector cleared explicitly (redundant with
  // the defaults above -- the point is to exercise apply_flatten/
  // apply_wiggle/apply_offset/apply_setvalue with genuinely empty inputs and
  // confirm they change nothing).
  SyntheticOptions explicit_empty = defaults;
  explicit_empty.flatten.clear();
  explicit_empty.wiggle.clear();
  explicit_empty.offset.clear();
  explicit_empty.seed.clear();
  explicit_empty.setvalue.clear();
  RawTable b = eeos::make_synthetic_table(explicit_empty);

  REQUIRE(a.field_names() == b.field_names());
  for (const std::string &name : a.field_names()) {
    CHECK(a.field(name) == b.field(name)); // vector<double>::operator== is bitwise exact
  }
  CHECK_FALSE(a.has_field("cs2"));
  CHECK_FALSE(a.has_field("gamma"));
  CHECK_FALSE(a.has_field("mu_e"));
}

// --- (f) dirty_synthetic_options(): the fixed LS220/SRO-mimicking preset ---
//
// Block coordinates below are the exact ones dirty_synthetic_options() uses
// (see synthetic.cpp): wiggle on "entropy" over irho[30,35] x kYe[6,8] x
// jT[10,20]; flatten on "logenergy" over irho[5,15] x kYe[2,4] x jT[3,12];
// offset on "entropy" over irho[0,3] x kYe[0,2] x jT[0,2]; setvalue on "cs2"
// at (20,15,5) (+Inf) and (21,16,5) (NaN), and on "gamma" at (20,15,5) (NaN).

TEST_CASE("dirty_synthetic_options: default grid, aux fields present") {
  RawTable table = eeos::make_synthetic_table(dirty_synthetic_options());
  CHECK(table.nrho() == 40);
  CHECK(table.ntemp() == 30);
  CHECK(table.nye() == 10);
  CHECK(table.has_field("cs2"));
  CHECK(table.has_field("gamma"));
  CHECK(table.has_field("mu_e"));
}

TEST_CASE("dirty_synthetic_options: planted Inf/NaN land at the exact coordinates") {
  RawTable table = eeos::make_synthetic_table(dirty_synthetic_options());
  const std::vector<double> &cs2 = table.field("cs2");
  const std::vector<double> &gamma = table.field("gamma");

  const double cs2_at_inf_point = cs2[table.index(20, 15, 5)];
  CHECK(std::isinf(cs2_at_inf_point));
  CHECK(cs2_at_inf_point > 0.0); // +Inf specifically, not -Inf
  CHECK(std::isnan(cs2[table.index(21, 16, 5)]));
  CHECK(std::isnan(gamma[table.index(20, 15, 5)]));
}

TEST_CASE("dirty_synthetic_options: offset block's entropy entries are all negative") {
  RawTable table = eeos::make_synthetic_table(dirty_synthetic_options());
  const std::vector<double> &entropy = table.field("entropy");
  for (size_t kYe = 0; kYe <= 2; ++kYe) {
    for (size_t irho = 0; irho <= 3; ++irho) {
      for (size_t jT = 0; jT <= 2; ++jT) {
        CHECK(entropy[table.index(irho, jT, kYe)] < 0.0);
      }
    }
  }
}

TEST_CASE("dirty_synthetic_options: wiggle window has at least one decreasing adjacent T-pair in "
          "entropy") {
  RawTable table = eeos::make_synthetic_table(dirty_synthetic_options());
  const std::vector<double> &entropy = table.field("entropy");

  bool found_decreasing_pair = false;
  for (size_t kYe = 6; kYe <= 8; ++kYe) {
    for (size_t irho = 30; irho <= 35; ++irho) {
      for (size_t jT = 10; jT < 20; ++jT) {
        const double v0 = entropy[table.index(irho, jT, kYe)];
        const double v1 = entropy[table.index(irho, jT + 1, kYe)];
        if (v1 <= v0) {
          found_decreasing_pair = true;
        }
      }
    }
  }
  CHECK(found_decreasing_pair);
}

TEST_CASE("dirty_synthetic_options: flatten block is constant along its T-range") {
  RawTable table = eeos::make_synthetic_table(dirty_synthetic_options());
  const std::vector<double> &logenergy = table.field("logenergy");
  for (size_t kYe = 2; kYe <= 4; ++kYe) {
    for (size_t irho = 5; irho <= 15; ++irho) {
      const double v0 = logenergy[table.index(irho, 3, kYe)];
      for (size_t jT = 3; jT <= 12; ++jT) {
        CHECK(logenergy[table.index(irho, jT, kYe)] == v0);
      }
    }
  }
}

TEST_CASE("dirty_synthetic_options: no defects outside the specified blocks") {
  // Same grid and aux fields, but no defects, for a fair bit-for-bit
  // comparison against dirty_synthetic_options()'s table.
  SyntheticOptions clean_opts;
  clean_opts.with_aux_fields = true;
  RawTable clean = eeos::make_synthetic_table(clean_opts);
  RawTable dirty = eeos::make_synthetic_table(dirty_synthetic_options());

  struct Point {
    size_t irho, jT, kYe;
  };
  // Chosen to sit outside every defect block: flatten is irho[5,15] x
  // kYe[2,4] x jT[3,12]; wiggle is irho[30,35] x kYe[6,8] x jT[10,20];
  // offset is irho[0,3] x kYe[0,2] x jT[0,2]; setvalue hits exactly
  // (20,15,5)/(21,16,5) in "cs2" and (20,15,5) in "gamma". Each point below
  // fails at least one of the three range conditions for every block (a
  // block only applies where irho, kYe, *and* jT are all in range at once),
  // and none coincides with a setvalue point.
  const std::vector<Point> far_points = {
      {25, 25, 0}, {39, 0, 9}, {0, 29, 9}, {20, 0, 0}, {36, 20, 3}, {5, 20, 9},
  };

  const std::vector<std::string> field_names = {"entropy",   "logenergy", "logpress",
                                                 "cs2",       "gamma",     "mu_e"};
  for (const std::string &name : field_names) {
    const std::vector<double> &clean_field = clean.field(name);
    const std::vector<double> &dirty_field = dirty.field(name);
    for (const Point &pt : far_points) {
      const size_t idx = clean.index(pt.irho, pt.jT, pt.kYe);
      CHECK(clean_field[idx] == dirty_field[idx]);
    }
  }
}
