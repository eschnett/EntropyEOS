// tests/test_check.cpp — unit tests for entropy_eos::check_table
// (entropy_eos/host/check.hpp), exercised against the synthetic ground-truth
// table (entropy_eos/host/synthetic.hpp) per CODE.md "Test harness".

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "entropy_eos/host/check.hpp"
#include "entropy_eos/host/synthetic.hpp"
#include "entropy_eos/host/table.hpp"
#include "entropy_eos/host/units.hpp"

using eeos::CheckClassResult;
using eeos::CheckOptions;
using eeos::CheckReport;
using eeos::RawTable;
using eeos::SeededViolation;
using eeos::Status;
using eeos::SyntheticOptions;

namespace {

const CheckClassResult *find_class(const CheckReport &report, const std::string &name) {
  for (const CheckClassResult &c : report.classes) {
    if (c.name == name) return &c;
  }
  return nullptr;
}

} // namespace

// --- 1. Clean synthetic table: ok, zero violations --------------------------

TEST_CASE("check_table: clean synthetic table is ok with zero range/monotonicity violations") {
  RawTable table = eeos::make_synthetic_table();
  CheckReport report = eeos::check_table(table);

  CHECK(report.status == Status::ok);
  CHECK(report.fatal_messages.empty());

  for (const char *name : {"entropy_negative", "entropy_nonmonotone_T", "logenergy_nonmonotone_T"}) {
    const CheckClassResult *c = find_class(report, name);
    REQUIRE(c != nullptr);
    CHECK(c->count == 0);
    CHECK(c->worst.empty());
  }
}

// --- 2. Consistency metrics: FD truncation only, shrinks with resolution ----

TEST_CASE("check_table: Maxwell consistency metrics are FD-truncation-small on both grids") {
  SUBCASE("fine grid (120 x 120 x 5): max residual below 2e-3") {
    SyntheticOptions opts;
    opts.nrho = 120;
    opts.ntemp = 120;
    opts.nye = 5;
    RawTable table = eeos::make_synthetic_table(opts);
    CheckReport report = eeos::check_table(table);

    for (const char *name : {"delta_T", "delta_p", "maxwell_s_rho"}) {
      const CheckClassResult *c = find_class(report, name);
      REQUIRE(c != nullptr);
      CHECK(c->max < 2e-3);
    }
  }

  SUBCASE("default grid (40 x 30 x 10): max residual below 3e-2") {
    RawTable table = eeos::make_synthetic_table(); // defaults
    CheckReport report = eeos::check_table(table);

    for (const char *name : {"delta_T", "delta_p", "maxwell_s_rho"}) {
      const CheckClassResult *c = find_class(report, name);
      REQUIRE(c != nullptr);
      CHECK(c->max < 3e-2);
    }
  }
}

// --- 3. Seeded violations are found at the right location --------------------

TEST_CASE("check_table: seeded entropy/logenergy violations are found at the right column") {
  SyntheticOptions opts;
  opts.nrho = 12;
  opts.ntemp = 10;
  opts.nye = 4;

  const size_t irho_s = 5, jT_s = 5, kYe_s = 2;
  const size_t irho_e = 3, jT_e = 6, kYe_e = 1;
  // Large negative deltas guarantee the seeded point drops below its
  // predecessor in T regardless of the model's natural step size.
  opts.seed = {
      SeededViolation{"entropy", irho_s, jT_s, kYe_s, -1000.0},
      SeededViolation{"logenergy", irho_e, jT_e, kYe_e, -1000.0},
  };
  RawTable table = eeos::make_synthetic_table(opts);
  CheckReport report = eeos::check_table(table);

  // The seeds are data violations, not structural problems.
  CHECK(report.status == Status::ok);
  CHECK(report.fatal_messages.empty());

  const CheckClassResult *cs = find_class(report, "entropy_nonmonotone_T");
  REQUIRE(cs != nullptr);
  CHECK(cs->count >= 1);
  {
    bool found = false;
    for (const CheckClassResult::Loc &loc : cs->worst) {
      if (loc.irho == irho_s && loc.kYe == kYe_s && (loc.jT == jT_s - 1 || loc.jT == jT_s)) {
        found = true;
        CHECK(loc.value < 0.0);
        CHECK(loc.rho == doctest::Approx(table.rho(irho_s)));
        CHECK(loc.ye == doctest::Approx(table.yev(kYe_s)));
      }
    }
    CHECK(found);
  }

  const CheckClassResult *ce = find_class(report, "logenergy_nonmonotone_T");
  REQUIRE(ce != nullptr);
  CHECK(ce->count >= 1);
  {
    bool found = false;
    for (const CheckClassResult::Loc &loc : ce->worst) {
      if (loc.irho == irho_e && loc.kYe == kYe_e && (loc.jT == jT_e - 1 || loc.jT == jT_e)) {
        found = true;
        CHECK(loc.value < 0.0);
        CHECK(loc.rho == doctest::Approx(table.rho(irho_e)));
        CHECK(loc.ye == doctest::Approx(table.yev(kYe_e)));
      }
    }
    CHECK(found);
  }
}

// --- 4. Structural failures ---------------------------------------------------

TEST_CASE("check_table: NaN injected into a field is fatal and names the field") {
  RawTable table = eeos::make_synthetic_table();
  table.field("entropy")[table.index(2, 2, 2)] = std::numeric_limits<double>::quiet_NaN();

  CheckReport report = eeos::check_table(table);
  CHECK(report.status == Status::fatal);
  REQUIRE(!report.fatal_messages.empty());

  bool names_entropy = false;
  for (const std::string &msg : report.fatal_messages) {
    if (msg.find("entropy") != std::string::npos) names_entropy = true;
  }
  CHECK(names_entropy);

  // Non-structural classes are skipped once a structural check fails.
  CHECK(report.classes.empty());
}

TEST_CASE("check_table: non-finite values in a non-interpreted field are reported, not fatal") {
  // Policy (CODE.md "Repair harness"): finiteness is fatal only for the
  // fields the pipeline interprets (logenergy, entropy). The shipped LS220
  // table carries Inf in cs2/gamma, and "logpress" is diagnostic-only, so an
  // Inf there must yield a "nonfinite_logpress" violation class, with the
  // Maxwell diagnostics still computed (poisoned stencil points skipped).
  RawTable table = eeos::make_synthetic_table();
  table.field("logpress")[table.index(3, 4, 5)] = std::numeric_limits<double>::infinity();

  CheckReport report = eeos::check_table(table);
  CHECK(report.status == Status::ok);
  CHECK(report.fatal_messages.empty());

  const CheckClassResult *nonfinite = nullptr;
  const CheckClassResult *delta_T = nullptr;
  for (const CheckClassResult &c : report.classes) {
    if (c.name == "nonfinite_logpress") nonfinite = &c;
    if (c.name == "delta_T") delta_T = &c;
  }
  REQUIRE(nonfinite != nullptr);
  CHECK(nonfinite->count == 1);
  REQUIRE(!nonfinite->worst.empty());
  CHECK(nonfinite->worst[0].irho == 3);
  CHECK(nonfinite->worst[0].jT == 4);
  CHECK(nonfinite->worst[0].kYe == 5);

  // Diagnostics still ran and stayed finite despite the poisoned point.
  REQUIRE(delta_T != nullptr);
  CHECK(std::isfinite(delta_T->max));
  CHECK(std::isfinite(delta_T->rms));
}

TEST_CASE("check_table: missing required field 'entropy' is fatal") {
  RawTable table;
  table.set_axes({5.0, 6.0, 7.0}, {-1.0, 0.0, 1.0}, {0.1, 0.3, 0.5});
  const size_t n = table.nrho() * table.ntemp() * table.nye();
  table.add_field("logenergy", std::vector<double>(n, 18.0));
  table.add_attribute("energy_shift", 1.5e18);
  // Deliberately no "entropy" field.

  CheckReport report = eeos::check_table(table);
  CHECK(report.status == Status::fatal);

  bool names_entropy = false;
  for (const std::string &msg : report.fatal_messages) {
    if (msg.find("entropy") != std::string::npos) names_entropy = true;
  }
  CHECK(names_entropy);
}

// --- 5. cs2 path: FD comparison and out-of-range counting ---------------------

TEST_CASE("check_table: cs2 diagnostic against FD, and out-of-range detection") {
  SyntheticOptions opts;
  opts.nrho = 120;
  opts.ntemp = 120;
  opts.nye = 5;
  RawTable table = eeos::make_synthetic_table(opts);

  // Default CheckOptions uses m_B_g = m_B_default_g, matching the m_B the
  // synthetic model itself is built on (see synthetic.cpp), so the analytic
  // cs2 constructed here and the FD-derived one check_table computes are
  // directly comparable.
  const CheckOptions check_opts;
  const double m_B = eeos::m_B_default_g;
  const double c = eeos::c_light_cm_s;

  // Exact ideal-gas result for this model (Gamma = 5/3): cs2 = (5/3) * g *
  // kT_erg / (m_B * h * c^2), g = 1 + Ye.
  std::vector<double> cs2(table.nrho() * table.ntemp() * table.nye());
  for (size_t kYe = 0; kYe < table.nye(); ++kYe) {
    const double ye = table.yev(kYe);
    const double g = 1.0 + ye;
    for (size_t jT = 0; jT < table.ntemp(); ++jT) {
      const double temp = table.temp(jT);
      const double kT_erg = temp * eeos::MeV_to_erg;
      for (size_t irho = 0; irho < table.nrho(); ++irho) {
        const double rho = table.rho(irho);
        const double eps = eeos::synthetic_eps(rho, temp, ye, opts);
        const double p = eeos::synthetic_p(rho, temp, ye, opts);
        const double h = 1.0 + (eps + p / rho) / (c * c);
        cs2[table.index(irho, jT, kYe)] = (5.0 / 3.0) * g * kT_erg / (m_B * h * c * c);
      }
    }
  }
  table.add_field("cs2", cs2);

  CheckReport report = eeos::check_table(table, check_opts);

  const CheckClassResult *vs_fd = find_class(report, "cs2_vs_fd");
  REQUIRE(vs_fd != nullptr);
  CHECK(vs_fd->max < 3e-2);

  const CheckClassResult *oor = find_class(report, "cs2_out_of_range");
  REQUIRE(oor != nullptr);
  CHECK(oor->count == 0);

  // Corrupt a single value to something clearly outside (0,1) and confirm it
  // alone is what gets counted.
  const size_t bad_idx = table.index(10, 10, 2);
  table.field("cs2")[bad_idx] = 1.5;
  CheckReport report2 = eeos::check_table(table, check_opts);
  const CheckClassResult *oor2 = find_class(report2, "cs2_out_of_range");
  REQUIRE(oor2 != nullptr);
  CHECK(oor2->count == 1);
  REQUIRE(oor2->worst.size() >= 1);
  CHECK(oor2->worst.front().irho == 10);
  CHECK(oor2->worst.front().jT == 10);
  CHECK(oor2->worst.front().kYe == 2);
}

// --- 6. print() smoke test -----------------------------------------------------

TEST_CASE("check_table: print() does not throw and mentions every reported class name") {
  RawTable table = eeos::make_synthetic_table();
  CheckReport report = eeos::check_table(table);

  std::ostringstream oss;
  CHECK_NOTHROW(report.print(oss));
  const std::string text = oss.str();
  REQUIRE_FALSE(report.classes.empty());
  for (const CheckClassResult &c : report.classes) {
    CHECK(text.find(c.name) != std::string::npos);
  }
}

TEST_CASE("check_table: print() on a fatal report does not throw") {
  RawTable table;
  table.set_axes({5.0, 6.0}, {-1.0, 0.0}, {0.1, 0.5});
  // No fields, no attributes at all: several structural failures at once.
  CheckReport report = eeos::check_table(table);
  CHECK(report.status == Status::fatal);

  std::ostringstream oss;
  CHECK_NOTHROW(report.print(oss));
  CHECK(oss.str().find("fatal") != std::string::npos);
}
