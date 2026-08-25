// tests/test_table.cpp — unit tests for entropy_eos::RawTable.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "entropy_eos/host/table.hpp"

using eeos::RawTable;

namespace {

// A small table: nrho=3, ntemp=2, nye=2, with a field whose value encodes
// its own (irho, jT, kYe) so verbatim storage and index layout are easy to
// check independently.
RawTable make_small_table() {
  RawTable t;
  t.set_axes(/*logrho=*/{5.0, 6.0, 7.0}, /*logtemp=*/{-1.0, 0.0}, /*ye=*/{0.1, 0.5});

  const size_t nrho = t.nrho();
  const size_t ntemp = t.ntemp();
  const size_t nye = t.nye();
  std::vector<double> data(nrho * ntemp * nye);
  for (size_t kYe = 0; kYe < nye; ++kYe) {
    for (size_t jT = 0; jT < ntemp; ++jT) {
      for (size_t irho = 0; irho < nrho; ++irho) {
        // Encode (irho, jT, kYe) into a single double, distinct per index.
        data[t.index(irho, jT, kYe)] =
            static_cast<double>(irho) + 100.0 * static_cast<double>(jT) +
            10000.0 * static_cast<double>(kYe);
      }
    }
  }
  t.add_field("probe", data);
  return t;
}

} // namespace

TEST_CASE("construction: default table has zero-sized axes") {
  RawTable t;
  CHECK(t.nrho() == 0);
  CHECK(t.ntemp() == 0);
  CHECK(t.nye() == 0);
  CHECK(t.field_names().empty());
  CHECK(t.attribute_names().empty());
}

TEST_CASE("construction: set_axes fixes nrho/ntemp/nye") {
  RawTable t = make_small_table();
  CHECK(t.nrho() == 3);
  CHECK(t.ntemp() == 2);
  CHECK(t.nye() == 2);
  CHECK(t.logrho().size() == 3);
  CHECK(t.logtemp().size() == 2);
  CHECK(t.ye().size() == 2);
}

TEST_CASE("index(): iRho is the fastest-varying axis") {
  RawTable t = make_small_table();
  // Fixing jT, kYe and stepping irho must give consecutive flat indices.
  CHECK(t.index(0, 0, 0) == 0);
  CHECK(t.index(1, 0, 0) == 1);
  CHECK(t.index(2, 0, 0) == 2);
  // Next jT starts right after a full rho row.
  CHECK(t.index(0, 1, 0) == t.nrho());
  // Next kYe starts right after a full (rho, T) plane.
  CHECK(t.index(0, 0, 1) == t.nrho() * t.ntemp());
}

TEST_CASE("verbatim storage: values read back bit-identical") {
  RawTable t = make_small_table();
  const std::vector<double> &probe = t.field("probe");
  for (size_t kYe = 0; kYe < t.nye(); ++kYe) {
    for (size_t jT = 0; jT < t.ntemp(); ++jT) {
      for (size_t irho = 0; irho < t.nrho(); ++irho) {
        const double expected = static_cast<double>(irho) + 100.0 * static_cast<double>(jT) +
                                 10000.0 * static_cast<double>(kYe);
        // Bit-identical, not just "close": no unit conversion may have
        // touched this value (CODE.md "Data model" verbatim requirement).
        CHECK(probe[t.index(irho, jT, kYe)] == expected);
      }
    }
  }

  // Also check a value with fractional bits round-trips exactly.
  RawTable t2;
  t2.set_axes({0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0});
  const double odd_bits = 1.0 / 3.0; // not exactly representable, but must
                                      // still be stored/retrieved bit-for-bit
  t2.add_field("f", std::vector<double>(8, odd_bits));
  CHECK(t2.field("f")[0] == odd_bits);
}

TEST_CASE("add_field: wrong-sized data throws std::invalid_argument") {
  RawTable t;
  t.set_axes({0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0});
  CHECK_THROWS_AS(t.add_field("bad", std::vector<double>(3)), std::invalid_argument);
}

TEST_CASE("physical accessors: rho/temp convert from log10 storage") {
  RawTable t = make_small_table();
  CHECK(t.rho(0) == doctest::Approx(1e5));
  CHECK(t.rho(1) == doctest::Approx(1e6));
  CHECK(t.rho(2) == doctest::Approx(1e7));
  CHECK(t.temp(0) == doctest::Approx(0.1));
  CHECK(t.temp(1) == doctest::Approx(1.0));
  CHECK(t.yev(0) == doctest::Approx(0.1));
  CHECK(t.yev(1) == doctest::Approx(0.5));
}

TEST_CASE("field_names(): insertion order preserved") {
  RawTable t;
  t.set_axes({0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0});
  t.add_field("zeta", std::vector<double>(8, 0.0));
  t.add_field("alpha", std::vector<double>(8, 0.0));
  t.add_field("middle", std::vector<double>(8, 0.0));
  const std::vector<std::string> &names = t.field_names();
  REQUIRE(names.size() == 3);
  CHECK(names[0] == "zeta");
  CHECK(names[1] == "alpha");
  CHECK(names[2] == "middle");

  // Re-adding an existing name overwrites in place, not appends.
  t.add_field("alpha", std::vector<double>(8, 5.0));
  CHECK(t.field_names().size() == 3);
  CHECK(t.field_names()[1] == "alpha");
  CHECK(t.field("alpha")[0] == 5.0);
}

TEST_CASE("field(): missing field throws std::out_of_range") {
  RawTable t = make_small_table();
  CHECK(t.has_field("probe"));
  CHECK_FALSE(t.has_field("nonexistent"));
  CHECK_THROWS_AS(t.field("nonexistent"), std::out_of_range);

  const RawTable &ct = t;
  CHECK_THROWS_AS(ct.field("nonexistent"), std::out_of_range);
}

TEST_CASE("attributes: insertion order, has/get, missing throws") {
  RawTable t;
  t.add_attribute("energy_shift", 1.5e18);
  t.add_attribute("aux", 42.0);
  REQUIRE(t.attribute_names().size() == 2);
  CHECK(t.attribute_names()[0] == "energy_shift");
  CHECK(t.attribute_names()[1] == "aux");
  CHECK(t.has_attribute("energy_shift"));
  CHECK_FALSE(t.has_attribute("nope"));
  CHECK(t.attribute("aux") == 42.0);
  CHECK(t.energy_shift() == 1.5e18);
  CHECK_THROWS_AS(t.attribute("nope"), std::out_of_range);

  RawTable t2;
  CHECK_THROWS_AS(t2.energy_shift(), std::out_of_range);
}

TEST_CASE("validate_axes: strictly increasing, finite axes pass") {
  RawTable t = make_small_table();
  CHECK_NOTHROW(t.validate_axes());
}

TEST_CASE("validate_axes: non-monotone axis throws") {
  RawTable t;
  t.set_axes(/*logrho=*/{5.0, 7.0, 6.0}, /*logtemp=*/{0.0, 1.0}, /*ye=*/{0.1, 0.5});
  CHECK_THROWS_AS(t.validate_axes(), std::runtime_error);
}

TEST_CASE("validate_axes: non-strictly-increasing (repeated value) axis throws") {
  RawTable t;
  t.set_axes(/*logrho=*/{5.0, 6.0, 6.0}, /*logtemp=*/{0.0, 1.0}, /*ye=*/{0.1, 0.5});
  CHECK_THROWS_AS(t.validate_axes(), std::runtime_error);
}

TEST_CASE("validate_axes: NaN axis throws") {
  RawTable t;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  t.set_axes(/*logrho=*/{5.0, nan, 7.0}, /*logtemp=*/{0.0, 1.0}, /*ye=*/{0.1, 0.5});
  CHECK_THROWS_AS(t.validate_axes(), std::runtime_error);
}

TEST_CASE("validate_axes: infinite axis throws") {
  RawTable t;
  const double inf = std::numeric_limits<double>::infinity();
  t.set_axes(/*logrho=*/{5.0, 6.0, 7.0}, /*logtemp=*/{0.0, inf}, /*ye=*/{0.1, 0.5});
  CHECK_THROWS_AS(t.validate_axes(), std::runtime_error);
}
