// tests/test_version.cpp -- unit tests for entropy_eos/core/version.hpp and
// entropy_eos/host/version.{hpp,cpp}: that the derived forms (the encoded
// integer, the string, the constexpr values) agree with the three numbers
// they are derived from, and that the compiled library agrees with the
// headers -- see CODE.md "Versioning".
//
// Deliberately free of any pinned CURRENT version: a release bump must not
// have to touch this file, or it stops being a check and becomes a second
// place to forget. Everything here is a relation between the mechanism's
// outputs; the one absolute is a >= 1.0.0 floor, which no future bump can
// falsify.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <string>

#include "entropy_eos/core/version.hpp"
#include "entropy_eos/host/version.hpp"

TEST_CASE("version: the encoded integer round-trips the three components") {
  CHECK(EEOS_VERSION / 10000 == EEOS_VERSION_MAJOR);
  CHECK(EEOS_VERSION / 100 % 100 == EEOS_VERSION_MINOR);
  CHECK(EEOS_VERSION % 100 == EEOS_VERSION_PATCH);

  // The comparison macro is the documented consumer-facing use, so check it
  // orders as advertised around the current version.
  CHECK(EEOS_VERSION >= EEOS_VERSION_ENCODE(1, 0, 0));
  CHECK(EEOS_VERSION >= EEOS_VERSION_ENCODE(EEOS_VERSION_MAJOR, EEOS_VERSION_MINOR,
                                            EEOS_VERSION_PATCH));
  CHECK(EEOS_VERSION < EEOS_VERSION_ENCODE(EEOS_VERSION_MAJOR, EEOS_VERSION_MINOR,
                                           EEOS_VERSION_PATCH + 1));
  CHECK(EEOS_VERSION < EEOS_VERSION_ENCODE(EEOS_VERSION_MAJOR + 1, 0, 0));

  // The two-digit encoding fields; see version.hpp on widening them.
  CHECK(EEOS_VERSION_MINOR < 100);
  CHECK(EEOS_VERSION_PATCH < 100);
}

TEST_CASE("version: the string is major.minor.patch") {
  const std::string expected = std::to_string(EEOS_VERSION_MAJOR) + "." +
                               std::to_string(EEOS_VERSION_MINOR) + "." +
                               std::to_string(EEOS_VERSION_PATCH);
  CHECK(std::string(EEOS_VERSION_STRING) == expected);
}

TEST_CASE("version: constexpr values mirror the macros and are usable at compile time") {
  static_assert(eeos::version == EEOS_VERSION, "eeos::version out of sync with EEOS_VERSION");
  static_assert(eeos::version_major == EEOS_VERSION_MAJOR, "major out of sync");
  static_assert(eeos::version_minor == EEOS_VERSION_MINOR, "minor out of sync");
  static_assert(eeos::version_patch == EEOS_VERSION_PATCH, "patch out of sync");
  CHECK(std::string(eeos::version_string) == EEOS_VERSION_STRING);
}

TEST_CASE("version: the compiled library matches these headers") {
  // In-tree this is tautological (one source tree, one compile). It earns its
  // keep against an installed PREFIX, where a stale libentropy_eos.a can sit
  // under updated headers -- which is exactly what version_matches() is for.
  CHECK(eeos::library_version() == EEOS_VERSION);
  CHECK(std::string(eeos::library_version_string()) == EEOS_VERSION_STRING);
  CHECK(eeos::version_matches());
}
