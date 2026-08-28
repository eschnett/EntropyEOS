// entropy_eos/host/version.hpp
//
// The one part of versioning that cannot be a macro: what version the
// COMPILED library was built from, as opposed to what version of the headers
// the caller is compiling against right now. Host-only (it needs a
// translation unit); core/version.hpp has the macros and constexpr values,
// and is all a header-only or device consumer needs.
//
// This exists for consumption mode (b) -- `make install PREFIX=...` (see
// CODE.md "Environment"). There, headers and libentropy_eos.a are two
// separately installed artifacts, so they can drift: a stale .a under an
// updated include tree links without complaint and then misbehaves in
// whatever way the version difference implies. Mode (a) (copied source)
// cannot drift, and the check is free there.

#pragma once

#include "entropy_eos/core/version.hpp"

namespace eeos {

// EEOS_VERSION / EEOS_VERSION_STRING as of the compilation of the library
// itself. Compare against the macros to detect a header/library mismatch.
int library_version() noexcept;
const char *library_version_string() noexcept;

// True iff the headers being compiled now and the library being linked
// against agree. Worth asserting once at startup in mode (b):
//
//   if (!eeos::version_matches()) {
//     std::cerr << "entropy_eos: headers " << EEOS_VERSION_STRING
//               << " vs. library " << eeos::library_version_string() << "\n";
//     return EXIT_FAILURE;
//   }
//
// Inline on purpose: EEOS_VERSION is baked in at the CALLER's compilation,
// which is the whole point -- the comparison is header-side against
// library-side.
inline bool version_matches() noexcept { return library_version() == EEOS_VERSION; }

} // namespace eeos
