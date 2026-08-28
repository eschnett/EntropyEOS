// entropy_eos/core/version.hpp
//
// The library's version, and nothing else. It lives in core/ rather than
// host/ because the version has to be visible to every consumer, including
// one that embeds only core/ (no HDF5, no .cpp files, nvcc) -- so this header
// is macros and constexpr integers only: no STL, no allocation, no
// exceptions, nothing to link (see CODE.md "Layout").
//
// Versioning contract, release checklist, and what a bump means for a
// numerics library: CODE.md "Versioning".

#pragma once

// The single place a release is recorded. Everything below is derived from
// these three numbers, so a bump means editing exactly this block.
#define EEOS_VERSION_MAJOR 1
#define EEOS_VERSION_MINOR 0
#define EEOS_VERSION_PATCH 0

// Encoded as major*10000 + minor*100 + patch, so that a consumer can compile
// conditionally against a version range with plain preprocessor arithmetic --
// no configure step, no CMake package version file:
//
//   #if EEOS_VERSION >= EEOS_VERSION_ENCODE(1, 1, 0)
//     ... use something that only exists from 1.1.0 on ...
//   #endif
//
// The two-digit fields cap minor and patch at 99; if either ever gets there,
// widen the multipliers (a monotone re-encoding, since comparisons are the
// only thing anyone may do with EEOS_VERSION).
#define EEOS_VERSION_ENCODE(major, minor, patch) ((major) * 10000 + (minor) * 100 + (patch))
#define EEOS_VERSION EEOS_VERSION_ENCODE(EEOS_VERSION_MAJOR, EEOS_VERSION_MINOR, EEOS_VERSION_PATCH)

// "1.0.0", built by stringifying the numbers above rather than spelled out a
// second time -- a hand-written literal here is exactly the kind of thing
// that silently fails to get bumped.
//
// The two helpers must stay defined (no #undef below): EEOS_VERSION_STRING
// expands at its point of use, not here, so it needs them then. They are
// EEOS_-prefixed for exactly that reason.
#define EEOS_STRINGIFY_(x) #x
#define EEOS_STRINGIFY(x) EEOS_STRINGIFY_(x)
#define EEOS_VERSION_STRING                                                                        \
  EEOS_STRINGIFY(EEOS_VERSION_MAJOR)                                                               \
  "." EEOS_STRINGIFY(EEOS_VERSION_MINOR) "." EEOS_STRINGIFY(EEOS_VERSION_PATCH)

namespace eeos {

// The same values as the macros, for code that would rather not use the
// preprocessor (`static_assert(eeos::version >= 10000)`, a version printed
// from device code, a template parameter). constexpr, so free at run time.
constexpr int version_major = EEOS_VERSION_MAJOR;
constexpr int version_minor = EEOS_VERSION_MINOR;
constexpr int version_patch = EEOS_VERSION_PATCH;
constexpr int version = EEOS_VERSION;
constexpr const char *version_string = EEOS_VERSION_STRING;

} // namespace eeos
