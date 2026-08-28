// entropy_eos/host/version.cpp -- see version.hpp.
//
// Deliberately the only place the version macros are baked into the library
// binary, so that "which version is this .a?" has exactly one answer.

#include "entropy_eos/host/version.hpp"

#include <cstddef>

namespace eeos {

namespace {

// The classic what(1)/SCCS ident string, so that
// `strings libentropy_eos.a | grep entropy_eos-` answers "which version is
// this artifact?" for a binary whose provenance has otherwise been lost --
// which a bare "1.0.0" in the string table would not. Referenced below, so
// the linker keeps it (and no compiler reports it unused).
const char kIdent[] = "@(#)entropy_eos-" EEOS_VERSION_STRING;
constexpr std::size_t kIdentPrefixLen = sizeof("@(#)entropy_eos-") - 1;

} // namespace

int library_version() noexcept { return EEOS_VERSION; }

const char *library_version_string() noexcept { return kIdent + kIdentPrefixLen; }

} // namespace eeos
