// tests/test_scale.hpp — sanitizer-aware scaling for the test suites.
//
// Project principle: under ASan/UBSan a test's value comes from the CODE
// PATHS it exercises, not from how many times it repeats them. The
// sanitizers already make every executed line correctness-checked (bounds,
// aliasing, UB, ...), so re-running the *same* code path thousands of extra
// times buys nothing beyond what a plain build's much cheaper repetition
// already buys -- it just burns sanitizer-slowed wall time. Accordingly,
// under a sanitized build the suites shrink sample counts, soak/state
// counts, property-test trial counts, and (for a few specific fine-grid
// convergence comparisons) grid resolutions, while every code path, every
// branch, and every numerical TOLERANCE/threshold assertion stays exactly
// as strict as in a plain build. The sole additional exception (not a
// tolerance weakening, a whole-variant skip) is the SRO real table (839 MB):
// it is skipped entirely under sanitizers via eeos_skip_big_table() below,
// while the LS220 real table keeps running (at a reduced sample count, like
// everything else) so its reader/build/solve code paths stay sanitizer-
// covered on real data.
//
// A plain (non-sanitized) build is bit-for-bit unaffected by this header:
// eeos_sanitized is false, eeos_n(full, sanitized) always returns `full`,
// and eeos_skip_big_table() always returns false.

#pragma once

#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>

// Compile-time sanitizer detection. __SANITIZE_ADDRESS__ is defined by GCC
// (and by clang when -fsanitize=address is on, alongside __has_feature);
// __has_feature(address_sanitizer)/__has_feature(undefined_behavior_sanitizer)
// cover clang's ASan and UBSan explicitly (UBSan defines no analogous
// GCC-style macro). Any one of the three being set means this test binary
// was built with SAN=1 (see the Makefile).
#if defined(__SANITIZE_ADDRESS__) || \
    (defined(__has_feature) && (__has_feature(address_sanitizer) || __has_feature(undefined_behavior_sanitizer)))
constexpr bool eeos_sanitized = true;
#else
constexpr bool eeos_sanitized = false;
#endif

// Returns `sanitized` when built with ASan/UBSan, `full` otherwise. Use this
// for every knob covered by this header's principle: random-sample counts,
// soak/state counts, property-test trial counts, and grid resolutions.
// Never use it to scale a tolerance, threshold, or other pass/fail bound.
constexpr size_t eeos_n(size_t full, size_t sanitized) { return eeos_sanitized ? sanitized : full; }

// Guards the SRO real-table variants (the 839 MB
// tables/LS220_3335_rho391_temp163_ye66.h5 file) -- the one variant this
// project skips outright under sanitizers rather than merely shrinking.
// Under a sanitized build, prints a "skipped" note naming `name` and
// returns true so the caller can bail out immediately (before even
// checking whether the file exists on disk, since the point is to avoid
// ever touching it under SAN); under a plain build always returns false, so
// the real-table path -- including the existing table_exists()-style guard
// that keeps CI green with no tables/ present -- is completely unaffected.
inline bool eeos_skip_big_table(const char *name) {
  if (eeos_sanitized) {
    std::cout << "skipped under sanitizers: " << name << "\n";
    return true;
  }
  return false;
}

// The path of the small (~20 MB) cropped LS220 fixture `make san-fixture`
// produces (tools/eos_crop.cpp; see the Makefile) -- a real-data box that
// still contains LS220's genuine Inf/NaN points in cs2/gamma and keeps the
// full temperature axis, so the tests that route through eeos_san_table()
// below keep exercising their reader/build/solve code paths, and the
// in-test repair paths, on genuine pathological real data even under SAN.
inline const char *eeos_san_ls220_crop_path() { return "tables/LS220_san_crop.h5"; }

// Routes a real-table path through the small SAN fixture when appropriate.
// Under a sanitized build, if `full_path` is the full-size LS220 table and
// the cropped fixture (tables/LS220_san_crop.h5) exists on disk, returns the
// fixture's path instead (printing a one-line note so a test log makes
// clear which table actually ran); otherwise returns `full_path` unchanged
// -- including on a plain build, where this function is a no-op by
// construction (eeos_sanitized is false), so a plain build is bit-for-bit
// unaffected by this header, per this file's header comment.
inline std::string eeos_san_table(const std::string &full_path) {
  static const std::string kLS220FullPath = "tables/LS220_234r_136t_50y_analmu_20091212_SVNr26.h5";
  if (eeos_sanitized && full_path == kLS220FullPath) {
    const std::string crop_path = eeos_san_ls220_crop_path();
    std::ifstream f(crop_path, std::ios::binary);
    if (f) {
      std::cout << "SAN: using cropped LS220 fixture; run 'make san-fixture' to regenerate\n";
      return crop_path;
    }
  }
  return full_path;
}
