// entropy_eos/host/check.hpp
//
// Table-level diagnostics: `check_table()` audits a RawTable against the
// hard/soft requirements of `eos-adapter-F-to-U.md` §8 (see CODE.md "Test
// harness"). It is a pure library function -- no I/O, no side effects -- so
// it can run in-process right after a table is loaded; `tools/eos_test` and
// `tools/eos_repair --check-only` are thin wrappers over it. Host-only: STL
// throughout, may throw (but check_table itself never propagates an
// exception from the table it is given -- see the "Structural" checks in
// check.cpp).

#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#include "entropy_eos/core/defs.hpp"
#include "entropy_eos/host/table.hpp"
#include "entropy_eos/host/units.hpp"

namespace eeos {

// Tunables for check_table(). Defaults match CODE.md's test-harness spec.
struct CheckOptions {
  // Baryon mass used to convert between per-baryon and per-gram quantities
  // in the unit-bearing consistency checks (class D). See units.hpp and
  // CODE.md "Open decisions" #2: callers with a table-specific m_B
  // convention should override this rather than rely on the placeholder
  // default.
  double m_B_g = m_B_default_g;

  // Relative threshold above which a point is counted as a violation in the
  // delta_T / delta_p / maxwell_s_rho / cs2_vs_fd classes (class D and the
  // cs2 diagnostic, class E).
  double tol_consistency = 0.05;

  // How many worst locations to keep per class (see CheckClassResult::worst).
  size_t worst_n = 10;
};

// One violation/diagnostic class, e.g. "entropy non-monotone in T". A class
// is either a *violation* class (most points are fine; `count`, `max`,
// `rms`, and `worst` describe only the points that violate the check -- see
// check.cpp for the exact per-class rule) or a *diagnostic* class (a
// continuous metric defined at every point, e.g. the Maxwell-consistency
// deltas; `max`/`rms` are then taken over every evaluated point, and `count`
// is just how many exceeded the threshold). Either way, `worst` lists up to
// `worst_n` of the most significant points, sorted by |value| descending.
struct CheckClassResult {
  std::string name;
  size_t count = 0; // points violating / exceeding threshold
  double max = 0;   // magnitude of the metric over the evaluated points
  double rms = 0;   // magnitude of the metric over the evaluated points

  struct Loc {
    size_t irho = 0, jT = 0, kYe = 0;
    double value = 0.0;                    // the metric at this point
    double rho = 0.0, temp = 0.0, ye = 0.0; // physical coordinates
  };
  std::vector<Loc> worst;

  // Sentinel: a class that needed a stored field the table doesn't have
  // (currently only "maxwell_consistency", when "logpress" is absent) is
  // still recorded, with `max`/`rms` set to NaN, so the report -- and
  // print() -- can say so explicitly rather than silently reporting a clean
  // "count = 0". See check.cpp.
};

// Overall result of check_table(). `status` is `fatal` only for structural
// problems (bad axes, non-finite data, missing required fields/attributes);
// `ok` covers everything else, including tables with violations reported in
// `classes` -- check_table never *repairs* anything, so `Status::repaired`
// is not used here (see tools/repair for that).
struct CheckReport {
  Status status = Status::ok;
  std::vector<std::string> fatal_messages; // structural failures (see above)
  std::vector<CheckClassResult> classes;

  // Human-readable summary: status, any fatal messages, then each class's
  // count/max/rms and its worst offenders in physical coordinates.
  void print(std::ostream &os) const;
};

// Runs the table-level checks of CODE.md "Test harness --level table" /
// eos-adapter-F-to-U.md §8 against `table`. Never throws: structural
// problems that would otherwise make later checks unsafe (bad axes,
// non-finite fields, missing required fields/attributes) are caught and
// reported as `fatal_messages` with `status = Status::fatal` instead, and in
// that case the non-structural classes (B-D below) are skipped since they
// cannot be safely evaluated.
//
//   A. Structural (fatal): axes strictly increasing and finite; every field
//      finite everywhere; required fields "logenergy"/"entropy" and
//      attribute "energy_shift" present.
//   B. Range/positivity: "entropy_negative" (entropy < 0). eps + shift > 0
//      and p > 0 are automatic once logenergy/logpress are finite (A already
//      checked that), so they are not separate classes.
//   C. Monotonicity in T: "entropy_nonmonotone_T", "logenergy_nonmonotone_T"
//      -- adjacent-pair violations per (irho, kYe) column.
//   D. Maxwell/thermodynamic-consistency diagnostics (need "logpress";
//      recorded as a single NaN-max "maxwell_consistency" placeholder class
//      if absent): "delta_T", "delta_p", "maxwell_s_rho".
//   E. Sound-speed diagnostic, only if a "cs2" field is present:
//      "cs2_vs_fd" (report-only) and "cs2_out_of_range" (stored cs2 outside
//      (0,1)).
CheckReport check_table(const RawTable &table, const CheckOptions &opts = CheckOptions());

} // namespace eeos
