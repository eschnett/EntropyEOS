// entropy_eos/host/synthetic.hpp
//
// An exactly thermodynamically consistent analytic ideal-gas EOS, tabulated
// into a RawTable. Used as ground truth by the unit tests and by
// `eos_test --synthetic` (see CODE.md "Test harness"): a clean synthetic
// table must report zero violations, and deliberately seeded violations must
// be found (and, later, repaired).

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "entropy_eos/host/table.hpp"

namespace eeos {

// Adds `delta` to the stored value of `field` at grid index (irho, jT, kYe)
// after the synthetic table has been generated, to exercise
// violation-detection code against a known, localized defect.
struct SeededViolation {
  std::string field;
  size_t irho = 0;
  size_t jT = 0;
  size_t kYe = 0;
  double delta = 0.0;
};

// Grid, model constants, and seeded defects for make_synthetic_table().
// Defaults match CODE.md's synthetic-table specification.
struct SyntheticOptions {
  // Grid: log-uniform in rho and T, linear in Ye.
  size_t nrho = 40;
  double rho_min_gcc = 1e5;
  double rho_max_gcc = 1e15;

  size_t ntemp = 30;
  double temp_min_MeV = 0.05;
  double temp_max_MeV = 50.0;

  size_t nye = 10;
  double ye_min = 0.05;
  double ye_max = 0.55;

  // energy_shift attribute, erg/g, applied so that logenergy = log10(eps +
  // energy_shift) is finite (eps can be small, and log10 requires > 0).
  double energy_shift = 1.5e18;

  // Model constants.
  double s0 = 12.0;       // kB/baryon: entropy scale at (T0_MeV, rho0_gcc)
  double T0_MeV = 0.05;   // MeV: entropy reference temperature
  double rho0_gcc = 1e16; // g/cm^3: entropy reference density

  // Defects to seed into the generated table, applied after generation.
  std::vector<SeededViolation> seed;
};

// The analytic model itself, exposed so tests can compare table values
// exactly against it. With g(Ye) = 1 + Ye and kT_erg = temp_MeV *
// MeV_to_erg, m_B = m_amu_g (see units.hpp):
//
//   eps(rho,T,Ye) = 1.5 * g * kT_erg / m_B                        [erg/g]
//   p(rho,T,Ye)   = g * rho * kT_erg / m_B                        [dyn/cm^2]
//   s(rho,T,Ye)   = g * (s0 + 1.5*ln(T/T0_MeV) - ln(rho/rho0))    [kB/baryon]
//
// This is exactly Maxwell-consistent (a single-potential ideal gas); the
// formulas must be kept exactly as given, not "simplified".
double synthetic_eps(double rho_gcc, double temp_MeV, double ye, const SyntheticOptions &opts);
double synthetic_p(double rho_gcc, double temp_MeV, double ye, const SyntheticOptions &opts);
double synthetic_s(double rho_gcc, double temp_MeV, double ye, const SyntheticOptions &opts);

// Builds a RawTable sampling the analytic model on the grid described by
// `opts`, storing:
//   "logenergy" = log10(eps + energy_shift)
//   "entropy"   = s
//   "logpress"  = log10(p)
// plus the "energy_shift" attribute. If opts.seed is non-empty, each
// SeededViolation is applied (added to the named stored field at its grid
// index) after generation.
RawTable make_synthetic_table(const SyntheticOptions &opts = SyntheticOptions());

} // namespace eeos
