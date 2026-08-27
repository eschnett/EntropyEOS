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

// Replaces every value in the inclusive block [irho0,irho1] x [kYe0,kYe1],
// for jT in [jT0, jT1], with the value the field already has at jT0: a
// plateau along T. Mimics a near-flat table region (e.g. SRO's logenergy
// plateaus): the resulting <=-pairs are exactly what PAVA strictifies.
struct FlattenDefect {
  std::string field;
  size_t irho0 = 0, irho1 = 0;
  size_t kYe0 = 0, kYe1 = 0;
  size_t jT0 = 0, jT1 = 0;
};

// Adds `amplitude * sin(2*pi*(jT - jT0)/period)` to every value in the same
// kind of inclusive block as FlattenDefect. With `amplitude` large enough
// relative to the field's natural per-step increment, this makes several
// adjacent T-pairs decrease inside the block. Mimics a clustered
// non-monotone-T patch (e.g. LS220's entropy transition cluster at high
// rho/Ye).
struct WiggleDefect {
  std::string field;
  size_t irho0 = 0, irho1 = 0;
  size_t kYe0 = 0, kYe1 = 0;
  size_t jT0 = 0, jT1 = 0;
  double amplitude = 0.0;
  double period = 1.0;
};

// Adds a constant `offset` to every value in the same kind of inclusive
// block as FlattenDefect. Mimics a systematic bias in a corner of the grid
// (e.g. SRO's slightly negative cold-corner entropies, via a negative
// offset).
struct OffsetDefect {
  std::string field;
  size_t irho0 = 0, irho1 = 0;
  size_t kYe0 = 0, kYe1 = 0;
  size_t jT0 = 0, jT1 = 0;
  double offset = 0.0;
};

// Adds a smooth power-law stiffening term to the *physical* specific energy:
//
//   eps_cgs(rho,T,Ye) += amplitude * (rho / rho_c_gcc)^alpha        [erg/g]
//
// applied at every grid point (no cutoff and no blending -- for alpha > 0 the
// term falls off as rho^-alpha, so it becomes negligible well below rho_c and
// eventually drops below the last bit of eps + energy_shift entirely, which
// makes "above rho_c" automatic while the added function stays analytic in
// rho and exactly independent of T and Ye). The field must be stored as
// log10(eps + energy_shift) (i.e.
// "logenergy"): the term is added to 10^value and the result re-logged, which
// is the same as adding it to eps since energy_shift is a constant.
//
// Purpose: mimic a stiff high-density corner. Since eps then grows like
// rho^alpha along every adiabat, c_s^2 = (eps_x + eps_xx)/h -> (alpha +
// alpha^2) eps / h at the top of the rho axis, which exceeds 1 -- i.e. the
// constructed U is superluminal -- once eps (in units of c^2) passes roughly
// 1/(alpha^2 - 1), for every T and Ye at once. That is the shape of the real
// LS220/SRO acausal corner (eos-causality-repair.md S2) and what the repair's
// causal-cap stage is built to fix. Being exactly T-independent, the term
// also leaves monotonicity in T untouched, so it exercises the causal-cap
// stage without perturbing the two monotonicity stages.
struct StiffenDefect {
  std::string field;
  double rho_c_gcc = 1.0;
  double alpha = 0.0;
  double amplitude = 0.0; // erg/g; the term's value exactly at rho = rho_c_gcc
};

// Sets the stored value of `field` at grid index (irho, jT, kYe) outright
// (replacing it, not adding to it). Unlike SeededViolation's additive delta,
// this can plant Inf/NaN cleanly -- used to mimic real tables' occasional
// non-finite points in fields the pipeline does not interpret (e.g. LS220's
// Inf points in cs2/gamma).
struct SetValue {
  std::string field;
  size_t irho = 0;
  size_t jT = 0;
  size_t kYe = 0;
  double value = 0.0;
};

// Grid, model constants, and defects for make_synthetic_table(). Defaults
// match CODE.md's synthetic-table specification: the grid is unperturbed and
// every defect list is empty, so a default-constructed SyntheticOptions
// always yields a clean table -- every defect mechanism below is opt-in.
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

  // If true, adds three auxiliary fields after the core three -- "cs2",
  // "gamma", "mu_e" (see make_synthetic_table()) -- fields the check/repair
  // pipeline never interprets, so defects planted in them can only ever
  // produce "nonfinite_<field>" violation classes, never a fatal one (see
  // check.hpp). Off by default: every existing caller/test that only looks
  // at "logenergy"/"entropy"/"logpress" is unaffected.
  bool with_aux_fields = false;

  // Defects to apply after the core (+ aux, if enabled) fields are
  // generated. All optional (default-empty) and independent of each other;
  // see make_synthetic_table()'s doc comment for the fixed order they are
  // applied in.
  std::vector<FlattenDefect> flatten;
  std::vector<WiggleDefect> wiggle;
  std::vector<OffsetDefect> offset;
  std::vector<StiffenDefect> stiffen;
  std::vector<SeededViolation> seed;
  std::vector<SetValue> setvalue;
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

// The exact ideal-gas (Gamma = 5/3) sound speed for this model, used to fill
// the optional "cs2" auxiliary field (opts.with_aux_fields):
//
//   g = 1 + Ye,  kT_erg = temp_MeV * MeV_to_erg
//   eps = synthetic_eps(...), p = synthetic_p(...)
//   h = 1 + (eps + p/rho) / c^2
//   cs2 = (5/3) * g * kT_erg / (m_B * h * c^2),  m_B = m_B_default_g, c = c_light_cm_s
//
// This exact expression (same operations, same order, same constants) is
// also used -- independently, inline -- by tests/test_check.cpp to build a
// "cs2" field for the cs2-diagnostic tests; that file is out of scope for
// this module, so its copy is kept bit-identical by construction rather than
// refactored to call this function. Keep the two in lockstep if either ever
// changes.
double synthetic_cs2(double rho_gcc, double temp_MeV, double ye, const SyntheticOptions &opts);

// Builds a RawTable sampling the analytic model on the grid described by
// `opts`, storing:
//   "logenergy" = log10(eps + energy_shift)
//   "entropy"   = s
//   "logpress"  = log10(p)
// plus, if opts.with_aux_fields, three more fields (diagnostic-only -- the
// check/repair pipeline never interprets them, see check.hpp):
//   "cs2"   = synthetic_cs2(...)
//   "gamma" = 5/3 everywhere
//   "mu_e"  = (kT_erg / m_B) * Ye  (a smooth dummy, m_B = m_amu_g)
// plus the "energy_shift" attribute.
//
// Defects are then applied, to the stored representation (post-log10 for
// "logenergy", exactly like SeededViolation always has), in this fixed
// order -- core + aux fields are all in place before any defect runs, and
// each defect list runs to completion before the next begins:
//   1. opts.flatten  (plateaus)
//   2. opts.wiggle    (oscillatory perturbation)
//   3. opts.offset    (constant shift)
//   4. opts.stiffen   (smooth high-density power-law energy excess)
//   5. opts.seed      (existing per-point additive violations)
//   6. opts.setvalue  (outright sets, last, so a planted Inf/NaN cannot be
//                       perturbed by any earlier or later step)
// Every FlattenDefect/WiggleDefect/OffsetDefect/SetValue index (and, for the
// block defects, irho0<=irho1, kYe0<=kYe1, jT0<=jT1) is validated against
// the table's actual grid size; an out-of-range index throws
// std::out_of_range (as does naming a field the table does not have, via
// RawTable::field()). A StiffenDefect with a non-positive rho_c_gcc throws
// std::invalid_argument. All defects are deterministic -- no RNG anywhere.
RawTable make_synthetic_table(const SyntheticOptions &opts = SyntheticOptions());

// Default grid (40 x 30 x 10), with opts.with_aux_fields = true and a fixed
// set of deterministic defects chosen to mimic pathologies actually observed
// in the real LS220/SRO stellarcollapse tables (see CODE.md), so
// `eos_test --synthetic-dirty` exercises them without needing any table
// file:
//   - a "entropy" WiggleDefect at high rho/Ye, clustering several
//     non-monotone-T pairs (mimics LS220's transition cluster);
//   - a "logenergy" FlattenDefect (mimics SRO's near-flat plateaus);
//   - a "entropy" OffsetDefect at the low-rho/low-T/low-Ye corner, driving
//     that whole block negative (mimics SRO's slightly negative cold-corner
//     entropies);
//   - a "logenergy" StiffenDefect making the top rho cells superluminal at
//     every T and Ye (mimics the LS220/SRO acausal high-density corner --
//     eos-causality-repair.md S2 -- and exercises repair's causal-cap stage);
//   - "cs2"/"gamma" SetValue defects planting Inf/NaN (mimics LS220's
//     handful of non-finite cs2/gamma points).
SyntheticOptions dirty_synthetic_options();

} // namespace eeos
