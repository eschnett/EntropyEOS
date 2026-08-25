// entropy_eos/host/adapter_build.hpp
//
// Builds an EntropyEOS (M2b) from a repaired RawTable: fits the two splines
// Sigma-hat = entropy and L-hat = log10(eps_cgs + energy_shift_cgs)
// (host/bspline_fit.hpp), determines the kappa density/energy rescaling
// that makes U >= 0 by construction (eos-adapter-F-to-U.md S5), relabels
// the fitted splines' x-origin by the resulting density shift so
// core/adapter_eval.hpp's evaluate() needs no further kappa bookkeeping at
// run time, and records a monotonicity audit (Sigma_u, L_u minima and
// violation locations on a refined grid) for later reporting (M2c's
// eos_test --level adapter). Host-only: STL throughout, may throw (see
// CODE.md "Environment"). Does not repair -- build_entropy_eos assumes a
// caller has already run repair_table() (CODE.md "M2 design notes", staged
// delivery item (b)).

#pragma once

#include <cstddef>
#include <vector>

#include "entropy_eos/core/adapter_eval.hpp"
#include "entropy_eos/host/bspline_fit.hpp"
#include "entropy_eos/host/table.hpp"
#include "entropy_eos/host/units.hpp"

namespace eeos {

// Tunables for build_entropy_eos(). Defaults match CODE.md's M2b spec.
struct BuildOptions {
  // Table's declared baryon-mass convention (CODE.md "Open decisions" #2);
  // m_B* = kappa * m_B_table_g is exported by the built EntropyEOS.
  double m_B_table_g = m_amu_g;

  // Refined-grid oversampling per table cell for the kappa-floor and
  // monotonicity-audit scans below: each axis is sampled at `refine` times
  // its native resolution (e.g. refine=4 turns an (nx-1) cubic-cell run
  // into (nx-1)*4+1 sample points).
  int refine = 4;

  // Axis uniformity tolerance: max relative deviation of any consecutive
  // spacing from the axis's mean spacing.
  double uniform_tol = 1e-8;

  // kappa floor safety margin: eps_floor = min(0, eps_hat_min - (abs +
  // rel*|eps_hat_min|)) (eos-adapter-F-to-U.md S5).
  double kappa_margin_rel = 1e-6;
  double kappa_margin_abs = 1e-12;
};

// One location recorded by the monotonicity audit below: the refined-grid
// point (in the *unshifted* table's own (x,u,y) coordinates -- the kappa
// density shift is determined only after this scan runs, see
// build_entropy_eos()) and the (Sigma_u or L_u) value sampled there.
struct AuditLoc {
  double x, u, y;
  double value;
};

// Sigma_u or L_u statistics over the refined grid (eos-adapter-F-to-U.md
// S10 "Invertibility"): the minimum value, how many sampled points were
// non-positive (a violation of the sigma_u>0 / e_u>0 requirement the
// T-solve relies on), and up to 10 of the worst (most negative) locations.
// Never causes build_entropy_eos to throw (CODE.md "M2 design notes": the
// safeguarded T-solve tolerates small local wiggles; this audit is the
// visibility into how many/where).
struct MonotonicityAudit {
  double min_value = 0.0;
  std::size_t violation_count = 0;
  std::vector<AuditLoc> worst; // up to 10, sorted by value ascending (worst first)
};

struct AdapterAudit {
  MonotonicityAudit sigma_u;
  MonotonicityAudit L_u;
};

// Owns the two fitted B-splines plus every scalar the adapter's run-time
// evaluate() needs (see core/adapter_eval.hpp::EntropyEOSView), and the
// audit metadata. Immutable once built; hands out a device-portable
// EntropyEOSView via view().
class EntropyEOS {
public:
  EntropyEOS(Bspline3 sigma, Bspline3 L, double kappa, double m_B_star_g, double m_B_table_g,
             double shift_hat, double conv_t, double x_lo, double x_hi, double u_lo, double u_hi,
             double y_lo, double y_hi, AdapterAudit audit, int max_iter = 50);

  EntropyEOSView view() const;

  double kappa() const { return kappa_; }
  double m_B_star_g() const { return m_B_star_g_; }
  double m_B_table_g() const { return m_B_table_g_; }
  double shift_hat() const { return shift_hat_; }
  double conv_t() const { return conv_t_; }

  double x_lo() const { return x_lo_; }
  double x_hi() const { return x_hi_; }
  double u_lo() const { return u_lo_; }
  double u_hi() const { return u_hi_; }
  double y_lo() const { return y_lo_; }
  double y_hi() const { return y_hi_; }

  int max_iter() const { return max_iter_; }
  const AdapterAudit &audit() const { return audit_; }

private:
  Bspline3 sigma_, L_;
  double kappa_, m_B_star_g_, m_B_table_g_, shift_hat_, conv_t_;
  double x_lo_, x_hi_, u_lo_, u_hi_, y_lo_, y_hi_;
  int max_iter_;
  AdapterAudit audit_;
};

// Builds an EntropyEOS from `table` (see the module comment). Steps (see
// eos-adapter-F-to-U.md S2-S5):
//   1. Validate axis uniformity/size and the required
//      fields/attribute (see below); does not repair.
//   2. Fit Sigma-hat over "entropy" and L-hat over "logenergy" on the
//      table's native (x,u,y) grid.
//   3. kappa re-zeroing: sample eps_hat on a refine-times-refined grid over
//      the whole box from the *fitted splines* (not the raw data -- splines
//      can undershoot); eps_floor = min(0, eps_hat_min - margin); kappa =
//      1 + eps_floor (<= 1; exactly 1 for an already-nonnegative table);
//      m_B* = kappa * m_B_table_g. The stored splines' x-origin is shifted
//      by log10(kappa) so a later evaluate() query at x = log10(rho*)
//      against the shifted grid automatically samples the correct
//      (unshifted) table point -- no refit needed.
//   4. Monotonicity audit (does not throw): Sigma_u/L_u minima and
//      violation locations on the same refined grid.
// Throws std::runtime_error if:
//   - the rho/T/Ye axis is not uniform to within opts.uniform_tol (relative
//     to that axis's own mean spacing), naming the offending axis;
//   - any axis has fewer than 4 points;
//   - "logenergy" or "entropy" is missing from `table`, or contains a
//     non-finite value;
//   - the "energy_shift" attribute is missing.
EntropyEOS build_entropy_eos(const RawTable &table, const BuildOptions &opts = BuildOptions());

} // namespace eeos
