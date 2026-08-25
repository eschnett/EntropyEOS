// entropy_eos/host/adapter_audit.hpp
//
// M2c adapter-level audit harness: `check_adapter()` runs the audit list of
// eos-adapter-F-to-U.md S10 against a *built* EntropyEOS, exercised through
// `tools/eos_test --level adapter` (CODE.md "Test harness"). Like
// `check_table()` (host/check.hpp), this is a library function first: pure
// (no I/O side effects), so an HPC consumer can validate a built adapter
// in-process. It reuses host/check.hpp's `CheckClassResult` so the two
// levels' reports share one shape (worst-offender lists, count/max/rms,
// print()-ability). Host-only: STL throughout, may throw; OpenMP where
// loops are independent (guarded by `_OPENMP`, serial otherwise -- see
// CODE.md "Environment").
//
// `check_adapter()` needs both the built `EntropyEOS` *and* the (typically
// repaired) `RawTable` it was built from: the table supplies node
// coordinates for the round-trip/fidelity audit (class B below) and the
// diagnostic-only "logpress" column, neither of which the built adapter
// retains itself.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#include "entropy_eos/core/defs.hpp"
#include "entropy_eos/host/adapter_build.hpp"
#include "entropy_eos/host/check.hpp"
#include "entropy_eos/host/table.hpp"

namespace eeos {

// Tunables for check_adapter().
struct AdapterCheckOptions {
  // Audit every node_stride-th table node in each of (irho, jT, kYe) for
  // the class B round-trip/fidelity audit below (1 = every node).
  size_t node_stride = 1;

  // Number of random points sampled for the class C physicality soak.
  size_t soak_n = 200000;

  // Seed for the soak's deterministic per-sample PRNG (see adapter_audit.cpp:
  // no std::random_device anywhere -- reproducibility is the point).
  unsigned soak_seed = 12345u;

  // How many worst locations to keep per class (see CheckClassResult::worst).
  // Note class A's worst lists are capped by however many the build-time
  // scan itself retained (host/adapter_build.cpp's kMaxWorst == 10), since
  // that scan is not recomputed here.
  size_t worst_n = 10;

  // Relative threshold for "roundtrip_T" (|T_F/T_node - 1|) entering the
  // violation class.
  double tol_roundtrip = 1e-8;
};

// Robust distribution summary for a diagnostic metric collected at every
// audited node (currently "delta_T"/"delta_p" -- see AdapterReport below).
// CheckClassResult's max/rms are dominated by any single remaining
// sigma_T~0 pocket (a delta_T/delta_p spike there can be many orders of
// magnitude above the bulk); these quantiles answer the "how good is the
// table fidelity, typically" question without being swamped by that one
// pocket. All fields are NaN together iff the underlying class was skipped
// (delta_p with no "logpress" -- see check.hpp's NaN-sentinel "skipped"
// convention, which this mirrors) or, degenerately, zero nodes were
// audited.
struct QuantileStats {
  double p50 = 0.0, p90 = 0.0, p99 = 0.0, p999 = 0.0, max = 0.0;
};

// Result of check_adapter(). `status` is `fatal` only if some evaluate()
// call in the audit produced a non-finite EOSPoint member (see
// `fatal_messages` for where); everything else -- monotonicity violations,
// round-trip/fidelity misses, physicality violations, iteration-cap hits --
// is reported through `classes`/`maxiter_count` without making the report
// fatal (see adapter_needs_attention() below for the "should a caller
// care?" summary).
struct AdapterReport {
  Status status = Status::ok;
  std::vector<std::string> fatal_messages; // located non-finite-EOSPoint reports, if any

  double kappa = 0.0;
  double m_B_star_g = 0.0;

  // One CheckClassResult per audit below, in this fixed order:
  //   A. "spline_sigma_u_nonpositive", "spline_L_u_nonpositive"   (violation; from the stored build audit)
  //   B. "roundtrip_T" (violation, cold-start -- see check_adapter()'s doc comment), "delta_T"
  //      (diagnostic), "delta_p" (diagnostic; the check.hpp NaN-sentinel "skipped" class if the
  //      table has no "logpress")
  //   C. "That_nonpositive", "p_nonpositive", "cs2_nonpositive", "cs2_acausal" (violations)
  std::vector<CheckClassResult> classes;

  // Quantiles {p50,p90,p99,p999,max} of "delta_T"/"delta_p" over every
  // finite-metric audited node (not just the CheckClassResult's worst_n
  // list) -- see QuantileStats's doc comment. delta_p_quantiles is the
  // all-NaN sentinel whenever the "delta_p" class itself is (no
  // "logpress" field).
  QuantileStats delta_T_quantiles;
  QuantileStats delta_p_quantiles;

  // Physicality-soak statistics (class C).
  size_t soak_n = 0;
  size_t maxiter_count = 0;
  size_t iters_hist[64] = {}; // cold-start T-solve iteration count histogram, index = min(iters, 63)
  double evals_per_sec = 0.0;

  // Human-readable summary: status, kappa/m_B*, any fatal messages, each
  // class's count/max/rms and worst offenders (physical rho/T/Ye), the
  // delta_T/delta_p quantiles, then the soak statistics and iteration
  // histogram.
  void print(std::ostream &os) const;
};

// Runs the eos-adapter-F-to-U.md S10 audit suite against `adapter` (built
// from `table`, typically after repair_table() -- see CODE.md "M2 design
// notes"). `table` must be the same table `adapter` was built from: its
// axes/entropy/logenergy/logpress columns supply node coordinates and the
// diagnostic-only pressure reference; check_adapter() does not re-fit or
// re-validate anything build_entropy_eos() already did.
//
//   A. Monotonicity (from the build's stored audit, not recomputed):
//      sigma_u/L_u minima, violation counts, worst locations (converted to
//      physical rho/T/Ye).
//   B. Node round trip and fidelity, every node_stride-th table node,
//      evaluate() called *cold* (u_guess = NaN, not warm-started at the
//      node's own T -- see adapter_audit.cpp's audit_nodes() for why: a
//      warm start at the exact answer made this class trivially green
//      regardless of solver robustness):
//      "roundtrip_T" (|T_F/T_node - 1| against opts.tol_roundtrip),
//      "delta_T" and "delta_p" (m_B*/consistency fidelity diagnostics of
//      eos-adapter-F-to-U.md S10, never affecting adapter_needs_attention();
//      AdapterReport::delta_T_quantiles/delta_p_quantiles report their
//      {p50,p90,p99,p999,max} distribution over every audited node, robust
//      to the max/rms being dominated by a single leftover pocket).
//   C. Physicality soak: opts.soak_n deterministic random cold-start
//      evaluate() calls uniform in the adapter's physical (x*, Ye) box and
//      in the pointwise entropy range from srange(); "That_nonpositive",
//      "p_nonpositive", "cs2_nonpositive", "cs2_acausal" violation classes,
//      flag_maxiter counted into maxiter_count, an iters_hist histogram, and
//      evals_per_sec timing (point generation is excluded from the timed
//      region; only the evaluate() calls are timed).
//
// Never throws on its own account: a non-finite EOSPoint anywhere in B or C
// is recorded in fatal_messages (status becomes Status::fatal) and that
// point is excluded from the surrounding class's statistics, but the audit
// continues over the remaining points.
AdapterReport check_adapter(const EntropyEOS &adapter, const RawTable &table,
                             const AdapterCheckOptions &opts = AdapterCheckOptions());

// True iff the report indicates something a caller should look at:
// any monotonicity/roundtrip_T/physicality violation class has count > 0,
// or maxiter_count > 0. (status == Status::fatal is a separate, more severe
// signal callers should check on its own -- see tools/eos_test.cpp.)
// "delta_T"/"delta_p" are diagnostics and never contribute here.
bool adapter_needs_attention(const AdapterReport &report);

} // namespace eeos
