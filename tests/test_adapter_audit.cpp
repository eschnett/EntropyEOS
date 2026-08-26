// tests/test_adapter_audit.cpp — unit tests for the M2c adapter-level audit
// harness: entropy_eos/host/adapter_audit.{hpp,cpp} (check_adapter(),
// adapter_needs_attention(), AdapterReport::print()), exercised against the
// synthetic ground-truth gas (entropy_eos/host/synthetic.hpp) -- see CODE.md
// "Test harness" / eos-adapter-F-to-U.md S10. Patterns (build_synthetic(),
// tolerances derived from the closed-form gas) follow tests/test_adapter.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "entropy_eos/host/adapter_audit.hpp"
#include "entropy_eos/host/adapter_build.hpp"
#include "entropy_eos/host/io_stellarcollapse.hpp"
#include "entropy_eos/host/repair.hpp"
#include "entropy_eos/host/synthetic.hpp"
#include "entropy_eos/host/table.hpp"
#include "entropy_eos/host/units.hpp"
#include "test_scale.hpp"

using eeos::AdapterCheckOptions;
using eeos::AdapterReport;
using eeos::CheckClassResult;
using eeos::EntropyEOS;
using eeos::RawTable;
using eeos::SyntheticOptions;

namespace {

const CheckClassResult &find_class(const AdapterReport &report, const std::string &name) {
  for (const CheckClassResult &c : report.classes) {
    if (c.name == name) return c;
  }
  REQUIRE_MESSAGE(false, "class '" << name << "' not found in AdapterReport::classes");
  static const CheckClassResult dummy; // unreachable (REQUIRE_MESSAGE aborts the test on failure)
  return dummy;
}

} // namespace

// ==========================================================================
// 1. Clean synthetic (default grid): everything zero except the two
//    spline-truncation fidelity diagnostics, which are small.
// ==========================================================================

TEST_CASE("check_adapter: clean synthetic (default grid) reports zero violations") {
  SyntheticOptions opts; // default 40x30x10, no defects
  RawTable table = eeos::make_synthetic_table(opts);
  EntropyEOS adapter = eeos::build_entropy_eos(table);
  // M2d-2: kappa is close to but no longer exactly 1.0 even on a table
  // that is eps>0 everywhere *physically* -- the eps-floor scan now also
  // covers the S7 extension zones, where the designed L u-low tail can dip
  // eps_hat slightly below the table's own minimum (see
  // tests/test_adapter.cpp test 1's doc comment).
  CHECK(adapter.kappa() > 1.0 - 1e-2);
  CHECK(adapter.kappa() <= 1.0);

  // Physicality soak (class C): 200000 states (10000 under sanitizers) --
  // see test_scale.hpp.
  AdapterCheckOptions aopts;
  aopts.soak_n = eeos_n(200000, 10000);
  const AdapterReport report = eeos::check_adapter(adapter, table, aopts);

  CHECK(report.status == eeos::Status::ok);
  CHECK(report.fatal_messages.empty());

  CHECK(find_class(report, "spline_sigma_u_nonpositive").count == 0);
  CHECK(find_class(report, "spline_L_u_nonpositive").count == 0);

  // roundtrip_T is now audited cold-start (u_guess = NaN, not warm-started
  // at the node's own T -- see adapter_audit.cpp's audit_nodes()): the
  // exact synthetic gas's safeguarded Newton/bisection solve still
  // converges to near machine precision regardless of the starting guess,
  // so this stays fully green (count 0) even without a warm start.
  const CheckClassResult &roundtrip = find_class(report, "roundtrip_T");
  CHECK(roundtrip.count == 0);
  CHECK(roundtrip.max <= 1e-9);

  const CheckClassResult &delta_T = find_class(report, "delta_T");
  const CheckClassResult &delta_p = find_class(report, "delta_p");
  MESSAGE("test_adapter_audit 1: delta_T max = " << delta_T.max << ", delta_p max = " << delta_p.max
                                                  << " (default 40x30x10 grid, pure spline error)");
  CHECK(delta_T.max <= 3e-3);
  CHECK(delta_p.max <= 3e-3);

  // Quantiles present (finite) and ordered p50 <= p90 <= p99 <= p999 <= max
  // for both fidelity diagnostics (see QuantileStats's doc comment).
  for (const eeos::QuantileStats &q : {report.delta_T_quantiles, report.delta_p_quantiles}) {
    CHECK(std::isfinite(q.p50));
    CHECK(std::isfinite(q.p90));
    CHECK(std::isfinite(q.p99));
    CHECK(std::isfinite(q.p999));
    CHECK(std::isfinite(q.max));
    CHECK(q.p50 <= q.p90);
    CHECK(q.p90 <= q.p99);
    CHECK(q.p99 <= q.p999);
    CHECK(q.p999 <= q.max);
  }
  // The quantiles' own max must agree with the CheckClassResult's max (both
  // are the same underlying metric's maximum over the same audited nodes).
  CHECK(report.delta_T_quantiles.max == delta_T.max);
  CHECK(report.delta_p_quantiles.max == delta_p.max);

  CHECK(find_class(report, "That_nonpositive").count == 0);
  CHECK(find_class(report, "p_nonpositive").count == 0);
  CHECK(find_class(report, "cs2_nonpositive").count == 0);
  CHECK(find_class(report, "cs2_acausal").count == 0);

  CHECK(report.maxiter_count == 0);
  CHECK_FALSE(eeos::adapter_needs_attention(report));
}

// ==========================================================================
// 2. Fine synthetic grid: delta_T/delta_p shrink with resolution.
// ==========================================================================

TEST_CASE("check_adapter: fine synthetic grid (120x120x12, node_stride=2) fidelity convergence") {
  SyntheticOptions opts_default; // 40x30x10

  RawTable table_default = eeos::make_synthetic_table(opts_default);
  EntropyEOS adapter_default = eeos::build_entropy_eos(table_default);
  AdapterCheckOptions aopts_default;
  aopts_default.soak_n = eeos_n(200000, 10000); // class C soak; see test_scale.hpp
  const AdapterReport report_default = eeos::check_adapter(adapter_default, table_default, aopts_default);

  const double dT_default = find_class(report_default, "delta_T").max;
  const double dp_default = find_class(report_default, "delta_p").max;
  MESSAGE("test_adapter_audit 2: default-grid delta_T=" << dT_default << " delta_p=" << dp_default);

  // The fine-grid (120x120x12) build/audit and the fine-vs-default
  // convergence-ratio checks below are skipped under sanitizers (printed
  // note): they are the same class-A/B fidelity code path at a second,
  // much more expensive resolution, not additional coverage -- see
  // test_scale.hpp. The default-grid fidelity checks above always run.
  if (eeos_sanitized) {
    MESSAGE("test_adapter_audit 2: fine-grid (120x120x12) build/audit and convergence-ratio "
            "checks skipped under sanitizers");
    return;
  }

  SyntheticOptions opts_fine = opts_default;
  opts_fine.nrho = 120;
  opts_fine.ntemp = 120;
  opts_fine.nye = 12;

  RawTable table_fine = eeos::make_synthetic_table(opts_fine);
  EntropyEOS adapter_fine = eeos::build_entropy_eos(table_fine);
  AdapterCheckOptions aopts_fine;
  aopts_fine.node_stride = 2;
  const AdapterReport report_fine = eeos::check_adapter(adapter_fine, table_fine, aopts_fine);

  const double dT_fine = find_class(report_fine, "delta_T").max;
  const double dp_fine = find_class(report_fine, "delta_p").max;

  MESSAGE("test_adapter_audit 2: fine-grid    delta_T=" << dT_fine << " delta_p=" << dp_fine);

  CHECK(dT_fine <= 1e-4);
  CHECK(dp_fine <= 1e-4);

  const double floor = 1e-300;
  CHECK(dT_default / std::max(dT_fine, floor) >= 5.0);
  CHECK(dp_default / std::max(dp_fine, floor) >= 5.0);
}

// ==========================================================================
// 3. Dirty synthetic (repaired in memory first): build succeeds despite
//    planted Inf/NaN in cs2/gamma (the adapter never reads those fields),
//    and check_adapter() runs to completion with finite report values.
// ==========================================================================

TEST_CASE("check_adapter: dirty synthetic, repaired in memory, builds and audits to completion") {
  RawTable table = eeos::make_synthetic_table(eeos::dirty_synthetic_options());

  const eeos::RepairResult repair_result = eeos::repair_table(table);
  std::cout << "test_adapter_audit 3: repair changed " << repair_result.entries.size() << " value(s)\n";

  // Must not throw: the adapter fits only "entropy"/"logenergy", so the
  // planted Inf/NaN in "cs2"/"gamma" (fields it never reads) cannot fail
  // the build.
  EntropyEOS adapter = eeos::build_entropy_eos(table);
  CHECK(std::isfinite(adapter.kappa()));
  CHECK(std::isfinite(adapter.m_B_star_g()));

  // Physicality soak (class C): 200000 states (10000 under sanitizers) --
  // see test_scale.hpp.
  AdapterCheckOptions aopts;
  aopts.soak_n = eeos_n(200000, 10000);
  const AdapterReport report = eeos::check_adapter(adapter, table, aopts);

  CHECK(std::isfinite(report.kappa));
  CHECK(std::isfinite(report.m_B_star_g));
  CHECK(std::isfinite(report.evals_per_sec));

  for (const CheckClassResult &c : report.classes) {
    if (std::isnan(c.max)) {
      // The check.hpp NaN-sentinel "skipped" convention (would-be "delta_p"
      // if "logpress" were absent; the dirty preset always has it, so this
      // branch is not expected to trigger, but is handled the same way
      // check.hpp's own print() handles it).
      CHECK(std::isnan(c.rms));
      continue;
    }
    CHECK(std::isfinite(c.max));
    CHECK(std::isfinite(c.rms));
  }

  // Monotonicity/roundtrip classes are allowed to be nonzero here (the
  // wiggle/flatten/offset defects, even after repair, can leave localized
  // spline overshoot) -- only finiteness is asserted, per this module's
  // contract. roundtrip_T in particular is audited cold-start now (see
  // test case 1 above), so a nonzero count here is expected, not a
  // regression.
  std::cout << "test_adapter_audit 3: spline_sigma_u_nonpositive.count="
            << find_class(report, "spline_sigma_u_nonpositive").count
            << " spline_L_u_nonpositive.count=" << find_class(report, "spline_L_u_nonpositive").count
            << " roundtrip_T.count=" << find_class(report, "roundtrip_T").count << "\n";

  // The dirty preset always carries "logpress" (dirty_synthetic_options()
  // always fills the core fields), so both quantile structs must be
  // present (finite) and ordered, same as test case 1.
  for (const eeos::QuantileStats &q : {report.delta_T_quantiles, report.delta_p_quantiles}) {
    CHECK(std::isfinite(q.p50));
    CHECK(std::isfinite(q.p90));
    CHECK(std::isfinite(q.p99));
    CHECK(std::isfinite(q.p999));
    CHECK(std::isfinite(q.max));
    CHECK(q.p50 <= q.p90);
    CHECK(q.p90 <= q.p99);
    CHECK(q.p99 <= q.p999);
    CHECK(q.p999 <= q.max);
  }

  std::ostringstream oss;
  report.print(oss);
  CHECK_FALSE(oss.str().empty());
}

// ==========================================================================
// 4. adapter_needs_attention()
// ==========================================================================

TEST_CASE("adapter_needs_attention: physicality/monotonicity/roundtrip/maxiter drive it, "
          "delta_T/delta_p never do") {
  AdapterReport clean;
  clean.status = eeos::Status::ok;
  CHECK_FALSE(eeos::adapter_needs_attention(clean));

  SUBCASE("a physicality violation class forces true") {
    AdapterReport r = clean;
    CheckClassResult c;
    c.name = "p_nonpositive";
    c.count = 3;
    r.classes.push_back(c);
    CHECK(eeos::adapter_needs_attention(r));
  }

  SUBCASE("cs2_acausal forces true") {
    AdapterReport r = clean;
    CheckClassResult c;
    c.name = "cs2_acausal";
    c.count = 1;
    r.classes.push_back(c);
    CHECK(eeos::adapter_needs_attention(r));
  }

  SUBCASE("a monotonicity violation class forces true") {
    AdapterReport r = clean;
    CheckClassResult c;
    c.name = "spline_sigma_u_nonpositive";
    c.count = 10;
    r.classes.push_back(c);
    CHECK(eeos::adapter_needs_attention(r));
  }

  SUBCASE("roundtrip_T forces true") {
    AdapterReport r = clean;
    CheckClassResult c;
    c.name = "roundtrip_T";
    c.count = 1;
    r.classes.push_back(c);
    CHECK(eeos::adapter_needs_attention(r));
  }

  SUBCASE("maxiter_count alone forces true") {
    AdapterReport r = clean;
    r.maxiter_count = 1;
    CHECK(eeos::adapter_needs_attention(r));
  }

  SUBCASE("delta_T/delta_p are diagnostics: a forced nonzero count never triggers it") {
    AdapterReport r = clean;
    CheckClassResult dT, dp;
    dT.name = "delta_T";
    dT.count = 100;
    dp.name = "delta_p";
    dp.count = 100;
    r.classes.push_back(dT);
    r.classes.push_back(dp);
    CHECK_FALSE(eeos::adapter_needs_attention(r));
  }

  SUBCASE("every class present but all zero-count is still clean") {
    AdapterReport r = clean;
    for (const char *name : {"spline_sigma_u_nonpositive", "spline_L_u_nonpositive", "roundtrip_T",
                              "delta_T", "delta_p", "That_nonpositive", "p_nonpositive",
                              "cs2_nonpositive", "cs2_acausal"}) {
      CheckClassResult c;
      c.name = name;
      c.count = 0;
      r.classes.push_back(c);
    }
    CHECK_FALSE(eeos::adapter_needs_attention(r));
  }
}

// ==========================================================================
// 5. AdapterReport::print() smoke test
// ==========================================================================

TEST_CASE("AdapterReport::print: mentions kappa, every class name, evals/sec") {
  SyntheticOptions opts; // default grid
  RawTable table = eeos::make_synthetic_table(opts);
  EntropyEOS adapter = eeos::build_entropy_eos(table);

  AdapterCheckOptions aopts;
  aopts.soak_n = 2000; // keep this smoke test fast; not exercising soak statistics here
  const AdapterReport report = eeos::check_adapter(adapter, table, aopts);

  std::ostringstream oss;
  report.print(oss);
  const std::string text = oss.str();

  CHECK(text.find("kappa") != std::string::npos);
  CHECK(text.find("evals/sec") != std::string::npos);
  for (const CheckClassResult &c : report.classes) {
    CHECK_MESSAGE(text.find(c.name) != std::string::npos,
                  "AdapterReport::print() is missing class name '" << c.name << "'");
  }

  // The delta_T/delta_p quantiles line (see QuantileStats's doc comment).
  CHECK(text.find("quantiles") != std::string::npos);
  CHECK(text.find("p50=") != std::string::npos);
  CHECK(text.find("p999=") != std::string::npos);
}

// ==========================================================================
// 6. M2d-2: extension_seam_jump is present and small, and soak_extended
//    runs clean, on the clean synthetic table.
// ==========================================================================

TEST_CASE("check_adapter: extension_seam_jump small and soak_extended clean on synthetic") {
  SyntheticOptions opts; // default 40x30x10, no defects
  RawTable table = eeos::make_synthetic_table(opts);
  EntropyEOS adapter = eeos::build_entropy_eos(table);

  AdapterCheckOptions aopts;
  aopts.soak_n = eeos_n(20000, 1000); // soak_extended; see test_scale.hpp
  aopts.soak_extended = true;
  const AdapterReport report = eeos::check_adapter(adapter, table, aopts);

  CHECK(report.status == eeos::Status::ok);
  CHECK(report.fatal_messages.empty());

  // "extension_seam_jump" (M2d-2's class D): diagnostic only -- never a
  // violation (count stays 0 regardless of magnitude) -- but small on a
  // clean table, since the tails are C2 by construction (see
  // core/adapter_eval.hpp's TAIL MATHEMATICS; the guard/slope-zero
  // overrides that would drop to C1/C0 do not activate on this table's
  // comfortably-positive boundary slopes).
  const CheckClassResult &seam = find_class(report, "extension_seam_jump");
  MESSAGE("test_adapter_audit 6: extension_seam_jump max = " << seam.max);
  CHECK(seam.count == 0);
  CHECK(seam.max <= 1e-6);

  // soak_extended (opts.soak_extended=true): the S7/M2d-2 extension zones
  // must be finite (status==ok already established that: a non-finite
  // EOSPoint anywhere would have flipped it to fatal) and monotone-solvable
  // -- zero maxiter_count, and the iteration histogram accounts for every
  // sampled point.
  CHECK(report.maxiter_count == 0);
  size_t hist_sum = 0;
  for (size_t i = 0; i < 64; ++i) hist_sum += report.iters_hist[i];
  CHECK(hist_sum == aopts.soak_n);

  // Physicality in the extension zone is not guaranteed by design (it is an
  // escape hatch for the solver, not a claim of physical validity far from
  // the table -- eos-adapter-F-to-U.md S7) so these are reported, not
  // asserted zero.
  MESSAGE("test_adapter_audit 6: soak_extended That_nonpositive="
          << find_class(report, "That_nonpositive").count
          << " p_nonpositive=" << find_class(report, "p_nonpositive").count
          << " cs2_nonpositive=" << find_class(report, "cs2_nonpositive").count
          << " cs2_acausal=" << find_class(report, "cs2_acausal").count);
}

// ==========================================================================
// 7. M2d-2: real tables (guarded, like tests/test_adapter.cpp test 8) --
//    soak_extended and the extension-seam-jump max.
// ==========================================================================

namespace {

const std::string kLS220Path = "tables/LS220_234r_136t_50y_analmu_20091212_SVNr26.h5";
const std::string kSROPath = "tables/LS220_3335_rho391_temp163_ye66.h5";

bool table_exists(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

void run_real_table_soak_extended(const std::string &full_path, const char *label) {
  const std::string path = eeos_san_table(full_path);
  if (!table_exists(path)) {
    WARN_MESSAGE(false, label << " table not found at '" << path << "' -- skipped ('skipped')");
    return;
  }

  RawTable table = eeos::read_stellarcollapse(path);
  const eeos::RepairResult repair_result = eeos::repair_table(table);
  (void)repair_result;
  EntropyEOS adapter = eeos::build_entropy_eos(table);

  // A real table's node grid is ~1e5-1e6 points; a large node_stride keeps
  // this guarded, local-only smoke test's class B pass fast while still
  // exercising it (unlike tests/test_adapter.cpp test 8, whose job is the
  // evaluate()-level soak, this test's job is check_adapter()'s new
  // soak_extended path and the seam-jump diagnostic, not a from-scratch
  // full audit -- that is tools/eos_test --level adapter's job, see
  // tests/integration.sh).
  AdapterCheckOptions aopts;
  aopts.node_stride = 25;
  aopts.soak_n = eeos_n(20000, 1000); // soak_extended; see test_scale.hpp
  aopts.soak_extended = true;
  const AdapterReport report = eeos::check_adapter(adapter, table, aopts);

  CHECK(report.status != eeos::Status::fatal);

  const CheckClassResult &seam = find_class(report, "extension_seam_jump");
  std::cout << "test_adapter_audit 7 (" << label << "): kappa=" << adapter.kappa()
            << " extension_seam_jump max=" << seam.max
            << " soak_extended maxiter_count=" << report.maxiter_count << "/" << aopts.soak_n << "\n";
}

} // namespace

TEST_CASE("check_adapter: LS220 real table, soak_extended + extension_seam_jump (guarded)") {
  run_real_table_soak_extended(kLS220Path, "LS220");
}

TEST_CASE("check_adapter: SRO real table, soak_extended + extension_seam_jump (guarded)") {
  if (eeos_skip_big_table("test_adapter_audit 7 (SRO): real table")) return;
  run_real_table_soak_extended(kSROPath, "SRO");
}
