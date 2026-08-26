// tests/test_con2prim_audit.cpp — unit tests for the M3b con2prim-level
// round-trip audit harness: entropy_eos/host/con2prim_audit.{hpp,cpp}
// (check_con2prim(), con2prim_needs_attention(), Con2PrimReport::print()),
// exercised against the synthetic ground-truth gas
// (entropy_eos/host/synthetic.hpp) -- see CODE.md "Test harness" /
// con2prim-entropy-rapidity.md S12 deliverable 2. Patterns
// (build_synthetic(), the guarded real-table smoke test) follow
// tests/test_adapter_audit.cpp and tests/test_con2prim.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include "entropy_eos/host/adapter_build.hpp"
#include "entropy_eos/host/con2prim_audit.hpp"
#include "entropy_eos/host/io_stellarcollapse.hpp"
#include "entropy_eos/host/repair.hpp"
#include "entropy_eos/host/synthetic.hpp"
#include "entropy_eos/host/table.hpp"
#include "test_scale.hpp"

using eeos::CheckClassResult;
using eeos::Con2PrimCheckOptions;
using eeos::Con2PrimReport;
using eeos::EntropyEOS;
using eeos::QuantileStats;
using eeos::RawTable;
using eeos::SyntheticOptions;

namespace {

const CheckClassResult &find_class(const Con2PrimReport &report, const std::string &name) {
  for (const CheckClassResult &c : report.classes) {
    if (c.name == name) return c;
  }
  REQUIRE_MESSAGE(false, "class '" << name << "' not found in Con2PrimReport::classes");
  static const CheckClassResult dummy;
  return dummy;
}

EntropyEOS build_synthetic(const SyntheticOptions &opts) {
  RawTable table = eeos::make_synthetic_table(opts);
  return eeos::build_entropy_eos(table);
}

void check_quantiles_ordered(const QuantileStats &q) {
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

} // namespace

// ==========================================================================
// 1. Clean synthetic (default grid, n_states=4000, 500 under sanitizers):
//    the core acceptance test (design doc deliverable 2).
// ==========================================================================

TEST_CASE("check_con2prim: clean synthetic (default grid, n=4000) round trips cleanly") {
  SyntheticOptions opts; // default 40x30x10, no defects
  EntropyEOS adapter = build_synthetic(opts);

  Con2PrimCheckOptions copts;
  copts.n_states = eeos_n(4000, 500);
  const Con2PrimReport report = eeos::check_con2prim(adapter, copts);

  CHECK(report.status == eeos::Status::ok);
  CHECK(report.n_states == copts.n_states);

  // Zero failed_* across both passes: the synthetic gas is smooth
  // everywhere in the sampled interior (5% margin), so both the coupled
  // Newton and the SS9 fallback should always converge.
  CHECK(report.n_failed_no_bracket == 0);
  CHECK(report.n_failed_max_iter == 0);
  CHECK(report.cold_n_failed_no_bracket == 0);
  CHECK(report.cold_n_failed_max_iter == 0);
  CHECK(find_class(report, "c2p_failed").count == 0);

  // Synthetic is smooth everywhere: the conservative-space round trip
  // should be near machine precision. Report the measured maxima (per this
  // module's contract) and assert the design doc's 1e-10 bar.
  MESSAGE("test_con2prim_audit 1: rt_D max=" << report.rt_D.max << " rt_tau max=" << report.rt_tau.max
                                              << " rt_S max=" << report.rt_S.max);
  CHECK(report.rt_D.max <= 1e-10);
  CHECK(report.rt_tau.max <= 1e-10);
  CHECK(report.rt_S.max <= 1e-10);

  CHECK(find_class(report, "c2p_roundtrip").count == 0);
  CHECK_FALSE(eeos::con2prim_needs_attention(report));

  // Quantiles ordered p50 <= p90 <= p99 <= p999 <= max, for every reported
  // QuantileStats.
  check_quantiles_ordered(report.rt_D);
  check_quantiles_ordered(report.rt_tau);
  check_quantiles_ordered(report.rt_S);
  check_quantiles_ordered(report.prim_rho);
  check_quantiles_ordered(report.prim_w);
  check_quantiles_ordered(report.prim_s);

  // Both iteration histograms are nonempty (n_states > 0, and every 10th
  // state gets a cold pass too).
  size_t warm_sum = 0, cold_sum = 0;
  for (size_t i = 0; i < 64; ++i) {
    warm_sum += report.iters_hist_warm[i];
    cold_sum += report.iters_hist_cold[i];
  }
  CHECK(warm_sum == report.n_states);
  CHECK(cold_sum > 0);

  MESSAGE("test_con2prim_audit 1: solves_per_sec_warm=" << report.solves_per_sec_warm
                                                          << " solves_per_sec_cold="
                                                          << report.solves_per_sec_cold);

  // print() smoke: mentions every field group.
  std::ostringstream oss;
  report.print(oss);
  const std::string text = oss.str();
  CHECK(text.find("status=") != std::string::npos);
  CHECK(text.find("n_states=") != std::string::npos);
  CHECK(text.find("warm:") != std::string::npos);
  CHECK(text.find("cold:") != std::string::npos);
  CHECK(text.find("iters_hist_warm") != std::string::npos);
  CHECK(text.find("iters_hist_cold") != std::string::npos);
  CHECK(text.find("rt_D") != std::string::npos);
  CHECK(text.find("rt_tau") != std::string::npos);
  CHECK(text.find("rt_S") != std::string::npos);
  CHECK(text.find("prim_rho") != std::string::npos);
  CHECK(text.find("prim_w") != std::string::npos);
  CHECK(text.find("prim_s") != std::string::npos);
  CHECK(text.find("c2p_failed") != std::string::npos);
  CHECK(text.find("c2p_roundtrip") != std::string::npos);
  CHECK(text.find("solves_per_sec_warm") != std::string::npos);
  CHECK(text.find("solves_per_sec_cold") != std::string::npos);
}

// ==========================================================================
// 2. Determinism: two runs with the same seed/options are bitwise
//    identical.
// ==========================================================================

TEST_CASE("check_con2prim: two runs with identical options are bitwise identical") {
  SyntheticOptions opts;
  EntropyEOS adapter = build_synthetic(opts);

  Con2PrimCheckOptions copts;
  copts.n_states = eeos_n(1500, 300); // determinism check: scale size, not logic

  const Con2PrimReport r1 = eeos::check_con2prim(adapter, copts);
  const Con2PrimReport r2 = eeos::check_con2prim(adapter, copts);

  CHECK(r1.status == r2.status);
  CHECK(r1.n_states == r2.n_states);
  CHECK(r1.n_newton == r2.n_newton);
  CHECK(r1.n_fallback == r2.n_fallback);
  CHECK(r1.n_failed_no_bracket == r2.n_failed_no_bracket);
  CHECK(r1.n_failed_max_iter == r2.n_failed_max_iter);
  CHECK(r1.cold_n_newton == r2.cold_n_newton);
  CHECK(r1.cold_n_fallback == r2.cold_n_fallback);
  CHECK(r1.cold_n_failed_no_bracket == r2.cold_n_failed_no_bracket);
  CHECK(r1.cold_n_failed_max_iter == r2.cold_n_failed_max_iter);

  for (size_t i = 0; i < 64; ++i) {
    CHECK(r1.iters_hist_warm[i] == r2.iters_hist_warm[i]);
    CHECK(r1.iters_hist_cold[i] == r2.iters_hist_cold[i]);
  }

  auto check_q_equal = [](const QuantileStats &a, const QuantileStats &b) {
    CHECK(a.p50 == b.p50);
    CHECK(a.p90 == b.p90);
    CHECK(a.p99 == b.p99);
    CHECK(a.p999 == b.p999);
    CHECK(a.max == b.max);
  };
  check_q_equal(r1.rt_D, r2.rt_D);
  check_q_equal(r1.rt_tau, r2.rt_tau);
  check_q_equal(r1.rt_S, r2.rt_S);
  check_q_equal(r1.prim_rho, r2.prim_rho);
  check_q_equal(r1.prim_w, r2.prim_w);
  check_q_equal(r1.prim_s, r2.prim_s);

  REQUIRE(r1.classes.size() == r2.classes.size());
  for (size_t i = 0; i < r1.classes.size(); ++i) {
    CHECK(r1.classes[i].name == r2.classes[i].name);
    CHECK(r1.classes[i].count == r2.classes[i].count);
    CHECK(r1.classes[i].max == r2.classes[i].max);
    CHECK(r1.classes[i].rms == r2.classes[i].rms);
    REQUIRE(r1.classes[i].worst.size() == r2.classes[i].worst.size());
    for (size_t j = 0; j < r1.classes[i].worst.size(); ++j) {
      CHECK(r1.classes[i].worst[j].rho == r2.classes[i].worst[j].rho);
      CHECK(r1.classes[i].worst[j].temp == r2.classes[i].worst[j].temp);
      CHECK(r1.classes[i].worst[j].ye == r2.classes[i].worst[j].ye);
      CHECK(r1.classes[i].worst[j].value == r2.classes[i].worst[j].value);
    }
  }

  CHECK(r1.solves_per_sec_warm > 0.0);
  CHECK(r1.solves_per_sec_cold > 0.0);
}

// ==========================================================================
// 3. con2prim_needs_attention() truth table.
// ==========================================================================

TEST_CASE("con2prim_needs_attention: failed_*/roundtrip drive it, c2p_failed's own class count does not") {
  Con2PrimReport clean;
  clean.status = eeos::Status::ok;
  CHECK_FALSE(eeos::con2prim_needs_attention(clean));

  SUBCASE("n_failed_no_bracket alone forces true") {
    Con2PrimReport r = clean;
    r.n_failed_no_bracket = 1;
    CHECK(eeos::con2prim_needs_attention(r));
  }

  SUBCASE("n_failed_max_iter alone forces true") {
    Con2PrimReport r = clean;
    r.n_failed_max_iter = 1;
    CHECK(eeos::con2prim_needs_attention(r));
  }

  SUBCASE("cold_n_failed_no_bracket alone forces true") {
    Con2PrimReport r = clean;
    r.cold_n_failed_no_bracket = 1;
    CHECK(eeos::con2prim_needs_attention(r));
  }

  SUBCASE("cold_n_failed_max_iter alone forces true") {
    Con2PrimReport r = clean;
    r.cold_n_failed_max_iter = 1;
    CHECK(eeos::con2prim_needs_attention(r));
  }

  SUBCASE("c2p_roundtrip class count forces true") {
    Con2PrimReport r = clean;
    CheckClassResult c;
    c.name = "c2p_roundtrip";
    c.count = 1;
    r.classes.push_back(c);
    CHECK(eeos::con2prim_needs_attention(r));
  }

  SUBCASE("c2p_failed's own class count is not consulted (redundant with the scalar counters)") {
    Con2PrimReport r = clean;
    CheckClassResult c;
    c.name = "c2p_failed";
    c.count = 5; // inconsistent with the (zero) scalar counters, but hand-built here on purpose
    r.classes.push_back(c);
    CHECK_FALSE(eeos::con2prim_needs_attention(r));
  }

  SUBCASE("status alone (fatal) does not force it -- a separate signal, per the header's doc comment") {
    Con2PrimReport r = clean;
    r.status = eeos::Status::fatal;
    CHECK_FALSE(eeos::con2prim_needs_attention(r));
  }

  SUBCASE("every class present but all zero-count, all scalars zero, is still clean") {
    Con2PrimReport r = clean;
    for (const char *name : {"c2p_failed", "c2p_roundtrip"}) {
      CheckClassResult c;
      c.name = name;
      c.count = 0;
      r.classes.push_back(c);
    }
    CHECK_FALSE(eeos::con2prim_needs_attention(r));
  }
}

// ==========================================================================
// 4. Guarded real-table smoke (LS220, default m_B, n_states=2000).
// ==========================================================================

namespace {

const std::string kLS220Path = "tables/LS220_234r_136t_50y_analmu_20091212_SVNr26.h5";

bool table_exists(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

} // namespace

TEST_CASE("check_con2prim: LS220 real table (guarded), default m_B, n=2000 -- runs to completion") {
  const std::string path = eeos_san_table(kLS220Path);
  if (!table_exists(path)) {
    WARN_MESSAGE(false, "LS220 table not found at '" << path << "' -- skipped ('skipped')");
    return;
  }

  RawTable table = eeos::read_stellarcollapse(path);
  const eeos::RepairResult repair_result = eeos::repair_table(table);
  (void)repair_result;
  EntropyEOS adapter = eeos::build_entropy_eos(table); // default m_B (amu)

  Con2PrimCheckOptions copts;
  copts.n_states = eeos_n(2000, 300);
  const Con2PrimReport report = eeos::check_con2prim(adapter, copts);

  CHECK(report.status != eeos::Status::fatal);
  CHECK(std::isfinite(report.rt_D.max));
  CHECK(std::isfinite(report.rt_tau.max));
  CHECK(std::isfinite(report.rt_S.max));
  CHECK(std::isfinite(report.solves_per_sec_warm));
  CHECK(std::isfinite(report.solves_per_sec_cold));

  // Residual multi-root pockets in the spline-safe-repaired real tables are
  // documented (CODE.md "M2 empirical findings" / "Open decisions" #4:
  // accept-and-guard) -- do not assert zero failures here, only report them.
  const CheckClassResult &failed = find_class(report, "c2p_failed");
  const CheckClassResult &roundtrip = find_class(report, "c2p_roundtrip");
  MESSAGE("test_con2prim_audit 4 (LS220): n_newton=" << report.n_newton
          << " n_fallback=" << report.n_fallback << " n_failed_no_bracket=" << report.n_failed_no_bracket
          << " n_failed_max_iter=" << report.n_failed_max_iter << " cold_n_newton=" << report.cold_n_newton
          << " cold_n_fallback=" << report.cold_n_fallback
          << " cold_n_failed_no_bracket=" << report.cold_n_failed_no_bracket
          << " cold_n_failed_max_iter=" << report.cold_n_failed_max_iter
          << " c2p_failed.count=" << failed.count << " c2p_roundtrip.count=" << roundtrip.count
          << " rt_D.max=" << report.rt_D.max << " rt_tau.p99=" << report.rt_tau.p99
          << " rt_tau.max=" << report.rt_tau.max << " rt_S.max=" << report.rt_S.max
          << " solves_per_sec_warm=" << report.solves_per_sec_warm
          << " solves_per_sec_cold=" << report.solves_per_sec_cold);

  std::ostringstream oss;
  report.print(oss);
  CHECK_FALSE(oss.str().empty());
}
