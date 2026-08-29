// tests/test_causal_cap.cpp — unit tests for the M3f causal-cap repair stage
// (entropy_eos/host/repair.hpp, eos-causality-repair.md).
//
// Two layers:
//   (1) causal_envelope(), the stage's edit primitive, against the closed
//       form of eos-causality-repair.md S3 -- the design's stated oracle --
//       plus the identities that closed form is supposed to satisfy
//       (h = 1 + eps + deps/dx, c_s^2 = dln h/dx = cs2_cap, p > 0), and the
//       no-op guarantee on an already-causal profile.
//   (2) repair_table() end to end on synthetic tables carrying the same
//       stiffened high-density corner the real LS220/SRO tables have
//       (host/synthetic.hpp's StiffenDefect): detect -> cap -> clean ->
//       idempotent, "entropy" never edited, untouched nodes bit-identical,
//       and the stage's opt-out behaving as documented.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "entropy_eos/host/repair.hpp"
#include "entropy_eos/host/synthetic.hpp"
#include "entropy_eos/host/table.hpp"
#include "test_scale.hpp"

using eeos::CausalEnvelope;
using eeos::RawTable;
using eeos::RepairOptions;
using eeos::RepairResult;
using eeos::StiffenDefect;
using eeos::SyntheticOptions;

namespace {

// The closed form of eos-causality-repair.md S3: the solution of
// deps/dx = h - 1 - eps with h(x) = h_a e^{c dx}, started from eps_a at
// dx = 0.
double closed_form_eps(double eps_a, double h_a, double c, double dx) {
  return (eps_a + 1.0) * std::exp(-dx) - 1.0 + h_a * (std::exp(c * dx) - std::exp(-dx)) / (1.0 + c);
}

// A manufactured stretch for causal_envelope(): h sampled DOWNWARD from the
// node, h[0] the node and h[steps] the anchor, with h purely exponential at
// rate `c_orig` per unit x = ln rho and step `delta`.
std::vector<double> exponential_stretch(double h_anchor, double c_orig, double delta, int steps) {
  std::vector<double> h(static_cast<size_t>(steps) + 1);
  for (int m = 0; m <= steps; ++m) {
    h[static_cast<size_t>(m)] =
        h_anchor * std::exp(c_orig * static_cast<double>(steps - m) * delta);
  }
  return h;
}

// The dedicated causal-cap synthetic table: the same smooth power-law energy
// excess dirty_synthetic_options() plants, but on a rho axis fine enough
// (160 cells over 10 decades, 0.0625 dex) that a capped profile -- ln h
// growing at cs2_cap per unit ln rho -- advances only ~0.14 per cell, so the
// fitted cubic's second-derivative error stays far below the 1%
// cs2_max/cs2_cap hysteresis and the stage can converge to zero. See
// synthetic.cpp's dirty_synthetic_options() for why the coarse default grid
// cannot.
// The grid size is a CORRECTNESS precondition here, not a sample count, so it
// is deliberately not routed through eeos_n(): shrinking the rho axis would
// coarsen the very resolution the test is about. The T and Ye axes are
// already the smallest the tensor fit and the defect need.
SyntheticOptions stiff_corner_options() {
  SyntheticOptions opts;
  opts.nrho = 160;
  opts.ntemp = 20;
  opts.nye = 8;
  opts.stiffen.push_back(StiffenDefect{"logenergy", 1e13, 2.5, 1.35e16});
  return opts;
}

size_t count_entries_for(const RepairResult &r, const std::string &field) {
  size_t n = 0;
  for (const eeos::RepairEntry &e : r.entries) {
    if (e.field == field) {
      ++n;
    }
  }
  return n;
}

} // namespace

// --- (1) causal_envelope(): the closed-form oracle ---------------------------

TEST_CASE("causal_envelope: reproduces the closed form exactly on a fully binding stretch") {
  const double delta = 0.05;
  const int steps = 30;
  const double h_anchor = 1.2;
  // Physical anchor state: p/rho = h - 1 - eps must be positive.
  const double eps_anchor = 0.05;
  const double c_orig = 1.5; // superluminal original
  const double cap = 0.99;
  const double dx = static_cast<double>(steps) * delta;

  const std::vector<double> h = exponential_stretch(h_anchor, c_orig, delta, steps);
  // The node's own stored eps is what the ORIGINAL profile has there.
  const double eps_node = closed_form_eps(eps_anchor, h_anchor, c_orig, dx);

  const CausalEnvelope env = eeos::causal_envelope(h, eps_node, cap, delta);
  REQUIRE(env.ok);
  CHECK(env.bound);

  const double expected = closed_form_eps(eps_anchor, h_anchor, cap, dx);
  // The per-step integrating-factor recursion telescopes algebraically for an
  // exponential h, so the only error here is floating-point roundoff over 30
  // steps.
  CHECK(std::fabs(env.eps_node - expected) <= 1e-13 * (1.0 + std::fabs(expected)));
  CHECK(env.eps_node < eps_node); // the projection always lowers eps
}

TEST_CASE("causal_envelope: the projected profile satisfies h = 1 + eps + deps/dx, p > 0 and "
          "c_s^2 = cs2_cap") {
  // Project the same stretch truncated at every depth, which gives the
  // projected eps at every point of the stretch, and check it against the
  // closed form there -- to roundoff, at every depth, not just at the node.
  // The identities the closed form is *supposed* to satisfy are then checked
  // analytically (its own exact derivative), so nothing here is weakened by a
  // finite-difference truncation error.
  const double delta = 0.02;
  const int steps = 40;
  const double h_anchor = 1.3;
  const double eps_anchor = 0.1; // p/rho = h - 1 - eps > 0 at the anchor
  const double c_orig = 1.6;
  const double cap = 0.99;

  const std::vector<double> h_full = exponential_stretch(h_anchor, c_orig, delta, steps);

  // First and second d/dx of closed_form_eps at dx, analytic.
  const auto closed_form_deps = [&](double c, double dx) {
    return -(eps_anchor + 1.0) * std::exp(-dx) +
           h_anchor * (c * std::exp(c * dx) + std::exp(-dx)) / (1.0 + c);
  };
  const auto closed_form_d2eps = [&](double c, double dx) {
    return (eps_anchor + 1.0) * std::exp(-dx) +
           h_anchor * (c * c * std::exp(c * dx) - std::exp(-dx)) / (1.0 + c);
  };

  double prev_eps = std::numeric_limits<double>::infinity();
  for (int m = steps; m >= 0; --m) { // walk UP the adiabat (x increasing)
    const std::vector<double> h_tail(h_full.begin() + m, h_full.end());
    const double dx = static_cast<double>(steps - m) * delta;
    const double eps_orig_m = closed_form_eps(eps_anchor, h_anchor, c_orig, dx);
    const CausalEnvelope env = eeos::causal_envelope(h_tail, eps_orig_m, cap, delta);
    REQUIRE(env.ok);

    const double expect = closed_form_eps(eps_anchor, h_anchor, cap, dx);
    CHECK(std::fabs(env.eps_node - expect) <= 1e-13 * (1.0 + std::fabs(expect)));

    // The capped profile's eps is strictly increasing along the adiabat.
    if (m < steps) {
      CHECK(env.eps_node > prev_eps);
    }
    prev_eps = env.eps_node;

    // Identity 1: h = 1 + eps + deps/dx is exactly the capped envelope.
    const double h_from_eps = 1.0 + expect + closed_form_deps(cap, dx);
    const double h_expect = h_anchor * std::exp(cap * dx);
    CHECK(std::fabs(h_from_eps - h_expect) <= 1e-14 * h_expect);

    // Identity 2: p/rho = h - 1 - eps = deps/dx > 0.
    CHECK(closed_form_deps(cap, dx) > 0.0);

    // Identity 3: c_s^2 = dln h/dx = (deps/dx + d2eps/dx2)/h == cap
    // (eos-causality-repair.md S3's "h c_s^2 = 2 rho U_rho + rho^2 U_rhorho"
    // in logs).
    const double cs2 = (closed_form_deps(cap, dx) + closed_form_d2eps(cap, dx)) / h_from_eps;
    CHECK(std::fabs(cs2 - cap) <= 1e-13 * cap);
  }
}

TEST_CASE("causal_envelope: no-op (bitwise) on an already-causal profile") {
  const double delta = 0.05;
  const int steps = 25;
  const double cap = 0.99;
  const double eps_node = 0.7531904;

  for (double c_orig : {0.0, 0.25, 0.5, 0.9}) {
    const std::vector<double> h = exponential_stretch(1.4, c_orig, delta, steps);
    const CausalEnvelope env = eeos::causal_envelope(h, eps_node, cap, delta);
    REQUIRE(env.ok);
    CHECK_FALSE(env.bound);
    // Bitwise: this is what keeps causal nodes untouched and the stage
    // idempotent (repair.hpp, causal-cap step 7c).
    CHECK(env.eps_node == eps_node);
  }
}

TEST_CASE("causal_envelope: a stretch that only binds near the top still lowers eps") {
  // h rises gently (causal) for the lower half and steeply (superluminal) for
  // the upper half; only the upper half may bind.
  const double delta = 0.05;
  const double cap = 0.99;
  std::vector<double> h(21);
  h[20] = 1.2; // anchor
  for (int m = 19; m >= 0; --m) {
    const double rate = (m >= 10) ? 0.3 : 1.8;
    h[static_cast<size_t>(m)] = h[static_cast<size_t>(m + 1)] * std::exp(rate * delta);
  }
  const CausalEnvelope env = eeos::causal_envelope(h, 0.9, cap, delta);
  REQUIRE(env.ok);
  CHECK(env.bound);
  CHECK(env.eps_node < 0.9);
}

TEST_CASE("causal_envelope: degenerate inputs") {
  const std::vector<double> tiny = {1.5};
  CausalEnvelope env = eeos::causal_envelope(tiny, 0.25, 0.99, 0.05);
  CHECK(env.ok);
  CHECK_FALSE(env.bound);
  CHECK(env.eps_node == 0.25);

  env = eeos::causal_envelope(std::vector<double>(), 0.25, 0.99, 0.05);
  CHECK(env.ok);
  CHECK_FALSE(env.bound);

  // Non-positive delta: ill-posed.
  const std::vector<double> h = exponential_stretch(1.2, 1.5, 0.05, 5);
  env = eeos::causal_envelope(h, 0.25, 0.99, 0.0);
  CHECK_FALSE(env.ok);

  // A non-positive h anywhere on the stretch: ill-posed (the stage gives up
  // on the node rather than editing it).
  std::vector<double> bad = h;
  bad[2] = -1.0;
  env = eeos::causal_envelope(bad, 0.25, 0.99, 0.05);
  CHECK_FALSE(env.ok);
}

// --- (2) repair_table() end to end ------------------------------------------

TEST_CASE("repair_table: causal-cap is a no-op on a clean synthetic table") {
  SyntheticOptions opts;
  opts.nrho = eeos_n(160, 60);
  RawTable table = eeos::make_synthetic_table(opts);
  const std::vector<double> logenergy_before = table.field("logenergy");

  const RepairResult r = eeos::repair_table(table);
  CHECK(r.causal_cap.ran);
  CHECK(r.causal_cap.violations_before == 0);
  CHECK(r.causal_cap.violations_after == 0);
  CHECK(r.causal_cap.nodes_capped == 0);
  CHECK_FALSE(r.causal_cap.reverted);
  CHECK(r.causal_cap.cs2_max_seen < 1.0);
  CHECK(r.entries.empty());
  CHECK(table.field("logenergy") == logenergy_before); // bitwise
}

TEST_CASE("repair_table: causal-cap detects and removes a stiffened high-density corner") {
  RawTable table = eeos::make_synthetic_table(stiff_corner_options());
  RawTable reference = eeos::make_synthetic_table(stiff_corner_options());

  const RepairResult r = eeos::repair_table(table);
  const RepairResult::CausalCapSummary &c = r.causal_cap;

  REQUIRE(c.ran);
  CHECK_FALSE(c.reverted);
  CHECK(c.violations_before > 0);       // the corner was detected ...
  CHECK(c.violations_after == 0);       // ... and is gone at (4,4,4)
  CHECK(c.nodes_capped > 0);
  CHECK(c.trace_giveups == 0);
  CHECK(c.interior_untouched == 0);
  CHECK(c.cs2_max_seen < 1.0);
  // The stage must never trade the T-solve's hard requirement for causality.
  CHECK(c.mono_logenergy_after <= c.mono_logenergy_before);

  // Only "logenergy" is ever edited by this stage; this table has no other
  // defect, so "entropy" must carry no entry at all.
  CHECK(count_entries_for(r, "entropy") == 0);
  CHECK(count_entries_for(r, "logenergy") == r.entries.size());
  CHECK(table.field("entropy") == reference.field("entropy")); // bitwise

  // Every changed node lowered the stored energy (the envelope is one-sided)
  // and sits in the high-density part of the grid.
  size_t min_irho = table.nrho();
  for (const eeos::RepairEntry &e : r.entries) {
    CHECK(e.new_value < e.old_value);
    min_irho = std::min(min_irho, e.irho);
  }
  CHECK(min_irho > table.nrho() / 2);

  // Nodes below the treated corner are bit-identical to the input.
  const std::vector<double> &now = table.field("logenergy");
  const std::vector<double> &was = reference.field("logenergy");
  for (size_t kYe = 0; kYe < table.nye(); ++kYe) {
    for (size_t jT = 0; jT < table.ntemp(); ++jT) {
      for (size_t irho = 0; irho + 1 < min_irho; ++irho) {
        const size_t idx = table.index(irho, jT, kYe);
        CHECK(now[idx] == was[idx]);
      }
    }
  }
}

TEST_CASE("repair_table: causal-cap is idempotent on the stiffened-corner table") {
  RawTable table = eeos::make_synthetic_table(stiff_corner_options());
  const RepairResult first = eeos::repair_table(table);
  REQUIRE(first.causal_cap.violations_after == 0);
  const std::vector<double> after_first = table.field("logenergy");

  const RepairResult second = eeos::repair_table(table);
  CHECK(second.entries.empty());
  CHECK(second.status == eeos::Status::ok);
  CHECK(second.causal_cap.ran);
  CHECK(second.causal_cap.nodes_capped == 0);
  CHECK(second.causal_cap.violations_before == 0);
  CHECK(second.causal_cap.violations_after == 0);
  CHECK(table.field("logenergy") == after_first); // bitwise
}

TEST_CASE("repair_table: the causal-cap loop runs while it improves, not on a fixed budget") {
  // M3j. The loop stops on a property of the DATA -- a clean audit, nothing
  // left in scope, a round that writes nothing, or two consecutive rounds
  // that fail to beat the best state seen -- and options.causal_rounds_max is
  // only a runaway backstop above that. Two consequences, both checked here.
  RawTable table = eeos::make_synthetic_table(stiff_corner_options());
  const RepairResult r = eeos::repair_table(table);
  REQUIRE(r.causal_cap.ran);
  REQUIRE(r.causal_cap.violations_after == 0);

  // (a) The loop stopped on its own, nowhere near the ceiling. The history
  // holds one entry per main-loop audit plus the final (4,4,4) verification,
  // so its length minus one is the number of main-loop rounds actually run.
  const RepairOptions defaults;
  CHECK(static_cast<int>(r.causal_cap.rounds_violation_history.size()) - 1 <
        defaults.causal_rounds_max);

  // (b) A ceiling that DOES cut the loop off discards the round it cut away:
  // that round's projection is never audited, so the best state is the one
  // before it. At the extreme -- a ceiling of 1 -- the whole stage becomes a
  // silent no-op: it reports no change and leaves the corner exactly as
  // acausal as it found it, while the same table under the default ceiling
  // comes back repaired and clean. This is the pre-M3j failure mode in
  // miniature: SFHo's loop was still descending when its 8-round budget
  // expired, so a single eos_repair run left work behind that only a second
  // run could pick up (CODE.md "DD2 / SFHo empirical findings").
  RawTable truncated = eeos::make_synthetic_table(stiff_corner_options());
  RepairOptions budget1;
  budget1.causal_rounds_max = 1;
  const RepairResult rt = eeos::repair_table(truncated, budget1);
  CHECK(rt.causal_cap.ran);
  CHECK_FALSE(rt.causal_cap.reverted);
  CHECK(rt.causal_cap.rounds_used == 0);
  CHECK(rt.causal_cap.nodes_capped == 0);
  CHECK(rt.causal_cap.violations_after == rt.causal_cap.violations_before);
  CHECK(rt.causal_cap.violations_after > r.causal_cap.violations_after);
  CHECK(count_entries_for(rt, "logenergy") == 0);
}

TEST_CASE("repair_table: --no-causal-cap leaves the corner alone and never changes entropy") {
  RawTable with_cap = eeos::make_synthetic_table(stiff_corner_options());
  RawTable without_cap = eeos::make_synthetic_table(stiff_corner_options());

  RepairOptions off;
  off.causal_cap = false;

  const RepairResult r_on = eeos::repair_table(with_cap);
  const RepairResult r_off = eeos::repair_table(without_cap, off);

  CHECK_FALSE(r_off.causal_cap.ran);
  CHECK(r_off.causal_cap.rounds_violation_history.empty());
  CHECK(r_off.causal_cap.nodes_capped == 0);

  // The stage is the ONLY difference between the two runs, and it edits only
  // "logenergy": the two runs' "entropy" fields must be bit-identical.
  CHECK(with_cap.field("entropy") == without_cap.field("entropy"));
  CHECK(count_entries_for(r_on, "entropy") == count_entries_for(r_off, "entropy"));
  CHECK(count_entries_for(r_on, "logenergy") > count_entries_for(r_off, "logenergy"));
}

TEST_CASE("repair_table: an unusable cs2_cap is a caller error, reported before anything moves") {
  SyntheticOptions opts;
  opts.nrho = 24;
  opts.ntemp = 12;
  opts.nye = 6;
  RawTable table = eeos::make_synthetic_table(opts);
  const std::vector<double> before = table.field("logenergy");

  for (double bad_cap : {0.0, -0.5, 1.0, 1.5}) {
    RepairOptions o;
    o.cs2_cap = bad_cap;
    CHECK_THROWS_AS(eeos::repair_table(table, o), std::invalid_argument);
    CHECK(table.field("logenergy") == before); // nothing was modified
  }

  // With the stage off the same value is simply irrelevant.
  RepairOptions off;
  off.causal_cap = false;
  off.cs2_cap = 1.5;
  CHECK_NOTHROW(eeos::repair_table(table, off));
}

TEST_CASE("repair_table: causal-cap is deterministic (independent builds agree bit for bit)") {
  RawTable a = eeos::make_synthetic_table(stiff_corner_options());
  RawTable b = eeos::make_synthetic_table(stiff_corner_options());

  const RepairResult ra = eeos::repair_table(a);
  const RepairResult rb = eeos::repair_table(b);

  REQUIRE(ra.entries.size() == rb.entries.size());
  for (size_t i = 0; i < ra.entries.size(); ++i) {
    CHECK(ra.entries[i].field == rb.entries[i].field);
    CHECK(ra.entries[i].irho == rb.entries[i].irho);
    CHECK(ra.entries[i].jT == rb.entries[i].jT);
    CHECK(ra.entries[i].kYe == rb.entries[i].kYe);
    CHECK(ra.entries[i].new_value == rb.entries[i].new_value);
  }
  CHECK(ra.causal_cap.nodes_capped == rb.causal_cap.nodes_capped);
  CHECK(ra.causal_cap.violations_after == rb.causal_cap.violations_after);
  CHECK(a.field("logenergy") == b.field("logenergy"));
}

TEST_CASE("repair_table: causal-cap reports, but never edits, the dirty preset's coarse-grid "
          "residual") {
  // dirty_synthetic_options()'s rho axis is deliberately too coarse for this
  // stage to converge (see synthetic.cpp): what must hold is that the corner
  // is detected, substantially reduced, never made worse, and that a second
  // run's causal stage changes nothing.
  RawTable table = eeos::make_synthetic_table(eeos::dirty_synthetic_options());
  const RepairResult first = eeos::repair_table(table);
  const RepairResult::CausalCapSummary &c = first.causal_cap;

  CHECK(c.ran);
  CHECK(c.violations_before > 0);
  CHECK(c.violations_after < c.violations_before);
  CHECK(c.mono_logenergy_after <= c.mono_logenergy_before);

  // Even on this messy table -- where the entropy WiggleDefect makes sigma
  // nearly flat in places and c_s^2 blows up in the resulting pockets -- the
  // stage must leave "entropy" byte-for-byte as the stage-free run does.
  RawTable no_cap = eeos::make_synthetic_table(eeos::dirty_synthetic_options());
  RepairOptions off;
  off.causal_cap = false;
  const RepairResult first_off = eeos::repair_table(no_cap, off);
  CHECK(table.field("entropy") == no_cap.field("entropy"));
  CHECK(count_entries_for(first, "entropy") == count_entries_for(first_off, "entropy"));

  const std::vector<double> after_first = table.field("logenergy");
  const RepairResult second = eeos::repair_table(table);
  CHECK(second.causal_cap.nodes_capped == 0);
  // The residual is entirely in interior (non-edge-anchored) runs, which this
  // stage reports and never edits.
  CHECK(second.causal_cap.interior_untouched > 0);
}
