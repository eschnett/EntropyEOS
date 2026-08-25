// tests/test_repair.cpp — unit tests for entropy_eos::repair_table /
// repair_column (entropy_eos/host/repair.hpp).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "entropy_eos/core/bspline_eval.hpp"
#include "entropy_eos/host/bspline_fit.hpp"
#include "entropy_eos/host/repair.hpp"
#include "entropy_eos/host/synthetic.hpp"
#include "entropy_eos/host/table.hpp"

using eeos::RawTable;
using eeos::RepairEntry;
using eeos::RepairOptions;
using eeos::RepairResult;
using eeos::SeededViolation;
using eeos::SyntheticOptions;

// --- (1) PAVA hand-worked cases ---------------------------------------------
//
// repair_column(col, 0.0) isolates the pure PAVA result: with min_slope ==
// 0, the strictification pass is v[j] = max(v[j], v[j-1]), which cannot
// change anything since the PAVA output is already non-decreasing (see the
// header comment on repair_column).

TEST_CASE("repair_column: hand-worked PAVA case {1,3,2} -> {1,2.5,2.5}") {
  std::vector<double> col = {1.0, 3.0, 2.0};
  eeos::repair_column(col, 0.0);
  REQUIRE(col.size() == 3);
  CHECK(col[0] == 1.0);
  CHECK(col[1] == 2.5);
  CHECK(col[2] == 2.5);
}

TEST_CASE("repair_column: hand-worked PAVA case {3,1,2} -> {2,2,2}") {
  std::vector<double> col = {3.0, 1.0, 2.0};
  eeos::repair_column(col, 0.0);
  REQUIRE(col.size() == 3);
  CHECK(col[0] == 2.0);
  CHECK(col[1] == 2.0);
  CHECK(col[2] == 2.0);
}

TEST_CASE("repair_column: already-sorted input is unchanged") {
  const std::vector<double> input = {-3.0, -1.0, 0.5, 0.5, 2.0, 7.25};
  std::vector<double> col = input;
  eeos::repair_column(col, 0.0);
  REQUIRE(col.size() == input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    CHECK(col[i] == input[i]);
  }
}

// --- (2) property test against an independent O(n^2) reference -------------

namespace {

// Independent reference PAVA: repeatedly scan for the first adjacent pair of
// blocks that is out of order, merge it, and rescan from the beginning,
// until no violation remains, then re-expand block means. O(n^2) or worse,
// but deliberately simple and structurally different from the stack-based
// production algorithm in repair_column -- except that restarting the scan
// from the left after every merge makes it resolve violations in the same
// left-to-right, cascade-backward order as the stack algorithm, so the two
// are expected to agree bit-for-bit, not just approximately.
std::vector<double> reference_pava(const std::vector<double> &input) {
  std::vector<double> mean(input.begin(), input.end());
  std::vector<size_t> count(input.size(), 1);

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i + 1 < mean.size(); ++i) {
      if (mean[i] > mean[i + 1]) {
        const size_t merged_count = count[i] + count[i + 1];
        const double merged_mean =
            (mean[i] * static_cast<double>(count[i]) + mean[i + 1] * static_cast<double>(count[i + 1])) /
            static_cast<double>(merged_count);
        mean[i] = merged_mean;
        count[i] = merged_count;
        mean.erase(mean.begin() + i + 1);
        count.erase(count.begin() + i + 1);
        changed = true;
        break;
      }
    }
  }

  std::vector<double> result;
  result.reserve(input.size());
  for (size_t p = 0; p < mean.size(); ++p) {
    for (size_t r = 0; r < count[p]; ++r) {
      result.push_back(mean[p]);
    }
  }
  return result;
}

} // namespace

TEST_CASE("repair_column: PAVA stage agrees exactly with an independent O(n^2) reference") {
  std::mt19937 rng(20260824);
  std::uniform_int_distribution<size_t> pick_len(1, 12);
  std::uniform_real_distribution<double> pick_val(-50.0, 50.0);

  for (int trial = 0; trial < 200; ++trial) {
    const size_t n = pick_len(rng);
    std::vector<double> input(n);
    for (size_t i = 0; i < n; ++i) {
      input[i] = pick_val(rng);
    }

    const std::vector<double> expected = reference_pava(input);

    // min_slope == 0 so strictification cannot perturb the PAVA output (see
    // repair_column's header comment) -- this isolates the PAVA stage.
    std::vector<double> got = input;
    eeos::repair_column(got, 0.0);

    REQUIRE(got.size() == expected.size());
    for (size_t i = 0; i < n; ++i) {
      CHECK(got[i] == expected[i]);
    }
  }
}

// --- (3) strict min-slope monotonicity --------------------------------------

TEST_CASE("repair_column: result satisfies the strict minimum slope") {
  std::mt19937 rng(7);
  std::uniform_int_distribution<size_t> pick_len(2, 25);
  std::uniform_real_distribution<double> pick_val(-10.0, 10.0);
  const double min_slope = 0.01;

  for (int trial = 0; trial < 200; ++trial) {
    const size_t n = pick_len(rng);
    std::vector<double> col(n);
    for (size_t i = 0; i < n; ++i) {
      col[i] = pick_val(rng);
    }
    eeos::repair_column(col, min_slope);
    for (size_t j = 1; j < n; ++j) {
      CHECK(col[j] >= col[j - 1] + min_slope * 0.999);
    }
  }
}

// --- helpers for the RawTable-level tests -----------------------------------

namespace {

RawTable make_small_table(size_t nrho, size_t ntemp, size_t nye) {
  std::vector<double> logrho(nrho), logtemp(ntemp), ye(nye);
  for (size_t i = 0; i < nrho; ++i) {
    logrho[i] = 5.0 + static_cast<double>(i);
  }
  for (size_t j = 0; j < ntemp; ++j) {
    logtemp[j] = -1.0 + 0.1 * static_cast<double>(j);
  }
  for (size_t k = 0; k < nye; ++k) {
    ye[k] = 0.1 + 0.05 * static_cast<double>(k);
  }

  RawTable t;
  t.set_axes(logrho, logtemp, ye);

  const size_t n = nrho * ntemp * nye;
  std::vector<double> entropy(n), logenergy(n);
  for (size_t kYe = 0; kYe < nye; ++kYe) {
    for (size_t jT = 0; jT < ntemp; ++jT) {
      for (size_t irho = 0; irho < nrho; ++irho) {
        const size_t idx = t.index(irho, jT, kYe);
        // Strictly increasing in jT by construction.
        entropy[idx] = 1.0 + static_cast<double>(jT) + 0.01 * static_cast<double>(irho) +
                       0.001 * static_cast<double>(kYe);
        logenergy[idx] = 18.0 + 0.1 * static_cast<double>(jT);
      }
    }
  }
  t.add_field("entropy", entropy);
  t.add_field("logenergy", logenergy);
  return t;
}

// Bitwise comparison (via memcpy into an integer of the same width) rather
// than `==`, so that a NaN preserved untouched still counts as "identical"
// -- IEEE 754 makes NaN != NaN under the ordinary comparison operators.
bool same_bits(double a, double b) {
  static_assert(sizeof(double) == sizeof(uint64_t), "double must be 64-bit");
  uint64_t bits_a, bits_b;
  std::memcpy(&bits_a, &a, sizeof(double));
  std::memcpy(&bits_b, &b, sizeof(double));
  return bits_a == bits_b;
}

bool fields_bit_identical(const RawTable &a, const RawTable &b, const std::string &field) {
  const std::vector<double> &fa = a.field(field);
  const std::vector<double> &fb = b.field(field);
  if (fa.size() != fb.size()) return false;
  for (size_t i = 0; i < fa.size(); ++i) {
    if (!same_bits(fa[i], fb[i])) return false;
  }
  return true;
}

bool entries_equal(const std::vector<RepairEntry> &a, const std::vector<RepairEntry> &b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].field != b[i].field || a[i].irho != b[i].irho || a[i].jT != b[i].jT ||
        a[i].kYe != b[i].kYe || a[i].old_value != b[i].old_value || a[i].new_value != b[i].new_value) {
      return false;
    }
  }
  return true;
}

} // namespace

// --- (4) synthetic end-to-end -----------------------------------------------

TEST_CASE("repair_table: synthetic table with seeded violations is repaired correctly") {
  SyntheticOptions opts;
  opts.nrho = 10;
  opts.ntemp = 12;
  opts.nye = 4;

  // Deltas negative and large relative to the natural scale of each field
  // (entropy ~ O(10) kB/baryon, logenergy ~ O(18-20)) and applied at jT > 0
  // so there is a predecessor to violate against.
  const size_t irho_s = 6, jT_s = 5, kYe_s = 2;
  const double delta_s = -1000.0;
  const size_t irho_e = 3, jT_e = 7, kYe_e = 1;
  const double delta_e = -50.0;

  opts.seed = {
      SeededViolation{"entropy", irho_s, jT_s, kYe_s, delta_s},
      SeededViolation{"logenergy", irho_e, jT_e, kYe_e, delta_e},
  };

  RawTable seeded = eeos::make_synthetic_table(opts);
  const RawTable seeded_copy = seeded; // snapshot before repair, per spec

  RepairResult result = eeos::repair_table(seeded);

  CHECK(result.status == eeos::Status::repaired);
  CHECK_FALSE(result.entries.empty());

  // (a) strict monotonicity holds on every column of both repaired fields.
  RepairOptions default_opts;
  for (const std::string &field : {std::string("entropy"), std::string("logenergy")}) {
    const double min_slope =
        field == "entropy" ? default_opts.min_slope_entropy : default_opts.min_slope_logenergy;
    const std::vector<double> &data = seeded.field(field);
    for (size_t kYe = 0; kYe < seeded.nye(); ++kYe) {
      for (size_t irho = 0; irho < seeded.nrho(); ++irho) {
        for (size_t jT = 1; jT < seeded.ntemp(); ++jT) {
          const double prev = data[seeded.index(irho, jT - 1, kYe)];
          const double cur = data[seeded.index(irho, jT, kYe)];
          CHECK(cur >= prev + min_slope * 0.999);
        }
      }
    }
  }

  // (b) entries include the seeded locations.
  auto has_entry_at = [&](const std::string &field, size_t irho, size_t jT, size_t kYe) {
    for (const RepairEntry &e : result.entries) {
      if (e.field == field && e.irho == irho && e.jT == jT && e.kYe == kYe) return true;
    }
    return false;
  };
  CHECK(has_entry_at("entropy", irho_s, jT_s, kYe_s));
  CHECK(has_entry_at("logenergy", irho_e, jT_e, kYe_e));

  // (c) compare against the seeded-but-unrepaired snapshot: every changed
  // index must be listed in result.entries with matching old/new values, and
  // every point NOT listed must be bit-identical to the snapshot.
  for (const std::string &field : {std::string("entropy"), std::string("logenergy")}) {
    const std::vector<double> &before = seeded_copy.field(field);
    const std::vector<double> &after = seeded.field(field);
    REQUIRE(before.size() == after.size());

    std::vector<bool> listed(before.size(), false);
    for (const RepairEntry &e : result.entries) {
      if (e.field != field) continue;
      const size_t idx = seeded.index(e.irho, e.jT, e.kYe);
      CHECK(e.old_value == before[idx]);
      CHECK(e.new_value == after[idx]);
      listed[idx] = true;
    }

    for (size_t idx = 0; idx < before.size(); ++idx) {
      if (before[idx] != after[idx]) {
        CHECK(listed[idx]);
      } else {
        CHECK_FALSE(listed[idx]); // an entry must only be recorded on an actual change
      }
    }
  }

  // logpress was not a listed field and must be completely untouched; a
  // freshly generated table with the same grid but no seeded defects has
  // the same logpress values (the seed only perturbs "entropy"/"logenergy").
  SyntheticOptions clean_opts = opts;
  clean_opts.seed.clear();
  RawTable clean = eeos::make_synthetic_table(clean_opts);
  CHECK(fields_bit_identical(clean, seeded, "logpress"));
}

// --- (5) idempotence ---------------------------------------------------------

TEST_CASE("repair_table: idempotent -- a second repair reports zero changes") {
  SyntheticOptions opts;
  opts.nrho = 8;
  opts.ntemp = 10;
  opts.nye = 3;
  opts.seed = {
      SeededViolation{"entropy", 4, 4, 1, -500.0},
      SeededViolation{"logenergy", 2, 6, 0, -20.0},
  };
  RawTable table = eeos::make_synthetic_table(opts);

  RepairResult first = eeos::repair_table(table);
  CHECK_FALSE(first.entries.empty());

  const RawTable after_first = table;

  RepairResult second = eeos::repair_table(table);
  CHECK(second.entries.empty());
  CHECK(second.status == eeos::Status::ok);

  CHECK(fields_bit_identical(after_first, table, "entropy"));
  CHECK(fields_bit_identical(after_first, table, "logenergy"));
}

// --- (6) determinism ---------------------------------------------------------

TEST_CASE("repair_table: deterministic -- identical inputs give identical entries") {
  SyntheticOptions opts;
  opts.nrho = 9;
  opts.ntemp = 11;
  opts.nye = 3;
  opts.seed = {
      SeededViolation{"entropy", 5, 5, 2, -300.0},
      SeededViolation{"entropy", 1, 8, 0, -300.0},
      SeededViolation{"logenergy", 7, 3, 1, -30.0},
  };

  RawTable table_a = eeos::make_synthetic_table(opts);
  RawTable table_b = eeos::make_synthetic_table(opts);

  RepairResult result_a = eeos::repair_table(table_a);
  RepairResult result_b = eeos::repair_table(table_b);

  CHECK(entries_equal(result_a.entries, result_b.entries));
  CHECK(fields_bit_identical(table_a, table_b, "entropy"));
  CHECK(fields_bit_identical(table_a, table_b, "logenergy"));
}

// --- (7) structural problems throw -------------------------------------------

TEST_CASE("repair_table: a missing listed field throws std::runtime_error") {
  RawTable table = make_small_table(3, 4, 2);
  RepairOptions opts;
  opts.fields = {"entropy", "nonexistent_field"};
  CHECK_THROWS_AS(eeos::repair_table(table, opts), std::runtime_error);
}

TEST_CASE("repair_table: a NaN in a listed field throws std::runtime_error") {
  RawTable table = make_small_table(3, 4, 2);
  table.field("entropy")[table.index(1, 2, 0)] = std::numeric_limits<double>::quiet_NaN();

  const RawTable before = table;
  CHECK_THROWS_AS(eeos::repair_table(table), std::runtime_error);

  // Nothing should have been modified by the failed attempt (validation
  // happens before any field is touched).
  CHECK(fields_bit_identical(before, table, "entropy"));
  CHECK(fields_bit_identical(before, table, "logenergy"));
}

TEST_CASE("repair_table: an unsupported field name throws std::invalid_argument") {
  RawTable table = make_small_table(3, 4, 2);
  table.add_field("mystery", std::vector<double>(3 * 4 * 2, 1.0));
  RepairOptions opts;
  opts.fields = {"mystery"};
  CHECK_THROWS_AS(eeos::repair_table(table, opts), std::invalid_argument);
}

// --- (8) print() smoke test --------------------------------------------------

TEST_CASE("RepairResult::print: produces non-empty, sane output") {
  SyntheticOptions opts;
  opts.nrho = 6;
  opts.ntemp = 8;
  opts.nye = 2;
  opts.seed = {SeededViolation{"entropy", 2, 3, 0, -200.0}};
  RawTable table = eeos::make_synthetic_table(opts);

  RepairResult result = eeos::repair_table(table);

  std::ostringstream out;
  result.print(out);
  const std::string text = out.str();
  CHECK_FALSE(text.empty());
  CHECK(text.find("repair_table") != std::string::npos);
  CHECK(text.find("entropy") != std::string::npos);
  CHECK(text.find("logenergy") != std::string::npos);

  // A clean table's print() should say so and report zero modifications.
  RawTable clean_table = eeos::make_synthetic_table(SyntheticOptions{});
  RepairResult clean_result = eeos::repair_table(clean_table);
  std::ostringstream clean_out;
  clean_result.print(clean_out);
  CHECK(clean_out.str().find("no changes") != std::string::npos);
}

// --- (9) M2c-prime spline-safe repair ---------------------------------------
//
// RepairOptions::spline_safe defaults to true (see repair.hpp); every test
// above already runs with it on and still passes, so it is not disabled
// anywhere in this file per this milestone's instructions ("if any existing
// test's expectations depend on plain PAVA output, set spline_safe=false
// there explicitly" -- none do here: the small hand-worked/synthetic
// columns above are all too short or too far from a plateau-then-jump shape
// to trigger the loop, so its presence changes nothing for them).

namespace {

// A single-(irho,kYe)-column table (nrho=1, nye=1, ntemp=col.size()) holding
// `col` under `field`, for tests that only care about repair_table()'s
// per-column behavior in isolation.
RawTable make_single_column_table(const std::string &field, const std::vector<double> &col) {
  const std::vector<double> logrho = {5.0};
  const std::vector<double> ye = {0.2};
  std::vector<double> logtemp(col.size());
  for (size_t j = 0; j < col.size(); ++j) {
    logtemp[j] = -1.0 + 0.1 * static_cast<double>(j);
  }

  RawTable t;
  t.set_axes(logrho, logtemp, ye);
  t.add_field(field, col);
  return t;
}

// The minimum S' of `data`'s fresh not-a-knot cubic B-spline fit (unit
// spacing, x0=0, h=1 -- same convention repair.cpp's spline-safe loop uses)
// over the same refined-sample set repair.hpp's doc comment describes:
// j + m/refine for j = 0..n-2, m = 0..refine-1, plus the last node.
double fresh_fit_min_slope(const std::vector<double> &data, int refine = 4) {
  const int n = static_cast<int>(data.size());
  const std::vector<double> coeffs = eeos::fit_bspline_1d(data);
  const eeos::BsplineView1 view{coeffs.data(), n, 0.0, 1.0};

  double min_fx = std::numeric_limits<double>::infinity();
  for (int j = 0; j + 1 < n; ++j) {
    for (int m = 0; m < refine; ++m) {
      const double x = static_cast<double>(j) + static_cast<double>(m) / static_cast<double>(refine);
      min_fx = std::min(min_fx, eeos::bspline_eval1(view, x).fx);
    }
  }
  min_fx = std::min(min_fx, eeos::bspline_eval1(view, static_cast<double>(n - 1)).fx);
  return min_fx;
}

// Hand-built column: a short smooth ramp, a long near-plateau (20 equal
// values -- PAVA leaves it flat, strictification only adds a
// min_slope-sized staircase), an abrupt +1.0 jump, then another short
// smooth ramp. This is the motivating pattern from eos-adapter-F-to-U.md
// S4 / the M2c-prime work order: the raw data is (after repair_column())
// strictly increasing, but the near-zero-slope plateau sits immediately
// next to a much steeper recovery, and a C^2 not-a-knot cubic B-spline fit
// of that shape rings non-monotone between nodes right at the boundary.
std::vector<double> plateau_then_jump_column() {
  std::vector<double> col;
  double v = 0.0;
  for (int i = 0; i < 5; ++i) {
    col.push_back(v);
    v += 0.05;
  }
  const double plateau_val = v;
  for (int i = 0; i < 20; ++i) {
    col.push_back(plateau_val);
  }
  col.push_back(plateau_val + 1.0);
  v = plateau_val + 1.0;
  for (int i = 0; i < 5; ++i) {
    v += 0.05;
    col.push_back(v);
  }
  return col;
}

} // namespace

TEST_CASE("repair_table: spline-safe removes a plateau-then-jump spline-monotonicity violation "
          "that plain PAVA + strictify leaves behind") {
  const std::vector<double> col = plateau_then_jump_column();

  // Sanity: the pattern is real, i.e. a *plain* repair_column() pass alone
  // (spline_safe's raw material) already produces a fresh fit with a
  // non-positive minimum slope -- otherwise this test would not be
  // exercising anything.
  {
    std::vector<double> base = col;
    eeos::repair_column(base, 1e-8);
    CHECK(fresh_fit_min_slope(base) <= 0.0);
  }

  // spline_safe = false: repair_table() only runs the base PAVA +
  // strictify pass, so the violation survives.
  {
    RawTable table = make_single_column_table("entropy", col);
    RepairOptions opts;
    opts.fields = {"entropy"};
    opts.spline_safe = false;
    eeos::repair_table(table, opts);
    CHECK(fresh_fit_min_slope(table.field("entropy")) <= 0.0);
  }

  // spline_safe = true (default): the smoothing loop removes it -- a fresh
  // fit of the repaired column has S' > spline_slope_floor (0.0) at every
  // refined sample.
  {
    RawTable table = make_single_column_table("entropy", col);
    RepairOptions opts;
    opts.fields = {"entropy"};
    opts.spline_safe = true;
    const RepairResult result = eeos::repair_table(table, opts);
    CHECK(fresh_fit_min_slope(table.field("entropy")) > 0.0);

    // The loop actually ran and reports it in the summary.
    REQUIRE(result.summaries.size() == 1);
    CHECK(result.summaries[0].spline_columns_smoothed == 1);
    CHECK(result.summaries[0].spline_rounds_used_max > 0);
    CHECK(result.summaries[0].spline_columns_still_violating == 0);
  }
}

TEST_CASE("repair_table: idempotent with spline_safe on (repair_table() and a synthetic-dirty "
          "end-to-end)") {
  // (a) repair_table() applied twice to the same seeded-violation table.
  {
    SyntheticOptions opts;
    opts.nrho = 8;
    opts.ntemp = 10;
    opts.nye = 3;
    opts.seed = {
        SeededViolation{"entropy", 4, 4, 1, -500.0},
        SeededViolation{"logenergy", 2, 6, 0, -20.0},
    };
    RawTable table = eeos::make_synthetic_table(opts);

    const RepairOptions ropts; // spline_safe = true (default)
    const RepairResult first = eeos::repair_table(table, ropts);
    CHECK_FALSE(first.entries.empty());

    const RawTable after_first = table;
    const RepairResult second = eeos::repair_table(table, ropts);
    CHECK(second.entries.empty());
    CHECK(second.status == eeos::Status::ok);
    CHECK(fields_bit_identical(after_first, table, "entropy"));
    CHECK(fields_bit_identical(after_first, table, "logenergy"));
  }

  // (b) synthetic-dirty end-to-end: the fixed LS220/SRO-mimicking defect
  // preset (near-plateaus, wiggle clusters, ...) is exactly the kind of
  // input the spline-safe loop targets, so this exercises idempotence
  // through the loop itself, not just the base pass.
  {
    RawTable table = eeos::make_synthetic_table(eeos::dirty_synthetic_options());
    const RepairResult first = eeos::repair_table(table);
    CHECK_FALSE(first.entries.empty());

    const RawTable after_first = table;
    const RepairResult second = eeos::repair_table(table);
    CHECK(second.entries.empty());
    CHECK(second.status == eeos::Status::ok);
    CHECK(fields_bit_identical(after_first, table, "entropy"));
    CHECK(fields_bit_identical(after_first, table, "logenergy"));
  }
}

TEST_CASE("repair_table: deterministic with spline_safe on (synthetic-dirty, independent builds)") {
  RawTable table_a = eeos::make_synthetic_table(eeos::dirty_synthetic_options());
  RawTable table_b = eeos::make_synthetic_table(eeos::dirty_synthetic_options());

  const RepairResult result_a = eeos::repair_table(table_a);
  const RepairResult result_b = eeos::repair_table(table_b);

  CHECK_FALSE(result_a.entries.empty());
  CHECK(entries_equal(result_a.entries, result_b.entries));
  CHECK(fields_bit_identical(table_a, table_b, "entropy"));
  CHECK(fields_bit_identical(table_a, table_b, "logenergy"));
}
