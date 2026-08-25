#!/usr/bin/env bash
# tests/integration.sh -- end-to-end integration tests for the M1 tools
# (tools/eos_repair, tools/eos_test) over the entropy_eos library.
#
# Run from the repository root (paths below are relative to the CWD);
# `make integration` does this after building the tools. Part A (synthetic)
# is table-free and always runs, including in CI. Its second half (steps
# 8-11 below) drives --synthetic-dirty, a fixed deterministic defect preset
# (entropy_eos/host/synthetic.hpp's dirty_synthetic_options()) that mirrors
# pathologies actually found in the real LS220/SRO stellarcollapse tables --
# a handful of non-finite cs2/gamma points, a clustered non-monotone-T
# entropy patch, a near-flat logenergy plateau, and slightly negative
# cold-corner entropies -- so CI (which has no tables/*.h5) still exercises
# eos_test/eos_repair against them. Part B exercises the two real
# stellarcollapse-format tables under tables/*.h5 when present on this
# machine, and is skipped gracefully (not failed) when they are absent.

set -euo pipefail

REPAIR=./tools/eos_repair
TEST=./tools/eos_test

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

# expect_exit N cmd... -- runs cmd..., compares its exit code against N,
# prints PASS/FAIL, and fails the whole script (exit 1) on a mismatch.
expect_exit() {
  local expected="$1"
  shift
  local actual=0
  set +e
  "$@"
  actual=$?
  set -e
  if [ "$actual" -eq "$expected" ]; then
    echo "PASS (exit $actual): $*"
  else
    echo "FAIL (expected exit $expected, got $actual): $*"
    exit 1
  fi
}

echo "=== Part A: synthetic (table-free, CI-safe) ==="

# 1-6: --synthetic-seeded (four seeded monotonicity violations) round trip.
expect_exit 0 "$TEST" --synthetic
expect_exit 1 "$TEST" --synthetic-seeded --write-synthetic "$T/seeded.h5"
expect_exit 1 "$REPAIR" --check-only "$T/seeded.h5"
expect_exit 1 "$REPAIR" "$T/seeded.h5" "$T/repaired.h5" --log "$T/repair.log"
expect_exit 0 "$TEST" "$T/repaired.h5"                     # clean after repair
expect_exit 0 "$REPAIR" --check-only "$T/repaired.h5"      # idempotent through files

# 7:
if grep -q "logenergy" "$T/repair.log"; then
  echo "PASS: repair.log mentions a repaired field (logenergy)"
else
  echo "FAIL: repair.log does not mention 'logenergy'"
  exit 1
fi

# 8-11: --synthetic-dirty (LS220/SRO-mimicking defect preset) round trip.
# Unlike the seeded case above, repair cannot make this table fully clean by
# design: "entropy_negative" (the offset cold corner) and "nonfinite_cs2"/
# "nonfinite_gamma" (the planted Inf/NaN) are untouched by repair_table(),
# which only ever acts on "entropy"/"logenergy" monotonicity -- so eos_test
# on the repaired table is expected to still exit 1, and its CSV dump is
# used below to confirm exactly which classes persist.
expect_exit 1 "$TEST" --synthetic-dirty --write-synthetic "$T/dirty.h5"
expect_exit 1 "$REPAIR" "$T/dirty.h5" "$T/dirty_rep.h5"
expect_exit 0 "$REPAIR" --check-only "$T/dirty_rep.h5"
expect_exit 1 "$TEST" "$T/dirty_rep.h5" --csv "$T/dirty_rep"

if [ -f "$T/dirty_rep_entropy_nonmonotone_T.csv" ] || [ -f "$T/dirty_rep_logenergy_nonmonotone_T.csv" ]; then
  echo "FAIL: repaired dirty table still shows entropy/logenergy nonmonotone-T violations"
  exit 1
fi
echo "PASS: repaired dirty table has zero entropy/logenergy nonmonotone-T violations"

if [ -f "$T/dirty_rep_entropy_negative.csv" ] && [ -f "$T/dirty_rep_nonfinite_cs2.csv" ]; then
  echo "PASS: repaired dirty table still reports entropy_negative and nonfinite_cs2, as designed" \
    "(repair_table() never touches either)"
else
  echo "FAIL: expected dirty_rep_entropy_negative.csv and dirty_rep_nonfinite_cs2.csv to exist"
  exit 1
fi

# 12-13: --level adapter, table-free (M2c). --synthetic is exactly consistent
# and defect-free, so a clean report (exit 0) is required; --synthetic-dirty
# plants monotonicity/physicality pathologies the in-memory repair pass
# (auto-run by --level adapter) cannot fully clean up, so -- like Part B's
# real tables below -- either "clean" (0) or "found something to flag" (1)
# is an accepted outcome.
expect_exit 0 "$TEST" --level adapter --synthetic
set +e
"$TEST" --level adapter --synthetic-dirty >"$T/dirty_adapter.report.txt"
dirty_adapter_exit=$?
set -e
case "$dirty_adapter_exit" in
  0 | 1) echo "PASS: eos_test --level adapter --synthetic-dirty exit=$dirty_adapter_exit" ;;
  *)
    echo "FAIL: eos_test --level adapter --synthetic-dirty exited $dirty_adapter_exit (expected 0 or 1)"
    exit 1
    ;;
esac

echo "=== Part B: real tables (skipped gracefully if absent) ==="

# run_real_table PATH NAME -- exercises eos_test/eos_repair on one real
# stellarcollapse-format table. eos_test/eos_repair on the *raw* table may
# legitimately report "clean" (0) or "found something to flag/repair" (1).
# Non-finite values in fields the pipeline does not interpret are a reported
# violation class, not fatal (the shipped LS220 table genuinely carries a few
# Inf points in "cs2"/"gamma"; they pass through the writer byte-identically
# and land in the exit-1 set). Fatal (2) -- non-finite/missing logenergy or
# entropy, broken axes -- is kept as a handled outcome for arbitrary tables a
# user may drop in: it has no repaired output to exercise further, so the
# rest of this function is skipped for it. Otherwise, what must hold after
# repair is: (a)
# the repaired output is itself clean under --check-only (idempotence), and
# (b) the two nonmonotone-T classes -- the ones repair_table() actually
# targets -- are gone, checked here via --csv (a class's CSV is only written
# when its count > 0, so the assertion is "the file must not exist"). Some
# real tables also carry tiny negative entropies at cold corners
# (entropy_negative), which repair_table() does not touch and so may persist
# after repair -- that's fine, it only affects eos_test's exit code (0 vs 1),
# not this table's structural cleanliness.
run_real_table() {
  local path="$1"
  local name="$2"

  if [ ! -f "$path" ]; then
    echo "skipped: $name ($path not found)"
    return
  fi

  local raw_report="$T/${name}.report.txt"
  local test_exit=0
  set +e
  "$TEST" "$path" >"$raw_report"
  test_exit=$?
  set -e
  case "$test_exit" in
    0 | 1 | 2) echo "PASS: eos_test $name (raw) exit=$test_exit" ;;
    *)
      echo "FAIL: eos_test $name (raw) exited $test_exit (expected 0, 1, or 2)"
      exit 1
      ;;
  esac

  if [ "$test_exit" -eq 2 ]; then
    echo "SUMMARY $name: raw table has a fatal structural problem (see $raw_report);" \
      "repair is refused by design, further checks skipped for this table"
    return
  fi

  local rep="$T/${name}_rep.h5"
  local repair_out="$T/${name}_repair.out"
  local repair_exit=0
  set +e
  "$REPAIR" "$path" "$rep" >"$repair_out"
  repair_exit=$?
  set -e
  case "$repair_exit" in
    0 | 1) echo "PASS: eos_repair $name exit=$repair_exit" ;;
    *)
      echo "FAIL: eos_repair $name exited $repair_exit (expected 0 or 1)"
      exit 1
      ;;
  esac

  expect_exit 0 "$REPAIR" --check-only "$rep" # repaired output is clean

  local rep_report="$T/${name}_rep.report.txt"
  local rep_test_exit=0
  set +e
  "$TEST" "$rep" --csv "$T/${name}_rep" >"$rep_report"
  rep_test_exit=$?
  set -e
  case "$rep_test_exit" in
    0 | 1) echo "PASS: eos_test $name (repaired) exit=$rep_test_exit" ;;
    *)
      echo "FAIL: eos_test $name (repaired) exited $rep_test_exit (expected 0 or 1)"
      exit 1
      ;;
  esac

  # The classes repair_table() actually targets must be gone; --csv only
  # writes a class's file when its count > 0, so "file absent" is the check.
  if [ -f "$T/${name}_rep_entropy_nonmonotone_T.csv" ] || [ -f "$T/${name}_rep_logenergy_nonmonotone_T.csv" ]; then
    echo "FAIL: repaired $name still shows entropy/logenergy nonmonotone-T violations"
    exit 1
  fi
  echo "PASS: repaired $name has zero entropy/logenergy nonmonotone-T violations"

  # --level adapter (M2c): auto-repair path (eos_test repairs in memory
  # itself, from the *original* raw table -- not $rep -- since that is the
  # tool's own documented default flow). "clean" (0) or "found something to
  # flag" (1) are both accepted outcomes, exactly like the raw check_table
  # run above; a real table's own audit statistics are pulled from the saved
  # report below for the summary line.
  local adapter_report="$T/${name}_adapter.report.txt"
  local adapter_test_exit=0
  set +e
  "$TEST" --level adapter "$path" >"$adapter_report"
  adapter_test_exit=$?
  set -e
  case "$adapter_test_exit" in
    0 | 1) echo "PASS: eos_test --level adapter $name exit=$adapter_test_exit" ;;
    *)
      echo "FAIL: eos_test --level adapter $name exited $adapter_test_exit (expected 0 or 1)"
      exit 1
      ;;
  esac

  # Best-effort extraction of "N value(s) changed" from eos_repair's summary
  # (RepairResult::print()'s "  <field>: N value(s) changed[, ...]" lines),
  # for the one-line summary below; "?" if the format ever changes.
  local s_changed e_changed
  s_changed=$( (grep -E '^  entropy:' "$repair_out" || true) | grep -oE '[0-9]+' | head -1)
  e_changed=$( (grep -E '^  logenergy:' "$repair_out" || true) | grep -oE '[0-9]+' | head -1)
  s_changed=${s_changed:-?}
  e_changed=${e_changed:-?}

  # Best-effort extraction of the adapter report's headline numbers
  # (AdapterReport::print()'s "kappa=...", "<class>: count=... max=...",
  # its "  quantiles: p50=... p90=... p99=... p999=... max=..." follow-up
  # line for delta_T/delta_p (see QuantileStats), and "physicality soak:
  # ... maxiter_count=... evals_per_sec=..." lines), for the summary line
  # below; "?" if a line is ever absent.
  local a_kappa a_rt_count a_dT_max a_dp_max a_dT_p99 a_dp_p99 a_sigma_count a_maxiter a_evals
  a_kappa=$( (grep -E '^kappa=' "$adapter_report" || true) | head -1 | cut -d= -f2)
  a_rt_count=$( (grep -E '^roundtrip_T:' "$adapter_report" || true) | grep -oE 'count=[0-9]+' | head -1 |
    cut -d= -f2)
  a_dT_max=$( (grep -E '^delta_T:' "$adapter_report" || true) | grep -oE 'max=[0-9.eE+-]+' | head -1 |
    cut -d= -f2)
  a_dp_max=$( (grep -E '^delta_p:' "$adapter_report" || true) | grep -oE 'max=[0-9.eE+-]+' | head -1 |
    cut -d= -f2)
  # p99 comes from the "quantiles:" line immediately following the
  # "delta_T:"/"delta_p:" header line -- -A1 grabs both lines, and "p99="
  # cannot match inside "p999=" (different key), so this is unambiguous.
  a_dT_p99=$( (grep -A1 -E '^delta_T:' "$adapter_report" || true) | grep -oE 'p99=[0-9.eE+-]+' | head -1 |
    cut -d= -f2)
  a_dp_p99=$( (grep -A1 -E '^delta_p:' "$adapter_report" || true) | grep -oE 'p99=[0-9.eE+-]+' | head -1 |
    cut -d= -f2)
  a_sigma_count=$( (grep -E '^spline_sigma_u_nonpositive:' "$adapter_report" || true) |
    grep -oE 'count=[0-9]+' | head -1 | cut -d= -f2)
  a_maxiter=$( (grep -E '^physicality soak:' "$adapter_report" || true) | grep -oE 'maxiter_count=[0-9]+' |
    head -1 | cut -d= -f2)
  a_evals=$( (grep -E '^physicality soak:' "$adapter_report" || true) | grep -oE 'evals_per_sec=[0-9.eE+-]+' |
    head -1 | cut -d= -f2)
  a_kappa=${a_kappa:-?}
  a_rt_count=${a_rt_count:-?}
  a_dT_max=${a_dT_max:-?}
  a_dp_max=${a_dp_max:-?}
  a_dT_p99=${a_dT_p99:-?}
  a_dp_p99=${a_dp_p99:-?}
  a_sigma_count=${a_sigma_count:-?}
  a_maxiter=${a_maxiter:-?}
  a_evals=${a_evals:-?}

  echo "SUMMARY $name: eos_test(raw)=$test_exit eos_repair=$repair_exit" \
    "eos_test(repaired)=$rep_test_exit entropy_repaired=$s_changed logenergy_repaired=$e_changed" \
    "eos_test(adapter)=$adapter_test_exit kappa=$a_kappa roundtrip_T_count=$a_rt_count" \
    "delta_T_max=$a_dT_max delta_T_p99=$a_dT_p99 delta_p_max=$a_dp_max delta_p_p99=$a_dp_p99" \
    "sigma_monotonicity_count=$a_sigma_count maxiter_count=$a_maxiter evals_per_sec=$a_evals"
}

run_real_table "tables/LS220_234r_136t_50y_analmu_20091212_SVNr26.h5" "LS220"
run_real_table "tables/LS220_3335_rho391_temp163_ye66.h5" "SRO"

echo "=== integration tests passed ==="
