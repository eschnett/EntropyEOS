#!/usr/bin/env bash
# tools/repair_tables.sh -- how each EOS table this package has been measured
# on is repaired, as one runnable record.
#
# For every known table present under the table directory (default tables/,
# see tables/README.md for downloads) this runs tools/eos_repair with this
# release's defaults, writes <file>_repaired.h5 and <NAME>_repair.log next to
# the input, and checks that the output is a one-run fixed point (a second
# eos_repair --check-only finds nothing to change). With --test it then runs
# tools/eos_test's table, adapter and con2prim checks on the repaired table
# with that table's baryon-mass convention, which is the one per-table fact
# the checks need: eos_repair itself is m_B-independent.
#
# Since 1.3.0 every table is repaired with eos_repair's defaults, so the
# table registry below carries no per-table repair options (CODE.md
# milestone M3l: one cs2_cap suits all of them). A table that ever needs one
# gets it in its table_info entry, so the record stays in this file.
#
# Run from anywhere; `make repair-tables` runs it after building the tools.
# Needs the tools built (`make tools`). Bash 3.2 compatible (macOS /bin/bash).

set -euo pipefail

cd "$(dirname "$0")/.."

REPAIR=./tools/eos_repair
TEST=./tools/eos_test
ALL_TABLES="LS220 SRO DD2 SFHo APRLDP"
M_NEUTRON="--m-B 1.67492749804e-24"

usage() {
  cat <<'EOF'
usage: tools/repair_tables.sh [--dir DIR] [--test] [--list] [NAME...]

Repairs the named tables (default: every known table found in DIR) with
tools/eos_repair's defaults, writing DIR/<file>_repaired.h5 and
DIR/<NAME>_repair.log, and checks that each output is a one-run fixed point.

  NAME       one or more of: LS220 SRO DD2 SFHo APRLDP. A table named here
             must exist; with no names, missing tables are skipped.
  --dir DIR  directory holding the input tables and receiving the outputs
             (default: tables)
  --test     also run tools/eos_test (table, adapter and con2prim levels) on
             each repaired table with its baryon mass, writing
             DIR/<NAME>_test_<level>.txt
  --list     print each table's file, baryon mass and source, then exit
  -h, --help print this message

Exit status: 0 if every table attempted was repaired (and, with --test,
checked) successfully; 1 otherwise; 64 on a usage error.
EOF
}

# table_info NAME -- sets FILE (input file name), MB (eos_test baryon-mass
# arguments; empty means the amu default), MB_NAME and SOURCE. Returns 1 for
# an unknown name.
table_info() {
  case "$1" in
    LS220)
      # Lattimer & Swesty K = 220 MeV, O'Connor & Ott's 2009 tabulation. Its
      # m_B convention is indeterminate (CODE.md "M2 empirical findings");
      # the amu default is used.
      FILE=LS220_234r_136t_50y_analmu_20091212_SVNr26.h5
      MB=""
      MB_NAME=amu
      SOURCE="https://stellarcollapse.org/EOS/$FILE.bz2 (tables/README.md)"
      ;;
    SRO)
      # Schneider, Roberts & Ott's LS220 re-implementation (SROEOS).
      FILE=LS220_3335_rho391_temp163_ye66.h5
      MB=$M_NEUTRON
      MB_NAME=m_n
      SOURCE="https://stellarcollapse.org/EOS/$FILE.bz2 (tables/README.md)"
      ;;
    DD2)
      FILE=Hempel_DD2EOS_rho234_temp180_ye60_version_1.1_20120817.h5
      MB=""
      MB_NAME=amu
      SOURCE="https://stellarcollapse.org/~evanoc/$FILE.bz2 (tables/README.md)"
      ;;
    SFHo)
      FILE=Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5
      MB=""
      MB_NAME=amu
      SOURCE="https://stellarcollapse.org/~evanoc/$FILE.bz2 (tables/README.md)"
      ;;
    APRLDP)
      # Schneider, Constantinou, Muccioli & Prakash's APR EOS, low-density
      # phase only, with NSE (3335 nuclides); SROEOS layout, m_B = m_n. Its
      # own cs2 exceeds 1 above rho ~ 1.55e15 g/cc, so the causal cap
      # rewrites that corner (CODE.md "M3l empirical findings").
      FILE=APRLDP_3335_rho393_temp133_ye66_gitM180edd5_20190225.h5
      MB=$M_NEUTRON
      MB_NAME=m_n
      SOURCE="https://stellarcollapse.org/APREOS.html, \"APRLDP with NSE\" (a .h5.tar.bz2; the page's Box links were dead on 2026-10-26)"
      ;;
    *)
      return 1
      ;;
  esac
}

DIR=tables
DO_TEST=0
DO_LIST=0
NAMES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dir)
      if [ $# -lt 2 ]; then
        echo "repair_tables.sh: --dir requires a value" >&2
        usage >&2
        exit 64
      fi
      DIR=$2
      shift 2
      ;;
    --test) DO_TEST=1; shift ;;
    --list) DO_LIST=1; shift ;;
    -h | --help) usage; exit 0 ;;
    -*)
      echo "repair_tables.sh: unknown option '$1'" >&2
      usage >&2
      exit 64
      ;;
    *)
      if ! table_info "$1"; then
        echo "repair_tables.sh: unknown table '$1' (known: $ALL_TABLES)" >&2
        exit 64
      fi
      NAMES="$NAMES $1"
      shift
      ;;
  esac
done

if [ "$DO_LIST" = 1 ]; then
  for name in $ALL_TABLES; do
    table_info "$name"
    printf '%-7s %s\n        m_B: %s\n        source: %s\n' "$name" "$FILE" "$MB_NAME" "$SOURCE"
  done
  exit 0
fi

EXPLICIT=1
if [ -z "$NAMES" ]; then
  NAMES=$ALL_TABLES
  EXPLICIT=0
fi

for tool in "$REPAIR" "$TEST"; do
  if [ ! -x "$tool" ]; then
    echo "repair_tables.sh: $tool not found; run 'make tools' first" >&2
    exit 1
  fi
done

# run_test NAME LEVEL OUT REPORT [ARGS...] -- one eos_test run; exit codes 0
# and 1 (clean / needs attention, the latter normal on real tables) pass,
# anything else fails.
run_test() {
  local name=$1 level=$2 out=$3 report=$4
  shift 4
  local rc=0
  "$TEST" --level "$level" "$out" "$@" >"$report" 2>&1 || rc=$?
  if [ "$rc" -gt 1 ]; then
    echo "FAIL: $name: eos_test --level $level exited $rc (see $report)"
    return 1
  fi
  return 0
}

status=0
summary=""
for name in $NAMES; do
  table_info "$name"
  in=$DIR/$FILE
  out=$DIR/${FILE%.h5}_repaired.h5
  log=$DIR/${name}_repair.log

  if [ ! -f "$in" ]; then
    if [ "$EXPLICIT" = 1 ]; then
      echo "FAIL: $name: $in not found (source: $SOURCE)"
      status=1
      summary="$summary\n  $name: missing"
    else
      echo "skipped: $name ($in not found; source: $SOURCE)"
      summary="$summary\n  $name: skipped (not present)"
    fi
    continue
  fi

  echo "=== $name: $in -> $out"
  rc=0
  "$REPAIR" "$in" "$out" --log "$log" || rc=$?
  if [ "$rc" -gt 1 ]; then
    echo "FAIL: $name: eos_repair exited $rc"
    rm -f "$out"
    status=1
    summary="$summary\n  $name: repair FAILED (exit $rc)"
    continue
  fi

  # One run must be a fixed point (CODE.md milestone M3j): a second pass on
  # the output finds nothing to change and exits 0.
  rc=0
  "$REPAIR" --check-only "$out" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "FAIL: $name: $out is not a fixed point (eos_repair --check-only exited $rc)"
    status=1
    summary="$summary\n  $name: NOT a fixed point"
    continue
  fi

  line="$name: repaired, fixed point; log $log"
  if [ "$DO_TEST" = 1 ]; then
    # $MB is deliberately unquoted: it is empty or two words.
    # shellcheck disable=SC2086
    if run_test "$name" table "$out" "$DIR/${name}_test_table.txt" $MB &&
      run_test "$name" adapter "$out" "$DIR/${name}_test_adapter.txt" $MB --no-repair &&
      run_test "$name" con2prim "$out" "$DIR/${name}_test_con2prim.txt" $MB --no-repair; then
      acausal=$(grep -E '^cs2_acausal: count=' "$DIR/${name}_test_adapter.txt" |
        sed -E 's/^cs2_acausal: count=([0-9]+).*/\1/')
      warm=$(grep -E '^warm: ' "$DIR/${name}_test_con2prim.txt" |
        sed -E 's/.*n_failed_total=([0-9]+).*/\1/')
      cold=$(grep -E '^cold: ' "$DIR/${name}_test_con2prim.txt" |
        sed -E 's/.*n_failed_total=([0-9]+).*/\1/')
      line="$line; m_B $MB_NAME, in-box cs2>=1 ${acausal:-?}, con2prim failed warm/cold ${warm:-?}/${cold:-?}"
    else
      status=1
      line="$name: repaired, fixed point; eos_test FAILED"
    fi
  fi
  summary="$summary\n  $line"
done

printf '\nsummary:%b\n' "$summary"
exit $status
