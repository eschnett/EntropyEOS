// tools/eos_test.cpp — the M1/M2c/M3b/M3e test harness: "--level table"
// (M1), "--level adapter" (M2c, CODE.md "Test harness" /
// eos-adapter-F-to-U.md S10), and "--level con2prim" (M3b,
// con2prim-entropy-rapidity.md S12 deliverable 2, which since M3e also
// carries the invalid-state policy section of that document's S11 --
// automatically, through check_con2prim(); the only new knobs here are
// --rho-atm and --w-cap). One binary that accreted all these stages as their
// milestones landed.
//
// A thin main() over entropy_eos: loads a table (from a stellarcollapse
// file, or an in-memory synthetic ideal-gas table, optionally with seeded
// violations or a fixed dirty-defect preset -- the same four input forms for
// every level), then either runs check_table() ("table"), or repairs the
// table in memory, builds an EntropyEOS, and runs check_adapter()
// ("adapter") or check_con2prim() ("con2prim"), printing a human-readable
// report and optionally dumping the worst offenders per class to CSV. No
// physics or table logic lives here -- see
// entropy_eos/host/{check,adapter_audit,con2prim_audit,synthetic}.hpp for
// that.
//
//   eos_test --level table (FILE.h5 | --synthetic | --synthetic-seeded |
//                            --synthetic-dirty)
//            [--write-synthetic PATH] [--csv PREFIX] [--tol X] [--worst N]
//            [--m-B GRAMS]
//   eos_test --level adapter (FILE.h5 | --synthetic | --synthetic-seeded |
//                              --synthetic-dirty)
//            [--write-synthetic PATH] [--csv PREFIX] [--worst N]
//            [--no-repair] [--node-stride N] [--soak N] [--m-B GRAMS]
//   eos_test --level con2prim (FILE.h5 | --synthetic | --synthetic-seeded |
//                               --synthetic-dirty)
//            [--write-synthetic PATH] [--csv PREFIX] [--worst N]
//            [--no-repair] [--m-B GRAMS]
//            [--states N] [--wmax X] [--sigma-max X] [--rt-tol X]
//            [--rho-atm X] [--w-cap X]
//
// Exit codes ("table"): 2 = fatal structural problem; 1 = "entropy_negative",
// "entropy_nonmonotone_T", "logenergy_nonmonotone_T", or any
// "nonfinite_<field>" class has count > 0 (the
// Maxwell-consistency and cs2 classes are diagnostics only and never affect
// the exit code); 0 = otherwise clean; 64 = usage error.
//
// Exit codes ("adapter"): 2 = build_entropy_eos() failed, or
// check_adapter()'s report.status is fatal (a non-finite EOSPoint turned up
// somewhere in the audit); 1 = adapter_needs_attention() (any monotonicity,
// roundtrip_T, or physicality violation class has count > 0, or
// maxiter_count > 0); 0 = otherwise clean; 64 = usage error. Like the table
// level's Maxwell-consistency classes, "delta_T"/"delta_p" are diagnostics
// only and never affect the exit code.
//
// Exit codes ("con2prim"): 2 = build_entropy_eos() failed, or
// check_con2prim()'s report.status is fatal (a non-finite recovered
// primitive/EOSPoint turned up somewhere in the audit); 1 =
// con2prim_needs_attention() (any warm/cold failed_* state, the
// "c2p_roundtrip" class has count > 0, or the M3e invalid-state policy
// section found a false positive / a non-verbatim no-touch path / an invalid
// output / a failing broken-state battery case); 0 = otherwise clean; 64 =
// usage error.

#include "entropy_eos/entropy_eos.hpp"
#include "entropy_eos/core/state_policy.hpp"
#include "entropy_eos/host/adapter_audit.hpp"
#include "entropy_eos/host/con2prim_audit.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kExitOk = 0;
constexpr int kExitViolation = 1;
constexpr int kExitFatal = 2;
constexpr int kExitUsage = 64;

void print_usage(std::ostream &os) {
  os << "usage:\n"
        "  eos_test [--level table|adapter|con2prim] FILE.h5 [options]\n"
        "  eos_test [--level table|adapter|con2prim] --synthetic [options]\n"
        "  eos_test [--level table|adapter|con2prim] --synthetic-seeded [options]\n"
        "  eos_test [--level table|adapter|con2prim] --synthetic-dirty [options]\n"
        "\n"
        "Runs check_table() (\"--level table\", CODE.md \"Test harness\"),\n"
        "check_adapter() (\"--level adapter\"), or check_con2prim() (\"--level\n"
        "con2prim\") against a stellarcollapse-format table, an in-memory synthetic\n"
        "ideal-gas table, a synthetic table with 4 deliberately seeded monotonicity\n"
        "violations, or a synthetic table with a fixed set of deterministic defects\n"
        "mimicking real LS220/SRO table pathologies, and prints a human-readable\n"
        "report to stdout.\n"
        "\n"
        "input (exactly one required):\n"
        "  FILE.h5              read a stellarcollapse-format table from FILE.h5\n"
        "  --synthetic          use make_synthetic_table() with default options\n"
        "  --synthetic-seeded   like --synthetic, plus 4 seeded violations (two on\n"
        "                       \"entropy\", two on \"logenergy\") at interior grid\n"
        "                       points, printed to stdout before the report\n"
        "  --synthetic-dirty    use dirty_synthetic_options(): default grid, aux\n"
        "                       fields (cs2/gamma/mu_e), and a fixed deterministic\n"
        "                       defect set (a wiggle-induced entropy\n"
        "                       non-monotone-T cluster, a logenergy plateau, a\n"
        "                       negative-entropy cold corner, and planted\n"
        "                       Inf/NaN in cs2/gamma) mimicking pathologies found\n"
        "                       in the real LS220/SRO tables\n"
        "\n"
        "options:\n"
        "  --level LEVEL        check level: \"table\" (default), \"adapter\", or\n"
        "                       \"con2prim\"\n"
        "  --write-synthetic PATH  write the generated synthetic table to PATH (only\n"
        "                          valid together with --synthetic,\n"
        "                          --synthetic-seeded, or --synthetic-dirty)\n"
        "  --csv PREFIX         for every check class with count > 0, write\n"
        "                       PREFIX_<class>.csv (header: irho,jT,kYe,rho,temp,ye,\n"
        "                       value), one row per worst-offender entry (--level\n"
        "                       adapter's class A/C and --level con2prim's classes'\n"
        "                       worst entries carry irho=jT=kYe=0, since those\n"
        "                       audits are not evaluated at table nodes; --level\n"
        "                       con2prim's \"c2p_failed\" class writes the sampled\n"
        "                       rapidity w into the \"value\" column, see\n"
        "                       con2prim_audit.hpp).\n"
        "                       The list is capped at --worst N; pass a large N for a\n"
        "                       full dump.\n"
        "  --worst N            CheckOptions::worst_n / AdapterCheckOptions::worst_n /\n"
        "                       Con2PrimCheckOptions::worst_n override (also caps\n"
        "                       --csv)\n"
        "  --tol X              CheckOptions::tol_consistency override (--level table\n"
        "                       only)\n"
        "  --m-B GRAMS          the table's baryon-mass convention: CheckOptions::m_B_g\n"
        "                       (--level table) or BuildOptions::m_B_table_g (--level\n"
        "                       adapter/con2prim). Default is the amu; SRO tables\n"
        "                       empirically use the neutron mass 1.67492749804e-24 g\n"
        "                       (see CODE.md)\n"
        "  --no-repair          --level adapter/con2prim only: skip the default\n"
        "                       in-memory repair_table() pass and build directly\n"
        "                       from the loaded table\n"
        "  --node-stride N      --level adapter only: AdapterCheckOptions::node_stride\n"
        "                       override (audit every Nth table node per axis;\n"
        "                       default 1)\n"
        "  --soak N             --level adapter only: AdapterCheckOptions::soak_n\n"
        "                       override (physicality-soak sample count; default\n"
        "                       200000)\n"
        "  --states N           --level con2prim only: Con2PrimCheckOptions::n_states\n"
        "                       override (total sampled states in the warm pass,\n"
        "                       ~10% of which also get a cold pass; default 20000)\n"
        "  --wmax X             --level con2prim only:\n"
        "                       Con2PrimCheckOptions::w_max_sample override (sampled\n"
        "                       rapidity range [0,X]; default 6.0)\n"
        "  --sigma-max X        --level con2prim only: Con2PrimCheckOptions::sigma_max\n"
        "                       override (magnetization B^2/(rho*h) sampled\n"
        "                       log-uniform in [1e-6,X]; default 1e4)\n"
        "  --rt-tol X           --level con2prim only:\n"
        "                       Con2PrimCheckOptions::tol_roundtrip override\n"
        "                       (conservative-space round-trip threshold entering\n"
        "                       \"c2p_roundtrip\"; default 1e-8)\n"
        "  --rho-atm X          --level con2prim only: the M3e policy section's\n"
        "                       atmosphere density, in adapter units (kappa-rescaled\n"
        "                       g/cc). The PRODUCTION default this tool reports before\n"
        "                       the audit is default_policy(view, 10^x_lo * 10), i.e.\n"
        "                       ten times the table's own density floor; the AUDIT\n"
        "                       lowers whatever it is given to half the smallest D of\n"
        "                       its own sampled set, because an atmosphere threshold\n"
        "                       sitting inside the sampled range would make the\n"
        "                       false-positive measurement vacuous (see\n"
        "                       Con2PrimCheckOptions::policy_rho_atm). The value\n"
        "                       actually used is printed in the report.\n"
        "  --w-cap X            --level con2prim only: the M3e policy section's rapidity\n"
        "                       cap. Production default: acosh(100) (W <= 100), which\n"
        "                       must stay below the solver's w_max = 12. The audit\n"
        "                       raises whatever it is given to max(that, --wmax + 0.5)\n"
        "                       for the same reason (and because\n"
        "                       PolicyOptions::D_max/tau_max are derived FROM w_cap);\n"
        "                       the value used is printed in the report.\n"
        "  -h, --help           print this message\n"
        "\n"
        "exit codes (--level table):\n"
        "  0   status ok and no entropy/monotonicity violations\n"
        "  1   \"entropy_negative\", \"entropy_nonmonotone_T\",\n"
        "      \"logenergy_nonmonotone_T\", or any \"nonfinite_<field>\" class has\n"
        "      count > 0 (the Maxwell-consistency and cs2 classes are diagnostics\n"
        "      only and never affect the exit code)\n"
        "  2   fatal structural problem in the table (see the printed report)\n"
        "  64  usage error\n"
        "\n"
        "exit codes (--level adapter):\n"
        "  0   report.status ok and adapter_needs_attention() is false\n"
        "  1   adapter_needs_attention(): any monotonicity ('spline_sigma_u_\n"
        "      nonpositive', 'spline_L_u_nonpositive'), 'roundtrip_T', or\n"
        "      physicality ('That_nonpositive', 'p_nonpositive', 'cs2_nonpositive',\n"
        "      'cs2_acausal') class has count > 0, or the physicality soak hit the\n"
        "      T-solve iteration cap at least once ('delta_T'/'delta_p' are\n"
        "      diagnostics only and never affect the exit code)\n"
        "  2   build_entropy_eos() failed (see stderr), or report.status is fatal\n"
        "      (a non-finite EOSPoint turned up somewhere in the audit)\n"
        "  64  usage error\n"
        "\n"
        "exit codes (--level con2prim):\n"
        "  0   report.status ok and con2prim_needs_attention() is false\n"
        "  1   con2prim_needs_attention(): any warm-pass failed_no_bracket/\n"
        "      failed_max_iter state, any cold-pass failed state, the\n"
        "      'c2p_roundtrip' class has count > 0, or the M3e policy section\n"
        "      reported a false positive (policy_n_valid_touched), a non-verbatim\n"
        "      no-touch path, an invalid output, or a broken-state battery case\n"
        "      that came out invalid or unflagged ('c2p_failed' is a redundant\n"
        "      worst-offender view of the same warm-pass failures, not consulted\n"
        "      independently; the raw policy intervention count is reported but\n"
        "      not consulted either -- on a real table it legitimately absorbs\n"
        "      the solver's own documented failure tail)\n"
        "  2   build_entropy_eos() failed (see stderr), or report.status is fatal\n"
        "      (a non-finite recovered primitive/EOSPoint turned up somewhere in\n"
        "      the audit)\n"
        "  64  usage error\n";
}

struct ParsedArgs {
  std::string level = "table";
  bool synthetic = false;
  bool synthetic_seeded = false;
  bool synthetic_dirty = false;
  std::string file_path;
  bool have_write_synthetic = false;
  std::string write_synthetic_path;
  bool have_csv = false;
  std::string csv_prefix;
  bool have_tol = false;
  double tol = 0.0;
  bool have_worst = false;
  long long worst = 0;
  bool have_m_B = false;
  double m_B = 0.0;
  bool no_repair = false;
  bool have_node_stride = false;
  long long node_stride = 0;
  bool have_soak = false;
  long long soak = 0;
  bool have_states = false;
  long long states = 0;
  bool have_wmax = false;
  double wmax = 0.0;
  bool have_sigma_max = false;
  double sigma_max = 0.0;
  bool have_rt_tol = false;
  double rt_tol = 0.0;
  bool have_rho_atm = false;
  double rho_atm = 0.0;
  bool have_w_cap = false;
  double w_cap = 0.0;
  std::vector<std::string> positionals;
};

bool parse_args(const std::vector<std::string> &args, ParsedArgs &out) {
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string &a = args[i];

    if (a == "-h" || a == "--help") {
      print_usage(std::cerr);
      return false;
    } else if (a == "--synthetic") {
      out.synthetic = true;
    } else if (a == "--synthetic-seeded") {
      out.synthetic_seeded = true;
    } else if (a == "--synthetic-dirty") {
      out.synthetic_dirty = true;
    } else if (a == "--level") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_test: option '--level' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.level = args[++i];
    } else if (a == "--write-synthetic") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_test: option '--write-synthetic' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.write_synthetic_path = args[++i];
      out.have_write_synthetic = true;
    } else if (a == "--csv") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_test: option '--csv' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.csv_prefix = args[++i];
      out.have_csv = true;
    } else if (a == "--tol") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_test: option '--tol' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      try {
        out.tol = std::stod(args[++i]);
      } catch (const std::exception &) {
        std::cerr << "eos_test: invalid number for --tol: '" << args[i] << "'\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.have_tol = true;
    } else if (a == "--worst") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_test: option '--worst' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      try {
        out.worst = std::stoll(args[++i]);
      } catch (const std::exception &) {
        std::cerr << "eos_test: invalid integer for --worst: '" << args[i] << "'\n\n";
        print_usage(std::cerr);
        return false;
      }
      if (out.worst < 0) {
        std::cerr << "eos_test: --worst must be >= 0\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.have_worst = true;
    } else if (a == "--m-B") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_test: option '--m-B' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      try {
        out.m_B = std::stod(args[++i]);
      } catch (const std::exception &) {
        std::cerr << "eos_test: invalid number for --m-B: '" << args[i] << "'\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.have_m_B = true;
    } else if (a == "--no-repair") {
      out.no_repair = true;
    } else if (a == "--node-stride") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_test: option '--node-stride' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      try {
        out.node_stride = std::stoll(args[++i]);
      } catch (const std::exception &) {
        std::cerr << "eos_test: invalid integer for --node-stride: '" << args[i] << "'\n\n";
        print_usage(std::cerr);
        return false;
      }
      if (out.node_stride < 1) {
        std::cerr << "eos_test: --node-stride must be >= 1\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.have_node_stride = true;
    } else if (a == "--soak") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_test: option '--soak' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      try {
        out.soak = std::stoll(args[++i]);
      } catch (const std::exception &) {
        std::cerr << "eos_test: invalid integer for --soak: '" << args[i] << "'\n\n";
        print_usage(std::cerr);
        return false;
      }
      if (out.soak < 0) {
        std::cerr << "eos_test: --soak must be >= 0\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.have_soak = true;
    } else if (a == "--states") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_test: option '--states' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      try {
        out.states = std::stoll(args[++i]);
      } catch (const std::exception &) {
        std::cerr << "eos_test: invalid integer for --states: '" << args[i] << "'\n\n";
        print_usage(std::cerr);
        return false;
      }
      if (out.states < 1) {
        std::cerr << "eos_test: --states must be >= 1\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.have_states = true;
    } else if (a == "--wmax") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_test: option '--wmax' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      try {
        out.wmax = std::stod(args[++i]);
      } catch (const std::exception &) {
        std::cerr << "eos_test: invalid number for --wmax: '" << args[i] << "'\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.have_wmax = true;
    } else if (a == "--sigma-max") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_test: option '--sigma-max' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      try {
        out.sigma_max = std::stod(args[++i]);
      } catch (const std::exception &) {
        std::cerr << "eos_test: invalid number for --sigma-max: '" << args[i] << "'\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.have_sigma_max = true;
    } else if (a == "--rt-tol") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_test: option '--rt-tol' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      try {
        out.rt_tol = std::stod(args[++i]);
      } catch (const std::exception &) {
        std::cerr << "eos_test: invalid number for --rt-tol: '" << args[i] << "'\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.have_rt_tol = true;
    } else if (a == "--rho-atm") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_test: option '--rho-atm' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      try {
        out.rho_atm = std::stod(args[++i]);
      } catch (const std::exception &) {
        std::cerr << "eos_test: invalid number for --rho-atm: '" << args[i] << "'\n\n";
        print_usage(std::cerr);
        return false;
      }
      if (!(out.rho_atm > 0.0)) {
        std::cerr << "eos_test: --rho-atm must be > 0\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.have_rho_atm = true;
    } else if (a == "--w-cap") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_test: option '--w-cap' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      try {
        out.w_cap = std::stod(args[++i]);
      } catch (const std::exception &) {
        std::cerr << "eos_test: invalid number for --w-cap: '" << args[i] << "'\n\n";
        print_usage(std::cerr);
        return false;
      }
      if (!(out.w_cap > 0.0)) {
        std::cerr << "eos_test: --w-cap must be > 0\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.have_w_cap = true;
    } else if (!a.empty() && a[0] == '-') {
      std::cerr << "eos_test: unknown option '" << a << "'\n\n";
      print_usage(std::cerr);
      return false;
    } else {
      out.positionals.push_back(a);
    }
  }

  if (out.positionals.size() > 1) {
    std::cerr << "eos_test: at most one positional argument (FILE.h5) is allowed\n\n";
    print_usage(std::cerr);
    return false;
  }
  out.file_path = out.positionals.empty() ? std::string() : out.positionals[0];

  const int n_inputs = (!out.file_path.empty() ? 1 : 0) + (out.synthetic ? 1 : 0) +
                       (out.synthetic_seeded ? 1 : 0) + (out.synthetic_dirty ? 1 : 0);
  if (n_inputs != 1) {
    std::cerr << "eos_test: specify exactly one of FILE.h5, --synthetic, --synthetic-seeded, "
                 "--synthetic-dirty\n\n";
    print_usage(std::cerr);
    return false;
  }

  if (out.have_write_synthetic && !(out.synthetic || out.synthetic_seeded || out.synthetic_dirty)) {
    std::cerr << "eos_test: --write-synthetic is only valid together with --synthetic, "
                 "--synthetic-seeded, or --synthetic-dirty\n\n";
    print_usage(std::cerr);
    return false;
  }

  if ((out.have_node_stride || out.have_soak) && out.level != "adapter") {
    std::cerr << "eos_test: --node-stride/--soak are only valid with --level adapter\n\n";
    print_usage(std::cerr);
    return false;
  }

  if (out.no_repair && out.level != "adapter" && out.level != "con2prim") {
    std::cerr << "eos_test: --no-repair is only valid with --level adapter or --level con2prim\n\n";
    print_usage(std::cerr);
    return false;
  }

  if ((out.have_states || out.have_wmax || out.have_sigma_max || out.have_rt_tol ||
       out.have_rho_atm || out.have_w_cap) &&
      out.level != "con2prim") {
    std::cerr << "eos_test: --states/--wmax/--sigma-max/--rt-tol/--rho-atm/--w-cap are only valid "
                 "with --level con2prim\n\n";
    print_usage(std::cerr);
    return false;
  }

  return true;
}

// Four seeded violations (two on "entropy", two on "logenergy") at interior
// grid points of the default SyntheticOptions grid (40 x 30 x 10). Delta
// magnitude (-2.0) is chosen to comfortably exceed the model's per-T-step
// increment in both fields (entropy: g*1.5*d(ln T) ~ 0.4-0.55 kB/baryon per
// step; logenergy: at most the log10(T) grid spacing ~0.1 dex per step, once
// eps dominates the energy_shift) -- large enough to break monotonicity at
// the seeded point -- while staying well within the model's comfortable
// entropy floor (~15 kB/baryon over the whole default grid, since
// rho_max_gcc < rho0_gcc keeps the density term non-negative everywhere), so
// that after repair_table()'s isotonic pooling the column-local average
// cannot be dragged into "entropy_negative" territory.
std::vector<eeos::SeededViolation> make_seeds(const eeos::SyntheticOptions &grid) {
  const double delta = -2.0;
  return {
      eeos::SeededViolation{"entropy", grid.nrho / 4, grid.ntemp / 3, grid.nye / 3, delta},
      eeos::SeededViolation{"entropy", 3 * grid.nrho / 4, 2 * grid.ntemp / 3, 2 * grid.nye / 3, delta},
      eeos::SeededViolation{"logenergy", grid.nrho / 3, grid.ntemp / 4, grid.nye / 4, delta},
      eeos::SeededViolation{"logenergy", 2 * grid.nrho / 3, 3 * grid.ntemp / 4, 3 * grid.nye / 4, delta},
  };
}

// Shared table loader for both --level table and --level adapter: exactly
// one of FILE.h5 / --synthetic / --synthetic-seeded / --synthetic-dirty
// (parse_args already enforced that), with the same stdout announcements
// and optional --write-synthetic side effect either level produces.
eeos::RawTable load_table(const ParsedArgs &pa) {
  eeos::RawTable table;
  if (!pa.file_path.empty()) {
    table = eeos::read_stellarcollapse(pa.file_path);
  } else if (pa.synthetic_dirty) {
    eeos::SyntheticOptions sopts = eeos::dirty_synthetic_options();
    std::cout << "eos_test --synthetic-dirty: dirty_synthetic_options() (" << sopts.flatten.size()
              << " flatten, " << sopts.wiggle.size() << " wiggle, " << sopts.offset.size()
              << " offset, " << sopts.setvalue.size() << " setvalue defect(s), aux fields "
              << (sopts.with_aux_fields ? "on" : "off") << "):\n";
    for (const eeos::FlattenDefect &d : sopts.flatten) {
      std::cout << "  flatten " << d.field << " irho=[" << d.irho0 << "," << d.irho1 << "] kYe=["
                 << d.kYe0 << "," << d.kYe1 << "] jT=[" << d.jT0 << "," << d.jT1 << "]\n";
    }
    for (const eeos::WiggleDefect &d : sopts.wiggle) {
      std::cout << "  wiggle " << d.field << " irho=[" << d.irho0 << "," << d.irho1 << "] kYe=["
                 << d.kYe0 << "," << d.kYe1 << "] jT=[" << d.jT0 << "," << d.jT1
                 << "] amplitude=" << d.amplitude << " period=" << d.period << "\n";
    }
    for (const eeos::OffsetDefect &d : sopts.offset) {
      std::cout << "  offset " << d.field << " irho=[" << d.irho0 << "," << d.irho1 << "] kYe=["
                 << d.kYe0 << "," << d.kYe1 << "] jT=[" << d.jT0 << "," << d.jT1
                 << "] offset=" << d.offset << "\n";
    }
    for (const eeos::SetValue &v : sopts.setvalue) {
      std::cout << "  setvalue " << v.field << " at (irho=" << v.irho << ", jT=" << v.jT
                 << ", kYe=" << v.kYe << ") value=" << v.value << "\n";
    }
    table = eeos::make_synthetic_table(sopts);
    if (pa.have_write_synthetic) {
      eeos::write_stellarcollapse(pa.write_synthetic_path, table);
    }
  } else {
    eeos::SyntheticOptions sopts; // defaults per CODE.md
    if (pa.synthetic_seeded) {
      sopts.seed = make_seeds(sopts);
      std::cout << "eos_test --synthetic-seeded: seeding " << sopts.seed.size() << " violation(s):\n";
      for (const eeos::SeededViolation &v : sopts.seed) {
        std::cout << "  " << v.field << " at (irho=" << v.irho << ", jT=" << v.jT << ", kYe=" << v.kYe
                   << ") delta=" << v.delta << "\n";
      }
    }
    table = eeos::make_synthetic_table(sopts);
    if (pa.have_write_synthetic) {
      eeos::write_stellarcollapse(pa.write_synthetic_path, table);
    }
  }
  return table;
}

// Shared CSV writer for both --level table (CheckReport::classes) and
// --level adapter (AdapterReport::classes): one PREFIX_<class>.csv per class
// with count > 0, one row per worst-offender entry (see print_usage()).
void write_csv(const std::vector<eeos::CheckClassResult> &classes, const std::string &prefix) {
  for (const eeos::CheckClassResult &c : classes) {
    if (c.count == 0) {
      continue;
    }
    const std::string path = prefix + "_" + c.name + ".csv";
    std::ofstream csv(path);
    if (!csv) {
      throw std::runtime_error("could not open CSV file '" + path + "' for writing");
    }
    csv << "irho,jT,kYe,rho,temp,ye,value\n";
    csv << std::setprecision(17);
    for (const eeos::CheckClassResult::Loc &loc : c.worst) {
      csv << loc.irho << "," << loc.jT << "," << loc.kYe << "," << loc.rho << "," << loc.temp << ","
          << loc.ye << "," << loc.value << "\n";
    }
  }
}

int run_table_level(const ParsedArgs &pa) {
  eeos::RawTable table = load_table(pa);

  eeos::CheckOptions check_opts;
  if (pa.have_m_B) {
    check_opts.m_B_g = pa.m_B;
  }
  if (pa.have_tol) {
    check_opts.tol_consistency = pa.tol;
  }
  if (pa.have_worst) {
    check_opts.worst_n = static_cast<size_t>(pa.worst);
  }

  const eeos::CheckReport report = eeos::check_table(table, check_opts);
  report.print(std::cout);

  if (pa.have_csv) {
    write_csv(report.classes, pa.csv_prefix);
  }

  if (report.status == eeos::Status::fatal) {
    return kExitFatal;
  }

  bool hard_violation = false;
  for (const eeos::CheckClassResult &c : report.classes) {
    if (c.count == 0) {
      continue;
    }
    if (c.name == "entropy_negative" || c.name == "entropy_nonmonotone_T" ||
        c.name == "logenergy_nonmonotone_T" || c.name.rfind("nonfinite_", 0) == 0) {
      hard_violation = true;
    }
  }
  return hard_violation ? kExitViolation : kExitOk;
}

int run_adapter_level(const ParsedArgs &pa) {
  eeos::RawTable table = load_table(pa);

  if (!pa.no_repair) {
    const eeos::RepairResult repair_result = eeos::repair_table(table);
    std::cout << "repair (in-memory): status="
              << (repair_result.status == eeos::Status::ok ? "ok" : "repaired")
              << " entries=" << repair_result.entries.size();
    for (const eeos::RepairResult::FieldSummary &s : repair_result.summaries) {
      std::cout << " " << s.field << "_changed=" << s.modified;
    }
    std::cout << "\n";
  } else {
    std::cout << "repair (in-memory): skipped (--no-repair)\n";
  }

  // build_entropy_eos() failures propagate to main()'s catch block, which
  // prints "eos_test: <message>" to stderr and exits kExitFatal (2) -- the
  // "build failure -> message to stderr, exit 2" contract.
  eeos::BuildOptions bopts;
  if (pa.have_m_B) {
    // The table's baryon-mass convention. Empirically (delta_T quantile
    // collapse, CODE.md "M2 design notes"): SRO tables use the neutron mass
    // m_neutron_g; pass --m-B 1.67492749804e-24 for those.
    bopts.m_B_table_g = pa.m_B;
  }
  const eeos::EntropyEOS adapter = eeos::build_entropy_eos(table, bopts);

  eeos::AdapterCheckOptions aopts;
  if (pa.have_node_stride) {
    aopts.node_stride = static_cast<size_t>(pa.node_stride);
  }
  if (pa.have_soak) {
    aopts.soak_n = static_cast<size_t>(pa.soak);
  }
  if (pa.have_worst) {
    aopts.worst_n = static_cast<size_t>(pa.worst);
  }

  const eeos::AdapterReport report = eeos::check_adapter(adapter, table, aopts);
  report.print(std::cout);

  if (pa.have_csv) {
    write_csv(report.classes, pa.csv_prefix);
  }

  if (report.status == eeos::Status::fatal) {
    return kExitFatal;
  }
  return eeos::adapter_needs_attention(report) ? kExitViolation : kExitOk;
}

int run_con2prim_level(const ParsedArgs &pa) {
  eeos::RawTable table = load_table(pa);

  // Same auto-repair-then-build plumbing as --level adapter (M3b reuses
  // it verbatim: check_con2prim() audits a built EntropyEOS, same as
  // check_adapter()).
  if (!pa.no_repair) {
    const eeos::RepairResult repair_result = eeos::repair_table(table);
    std::cout << "repair (in-memory): status="
              << (repair_result.status == eeos::Status::ok ? "ok" : "repaired")
              << " entries=" << repair_result.entries.size();
    for (const eeos::RepairResult::FieldSummary &s : repair_result.summaries) {
      std::cout << " " << s.field << "_changed=" << s.modified;
    }
    std::cout << "\n";
  } else {
    std::cout << "repair (in-memory): skipped (--no-repair)\n";
  }

  // build_entropy_eos() failures propagate to main()'s catch block, which
  // prints "eos_test: <message>" to stderr and exits kExitFatal (2) -- the
  // "build failure -> message to stderr, exit 2" contract.
  eeos::BuildOptions bopts;
  if (pa.have_m_B) {
    // The table's baryon-mass convention. Empirically (delta_T quantile
    // collapse, CODE.md "M2 design notes"): SRO tables use the neutron mass
    // m_neutron_g; pass --m-B 1.67492749804e-24 for those.
    bopts.m_B_table_g = pa.m_B;
  }
  const eeos::EntropyEOS adapter = eeos::build_entropy_eos(table, bopts);

  eeos::Con2PrimCheckOptions copts;
  if (pa.have_states) {
    copts.n_states = static_cast<size_t>(pa.states);
  }
  if (pa.have_wmax) {
    copts.w_max_sample = pa.wmax;
  }
  if (pa.have_sigma_max) {
    copts.sigma_max = pa.sigma_max;
  }
  if (pa.have_rt_tol) {
    copts.tol_roundtrip = pa.rt_tol;
  }
  if (pa.have_worst) {
    copts.worst_n = static_cast<size_t>(pa.worst);
  }
  if (pa.have_rho_atm) {
    copts.policy_rho_atm = pa.rho_atm;
  }
  if (pa.have_w_cap) {
    copts.policy_w_cap = pa.w_cap;
  }

  // M3e: report the PRODUCTION policy defaults -- default_policy() with
  // rho_atm = 10^x_lo * 10, ten times the table's own density floor, which is
  // the value a caller of this library would plausibly start from (an
  // atmosphere one decade above the tabulated minimum). The AUDIT's own
  // policy differs deliberately, and reports its own rho_atm/w_cap in the
  // report below -- see con2prim_audit.hpp's Con2PrimCheckOptions::
  // policy_rho_atm/_w_cap for why (an atmosphere or rapidity threshold inside
  // the audit's sampled range would make its false-positive measurement
  // vacuous rather than informative).
  {
    const eeos::EntropyEOSView view = adapter.view();
    const eeos::PolicyOptions prod = eeos::default_policy(view, 10.0 * std::pow(10.0, view.x_lo));
    std::cout << "policy defaults (production, rho_atm = 10^x_lo * 10): rho_atm=" << prod.rho_atm
              << " rho_ceiling=" << prod.rho_ceiling << " w_cap=" << prod.w_cap
              << " D_max=" << prod.D_max << " tau_max=" << prod.tau_max
              << " collapse_to_atmosphere=" << (prod.collapse_to_atmosphere ? "true" : "false")
              << "\n";
  }

  const eeos::Con2PrimReport report = eeos::check_con2prim(adapter, copts);
  report.print(std::cout);

  if (pa.have_csv) {
    write_csv(report.classes, pa.csv_prefix);
  }

  if (report.status == eeos::Status::fatal) {
    return kExitFatal;
  }
  return eeos::con2prim_needs_attention(report) ? kExitViolation : kExitOk;
}

} // namespace

int main(int argc, char **argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);

  ParsedArgs pa;
  if (!parse_args(args, pa)) {
    return kExitUsage;
  }

  if (pa.level != "table" && pa.level != "adapter" && pa.level != "con2prim") {
    std::cerr << "eos_test: unknown --level '" << pa.level << "' (expected table, adapter, or "
                 "con2prim)\n";
    return kExitUsage;
  }

  try {
    if (pa.level == "table") {
      return run_table_level(pa);
    }
    if (pa.level == "adapter") {
      return run_adapter_level(pa);
    }
    return run_con2prim_level(pa);
  } catch (const std::exception &e) {
    std::cerr << "eos_test: " << e.what() << "\n";
    return kExitFatal;
  }
}
