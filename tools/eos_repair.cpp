// tools/eos_repair.cpp — M1 repair harness (CODE.md "Repair harness").
//
// A thin main() over entropy_eos: reads a stellarcollapse-format table,
// checks it (check_table), repairs it in memory (repair_table), and either
// reports what would change (--check-only) or writes a repaired copy plus a
// "/repair" provenance group and an optional human-readable log. No physics
// or table logic lives here -- see entropy_eos/host/{check,repair}.hpp for
// that.
//
//   eos_repair IN.h5 OUT.h5 [--min-slope-s X] [--min-slope-loge X]
//              [--no-spline-safe] [--no-spline-safe-3d]
//              [--no-causal-cap] [--cs2-cap X] [--log FILE]
//   eos_repair --check-only IN.h5 [--min-slope-s X] [--min-slope-loge X]
//              [--no-spline-safe] [--no-spline-safe-3d]
//              [--no-causal-cap] [--cs2-cap X]
//
// Exit codes: 0 = already clean, 1 = repaired (or would repair, under
// --check-only), 2 = fatal structural problem or other runtime error,
// 64 = usage error.

#include "entropy_eos/entropy_eos.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kExitOk = 0;
constexpr int kExitChanged = 1;
constexpr int kExitFatal = 2;
constexpr int kExitUsage = 64;

const char *const kToolVersion = "entropy_eos eos_repair 0.1";

void print_usage(std::ostream &os) {
  os << "usage:\n"
        "  eos_repair IN.h5 OUT.h5 [--min-slope-s X] [--min-slope-loge X]\n"
        "             [--no-spline-safe] [--no-spline-safe-3d]\n"
        "             [--no-causal-cap] [--cs2-cap X] [--log FILE]\n"
        "  eos_repair --check-only IN.h5 [--min-slope-s X] [--min-slope-loge X]\n"
        "             [--no-spline-safe] [--no-spline-safe-3d]\n"
        "             [--no-causal-cap] [--cs2-cap X]\n"
        "\n"
        "Repairs a stellarcollapse-format EOS table so that \"entropy\" and\n"
        "\"logenergy\" are strictly monotone increasing in temperature at every\n"
        "(rho, Ye): L2 isotonic regression (PAVA) followed by a minimum-slope\n"
        "strictification pass (see CODE.md \"Repair harness\"), then -- unless\n"
        "--no-spline-safe -- a per-column audit-driven smoothing loop\n"
        "(M2c-prime, eos-adapter-F-to-U.md S4) that fits the same C^2\n"
        "not-a-knot cubic B-spline the adapter build uses to each (rho, Ye)\n"
        "column and nudges any cell where the fitted spline's derivative dips\n"
        "to <= 0 between nodes (a near-plateau immediately adjacent to a steep\n"
        "recovery can ring the spline non-monotone there even though the raw\n"
        "data is monotone), then re-runs PAVA + strictification and re-audits,\n"
        "up to a bounded number of rounds. Then -- unless --no-spline-safe-3d\n"
        "-- a tensor-product 3D audit-driven diffusion stage (M2d-1, CODE.md\n"
        "\"open decision 4\") that fits the *whole* field as one tensor-product\n"
        "spline and fixes cross-column violations (two neighboring columns\n"
        "individually monotone but blending non-monotone between them) that no\n"
        "per-column repair can see, let alone fix. Finally -- unless\n"
        "--no-causal-cap -- the causal-cap stage (M3f,\n"
        "eos-causality-repair.md), which enforces the other hard requirement\n"
        "of eos-adapter-F-to-U.md S8: causality, 0 < c_s^2 < 1, of the\n"
        "potential U the adapter builds from those same two splines. It audits\n"
        "c_s^2 on a refined grid through the analytic chain rule and, wherever\n"
        "it finds a violation run reaching the rho_max edge, projects\n"
        "\"logenergy\" (only) onto the causal Lipschitz envelope along adiabats\n"
        "-- a logged physics edit to a region that is already unphysical by the\n"
        "table's own stored cs2 column, exactly like the monotonicity repair.\n"
        "Interior violation spikes (the sigma_T pockets) and c_s^2 <= 0 samples\n"
        "are reported, never edited. Structural problems (bad axes, non-finite\n"
        "values, missing required fields or attributes) are fatal and are never\n"
        "repaired -- they indicate a broken file, not physics noise.\n"
        "\n"
        "positional arguments:\n"
        "  IN.h5               input table (stellarcollapse.org / O'Connor-Ott\n"
        "                      HDF5 layout); never modified\n"
        "  OUT.h5              repaired output table; must differ from IN.h5\n"
        "\n"
        "options:\n"
        "  --check-only        report what would be repaired; write nothing\n"
        "  --min-slope-s X     minimum entropy slope per T grid step (absolute,\n"
        "                      kB/baryon; default: RepairOptions::min_slope_entropy)\n"
        "  --min-slope-loge X  minimum logenergy slope per T grid step (absolute,\n"
        "                      log10(erg/g); default: RepairOptions::min_slope_logenergy)\n"
        "  --no-spline-safe    skip the per-column spline-safe smoothing loop\n"
        "                      (RepairOptions::spline_safe = false); repair is\n"
        "                      then plain PAVA + strictification, as before\n"
        "                      M2c-prime. On by default.\n"
        "  --no-spline-safe-3d skip the tensor-product 3D spline-safe stage\n"
        "                      (RepairOptions::spline_safe_3d = false). On by\n"
        "                      default.\n"
        "  --no-causal-cap     skip the causal-cap stage\n"
        "                      (RepairOptions::causal_cap = false); the table's\n"
        "                      acausal high-density corner then passes through\n"
        "                      unchanged. On by default.\n"
        "  --cs2-cap X         target slope of the causality projection, in\n"
        "                      units of c^2 (RepairOptions::cs2_cap; default\n"
        "                      0.99 -- deliberately below the audit threshold\n"
        "                      cs2_max = 1.0, so the gap absorbs refit ringing\n"
        "                      and leaves the con2prim Newton some 1 - c_s^2\n"
        "                      conditioning headroom)\n"
        "  --log FILE          write a human-readable repair log to FILE (not\n"
        "                      valid together with --check-only)\n"
        "  -h, --help          print this message\n"
        "\n"
        "exit codes:\n"
        "  0   table already satisfied monotonicity (no changes needed/made)\n"
        "  1   one or more values were repaired (or would be, under --check-only)\n"
        "  2   fatal structural problem in IN.h5, or another runtime error (see stderr)\n"
        "  64  usage error\n";
}

struct ParsedArgs {
  bool check_only = false;
  bool have_min_slope_s = false;
  double min_slope_s = 0.0;
  bool have_min_slope_loge = false;
  double min_slope_loge = 0.0;
  bool no_spline_safe = false;
  bool no_spline_safe_3d = false;
  bool no_causal_cap = false;
  bool have_cs2_cap = false;
  double cs2_cap = 0.0;
  bool have_log = false;
  std::string log_path;
  std::vector<std::string> positionals;
};

// Parses argv into `out`. On success returns true; on -h/--help or any
// argument error, prints usage to stderr and returns false (main() maps that
// to exit 64, uniformly for both cases per CODE.md/the tool spec).
bool parse_args(const std::vector<std::string> &args, ParsedArgs &out) {
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string &a = args[i];

    if (a == "-h" || a == "--help") {
      print_usage(std::cerr);
      return false;
    } else if (a == "--check-only") {
      out.check_only = true;
    } else if (a == "--min-slope-s") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_repair: option '--min-slope-s' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      try {
        out.min_slope_s = std::stod(args[++i]);
      } catch (const std::exception &) {
        std::cerr << "eos_repair: invalid number for --min-slope-s: '" << args[i] << "'\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.have_min_slope_s = true;
    } else if (a == "--min-slope-loge") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_repair: option '--min-slope-loge' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      try {
        out.min_slope_loge = std::stod(args[++i]);
      } catch (const std::exception &) {
        std::cerr << "eos_repair: invalid number for --min-slope-loge: '" << args[i] << "'\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.have_min_slope_loge = true;
    } else if (a == "--no-spline-safe") {
      out.no_spline_safe = true;
    } else if (a == "--no-spline-safe-3d") {
      out.no_spline_safe_3d = true;
    } else if (a == "--no-causal-cap") {
      out.no_causal_cap = true;
    } else if (a == "--cs2-cap") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_repair: option '--cs2-cap' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      try {
        out.cs2_cap = std::stod(args[++i]);
      } catch (const std::exception &) {
        std::cerr << "eos_repair: invalid number for --cs2-cap: '" << args[i] << "'\n\n";
        print_usage(std::cerr);
        return false;
      }
      if (!(out.cs2_cap > 0.0) || !(out.cs2_cap < 1.0)) {
        std::cerr << "eos_repair: --cs2-cap must lie strictly between 0 and 1, got '" << args[i]
                  << "'\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.have_cs2_cap = true;
    } else if (a == "--log") {
      if (i + 1 >= args.size()) {
        std::cerr << "eos_repair: option '--log' requires a value\n\n";
        print_usage(std::cerr);
        return false;
      }
      out.log_path = args[++i];
      out.have_log = true;
    } else if (!a.empty() && a[0] == '-') {
      std::cerr << "eos_repair: unknown option '" << a << "'\n\n";
      print_usage(std::cerr);
      return false;
    } else {
      out.positionals.push_back(a);
    }
  }

  if (out.check_only) {
    if (out.have_log) {
      std::cerr << "eos_repair: --log is not valid together with --check-only\n\n";
      print_usage(std::cerr);
      return false;
    }
    if (out.positionals.size() != 1) {
      std::cerr << "eos_repair: --check-only takes exactly one positional argument (IN.h5)\n\n";
      print_usage(std::cerr);
      return false;
    }
  } else {
    if (out.positionals.size() != 2) {
      std::cerr << "eos_repair: expected exactly two positional arguments (IN.h5 OUT.h5)\n\n";
      print_usage(std::cerr);
      return false;
    }
  }

  return true;
}

// Human-readable log per CODE.md "Repair harness": header (tool version,
// input path, fnv1a in hex, options, timestamp), the RepairResult summary,
// then one line per RepairEntry.
void write_log(const std::string &log_path, const std::string &in_path, const std::string &out_path,
               unsigned long long input_fnv1a, const eeos::RepairOptions &options,
               const eeos::RepairResult &result) {
  std::ofstream log(log_path);
  if (!log) {
    throw std::runtime_error("could not open log file '" + log_path + "' for writing");
  }

  const std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm now_tm{};
#if defined(_WIN32)
  gmtime_s(&now_tm, &now_time);
#else
  gmtime_r(&now_time, &now_tm);
#endif

  log << "eos_repair log\n";
  log << "tool_version:        " << kToolVersion << "\n";
  log << "input:               " << in_path << "\n";
  log << "output:              " << out_path << "\n";
  log << "input_fnv1a:         0x" << std::hex << std::setw(16) << std::setfill('0') << input_fnv1a
      << std::dec << std::setfill(' ') << "\n";
  log << "min_slope_entropy:   " << options.min_slope_entropy << "\n";
  log << "min_slope_logenergy: " << options.min_slope_logenergy << "\n";
  log << "spline_safe:         " << (options.spline_safe ? "on" : "off") << "\n";
  log << "spline_safe_3d:      " << (options.spline_safe_3d ? "on" : "off") << "\n";
  log << "causal_cap:          " << (options.causal_cap ? "on" : "off") << "\n";
  log << "cs2_max:             " << options.cs2_max << "\n";
  log << "cs2_cap:             " << options.cs2_cap << "\n";
  log << "causal_rounds_max:   " << options.causal_rounds_max << "\n";
  log << "trace_depth_max:     " << options.trace_depth_max << "\n";
  log << "anchor_pad:          " << options.anchor_pad << "\n";
  log << "timestamp:           " << std::put_time(&now_tm, "%Y-%m-%dT%H:%M:%SZ") << "\n";
  log << "\n";

  result.print(log);

  if (!result.entries.empty()) {
    log << "\nentries (field (irho,jT,kYe) old -> new):\n";
    log << std::setprecision(10);
    for (const eeos::RepairEntry &e : result.entries) {
      log << e.field << " (" << e.irho << "," << e.jT << "," << e.kYe << ") " << e.old_value << " -> "
          << e.new_value << "\n";
    }
  }
}

// Per-round progress lines for the M2d-1 tensor-product 3D spline-safe
// stage (repair.hpp: "the library itself stays quiet ... tools/eos_repair
// prints these"): one line per field that ran the stage (non-empty
// rounds3d_violation_history), listing every recorded audit's violation
// *sample* count in order (main-loop rounds, then the final (4,4,4)
// verification's own audits -- see RepairResult::FieldSummary's doc
// comment for exactly what each entry means).
void print_rounds3d_progress(std::ostream &os, const eeos::RepairResult &result) {
  for (const eeos::RepairResult::FieldSummary &s : result.summaries) {
    if (s.rounds3d_violation_history.empty()) {
      continue;
    }
    os << "spline-safe-3d " << s.field << ":";
    for (size_t r = 0; r < s.rounds3d_violation_history.size(); ++r) {
      os << " round" << (r + 1) << "_violations=" << s.rounds3d_violation_history[r];
    }
    os << "\n";
  }
}

// The same per-round progress line for the M3f causal-cap stage (repair.hpp:
// the library stays quiet, the tool prints): every main-loop round's c_s^2
// violation *sample* count in order, then the final (4,4,4) verification's
// own count as the last entry.
void print_causal_cap_progress(std::ostream &os, const eeos::RepairResult &result) {
  const eeos::RepairResult::CausalCapSummary &c = result.causal_cap;
  if (c.rounds_violation_history.empty()) {
    return;
  }
  os << "causal-cap:";
  for (size_t r = 0; r + 1 < c.rounds_violation_history.size(); ++r) {
    os << " round" << (r + 1) << "_violations=" << c.rounds_violation_history[r];
  }
  os << " verify444_violations=" << c.rounds_violation_history.back() << "\n";
}

} // namespace

int main(int argc, char **argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);

  ParsedArgs pa;
  if (!parse_args(args, pa)) {
    return kExitUsage;
  }

  const std::string in_path = pa.positionals[0];
  const std::string out_path = pa.check_only ? std::string() : pa.positionals[1];

  if (!pa.check_only && out_path == in_path) {
    std::cerr << "eos_repair: OUT.h5 must differ from IN.h5 -- refusing to modify the input\n\n";
    print_usage(std::cerr);
    return kExitUsage;
  }

  try {
    eeos::RawTable table = eeos::read_stellarcollapse(in_path);

    // Structural problems are fatal, never repaired (CODE.md "Repair
    // harness"): check first, and bail before touching anything.
    const eeos::CheckReport report = eeos::check_table(table);
    if (report.status == eeos::Status::fatal) {
      report.print(std::cerr);
      return kExitFatal;
    }

    eeos::RepairOptions repair_opts;
    if (pa.have_min_slope_s) {
      repair_opts.min_slope_entropy = pa.min_slope_s;
    }
    if (pa.have_min_slope_loge) {
      repair_opts.min_slope_logenergy = pa.min_slope_loge;
    }
    if (pa.no_spline_safe) {
      repair_opts.spline_safe = false;
    }
    if (pa.no_spline_safe_3d) {
      repair_opts.spline_safe_3d = false;
    }
    if (pa.no_causal_cap) {
      repair_opts.causal_cap = false;
    }
    if (pa.have_cs2_cap) {
      repair_opts.cs2_cap = pa.cs2_cap;
    }

    eeos::RepairResult result = eeos::repair_table(table, repair_opts);
    print_rounds3d_progress(std::cout, result);
    print_causal_cap_progress(std::cout, result);

    if (pa.check_only) {
      std::cout << "eos_repair --check-only: "
                << (result.entries.empty()
                        ? std::string("table already satisfies monotonicity (0 changes)")
                        : "would repair " + std::to_string(result.entries.size()) + " value(s)")
                << "\n";
      result.print(std::cout);
      return result.entries.empty() ? kExitOk : kExitChanged;
    }

    eeos::write_stellarcollapse(out_path, table, in_path);
    const unsigned long long input_fnv1a = eeos::fnv1a_file(in_path);
    eeos::append_repair_group(out_path, result, repair_opts, in_path, input_fnv1a, kToolVersion);

    std::cout << "eos_repair: wrote '" << out_path << "'\n";
    result.print(std::cout);

    if (pa.have_log) {
      write_log(pa.log_path, in_path, out_path, input_fnv1a, repair_opts, result);
    }

    return result.entries.empty() ? kExitOk : kExitChanged;
  } catch (const std::exception &e) {
    std::cerr << "eos_repair: " << e.what() << "\n";
    return kExitFatal;
  }
}
