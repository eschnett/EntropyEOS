// tools/eos_test.cpp — M1 test harness, "--level table" (CODE.md "Test
// harness"). One binary that will accrete "--level adapter" / "--level
// con2prim" stages in M2/M3; this milestone implements only "table".
//
// A thin main() over entropy_eos: loads a table (from a stellarcollapse
// file, or an in-memory synthetic ideal-gas table, optionally with seeded
// violations), runs check_table(), prints its report, optionally dumps the
// worst offenders per class to CSV, and exits with a status summarizing
// whether any hard violation was found. No physics or table logic lives here
// -- see entropy_eos/host/{check,synthetic}.hpp for that.
//
//   eos_test [--level table] (FILE.h5 | --synthetic | --synthetic-seeded)
//            [--write-synthetic PATH] [--csv PREFIX] [--tol X] [--worst N]
//            [--m-B GRAMS]
//
// Exit codes: 2 = fatal structural problem; 1 = "entropy_negative",
// "entropy_nonmonotone_T", "logenergy_nonmonotone_T", or any
// "nonfinite_<field>" class has count > 0 (the
// Maxwell-consistency and cs2 classes are diagnostics only and never affect
// the exit code); 0 = otherwise clean; 64 = usage error.

#include "entropy_eos/entropy_eos.hpp"

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
        "  eos_test [--level table] FILE.h5 [options]\n"
        "  eos_test [--level table] --synthetic [options]\n"
        "  eos_test [--level table] --synthetic-seeded [options]\n"
        "\n"
        "Runs check_table() (CODE.md \"Test harness\") against a stellarcollapse-\n"
        "format table, an in-memory synthetic ideal-gas table, or a synthetic\n"
        "table with 4 deliberately seeded monotonicity violations, and prints a\n"
        "human-readable report to stdout.\n"
        "\n"
        "input (exactly one required):\n"
        "  FILE.h5              read a stellarcollapse-format table from FILE.h5\n"
        "  --synthetic          use make_synthetic_table() with default options\n"
        "  --synthetic-seeded   like --synthetic, plus 4 seeded violations (two on\n"
        "                       \"entropy\", two on \"logenergy\") at interior grid\n"
        "                       points, printed to stdout before the report\n"
        "\n"
        "options:\n"
        "  --level LEVEL        check level; only \"table\" is implemented (default).\n"
        "                       Other values exit 64 with a stub message (M2/M3 add\n"
        "                       \"adapter\" / \"con2prim\").\n"
        "  --write-synthetic PATH  write the generated synthetic table to PATH (only\n"
        "                          valid together with --synthetic or\n"
        "                          --synthetic-seeded)\n"
        "  --csv PREFIX         for every check class with count > 0, write\n"
        "                       PREFIX_<class>.csv (header: irho,jT,kYe,rho,temp,ye,\n"
        "                       value), one row per worst-offender entry. The list\n"
        "                       is capped at --worst N; pass a large N for a full\n"
        "                       dump.\n"
        "  --tol X              CheckOptions::tol_consistency override\n"
        "  --worst N            CheckOptions::worst_n override (also caps --csv)\n"
        "  --m-B GRAMS          CheckOptions::m_B_g override\n"
        "  -h, --help           print this message\n"
        "\n"
        "exit codes:\n"
        "  0   status ok and no entropy/monotonicity violations\n"
        "  1   \"entropy_negative\", \"entropy_nonmonotone_T\",\n"
        "      \"logenergy_nonmonotone_T\", or any \"nonfinite_<field>\" class has\n"
        "      count > 0 (the Maxwell-consistency and cs2 classes are diagnostics\n"
        "      only and never affect the exit code)\n"
        "  2   fatal structural problem in the table (see the printed report)\n"
        "  64  usage error\n";
}

struct ParsedArgs {
  std::string level = "table";
  bool synthetic = false;
  bool synthetic_seeded = false;
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

  const int n_inputs =
      (!out.file_path.empty() ? 1 : 0) + (out.synthetic ? 1 : 0) + (out.synthetic_seeded ? 1 : 0);
  if (n_inputs != 1) {
    std::cerr << "eos_test: specify exactly one of FILE.h5, --synthetic, --synthetic-seeded\n\n";
    print_usage(std::cerr);
    return false;
  }

  if (out.have_write_synthetic && !(out.synthetic || out.synthetic_seeded)) {
    std::cerr << "eos_test: --write-synthetic is only valid together with --synthetic or "
                 "--synthetic-seeded\n\n";
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

} // namespace

int main(int argc, char **argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);

  ParsedArgs pa;
  if (!parse_args(args, pa)) {
    return kExitUsage;
  }

  if (pa.level != "table") {
    std::cerr << "eos_test: level not implemented yet (lands with M2/M3)\n";
    return kExitUsage;
  }

  try {
    eeos::RawTable table;
    if (!pa.file_path.empty()) {
      table = eeos::read_stellarcollapse(pa.file_path);
    } else {
      eeos::SyntheticOptions sopts; // defaults per CODE.md
      if (pa.synthetic_seeded) {
        sopts.seed = make_seeds(sopts);
        std::cout << "eos_test --synthetic-seeded: seeding " << sopts.seed.size()
                  << " violation(s):\n";
        for (const eeos::SeededViolation &v : sopts.seed) {
          std::cout << "  " << v.field << " at (irho=" << v.irho << ", jT=" << v.jT
                     << ", kYe=" << v.kYe << ") delta=" << v.delta << "\n";
        }
      }
      table = eeos::make_synthetic_table(sopts);
      if (pa.have_write_synthetic) {
        eeos::write_stellarcollapse(pa.write_synthetic_path, table);
      }
    }

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
      for (const eeos::CheckClassResult &c : report.classes) {
        if (c.count == 0) {
          continue;
        }
        const std::string path = pa.csv_prefix + "_" + c.name + ".csv";
        std::ofstream csv(path);
        if (!csv) {
          throw std::runtime_error("could not open CSV file '" + path + "' for writing");
        }
        csv << "irho,jT,kYe,rho,temp,ye,value\n";
        csv << std::setprecision(17);
        for (const eeos::CheckClassResult::Loc &loc : c.worst) {
          csv << loc.irho << "," << loc.jT << "," << loc.kYe << "," << loc.rho << "," << loc.temp
              << "," << loc.ye << "," << loc.value << "\n";
        }
      }
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
          c.name == "logenergy_nonmonotone_T" ||
          c.name.rfind("nonfinite_", 0) == 0) {
        hard_violation = true;
      }
    }
    return hard_violation ? kExitViolation : kExitOk;
  } catch (const std::exception &e) {
    std::cerr << "eos_test: " << e.what() << "\n";
    return kExitFatal;
  }
}
