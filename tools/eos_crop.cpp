// tools/eos_crop.cpp — crops a stellarcollapse-format EOS table to a small
// index-range box, for use as a fast sanitizer-build test fixture (see
// CODE.md "Test harness": "committed ~10 MB crops of the pathological
// regions"). A thin main() over entropy_eos, in the tools/eos_repair.cpp
// style: no physics or table logic lives here, only argument parsing and
// calls into read_stellarcollapse()/write_stellarcollapse().
//
//   eos_crop IN.h5 OUT.h5 --irho A B --jt A B --kye A B
//
// Reads IN.h5 via read_stellarcollapse(), slices the three axes and every
// field to the given INCLUSIVE index ranges (values copied verbatim -- no
// unit conversion, no reinterpretation), copies every attribute
// (energy_shift, and have_rel_cs2 if present) unchanged, and writes the
// result via the source-less write_stellarcollapse(path, table) overload.
//
// That overload has no source file to draw passthrough datasets from, so
// anything read_stellarcollapse() does not surface as a RawTable field or
// attribute -- opaque provenance blobs, "points*" (regenerated fresh from
// the cropped axis sizes) -- is intentionally NOT carried into OUT.h5. This
// is fine for a numerics test fixture (axes, fields, and energy_shift are
// exactly what the library reads) but OUT.h5 is not a byte-faithful
// sub-table of IN.h5 in the way eos_repair's output is a byte-faithful
// superset of its input.
//
// Exit codes: 0 = wrote OUT.h5; 64 = usage error (bad/missing arguments, an
// index range outside the input axis, or a cropped axis with fewer than 4
// points -- too small for the library's uniform-knot cubic B-spline fit);
// 2 = a read/write error at runtime (bad file, HDF5 failure, ...).

#include "entropy_eos/entropy_eos.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 2;
constexpr int kExitUsage = 64;

constexpr size_t kMinAxisPoints = 4;

void print_usage(std::ostream &os) {
  os << "usage:\n"
        "  eos_crop IN.h5 OUT.h5 --irho A B --jt A B --kye A B\n"
        "  eos_crop --version\n"
        "\n"
        "Crops a stellarcollapse-format (O'Connor-Ott) EOS table to the\n"
        "inclusive index box [A,B] on each of the three axes (0-based:\n"
        "irho into \"logrho\", jt into \"logtemp\", kye into \"ye\") and\n"
        "writes the result as a new, smaller stellarcollapse-format table --\n"
        "intended as a small local test fixture standing in for a full real\n"
        "table under sanitizer-instrumented test runs (see CODE.md \"Test\n"
        "harness\").\n"
        "\n"
        "Every field is sliced verbatim (the stored values are copied\n"
        "as-is; no unit conversion, no reinterpretation), and every\n"
        "attribute (\"energy_shift\", and \"have_rel_cs2\" if present) is\n"
        "copied unchanged. The output is written with the source-less\n"
        "write_stellarcollapse(path, table) overload, which has no input\n"
        "file to copy passthrough datasets from: \"pointsrho\"/\"pointstemp\"/\n"
        "\"pointsye\" are regenerated fresh from the cropped axis sizes, and\n"
        "opaque provenance blobs that read_stellarcollapse() does not\n"
        "surface as a RawTable field/attribute are intentionally NOT carried\n"
        "into the output.\n"
        "\n"
        "positional arguments:\n"
        "  IN.h5               input table (stellarcollapse.org / O'Connor-Ott\n"
        "                      HDF5 layout); never modified\n"
        "  OUT.h5              cropped output table; must differ from IN.h5\n"
        "\n"
        "options (all required):\n"
        "  --irho A B          inclusive index range on the rho axis (0-based,\n"
        "                      into \"logrho\"/every field's fastest index)\n"
        "  --jt A B            inclusive index range on the temperature axis\n"
        "                      (0-based, into \"logtemp\")\n"
        "  --kye A B           inclusive index range on the Ye axis (0-based,\n"
        "                      into \"ye\")\n"
        "  -h, --help          print this message\n"
        "\n"
        "exit codes:\n"
        "  0   wrote OUT.h5\n"
        "  2   a read/write error at runtime (bad file, HDF5 failure, ...)\n"
        "  64  usage error: bad/missing arguments, an index range outside the\n"
        "      input axis (or A > B), or a cropped axis with fewer than "
     << kMinAxisPoints
     << "\n"
        "      points (too small for the library's uniform-knot cubic\n"
        "      B-spline fit)\n";
}

struct AxisRange {
  bool have = false;
  long a = 0, b = 0;
};

struct ParsedArgs {
  std::vector<std::string> positionals;
  AxisRange irho, jt, kye;
};

bool parse_range(const std::vector<std::string> &args, size_t &i, const char *opt_name, AxisRange &out) {
  if (i + 2 >= args.size()) {
    std::cerr << "eos_crop: option '" << opt_name << "' requires two values (A B)\n\n";
    return false;
  }
  try {
    out.a = std::stol(args[i + 1]);
    out.b = std::stol(args[i + 2]);
  } catch (const std::exception &) {
    std::cerr << "eos_crop: invalid integer for '" << opt_name << "': '" << args[i + 1] << "' '"
              << args[i + 2] << "'\n\n";
    return false;
  }
  out.have = true;
  i += 2;
  return true;
}

// Parses argv into `out`. On success returns true; on -h/--help or any
// argument error, prints usage to stderr and returns false (main() maps
// that to exit 64, uniformly for both cases, per tools/eos_repair.cpp's
// convention).
bool parse_args(const std::vector<std::string> &args, ParsedArgs &out) {
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string &a = args[i];

    if (a == "-h" || a == "--help") {
      print_usage(std::cerr);
      return false;
    } else if (a == "--irho") {
      if (!parse_range(args, i, "--irho", out.irho)) {
        print_usage(std::cerr);
        return false;
      }
    } else if (a == "--jt") {
      if (!parse_range(args, i, "--jt", out.jt)) {
        print_usage(std::cerr);
        return false;
      }
    } else if (a == "--kye") {
      if (!parse_range(args, i, "--kye", out.kye)) {
        print_usage(std::cerr);
        return false;
      }
    } else if (!a.empty() && a[0] == '-') {
      std::cerr << "eos_crop: unknown option '" << a << "'\n\n";
      print_usage(std::cerr);
      return false;
    } else {
      out.positionals.push_back(a);
    }
  }

  if (out.positionals.size() != 2) {
    std::cerr << "eos_crop: expected exactly two positional arguments (IN.h5 OUT.h5)\n\n";
    print_usage(std::cerr);
    return false;
  }
  if (!out.irho.have || !out.jt.have || !out.kye.have) {
    std::cerr << "eos_crop: --irho, --jt, and --kye are all required\n\n";
    print_usage(std::cerr);
    return false;
  }

  return true;
}

// Validates one axis range against the input axis size: A <= B, both within
// [0, n), and the resulting inclusive count >= kMinAxisPoints. Prints a
// specific message and usage on failure.
bool validate_range(const AxisRange &r, size_t n, const char *label) {
  if (r.a < 0 || r.b < 0 || static_cast<size_t>(r.a) >= n || static_cast<size_t>(r.b) >= n || r.a > r.b) {
    std::cerr << "eos_crop: --" << label << " " << r.a << " " << r.b << " is not a valid inclusive range "
              << "into an axis of size " << n << "\n\n";
    print_usage(std::cerr);
    return false;
  }
  const size_t count = static_cast<size_t>(r.b - r.a) + 1;
  if (count < kMinAxisPoints) {
    std::cerr << "eos_crop: --" << label << " " << r.a << " " << r.b << " yields only " << count
              << " point(s); at least " << kMinAxisPoints
              << " are required (the library's cubic B-spline fit needs a uniform-knot axis of at "
                 "least this many points)\n\n";
    print_usage(std::cerr);
    return false;
  }
  return true;
}

// Slices `in`'s axes and every field to the inclusive index box
// [ia,ib]x[ja,jb]x[ka,kb] (rho x T x Ye), copying values verbatim, and
// copies every attribute unchanged.
eeos::RawTable crop_table(const eeos::RawTable &in, size_t ia, size_t ib, size_t ja, size_t jb, size_t ka,
                           size_t kb) {
  eeos::RawTable out;

  std::vector<double> logrho(in.logrho().begin() + static_cast<long>(ia),
                              in.logrho().begin() + static_cast<long>(ib) + 1);
  std::vector<double> logtemp(in.logtemp().begin() + static_cast<long>(ja),
                               in.logtemp().begin() + static_cast<long>(jb) + 1);
  std::vector<double> ye(in.ye().begin() + static_cast<long>(ka), in.ye().begin() + static_cast<long>(kb) + 1);
  out.set_axes(std::move(logrho), std::move(logtemp), std::move(ye));

  for (const std::string &name : in.field_names()) {
    const std::vector<double> &src = in.field(name);
    std::vector<double> dst(out.nrho() * out.ntemp() * out.nye());
    for (size_t k = 0; k < out.nye(); ++k) {
      for (size_t j = 0; j < out.ntemp(); ++j) {
        for (size_t i = 0; i < out.nrho(); ++i) {
          dst[out.index(i, j, k)] = src[in.index(ia + i, ja + j, ka + k)];
        }
      }
    }
    out.add_field(name, std::move(dst));
  }

  for (const std::string &attr_name : in.attribute_names()) {
    out.add_attribute(attr_name, in.attribute(attr_name));
  }

  return out;
}

} // namespace

int main(int argc, char **argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);

  // Answered before argument validation (GNU convention): asking which
  // entropy_eos a binary came from should not require valid arguments.
  if (std::find(args.begin(), args.end(), "--version") != args.end()) {
    std::cout << "eos_crop (entropy_eos) " EEOS_VERSION_STRING "\n";
    return kExitOk;
  }

  ParsedArgs pa;
  if (!parse_args(args, pa)) {
    return kExitUsage;
  }

  const std::string in_path = pa.positionals[0];
  const std::string out_path = pa.positionals[1];

  if (out_path == in_path) {
    std::cerr << "eos_crop: OUT.h5 must differ from IN.h5 -- refusing to modify the input\n\n";
    print_usage(std::cerr);
    return kExitUsage;
  }

  try {
    const eeos::RawTable in_table = eeos::read_stellarcollapse(in_path);

    if (!validate_range(pa.irho, in_table.nrho(), "irho") ||
        !validate_range(pa.jt, in_table.ntemp(), "jt") || !validate_range(pa.kye, in_table.nye(), "kye")) {
      return kExitUsage;
    }

    const auto ia = static_cast<size_t>(pa.irho.a), ib = static_cast<size_t>(pa.irho.b);
    const auto ja = static_cast<size_t>(pa.jt.a), jb = static_cast<size_t>(pa.jt.b);
    const auto ka = static_cast<size_t>(pa.kye.a), kb = static_cast<size_t>(pa.kye.b);

    const eeos::RawTable out_table = crop_table(in_table, ia, ib, ja, jb, ka, kb);

    const unsigned long long source_fnv1a = eeos::fnv1a_file(in_path);

    eeos::write_stellarcollapse(out_path, out_table);

    std::ifstream out_check(out_path, std::ios::binary | std::ios::ate);
    const std::streamoff out_size =
        out_check.is_open() ? static_cast<std::streamoff>(out_check.tellg()) : std::streamoff(-1);

    std::cout << "eos_crop: source='" << in_path << "' source_fnv1a=0x" << std::hex << source_fnv1a
               << std::dec << "\n"
               << "  irho=[" << ia << "," << ib << "] jt=[" << ja << "," << jb << "] kye=[" << ka << ","
               << kb << "]\n"
               << "  cropped axes: " << out_table.nrho() << " x " << out_table.ntemp() << " x "
               << out_table.nye() << "\n"
               << "  wrote '" << out_path << "' (" << out_size << " bytes)\n";

    return kExitOk;
  } catch (const std::exception &e) {
    std::cerr << "eos_crop: " << e.what() << "\n";
    return kExitError;
  }
}
