// tests/test_io_stellarcollapse.cpp — unit tests for
// entropy_eos::{read,write}_stellarcollapse, append_repair_group, and
// fnv1a_file (entropy_eos/host/io_stellarcollapse.hpp).
//
// This test file itself includes <hdf5.h> and makes raw HDF5 calls to
// independently verify what the module under test writes -- that is a test
// concern only; the confinement of HDF5 to a single translation unit
// (CODE.md "Environment") applies to the *library*, not to tests/, which
// (per CODE.md "Unit tests and CI") is never copied into a downstream
// consumer's build.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <hdf5.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "entropy_eos/host/io_stellarcollapse.hpp"
#include "entropy_eos/host/repair.hpp"
#include "entropy_eos/host/synthetic.hpp"
#include "entropy_eos/host/table.hpp"

using eeos::RawTable;
using eeos::RepairEntry;
using eeos::RepairOptions;
using eeos::RepairResult;
using eeos::SyntheticOptions;

namespace {

// Scratch files go here, per the task's sandboxing convention -- never into
// the repo's own tables/ or tests/ directories.
const std::string kScratchDir =
    "/private/tmp/claude-505/-Users-eschnett-src-EntropyEOS/934472f4-9180-46bf-b6f7-d5c721742315/"
    "scratchpad/";

std::string scratch_path(const std::string &name) { return kScratchDir + name; }

// The two real ground-truth files (see the task description); tests that
// use them are guarded by table_exists() so CI (which has no tables/)
// skips them cleanly.
const std::string kLS220Path = "tables/LS220_234r_136t_50y_analmu_20091212_SVNr26.h5";
const std::string kSROPath = "tables/LS220_3335_rho391_temp163_ye66.h5";

bool table_exists(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

// --- bitwise comparison helpers (verbatim-storage checks; see table.hpp) ---

bool same_bits(double a, double b) {
  static_assert(sizeof(double) == sizeof(std::uint64_t), "double must be 64-bit");
  std::uint64_t bits_a, bits_b;
  std::memcpy(&bits_a, &a, sizeof(double));
  std::memcpy(&bits_b, &b, sizeof(double));
  return bits_a == bits_b;
}

bool same_bits_vec(const std::vector<double> &a, const std::vector<double> &b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (!same_bits(a[i], b[i])) {
      return false;
    }
  }
  return true;
}

// --- tiny RAII guard for the raw HDF5 calls this test file makes directly
// (separate from, and no smaller than, the one in io_stellarcollapse.cpp --
// duplicated here rather than exposed from the library, since it is purely
// a test-verification convenience) ---

class H5Guard {
public:
  H5Guard(hid_t id, herr_t (*closer)(hid_t)) : id_(id), closer_(closer) {}
  ~H5Guard() {
    if (id_ >= 0) {
      closer_(id_);
    }
  }
  H5Guard(const H5Guard &) = delete;
  H5Guard &operator=(const H5Guard &) = delete;
  hid_t get() const { return id_; }
  bool valid() const { return id_ >= 0; }

private:
  hid_t id_;
  herr_t (*closer_)(hid_t);
};

// --- raw-HDF5 verification helpers -----------------------------------------

size_t count_root_datasets(const std::string &path) {
  H5Guard file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!file.valid()) {
    throw std::runtime_error("test helper: cannot open '" + path + "'");
  }
  size_t count = 0;
  auto cb = [](hid_t, const char *, const H5L_info_t *, void *op_data) -> herr_t {
    ++(*static_cast<size_t *>(op_data));
    return 0;
  };
  if (H5Literate(file.get(), H5_INDEX_NAME, H5_ITER_INC, nullptr, cb, &count) < 0) {
    throw std::runtime_error("test helper: failed to enumerate '" + path + "'");
  }
  return count;
}

long long read_scalar_number(const std::string &path, const std::string &name) {
  H5Guard file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!file.valid()) {
    throw std::runtime_error("test helper: cannot open '" + path + "'");
  }
  H5Guard dset(H5Dopen(file.get(), name.c_str(), H5P_DEFAULT), H5Dclose);
  if (!dset.valid()) {
    throw std::runtime_error("test helper: no dataset '" + name + "' in '" + path + "'");
  }
  double value = 0.0;
  if (H5Dread(dset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value) < 0) {
    throw std::runtime_error("test helper: failed reading '" + name + "'");
  }
  return std::llround(value);
}

std::vector<unsigned char> read_raw_dataset_bytes(const std::string &path, const std::string &name) {
  H5Guard file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!file.valid()) {
    throw std::runtime_error("test helper: cannot open '" + path + "'");
  }
  H5Guard dset(H5Dopen(file.get(), name.c_str(), H5P_DEFAULT), H5Dclose);
  if (!dset.valid()) {
    throw std::runtime_error("test helper: no dataset '" + name + "' in '" + path + "'");
  }
  H5Guard space(H5Dget_space(dset.get()), H5Sclose);
  const hssize_t npoints = H5Sget_simple_extent_npoints(space.get());
  H5Guard dtype(H5Dget_type(dset.get()), H5Tclose);
  const size_t elem_size = H5Tget_size(dtype.get());
  std::vector<unsigned char> buf(static_cast<size_t>(npoints) * elem_size);
  // Read with the dataset's own stored type as the memory type: for the
  // opaque 1-byte blobs this test uses this on, that's a plain byte-for-byte
  // copy, no conversion involved.
  if (H5Dread(dset.get(), dtype.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data()) < 0) {
    throw std::runtime_error("test helper: failed reading '" + name + "'");
  }
  return buf;
}

std::vector<unsigned> read_1d_uint_dataset(hid_t loc, const std::string &name) {
  H5Guard dset(H5Dopen(loc, name.c_str(), H5P_DEFAULT), H5Dclose);
  if (!dset.valid()) {
    throw std::runtime_error("test helper: no dataset '" + name + "'");
  }
  H5Guard space(H5Dget_space(dset.get()), H5Sclose);
  const hssize_t n = H5Sget_simple_extent_npoints(space.get());
  std::vector<unsigned> data(static_cast<size_t>(n));
  if (H5Dread(dset.get(), H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
    throw std::runtime_error("test helper: failed reading '" + name + "'");
  }
  return data;
}

std::vector<double> read_1d_double_dataset(hid_t loc, const std::string &name) {
  H5Guard dset(H5Dopen(loc, name.c_str(), H5P_DEFAULT), H5Dclose);
  if (!dset.valid()) {
    throw std::runtime_error("test helper: no dataset '" + name + "'");
  }
  H5Guard space(H5Dget_space(dset.get()), H5Sclose);
  const hssize_t n = H5Sget_simple_extent_npoints(space.get());
  std::vector<double> data(static_cast<size_t>(n));
  if (H5Dread(dset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
    throw std::runtime_error("test helper: failed reading '" + name + "'");
  }
  return data;
}

// Length of the NUL-terminated string stored in the first `maxlen` bytes at
// `s` (a hand-rolled strnlen: avoids depending on the POSIX extension).
size_t fixed_string_length(const char *s, size_t maxlen) {
  size_t n = 0;
  while (n < maxlen && s[n] != '\0') {
    ++n;
  }
  return n;
}

std::vector<std::string> read_fixed_strings_dataset(hid_t loc, const std::string &name) {
  H5Guard dset(H5Dopen(loc, name.c_str(), H5P_DEFAULT), H5Dclose);
  if (!dset.valid()) {
    throw std::runtime_error("test helper: no dataset '" + name + "'");
  }
  H5Guard space(H5Dget_space(dset.get()), H5Sclose);
  const hssize_t n = H5Sget_simple_extent_npoints(space.get());
  H5Guard dtype(H5Dget_type(dset.get()), H5Tclose);
  const size_t len = H5Tget_size(dtype.get());
  std::vector<char> buf(static_cast<size_t>(n) * len);
  if (H5Dread(dset.get(), dtype.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data()) < 0) {
    throw std::runtime_error("test helper: failed reading '" + name + "'");
  }
  std::vector<std::string> result(static_cast<size_t>(n));
  for (size_t i = 0; i < result.size(); ++i) {
    const char *p = buf.data() + i * len;
    result[i].assign(p, fixed_string_length(p, len));
  }
  return result;
}

double read_double_attr(hid_t loc, const std::string &name) {
  H5Guard attr(H5Aopen(loc, name.c_str(), H5P_DEFAULT), H5Aclose);
  if (!attr.valid()) {
    throw std::runtime_error("test helper: no attribute '" + name + "'");
  }
  double value = 0.0;
  if (H5Aread(attr.get(), H5T_NATIVE_DOUBLE, &value) < 0) {
    throw std::runtime_error("test helper: failed reading attribute '" + name + "'");
  }
  return value;
}

std::uint64_t read_uint64_attr(hid_t loc, const std::string &name) {
  H5Guard attr(H5Aopen(loc, name.c_str(), H5P_DEFAULT), H5Aclose);
  if (!attr.valid()) {
    throw std::runtime_error("test helper: no attribute '" + name + "'");
  }
  std::uint64_t value = 0;
  if (H5Aread(attr.get(), H5T_NATIVE_UINT64, &value) < 0) {
    throw std::runtime_error("test helper: failed reading attribute '" + name + "'");
  }
  return value;
}

std::string read_string_attr(hid_t loc, const std::string &name) {
  H5Guard attr(H5Aopen(loc, name.c_str(), H5P_DEFAULT), H5Aclose);
  if (!attr.valid()) {
    throw std::runtime_error("test helper: no attribute '" + name + "'");
  }
  H5Guard dtype(H5Aget_type(attr.get()), H5Tclose);
  const size_t len = H5Tget_size(dtype.get());
  std::vector<char> buf(len);
  if (H5Aread(attr.get(), dtype.get(), buf.data()) < 0) {
    throw std::runtime_error("test helper: failed reading attribute '" + name + "'");
  }
  return std::string(buf.data(), fixed_string_length(buf.data(), len));
}

} // namespace

// --- (1) synthetic round trip (no source file) ------------------------------

TEST_CASE("write/read_stellarcollapse (source-less): synthetic round trip is exact") {
  SyntheticOptions opts;
  opts.nrho = 6;
  opts.ntemp = 5;
  opts.nye = 3;
  RawTable original = eeos::make_synthetic_table(opts);

  const std::string path = scratch_path("synthetic_roundtrip.h5");
  eeos::write_stellarcollapse(path, original);

  RawTable back = eeos::read_stellarcollapse(path);

  REQUIRE(back.nrho() == original.nrho());
  REQUIRE(back.ntemp() == original.ntemp());
  REQUIRE(back.nye() == original.nye());
  CHECK(same_bits_vec(original.logrho(), back.logrho()));
  CHECK(same_bits_vec(original.logtemp(), back.logtemp()));
  CHECK(same_bits_vec(original.ye(), back.ye()));

  // Field order: make_synthetic_table() adds "logenergy", "entropy",
  // "logpress" in that (non-alphabetical) order. A plain by-name HDF5
  // listing would come back alphabetized ("entropy", "logenergy",
  // "logpress"); this module tracks link creation order on files it writes
  // specifically so this round trip reproduces the original order exactly
  // (see list_root_names_in_file_order()'s comment in the .cpp).
  CHECK(back.field_names() == original.field_names());
  for (const std::string &name : original.field_names()) {
    REQUIRE(back.has_field(name));
    CHECK(same_bits_vec(original.field(name), back.field(name)));
  }

  REQUIRE(back.has_attribute("energy_shift"));
  CHECK(back.energy_shift() == original.energy_shift());
}

// --- (2) passthrough round trip ---------------------------------------------

TEST_CASE("write_stellarcollapse (with source): passthrough round trip is exact") {
  SyntheticOptions opts;
  opts.nrho = 5;
  opts.ntemp = 4;
  opts.nye = 3;
  RawTable source_table = eeos::make_synthetic_table(opts);

  const std::string path_a = scratch_path("passthrough_a.h5");
  eeos::write_stellarcollapse(path_a, source_table);
  RawTable a = eeos::read_stellarcollapse(path_a);

  const std::string path_b = scratch_path("passthrough_b.h5");
  eeos::write_stellarcollapse(path_b, a, path_a);
  RawTable b = eeos::read_stellarcollapse(path_b);

  REQUIRE(a.nrho() == b.nrho());
  REQUIRE(a.ntemp() == b.ntemp());
  REQUIRE(a.nye() == b.nye());
  CHECK(same_bits_vec(a.logrho(), b.logrho()));
  CHECK(same_bits_vec(a.logtemp(), b.logtemp()));
  CHECK(same_bits_vec(a.ye(), b.ye()));
  CHECK(a.field_names() == b.field_names());
  for (const std::string &name : a.field_names()) {
    REQUIRE(b.has_field(name));
    CHECK(same_bits_vec(a.field(name), b.field(name)));
  }
  REQUIRE(b.has_attribute("energy_shift"));
  CHECK(a.energy_shift() == b.energy_shift());

  // points* datasets exist in path_b with the correct values -- verified
  // with raw HDF5 calls, since RawTable never stores them (they are
  // pure passthrough / re-derived-on-write, never a RawTable field).
  CHECK(read_scalar_number(path_b, "pointsrho") == static_cast<long long>(b.nrho()));
  CHECK(read_scalar_number(path_b, "pointstemp") == static_cast<long long>(b.ntemp()));
  CHECK(read_scalar_number(path_b, "pointsye") == static_cast<long long>(b.nye()));
}

// --- (2b) opaque-blob passthrough without a real table -----------------------
//
// Exercises the same opaque-provenance-blob passthrough guarantee as (5)'s
// "SRO opaque blob survives byte-identical" test, but entirely with a
// synthetic table and a hand-added dataset -- no tables/*.h5 file required,
// so this runs (and matters) in CI, which has no real tables.

TEST_CASE("write_stellarcollapse (with source): a hand-added opaque uint8 blob survives "
          "byte-identical with no real table present") {
  SyntheticOptions opts;
  opts.nrho = 6;
  opts.ntemp = 5;
  opts.nye = 3;
  RawTable table = eeos::make_synthetic_table(opts);

  const std::string path_in = scratch_path("fake_provenance_in.h5");
  eeos::write_stellarcollapse(path_in, table);

  // A few hundred fixed, non-constant bytes (not all-zero/all-equal, so a
  // bug that zeroes or constant-fills the copy would still be caught).
  std::vector<unsigned char> blob(300);
  for (size_t i = 0; i < blob.size(); ++i) {
    blob[i] = static_cast<unsigned char>((i * 37 + 11) % 256);
  }

  {
    H5Guard file(H5Fopen(path_in.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose);
    REQUIRE(file.valid());
    const hsize_t dim = static_cast<hsize_t>(blob.size());
    H5Guard space(H5Screate_simple(1, &dim, nullptr), H5Sclose);
    REQUIRE(space.valid());
    H5Guard dset(H5Dcreate(file.get(), "FAKE-provenance.in", H5T_NATIVE_UCHAR, space.get(),
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                 H5Dclose);
    REQUIRE(dset.valid());
    REQUIRE(H5Dwrite(dset.get(), H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, blob.data()) >= 0);
  }

  // read_stellarcollapse() must leave the blob out of the RawTable entirely
  // (it matches none of the axis/field/attribute rules), while the real
  // fields are unaffected.
  RawTable read_back = eeos::read_stellarcollapse(path_in);
  CHECK_FALSE(read_back.has_field("FAKE-provenance.in"));
  CHECK(read_back.has_field("logenergy"));
  CHECK(read_back.has_field("entropy"));
  CHECK(read_back.has_field("logpress"));

  const std::string path_out = scratch_path("fake_provenance_out.h5");
  eeos::write_stellarcollapse(path_out, read_back, path_in);

  const std::vector<unsigned char> blob_out = read_raw_dataset_bytes(path_out, "FAKE-provenance.in");
  REQUIRE(blob_out.size() == blob.size());
  CHECK(blob_out == blob);
}

// --- (3) append_repair_group -------------------------------------------------

TEST_CASE("append_repair_group: writes group contents and attributes correctly") {
  SyntheticOptions opts;
  opts.nrho = 4;
  opts.ntemp = 4;
  opts.nye = 2;
  RawTable t = eeos::make_synthetic_table(opts);
  const std::string path = scratch_path("repair_group.h5");
  eeos::write_stellarcollapse(path, t);

  RepairResult result;
  result.status = eeos::Status::repaired;
  result.entries = {
      RepairEntry{"entropy", 1, 2, 0, 3.5, 4.5},
      RepairEntry{"logenergy", 0, 3, 1, 18.0, 18.25},
  };

  RepairOptions options;
  options.min_slope_entropy = 1e-6;
  options.min_slope_logenergy = 1e-9;

  const unsigned long long fake_fnv1a = 0xDEADBEEFCAFEULL;
  eeos::append_repair_group(path, result, options, "some_input.h5", fake_fnv1a,
                             "eos_repair-test-1.2.3");

  H5Guard file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  REQUIRE(file.valid());
  H5Guard group(H5Gopen(file.get(), "repair", H5P_DEFAULT), H5Gclose);
  REQUIRE(group.valid());

  const std::vector<unsigned> irho = read_1d_uint_dataset(group.get(), "irho");
  const std::vector<unsigned> jT = read_1d_uint_dataset(group.get(), "jT");
  const std::vector<unsigned> kYe = read_1d_uint_dataset(group.get(), "kYe");
  const std::vector<double> old_value = read_1d_double_dataset(group.get(), "old_value");
  const std::vector<double> new_value = read_1d_double_dataset(group.get(), "new_value");
  const std::vector<std::string> field = read_fixed_strings_dataset(group.get(), "field");

  REQUIRE(irho.size() == 2);
  REQUIRE(jT.size() == 2);
  REQUIRE(kYe.size() == 2);
  REQUIRE(old_value.size() == 2);
  REQUIRE(new_value.size() == 2);
  REQUIRE(field.size() == 2);

  CHECK(irho[0] == 1);
  CHECK(jT[0] == 2);
  CHECK(kYe[0] == 0);
  CHECK(old_value[0] == 3.5);
  CHECK(new_value[0] == 4.5);
  CHECK(field[0] == "entropy");

  CHECK(irho[1] == 0);
  CHECK(jT[1] == 3);
  CHECK(kYe[1] == 1);
  CHECK(old_value[1] == 18.0);
  CHECK(new_value[1] == 18.25);
  CHECK(field[1] == "logenergy");

  CHECK(read_double_attr(group.get(), "min_slope_entropy") == 1e-6);
  CHECK(read_double_attr(group.get(), "min_slope_logenergy") == 1e-9);
  CHECK(read_string_attr(group.get(), "tool_version") == "eos_repair-test-1.2.3");
  CHECK(read_string_attr(group.get(), "input_path") == "some_input.h5");
  CHECK(read_uint64_attr(group.get(), "input_fnv1a") == fake_fnv1a);
  CHECK(read_uint64_attr(group.get(), "n_modified") == 2);
}

TEST_CASE("append_repair_group: empty result writes n_modified=0 and no entry datasets") {
  SyntheticOptions opts;
  opts.nrho = 3;
  opts.ntemp = 3;
  opts.nye = 2;
  RawTable t = eeos::make_synthetic_table(opts);
  const std::string path = scratch_path("repair_group_empty.h5");
  eeos::write_stellarcollapse(path, t);

  RepairResult empty_result; // default-constructed: entries empty, status ok
  RepairOptions options;
  eeos::append_repair_group(path, empty_result, options, "clean_input.h5", 0ULL,
                             "eos_repair-test-0.0");

  H5Guard file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  REQUIRE(file.valid());
  H5Guard group(H5Gopen(file.get(), "repair", H5P_DEFAULT), H5Gclose);
  REQUIRE(group.valid());

  CHECK(H5Lexists(group.get(), "irho", H5P_DEFAULT) <= 0);
  CHECK(H5Lexists(group.get(), "field", H5P_DEFAULT) <= 0);
  CHECK(read_uint64_attr(group.get(), "n_modified") == 0);
  CHECK(read_string_attr(group.get(), "input_path") == "clean_input.h5");
}

TEST_CASE("append_repair_group: a second call on the same file throws") {
  SyntheticOptions opts;
  opts.nrho = 3;
  opts.ntemp = 3;
  opts.nye = 2;
  RawTable t = eeos::make_synthetic_table(opts);
  const std::string path = scratch_path("repair_group_twice.h5");
  eeos::write_stellarcollapse(path, t);

  RepairResult empty_result;
  RepairOptions options;
  eeos::append_repair_group(path, empty_result, options, "in.h5", 1ULL, "v1");
  CHECK_THROWS_AS(eeos::append_repair_group(path, empty_result, options, "in.h5", 1ULL, "v1"),
                  std::runtime_error);
}

// --- (4) fnv1a_file: known-answer tests -------------------------------------

TEST_CASE("fnv1a_file: known-answer tests against published FNV-1a-64 test vectors") {
  // 64-bit FNV-1a: hash = 14695981039346656037ULL (offset basis); for each
  // input byte b: hash = (hash XOR b) * 1099511628211ULL (prime), mod 2^64.
  //
  // By hand for the 1-byte input "a" (0x61): offset_basis in hex is
  // 0xcbf29ce484222325; XOR-ing in the low byte 0x61 gives
  // 0xcbf29ce484222325 XOR 0x61 = 0xcbf29ce484222344 (0x25 XOR 0x61 = 0x44,
  // no other byte changes), which is then multiplied by the prime mod 2^64.
  // That product is exactly the published FNV-1a-64 test vector for "a":
  // 0xaf63dc4c8601ec8c = 12638187200555641996ULL. "123456789" (9 bytes) is
  // likewise a published FNV-1a-64 test vector:
  // 0x06d5573923c6cdfc = 492395637191921148ULL. The empty input leaves the
  // hash at the offset basis (no bytes to fold in).
  {
    const std::string path = scratch_path("fnv1a_a.bin");
    std::ofstream out(path, std::ios::binary);
    REQUIRE(static_cast<bool>(out));
    out << "a";
    out.close();
    CHECK(eeos::fnv1a_file(path) == 12638187200555641996ULL);
  }
  {
    const std::string path = scratch_path("fnv1a_123456789.bin");
    std::ofstream out(path, std::ios::binary);
    REQUIRE(static_cast<bool>(out));
    out << "123456789";
    out.close();
    CHECK(eeos::fnv1a_file(path) == 492395637191921148ULL);
  }
  {
    const std::string path = scratch_path("fnv1a_empty.bin");
    std::ofstream out(path, std::ios::binary);
    REQUIRE(static_cast<bool>(out));
    out.close();
    CHECK(eeos::fnv1a_file(path) == 14695981039346656037ULL);
  }
}

TEST_CASE("fnv1a_file: nonexistent file throws std::runtime_error") {
  CHECK_THROWS_AS(eeos::fnv1a_file(scratch_path("does_not_exist_fnv1a.bin")), std::runtime_error);
}

// --- (5) real-table tests (skipped cleanly if tables/ is absent) -----------

TEST_CASE("read_stellarcollapse: LS220 real table") {
  if (!table_exists(kLS220Path)) {
    WARN_MESSAGE(false, "LS220 table not found at '" << kLS220Path << "' -- skipped ('skipped')");
    return;
  }

  RawTable t = eeos::read_stellarcollapse(kLS220Path);

  CHECK(t.nrho() == 234);
  CHECK(t.ntemp() == 136);
  CHECK(t.nye() == 50);

  for (size_t i = 1; i < t.nrho(); ++i) {
    CHECK(t.logrho()[i] > t.logrho()[i - 1]);
  }
  for (size_t j = 1; j < t.ntemp(); ++j) {
    CHECK(t.logtemp()[j] > t.logtemp()[j - 1]);
  }
  for (size_t k = 1; k < t.nye(); ++k) {
    CHECK(t.ye()[k] > t.ye()[k - 1]);
  }

  CHECK(t.has_field("logenergy"));
  CHECK(t.has_field("entropy"));
  CHECK(t.has_field("logpress"));
  CHECK(t.has_field("cs2"));

  REQUIRE(t.has_attribute("energy_shift"));
  CHECK(t.energy_shift() > 0.0);
}

TEST_CASE("write_stellarcollapse (with source): LS220 write-back round trip") {
  if (!table_exists(kLS220Path)) {
    WARN_MESSAGE(false, "LS220 table not found at '" << kLS220Path << "' -- skipped ('skipped')");
    return;
  }

  RawTable a = eeos::read_stellarcollapse(kLS220Path);

  const std::string out = scratch_path("ls220_roundtrip.h5");
  eeos::write_stellarcollapse(out, a, kLS220Path);

  RawTable b = eeos::read_stellarcollapse(out);

  REQUIRE(a.nrho() == b.nrho());
  REQUIRE(a.ntemp() == b.ntemp());
  REQUIRE(a.nye() == b.nye());
  CHECK(same_bits_vec(a.logrho(), b.logrho()));
  CHECK(same_bits_vec(a.logtemp(), b.logtemp()));
  CHECK(same_bits_vec(a.ye(), b.ye()));

  REQUIRE(a.field_names().size() == b.field_names().size());
  for (const std::string &name : a.field_names()) {
    REQUIRE(b.has_field(name));
    CHECK(same_bits_vec(a.field(name), b.field(name)));
  }
  REQUIRE(b.has_attribute("energy_shift"));
  CHECK(a.energy_shift() == b.energy_shift());

  // h5ls-level dataset count of B equals A (H5Literate count, not the h5ls
  // binary).
  CHECK(count_root_datasets(kLS220Path) == count_root_datasets(out));
}

TEST_CASE("read_stellarcollapse: SRO real table") {
  if (!table_exists(kSROPath)) {
    WARN_MESSAGE(false, "SRO table not found at '" << kSROPath << "' -- skipped ('skipped')");
    return;
  }

  RawTable t = eeos::read_stellarcollapse(kSROPath);

  CHECK(t.nrho() == 391);
  CHECK(t.ntemp() == 163);
  CHECK(t.nye() == 66);
  CHECK(t.has_attribute("have_rel_cs2"));
}

TEST_CASE("write_stellarcollapse (with source): SRO opaque blob survives byte-identical") {
  if (!table_exists(kSROPath)) {
    WARN_MESSAGE(false, "SRO table not found at '" << kSROPath << "' -- skipped ('skipped')");
    return;
  }

  RawTable a = eeos::read_stellarcollapse(kSROPath);

  const std::string out = scratch_path("sro_roundtrip.h5");
  eeos::write_stellarcollapse(out, a, kSROPath);

  RawTable b = eeos::read_stellarcollapse(out);
  REQUIRE(a.field_names().size() == b.field_names().size());
  for (const std::string &name : a.field_names()) {
    REQUIRE(b.has_field(name));
    CHECK(same_bits_vec(a.field(name), b.field(name)));
  }

  const std::vector<unsigned char> blob_a = read_raw_dataset_bytes(kSROPath, "NSE-partition.in");
  const std::vector<unsigned char> blob_b = read_raw_dataset_bytes(out, "NSE-partition.in");
  REQUIRE(blob_a.size() == blob_b.size());
  CHECK(blob_a == blob_b);
}

// --- (6) error paths ---------------------------------------------------------

TEST_CASE("read_stellarcollapse: nonexistent file throws std::runtime_error") {
  CHECK_THROWS_AS(eeos::read_stellarcollapse(scratch_path("does_not_exist_12345.h5")),
                  std::runtime_error);
}

TEST_CASE("read_stellarcollapse: file missing 'logrho' throws std::runtime_error") {
  const std::string path = scratch_path("broken_missing_logrho.h5");
  {
    H5Guard file(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
    REQUIRE(file.valid());

    const hsize_t dim = 2;
    H5Guard space(H5Screate_simple(1, &dim, nullptr), H5Sclose);
    REQUIRE(space.valid());
    const double vals[2] = {0.0, 1.0};

    H5Guard d1(H5Dcreate(file.get(), "logtemp", H5T_NATIVE_DOUBLE, space.get(), H5P_DEFAULT,
                          H5P_DEFAULT, H5P_DEFAULT),
               H5Dclose);
    REQUIRE(d1.valid());
    REQUIRE(H5Dwrite(d1.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals) >= 0);

    H5Guard d2(H5Dcreate(file.get(), "ye", H5T_NATIVE_DOUBLE, space.get(), H5P_DEFAULT, H5P_DEFAULT,
                          H5P_DEFAULT),
               H5Dclose);
    REQUIRE(d2.valid());
    REQUIRE(H5Dwrite(d2.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals) >= 0);
    // Deliberately no "logrho" dataset.
  }

  CHECK_THROWS_AS(eeos::read_stellarcollapse(path), std::runtime_error);
}

TEST_CASE("write_stellarcollapse (source-less): table without 'energy_shift' throws") {
  RawTable t;
  t.set_axes({0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0});
  t.add_field("entropy", std::vector<double>(8, 1.0));
  CHECK_THROWS_AS(eeos::write_stellarcollapse(scratch_path("no_energy_shift.h5"), t),
                  std::runtime_error);
}

TEST_CASE("write_stellarcollapse (with source): nonexistent source file throws") {
  RawTable t;
  t.set_axes({0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0});
  CHECK_THROWS_AS(eeos::write_stellarcollapse(scratch_path("out_from_missing_src.h5"), t,
                                               scratch_path("does_not_exist_src.h5")),
                  std::runtime_error);
}
