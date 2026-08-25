#include "entropy_eos/host/io_stellarcollapse.hpp"

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace eeos {

namespace {

// ---------------------------------------------------------------------
// RAII handle for an hid_t: a release-function pointer plus the id, closed
// exactly once (in the destructor, or early via reset()) so no error path
// -- an exception thrown after an H5*open()/H5*create() succeeds but before
// the enclosing function returns normally -- can leak an HDF5 resource. One
// generic class covers every handle kind used below (file/dataset/group/
// dataspace/datatype/attribute/property-list): they all close via a plain
// `herr_t (*)(hid_t)` function (H5Fclose, H5Dclose, H5Gclose, H5Sclose,
// H5Tclose, H5Aclose, H5Oclose, H5Pclose).
// ---------------------------------------------------------------------
class Handle {
public:
  Handle() = default;
  Handle(hid_t id, herr_t (*closer)(hid_t)) : id_(id), closer_(closer) {}
  ~Handle() { reset(); }

  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;

  Handle(Handle &&other) noexcept : id_(other.id_), closer_(other.closer_) {
    other.id_ = -1;
    other.closer_ = nullptr;
  }
  Handle &operator=(Handle &&other) noexcept {
    if (this != &other) {
      reset();
      id_ = other.id_;
      closer_ = other.closer_;
      other.id_ = -1;
      other.closer_ = nullptr;
    }
    return *this;
  }

  hid_t get() const { return id_; }
  bool valid() const { return id_ >= 0; }

  void reset() {
    if (id_ >= 0 && closer_ != nullptr) {
      closer_(id_);
    }
    id_ = -1;
  }

private:
  hid_t id_ = -1;
  herr_t (*closer_)(hid_t) = nullptr;
};

// RAII guard: suppresses HDF5's default error-printing-to-stderr handler for
// its lifetime, restoring whatever handler was previously installed on
// destruction (even when unwinding through an exception). Every public
// function below wraps its whole body in one of these, because every one of
// them contains at least one probe that fails in ordinary, non-exceptional
// operation on real files -- opening a file that may not exist, or (see
// list_root_names_in_file_order() below) iterating a group's
// creation-order index when the file has none, which is the normal case for
// real stellarcollapse.org tables. Without this, a clean test run still
// spews HDF5's own diagnostic dump to stderr; the std::runtime_error message
// this module throws instead is meant to carry the diagnostic.
class SilenceHDF5Errors {
public:
  SilenceHDF5Errors() {
    H5Eget_auto2(H5E_DEFAULT, &func_, &data_);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  }
  ~SilenceHDF5Errors() { H5Eset_auto2(H5E_DEFAULT, func_, data_); }

  SilenceHDF5Errors(const SilenceHDF5Errors &) = delete;
  SilenceHDF5Errors &operator=(const SilenceHDF5Errors &) = delete;

private:
  H5E_auto2_t func_ = nullptr;
  void *data_ = nullptr;
};

// Lists the names of every member of `loc`'s root group, in "file order".
// Real stellarcollapse.org files carry no link-creation-order index (only a
// name index), so h5ls -- and this function -- lists them alphabetically;
// files this module writes (see create_output_file() below) always enable
// creation-order tracking, so a round trip through write_stellarcollapse()
// reproduces the exact field order the RawTable was built in, not an
// alphabetized version of it. H5_INDEX_CRT_ORDER is tried first and its
// (expected, on real files) failure is silenced; H5_INDEX_NAME is the
// fallback and the only other deterministic order HDF5 exposes.
std::vector<std::string> list_root_names_in_file_order(hid_t loc, const std::string &path) {
  std::vector<std::string> names;
  auto collect = [](hid_t, const char *name, const H5L_info_t *, void *op_data) -> herr_t {
    static_cast<std::vector<std::string> *>(op_data)->push_back(name);
    return 0;
  };

  herr_t status;
  {
    SilenceHDF5Errors silence;
    status = H5Literate(loc, H5_INDEX_CRT_ORDER, H5_ITER_INC, nullptr, collect, &names);
  }
  if (status < 0) {
    names.clear();
    status = H5Literate(loc, H5_INDEX_NAME, H5_ITER_INC, nullptr, collect, &names);
  }
  if (status < 0) {
    throw std::runtime_error("read_stellarcollapse: '" + path + "': failed to enumerate datasets");
  }
  return names;
}

// Reads a required 1-D double dataset (an axis). Throws if missing, not
// 1-D, or unreadable.
std::vector<double> read_1d_double_dataset(hid_t loc, const char *name, const std::string &path) {
  if (H5Lexists(loc, name, H5P_DEFAULT) <= 0) {
    throw std::runtime_error("read_stellarcollapse: '" + path + "' has no dataset '" + name + "'");
  }
  Handle dset(H5Dopen(loc, name, H5P_DEFAULT), H5Dclose);
  if (!dset.valid()) {
    throw std::runtime_error("read_stellarcollapse: '" + path + "': failed to open dataset '" +
                              std::string(name) + "'");
  }
  Handle space(H5Dget_space(dset.get()), H5Sclose);
  if (!space.valid() || H5Sget_simple_extent_ndims(space.get()) != 1) {
    throw std::runtime_error("read_stellarcollapse: '" + path + "': dataset '" + std::string(name) +
                              "' is not 1-D");
  }
  hsize_t dim = 0;
  H5Sget_simple_extent_dims(space.get(), &dim, nullptr);
  std::vector<double> data(static_cast<size_t>(dim));
  if (H5Dread(dset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
    throw std::runtime_error("read_stellarcollapse: '" + path + "': failed to read dataset '" +
                              std::string(name) + "'");
  }
  return data;
}

// True iff `dset`'s dataspace holds exactly one element (a stellarcollapse
// scalar {1} dataset, whatever its rank).
bool dataset_is_scalar(hid_t dset) {
  Handle space(H5Dget_space(dset), H5Sclose);
  if (!space.valid()) {
    return false;
  }
  return H5Sget_simple_extent_npoints(space.get()) == 1;
}

// Reads a scalar dataset's single value as a double, regardless of its
// stored type (HDF5 converts int<->float transparently).
double read_scalar_as_double(hid_t dset, const std::string &path, const std::string &name) {
  double value = 0.0;
  if (H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value) < 0) {
    throw std::runtime_error("read_stellarcollapse: '" + path + "': failed to read scalar '" + name +
                              "'");
  }
  return value;
}

// Cross-checks an optional "points*" scalar dataset against the
// corresponding axis size already read; a mismatch throws. Absent is not an
// error (only checked "when present", per the API contract).
void check_points_dataset(hid_t loc, const char *name, size_t expected, const std::string &path) {
  if (H5Lexists(loc, name, H5P_DEFAULT) <= 0) {
    return;
  }
  Handle dset(H5Dopen(loc, name, H5P_DEFAULT), H5Dclose);
  if (!dset.valid()) {
    throw std::runtime_error("read_stellarcollapse: '" + path + "': failed to open '" +
                              std::string(name) + "'");
  }
  const double value = read_scalar_as_double(dset.get(), path, name);
  const long long rounded = std::llround(value);
  if (rounded < 0 || static_cast<size_t>(rounded) != expected) {
    throw std::runtime_error("read_stellarcollapse: '" + path + "': '" + name + "' = " +
                              std::to_string(rounded) + " does not match axis size " +
                              std::to_string(expected));
  }
}

// True iff `dset` is 3-D with dims exactly (nye, ntemp, nrho). That is C
// order with iRho fastest -- exactly RawTable::index(irho, jT, kYe) =
// irho + nrho*(jT + ntemp*kYe) (table.hpp) -- so a straight, contiguous
// H5Dread into an nrho*ntemp*nye buffer lands every element at the index
// RawTable expects with no reordering.
bool dataset_shape_matches_field(hid_t dset, hsize_t nye, hsize_t ntemp, hsize_t nrho) {
  Handle space(H5Dget_space(dset), H5Sclose);
  if (!space.valid() || H5Sget_simple_extent_ndims(space.get()) != 3) {
    return false;
  }
  hsize_t dims[3] = {0, 0, 0};
  H5Sget_simple_extent_dims(space.get(), dims, nullptr);
  return dims[0] == nye && dims[1] == ntemp && dims[2] == nrho;
}

// True iff `dset`'s native type is a float type (f4 or f8). Both are read
// identically below: H5Dread with a H5T_NATIVE_DOUBLE memory type is a
// straight copy for an f8 source and a lossless upconversion (HDF5's own
// numeric type conversion) for an f4 source, so there is no separate code
// path for "single" vs "double" fields.
bool dataset_is_float_class(hid_t dset) {
  Handle dtype(H5Dget_type(dset), H5Tclose);
  if (!dtype.valid()) {
    return false;
  }
  return H5Tget_class(dtype.get()) == H5T_FLOAT;
}

// Creates `path_out` (overwriting it) with link-creation-order tracking
// enabled on its root group, so that a table written through this module
// and later re-read through list_root_names_in_file_order() reproduces the
// exact field/axis order it was written in -- see that function's comment.
Handle create_output_file(const std::string &path_out) {
  Handle fcpl(H5Pcreate(H5P_FILE_CREATE), H5Pclose);
  if (!fcpl.valid() ||
      H5Pset_link_creation_order(fcpl.get(), H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED) < 0) {
    throw std::runtime_error("write_stellarcollapse: failed to configure creation-order tracking for '" +
                              path_out + "'");
  }
  Handle file(H5Fcreate(path_out.c_str(), H5F_ACC_TRUNC, fcpl.get(), H5P_DEFAULT), H5Fclose);
  if (!file.valid()) {
    throw std::runtime_error("write_stellarcollapse: cannot create file '" + path_out + "'");
  }
  return file;
}

void write_1d_double(hid_t loc, const std::string &name, const std::vector<double> &data) {
  const hsize_t dim = static_cast<hsize_t>(data.size());
  Handle space(H5Screate_simple(1, &dim, nullptr), H5Sclose);
  if (!space.valid()) {
    throw std::runtime_error("write_stellarcollapse: failed to create dataspace for '" + name + "'");
  }
  Handle dset(
      H5Dcreate(loc, name.c_str(), H5T_NATIVE_DOUBLE, space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      H5Dclose);
  if (!dset.valid()) {
    throw std::runtime_error("write_stellarcollapse: failed to create dataset '" + name + "'");
  }
  if (H5Dwrite(dset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
    throw std::runtime_error("write_stellarcollapse: failed to write dataset '" + name + "'");
  }
}

void write_3d_double(hid_t loc, const std::string &name, const std::vector<double> &data, hsize_t nye,
                      hsize_t ntemp, hsize_t nrho) {
  const hsize_t dims[3] = {nye, ntemp, nrho};
  Handle space(H5Screate_simple(3, dims, nullptr), H5Sclose);
  if (!space.valid()) {
    throw std::runtime_error("write_stellarcollapse: failed to create dataspace for '" + name + "'");
  }
  Handle dset(
      H5Dcreate(loc, name.c_str(), H5T_NATIVE_DOUBLE, space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      H5Dclose);
  if (!dset.valid()) {
    throw std::runtime_error("write_stellarcollapse: failed to create dataset '" + name + "'");
  }
  if (H5Dwrite(dset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
    throw std::runtime_error("write_stellarcollapse: failed to write dataset '" + name + "'");
  }
}

void write_scalar_int(hid_t loc, const std::string &name, int value) {
  const hsize_t dim = 1;
  Handle space(H5Screate_simple(1, &dim, nullptr), H5Sclose);
  if (!space.valid()) {
    throw std::runtime_error("write_stellarcollapse: failed to create dataspace for '" + name + "'");
  }
  Handle dset(
      H5Dcreate(loc, name.c_str(), H5T_NATIVE_INT, space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      H5Dclose);
  if (!dset.valid()) {
    throw std::runtime_error("write_stellarcollapse: failed to create dataset '" + name + "'");
  }
  if (H5Dwrite(dset.get(), H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value) < 0) {
    throw std::runtime_error("write_stellarcollapse: failed to write dataset '" + name + "'");
  }
}

void write_scalar_double(hid_t loc, const std::string &name, double value) {
  const hsize_t dim = 1;
  Handle space(H5Screate_simple(1, &dim, nullptr), H5Sclose);
  if (!space.valid()) {
    throw std::runtime_error("write_stellarcollapse: failed to create dataspace for '" + name + "'");
  }
  Handle dset(
      H5Dcreate(loc, name.c_str(), H5T_NATIVE_DOUBLE, space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      H5Dclose);
  if (!dset.valid()) {
    throw std::runtime_error("write_stellarcollapse: failed to create dataset '" + name + "'");
  }
  if (H5Dwrite(dset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value) < 0) {
    throw std::runtime_error("write_stellarcollapse: failed to write dataset '" + name + "'");
  }
}

void write_1d_uint(hid_t loc, const std::string &name, const std::vector<unsigned> &data) {
  const hsize_t dim = static_cast<hsize_t>(data.size());
  Handle space(H5Screate_simple(1, &dim, nullptr), H5Sclose);
  if (!space.valid()) {
    throw std::runtime_error("append_repair_group: failed to create dataspace for '" + name + "'");
  }
  Handle dset(
      H5Dcreate(loc, name.c_str(), H5T_NATIVE_UINT, space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      H5Dclose);
  if (!dset.valid()) {
    throw std::runtime_error("append_repair_group: failed to create dataset '" + name + "'");
  }
  if (H5Dwrite(dset.get(), H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
    throw std::runtime_error("append_repair_group: failed to write dataset '" + name + "'");
  }
}

// Fixed-size (NUL-padded/terminated) C-string datatype of `len` bytes,
// including the terminator -- used for both the "/repair/field" dataset and
// the "tool_version"/"input_path" attributes (see append_repair_group()).
// Chosen over HDF5 variable-length strings to avoid H5Dvlen_reclaim-style
// memory management for what are always short, known-in-advance strings.
// Returns an invalid (negative) handle on failure; caller checks .valid().
Handle create_fixed_string_type(size_t len) {
  Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
  if (!type.valid()) {
    return type;
  }
  if (H5Tset_size(type.get(), len) < 0 || H5Tset_strpad(type.get(), H5T_STR_NULLTERM) < 0) {
    return Handle();
  }
  return type;
}

constexpr size_t kFieldNameStringLen = 32; // ample for "entropy"/"logenergy" plus headroom

void write_fixed_strings_dataset(hid_t loc, const std::string &dset_name,
                                  const std::vector<std::string> &values, size_t fixed_len) {
  Handle dtype = create_fixed_string_type(fixed_len);
  if (!dtype.valid()) {
    throw std::runtime_error("append_repair_group: failed to create string type for '" + dset_name +
                              "'");
  }
  std::vector<char> buf(values.size() * fixed_len, '\0');
  for (size_t i = 0; i < values.size(); ++i) {
    const std::string &v = values[i];
    const size_t n = std::min(v.size(), fixed_len - 1); // leave room for the NUL terminator
    std::memcpy(buf.data() + i * fixed_len, v.data(), n);
  }
  const hsize_t dim = static_cast<hsize_t>(values.size());
  Handle space(H5Screate_simple(1, &dim, nullptr), H5Sclose);
  if (!space.valid()) {
    throw std::runtime_error("append_repair_group: failed to create dataspace for '" + dset_name + "'");
  }
  Handle dset(H5Dcreate(loc, dset_name.c_str(), dtype.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT,
                         H5P_DEFAULT),
              H5Dclose);
  if (!dset.valid()) {
    throw std::runtime_error("append_repair_group: failed to create dataset '" + dset_name + "'");
  }
  if (H5Dwrite(dset.get(), dtype.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data()) < 0) {
    throw std::runtime_error("append_repair_group: failed to write dataset '" + dset_name + "'");
  }
}

void write_attr_double(hid_t loc, const std::string &name, double value) {
  Handle space(H5Screate(H5S_SCALAR), H5Sclose);
  if (!space.valid()) {
    throw std::runtime_error("append_repair_group: failed to create dataspace for attribute '" + name +
                              "'");
  }
  Handle attr(H5Acreate(loc, name.c_str(), H5T_NATIVE_DOUBLE, space.get(), H5P_DEFAULT, H5P_DEFAULT),
              H5Aclose);
  if (!attr.valid()) {
    throw std::runtime_error("append_repair_group: failed to create attribute '" + name + "'");
  }
  if (H5Awrite(attr.get(), H5T_NATIVE_DOUBLE, &value) < 0) {
    throw std::runtime_error("append_repair_group: failed to write attribute '" + name + "'");
  }
}

void write_attr_uint64(hid_t loc, const std::string &name, std::uint64_t value) {
  Handle space(H5Screate(H5S_SCALAR), H5Sclose);
  if (!space.valid()) {
    throw std::runtime_error("append_repair_group: failed to create dataspace for attribute '" + name +
                              "'");
  }
  Handle attr(H5Acreate(loc, name.c_str(), H5T_NATIVE_UINT64, space.get(), H5P_DEFAULT, H5P_DEFAULT),
              H5Aclose);
  if (!attr.valid()) {
    throw std::runtime_error("append_repair_group: failed to create attribute '" + name + "'");
  }
  if (H5Awrite(attr.get(), H5T_NATIVE_UINT64, &value) < 0) {
    throw std::runtime_error("append_repair_group: failed to write attribute '" + name + "'");
  }
}

void write_attr_string(hid_t loc, const std::string &name, const std::string &value) {
  Handle dtype = create_fixed_string_type(value.size() + 1);
  if (!dtype.valid()) {
    throw std::runtime_error("append_repair_group: failed to create string type for attribute '" + name +
                              "'");
  }
  Handle space(H5Screate(H5S_SCALAR), H5Sclose);
  if (!space.valid()) {
    throw std::runtime_error("append_repair_group: failed to create dataspace for attribute '" + name +
                              "'");
  }
  Handle attr(H5Acreate(loc, name.c_str(), dtype.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
  if (!attr.valid()) {
    throw std::runtime_error("append_repair_group: failed to create attribute '" + name + "'");
  }
  if (H5Awrite(attr.get(), dtype.get(), value.c_str()) < 0) {
    throw std::runtime_error("append_repair_group: failed to write attribute '" + name + "'");
  }
}

} // namespace

RawTable read_stellarcollapse(const std::string &path) {
  SilenceHDF5Errors silence;

  Handle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!file.valid()) {
    throw std::runtime_error("read_stellarcollapse: cannot open file '" + path + "'");
  }

  std::vector<double> logrho = read_1d_double_dataset(file.get(), "logrho", path);
  std::vector<double> logtemp = read_1d_double_dataset(file.get(), "logtemp", path);
  std::vector<double> ye = read_1d_double_dataset(file.get(), "ye", path);

  const size_t nrho = logrho.size();
  const size_t ntemp = logtemp.size();
  const size_t nye = ye.size();

  RawTable table;
  table.set_axes(std::move(logrho), std::move(logtemp), std::move(ye));

  check_points_dataset(file.get(), "pointsrho", nrho, path);
  check_points_dataset(file.get(), "pointstemp", ntemp, path);
  check_points_dataset(file.get(), "pointsye", nye, path);

  const hsize_t h_nye = static_cast<hsize_t>(nye);
  const hsize_t h_ntemp = static_cast<hsize_t>(ntemp);
  const hsize_t h_nrho = static_cast<hsize_t>(nrho);

  for (const std::string &name : list_root_names_in_file_order(file.get(), path)) {
    if (name == "logrho" || name == "logtemp" || name == "ye") {
      continue; // already consumed as an axis above
    }

    Handle obj(H5Oopen(file.get(), name.c_str(), H5P_DEFAULT), H5Oclose);
    if (!obj.valid() || H5Iget_type(obj.get()) != H5I_DATASET) {
      continue; // not a dataset (e.g. a "/repair" group) -- ignore
    }
    const hid_t dset = obj.get();

    if (name == "energy_shift" || name == "have_rel_cs2") {
      if (dataset_is_scalar(dset)) {
        table.add_attribute(name, read_scalar_as_double(dset, path, name));
      }
      continue;
    }

    if (name == "pointsrho" || name == "pointstemp" || name == "pointsye") {
      continue; // already cross-checked above; not stored in the RawTable
    }

    if (dataset_shape_matches_field(dset, h_nye, h_ntemp, h_nrho) && dataset_is_float_class(dset)) {
      std::vector<double> data(nrho * ntemp * nye);
      if (H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
        throw std::runtime_error("read_stellarcollapse: '" + path + "': failed to read field '" + name +
                                  "'");
      }
      table.add_field(name, std::move(data));
    }
    // Otherwise: an opaque provenance blob, or a dataset whose shape/type
    // don't match a field -- left out of the RawTable for write-time
    // passthrough (see write_stellarcollapse(path_out, table, path_in)).
  }

  return table;
}

void write_stellarcollapse(const std::string &path_out, const RawTable &table,
                            const std::string &path_in) {
  SilenceHDF5Errors silence;

  Handle src(H5Fopen(path_in.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!src.valid()) {
    throw std::runtime_error("write_stellarcollapse: cannot open source file '" + path_in + "'");
  }

  Handle dst = create_output_file(path_out);

  const hsize_t nye = static_cast<hsize_t>(table.nye());
  const hsize_t ntemp = static_cast<hsize_t>(table.ntemp());
  const hsize_t nrho = static_cast<hsize_t>(table.nrho());

  std::unordered_set<std::string> written;

  for (const std::string &name : list_root_names_in_file_order(src.get(), path_in)) {
    if (name == "logrho") {
      write_1d_double(dst.get(), name, table.logrho());
    } else if (name == "logtemp") {
      write_1d_double(dst.get(), name, table.logtemp());
    } else if (name == "ye") {
      write_1d_double(dst.get(), name, table.ye());
    } else if (table.has_field(name)) {
      write_3d_double(dst.get(), name, table.field(name), nye, ntemp, nrho);
    } else {
      // Passthrough: opaque provenance blobs, "points*", "energy_shift",
      // "have_rel_cs2", or anything else the RawTable doesn't carry --
      // copied byte-for-byte so it round-trips exactly. This is also why
      // mutating table.attribute("energy_shift") has no effect here: that
      // scalar dataset is never in `table.field_names()`/axes, so it always
      // takes this branch (see the header comment on write_stellarcollapse).
      if (H5Ocopy(src.get(), name.c_str(), dst.get(), name.c_str(), H5P_DEFAULT, H5P_DEFAULT) < 0) {
        throw std::runtime_error("write_stellarcollapse: failed to copy dataset '" + name + "' from '" +
                                  path_in + "' to '" + path_out + "'");
      }
      continue; // not a table field/axis; nothing to mark "written"
    }
    written.insert(name);
  }

  // Fields present in the table but absent from path_in are appended at the end.
  for (const std::string &field_name : table.field_names()) {
    if (written.find(field_name) == written.end()) {
      write_3d_double(dst.get(), field_name, table.field(field_name), nye, ntemp, nrho);
    }
  }
}

void write_stellarcollapse(const std::string &path_out, const RawTable &table) {
  SilenceHDF5Errors silence;

  if (!table.has_attribute("energy_shift")) {
    throw std::runtime_error("write_stellarcollapse: table has no 'energy_shift' attribute (required "
                              "to write '" +
                              path_out + "')");
  }

  Handle file = create_output_file(path_out);

  write_1d_double(file.get(), "logrho", table.logrho());
  write_1d_double(file.get(), "logtemp", table.logtemp());
  write_1d_double(file.get(), "ye", table.ye());

  write_scalar_int(file.get(), "pointsrho", static_cast<int>(table.nrho()));
  write_scalar_int(file.get(), "pointstemp", static_cast<int>(table.ntemp()));
  write_scalar_int(file.get(), "pointsye", static_cast<int>(table.nye()));

  for (const std::string &attr_name : table.attribute_names()) {
    write_scalar_double(file.get(), attr_name, table.attribute(attr_name));
  }

  const hsize_t nye = static_cast<hsize_t>(table.nye());
  const hsize_t ntemp = static_cast<hsize_t>(table.ntemp());
  const hsize_t nrho = static_cast<hsize_t>(table.nrho());
  for (const std::string &field_name : table.field_names()) {
    write_3d_double(file.get(), field_name, table.field(field_name), nye, ntemp, nrho);
  }
}

void append_repair_group(const std::string &path, const RepairResult &result,
                          const RepairOptions &options, const std::string &input_path,
                          unsigned long long input_fnv1a, const std::string &tool_version) {
  SilenceHDF5Errors silence;

  Handle file(H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose);
  if (!file.valid()) {
    throw std::runtime_error("append_repair_group: cannot open '" + path + "' for writing");
  }
  if (H5Lexists(file.get(), "repair", H5P_DEFAULT) > 0) {
    throw std::runtime_error("append_repair_group: '" + path + "' already has a '/repair' group");
  }

  Handle group(H5Gcreate(file.get(), "repair", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
  if (!group.valid()) {
    throw std::runtime_error("append_repair_group: failed to create '/repair' group in '" + path + "'");
  }

  const size_t n = result.entries.size();
  if (n > 0) {
    std::vector<unsigned> irho(n), jT(n), kYe(n);
    std::vector<double> old_value(n), new_value(n);
    std::vector<std::string> field(n);
    for (size_t i = 0; i < n; ++i) {
      const RepairEntry &e = result.entries[i];
      irho[i] = static_cast<unsigned>(e.irho);
      jT[i] = static_cast<unsigned>(e.jT);
      kYe[i] = static_cast<unsigned>(e.kYe);
      old_value[i] = e.old_value;
      new_value[i] = e.new_value;
      field[i] = e.field;
    }
    write_1d_uint(group.get(), "irho", irho);
    write_1d_uint(group.get(), "jT", jT);
    write_1d_uint(group.get(), "kYe", kYe);
    write_1d_double(group.get(), "old_value", old_value);
    write_1d_double(group.get(), "new_value", new_value);
    write_fixed_strings_dataset(group.get(), "field", field, kFieldNameStringLen);
  }

  write_attr_double(group.get(), "min_slope_entropy", options.min_slope_entropy);
  write_attr_double(group.get(), "min_slope_logenergy", options.min_slope_logenergy);
  write_attr_string(group.get(), "tool_version", tool_version);
  write_attr_string(group.get(), "input_path", input_path);
  write_attr_uint64(group.get(), "input_fnv1a", static_cast<std::uint64_t>(input_fnv1a));
  write_attr_uint64(group.get(), "n_modified", static_cast<std::uint64_t>(n));
}

unsigned long long fnv1a_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("fnv1a_file: cannot open '" + path + "'");
  }

  // 64-bit FNV-1a: hash = offset_basis; for each byte b: hash ^= b; hash *=
  // prime. Dependency-free, not cryptographic (see header comment).
  constexpr unsigned long long offset_basis = 14695981039346656037ULL;
  constexpr unsigned long long prime = 1099511628211ULL;
  unsigned long long hash = offset_basis;

  std::vector<char> buf(1 << 16);
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const std::streamsize count = in.gcount();
    for (std::streamsize i = 0; i < count; ++i) {
      hash ^= static_cast<unsigned char>(buf[static_cast<size_t>(i)]);
      hash *= prime;
    }
  }
  return hash;
}

} // namespace eeos
