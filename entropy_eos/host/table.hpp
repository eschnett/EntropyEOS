// entropy_eos/host/table.hpp
//
// RawTable: an EOS table's content, stored verbatim (see CODE.md "Data
// model"). Host-only: STL containers throughout, and it may throw.

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace eeos {

// RawTable holds a table's axes, named 3D fields, and named scalar
// attributes exactly as found in the source file. It performs no unit
// conversion and no physics interpretation beyond the handful of named
// convenience accessors documented below (rho/temp/yev, energy_shift());
// everything else is looked up by name. Insertion order of fields and
// attributes is preserved so a later write-back (the repair tool) can
// reproduce the source file's dataset order faithfully.
class RawTable {
public:
  RawTable() = default;

  // Sets the three table axes. Sizes fix nrho()/ntemp()/nye() for every
  // field added afterwards. Does not validate; call validate_axes()
  // explicitly once axes and fields are in place.
  void set_axes(std::vector<double> logrho_in, std::vector<double> logtemp_in,
                std::vector<double> ye_in);

  // Axis sizes.
  size_t nrho() const { return logrho_.size(); }
  size_t ntemp() const { return logtemp_.size(); }
  size_t nye() const { return ye_.size(); }

  // Axes, verbatim as stored: log10(rho [g/cm^3]), log10(T [MeV]), Ye
  // (linear).
  const std::vector<double> &logrho() const { return logrho_; }
  const std::vector<double> &logtemp() const { return logtemp_; }
  const std::vector<double> &ye() const { return ye_; }

  // Physical convenience accessors, converting on demand from the stored
  // (verbatim) axes.
  double rho(size_t i) const;  // 10^logrho[i], g/cm^3
  double temp(size_t j) const; // 10^logtemp[j], MeV
  double yev(size_t k) const;  // ye[k] (already linear)

  // Throws std::runtime_error if any axis contains a non-finite value or is
  // not strictly increasing. Axes only; field-level checks (finiteness,
  // physical bounds, monotonicity in T of stored fields) belong to a later
  // module (host/check).
  void validate_axes() const;

  // Flat index into a field's storage: dims [nYe][nT][nRho], iRho fastest,
  // matching the stellarcollapse.org layout (see CODE.md "Data model").
  size_t index(size_t irho, size_t jT, size_t kYe) const {
    return irho + nrho() * (jT + ntemp() * kYe);
  }

  // Named 3D fields (dims [nYe][nT][nRho], iRho fastest; size must equal
  // nrho()*ntemp()*nye()). Throws std::invalid_argument if `data` has the
  // wrong size. Re-adding an existing name overwrites its data in place,
  // preserving its original position in field_names().
  void add_field(const std::string &name, std::vector<double> data);

  bool has_field(const std::string &name) const;

  // Throws std::out_of_range if `name` is not present.
  const std::vector<double> &field(const std::string &name) const;
  std::vector<double> &field(const std::string &name);

  // Field names in insertion order.
  const std::vector<std::string> &field_names() const { return field_names_; }

  // Named scalar attributes (e.g. "energy_shift"), insertion order
  // preserved. Re-adding an existing name overwrites its value.
  void add_attribute(const std::string &name, double value);
  bool has_attribute(const std::string &name) const;

  // Throws std::out_of_range if `name` is not present.
  double attribute(const std::string &name) const;

  // Attribute names in insertion order.
  const std::vector<std::string> &attribute_names() const { return attribute_names_; }

  // Convenience for the "energy_shift" attribute. Throws std::out_of_range
  // if absent.
  double energy_shift() const;

private:
  std::vector<double> logrho_, logtemp_, ye_;

  std::vector<std::string> field_names_;
  std::vector<std::vector<double>> field_data_;
  std::unordered_map<std::string, size_t> field_index_;

  std::vector<std::string> attribute_names_;
  std::vector<double> attribute_values_;
  std::unordered_map<std::string, size_t> attribute_index_;
};

} // namespace eeos
