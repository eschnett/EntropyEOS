#include "entropy_eos/host/table.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace eeos {

void RawTable::set_axes(std::vector<double> logrho_in, std::vector<double> logtemp_in,
                         std::vector<double> ye_in) {
  logrho_ = std::move(logrho_in);
  logtemp_ = std::move(logtemp_in);
  ye_ = std::move(ye_in);
}

double RawTable::rho(size_t i) const { return std::pow(10.0, logrho_.at(i)); }
double RawTable::temp(size_t j) const { return std::pow(10.0, logtemp_.at(j)); }
double RawTable::yev(size_t k) const { return ye_.at(k); }

namespace {

void validate_one_axis(const std::vector<double> &axis, const char *name) {
  for (size_t i = 0; i < axis.size(); ++i) {
    if (!std::isfinite(axis[i])) {
      throw std::runtime_error(std::string("RawTable::validate_axes: axis '") + name +
                                "' has a non-finite value at index " + std::to_string(i));
    }
    if (i > 0 && !(axis[i] > axis[i - 1])) {
      throw std::runtime_error(std::string("RawTable::validate_axes: axis '") + name +
                                "' is not strictly increasing at index " + std::to_string(i));
    }
  }
}

} // namespace

void RawTable::validate_axes() const {
  validate_one_axis(logrho_, "logrho");
  validate_one_axis(logtemp_, "logtemp");
  validate_one_axis(ye_, "ye");
}

void RawTable::add_field(const std::string &name, std::vector<double> data) {
  const size_t expected = nrho() * ntemp() * nye();
  if (data.size() != expected) {
    throw std::invalid_argument("RawTable::add_field: field '" + name + "' has size " +
                                 std::to_string(data.size()) + ", expected " +
                                 std::to_string(expected));
  }
  auto it = field_index_.find(name);
  if (it != field_index_.end()) {
    field_data_[it->second] = std::move(data);
    return;
  }
  field_index_.emplace(name, field_names_.size());
  field_names_.push_back(name);
  field_data_.push_back(std::move(data));
}

bool RawTable::has_field(const std::string &name) const {
  return field_index_.find(name) != field_index_.end();
}

const std::vector<double> &RawTable::field(const std::string &name) const {
  auto it = field_index_.find(name);
  if (it == field_index_.end()) {
    throw std::out_of_range("RawTable::field: no such field '" + name + "'");
  }
  return field_data_[it->second];
}

std::vector<double> &RawTable::field(const std::string &name) {
  auto it = field_index_.find(name);
  if (it == field_index_.end()) {
    throw std::out_of_range("RawTable::field: no such field '" + name + "'");
  }
  return field_data_[it->second];
}

void RawTable::add_attribute(const std::string &name, double value) {
  auto it = attribute_index_.find(name);
  if (it != attribute_index_.end()) {
    attribute_values_[it->second] = value;
    return;
  }
  attribute_index_.emplace(name, attribute_names_.size());
  attribute_names_.push_back(name);
  attribute_values_.push_back(value);
}

bool RawTable::has_attribute(const std::string &name) const {
  return attribute_index_.find(name) != attribute_index_.end();
}

double RawTable::attribute(const std::string &name) const {
  auto it = attribute_index_.find(name);
  if (it == attribute_index_.end()) {
    throw std::out_of_range("RawTable::attribute: no such attribute '" + name + "'");
  }
  return attribute_values_[it->second];
}

double RawTable::energy_shift() const { return attribute("energy_shift"); }

RawTable crop_table(const RawTable &in, size_t ia, size_t ib, size_t ja, size_t jb, size_t ka,
                    size_t kb, const std::vector<std::string> *fields) {
  if (ia > ib || ib >= in.nrho() || ja > jb || jb >= in.ntemp() || ka > kb || kb >= in.nye()) {
    throw std::out_of_range("crop_table: index box [" + std::to_string(ia) + "," +
                            std::to_string(ib) + "]x[" + std::to_string(ja) + "," +
                            std::to_string(jb) + "]x[" + std::to_string(ka) + "," +
                            std::to_string(kb) + "] does not fit the table's " +
                            std::to_string(in.nrho()) + "x" + std::to_string(in.ntemp()) + "x" +
                            std::to_string(in.nye()) + " grid");
  }

  RawTable out;
  std::vector<double> logrho(in.logrho().begin() + static_cast<long>(ia),
                             in.logrho().begin() + static_cast<long>(ib) + 1);
  std::vector<double> logtemp(in.logtemp().begin() + static_cast<long>(ja),
                              in.logtemp().begin() + static_cast<long>(jb) + 1);
  std::vector<double> ye(in.ye().begin() + static_cast<long>(ka),
                         in.ye().begin() + static_cast<long>(kb) + 1);
  out.set_axes(std::move(logrho), std::move(logtemp), std::move(ye));

  for (const std::string &name : in.field_names()) {
    if (fields != nullptr && std::find(fields->begin(), fields->end(), name) == fields->end()) {
      continue;
    }
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
  if (fields != nullptr) {
    for (const std::string &name : *fields) {
      (void)in.field(name); // throws std::out_of_range for a name `in` lacks
    }
  }

  for (const std::string &attr_name : in.attribute_names()) {
    out.add_attribute(attr_name, in.attribute(attr_name));
  }
  return out;
}

} // namespace eeos
