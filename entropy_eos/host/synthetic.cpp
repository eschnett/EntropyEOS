#include "entropy_eos/host/synthetic.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "entropy_eos/host/units.hpp"

namespace eeos {

double synthetic_eps(double rho_gcc, double temp_MeV, double ye, const SyntheticOptions &opts) {
  (void)rho_gcc;
  (void)opts;
  const double g = 1.0 + ye;
  const double kT_erg = temp_MeV * MeV_to_erg;
  return 1.5 * g * kT_erg / m_amu_g;
}

double synthetic_p(double rho_gcc, double temp_MeV, double ye, const SyntheticOptions &opts) {
  (void)opts;
  const double g = 1.0 + ye;
  const double kT_erg = temp_MeV * MeV_to_erg;
  return g * rho_gcc * kT_erg / m_amu_g;
}

double synthetic_s(double rho_gcc, double temp_MeV, double ye, const SyntheticOptions &opts) {
  const double g = 1.0 + ye;
  return g * (opts.s0 + 1.5 * std::log(temp_MeV / opts.T0_MeV) - std::log(rho_gcc / opts.rho0_gcc));
}

// Kept bit-identical, on purpose, to the independent inline copy of this
// formula in tests/test_check.cpp (see synthetic.hpp's doc comment on
// synthetic_cs2()): same operations, same order, same constants.
double synthetic_cs2(double rho_gcc, double temp_MeV, double ye, const SyntheticOptions &opts) {
  const double g = 1.0 + ye;
  const double kT_erg = temp_MeV * MeV_to_erg;
  const double eps = synthetic_eps(rho_gcc, temp_MeV, ye, opts);
  const double p = synthetic_p(rho_gcc, temp_MeV, ye, opts);
  const double m_B = m_B_default_g;
  const double c = c_light_cm_s;
  const double h = 1.0 + (eps + p / rho_gcc) / (c * c);
  return (5.0 / 3.0) * g * kT_erg / (m_B * h * c * c);
}

namespace {

// n values log-uniform in [lo, hi], returned as log10 of the value (n==1
// returns {log10(lo)}).
std::vector<double> log_uniform_log10_axis(size_t n, double lo, double hi) {
  std::vector<double> axis(n);
  const double log_lo = std::log10(lo);
  const double log_hi = std::log10(hi);
  for (size_t i = 0; i < n; ++i) {
    const double t = (n > 1) ? static_cast<double>(i) / static_cast<double>(n - 1) : 0.0;
    axis[i] = log_lo + t * (log_hi - log_lo);
  }
  return axis;
}

// n values linear in [lo, hi].
std::vector<double> linear_axis(size_t n, double lo, double hi) {
  std::vector<double> axis(n);
  for (size_t i = 0; i < n; ++i) {
    const double t = (n > 1) ? static_cast<double>(i) / static_cast<double>(n - 1) : 0.0;
    axis[i] = lo + t * (hi - lo);
  }
  return axis;
}

// 2*pi, used by apply_wiggle(); a local constant rather than <cmath>'s
// non-standard M_PI (not guaranteed by C++17).
constexpr double kTwoPi = 6.283185307179586476925286766559005768394338798750;

// Validates one FlattenDefect/WiggleDefect/OffsetDefect's inclusive block
// against the table's actual grid size; throws std::out_of_range naming
// `what` (the struct name) on any violation. Field-name existence is left to
// RawTable::field() (also std::out_of_range), called by each apply_*()
// below.
void check_block(const RawTable &table, size_t irho0, size_t irho1, size_t kYe0, size_t kYe1,
                  size_t jT0, size_t jT1, const char *what) {
  if (irho0 > irho1 || irho1 >= table.nrho()) {
    throw std::out_of_range(std::string(what) + ": irho range [" + std::to_string(irho0) + ", " +
                             std::to_string(irho1) + "] invalid for nrho=" +
                             std::to_string(table.nrho()));
  }
  if (kYe0 > kYe1 || kYe1 >= table.nye()) {
    throw std::out_of_range(std::string(what) + ": kYe range [" + std::to_string(kYe0) + ", " +
                             std::to_string(kYe1) + "] invalid for nye=" +
                             std::to_string(table.nye()));
  }
  if (jT0 > jT1 || jT1 >= table.ntemp()) {
    throw std::out_of_range(std::string(what) + ": jT range [" + std::to_string(jT0) + ", " +
                             std::to_string(jT1) + "] invalid for ntemp=" +
                             std::to_string(table.ntemp()));
  }
}

// Step 1 of make_synthetic_table()'s defect pipeline: plateaus (see
// FlattenDefect).
void apply_flatten(RawTable &table, const std::vector<FlattenDefect> &defects) {
  for (const FlattenDefect &d : defects) {
    check_block(table, d.irho0, d.irho1, d.kYe0, d.kYe1, d.jT0, d.jT1, "FlattenDefect");
    std::vector<double> &data = table.field(d.field);
    for (size_t kYe = d.kYe0; kYe <= d.kYe1; ++kYe) {
      for (size_t irho = d.irho0; irho <= d.irho1; ++irho) {
        const double v0 = data[table.index(irho, d.jT0, kYe)];
        for (size_t jT = d.jT0; jT <= d.jT1; ++jT) {
          data[table.index(irho, jT, kYe)] = v0;
        }
      }
    }
  }
}

// Step 2: oscillatory perturbation (see WiggleDefect).
void apply_wiggle(RawTable &table, const std::vector<WiggleDefect> &defects) {
  for (const WiggleDefect &d : defects) {
    check_block(table, d.irho0, d.irho1, d.kYe0, d.kYe1, d.jT0, d.jT1, "WiggleDefect");
    std::vector<double> &data = table.field(d.field);
    for (size_t kYe = d.kYe0; kYe <= d.kYe1; ++kYe) {
      for (size_t irho = d.irho0; irho <= d.irho1; ++irho) {
        for (size_t jT = d.jT0; jT <= d.jT1; ++jT) {
          const double phase = kTwoPi * static_cast<double>(jT - d.jT0) / d.period;
          data[table.index(irho, jT, kYe)] += d.amplitude * std::sin(phase);
        }
      }
    }
  }
}

// Step 3: constant shift (see OffsetDefect).
void apply_offset(RawTable &table, const std::vector<OffsetDefect> &defects) {
  for (const OffsetDefect &d : defects) {
    check_block(table, d.irho0, d.irho1, d.kYe0, d.kYe1, d.jT0, d.jT1, "OffsetDefect");
    std::vector<double> &data = table.field(d.field);
    for (size_t kYe = d.kYe0; kYe <= d.kYe1; ++kYe) {
      for (size_t irho = d.irho0; irho <= d.irho1; ++irho) {
        for (size_t jT = d.jT0; jT <= d.jT1; ++jT) {
          data[table.index(irho, jT, kYe)] += d.offset;
        }
      }
    }
  }
}

// Step 4: smooth high-density power-law energy excess (see StiffenDefect).
// Applied to the stored log10 representation: 10^v + A*(rho/rho_c)^alpha,
// re-logged. No cutoff -- for alpha > 0 the term is negligible well below
// rho_c by construction, which is what keeps it analytic in rho.
void apply_stiffen(RawTable &table, const std::vector<StiffenDefect> &defects) {
  for (const StiffenDefect &d : defects) {
    if (!(d.rho_c_gcc > 0.0)) {
      throw std::invalid_argument("StiffenDefect: rho_c_gcc must be positive, got " +
                                   std::to_string(d.rho_c_gcc));
    }
    std::vector<double> &data = table.field(d.field);
    for (size_t irho = 0; irho < table.nrho(); ++irho) {
      const double term = d.amplitude * std::pow(table.rho(irho) / d.rho_c_gcc, d.alpha);
      for (size_t kYe = 0; kYe < table.nye(); ++kYe) {
        for (size_t jT = 0; jT < table.ntemp(); ++jT) {
          const size_t idx = table.index(irho, jT, kYe);
          data[idx] = std::log10(std::pow(10.0, data[idx]) + term);
        }
      }
    }
  }
}

// Step 6 (last): outright sets, so a planted Inf/NaN cannot be perturbed by
// any earlier step (see SetValue).
void apply_setvalue(RawTable &table, const std::vector<SetValue> &defects) {
  for (const SetValue &v : defects) {
    if (v.irho >= table.nrho() || v.jT >= table.ntemp() || v.kYe >= table.nye()) {
      throw std::out_of_range("SetValue: index (irho=" + std::to_string(v.irho) +
                               ", jT=" + std::to_string(v.jT) + ", kYe=" + std::to_string(v.kYe) +
                               ") out of range for grid (nrho=" + std::to_string(table.nrho()) +
                               ", ntemp=" + std::to_string(table.ntemp()) +
                               ", nye=" + std::to_string(table.nye()) + ")");
    }
    table.field(v.field)[table.index(v.irho, v.jT, v.kYe)] = v.value;
  }
}

} // namespace

RawTable make_synthetic_table(const SyntheticOptions &opts) {
  RawTable table;

  std::vector<double> logrho = log_uniform_log10_axis(opts.nrho, opts.rho_min_gcc, opts.rho_max_gcc);
  std::vector<double> logtemp =
      log_uniform_log10_axis(opts.ntemp, opts.temp_min_MeV, opts.temp_max_MeV);
  std::vector<double> ye = linear_axis(opts.nye, opts.ye_min, opts.ye_max);

  table.set_axes(logrho, logtemp, ye);

  const size_t nrho = opts.nrho;
  const size_t ntemp = opts.ntemp;
  const size_t nye = opts.nye;

  std::vector<double> logenergy(nrho * ntemp * nye);
  std::vector<double> entropy(nrho * ntemp * nye);
  std::vector<double> logpress(nrho * ntemp * nye);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t kYe = 0; kYe < nye; ++kYe) {
    const double ye_val = ye[kYe];
    for (size_t jT = 0; jT < ntemp; ++jT) {
      const double temp_MeV = std::pow(10.0, logtemp[jT]);
      for (size_t irho = 0; irho < nrho; ++irho) {
        const double rho_gcc = std::pow(10.0, logrho[irho]);
        const double eps = synthetic_eps(rho_gcc, temp_MeV, ye_val, opts);
        const double p = synthetic_p(rho_gcc, temp_MeV, ye_val, opts);
        const double s = synthetic_s(rho_gcc, temp_MeV, ye_val, opts);
        assert(s > 0.0 && "synthetic entropy model must stay positive over the grid");

        const size_t idx = table.index(irho, jT, kYe);
        logenergy[idx] = std::log10(eps + opts.energy_shift);
        entropy[idx] = s;
        logpress[idx] = std::log10(p);
      }
    }
  }

  table.add_field("logenergy", std::move(logenergy));
  table.add_field("entropy", std::move(entropy));
  table.add_field("logpress", std::move(logpress));
  table.add_attribute("energy_shift", opts.energy_shift);

  if (opts.with_aux_fields) {
    std::vector<double> cs2(nrho * ntemp * nye);
    std::vector<double> gamma(nrho * ntemp * nye);
    std::vector<double> mu_e(nrho * ntemp * nye);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t kYe = 0; kYe < nye; ++kYe) {
      const double ye_val = ye[kYe];
      for (size_t jT = 0; jT < ntemp; ++jT) {
        const double temp_MeV = std::pow(10.0, logtemp[jT]);
        const double kT_erg = temp_MeV * MeV_to_erg;
        for (size_t irho = 0; irho < nrho; ++irho) {
          const double rho_gcc = std::pow(10.0, logrho[irho]);
          const size_t idx = table.index(irho, jT, kYe);
          cs2[idx] = synthetic_cs2(rho_gcc, temp_MeV, ye_val, opts);
          gamma[idx] = 5.0 / 3.0;
          mu_e[idx] = (kT_erg / m_amu_g) * ye_val;
        }
      }
    }

    table.add_field("cs2", std::move(cs2));
    table.add_field("gamma", std::move(gamma));
    table.add_field("mu_e", std::move(mu_e));
  }

  // Defect pipeline, fixed order (see make_synthetic_table()'s doc comment):
  // flatten, wiggle, offset, seed, setvalue (setvalue last so a planted
  // Inf/NaN cannot be perturbed by anything after it).
  apply_flatten(table, opts.flatten);
  apply_wiggle(table, opts.wiggle);
  apply_offset(table, opts.offset);
  apply_stiffen(table, opts.stiffen);
  for (const SeededViolation &v : opts.seed) {
    table.field(v.field)[table.index(v.irho, v.jT, v.kYe)] += v.delta;
  }
  apply_setvalue(table, opts.setvalue);

  return table;
}

SyntheticOptions dirty_synthetic_options() {
  SyntheticOptions opts; // default grid (40 x 30 x 10) and model constants
  opts.with_aux_fields = true;

  // Mimics LS220's clustered non-monotone entropy across a T-window at high
  // rho/Ye: amplitude (0.5) is comparable to the model's per-T-step entropy
  // increment there (g*1.5*d(ln T) ~ 0.35-0.5 kB/baryon per step for
  // kYe in [6,8]), so several adjacent pairs inside the window decrease.
  opts.wiggle.push_back(WiggleDefect{"entropy", 30, 35, 6, 8, 10, 20, 0.5, 4.0});

  // Mimics SRO's near-flat logenergy plateaus, which repair's PAVA
  // strictifies.
  opts.flatten.push_back(FlattenDefect{"logenergy", 5, 15, 2, 4, 3, 12});

  // Mimics SRO's slightly negative cold-corner entropies. The offset must
  // clear this block's actual maximum: with rho0_gcc (1e16) above the whole
  // default rho grid (up to 1e15), synthetic_s's density term
  // -ln(rho/rho0) is positive throughout and largest at the smallest rho, so
  // this low-rho/low-T/low-Ye corner's s ranges up to ~44.2 kB/baryon over
  // the default grid, not near zero -- -50.0 (rather than a smaller-looking
  // round number) is chosen with that margin in mind so every entry in the
  // block lands below zero.
  opts.offset.push_back(OffsetDefect{"entropy", 0, 3, 0, 2, 0, 2, -50.0});

  // Mimics the LS220/SRO acausal high-density corner (eos-causality-repair.md
  // S2): a smooth rho^2 energy excess that makes the *constructed* U
  // superluminal over the top two rho layers at EVERY T and Ye, while leaving
  // monotonicity in T (which it does not depend on) untouched. Numbers set on
  // the default grid (rho 1e5..1e15 over 40 cells): at rho = 1e15 the term is
  // 1.35e17 * (100)^2 = 1.35e21 erg/g, i.e. ~1.5 c^2, giving c_s^2 up to
  // ~1.63 at the top two nodes and comfortably causal values below them.
  //
  // The default grid's rho axis is coarse for this defect on purpose --
  // 0.256 dex per cell, so a capped profile (ln h growing at cs2_cap per
  // unit ln rho) advances 0.585 per cell and a cubic spline's second
  // derivative, which is what c_s^2 is built from, carries an O((0.585)^2/12)
  // ~ 3% error there, larger than the 1% cs2_max/cs2_cap hysteresis. The
  // causal-cap stage therefore collapses this corner's severity (max c_s^2
  // ~1.63 -> ~1.01) but cannot drive the count to zero on THIS grid; the
  // dedicated unit test (tests/test_causal_cap.cpp) plants the same defect
  // on a rho-resolved grid, where the same stage does reach zero. See
  // tests/integration.sh for the narrative.
  opts.stiffen.push_back(StiffenDefect{"logenergy", 1e13, 2.0, 1.35e17});

  // Mimics LS220's corruption: a few Inf/NaN points in cs2/gamma.
  opts.setvalue.push_back(SetValue{"cs2", 20, 15, 5, std::numeric_limits<double>::infinity()});
  opts.setvalue.push_back(SetValue{"cs2", 21, 16, 5, std::numeric_limits<double>::quiet_NaN()});
  opts.setvalue.push_back(SetValue{"gamma", 20, 15, 5, std::numeric_limits<double>::quiet_NaN()});

  return opts;
}

} // namespace eeos
