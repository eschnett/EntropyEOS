#include "entropy_eos/host/synthetic.hpp"

#include <cassert>
#include <cmath>

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

  for (const SeededViolation &v : opts.seed) {
    table.field(v.field)[table.index(v.irho, v.jT, v.kYe)] += v.delta;
  }

  return table;
}

} // namespace eeos
