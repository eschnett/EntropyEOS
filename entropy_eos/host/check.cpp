#include "entropy_eos/host/check.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace eeos {

namespace {

using Loc = CheckClassResult::Loc;

// ln(10), computed once; used for the d/dT = (1/(T ln10)) d/dlog10T and
// d/drho = (1/(rho ln10)) d/dlog10rho chain-rule factors (see
// eos-adapter-F-to-U.md and CODE.md "Test harness").
const double kLn10 = std::log(10.0);

// Denominator floor for the relative-difference metrics of class D/E, so a
// point where the reference derivative happens to vanish doesn't produce a
// spurious division by zero (CheckOptions doc: "max(|...|, tiny)").
constexpr double kTiny = 1e-300;

// Flat index -> (irho, jT, kYe), inverting RawTable::index() (irho fastest,
// then jT, then kYe). Only called for indices into an existing field, whose
// size is guaranteed by RawTable::add_field() to be nrho*ntemp*nye, so nrho
// and ntemp are guaranteed nonzero here.
void unflatten(const RawTable &table, size_t idx, size_t &irho, size_t &jT, size_t &kYe) {
  const size_t nrho = table.nrho();
  const size_t ntemp = table.ntemp();
  irho = idx % nrho;
  const size_t rem = idx / nrho;
  jT = rem % ntemp;
  kYe = rem / ntemp;
}

Loc make_loc(const RawTable &table, size_t irho, size_t jT, size_t kYe, double value) {
  Loc loc;
  loc.irho = irho;
  loc.jT = jT;
  loc.kYe = kYe;
  loc.value = value;
  loc.rho = table.rho(irho);
  loc.temp = table.temp(jT);
  loc.ye = table.yev(kYe);
  return loc;
}

// Bounded "top worst_n by |value|" collector. Insertion is O(worst_n), which
// is fine: worst_n is a small constant (default 10) and this is called at
// most once per evaluated point.
class TopK {
public:
  explicit TopK(size_t n) : n_(n) {}

  void consider(const Loc &loc) {
    if (n_ == 0) return;
    if (items_.size() < n_) {
      items_.push_back(loc);
      return;
    }
    size_t min_idx = 0;
    double min_abs = std::fabs(items_[0].value);
    for (size_t i = 1; i < items_.size(); ++i) {
      const double v = std::fabs(items_[i].value);
      if (v < min_abs) {
        min_abs = v;
        min_idx = i;
      }
    }
    if (std::fabs(loc.value) > min_abs) items_[min_idx] = loc;
  }

  void merge_from(const TopK &other) {
    for (const Loc &l : other.items_) consider(l);
  }

  std::vector<Loc> sorted() const {
    std::vector<Loc> v = items_;
    std::sort(v.begin(), v.end(),
              [](const Loc &a, const Loc &b) { return std::fabs(a.value) > std::fabs(b.value); });
    return v;
  }

private:
  size_t n_;
  std::vector<Loc> items_;
};

// Running statistics for one CheckClassResult, accumulated per-thread under
// OpenMP and merged afterward (see CODE.md "Environment": OpenMP over
// independent columns; merging per-thread accumulators avoids any shared
// mutable state inside the parallel region).
struct Accum {
  size_t count = 0;
  double max_abs = 0.0;
  double sum_sq = 0.0;
  size_t n_points = 0;
  TopK topk;

  explicit Accum(size_t worst_n) : topk(worst_n) {}

  // "Violation" class (B, C, cs2_out_of_range): most points are benign and
  // contribute nothing but their weight to the rms denominator; only
  // violating points feed count/max/rms and are offered to the worst list,
  // so the worst list isn't padded with meaningless zero-metric entries.
  void add_violation(double metric, bool is_violation, const Loc &loc) {
    ++n_points;
    if (!is_violation) return;
    ++count;
    max_abs = std::max(max_abs, std::fabs(metric));
    sum_sq += metric * metric;
    topk.consider(loc);
  }

  // "Diagnostic" class (D, cs2_vs_fd): the metric is meaningful at every
  // point, so max/rms/worst are taken over all evaluated points regardless
  // of whether the point crosses the threshold; `is_violation` only affects
  // `count` (eos-adapter-F-to-U.md / CODE.md: "count = points exceeding
  // tol_consistency; max/rms over all points").
  void add_diagnostic(double metric, bool is_violation, const Loc &loc) {
    ++n_points;
    if (is_violation) ++count;
    max_abs = std::max(max_abs, std::fabs(metric));
    sum_sq += metric * metric;
    topk.consider(loc);
  }

  void merge_from(const Accum &other) {
    count += other.count;
    max_abs = std::max(max_abs, other.max_abs);
    sum_sq += other.sum_sq;
    n_points += other.n_points;
    topk.merge_from(other.topk);
  }

  CheckClassResult finalize(const std::string &name) const {
    CheckClassResult r;
    r.name = name;
    r.count = count;
    r.max = max_abs;
    r.rms = n_points > 0 ? std::sqrt(sum_sq / static_cast<double>(n_points)) : 0.0;
    r.worst = topk.sorted();
    return r;
  }
};

// Number of OpenMP threads that will run the parallel loops below (1 when
// built without OpenMP, so the accumulator-per-thread pattern degenerates to
// a single serial accumulator).
int max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

int this_thread() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

// Three-point finite-difference derivative of `f(idx)` with respect to
// `x[idx]`, central in the interior and one-sided (still second-order
// accurate; standard 3-point Lagrange-derivative formulas) at the two edges.
// Works for a non-uniformly spaced `x` (it reduces to the textbook
// central/one-sided formulas when the spacing is uniform, which is the case
// for the log-uniform synthetic and stellarcollapse.org axes). Requires
// x.size() >= 3.
template <typename F>
double fd3(const F &f, const std::vector<double> &x, size_t j) {
  const size_t n = x.size();
  if (j == 0) {
    const double h1 = x[1] - x[0];
    const double h2 = x[2] - x[1];
    return -((2.0 * h1 + h2) / (h1 * (h1 + h2))) * f(0) + ((h1 + h2) / (h1 * h2)) * f(1) -
           (h1 / (h2 * (h1 + h2))) * f(2);
  }
  if (j == n - 1) {
    const double h1 = x[n - 2] - x[n - 3];
    const double h2 = x[n - 1] - x[n - 2];
    return (h2 / (h1 * (h1 + h2))) * f(n - 3) - ((h1 + h2) / (h1 * h2)) * f(n - 2) +
           ((h1 + 2.0 * h2) / (h2 * (h1 + h2))) * f(n - 1);
  }
  const double h1 = x[j] - x[j - 1];
  const double h2 = x[j + 1] - x[j];
  return -(h2 / (h1 * (h1 + h2))) * f(j - 1) + ((h2 - h1) / (h1 * h2)) * f(j) +
         (h1 / (h2 * (h1 + h2))) * f(j + 1);
}

// A CheckClassResult standing in for a class that could not be evaluated
// (currently: the Maxwell-consistency diagnostics when "logpress" is
// absent). `max`/`rms` are NaN, a sentinel print() recognizes and renders as
// "skipped" rather than a (misleadingly clean) "0" -- see CheckClassResult's
// doc comment in check.hpp.
CheckClassResult skipped_class(const std::string &name) {
  CheckClassResult r;
  r.name = name;
  r.count = 0;
  r.max = std::numeric_limits<double>::quiet_NaN();
  r.rms = std::numeric_limits<double>::quiet_NaN();
  return r;
}

// Non-finite values in a field the pipeline does not *interpret* (anything
// other than "logenergy"/"entropy") are a reported violation class rather
// than a structural fatal: repair and the adapter never read those fields,
// and the writer passes them through byte-identically (CODE.md "Repair
// harness"). This is not hypothetical -- the shipped stellarcollapse LS220
// table carries Inf points in its cs2 and gamma fields.
CheckClassResult check_nonfinite_field(const RawTable &table, const CheckOptions &opts,
                                       const std::string &name) {
  Accum acc(opts.worst_n);
  const std::vector<double> &data = table.field(name);
  for (size_t idx = 0; idx < data.size(); ++idx) {
    const double v = data[idx];
    if (!std::isfinite(v)) {
      size_t irho, jT, kYe;
      unflatten(table, idx, irho, jT, kYe);
      // Metric 1.0, not v: feeding Inf/NaN into max/rms would make the class
      // statistics themselves non-finite. The offending value is still shown
      // via Loc.value.
      acc.add_violation(1.0, true, make_loc(table, irho, jT, kYe, v));
    } else {
      acc.add_violation(0.0, false, Loc{});
    }
  }
  return acc.finalize("nonfinite_" + name);
}

// --- B. Range/positivity ----------------------------------------------------
//
// eps + shift > 0 and p > 0 are automatic once logenergy/logpress are
// finite: eps = 10^logenergy - shift and p = 10^logpress are then finite,
// and pow(10, finite) is always strictly positive, so neither needs its own
// check here (the structural finiteness pass already covers it).
CheckClassResult check_entropy_negative(const RawTable &table, const CheckOptions &opts) {
  Accum acc(opts.worst_n);
  const std::vector<double> &entropy = table.field("entropy");
  for (size_t idx = 0; idx < entropy.size(); ++idx) {
    const double s = entropy[idx];
    const bool bad = s < 0.0;
    if (bad) {
      size_t irho, jT, kYe;
      unflatten(table, idx, irho, jT, kYe);
      acc.add_violation(s, true, make_loc(table, irho, jT, kYe, s));
    } else {
      acc.add_violation(0.0, false, Loc{});
    }
  }
  return acc.finalize("entropy_negative");
}

// --- C. Monotonicity in T ---------------------------------------------------
//
// Per (irho, kYe) column (independent -> OpenMP over kYe), count adjacent
// pairs with entropy[j+1] <= entropy[j] resp. logenergy[j+1] <= logenergy[j].
// The metric/Loc value is the (negative or zero) difference; the worst list
// is the most negative (largest-magnitude violation) pairs.
void check_monotonicity_T(const RawTable &table, const CheckOptions &opts, CheckClassResult &out_entropy,
                           CheckClassResult &out_logenergy) {
  const std::vector<double> &entropy = table.field("entropy");
  const std::vector<double> &logenergy = table.field("logenergy");
  const size_t nrho = table.nrho();
  const size_t ntemp = table.ntemp();
  const size_t nye = table.nye();

  const int nthreads = max_threads();
  std::vector<Accum> local_s(nthreads, Accum(opts.worst_n));
  std::vector<Accum> local_e(nthreads, Accum(opts.worst_n));

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (size_t kYe = 0; kYe < nye; ++kYe) {
    for (size_t irho = 0; irho < nrho; ++irho) {
      const int tid = this_thread();
      for (size_t jT = 0; jT + 1 < ntemp; ++jT) {
        const size_t idx0 = table.index(irho, jT, kYe);
        const size_t idx1 = table.index(irho, jT + 1, kYe);

        const double ds = entropy[idx1] - entropy[idx0];
        local_s[tid].add_violation(ds, ds <= 0.0, make_loc(table, irho, jT, kYe, ds));

        const double de = logenergy[idx1] - logenergy[idx0];
        local_e[tid].add_violation(de, de <= 0.0, make_loc(table, irho, jT, kYe, de));
      }
    }
  }

  Accum acc_s(opts.worst_n), acc_e(opts.worst_n);
  for (int t = 0; t < nthreads; ++t) {
    acc_s.merge_from(local_s[t]);
    acc_e.merge_from(local_e[t]);
  }
  out_entropy = acc_s.finalize("entropy_nonmonotone_T");
  out_logenergy = acc_e.finalize("logenergy_nonmonotone_T");
}

// --- D. Maxwell/thermodynamic-consistency diagnostics -----------------------
//
// Needs "logpress" and at least 3 points on each of the rho/T axes (fd3()'s
// requirement); the caller substitutes skipped_class() placeholders when
// that's not met. Evaluated at *every* (irho, jT, kYe) -- the one-sided fd3
// formulas at the edges make that safe -- parallelized over kYe.
void check_consistency(const RawTable &table, const CheckOptions &opts, CheckClassResult &out_delta_T,
                        CheckClassResult &out_delta_p, CheckClassResult &out_maxwell_s_rho) {
  const std::vector<double> &logenergy = table.field("logenergy");
  const std::vector<double> &entropy = table.field("entropy");
  const std::vector<double> &logpress = table.field("logpress");
  const double shift = table.energy_shift();
  const double m_B = opts.m_B_g;

  const size_t nrho = table.nrho();
  const size_t ntemp = table.ntemp();
  const size_t nye = table.nye();
  const std::vector<double> &logrho_axis = table.logrho();
  const std::vector<double> &logtemp_axis = table.logtemp();

  const int nthreads = max_threads();
  std::vector<Accum> local_T(nthreads, Accum(opts.worst_n));
  std::vector<Accum> local_p(nthreads, Accum(opts.worst_n));
  std::vector<Accum> local_sr(nthreads, Accum(opts.worst_n));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t kYe = 0; kYe < nye; ++kYe) {
    const int tid = this_thread();
    for (size_t jT = 0; jT < ntemp; ++jT) {
      for (size_t irho = 0; irho < nrho; ++irho) {
        const size_t idx = table.index(irho, jT, kYe);
        const double T = table.temp(jT);
        const double rho = table.rho(irho);
        const double p = std::pow(10.0, logpress[idx]);

        auto eps_T = [&](size_t jj) {
          return std::pow(10.0, logenergy[table.index(irho, jj, kYe)]) - shift;
        };
        auto s_T = [&](size_t jj) { return entropy[table.index(irho, jj, kYe)]; };
        auto p_T = [&](size_t jj) { return std::pow(10.0, logpress[table.index(irho, jj, kYe)]); };
        auto eps_R = [&](size_t ii) {
          return std::pow(10.0, logenergy[table.index(ii, jT, kYe)]) - shift;
        };
        auto s_R = [&](size_t ii) { return entropy[table.index(ii, jT, kYe)]; };

        const double dEps_dT = fd3(eps_T, logtemp_axis, jT) / (T * kLn10);
        const double dS_dT = fd3(s_T, logtemp_axis, jT) / (T * kLn10);
        const double dP_dT = fd3(p_T, logtemp_axis, jT) / (T * kLn10);
        const double dEps_dRho = fd3(eps_R, logrho_axis, irho) / (rho * kLn10);
        const double dS_dRho = fd3(s_R, logrho_axis, irho) / (rho * kLn10);

        const double kT_erg = T * MeV_to_erg;

        const double delta_T =
            std::fabs(dEps_dT - (kT_erg / m_B) * dS_dT) / std::max(std::fabs(dEps_dT), kTiny);
        const double delta_p = std::fabs(rho * rho * dEps_dRho - (p - T * dP_dT)) / p;
        const double maxwell_s_rho = std::fabs(dS_dRho + dP_dT * m_B / (rho * rho * MeV_to_erg)) /
                                      std::max(std::fabs(dS_dRho), kTiny);

        // A non-finite metric means a non-finite input reached the FD stencil
        // (possible in "logpress", which is diagnostic-only and therefore not
        // structurally fatal); such points are skipped here and reported by
        // the "nonfinite_<field>" class instead.
        if (std::isfinite(delta_T))
          local_T[tid].add_diagnostic(delta_T, delta_T > opts.tol_consistency,
                                       make_loc(table, irho, jT, kYe, delta_T));
        if (std::isfinite(delta_p))
          local_p[tid].add_diagnostic(delta_p, delta_p > opts.tol_consistency,
                                       make_loc(table, irho, jT, kYe, delta_p));
        if (std::isfinite(maxwell_s_rho))
          local_sr[tid].add_diagnostic(maxwell_s_rho, maxwell_s_rho > opts.tol_consistency,
                                        make_loc(table, irho, jT, kYe, maxwell_s_rho));
      }
    }
  }

  Accum acc_T(opts.worst_n), acc_p(opts.worst_n), acc_sr(opts.worst_n);
  for (int t = 0; t < nthreads; ++t) {
    acc_T.merge_from(local_T[t]);
    acc_p.merge_from(local_p[t]);
    acc_sr.merge_from(local_sr[t]);
  }
  out_delta_T = acc_T.finalize("delta_T");
  out_delta_p = acc_p.finalize("delta_p");
  out_maxwell_s_rho = acc_sr.finalize("maxwell_s_rho");
}

// --- E. Sound speed (report-only diagnostic) --------------------------------
//
// Stored-cs2 unit conventions vary between table providers (some store
// c_s^2 in units of c^2, some don't even claim causal normalization), so
// "cs2_vs_fd" is a diagnostic comparison against the FD-derived value, never
// a pass/fail; "cs2_out_of_range" is the only cs2-based class that can ever
// fail a table.
CheckClassResult check_cs2_out_of_range(const RawTable &table, const CheckOptions &opts) {
  Accum acc(opts.worst_n);
  const std::vector<double> &cs2 = table.field("cs2");
  for (size_t idx = 0; idx < cs2.size(); ++idx) {
    const double v = cs2[idx];
    if (!std::isfinite(v)) {
      // Reported by the "nonfinite_cs2" class; counting Inf here as well
      // would double-report it as a range violation.
      acc.add_violation(0.0, false, Loc{});
      continue;
    }
    double metric = 0.0;
    bool bad = false;
    if (v <= 0.0) {
      metric = v;
      bad = true;
    } else if (v >= 1.0) {
      metric = v - 1.0;
      bad = true;
    }
    if (bad) {
      size_t irho, jT, kYe;
      unflatten(table, idx, irho, jT, kYe);
      acc.add_violation(metric, true, make_loc(table, irho, jT, kYe, metric));
    } else {
      acc.add_violation(0.0, false, Loc{});
    }
  }
  return acc.finalize("cs2_out_of_range");
}

CheckClassResult check_cs2_vs_fd(const RawTable &table, const CheckOptions &opts) {
  const std::vector<double> &logenergy = table.field("logenergy");
  const std::vector<double> &entropy = table.field("entropy");
  const std::vector<double> &logpress = table.field("logpress");
  const std::vector<double> &cs2 = table.field("cs2");
  const double shift = table.energy_shift();
  const double c = c_light_cm_s;

  const size_t nrho = table.nrho();
  const size_t ntemp = table.ntemp();
  const size_t nye = table.nye();
  const std::vector<double> &logrho_axis = table.logrho();
  const std::vector<double> &logtemp_axis = table.logtemp();

  const int nthreads = max_threads();
  std::vector<Accum> local(nthreads, Accum(opts.worst_n));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t kYe = 0; kYe < nye; ++kYe) {
    const int tid = this_thread();
    for (size_t jT = 0; jT < ntemp; ++jT) {
      for (size_t irho = 0; irho < nrho; ++irho) {
        const size_t idx = table.index(irho, jT, kYe);
        const double rho = table.rho(irho);

        auto s_T = [&](size_t jj) { return entropy[table.index(irho, jj, kYe)]; };
        auto p_T = [&](size_t jj) { return std::pow(10.0, logpress[table.index(irho, jj, kYe)]); };
        auto s_R = [&](size_t ii) { return entropy[table.index(ii, jT, kYe)]; };
        auto p_R = [&](size_t ii) { return std::pow(10.0, logpress[table.index(ii, jT, kYe)]); };

        const double T = table.temp(jT);
        const double p = std::pow(10.0, logpress[idx]);
        const double eps = std::pow(10.0, logenergy[idx]) - shift;

        const double dS_dT = fd3(s_T, logtemp_axis, jT) / (T * kLn10);
        const double dP_dT = fd3(p_T, logtemp_axis, jT) / (T * kLn10);
        const double dS_dRho = fd3(s_R, logrho_axis, irho) / (rho * kLn10);
        const double dP_dRho = fd3(p_R, logrho_axis, irho) / (rho * kLn10);

        const double h = 1.0 + (eps + p / rho) / (c * c);
        const double cs2_fd = (dP_dRho - dP_dT * (dS_dRho / dS_dT)) / (h * c * c);

        const double metric = std::fabs(cs2[idx] - cs2_fd) / std::max(std::fabs(cs2_fd), kTiny);
        // Skip points poisoned by non-finite inputs (stored cs2 or logpress
        // feeding the stencil); those are reported by "nonfinite_<field>".
        if (std::isfinite(metric))
          local[tid].add_diagnostic(metric, metric > opts.tol_consistency,
                                     make_loc(table, irho, jT, kYe, metric));
      }
    }
  }

  Accum acc(opts.worst_n);
  for (int t = 0; t < nthreads; ++t) acc.merge_from(local[t]);
  return acc.finalize("cs2_vs_fd");
}

} // namespace

CheckReport check_table(const RawTable &table, const CheckOptions &opts) {
  CheckReport report;

  // --- A. Structural -----------------------------------------------------
  //
  // Every sub-check below runs unconditionally (never short-circuits on the
  // first failure), so a single call reports everything wrong with a badly
  // broken table at once; only afterward do we decide whether to continue
  // to B-D.

  try {
    table.validate_axes();
  } catch (const std::exception &e) {
    report.fatal_messages.push_back(std::string("axes: ") + e.what());
  }

  for (const char *required : {"logenergy", "entropy"}) {
    if (!table.has_field(required)) {
      report.fatal_messages.push_back(std::string("missing required field '") + required + "'");
    }
  }
  if (!table.has_attribute("energy_shift")) {
    report.fatal_messages.push_back("missing required attribute 'energy_shift'");
  }

  // Finiteness is *fatal* only for the fields the pipeline interprets
  // (repairs, or feeds to the M2 adapter): "logenergy" and "entropy".
  // Non-finite values anywhere else become "nonfinite_<field>" violation
  // classes in section B below -- see check_nonfinite_field().
  for (const char *name : {"logenergy", "entropy"}) {
    if (!table.has_field(name)) continue; // absence already reported above
    const std::vector<double> &data = table.field(name);
    size_t bad_count = 0;
    size_t first_bad = 0;
    bool have_first = false;
    for (size_t idx = 0; idx < data.size(); ++idx) {
      if (!std::isfinite(data[idx])) {
        ++bad_count;
        if (!have_first) {
          first_bad = idx;
          have_first = true;
        }
      }
    }
    if (bad_count > 0) {
      size_t irho, jT, kYe;
      unflatten(table, first_bad, irho, jT, kYe);
      std::ostringstream msg;
      msg << "field '" << name << "' has " << bad_count << " non-finite value(s), first at irho="
          << irho << " jT=" << jT << " kYe=" << kYe << " (rho=" << table.rho(irho)
          << " T=" << table.temp(jT) << " Ye=" << table.yev(kYe) << ")";
      report.fatal_messages.push_back(msg.str());
    }
  }

  if (!report.fatal_messages.empty()) {
    report.status = Status::fatal;
    return report;
  }

  // --- B. Range/positivity -------------------------------------------------
  report.classes.push_back(check_entropy_negative(table, opts));

  // Non-finite values in non-interpreted fields (see check_nonfinite_field).
  // Only offending fields get a class, so a clean table isn't padded with
  // one zero-count class per auxiliary field.
  for (const std::string &name : table.field_names()) {
    if (name == "logenergy" || name == "entropy") continue; // fatal above
    CheckClassResult r = check_nonfinite_field(table, opts, name);
    if (r.count > 0) report.classes.push_back(std::move(r));
  }

  // --- C. Monotonicity in T -------------------------------------------------
  CheckClassResult mono_s, mono_e;
  check_monotonicity_T(table, opts, mono_s, mono_e);
  report.classes.push_back(std::move(mono_s));
  report.classes.push_back(std::move(mono_e));

  // --- D. Maxwell/thermodynamic-consistency diagnostics ---------------------
  if (!table.has_field("logpress")) {
    report.classes.push_back(skipped_class("maxwell_consistency"));
  } else if (table.nrho() < 3 || table.ntemp() < 3) {
    // fd3() needs at least 3 points on each differentiated axis.
    report.classes.push_back(skipped_class("maxwell_consistency"));
  } else {
    CheckClassResult delta_T, delta_p, maxwell_s_rho;
    check_consistency(table, opts, delta_T, delta_p, maxwell_s_rho);
    report.classes.push_back(std::move(delta_T));
    report.classes.push_back(std::move(delta_p));
    report.classes.push_back(std::move(maxwell_s_rho));
  }

  // --- E. Sound speed (report-only) -----------------------------------------
  if (table.has_field("cs2")) {
    report.classes.push_back(check_cs2_out_of_range(table, opts));
    if (table.has_field("logpress") && table.nrho() >= 3 && table.ntemp() >= 3) {
      report.classes.push_back(check_cs2_vs_fd(table, opts));
    }
  }

  return report;
}

void CheckReport::print(std::ostream &os) const {
  const std::ios::fmtflags saved_flags = os.flags();
  const std::streamsize saved_precision = os.precision();

  os << "check_table report: status = "
     << (status == Status::ok ? "ok" : status == Status::repaired ? "repaired" : "fatal") << "\n";

  if (!fatal_messages.empty()) {
    os << "fatal structural problems:\n";
    for (const std::string &msg : fatal_messages) {
      os << "  - " << msg << "\n";
    }
  }

  for (const CheckClassResult &c : classes) {
    os << "\n" << c.name << ": count=" << c.count;
    if (std::isnan(c.max)) {
      os << " (skipped: required field not present)\n";
      continue;
    }
    os << " max=" << std::scientific << std::setprecision(4) << c.max << " rms=" << c.rms << "\n";
    if (!c.worst.empty()) {
      os << "  worst offenders (i,j,k -> rho [g/cc], T [MeV], Ye : value):\n";
      for (const CheckClassResult::Loc &loc : c.worst) {
        os << "    (" << std::setw(4) << loc.irho << "," << std::setw(4) << loc.jT << ","
           << std::setw(3) << loc.kYe << ") -> rho=" << std::scientific << std::setprecision(4)
           << loc.rho << " T=" << loc.temp << " Ye=" << std::fixed << std::setprecision(4) << loc.ye
           << " : value=" << std::scientific << std::setprecision(4) << loc.value << "\n";
      }
    }
  }

  os.flags(saved_flags);
  os.precision(saved_precision);
}

} // namespace eeos
