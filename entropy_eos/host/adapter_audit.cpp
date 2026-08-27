#include "entropy_eos/host/adapter_audit.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>

#include "entropy_eos/core/adapter_eval.hpp"
#include "entropy_eos/host/units.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace eeos {

namespace {

using Loc = CheckClassResult::Loc;

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

// --- TopK/Accum: same shape as host/check.cpp's (that file's copies are
// anonymous-namespace-local to its own translation unit, so this audit --
// a separate .cpp -- keeps its own; see host/adapter_build.cpp's
// AuditAccum for the same "each host/*.cpp keeps its own small
// per-thread-accumulator-then-merge helper" pattern). -------------------

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

// Running statistics for one CheckClassResult, accumulated per-thread and
// merged afterward (see host/check.cpp's Accum, which this mirrors).
struct Accum {
  size_t count = 0;
  double max_abs = 0.0;
  double sum_sq = 0.0;
  size_t n_points = 0;
  TopK topk;

  explicit Accum(size_t worst_n) : topk(worst_n) {}

  // Violation class: only violating points feed count/max/rms/worst.
  void add_violation(double metric, bool is_violation, const Loc &loc) {
    ++n_points;
    if (!is_violation) return;
    ++count;
    max_abs = std::max(max_abs, std::fabs(metric));
    sum_sq += metric * metric;
    topk.consider(loc);
  }

  // Diagnostic class: max/rms/worst over every point regardless of
  // `is_violation`; only `count` is threshold-gated.
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

// A CheckClassResult standing in for a class that could not be evaluated
// (here: "delta_p" when "logpress" is absent from `table`) -- see
// check.hpp's CheckClassResult doc comment for the NaN-max sentinel
// convention print() recognizes.
CheckClassResult skipped_class(const std::string &name) {
  CheckClassResult r;
  r.name = name;
  r.count = 0;
  r.max = std::numeric_limits<double>::quiet_NaN();
  r.rms = std::numeric_limits<double>::quiet_NaN();
  return r;
}

// The all-NaN QuantileStats sentinel (see its doc comment): a skipped class,
// or degenerately zero finite-metric values collected.
QuantileStats skipped_quantiles() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  return QuantileStats{nan, nan, nan, nan, nan};
}

// Nearest-rank quantile of `values` at probability p in [0,1]: values must
// already be sorted ascending and non-empty. Deterministic, and monotone
// non-decreasing in p by construction, so a QuantileStats built by calling
// this at increasing p is ordered by construction (p50 <= p90 <= ... <=
// max), as tests/test_adapter_audit.cpp checks.
double nearest_rank_quantile(const std::vector<double> &sorted_values, double p) {
  const size_t n = sorted_values.size();
  size_t rank = static_cast<size_t>(std::ceil(p * static_cast<double>(n)));
  rank = std::max<size_t>(rank, 1);
  rank = std::min(rank, n);
  return sorted_values[rank - 1];
}

// Builds the {p50,p90,p99,p999,max} QuantileStats over every value in
// `values` (order not required on entry; sorted in place). Returns the
// all-NaN sentinel if `values` is empty (see skipped_quantiles()).
QuantileStats compute_quantiles(std::vector<double> &values) {
  if (values.empty()) {
    return skipped_quantiles();
  }
  std::sort(values.begin(), values.end());
  QuantileStats q;
  q.p50 = nearest_rank_quantile(values, 0.50);
  q.p90 = nearest_rank_quantile(values, 0.90);
  q.p99 = nearest_rank_quantile(values, 0.99);
  q.p999 = nearest_rank_quantile(values, 0.999);
  q.max = values.back();
  return q;
}

// --- deterministic per-sample PRNG for the class C soak -----------------
//
// No std::random_device anywhere (AdapterCheckOptions::soak_seed is the
// sole source of randomness, so a report is exactly reproducible). Each
// sampled point gets its own generator, seeded from (soak_seed, index) via
// splitmix64 (the standard fixup for xorshift's weak-seed sensitivity),
// so the soak loop has no state threaded across iterations -- trivially
// OpenMP-parallel and bit-identical regardless of thread count/schedule.

inline uint64_t splitmix64_step(uint64_t &state) {
  uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

class SampleRng {
public:
  SampleRng(unsigned seed, size_t index) {
    uint64_t mix = static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(index) + 1;
    state_ = splitmix64_step(mix);
    if (state_ == 0) state_ = 0x9E3779B97F4A7C15ULL; // xorshift64* requires a nonzero state
  }

  // xorshift64* (Vigna): a fast, well-mixed generator once seeded via
  // splitmix64 above.
  double next_unit() {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    const uint64_t r = state_ * 0x2545F4914F6CDD1DULL;
    // Top 53 bits -> a double uniform in [0,1).
    return static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0); // 2^53
  }

private:
  uint64_t state_;
};

bool eos_point_finite(const EOSPoint &pt) {
  return std::isfinite(pt.U) && std::isfinite(pt.U_rho) && std::isfinite(pt.U_s) &&
         std::isfinite(pt.U_rhorho) && std::isfinite(pt.U_rhos) && std::isfinite(pt.That) &&
         std::isfinite(pt.p) && std::isfinite(pt.h) && std::isfinite(pt.cs2) &&
         std::isfinite(pt.T_F_MeV) && std::isfinite(pt.mu_tilde) && std::isfinite(pt.u_solved);
}

// --- A. Monotonicity, from the build's stored audit (not recomputed) ----
//
// AuditLoc's (x,u,y) are the *unshifted* table's own (log10 rho, log10 T,
// Ye) coordinates (see host/adapter_build.hpp's AuditLoc doc comment), so
// converting to physical rho/T is a plain pow(10,.); no grid-node index
// applies (the scan runs on a refined, off-node grid), so Loc::irho/jT/kYe
// are left at their default 0.
CheckClassResult class_from_monotonicity_audit(const MonotonicityAudit &audit, const std::string &name,
                                                size_t worst_n) {
  CheckClassResult r;
  r.name = name;
  r.count = audit.violation_count;
  r.max = std::fabs(audit.min_value);
  // Not tracked by the build-time scan (a running min, not a sum of
  // squares) and not recomputed here per this module's contract -- left at
  // 0 rather than a misleading placeholder.
  r.rms = 0.0;

  const size_t n = std::min(worst_n, audit.worst.size());
  r.worst.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const AuditLoc &loc = audit.worst[i];
    Loc l;
    l.rho = std::pow(10.0, loc.x);
    l.temp = std::pow(10.0, loc.u);
    l.ye = loc.y;
    l.value = loc.value;
    r.worst.push_back(l);
  }
  return r;
}

// --- B. Node round trip and fidelity -------------------------------------
void audit_nodes(const EntropyEOSView &view, const RawTable &table, double kappa, double conv_t,
                  size_t node_stride, size_t worst_n, double tol_roundtrip, CheckClassResult &out_roundtrip,
                  CheckClassResult &out_delta_T, CheckClassResult &out_delta_p,
                  QuantileStats &out_delta_T_q, QuantileStats &out_delta_p_q, bool &any_fatal,
                  std::vector<std::string> &fatal_messages) {
  const size_t nrho = table.nrho();
  const size_t ntemp = table.ntemp();
  const size_t nye = table.nye();
  const std::vector<double> &entropy = table.field("entropy");
  const bool have_logpress = table.has_field("logpress");
  const std::vector<double> *logpress = have_logpress ? &table.field("logpress") : nullptr;
  const size_t stride = std::max<size_t>(node_stride, 1);
  const double c2 = c_light_cm_s * c_light_cm_s;
  // Cold start (u_guess = NaN, EntropyEOSView::evaluate()'s documented
  // "no warm start" sentinel -- see core/adapter_eval.hpp's
  // detail::aeval_is_nan() branch): warm-starting the T-solve at the exact
  // node answer (u_guess = table.logtemp()[jT]) made this audit trivially
  // green regardless of solver robustness -- every solve started already
  // converged. Cold starts instead exercise the same secant-then-Newton
  // path a real out-of-nowhere evaluate() call takes, so this audit
  // actually probes solve robustness in wiggle regions (a spline-safe
  // repair's near-flat pockets, extension seams, ...), not just the chain
  // rule at a point already found.
  const double u_guess = std::numeric_limits<double>::quiet_NaN();

  const int nthreads = max_threads();
  std::vector<Accum> local_rt(static_cast<size_t>(nthreads), Accum(worst_n));
  std::vector<Accum> local_dT(static_cast<size_t>(nthreads), Accum(worst_n));
  std::vector<Accum> local_dp(static_cast<size_t>(nthreads), Accum(worst_n));
  std::vector<char> local_fatal(static_cast<size_t>(nthreads), 0);
  std::vector<std::vector<std::string>> local_msgs(static_cast<size_t>(nthreads));
  // Every finite delta_T/delta_p metric value, one double per audited node
  // (host-side memory, acceptable per eos-adapter-F-to-U.md S10's robust-
  // fidelity-statistics ask), collected per-thread and merged/sorted below
  // into the report's {p50,p90,p99,p999,max} quantiles.
  std::vector<std::vector<double>> local_dT_vals(static_cast<size_t>(nthreads));
  std::vector<std::vector<double>> local_dp_vals(static_cast<size_t>(nthreads));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t kYe = 0; kYe < nye; kYe += stride) {
    const size_t tid = static_cast<size_t>(this_thread());
    for (size_t jT = 0; jT < ntemp; jT += stride) {
      const double T_j = table.temp(jT);
      for (size_t irho = 0; irho < nrho; irho += stride) {
        const size_t idx = table.index(irho, jT, kYe);
        const double s = entropy[idx];
        const double rho_i = table.rho(irho);
        const double ye_k = table.yev(kYe);

        const EOSPoint pt = view.evaluate(kappa * rho_i, s, ye_k, u_guess);

        if (!eos_point_finite(pt)) {
          local_fatal[tid] = 1;
          std::ostringstream msg;
          msg << "non-finite EOSPoint at table node (irho=" << irho << ",jT=" << jT << ",kYe=" << kYe
              << ") rho=" << rho_i << " T=" << T_j << " Ye=" << ye_k;
          local_msgs[tid].push_back(msg.str());
          continue;
        }

        Loc loc0;
        loc0.irho = irho;
        loc0.jT = jT;
        loc0.kYe = kYe;
        loc0.rho = rho_i;
        loc0.temp = T_j;
        loc0.ye = ye_k;

        const double m_rt = std::fabs(pt.T_F_MeV / T_j - 1.0);
        if (std::isfinite(m_rt)) {
          Loc loc = loc0;
          loc.value = m_rt;
          local_rt[tid].add_violation(m_rt, m_rt > tol_roundtrip, loc);
        }

        const double m_dT = std::fabs(pt.That * kappa / (conv_t * T_j) - 1.0);
        if (std::isfinite(m_dT)) {
          Loc loc = loc0;
          loc.value = m_dT;
          local_dT[tid].add_diagnostic(m_dT, false, loc);
          local_dT_vals[tid].push_back(m_dT);
        }

        if (logpress != nullptr) {
          const double p_table = std::pow(10.0, (*logpress)[idx]);
          const double m_dp = std::fabs(pt.p * c2 / p_table - 1.0);
          if (std::isfinite(m_dp)) {
            Loc loc = loc0;
            loc.value = m_dp;
            local_dp[tid].add_diagnostic(m_dp, false, loc);
            local_dp_vals[tid].push_back(m_dp);
          }
        }
      }
    }
  }

  Accum acc_rt(worst_n), acc_dT(worst_n), acc_dp(worst_n);
  std::vector<double> all_dT_vals, all_dp_vals;
  for (int t = 0; t < nthreads; ++t) {
    const size_t ti = static_cast<size_t>(t);
    acc_rt.merge_from(local_rt[ti]);
    acc_dT.merge_from(local_dT[ti]);
    acc_dp.merge_from(local_dp[ti]);
    all_dT_vals.insert(all_dT_vals.end(), local_dT_vals[ti].begin(), local_dT_vals[ti].end());
    all_dp_vals.insert(all_dp_vals.end(), local_dp_vals[ti].begin(), local_dp_vals[ti].end());
    if (local_fatal[ti]) any_fatal = true;
    for (std::string &m : local_msgs[ti]) fatal_messages.push_back(std::move(m));
  }
  out_roundtrip = acc_rt.finalize("roundtrip_T");
  out_delta_T = acc_dT.finalize("delta_T");
  out_delta_p = acc_dp.finalize("delta_p");
  out_delta_T_q = compute_quantiles(all_dT_vals);
  out_delta_p_q = have_logpress ? compute_quantiles(all_dp_vals) : skipped_quantiles();
}

// --- C. Physicality soak --------------------------------------------------
void audit_soak(const EntropyEOSView &view, const AdapterCheckOptions &opts, CheckClassResult &out_that,
                 CheckClassResult &out_p, CheckClassResult &out_cs2_pos, CheckClassResult &out_cs2_acaus,
                 size_t &maxiter_count, size_t iters_hist[64], double &evals_per_sec, bool &any_fatal,
                 std::vector<std::string> &fatal_messages) {
  const size_t n = opts.soak_n;
  // M2d-2: opts.soak_extended widens the sampled box to the S7 extension
  // zones (EntropyEOSView::x_ext_lo/hi and srange_extended()) instead of
  // the physical one -- everything below (finiteness, flag_maxiter, the
  // physicality classes) is unchanged and now exercises the domain
  // extensions themselves.
  const double x_lo = opts.soak_extended ? view.x_ext_lo : view.x_lo;
  const double x_hi = opts.soak_extended ? view.x_ext_hi : view.x_hi;
  const double y_lo = view.y_lo, y_hi = view.y_hi;

  // Pass 1 (untimed): generate every query point (rho*, s, Ye) up front --
  // this is "PRNG/bookkeeping", not the evaluate() cost the soak measures.
  std::vector<double> q_rho(n), q_s(n), q_ye(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t k = 0; k < n; ++k) {
    SampleRng rng(opts.soak_seed, k);
    const double x = x_lo + rng.next_unit() * (x_hi - x_lo);
    const double y = y_lo + rng.next_unit() * (y_hi - y_lo);
    const double rho_star = std::pow(10.0, x);
    const SRange sr = opts.soak_extended ? view.srange_extended(rho_star, y) : view.srange(rho_star, y);
    const double s = sr.s_min + rng.next_unit() * (sr.s_max - sr.s_min);
    q_rho[k] = rho_star;
    q_ye[k] = y;
    q_s[k] = s;
  }

  const int nthreads = max_threads();
  std::vector<Accum> local_that(static_cast<size_t>(nthreads), Accum(opts.worst_n));
  std::vector<Accum> local_p(static_cast<size_t>(nthreads), Accum(opts.worst_n));
  std::vector<Accum> local_cs2_pos(static_cast<size_t>(nthreads), Accum(opts.worst_n));
  std::vector<Accum> local_cs2_acaus(static_cast<size_t>(nthreads), Accum(opts.worst_n));
  std::vector<size_t> local_maxiter(static_cast<size_t>(nthreads), 0);
  std::vector<std::vector<size_t>> local_hist(static_cast<size_t>(nthreads), std::vector<size_t>(64, 0));
  std::vector<char> local_fatal(static_cast<size_t>(nthreads), 0);
  std::vector<std::vector<std::string>> local_msgs(static_cast<size_t>(nthreads));

  const double nan_guess = std::numeric_limits<double>::quiet_NaN();

  // Pass 2 (timed): cold-start evaluate() calls only.
  const auto t0 = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t k = 0; k < n; ++k) {
    const size_t tid = static_cast<size_t>(this_thread());
    const EOSPoint pt = view.evaluate(q_rho[k], q_s[k], q_ye[k], nan_guess);

    if (!eos_point_finite(pt)) {
      local_fatal[tid] = 1;
      std::ostringstream msg;
      msg << "non-finite EOSPoint in physicality soak at rho*=" << q_rho[k] << " s=" << q_s[k]
          << " Ye=" << q_ye[k];
      local_msgs[tid].push_back(msg.str());
      continue;
    }

    const size_t hb = std::min<size_t>(static_cast<size_t>(pt.iters < 0 ? 0 : pt.iters), 63);
    ++local_hist[tid][hb];
    if (pt.flags & flag_maxiter) ++local_maxiter[tid];

    Loc loc;
    loc.rho = q_rho[k];
    loc.temp = pt.T_F_MeV;
    loc.ye = q_ye[k];

    {
      Loc l = loc;
      l.value = pt.That;
      local_that[tid].add_violation(pt.That, pt.That <= 0.0, l);
    }
    {
      Loc l = loc;
      l.value = pt.p;
      local_p[tid].add_violation(pt.p, pt.p <= 0.0, l);
    }
    {
      Loc l = loc;
      l.value = pt.cs2;
      local_cs2_pos[tid].add_violation(pt.cs2, pt.cs2 <= 0.0, l);
    }
    {
      Loc l = loc;
      l.value = pt.cs2 - 1.0;
      local_cs2_acaus[tid].add_violation(pt.cs2 - 1.0, pt.cs2 >= 1.0, l);
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(t1 - t0).count();
  evals_per_sec = static_cast<double>(n) / std::max(seconds, 1e-12);

  Accum acc_that(opts.worst_n), acc_p(opts.worst_n), acc_cs2_pos(opts.worst_n), acc_cs2_acaus(opts.worst_n);
  maxiter_count = 0;
  for (size_t i = 0; i < 64; ++i) iters_hist[i] = 0;
  for (int t = 0; t < nthreads; ++t) {
    const size_t ti = static_cast<size_t>(t);
    acc_that.merge_from(local_that[ti]);
    acc_p.merge_from(local_p[ti]);
    acc_cs2_pos.merge_from(local_cs2_pos[ti]);
    acc_cs2_acaus.merge_from(local_cs2_acaus[ti]);
    maxiter_count += local_maxiter[ti];
    for (size_t i = 0; i < 64; ++i) iters_hist[i] += local_hist[ti][i];
    if (local_fatal[ti]) any_fatal = true;
    for (std::string &m : local_msgs[ti]) fatal_messages.push_back(std::move(m));
  }
  out_that = acc_that.finalize("That_nonpositive");
  out_p = acc_p.finalize("p_nonpositive");
  out_cs2_pos = acc_cs2_pos.finalize("cs2_nonpositive");
  out_cs2_acaus = acc_cs2_acaus.finalize("cs2_acausal");
}

// --- D. Extension seam continuity (diagnostic only, M2d-2) ----------------
//
// See check_adapter()'s doc comment for the design: a ~40x20 grid per seam
// (all four of u_lo, u_hi, x_lo, x_hi) across the seam's two non-seam axes,
// evaluating U/U_s just inside/outside the seam (offset kOffset in the seam
// coordinate) through the *public* evaluate()/sigma_extended() API -- the
// same interface a con2prim caller sees, so this measures what the caller
// actually experiences at the seam, not an internal spline detail.
// sigma_extended() converts a chosen u just inside/outside a u_lo/u_hi seam
// (or, for an x_lo/x_hi seam, a u picked anywhere in the physical range)
// into the matching s, so both the "inside" and "outside" evaluate() calls
// land at the intended (x,u) pair via the same T-solve path a real caller
// takes -- not a back door into the spline internals.
void audit_seam_jumps(const EntropyEOSView &view, size_t worst_n, CheckClassResult &out) {
  constexpr double kOffset = 1e-7;
  constexpr int kNa = 40, kNb = 20;
  constexpr double kTiny = 1e-300;
  const double nan_guess = std::numeric_limits<double>::quiet_NaN();

  const int nthreads = max_threads();
  std::vector<Accum> local(static_cast<size_t>(nthreads), Accum(worst_n));

  // Evaluates just inside/outside one seam point and records the worse of
  // U's and U_s's relative jump (diagnostic: never a "violation", count
  // stays 0 -- see Accum::add_diagnostic()). Skips a point outright if
  // either side is non-finite (that is B/C's fatal-detection job, not this
  // diagnostic's).
  auto record = [&](Accum &acc, double rho_in, double s_in, double rho_out, double s_out, double ye,
                     double rho_loc, double temp_loc) {
    const EOSPoint pin = view.evaluate(rho_in, s_in, ye, nan_guess);
    const EOSPoint pout = view.evaluate(rho_out, s_out, ye, nan_guess);
    if (!eos_point_finite(pin) || !eos_point_finite(pout)) return;
    const double jump_U = std::fabs(pout.U - pin.U) / std::max(std::fabs(pin.U), kTiny);
    const double jump_Us = std::fabs(pout.U_s - pin.U_s) / std::max(std::fabs(pin.U_s), kTiny);
    const double metric = std::max(jump_U, jump_Us);
    Loc loc;
    loc.rho = rho_loc;
    loc.temp = temp_loc;
    loc.ye = ye;
    loc.value = metric;
    acc.add_diagnostic(metric, false, loc);
  };

  // u_lo / u_hi seams: grid over (x* in [x_lo,x_hi], Ye in [y_lo,y_hi]); Ye
  // varies inside the parallel-for so both seams share the same rho column.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int ia = 0; ia < kNa; ++ia) {
    Accum &acc = local[static_cast<size_t>(this_thread())];
    const double x =
        view.x_lo + (view.x_hi - view.x_lo) * (static_cast<double>(ia) + 0.5) / static_cast<double>(kNa);
    const double rho = std::pow(10.0, x);
    for (int side = 0; side < 2; ++side) {
      const double u_seam = side == 0 ? view.u_lo : view.u_hi;
      const double u_in = side == 0 ? u_seam + kOffset : u_seam - kOffset;
      const double u_out = side == 0 ? u_seam - kOffset : u_seam + kOffset;
      for (int ib = 0; ib < kNb; ++ib) {
        const double ye = view.y_lo +
                           (view.y_hi - view.y_lo) * (static_cast<double>(ib) + 0.5) / static_cast<double>(kNb);
        const double s_in = view.sigma_extended(rho, u_in, ye);
        const double s_out = view.sigma_extended(rho, u_out, ye);
        record(acc, rho, s_in, rho, s_out, ye, rho, std::pow(10.0, u_seam));
      }
    }
  }

  // x_lo / x_hi seams: grid over (u=log10(T) in [u_lo,u_hi], Ye).
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int ia = 0; ia < kNa; ++ia) {
    Accum &acc = local[static_cast<size_t>(this_thread())];
    const double u =
        view.u_lo + (view.u_hi - view.u_lo) * (static_cast<double>(ia) + 0.5) / static_cast<double>(kNa);
    for (int side = 0; side < 2; ++side) {
      const double x_seam = side == 0 ? view.x_lo : view.x_hi;
      const double rho_seam = std::pow(10.0, x_seam);
      const double rho_in =
          side == 0 ? std::pow(10.0, x_seam + kOffset) : std::pow(10.0, x_seam - kOffset);
      const double rho_out =
          side == 0 ? std::pow(10.0, x_seam - kOffset) : std::pow(10.0, x_seam + kOffset);
      for (int ib = 0; ib < kNb; ++ib) {
        const double ye = view.y_lo +
                           (view.y_hi - view.y_lo) * (static_cast<double>(ib) + 0.5) / static_cast<double>(kNb);
        const double s = view.sigma_extended(rho_seam, u, ye);
        record(acc, rho_in, s, rho_out, s, ye, rho_seam, std::pow(10.0, u));
      }
    }
  }

  Accum total(worst_n);
  for (int t = 0; t < nthreads; ++t) total.merge_from(local[static_cast<size_t>(t)]);
  out = total.finalize("extension_seam_jump");
}

// --- E. Extension-band c_s^2 map (M3g, eos-causal-tail.md S5) -------------
//
// See check_adapter()'s doc comment for the design. One band = one of the
// four extension zones; the band axis is walked at opts.ext_band_refine
// samples per grid cell (skipping the seam itself, which is in-box), the two
// other axes at opts.ext_band_other_refine over their physical ranges.
// Points are evaluated through EntropyEOSView::eval_at() -- native (x,u,y)
// coordinates, no T-solve -- plus one extended sigma sample for the tail's
// own sigma_u.

// The four bands, in the report's fixed order.
enum class Band { u_low = 0, u_high = 1, x_low = 2, x_high = 3 };
const char *const kBandNames[4] = {"u_low", "u_high", "x_low", "x_high"};

// Sample count for one axis spanned by `n_nodes` table nodes at `refine`
// samples per cell.
size_t axis_samples(int n_nodes, size_t refine) {
  return static_cast<size_t>(n_nodes - 1) * refine + 1;
}

void audit_ext_band(const EntropyEOSView &view, const AdapterCheckOptions &opts, Band band,
                     std::vector<CheckClassResult> &out, size_t &n_samples) {
  const std::string suffix = std::string("ext_") + kBandNames[static_cast<int>(band)] + "_";
  const size_t refine = opts.ext_band_refine;
  const size_t other = opts.ext_band_other_refine;

  if (refine == 0 || other == 0) {
    n_samples = 0;
    out.push_back(skipped_class(suffix + "cs2_acausal"));
    out.push_back(skipped_class(suffix + "p_nonpositive"));
    out.push_back(skipped_class(suffix + "sigma_u_nonpositive"));
    out.push_back(skipped_class(suffix + "cs2_nonpositive"));
    return;
  }

  const bool u_band = band == Band::u_low || band == Band::u_high;
  // Band axis: `n_band` samples strictly inside the extension zone, from the
  // seam outward (index 1..n_band, so the in-box seam itself is excluded).
  const double seam = band == Band::u_low   ? view.u_lo
                      : band == Band::u_high ? view.u_hi
                      : band == Band::x_low  ? view.x_lo
                                              : view.x_hi;
  const double edge = band == Band::u_low   ? view.u_ext_lo
                      : band == Band::u_high ? view.u_ext_hi
                      : band == Band::x_low  ? view.x_ext_lo
                                              : view.x_ext_hi;
  const double h = u_band ? view.sigma.hu : view.sigma.hx;
  const int ext_cells = std::max(1, static_cast<int>(std::lround(std::fabs(edge - seam) / h)));
  const size_t n_band = static_cast<size_t>(ext_cells) * refine;
  const double d_band = (edge - seam) / static_cast<double>(n_band);

  // The two other axes, over their physical ranges.
  const size_t n_a = u_band ? axis_samples(view.sigma.nx, other) : axis_samples(view.sigma.nu, other);
  const double a_lo = u_band ? view.x_lo : view.u_lo;
  const double a_hi = u_band ? view.x_hi : view.u_hi;
  const size_t n_y = axis_samples(view.sigma.ny, other);

  n_samples = n_band * n_a * n_y;

  const int nthreads = max_threads();
  std::vector<Accum> local_cs2(static_cast<size_t>(nthreads), Accum(opts.worst_n));
  std::vector<Accum> local_p(static_cast<size_t>(nthreads), Accum(opts.worst_n));
  std::vector<Accum> local_sig(static_cast<size_t>(nthreads), Accum(opts.worst_n));
  std::vector<Accum> local_cs2neg(static_cast<size_t>(nthreads), Accum(opts.worst_n));

  const detail::ExtSpec sig_spec = view.sigma_ext_spec();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long long ia_ll = 0; ia_ll < static_cast<long long>(n_a); ++ia_ll) {
    const size_t ia = static_cast<size_t>(ia_ll);
    const size_t tid = static_cast<size_t>(this_thread());
    const double a = n_a > 1 ? a_lo + (a_hi - a_lo) * static_cast<double>(ia) /
                                          static_cast<double>(n_a - 1)
                             : a_lo;
    for (size_t iy = 0; iy < n_y; ++iy) {
      const double y = n_y > 1 ? view.y_lo + (view.y_hi - view.y_lo) * static_cast<double>(iy) /
                                                 static_cast<double>(n_y - 1)
                                : view.y_lo;
      for (size_t ib = 1; ib <= n_band; ++ib) {
        const double c = seam + d_band * static_cast<double>(ib);
        const double x = u_band ? a : c;
        const double u = u_band ? c : a;

        const EOSPoint pt = view.eval_at(x, u, y, 0u, 0);
        const double sigma_u = detail::aeval_extended(view.sigma, x, u, y, sig_spec).fu;

        Loc loc;
        loc.rho = std::pow(10.0, x);
        loc.temp = std::pow(10.0, u);
        loc.ye = y;

        if (std::isfinite(pt.cs2)) {
          Loc l = loc;
          l.value = pt.cs2 - 1.0;
          local_cs2[tid].add_violation(pt.cs2 - 1.0, pt.cs2 >= 1.0, l);
          Loc l2 = loc;
          l2.value = pt.cs2;
          local_cs2neg[tid].add_violation(pt.cs2, pt.cs2 <= 0.0, l2);
        }
        if (std::isfinite(pt.p)) {
          Loc l = loc;
          l.value = pt.p;
          local_p[tid].add_violation(pt.p, pt.p <= 0.0, l);
        }
        if (std::isfinite(sigma_u)) {
          Loc l = loc;
          l.value = sigma_u;
          local_sig[tid].add_violation(sigma_u, sigma_u <= 0.0, l);
        }
      }
    }
  }

  Accum acc_cs2(opts.worst_n), acc_p(opts.worst_n), acc_sig(opts.worst_n), acc_cs2neg(opts.worst_n);
  for (int t = 0; t < nthreads; ++t) {
    acc_cs2.merge_from(local_cs2[static_cast<size_t>(t)]);
    acc_p.merge_from(local_p[static_cast<size_t>(t)]);
    acc_sig.merge_from(local_sig[static_cast<size_t>(t)]);
    acc_cs2neg.merge_from(local_cs2neg[static_cast<size_t>(t)]);
  }
  out.push_back(acc_cs2.finalize(suffix + "cs2_acausal"));
  out.push_back(acc_p.finalize(suffix + "p_nonpositive"));
  out.push_back(acc_sig.finalize(suffix + "sigma_u_nonpositive"));
  out.push_back(acc_cs2neg.finalize(suffix + "cs2_nonpositive"));
}

// The u_hi seam walk that reports where M3g's causal slope clamp binds and
// where it lost to the monotonicity floor (eos-causal-tail.md S3).
void audit_causal_clamp_seam(const EntropyEOSView &view, const AdapterCheckOptions &opts, size_t &n,
                              size_t &active, size_t &floor_wins) {
  n = active = floor_wins = 0;
  const size_t other = opts.ext_band_other_refine;
  if (other == 0 || opts.ext_band_refine == 0) return;

  const size_t n_x = axis_samples(view.sigma.nx, other);
  const size_t n_y = axis_samples(view.sigma.ny, other);

  const int nthreads = max_threads();
  std::vector<size_t> local_active(static_cast<size_t>(nthreads), 0),
      local_floor(static_cast<size_t>(nthreads), 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long long ix_ll = 0; ix_ll < static_cast<long long>(n_x); ++ix_ll) {
    const size_t ix = static_cast<size_t>(ix_ll);
    const size_t tid = static_cast<size_t>(this_thread());
    const double x = n_x > 1 ? view.x_lo + (view.x_hi - view.x_lo) * static_cast<double>(ix) /
                                               static_cast<double>(n_x - 1)
                             : view.x_lo;
    const double rho = std::pow(10.0, x);
    for (size_t iy = 0; iy < n_y; ++iy) {
      const double y = n_y > 1 ? view.y_lo + (view.y_hi - view.y_lo) * static_cast<double>(iy) /
                                                 static_cast<double>(n_y - 1)
                                : view.y_lo;
      const UHighTailInfo info = view.u_high_tail_info(rho, y);
      if (info.clamped) ++local_active[tid];
      if (info.floor_wins) ++local_floor[tid];
    }
  }

  n = n_x * n_y;
  for (int t = 0; t < nthreads; ++t) {
    active += local_active[static_cast<size_t>(t)];
    floor_wins += local_floor[static_cast<size_t>(t)];
  }
}

} // namespace

AdapterReport check_adapter(const EntropyEOS &adapter, const RawTable &table,
                             const AdapterCheckOptions &opts) {
  AdapterReport report;
  report.kappa = adapter.kappa();
  report.m_B_star_g = adapter.m_B_star_g();
  report.soak_n = opts.soak_n;

  const EntropyEOSView view = adapter.view();

  // Reconstruct conv_t (== MeV_to_erg / (m_B_table_g * c^2)) from the two
  // scalars the report itself carries (kappa, m_B_star_g) rather than
  // reaching back into `adapter` for its own conv_t()/m_B_table_g() --
  // this is the delta_T fidelity metric's reference m_B*/consistency
  // conversion (eos-adapter-F-to-U.md S10, CODE.md "M2 design notes").
  const double m_B_table_g = adapter.m_B_star_g() / adapter.kappa();
  const double conv_t = MeV_to_erg / (m_B_table_g * c_light_cm_s * c_light_cm_s);

  bool any_fatal = false;
  std::vector<std::string> fatal_messages;

  // --- A. Monotonicity (stored build audit) -------------------------------
  report.classes.push_back(
      class_from_monotonicity_audit(adapter.audit().sigma_u, "spline_sigma_u_nonpositive", opts.worst_n));
  report.classes.push_back(
      class_from_monotonicity_audit(adapter.audit().L_u, "spline_L_u_nonpositive", opts.worst_n));

  // --- B. Node round trip and fidelity -------------------------------------
  CheckClassResult roundtrip_T, delta_T, delta_p;
  audit_nodes(view, table, adapter.kappa(), conv_t, opts.node_stride, opts.worst_n, opts.tol_roundtrip,
              roundtrip_T, delta_T, delta_p, report.delta_T_quantiles, report.delta_p_quantiles, any_fatal,
              fatal_messages);
  report.classes.push_back(std::move(roundtrip_T));
  report.classes.push_back(std::move(delta_T));
  if (table.has_field("logpress")) {
    report.classes.push_back(std::move(delta_p));
  } else {
    report.classes.push_back(skipped_class("delta_p"));
  }

  // --- C. Physicality soak --------------------------------------------------
  CheckClassResult that_np, p_np, cs2_pos, cs2_acaus;
  audit_soak(view, opts, that_np, p_np, cs2_pos, cs2_acaus, report.maxiter_count, report.iters_hist,
             report.evals_per_sec, any_fatal, fatal_messages);
  report.classes.push_back(std::move(that_np));
  report.classes.push_back(std::move(p_np));
  report.classes.push_back(std::move(cs2_pos));
  report.classes.push_back(std::move(cs2_acaus));

  // --- D. Extension seam continuity (diagnostic only, M2d-2) ---------------
  CheckClassResult seam_jump;
  audit_seam_jumps(view, opts.worst_n, seam_jump);
  report.classes.push_back(std::move(seam_jump));

  // --- E. Extension-band map (M3g) -----------------------------------------
  for (int b = 0; b < 4; ++b) {
    audit_ext_band(view, opts, static_cast<Band>(b), report.classes,
                   report.ext_band_n[static_cast<size_t>(b)]);
  }
  audit_causal_clamp_seam(view, opts, report.ext_clamp_seam_n, report.ext_clamp_active,
                           report.ext_clamp_floor_wins);

  report.fatal_messages = std::move(fatal_messages);
  report.status = any_fatal ? Status::fatal : Status::ok;
  return report;
}

bool adapter_needs_attention(const AdapterReport &report) {
  if (report.maxiter_count > 0) return true;
  static const char *const kRelevant[] = {"spline_sigma_u_nonpositive", "spline_L_u_nonpositive",
                                           "roundtrip_T",                "That_nonpositive",
                                           "p_nonpositive",              "cs2_nonpositive",
                                           "cs2_acausal",
                                           // M3g class E: the three bands a converged, flagged,
                                           // legitimately-used state can live in. The x_high band is
                                           // deliberately absent -- see adapter_audit.hpp.
                                           "ext_u_low_cs2_acausal",
                                           "ext_u_low_p_nonpositive",
                                           "ext_u_low_sigma_u_nonpositive",
                                           "ext_u_high_cs2_acausal",
                                           "ext_u_high_p_nonpositive",
                                           "ext_u_high_sigma_u_nonpositive",
                                           "ext_x_low_cs2_acausal",
                                           "ext_x_low_p_nonpositive",
                                           "ext_x_low_sigma_u_nonpositive"};
  for (const CheckClassResult &c : report.classes) {
    for (const char *name : kRelevant) {
      if (c.name == name && c.count > 0) return true;
    }
  }
  return false;
}

void AdapterReport::print(std::ostream &os) const {
  const std::ios::fmtflags saved_flags = os.flags();
  const std::streamsize saved_precision = os.precision();

  os << "check_adapter report: status="
     << (status == Status::ok ? "ok" : status == Status::repaired ? "repaired" : "fatal") << "\n";
  os << "kappa=" << std::scientific << std::setprecision(15) << kappa << "\n";
  os << "m_B_star_g=" << m_B_star_g << "\n";

  if (!fatal_messages.empty()) {
    os << "fatal problems (non-finite EOSPoint):\n";
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
    os << " max=" << std::scientific << std::setprecision(6) << c.max << " rms=" << c.rms << "\n";
    if (c.name == "delta_T" || c.name == "delta_p") {
      const QuantileStats &q = c.name == "delta_T" ? delta_T_quantiles : delta_p_quantiles;
      os << "  quantiles: p50=" << std::scientific << std::setprecision(6) << q.p50 << " p90=" << q.p90
         << " p99=" << q.p99 << " p999=" << q.p999 << " max=" << q.max << "\n";
    }
    if (!c.worst.empty()) {
      os << "  worst offenders (rho [g/cc], T [MeV], Ye : value):\n";
      for (const CheckClassResult::Loc &loc : c.worst) {
        os << "    rho=" << std::scientific << std::setprecision(6) << loc.rho << " T=" << loc.temp
           << " Ye=" << std::fixed << std::setprecision(4) << loc.ye << " : value=" << std::scientific
           << std::setprecision(6) << loc.value << "\n";
      }
    }
  }

  os << "\nextension-band map (M3g): samples u_low=" << ext_band_n[0] << " u_high=" << ext_band_n[1]
     << " x_low=" << ext_band_n[2] << " x_high=" << ext_band_n[3]
     << " (x_high is report-only, never an exit-code violation)\n";
  os << "  u_hi causal slope clamp: seam points=" << ext_clamp_seam_n << " clamp active="
     << ext_clamp_active << " monotonicity floor won=" << ext_clamp_floor_wins << "\n";

  os << "\nphysicality soak: n=" << soak_n << " maxiter_count=" << maxiter_count
     << " evals_per_sec=" << std::scientific << std::setprecision(6) << evals_per_sec << " evals/sec\n";
  os << "  cold-start iteration histogram:\n";
  for (size_t i = 0; i < 64; ++i) {
    if (iters_hist[i] > 0) {
      os << "    iters=" << i << ": " << iters_hist[i] << "\n";
    }
  }

  os.flags(saved_flags);
  os.precision(saved_precision);
}

} // namespace eeos
