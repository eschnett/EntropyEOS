#include "entropy_eos/host/con2prim_audit.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>

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

// --- TopK/Accum/SampleRng: same shape as host/adapter_audit.cpp's (that
// file's copies are anonymous-namespace-local to its own translation unit,
// so this audit -- a separate .cpp -- keeps its own; see that file's own
// comment for the "each host/*.cpp keeps its own small per-thread-
// accumulator-then-merge helper" convention). Only the violation-class half
// of Accum is needed here (both "c2p_failed" and "c2p_roundtrip" are
// violation classes: only violating states feed count/max/rms/worst). -----

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

// Deterministic per-sample PRNG (splitmix64 seed fixup + xorshift64*),
// identical construction to adapter_audit.cpp's SampleRng: each sampled
// state gets its own generator seeded from (seed, index), so the sampling
// loop has no state threaded across iterations -- trivially OpenMP-parallel
// and bit-identical regardless of thread count/schedule.
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
    if (state_ == 0) state_ = 0x9E3779B97F4A7C15ULL;
  }

  double next_unit() {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    const uint64_t r = state_ * 0x2545F4914F6CDD1DULL;
    return static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0); // 2^53
  }

private:
  uint64_t state_;
};

// --- Quantile helpers (same construction as adapter_audit.cpp's; no
// "skipped" sentinel needed here since every QuantileStats field this
// module reports is always collected, never conditional on an optional
// table field). ------------------------------------------------------------

double nearest_rank_quantile(const std::vector<double> &sorted_values, double p) {
  const size_t n = sorted_values.size();
  size_t rank = static_cast<size_t>(std::ceil(p * static_cast<double>(n)));
  rank = std::max<size_t>(rank, 1);
  rank = std::min(rank, n);
  return sorted_values[rank - 1];
}

QuantileStats compute_quantiles(std::vector<double> &values) {
  if (values.empty()) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return QuantileStats{nan, nan, nan, nan, nan};
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

// --- Finiteness guards (mirrors adapter_audit.cpp's eos_point_finite();
// this module keeps its own copy for the same "each host/*.cpp keeps its
// own" reason, plus the con2prim/prim2con-specific output shapes). ---------

bool eos_point_finite(const EOSPoint &pt) {
  return std::isfinite(pt.U) && std::isfinite(pt.U_rho) && std::isfinite(pt.U_s) &&
         std::isfinite(pt.U_rhorho) && std::isfinite(pt.U_rhos) && std::isfinite(pt.That) &&
         std::isfinite(pt.p) && std::isfinite(pt.h) && std::isfinite(pt.cs2) &&
         std::isfinite(pt.T_F_MeV) && std::isfinite(pt.mu_tilde) && std::isfinite(pt.u_solved);
}

bool c2p_out_finite(const Con2PrimOut &out) {
  return std::isfinite(out.rho) && std::isfinite(out.s) && std::isfinite(out.ye) &&
         std::isfinite(out.w) && std::isfinite(out.W) && std::isfinite(out.v_par) &&
         std::isfinite(out.v_perp) && eos_point_finite(out.eos);
}

bool prim2con_out_finite(const Prim2ConOut &out) {
  return std::isfinite(out.D) && std::isfinite(out.tau) && std::isfinite(out.D_Y) &&
         std::isfinite(out.S_par) && std::isfinite(out.S_perp) && std::isfinite(out.B2);
}

// --- State sampling (deterministic; see con2prim_audit.hpp's doc comment
// for the design rationale). One state per index k in [0, opts.n_states):
// a per-index SampleRng draws, in order, the log-density fraction, Ye
// fraction, entropy fraction, rapidity fraction, a B2-zero coin, the
// log-magnetization fraction, cos_vB, and three warm-start perturbation
// fractions -- always in this fixed order, so the sequence (and hence every
// sampled state) is independent of thread count/schedule. ------------------

struct SampledState {
  real rho = 0, ye = 0, s = 0, w = 0, B2 = 0, cos_vB = 0;
  real s_guess = 0, w_guess = 0, u_guess = 0;
  real T_input = 0; // pt0.T_F_MeV at (rho,s,ye) -- Loc::temp for this state's worst-offender entries
  Con2PrimIn cin{};
  Prim2ConOut truth{};
  bool cold = false; // true if this state also gets a cold-start pass (every 10th)
};

SampledState sample_state(const EntropyEOSView &view, unsigned seed, size_t k,
                           const Con2PrimCheckOptions &opts) {
  SampledState st;
  SampleRng rng(seed, k);
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // rho: log-uniform inside a 5%-per-side margin of the physical x box.
  const double margin = 0.05;
  const double xspan = view.x_hi - view.x_lo;
  const double xlo = view.x_lo + margin * xspan;
  const double xhi = view.x_hi - margin * xspan;
  const double x = xlo + rng.next_unit() * (xhi - xlo);
  st.rho = std::pow(10.0, x);

  // Ye: uniform over the full physical range.
  st.ye = view.y_lo + rng.next_unit() * (view.y_hi - view.y_lo);

  // s: uniform inside a 5%-per-side margin of the pointwise srange().
  const SRange sr = view.srange(st.rho, st.ye);
  const double sspan = sr.s_max - sr.s_min;
  st.s = sr.s_min + margin * sspan + rng.next_unit() * (sspan - 2.0 * margin * sspan);

  // w: uniform in [0, w_max_sample].
  st.w = rng.next_unit() * opts.w_max_sample;

  // B2: a 10% coin (independent of the sigma draw below, so the B2==0
  // subset and the cold-pass subset -- chosen by state index, see below --
  // are not artificially correlated) gives B2=0 exactly; otherwise
  // sigma = B2/(rho*h) log-uniform in [1e-6, sigma_max], with h from a
  // preliminary evaluate() at (rho,s,ye).
  const double b2_coin = rng.next_unit();
  const bool b2_zero = b2_coin < 0.10;
  const double log_sigma_lo = std::log(1e-6);
  const double log_sigma_hi = std::log(opts.sigma_max);
  const double sigma = std::exp(log_sigma_lo + rng.next_unit() * (log_sigma_hi - log_sigma_lo));

  // cos_vB: uniform in [-1, 1].
  st.cos_vB = -1.0 + 2.0 * rng.next_unit();

  // Warm-start perturbation fractions, each in [-1, 1].
  const double u1 = -1.0 + 2.0 * rng.next_unit();
  const double u2 = -1.0 + 2.0 * rng.next_unit();
  const double u3 = -1.0 + 2.0 * rng.next_unit();

  const EOSPoint pt0 = view.evaluate(st.rho, st.s, st.ye, nan);
  st.T_input = pt0.T_F_MeV;
  st.B2 = b2_zero ? 0.0 : sigma * st.rho * pt0.h;

  st.truth = prim2con(view, st.rho, st.s, st.ye, st.w, st.B2, st.cos_vB, pt0.u_solved);
  st.cin = Con2PrimIn{st.truth.D, st.truth.tau, st.truth.D_Y, st.truth.S_par, st.truth.S_perp, st.truth.B2};

  // Warm-start guesses: s_guess multiplicative (matches how s is compared
  // in prim-space, |ds|/max(|s|,1)), w_guess additive (w ranges over
  // [0,w_max], including through 0, where a multiplicative perturbation
  // would degenerate), u_guess additive on log10(T) (mirrors w_guess: like
  // w, u_solved is an unbounded-sign natural coordinate, not a
  // always-positive one where a multiplicative perturbation makes sense).
  st.s_guess = st.s * (1.0 + opts.warm_rel * u1);
  st.w_guess = st.w + opts.warm_rel * u2;
  st.u_guess = pt0.u_solved + opts.warm_rel * u3;

  st.cold = (k % 10 == 0);
  return st;
}

} // namespace

Con2PrimReport check_con2prim(const EntropyEOS &adapter, const Con2PrimCheckOptions &opts) {
  Con2PrimReport report;
  report.n_states = opts.n_states;

  const EntropyEOSView view = adapter.view();
  const size_t n = opts.n_states;
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // --- Pass 1 (untimed): sample every state and build its truth
  // conservative state (prim2con() plus the preliminary evaluate() it
  // needs) -- this is "point generation", not the con2prim() solve cost the
  // timed passes below measure.
  std::vector<SampledState> states(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t k = 0; k < n; ++k) {
    states[static_cast<size_t>(k)] = sample_state(view, opts.seed, k, opts);
  }

  std::vector<size_t> cold_idx;
  cold_idx.reserve(n / 10 + 1);
  for (size_t k = 0; k < n; ++k) {
    if (states[k].cold) cold_idx.push_back(k);
  }
  const size_t ncold = cold_idx.size();

  // --- Pass 2 (TIMED): warm con2prim() over every state.
  std::vector<Con2PrimOut> warm(n);
  const auto tw0 = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t k = 0; k < n; ++k) {
    const SampledState &s = states[k];
    warm[k] = con2prim(view, s.cin, opts.solver, s.s_guess, s.w_guess, s.u_guess);
  }
  const auto tw1 = std::chrono::steady_clock::now();
  const double warm_seconds = std::chrono::duration<double>(tw1 - tw0).count();
  report.solves_per_sec_warm = static_cast<double>(n) / std::max(warm_seconds, 1e-12);

  // --- Pass 3 (TIMED): cold con2prim() over the ~10% cold subset.
  std::vector<Con2PrimOut> cold(ncold);
  const auto tc0 = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t ci = 0; ci < ncold; ++ci) {
    const SampledState &s = states[cold_idx[ci]];
    cold[ci] = con2prim(view, s.cin, opts.solver, nan, nan, nan);
  }
  const auto tc1 = std::chrono::steady_clock::now();
  const double cold_seconds = std::chrono::duration<double>(tc1 - tc0).count();
  report.solves_per_sec_cold = static_cast<double>(ncold) / std::max(cold_seconds, 1e-12);

  // --- Pass 4 (untimed): classify warm results, round-trip metrics,
  // histograms, per-thread accumulated then merged.
  const int nthreads = max_threads();
  const size_t nt = static_cast<size_t>(nthreads);

  std::vector<Accum> local_failed(nt, Accum(opts.worst_n));
  std::vector<Accum> local_roundtrip(nt, Accum(opts.worst_n));
  std::vector<std::vector<double>> local_rt_D(nt), local_rt_tau(nt), local_rt_S(nt);
  std::vector<std::vector<double>> local_prim_rho(nt), local_prim_w(nt), local_prim_s(nt);
  std::vector<size_t> local_n_newton(nt, 0), local_n_fallback(nt, 0), local_n_no_bracket(nt, 0),
      local_n_max_iter(nt, 0);
  std::vector<std::vector<size_t>> local_hist_warm(nt, std::vector<size_t>(64, 0));
  std::vector<char> local_fatal(nt, 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t k = 0; k < n; ++k) {
    const size_t tid = static_cast<size_t>(this_thread());
    const SampledState &s = states[k];
    const Con2PrimOut &rec = warm[k];

    switch (rec.result) {
      case C2PResult::converged_newton: ++local_n_newton[tid]; break;
      case C2PResult::converged_fallback: ++local_n_fallback[tid]; break;
      case C2PResult::failed_no_bracket: ++local_n_no_bracket[tid]; break;
      case C2PResult::failed_max_iter: ++local_n_max_iter[tid]; break;
    }

    const int total_iters = rec.iters_newton + rec.iters_fallback;
    const size_t hb = static_cast<size_t>(std::min(std::max(total_iters, 0), 63));
    ++local_hist_warm[tid][hb];

    const bool failed =
        (rec.result == C2PResult::failed_no_bracket || rec.result == C2PResult::failed_max_iter);

    Loc loc0;
    loc0.rho = s.rho;
    loc0.temp = s.T_input;
    loc0.ye = s.ye;

    {
      // "c2p_failed": Loc::value carries the sampled w (no dedicated
      // rapidity field on CheckClassResult::Loc -- see the header's doc
      // comment).
      Loc l = loc0;
      l.value = s.w;
      local_failed[tid].add_violation(s.w, failed, l);
    }

    if (!c2p_out_finite(rec)) {
      local_fatal[tid] = 1;
      continue; // excluded from round-trip/prim-space stats, per the header's doc comment
    }

    // Round trip: re-run prim2con() on the RECOVERED state (rho,s,ye,w),
    // the SAME B2, and cos_vB reconstructed sign-correctly from V =
    // sqrt(v_par^2+v_perp^2) -- falling back to the original sampled
    // cos_vB when V is degenerately small (matching prim2con.hpp's own
    // note that cos_vB is irrelevant whenever B2==0 or w==0, i.e. exactly
    // when V->0).
    const double V = std::sqrt(rec.v_par * rec.v_par + rec.v_perp * rec.v_perp);
    const double cos_vB_rec = V > 1e-10 ? rec.v_par / V : s.cos_vB;
    const Prim2ConOut back =
        prim2con(view, rec.rho, rec.s, rec.ye, rec.w, s.B2, cos_vB_rec, rec.eos.u_solved);

    if (!prim2con_out_finite(back)) {
      local_fatal[tid] = 1;
      continue;
    }

    const double floor = 1e-30;
    const double rt_D = std::fabs(back.D - s.truth.D) / std::max(std::fabs(s.truth.D), floor);
    const double rt_tau = std::fabs(back.tau - s.truth.tau) / std::max(std::fabs(s.truth.tau), floor);
    const double rt_Spar =
        std::fabs(back.S_par - s.truth.S_par) / std::max(std::fabs(s.truth.S_par), floor);
    const double rt_Sperp =
        std::fabs(back.S_perp - s.truth.S_perp) / std::max(std::fabs(s.truth.S_perp), floor);
    const double rt_S = std::max(rt_Spar, rt_Sperp);

    local_rt_D[tid].push_back(rt_D);
    local_rt_tau[tid].push_back(rt_tau);
    local_rt_S[tid].push_back(rt_S);

    const double prim_rho = std::fabs(rec.rho - s.rho) / std::max(s.rho, floor);
    const double prim_w = std::fabs(rec.w - s.w) / (1.0 + s.w);
    const double prim_s = std::fabs(rec.s - s.s) / std::max(std::fabs(s.s), 1.0);
    local_prim_rho[tid].push_back(prim_rho);
    local_prim_w[tid].push_back(prim_w);
    local_prim_s[tid].push_back(prim_s);

    const double rt_max = std::max(rt_D, std::max(rt_tau, rt_S));
    Loc lr = loc0;
    lr.value = rt_max;
    local_roundtrip[tid].add_violation(rt_max, rt_max > opts.tol_roundtrip, lr);
  }

  Accum acc_failed(opts.worst_n), acc_roundtrip(opts.worst_n);
  std::vector<double> all_rt_D, all_rt_tau, all_rt_S, all_prim_rho, all_prim_w, all_prim_s;
  bool any_fatal = false;
  for (size_t t = 0; t < nt; ++t) {
    acc_failed.merge_from(local_failed[t]);
    acc_roundtrip.merge_from(local_roundtrip[t]);
    all_rt_D.insert(all_rt_D.end(), local_rt_D[t].begin(), local_rt_D[t].end());
    all_rt_tau.insert(all_rt_tau.end(), local_rt_tau[t].begin(), local_rt_tau[t].end());
    all_rt_S.insert(all_rt_S.end(), local_rt_S[t].begin(), local_rt_S[t].end());
    all_prim_rho.insert(all_prim_rho.end(), local_prim_rho[t].begin(), local_prim_rho[t].end());
    all_prim_w.insert(all_prim_w.end(), local_prim_w[t].begin(), local_prim_w[t].end());
    all_prim_s.insert(all_prim_s.end(), local_prim_s[t].begin(), local_prim_s[t].end());
    report.n_newton += local_n_newton[t];
    report.n_fallback += local_n_fallback[t];
    report.n_failed_no_bracket += local_n_no_bracket[t];
    report.n_failed_max_iter += local_n_max_iter[t];
    for (size_t i = 0; i < 64; ++i) report.iters_hist_warm[i] += local_hist_warm[t][i];
    if (local_fatal[t]) any_fatal = true;
  }

  report.classes.push_back(acc_failed.finalize("c2p_failed"));
  report.classes.push_back(acc_roundtrip.finalize("c2p_roundtrip"));

  report.rt_D = compute_quantiles(all_rt_D);
  report.rt_tau = compute_quantiles(all_rt_tau);
  report.rt_S = compute_quantiles(all_rt_S);
  report.prim_rho = compute_quantiles(all_prim_rho);
  report.prim_w = compute_quantiles(all_prim_w);
  report.prim_s = compute_quantiles(all_prim_s);

  // --- Cold-pass classification (results + histogram + fatal check only --
  // no round-trip/prim-space stats or worst-offender list, see the header's
  // doc comment).
  std::vector<size_t> local_cold_newton(nt, 0), local_cold_fallback(nt, 0), local_cold_failed(nt, 0);
  std::vector<std::vector<size_t>> local_hist_cold(nt, std::vector<size_t>(64, 0));
  std::vector<char> local_cold_fatal(nt, 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t ci = 0; ci < ncold; ++ci) {
    const size_t tid = static_cast<size_t>(this_thread());
    const Con2PrimOut &rec = cold[ci];
    if (!c2p_out_finite(rec)) {
      local_cold_fatal[tid] = 1;
      continue;
    }
    switch (rec.result) {
      case C2PResult::converged_newton: ++local_cold_newton[tid]; break;
      case C2PResult::converged_fallback: ++local_cold_fallback[tid]; break;
      case C2PResult::failed_no_bracket:
      case C2PResult::failed_max_iter: ++local_cold_failed[tid]; break;
    }
    const int total_iters = rec.iters_newton + rec.iters_fallback;
    const size_t hb = static_cast<size_t>(std::min(std::max(total_iters, 0), 63));
    ++local_hist_cold[tid][hb];
  }

  for (size_t t = 0; t < nt; ++t) {
    report.cold_n_newton += local_cold_newton[t];
    report.cold_n_fallback += local_cold_fallback[t];
    report.cold_n_failed += local_cold_failed[t];
    for (size_t i = 0; i < 64; ++i) report.iters_hist_cold[i] += local_hist_cold[t][i];
    if (local_cold_fatal[t]) any_fatal = true;
  }

  report.status = any_fatal ? Status::fatal : Status::ok;
  return report;
}

bool con2prim_needs_attention(const Con2PrimReport &report) {
  if (report.n_failed_no_bracket > 0 || report.n_failed_max_iter > 0 || report.cold_n_failed > 0) {
    return true;
  }
  for (const CheckClassResult &c : report.classes) {
    if (c.name == "c2p_roundtrip" && c.count > 0) return true;
  }
  return false;
}

void Con2PrimReport::print(std::ostream &os) const {
  const std::ios::fmtflags saved_flags = os.flags();
  const std::streamsize saved_precision = os.precision();

  os << "check_con2prim report: status="
     << (status == Status::ok ? "ok" : status == Status::repaired ? "repaired" : "fatal")
     << " n_states=" << n_states << "\n";

  os << "warm: n_newton=" << n_newton << " n_fallback=" << n_fallback
     << " n_failed_no_bracket=" << n_failed_no_bracket << " n_failed_max_iter=" << n_failed_max_iter
     << " n_failed_total=" << (n_failed_no_bracket + n_failed_max_iter) << "\n";
  os << "cold: n_newton=" << cold_n_newton << " n_fallback=" << cold_n_fallback
     << " n_failed=" << cold_n_failed << "\n";

  os << "iters_hist_warm:\n";
  for (size_t i = 0; i < 64; ++i) {
    if (iters_hist_warm[i] > 0) os << "    iters=" << i << ": " << iters_hist_warm[i] << "\n";
  }
  os << "iters_hist_cold:\n";
  for (size_t i = 0; i < 64; ++i) {
    if (iters_hist_cold[i] > 0) os << "    iters=" << i << ": " << iters_hist_cold[i] << "\n";
  }

  auto print_q = [&](const char *name, const QuantileStats &q) {
    os << name << " quantiles: p50=" << std::scientific << std::setprecision(6) << q.p50
       << " p90=" << q.p90 << " p99=" << q.p99 << " p999=" << q.p999 << " max=" << q.max << "\n";
  };
  os << "\n";
  print_q("rt_D", rt_D);
  print_q("rt_tau", rt_tau);
  print_q("rt_S", rt_S);
  print_q("prim_rho", prim_rho);
  print_q("prim_w", prim_w);
  print_q("prim_s", prim_s);

  for (const CheckClassResult &c : classes) {
    os << "\n" << c.name << ": count=" << c.count << " max=" << std::scientific << std::setprecision(6)
       << c.max << " rms=" << c.rms << "\n";
    if (!c.worst.empty()) {
      const char *label = (c.name == "c2p_failed") ? "w" : "value";
      os << "  worst offenders (rho [g/cc], T [MeV], Ye : " << label << "):\n";
      for (const CheckClassResult::Loc &loc : c.worst) {
        os << "    rho=" << std::scientific << std::setprecision(6) << loc.rho << " T=" << loc.temp
           << " Ye=" << std::fixed << std::setprecision(4) << loc.ye << " : " << label << "="
           << std::scientific << std::setprecision(6) << loc.value << "\n";
      }
    }
  }

  os << "\nsolves_per_sec_warm=" << std::scientific << std::setprecision(6) << solves_per_sec_warm
     << " solves_per_sec_cold=" << solves_per_sec_cold << "\n";

  os.flags(saved_flags);
  os.precision(saved_precision);
}

} // namespace eeos
