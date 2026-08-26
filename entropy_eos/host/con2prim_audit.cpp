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

// --- M3e policy section helpers (core/state_policy.hpp) -------------------

// Re-solves `back` with the PLAIN solver and returns the conservative-space
// round-trip error of its answer. `ok` reports whether the re-solve converged
// to a finite state at all; the returned error is +Inf when it did not.
//
// The forward map here is the SOLVER'S OWN -- (D, tau, S_par, S_perp) rebuilt
// from the recovered (rho, w, v_par, v_perp, eos) by the design doc's S4/S5/S7
// formulas -- rather than a prim2con() call with cos_vB reconstructed from
// v_par/V, which is what pass 4 below uses for the sampled warm states. The
// reason is specific to the states this section feeds it: an excised or
// ceiling-projected state can have V of order 1e-12 (a collapse state with
// almost no momentum relative to the ceiling inertia), and there the
// reconstruction V > 1e-10 ? v_par/V : 0 degenerates exactly as
// prim2con.hpp's own note says it must -- it puts all of the (negligible)
// momentum into S_perp and none into S_par, which shows up as a RELATIVE
// S_par error of 1 even though both values are numerically nothing. (Measured
// on LS220: this alone failed the "collapse_ceiling_D_only" battery case.)
// The solver's own forward map has no such degeneracy, and is exactly the
// construction tests/test_con2prim.cpp's run_one() uses, for the same reason.
//
// The per-component denominators carry the same D-scaled floor as
// tests/test_con2prim.cpp's run_one(): a momentum component that is genuinely
// negligible next to D must not dominate the metric.
double policy_resolve_err(const EntropyEOSView &view, const Con2PrimOptions &solver,
                          const Con2PrimIn &back, double s_g, double w_g, double u_g, bool &ok) {
  const double inf = std::numeric_limits<double>::infinity();
  const Con2PrimOut re = con2prim(view, back, solver, s_g, w_g, u_g);
  ok = (re.result == C2PResult::converged_newton || re.result == C2PResult::converged_fallback) &&
       c2p_out_finite(re);
  if (!ok) return inf;

  const double W = re.W;
  const double D = re.rho * W;
  const double z = re.rho * re.eos.h * W * W;
  const double v2 = re.v_par * re.v_par + re.v_perp * re.v_perp;
  const double half_sinh = std::sinh(0.5 * re.w);
  const double sinh_w = std::sinh(re.w);
  const double tau = 2.0 * D * half_sinh * half_sinh + re.rho * re.eos.U * W * W +
                     re.eos.p * sinh_w * sinh_w + 0.5 * back.B2 * (1.0 + v2) -
                     0.5 * back.B2 * re.v_par * re.v_par;
  const Prim2ConOut fw{D, tau, D * re.ye, z * re.v_par, (z + back.B2) * re.v_perp, back.B2};
  if (!prim2con_out_finite(fw)) {
    ok = false;
    return inf;
  }

  const double scale = 1e-12 * std::max(std::fabs(back.D), 1e-300);
  double m = 0.0;
  m = std::max(m, std::fabs(fw.D - back.D) / std::max(std::fabs(back.D), scale));
  m = std::max(m, std::fabs(fw.tau - back.tau) / std::max(std::fabs(back.tau), scale));
  m = std::max(m, std::fabs(fw.S_par - back.S_par) / std::max(std::fabs(back.S_par), scale));
  m = std::max(m, std::fabs(fw.S_perp - back.S_perp) / std::max(std::fabs(back.S_perp), scale));
  return m;
}

// "Is this con2prim_safe() output acceptable?" -- the never-fails contract,
// checked mechanically.
struct PolicyJudge {
  bool valid = false;   // finite, D > 0, and the returned primitives are policy-valid
  bool flagged = false; // the policy actually reported what it did
  bool warm_ok = false; // the returned conservatives re-solve when warm-started
  bool cold_ok = false; // ... and when cold-started (diagnostic only, see the header)
  double warm_err = 0.0;
};

PolicyJudge judge_policy_output(const EntropyEOSView &view, const PolicyOptions &pol,
                                const Con2PrimOptions &solver, const Con2PrimSafeOut &so) {
  PolicyJudge j;
  j.flagged = so.policy_flags != 0u;

  const PrimState ps{so.base.rho, so.base.s, so.base.ye, so.base.w};
  j.valid = c2p_out_finite(so.base) && prim2con_out_finite(so.cons) && so.cons.D > 0.0 &&
            check_prim_state(view, ps, pol) == 0u;

  const Con2PrimIn back{so.cons.D, so.cons.tau, so.cons.D_Y, so.cons.S_par, so.cons.S_perp, so.cons.B2};
  j.warm_err = policy_resolve_err(view, solver, back, so.base.s, so.base.w, so.base.eos.u_solved, j.warm_ok);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  bool cold_ok = false;
  (void)policy_resolve_err(view, solver, back, nan, nan, nan, cold_ok);
  j.cold_ok = cold_ok;
  return j;
}

// One deterministic broken-state battery case.
struct PolicyBatteryCase {
  const char *name;
  Con2PrimIn cin;
  PolicyOptions pol;
};

// The fixed broken-state battery: the con2prim-entropy-rapidity.md S11
// taxonomy, one case per failure mode, built from the view and the audit's
// own PolicyOptions so it scales with whatever table is being audited.
// Deterministic by construction (no PRNG).
std::vector<PolicyBatteryCase> make_policy_battery(const EntropyEOSView &view, const PolicyOptions &pol) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();

  const double ye_mid = 0.5 * (view.y_lo + view.y_hi);
  const double rho_mid = std::pow(10.0, 0.5 * (view.x_lo + view.x_hi));
  const SRange sr = view.srange(rho_mid, ye_mid);
  const double s_mid = 0.5 * (sr.s_min + sr.s_max);
  const EOSPoint pt = view.evaluate(rho_mid, s_mid, ye_mid, nan);

  // A genuinely valid, magnetized base state -- the cases that plant a
  // single defect start from this, so that the defect is the only thing
  // wrong.
  const Prim2ConOut base =
      prim2con(view, rho_mid, s_mid, ye_mid, 2.0, 3.0 * rho_mid * pt.h, 0.4, pt.u_solved);
  const Con2PrimIn base_in{base.D, base.tau, base.D_Y, base.S_par, base.S_perp, base.B2};

  PolicyOptions pol_ceil = pol;
  pol_ceil.collapse_to_atmosphere = false;

  // w_cap tightened WITHOUT re-deriving D_max/tau_max: those are derived
  // FROM w_cap (tau_max ~ cosh(w_cap)^2), so re-deriving them here would
  // classify the w = 5 state below as a step-1 collapse and the w-cap path
  // would never be exercised. See core/state_policy.hpp.
  PolicyOptions pol_cap = pol;
  pol_cap.w_cap = std::acosh(10.0);

  const double D_atm = 0.5 * pol.rho_atm;
  const double D_atm_Y = D_atm * ye_mid;

  std::vector<PolicyBatteryCase> out;

  // (1) Vacuum / sub-atmosphere, including garbage tau and momentum.
  out.push_back({"atm_vacuum", Con2PrimIn{D_atm, 0.0, D_atm_Y, 0.0, 0.0, 0.0}, pol});
  out.push_back({"atm_tau_absurd", Con2PrimIn{D_atm, 1e30, D_atm_Y, 1e20, 1e20, 0.0}, pol});
  out.push_back(
      {"atm_tau_negative", Con2PrimIn{D_atm, -1e10 * D_atm, D_atm_Y, 1e3 * D_atm, 1e3 * D_atm, 0.0}, pol});
  out.push_back({"atm_magnetized",
                 Con2PrimIn{D_atm, 1e3 * D_atm, D_atm_Y, 1e2 * D_atm, 1e2 * D_atm, 1e6 * D_atm}, pol});
  out.push_back({"atm_tiny_D", Con2PrimIn{1e-30 * D_atm, 0.0, 0.0, 0.0, 0.0, 0.0}, pol});

  // (2) Collapse: D and/or tau growing without bound, both policies.
  const Con2PrimIn col_both{1e6 * pol.D_max, 1e6 * pol.tau_max, 1e6 * pol.D_max * ye_mid, 0.0, 0.0, 0.0};
  const Con2PrimIn col_mom{1e6 * pol.D_max, 1e6 * pol.tau_max, 1e6 * pol.D_max * ye_mid, 1e30, 1e30, 0.0};
  const Con2PrimIn col_tau{rho_mid, 1e6 * pol.tau_max, rho_mid * ye_mid, 1e20, 1e20, 0.0};
  const Con2PrimIn col_D{1e6 * pol.D_max, base.tau, 1e6 * pol.D_max * ye_mid, 1e10, 1e10, 0.0};
  out.push_back({"collapse_atm_D_and_tau", col_both, pol});
  out.push_back({"collapse_atm_superluminal", col_mom, pol});
  out.push_back({"collapse_atm_tau_only", col_tau, pol});
  out.push_back({"collapse_atm_D_only", col_D, pol});
  out.push_back({"collapse_ceiling_D_and_tau", col_both, pol_ceil});
  out.push_back({"collapse_ceiling_superluminal", col_mom, pol_ceil});
  out.push_back({"collapse_ceiling_tau_only", col_tau, pol_ceil});
  out.push_back({"collapse_ceiling_D_only", col_D, pol_ceil});

  // (3) Non-finite: NaN and Inf planted in each conservative in turn.
  static const char *const field_names[12] = {
      "nonfinite_nan_D",     "nonfinite_nan_tau",   "nonfinite_nan_D_Y",   "nonfinite_nan_S_par",
      "nonfinite_nan_S_perp", "nonfinite_nan_B2",   "nonfinite_inf_D",     "nonfinite_inf_tau",
      "nonfinite_inf_D_Y",   "nonfinite_inf_S_par", "nonfinite_inf_S_perp", "nonfinite_inf_B2"};
  for (int bv = 0; bv < 2; ++bv) {
    for (int field = 0; field < 6; ++field) {
      Con2PrimIn cin = base_in;
      double *slots[6] = {&cin.D, &cin.tau, &cin.D_Y, &cin.S_par, &cin.S_perp, &cin.B2};
      *slots[field] = (bv == 0) ? nan : inf;
      out.push_back({field_names[bv * 6 + field], cin, pol});
    }
  }

  // (4) Broken-but-finite input: D <= 0, and a negative B^2 (a squared norm
  // cannot be negative -- core/state_policy.hpp step 0).
  {
    Con2PrimIn cin = base_in;
    cin.D = 0.0;
    out.push_back({"nonpositive_D_zero", cin, pol});
    cin.D = -base.D;
    out.push_back({"nonpositive_D_negative", cin, pol});
    cin = base_in;
    cin.B2 = -base.B2;
    out.push_back({"negative_B2", cin, pol});
  }

  // (5) tau below the coldest state this (D, |S|) can express.
  {
    Con2PrimIn cin = base_in;
    cin.tau = base.tau / 10.0;
    out.push_back({"tau_below_cold_floor", cin, pol});
  }

  // (6) A rapidity above the cap (a legitimate state under a tighter cap).
  {
    const Prim2ConOut t5 = prim2con(view, rho_mid, s_mid, ye_mid, 5.0, 0.0, 0.3, pt.u_solved);
    out.push_back({"w_above_cap",
                   Con2PrimIn{t5.D, t5.tau, t5.D_Y, t5.S_par, t5.S_perp, t5.B2}, pol_cap});
  }

  // (7) Ye outside the table range (Ye = D_Y/D is exact, so this is purely a
  // D_Y defect).
  {
    Con2PrimIn cin = base_in;
    cin.D_Y = 0.0;
    out.push_back({"ye_below_range", cin, pol});
    cin.D_Y = base.D * (view.y_hi + 0.2);
    out.push_back({"ye_above_range", cin, pol});
  }

  return out;
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
  std::vector<size_t> local_cold_newton(nt, 0), local_cold_fallback(nt, 0);
  std::vector<size_t> local_cold_no_bracket(nt, 0), local_cold_max_iter(nt, 0);
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
      case C2PResult::failed_no_bracket: ++local_cold_no_bracket[tid]; break;
      case C2PResult::failed_max_iter: ++local_cold_max_iter[tid]; break;
    }
    const int total_iters = rec.iters_newton + rec.iters_fallback;
    const size_t hb = static_cast<size_t>(std::min(std::max(total_iters, 0), 63));
    ++local_hist_cold[tid][hb];
  }

  for (size_t t = 0; t < nt; ++t) {
    report.cold_n_newton += local_cold_newton[t];
    report.cold_n_fallback += local_cold_fallback[t];
    report.cold_n_failed_no_bracket += local_cold_no_bracket[t];
    report.cold_n_failed_max_iter += local_cold_max_iter[t];
    for (size_t i = 0; i < 64; ++i) report.iters_hist_cold[i] += local_hist_cold[t][i];
    if (local_cold_fatal[t]) any_fatal = true;
  }

  // --- M3e policy section (con2prim_audit.hpp's "M3e POLICY SECTION") -----
  if (opts.policy) {
    report.policy_ran = true;

    // rho_atm: half the smallest D the sampled set contains, so no valid
    // sampled state can trip the atmosphere trigger (a caller-supplied value
    // is lowered to that when it is larger -- see the header).
    double min_D = std::numeric_limits<double>::infinity();
    for (size_t k = 0; k < n; ++k) min_D = std::min(min_D, static_cast<double>(states[k].cin.D));
    double rho_atm = 0.5 * min_D;
    if (!std::isfinite(rho_atm) || rho_atm <= 0.0) rho_atm = std::pow(10.0, view.x_lo);
    if (std::isfinite(opts.policy_rho_atm) && opts.policy_rho_atm > 0.0) {
      rho_atm = std::min(rho_atm, opts.policy_rho_atm);
    }

    // w_cap: above the sampled rapidity range, so neither the cap nor the
    // D_max/tau_max bounds DERIVED from it can fire on a valid sampled state.
    double w_cap = std::max(std::acosh(100.0), opts.w_max_sample + 0.5);
    if (std::isfinite(opts.policy_w_cap) && opts.policy_w_cap > 0.0) {
      w_cap = std::max(w_cap, opts.policy_w_cap);
    }
    w_cap = std::min(w_cap, 0.9 * opts.solver.w_max);

    PolicyOptions pol = default_policy(view, rho_atm);
    pol.w_cap = w_cap;
    policy_derive_bounds(view, pol);
    report.policy_rho_atm_used = pol.rho_atm;
    report.policy_w_cap_used = pol.w_cap;

    // Pass A (TIMED): con2prim_safe() over the SAME warm state set, with the
    // same warm-start guesses, so this is directly comparable to
    // solves_per_sec_warm.
    std::vector<Con2PrimSafeOut> safe(n);
    const auto tp0 = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t k = 0; k < n; ++k) {
      const SampledState &s = states[k];
      safe[k] = con2prim_safe(view, s.cin, opts.solver, pol, s.s_guess, s.w_guess, s.u_guess);
    }
    const auto tp1 = std::chrono::steady_clock::now();
    const double policy_seconds = std::chrono::duration<double>(tp1 - tp0).count();
    report.solves_per_sec_policy = static_cast<double>(n) / std::max(policy_seconds, 1e-12);

    // Pass B (untimed): classify.
    std::vector<size_t> loc_interv(nt, 0), loc_valid_touched(nt, 0), loc_mismatch(nt, 0),
        loc_invalid(nt, 0);
    std::vector<std::vector<size_t>> loc_flags(nt, std::vector<size_t>(8, 0));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t k = 0; k < n; ++k) {
      const size_t tid = static_cast<size_t>(this_thread());
      const SampledState &st = states[k];
      const Con2PrimSafeOut &so = safe[k];

      // The returned state must ALWAYS be valid -- the never-fails contract.
      const PrimState out_ps{so.base.rho, so.base.s, so.base.ye, so.base.w};
      if (!(c2p_out_finite(so.base) && prim2con_out_finite(so.cons) && so.cons.D > 0.0 &&
            check_prim_state(view, out_ps, pol) == 0u)) {
        ++loc_invalid[tid];
      }

      if (so.policy_flags == 0u) {
        // The no-touch path is contractually a verbatim copy of the input.
        if (!(so.cons.D == st.cin.D && so.cons.tau == st.cin.tau && so.cons.D_Y == st.cin.D_Y &&
              so.cons.S_par == st.cin.S_par && so.cons.S_perp == st.cin.S_perp &&
              so.cons.B2 == st.cin.B2)) {
          ++loc_mismatch[tid];
        }
        continue;
      }

      ++loc_interv[tid];
      for (int b = 0; b < 8; ++b) {
        if (so.policy_flags & (1u << (8 + b))) ++loc_flags[tid][static_cast<size_t>(b)];
      }

      // False positive? Only if the state was valid by construction AND the
      // plain solver handled it AND its answer was itself policy-valid --
      // i.e. the layer had nothing whatsoever to repair. See the header.
      const Con2PrimOut &plain = warm[k];
      const bool plain_converged = (plain.result == C2PResult::converged_newton ||
                                    plain.result == C2PResult::converged_fallback);
      const PrimState truth_ps{st.rho, st.s, st.ye, st.w};
      const PrimState solved_ps{plain.rho, plain.s, plain.ye, plain.w};
      if (plain_converged && c2p_out_finite(plain) && check_prim_state(view, truth_ps, pol) == 0u &&
          check_prim_state(view, solved_ps, pol) == 0u) {
        ++loc_valid_touched[tid];
      }
    }

    for (size_t t = 0; t < nt; ++t) {
      report.policy_n_interventions += loc_interv[t];
      report.policy_n_valid_touched += loc_valid_touched[t];
      report.policy_n_cons_mismatch += loc_mismatch[t];
      report.policy_n_invalid_out += loc_invalid[t];
      for (size_t b = 0; b < 8; ++b) report.policy_flag_counts[b] += loc_flags[t][b];
    }

    // Pass C: the deterministic broken-state battery.
    const std::vector<PolicyBatteryCase> battery = make_policy_battery(view, pol);
    report.policy_battery_n = battery.size();
    for (const PolicyBatteryCase &c : battery) {
      const Con2PrimSafeOut so = con2prim_safe(view, c.cin, opts.solver, c.pol);
      const PolicyJudge j = judge_policy_output(view, c.pol, opts.solver, so);
      if (!j.valid || !j.warm_ok || !(j.warm_err <= opts.policy_tol_resolve)) {
        ++report.policy_battery_n_invalid;
      }
      if (!j.flagged) ++report.policy_battery_n_no_flag;
      if (!j.cold_ok) ++report.policy_battery_n_cold_missed;
    }
  }

  report.status = any_fatal ? Status::fatal : Status::ok;
  return report;
}

bool con2prim_needs_attention(const Con2PrimReport &report) {
  if (report.n_failed_no_bracket > 0 || report.n_failed_max_iter > 0 ||
      report.cold_n_failed_no_bracket > 0 || report.cold_n_failed_max_iter > 0) {
    return true;
  }
  // M3e policy section (see con2prim_audit.hpp): a false positive, a
  // non-verbatim no-touch path, an invalid output, or a battery case that
  // came out invalid or unflagged. policy_n_interventions itself is NOT
  // consulted -- on a real table it legitimately absorbs the M3 solver's
  // documented residual failure tail.
  if (report.policy_n_valid_touched > 0 || report.policy_n_cons_mismatch > 0 ||
      report.policy_n_invalid_out > 0 || report.policy_battery_n_invalid > 0 ||
      report.policy_battery_n_no_flag > 0) {
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
     << " n_failed_no_bracket=" << cold_n_failed_no_bracket
     << " n_failed_max_iter=" << cold_n_failed_max_iter
     << " n_failed_total=" << (cold_n_failed_no_bracket + cold_n_failed_max_iter) << "\n";

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

  // --- M3e policy section (self-contained: the lines below say what each
  // counter means, so the report can be read without the header at hand).
  os << "\npolicy: " << (policy_ran ? "ran" : "skipped");
  if (policy_ran) {
    os << " rho_atm=" << std::scientific << std::setprecision(6) << policy_rho_atm_used
       << " w_cap=" << policy_w_cap_used;
  }
  os << "\n";
  if (policy_ran) {
    os << "policy warm-set: n_interventions=" << policy_n_interventions
       << " (a repair of any kind; on a real table this legitimately absorbs the solver's own "
          "failure tail)\n";
    os << "policy warm-set: n_valid_touched=" << policy_n_valid_touched
       << " (FALSE POSITIVES: the state was valid, the solver converged to a valid answer, and the "
          "policy still intervened -- must be 0)\n";
    os << "policy warm-set: n_cons_mismatch=" << policy_n_cons_mismatch
       << " (policy_flags==0 but the returned conservatives were not the input bit-identically -- "
          "must be 0)"
       << " n_invalid_out=" << policy_n_invalid_out
       << " (returned state not policy-valid/finite -- must be 0)\n";
    static const char *const flag_names[8] = {"atmosphere", "ceiling",     "s_floored",  "s_ceiled",
                                              "w_capped",   "rho_clamped", "ye_clamped", "nonfinite"};
    os << "policy flag counts:";
    for (size_t i = 0; i < 8; ++i) os << " " << flag_names[i] << "=" << policy_flag_counts[i];
    os << "\n";
    os << "policy battery: n=" << policy_battery_n << " n_invalid=" << policy_battery_n_invalid
       << " (must be 0) n_no_flag=" << policy_battery_n_no_flag << " (must be 0) n_cold_missed="
       << policy_battery_n_cold_missed
       << " (diagnostic only: a COLD re-solve of the repaired state missed it; the WARM re-solve is "
          "the contract)\n";
    os << "policy battery result: "
       << ((policy_battery_n > 0 && policy_battery_n_invalid == 0 && policy_battery_n_no_flag == 0)
               ? "PASS"
               : (policy_battery_n == 0 ? "EMPTY" : "FAIL"))
       << "\n";
  }

  os << "\nsolves_per_sec_warm=" << std::scientific << std::setprecision(6) << solves_per_sec_warm
     << " solves_per_sec_cold=" << solves_per_sec_cold
     << " solves_per_sec_policy=" << solves_per_sec_policy << "\n";

  os.flags(saved_flags);
  os.precision(saved_precision);
}

} // namespace eeos
