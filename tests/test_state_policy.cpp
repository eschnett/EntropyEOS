// tests/test_state_policy.cpp — unit tests for the M3e invalid-state policy
// layer: entropy_eos/core/state_policy.hpp (PolicyOptions/default_policy(),
// PrimState, check_prim_state()/project_prim_state(), check_con_state(),
// con2prim_safe()), exercised against the synthetic ground-truth gas
// (entropy_eos/host/synthetic.hpp) plus a guarded LS220 real-table pass.
// Patterns (build_synthetic(), InteriorSampler, the solver's-own-forward-map
// round trip, the guarded real-table smoke test, eeos_n() sanitizer scaling)
// follow tests/test_con2prim.cpp and tests/test_con2prim_audit.cpp -- see
// CODE.md "Test harness" / con2prim-entropy-rapidity.md S11.
//
// The layer's contract is that it NEVER fails: every test below therefore
// ends in the same two questions, asked through the shared helpers
// `check_policy_valid_output()` and `check_resolvable()`:
//   (a) are the returned primitives policy-valid and finite?
//   (b) are the returned conservatives EXACTLY solvable -- i.e. does the
//       PLAIN solver, re-run on them, converge and reproduce them? (Both
//       warm- and cold-started; see check_resolvable() for which is asserted
//       where and why.)
// (b) is the real acceptance bar: it is what makes "the caller adopts
// Con2PrimSafeOut::cons" a safe operation rather than a hope.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "entropy_eos/core/adapter_eval.hpp"
#include "entropy_eos/core/con2prim.hpp"
#include "entropy_eos/core/prim2con.hpp"
#include "entropy_eos/core/state_policy.hpp"
#include "entropy_eos/host/adapter_build.hpp"
#include "entropy_eos/host/io_stellarcollapse.hpp"
#include "entropy_eos/host/repair.hpp"
#include "entropy_eos/host/synthetic.hpp"
#include "entropy_eos/host/table.hpp"
#include "test_scale.hpp"

using eeos::C2PResult;
using eeos::Con2PrimIn;
using eeos::Con2PrimOptions;
using eeos::Con2PrimOut;
using eeos::Con2PrimSafeOut;
using eeos::EntropyEOS;
using eeos::EntropyEOSView;
using eeos::EOSPoint;
using eeos::PolicyOptions;
using eeos::Prim2ConOut;
using eeos::PrimState;
using eeos::RawTable;
using eeos::SRange;
using eeos::SyntheticOptions;

namespace {

double nan_guess() { return std::numeric_limits<double>::quiet_NaN(); }
double inf_val() { return std::numeric_limits<double>::infinity(); }

double rel_err(double got, double ref, double floor = 1e-300) {
  return std::fabs(got - ref) / std::max(std::fabs(ref), floor);
}

EntropyEOS build_synthetic(const SyntheticOptions &opts) {
  RawTable table = eeos::make_synthetic_table(opts);
  return eeos::build_entropy_eos(table);
}

bool cons_finite(const Prim2ConOut &c) {
  return std::isfinite(c.D) && std::isfinite(c.tau) && std::isfinite(c.D_Y) && std::isfinite(c.S_par) &&
         std::isfinite(c.S_perp) && std::isfinite(c.B2);
}

bool eos_point_finite(const EOSPoint &pt) {
  return std::isfinite(pt.U) && std::isfinite(pt.U_rho) && std::isfinite(pt.U_s) &&
         std::isfinite(pt.U_rhorho) && std::isfinite(pt.U_rhos) && std::isfinite(pt.p) &&
         std::isfinite(pt.h) && std::isfinite(pt.cs2) && std::isfinite(pt.T_F_MeV) &&
         std::isfinite(pt.u_solved);
}

bool prims_finite(const Con2PrimOut &o) {
  return std::isfinite(o.rho) && std::isfinite(o.s) && std::isfinite(o.ye) && std::isfinite(o.w) &&
         std::isfinite(o.W) && std::isfinite(o.v_par) && std::isfinite(o.v_perp) && eos_point_finite(o.eos);
}

bool cons_bit_identical(const Prim2ConOut &a, const Con2PrimIn &b) {
  return a.D == b.D && a.tau == b.tau && a.D_Y == b.D_Y && a.S_par == b.S_par && a.S_perp == b.S_perp &&
         a.B2 == b.B2;
}

Con2PrimIn as_in(const Prim2ConOut &c) { return Con2PrimIn{c.D, c.tau, c.D_Y, c.S_par, c.S_perp, c.B2}; }

// The solver's OWN forward map applied to its OWN answer -- identical
// construction to tests/test_con2prim.cpp's run_one() (see its comment: the
// residuals constrain V = tanh(w) and the energy balance, not that v_par/V
// reproduces any particular incoming direction, so comparing through the
// solver's own reported (rho, w, v_par, v_perp, eos) isolates genuine
// imprecision from a same-conservative-state alternate root).
Prim2ConOut forward_of(const Con2PrimOut &r, double B2) {
  const double W = r.W;
  const double D = r.rho * W;
  const double z = r.rho * r.eos.h * W * W;
  const double v2 = r.v_par * r.v_par + r.v_perp * r.v_perp;
  const double half_sinh = std::sinh(0.5 * r.w);
  const double sinh_w = std::sinh(r.w);
  const double tau = 2.0 * D * half_sinh * half_sinh + r.rho * r.eos.U * W * W +
                     r.eos.p * sinh_w * sinh_w + 0.5 * B2 * (1.0 + v2) - 0.5 * B2 * r.v_par * r.v_par;
  return Prim2ConOut{D, tau, D * r.ye, z * r.v_par, (z + B2) * r.v_perp, B2};
}

double roundtrip_err(const Prim2ConOut &back, const Con2PrimIn &ref) {
  double m = 0.0;
  m = std::max(m, rel_err(back.D, ref.D));
  m = std::max(m, rel_err(back.tau, ref.tau));
  m = std::max(m, rel_err(back.S_par, ref.S_par));
  m = std::max(m, rel_err(back.S_perp, ref.S_perp));
  return m;
}

// (a) The returned primitives are finite and policy-valid, and the returned
// conservatives are finite with D > 0. Every con2prim_safe() output must
// satisfy this, whatever the input was.
void check_policy_valid_output(const EntropyEOSView &view, const PolicyOptions &pol,
                               const Con2PrimSafeOut &so, const std::string &label) {
  INFO(label);
  CHECK(prims_finite(so.base));
  CHECK(cons_finite(so.cons));
  CHECK(so.cons.D > 0.0);
  const PrimState ps{so.base.rho, so.base.s, so.base.ye, so.base.w};
  CHECK(eeos::check_prim_state(view, ps, pol) == 0u);
}

// (b) The returned conservatives are EXACTLY solvable: the PLAIN solver run
// on them converges and its own forward map reproduces them.
//
// Two flavours, because they measure different things:
//   WARM -- seeded with the returned primitives themselves. This is the
//     production situation (the caller has them in hand as the next step's
//     guess) and it is the sharp test of THIS layer: it asks whether the
//     returned (prims, cons) pair really is a converged solution of the
//     residuals, with no dependence on the solver's cold-start machinery.
//     Asserted everywhere.
//   COLD -- no guesses at all. A strictly stronger statement, but it also
//     re-tests the M3d seed and the S9 bracket scan, which have a documented
//     residual failure tail on the REAL tables (CODE.md M3 open item (i)) in
//     exactly the extreme corners this layer projects into. Asserted on the
//     synthetic gas (where it holds outright) and only REPORTED on LS220 --
//     see test 8's own comment for the two measured cases.
struct ResolveOutcome {
  bool warm_ok = false, cold_ok = false;
  double warm_err = 0.0, cold_err = 0.0;
};

ResolveOutcome resolve_both(const EntropyEOSView &view, const Con2PrimOptions &copts,
                            const Con2PrimSafeOut &so) {
  const Con2PrimIn back_in = as_in(so.cons);
  ResolveOutcome r;
  const Con2PrimOut warm =
      eeos::con2prim(view, back_in, copts, so.base.s, so.base.w, so.base.eos.u_solved);
  r.warm_ok = (warm.result == C2PResult::converged_newton ||
               warm.result == C2PResult::converged_fallback) &&
              prims_finite(warm);
  r.warm_err = roundtrip_err(forward_of(warm, back_in.B2), back_in);
  const Con2PrimOut cold = eeos::con2prim(view, back_in, copts, nan_guess(), nan_guess(), nan_guess());
  r.cold_ok = (cold.result == C2PResult::converged_newton ||
               cold.result == C2PResult::converged_fallback) &&
              prims_finite(cold);
  r.cold_err = roundtrip_err(forward_of(cold, back_in.B2), back_in);
  return r;
}

ResolveOutcome check_resolvable(const EntropyEOSView &view, const Con2PrimOptions &copts,
                                const Con2PrimSafeOut &so, const std::string &label,
                                bool require_cold = true, double tol = 1e-10) {
  INFO(label);
  const ResolveOutcome r = resolve_both(view, copts, so);
  CHECK(r.warm_ok);
  CHECK(r.warm_err <= tol);
  if (require_cold) {
    CHECK(r.cold_ok);
    CHECK(r.cold_err <= tol);
  }
  return r;
}

// Interior sampler for (rho, Ye, s), same construction as
// tests/test_con2prim.cpp's.
struct InteriorSampler {
  std::mt19937 rng;
  std::uniform_real_distribution<double> xq, yq, frac;

  InteriorSampler(const EntropyEOSView &view, unsigned seed, double margin_frac = 0.05)
      : rng(seed), frac(0.0, 1.0) {
    const double xspan = view.x_hi - view.x_lo;
    xq = std::uniform_real_distribution<double>(view.x_lo + margin_frac * xspan,
                                                view.x_hi - margin_frac * xspan);
    const double yspan = view.y_hi - view.y_lo;
    yq = std::uniform_real_distribution<double>(view.y_lo + margin_frac * yspan,
                                                view.y_hi - margin_frac * yspan);
  }

  double rho() { return std::pow(10.0, xq(rng)); }
  double ye() { return yq(rng); }
  double s_in(const SRange &sr, double s_margin_frac) {
    const double span = sr.s_max - sr.s_min;
    const double m = s_margin_frac * span;
    return sr.s_min + m + frac(rng) * (span - 2.0 * m);
  }
};

// The policy used by every synthetic test below. rho_atm = the table's own
// physical density floor 10^x_lo, deliberately BELOW the samplers' range
// (which starts 5% of the x-span above x_lo), so that a valid sampled state
// can never trip the atmosphere trigger -- an "intervention" caused by the
// test's own sampling choice would tell us nothing about the policy.
PolicyOptions test_policy(const EntropyEOSView &view) {
  return eeos::default_policy(view, std::pow(10.0, view.x_lo));
}

// ==========================================================================
// The no-false-positive body, shared by test 1 (synthetic) and test 8
// (LS220): sample `npts` states that are valid BY CONSTRUCTION (prim2con of
// in-range primitives, the tests/test_con2prim.cpp pattern) and demand that
// the policy layer be transparent on them.
//
// On a real table the demand cannot be literally "zero interventions": the
// M3 solver has a documented residual failure tail in the hot-edge/acausal
// corner (CODE.md M3 open item (i)) and can, rarely, converge onto an
// alternate root outside the physical box -- and repairing exactly those is
// this layer's JOB, not a false positive. So the assertion is the sharper
// statement, which holds unconditionally:
//
//   if the plain solver converged AND its recovered primitives are
//   policy-valid, then con2prim_safe() must have done nothing at all
//   (policy_flags == 0, cons bit-identical, base identical field by field).
//
// A wrongly-derived threshold -- a tau_max that is too tight, an atmosphere
// trigger above the sampled range, a sign slip in the D/tau tests -- fires
// at step 0/1 BEFORE the solve and is therefore caught by this even though
// it is phrased conditionally. The unconditional intervention count is
// reported, and bounded loosely (1%) so a wholesale regression still fails.
// ==========================================================================
struct NoFalsePositiveResult {
  int n = 0, n_interventions = 0, n_solver_failed = 0;
  size_t flag_counts[8] = {};
};

NoFalsePositiveResult run_no_false_positives(const EntropyEOSView &view, unsigned seed, int npts,
                                             double w_sample_max) {
  const Con2PrimOptions copts;
  const PolicyOptions pol = test_policy(view);

  // The sampled rapidity must stay below the policy cap: a state with
  // w > w_cap is not a valid state AS FAR AS THE POLICY IS CONCERNED (that
  // is what the cap means), so sampling above it would be asking the test to
  // contradict itself. 5.0 < acosh(100) = 5.2983.
  REQUIRE(w_sample_max < pol.w_cap);

  InteriorSampler sampler(view, seed);
  std::uniform_real_distribution<double> wq(0.0, w_sample_max);
  std::uniform_real_distribution<double> cq(-1.0, 1.0);
  std::uniform_real_distribution<double> log_sigma(std::log(1e-6), std::log(1e4));

  NoFalsePositiveResult res;
  res.n = npts;

  for (int k = 0; k < npts; ++k) {
    const double rho = sampler.rho();
    const double ye = sampler.ye();
    const SRange sr = view.srange(rho, ye);
    const double s = sampler.s_in(sr, 0.05);
    const double w = wq(sampler.rng);
    const double cos_vB = cq(sampler.rng);

    const EOSPoint pt0 = view.evaluate(rho, s, ye, nan_guess());
    const double B2 = (k % 10 == 0) ? 0.0 : std::exp(log_sigma(sampler.rng)) * rho * pt0.h;

    const Prim2ConOut truth = eeos::prim2con(view, rho, s, ye, w, B2, cos_vB, pt0.u_solved);
    const Con2PrimIn cin = as_in(truth);

    // Evolution-like warm start (test_con2prim's pattern), so the common
    // production path is what gets exercised.
    const double s_guess = s * (1.0 + 1e-3);
    const double w_guess = w + 1e-3;

    // The truth primitives themselves must be policy-valid -- if they were
    // not, the sampling (not the policy) would be at fault.
    const PrimState truth_ps{rho, s, ye, w};
    CHECK(eeos::check_prim_state(view, truth_ps, pol) == 0u);

    const Con2PrimOut plain = eeos::con2prim(view, cin, copts, s_guess, w_guess, pt0.u_solved);
    const Con2PrimSafeOut safe =
        eeos::con2prim_safe(view, cin, copts, pol, s_guess, w_guess, pt0.u_solved);

    const bool plain_converged =
        (plain.result == C2PResult::converged_newton || plain.result == C2PResult::converged_fallback);
    if (!plain_converged) ++res.n_solver_failed;

    const PrimState solved_ps{plain.rho, plain.s, plain.ye, plain.w};
    const bool solved_valid = prims_finite(plain) && eeos::check_prim_state(view, solved_ps, pol) == 0u;

    if (safe.policy_flags != 0u) {
      ++res.n_interventions;
      for (int b = 0; b < 8; ++b) {
        if (safe.policy_flags & (1u << (8 + b))) ++res.flag_counts[static_cast<size_t>(b)];
      }
    }

    if (plain_converged && solved_valid) {
      // The sharp assertion (see this function's doc comment).
      CHECK(safe.policy_flags == 0u);
      CHECK(safe.solved);
      CHECK(cons_bit_identical(safe.cons, cin));
      CHECK(safe.base.rho == plain.rho);
      CHECK(safe.base.s == plain.s);
      CHECK(safe.base.ye == plain.ye);
      CHECK(safe.base.w == plain.w);
      CHECK(safe.base.W == plain.W);
      CHECK(safe.base.v_par == plain.v_par);
      CHECK(safe.base.v_perp == plain.v_perp);
      CHECK(safe.base.result == plain.result);
      CHECK(safe.base.iters_newton == plain.iters_newton);
      CHECK(safe.base.iters_fallback == plain.iters_fallback);
      CHECK(safe.base.flags == plain.flags);
      CHECK(safe.base.eos.U == plain.eos.U);
      CHECK(safe.base.eos.p == plain.eos.p);
      CHECK(safe.base.eos.h == plain.eos.h);
      CHECK(safe.base.eos.T_F_MeV == plain.eos.T_F_MeV);
    }

    // Whatever happened, the output must be valid.
    check_policy_valid_output(view, pol, safe, "no-false-positive state " + std::to_string(k));
  }

  return res;
}

void report_nfp(const std::string &label, const NoFalsePositiveResult &res) {
  std::cout << label << ": n=" << res.n << " interventions=" << res.n_interventions
            << " solver_failed=" << res.n_solver_failed << " flags[atm,ceil,s_lo,s_hi,wcap,rho,ye,nonfin]=[";
  for (int b = 0; b < 8; ++b) std::cout << res.flag_counts[static_cast<size_t>(b)] << (b < 7 ? " " : "");
  std::cout << "]\n";
}

} // namespace

// ==========================================================================
// 1. NO FALSE POSITIVES (the critical test): the policy layer is completely
//    transparent on valid states.
// ==========================================================================

TEST_CASE("state_policy: no false positives on 1000 (200 under sanitizers) valid random states") {
  SyntheticOptions opts; // default 40x30x10
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();

  const int npts = static_cast<int>(eeos_n(1000, 200));
  const NoFalsePositiveResult res = run_no_false_positives(view, 0x50117C00u, npts, 5.0);
  report_nfp("test_state_policy 1 (synthetic)", res);

  // The synthetic gas is smooth over the whole sampled interior, so here the
  // strong form holds outright: not one intervention, and not one solver
  // failure either.
  CHECK(res.n_solver_failed == 0);
  CHECK(res.n_interventions == 0);
}

// ==========================================================================
// 2. Atmosphere (vacuum / below-floor excision).
// ==========================================================================

TEST_CASE("state_policy: sub-atmosphere D is excised to the atmosphere spec and stays solvable") {
  SyntheticOptions opts;
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();
  const Con2PrimOptions copts;

  // rho_atm well inside the table here (10x the floor), so that "half the
  // atmosphere density" is still a representable density and the atmosphere
  // point itself sits comfortably in the box.
  const PolicyOptions pol = eeos::default_policy(view, 10.0 * std::pow(10.0, view.x_lo));
  const double D = 0.5 * pol.rho_atm;
  const double ye_in = 0.5 * (view.y_lo + view.y_hi);
  const PrimState atm = eeos::policy_atmosphere(view, pol, ye_in);

  struct Case {
    const char *name;
    double tau, S_par, S_perp, B2;
    bool nonfinite; // expect flag_pol_nonfinite too
  };
  const Case cases[] = {
      {"tau=0, S=0, B=0", 0.0, 0.0, 0.0, 0.0, false},
      {"tau tiny", 1e-8 * D, 0.0, 0.0, 0.0, false},
      {"tau absurd (1e30)", 1e30, 1e20, 1e20, 0.0, false},
      {"tau negative garbage", -1e5 * D, 1e3 * D, 1e3 * D, 0.0, false},
      {"huge B2", 1e3 * D, 1e2 * D, 1e2 * D, 1e6 * D, false},
      // A negative B^2 is corrupt data (a squared norm cannot be negative),
      // so it joins the step-0 broken-input class rather than being quietly
      // zeroed -- see con2prim_safe()'s decision tree.
      {"negative B2 (corrupt: a squared norm)", D, 0.0, 0.0, -7.0, true},
      {"garbage tau = NaN", nan_guess(), 0.0, 0.0, 0.0, true},
      {"garbage tau = +Inf", inf_val(), 0.0, 0.0, 0.0, true},
  };

  for (const Case &c : cases) {
    const Con2PrimIn cin{D, c.tau, D * ye_in, c.S_par, c.S_perp, c.B2};
    const Con2PrimSafeOut so = eeos::con2prim_safe(view, cin, copts, pol);
    INFO("atmosphere case: " << c.name);

    const unsigned expect = c.nonfinite ? (eeos::flag_pol_atmosphere | eeos::flag_pol_nonfinite)
                                        : eeos::flag_pol_atmosphere;
    CHECK(so.policy_flags == expect);
    CHECK_FALSE(so.solved); // steps 0-1 never reach the solver

    // Output prims are exactly the atmosphere spec.
    CHECK(so.base.rho == atm.rho);
    CHECK(so.base.s == atm.s);
    CHECK(so.base.ye == atm.ye);
    CHECK(so.base.w == atm.w);
    CHECK(so.base.v_par == 0.0);
    CHECK(so.base.v_perp == 0.0);

    // B^2 passes through when it is a field, and is zeroed when it is not.
    CHECK(so.cons.B2 == (c.B2 > 0.0 && std::isfinite(c.B2) ? c.B2 : 0.0));

    check_policy_valid_output(view, pol, so, std::string("atmosphere ") + c.name);
    check_resolvable(view, copts, so, std::string("atmosphere ") + c.name);
  }

  SUBCASE("an out-of-range incoming Ye is clamped into the table range") {
    for (double ye_bad : {-1.0, 0.0, 3.0}) {
      const Con2PrimIn cin{D, 0.0, D * ye_bad, 0.0, 0.0, 0.0};
      const Con2PrimSafeOut so = eeos::con2prim_safe(view, cin, copts, pol);
      INFO("atmosphere with incoming Ye = " << ye_bad);
      CHECK(so.policy_flags == eeos::flag_pol_atmosphere);
      CHECK(so.base.ye >= view.y_lo);
      CHECK(so.base.ye <= view.y_hi);
      check_policy_valid_output(view, pol, so, "atmosphere with bad Ye");
      check_resolvable(view, copts, so, "atmosphere with bad Ye");
    }
  }

  SUBCASE("a caller-pinned s_atm/ye_atm is honoured (and clamped into range)") {
    PolicyOptions p2 = pol;
    p2.ye_atm = 0.5 * (view.y_lo + view.y_hi);
    const SRange sr = view.srange(p2.rho_atm, p2.ye_atm);
    p2.s_atm = sr.s_min + 0.25 * (sr.s_max - sr.s_min);
    const Con2PrimIn cin{D, 0.0, D * 0.9 * view.y_hi, 0.0, 0.0, 0.0};
    Con2PrimSafeOut so = eeos::con2prim_safe(view, cin, copts, p2);
    CHECK(so.base.s == p2.s_atm);
    CHECK(so.base.ye == p2.ye_atm);
    check_policy_valid_output(view, pol, so, "pinned atmosphere");

    // An out-of-range s_atm is clamped rather than trusted, so the
    // atmosphere state is policy-valid whatever the caller passes.
    p2.s_atm = 1e30;
    so = eeos::con2prim_safe(view, cin, copts, p2);
    CHECK(so.base.s == view.srange(p2.rho_atm, p2.ye_atm).s_max);
    check_policy_valid_output(view, pol, so, "pinned atmosphere, s_atm out of range");
    check_resolvable(view, copts, so, "pinned atmosphere, s_atm out of range");
  }
}

// ==========================================================================
// 3. Collapse / hydro excision (D and tau growing without bound), both
//    collapse_to_atmosphere settings.
// ==========================================================================

TEST_CASE("state_policy: collapse states are excised (atmosphere) or projected (ceiling)") {
  SyntheticOptions opts;
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();
  const Con2PrimOptions copts;
  const PolicyOptions pol = test_policy(view);

  REQUIRE(pol.D_max > 0.0);
  REQUIRE(pol.tau_max > 0.0);

  const double ye_in = 0.5 * (view.y_lo + view.y_hi);

  struct Case {
    const char *name;
    double D, tau, S_par, S_perp, B2;
  };
  const double D_sane = std::pow(10.0, 0.5 * (view.x_lo + view.x_hi));
  const Case cases[] = {
      {"D and tau both 1e6x over", 1e6 * pol.D_max, 1e6 * pol.tau_max, 0.0, 0.0, 0.0},
      {"D and tau over, huge momentum", 1e6 * pol.D_max, 1e6 * pol.tau_max, 1e30, 1e30, 0.0},
      {"D and tau over, magnetized", 1e6 * pol.D_max, 1e6 * pol.tau_max, 1e25, 1e25, 1e20},
      {"tau alone 1e6x over, D sane", D_sane, 1e6 * pol.tau_max, 1e20, 1e20, 0.0},
      {"D alone 1e6x over", 1e6 * pol.D_max, 1e3 * D_sane, 1e10, 1e10, 0.0},
  };

  for (int mode = 0; mode < 2; ++mode) {
    PolicyOptions p = pol;
    p.collapse_to_atmosphere = (mode == 0);
    const PrimState atm = eeos::policy_atmosphere(view, p, ye_in);

    for (const Case &c : cases) {
      const Con2PrimIn cin{c.D, c.tau, c.D * ye_in, c.S_par, c.S_perp, c.B2};
      const Con2PrimSafeOut so = eeos::con2prim_safe(view, cin, copts, p);
      INFO("collapse case: " << c.name << " collapse_to_atmosphere=" << p.collapse_to_atmosphere);

      CHECK((so.policy_flags & eeos::flag_pol_ceiling) != 0u);
      CHECK_FALSE(so.solved);

      if (p.collapse_to_atmosphere) {
        CHECK((so.policy_flags & eeos::flag_pol_atmosphere) != 0u);
        CHECK(so.base.rho == atm.rho);
        CHECK(so.base.s == atm.s);
        CHECK(so.base.w == 0.0);
      } else {
        CHECK((so.policy_flags & eeos::flag_pol_atmosphere) == 0u);
        CHECK(so.base.rho == pol.rho_ceiling);
        CHECK(so.base.s == view.srange(pol.rho_ceiling, so.base.ye).s_max);
        CHECK(so.base.w >= 0.0);
        CHECK(so.base.w <= pol.w_cap);
        // A superluminal momentum demand saturates the cap and says so.
        if (so.base.w == pol.w_cap) CHECK((so.policy_flags & eeos::flag_pol_w_capped) != 0u);
      }

      check_policy_valid_output(view, p, so, std::string("collapse ") + c.name);
      check_resolvable(view, copts, so, std::string("collapse ") + c.name);
    }
  }

  SUBCASE("the atmosphere trigger wins over the collapse ceilings (step order)") {
    // D below the atmosphere trigger AND tau above tau_max: step 1 tests the
    // atmosphere first, so this is a pure atmosphere reset with no ceiling
    // bit -- the documented ordering.
    const Con2PrimIn cin{0.5 * pol.rho_atm, 1e6 * pol.tau_max, 0.5 * pol.rho_atm * ye_in, 0.0, 0.0, 0.0};
    const Con2PrimSafeOut so = eeos::con2prim_safe(view, cin, copts, pol);
    CHECK(so.policy_flags == eeos::flag_pol_atmosphere);
    check_policy_valid_output(view, pol, so, "atmosphere-before-ceiling");
  }

  SUBCASE("check_con_state agrees with con2prim_safe's step-0/1 verdicts, for free") {
    const Con2PrimIn over{1e6 * pol.D_max, 1e6 * pol.tau_max, 1e6 * pol.D_max * ye_in, 0.0, 0.0, 0.0};
    CHECK(eeos::check_con_state(view, over, pol) == eeos::flag_pol_ceiling);
    const Con2PrimIn low{0.5 * pol.rho_atm, 0.0, 0.5 * pol.rho_atm * ye_in, 0.0, 0.0, 0.0};
    CHECK(eeos::check_con_state(view, low, pol) == eeos::flag_pol_atmosphere);
    const Con2PrimIn bad{nan_guess(), 0.0, 0.0, 0.0, 0.0, 0.0};
    CHECK(eeos::check_con_state(view, bad, pol) ==
          (eeos::flag_pol_nonfinite | eeos::flag_pol_atmosphere));
    const Con2PrimIn nonpos{-1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    CHECK(eeos::check_con_state(view, nonpos, pol) ==
          (eeos::flag_pol_nonfinite | eeos::flag_pol_atmosphere));
  }
}

// ==========================================================================
// 4. Non-finite inputs: NaN and Inf planted in each conservative in turn.
// ==========================================================================

TEST_CASE("state_policy: NaN/Inf in any conservative is excised to atmosphere, all outputs finite") {
  SyntheticOptions opts;
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();
  const Con2PrimOptions copts;
  const PolicyOptions pol = test_policy(view);

  // A genuinely valid base state, so the ONLY thing wrong is the planted
  // value.
  const double rho = std::pow(10.0, 0.5 * (view.x_lo + view.x_hi));
  const double ye = 0.5 * (view.y_lo + view.y_hi);
  const SRange sr = view.srange(rho, ye);
  const double s = 0.5 * (sr.s_min + sr.s_max);
  const EOSPoint pt0 = view.evaluate(rho, s, ye, nan_guess());
  const Prim2ConOut truth = eeos::prim2con(view, rho, s, ye, 2.0, 3.0 * rho * pt0.h, 0.4, pt0.u_solved);
  const Con2PrimIn base = as_in(truth);

  const char *names[6] = {"D", "tau", "D_Y", "S_par", "S_perp", "B2"};
  const double bad_values[3] = {std::numeric_limits<double>::quiet_NaN(), inf_val(), -inf_val()};

  for (int field = 0; field < 6; ++field) {
    for (int bv = 0; bv < 3; ++bv) {
      Con2PrimIn cin = base;
      double *slots[6] = {&cin.D, &cin.tau, &cin.D_Y, &cin.S_par, &cin.S_perp, &cin.B2};
      *slots[field] = bad_values[bv];

      const Con2PrimSafeOut so = eeos::con2prim_safe(view, cin, copts, pol);
      INFO("planted " << (bv == 0 ? "NaN" : (bv == 1 ? "+Inf" : "-Inf")) << " in " << names[field]);

      CHECK((so.policy_flags & eeos::flag_pol_nonfinite) != 0u);
      CHECK((so.policy_flags & eeos::flag_pol_atmosphere) != 0u);
      CHECK_FALSE(so.solved);
      check_policy_valid_output(view, pol, so, std::string("nonfinite ") + names[field]);
      check_resolvable(view, copts, so, std::string("nonfinite ") + names[field]);
    }
  }

  SUBCASE("D <= 0 shares the non-finite verdict") {
    for (double D_bad : {0.0, -1.0, -1e30}) {
      Con2PrimIn cin = base;
      cin.D = D_bad;
      const Con2PrimSafeOut so = eeos::con2prim_safe(view, cin, copts, pol);
      INFO("D = " << D_bad);
      CHECK(so.policy_flags == (eeos::flag_pol_nonfinite | eeos::flag_pol_atmosphere));
      check_policy_valid_output(view, pol, so, "D <= 0");
      check_resolvable(view, copts, so, "D <= 0");
    }
  }
}

// ==========================================================================
// 5. tau below the cold floor -> s-floor projection.
// ==========================================================================

TEST_CASE("state_policy: tau shrunk 10x below the cold floor lands exactly on the physical s_min") {
  SyntheticOptions opts;
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();
  const Con2PrimOptions copts;
  const PolicyOptions pol = test_policy(view);

  InteriorSampler sampler(view, 0x7A0710F0u);
  std::uniform_real_distribution<double> wq(0.5, 3.0);
  std::uniform_real_distribution<double> cq(-1.0, 1.0);
  std::uniform_real_distribution<double> log_sigma(std::log(1e-6), std::log(1e3));

  const int npts = static_cast<int>(eeos_n(150, 40));
  int n_floored = 0;
  // check_con_state()'s opt-in endpoint diagnosis, measured on the same
  // states: it must be silent on the ORIGINAL (valid) state and must report
  // flag_pol_s_floored on the tau/10 one. This is the expensive branch (two
  // full inner w-solves), so this is the one place it is exercised.
  int n_ep_clean = 0, n_ep_floored = 0, n_ep_hot = 0;
  double max_rt = 0.0, max_s_err = 0.0;

  for (int k = 0; k < npts; ++k) {
    const double rho = sampler.rho();
    const double ye = sampler.ye();
    const SRange sr = view.srange(rho, ye);
    const double s = sampler.s_in(sr, 0.05);
    const double w = wq(sampler.rng);
    const double cos_vB = cq(sampler.rng);
    const EOSPoint pt0 = view.evaluate(rho, s, ye, nan_guess());
    const double B2 = (k % 3 == 0) ? 0.0 : std::exp(log_sigma(sampler.rng)) * rho * pt0.h;

    const Prim2ConOut truth = eeos::prim2con(view, rho, s, ye, w, B2, cos_vB, pt0.u_solved);
    const Con2PrimIn valid_in = as_in(truth);
    Con2PrimIn cin = valid_in;
    cin.tau = truth.tau / 10.0; // below the coldest state this (D, S) can support

    // The cheap diagnosis must be silent on all three (none of them trips a
    // finiteness/D/tau bound), and the endpoint diagnosis must separate them.
    CHECK(eeos::check_con_state(view, valid_in, pol) == 0u);
    CHECK(eeos::check_con_state(view, cin, pol) == 0u);
    if (eeos::check_con_state(view, valid_in, pol, /*check_endpoints=*/true, copts) == 0u) ++n_ep_clean;
    if (eeos::check_con_state(view, cin, pol, /*check_endpoints=*/true, copts) ==
        eeos::flag_pol_s_floored) {
      ++n_ep_floored;
    }
    {
      Con2PrimIn hot = valid_in;
      hot.tau = truth.tau * 1e6;
      if (eeos::check_con_state(view, hot, pol, /*check_endpoints=*/true, copts) &
          eeos::flag_pol_ceiling) {
        ++n_ep_hot;
      }
    }

    const Con2PrimSafeOut so = eeos::con2prim_safe(view, cin, copts, pol);
    INFO("tau/10 state " << k);

    if (so.policy_flags & eeos::flag_pol_s_floored) {
      ++n_floored;
      // The recovered s is EXACTLY the physical floor at its own (rho, Ye)
      // -- the branch sets it there by construction rather than near it.
      const SRange sr_out = view.srange(so.base.rho, so.base.ye);
      const double s_err = rel_err(so.base.s, sr_out.s_min, 1.0);
      max_s_err = std::max(max_s_err, s_err);
      CHECK(s_err <= 1e-12);
    }
    check_policy_valid_output(view, pol, so, "tau/10");
    max_rt = std::max(max_rt, check_resolvable(view, copts, so, "tau/10").cold_err);
  }

  std::cout << "test_state_policy 5: " << n_floored << "/" << npts
            << " states took the s-floor branch; max |s - s_min|/max(|s_min|,1) = " << max_s_err
            << " max re-solve round trip = " << max_rt << "\n";
  std::cout << "test_state_policy 5: check_con_state endpoint diagnosis: " << n_ep_clean << "/" << npts
            << " silent on the valid state, " << n_ep_floored << "/" << npts << " s_floored on tau/10, "
            << n_ep_hot << "/" << npts << " ceiling on tau*1e6\n";
  CHECK(n_ep_clean == npts);
  CHECK(n_ep_floored == npts);
  CHECK(n_ep_hot == npts);
  // Measured: 150/150 on this sampler (and 200/200 at every entropy
  // percentile probed, s in {1%, 10%, 50%} of the srange). Dividing tau by
  // 10 removes far more energy than the coldest adiabat at this (D, |S|) can
  // give up, so the cold-floor diagnosis is not a marginal call here.
  CHECK(n_floored == npts);
}

// ==========================================================================
// 6. w cap: a legitimate high-rapidity state under a tightened cap.
// ==========================================================================

TEST_CASE("state_policy: a solved w above w_cap is capped, D is preserved, and the result re-solves") {
  SyntheticOptions opts;
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();
  const Con2PrimOptions copts;

  // w_cap = acosh(10) (W <= 10), so the w = 5 states below are genuinely
  // over it. D_max/tau_max are DELIBERATELY left at the wide default_policy
  // values (which were derived from w_cap = acosh(100)): both bounds are
  // derived FROM w_cap -- tau_max ~ cosh(w_cap)^2 -- so re-deriving them at
  // the tightened cap would classify these very states as collapse at step 1
  // and the w-cap path would never be exercised at all (measured on this
  // synthetic gas: 100/100 states excised to atmosphere instead). Tightening
  // only the cap is a coherent production configuration in its own right:
  // "the table may hold states this energetic, but the evolution must not be
  // asked to carry a Lorentz factor above 10".
  PolicyOptions pol = test_policy(view);
  const double D_max_wide = pol.D_max, tau_max_wide = pol.tau_max;
  pol.w_cap = std::acosh(10.0);
  REQUIRE(pol.w_cap < copts.w_max);
  REQUIRE(pol.D_max == D_max_wide);
  REQUIRE(pol.tau_max == tau_max_wide);

  InteriorSampler sampler(view, 0x3CA93E00u);
  std::uniform_real_distribution<double> cq(-1.0, 1.0);
  std::uniform_real_distribution<double> log_sigma(std::log(1e-6), std::log(1e3));

  const int npts = static_cast<int>(eeos_n(100, 25));
  int n_capped = 0, n_rho_clamped = 0;
  double max_rt = 0.0;

  for (int k = 0; k < npts; ++k) {
    const double rho = sampler.rho();
    const double ye = sampler.ye();
    const SRange sr = view.srange(rho, ye);
    const double s = sampler.s_in(sr, 0.05);
    const double cos_vB = cq(sampler.rng);
    const EOSPoint pt0 = view.evaluate(rho, s, ye, nan_guess());
    const double B2 = (k % 4 == 0) ? 0.0 : std::exp(log_sigma(sampler.rng)) * rho * pt0.h;

    const Prim2ConOut truth = eeos::prim2con(view, rho, s, ye, 5.0, B2, cos_vB, pt0.u_solved);
    const Con2PrimIn cin = as_in(truth);
    const Con2PrimSafeOut so = eeos::con2prim_safe(view, cin, copts, pol, s, 5.0, pt0.u_solved);
    INFO("w-cap state " << k);

    CHECK((so.policy_flags & eeos::flag_pol_w_capped) != 0u);
    CHECK(so.solved); // this repair happens AFTER a successful solve
    ++n_capped;
    CHECK(so.base.w == pol.w_cap);

    // D is preserved exactly (the documented refinement: rho is recomputed
    // as D/cosh(w_cap) rather than left at the solved value), while the
    // momentum is implicitly rescaled -- so cons must differ from the input.
    // The one exception is documented in state_policy.hpp: capping w RAISES
    // rho by cosh(w_solved)/cosh(w_cap), which can push it past the density
    // ceiling; the rho clamp then takes precedence over D preservation and
    // says so with flag_pol_rho_clamped.
    if (so.policy_flags & eeos::flag_pol_rho_clamped) {
      ++n_rho_clamped;
      CHECK(so.base.rho == pol.rho_ceiling);
      CHECK(so.cons.D < cin.D);
    } else {
      CHECK(rel_err(so.cons.D, cin.D) <= 1e-14);
    }
    CHECK(so.cons.S_perp != cin.S_perp);
    CHECK_FALSE(cons_bit_identical(so.cons, cin));

    check_policy_valid_output(view, pol, so, "w-cap");
    max_rt = std::max(max_rt, check_resolvable(view, copts, so, "w-cap").cold_err);

    // Re-solving the output must land at or below the cap.
    const Con2PrimIn back = as_in(so.cons);
    const Con2PrimOut re = eeos::con2prim(view, back, copts, nan_guess(), nan_guess(), nan_guess());
    CHECK(re.w <= pol.w_cap * (1.0 + 1e-10));
  }

  std::cout << "test_state_policy 6: " << n_capped << "/" << npts
            << " w=5 states capped at w_cap=" << pol.w_cap << " (" << n_rho_clamped
            << " of them also hit the density ceiling, so D could not be preserved there); max re-solve "
               "round trip = "
            << max_rt << "\n";
  CHECK(n_capped == npts);
}

// ==========================================================================
// 7. project_prim_state()/check_prim_state(): the flag table, and
//    idempotence.
// ==========================================================================

TEST_CASE("state_policy: project_prim_state flags each violation exactly and is idempotent") {
  SyntheticOptions opts;
  EntropyEOS adapter = build_synthetic(opts);
  const EntropyEOSView view = adapter.view();
  // rho_atm 10x above the table floor, so a "too low but still tabulated"
  // rho exists and the ceiling/floor cases are distinguishable.
  const PolicyOptions pol = eeos::default_policy(view, 10.0 * std::pow(10.0, view.x_lo));

  const double rho_mid = std::pow(10.0, 0.5 * (view.x_lo + view.x_hi));
  const double ye_mid = 0.5 * (view.y_lo + view.y_hi);
  const SRange sr_mid = view.srange(rho_mid, ye_mid);
  const double s_mid = 0.5 * (sr_mid.s_min + sr_mid.s_max);

  // The physical srange moves with BOTH rho and Ye, so every case below whose
  // rho or Ye gets clamped carries an s that is mid-range at the CLAMPED
  // (rho, Ye) -- otherwise the s clamp would fire too and the "each flag
  // exact" assertion would be measuring the test's own choice of s rather
  // than the policy.
  const SRange sr_atm = view.srange(pol.rho_atm, ye_mid);
  const double s_atm_mid = 0.5 * (sr_atm.s_min + sr_atm.s_max);
  const SRange sr_ceil = view.srange(pol.rho_ceiling, ye_mid);
  const double s_ceil_mid = 0.5 * (sr_ceil.s_min + sr_ceil.s_max);
  const SRange sr_ylo = view.srange(rho_mid, view.y_lo);
  const double s_ylo_mid = 0.5 * (sr_ylo.s_min + sr_ylo.s_max);
  const SRange sr_yhi = view.srange(rho_mid, view.y_hi);
  const double s_yhi_mid = 0.5 * (sr_yhi.s_min + sr_yhi.s_max);

  struct Case {
    const char *name;
    PrimState ps;
    unsigned expect;
  };
  const Case cases[] = {
      {"valid", {rho_mid, s_mid, ye_mid, 1.0}, 0u},
      {"valid at w = 0", {rho_mid, s_mid, ye_mid, 0.0}, 0u},
      {"valid at w = w_cap exactly", {rho_mid, s_mid, ye_mid, pol.w_cap}, 0u},
      {"valid at rho = rho_atm exactly", {pol.rho_atm, s_atm_mid, ye_mid, 0.0}, 0u},
      {"valid at rho = rho_ceiling exactly", {pol.rho_ceiling, s_ceil_mid, ye_mid, 1.0}, 0u},
      {"valid at s = s_min exactly", {rho_mid, sr_mid.s_min, ye_mid, 1.0}, 0u},
      {"valid at s = s_max exactly", {rho_mid, sr_mid.s_max, ye_mid, 1.0}, 0u},
      {"rho below the floor", {0.5 * pol.rho_atm, s_mid, ye_mid, 1.0}, eeos::flag_pol_atmosphere},
      {"rho = 0", {0.0, s_mid, ye_mid, 1.0}, eeos::flag_pol_atmosphere},
      {"rho negative", {-1.0, s_mid, ye_mid, 1.0}, eeos::flag_pol_atmosphere},
      {"rho above the ceiling", {10.0 * pol.rho_ceiling, s_ceil_mid, ye_mid, 1.0},
       eeos::flag_pol_rho_clamped},
      {"s below the physical floor", {rho_mid, sr_mid.s_min - 5.0, ye_mid, 1.0},
       eeos::flag_pol_s_floored},
      {"s above the physical ceiling", {rho_mid, sr_mid.s_max + 5.0, ye_mid, 1.0},
       eeos::flag_pol_s_ceiled},
      {"Ye below the table range", {rho_mid, s_ylo_mid, view.y_lo - 0.1, 1.0},
       eeos::flag_pol_ye_clamped},
      {"Ye above the table range", {rho_mid, s_yhi_mid, view.y_hi + 0.1, 1.0},
       eeos::flag_pol_ye_clamped},
      {"w above the cap", {rho_mid, s_mid, ye_mid, 2.0 * pol.w_cap}, eeos::flag_pol_w_capped},
      {"w negative", {rho_mid, s_mid, ye_mid, -1.0}, eeos::flag_pol_w_capped},
      {"rho above ceiling AND w above cap", {10.0 * pol.rho_ceiling, s_ceil_mid, ye_mid, 2.0 * pol.w_cap},
       eeos::flag_pol_rho_clamped | eeos::flag_pol_w_capped},
      {"s below floor AND Ye above range AND w above cap",
       {rho_mid, sr_yhi.s_min - 5.0, view.y_hi + 0.1, 2.0 * pol.w_cap},
       eeos::flag_pol_s_floored | eeos::flag_pol_ye_clamped | eeos::flag_pol_w_capped},
      {"NaN rho", {nan_guess(), s_mid, ye_mid, 1.0},
       eeos::flag_pol_nonfinite | eeos::flag_pol_atmosphere},
      {"NaN s", {rho_mid, nan_guess(), ye_mid, 1.0},
       eeos::flag_pol_nonfinite | eeos::flag_pol_atmosphere},
      {"NaN Ye", {rho_mid, s_mid, nan_guess(), 1.0},
       eeos::flag_pol_nonfinite | eeos::flag_pol_atmosphere},
      {"NaN w", {rho_mid, s_mid, ye_mid, nan_guess()},
       eeos::flag_pol_nonfinite | eeos::flag_pol_atmosphere},
      {"Inf w", {rho_mid, s_mid, ye_mid, inf_val()},
       eeos::flag_pol_nonfinite | eeos::flag_pol_atmosphere},
      {"-Inf rho", {-inf_val(), s_mid, ye_mid, 1.0},
       eeos::flag_pol_nonfinite | eeos::flag_pol_atmosphere},
  };

  for (const Case &c : cases) {
    INFO("prim case: " << c.name);
    // check_prim_state() returns exactly what project_prim_state() sets.
    CHECK(eeos::check_prim_state(view, c.ps, pol) == c.expect);

    PrimState ps = c.ps;
    const unsigned got = eeos::project_prim_state(view, ps, pol);
    CHECK(got == c.expect);

    // The projected state is finite, in range, and a FIXED POINT.
    CHECK(std::isfinite(ps.rho));
    CHECK(std::isfinite(ps.s));
    CHECK(std::isfinite(ps.ye));
    CHECK(std::isfinite(ps.w));
    CHECK(ps.rho >= pol.rho_atm);
    CHECK(ps.rho <= pol.rho_ceiling);
    CHECK(ps.ye >= view.y_lo);
    CHECK(ps.ye <= view.y_hi);
    CHECK(ps.w >= 0.0);
    CHECK(ps.w <= pol.w_cap);
    const SRange sr_out = view.srange(ps.rho, ps.ye);
    CHECK(ps.s >= sr_out.s_min);
    CHECK(ps.s <= sr_out.s_max);

    PrimState ps2 = ps;
    CHECK(eeos::project_prim_state(view, ps2, pol) == 0u);
    CHECK(eeos::check_prim_state(view, ps, pol) == 0u);
    CHECK(ps2.rho == ps.rho);
    CHECK(ps2.s == ps.s);
    CHECK(ps2.ye == ps.ye);
    CHECK(ps2.w == ps.w);
  }

  SUBCASE("default_policy's derived defaults") {
    CHECK(pol.rho_ceiling == doctest::Approx(std::pow(10.0, view.x_hi)));
    CHECK(pol.w_cap == doctest::Approx(std::acosh(100.0)));
    CHECK(pol.D_max == doctest::Approx(pol.rho_ceiling * std::cosh(pol.w_cap)));
    CHECK(pol.tau_max > 0.0);
    CHECK(std::isfinite(pol.tau_max));
    // tau_max must bound the fluid tau of every state the policy admits --
    // spot-check the hot corner it is derived from.
    const SRange sr_hot = view.srange(pol.rho_ceiling, ye_mid);
    const EOSPoint hot = view.evaluate(pol.rho_ceiling, sr_hot.s_max, ye_mid, nan_guess());
    const double W = std::cosh(pol.w_cap);
    CHECK(pol.tau_max >= pol.rho_ceiling * hot.h * W * W * (1.0 - 1e-12));
    // The cap must leave the solver room to exceed it (else it never fires).
    CHECK(pol.w_cap < Con2PrimOptions().w_max);
  }
}

// ==========================================================================
// 8. Guarded LS220 real table: the no-false-positive scan on real data, then
//    the full broken-state battery of tests 2-6 on the real adapter.
//    SRO is skipped entirely here (839 MB; tests/test_scale.hpp).
// ==========================================================================

namespace {

const std::string kLS220Path = "tables/LS220_234r_136t_50y_analmu_20091212_SVNr26.h5";

bool table_exists(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

} // namespace

TEST_CASE("state_policy: LS220 real table (guarded) -- no false positives, then the broken-state battery") {
  // The 839 MB SRO table is not exercised in this file at all (see the module
  // header): the policy layer's real-table coverage is LS220, at the reduced
  // state count eeos_n() gives under sanitizers. Announcing it through the
  // shared helper keeps the "which variants did not run" line in the output
  // identical in shape to every other suite's.
  (void)eeos_skip_big_table("state_policy SRO variant (LS220 below covers the real-table path)");

  if (!table_exists(kLS220Path)) {
    WARN_MESSAGE(false, "LS220 table not found at '" << kLS220Path << "' -- skipped ('skipped')");
    return;
  }

  RawTable table = eeos::read_stellarcollapse(kLS220Path);
  const eeos::RepairResult repair_result = eeos::repair_table(table);
  (void)repair_result;
  EntropyEOS adapter = eeos::build_entropy_eos(table); // default m_B (amu)
  const EntropyEOSView view = adapter.view();
  const Con2PrimOptions copts;

  SUBCASE("no false positives over 2000 (300 under sanitizers) valid states") {
    const int npts = static_cast<int>(eeos_n(2000, 300));
    const NoFalsePositiveResult res = run_no_false_positives(view, 0x15220A11u, npts, 5.0);
    report_nfp("test_state_policy 8 (LS220)", res);

    // Measured (LS220, repaired, default m_B, n=2000): 9 interventions, 8 of
    // them on states where the plain solver itself failed -- the documented
    // M3 hot-edge/acausal residual tail (CODE.md M3 open item (i)), which is
    // what this layer EXISTS to absorb, not a false positive. The remaining
    // one is a state where the solver converged onto an alternate root above
    // the rapidity cap, and capping it is again the layer doing its job. The
    // sharp per-state assertion inside run_no_false_positives() (solver
    // converged AND its answer policy-valid => the layer did nothing at all)
    // holds for every one of the 2000; the loose bound here only guards
    // against a wholesale regression.
    CHECK(res.n_interventions <= res.n / 100);
  }

  SUBCASE("broken-state battery on the real adapter") {
    const PolicyOptions pol = eeos::default_policy(view, 10.0 * std::pow(10.0, view.x_lo));
    const double ye_in = 0.5 * (view.y_lo + view.y_hi);

    // A valid real-table base state for the derived cases (tau/10, w-cap).
    const double rho = std::pow(10.0, 0.5 * (view.x_lo + view.x_hi));
    const SRange sr = view.srange(rho, ye_in);
    const double s = 0.5 * (sr.s_min + sr.s_max);
    const EOSPoint pt0 = view.evaluate(rho, s, ye_in, nan_guess());

    // On the REAL adapter the cold re-solve is only REPORTED, not asserted
    // (see check_resolvable()'s doc comment): two of the cases below --
    // the ceiling projection at the table's hot high-density corner, and the
    // w-capped state, whose recovered entropy is O(1e5) kB/baryon in LS220's
    // radiation-dominated pocket -- are exactly the corners where the M3d
    // seed plus the S9 bracket scan have their documented residual failure
    // tail (CODE.md M3 open item (i)). Both re-solve to 1.5e-16 WARM, i.e.
    // the states this layer produced are genuine converged solutions; it is
    // the cold start that misses them.
    int n_cases = 0, n_cold_missed = 0;

    // 2. atmosphere (incl. garbage tau) -----------------------------------
    for (double tau : {0.0, 1e30, -1e20}) {
      const double D = 0.5 * pol.rho_atm;
      const Con2PrimIn cin{D, tau, D * ye_in, 1e10, 1e10, 0.0};
      const Con2PrimSafeOut so = eeos::con2prim_safe(view, cin, copts, pol);
      INFO("LS220 atmosphere tau=" << tau);
      CHECK((so.policy_flags & eeos::flag_pol_atmosphere) != 0u);
      check_policy_valid_output(view, pol, so, "LS220 atmosphere");
      if (!check_resolvable(view, copts, so, "LS220 atmosphere", /*require_cold=*/false).cold_ok) ++n_cold_missed;
      ++n_cases;
    }

    // 3. collapse, both settings ------------------------------------------
    for (int mode = 0; mode < 2; ++mode) {
      PolicyOptions p = pol;
      p.collapse_to_atmosphere = (mode == 0);
      const Con2PrimIn a{1e6 * pol.D_max, 1e6 * pol.tau_max, 1e6 * pol.D_max * ye_in, 1e30, 1e30, 0.0};
      const Con2PrimIn b{rho, 1e6 * pol.tau_max, rho * ye_in, 1e20, 1e20, 0.0};
      for (const Con2PrimIn &cin : {a, b}) {
        const Con2PrimSafeOut so = eeos::con2prim_safe(view, cin, copts, p);
        INFO("LS220 collapse mode=" << mode);
        CHECK((so.policy_flags & eeos::flag_pol_ceiling) != 0u);
        check_policy_valid_output(view, p, so, "LS220 collapse");
        if (!check_resolvable(view, copts, so, "LS220 collapse", /*require_cold=*/false).cold_ok) ++n_cold_missed;
        ++n_cases;
      }
    }

    // 4. non-finite in each conservative ----------------------------------
    const Prim2ConOut truth = eeos::prim2con(view, rho, s, ye_in, 2.0, 3.0 * rho * pt0.h, 0.4, pt0.u_solved);
    const Con2PrimIn base = as_in(truth);
    for (int field = 0; field < 6; ++field) {
      for (int bv = 0; bv < 2; ++bv) {
        Con2PrimIn cin = base;
        double *slots[6] = {&cin.D, &cin.tau, &cin.D_Y, &cin.S_par, &cin.S_perp, &cin.B2};
        *slots[field] = (bv == 0) ? nan_guess() : inf_val();
        const Con2PrimSafeOut so = eeos::con2prim_safe(view, cin, copts, pol);
        INFO("LS220 nonfinite field=" << field << " bv=" << bv);
        CHECK((so.policy_flags & eeos::flag_pol_nonfinite) != 0u);
        check_policy_valid_output(view, pol, so, "LS220 nonfinite");
        if (!check_resolvable(view, copts, so, "LS220 nonfinite", /*require_cold=*/false).cold_ok) ++n_cold_missed;
        ++n_cases;
      }
    }

    // 5. tau below the cold floor -----------------------------------------
    {
      Con2PrimIn cin = base;
      cin.tau = base.tau / 10.0;
      const Con2PrimSafeOut so = eeos::con2prim_safe(view, cin, copts, pol);
      INFO("LS220 tau/10");
      CHECK(so.policy_flags != 0u);
      check_policy_valid_output(view, pol, so, "LS220 tau/10");
      if (!check_resolvable(view, copts, so, "LS220 tau/10", /*require_cold=*/false).cold_ok) ++n_cold_missed;
      ++n_cases;
    }

    // 6. w cap -------------------------------------------------------------
    {
      PolicyOptions p = pol;
      p.w_cap = std::acosh(10.0); // D_max/tau_max left wide, see test 6's comment
      const Prim2ConOut t5 = eeos::prim2con(view, rho, s, ye_in, 5.0, 0.0, 0.3, pt0.u_solved);
      const Con2PrimIn cin = as_in(t5);
      const Con2PrimSafeOut so = eeos::con2prim_safe(view, cin, copts, p, s, 5.0, pt0.u_solved);
      INFO("LS220 w-cap");
      CHECK((so.policy_flags & eeos::flag_pol_w_capped) != 0u);
      CHECK(so.base.w == p.w_cap);
      CHECK(rel_err(so.cons.D, cin.D) <= 1e-14);
      check_policy_valid_output(view, p, so, "LS220 w-cap");
      if (!check_resolvable(view, copts, so, "LS220 w-cap", /*require_cold=*/false).cold_ok) ++n_cold_missed;
      ++n_cases;
    }

    std::cout << "test_state_policy 8 (LS220): broken-state battery ran " << n_cases
              << " cases, all outputs policy-valid and exactly re-solvable when warm-started ("
              << n_cold_missed << " missed by a COLD re-solve; see the comment above)\n";
    CHECK(n_cases >= 20);
  }
}
