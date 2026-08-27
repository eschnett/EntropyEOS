# Causal extension tails: the log-σ u-high tail (M3g)

Draft v0.2, August 27, 2026. Companion to `eos-adapter-F-to-U.md` (§7 domain
extensions), `core/adapter_eval.hpp` ("TAIL MATHEMATICS" / "CAUSAL TAILS"), and
CODE.md's "M3 failure-tail root cause" findings block (the evidence base).
**Approved 2026-08-27** (`cs2_ext_cap = 0.99`; §7 principle amendment ships with
the implementation). **Implemented 2026-08-27** — measurements in CODE.md's
"M3g empirical findings" block; §3 as specified, §5's map added, §7 still open.

**Results in one paragraph.** The u-high band's `c_s² ≥ 1` fraction goes
**88.4% → 0.007%** (LS220) and **88.0% → 0.009%** (SRO), with the far tail
measuring `c_s² = 0.339 / 0.340` — the predicted radiation asymptote `b/α − 1`,
not merely a bound — and the only survivors sitting in the blend cell above the
known residual ρ_max/T_max *data* corner. At 40,000 con2prim states the warm
failure count falls 86 → 31 (LS220) and 163 → 82 (SRO), cold 16 → 8 and 30 → 15,
the **silent wrong-root class 18 → 0 and 25 → 0**, and `rt_tau`'s p999 collapses
from 4.6e-3 / 9.5e-1 to 9.6e-13 / 9.9e-13, i.e. into the 1e-13 bulk. The whole
residual now carries the class-B signature (f2 precision floor at high rapidity;
35/39 resp. 95/97 recover on a bigger iteration budget), so §7's solver
follow-ups are the next lever, not this design. The causal slope clamp never
fires on either real table (b/α − 1 ≈ 1/3 is far below 0.99) and fires at every
seam point of the synthetic *ideal* gas, which is what keeps it covered in CI.
The §5 map's real news: the **x-low band is ~50–58% acausal** — the follow-up
that §7 anticipated, and where the last acausal failure paths live. The one
thing the fix makes worse: ~0.5% of the u-high band acquires `c_s² ≤ 0` (and
~0.02% `p ≤ 0`) deep in the tail, where it previously had `c_s² ≥ 1`; that is
harmless for the §9 inner-solve proof (which needs only `c_s² < 1`) but the map
carries a report-only `cs2_nonpositive` class per band so it stays visible.

## 1. Problem

The M3 con2prim failure tail (~0.2–0.8% of audit states, plus ~0.05% *silent
wrong-root convergences*) is ~95% caused by the u-high (hot-entropy) extension:
its `c_s²` crosses 1 about one grid cell past the s_max seam (measured 0.45 /
0.78 / 1.55 at 1.02× / 1.1× / 1.3× s_max, saturating ≈ 3.8; numerically identical
on LS220 and SRO, so it is the tail *construction*, not the data). Any hot-edge
state whose trial iterate satisfies s > s_max(D/cosh w) evaluates there; with
`c_s² > 1`, `z_w = z(1−c_s²)tanh w < 0` and f1(w; s) is non-monotone (measured
2–6 slope sign flips per failing path), so the §9 inner-solve proof — the
fallback's correctness guarantee — does not hold on the extended domain. The §7
principle "during iteration the adapter must always return finite, smooth,
monotone values" is missing one word: **causal**. Data-side repair cannot reach
this (M3f capped the box; the tail is rebuilt from seam tracks at run time), and
run-time clamping of `cs2` stays forbidden for the usual single-potential reason.

## 2. Why tuning the existing tail cannot fix it

The current u-high tail continues both σ and L(= log₁₀(ε+Δ)) **linearly in
u = log₁₀T**. For L that is physically right at a hot boundary (radiation:
ε ∝ T⁴/ρ ⇒ L linear in u, slope ≈ 4). For σ it is structurally wrong: physical
entropy grows *exponentially* in u (s ∝ T³/ρ), so a linear σ-tail forces T — and
with it ε — to climb far too fast along adiabats as ρ rises at fixed s.
Quantitatively, with q ≡ ∂lnW/∂x|_s (W = 1 + ε, x = lnρ; p > 0 ⟺ q > 0) the
causality condition c_s² ≤ 1 reads q′ + q² ≤ 1. In the hot-seam radiation regime
(seam values σ_b, σ_u, ε_b all ∝ 1/ρ) the linear-σ tail gives q′ = q + 1, so the
condition becomes q(1+q) ≤ 0 — **incompatible with p > 0 except at the measure-zero
point q = 0**. No slope choice rescues it; the tail family itself must change.

## 3. The fix

**Replace the u-HIGH tail of σ (only) by the same curvature-ramp construction
applied in log space**, and add a **causal slope clamp** to L's u-high tail:

- **Log-σ tail.** Transform the σ primary u-track at the seam,
  `g = lnσ: g_b = lnσ_b, g′_b = σ_u/σ, g″_b = σ_uu/σ − (σ_u/σ)²`, run the existing
  `aeval_ramp_track()` phases 1–2 on g unchanged, and map back
  (`σ = e^g, σ_u = σg′, σ_uu = σ(g″ + g′²)`). σ_b > 0 always holds at the u_hi
  seam (it is the column's largest entropy). The secondary (fx) track and the
  frozen fxx/fy tracks transform the same way (`g_x = σ_x/σ` etc., mapped back
  with the tail-evolved σ), so the mixed-derivative composition table of
  adapter_eval's module comment is unchanged in shape. Corner composition order
  (u-tail first, then x-tail on the extended tracks) is untouched — the log
  transform lives entirely inside the u-high tail operator, σ field only.
- **Why it works.** With lnσ and lnε both asymptotically linear in u (growth
  rates α ≡ dlnσ/du and b ≡ dlnε/du) and 1/ρ seam scaling, the far-tail
  fixed-s slope is *constant*: q = b/α − 1, q′ = 0, hence
  **c_s²(tail) = b/α − 1 exactly**. Radiation slopes (b = 4·ln10, α = 3·ln10 per
  u) give c_s² = 1/3 — the tail becomes not merely causal but the physically
  correct hot-gas asymptotic. p > 0 ⟺ b > α (true: 4 > 3).
- **Causal slope clamp on L's u-high tail.** Enforce, per evaluation point at the
  seam, `b_eff ≤ (1 + cs2_ext_cap)·α_eff` with
  `b_eff = ln10·m_L·(ε_b+Δ)/ε_b` and `α_eff = m_lnσ` (the two phase-2 slopes):
  where the raw track violates it, lower the effective L-slope to the bound —
  the exact analogue of the existing monotonicity guard, same effective-track
  mechanism, C² dropping to C1 only inside flagged territory. Guard priority is
  lexicographic, as everywhere else: the monotonicity floor
  (`ext_slope_floor_L`) wins over the causal clamp if they ever conflict (not
  expected on real tables — α is large at hot seams; the audit reports any such
  point).
- **Monotonicity guard in log space.** σ_u ≥ ext_slope_floor_sigma is preserved
  by flooring the log-slope at `ext_slope_floor_sigma/σ_b` (since σ ≥ σ_b in the
  growing tail, σ_u = σ·g′ ≥ σ_b·g′).

Unchanged: the u-LOW tails (σ must run linearly to −∞ so every s < s_min maps to
a finite T — the escape-hatch design stands, and no failure implicates it), both
x-tails' construction (but see §5), all flag semantics, `srange()`/physical-box
logic, and every in-box evaluation (the tail operator is bypassed identically —
assert bit-identity in tests).

## 4. Consequences to check, not fear

- `srange_extended().s_max` grows: the 8-cell extension gives ~10^0.8 ≈ 6× s_max
  (log-tail) where the linear tail gave ~1.8×. The bracket scan is log-spaced and
  the T-solve is a safeguarded Newton on a *still strictly monotone* σ — no
  mechanics change. Class B (see CODE.md) worsens only if scan windows are sized
  to srange spans; its fix is solver-side and out of scope here.
- The phase-1 blend cell can still overshoot `cs2_ext_cap` transiently (seam
  curvature ramps off over one cell). The acceptance audit (§6) is the arbiter;
  if it objects, cap the effective g″/L″ in the blend the same way the slope
  clamp works. Expected benign: the box side of the seam is already capped at
  0.99 by M3f.
- Throughput: one extra exp/log pair on the σ u-high tail path only; in-box
  evaluations untouched.

## 5. Secondary scope: the other extension bands

One failure path (LS220 k=21010) showed c_s² = 1.75 with only `flag_ext_rho_low`
— the x-low band has at least one acausal region too. This design does not
redesign the x-tails; instead it adds the missing *visibility*: a deterministic
refined **extension-band c_s² map** (all four bands: u-low, u-high, x-low, x-hi,
each scanned at (4,4)-refined resolution over its band × the other axes,
reporting c_s² ≥ 1 / p ≤ 0 / σ_u ≤ 0 counts and worst locations) added to
`check_adapter()` as a new violation class, run in `eos_test --level adapter`.
If the map shows the x-low band matters beyond that one path, its fix is a
follow-up (same effective-track clamp toolbox). The x-hi band stays hard-invalid
(`flag_oob_rho_high`) regardless — causality there is nice-to-have, not load-
bearing.

**Measured (2026-08-27, first run of the map).** It matters: **50.5% (SRO) /
58.3% (LS220) of the x-low band has `c_s² ≥ 1`** (worst excess 0.84 / 0.79), and
after M3g every remaining acausal con2prim failure path lands there — including
the same three sampled states on *both* tables, which is the signature of a tail
construction rather than of table data. The u-low band is nearly clean (23 / 46
points), x-hi is 98–99% acausal and stays report-only. Implementation choice made
here: the u-low/u-high/x-low bands count toward `adapter_needs_attention()`, x-hi
does not, since §7 of `eos-adapter-F-to-U.md` already makes a converged state
there invalid outright. Full numbers in CODE.md's M3g findings block.

## 6. Tests and acceptance

- **Unit (exactness):** on the synthetic hot gas (radiation-dominated boundary),
  the far u-high tail must measure c_s² → b/α − 1 (≈ 1/3) to the audit's
  tolerance; a hand-built σ/L pair with known slopes checks the log-track
  transforms and the clamp arithmetic to roundoff. In-box bit-identity before vs
  after (both fields, all 12 derivatives, at random in-box points).
- **Extension-band map (both real tables):** u-high band c_s² ≥ 1 count → 0
  (currently: crossing at ~1 cell past every hot seam point); report the other
  three bands.
- **The failure tail (the actual acceptance):** `eos_test --level con2prim
  --states 40000` — expected warm failures LS220 86 → ~5, SRO 163 → ~8 (the
  class-B remainder), cold 16 → ~2 / 30 → ~4, and the wrong-root class
  (round-trip errors ≈ 1) → ~0. Path probes (scratchpad `c2p_tail.cpp`) should
  show zero f1-slope flips and z_w > 0 on every previously failing path.
- **No regressions:** class D seam jumps unchanged on synthetic (~5e-7); extended
  soak (`soak_extended`) zero maxiter; adapter class B round trips unchanged;
  warm-path solves bit-identical for states whose solve never leaves the box;
  `make test` and `integration.sh` green, serial and OpenMP.
- Record measured outcomes in CODE.md as the M3g findings block.

## 7. Out of scope (tracked follow-ups)

- Class B solver work: relative-width local scan window; Newton-path f2
  precision-floor acceptance (mirror of the outer bisection's polish).
- Inner-solve robustness to non-monotone f1 as defense in depth (bracketed
  sign-bisection) — worth having even with causal tails, but it must not mask
  this fix's acceptance numbers; if built, land it behind a flag or after M3g's
  measurements.
- x-low band repair, pending the §5 map.

## 8. Open decisions

1. ~~`cs2_ext_cap` default~~ — decided 2026-08-27: **0.99** (match `cs2_cap`).
2. ~~§7 principle amendment~~ — decided 2026-08-27: yes — amend
   `eos-adapter-F-to-U.md` §7's principle sentence to "finite, smooth, monotone,
   **causal**" as part of the implementation.
