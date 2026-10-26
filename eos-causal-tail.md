# Causal extension tails: the log-σ u-high tail (M3g) and x-low tails (M3i)

Draft v0.2, August 27, 2026. Companion to `eos-adapter-F-to-U.md` (§7 domain
extensions), `core/adapter_eval.hpp` ("TAIL MATHEMATICS" / "CAUSAL TAILS" /
"M3i X-LOW CAUSAL TAILS"), and CODE.md's "M3 failure-tail root cause" findings
block (the evidence base).
**Approved 2026-08-27** (`cs2_ext_cap = 0.99`; §7 principle amendment ships with
the implementation). **Implemented 2026-08-27** — measurements in CODE.md's
"M3g empirical findings" block; §3 as specified, §5's map added.
**§5's x-low follow-up implemented 2026-08-28 (M3i)** — same family argument on
the x axis; measurements in CODE.md's "M3i empirical findings" block.
**§9's u-high ratio lock implemented 2026-10-25 (M3k, 1.2.0)** — removes the tail's
depth-linear drift where the slope ratio varies with ρ, gated off where the locked tail
would not be causal; measurements in CODE.md's "M3k empirical findings" block.

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

**Repaired 2026-08-28 (M3i)** — same family argument as §2–§3, one axis over, and
the map is the acceptance gauge. At these densities radiation dominates for
T ≳ 0.05 MeV, so `ln σ` and `ln ε` are linear in `x = log₁₀ρ` (slope −ln10):
*exponential in x for the values*, exactly as they were in `u`. Two changes,
x-LOW only, both in `core/adapter_eval.hpp`:

- **Log-σ x-low tail** (σ only), the mirror of §3's construction: log-sample,
  run the unchanged curvature-ramp machinery on `g = lnσ`, map back. Unlike the
  `u_hi` seam, `σ_b > 0` is *not* guaranteed here — the u-low tail drives σ
  linearly toward −∞ by design — so `aeval_xlow_log_ok()` admits the log branch
  only where `σ_b > 0` and the log track's slopes keep the band's total log
  excursion bounded, and falls back to the plain linear tail otherwise. It never
  fires on either real table or on the synthetic gas (min seam σ 1.64 / 0.80 /
  37.2; largest excursion 1.03 against the bound of 40).
- **L's slope-to-zero override is gone.** L is already a log-space field, so its
  plain linear tail *is* the correct exponential family for ε; the override forced
  `b_x = 0` and hence `q = 0 + 4/3` and `c_s² = 4/3` — the measured 1.35 —
  everywhere, while its C1 curvature override (`L_xx_eff = 2L_x/h_x ≈ −36` at a
  radiation seam) was the band's whole `c_s² ≤ 0` population. With both families
  right, `q = −1 + 4/3 = 1/3` and `c_s² = 1/3`, and the old "ε becomes
  ρ-independent" behaviour re-emerges by itself at cold seams, where the measured
  `L_x` is 0.

Measured: x_low `c_s² ≥ 1` **500,561 → 0** (LS220) and **688,076 → 0** (SRO);
`c_s² ≤ 0` **79,634 → 683** and **107,006 → 25,681**, with the worst excess falling
from 5.69 / 10.14 to **3.2e-4 / 1.5e-3**. The other three bands are unchanged
number for number. **No causal slope clamp was added on this side** (the mirror of
§3's `u_high_b_cap`): the measured asymptote is 1/3, and unlike the u-high clamp —
which the synthetic ideal gas trips at every seam point, keeping it covered in CI —
an x-low clamp would be dead code on every table available. Full numbers in CODE.md's
M3i findings block.

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

- ~~Class B solver work: relative-width local scan window; Newton-path f2
  precision-floor acceptance (mirror of the outer bisection's polish).~~ — **done
  (M3h, 2026-08-27)**: the local scan window is now a geometric relative ladder, and the
  precision-floor acceptance turned out to be needed on *f1*, not f2 (an absolute
  tolerance on a residual that is O(cosh w)); with the Newton's best-iterate handoff
  these take the residual failure tail to 1 state per table in 44,000 — see CODE.md
  "M3h empirical findings".
- Inner-solve robustness to non-monotone f1 as defense in depth (bracketed
  sign-bisection) — worth having even with causal tails, but it must not mask
  this fix's acceptance numbers; if built, land it behind a flag or after M3g's
  measurements.
- ~~x-low band repair, pending the §5 map.~~ — **done (M3i, 2026-08-28)**: log-σ
  x-low tail plus the removal of L's slope-to-zero override take that band's
  `c_s² ≥ 1` count to 0 on both tables and its far tail to the same 1/3 asymptote
  the u-high side reaches — see §5 and CODE.md "M3i empirical findings". What is
  left of the original M3 tail: one pathological LS220 state and one SRO state on
  the **x-HIGH** (hard-invalid `flag_oob_rho_high`) band, per 44,000; the x-high
  band itself stays report-only by design.

## 8. Open decisions

1. ~~`cs2_ext_cap` default~~ — decided 2026-08-27: **0.99** (match `cs2_cap`).
2. ~~§7 principle amendment~~ — decided 2026-08-27: yes — amend
   `eos-adapter-F-to-U.md` §7's principle sentence to "finite, smooth, monotone,
   **causal**" as part of the implementation.

## 9. The u-high ratio lock (M3k, 1.2.0)

**Problem.** §3's asymptote `c_s² = b/α − 1` assumes 1/ρ seam scaling: the two log
slopes' *ratio* is the same at neighbouring densities. Where it is not, the tail drifts.
In phase 2 (`L_u = m_L`, `g_u = α`, `g = lnσ`, `d = u − u_hi`, no u-curvature), the
secondary tracks are linear in depth, `L_x(d) = L_x + L_xu d` and `g_x(d) = g_x + g_xu d`,
so along an adiabat

```
∂L/∂x|_s = (L_x − r g_x) + (L_xu − r g_xu) d,     r = m_L / α,
```

and the second term grows without bound. It vanishes at a radiation seam
(`L_xu = g_xu = 0`) and nowhere else. Measured where it bites: the seam-blend window
`repair_table()` rebuilds at ρ ≈ 1.4e7 on DD2/SFHo (the ratio varies by ±0.1 across
it, so `c_s² ≤ 0` from the seam on and `p ≤ 0` at 6–7 cells), and the matter-dominated
hot seam at ρ ~ 1e15 (`c_s²` 0.70 → 0.18 over the 8-cell band, ≤ 0 at ~10 cells). The
T_max trim (`RepairOptions::trim_t_hi_rows`) made the first one visible: DD2's u-high
`p ≤ 0` went 0 → 6,307 when the tail was launched from 152 MeV instead of 158.

**The lock.** L's u-high secondary track gets the effective curvature that ramps its
slope, over the same one-cell blend the primary track uses, from the spline's `L_xu` to
the drift-free `r g_xu` — the §3 causal clamp's own effective-track trick, so the tail
stays C¹ at the seam. Beyond phase 1 the drift term is identically zero, and with
`L_uu = g_uu = 0` there the second fixed-s derivative is depth-independent too,
`∂²L/∂x²|_s = L_xx − r g_xx`. Only L's secondary track changes: its value track, and with
it ε, U, κ's floor scan, `srange()` and the T-solve, are bit-identical.

**The exact phase-2 tail, and why the lock must be gated.** With the lock, the
phase-2 tail has constant fixed-s slopes `k0 = A − rB` (of ln E, E = ε̂ + Δ̂) and
`j0 = C − rD` (of L), where `A = L_x + (w/3)L_xu`, `B = g_x + (w/3)g_xu` (the phase-1
offsets), `C = L_xx`, `D = g_xx` at the seam, and

```
c_s² = E [k0(1 + k0) + j0/ln10] / [(1 − Δ̂) + E(1 + k0)]  →  c_inf = k0 + j0 / (ln10 (1 + k0)),
```

monotonically as ε grows into the tail (on LS220 at ρ ≈ 10^15.5: 1.57 predicted, 1.58 evaluated 64 cells
deep). At a radiation seam (`A = −1`, `B = −ln10`, `C = D = 0`) `c_inf ≤ cap`
is exactly §3's `b ≤ (1 + cap) α`. A cap on r was tried first and fails where it
matters: at the ρ ~ 1e15 seams of LS220 and DD2 the locked `c_inf` is 1.2–1.6, and it
comes from the seam's own x-structure (`A`, `C`), so `c_inf` is already above 1 at
`r = 0` and no slope cap reaches it. Freezing that stiff structure while ε keeps
growing is simply the wrong extrapolation there. So the lock is gated: it acts with
weight `λ = S((cs2_ext_cap − c_inf)/0.1)`, S the quintic smoothstep clamped to [0, 1] —
full where the locked tail is causal with room to spare, none (the §3 tail, whose drift
is soft) where it is not, smooth in between because `c_inf` is a smooth function of the
seam sample.

**Measured** (CODE.md "M3k empirical findings"): DD2 u-high `p ≤ 0` 6,307 → **0**,
`c_s² ≤ 0` 43,299 → 1,891, `c_s² ≥ 1` stays 0; SFHo `c_s² ≤ 0` 30,940 → **0**; LS220/SRO
move by a few samples (SRO u-high `c_s² ≤ 0` 16,573 → 16,537 with the trim off). The
ρ ~ 1e15 hot corners keep the §3 tail and its soft drift, which is what CODE.md's M3g
open item described; nothing on any table goes acausal because of the lock.
`tests/test_adapter_tail.cpp` asserts the depth-independence of `∂L/∂x|_s`, C¹ at the
seam, the unchanged value track, and that a closed gate reproduces the 1.1.0 tail
bit for bit.
