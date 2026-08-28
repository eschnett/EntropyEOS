# Causality repair: the causal-cap stage for `eos_repair`

Draft v0.2, August 27, 2026. Companion to `eos-adapter-F-to-U.md` (§8 requirement
`0 < c_s² < 1`, §10 causality audit) and `con2prim-entropy-rapidity.md` (the
`z_w = z(1−c_s²)tanh w` conditioning that motivates this). **Approved 2026-08-27**
(stage default on, `cs2_cap = 0.99`); **implemented 2026-08-27** as the M3f
`RepairOptions::causal_cap` stage of `entropy_eos/host/repair.{hpp,cpp}`, with the
measured results in §10 below and in CODE.md's "M3f empirical findings".

## 1. Problem

The con2prim Newton's diagonal Jacobian entry is `z_w = z(1−c_s²)tanh w`: it is
positive — and the 2×2 solve well-conditioned — only where the EOS is causal. Where
the *fitted adapter* returns `c_s² ≥ 1`, Newton degrades and the M3 measured failure
tail (~0.2–0.8% of random states, hot-edge/high-w corner) lives exactly there. The
run-time guards (safeguarded fallback, `state_policy`'s ceiling projection) keep the
evolution alive but cannot restore solver conditioning; and a run-time clamp of the
returned `cs2` is not an option at all — the Newton consumes `U_ρρ` and `U_ρ`
separately, so a clamped `cs2` would no longer correspond to any single potential
`U` and would silently break the solver's identities. The only self-consistent fix
is in the *data*: make the table itself causal, before the fit, and log the edit.

## 2. Measured shape of the defect (design driver)

Node map of the fitted adapter's `c_s²` on the repaired real tables (round-trip
`s = σ(node)`, evaluate warm-started at the node's own T). LS220-2009
(234×136×50 nodes):

- **67,107 acausal nodes (4.2%), forming a clean high-ρ corner suffix**: irho ∈
  [223, 233] (ρ ≳ 2.8×10¹⁵ g/cc, i.e. ≳ 10 ρ_sat; table max 10¹⁶), at **all** 136
  temperatures and **all** 50 Ye. Of the 6,800 affected (T, Ye) columns, 6,792 are
  *contiguous runs ending at the ρ_max edge*; max depth 11 of 234 ρ cells
  (~0.6 decades). Above ρ ≈ 4.6×10¹⁵ every single node is acausal.
- **The corner is genuine LS220 physics, not spline artifact**: the table's own
  stored `cs2` column is acausal at all 81,600 nodes with irho ≥ 222 (stored max
  18.4 c²). LS220's known superluminality at high density — the repair is a genuine
  physics edit to a region that is already unphysical, exactly like the
  monotonicity repair, and must be logged the same way. (The high-density superluminality
  of Skyrme-type EOS is discussed in Schneider, Roberts & Ott 2017; constant-$c_s$
  extrapolation is the established precedent for editing a table's high-density end to
  restore causality. Servignat et al. 2024 attack the same defect from the other side, by
  replacing the table with a smooth fit having continuous $c_s$. See
  [`RELATED.md`](RELATED.md).)
- **9 isolated interior acausal nodes** (c_s² up to 151 at ρ ~ 10¹⁴, T ≤ 0.32 MeV,
  extreme Ye) — chain-rule blowups in the known σ_T ≈ 0 nuclear-transition pockets
  (CODE.md open decision 4, resolved accept-and-guard). Not part of this stage's
  target: the defect there is σ's flatness, not ε's stiffness.
- **12,022 nodes with c_s² ≤ 0** — scattered mid-table bands (ρ ~ 10⁸ and the
  transition region), not edge-anchored. A different defect class (∂p/∂ρ|_s ≤ 0
  from fit noise in near-flat regions / the same σ pockets); `z_w` stays positive
  there, so con2prim conditioning is unharmed. Reported, not repaired (v1).

SRO-LS220 (391×163×66 nodes, m_B = m_n) shows the same structure:

- **182,269 acausal nodes (4.3%), same corner suffix**: irho ∈ [373, 390]
  (ρ ≳ 2.9×10¹⁵), all T, all Ye, saturating (all 10,758 (T, Ye) columns) from
  irho = 379; 10,587 of the affected columns are contiguous suffixes to ρ_max;
  max depth 18 of 391 cells. The onset density varies by ~6 cells across Ye —
  the per-adiabat anchor (§4) absorbs that automatically. Here the stored `cs2`
  is the *relativistic* one (`have_rel_cs2 = 1`) and agrees with the derived map
  at **all** 129,096 corner nodes — the strongest possible confirmation that the
  corner is table physics.
- **~200 isolated interior acausal nodes**, again the high-Ye/low-T pocket band:
  a full-T sliver at irho 311–312 (ρ ≈ 2.7×10¹³, Ye = 0.645, c_s² ≈ 1.585 nearly
  T-independent — likely a data seam at one ρ column) plus ~20 scattered
  chain-rule spikes (max c_s² = 753 at ρ ≈ 10¹⁴, T = 0.014 MeV, Ye = 0.565). All
  at Ye ≥ 0.56 and/or in the σ_T pocket band; none edge-anchored.
- **15,895 nodes with c_s² ≤ 0**, of which 9,698 on the irho = 0 edge row (x_lo
  boundary-fit artifact) and the rest in the ρ ~ 10¹³·⁵–10¹⁴ transition band.

Why the literal "same local smoothing as the monotonicity stage" is ruled out: the
M2d-1 diffusion stage's own log on this table shows the runaway it was bounded
against (logenergy violations 4,256 → 185,951 over six rounds before the backstop
reverted everything). Diffusion removes curvature *noise*; the acausal corner is a
*systematic* stiffness over ~10 cells × all T × all Ye. What carries over from
M2d-1 is the **harness** — audit-driven rounds, per-column re-repair of touched
columns, best-state tracking, revert-if-worse backstop, full logging, idempotence —
with a different **edit primitive** that matches the constraint's structure.

## 3. The mathematics

With `x = ln ρ` at fixed `(s, Ye)`, and `W = 1 + ε` (ε dimensionless, per unit
rest mass; any baryon-mass/κ convention — c_s² is invariant under all of them):

```
p/ρ = ∂ε/∂x|_s ,   h = W + ∂W/∂x|_s ,   c_s² = ∂ ln h / ∂x |_s .
```

(The last identity is the adapter doc's `h c_s² = 2ρU_ρ + ρ²U_ρρ` rewritten in
logs; equivalently causality is `∂²U/∂x²|_s ≤ 1 + U`.) So **causality is a slope
cap on ln h along adiabats**, and the stiffest causal EOS is `h ∝ ρ^{c̄}` with
`c̄ = cs2_cap`. The repair is the one-sided Lipschitz-envelope projection the NS
literature uses to cap EOSs at the causal limit:

- **Envelope**: marching up in x along an adiabat with refined step δ,
  `h_env(x+δ) = min( h_orig(x+δ), h_env(x)·e^{c̄δ} )`. Continuous, ≤ original,
  untouched wherever the original is causal.
- **Energy consistency**: ε must follow `∂ε/∂x|_s = p/ρ = h − 1 − ε`, a linear
  ODE. Integrate it with h = h_env from the first binding point (the *anchor*),
  initial value ε_orig(anchor); per-step integrating-factor update (exact when
  c_s² is piecewise constant, unconditionally stable). Gronwall gives
  ε_env ≤ ε_orig everywhere. Then p = ρ(h_env − 1 − ε_env) > 0 is implied, ε is
  still strictly increasing along the adiabat, and ε + Δ > 0 is preserved
  (ε_env ≥ ε_anchor > −Δ).
- **Closed form** (unit-test oracle; also what the implementation reduces to on a
  pure binding stretch): with anchor values (ε_a, h_a) at x_a and Δx = x − x_a,

  ```
  h(x) = h_a e^{c̄Δx}
  ε(x) = (ε_a + 1) e^{−Δx} − 1 + h_a (e^{c̄Δx} − e^{−Δx}) / (1 + c̄)
  ```

**Only `logenergy` is edited.** σ is untouched, so: node adiabat labels
`s = σ(node)` are stable across rounds, the T-solve's `σ_u > 0` guarantee is
undisturbed, and the entropy field keeps full fidelity. The capped profile lowers
ε (hence p, h) only above the anchor — for LS220 that means only at
ρ ≳ 2.8×10¹⁵ g/cc, beyond stable-NS central densities, where states are en route
to collapse (`state_policy` excision territory) anyway.

All quantities above come from the *fitted splines of the current data* (the same
`fit_bspline_3d` the adapter uses), with log10-axis factors handled analytically
as elsewhere: first-derivative chain rule for h and the trace
(`∂ε/∂x|_s = ε_x − ε_u σ_x/σ_u`), full second-derivative chain rule (the §3.1
formulas of the adapter doc, κ-free) for the c_s² audit. No T-solve is needed to
*audit* a sample (x, u, y) — its adiabat is implicit; the *trace* needs cheap 1D
monotone u-solves on the σ spline. Because c_s² is κ- and m_B-invariant, auditing
the raw-variable fit is exactly auditing the production adapter's interior — no
`BuildOptions`, no unit decisions beyond c² and `energy_shift`.

## 4. The stage, step by step

Runs inside `repair_table()` as the last stage, per Ye slice for the traces
(slices independent → OpenMP), after the per-column and 3D-monotonicity stages.
Requires both fields; edits only `logenergy`.

1. **Audit**: fit σ and L with `fit_bspline_3d` on the current data; sample c_s²
   via the chain rule on the refined grid (main loop at (refine3d_xy, refine3d_u,
   refine3d_xy), final verification at (4,4,4), mirroring the 3D stage). A sample
   violates iff `c_s² ≥ cs2_max` (default 1.0). Also count (report-only) the
   `c_s² ≤ 0` samples and, for visibility, `c_s² ≥ cs2_cap`.
2. **Scope**: group violating samples by (refined u, y) row into maximal runs
   along x. Only runs that **reach the x_hi edge** are treated ("edge-anchored");
   interior runs (the σ-pocket spikes) are counted and reported as
   `cs2_interior_untouched`, never edited — capping ε there cannot fix a defect
   that lives in σ_T, and chasing them is what burns round budgets (M2d-1's
   lesson).
3. **Project**: for every *node* (i, j, k) lying in or above a treated run's x
   range: trace its adiabat s = σ_data(i,j,k) downward in x (refined steps; solve
   u(x') from the σ spline; where the adiabat exits the box through u_min, follow
   the u_min edge — the degenerate regime where T-dependence is negligible, see
   §6) until c_s² ≤ cs2_cap has held for anchor_pad consecutive steps; that is the
   anchor. March back up computing the envelope + ε-ODE of §3; write the node's
   new value `L = log10(ε_env·c² + Δ)` iff ε_env < ε_orig (bitwise-untouched
   otherwise). Traces are per-node and independent → deterministic under OpenMP.
4. **Restore monotonicity**: re-run the per-column pipeline (PAVA + strictify +
   1D spline-safe) on every column the projection touched, exactly as the 3D
   stage's step (d) does.
5. **Loop**: re-audit; rounds continue while the violation count improves, up to
   `causal_rounds_max`. Residual seam ringing (the spline smooths the anchor kink;
   cap vs. threshold hysteresis absorbs most of it) is handled by later rounds
   re-tracing with anchors that land deeper. Best-state tracking and the
   revert-after-non-improving rounds rule are carried over from M2d-1 verbatim.
6. **Verification + lexicographic backstop**: final audit at (4,4,4) of **both**
   c_s² and the σ_u/L_u monotonicity counts. The kept state must satisfy, in
   order: (a) monotonicity violation counts no worse than the pre-stage state,
   (b) c_s² violation count minimal among audited states. If (a) fails, revert to
   the pre-stage state (never trade the T-solve's hard requirement for
   causality); the stage then reports itself reverted.

Write-back, provenance, determinism, and `--check-only` semantics are unchanged:
every changed value becomes a `RepairEntry` (field `"logenergy"`, old/new vs. the
original input), `/repair` gains the stage's parameters, exit codes keep their
meaning.

## 5. Options (RepairOptions additions) and defaults

| option | default | meaning |
|---|---|---|
| `causal_cap` | on (decided) | run the stage at all |
| `cs2_max` | 1.0 | audit threshold: a sample with c_s² ≥ this is a violation |
| `cs2_cap` | 0.99 (decided) | target slope c̄ of the projection (hysteresis vs. cs2_max absorbs refit ringing; `1−cs2` conditioning headroom for the Newton) |
| `causal_rounds_max` | 8 | round budget (round 1 does the bulk; later rounds chase seam ringing) |
| `trace_depth_max` | 64 | cells the anchor search may descend before giving up on a node (report, don't edit) — measured depth: ≤ 11 (LS220), ≤ 18 (SRO) |
| `anchor_pad` | 2 | consecutive causal refined steps required before anchoring (hysteresis) |

`eos_repair` grows `--no-causal-cap` and `--cs2-cap X`. The default is **on**
(decided 2026-08-27): the tool's contract is "make the table satisfy the hard
requirements", and `0 < c_s² < 1` is on the §8 list; like the monotonicity repair
it is a logged physics edit, and it acts only where the table is already
unphysical by its own stored `cs2` column.

## 6. Approximations, accepted deliberately

- **Once capped, capped to the edge.** The envelope never "un-binds": if the
  original dipped back below the cap inside a treated run, the minimal projection
  would rejoin it, but ε would then need a discontinuous jump (the ODE deficit
  decays only as e^{−Δx}, ~18 cells). Measured shape shows single binding
  stretches running to the edge, so the simple monotone envelope is both smooth
  and (in practice) minimal.
- **Cold-edge path.** Adiabats traced down in x exit the box through u_min
  (du/dx|_s > 0). The trace then follows the u_min edge instead. In that regime
  (fully degenerate, σ ≈ 0, T-dependence negligible) the edge path approximates
  the adiabat excellently, and the audit — which needs no path — remains the
  arbiter of the result.
- **The projection is node-level; causality is enforced sample-level.** The refit
  can ring above cs2_cap near seams; the loop plus the cap-vs-threshold gap is
  the mechanism that converges this, same as the monotonicity stages.
- **c_s² ≤ 0 and interior spikes are out of scope (v1).** They do not harm `z_w`
  conditioning; their root cause (σ_T pockets) is the documented accept-and-guard
  residual. A v2 could add the mirrored *lower* envelope (c_s² ≥ floor along
  adiabats) with the same machinery **[decide: ever needed?]**.

## 7. Tests and acceptance

- **Unit (exactness)**: the closed form of §3 — project a manufactured
  superluminal 1D profile (e.g. h ∝ ρ^{1.5} above x_c) and verify c_s² = cap and
  the ε/h/p identities to roundoff; envelope no-op on a causal profile.
- **Synthetic (CI)**: extend the dirty preset with a stiffened high-ρ corner
  (add a smooth `A·(ρ/ρ_c)^α` term to ε above ρ_c, α chosen so the constructed
  U is superluminal at the top cells for all T). Full narrative: detect →
  causal-cap → adapter audit class C clean in-box → idempotent second run (zero
  changes).
- **Real tables (local integration)**: on LS220-2009 and SRO — (a) node map
  before/after: corner counts (67,107 LS220 / 182,269 SRO) → 0 expected,
  interior spikes reported untouched; (b) `eos_test --level adapter` class C `cs2_acausal` reduced to the
  pocket residuals; (c) **the M3 acceptance metric**: `eos_test --level con2prim`
  cold/warm failure tails on the capped table — the hot-edge/high-w tail should
  collapse; report the measured numbers in CODE.md like every other milestone.
- **Idempotence**: second `eos_repair` run reports zero changes when the first
  ended clean at (4,4,4) — same divisibility argument as the 3D stage (main-loop
  refine divides 4), plus the projection's own `ε_new < ε_old` write condition.
- **Byte-faithfulness**: untouched datasets and untouched `logenergy` nodes remain
  bit-identical; `entropy` never gains an entry from this stage.

## 8. Alternatives considered and rejected

- **Local diffusion (the literal reuse)**: measured runaway on this exact defect
  (§2); wrong tool for a systematic slope excess.
- **Shape-constrained (QP) refit with a causality constraint**: the constraint is
  nonlinear in the coefficients (ratio of second derivatives through the σ chain
  rule); heavier than the M2 QP idea that was already deferred, for no clearer
  guarantee than the audit loop provides.
- **Run-time cs2 clamp in the adapter**: breaks single-potential consistency
  (§1); the Newton would consume derivatives that belong to no U.
- **Leave it entirely to `state_policy` guards**: works (never fails) but leaves
  the measured failure tail and its fallback cost in place, and a GPU
  fixed-iteration path (M4) really wants conditioning everywhere in-box.

## 9. Open items

1. ~~Stage default~~ — decided 2026-08-27: **on**, with `--no-causal-cap` to opt
   out.
2. ~~`cs2_cap`~~ — decided 2026-08-27: **0.99** (revisit only if the measured
   fidelity cost or residual ringing says otherwise).
3. ~~SRO map~~ — done (§2): same suffix corner, depth ≤ 18, stored relativistic
   cs2 agrees 100%; `trace_depth_max = 64` stands. SRO's interior sliver at
   (ρ ≈ 2.7×10¹³, Ye = 0.645) is v1-untouched by the scoping rule; if M3 testing
   ever shows real recovery failures there, it is a candidate for a *bounded*
   local x-smoothing of the seam (narrow, so no runaway) — deferred with the
   other pocket residuals.
4. Whether the adapter's x_hi extension tail, relaunched from capped boundary
   values, is itself causal — re-measure with the extended soak
   (`soak_extended`) after the repair; the OOB_RHO_HIGH hard-invalid contract is
   unchanged either way. **Now more pressing** (§10): the con2prim failure tail
   survived the in-box repair, so the extension tail is one of the few remaining
   suspects.

## 10. Implemented — measured results (2026-08-27)

The stage landed as specified: audit of `c_s²` on the fitted splines through the
§3 chain rule, edge-anchored scoping, per-node adiabat trace to a causal anchor,
Lipschitz envelope plus the ε-ODE (integrated for the deficit `D = ε_orig − ε_env`,
which is algebraically the §3 ODE but leaves an already-causal stretch bitwise
untouched — the one implementation refinement worth naming, since it is what makes
the stage idempotent), per-column re-repair of touched columns, best-state rounds,
and the lexicographic backstop. On the two local tables it removes the corner this
document was written about: acausal **nodes 67,107 → 206 (LS220-2009)** and
**182,269 → 575 (SRO)**, with the defining structure — 6,792 (T, Ye) columns whose
acausal set is a contiguous suffix to ρ_max — reduced to **zero such columns**;
refined (4,4,4) violation samples 4,076,091 → 7,622 and 11,188,863 → 21,283, of
which 98.8% / 99.5% are the interior σ_T-pocket runs §4 step 2 scopes out on
purpose. Only `logenergy` moves (66,419 / 182,210 nodes, max |Δ log₁₀(ε+Δ)| ≈ 0.08);
`entropy` gains no entry, `c_s² ≤ 0` counts are unchanged, `L_u` monotonicity is
unchanged (LS220) or better (SRO), both tables stay idempotent and thread-count
independent, and the whole `eos_repair` run costs 24 s / 76 s serially. The
adapter's own audit confirms it independently: class C `cs2_acausal` 7,364 → 132 and
7,639 → 135. Two findings argue for revisiting decisions this document closed.
First, **`cs2_cap = 0.99` is not the best point on its own curve**: 0.95 leaves only
16 residual nodes on LS220 (vs 206), while 0.97 and 0.90 are *reverted wholesale* by
the §4 step 6 backstop because their projections regress the (4,4,4) `L_u` count by
2 and 33 samples out of 20,650 — a rule that, as written, trades away a 5,000×
causality improvement for a 0.01% monotonicity regression, and makes the outcome
discontinuous in the cap. Second, **§7's acceptance metric did not move**: the M3
con2prim failure tail is 87 → 86 (LS220 warm) and 162 → 163 (SRO warm) per 40,000
states, so that tail is *not* caused by in-box `c_s² ≥ 1`. The stage's justification
therefore rests on the conditioning and table-validity arguments of §1, which stand,
rather than on a measured failure-rate win; the tail's remaining suspects are the σ_T
pockets, the x_hi extension tail relaunched from capped values (item 4 above), and
the bracketing paths themselves. A third, smaller finding: the projection controls ε
at nodes while `c_s²` is a second derivative of the fit, so the stage needs
`(cs2_cap·Δx)²/12` to sit inside the `cs2_max`/`cs2_cap` hysteresis — true on real
tables (~0.1%), false on coarse synthetic grids (~3% at 0.26 dex/cell), which is why
the CI preset only asserts detection and reduction while the ρ-resolved unit test
asserts zero.
