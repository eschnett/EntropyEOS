# EOS Adapter: $F(\rho, T, Y_e) \to U(\rho, s, Y_e)$ — Design Notes

Draft v0.1, August 24, 2026. Companion to `con2prim-entropy-rapidity.md` (this is its
deliverable 1). Working document; everything is up for iteration.

## 1. Problem statement

The con2prim design consumes the EOS through a single thermodynamic potential

$$\epsilon = U(\rho, s, Y_e), \qquad
p = \rho^2 U_\rho, \quad \hat T = U_s, \quad \tilde\mu = U_{Y_e},$$

and needs $U, U_\rho, U_s$ per evaluation plus $U_{\rho\rho}, U_{\rho s}$ for the Jacobian,
with $U \in C^2$, exact internal consistency, $0 < c_s^2 < 1$, and (per the resolved open
questions of the parent document) $\epsilon \ge 0$ everywhere, per-baryon entropy $s$ in
$k_B$ iterated linearly, $Y_e$ as the composition variable, the sub-$\rho_{\min}$ extension
owned by the EOS layer, and a CPU prototype (C++17; see `CODE.md`) before a GPU port.

What is actually available is "the usual" table $F$: a rectangular grid
$(\rho_i, T_j, Y_{e,k})$, typically log-spaced in $\rho$ and $T$, held in memory, storing
at least

- specific internal energy $\epsilon_F(\rho, T, Y_e)$, stored as
  $\log(\epsilon_F + \Delta)$ with a constant `energy_shift` $\Delta > 0$ chosen so the
  argument is positive ($\epsilon_F$ itself may be negative at low $T$),
- entropy per baryon $s = \sigma_F(\rho, T, Y_e)$ in $k_B$,
- pressure $p_F$ and more (chemical potentials, $c_s^2$, …), which the adapter uses **only
  for audits**, not for construction,
- metadata: units, the baryon mass $m_B$ used to define $\rho = m_B n_B$, and $\Delta$.

This note designs the adapter that wraps such an $F$ and presents $U$ with its required
derivatives. Headline decisions: derivatives are computed **on the fly** through an
analytic chain rule around a 1D temperature inversion (alternatives in §9); the
`energy_shift` and the energy zero point are absorbed at build time by a **baryon-mass
rescaling**, so $U$ is shift-free and $\epsilon \ge 0$ by construction (§5); out-of-bounds
is handled by **smooth extension plus flags, never clamping mid-iteration** (§7).

## 2. Core construction

Fix $Y_e$ (it enters as a third tensor-product direction but is not iterated). Let
$e(\rho, T, Y_e)$ and $\sigma(\rho, T, Y_e)$ be smooth interpolants of the table's energy
and entropy columns. Define $T(\rho, s, Y_e)$ implicitly by

$$\sigma(\rho,\, T(\rho,s,Y_e),\, Y_e) = s,$$

which exists and is unique iff $\sigma_T > 0$ (the $c_v > 0$ condition, §8). Then

$$\boxed{\;U(\rho, s, Y_e) \;=\; e\big(\rho,\, T(\rho,s,Y_e),\, Y_e\big).\;}$$

Two observations do most of the work:

1. **Consistency is structural, not inherited.** $U$ so defined is a genuine single
   function of $(\rho, s, Y_e)$; its mixed partials commute and $p := \rho^2 U_\rho$,
   $\hat T := U_s$ are exactly consistent with $U$ *regardless of whether $F$ satisfies
   its own Maxwell relations*. Table inconsistency does not break the solver — it only
   makes the derived $p, \hat T$ deviate from the table's stored $p_F, T$ columns
   (a fidelity issue, measured by the audits of §10, with an upgrade path in §9.4).
2. **Only $e$ and $\sigma$ are used.** The construction deliberately ignores $p_F$; using
   it would reintroduce a second, potentially disagreeing source of pressure — exactly
   what the single-potential interface is designed to eliminate.

The adapter therefore splits into a **build stage** (once per table: read, convert units,
repair, fit splines, extend, re-zero energy, audit — §§4–7) and a **run stage** (per
evaluation: one warm-started 1D solve plus one chain-rule assembly — §3).

## 3. Run-time evaluation

Work in log variables $x = \ln\rho$, $u = \ln T$. One call:

**`evaluate(ρ, s, Y_e; T_guess) → EOSPoint`**

1. Clamp $Y_e$ to $[Y_{\min}, Y_{\max}]$; set flag if clamped ($Y_e$ is exact and not
   iterated, so a clamp here is a property of the input state, not of an iterate).
2. Solve $\sigma(x, u, Y_e) = s$ for $u$: safeguarded Newton in $u$ on the (extended,
   §7) bracket, warm-started from `T_guess`. Monotonicity $\sigma_u > 0$ makes the
   bracketed solve globally convergent; warm starts (previous con2prim iterate, previous
   timestep's $T$) make 2–3 iterations typical. Solve to near machine precision so the
   inversion contributes no error visible to the con2prim Newton.
3. Evaluate $e, \sigma$ and their $(\rho, T)$ derivatives up to second order (plus first
   $Y_e$ derivatives if $\tilde\mu$ is requested) at the solved point.
4. Assemble $U$ and derivatives by the chain rule below.
5. Return derived quantities and flags.

### 3.1 Chain rule

With all $e, \sigma$ partials taken at fixed $(\rho, T, Y_e)$-arguments, the implicit
function theorem gives the derivatives of the inverse map,

$$T_s = \frac{1}{\sigma_T}, \qquad
T_\rho = -\frac{\sigma_\rho}{\sigma_T}, \qquad
T_{ss} = -\frac{\sigma_{TT}}{\sigma_T^3}, \qquad
T_{\rho s} = -\frac{\sigma_{\rho T} + \sigma_{TT} T_\rho}{\sigma_T^2}, \qquad
T_{\rho\rho} = -\frac{\sigma_{\rho\rho} + 2\sigma_{\rho T} T_\rho + \sigma_{TT} T_\rho^2}{\sigma_T},$$

and then

$$U = e, \qquad
U_s = e_T\, T_s, \qquad
U_\rho = e_\rho + e_T\, T_\rho,$$

$$U_{\rho s} = e_{\rho T} T_s + e_{TT}\, T_\rho T_s + e_T\, T_{\rho s},$$

$$U_{\rho\rho} = e_{\rho\rho} + 2 e_{\rho T} T_\rho + e_{TT} T_\rho^2 + e_T\, T_{\rho\rho},$$

$$U_{ss} = e_{TT} T_s^2 + e_T T_{ss} \quad\text{(not needed by the solver Jacobian; used
by extensions and audits)}, \qquad
U_{Y_e} = e_{Y_e} + e_T\, T_{Y_e}, \quad T_{Y_e} = -\sigma_{Y_e}/\sigma_T.$$

Sanity identities (cheap to verify in tests): computing $U_{\rho s}$ as
$\partial_s U_\rho$ instead gives
$e_{\rho T} T_s - (e_{TT}\sigma_T - e_T \sigma_{TT})\sigma_\rho T_s/\sigma_T^2
- e_T \sigma_{\rho T} T_s / \sigma_T$, which is identical — mixed partials commute by
construction. If $F$ happens to be exactly consistent
($e_T = \hat T \sigma_T$, $e_\rho = \hat T \sigma_\rho + p/\rho^2$), the formulas collapse
to $U_s = \hat T$ and $U_\rho = p/\rho^2$, i.e. the adapter reproduces the table's own
$p, T$.

Everything the solver needs per evaluation is therefore **six values from each of two
interpolants** ($e, e_x, e_u, e_{xx}, e_{xu}, e_{uu}$ and the same for $\sigma$) plus a
handful of arithmetic. Log-axis conversions:
$e_\rho = e_x/\rho$, $e_T = e_u/T$, $e_{\rho\rho} = (e_{xx} - e_x)/\rho^2$,
$e_{TT} = (e_{uu} - e_u)/T^2$, $e_{\rho T} = e_{xu}/(\rho T)$.

### 3.2 Derived quantities and outputs

$$\hat T = U_s, \qquad p = \rho^2 U_\rho, \qquad h = 1 + U + \rho U_\rho, \qquad
h c_s^2 = 2\rho U_\rho + \rho^2 U_{\rho\rho}.$$

The returned `EOSPoint` carries
$(U, U_\rho, U_s, U_{\rho\rho}, U_{\rho s};\; \hat T, p, h, c_s^2;\; T_F;\;
\tilde\mu\ \text{optional};\; \text{flags};\; u\ \text{for warm-starting the next call})$.

**Two temperatures.** $T_F$ is the solved table temperature (argument of $F$); $\hat T =
U_s$ is the thermodynamic conjugate of $s$ under $U$. They agree exactly iff $F$ is
consistent; otherwise they differ at the level of the table's inconsistency. The solver
must use $\hat T$ internally (its identities $z_s = D\cosh w\,(\hat T + \rho U_{\rho s})$
etc. assume it), while $T_F$ is the right value to report as the primitive $T$ output,
because downstream tabulated physics (neutrino opacities etc.) is indexed by the table's
own temperature coordinate. Return both; audit their relative difference (§10).

### 3.3 Cost and GPU notes

Per evaluation: $N_{\rm Newton}$ cheap 1D evaluations of $(\sigma, \sigma_u)$ plus one
full 12-derivative evaluation. Optimization: during the $u$-solve, $x$ and $Y_e$ are
fixed, so the tensor-product spline can be collapsed once per visited cell to a 1D cubic
in $u$, making Newton iterations nearly free. Warm-started, the whole call should cost a
small multiple of a plain tricubic lookup. For GPU (fixed-cost, branch-free), a fixed
iteration count (e.g. warm start + 3 safeguarded Newton steps, with the safeguard as a
predicated bisection step, not a branch) is viable because $\sigma_u > 0$ is guaranteed by
construction; measure the worst-case residual in the audit suite before trusting it, and
see §9 for tabulated alternatives that remove the inner solve entirely.

## 4. Table representation

- **Spline family:** tensor-product **cubic B-splines** with knots at the data points, in
  $(x, u, Y_e)$. B-splines are $C^2$; local schemes (Catmull–Rom, Lekien–Marsden tricubic)
  are only $C^1$ and would put jumps into $U_{\rho\rho}, U_{\rho s}$, degrading the
  con2prim Newton — they are not acceptable here. Cubic B-spline second derivatives are
  piecewise linear (continuous, kinked at knots), which is sufficient; quintic B-splines
  are the smoothness upgrade if audits show the Jacobian quality limits convergence.
- **Fitted quantities:** fit $\hat L(x,u,Y_e) = \ln(\epsilon_F + \Delta)$ (the log-stored
  energy, well-conditioned across its many decades and automatically respecting
  $\epsilon_F + \Delta > 0$) and $\sigma(x,u,Y_e) = s$ linearly (its range is modest and
  it must be allowed to approach 0). The exponential is undone analytically:
  $\epsilon_F = e^{\hat L} - \Delta$,
  $\partial \epsilon_F = (\epsilon_F + \Delta)\,\partial\hat L$,
  $\partial^2 \epsilon_F = (\epsilon_F + \Delta)(\partial^2\hat L + \partial\hat L\,
  \partial\hat L)$, composed before the §3.1 chain rule.
- **Monotonicity repair:** before fitting, enforce $\sigma$ and $\epsilon_F$ strictly
  increasing in $T$ along every $(\rho, Y_e)$ column (isotonic regression with a small
  minimum slope, applied to the data; both columns need it, since $U_s = e_T/\sigma_T$
  must be positive). Then fit unconstrained B-splines and **audit** $\sigma_u > 0$,
  $e_u > 0$ on a fine grid; where the spline overshoots into non-monotonicity, apply
  local smoothing/tension and refit. (The alternative — shape-constrained $C^2$ spline
  fitting as a small QP — is cleaner but heavier; start with the audit loop.) Log every
  modified cell: repair is a physics edit and must be visible.
- **$Y_e$ direction:** cubic B-spline as well (gives smooth $U$ across $Y_e$ cells and a
  usable $\tilde\mu$); only first $Y_e$-derivatives are ever needed.

## 5. Energy shift and the energy zero point

Two distinct issues hide under "energy_shift":

1. **Storage shift.** $\Delta$ exists only to make $\log(\epsilon_F + \Delta)$ well
   defined. It is undone analytically inside the adapter (§4) and never appears in the
   interface: $U$ is shift-free as required.
2. **Physical zero point.** After unshifting, $\epsilon_F$ is still negative in cold
   regions for typical tables (energy measured relative to a free-nucleon or amu
   baseline), while the cancellation-free $\tau$ residual of the con2prim design wants
   $\epsilon \ge 0$. A constant shift of $\epsilon$ alone is *not* free — it changes
   $\rho(1+\epsilon)$ and hence the physics. The invariant way to re-zero is to **rescale
   the baryon mass**: the true energy per baryon is $m_B(1 + \epsilon_F)$; choose

   $$m_B^\ast = \kappa\, m_B, \qquad
   \kappa = 1 + \epsilon_{\rm floor}, \qquad
   \epsilon_{\rm floor} \le \min \epsilon_F \ \text{(see below)},$$

   and define the adapter's density and energy as

   $$\rho^\ast = \kappa\,\rho, \qquad
   U = \frac{1 + \epsilon_F}{\kappa} - 1 \;\ge\; 0.$$

   This is exact, not an approximation: per baryon, $m_B^\ast(1 + U) = m_B(1+\epsilon_F)$.
   Consequently $p$, $T$, $s$, $Y_e$, $c_s^2$, the total energy density
   $\rho^\ast(1+U) = \rho(1+\epsilon_F)$, and $\rho^\ast h^\ast = \rho h$ are all
   invariant; only the rest-mass bookkeeping moves: $D^\ast = \kappa D$,
   $\tau^\ast = E - D^\ast$, $\hat T^\ast = k_B T/(m_B^\ast c^2)$, $h^\ast = h/\kappa$.
   Implementation is a relabeling at build time: shift the log-density grid by
   $\ln\kappa$ and transform the stored energy affinely; no refit needed.

**Choosing $\epsilon_{\rm floor}$:** take the minimum of $\epsilon_F$ over a *fine
sampling of the fitted spline including the §7 extensions* — not over the raw data, since
$C^2$ splines can undershoot the data minimum and the cold extension can dip below
$\epsilon_F(T_{\min})$ — then subtract a small safety margin. Store $\kappa$ (and
$m_B^\ast$, exactly) in the adapter's metadata.

**Contract with the evolution code:** $D$, $\rho$, and the initial data must use the same
$m_B^\ast$ convention, and $\kappa$ is part of the EOS identity (checkpoint compatibility:
a table swap that changes $\kappa$ changes $D$). The adapter exports $\kappa$ and
$m_B^\ast$; prim2con and importers consume them. From here on, all solver-facing formulas
say $\rho$ and mean $\rho^\ast$.

## 6. Units and conventions

- Table-native units (e.g. $\rho$ in g cm⁻³, $T$ in MeV, $\epsilon$ in erg g⁻¹, $s$ in
  $k_B$/baryon) are converted to the solver's geometric units once, at build time; the
  run-time path is unit-free.
- With $s$ per baryon in $k_B$ and $\epsilon$ per unit rest mass, the conjugate
  temperature is dimensionless: $\hat T = U_s = k_B T/(m_B^\ast c^2)$
  ($\approx T[\mathrm{MeV}]/931.494\,\kappa^{-1}\dots$ — fold the exact factor into build).
  This *is* the $T$ appearing in the parent document's identities. $T_F$ in physical
  units is reported alongside (§3.2).
- $\tilde\mu = U_{Y_e}$ is likewise per baryon in units of $m_B^\ast c^2$. It is derived
  from $e, \sigma$ alone; the table's chemical-potential columns (with their zoo of
  rest-mass conventions) are never consumed.
- The table's stated $m_B$ (its `mass_factor` relating $\rho$ to $n_B$) must be read and
  honored — tables differ ($m_n$, $m_u$, …), and a silent mismatch is a classic
  percent-level bug.

## 7. Out-of-bounds handling

Guiding principle: **during iteration, the adapter must always return finite, smooth,
monotone, causal values — extension plus flag, never a hard clamp** (a clamp zeroes
derivatives and stalls or misleads Newton). Validity is judged once, on the *converged*
state, by the con2prim failure policies (parent §11): a converged state carrying an
extension flag is handled there (clamp-and-flag, atmosphere, or error), not inside the EOS
call.

("Causal" was added in M3g: the inner $w$-solve's monotonicity proof rests on
$c_s^2 < 1$, so an acausal *extension* breaks it just as surely as acausal table data
would. See `eos-causal-tail.md` for the log-$\sigma$ hot tail and the causal slope clamp
that make the $u$-high extension obey it, and its §5 for the M3i follow-up that does the
same for the $\rho$-low one.)

All extensions live at the **$F$ level** — the splines $\hat L$ and $\sigma$ are extended
in $(x, u)$ beyond the physical box — so the run-time path (§3) stays uniform and the
chain rule never sees a special case. Flags are set by comparing the solved $(x, u, Y_e)$
against the physical box.

Per boundary:

- **$s$ below range ($T < T_{\min}$), the common cold case.** Extend both splines below
  $u_{\min}$: $\sigma$ with a linear-in-$u$ tail (slope $\sigma_u(u_{\min}) > 0$, blended
  $C^2$ over a buffer), so $\sigma \to -\infty$ as $u \to -\infty$ and *every* $s <
  s_{\min}(\rho, Y_e)$ maps to a finite $T$ — the inner solve cannot fail; and $\hat L$
  (i.e. $\epsilon$) with an exponentially *decaying* slope,
  $e_u(u) = e_u(u_{\min})\, e^{(u - u_{\min})/\lambda}$ with $\lambda$ matched to the
  boundary curvature (clamped positive), so $\epsilon$ tends to a finite limit
  $\epsilon(T_{\min}) - \lambda\, e_u(u_{\min})$ instead of diverging to $-\infty$. That
  limit participates in the $\epsilon_{\rm floor}$ scan (§5), preserving $U \ge 0$; the
  decaying $e_u$ keeps $\hat T > 0$. Cap the solve at some $u_{\rm floor}$
  ($T \sim 10^{-6}$ MeV) and flag `EXT_S_LOW`.
- **$s$ above range ($T > T_{\max}$).** Same construction mirrored: linear-in-$u$ growth
  of $\sigma$ and of $\hat L$ (monotone, $C^2$-blended); flag `EXT_S_HIGH`. Excursions
  here are transient Newton overshoots or genuinely pathological states; the $w$-cap and
  §11 policies own the latter.
- **$\rho$ below range.** The adapter owns this per the resolved open question 7: extend
  in $x$ below $x_{\min}$ toward the correct **low-density asymptotics**, as $C^2$
  curvature-limited blends to linear tails. Since M3i that means *log-linear in the
  values on both fields*: $\ln\sigma$ and $\ln\epsilon$ are continued affinely in
  $x = \log_{10}\rho$, i.e. $\sigma$ and $\epsilon$ become power laws in $\rho$ with the
  seam's own exponents. That is the radiation-dominated behaviour at a hot seam
  ($\sigma, \epsilon \propto 1/\rho$, so $q = \partial\ln W/\partial x|_s = -1 + 4/3 = 1/3$
  and $c_s^2 \to 1/3$), and it *degenerates* to the old ideal-gas target at a cold seam,
  where the measured seam slopes are $\sigma_x < 0$ (entropy grows as density drops) and
  $e_x \approx 0$ (energy density-independent at fixed $T$) all by themselves. Before
  M3i, $e_x \to 0$ was **imposed** at every seam by a slope-to-zero override; on both
  real tables that was the measured cause of $c_s^2 = 4/3$ across ~50–58% of this band,
  and of a $c_s^2 \le 0$ population inside its blend cell (the override reaches its
  target by overriding the *curvature*). See `eos-causal-tail.md` §5 and CODE.md's "M3i
  empirical findings". $\sigma$'s log tail carries a guard for the deep-cold corner where
  the $u$-low tail has already driven $\sigma$ to (or through) zero and $\ln\sigma$ has
  no meaning: there it falls back to the plain linear tail
  (`core/adapter_eval.hpp`'s `aeval_xlow_log_ok()`; it never fires on the real tables).
  This is where the atmosphere lives; the *policy* (floors, resets) stays with the
  caller. Note that common tables reach $\rho_{\min} \sim 10^3\,$g cm⁻³, below typical
  atmosphere floors, in which case this extension is rarely exercised — but it must exist
  so Newton can pass through. Flag `EXT_RHO_LOW`.
- **$\rho$ above range.** No defensible physical extension (causality is at risk).
  Extend $C^1$-linearly in $x$ just so the iteration remains finite and can converge to a
  reportable point, and flag `OOB_RHO_HIGH` as a hard failure: a converged state here is
  invalid, full stop.
- **$Y_e$ outside range.** $Y_e = D_Y/D$ is computed once, before the solve; clamp to the
  table range and flag `CLAMP_YE`. No smooth extension needed since $Y_e$ is never
  iterated.

For the solver's bracketing and validity tests, the adapter also exports the pointwise
physical entropy range,

$$s_{\min}(\rho, Y_e) = \sigma(x, u_{\min}, Y_e), \qquad
s_{\max}(\rho, Y_e) = \sigma(x, u_{\max}, Y_e),$$

as cheap spline evaluations, plus the global box. (The physical domain in $(\rho, s)$ is
*not* rectangular — $s_{\max}$ at $\rho_{\max}$ is far below $s_{\max}$ at $\rho_{\min}$ —
which matters for the retabulation alternatives of §9.)

## 8. Requirements on $F$

Mapping the parent document's requirements on $U$ to sufficient conditions on $F$:

| $U$ requirement (parent §3, §9) | Condition on $F$ |
|---|---|
| $T(\rho,s,Y_e)$ exists, unique; inner solve well-posed | $\sigma_T > 0$ everywhere ($c_v > 0$), after repair |
| $\hat T = U_s > 0$ | $e_T > 0$ everywhere, after repair |
| $U \in C^2$ | data sampled from a (piecewise) $C^2$ surface; $C^2$ spline representation (§4); adequate grid resolution |
| $p = \rho^2 U_\rho > 0$ (τ-form positivity) | $\partial\epsilon/\partial\rho|_{s} > 0$; for a consistent table this is $p_F > 0$ — audited on the constructed $U$ |
| $0 < c_s^2 < 1$ (causality; $z_w > 0$ conditioning) | condition on second derivatives of the constructed $U$ — audited, not assumed |
| $\epsilon \ge 0$ (cancellation-free $\tau$) | $\epsilon_F$ bounded below; handled by the $\kappa$ re-zeroing (§5) |
| exact $p, T, \epsilon$ mutual consistency | automatic for any single-potential construction — **no** consistency requirement on $F$ for solver correctness |

Hard requirements on the input table:

1. Complete rectangular grid, strictly monotone axes, no NaN/masked holes (β-equilibrium
   or otherwise masked regions must be filled before use).
2. Declared conventions: units, $m_B$ (mass factor), `energy_shift` $\Delta$ with
   $\epsilon_F + \Delta > 0$ on all points, $s$ per baryon in $k_B$.
3. Monotonicity in $T$ of both $\epsilon_F$ and $s$ at every $(\rho, Y_e)$ — up to
   repairable, localized violations (noise, non-NSE seams). Widespread violations mean
   the table cannot support an entropy-based scheme at all.
4. Domain covering every $(\rho, T, Y_e)$ the evolution can reach, with $T_{\min}$ low
   enough that the coldest expected physical states lie *inside* the table ($s_{\min}$
   comfortably below the entropies of cold NS matter), so the cold extension is an escape
   hatch, not a running mode.
5. First-order phase transitions, if present, entering the table already
   Maxwell-constructed (mixed-phase smoothed); raw two-branch data would alias into
   spline ringing that no generic repair fixes.

Soft requirements (quality, measured not assumed):

6. Approximate thermodynamic consistency — the Maxwell checks
   $e_T \approx \hat T \sigma_T$ and
   $e_\rho \approx (p_F - T\, \partial p_F/\partial T)/\rho^2$
   (equivalently $\sigma_\rho \approx -(\partial p_F/\partial T)/\rho^2$). Deviations do
   not break the solver but open a gap between $(\hat T, p)$ and the table's stored
   columns; if the audit exceeds tolerance, use the §9.4 refit.
7. $T$-resolution sufficient that grid-halving changes interpolated $s, \epsilon$ below
   the target tolerance, in particular across the nuclear-dissociation region where $c_v$
   has structure.

## 9. Alternatives to the on-the-fly inversion

The nested $T$-solve is the reference design (simple, exact, no new tables). Its cost is
modest on CPU but its variable iteration count is the one genuinely GPU-unfriendly
feature. Alternatives, all built and validated against the reference adapter:

1. **Tabulated inverse map.** Precompute $\tilde T(x, \theta, Y_e)$ as a $C^2$ B-spline,
   where $\theta = (s - s_{\min}(x,Y_e))/(s_{\max}(x,Y_e) - s_{\min}(x,Y_e))$ is a
   normalized entropy coordinate (with $s_{\min}, s_{\max}$ themselves stored as $C^2$
   splines — this sidesteps the non-rectangular $(\rho,s)$ domain of §7). Evaluation
   composes $e(\rho, \tilde T(\cdot), Y_e)$ with the same chain rule, now needing
   $\tilde T$'s derivatives instead of a Newton loop: **fixed cost, no solve**. Crucially,
   consistency survives: $U$ is still a single smooth function (a composition of splines),
   merely with $\tilde T$ an *approximate* inverse — the error appears as a small offset
   between the $s$-argument and true entropy, of the same order as any retabulation error.
2. **Full retabulation.** Sample the reference adapter on an $(x, \theta, Y_e)$ grid and
   fit one direct B-spline for $U$. Cheapest possible evaluation (one lookup, exactly the
   parent document's "tensor-product splines in $(\log\rho, s, Y_e)$" with $\theta$
   standing in for $s$); costs a second layer of interpolation error and the
   coordinate-transform chain-rule terms from $s_{\min}, s_{\max}$. Consistency again
   exact by construction.
3. **Recommendation:** reference adapter (on-the-fly) for the CPU prototype and as ground
   truth; then benchmark 1 vs 2 for the GPU port and promote whichever meets the accuracy
   audit — likely 2, with 1 as the fallback if the double interpolation layer is too
   lossy.
4. **Orthogonal upgrade — consistent refit at the $F$ level.** If the Maxwell audits of
   §8/§10 show unacceptable inconsistency, fit a single free-energy potential
   $f(\rho, T, Y_e) = \epsilon - \hat T s$ to the table (Timmes–Swesty style) and *define*
   $\sigma := -f_{\hat T}$, $e := f - \hat T f_{\hat T}$ before applying §2. Then $F$
   itself becomes consistent, $\hat T = T_F$ exactly, and the two-temperatures issue of
   §3.2 disappears — at the price of least-squares deviations from the raw $\epsilon, s$
   columns. This slots in without changing anything downstream. Two caveats: (i) no
   standard format stores $f$, so it is assembled pointwise from the same two columns —
   the single-potential route reduces the number of *fitted* objects, not the number of
   columns read; (ii) since $e, \sigma$ become first derivatives of $f$, the solver's
   $U_{\rho\rho}, U_{\rho s}$ need **third** derivatives of $f$
   ($\sigma_{TT} = -f_{\hat T\hat T\hat T}$), which a cubic spline cannot supply
   continuously — the refit requires quintic ($C^4$) tensor-product splines
   ($6^3$-point evaluation instead of $4^3$), exactly why the Helmholtz EOS of Timmes &
   Swesty interpolates biquintically. The two-spline default needs only $C^2$ cubics
   because it never differentiates a fitted quantity more than twice.

## 10. Audits and tests (build-time artifacts)

- **Invertibility:** $\min \sigma_u$ and $\min e_u$ over a fine grid (post-repair,
  post-fit) strictly positive; map of repaired cells with magnitudes.
- **Causality/convexity:** maps of $c_s^2$ and $p$ from the constructed $U$; flag
  $c_s^2 \notin (0,1)$, $p \le 0$, $U_s \le 0$; iterate smoothing where violated (this is
  the parent document's "spline-quality checks" made concrete).
- **Consistency (fidelity):** maps of $\delta_T = \hat T/(k_B T_F/m_B^\ast c^2) - 1$ and
  $\delta_p = \rho^2 U_\rho / p_F - 1$; thresholds decide whether §9.4 is needed.
- **Zero point:** spline-sampled $\min \epsilon_F$ including extensions; verify
  $U \ge 0$ after re-zeroing.
- **Derivative correctness:** finite-difference checks of $U_\rho, U_s, U_{\rho\rho},
  U_{\rho s}$ against the analytic chain rule; the two independent expansions of
  $U_{\rho s}$ (§3.1) agree to roundoff.
- **Round trip:** for table nodes $(\rho, T, Y_e)$: $s := \sigma(\rho,T,Y_e)$, then
  `evaluate`$(\rho, s, Y_e)$ returns $U = e(\rho,T,Y_e)$ and $T_F = T$ to solve tolerance.
- **Manufactured EOS:** run the full build on an analytic gas (ideal nucleons +
  electrons + photons) tabulated at matching resolution; measure convergence of
  $U$ and all derivatives under grid refinement.
- **Warm-start statistics:** iteration-count histograms for cold starts vs. warm starts,
  feeding the GPU fixed-count decision (§3.3).

## 11. API sketch (C++17; layout and build rules in `CODE.md`)

```cpp
namespace eeos {

struct EOSPoint {
  real U, U_rho, U_s, U_rhorho, U_rhos; // what the con2prim Newton consumes
  real That, p, h, cs2;                 // derived (That = U_s)
  real T_F;                             // solved table temperature (cross-table physics)
  real mu_tilde;                        // optional
  real u_solved;                        // ln T, warm start for the next call
  unsigned flags;   // CLAMP_YE | EXT_S_LOW | EXT_S_HIGH | EXT_RHO_LOW | OOB_RHO_HIGH | MAXITER
};

// Host side: owns spline coefficients; built once from a (repaired) RawTable.
class EntropyEOS;   // holds kappa, m_B_star, physical box, audit metadata
EntropyEOS build_entropy_eos(const RawTable&, const BuildOptions&);

// Device-portable POD view (raw pointers into the coefficient arrays):
struct EntropyEOSView {
  EEOS_HOST_DEVICE EOSPoint evaluate(real rho, real s, real ye, real u_guess) const;
  EEOS_HOST_DEVICE SRange   srange(real rho, real ye) const;  // solver bracketing
  // global box, kappa, m_B_star as public members
};

} // namespace eeos
```

Warm-start state is threaded explicitly through arguments/returns (no hidden mutable
state); `evaluate` is pure, allocation-free, exception-free, and host/device-compilable —
the con2prim kernel keeps `u_solved` in a register between its Newton iterations.

## 12. Open questions

1. Extension asymptotics: are the ideal-gas targets for the low-$\rho$ tail (§7) good
   enough, or should the tail be matched to an actual analytic low-density EOS
   (nucleons + e± + γ) per $(T, Y_e)$?
2. Consistency threshold: at what $\delta_T, \delta_p$ do we trigger the free-energy
   refit (§9.4)? Proposal: $10^{-3}$ RMS / $10^{-2}$ max over the region
   $\rho > 10^{10}\,$g cm⁻³.
3. Monotone fitting: audit-loop smoothing (simple) vs. shape-constrained QP fit (clean) —
   decide after seeing how noisy real tables are under cubic B-splines.
4. Quintic B-splines for a $C^2$-smooth Jacobian (vs. cubic's knot kinks in second
   derivatives): measure whether con2prim iteration counts care before paying for it.
5. Retabulation grid ($\theta$ resolution, $s$ vs. $\log$-like spacing inside $\theta$)
   for the GPU path — defer to benchmarks against the reference adapter.
6. Should `evaluate` optionally skip second derivatives (residual-only calls in a
   line-search / bisection fallback) as a fast path?

## 13. Relation to prior work

Deriving all thermodynamics from one interpolated potential follows Timmes & Swesty
(2000), whose Helmholtz EOS tabulates $f(\rho, T)$ and differentiates a biquintic
interpolant precisely to guarantee consistency; the adapter here applies the same
philosophy one variable-change later, in $(\rho, s, Y_e)$. The input-table conventions
(`energy_shift`, log-stored energy, mass factor) follow the stellarcollapse.org format of
O'Connor & Ott (2010); CompOSE tables are analogous modulo metadata. Temperature-recovery
by inverting $s(\rho, T, Y_e)$ is the entropy analogue of the standard
$\epsilon \to T$ inversion every tabulated-EOS con2prim already performs (Siegel et al.
2018); the differences are that the inversion lives *inside the EOS layer* behind a
potential interface, that monotonicity is guaranteed by construction rather than hoped
for, and that the baryon-mass rescaling makes the $\epsilon \ge 0$ zero point exact
rather than an ad-hoc shift.

The lineage runs back one step further, to Swesty (1996), which introduced the
consistent-by-differentiation construction that Timmes & Swesty (2000) then applied. The
pair of requirements in §8 — consistency (derivability from a potential) and stability
($c_s^2 > 0$) — is stated in exactly that form by Dilts (2006), which also observes that
table interfaces typically enforce the second and not the first, and proposes fitting under
thermodynamic constraints; that is the closest published relative of the repair stage.
Baturin et al. (2019) is the methodological template for the fidelity audits of §10. Full
citations and the wider landscape are in [`RELATED.md`](RELATED.md).
