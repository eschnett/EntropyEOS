# EntropyEOS: Entropy-based equation-of-state functions

[![CI](https://github.com/eschnett/EntropyEOS/actions/workflows/ci.yml/badge.svg)](https://github.com/eschnett/EntropyEOS/actions/workflows/ci.yml)

This package `EntropyEOS` provides reliable and efficient functions to
handle tabulated equations of state (EOS), meant for
general-relativistic hydrodynamics calculations running on
high-performance computing systems.

It contains three things that build on each other: tools to check a tabulated EOS and
repair it where it violates the assumptions a hydrodynamics code needs; an adapter that
turns the usual $F(\rho, T, Y_e)$ table into the single thermodynamic potential
$U(\rho, s, Y_e)$ with all the derivatives a solver wants; and a `prim2con`/`con2prim`
pair built on that potential, which uses entropy and rapidity as its iteration variables
and offers a safe path that never fails. The run-time half of the library is
header-only, allocation-free and exception-free, so it compiles for a GPU as well as a
CPU.

The design is written up in three companion documents:
[`con2prim-entropy-rapidity.md`](con2prim-entropy-rapidity.md) for the physics of the
inversion, [`eos-adapter-F-to-U.md`](eos-adapter-F-to-U.md) for the $F \to U$ adapter,
and [`CODE.md`](CODE.md) for the code design, milestones, and measured results.
[`RELATED.md`](RELATED.md) surveys the related software and publications, and says where
this work sits among them.

## Background

### Relativistic Magneto-Hydrodynamics

A GRMHD code evolves *conserved* variables but needs *primitive* variables at every
point of every timestep — to compute fluxes, to call the EOS, to output anything
physical. The conversion between them sits in the innermost loop of the whole
simulation, so it has to be both fast and unconditionally robust. Working with the
undensitized, Valencia-type conserved state measured by the normal observer,

$$(D,\; S_i,\; \tau,\; B^i,\; D_Y), \qquad
D = \rho W, \quad D_Y = \rho Y_e W, \quad \tau = E - D,$$

and writing $z \equiv \rho h W^2$ for the total enthalpy density, the momentum and
energy are

$$S_i = (z + B^2)\, v_i - (B^j v_j)\, B_i, \qquad
E = z - p + \tfrac12 B^2 (1 + v^2) - \tfrac12 (B^i v_i)^2 .$$

The formulation used here consumes the EOS as a *single thermodynamic potential*, the
specific internal energy in its natural variables,

$$\epsilon = U(\rho, s, Y_e), \qquad
p = \rho^2 U_\rho, \quad \hat T = U_s, \quad \tilde\mu = U_{Y_e},
\quad h = 1 + U + \rho U_\rho,$$

and iterates on entropy $s$ (per baryon, in $k_B$) together with the **rapidity** $w$,
where $W = \cosh w$, $v = \tanh w$. Both choices remove constraints from the iterate
space. Any $(s, w)$ in the table domain is a physical state, since $v < 1$ and
$\rho > 0$ are built into the parametrization; $w$ stays well conditioned in both the $v \to 0$ and
$v \to 1$ limits, where velocity- or $W$-based schemes go stiff; and entropy is the
natural coordinate of the table's physics as well as a slowly varying quantity along
smooth flow, which makes the previous timestep an excellent initial guess.

In this form `prim2con` is simple and always works: every conserved variable is an
explicit formula in the primitives, with no iteration and no failure mode. The one
subtlety is arithmetic rather than algebra. Naively forming $\tau = E - D$ subtracts two
large numbers to leave a small one, and in cold slow flow, where $\tau \ll D$, the
result is mostly rounding error. Using $\rho \cosh w = D$ and
$\cosh w - 1 = 2\sinh^2(w/2)$ gives an equivalent expression,

$$\tau = 2D\sinh^2(w/2) + \rho\,\epsilon \cosh^2 w + p \sinh^2 w
+ \tfrac12 B^2 (1 + \tanh^2 w) - \tfrac12 B^2 v_\parallel^2 ,$$

in which every term is individually small and (for $\epsilon, p \ge 0$) nonnegative, so
there is no cancellation.

`con2prim` is the direction that requires numerical inversion. Composition drops out
exactly, $Y_e = D_Y/D$, and $\rho = D/\cosh w$ eliminates the density, leaving two
unknowns $(s, w)$ implicitly defined by two residuals: a momentum residual
$f_1 = \sinh w - \cosh w\, V$ — deliberately not squared, so the $S \to 0$ root stays
simple rather than double — and a relative energy residual $f_2$ built on the $\tau$
form above.

The advantage of this particular formulation is that it guarantees existence and uniqueness
of the solution rather than hoping for it. At fixed $s$,

$$\frac{\partial f_1}{\partial w} \;\ge\; \cosh w\,(1 - V \tanh w) \;>\; 0,
\qquad f_1(0) = -V \le 0, \qquad f_1 \to +\infty \ \text{as}\ w \to \infty,$$

so there is exactly one $w_\star(s)$, bracketed and safe for a bisection-guarded Newton algorithm.
That reduces the outer problem to one dimension in $s$ on $[s_{\min}, s_{\max}]$, where
a sign change can be found by scanning. The production path is still the coupled
$2\times2$ Newton with an analytic Jacobian, for its quadratic convergence and single
EOS call per iteration. The nested scheme is a fallback that is guaranteed to work,
used only when the Newton algorithm fails.

These guarantees are inherited from the EOS, and they hold
*only* if it has the right thermodynamic properties.

| property | why the solver needs it |
|---|---|
| $\sigma_T = \partial s/\partial T > 0$ (i.e. $c_v > 0$) | $T(\rho,s,Y_e)$ exists and is unique, so the inner inversion is well posed |
| $e_T = \partial\epsilon/\partial T > 0$ | $\hat T = U_s > 0$ |
| $p > 0$ | positivity of the $\tau$ form |
| $0 < c_s^2 < 1$ | causality — and conditioning, since the Jacobian's diagonal is $z_w = z(1 - c_s^2)\tanh w$ and the monotonicity argument above rests on $V < 1$ |

Thermodynamic consistency is the one requirement that does not appear in that list,
because the single-potential interface makes it structural: $p$ and $\hat T$ are
derivatives of one function, so they cannot disagree with $\epsilon$ no matter what the
table does. (A table that violates its own Maxwell relations in its stored pressure and
sound-speed columns has an internal inconsistency, but it does not break the solver because these
columns are never used.)

### EOS tables

What is actually available in practice is not a function providing the internal energy $U(\rho, s, Y_e)$,
but a table of thermodynamic functions $F(\rho, T, Y_e)$ on a rectangular grid,
typically log-spaced in $\rho$ and $T$ and linear in $Y_e$, storing
$\log_{10}(\epsilon + \Delta)$ with a constant `energy_shift` $\Delta > 0$ (so the
argument stays positive where $\epsilon$ itself is not), entropy per baryon, pressure,
and a zoo of further columns — chemical potentials, $c_s^2$, $\Gamma$, mass fractions.
Here $F$ names that whole bundle of columns, *not* a Helmholtz free energy. The distinction
matters: a table that did store a free energy would already be a thermodynamic potential,
and consistency would come for free by differentiating it (Swesty 1996). These tables are
precisely not that, which is why consistency has to be re-established by construction
below rather than inherited from the file.
Behind those numbers is a significant amount of nuclear physics: nuclear statistical
equilibrium and its composition, nuclear dissociation, the transition to uniform
matter, possible first-order phase transitions, and the $e^\pm$ and photon
contributions on top.

Two properties of how such tables are made matter more than any of that detail. They
are created **pointwise** — each grid point is an independent solve of the nuclear
model, with nothing in the construction relating one point to the next — and they are
**approximate**, so the stored $p$ and $c_s^2$ need not agree with the stored $\epsilon$
and $s$ to better than the percent level. Smoothness across neighbouring points is a
hope, not a property of the file.

A large domain is also wanted, and that pulls in the opposite direction. The table has
to cover everything the evolution can reach: LS220 spans $\rho$ from $10^3$ to
$10^{16}\,$g cm⁻³, $T$ from $10^{-2}$ to $250\,$MeV, and $Y_e$ from $0.035$ to $0.55$,
which necessarily extends well past where the underlying model is trustworthy. Tables
therefore have unphysical corners *by design*. The high-density corner of both tables
tested here is genuinely acausal — $c_s^2 \ge 1$ at 4.2% of nodes (LS220-2009) and 4.3%
(SRO), as a contiguous suffix in $\rho$ at nearly every $(T, Y_e)$, and confirmed by
those tables' own stored `cs2` columns. This is what the
physics model says out there, it is not tabulation noise.

Nothing makes strong guarantees about gradients and second derivatives either. Pathologies
measured in shipped tables include non-monotone entropy and energy through the nuclear
transition just below saturation density, near-flat `logenergy` plateaus, negative
entropy in a cold high-density corner (6,826 nodes in SRO, none in LS220), and Inf/NaN
values in auxiliary columns. Node-to-node monotonicity
is not even sufficient, because the adapter differentiates a *fitted* $C^2$ spline
rather than the raw data: a long near-plateau followed by a steep recovery makes that
spline oscillate non-monotonically *between* nodes at which every secant slope is positive.

Finally, the table has to be converted from $F$ to $U$. Given smooth interpolants $e$
and $\sigma$ of the energy and entropy columns, we define $T(\rho, s, Y_e)$ implicitly by
$\sigma(\rho, T, Y_e) = s$ — which exists and is unique exactly when $\sigma_T > 0$ —
and set

$$U(\rho, s, Y_e) = e\big(\rho,\, T(\rho, s, Y_e),\, Y_e\big).$$

This definition has two consequences. Consistency becomes structural (i.e. always true), since $U$ is a
genuine single function of $(\rho, s, Y_e)$ whether or not $F$ satisfies its own Maxwell
relations. And only $e$ and $\sigma$ are used: we ignore the stored pressure column,
because consuming it would reintroduce exactly the second,
potentially disagreeing source of $p$ that the single-potential interface
eliminates.

The energy zero point needs one more step. Real tables have $\epsilon_F < 0$ in cold
regions, since energy is measured relative to a free-nucleon or amu baseline, while the
cancellation-free $\tau$ form wants $\epsilon \ge 0$ — and simply shifting $\epsilon$ by
a constant is not free, because it changes $\rho(1+\epsilon)$ and hence the physics. The
invariant fix is to rescale the baryon mass. With $m_B^\ast = \kappa\, m_B$ and
$\kappa = 1 + \epsilon_{\rm floor}$,

$$\rho^\ast = \kappa\rho, \qquad U = \frac{1 + \epsilon_F}{\kappa} - 1 \;\ge\; 0 ,$$

which is exact rather than approximate: per baryon $m_B^\ast(1 + U) = m_B(1+\epsilon_F)$,
so $p$, $T$, $s$, $Y_e$, $c_s^2$, $\rho h$ and the total energy density are all
unchanged and only rest-mass bookkeeping moves ($D^\ast = \kappa D$). Since $\kappa$
becomes part of the EOS identity — a table swap that changes it changes $D$, and hence
checkpoint compatibility — the adapter exports it, so the evolution code and the initial
data can adopt the same convention.

Tables in the stellarcollapse.org (O'Connor–Ott) HDF5 layout can be read directly; LS220,
SRO, DD2 and SFHo are all exercised by the test suite (see
[`tables/README.md`](tables/README.md) for where to download them). CompOSE/MUSES tables
use the same entry point and could be added without touching anything else.

### Numerical inversion

The two obvious algorithms each fall short on their own. Newton's method is fast — quadratic
convergence, with only one EOS evaluation per iteration — but has no global convergence guarantee: it can
overshoot out of the table, converge into the wrong basin, or stall where the residual
surface is not convex. Bisection is unconditionally safe but needs a monotone
function of a single variable and a bracket containing the root, and a
two-dimensional residual pair provides neither. Hence we chose the nested structure described above, with a provably monotone and
bracketed inner $w$-solve under a one-dimensional outer solve in $s$. Even there,
checking only the endpoint signs of the outer function is not enough — it is
non-monotone near domain-extension seams, and the extended $s$-bracket can span ten
decades — so we use a multi-point scan as bracket search: a global sweep plus a local
geometric ladder of relative offsets around the best iterate the Newton reached.

Whether an initial guess is available changes the picture completely. Along smooth flow
entropy and rapidity barely move, so the previous timestep's $(s, w)$ is an excellent
warm start, and 98.8% of warm solves converge in the Newton alone, typically in one or
two steps. A cold start has to manufacture a seed instead, and does so exactly rather
than by tuning: the energy relation is solved exactly for $z$ as a bracketed cubic with
exact physical bounds (so the seed is aware of $B^2$, which a hydro-limit seed is not),
$w$ follows from the momentum projections, $\epsilon$ from the $\tau$ identity, and $s$
from a guaranteed-monotone bisection on $U$. $p$ is calculated last.

Floating-point error influences the residuals significantly. The
$\tau \ll D$ cancellation is what forces the $\tau$ form given above. Less obviously, the
raw momentum residual is $O(\cosh w)$, so an absolute bound on it silently demands
$\cosh w$ times more accuracy at high rapidity — more than double precision can supply
at all past $w \approx 9$. The convergence test is therefore taken on the normalized
$f_1/\cosh w = \tanh w - V$, so that a requested tolerance means the same thing at every
rapidity.

## This package

### Core functionality

Everything rests on the notion of a **valid state**: one the library can accept,
evaluate and return without failing. Validity is a set of bounds on the primitive
variables,

$$\rho \in [\rho_{\rm atm},\, \rho_{\rm ceiling}], \qquad
Y_e \in [Y_{\min}, Y_{\max}], \qquad
w \in [0, w_{\rm cap}], \qquad
s \in [\,s_{\min}(\rho, Y_e),\, s_{\max}(\rho, Y_e)\,],$$

of which the last is the odd one out. The first three are constants, but the entropy
window is not: $s_{\min}$ and $s_{\max}$ are the images of the table's $T_{\min}$ and
$T_{\max}$ under $\sigma$ at that particular $(\rho, Y_e)$, and $s_{\max}$ at
$\rho_{\max}$ lies far below $s_{\max}$ at $\rho_{\min}$. The valid set is therefore a
rectangular box in $(\rho, T, Y_e, w)$ — the variables the *table* is written in — but
is not a box in $(\rho, s, Y_e, w)$, the variables the solver actually
iterates on. So the adapter exports the pointwise window `srange(rho, Ye)` as a cheap
pair of spline evaluations, and any clamp of $s$ is taken at the already-clamped
$(\rho, Y_e)$ rather than against a global range.

Those edges are the physics a caller actually cares about. The **EOS table range** sets
the temperature limits and the top of the density range; the **atmosphere**
$\rho_{\rm atm}$ sets the bottom, below which there is no fluid worth evolving; and
**gravitational collapse** sets the ceilings — inside a forming horizon $D$ and $\tau$
grow without bound, and replacing such a point by atmosphere is the converse of
atmosphere handling, a kind of hydro excision.

The contract of a function processing a hydrodynamic state (not just
in this library, but in the whole evolution code) should then be:

1. every function must accept any valid state as input;
2. every function must only produce valid states as output, and cannot fail.

Detecting invalid states is cheap and explicit. `check_prim_state()` diagnoses a
primitive state with no mutation and no solves, and `check_con_state()` diagnoses a
*conservative* state using pure arithmetic by default — no EOS evaluation, no solve — so
it can run on every point of every timestep. Beyond that, every `EOSPoint` and
`Con2PrimOut` carries flags recording which domain-extension band, if any, the answer
came from. Mapping invalid states back to valid ones is the job of the policy layer:
`PolicyOptions` holds the thresholds, `default_policy(view, rho_atm)` derives all of
them but $\rho_{\rm atm}$ from the built adapter, and `project_prim_state()` clamps a
state into validity in place and reports what it changed. Projection is idempotent, so
projecting an already-valid state is a no-op that reports nothing.

For valid states `prim2con` is then straightforward, it has a closed form and cannot fail.
`con2prim` is where the contract has a cost. In the good case it is
simply the inverse: it converges, the recovered primitives are valid, and `prim2con` of
them reproduces the conserved input. But it *cannot* decide in advance whether a
conserved state is valid at all. Cheap necessary conditions exist — non-finite input,
$D \le 0$, $D$ or $\tau$ past the collapse ceilings, $D$ below the atmosphere trigger —
yet the real question, whether some $(s, w)$ in the box maps to these conservatives, is
answered only by attempting the solve. So either `con2prim_safe()` succeeds, producing a
valid state that reproduces its input, or `con2prim_safe()` "fails" and chooses a new
valid state by policy: an $s$-floor projection that keeps the inner solve's own $w$, a
ceiling projection along the adiabat, a $w$-cap, or a full atmosphere reset.

What makes the second ("failure") branch usable is that it returns *both* the new primitives and the
conserved state that `prim2con` generates from them. The caller adopts those
conservatives. Because every repair goes through
primitives, the identity
`con2prim(returned cons) == returned prims` holds by construction,
and it is verified as a property in the self-tests. `con2prim_safe()`
never fails for any input whatsoever, including non-finite garbage. Physics fidelity is
explicitly *not* a goal in the excision regime; the point is to keep the evolution going
with a state that is valid and exactly solvable, however bad the physics has become.

Two practical notes for consumers. Everything the run-time path needs lives in
`entropy_eos/core/`: header-only, no STL containers, no exceptions, no allocation, no
I/O, every function marked `EEOS_HOST_DEVICE` and operating on POD view structs, so the
same evaluation and inversion code compiles for a GPU once the coefficient arrays are
mirrored. Warm-start state is threaded explicitly through arguments and return values —
`evaluate()` is pure and re-entrant, with no hidden mutable state anywhere — so calls
are safely parallel across grid points. And the solver-facing units are uniform:
$\rho$ means $\rho^\ast$ in $\kappa$-rescaled g cm⁻³, $s$ is in $k_B$ per baryon, $w$ is
the rapidity, and $D$, $\tau$, $S_\parallel$, $S_\perp$, $B^2$ are all g cm⁻³. Unit
conversion happens once, at the adapter-build boundary; the run-time path is unit-free.
HDF5 is the only external dependency and is confined to the table-I/O translation units,
so a code embedding just the adapter and the solver needs no HDF5 at all.

### Producing valid tables

Since real tables have all the problems catalogued above, the library checks and repairs
them. Both are library functions first, with the command-line tools as thin wrappers.
`check_table()` is pure — no I/O side effects, cheap enough to run at startup right
after loading a table — and reports structural problems (axes, finiteness, required
fields), range and positivity, monotonicity in $T$ of entropy and energy with violation
counts and locations, Maxwell-consistency diagnostics against the stored pressure, and
the stored $c_s^2$ against a finite-difference estimate. `check_adapter()` and
`check_con2prim()` extend the same pattern to the built adapter and to the solver, so a
table can be audited at every level it will actually be used at.

`repair_table()`, and the `eos_repair` tool over it, then makes a table satisfy the hard
requirements. The intent is that this happens **once, before a simulation campaign**:
repair is a build-time activity on the data, and the run-time path contains no repair
logic at all. The tool never modifies its input; it writes a faithful copy of every
untouched dataset alongside the repaired fields, a `/repair` provenance group recording
per-field indices, old and new values, parameters, tool version and an input checksum,
and a human-readable log. Only `entropy` and `logenergy` are ever edited. A missing or
non-finite value in one of *those* is fatal — a broken file, not physics noise, and the
run throws before touching anything — whereas non-finite values in columns the pipeline
never interprets are reported and passed through byte-identically, which is what lets
LS220 be repaired at all despite carrying three Inf/NaN points in `cs2` and `gamma`.

Three stages then run in order, each aimed at one failure mode.

**Isotonic repair**, per $(\rho, Y_e)$ column along $T$, applies an $L^2$ isotonic
regression (pool-adjacent-violators, Barlow et al. 1972) to `entropy` and `logenergy`
separately, followed by
a strictification pass imposing a minimum slope — PAVA only makes a column
non-decreasing, while the adapter needs it strictly increasing. Repairs act on the
*stored* variables directly, since monotonicity of `logenergy` in $T$ is equivalent to
monotonicity of $\epsilon$, so no unit round trip is needed and the output stays
in-format. Columns are independent, which makes this stage embarrassingly parallel. On
the real tables the violations it targets are few and sharply localized: 62 entropy and
444 `logenergy` adjacent-pair violations in LS220, 23 and 68 in SRO — and on both
tables **100% of them lie above $\rho = 10^{13}$ g cm⁻³**, clustered in the nuclear
transition just below saturation density and spread across a wide range of temperature.
(The untouched Inf/NaN auxiliary points sit in the low-density, low-temperature corner, at $\rho \approx 10^7$ g cm⁻³ and
$T \approx 0.03$ MeV.)

**Spline-safe smoothing** follows, because monotone *data* is not what the adapter
actually needs — it differentiates the fitted spline. The per-column half of this stage
fits each repaired column, audits the fitted slope on a refined grid, and wherever a
sample violates, nudges the data with one small local Jacobi diffusion step before
re-repairing and re-auditing, up to a round cap. That handles the real cases: on LS220 it
smooths 38 entropy and 41 `logenergy` columns and leaves none violating. A second,
field-wide half does the same against the full tensor-product fit, which sees
cross-column violations — two individually monotone neighbours whose smooth blend becomes
non-monotone at an off-node position. That 3D loop is *not* monotonically improving, so it tracks the best state
it has seen, gives up once four consecutive rounds fail to beat it, and is finally
guarded by a backstop that re-audits the state from before the stage ran and reverts
wholesale unless the stage improved on it. On LS220 that backstop fires for both fields:
the diffusion makes the authoritative violation count worse at every round, so the stage
is discarded in full and its residual — 275 entropy and 20,650 `logenergy` samples — is
reported.

[what happens with these failures? are they bad at run time?]

**The causal cap** finally addresses $c_s^2 \ge 1$. It audits $c_s^2$ of the fitted
splines along adiabats through the analytic chain rule, groups violating samples into
runs along $\rho$, and treats only those runs reaching the $\rho_{\max}$ edge. For each
node in a treated run it traces the adiabat downward to a still-causal point, then marches
back up imposing a one-sided Lipschitz envelope on $\ln h$ and integrating the
energy-consistency ODE, writing only where the envelope actually bounds and $\epsilon$
drops. Only `logenergy` is touched, so the adiabat labels
are stable from round to round. Interior violation runs are counted and deliberately
never edited, since their defect lives in $\sigma_T$'s flatness rather than in
$\epsilon$'s stiffness, and so are samples with $c_s^2 \le 0$. On LS220 the stage runs
seven rounds and caps 66,419 nodes, covering the top 18% of the density axis at *every*
temperature, and takes the violation count from 4,076,091 to 7,622 — of which 7,532 are
exactly those untouched interior runs. A lexicographic backstop then requires that
monotonicity counts have not regressed and the $c_s^2$ count has not risen, reverting
the entire stage otherwise: causality is never enforced at the price of the $T$-solve's
hard requirement.

Repairing the *data* this way, rather than clamping $c_s^2$ at run time, is deliberate: a
clamped $c_s^2$ corresponds to no single potential $U$, and would silently break the
identities the Newton algorithm is built on.

Two properties tie the stages together. The result is **deterministic** — every stage's
outcome is independent of thread count and scheduling, so one input always yields the
same bytes. And repair is **idempotent** in the case that matters: a second run over a
repaired table reports zero changes and exits clean, which holds on the real tables as
well as in the unit tests. That guarantee is conditional, because a
stage that leaves residual violations may explore further from wherever the data are now.

With a valid table in hand, `build_entropy_eos()` performs the $F \to U$ conversion once:
convert units, fit $C^2$ tensor-product cubic B-splines to entropy and
$\log(\epsilon + \Delta)$, choose $\kappa$ from a fine scan of the *fitted* splines
(which can undershoot the data minimum, so scanning the raw data would not be enough),
and extend both fields beyond the physical box — eight grid cells per side in $\log\rho$
and $\log T$ — with $C^2$, monotone, *causal* tails. Out of bounds is handled by smooth
extension plus a flag, not by a hard clamp, since a clamp zeroes
derivatives and stalls or misleads Newton. Validity is judged only at the end, on the converged
state, by the policies above. The tails are also built to approach the right asymptotics
rather than merely to stay finite, since an *acausal* extension breaks the inner solve's
monotonicity argument just as surely as acausal table data would. In the density
direction that means continuing $\ln\sigma$ and $\ln\epsilon$ affinely in $\log\rho$, so
both become power laws in $\rho$ with the seam's own exponents: the radiation limit
$c_s^2 \to 1/3$ at a hot seam, degenerating to the ideal-gas behaviour at a cold one. In
the temperature direction the hot tail is built in $\ln\sigma$ for the same reason, with
an explicit causal clamp on the energy tail's slope. $Y_e$ is the exception to all of this: it is
exact and never iterated, so it is simply clamped and flagged.

## Example

Check and repair a table, then audit the result end to end:

```bash
make tools
T=tables/LS220_234r_136t_50y_analmu_20091212_SVNr26.h5
./tools/eos_repair --check-only $T          # exit 0 = clean, 1 = needs repair
./tools/eos_repair $T tables/LS220_repaired.h5 --log repair.log
./tools/eos_test --level con2prim tables/LS220_repaired.h5
```

`eos_test` also runs at `--level table` and `--level adapter`, and against a built-in
synthetic table (`--synthetic`, `--synthetic-dirty`) when no real table is at hand;
[`tables/README.md`](tables/README.md) says where to download the real ones, which are
not committed.

The same operations from C++ — check validity, convert primitives to conservatives and
back, and fall through to the never-fails path:

```c++
#include <cmath>
#include <limits>
#include <cstdio>

#include <entropy_eos/entropy_eos.hpp>

static const double kNaN = std::numeric_limits<double>::quiet_NaN();

int main() {
  // --- 1. Load, check and repair a table (once, at startup) ----------------
  eeos::RawTable table = eeos::make_synthetic_table();  // or read_stellarcollapse("LS220.h5")

  const eeos::CheckReport report = eeos::check_table(table);
  if (report.status == eeos::Status::fatal) return 1;   // broken file, not physics noise
  eeos::repair_table(table);                            // no-op on an already-clean table

  // --- 2. Build the adapter: F(rho,T,Ye) -> U(rho*,s,Ye) -------------------
  const eeos::EntropyEOS eos = eeos::build_entropy_eos(table);
  const eeos::EntropyEOSView view = eos.view();         // POD, device-ready
  const eeos::PolicyOptions pol =
      eeos::default_policy(view, /*rho_atm=*/1e3 * eos.kappa());

  // --- 3. A valid primitive state ------------------------------------------
  const double ye = 0.4;
  const double rho = std::pow(10.0, 0.5 * (view.x_lo + view.x_hi));  // rho* = kappa*rho
  const eeos::SRange sr = view.srange(rho, ye);
  eeos::PrimState ps{rho, 0.5 * (sr.s_min + sr.s_max), ye, /*w=*/0.8};

  if (eeos::check_prim_state(view, ps, pol))
    eeos::project_prim_state(view, ps, pol);            // clamp into validity

  const eeos::EOSPoint pt = view.evaluate(ps.rho, ps.s, ps.ye, /*u_guess=*/kNaN);
  std::printf("p = %.6e   T = %.4f MeV   cs2 = %.4f\n", pt.p, pt.T_F_MeV, pt.cs2);

  // --- 4. prim2con: always succeeds ----------------------------------------
  const double B2 = 0.1 * ps.rho, cos_vB = 0.3;
  const eeos::Prim2ConOut c =
      eeos::prim2con(view, ps.rho, ps.s, ps.ye, ps.w, B2, cos_vB, pt.u_solved);

  // --- 5. con2prim: recovers the primitives --------------------------------
  const eeos::Con2PrimIn in{c.D, c.tau, c.D_Y, c.S_par, c.S_perp, c.B2};
  const eeos::Con2PrimOptions opts;                     // tol 1e-12
  const eeos::Con2PrimOut rec =
      eeos::con2prim(view, in, opts, ps.s, ps.w, pt.u_solved);   // warm start
  std::printf("rho %.3e -> %.3e   s %.6f -> %.6f   w %.6f -> %.6f\n",
              ps.rho, rec.rho, ps.s, rec.s, ps.w, rec.w);

  // --- 6. con2prim_safe: never fails ---------------------------------------
  const eeos::Con2PrimIn garbage{c.D, -1e30, c.D_Y, 1e30, 0.0, c.B2};
  const eeos::Con2PrimSafeOut safe = eeos::con2prim_safe(view, garbage, opts, pol);
  if (safe.policy_flags & eeos::flag_pol_any)
    std::printf("state repaired (flags 0x%x); adopt safe.cons as the new conserved state\n",
                safe.policy_flags);
  return 0;
}
```

Build and run it against the static library:

```bash
make lib
c++ -O2 -std=c++17 -I. example.cpp libentropy_eos.a -lhdf5 -o example && ./example
```

which prints

```
p = 2.376396e+07   T = 1.5811 MeV   cs2 = 0.0039
rho 9.999e+09 -> 9.999e+09   s 43.394858 -> 43.394858   w 0.800000 -> 0.800000
state repaired (flags 0x400); adopt safe.cons as the new conserved state
```

Both usage modes are supported: either link `libentropy_eos.a` as above, with
`make install PREFIX=…` to install the headers alongside it, or copy the `entropy_eos/`
directory into your own tree and add its `host/*.cpp` files to your build. Users
write `#include <entropy_eos/entropy_eos.hpp>` either way. The build is deliberately
plain C++17 with no cmake or configure step. OpenMP is opt-in (`make OPENMP=-fopenmp`)
and the code compiles and runs serially without it.

## Status and testing

This is version **1.0.0**. The version is available to consumers as
`EEOS_VERSION_MAJOR/_MINOR/_PATCH`, as an encoded integer `EEOS_VERSION` for
`#if EEOS_VERSION >= EEOS_VERSION_ENCODE(1, 1, 0)` feature guards, as
`EEOS_VERSION_STRING`, and as `constexpr eeos::version*` values, all from
[`entropy_eos/core/version.hpp`](entropy_eos/core/version.hpp) (header-only and
device-safe, so a consumer embedding only `core/` still has it). When the library is
consumed as an installed `libentropy_eos.a`, `eeos::version_matches()` checks the
archive against the headers it is being compiled against, and every tool answers
`--version`. What a major/minor/patch bump does and does not promise — in particular
what it means for computed numbers, which are pinned by table provenance rather than by
the release number — is written up in [`CODE.md`](CODE.md) under "Versioning".

The table layer, the adapter, and the solver with its invalid-state policies are
complete and measured; the remaining milestone is the CUDA port, for which `core/` is
already written but not yet compiled under `nvcc`. Correctness rests on three
independent legs: unit tests against a manufactured analytic EOS with known closed-form
answers, a deterministic "dirty" synthetic preset that reproduces the pathologies found
in the real tables so the whole detect-repair-audit narrative runs in CI, and
integration runs over the four real tables when they are present locally. `make test`
runs the unit tests, `make integration` the end-to-end suite; CI runs both, serially and
with OpenMP, and skips the real-table half gracefully.

## References

- E. O'Connor & C. D. Ott, *Class. Quantum Grav.* **27**, 114103 (2010) — the HDF5 table
  format read here.
- F. D. Swesty, *J. Comput. Phys.* **127**, 118 (1996); F. X. Timmes & F. D. Swesty,
  *Astrophys. J. Suppl.* **126**, 501 (2000) — thermodynamically consistent interpolation
  by differentiating a single potential, in the free-energy form $F(\rho, T)$.
- A. S. Schneider, L. F. Roberts & C. D. Ott, *Phys. Rev. C* **96**, 065802 (2017) — the
  SRO framework, and the high-density causality behaviour of Skyrme-type EOS that the
  acausal corner reported above reflects.
- S. C. Noble et al., *Astrophys. J.* **641**, 626 (2006) — the standard 1D/2D con2prim
  schemes.
- W. I. Newman & N. D. Hamlin, *SIAM J. Sci. Comput.* **36**, B661 (2014); D. M. Siegel
  et al., *Astrophys. J.* **859**, 71 (2018) — the tabulated-EOS con2prim landscape.
- W. Kastaun, J. V. Kalinani & R. Ciolfi, *Phys. Rev. D* **103**, 023018 (2021) — the
  guaranteed-convergence master function behind RePrimAnd, the closest relative of the
  scheme used here. It solves a one-dimensional master function for $\mu = 1/(Wh)$ on the
  bounded interval $0 < \mu \le 1/h_0$.
- L. R. Werneck et al., [arXiv:2208.14487](https://arxiv.org/abs/2208.14487) (2023) —
  tabulated EOS in IllinoisGRMHD, and the evolved-entropy backup solvers referred to below.
- R. E. Barlow, D. J. Bartholomew, J. M. Bremner & H. D. Brunk, *Statistical Inference
  Under Order Restrictions* (Wiley, 1972) — the pool-adjacent-violators algorithm used by
  the isotonic repair stage.

Entropy has usually appeared in these codes as an evolved *fallback* tracer: passively
advected alongside the other variables, and unreliable precisely at shocks, where it is
not conserved. The approach here differs in making it the primary thermal recovery
variable, obtained from the conserved state rather than carried separately; and in
pairing it with rapidity, which is unbounded, so that the iterate space has no kinematic
boundary at all rather than a relocated one.

The remaining ingredient is older than this work and worth attributing plainly. That
pressure and temperature should be *derivatives of one potential* rather than independent
interpolated tables is Swesty's idea, realized in the free energy $F(\rho, T)$ and standard
practice since. What is new here is the choice of variables: taking the potential as
$U(\rho, s, Y_e)$, the variables con2prim actually iterates on, and obtaining it by
composition through a temperature inversion from a table that is not itself a potential
table — which is also what makes the table's own thermodynamic defects, and hence the
repair stage, unavoidable rather than incidental.

A fuller survey of related software and publications — consistency by construction,
con2prim libraries, table repair, causality, and performance portability — is in
[`RELATED.md`](RELATED.md). Table citations for LS220, SRO, DD2 and SFHo are in
[`tables/README.md`](tables/README.md).
