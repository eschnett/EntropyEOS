# Con2Prim via Entropy and Rapidity — Design Task

Draft v0.1, August 24, 2026. Working document; everything is up for iteration.

## 1. Scope

Design a pointwise conservative-to-primitive inversion for ideal GRMHD with a tabulated,
composition-dependent EOS. Input is the undensitized conserved state (all $\sqrt{\gamma}$
factors removed; the spatial metric $\gamma_{ij}$ is available to form scalars)

$$(D,\; S_i,\; \tau,\; B^i,\; D_Y),$$

output is the primitive state

$$(\rho,\; v^i,\; p,\; \epsilon,\; B^i,\; Y_e,\; T,\; s).$$

Conventions: Valencia-type conservatives measured by the normal observer; $\tau$ excludes
rest mass and we write $E \equiv \tau + D$ for total energy density; $S_i$ covariant,
$v^i, B^i$ contravariant; $B^i$ is the normal-observer field and passes through unchanged.
Units $c = G = 1$; $s$ is entropy per baryon in $k_B$; $\rho = m_B n_B$ is rest-mass density.
Scalars used below: $B^2 = \gamma_{ij}B^iB^j$, $S^2 = \gamma^{ij}S_iS_j$,
$S_{\parallel} = (S_iB^i)/|B|$ (signed), $S_{\perp} = \sqrt{S^2 - S_{\parallel}^2}$.

**Notation changes relative to the brainstorm.** The brainstorm used $P$ for momentum and
$T$ for total energy; both clash fatally with pressure and temperature in an EOS-heavy
context, so: momentum magnitude $\to |S|$ (projections $S_\parallel, S_\perp$), total
energy $\to E$, and $D_e = N_e\cosh w \to D_Y = \rho Y_e \cosh w$ (i.e. the intensive
composition variable is $Y_e$, not a number density; see open question 1).

## 2. Central idea

Iterate on a primitive state whose components are thermodynamically natural for a table and
kinematically unconstrained:

- **Thermal variables $(\rho, s, Y_e)$** — entropy instead of $\epsilon$ or $T$.
- **Rapidity $w$** instead of velocity: $W = \cosh w$, $Wv = \sinh w$, $v = \tanh w$,
  with $w \in [0,\infty)$ and all directional information carried algebraically by
  $S_i$ and $B^i$.

Every iterate $(s, w)$ in the table domain corresponds to a physical state: $v < 1$ and
$\rho > 0$ are built into the parametrization, so Newton never has to be fenced away from a
manifold boundary. The rapidity is well conditioned in both limits: $w \simeq v$ as
$v \to 0$ and $w \simeq \log(2W)$ as $v \to 1$, where velocity- or $W$-based schemes become
stiff.

The closest published scheme, Kastaun et al. (2021), also removes the velocity constraint,
but by solving for $\mu = 1/(Wh)$ on the *bounded* interval $0 < \mu \le 1/h_0$; the
rapidity is unbounded instead. Note also that entropy here is recovered from the conserved
state, unlike the evolved-entropy backup solvers of IllinoisGRMHD and HARM, which advect
$s$ separately and are therefore unreliable at shocks. See [`RELATED.md`](RELATED.md).

## 3. EOS interface

The EOS is supplied as a single function

$$\epsilon = U(\rho, s, Y_e),$$

the specific internal energy **in its natural variables** — i.e. a complete thermodynamic
potential. The first law per unit rest mass,
$d\epsilon = T\,ds - p\,d(1/\rho) + \tilde\mu\,dY_e$, gives everything else as derivatives:

$$p = \rho^2\, U_\rho, \qquad T = U_s, \qquad \tilde\mu = U_{Y_e},$$

with $h = 1 + U + \rho U_\rho$ and the useful identities

$$\left.\frac{\partial p}{\partial\rho}\right|_{s,Y_e} = 2\rho U_\rho + \rho^2 U_{\rho\rho} = h c_s^2,
\qquad
\left.\frac{\partial h}{\partial\rho}\right|_{s} = \frac{h c_s^2}{\rho},
\qquad
\left.\frac{\partial h}{\partial s}\right|_{\rho} = T + \rho U_{\rho s},
\qquad
\left.\frac{\partial p}{\partial s}\right|_{\rho} = \rho^2 U_{\rho s}.$$

Thermodynamic consistency is exact by construction — there is no separate $p$ table to
disagree with $\epsilon$. The solver needs $U, U_\rho, U_s$ per evaluation and
$U_{\rho\rho}, U_{\rho s}$ for the Jacobian, so the table representation must be $C^2$
(tensor-product splines in, say, $(\log\rho, s, Y_e)$). Causality/convexity checks
($0 < c_s^2 < 1$, $c_v > 0$) become spline-quality checks. Converting a standard
$(\rho, T, Y_e)$ table to this form — including monotonicity repair where $c_v \le 0$ —
is a prerequisite subproject (deliverable 1).

## 4. Prim2con in rapidity form (pure hydro)

The ansatz, in the corrected notation:

$$D = \rho\cosh w$$
$$D_Y = \rho Y_e \cosh w$$
$$|S| = (\rho + \rho\epsilon + p)\sinh w\cosh w = \rho h W^2 v$$
$$E = (\rho + \rho\epsilon)\cosh^2 w + p\sinh^2 w = \rho h W^2 - p$$

Two exact consequences worth noting. First, composition decouples completely:

$$Y_e = D_Y / D \quad\text{(exact, no iteration).}$$

Second, the hydro system has a clean exponential structure,

$$h\, e^{\pm w} = \frac{E + p \pm |S|}{D},
\qquad \tanh w = \frac{|S|}{E+p},
\qquad (E+p)^2 - |S|^2 = (Dh)^2,$$

which is useful for initial guesses and for analysis (these hold only at $B = 0$; MHD
modifies them through §5).

## 5. Magnetic terms

With $z \equiv \rho h W^2 = D\,h\cosh w$, the ideal-MHD momentum and energy are

$$S_i = (z + B^2)\,v_i - (B^jv_j)\,B_i,
\qquad
E = z - p + \tfrac12 B^2(1+v^2) - \tfrac12 (B^iv_i)^2.$$

Projecting the momentum along $\hat b = B/|B|$, the magnetic terms cancel in the parallel
direction and add inertia in the perpendicular one:

$$S_\parallel = z\, v_\parallel, \qquad S_\perp = (z + B^2)\, v_\perp,$$

so given a scalar $z$ the velocity is fully determined,

$$v^i = \frac{S^i + (S_jB^j)\,B^i/z}{z + B^2},
\qquad (B^iv_i)^2 = B^2 v_\parallel^2 .$$

## 6. Reduction to two unknowns

$Y_e = D_Y/D$ (exact); $\rho = D/\cosh w$ (eliminates $\rho$); $B^i$ pass through. The
remaining unknowns are $(s, w)$. Given a trial $(s, w)$:

1. $\rho = D/\cosh w$, $Y_e = D_Y/D$; EOS gives $\epsilon, p, h, T$ and derivatives.
2. $z = D\,h\cosh w$; then $v_\parallel = S_\parallel/z$, $v_\perp = S_\perp/(z+B^2)$,
   $V \equiv \sqrt{v_\parallel^2 + v_\perp^2}$.

## 7. Residuals

**Momentum** (kinematic consistency, deliberately *not* squared so that the $S \to 0$,
$w \to 0$ root stays simple rather than double):

$$f_1(s,w) = \sinh w - \cosh w\; V(s,w).$$

**Energy, in cancellation-free $\tau$ form.** Naively matching $E = \tau + D$ loses
precision in cold, slow flows where $\tau \ll D$. Using $\rho\cosh w = D$ and
$\cosh w - 1 = 2\sinh^2(w/2)$,

$$\tau_{\rm model} = 2D\sinh^2(w/2) \;+\; \rho\,\epsilon\cosh^2 w \;+\; p\sinh^2 w
\;+\; \tfrac12 B^2(1+\tanh^2 w) - \tfrac12 B^2 v_\parallel^2,$$

in which every term is individually small in the cold slow limit and (for
$\epsilon, p \ge 0$) manifestly nonnegative, including the magnetic part, which is bounded
below by $B^2/2$. Then

$$f_2(s,w) = \frac{\tau_{\rm model} - \tau}{\max(\tau,\, \delta D)}$$

with the normalization to be settled (the point is that the convergence tolerance must be
relative to $\tau$, not to $D$). Caveat: some tables have $\epsilon < 0$ at low $T$
depending on the energy zero point (`energy_shift`); see open question 3.

## 8. Jacobian

All entries are compact. With $\rho_w = -\rho\tanh w$ and the EOS identities of §3:

$$z_w = z\,(1 - c_s^2)\tanh w, \qquad z_s = D\cosh w\,(T + \rho U_{\rho s}),$$
$$p_w = -\rho h c_s^2 \tanh w, \qquad p_s = \rho^2 U_{\rho s},$$
$$\partial_\bullet v_\parallel = -\frac{v_\parallel}{z} z_\bullet, \qquad
\partial_\bullet v_\perp = -\frac{v_\perp}{z + B^2} z_\bullet,$$

and $f_1, f_2$ assemble by the chain rule. Note $z_w > 0$ for any causal EOS
($c_s^2 < 1$) and $w > 0$ — the factor $(1 - c_s^2)$ appearing naturally is a good sign
for conditioning.

## 9. Structure of the root: a provable inner solve

At fixed $s$, $f_1$ is **strictly monotone in $w$**:

$$\frac{\partial f_1}{\partial w}
= \cosh w - \sinh w\, V - \cosh w\, \frac{\partial V}{\partial w},
\qquad
\frac{\partial V}{\partial w}
= -\frac{z_w}{V}\left(\frac{v_\parallel^2}{z} + \frac{v_\perp^2}{z+B^2}\right) \le 0,$$

so $\partial f_1/\partial w \ge \cosh w\,(1 - V\tanh w) > 0$ since $V < 1$. Moreover
$f_1(0) = -V \le 0$ and $f_1 \to +\infty$ as $w \to \infty$ (because $z \to \infty$
drives $V \to 0$). Hence for every $s$ in the table range there is a **unique**
$w_\star(s) \in [0, w_{\max}]$, bracketed and safe for bisection-guarded Newton. The outer
problem is then one-dimensional: find $s$ with $f_2(s, w_\star(s)) = 0$, bracketed on
$[s_{\min}, s_{\max}]$ by sign change. Whether the outer function is also monotone (it
should be, since $\partial f_2/\partial s$ contains $\rho T\cosh^2 w > 0$ plus terms with
the sign of the Grüneisen-like $U_{\rho s}$, and $dw_\star/ds \le 0$) is to be established;
even without a proof, the nested scheme with outer bisection fallback is globally
convergent in practice. The production path is still the coupled $2\times2$ Newton of
§§7–8 (quadratic convergence, one EOS call per iteration); the nested scheme is the
guaranteed fallback, in the spirit of Kastaun et al.'s master function but with entropy as
the thermal variable.

## 10. Why this should work well

Rapidity makes the iterate space free of kinematic constraints and well conditioned at both
$v \to 0$ and $v \to 1$; the $2\sinh^2(w/2)$ identity gives the cancellation-free energy
residual essentially for free. Entropy makes the EOS surface smooth (adiabats are the
natural coordinates of the table's physics), gives $p$ and $T$ as derivatives of one $C^2$
potential with exact thermodynamic consistency, and is slowly varying along smooth flow —
the previous timestep's $(s, w)$ is an excellent initial guess, so the common case should
converge in 1–2 Newton steps. $Y_e$ is exact and outside the iteration. The kernel is a
fixed-size $2\times2$ Newton with analytic Jacobian and bracketed fallback: branch-light
and fixed-iteration-friendly, i.e. GPU-suitable.

## 11. Failure modes and policies (to decide)

Invalid conserved states from evolution error need detection and a repair policy
(rescale $S_i$, adjust $\tau$, atmosphere reset) — the natural validity test here is
whether the outer bracket $[s_{\min}, s_{\max}]$ contains a sign change of $f_2$ after the
inner solve, plus a $w_{\max}$ (equivalently $W_{\max}$) cap. Entropy leaving the table
range must be clamped and flagged, and the low-density side needs an atmosphere
prescription plus a table extension below $\rho_{\min}$. The $B \to 0$ and $S \to 0$
limits are smooth in the formulas above but should be implemented branch-free
(e.g. $S_\parallel$ via $S_iB^i$ with a guarded $|B|$, $S_\perp^2$ clamped at 0).

## 12. Deliverables and test plan

1. **EOS layer (Julia):** $C^2$ tensor-product spline $U(\log\rho, s, Y_e)$ built from a
   $(\rho, T, Y_e)$ table, with $c_v > 0$ monotonicity repair, derivative evaluation up to
   second order, and causality/convexity audits over the whole domain.
2. **Reference prim2con** in rapidity form, plus property tests: round trip
   con2prim∘prim2con = id to near machine precision over table samples ×
   $w \in [0, 6]$ × magnetization $B^2/(\rho h) \in [0, 10^4]$ × arbitrary
   $\angle(S, B)$.
3. **Production solver:** coupled $2\times2$ Newton with analytic Jacobian, safeguards
   (clamps, damping), warm starts from the previous step.
4. **Guaranteed fallback:** nested 1D scheme of §9 with bisection guards; invalid-state
   policies of §11.
5. **Benchmarks:** accuracy/iteration-count/runtime against RePrimAnd on the same states;
   then a GPU port (fixed iteration count, branch-free formulation).

## 13. Open questions

1. EOS signature: $Y_e$ (intensive, bounded) vs. electron number density $N_e$ as in the
   brainstorm — I propose $Y_e$ and absorbing $m_B$ conventions into the EOS layer.
2. Entropy variable: per-baryon $s$ in $k_B$, iterated linearly — agreed? Any reason to
   prefer entropy density or a log variable?
3. Energy zero point: can we shift $U$ so $\epsilon \ge 0$ everywhere (preserving the
   cancellation-free $\tau$ form), or do we handle signed $\epsilon$?
4. Do the evolution code's conservatives match §1 exactly ($\tau$ without rest mass,
   covariant $S_i$, undensitized at con2prim entry)?
5. Unknowns: eliminate $\rho$ (2×2 in $(s,w)$, as here) vs. keep $(ρ, s, w)$ with a
   $D$-residual for a simpler Jacobian?
6. Fallback philosophy: nested scheme of §9, or additionally an evolved auxiliary entropy
   for guesses/rescue, or adapt RePrimAnd's master function to $U(\rho,s,Y_e)$ calls?
7. Atmosphere/floors and sub-$\rho_{\min}$ table extension: whose responsibility —
   con2prim or the caller?
8. Target order: CPU Julia prototype first, then CUDA port for the GPU GRMHD code?
9. Policy values: $W_{\max}$, tolerances (relative to $\tau$), maximum iteration counts
   for fixed-cost GPU execution.

## 14. Relation to prior work

Noble et al. (2006) established the standard 1D/2D schemes; Newman & Hamlin (2014) and
Siegel et al. (2018) cover the tabulated-EOS landscape; Kastaun, Kalinani & Ciolfi (2021)
give the guaranteed-convergence 1D master function behind RePrimAnd. Entropy has mostly
appeared as an evolved *fallback* tracer in existing codes. The proposal here differs in
using entropy as the *primary* thermal recovery variable, pairing it with rapidity so the
iterate space has no kinematic boundary, and consuming the EOS through a single potential
$U(\rho, s, Y_e)$ so that pressure and temperature are derivatives rather than independent
tables.
