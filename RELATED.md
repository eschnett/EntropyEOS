# Related work

Prior art bearing on the three technical choices `EntropyEOS` makes: the single
thermodynamic potential $U(\rho, s, Y_e)$ as the EOS interface, rapidity $w$ as the
kinematic iterate, and repairing table *data* rather than clamping at run time.

Each entry says how the work bears on a decision made here, not what the paper is about.
Citations for the *physics* of the tables themselves — LS220, SRO, DD2, SFHo, and the
O'Connor–Ott HDF5 format — are in [`tables/README.md`](tables/README.md) and are not
repeated here.

## 1. Thermodynamic consistency by construction

The oldest and closest thread. The adapter's claim that consistency is *structural* — that
$p$ and $\hat T$ cannot disagree with $\epsilon$ because all three are derivatives of one
function — is an established idea, previously realized in different variables.

- **F. D. Swesty**, *J. Comput. Phys.* **127**, 118 (1996) — *Thermodynamically Consistent
  Interpolation for Equation of State Tables.* Bi-quintic Hermite interpolation of the
  **Helmholtz free energy** $F(\rho, T)$, with pressure and entropy obtained by
  differentiation, so the first law and the Maxwell relations hold exactly *between* mesh
  points as well as on them. The direct ancestor of the construction used here: EntropyEOS
  applies the same principle to $U(\rho, s, Y_e)$, buying con2prim-natural variables at the
  cost of a temperature inversion.
- **F. X. Timmes & F. D. Swesty**, *Astrophys. J. Suppl.* **126**, 501 (2000) — the same
  construction applied to the $e^\pm$ EOS (`helmholtz`); the paper that made "interpolate one
  potential, differentiate for the rest" standard practice in stellar astrophysics.
- **G. A. Dilts**, *Phys. Rev. E* **73**, 066704 (2006),
  [arXiv:physics/0510195](https://arxiv.org/abs/physics/0510195) — *Consistent thermodynamic
  derivative estimates for tabular equations of state.* States the same pair of requirements
  as [`eos-adapter-F-to-U.md`](eos-adapter-F-to-U.md) §8 — **consistency** (derivable from a
  free energy) and **stability** ($c_s^2 > 0$) — observes that standard table interfaces
  enforce at most the second, and proposes a constrained local least-squares ("tuned
  regression") interface. The closest published work to *fitting a table subject to
  thermodynamic constraints*, and hence to the repair stage here.
- **V. A. Baturin, W. Däppen, A. V. Oreshina, S. V. Ayukov & A. B. Gorshkov**,
  *Astron. Astrophys.* **626**, A108 (2019),
  [arXiv:1905.08303](https://arxiv.org/abs/1905.08303) — quintic Hermite 2D splines
  interpolating pressure *together with its derivatives*, so the thermodynamic identities
  hold off-node; quantifies interpolation error against model-to-model differences. The
  methodological template for the adapter's fidelity audits.

**The contrast this project is arguing against.** Mainstream simulation practice interpolates
trilinearly in $(\log\rho, \log T, Y_e)$ and obtains derivatives by analytically
differentiating the trilinear formula — `nuc_eos`/stellarcollapse, `weaklib`, `thornado`.
That is not Maxwell-consistent by construction, and most tabulations of the LS EOS are not
built to guarantee consistency either, with the documented consequence that spurious entropy
generation gets attributed to the EOS rather than to the tabulation.

## 2. Conservative-to-primitive inversion

- **W. Kastaun, J. V. Kalinani & R. Ciolfi**, *Phys. Rev. D* **103**, 023018 (2021),
  [arXiv:2005.01821](https://arxiv.org/abs/2005.01821);
  [RePrimAnd](https://github.com/wokast/RePrimAnd), [ASCL 2107.021](https://ascl.net/2107.021).
  The guaranteed-convergence scheme closest in spirit to this one. Worth stating precisely
  how it differs: RePrimAnd solves a one-dimensional master function $f(\mu)$ for
  $\mu \equiv 1/(Wh)$, whose range is the **bounded** interval $0 < \mu \le 1/h_0$; the
  quantity $z \equiv Wv$ appears in that paper as notation for bounding the velocity and for
  presenting results, not as the iterate. The rapidity used here is unbounded,
  $w \in [0,\infty)$, which is what removes the kinematic boundary from the iterate space
  rather than relocating it. RePrimAnd's EOS layer enforces validity *at the interface*,
  where EntropyEOS instead makes the underlying data valid. Its runtime polymorphism is
  awkward inside GPU kernels.
- [**primitive-solver**](https://github.com/jfields7/primitive-solver) — J. Fields et al.,
  *Performance-Portable Binary Neutron Star Mergers with AthenaK*,
  [arXiv:2409.10384](https://arxiv.org/abs/2409.10384). The same Kastaun inversion rebuilt
  around compile-time `EOSPolicy` and `ErrorPolicy` template parameters, explicitly because
  virtual functions are costly in GPU kernels. Independent convergence on the design used in
  [`entropy_eos/core/`](entropy_eos/core/): POD views, no virtual dispatch, error handling as
  a policy rather than an exception. The most useful design comparison available.
- **D. M. Siegel, P. Mösta, D. Desai & S. Wu**, *Astrophys. J.* **859**, 71 (2018),
  [arXiv:1712.07538](https://arxiv.org/abs/1712.07538) — the systematic comparison of recovery
  schemes for analytic and tabulated EOS. The benchmark protocol that `check_con2prim()`
  measurements should be read against.
- **S. C. Noble et al.**, *Astrophys. J.* **641**, 626 (2006);
  [PVS-GRMHD, ASCL 1210.026](https://ascl.net/1210.026) — the standard 1D/2D schemes; the
  $2\times2$ Newton here is their entropy/rapidity analogue.
- **W. I. Newman & N. D. Hamlin**, *SIAM J. Sci. Comput.* **36**, B661 (2014) — effective-1D
  pressure iteration, independent of the initial guess, with an inner EOS inversion per outer
  step: structurally the same cost model as the nested $\sigma(\rho,T,Y_e) = s$ solve used
  here, and reported robust at large Lorentz factor.
- **C. Cai, J. Qiu & K. Wu** (2024), [arXiv:2404.05531](https://arxiv.org/abs/2404.05531) —
  a provably quadratically convergent, physical-constraint-preserving Newton–Raphson recovery
  for RMHD, with a constructive theory of admissible initial guesses. The strongest available
  statement about *when* a seed is safe, and the natural comparison for the cold-start seed
  construction.
- **L. R. Werneck et al.** (2023), [arXiv:2208.14487](https://arxiv.org/abs/2208.14487) —
  tabulated EOS and neutrino leakage in IllinoisGRMHD. Documents the **evolved-entropy backup
  solvers** and why they are backups only: the entropy is passively advected and is not
  conserved at shocks. This is the reference point for the distinction drawn in
  [`README.md`](README.md) — entropy here is recovered from the conserved state, not carried
  as a separate advected tracer, so it is not subject to that failure mode.
- **J. V. Kalinani et al.** (2021), [arXiv:2107.10620](https://arxiv.org/abs/2107.10620) —
  adoption of the Kastaun scheme in the Spritz code; a useful account of what changes in an
  existing code when the recovery scheme is replaced.
- **T. Dieselhorst et al.** (2021), [arXiv:2109.02679](https://arxiv.org/abs/2109.02679), and
  **S. Kacmaz et al.** (2024), [arXiv:2412.07836](https://arxiv.org/abs/2412.07836) —
  machine-learning surrogates for con2prim with tabulated EOS, benchmarked against RePrimAnd.
  The opposite bet to the one made here: learn the inverse map, rather than make the forward
  map cheap and well conditioned.

## 3. Table repair and regularization

No published tool was found that does what [`eos_repair`](tools/eos_repair.cpp) does — an
audit-driven, logged, provenance-recording, idempotent repair pass over a *delivered* 3D
nuclear table, for monotonicity and causality. The nearest work approaches the same problem
from four other directions.

- **G. Servignat, P. J. Davis, J. Novak, M. Oertel & J. A. Pons**, *Phys. Rev. D* **109**,
  103022 (2024), [arXiv:2311.02653](https://arxiv.org/abs/2311.02653) — *One- and two-argument
  equation of state parametrizations with continuous sound speed.* Motivated by exactly the
  defect catalogued in [`README.md`](README.md): realistic tables have poor derivative
  precision at high density, and finite-difference sound speeds show non-physical spikes that
  can crash evolution codes. Their answer is to *replace* the table with a smooth fit having
  continuous $c_s$; the answer here is to repair the table and then fit $C^2$ splines. The
  most direct "same problem, different solution" comparison.
- **I. Legred et al.** (2023), [arXiv:2301.13818](https://arxiv.org/abs/2301.13818) — a
  flexible enthalpy-based EOS parametrization in SpECTRE; the same smooth-representation
  strategy expressed in a different potential.
- **Isotonic regression.** R. E. Barlow, D. J. Bartholomew, J. M. Bremner & H. D. Brunk,
  *Statistical Inference Under Order Restrictions* (Wiley, 1972) — the pool-adjacent-violators
  algorithm used by the isotonic stage. M. J. Best & N. Chakravarti, *Math. Program.* **47**,
  425 (1990), for PAVA as an active-set method with its $O(n)$ bound.
- **Shape-preserving interpolation**, background for why monotone *data* is not sufficient
  when a fitted spline is what gets differentiated: F. N. Fritsch & R. E. Carlson,
  *SIAM J. Numer. Anal.* **17**, 238 (1980); H. T. Huynh, *Accurate monotonicity-preserving
  cubic interpolation*, [OSTI 5328033](https://www.osti.gov/servlets/purl/5328033).

Taken together, the literature enforces consistency in the *interpolant* (Swesty, Dilts,
Baturin), or validity at the *interface* (RePrimAnd), or replaces the table with a smooth
*fit* (Servignat, Legred), or silently fixes points during *table construction*. A reusable,
audited, logged repair pass over a table as shipped appears not to have been published.

## 4. Causality

- Skyrme-type and LS-family equations of state are known to become **superluminal at high
  density**, so the edge-anchored acausal corner reported in [`README.md`](README.md) —
  4.2% of LS220 nodes, 4.3% of SRO — is expected physics rather than a fitting artifact. See
  the causality discussion in **A. S. Schneider, L. F. Roberts & C. D. Ott**,
  *Phys. Rev. C* **96**, 065802 (2017), [arXiv:1707.01527](https://arxiv.org/abs/1707.01527)
  (cited in [`tables/README.md`](tables/README.md) for SRO itself, here for a different
  reason). The internal confirmation is independent: both tables' own stored `cs2` columns
  agree with the derived map across the corner.
- **Constant-sound-speed extrapolation** is the established remedy in a neighbouring context —
  quark-matter tables whose $p(e)$ back-bends are repaired by a constant-$c_s$ extension.
  Precedent for editing the high-density end to restore causality, which is what the causal
  cap's per-adiabat construction does.
- Bounds on $c_s$ for context: **P. Bedaque & A. W. Steiner**, *Phys. Rev. Lett.* **114**,
  031103 (2015); **Bounds on the speed of sound in dense matter, and neutron star structure**,
  *Phys. Rev. C* **95**, 045801 (2017); **C. E. Rhoades & R. Ruffini**,
  *Phys. Rev. Lett.* **32**, 324 (1974) for the classical causal maximum-mass argument;
  **S. Altiparmak et al.**, [arXiv:2203.14974](https://arxiv.org/abs/2203.14974).
- Why smoothness of $c_s$ matters downstream: sound-speed discontinuities increase inspiral
  phase error in binary-neutron-star merger simulations, and hydrodynamics codes generally do
  not support non-monotone $c_s$ in the crust — supporting evidence for repairing the data
  rather than clamping at run time.

## 5. Table formats and consuming codes

- **CompOSE** — S. Typel et al., *Eur. Phys. J. A* **58**, 221 (2022);
  [arXiv:1307.5715](https://arxiv.org/abs/1307.5715);
  [reference manual](https://compose.obspm.fr/download/pdf/manual_v3.00.pdf). The format
  behind the planned `io_compose` backend. §4.1.2 of the manual specifies thermodynamic
  consistency checks, and the database deliberately stores pressure, entropy, three chemical
  potentials and both energies *independently* so that consistency can be tested — the same
  redundancy `check_table()` exploits. See also the MUSES CompOSE modules.
- [**PyCompOSE**](https://github.com/computationalrelativity/PyCompOSE) — unofficial Python
  reader for CompOSE ASCII tables; a practical format reference and a cross-check for
  `io_compose`.
- [**weaklib**](https://github.com/starkiller-astro/weaklib) and **thornado** —
  D. Pochik et al., [arXiv:2011.04680](https://arxiv.org/abs/2011.04680). The ORNL EOS and
  opacity table library with its discontinuous-Galerkin consumer; the comparison point for
  derivatives obtained by differentiating a trilinear interpolant.
- Codes that consume tabulated EOS and are the eventual consumers of this library: GRaM-X
  (S. Shankar et al., [arXiv:2210.17509](https://arxiv.org/abs/2210.17509)), AsterX
  (J. V. Kalinani et al., [arXiv:2406.11669](https://arxiv.org/abs/2406.11669)), AthenaK
  (J. Fields et al., [arXiv:2409.10384](https://arxiv.org/abs/2409.10384)), GR-Athena++
  (B. Daszuta et al., [arXiv:2406.05126](https://arxiv.org/abs/2406.05126)), HARM3D+NUC
  (A. Murguia-Berthier et al., [arXiv:2106.05356](https://arxiv.org/abs/2106.05356)), and
  WhiskyTHC.

## 6. Performance portability and the GPU path

- [**singularity-eos**](https://github.com/lanl/singularity-eos) — J. M. Miller et al.,
  *J. Open Source Softw.* **9**(103), 6805 (2024) — and
  [**spiner**](https://github.com/lanl/spiner), its performance-portable interpolation layer.
  The reference answer to "tabulated EOS behind one interface, host and device, many
  architectures", and the closest existing software to the M4 target. The strongest external
  argument for the header-only, allocation-free, POD-view split enforced in
  [`entropy_eos/core/`](entropy_eos/core/). Consumed by Phoebus (B. Barker et al.,
  [arXiv:2410.09146](https://arxiv.org/abs/2410.09146)).
- **Not-quite-transcendental (NQT) functions** — P. C. Hammond et al.,
  *Astrophys. J. Suppl.* (2025), [arXiv:2501.05410](https://arxiv.org/abs/2501.05410); earlier
  J. M. Miller et al., [arXiv:2206.08957](https://arxiv.org/abs/2206.08957). Cheap
  exponent-bit substitutes for `log`/`exp` that retain the properties making logarithmic grids
  good for interpolation: measurably faster than the transcendental versions *and* lower
  error, with the second-order variant integer-aliased, implemented in singularity-EOS and
  benchmarked in AthenaK neutron-star runs. Directly relevant to the choice of logarithm base
  on the GPU path, where it is likely a better answer than either `log` or `log2`.
