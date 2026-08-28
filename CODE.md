# Code design

Draft v0.2, August 24, 2026. Iterated from v0.1. Companion to
`con2prim-entropy-rapidity.md` (physics) and `eos-adapter-F-to-U.md` (adapter design;
its §10 audit list and §11 API are implemented here). Items marked **[decide]** are open.

## Environment

- C++17, multi-core CPU. Parallelism via OpenMP pragmas, used liberally wherever loops
  are independent (guarded by `_OPENMP`; the code still compiles and runs serially
  without it). No `std::thread` machinery.
- C++17 rather than C: `core/` already follows a C-like discipline (POD views, no
  exceptions, no allocation), so the C++ benefit concentrates where it is cheapest to
  keep — RAII around HDF5 handles and memory in the host code, containers for the
  generic table, small templates in the spline/solver kernels, and CUDA (which is C++)
  for M4. If a C or Fortran consumer appears, the plan is a thin `extern "C"` shim over
  the library API, not a rewrite.
- Future: CUDA C++. The library is split along the host/device boundary from day one
  (see layout rules below) so the port is a compiler switch, not a rewrite.
- Code is a library; the standalone executables are thin `main()`s on top of it.
- Naming: library `entropy_eos`, namespace `eeos`.
- Two supported consumption modes, both first-class: (a) **copying the source** — one
  directory to copy, a small fixed set of `.cpp` files added to the consumer's build,
  one umbrella header; (b) **conventional library** — the Makefile builds
  `libentropy_eos.a`, and `make install PREFIX=…` installs it with the headers.
  Consumers write `#include <entropy_eos/…>` in both modes. No
  cmake/configure/autodetection either way. The documented manual compile line must
  always work:
  `c++ -O2 -std=c++17 -fopenmp -I. entropy_eos/host/*.cpp tools/eos_repair.cpp -lhdf5 -o eos_repair`
- **HDF5 is the only external dependency**, confined to the `io_*` translation units.
  Consumers that don't do table I/O themselves (e.g. a GRMHD code embedding only the
  adapter + con2prim) need no HDF5.
- Precision: `double` throughout (`eeos::real` typedef in `defs.hpp`); revisit `float`
  variants for the GPU port.
- Error handling: host-side build/repair/I-O code may throw; `core/` never throws,
  never allocates — errors are status flags in return structs (device-compatible).

## Layout

```
EntropyEOS/
  CODE.md  con2prim-entropy-rapidity.md  eos-adapter-F-to-U.md
  Makefile                     # trivial: HDF5_DIR variable; tools, tests, lib, install
  entropy_eos/
    entropy_eos.hpp            # umbrella header
    core/                      # header-only, device-ready (the CUDA boundary)
      defs.hpp                 #   real, EEOS_HOST_DEVICE macro, flag bits, status enums
      bspline_eval.hpp         #   tensor-product cubic B-spline evaluation, derivs ≤ 2nd
      adapter_eval.hpp         #   EntropyEOSView, EOSPoint, evaluate(), srange()
      prim2con.hpp             #   rapidity-form prim2con                    (M3)
      con2prim.hpp             #   2×2 Newton + nested 1D fallback           (M3)
    host/                      # host-only: owns memory, may use STL/exceptions
      table.hpp/.cpp           #   RawTable: axes + generic named 3D fields + attributes
      units.hpp/.cpp           #   constants, unit conversions, m_B conventions, κ
      synthetic.hpp/.cpp       #   manufactured analytic EOS → RawTable (ground truth)
      check.hpp/.cpp           #   table-level diagnostics, library-first (see test harness)
      repair.hpp/.cpp          #   isotonic + spline-safe + causal-cap repair, change log
      bspline_fit.hpp/.cpp     #   B-spline coefficient fitting (banded solves)  (M2)
      adapter_build.hpp/.cpp   #   RawTable → EntropyEOS (fit, extend, κ, audits) (M2)
      io_stellarcollapse.hpp/.cpp  # the only files including hdf5.h
      io_compose.hpp/.cpp      #   CompOSE/MUSES backend (planned only; same interface)
  tools/
    eos_repair.cpp             # M1
    eos_test.cpp               # M1 (staged; grows with M2/M3)
  tests/
    doctest.h                  # vendored single-header framework (committed; nothing to install)
    test_*.cpp
```

Rules that keep the CUDA port honest:

- `core/` is header-only: no STL containers, no exceptions, no virtual functions, no
  I/O, no allocation. Every function is marked `EEOS_HOST_DEVICE` (expands to
  `__host__ __device__` under `__CUDACC__`, empty otherwise). It operates on POD *view*
  structs — raw pointers + extents into coefficient arrays — so the same evaluation code
  runs on host or device once the arrays are mirrored.
- `host/` owns all memory (owning `EntropyEOS` holds `std::vector`s and hands out an
  `EntropyEOSView`), does all fitting/repair/I-O, and is never needed on the device.
- Warm-start state (`u_solved` = ln T) is threaded explicitly through arguments and
  return values; no hidden mutable state anywhere. `evaluate()` is pure and re-entrant.

## Data model

- `RawTable` is the file's content **verbatim**: axes and fields exactly as stored
  (log10 ρ, log10 T, Ye linear; `logenergy`, `entropy`, … in file units), every dataset
  carried (not just the ones we interpret), attributes included. Verbatim matters: any
  convert-at-read/convert-back-at-write round trip would perturb untouched datasets in
  the last bit and silently break the repair tool's byte-faithful-copy guarantee. Typed
  accessors convert on demand (physical ρ, T, ε + shift, …); wholesale unit conversion
  happens once, at the adapter-build boundary (M2).
- Typed accessors exist for the fields the library interprets: `logenergy`
  (+ `energy_shift`) and `entropy` are load-bearing; `logpress` (and `cs2`, `munu`, …
  if present) are consumed by audits only, per the adapter design.
- Storage order: normalized to `field[kYe][jT][iRho]` (ρ fastest, matching the
  stellarcollapse layout) with `at(i,j,k)` helpers. Table sizes (~10⁶ points) make M1
  performance a non-issue; the spline-*coefficient* layout for the M2 hot loop is a
  separate choice (T fastest, so the inner T-solve walks contiguously).
- `units.hpp` holds the physical constants, unit conversions, and the per-format m_B
  convention. **[decide]** m_B source per table family: read from the file where an
  attribute exists, else a documented per-format constant passed explicitly to the
  reader (a silent mismatch is a classic percent-level bug).

## Table formats

- **LS220 and SRO both ship in the stellarcollapse.org (O'Connor–Ott) HDF5 layout**
  (SRO = Schneider–Roberts–Ott 2017; their code emits this format), so
  `io_stellarcollapse` is the first and only M1 backend: log10-stored axes and
  energy/pressure, linear entropy, `energy_shift` attribute. Use the HDF5 C API (more
  portable than the C++ API, which many installs don't build).
- CompOSE/MUSES (named in v0.1) differ in layout, units, and normalization
  (per-baryon quantities, different energy zero). They sit behind the same
  `RawTable read_table(path, Format)` entry point and can be added without touching
  anything downstream. **Planned only** — no committed milestone.

## Repair harness — `tools/eos_repair` (M1)

```
eos_repair in.h5 out.h5 [--check-only] [--min-slope-s X] [--min-slope-loge X]
           [--no-spline-safe] [--no-spline-safe-3d] [--no-causal-cap] [--cs2-cap X]
           [--log FILE]
```

Purpose: make a table satisfy the hard requirements of `eos-adapter-F-to-U.md` §8 —
above all strict monotonicity in T of entropy and energy at every (ρ, Ye), and (M3f)
causality `0 < c_s² < 1` of the constructed potential — and leave an auditable trail.
Design points:

- **Repairs act on the stored variables** (`entropy`, `logenergy`) directly:
  monotonicity of `logenergy` in T is equivalent to monotonicity of ε (log is monotone,
  the shift constant), so no unit round-trip is needed and the output stays in-format.
- Algorithm, per (iRho, kYe) column along T (columns are independent → OpenMP):
  L2 isotonic regression (pool-adjacent-violators) on `entropy` and `logenergy`
  separately, then a strictification pass imposing the minimum slopes (PAVA leaves flat
  runs; the adapter needs strictly positive σ_T, e_T). Min-slope defaults are still open:
  tune on real LS220 violations first.
- **Causality (M3f, `eos-causality-repair.md`)**: a final stage, on by default
  (`--no-causal-cap` opts out), audits `c_s²` of the *fitted* entropy/logenergy splines
  on the refined grid via the analytic chain rule (κ-free, m_B-free — only `c²` and
  `energy_shift` enter, so auditing the raw-variable fit *is* auditing the adapter's
  interior) and, for violation runs that reach the ρ_max edge, projects `logenergy`
  (only) onto the causal Lipschitz envelope along adiabats: trace each node's adiabat
  down to a causal anchor, cap `∂ln h/∂x|_s` at `cs2_cap = 0.99`, integrate the
  energy-consistency ODE back up, write only where ε drops. Interior violation runs (the
  σ_T pockets) and `c_s² ≤ 0` samples are reported, never edited. The stage carries the
  M2d-1 harness verbatim (audit-driven rounds, per-column re-repair of touched columns,
  best-state tracking) plus a *lexicographic* backstop: the kept state's (4,4,4)
  monotonicity counts must not regress, else the whole stage reverts.
- Structural problems are *fatal*, not repaired — they indicate a broken file, not
  physics noise. Fatal means: non-monotone/non-finite axes, missing `energy_shift`, or
  a missing or non-finite **interpreted** field (`logenergy`, `entropy`). Non-finite
  values in fields the pipeline never reads (the shipped LS220 table carries Inf points
  in `cs2`/`gamma`) are a reported `nonfinite_<field>` violation class instead: they
  pass through the writer byte-identically and must not block repairing the fields we
  do interpret.
- Guarantees: input never modified; deterministic; **idempotent** (a second run reports
  zero changes — enforced by tests). Output = faithful copy of every untouched dataset,
  the two repaired fields, plus a `/repair` provenance group (per-field indices /
  old / new values, parameters, tool version, input checksum) and a human-readable log.
- Exit codes: 0 = already clean, 1 = repaired (or would repair, under `--check-only`),
  2 = fatal structural problem.

## Test harness — `tools/eos_test` (staged; M1 ships the first stage)

The table-level checks are a **library function first**:
`CheckReport check_table(const RawTable&, const CheckOptions&)` in `host/check` — pure
(no I/O side effects), cheap enough to run at startup, with a printable/queryable
report — so an HPC consumer can validate a table in-process right after loading it.
The tools are thin wrappers over it.

`eos_test` is one binary that accretes stages as milestones land ("check F" from v0.1
lives here, plus in `eos_repair --check-only`):

- `--level table` (M1, read-only): axes strictly monotone and finite; all fields
  finite; ε + Δ > 0; s ≥ 0; finite-difference monotonicity maps (σ_T > 0, e_T > 0) with
  violation counts and locations (run on raw and repaired tables to quantify the
  repair); finite-difference Maxwell/consistency metrics δ_T, δ_p against the
  `logpress` column; p > 0; stored `cs2` (if present) vs. FD-derived. Output is
  human-readable: text summary on stdout, optional CSV files for the maps (no HDF5
  report).
- `--level adapter` (M2): build the adapter in memory and run the audit suite of
  `eos-adapter-F-to-U.md` §10 — spline monotonicity on a refined grid, ε_floor/κ
  report, FD-vs-analytic checks of all five U-derivatives, the U_ρs symmetry identity,
  round trips at nodes and random points, c_s² ∈ (0,1) / p > 0 / T̂ > 0 maps,
  extension-seam continuity, and a random soak over (ρ, s, Ye) including out-of-bounds
  probes, with flag statistics, warm/cold iteration histograms, and timings.
- `--level con2prim` (M3): random-state prim2con → con2prim round trips per the parent
  document's deliverable 2 (table samples × w ∈ [0,6] × magnetization ×
  arbitrary ∠(S,B)), accuracy/iteration/timing statistics against tolerances.

**Synthetic ground truth:** `host/synthetic` generates an analytic ideal-gas
(optionally + e± + photon) table as a `RawTable` at any resolution. Unit tests use it
for exactness and grid-convergence checks; `eos_test --synthetic` runs the whole suite
against it (a clean synthetic table must report zero violations; deliberately seeded
violations must be found and repaired). This makes M1 fully testable without any real
table and without con2prim. A deterministic **dirty preset**
(`eos_test --synthetic-dirty`) additionally fabricates the pathology patterns observed
in the real tables — clustered non-monotone entropy across a T-window (LS220's nuclear
transition), flat logenergy plateaus (SRO), a negative-entropy cold corner (SRO), a
smooth `ρ²` energy excess making the constructed `U` superluminal over the top two ρ
layers at every (T, Ye) (the M3f acausal corner), and planted Inf/NaN in auxiliary
fields (LS220's cs2/gamma) — so the full detect → repair → residual-classes-persist
narrative runs in CI.

**Real tables are local-only.** CI never downloads or stores the LS220/SRO files
(~1 GB; Git LFS quotas and stellarcollapse.org bandwidth both rule it out);
`tests/integration.sh` runs them when present under `tables/` (see
`tables/README.md`) and skips them gracefully otherwise. Future options if per-PR
real-data coverage is ever wanted: committed ~10 MB crops of the pathological regions
(via an `eos_crop` tool with provenance attributes), and/or a scheduled weekly workflow
restoring the full tables from the Actions cache.

**Unit tests and CI:** doctest, as a single vendored header committed to `tests/` —
nothing to install, fast to compile, and since `tests/` is not copied into downstream
projects the tests carry no weight for consumers and may grow arbitrarily heavy. CI is
a minimal GitHub Actions workflow: `apt-get install libhdf5-dev`, `make test` (serial
and OpenMP builds).

## M2 design notes

- **Uniform knots.** The B-spline layer requires uniformly spaced axes (in log10 ρ,
  log10 T, linear Ye) — true for stellarcollapse tables and the synthetic generator,
  asserted at adapter build. Uniformity makes evaluation branch-light and GPU-friendly
  (cell index by one multiply, fixed basis polynomials); non-uniform axes (CompOSE?)
  would need a general de Boor layer later.
- **Fitted quantities are the stored ones**: `logenergy` (log10(ε+Δ)) and `entropy`,
  bit-verbatim, with the log10 factors handled analytically in the chain rule — no
  conversion noise enters the fit.
- **Unit contract of the adapter** (where percent-level bugs live, so fixed here):
  ρ in g/cm³ *rescaled by κ* (§ energy zero point of the adapter doc), U = ε/c²
  (dimensionless), p reported as p/c² in g/cm³ (so h = 1 + U + p/ρ is dimensionless),
  s in k_B/baryon, T̂ = U_s = k_B T/(m_B\* c²) dimensionless; the solved table
  temperature T_F is reported alongside in MeV. m_B\* = κ·m_B with m_B the table
  convention (default amu; see open decision) — note that the δ_T audit effectively
  *measures* the table's true m_B, and its value is reported.
- **Banded solver in-tree, not LAPACK.** The spline-fit solves are build-time only
  (milliseconds even for SRO; the hot loop does no linear algebra), the collocation
  matrix is benign (diagonally dominant 1-4-1 rows, fixed bandwidth 4, O(1) condition),
  and LAPACK would break the HDF5-only-dependency and copy-in rules for real platform
  friction (provider variance, Fortran mangling, LAPACKE availability). ~150 tested
  lines instead; the solve sits behind one signature, so swapping LAPACK in later is
  cheap if ever warranted.
- **Staged delivery**: (a) B-spline fit/eval with exactness+convergence tests;
  (b) adapter core — κ re-zeroing, warm-started monotone T-solve, chain rule, EOSPoint,
  srange, flags — with out-of-range s/ρ *clamped and flagged* as a stopgap;
  (c) `eos_test --level adapter` audit suite (the adapter doc §10 list), validated
  against the closed-form U(ρ,s,Ye) of the synthetic gas; (c′) spline-safe repair —
  the §4 audit-driven diffusion loop, per T-column; (d) the smooth C¹/C²
  domain extensions of adapter doc §7 replacing the clamps, re-audited.

**M2 empirical findings (LS220-2009 and SRO-LS220):**

- **m_B convention measured**: with m_B = amu, SRO's δ_T fidelity quantiles sit *flat*
  at ~8.7e-3 = m_n/m_u − 1; rebuilding with the neutron mass collapses them to
  p50 = 1.6e-5, p90 = 1.3e-3 — **SRO's baryon mass is m_n** (`units.hpp
  m_neutron_g`; pass `--m-B` / `BuildOptions::m_B_table_g`). LS220-2009's convention is
  indeterminate: its intrinsic inconsistency floor (δ_T p50 ≈ 0.5–0.9% either way,
  p90 ≈ 40%) buries the signal.
- **Fidelity verdict (the quintic free-energy question)**: SRO is thermodynamically
  consistent at the 1e-5–1e-3 level in the bulk (δ_p p50 = 3e-5) — the two-cubic-spline
  design stands, no quintic refit warranted. LS220-2009 carries genuine percent-level
  inconsistency with heavy tails; a refit would *redefine* its p/T rather than fix
  anything — prefer SRO-family tables for production, LS220-2009 for benchmarks.
- **Residual spline pathologies after spline-safe repair** (per-column loop is 100%
  clean; these are cross-column ρ/Ye tensor-blending artifacts a per-column repair
  cannot reach): LS220 275 σ_u / 20650 L_u refined-sample violations, SRO 6201/221537;
  cold-start round trips land on wrong roots at 11 (LS220) / 711 (SRO) of ~4M nodes;
  soak physicality: T̂≤0 and cs²≤0 at ≤1 random point per 50k, cs²≥1 at ~4% of
  uniform-in-log samples (concentrated at the extreme-ρ corner — map before M3).
- **M2d-1 outcome — local diffusion has a safe limit.** A 3D audit-driven diffusion
  stage was built and *empirically bounded*: in wide near-flat regions (logenergy at
  low T, where ε ≪ energy_shift makes the margin razor-thin) a diffusion step aimed at
  one violation tips barely-positive neighbors negative and the counts run away
  10–100×. The stage therefore tracks the best state seen, reverts after 4
  non-improving rounds, and is backstopped to never leave a field worse than skipping
  it. Net effect on real tables: SRO σ_u 6201→3844 (multi-root nodes 711→610),
  everything else safely reverted to baseline; idempotence holds on both real tables
  (second run: zero changes). The adversarial synthetic wiggle defect does not reach a
  single-run fixed point and is documented as such in integration.sh.

**M3f empirical findings — causal-cap outcome** (`eos-causality-repair.md`, measured
2026-08-27 on the two local tables; "nodes" are the fitted `c_s²` evaluated at table
nodes, "samples" are the stage's own (4,4,4) refined audit):

- **The systematic acausal corner is gone.** LS220-2009: acausal nodes
  **67,107 (4.2%) → 206 (0.013%)**, and the structure that motivated the stage — 6,792
  (T, Ye) columns whose acausal set is a contiguous suffix reaching ρ_max — goes to
  **0 columns**. SRO-LS220: **182,269 (4.3%) → 575 (0.014%)**. Refined samples:
  4,076,091 → 7,622 (LS220), 11,188,863 → 21,283 (SRO). Cost: 24 s (LS220) / 76 s (SRO)
  for the whole `eos_repair` run, serial; 11 s for LS220 on 8 OpenMP threads.
- **What the residual is.** 98.8% (LS220) / 99.5% (SRO) of the remaining refined
  violations sit in *interior* runs — the σ_T-pocket spikes the stage reports and never
  edits, exactly as scoped. Of the residual nodes, LS220's 9 and SRO's 194 (172 of them
  the known full-T sliver at irho 311–312, Ye = 0.645) are those documented pockets; the
  rest (197 LS220 / 381 SRO) are a small new residue at the *joint* ρ_max/T_max corner
  (LS220 irho 226–233 × jT 132–135, i.e. T ≳ 200 MeV; SRO irho 377–390 × jT 159–162),
  where after capping the violating run no longer reaches the ρ_max edge sample and so
  falls outside the edge-anchored scoping rule. `c_s² ≤ 0` counts are untouched
  (LS220 12,022 → 12,022; SRO 6,197 → 6,197 for the in-box fit), i.e. the stage creates
  no new defect of that class.
- **Nothing was traded for it.** Only `logenergy` is edited (66,419 / 182,210 nodes;
  max |Δ log₁₀(ε+Δ)| = 0.081 / 0.085, rms 0.036 / 0.032); `entropy` gains no entry.
  (4,4,4) `L_u` monotonicity: LS220 20,650 → 20,650 (unchanged), SRO 221,537 → 219,886
  (improved); `σ_u` untouched by construction. Both tables stay **idempotent** (a second
  `eos_repair --check-only` reports 0 changes, exit 0), and the output is bit-identical
  across 1/4/8 OpenMP threads.
- **Independent confirmation, via the adapter audit.** `eos_test --level adapter` class C
  `cs2_acausal`: LS220 **7,364 → 132** (worst excess over 1 falls 0.328 → 0.023), SRO
  **7,639 → 135** (0.333 → 0.016). The plain in-box fit and the adapter's own extended
  build agree node for node, so the residual is real data, not an extension artifact.
- **`cs2_cap` sits on a knife edge (open item).** At the approved default 0.99 the stage
  is kept. At **0.95** it does much better — LS220 residual nodes 206 → **16** (the 9
  pockets + 7 at irho 233), refined 7,622 → 248 — and is still kept. At **0.97 or 0.90
  the lexicographic backstop reverts the entire stage**: those projected states would
  have had 840 / 13,461 refined violations (far better than 0.99's 7,622) but `L_u`
  monotonicity counts of 20,652 / 20,683 against the 20,650 baseline. A *2-sample*
  monotonicity regression out of 20,650 therefore discards a ~5,000× causality
  improvement. That is the design's rule working as written (never trade the T-solve's
  hard requirement for causality), but it makes the outcome discontinuous in `cs2_cap` —
  worth revisiting together with the 0.99-vs-0.95 choice.
- **The M3 acceptance metric did *not* move.** `eos_test --level con2prim` at 40,000
  states, before → after: LS220 warm failures 87 → 86, cold 18 → 16; SRO warm 162 → 163,
  cold 32 → 30; `rt_tau` p999/max unchanged on both. So the ~0.2–0.8% failure tail of M3
  open item (i) is **not** caused by in-box `c_s² ≥ 1`: it survives a 55× reduction of
  the adapter's causality violations. The stage's value is therefore the conditioning
  argument (`z_w = z(1−c_s²)tanh w` positive in-box, which is what a GPU
  fixed-iteration path needs) and table validity — not a measured failure-rate win.
  (The tail's actual root cause was found the same day — see the next findings block.)

**M3 failure-tail root cause (post-M3f investigation, 2026-08-27).** The residual
con2prim failure tail (LS220 86 warm + 16 cold, SRO 163 + 30, per 40k warm / 4k cold
states) was dissected state by state: the audit's sampling replicated exactly, every
failing state probed along its trial path `ρ = D/cosh w`, w ∈ [0, 6.5], at fixed trial
s, plus retry ladders. Two disjoint classes, neither of them in-box table physics —
which is why the M3f cap could not move the tail:

- **Class A (~95% of failures; plus 18/40k LS220 and 25/40k SRO *silent wrong-root
  convergences*): the u-high (hot-entropy) extension is acausal, and that breaks the
  fallback's correctness proof.** Every truth state is healthy (c_s² ≈ 0.33, no
  flags), but 86/102 (LS220) / 174/193 (SRO) sit in the top ~5% of the log-T axis. A
  trial w below truth raises ρ_trial = D/cosh(w); s_max(ρ) falls with ρ, so the fixed
  trial s exceeds s_max(ρ_trial) and the T-solve enters the u-high tail — where
  measured c_s² = 0.45 / 0.78 / **1.55** at 1.02× / 1.1× / 1.3× s_max: it crosses 1
  about one grid cell past the seam and saturates ≈ 3.8, numerically identical on
  both tables, so it is the *tail construction*, not the data (the curvature-ramp
  tail guards σ_u/L_u monotonicity but nothing guards causality). With c_s² > 1,
  `z_w < 0` and f1(w; s) is measurably non-monotone (2–6 slope sign flips per path in
  97/102 resp. 185/193 failures) — the §9 inner-solve monotonicity proof does not
  hold on the extended domain, the inner w-solve lands on wrong branches, the outer
  g(s) becomes effectively garbage, and the outcomes are `no_bracket`, `max_iter`,
  and fallback "convergences" onto spurious roots (s off by orders of magnitude,
  round-trip errors ≈ 1; all but one carry `flag_ext_s_high`/`flag_ext_rho_low` on
  the output, so a flag-honoring caller catches them). Fix designed in
  `eos-causal-tail.md` (approval pending).
- **Class B (~5%; LS220 5, SRO ~8 per 40k): radiation-dominated bracket coarseness
  plus an f2 precision floor.** These never touch an extension (paths fully causal,
  z_w > 0): ρ ~ 10⁴–10⁷ g/cc, T ≈ 100–135 MeV, where srange spans 13–14.5 decades. A
  dense scan shows g(s) has ~4 roots there and the true root's sign window is
  10⁻⁷–10⁻³ of the bracket — unresolvable by the 17-point scan, whose local half is
  sized to the local srange; g's s-sensitivity itself is healthy (s·∂g/∂s ≈ 1.3).
  Newton's f2 floor sits between 1e-12 and 1e-10 at these states. Retries with
  tol = 1e-10, a 65-point scan, or 200 Newton iterations recover essentially all of
  class B (and the retry ladders recover 94–97/102 resp. 188–192/193 overall).
  Solver-side follow-ups (out of `eos-causal-tail.md`'s scope): a relative-width
  local scan window, and the same precision-floor acceptance for the Newton path's
  f2 that the outer bisection's polish already has.
- **Grid resolution is the stage's real limit.** The projection controls ε at *nodes*
  while `c_s²` is a second derivative of the fitted spline, so on a grid where a capped
  profile advances `cs2_cap·Δx` per cell the cubic's second-derivative error is
  ~(cs2_cap·Δx)²/12 relative. Real tables (Δlog₁₀ρ ≈ 0.056) sit at ~0.1%, comfortably
  inside the 1% `cs2_max`/`cs2_cap` hysteresis; the synthetic dirty preset's 0.256 dex
  axis sits at ~3% and provably cannot converge, which is why
  `tests/test_causal_cap.cpp` plants the same defect on a ρ-resolved grid (where the
  identical stage reaches exactly zero) while `integration.sh` asserts only detection
  plus reduction on the coarse preset.

**M3g empirical findings — causal extension tails** (`eos-causal-tail.md`, measured
2026-08-27 on the two local tables; "before" is the same binary built at the pre-M3g
commit, same seeds, same states, so every pair below is apples to apples):

- **The u-high extension is causal now.** Public-API scan of the whole u-HIGH band
  (120 ρ columns × 24 Ye × 32 samples across the 8-cell extension = 92,160 points):
  LS220 `c_s² ≥ 1` **88.4% → 0.007%** (max `c_s²` **3.814 → 1.012**), SRO
  **88.0% → 0.009%** (max **3.857 → 1.009**). Before, the crossing began in the first
  blend cell (45% of it) and saturated ≈ 3.8 everywhere beyond; after, cells 1–8 past the
  seam are *exactly* clean and every survivor sits in the blend cell above the known
  residual ρ_max/T_max acausal corner — i.e. the tail now inherits the data's own residue
  (in-box `cs2_acausal` max excess 0.023 / 0.016, unchanged by M3g) instead of
  manufacturing a defect of its own. Deep in the tail `c_s²` measures **0.339 / 0.340**:
  the fitted slopes are b = 4·ln10 and α ≈ 3·ln10 per decade of T, so the far-tail value
  is b/α − 1 = 1/3 — the design's claim, and the physically correct hot-gas asymptote,
  not merely a bound.
- **What the u-high band trades for it: a little `c_s² ≤ 0`, in place of a lot of
  `c_s² ≥ 1`.** Same 92,160-point scan, before → after: `c_s² ≤ 0` **0 → 424 (0.46%)**
  on LS220 and **0 → 459 (0.50%)** on SRO, `p ≤ 0` **0 → 15 (0.016%)** and
  **0 → 26 (0.028%)**, both deep in the tail where the log-σ tail's fixed-s slope turns
  the pressure gradient over. This is a *good* trade, not a hidden cost: the con2prim §9
  inner-solve proof needs `z_w = z(1 − c_s²)tanh w > 0`, i.e. only `c_s² < 1` — a negative
  `c_s²` makes `z_w` more positive, while `c_s² > 1` flipped its sign, which is what broke
  the proof in the first place. Physicality in the extensions is explicitly not guaranteed
  (`eos-adapter-F-to-U.md` §7), the whole band is flagged, and the measured con2prim
  outcome below is the arbiter. It is nonetheless the one thing this change makes *worse*,
  so class E carries a report-only `ext_<band>_cs2_nonpositive` per band (one metric more
  than `eos-causal-tail.md` §5 asked for) to keep it visible.
- **The causal slope clamp never fires on a real table — and always fires on the
  synthetic one.** Seam scan over (ρ*, Ye) at 46,233 (LS220) / 102,311 (SRO) points:
  clamp active **0**, monotonicity-floor conflicts **0**. Both hot seams have
  b/α − 1 ≈ 1/3, a factor 3 below `cs2_ext_cap = 0.99`. The synthetic ideal gas is the
  opposite case — its entropy is *linear* in u, so α = σ_u/σ is small and the raw
  b/α − 1 ≈ 21 — and trips the clamp at 100% of its seam points, which is what keeps the
  clamp arithmetic covered in CI. What it costs there, and only there: class D
  `extension_seam_jump` **4.60e-7 → 1.36e-6** (the clamp is C1, not C2 — it lowers L's
  tail *curvature* and leaves the seam value and slope alone, so U and U_s stay continuous
  and only their derivatives jump), and 456/20,000 `soak_extended` samples acquire
  `c_s² ≤ 0` (worst −0.25) inside that one blend cell, where the strongly negative L_uu
  lives. Both are flagged territory only; `c_s² ≤ 0` does not endanger the §9 inner-solve
  proof, which needs `c_s² < 1`. On the real tables `extension_seam_jump` is
  **bit-identical** before and after (2.301928 / 94.58427).
- **The failure tail — the acceptance metric — at 40,000 states**
  (`eos_test --level con2prim`, before → after):

  | metric | LS220 | SRO |
  |---|---|---|
  | warm failures (no_bracket + max_iter) | 86 → **31** | 163 → **82** |
  | cold failures (of 4,000) | 16 → **8** | 30 → **15** |
  | silent wrong-root convergences | 18 → **0** | 25 → **0** |
  | `c2p_roundtrip` count | 94 → **14** | 164 → **23** |
  | `rt_tau` p999 | 4.61e-3 → **9.63e-13** | 9.46e-1 → **9.88e-13** |
  | `rt_tau` max | 1.000 → 0.968 | 2.118 → 0.481 |
  | `rt_S` p999 | 1.47e-12 → 1.23e-12 | 9.58e-1 → **1.21e-12** |
  | M3e policy interventions | 103 → 31 | 183 → 82 |
  | warm throughput (solves/s) | 1.97e5 → 1.87e5 | 1.58e5 → 1.69e5 |

  (The throughput row is run-to-run noise on a loaded laptop, in both directions; the
  controlled measurement is the micro-benchmark in the "Nothing inside the box moved"
  bullet below.)

  The wrong-root class is gone outright, and `rt_tau`'s p999 collapses by ~5×10⁹ (LS220)
  and ~10¹² (SRO) into the 1e-13 bulk — the design's sharpest prediction, confirmed. The
  policy layer's `ceiling` interventions (the τ-runaway signature of a wrong root) go
  **44 → 0** and **73 → 0**, and its battery's cold-re-solve misses **2 → 0** / **1 → 0**.
- **What the residual is: class B, as scoped.** Path probes at fixed truth s over
  w ∈ [0, 6.5] (scratch `c2p_tail`): failing paths that go acausal anywhere
  **97/102 → 4/39** (LS220) and **185/193 → 5/97** (SRO); paths with `z_w ≤ 0` anywhere
  **97 → 4** and **185 → 5**. Every remaining failure carries residuals at the f2
  precision floor (|f1| ~ 1e-12, |f2| ~ 1e-14) at high rapidity, and **35/39** resp.
  **95/97** converge on a retry with a larger iteration budget — the class-B signature of
  the previous findings block, whose fix is solver-side and out of this design's scope.
  The 4 (LS220) resp. 5 (SRO) still-acausal paths reach an **x**-tail, never the u-high
  one: all 4 LS220 and 4 of the 5 SRO carry `flag_ext_rho_low` (the **x-low** band —
  `eos-causal-tail.md` §5's follow-up, and the same three sampled states k = 20628, 21010,
  24201 on *both* tables, which is the signature of a tail construction rather than of
  table data), the fifth `flag_oob_rho_high` (the hard-invalid x-high band). (The probe's
  `f1_slope_flips` counter stays nonzero on many paths, but it is a finite-difference sign
  test applied to an f1 that is itself ~1e-12 there — noise, not non-monotonicity.
  `z_w > 0` is the meaningful statement, and it now holds on 35/39 resp. 92/97.)
- **The new extension-band map (check_adapter class E) is what makes the rest visible.**
  Deterministic (4-per-cell along the band, 2-per-cell along the other two axes) scan of
  all four bands. M3g changed only the u-HIGH band's construction, so for the other three
  this is a *first measurement*, not a before/after:

  | LS220 band | samples | `c_s² ≥ 1` | `c_s² ≤ 0` | `p ≤ 0` | `σ_u ≤ 0` |
  |---|---|---|---|---|---|
  | u_low  | 1,479,456 | 23 | 11,992 | 2,440 | 0 |
  | u_high | 1,479,456 | 237 | 6,883 | 226 | 0 |
  | x_low  | 858,528 | 500,561 | 79,634 | 0 | 0 |
  | x_high | 858,528 | 846,501 | 0 | 0 | 0 |

  | SRO band | samples | `c_s² ≥ 1` | `c_s² ≤ 0` | `p ≤ 0` | `σ_u ≤ 0` |
  |---|---|---|---|---|---|
  | u_low  | 3,273,952 | 46 | 6,932 | 0 | 0 |
  | u_high | 3,273,952 | 262 | 16,573 | 933 | 0 |
  | x_low  | 1,362,400 | 688,076 | 107,006 | 0 | 0 |
  | x_high | 1,362,400 | 1,353,891 | 0 | 0 | 0 |

  (`c_s² ≤ 0` is the report-only fourth metric; in the u_low and both x bands, which M3g
  does not touch, it is pre-existing.) u_high's `c_s² ≥ 1` residue is the ρ_max/T_max data
  corner (max excess over 1: 0.040 / 0.015).
  **x_low is the real news**: ~50–58% of that band is acausal (max excess 0.79 / 0.84),
  confirming §5's suspicion from a single failing path — that band is a genuine follow-up,
  and it is exactly where the surviving acausal failure paths live. x_high is 98–99%
  acausal but is *report-only* by design: `flag_oob_rho_high` already makes any converged
  state there invalid outright (`eos-adapter-F-to-U.md` §7), so its tail exists only to
  keep an iterate finite until the caller discards it; the other three bands are places a
  converged, flagged, legitimately-used state can live, so they count as violations.
  `σ_u ≤ 0` is zero in every band on both tables — the monotonicity guard holds, in log
  space as it did in linear space.
- **Nothing inside the box moved.** `aeval_extended()` is *bit-identical* to a plain
  `bspline_eval3()` at every in-box point (asserted bit-for-bit, both fields, all 7 spline
  outputs, in `tests/test_adapter_tail.cpp`), and every in-box **warm** `evaluate()` is
  bit-identical across the change at 6,000 random points (4,000 synthetic + 2,000 LS220,
  all 12 `EOSPoint` members compared as raw bits). In-box **cold** solves do move, in
  their last digits only, because the secant seed is built from the extended bracket's
  endpoints and `srange_extended().s_max` legitimately grew: LS220 mean cold iterations
  9.23 → 9.28, 11 of 2,000 points differing in `u_solved` by more than 1e-12 relative
  (all inside the T-solve's own 1e-12 convergence tolerance). Adapter class B
  `roundtrip_T` LS220 **10 → 8**, SRO **3000 → 3001**; `cs2_acausal` 132 → 132 and
  135 → 135; `cs2_nonpositive` 5 → 4 and 4 → 4. `evaluate()` throughput is unchanged
  (micro-benchmark on LS220: cold 0.59 → 0.59 Meval/s, warm 1.34 → 1.52 Meval/s).
- **`srange_extended().s_max` grows, as designed.** LS220 ratio to the physical `s_max`
  is **2.3–3.0× → 3.3–6.0×** (the design's ~10^0.8 ≈ 6 at the top of the range); on
  the synthetic ideal gas it barely moves (1.06–1.11×), because the growth factor is
  `exp(α·(u_ext_hi − u_hi))` and α is small there. Nothing downstream is sized to it;
  `tests/test_adapter_tail.cpp` pins both numbers deliberately.
- **Two 1-in-10⁵ leftovers, both benign and both flagged.** The LS220 physicality soak
  picks up one `flag_maxiter` per 200k–400k samples where it had none: (a) in-box, a cold
  Newton oscillation inside one of the documented σ_u pockets at ρ ≈ 1e14 (10 of 11 warm
  starts at the same state converge; the whole solve is in-box, so the tails cannot be the
  cause — only the seed changed); (b) in the extended soak, a point in the *hard-invalid*
  ρ > ρ_max corner that converged to |σ_ext − s| = 2.5e-12 against a 1.7e-12 threshold,
  i.e. a sub-2× tolerance miss at the corner's roundoff floor. Both are reported, flagged,
  and unrelated to causality.

**M3h empirical findings — class-B solver hardening** (`entropy_eos/core/con2prim.hpp`,
measured 2026-08-27 on the two local tables; "before" is the same binary built at the
M3g commit, same seeds, same states, so every pair below is apples to apples).
Three changes, all comparisons-only or exit-state bookkeeping: **(a)** the convergence
tests on the momentum residual are taken on the *normalized* `f1/cosh(w) = tanh w − V`;
**(b)** the S9 bracket scan's LOCAL half is a geometric ladder of *relative* offsets from
its anchor; **(c)** the Newton loop hands its *best* iterate, not its last, to the S9
fallback.

- **The residual tail is essentially gone, and (a) is what does it.** `eos_test --level
  con2prim --states 40000` (40,000 warm + 4,000 cold), before → after:

  | metric | LS220 | SRO |
  |---|---|---|
  | warm failures (no_bracket + max_iter) | 31 (12+19) → **1 (0+1)** | 82 (23+59) → **0** |
  | cold failures (of 4,000) | 8 (4+4) → **0** | 15 (5+10) → **1 (1+0)** |
  | `c2p_roundtrip` count | 14 → **1** | 23 → **0** |
  | silent wrong-root convergences | 0 → **0** | 0 → **0** |
  | M3e policy interventions | 31 → **1** | 82 → **0** |
  | warm states needing ≥ 5 iterations | 9,953 → **536** | 10,941 → **554** |
  | warm throughput (solves/s) | 2.15e5 → **4.64e5** | 1.76e5 → **4.61e5** |

  `policy_n_valid_touched` stays 0 and the policy battery still PASSes on both. The
  wrong-root row is the scratch `c2p_tail` criterion (a converged warm state whose
  recovered s or w is more than 10% from the truth it was built from) — M3g had already
  taken it to 0, and M3h keeps it there. The throughput row is not noise this time: it is
  the iteration row, one line up (the mis-scaled test was buying ~1 extra Newton
  iteration on most warm solves and a full 60-iteration inner-solve budget on the ones
  that reached the scan).
- **The diagnosis, restated as the fix.** `f1 = sinh w − cosh w·V` is a difference of two
  O(cosh w) terms, so `|f1| ≤ tol` demanded `cosh w` times more accuracy at w = 6 than at
  w = 0 — and *more than double precision can deliver*: V comes from the EOS, so `|f1|`
  has a floor of order `ε·cosh w` times the chain's own amplification. Measured on the
  M3g failure set: the returned states carried `|f2| = 1e-14…1e-15` (converged) against
  `|f1| = 1.2…4.2e-12` at w = 5.5…6.0 where cosh w ≈ 180–200, i.e. `|tanh w − V| ≈ 1e-14`.
  Those states were converged and were being rejected by the scaling alone. At trial
  w ≳ 9, which the inner solve's bisection half reaches routinely, `ε·cosh w` exceeds
  tol = 1e-12 outright, so those inner solves could never terminate and burned their full
  60-iteration budget at every one of the 17 scan points (invisible in the iteration
  histogram, which counts only Newton + *outer* iterations — it shows up in the
  throughput). A directed check on the synthetic gas: of 150 states at w ∈ [8, 11.5],
  **86 fail before, 0 after** (`tests/test_con2prim.cpp` test 9).
- **`tol` now means what it says, and the bulk residuals rise to meet it — the one thing
  this change makes worse.** The old test was, in the bulk, an accidental demand for
  `tol/cosh w`, which bought roughly one extra Newton iteration per solve and with it
  ~100× more accuracy than was asked for. With it gone the Newton stops at the requested
  tolerance: LS220 `rt_tau` p50 1.07e-14 → 2.63e-14, p90 7.36e-14 → 4.00e-13, p99
  5.98e-13 → 9.06e-13 (SRO p90 8.12e-14 → 4.01e-13), and the *prim*-space quantiles move
  with them (LS220 `prim_rho` p99 3.4e-11 → 2.3e-9, `prim_w` p99 5.2e-12 → 3.7e-10) — the
  coupled solve's condition number is ~1e3, so recovered primitives track ~1e3·tol either
  way. Everything stays inside the contract (`rt_tau` p999 9.63e-13 → 9.95e-13, still
  ≈ 1e-12, and the audit's own 1e-8 round-trip threshold is met by all but one state),
  and the tails — which is what a failure metric sees — improve by orders of magnitude:
  `rt_tau` max 9.68e-1 → 1.30e-4 (LS220) and 4.81e-1 → **1.30e-12** (SRO), `rt_S` max
  9.72e-1 → 5.51e-6 and 2.28e-12 → 2.04e-12.
- **The bulk and the tail trade against each other through `tol`, and 1e-12 is the right
  side of it.** Measured on LS220 at 40k warm + 4k cold, varying only
  `Con2PrimOptions::tol`: at **1e-13** the bulk comes back *better than the M3g baseline*
  (`rt_tau` p90 5.57e-14, p99 9.80e-14; `prim_rho` p99 7.2e-11; `prim_w` p99 1.2e-11) but
  **44 warm + 2 cold** states fail; at **1e-14**, **1,115 + 89**. That is the same
  double-precision floor seen from the other side: past ~1e-13 the request outruns what
  the EOS-derived residuals can deliver, and states start failing again. So the recovered
  primitives' ~1e-9 (p99) accuracy at the default is not a bug to be tuned away — it is
  the price of an empty failure tail, and a caller who needs the tighter bulk can buy it
  at 1e-13 with 46 failures per 44,000 states, eyes open.
- **(b) is measurement-neutral on today's tables, and kept on its merits.** With (a) and
  (c) in place, *no* state on either table depends on the scan any more: the pre-M3h
  linear window, the new ladder, and ladders with δ_min ∈ {1e-5, 1e-4, 1e-3} all produce
  **identical** failure, fallback and round-trip counts at 40k states, and a
  forced-fallback stress (`max_iter_newton = 0`, so every state goes through the scan;
  8,000 states) gives 2 / 0 `no_bracket` on LS220 / SRO for all four. Where it does show
  is the geometry that motivated it — an anchor sitting at 1e-5…1e-4 of its own physical
  srange span, which is the real tables' radiation band (s ≈ 1e9 inside a span of 5e13).
  On a synthetic table reproducing that (span/s ≈ 41), with the root 1e-4 away from the
  anchor, the ladder returns a bracket **2.4e-3 of s** while the span-sized window cannot
  get below its own point spacing of **1.19 of s** — a factor 500 (`test_con2prim` test
  10). One measured trap along the way: a *one-sided* ladder with alternating signs (the
  first thing tried) halves each side's REACH, and SRO audit state k = 17457, whose root
  sits at +0.157 of the anchor, then fails where the pre-M3h window succeeded. Emitting
  the ladder in ± PAIRS, so both sides span the full [1e-5, 0.6], recovers it. Coverage
  beats resolution here: the outer Illinois solve resolves a wide bracket cheaply, while
  a missed root is a `failed_no_bracket` outright.
- **(c) is bookkeeping, and its value is visible on the exhaustion path.** With
  `max_iter_newton = 3` on 200 synthetic states, **74 trajectories end more than 2×
  worse than their best** (worst 3,960×) — the unconditional-clamped-step design cutting
  off mid-excursion — and on every one of them the state a deliberately crippled solve
  reports is now the best iterate rather than that stale position (`test_con2prim` test
  11). It also makes the two remaining real-table failures report far better states:
  LS220 k = 29472 returns `f2 = −1.3e-4` where M3g returned 2.7e-3.
- **What the two survivors are — exactly the two classes scoped as out of reach.**
  (i) LS220 k = 29472 (warm, `max_iter`): a fully causal path (`c_s²` max 0.361,
  `z_w > 0` everywhere), healthy truth state, and **no** retry recovers it — 200 Newton
  iterations, a 65-point scan and tol = 1e-10 all fail, exactly as they did before M3h.
  This is the "pathological" residue of the M3g findings block. (ii) SRO k = 14800 (cold,
  `no_bracket`): ρ = 2.3e15 at the top of the ρ axis, path `c_s²` max 1.083 with
  `z_w min = −2.4e18` and `flag_oob_rho_high` — the **x-tail** class, and it converges on
  any of the three retries. Both were failures before M3h too; M3h introduces no new
  failure on either table.
- **Bit-identity, censused rather than argued.** The residuals, the Jacobian and the step
  are untouched, and the relative test is strictly weaker than the absolute one, so a
  solve changes only if its trajectory reaches an iterate satisfying
  `|f2| ≤ tol < |f1| ≤ tol·cosh w` before it would have converged — or if it enters the
  S9 fallback, whose inner-solve stopping test, local scan window and anchor all moved.
  Comparing the recovered (s, w, T) **bit for bit** between the M3g binary and this one
  over the same 40,000 warm + 4,000 cold states: **27,232 / 40,000 warm and 2,620 / 4,000
  cold are bit-identical on LS220** (26,348 and 2,541 on SRO), i.e. ~2/3. Of the states
  that do move, essentially all move because they converged one Newton iteration earlier
  (12,679 of the 12,768 differing LS220 warm states changed their iteration count), by
  `|Δs|/s` p50 6.0e-12, p99 3.5e-9, max 4.0e-8 and `|Δw|` p50 1.1e-11, max 4.7e-8 across
  every state that converged in both builds — the same root, one step less deep. The
  result *enum* changes for 395 (LS220) / 466 (SRO) warm states, almost all of them
  fallback → Newton.

## Milestones

- **M1 (initial deliverable):** `defs`, `table`, `units`, `synthetic`,
  `io_stellarcollapse`, `check`, `repair`; tools `eos_repair` and
  `eos_test --level table`; unit tests. Acceptance: clean runs and repair reports on
  LS220 and SRO tables; idempotence; synthetic clean/seeded-violation tests pass.
- **M2:** ✅ complete (stages a–d landed: B-splines, adapter core, audit harness,
  spline-safe repair with its measured limits, designed domain extensions). The
  adapter presents U(ρ,s,Ye) on an extended domain (ext_cells=8 per side) with C²
  monotone tails, flags judged on the solved state, and U ≥ 0 over the whole extended
  box; synthetic seam jumps ~5e-7, real-table seam maxima confined to the documented
  residual pockets (accept-and-guard, open decision 4).
- **M3:** ✅ core complete (a: prim2con + 2×2 Newton + nested fallback; b: `eos_test
  --level con2prim`; c: fallback hardening). Measured state: warm starts 98.8% Newton
  with conservative-space round trips at p99 ≈ 1e-12 and ~2×10⁵ solves/s; the
  guaranteed-fallback bracket uses a global+local multi-point scan at full precision
  (endpoints-only sign checks provably miss roots — g(s) is non-monotone near extension
  seams and the extended s-bracket can span 10+ decades). Cold starts (M3d): a
  pressure-lagged exact seed — the energy relation solved exactly for z (bracketed
  cubic in q = z+B², exact physical bounds, hydro limit recovered rather than
  special-cased), rapidity exact from the momentum projections, ε from the
  τ-identity, s from the guaranteed-monotone U-bisection; only p is lagged
  (sensitivity dw = tanh w·dp/(ρh) < 1/4·dp/p-ish, bounded). Measured: cold failures
  143→0 (LS220) / 131→4 (SRO) per 4000 states; at 40k statistics cold is within 2×
  of warm, sharing only the acausal-corner tail; warm path bit-identical; cold
  throughput ×9; no tuned constants (seed_passes = 3; s-bisection cap 16, 8 suffices
  for a GPU fixed-trip count). Known open items: (i) ~~the shared ~0.2–0.8% failure
  tail~~ — **root-caused 2026-08-27** (see "M3 failure-tail root cause" above) and now
  **closed**: ~95% was the acausal u-high extension breaking the inner solve's
  monotonicity proof (fixed by **M3g** below — that class is gone, the wrong-root class
  with it), the remaining ~5% was a mis-scaled momentum-residual tolerance plus
  bracket-scan coarseness (fixed by **M3h** below). What remains is **2 states in 88,000**
  across both tables, and neither is class B: one pathological LS220 state that no retry
  ladder recovers, and one SRO state on the acausal **x-tail** — i.e. the residual is now
  (ii) the x-low/x-high band follow-up of `eos-causal-tail.md` §7, and (iii) the RePrimAnd
  benchmark (external library build — decide separately).
- **M3f:** ✅ complete — the causal-cap repair stage of `eos-causality-repair.md`
  (`RepairOptions::causal_cap`, on by default; `eos_repair --no-causal-cap` / `--cs2-cap
  X`), which makes the *data* causal before the fit rather than clamping `c_s²` at run
  time (a clamped `cs2` would correspond to no single potential `U` and would silently
  break the Newton's identities). Measured outcome above.
- **M3g:** ✅ complete — the causal extension tails of `eos-causal-tail.md`, the run-time
  counterpart of M3f: the u-HIGH σ tail is built in **log space** (`g = lnσ` run through
  the same curvature-ramp machinery, mapped back), which makes the far tail's fixed-s
  slope constant and equal to the radiation asymptote `b/α − 1 = 1/3` instead of crossing
  `c_s² = 1` one grid cell past every hot seam; and L's u-high tail carries a **causal
  slope clamp** (`b_eff ≤ (1 + cs2_ext_cap)·α_eff`, `BuildOptions::cs2_ext_cap = 0.99`,
  the monotonicity floor winning any conflict). `check_adapter()` grew the class E
  **extension-band map** over all four bands. `eos-adapter-F-to-U.md` §7's guiding
  principle gained the word *causal*. Measured outcome above; the x-low band's acausality
  and the class-B solver work are the tracked follow-ups (`eos-causal-tail.md` §7).
- **M3h:** ✅ complete — the class-B solver work of `eos-causal-tail.md` §7, entirely
  inside `core/con2prim.hpp` and entirely in the *comparisons*: the convergence tests on
  the momentum residual are taken on the normalized `f1/cosh(w) = tanh w − V` (so
  `Con2PrimOptions::tol` means the same thing at every rapidity, instead of demanding
  `cosh w` times more accuracy than double precision can supply above w ≈ 9); the S9
  bracket scan's LOCAL half is a geometric ladder of *relative* offsets from its anchor
  rather than a linear window sized to a physical srange span that can be 5×10⁴ times the
  anchor; and the Newton loop hands its *best* iterate, not its last, to the fallback as
  scan anchor and warm start, with every failure exit reporting a state no worse than it.
  No new tunables, no budget changes, no change to the residuals, the Jacobian, the step,
  or the M3d seed. Measured outcome above: warm+cold failures 39 → 1 (LS220) and 97 → 1
  (SRO) per 44,000 states, `c2p_roundtrip` 14 → 1 and 23 → 0, warm throughput ×2.1–2.6,
  and the two survivors are the pathological and x-tail states scoped as out of reach.
  The cost, documented in the findings block: bulk round-trip residuals rise to the
  requested `tol` (`rt_tau` p90 7e-14 → 4e-13) because the old test was over-strict by
  `cosh w`; the `tol` sweep there shows that buying the old bulk back (1e-13) costs 46
  failures per 44,000 states, so the default stays 1e-12.
- **M4:** CUDA: compile `core/` under nvcc, mirror coefficient arrays to device,
  fixed-iteration evaluate/con2prim variants.

## Open decisions

1. ~~Repair min-slope defaults~~ — superseded: the spline-safe audit-driven loop (M2c′)
   adapts to local jump sizes, which no fixed min-slope could; the strict-slope floors
   remain as cheap placeholders beneath it.
2. ~~m_B convention~~ — resolved empirically for SRO (neutron mass; see M2 findings);
   LS220-2009 indeterminate below its own inconsistency floor. Default stays amu;
   per-table override via `BuildOptions::m_B_table_g` / `eos_test --m-B`.
3. Whether/when an `extern "C"` shim for C/Fortran consumers is warranted (not before a
   concrete consumer asks).
4. ~~Residual cross-column spline violations and multi-root pockets~~ — resolved:
   **accept and guard**. Data-side local diffusion has hit its safe limit (M2d-1
   outcome above); the residual pockets are mapped by the audits, harm random states
   at the ≤1-per-50k level, and the safeguarded T-solve never fails on them — so
   con2prim (M3) guards against them (flags, fallback), and shape-constrained
   monotone tensor fitting is deferred unless M3 testing shows actual recovery
   failures.
5. ~~Whether the acausal high-density corner is repairable in the data~~ — resolved by
   M3f (see the causal-cap outcome above): yes, as a logged physics edit to `logenergy`,
   with the residual and the `cs2_cap` sensitivity measured. Still open under it: whether
   the default `cs2_cap = 0.99` should move to 0.95, and whether the lexicographic
   backstop's "no monotonicity regression at all" rule wants a tolerance.
