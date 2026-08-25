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
      repair.hpp/.cpp          #   isotonic repair + change log (see repair harness)
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
eos_repair in.h5 out.h5 [--check-only] [--min-slope-s X] [--min-slope-loge X] [--log FILE]
```

Purpose: make a table satisfy the hard requirements of `eos-adapter-F-to-U.md` §8 —
above all strict monotonicity in T of entropy and energy at every (ρ, Ye) — and leave an
auditable trail. Design points:

- **Repairs act on the stored variables** (`entropy`, `logenergy`) directly:
  monotonicity of `logenergy` in T is equivalent to monotonicity of ε (log is monotone,
  the shift constant), so no unit round-trip is needed and the output stays in-format.
- Algorithm, per (iRho, kYe) column along T (columns are independent → OpenMP):
  L2 isotonic regression (pool-adjacent-violators) on `entropy` and `logenergy`
  separately, then a strictification pass imposing the minimum slopes (PAVA leaves flat
  runs; the adapter needs strictly positive σ_T, e_T). Min-slope defaults are still open:
  tune on real LS220 violations first.
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
transition), flat logenergy plateaus (SRO), a negative-entropy cold corner (SRO), and
planted Inf/NaN in auxiliary fields (LS220's cs2/gamma) — so the full
detect → repair → residual-classes-persist narrative runs in CI.

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
  Addressing the residual (2D/3D audit-driven smoothing vs. flag-and-guard in
  con2prim) is folded into stage (d).

## Milestones

- **M1 (initial deliverable):** `defs`, `table`, `units`, `synthetic`,
  `io_stellarcollapse`, `check`, `repair`; tools `eos_repair` and
  `eos_test --level table`; unit tests. Acceptance: clean runs and repair reports on
  LS220 and SRO tables; idempotence; synthetic clean/seeded-violation tests pass.
- **M2:** `bspline_fit`/`bspline_eval`, `adapter_build`/`adapter_eval` (T-solve, chain
  rule, extensions, κ); `eos_test --level adapter`.
- **M3:** `prim2con`, `con2prim` (coupled 2×2 Newton + nested 1D fallback);
  `eos_test --level con2prim`; benchmarks vs. RePrimAnd.
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
4. Residual cross-column spline violations and multi-root pockets: 2D/3D audit-driven
   smoothing vs. flag-and-guard in con2prim — decide in M2 stage (d).
