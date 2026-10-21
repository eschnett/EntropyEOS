# CLAUDE.md

Guidance for Claude Code when working in this repository. The `*.md` design documents
are the authoritative description of the package; this file is a map to them plus the
working rules that are easy to miss.

## What this is

`EntropyEOS` (library `entropy_eos`, namespace `eeos`) is a C++17 library for tabulated
nuclear equations of state in GRMHD codes: it checks and repairs stellarcollapse-format
HDF5 tables, turns the $F(\rho, T, Y_e)$ table into a single potential $U(\rho, s, Y_e)$
(B-spline fit with C² extension tails), and provides `prim2con`/`con2prim` built on
that potential, iterating on entropy $s$ and rapidity $w$. Version 1.1.0 (M1–M4
complete).

## Documents

| file | contents |
|---|---|
| `README.md` | user-facing overview, physics background, example, GPU usage, status |
| `CODE.md` | code design: environment, versioning, layout, data model, harnesses, milestones with measured results, open decisions |
| `con2prim-entropy-rapidity.md` | physics of the inversion |
| `eos-adapter-F-to-U.md` | the $F \to U$ adapter design (§10 audits, §11 API) |
| `eos-causality-repair.md` | the causal-cap repair stage (M3f) |
| `eos-causal-tail.md` | causal extension tails (M3g u-high, M3i x-low) |
| `eos-device-interface.md` | CUDA/HIP/SYCL device interface (M4); §8 lists deferred follow-ups |
| `RELATED.md` | related software and literature |
| `tables/README.md` | how to download the EOS tables, citations |
| `TODO.md` | **Erik's working notes — never modify this file** |

Read the relevant design document before changing the corresponding code; the source
comments refer back to their sections. When a change alters behavior or measured
results, update the documents (in particular CODE.md's "Milestones" and findings
blocks) in the same change.

## Build and test

```bash
make                  # libentropy_eos.a
make test             # build and run every tests/test_*.cpp (doctest)
make tools            # tools/eos_repair, tools/eos_test, tools/eos_crop
make integration      # tests/integration.sh; real-table part runs only if tables/*.h5 exist
make test SAN=1       # ASan + UBSan
make clean
```

- On macOS the default `c++` is Apple clang without OpenMP, so `OPENMP` is empty by
  default; with gcc pass `OPENMP=-fopenmp CXX=g++`. HDF5 defaults to MacPorts
  (`HDF5_DIR=/opt/local`); other systems set `HDF5_INC`/`HDF5_LIB` (see
  `.github/workflows/ci.yml` for Ubuntu).
- A single test: `make tests/test_con2prim && ./tests/test_con2prim`; doctest filters
  such as `-tc="con2prim*"` select test cases.
- Tools document their options under `--help`.
- Every object and binary depends on every header, so a header edit rebuilds everything;
  `make clean` is never needed to get a correct build.
- CI (`.github/workflows/ci.yml`) runs `tests/check_mirror_parallel.sh`, then
  `make test` and `make integration` with OpenMP, then `make test SAN=1`.

GPU (opt-in, never part of `make`/`make test`): `make tests/test_device_cuda` (synthetic
table) or `make tests/test_device_cuda_hdf5` (adds `--table`), with `NVCC=` and
`GPU_ARCH=`; the Makefile comments give the exact lines. Runs happen on the Symmetry
cluster (H200, `h200q` partition).

## Ground rules

- **Plain build, no machinery.** No cmake, configure, or autodetection. Wildcards mean a
  new `entropy_eos/host/*.cpp`, `tests/test_*.cpp`, `tools/*.cpp`, or header needs no
  Makefile edit. The manual compile line must keep working:
  `c++ -O2 -std=c++17 -fopenmp -I. entropy_eos/host/*.cpp tools/eos_repair.cpp -lhdf5 -o eos_repair`.
  GPU recipes name their sources explicitly, never `$^` (which would pass headers to nvcc).
- **`core/` is device code.** Header-only; no STL containers, no exceptions, no virtual
  functions, no I/O, no allocation; every function `EEOS_HOST_DEVICE`; state lives in POD
  views; errors are status flags in return structs; no hidden mutable state (warm-start
  state goes through arguments and return values). Never use fast-math: the NaN/finite
  probes in `core/` (`v != v`, `(v - v) == 0`) break under it.
- **`host/`** owns memory, does fitting/repair/I-O, may use STL and exceptions, and is
  never needed on the device.
- **HDF5 is the only external dependency**, and only `io_stellarcollapse.{hpp,cpp}`
  includes `hdf5.h`.
- **Tools are thin `main()`s**; physics and table logic belong in the library.
- **`RawTable` is verbatim** file content (file units, every dataset and attribute);
  unit conversion happens at the adapter-build boundary. Do not convert at read time.
- **`eeos::real`** (`double`, in `defs.hpp`) should be used throughout rather than
  `double`; watch literal constants.
- **OpenMP** pragmas are guarded by `_OPENMP`; the code must also build and run serially.
- **Device mirrors:** `device/mirror_hip.hpp` must stay the mechanical `cuda`→`hip`
  rename of `mirror_cuda.hpp` (enforced by `tests/check_mirror_parallel.sh`). Edit
  both together. The umbrella header never includes `device/`.
- **Tests** use the vendored `tests/doctest.h`. The synthetic EOS (`host/synthetic`)
  supplies closed-form ground truth, and the `--synthetic-dirty` preset reproduces
  real-table pathologies so that CI covers them without tables.

## Versioning and releases

- The version lives only in `entropy_eos/core/version.hpp`; follow CODE.md
  "Versioning" → "Releasing" for a bump (also update the prose in CODE.md and README's
  "Status and testing").
- **No git tags, ever.** The `main` history plus `version.hpp` are the record.
- Changing a default in a way that moves results beyond round-off requires at least a
  MINOR bump and a Milestones entry with measured before/after numbers. Options stay
  settable so the old behavior can be pinned.

## Tables

`tables/*.h5` (LS220, SRO, DD2, SFHo; a few hundred MB each) are not committed; see
`tables/README.md`. Tests that need them skip themselves when they are absent.
`tables/LS220_san_crop.h5` is a cropped fixture for sanitizer runs, regenerated by
`make san-fixture`. DD2/SFHo use the amu baryon mass (the library default); SRO uses
m_n.
