# The device interface: CUDA, AMD (HIP), and Intel (SYCL) GPUs (M4)

Draft v0.1, August 28, 2026. Companion to CODE.md's "Layout" ("Rules that keep
the CUDA port honest") and "Environment" sections, whose ground rules this
design inherits, and to RELATED.md §6 (performance portability and the GPU
path), whose prior art it leans on. The milestone text this implements —
amended below in §8 — is CODE.md's M4.

**Design in one paragraph.** The product of M4 is not a GPU runtime; it is the
guarantee that `entropy_eos/core/` compiles under nvcc, hipcc, and icpx `-fsycl`,
plus a two-pointer mirroring contract that lets any consumer who owns device
memory build a device-valid `EntropyEOSView` from two coefficient uploads.
`core/` was split for this from day one: header-only, no STL containers, no
exceptions, no virtual functions, no allocation, every function
`EEOS_HOST_DEVICE`, all state in POD views. The entire device-side state of a
built EOS is one `EntropyEOSView` — two `BsplineView3` members whose single
`const real *c` each is the only pointer in the struct, everything else scalars
passed by value. Mirroring is therefore: upload two contiguous `double` arrays
(13–35 MB per field for the real tables), rebind two pointers, pass the view by
value into kernels. Everything vendor-specific (the RAII mirror classes, the
GPU test) is opt-in convenience layered on that contract and kept out of every
existing build path.

## 1. Goal and scope

- `core/` compiles and runs under **nvcc** (tested: Symmetry h200q, H200/sm_90,
  CUDA 13.2), **hipcc** (AMD), and **icpx -fsycl** (Intel) — one source, no
  forks. Only CUDA hardware is available; §5 is the trustworthiness argument
  for the untested backends.
- A vendor-neutral mirroring API on `EntropyEOS` (host side) for consumers
  whose framework owns device memory (Kokkos, RAJA, AMReX/CarpetX, plain
  runtime): export the two coefficient blobs, rebind a view.
- Three opt-in, header-only RAII mirror classes in a new `entropy_eos/device/`
  directory for framework-free consumers and for our own tests, one per vendor
  runtime.
- A GPU validation + throughput harness (`tests/test_device_cuda.cu`), gated on
  ground truth, run on Symmetry; measured numbers recorded in CODE.md per
  sub-stage.

The library never launches kernels for a consumer and owns no batch API: a
GRMHD code calls `evaluate()`/`prim2con()`/`con2prim()`/`con2prim_safe()`
inside its own kernels, exactly as it does inside its own OpenMP loops on the
CPU. The test harness's kernels are the reference for what such a consumer
writes.

## 2. Ground rules (inherited, restated as they bind M4)

- **No cmake, no configure, no autodetection** (CODE.md "Environment"). The
  GPU test builds through explicit opt-in Makefile variables
  (`make tests/test_device_cuda NVCC=... GPU_ARCH=...`) and a documented manual
  compile line that must always work. `make`/`make test`/`make tools` never
  touch a GPU toolchain.
- **`core/` discipline unchanged.** M4's only `core/` edit is the
  `EEOS_HOST_DEVICE` generalization in `defs.hpp` (§3). Host behavior must stay
  bit-identical: the macro still expands empty for plain host compilers.
- **Fast-math is forbidden**, now a documented contract: `core/` probes NaN and
  finiteness by hand (`(v != v)` in `aeval_is_nan()`, `((v - v) == 0)` in
  `c2p_is_finite()`) precisely so no `<cmath>` classification functions are
  needed in device code — and both probes are constant-folded to the wrong
  answer under `-ffast-math`/`--use_fast_math`/`-ffinite-math-only`. Consumers
  must not compile `core/` with any of these. (FMA contraction is fine and
  stays on; it is the production configuration everywhere.)
- **No new dependencies.** `device/` headers include only their own vendor
  runtime header (`<cuda_runtime.h>` / `<hip/hip_runtime.h>` / `<sycl/sycl.hpp>`)
  and are never included by `entropy_eos.hpp` nor compiled into
  `libentropy_eos.a`; a consumer that ignores them builds exactly as before.

## 3. Compiler matrix and the macro layer

The only portability construct `core/` needs is the function annotation, and
only two of the three vendors need even that:

| Compiler | Defines | `EEOS_HOST_DEVICE` |
|---|---|---|
| nvcc, clang `-x cuda` | `__CUDACC__` | `__host__ __device__` |
| hipcc / amdclang (AMD) | `__HIPCC__` (hipcc-on-NVIDIA also defines `__CUDACC__`) | `__host__ __device__` |
| icpx `-fsycl` (Intel) | `SYCL_LANGUAGE_VERSION`, neither of the above | *(empty — correct, see below)* |
| plain host compiler | — | *(empty, unchanged)* |

```cpp
#if defined(__CUDACC__) || defined(__HIPCC__)
#define EEOS_HOST_DEVICE __host__ __device__
#else
#define EEOS_HOST_DEVICE
#endif
```

SYCL needs no annotation at all: device code is whatever the kernel lambda
reaches, and a header-inline function defined in the kernel's own translation
unit is compiled for device automatically. `SYCL_EXTERNAL` exists only for
functions *defined in a different TU*, which cannot happen here — `core/` is
header-only by rule. This is why the SYCL column is empty rather than a third
macro branch, and why the SYCL backend needs no `core/` support code
whatsoever.

No further macros are introduced — no `EEOS_DEVICE_COMPILE`, no forceinline.
Nothing in `core/` needs a device-pass-only branch today, and unused
portability macros rot.

Known per-compiler seams, handled empirically (fix on demonstrated failure,
never preemptively):

- `detail::p2c_nan()` (`core/prim2con.hpp`) calls the constexpr host function
  `std::numeric_limits<real>::quiet_NaN()` from device-annotated code; nvcc
  wants `--expt-relaxed-constexpr` for that (a long-stable flag —
  singularity-eos carries the same one). The flag rides in the documented
  compile line, not in code. Fallback if it ever becomes objectionable: a
  guarded `__builtin_nan("")` in `p2c_nan()`.
- `core/`'s complete std-math inventory is `fabs cosh pow sqrt tanh sinh log10
  cbrt atanh acosh log exp floor` (plus `<limits>`), all std-qualified. All
  thirteen have device overloads under nvcc/hipcc and sit in DPC++'s supported
  std-math subset. If a compiler rejects one, the fix is a minimal
  per-function shim in `defs.hpp` under that compiler's guard.

## 4. Mirroring

### 4a. The vendor-neutral contract (primary)

The two coefficient blobs are the *only* memory `EntropyEOS::view()` points at
that outlives the call; everything else in an `EntropyEOSView` is scalars
copied by value. `EntropyEOS` therefore gains exactly three members:

```cpp
const std::vector<double> &sigma_coeffs() const;   // (nx+2)(nu+2)(ny+2) doubles
const std::vector<double> &L_coeffs() const;
// view() with the two coefficient pointers replaced by caller-owned copies.
EntropyEOSView view_with(const real *sigma_c, const real *L_c) const;
```

A consumer uploads the two vectors with whatever allocator its framework
provides, calls `view_with()` with the device pointers, and passes the
resulting view **by value** into kernels (~300 bytes of kernel argument). That
is the entire interop story with Kokkos/RAJA/AMReX — a POD view passed by
value needs no interop layer. The same contract also covers exotic cases
(pinned host memory, managed memory, multi-GPU: one `view_with()` per device
copy). `view_with(view().sigma.c, view().L.c)` is `view()` — asserted
bit-identically in `tests/test_device_api.cpp`.

### 4b. Vendor RAII mirrors (opt-in convenience)

New directory `entropy_eos/device/`, header-only, one class per vendor with
deliberately distinct names (no ODR/namespace games; `eeos::sycl` in
particular is avoided because it would shadow `::sycl`):

- `device/mirror_cuda.hpp` — `class CudaEntropyEOS`. Ctor
  `explicit CudaEntropyEOS(const EntropyEOS &)`: `cudaMalloc` + `cudaMemcpy`
  per blob, then `view_ = eos.view_with(...)`. `view()` returns the rebound
  view (pass it by value to kernels; the mirror object itself stays on the
  host). Dtor `cudaFree`s (errors ignored — dtors don't throw); move-only;
  throws `std::runtime_error` with the `cudaGetErrorString` text on any
  runtime error, freeing the first blob if the second upload fails (host-side
  convention: `host/` may throw). Uploads are synchronous by design: a table
  mirror is a build-once ~30–70 MB cost, and an async variant would let the
  source `EntropyEOS` die before the copy lands. Includes only
  `<cuda_runtime.h>`: a plain host compiler with `-I$CUDA/include -lcudart`
  builds it — nvcc is needed for the consumer's kernels, not for the mirror.
- `device/mirror_hip.hpp` — `class HipEntropyEOS`, a **mechanical `cuda→hip`
  rename** of the CUDA header. The runtime subset used (Malloc, Memcpy, Free,
  GetErrorString, GetLastError) is exactly the subset HIP defines 1:1.
- `device/mirror_sycl.hpp` — `class SyclEntropyEOS`, same shape with the two
  deltas SYCL forces: the ctor takes `sycl::queue` (copied and kept —
  `sycl::malloc_device` needs it and the dtor's `sycl::free` needs the same
  context; the ctor `wait()`s so the view is usable on any queue of that
  context), and allocation failure is a null return, turned into the same
  `std::runtime_error`.

## 5. Why the untested backends can be trusted

No AMD or Intel GPU is available to this project today. The design makes the
untested code either *nonexistent* or *mechanically derived from tested code*:

- **SYCL:** the backend consists of `mirror_sycl.hpp` alone. `core/` needs no
  SYCL-specific line (§3) — what runs in a SYCL kernel is the same
  header-inline C++ that the CPU test suite executes on every commit.
- **HIP:** `mirror_hip.hpp` must equal `sed 's/hip/cuda/g; s/Hip/Cuda/g;
  s/HIP/CUDA/g'` of the *tested* `mirror_cuda.hpp` (comment lines aside),
  enforced by `tests/check_mirror_parallel.sh` in CI — pure text, no toolchain.
  Any future edit to the CUDA mirror that is not mirrored into the HIP header
  fails CI. This is a stronger guarantee than a shared macro-mapped
  implementation (§7), because each header stays readable, vendors' documented
  names stay greppable, and the derivation is checked rather than assumed.
- Both are labeled "compile-untested until first user report" in README, and
  `icpx -fsycl -fsyntax-only` / hipcc syntax checks run opportunistically
  wherever a toolchain is available (non-gating).

## 6. Validation and measurement

`tests/test_device_cuda.cu` — plain `main()`, no doctest (its job is a
measured report run manually under SLURM; precedent: `tests/integration.sh`).
The `.cu` extension keeps it invisible to the `tests/test_*.cpp` wildcard, so
`make test` never sees it.

- Synthetic mode (the milestone gate, HDF5-free by construction — its host
  closure is exactly `host/{table,synthetic,bspline_fit,adapter_build}.cpp`):
  build the analytic ideal-gas EOS at real-table scale, sample N states
  (default 2^20) across the physical box with a magnetized subset and
  rapidities w ∈ [0, 6), produce ground-truth conservatives via host
  `prim2con`, then run evaluate / cold `con2prim` / warm `con2prim` /
  `con2prim_safe` on CPU (serial reference) and GPU (one thread per state) and
  compare.
- **Gates (exit nonzero), anchored on ground truth, not bitwise:** every
  CPU-converged state must converge on GPU, and GPU round-trip primitive
  errors must meet the same absolute bars the CPU baseline meets. Bitwise
  GPU=CPU equality is impossible by construction — glibc and CUDA libm differ
  by ULPs in `pow`/`cosh`/`atanh`, and FMA contraction differs — so
  |GPU − CPU| per primitive (max, p99) is *reported*, expected in the
  1e-12…1e-9 band, alongside the flag/`C2PResult` agreement rate
  (disagreements listed, not gated: a borderline newton/fallback flip under
  ULP-level libm differences is legitimate).
- Throughput: evaluate/s and cold/warm/safe con2prim/s on the H200 vs the
  measured ~2e5 warm solves/s on one CPU core (CODE.md M3), plus the
  `-Xptxas -v` register/spill report. Registers are reported, not tuned: the
  library exposes no `__launch_bounds__`; occupancy is the consumer's knob.
- Real-table mode (recorded alongside, not gating): `--table <path>` behind
  `#ifdef EEOS_GPU_TEST_HDF5`, adding `io_stellarcollapse.cpp` and `-lhdf5` to
  the compile line; run on Symmetry against the LS220 table in
  `/mnt/beegfs/eschnetter/EOS/`. Real-table spline pathologies are where
  con2prim divergence actually lives, so the warm/cold split there is the
  number a GRMHD consumer wants.

## 7. Alternatives rejected

- **Kokkos/RAJA (or any portability framework) as the interface** — a heavy
  dependency against the no-dependency rule, and unnecessary: consumers who
  use those frameworks pass the POD view into their kernels as-is. The
  singularity-eos comparison in RELATED.md §6 cuts both ways — it is the
  closest existing software *and* its Kokkos dependency is what this library's
  consumers don't all share.
- **One macro-mapped CUDA/HIP mirror header** (`EEOS_RT(Malloc)` expanding per
  vendor) — trades two readable files in the vendors' own documented dialects
  for a third private dialect; the sed-diff CI check gives the same
  no-drift guarantee without it (§5).
- **`__CUDACC__`-only macro kept, HIP via hipcc's CUDA-compatibility** —
  hipcc targeting AMD defines `__HIPCC__`, not `__CUDACC__`; the old macro
  silently compiles `core/` host-only under hipcc, which is exactly the M4
  defect being fixed.
- **Texture objects / `__ldg` for coefficient fetches** — `const real *`
  loads already ride the read-only data path on sm_90+; revisit only if
  profiling shows fetch-bound kernels (seam: `BsplineView3` is the single
  coefficient-access type, `bspline_eval3()` the single reader).
- **`__constant__` memory for the view** — the view is a by-value kernel
  argument, which lands in the same constant-bank storage without the
  library owning a symbol.
- **A batch/kernel-launch API in the library** — consumers own their kernels
  and launch configuration; the test harness's kernels are the documented
  reference. Revisit if a framework-free consumer asks.

## 8. Deferred (with seams), and the milestone amendment

- **Fixed-iteration variants** — deferred out of M4 entirely (decision
  2026-08-28): the fixed-trip option lands as its own later stage once GPU
  profiling data exists to size it. The seams are already marked in code:
  `con2prim.hpp`'s seed loops carry the "drop the two breaks" note and the
  measured `seed_s_iters = 8` GPU budget; `evaluate()`'s T-solve exits and the
  2×2 Newton's convergence test are warp-uniform caps whose removal costs real
  work (6–12× more spline samples for the T-solve), so any future design must
  bring an H200 measurement, not an argument. CODE.md's M4 line is amended
  accordingly (compile + mirror + validation; fixed-iteration split out).
- **`real = float`** — a GPU float path needs its own accuracy study first
  (con2prim's 1e-12 tolerances are meaningless in float). Seam: the single
  `eeos::real` typedef; nothing in M4 hard-codes `double`.
- **NQT / log2 transcendentals** (RELATED.md §6) — accuracy study first.
  Seam: the `log10`/`pow` call sites in the cell mapping and chain rule; a
  `defs.hpp` shim can interpose without API change.
- **CI GPU runner** — none exists; the Symmetry procedure recorded in CODE.md
  is the documented substitute. The HIP rename check does run in CI (text
  only).

## 9. Implemented — measured results

*(appended per sub-stage as M4a–M4d land; see CODE.md "Milestones" for the
running record)*
