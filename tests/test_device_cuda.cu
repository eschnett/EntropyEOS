// tests/test_device_cuda.cu
//
// M4 GPU validation + throughput harness (eos-device-interface.md S6):
// compiles core/ under nvcc, mirrors a built EntropyEOS to the device
// (device/mirror_cuda.hpp), runs evaluate / cold con2prim / warm con2prim /
// con2prim_safe over N sampled states in device code, and gates the results
// against the host-side GROUND TRUTH (not bitwise against the CPU: device
// libm differs from glibc by ULPs, so |GPU-CPU| is reported, never gated).
// Plain main(), no doctest: this binary's job is a measured report run
// manually under the scheduler, never part of `make test` (the .cu extension
// keeps it out of the tests/test_*.cpp wildcard). Exit code: 0 pass,
// 1 gate failure, 2 CUDA/setup error.
//
// Build (opt-in Makefile target, or this documented manual line, which must
// always work -- no HDF5, no OpenMP, no static lib):
//
//   nvcc -O2 -std=c++17 -arch=sm_90 --expt-relaxed-constexpr -I. \
//     tests/test_device_cuda.cu entropy_eos/host/table.cpp \
//     entropy_eos/host/synthetic.cpp entropy_eos/host/bspline_fit.cpp \
//     entropy_eos/host/adapter_build.cpp -o tests/test_device_cuda
//
// Real-table mode (-DEEOS_GPU_TEST_HDF5 plus io/check/repair TUs and -lhdf5;
// see the Makefile's gpu-test block):
//
//   nvcc -O2 -std=c++17 -arch=sm_90 --expt-relaxed-constexpr -I. \
//     -DEEOS_GPU_TEST_HDF5 tests/test_device_cuda.cu \
//     entropy_eos/host/table.cpp entropy_eos/host/synthetic.cpp \
//     entropy_eos/host/bspline_fit.cpp entropy_eos/host/adapter_build.cpp \
//     entropy_eos/host/io_stellarcollapse.cpp entropy_eos/host/check.cpp \
//     entropy_eos/host/repair.cpp -lhdf5 -o tests/test_device_cuda
//
// Run: srun -p h200q --gpus=1 ./tests/test_device_cuda [--n N] [--block B]
//                                                      [--table LS220.h5]
//
// --expt-relaxed-constexpr: detail::p2c_nan() calls the constexpr host
// function std::numeric_limits<real>::quiet_NaN() from device code (see
// eos-device-interface.md S3). Fast-math must stay off (same section).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "entropy_eos/core/adapter_eval.hpp"
#include "entropy_eos/core/con2prim.hpp"
#include "entropy_eos/core/prim2con.hpp"
#include "entropy_eos/core/state_policy.hpp"
#include "entropy_eos/device/mirror_cuda.hpp"
#include "entropy_eos/host/adapter_build.hpp"
#include "entropy_eos/host/synthetic.hpp"
#ifdef EEOS_GPU_TEST_HDF5
#include "entropy_eos/host/io_stellarcollapse.hpp"
#include "entropy_eos/host/repair.hpp"
#endif

namespace {

#define CUDA_CHECK(call)                                                                    \
  do {                                                                                      \
    const cudaError_t eeos_err_ = (call);                                                   \
    if (eeos_err_ != cudaSuccess) {                                                        \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                 \
                   cudaGetErrorString(eeos_err_));                                          \
      std::exit(2);                                                                         \
    }                                                                                       \
  } while (0)

// Relative difference with an absolute floor, so exact zeros compare clean.
double rel_diff(double a, double b) {
  const double scale = std::fmax(std::fmax(std::fabs(a), std::fabs(b)), 1e-300);
  return std::fabs(a - b) / scale;
}

// Round-trip error of a recovered primitive against its ground truth. w gets
// an absolute floor of 1 in the denominator (w -> 0 states are legitimate and
// exactly representable; a pure relative error there measures noise).
double rt_err(double got, double truth, double floor_ = 0.0) {
  return std::fabs(got - truth) / std::fmax(std::fabs(truth), std::fmax(floor_, 1e-300));
}

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  const size_t k = static_cast<size_t>(p * (v.size() - 1));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

double maxv(const std::vector<double> &v) {
  double m = 0.0;
  for (const double x : v) m = std::fmax(m, x);
  return m;
}

bool converged(eeos::C2PResult r) {
  return r == eeos::C2PResult::converged_newton || r == eeos::C2PResult::converged_fallback;
}

struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
};

} // namespace

// --- Device kernels: one thread per state, the same core/ calls the CPU
// reference makes. The EntropyEOSView is passed BY VALUE (a ~300-byte kernel
// argument); its two coefficient pointers are the device copies.

__global__ void eval_kernel(eeos::EntropyEOSView eos, const eeos::real *rho, const eeos::real *s,
                            const eeos::real *ye, eeos::EOSPoint *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = eos.evaluate(rho[i], s[i], ye[i], eeos::detail::p2c_nan());
}

__global__ void con2prim_cold_kernel(eeos::EntropyEOSView eos, const eeos::Con2PrimIn *in,
                                     eeos::Con2PrimOptions opts, eeos::Con2PrimOut *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = eeos::con2prim(eos, in[i], opts);
}

__global__ void con2prim_warm_kernel(eeos::EntropyEOSView eos, const eeos::Con2PrimIn *in,
                                     eeos::Con2PrimOptions opts, const eeos::real *s_g,
                                     const eeos::real *w_g, const eeos::real *u_g,
                                     eeos::Con2PrimOut *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = eeos::con2prim(eos, in[i], opts, s_g[i], w_g[i], u_g[i]);
}

__global__ void con2prim_safe_kernel(eeos::EntropyEOSView eos, const eeos::Con2PrimIn *in,
                                     eeos::Con2PrimOptions opts, eeos::PolicyOptions pol,
                                     eeos::Con2PrimSafeOut *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = eeos::con2prim_safe(eos, in[i], opts, pol);
}

int main(int argc, char **argv) {
  int n = 1 << 20;
  int block = 256;
  std::string table_path;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--n") && a + 1 < argc) n = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--block") && a + 1 < argc) block = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--table") && a + 1 < argc) table_path = argv[++a];
    else {
      std::fprintf(stderr, "usage: %s [--n N] [--block B] [--table file.h5]\n", argv[0]);
      return 2;
    }
  }

  // --- Build the EOS on the host. ------------------------------------------
  // Synthetic default: the analytic ideal gas at real-table scale, so
  // coefficient-array traffic is realistic. refine=2 keeps the (serial in
  // this HDF5/OpenMP-free build) kappa/monotonicity scans to seconds -- the
  // refine level only affects build-time audit resolution, never the fitted
  // splines, so CPU and GPU see the identical EOS either way.
  eeos::EntropyEOS eos = [&]() {
    if (table_path.empty()) {
      eeos::SyntheticOptions sopts;
      sopts.nrho = 234;
      sopts.ntemp = 180;
      sopts.nye = 50;
      const eeos::RawTable table = eeos::make_synthetic_table(sopts);
      eeos::BuildOptions bopts;
      bopts.refine = 2;
      return eeos::build_entropy_eos(table, bopts);
    }
#ifdef EEOS_GPU_TEST_HDF5
    eeos::RawTable table = eeos::read_stellarcollapse(table_path);
    const eeos::RepairResult rep = eeos::repair_table(table);
    std::printf("repair: status %d, %zu entries\n", static_cast<int>(rep.status),
                rep.entries.size());
    return eeos::build_entropy_eos(table);
#else
    std::fprintf(stderr, "--table requires a -DEEOS_GPU_TEST_HDF5 build (see file header)\n");
    std::exit(2);
#endif
  }();
  const eeos::EntropyEOSView hview = eos.view();
  std::printf("eos: %s, box x [%g, %g] u [%g, %g] y [%g, %g]\n",
              table_path.empty() ? "synthetic ideal gas (234x180x50)" : table_path.c_str(),
              double(hview.x_lo), double(hview.x_hi), double(hview.u_lo), double(hview.u_hi),
              double(hview.y_lo), double(hview.y_hi));

  // --- Report the device, mirror the EOS. ----------------------------------
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s (sm_%d%d), %d states, block %d\n", prop.name, prop.major, prop.minor, n,
              block);
  const eeos::CudaEntropyEOS mirror(eos);
  const eeos::EntropyEOSView dview = mirror.view();
  std::printf("mirrored %.1f MB of coefficients\n",
              (eos.sigma_coeffs().size() + eos.L_coeffs().size()) * sizeof(double) / 1048576.0);

  // --- Sample N ground-truth states. ----------------------------------------
  // Interior in (x, s, ye) with a 2% srange margin; rapidity uniform in
  // [0, 6); half the states magnetized with B^2/p log-uniform in [1e-3, 10].
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::vector<eeos::real> rho(n), s(n), ye(n), w_true(n), B2(n), cos_vB(n), u_true(n);
  std::vector<eeos::Con2PrimIn> cin(n);
  for (int i = 0; i < n; ++i) {
    const eeos::real x = hview.x_lo + (hview.x_hi - hview.x_lo) * uni(rng);
    rho[i] = std::pow(eeos::real(10), x);
    ye[i] = hview.y_lo + (hview.y_hi - hview.y_lo) * uni(rng);
    const eeos::SRange sr = hview.srange(rho[i], ye[i]);
    s[i] = sr.s_min + (sr.s_max - sr.s_min) * (eeos::real(0.02) + eeos::real(0.96) * uni(rng));
    w_true[i] = eeos::real(6) * uni(rng);
    cos_vB[i] = eeos::real(2) * uni(rng) - eeos::real(1);
    const eeos::EOSPoint pt = hview.evaluate(rho[i], s[i], ye[i], eeos::detail::p2c_nan());
    u_true[i] = pt.u_solved;
    B2[i] = (i % 2) ? pt.p * std::pow(eeos::real(10), eeos::real(-3) + eeos::real(4) * uni(rng))
                    : eeos::real(0);
    const eeos::Prim2ConOut pc =
        eeos::prim2con(hview, rho[i], s[i], ye[i], w_true[i], B2[i], cos_vB[i], u_true[i]);
    cin[i] = eeos::Con2PrimIn{pc.D, pc.tau, pc.D_Y, pc.S_par, pc.S_perp, pc.B2};
  }

  const eeos::Con2PrimOptions opts;
  const eeos::PolicyOptions pol =
      eeos::default_policy(hview, std::pow(eeos::real(10), hview.x_lo));

  // --- CPU reference passes (serial, timed). --------------------------------
  std::vector<eeos::EOSPoint> pt_cpu(n);
  std::vector<eeos::Con2PrimOut> cold_cpu(n), warm_cpu(n);
  std::vector<eeos::Con2PrimSafeOut> safe_cpu(n);
  double t_eval_cpu, t_cold_cpu, t_warm_cpu, t_safe_cpu;
  {
    Timer t;
    for (int i = 0; i < n; ++i)
      pt_cpu[i] = hview.evaluate(rho[i], s[i], ye[i], eeos::detail::p2c_nan());
    t_eval_cpu = t.seconds();
  }
  {
    Timer t;
    for (int i = 0; i < n; ++i) cold_cpu[i] = eeos::con2prim(hview, cin[i], opts);
    t_cold_cpu = t.seconds();
  }
  {
    Timer t;
    for (int i = 0; i < n; ++i)
      warm_cpu[i] = eeos::con2prim(hview, cin[i], opts, s[i], w_true[i], u_true[i]);
    t_warm_cpu = t.seconds();
  }
  {
    Timer t;
    for (int i = 0; i < n; ++i) safe_cpu[i] = eeos::con2prim_safe(hview, cin[i], opts, pol);
    t_safe_cpu = t.seconds();
  }

  // --- GPU passes (one warm-up launch each, then cudaEvent-timed). ---------
  auto dalloc = [](size_t bytes) {
    void *p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    return p;
  };
  auto *rho_d = static_cast<eeos::real *>(dalloc(n * sizeof(eeos::real)));
  auto *s_d = static_cast<eeos::real *>(dalloc(n * sizeof(eeos::real)));
  auto *ye_d = static_cast<eeos::real *>(dalloc(n * sizeof(eeos::real)));
  auto *w_d = static_cast<eeos::real *>(dalloc(n * sizeof(eeos::real)));
  auto *u_d = static_cast<eeos::real *>(dalloc(n * sizeof(eeos::real)));
  auto *cin_d = static_cast<eeos::Con2PrimIn *>(dalloc(n * sizeof(eeos::Con2PrimIn)));
  auto *pt_d = static_cast<eeos::EOSPoint *>(dalloc(n * sizeof(eeos::EOSPoint)));
  auto *cold_d = static_cast<eeos::Con2PrimOut *>(dalloc(n * sizeof(eeos::Con2PrimOut)));
  auto *warm_d = static_cast<eeos::Con2PrimOut *>(dalloc(n * sizeof(eeos::Con2PrimOut)));
  auto *safe_d = static_cast<eeos::Con2PrimSafeOut *>(dalloc(n * sizeof(eeos::Con2PrimSafeOut)));
  CUDA_CHECK(cudaMemcpy(rho_d, rho.data(), n * sizeof(eeos::real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(s_d, s.data(), n * sizeof(eeos::real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ye_d, ye.data(), n * sizeof(eeos::real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(w_d, w_true.data(), n * sizeof(eeos::real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(u_d, u_true.data(), n * sizeof(eeos::real), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(cin_d, cin.data(), n * sizeof(eeos::Con2PrimIn), cudaMemcpyHostToDevice));

  const int grid = (n + block - 1) / block;
  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  auto gpu_time = [&](auto launch) {
    launch(); // warm-up (JIT/caches); results simply overwritten below
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev0));
    launch();
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    return ms / 1e3;
  };

  const double t_eval_gpu =
      gpu_time([&] { eval_kernel<<<grid, block>>>(dview, rho_d, s_d, ye_d, pt_d, n); });
  const double t_cold_gpu =
      gpu_time([&] { con2prim_cold_kernel<<<grid, block>>>(dview, cin_d, opts, cold_d, n); });
  const double t_warm_gpu = gpu_time(
      [&] { con2prim_warm_kernel<<<grid, block>>>(dview, cin_d, opts, s_d, w_d, u_d, warm_d, n); });
  const double t_safe_gpu = gpu_time(
      [&] { con2prim_safe_kernel<<<grid, block>>>(dview, cin_d, opts, pol, safe_d, n); });

  std::vector<eeos::EOSPoint> pt_gpu(n);
  std::vector<eeos::Con2PrimOut> cold_gpu(n), warm_gpu(n);
  std::vector<eeos::Con2PrimSafeOut> safe_gpu(n);
  CUDA_CHECK(cudaMemcpy(pt_gpu.data(), pt_d, n * sizeof(eeos::EOSPoint), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(cold_gpu.data(), cold_d, n * sizeof(eeos::Con2PrimOut), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(warm_gpu.data(), warm_d, n * sizeof(eeos::Con2PrimOut), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(safe_gpu.data(), safe_d, n * sizeof(eeos::Con2PrimSafeOut),
                        cudaMemcpyDeviceToHost));

  // --- Gates and reports. ----------------------------------------------------
  int failures = 0;

  // Gate 1: convergence parity, statistically. Boundary states of the
  // solver's own failure tail flip in BOTH directions under ULP-level libm
  // differences, so a per-state gate is wrong by construction (measured on
  // the first M4c H200 runs at 2^20 states: synthetic 15 cpu / 9 gpu cold
  // failures with 3 one-sided flips; LS220 under this harness's deliberately
  // extreme sampling -- w uniform in [0,6), B^2/p up to 10 -- 3591 cpu /
  // 3608 gpu with 195 one-sided flips, all in the documented hard class at
  // w ~ 4-5.5). The invariant a broken device path would violate is the
  // failure RATE: gate the GPU's total per pass to within 5% + 10 of the
  // CPU's.
  int cpu_cold_conv = 0, cpu_warm_conv = 0, gpu_cold_conv = 0, gpu_warm_conv = 0;
  int parity_cold = 0, parity_warm = 0;
  for (int i = 0; i < n; ++i) {
    const bool cc = converged(cold_cpu[i].result), cg = converged(cold_gpu[i].result);
    const bool wc = converged(warm_cpu[i].result), wg = converged(warm_gpu[i].result);
    cpu_cold_conv += cc;
    gpu_cold_conv += cg;
    cpu_warm_conv += wc;
    gpu_warm_conv += wg;
    if (cc && !cg) {
      if (++parity_cold <= 5)
        std::printf("  parity(cold) state %d: cpu %d gpu %d  D %.6e tau %.6e B2 %.3e w %.3f\n", i,
                    int(cold_cpu[i].result), int(cold_gpu[i].result), double(cin[i].D),
                    double(cin[i].tau), double(cin[i].B2), double(w_true[i]));
    }
    if (wc && !wg) {
      if (++parity_warm <= 5)
        std::printf("  parity(warm) state %d: cpu %d gpu %d\n", i, int(warm_cpu[i].result),
                    int(warm_gpu[i].result));
    }
  }
  std::printf("convergence: cold cpu %d/%d gpu %d/%d (cpu-conv-but-gpu-not: %d); warm cpu %d/%d "
              "gpu %d/%d (cpu-conv-but-gpu-not: %d)\n",
              cpu_cold_conv, n, gpu_cold_conv, n, parity_cold, cpu_warm_conv, n, gpu_warm_conv, n,
              parity_warm);
  const auto fail_bar = [&](int cpu_conv) { return (n - cpu_conv) + (n - cpu_conv) / 20 + 10; };
  if (n - gpu_cold_conv > fail_bar(cpu_cold_conv) || n - gpu_warm_conv > fail_bar(cpu_warm_conv)) {
    std::printf("  GATE FAIL: GPU failure total above the CPU's + 5%% + 10 (cold bar %d, warm "
                "bar %d)\n",
                fail_bar(cpu_cold_conv), fail_bar(cpu_warm_conv));
    ++failures;
  }

  // Gate 2: GPU round-trip primitive errors against ground truth, over that
  // side's converged states, held to the CPU's OWN distribution on the same
  // pass (10x + 1e-12 headroom for ULP-level libm scatter at the p99 level,
  // where the measured GPU/CPU ratio is ~1.1). Absolute solver quality --
  // including the converged-but-far tail this harness's extreme sampling
  // reaches on real tables (max ~1e1, identical on both sides) -- is the M3
  // audits' ledger, not M4's: the device claim is "indistinguishable from
  // the host solver", and that is what is gated.
  struct RtStats {
    double p99_rho, max_rho, p99_s, max_s, p99_w, max_w;
  };
  auto rt_stats = [&](const char *what, const std::vector<eeos::Con2PrimOut> &out) {
    std::vector<double> e_rho, e_s, e_w;
    for (int i = 0; i < n; ++i) {
      if (!converged(out[i].result)) continue;
      e_rho.push_back(rt_err(out[i].rho, rho[i]));
      e_s.push_back(rt_err(out[i].s, s[i]));
      e_w.push_back(rt_err(out[i].w, w_true[i], 1.0));
    }
    const RtStats st{percentile(e_rho, 0.99), maxv(e_rho), percentile(e_s, 0.99),
                     maxv(e_s),               percentile(e_w, 0.99), maxv(e_w)};
    std::printf("%s round trip: rho p99 %.2e max %.2e | s p99 %.2e max %.2e | w p99 %.2e max "
                "%.2e\n",
                what, st.p99_rho, st.max_rho, st.p99_s, st.max_s, st.p99_w, st.max_w);
    return st;
  };
  auto rt_gate = [&](const char *what, const RtStats &gpu, const RtStats &cpu) {
    const auto bar = [](double c) { return 10 * c + 1e-12; };
    if (gpu.p99_rho > bar(cpu.p99_rho) || gpu.max_rho > bar(cpu.max_rho) ||
        gpu.p99_s > bar(cpu.p99_s) || gpu.max_s > bar(cpu.max_s) || gpu.p99_w > bar(cpu.p99_w) ||
        gpu.max_w > bar(cpu.max_w)) {
      std::printf("  GATE FAIL: %s GPU round-trip above 10x the CPU's own distribution\n", what);
      ++failures;
    }
  };
  const RtStats rt_cold_cpu = rt_stats("cpu cold", cold_cpu);
  const RtStats rt_cold_gpu = rt_stats("gpu cold", cold_gpu);
  rt_gate("cold", rt_cold_gpu, rt_cold_cpu);
  const RtStats rt_warm_cpu = rt_stats("cpu warm", warm_cpu);
  const RtStats rt_warm_gpu = rt_stats("gpu warm", warm_gpu);
  rt_gate("warm", rt_warm_gpu, rt_warm_cpu);

  // Report: |GPU - CPU| per evaluated quantity (ULP-level libm scatter;
  // never gated) plus result/flag agreement rates.
  {
    std::vector<double> d_p, d_cs2, d_u, d_rho, d_w;
    int res_agree = 0, flag_agree = 0, safe_flag_agree = 0, safe_res_agree = 0;
    for (int i = 0; i < n; ++i) {
      d_p.push_back(rel_diff(pt_cpu[i].p, pt_gpu[i].p));
      d_cs2.push_back(rel_diff(pt_cpu[i].cs2, pt_gpu[i].cs2));
      d_u.push_back(rel_diff(pt_cpu[i].u_solved, pt_gpu[i].u_solved));
      if (converged(cold_cpu[i].result) && converged(cold_gpu[i].result)) {
        d_rho.push_back(rel_diff(cold_cpu[i].rho, cold_gpu[i].rho));
        d_w.push_back(rel_diff(cold_cpu[i].w, cold_gpu[i].w));
      }
      res_agree += cold_cpu[i].result == cold_gpu[i].result;
      flag_agree += cold_cpu[i].flags == cold_gpu[i].flags;
      safe_res_agree += safe_cpu[i].base.result == safe_gpu[i].base.result;
      safe_flag_agree += safe_cpu[i].policy_flags == safe_gpu[i].policy_flags;
    }
    std::printf("gpu-cpu deltas: eval p %.2e/%.2e cs2 %.2e/%.2e u %.2e/%.2e | cold rho %.2e/%.2e "
                "w %.2e/%.2e (p99/max)\n",
                percentile(d_p, 0.99), maxv(d_p), percentile(d_cs2, 0.99), maxv(d_cs2),
                percentile(d_u, 0.99), maxv(d_u), percentile(d_rho, 0.99), maxv(d_rho),
                percentile(d_w, 0.99), maxv(d_w));
    std::printf("agreement: cold result %.4f%% flags %.4f%% | safe result %.4f%% policy_flags "
                "%.4f%%\n",
                100.0 * res_agree / n, 100.0 * flag_agree / n, 100.0 * safe_res_agree / n,
                100.0 * safe_flag_agree / n);
  }

  // Gate 3: con2prim_safe never fails -- every returned state must be finite
  // on both sides (its contract; policy_flags reports what was repaired).
  {
    int bad = 0;
    for (int i = 0; i < n; ++i) {
      const eeos::Con2PrimSafeOut &o = safe_gpu[i];
      if (!eeos::detail::c2p_is_finite(o.base.rho) || !eeos::detail::c2p_is_finite(o.base.s) ||
          !eeos::detail::c2p_is_finite(o.base.w) || !eeos::detail::c2p_is_finite(o.cons.tau))
        ++bad;
    }
    std::printf("con2prim_safe: %d/%d non-finite outputs on GPU\n", bad, n);
    if (bad) ++failures;
  }

  // Throughput.
  const auto rate = [&](double t) { return n / t / 1e6; };
  std::printf("throughput (Mstates/s, %d states):\n", n);
  std::printf("  %-14s %10s %10s %8s\n", "", "cpu(1core)", "gpu", "ratio");
  std::printf("  %-14s %10.3f %10.1f %8.0f\n", "evaluate", rate(t_eval_cpu), rate(t_eval_gpu),
              t_eval_cpu / t_eval_gpu);
  std::printf("  %-14s %10.3f %10.1f %8.0f\n", "con2prim cold", rate(t_cold_cpu), rate(t_cold_gpu),
              t_cold_cpu / t_cold_gpu);
  std::printf("  %-14s %10.3f %10.1f %8.0f\n", "con2prim warm", rate(t_warm_cpu), rate(t_warm_gpu),
              t_warm_cpu / t_warm_gpu);
  std::printf("  %-14s %10.3f %10.1f %8.0f\n", "con2prim_safe", rate(t_safe_cpu), rate(t_safe_gpu),
              t_safe_cpu / t_safe_gpu);

  CUDA_CHECK(cudaFree(rho_d));
  CUDA_CHECK(cudaFree(s_d));
  CUDA_CHECK(cudaFree(ye_d));
  CUDA_CHECK(cudaFree(w_d));
  CUDA_CHECK(cudaFree(u_d));
  CUDA_CHECK(cudaFree(cin_d));
  CUDA_CHECK(cudaFree(pt_d));
  CUDA_CHECK(cudaFree(cold_d));
  CUDA_CHECK(cudaFree(warm_d));
  CUDA_CHECK(cudaFree(safe_d));
  CUDA_CHECK(cudaEventDestroy(ev0));
  CUDA_CHECK(cudaEventDestroy(ev1));

  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
