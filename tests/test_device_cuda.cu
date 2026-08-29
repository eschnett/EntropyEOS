// tests/test_device_cuda.cu
//
// M4 GPU validation harness (eos-device-interface.md S6): compiles core/
// under nvcc, mirrors a built EntropyEOS's two coefficient arrays to the
// device, and runs evaluate / prim2con / con2prim / con2prim_safe in device
// code against the CPU reference. Plain main(), no doctest: this binary's job
// is a measured report run manually under the scheduler, never part of
// `make test` (the .cu extension keeps it out of the tests/test_*.cpp
// wildcard). Exit code: 0 pass, 1 mismatch, 2 CUDA/setup error.
//
// Build (opt-in Makefile target, or this documented manual line, which must
// always work -- no HDF5, no OpenMP, no static lib):
//
//   nvcc -O2 -std=c++17 -arch=sm_90 --expt-relaxed-constexpr -I. \
//     tests/test_device_cuda.cu entropy_eos/host/table.cpp \
//     entropy_eos/host/synthetic.cpp entropy_eos/host/bspline_fit.cpp \
//     entropy_eos/host/adapter_build.cpp -o tests/test_device_cuda
//
// Run: srun -p h200q --gpus=1 ./tests/test_device_cuda
//
// --expt-relaxed-constexpr: detail::p2c_nan() calls the constexpr host
// function std::numeric_limits<real>::quiet_NaN() from device code (see
// eos-device-interface.md S3). Fast-math must stay off (same section).

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "entropy_eos/core/adapter_eval.hpp"
#include "entropy_eos/core/con2prim.hpp"
#include "entropy_eos/core/prim2con.hpp"
#include "entropy_eos/core/state_policy.hpp"
#include "entropy_eos/host/adapter_build.hpp"
#include "entropy_eos/host/synthetic.hpp"

namespace {

#define CUDA_CHECK(call)                                                                    \
  do {                                                                                      \
    const cudaError_t eeos_err_ = (call);                                                   \
    if (eeos_err_ != cudaSuccess) {                                                         \
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

__global__ void prim2con_kernel(eeos::EntropyEOSView eos, const eeos::real *rho, const eeos::real *s,
                                const eeos::real *ye, const eeos::real *w, const eeos::real *B2,
                                const eeos::real *cos_vB, eeos::Prim2ConOut *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = eeos::prim2con(eos, rho[i], s[i], ye[i], w[i], B2[i], cos_vB[i]);
}

__global__ void con2prim_kernel(eeos::EntropyEOSView eos, const eeos::Con2PrimIn *in,
                                eeos::Con2PrimOptions opts, eeos::Con2PrimOut *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = eeos::con2prim(eos, in[i], opts);
}

__global__ void con2prim_safe_kernel(eeos::EntropyEOSView eos, const eeos::Con2PrimIn *in,
                                     eeos::Con2PrimOptions opts, eeos::PolicyOptions pol,
                                     eeos::Con2PrimSafeOut *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = eeos::con2prim_safe(eos, in[i], opts, pol);
}

int main() {
  // --- Build the synthetic ideal-gas EOS on the host (HDF5-free). ---------
  const eeos::RawTable table = eeos::make_synthetic_table();
  const eeos::EntropyEOS eos = eeos::build_entropy_eos(table);
  const eeos::EntropyEOSView hview = eos.view();

  // --- Report the device. --------------------------------------------------
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  // --- Mirror the two coefficient arrays and rebind the view (M4a does this
  // by hand; the M4b CudaEntropyEOS mirror wraps exactly these lines). ------
  const auto coeff_count = [](const eeos::BsplineView3 &b) {
    return static_cast<size_t>(b.nx + 2) * (b.nu + 2) * (b.ny + 2);
  };
  const size_t nsig = coeff_count(hview.sigma), nL = coeff_count(hview.L);
  eeos::real *sigma_dev = nullptr, *L_dev = nullptr;
  CUDA_CHECK(cudaMalloc(&sigma_dev, nsig * sizeof(eeos::real)));
  CUDA_CHECK(cudaMalloc(&L_dev, nL * sizeof(eeos::real)));
  CUDA_CHECK(cudaMemcpy(sigma_dev, hview.sigma.c, nsig * sizeof(eeos::real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(L_dev, hview.L.c, nL * sizeof(eeos::real), cudaMemcpyHostToDevice));
  eeos::EntropyEOSView dview = hview;
  dview.sigma.c = sigma_dev;
  dview.L.c = L_dev;
  std::printf("mirrored %.1f MB of coefficients\n", (nsig + nL) * sizeof(eeos::real) / 1048576.0);

  // --- One mid-box magnetized state, end to end. ---------------------------
  const int n = 1;
  const eeos::real rho = std::pow(eeos::real(10), (hview.x_lo + hview.x_hi) / 2);
  const eeos::real ye = (hview.y_lo + hview.y_hi) / 2;
  const eeos::SRange sr = hview.srange(rho, ye);
  const eeos::real s = (sr.s_min + sr.s_max) / 2;
  const eeos::real w = eeos::real(0.5);
  const eeos::real cos_vB = eeos::real(0.7);

  // CPU reference.
  const eeos::EOSPoint pt_cpu = hview.evaluate(rho, s, ye, eeos::detail::p2c_nan());
  const eeos::real B2 = eeos::real(0.3) * pt_cpu.p; // moderately magnetized
  const eeos::Prim2ConOut p2c_cpu = eeos::prim2con(hview, rho, s, ye, w, B2, cos_vB);
  const eeos::Con2PrimIn cin{p2c_cpu.D, p2c_cpu.tau, p2c_cpu.D_Y, p2c_cpu.S_par, p2c_cpu.S_perp,
                             p2c_cpu.B2};
  const eeos::Con2PrimOptions opts;
  const eeos::Con2PrimOut c2p_cpu = eeos::con2prim(hview, cin, opts);
  const eeos::PolicyOptions pol = eeos::default_policy(hview, std::pow(eeos::real(10), hview.x_lo));
  const eeos::Con2PrimSafeOut safe_cpu = eeos::con2prim_safe(hview, cin, opts, pol);

  // Device inputs/outputs.
  eeos::real *rho_d, *s_d, *ye_d, *w_d, *B2_d, *cos_d;
  eeos::EOSPoint *pt_d;
  eeos::Prim2ConOut *p2c_d;
  eeos::Con2PrimIn *cin_d;
  eeos::Con2PrimOut *c2p_d;
  eeos::Con2PrimSafeOut *safe_d;
  CUDA_CHECK(cudaMalloc(&rho_d, sizeof(eeos::real)));
  CUDA_CHECK(cudaMalloc(&s_d, sizeof(eeos::real)));
  CUDA_CHECK(cudaMalloc(&ye_d, sizeof(eeos::real)));
  CUDA_CHECK(cudaMalloc(&w_d, sizeof(eeos::real)));
  CUDA_CHECK(cudaMalloc(&B2_d, sizeof(eeos::real)));
  CUDA_CHECK(cudaMalloc(&cos_d, sizeof(eeos::real)));
  CUDA_CHECK(cudaMalloc(&pt_d, sizeof(eeos::EOSPoint)));
  CUDA_CHECK(cudaMalloc(&p2c_d, sizeof(eeos::Prim2ConOut)));
  CUDA_CHECK(cudaMalloc(&cin_d, sizeof(eeos::Con2PrimIn)));
  CUDA_CHECK(cudaMalloc(&c2p_d, sizeof(eeos::Con2PrimOut)));
  CUDA_CHECK(cudaMalloc(&safe_d, sizeof(eeos::Con2PrimSafeOut)));
  CUDA_CHECK(cudaMemcpy(rho_d, &rho, sizeof(eeos::real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(s_d, &s, sizeof(eeos::real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ye_d, &ye, sizeof(eeos::real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(w_d, &w, sizeof(eeos::real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B2_d, &B2, sizeof(eeos::real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(cos_d, &cos_vB, sizeof(eeos::real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(cin_d, &cin, sizeof(eeos::Con2PrimIn), cudaMemcpyHostToDevice));

  eval_kernel<<<1, 32>>>(dview, rho_d, s_d, ye_d, pt_d, n);
  prim2con_kernel<<<1, 32>>>(dview, rho_d, s_d, ye_d, w_d, B2_d, cos_d, p2c_d, n);
  con2prim_kernel<<<1, 32>>>(dview, cin_d, opts, c2p_d, n);
  con2prim_safe_kernel<<<1, 32>>>(dview, cin_d, opts, pol, safe_d, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  eeos::EOSPoint pt_gpu;
  eeos::Prim2ConOut p2c_gpu;
  eeos::Con2PrimOut c2p_gpu;
  eeos::Con2PrimSafeOut safe_gpu;
  CUDA_CHECK(cudaMemcpy(&pt_gpu, pt_d, sizeof(pt_gpu), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&p2c_gpu, p2c_d, sizeof(p2c_gpu), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&c2p_gpu, c2p_d, sizeof(c2p_gpu), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&safe_gpu, safe_d, sizeof(safe_gpu), cudaMemcpyDeviceToHost));

  // --- Compare (relative, 1e-10 bar: CUDA libm vs glibc differ by ULPs, so
  // bitwise equality is not the contract -- eos-device-interface.md S6). ----
  const double bar = 1e-10;
  int failures = 0;
  const auto check = [&](const char *what, double cpu, double gpu) {
    const double d = rel_diff(cpu, gpu);
    const bool ok = d <= bar;
    if (!ok) ++failures;
    std::printf("  %-16s cpu % .17e  gpu % .17e  rel %.2e%s\n", what, cpu, gpu, d,
                ok ? "" : "  MISMATCH");
  };

  std::printf("state: rho %.6e  s %.6f  ye %.4f  w %.2f  B2 %.3e\n", double(rho), double(s),
              double(ye), double(w), double(B2));
  std::printf("evaluate:\n");
  check("U", pt_cpu.U, pt_gpu.U);
  check("p", pt_cpu.p, pt_gpu.p);
  check("h", pt_cpu.h, pt_gpu.h);
  check("cs2", pt_cpu.cs2, pt_gpu.cs2);
  check("u_solved", pt_cpu.u_solved, pt_gpu.u_solved);
  std::printf("prim2con:\n");
  check("D", p2c_cpu.D, p2c_gpu.D);
  check("tau", p2c_cpu.tau, p2c_gpu.tau);
  check("S_par", p2c_cpu.S_par, p2c_gpu.S_par);
  check("S_perp", p2c_cpu.S_perp, p2c_gpu.S_perp);
  std::printf("con2prim (cpu result %d gpu result %d, cpu iters %d gpu iters %d):\n",
              int(c2p_cpu.result), int(c2p_gpu.result), c2p_cpu.iters_newton, c2p_gpu.iters_newton);
  check("rho", c2p_cpu.rho, c2p_gpu.rho);
  check("s", c2p_cpu.s, c2p_gpu.s);
  check("ye", c2p_cpu.ye, c2p_gpu.ye);
  check("w", c2p_cpu.w, c2p_gpu.w);
  std::printf("con2prim_safe (cpu policy_flags %#x gpu %#x):\n", safe_cpu.policy_flags,
              safe_gpu.policy_flags);
  check("rho", safe_cpu.base.rho, safe_gpu.base.rho);
  check("s", safe_cpu.base.s, safe_gpu.base.s);
  check("w", safe_cpu.base.w, safe_gpu.base.w);
  check("tau(cons)", safe_cpu.cons.tau, safe_gpu.cons.tau);
  if (c2p_cpu.result != c2p_gpu.result || safe_cpu.policy_flags != safe_gpu.policy_flags)
    ++failures;

  CUDA_CHECK(cudaFree(rho_d));
  CUDA_CHECK(cudaFree(s_d));
  CUDA_CHECK(cudaFree(ye_d));
  CUDA_CHECK(cudaFree(w_d));
  CUDA_CHECK(cudaFree(B2_d));
  CUDA_CHECK(cudaFree(cos_d));
  CUDA_CHECK(cudaFree(pt_d));
  CUDA_CHECK(cudaFree(p2c_d));
  CUDA_CHECK(cudaFree(cin_d));
  CUDA_CHECK(cudaFree(c2p_d));
  CUDA_CHECK(cudaFree(safe_d));
  CUDA_CHECK(cudaFree(sigma_dev));
  CUDA_CHECK(cudaFree(L_dev));

  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
