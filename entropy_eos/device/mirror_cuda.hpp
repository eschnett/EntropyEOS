// entropy_eos/device/mirror_cuda.hpp
//
// M4 CUDA RAII mirror (eos-device-interface.md S4b): owns device copies of a
// built EntropyEOS's two coefficient arrays and hands out an EntropyEOSView
// bound to them. Opt-in: never included by entropy_eos.hpp and never compiled
// into the library -- include it only from translation units that link the
// CUDA runtime. Only the runtime API is used, so a PLAIN host compiler with
// -I$CUDA/include and -lcudart builds this header; nvcc is needed for the
// consumer's kernels, not for the mirror. Host-side code by the layout rules
// (CODE.md "Layout"): may allocate and throw.
//
// Uploads are synchronous by design: a table mirror is a build-once ~30-70 MB
// cost, and an async variant would let the source EntropyEOS die before the
// copy lands.
//
// mirror_hip.hpp is the mechanical s/cuda/hip/ rename of this file, enforced
// by tests/check_mirror_parallel.sh in CI (eos-device-interface.md S5): when
// editing here, mirror the edit there.

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "entropy_eos/core/adapter_eval.hpp"
#include "entropy_eos/host/adapter_build.hpp"

namespace eeos {

namespace detail {

// cudaMalloc + host-to-device cudaMemcpy of one coefficient blob, throwing
// std::runtime_error (with the runtime's error string, `what` naming the
// blob) on failure. Returns the device pointer; the caller owns it.
inline real *cuda_mirror_upload(const std::vector<double> &host, const char *what) {
  // The straight memcpy below assumes the device view's element type is the
  // host fit's double; a future `real = float` port must convert instead
  // (eos-device-interface.md S8).
  static_assert(sizeof(real) == sizeof(double), "mirror memcpy assumes real == double");
  const std::size_t bytes = host.size() * sizeof(real);
  real *dev = nullptr;
  cudaError_t err = cudaMalloc(&dev, bytes);
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("CudaEntropyEOS: cudaMalloc(") + what +
                             ") failed: " + cudaGetErrorString(err));
  err = cudaMemcpy(dev, host.data(), bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(dev);
    throw std::runtime_error(std::string("CudaEntropyEOS: cudaMemcpy(") + what +
                             ") failed: " + cudaGetErrorString(err));
  }
  return dev;
}

} // namespace detail

// Move-only owner of the device-side coefficient copies. The view it hands
// out is valid for the mirror's lifetime; pass it BY VALUE into kernels (it
// is a ~300-byte POD -- see eos-device-interface.md S4a). The source
// EntropyEOS is not needed after construction.
class CudaEntropyEOS {
public:
  explicit CudaEntropyEOS(const EntropyEOS &eos)
      : sigma_dev_(detail::cuda_mirror_upload(eos.sigma_coeffs(), "sigma")) {
    try {
      L_dev_ = detail::cuda_mirror_upload(eos.L_coeffs(), "L");
    } catch (...) {
      cudaFree(sigma_dev_);
      throw;
    }
    view_ = eos.view_with(sigma_dev_, L_dev_);
  }

  // Errors deliberately ignored (destructors must not throw); freeing a
  // nullptr after a move-out is a documented no-op in the runtime.
  ~CudaEntropyEOS() {
    cudaFree(L_dev_);
    cudaFree(sigma_dev_);
  }

  CudaEntropyEOS(const CudaEntropyEOS &) = delete;
  CudaEntropyEOS &operator=(const CudaEntropyEOS &) = delete;

  CudaEntropyEOS(CudaEntropyEOS &&other) noexcept
      : sigma_dev_(other.sigma_dev_), L_dev_(other.L_dev_), view_(other.view_) {
    other.sigma_dev_ = nullptr;
    other.L_dev_ = nullptr;
  }
  CudaEntropyEOS &operator=(CudaEntropyEOS &&other) noexcept {
    if (this != &other) {
      cudaFree(L_dev_);
      cudaFree(sigma_dev_);
      sigma_dev_ = other.sigma_dev_;
      L_dev_ = other.L_dev_;
      view_ = other.view_;
      other.sigma_dev_ = nullptr;
      other.L_dev_ = nullptr;
    }
    return *this;
  }

  const EntropyEOSView &view() const { return view_; }

private:
  real *sigma_dev_ = nullptr;
  real *L_dev_ = nullptr;
  EntropyEOSView view_{};
};

} // namespace eeos
