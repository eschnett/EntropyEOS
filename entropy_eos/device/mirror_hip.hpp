// entropy_eos/device/mirror_hip.hpp
//
// M4 HIP (AMD ROCm) RAII mirror (eos-device-interface.md S4b): owns device
// copies of a built EntropyEOS's two coefficient arrays and hands out an
// EntropyEOSView bound to them. Opt-in: never included by entropy_eos.hpp and
// never compiled into the library -- include it only from translation units
// that link the HIP runtime (-I$ROCM/include, -lamdhip64; or build the TU
// with hipcc). Host-side code by the layout rules (CODE.md "Layout"): may
// allocate and throw.
//
// Uploads are synchronous by design: a table mirror is a build-once ~30-70 MB
// cost, and an async variant would let the source EntropyEOS die before the
// copy lands.
//
// THIS FILE IS THE MECHANICAL s/cuda/hip/ RENAME of the tested
// mirror_cuda.hpp (eos-device-interface.md S5: HIP hardware is not available
// to this project; the rename of a tested file is the trustworthiness
// argument). Enforced by tests/check_mirror_parallel.sh in CI: never edit
// this file directly -- edit mirror_cuda.hpp and re-derive.

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "entropy_eos/core/adapter_eval.hpp"
#include "entropy_eos/host/adapter_build.hpp"

namespace eeos {

namespace detail {

// hipMalloc + host-to-device hipMemcpy of one coefficient blob, throwing
// std::runtime_error (with the runtime's error string, `what` naming the
// blob) on failure. Returns the device pointer; the caller owns it.
inline real *hip_mirror_upload(const std::vector<double> &host, const char *what) {
  // The straight memcpy below assumes the device view's element type is the
  // host fit's double; a future `real = float` port must convert instead
  // (eos-device-interface.md S8).
  static_assert(sizeof(real) == sizeof(double), "mirror memcpy assumes real == double");
  const std::size_t bytes = host.size() * sizeof(real);
  real *dev = nullptr;
  hipError_t err = hipMalloc(&dev, bytes);
  if (err != hipSuccess)
    throw std::runtime_error(std::string("HipEntropyEOS: hipMalloc(") + what +
                             ") failed: " + hipGetErrorString(err));
  err = hipMemcpy(dev, host.data(), bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    hipFree(dev);
    throw std::runtime_error(std::string("HipEntropyEOS: hipMemcpy(") + what +
                             ") failed: " + hipGetErrorString(err));
  }
  return dev;
}

} // namespace detail

// Move-only owner of the device-side coefficient copies. The view it hands
// out is valid for the mirror's lifetime; pass it BY VALUE into kernels (it
// is a ~300-byte POD -- see eos-device-interface.md S4a). The source
// EntropyEOS is not needed after construction.
class HipEntropyEOS {
public:
  explicit HipEntropyEOS(const EntropyEOS &eos)
      : sigma_dev_(detail::hip_mirror_upload(eos.sigma_coeffs(), "sigma")) {
    try {
      L_dev_ = detail::hip_mirror_upload(eos.L_coeffs(), "L");
    } catch (...) {
      hipFree(sigma_dev_);
      throw;
    }
    view_ = eos.view_with(sigma_dev_, L_dev_);
  }

  // Errors deliberately ignored (destructors must not throw); freeing a
  // nullptr after a move-out is a documented no-op in the runtime.
  ~HipEntropyEOS() {
    hipFree(L_dev_);
    hipFree(sigma_dev_);
  }

  HipEntropyEOS(const HipEntropyEOS &) = delete;
  HipEntropyEOS &operator=(const HipEntropyEOS &) = delete;

  HipEntropyEOS(HipEntropyEOS &&other) noexcept
      : sigma_dev_(other.sigma_dev_), L_dev_(other.L_dev_), view_(other.view_) {
    other.sigma_dev_ = nullptr;
    other.L_dev_ = nullptr;
  }
  HipEntropyEOS &operator=(HipEntropyEOS &&other) noexcept {
    if (this != &other) {
      hipFree(L_dev_);
      hipFree(sigma_dev_);
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
