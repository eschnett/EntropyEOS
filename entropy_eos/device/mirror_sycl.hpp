// entropy_eos/device/mirror_sycl.hpp
//
// M4 SYCL (Intel oneAPI) RAII mirror (eos-device-interface.md S4b): owns
// device copies of a built EntropyEOS's two coefficient arrays and hands out
// an EntropyEOSView bound to them. Opt-in: never included by entropy_eos.hpp
// and never compiled into the library -- include it only from translation
// units compiled as SYCL (icpx -fsycl). Host-side code by the layout rules
// (CODE.md "Layout"): may allocate and throw.
//
// This is the whole SYCL backend: core/ needs no SYCL-specific line at all
// (eos-device-interface.md S3 -- header-inline functions reached from a
// kernel lambda in the same TU are compiled for device automatically, and
// SYCL_EXTERNAL is cross-TU only, which core/ never is).
//
// Same shape as the tested mirror_cuda.hpp with the two deltas SYCL forces:
// the constructor takes a sycl::queue (copied and kept -- malloc_device needs
// it and the destructor's sycl::free needs the same context; the constructor
// wait()s, so the view is then usable from any queue of that context), and an
// allocation failure is a null return rather than an error code. Uploads are
// synchronous by design, as in the CUDA mirror.

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "entropy_eos/core/adapter_eval.hpp"
#include "entropy_eos/host/adapter_build.hpp"

namespace eeos {

namespace detail {

// malloc_device + host-to-device memcpy of one coefficient blob on `q`,
// throwing std::runtime_error (`what` names the blob) on failure. Returns
// the device pointer; the caller owns it (free with sycl::free on the same
// context).
inline real *sycl_mirror_upload(const std::vector<double> &host, const char *what,
                                ::sycl::queue &q) {
  // The straight memcpy below assumes the device view's element type is the
  // host fit's double; a future `real = float` port must convert instead
  // (eos-device-interface.md S8).
  static_assert(sizeof(real) == sizeof(double), "mirror memcpy assumes real == double");
  real *dev = ::sycl::malloc_device<real>(host.size(), q);
  if (dev == nullptr)
    throw std::runtime_error(std::string("SyclEntropyEOS: malloc_device(") + what + ") failed");
  try {
    q.memcpy(dev, host.data(), host.size() * sizeof(real)).wait();
  } catch (...) {
    ::sycl::free(dev, q);
    throw;
  }
  return dev;
}

} // namespace detail

// Move-only owner of the device-side coefficient copies. The view it hands
// out is valid for the mirror's lifetime; pass it BY VALUE into kernels (it
// is a ~300-byte POD -- see eos-device-interface.md S4a). The source
// EntropyEOS is not needed after construction.
class SyclEntropyEOS {
public:
  SyclEntropyEOS(const EntropyEOS &eos, ::sycl::queue q)
      : q_(std::move(q)), sigma_dev_(detail::sycl_mirror_upload(eos.sigma_coeffs(), "sigma", q_)) {
    try {
      L_dev_ = detail::sycl_mirror_upload(eos.L_coeffs(), "L", q_);
    } catch (...) {
      ::sycl::free(sigma_dev_, q_);
      throw;
    }
    view_ = eos.view_with(sigma_dev_, L_dev_);
  }

  // The null guards keep a moved-from object from touching its moved-from
  // queue; sycl::free must not be given the pointers of a live sibling.
  ~SyclEntropyEOS() {
    if (L_dev_) ::sycl::free(L_dev_, q_);
    if (sigma_dev_) ::sycl::free(sigma_dev_, q_);
  }

  SyclEntropyEOS(const SyclEntropyEOS &) = delete;
  SyclEntropyEOS &operator=(const SyclEntropyEOS &) = delete;

  SyclEntropyEOS(SyclEntropyEOS &&other)
      : q_(std::move(other.q_)), sigma_dev_(other.sigma_dev_), L_dev_(other.L_dev_),
        view_(other.view_) {
    other.sigma_dev_ = nullptr;
    other.L_dev_ = nullptr;
  }
  SyclEntropyEOS &operator=(SyclEntropyEOS &&other) {
    if (this != &other) {
      if (L_dev_) ::sycl::free(L_dev_, q_);
      if (sigma_dev_) ::sycl::free(sigma_dev_, q_);
      q_ = std::move(other.q_);
      sigma_dev_ = other.sigma_dev_;
      L_dev_ = other.L_dev_;
      view_ = other.view_;
      other.sigma_dev_ = nullptr;
      other.L_dev_ = nullptr;
    }
    return *this;
  }

  const EntropyEOSView &view() const { return view_; }

  // The queue the coefficient copies live on (i.e. their context).
  const ::sycl::queue &queue() const { return q_; }

private:
  ::sycl::queue q_;
  real *sigma_dev_ = nullptr;
  real *L_dev_ = nullptr;
  EntropyEOSView view_{};
};

} // namespace eeos
