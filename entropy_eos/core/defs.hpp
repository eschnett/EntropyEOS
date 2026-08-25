// entropy_eos/core/defs.hpp
//
// Core definitions shared by every header in entropy_eos/core/. This header is
// part of the device-ready boundary (see CODE.md "Layout"): header-only, no
// STL containers, no exceptions, no allocation, so it can be compiled by
// nvcc unchanged once the CUDA port (M4) lands.

#pragma once

namespace eeos {

// Working precision throughout the library. `float` variants are a possible
// future GPU specialization (see CODE.md "Environment"); not attempted yet.
using real = double;

} // namespace eeos

// EEOS_HOST_DEVICE marks every function in core/ so the same source compiles
// for host and device once nvcc is in the loop (M4). Under a CUDA compiler it
// expands to the usual dual-callable qualifiers; otherwise it is empty.
#if defined(__CUDACC__)
#define EEOS_HOST_DEVICE __host__ __device__
#else
#define EEOS_HOST_DEVICE
#endif

namespace eeos {

// Bit flags reported by core:: evaluation and host:: repair/check code to
// describe what happened at a point without throwing or logging (core/ must
// stay allocation- and exception-free). Bits may be OR-ed together.
constexpr unsigned flag_clamp_ye = 1u << 0;      // Ye was clamped into range
constexpr unsigned flag_ext_s_low = 1u << 1;     // point used the low-entropy extension
constexpr unsigned flag_ext_s_high = 1u << 2;    // point used the high-entropy extension
constexpr unsigned flag_ext_rho_low = 1u << 3;   // point used the low-density extension
constexpr unsigned flag_oob_rho_high = 1u << 4;  // rho above the table's high edge (no extension)
constexpr unsigned flag_maxiter = 1u << 5;       // inner (T-)solve hit the iteration cap

// Coarse-grained outcome of an operation (evaluation, repair, ...). `ok` means
// nothing notable happened; `repaired` means the result is usable but some
// flag above was set or a value was adjusted; `fatal` means the result must
// not be trusted (e.g. a structural problem in the input).
enum class Status { ok, repaired, fatal };

} // namespace eeos
