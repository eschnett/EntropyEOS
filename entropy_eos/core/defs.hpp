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
// Bits 6-7 are reserved for future core/con2prim.hpp solver flags, so that the
// M3e policy bits below can be masked off as one contiguous group.

// M3e invalid-state policy flags (core/state_policy.hpp; design doc
// con2prim-entropy-rapidity.md S11). These describe what the POLICY LAYER did
// to a state, never what the EOS evaluation found -- a caller can therefore
// separate "the table extension was used" (bits 0-5 above) from "the state was
// repaired/excised" (bits 8-15) by masking. All of them are set on
// Con2PrimSafeOut::policy_flags, whose value 0 means "the input was already
// policy-valid and the output conservatives are the input, bit-identically".
//
// Bits 8-15 (contiguous group, see flag_pol_any below):
constexpr unsigned flag_pol_atmosphere = 1u << 8;   // the whole state was replaced by the atmosphere
                                                    // (hydro excision: rho = rho_atm, s = s_atm, v = 0,
                                                    // Ye per policy, B^2 passed through)
constexpr unsigned flag_pol_ceiling = 1u << 9;      // a collapse ceiling (D_max/tau_max, or the
                                                    // outer-function "tau above the hottest state"
                                                    // diagnosis) was exceeded; the state was excised to
                                                    // atmosphere (then flag_pol_atmosphere is set too) or
                                                    // projected onto the ceiling primitives
constexpr unsigned flag_pol_s_floored = 1u << 10;   // s was raised to the PHYSICAL srange(rho,Ye).s_min
                                                    // (also: tau below the coldest compatible value)
constexpr unsigned flag_pol_s_ceiled = 1u << 11;    // s was lowered to the PHYSICAL srange(rho,Ye).s_max
constexpr unsigned flag_pol_w_capped = 1u << 12;    // rapidity was clamped into [0, w_cap] (either end)
constexpr unsigned flag_pol_rho_clamped = 1u << 13; // rho was clamped into [rho_atm, rho_ceiling] without
                                                    // a full atmosphere reset (in practice: the ceiling
                                                    // side; see core/state_policy.hpp)
constexpr unsigned flag_pol_ye_clamped = 1u << 14;  // Ye was clamped into the table's [y_lo, y_hi]
constexpr unsigned flag_pol_nonfinite = 1u << 15;   // a non-finite (or D <= 0) input was seen; the state
                                                    // was excised to atmosphere (flag_pol_atmosphere too)

// Mask of every M3e policy bit -- `flags & flag_pol_any` answers "did the
// policy layer touch this point?" without enumerating the bits.
constexpr unsigned flag_pol_any = flag_pol_atmosphere | flag_pol_ceiling | flag_pol_s_floored |
                                  flag_pol_s_ceiled | flag_pol_w_capped | flag_pol_rho_clamped |
                                  flag_pol_ye_clamped | flag_pol_nonfinite;

// Coarse-grained outcome of an operation (evaluation, repair, ...). `ok` means
// nothing notable happened; `repaired` means the result is usable but some
// flag above was set or a value was adjusted; `fatal` means the result must
// not be trusted (e.g. a structural problem in the input).
enum class Status { ok, repaired, fatal };

} // namespace eeos
