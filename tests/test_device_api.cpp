// tests/test_device_api.cpp — unit tests for the M4 device-mirroring hooks
// (eos-device-interface.md S4a): EntropyEOS::sigma_coeffs()/L_coeffs()/
// view_with(). Pure host: the contract under test is that a view rebound
// onto CALLER-OWNED COPIES of the two coefficient blobs is indistinguishable
// from view() -- asserted BIT-IDENTICALLY, because on the same host the same
// code reading equal bytes must produce equal bytes. (The GPU side of the
// mirror -- where results are compared against ground truth rather than
// bitwise, since device libm differs by ULPs -- is tests/test_device_cuda.cu,
// which never runs in CI; THIS test is what CI exercises of M4.)

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "entropy_eos/core/adapter_eval.hpp"
#include "entropy_eos/core/con2prim.hpp"
#include "entropy_eos/core/prim2con.hpp"
#include "entropy_eos/host/adapter_build.hpp"
#include "entropy_eos/host/synthetic.hpp"

using eeos::Con2PrimIn;
using eeos::Con2PrimOptions;
using eeos::Con2PrimOut;
using eeos::EntropyEOS;
using eeos::EntropyEOSView;
using eeos::EOSPoint;
using eeos::Prim2ConOut;
using eeos::real;
using eeos::SRange;

namespace {

EntropyEOS build_synthetic() {
  const eeos::RawTable table = eeos::make_synthetic_table(); // clean default 40x30x10
  return eeos::build_entropy_eos(table);
}

std::size_t coeff_count(const eeos::BsplineView3 &b) {
  return static_cast<std::size_t>(b.nx + 2) * (b.nu + 2) * (b.ny + 2);
}

} // namespace

TEST_CASE("sigma_coeffs()/L_coeffs() are the blobs view() points at") {
  const EntropyEOS eos = build_synthetic();
  const EntropyEOSView v = eos.view();

  CHECK(eos.sigma_coeffs().size() == coeff_count(v.sigma));
  CHECK(eos.L_coeffs().size() == coeff_count(v.L));
  CHECK(eos.sigma_coeffs().data() == v.sigma.c);
  CHECK(eos.L_coeffs().data() == v.L.c);

  // view_with over view()'s own pointers IS view(): every field, not just
  // the two it rebinds.
  const EntropyEOSView w = eos.view_with(v.sigma.c, v.L.c);
  CHECK(w.sigma.c == v.sigma.c);
  CHECK(w.L.c == v.L.c);
  CHECK(w.kappa == v.kappa);
  CHECK(w.shift_hat == v.shift_hat);
  CHECK(w.conv_t == v.conv_t);
  CHECK(w.inv_c2 == v.inv_c2);
  CHECK(w.x_lo == v.x_lo);
  CHECK(w.x_hi == v.x_hi);
  CHECK(w.u_lo == v.u_lo);
  CHECK(w.u_hi == v.u_hi);
  CHECK(w.y_lo == v.y_lo);
  CHECK(w.y_hi == v.y_hi);
  CHECK(w.x_ext_lo == v.x_ext_lo);
  CHECK(w.x_ext_hi == v.x_ext_hi);
  CHECK(w.u_ext_lo == v.u_ext_lo);
  CHECK(w.u_ext_hi == v.u_ext_hi);
  CHECK(w.ext_slope_floor_sigma == v.ext_slope_floor_sigma);
  CHECK(w.ext_slope_floor_L == v.ext_slope_floor_L);
  CHECK(w.cs2_ext_cap == v.cs2_ext_cap);
  CHECK(w.max_iter == v.max_iter);
}

TEST_CASE("a view rebound onto copies of the blobs is bit-identical in use") {
  const EntropyEOS eos = build_synthetic();
  const EntropyEOSView v = eos.view();

  // The mirror operation, done on the host: copy the blobs (as a device
  // upload would), rebind. Every downstream difference would be a bug in the
  // rebinding, since the bytes are equal by construction.
  const std::vector<double> sigma_copy = eos.sigma_coeffs();
  const std::vector<double> L_copy = eos.L_coeffs();
  const EntropyEOSView m = eos.view_with(sigma_copy.data(), L_copy.data());

  const Con2PrimOptions opts;
  int evaluated = 0, solved = 0;

  // A coarse interior sweep is enough: identical code on identical bytes.
  const int nx = 9, ns = 7, ny = 5;
  for (int ix = 0; ix < nx; ++ix) {
    const real x = v.x_lo + (v.x_hi - v.x_lo) * (ix + real(0.5)) / nx;
    const real rho = std::pow(real(10), x);
    for (int iy = 0; iy < ny; ++iy) {
      const real ye = v.y_lo + (v.y_hi - v.y_lo) * (iy + real(0.5)) / ny;
      const SRange sr = v.srange(rho, ye);
      for (int is = 0; is < ns; ++is) {
        const real s = sr.s_min + (sr.s_max - sr.s_min) * (is + real(0.5)) / ns;

        const EOSPoint a = v.evaluate(rho, s, ye, eeos::detail::p2c_nan());
        const EOSPoint b = m.evaluate(rho, s, ye, eeos::detail::p2c_nan());
        CHECK(a.U == b.U);
        CHECK(a.U_rho == b.U_rho);
        CHECK(a.U_s == b.U_s);
        CHECK(a.U_rhorho == b.U_rhorho);
        CHECK(a.U_rhos == b.U_rhos);
        CHECK(a.That == b.That);
        CHECK(a.p == b.p);
        CHECK(a.h == b.h);
        CHECK(a.cs2 == b.cs2);
        CHECK(a.T_F_MeV == b.T_F_MeV);
        CHECK(a.mu_tilde == b.mu_tilde);
        CHECK(a.u_solved == b.u_solved);
        CHECK(a.iters == b.iters);
        CHECK(a.flags == b.flags);
        ++evaluated;

        // One magnetized con2prim round trip per point, cold-started on both
        // views so the whole seed/Newton/fallback path runs twice on equal
        // bytes.
        const real w = real(0.4) + real(0.2) * is;
        const Prim2ConOut pc = eeos::prim2con(v, rho, s, ye, w, real(0.3) * a.p, real(0.7));
        const Con2PrimIn cin{pc.D, pc.tau, pc.D_Y, pc.S_par, pc.S_perp, pc.B2};
        const Con2PrimOut ca = eeos::con2prim(v, cin, opts);
        const Con2PrimOut cb = eeos::con2prim(m, cin, opts);
        CHECK(ca.result == cb.result);
        CHECK(ca.rho == cb.rho);
        CHECK(ca.s == cb.s);
        CHECK(ca.ye == cb.ye);
        CHECK(ca.w == cb.w);
        CHECK(ca.iters_newton == cb.iters_newton);
        CHECK(ca.iters_fallback == cb.iters_fallback);
        CHECK(ca.flags == cb.flags);
        ++solved;
      }
    }
  }

  CHECK(evaluated == nx * ns * ny);
  CHECK(solved == nx * ns * ny);
}
