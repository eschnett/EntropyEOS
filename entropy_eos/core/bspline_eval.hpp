// entropy_eos/core/bspline_eval.hpp
//
// Tensor-product uniform cubic B-spline evaluation: value and derivatives up
// to 2nd order, in 1D and in the axis-separable 3D case the M2 adapter
// needs. Header-only, device-ready (see CODE.md "Layout"): no STL
// containers, no exceptions, no allocation, every function
// `EEOS_HOST_DEVICE`, operating on POD *view* structs (raw pointer +
// extents) into a coefficient array owned by entropy_eos/host/bspline_fit.
//
// --- Uniform cubic B-spline on n data points --------------------------
//
// Data points sit at x_i = x0 + i*h, i = 0..n-1 (n >= 4). The spline has
// n+2 control coefficients c_0..c_{n+1} -- two more than data points,
// fixed at fit time by not-a-knot boundary conditions (see
// host/bspline_fit.hpp for the linear system that determines them). On
// cell i (x in [x_i, x_{i+1}], i = 0..n-2), with t = (x-x0)/h - i in
// [0,1]:
//
//   S(x) = c_i*b0(t) + c_{i+1}*b1(t) + c_{i+2}*b2(t) + c_{i+3}*b3(t)
//
//   b0(t) = (1-t)^3 / 6
//   b1(t) = (3t^3 - 6t^2 + 4) / 6
//   b2(t) = (-3t^3 + 3t^2 + 3t + 1) / 6
//   b3(t) = t^3 / 6
//
// and, by the chain rule (dt/dx = 1/h):
//
//   S'(x)  = (1/h)   * sum_k c_{i+k} * b_k'(t)
//   S''(x) = (1/h^2) * sum_k c_{i+k} * b_k''(t)
//
//   b0'(t)  = -(1-t)^2 / 2      b0''(t) = 1 - t
//   b1'(t)  = (3t^2 - 4t) / 2   b1''(t) = 3t - 2
//   b2'(t)  = (-3t^2 + 2t + 1) / 2   b2''(t) = 1 - 3t
//   b3'(t)  = t^2 / 2           b3''(t) = t
//
// (each basis quadruple and its two derivative quadruples sum to 1, 0, 0
// respectively for every t, as they must since a B-spline reproduces
// constants.)
//
// Not-a-knot fitting makes S reproduce any global cubic exactly (see
// bspline_fit.hpp and tests/test_bspline.cpp).
//
// Cell index for an arbitrary x: i = clamp((int)floor((x-x0)/h), 0, n-2),
// t = (x-x0)/h - i. For x outside [x0, x0+(n-1)*h] this evaluates the
// boundary cell's cubic -- smooth polynomial extrapolation, *not* a
// designed extension. Callers are responsible for clamping/flagging
// out-of-range arguments before calling (the M2 adapter clamps as a
// stopgap; the M2d domain-extension work of eos-adapter-F-to-U.md §7
// replaces that with designed extensions re-using this same evaluator).

#pragma once

#include <cmath>

#include "entropy_eos/core/defs.hpp"

namespace eeos {

// 1D view: n+2 coefficients for n data points at x0 + i*h, i = 0..n-1.
struct BsplineView1 {
  const real *c; // coefficient array, n+2 entries
  int n;         // data point count (coefficient array has n+2 entries)
  real x0, h;
};

struct BsplineEval1 {
  real f, fx, fxx;
};

// 3D tensor-product view: coefficient array dims (nx+2, nu+2, ny+2), ix
// fastest (same "fastest axis first" convention as RawTable's storage
// order, renamed rho/T/Ye -> x/u/y).
struct BsplineView3 {
  const real *c;            // coefficient array, dims (nx+2, nu+2, ny+2), ix fastest
  int nx, nu, ny;            // DATA point counts (coefficient dims are +2)
  real x0, hx, u0, hu, y0, hy;
};

// Value and the derivative set the M2 adapter's chain rule needs: no fyy,
// fxy, or fuy (not needed downstream; skipping them halves the per-axis
// basis work for y, see bspline_eval3()).
struct BsplineEval3 {
  real f, fx, fu, fy, fxx, fxu, fuu;
};

namespace detail {

EEOS_HOST_DEVICE inline int bspline_clamp_int(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// One axis's local cell index and local coordinate for a query point.
struct BsplineCell {
  int i;  // cell index, clamped to [0, n-2]
  real t; // local coordinate, in [0,1] inside the grid, extrapolated outside
};

EEOS_HOST_DEVICE inline BsplineCell bspline_cell(real x, real x0, real h, int n) {
  const real xi = (x - x0) / h;
  const int i = bspline_clamp_int(static_cast<int>(std::floor(xi)), 0, n - 2);
  return BsplineCell{i, xi - static_cast<real>(i)};
}

// The four basis functions (or a derivative of them) evaluated at one t,
// b0..b3 in order (see the module header comment for the formulas).
struct Basis4 {
  real b0, b1, b2, b3;
};

EEOS_HOST_DEVICE inline Basis4 bspline_basis(real t) {
  const real t2 = t * t;
  const real t3 = t2 * t;
  const real omt = real(1) - t;
  return Basis4{omt * omt * omt / real(6), (real(3) * t3 - real(6) * t2 + real(4)) / real(6),
                (real(-3) * t3 + real(3) * t2 + real(3) * t + real(1)) / real(6), t3 / real(6)};
}

EEOS_HOST_DEVICE inline Basis4 bspline_dbasis(real t) {
  const real t2 = t * t;
  const real omt = real(1) - t;
  return Basis4{-omt * omt / real(2), (real(3) * t2 - real(4) * t) / real(2),
                (real(-3) * t2 + real(2) * t + real(1)) / real(2), t2 / real(2)};
}

EEOS_HOST_DEVICE inline Basis4 bspline_d2basis(real t) {
  return Basis4{real(1) - t, real(3) * t - real(2), real(1) - real(3) * t, t};
}

} // namespace detail

EEOS_HOST_DEVICE inline BsplineEval1 bspline_eval1(const BsplineView1 &v, real x) {
  const detail::BsplineCell cell = detail::bspline_cell(x, v.x0, v.h, v.n);
  const detail::Basis4 b = detail::bspline_basis(cell.t);
  const detail::Basis4 d1 = detail::bspline_dbasis(cell.t);
  const detail::Basis4 d2 = detail::bspline_d2basis(cell.t);

  const real c0 = v.c[cell.i + 0];
  const real c1 = v.c[cell.i + 1];
  const real c2 = v.c[cell.i + 2];
  const real c3 = v.c[cell.i + 3];

  const real inv_h = real(1) / v.h;

  BsplineEval1 e;
  e.f = c0 * b.b0 + c1 * b.b1 + c2 * b.b2 + c3 * b.b3;
  e.fx = (c0 * d1.b0 + c1 * d1.b1 + c2 * d1.b2 + c3 * d1.b3) * inv_h;
  e.fxx = (c0 * d2.b0 + c1 * d2.b1 + c2 * d2.b2 + c3 * d2.b3) * (inv_h * inv_h);
  return e;
}

// Evaluates the tensor-product spline at (x,u,y) by contracting the 4x4x4
// block of coefficients around the cell (cx,cu,cy) with the per-axis basis
// vectors. x and u need value + 1st + 2nd derivative bases (fxx, fxu, fuu
// are wanted); y only needs value + 1st derivative (fy is wanted, fyy/fxy/
// fuy are not) -- computing that one extra vector costs nothing so it is
// computed unconditionally rather than gated. Fixed cost (64 coefficient
// reads, no data-dependent branches beyond the three per-axis index
// clamps in bspline_cell()).
EEOS_HOST_DEVICE inline BsplineEval3 bspline_eval3(const BsplineView3 &v, real x, real u, real y) {
  const detail::BsplineCell cx = detail::bspline_cell(x, v.x0, v.hx, v.nx);
  const detail::BsplineCell cu = detail::bspline_cell(u, v.u0, v.hu, v.nu);
  const detail::BsplineCell cy = detail::bspline_cell(y, v.y0, v.hy, v.ny);

  const detail::Basis4 Bx = detail::bspline_basis(cx.t);
  const detail::Basis4 Dx = detail::bspline_dbasis(cx.t);
  const detail::Basis4 Hx = detail::bspline_d2basis(cx.t);

  const detail::Basis4 Bu = detail::bspline_basis(cu.t);
  const detail::Basis4 Du = detail::bspline_dbasis(cu.t);
  const detail::Basis4 Hu = detail::bspline_d2basis(cu.t);

  const detail::Basis4 By = detail::bspline_basis(cy.t);
  const detail::Basis4 Dy = detail::bspline_dbasis(cy.t);

  const real bx[4] = {Bx.b0, Bx.b1, Bx.b2, Bx.b3};
  const real dx[4] = {Dx.b0, Dx.b1, Dx.b2, Dx.b3};
  const real hx4[4] = {Hx.b0, Hx.b1, Hx.b2, Hx.b3};
  const real bu[4] = {Bu.b0, Bu.b1, Bu.b2, Bu.b3};
  const real du[4] = {Du.b0, Du.b1, Du.b2, Du.b3};
  const real hu4[4] = {Hu.b0, Hu.b1, Hu.b2, Hu.b3};
  const real by[4] = {By.b0, By.b1, By.b2, By.b3};
  const real dy[4] = {Dy.b0, Dy.b1, Dy.b2, Dy.b3};

  const int nxp2 = v.nx + 2;
  const int nup2 = v.nu + 2;

  real f = real(0), fx = real(0), fu = real(0), fy = real(0);
  real fxx = real(0), fxu = real(0), fuu = real(0);

  for (int r = 0; r < 4; ++r) {
    const int iy = cy.i + r;
    for (int q = 0; q < 4; ++q) {
      const int iu = cu.i + q;
      const int base = nxp2 * (iu + nup2 * iy);

      // Contract along x first (the fastest-varying axis) for this (q,r):
      // value, 1st, and 2nd derivative sums.
      real sf = real(0), sdx = real(0), shx = real(0);
      for (int p = 0; p < 4; ++p) {
        const real cc = v.c[base + cx.i + p];
        sf += bx[p] * cc;
        sdx += dx[p] * cc;
        shx += hx4[p] * cc;
      }

      const real wuy = bu[q] * by[r];
      f += wuy * sf;
      fx += wuy * sdx;
      fxx += wuy * shx;
      fu += du[q] * by[r] * sf;
      fy += bu[q] * dy[r] * sf;
      fxu += du[q] * by[r] * sdx;
      fuu += hu4[q] * by[r] * sf;
    }
  }

  BsplineEval3 e;
  e.f = f;
  e.fx = fx / v.hx;
  e.fu = fu / v.hu;
  e.fy = fy / v.hy;
  e.fxx = fxx / (v.hx * v.hx);
  e.fxu = fxu / (v.hx * v.hu);
  e.fuu = fuu / (v.hu * v.hu);
  return e;
}

} // namespace eeos
