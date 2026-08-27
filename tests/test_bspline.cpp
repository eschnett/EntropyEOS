// tests/test_bspline.cpp — unit tests for the M2a B-spline layer:
// entropy_eos/host/bspline_fit.{hpp,cpp} (BandedLU, fit_bspline_1d,
// Bspline3/fit_bspline_3d) and entropy_eos/core/bspline_eval.hpp
// (bspline_eval1, bspline_eval3).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "entropy_eos/core/bspline_eval.hpp"
#include "entropy_eos/host/bspline_fit.hpp"
#include "test_scale.hpp"

using eeos::BandedLU;
using eeos::Bspline3;
using eeos::BsplineEval1;
using eeos::BsplineEval3;
using eeos::BsplineView1;
using eeos::BsplineView3;

namespace {

// 2*pi, used by the sin(x) convergence test; a local constant rather than
// <cmath>'s non-standard M_PI (see synthetic.cpp's kTwoPi for the same
// reasoning).
constexpr double kPi = 3.14159265358979323846264338327950288;

// Relative error, guarding against a reference value near zero (per-test
// tolerances below all follow this "max(|ref|, floor)" convention).
double rel_err(double got, double ref, double floor = 1e-12) {
  return std::fabs(got - ref) / std::max(std::fabs(ref), floor);
}

// --- dense Gaussian-elimination reference solver, for test 1 ------------

// Solves A x = b by dense Gaussian elimination with partial pivoting. `A`
// is modified in place; independent of BandedLU by construction, so it is
// a genuine cross-check.
std::vector<double> dense_gauss_solve(std::vector<std::vector<double>> A, std::vector<double> b) {
  const int n = static_cast<int>(b.size());
  for (int col = 0; col < n; ++col) {
    int piv = col;
    double best = std::fabs(A[static_cast<size_t>(col)][static_cast<size_t>(col)]);
    for (int row = col + 1; row < n; ++row) {
      const double v = std::fabs(A[static_cast<size_t>(row)][static_cast<size_t>(col)]);
      if (v > best) {
        best = v;
        piv = row;
      }
    }
    std::swap(A[static_cast<size_t>(col)], A[static_cast<size_t>(piv)]);
    std::swap(b[static_cast<size_t>(col)], b[static_cast<size_t>(piv)]);

    const double diag = A[static_cast<size_t>(col)][static_cast<size_t>(col)];
    for (int row = col + 1; row < n; ++row) {
      const double factor = A[static_cast<size_t>(row)][static_cast<size_t>(col)] / diag;
      if (factor == 0.0) continue;
      for (int c = col; c < n; ++c) {
        A[static_cast<size_t>(row)][static_cast<size_t>(c)] -=
            factor * A[static_cast<size_t>(col)][static_cast<size_t>(c)];
      }
      b[static_cast<size_t>(row)] -= factor * b[static_cast<size_t>(col)];
    }
  }

  std::vector<double> x(static_cast<size_t>(n));
  for (int row = n - 1; row >= 0; --row) {
    double sum = b[static_cast<size_t>(row)];
    for (int c = row + 1; c < n; ++c) {
      sum -= A[static_cast<size_t>(row)][static_cast<size_t>(c)] * x[static_cast<size_t>(c)];
    }
    x[static_cast<size_t>(row)] = sum / A[static_cast<size_t>(row)][static_cast<size_t>(row)];
  }
  return x;
}

// --- test-local reverse-order tensor fit, for test 7 ---------------------

// Fits fit_bspline_1d() along one axis of a 3D array (dims[0] fastest,
// dims[1] next, dims[2] slowest -- the same convention as
// Bspline3/RawTable, generalized to 3 arbitrary dims rather than
// literally nx/nu/ny), replacing that axis's extent with n+2. This is an
// independent, test-local re-implementation of what
// host/bspline_fit.cpp's private gather_line/scatter_line do internally,
// used only to apply the three axis passes in a different order than
// fit_bspline_3d() does (see test 7) -- it does not reuse any of
// fit_bspline_3d()'s internals, so an order-independence agreement is a
// real cross-check, not a tautology.
std::vector<double> fit_along_axis(const std::vector<double> &data, int d0, int d1, int d2, int axis) {
  const int dims[3] = {d0, d1, d2};
  const int n = dims[axis];
  const int strides[3] = {1, d0, d0 * d1};

  int out_dims[3] = {d0, d1, d2};
  out_dims[axis] = n + 2;
  const int out_strides[3] = {1, out_dims[0], out_dims[0] * out_dims[1]};

  int other[2];
  int oi = 0;
  for (int a = 0; a < 3; ++a) {
    if (a != axis) other[oi++] = a;
  }

  std::vector<double> result(static_cast<size_t>(out_dims[0]) * static_cast<size_t>(out_dims[1]) *
                              static_cast<size_t>(out_dims[2]));

  for (int i1 = 0; i1 < dims[other[1]]; ++i1) {
    for (int i0 = 0; i0 < dims[other[0]]; ++i0) {
      const size_t base_in = static_cast<size_t>(i0) * static_cast<size_t>(strides[other[0]]) +
                              static_cast<size_t>(i1) * static_cast<size_t>(strides[other[1]]);
      std::vector<double> line(static_cast<size_t>(n));
      for (int k = 0; k < n; ++k) {
        line[static_cast<size_t>(k)] = data[base_in + static_cast<size_t>(k) * static_cast<size_t>(strides[axis])];
      }
      const std::vector<double> c = eeos::fit_bspline_1d(line);

      const size_t base_out = static_cast<size_t>(i0) * static_cast<size_t>(out_strides[other[0]]) +
                               static_cast<size_t>(i1) * static_cast<size_t>(out_strides[other[1]]);
      for (int k = 0; k < n + 2; ++k) {
        result[base_out + static_cast<size_t>(k) * static_cast<size_t>(out_strides[axis])] = c[static_cast<size_t>(k)];
      }
    }
  }
  return result;
}

} // namespace

// ==========================================================================
// 1. BandedLU vs. dense Gaussian elimination
// ==========================================================================

TEST_CASE("BandedLU: matches dense Gaussian elimination on random banded systems "
          "(50 trials, 10 under sanitizers)") {
  std::mt19937 rng(20260825u);
  std::uniform_int_distribution<int> n_dist(6, 40);
  std::uniform_real_distribution<double> val_dist(-1.0, 1.0);

  const int n_trials = static_cast<int>(eeos_n(50, 10));
  for (int trial = 0; trial < n_trials; ++trial) {
    const int n = n_dist(rng);
    const int kl = BandedLU::kl;

    std::vector<std::vector<double>> dense(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(n), 0.0));
    BandedLU lu(n);
    for (int row = 0; row < n; ++row) {
      const int lo = std::max(0, row - kl);
      const int hi = std::min(n - 1, row + kl);
      for (int col = lo; col <= hi; ++col) {
        double v = val_dist(rng);
        if (row == col) v += static_cast<double>(n); // diagonal dominance -> well conditioned
        dense[static_cast<size_t>(row)][static_cast<size_t>(col)] = v;
        lu.at(row, col) = v;
      }
    }
    lu.factor();

    std::vector<double> b(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) b[static_cast<size_t>(i)] = val_dist(rng);

    std::vector<double> x_banded = b;
    lu.solve(x_banded);

    const std::vector<double> x_dense = dense_gauss_solve(dense, b);

    REQUIRE(x_banded.size() == x_dense.size());
    for (size_t i = 0; i < x_dense.size(); ++i) {
      CHECK(rel_err(x_banded[i], x_dense[i]) <= 1e-11);
    }
  }
}

// ==========================================================================
// 2. 1D exactness: a global cubic is reproduced exactly (not-a-knot)
// ==========================================================================

TEST_CASE("fit_bspline_1d: not-a-knot reproduces a global cubic exactly") {
  auto f = [](double x) { return 3.0 * x * x * x - 2.0 * x * x + x - 5.0; };
  auto fp = [](double x) { return 9.0 * x * x - 4.0 * x + 1.0; };
  auto fpp = [](double x) { return 18.0 * x - 4.0; };

  std::mt19937 rng(1001u);
  std::uniform_real_distribution<double> x0_dist(-5.0, 5.0);
  std::uniform_real_distribution<double> h_dist(0.05, 2.0);

  for (const int n : {4, 7, 20}) {
    const double x0 = x0_dist(rng);
    const double h = h_dist(rng);

    std::vector<double> data(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) data[static_cast<size_t>(i)] = f(x0 + i * h);

    const std::vector<double> c = eeos::fit_bspline_1d(data);
    const BsplineView1 view{c.data(), n, x0, h};

    const double xend = x0 + (n - 1) * h;
    std::uniform_real_distribution<double> xq(x0, xend);
    const int n_samp = static_cast<int>(eeos_n(100, 20));
    for (int k = 0; k < n_samp; ++k) {
      const double x = xq(rng);
      const BsplineEval1 e = eeos::bspline_eval1(view, x);
      CHECK(rel_err(e.f, f(x)) <= 1e-10);
      CHECK(rel_err(e.fx, fp(x)) <= 1e-10);
      CHECK(rel_err(e.fxx, fpp(x)) <= 1e-10);
    }
  }
}

// ==========================================================================
// 3. 1D node interpolation
// ==========================================================================

TEST_CASE("fit_bspline_1d: S(x_i) == f_i at every node") {
  std::mt19937 rng(2002u);
  std::uniform_real_distribution<double> val_dist(-100.0, 100.0);

  for (const int n : {4, 5, 10, 30}) {
    std::vector<double> data(static_cast<size_t>(n));
    for (double &v : data) v = val_dist(rng);

    const std::vector<double> c = eeos::fit_bspline_1d(data);
    const double x0 = 0.0, h = 1.0;
    const BsplineView1 view{c.data(), n, x0, h};

    for (int i = 0; i < n; ++i) {
      const BsplineEval1 e = eeos::bspline_eval1(view, x0 + i * h);
      CHECK(rel_err(e.f, data[static_cast<size_t>(i)]) <= 1e-13);
    }
  }
}

// ==========================================================================
// 4. 1D convergence on sin(x)
// ==========================================================================

TEST_CASE("fit_bspline_1d: grid convergence on sin(x) over [0,3] "
          "(n = 20,40,80,160; only 20,40 -- first order pair -- under sanitizers)") {
  const double x0 = 0.0, xend = 3.0;
  // Under sanitizers only the first (cheapest) resolution pair runs, so only
  // the first observed convergence order gets asserted below -- at the same
  // thresholds as a full run, since the order between any adjacent pair is
  // independently meaningful (see test_scale.hpp: shrink the grid, not the
  // tolerance).
  const std::vector<int> ns = eeos_sanitized ? std::vector<int>{20, 40} : std::vector<int>{20, 40, 80, 160};
  std::vector<double> err_f, err_fp, err_fpp;

  for (const int n : ns) {
    const double h = (xend - x0) / (n - 1);
    std::vector<double> data(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) data[static_cast<size_t>(i)] = std::sin(x0 + i * h);

    const std::vector<double> c = eeos::fit_bspline_1d(data);
    const BsplineView1 view{c.data(), n, x0, h};

    double max_f = 0.0, max_fp = 0.0, max_fpp = 0.0;
    const int nsamp = 2000;
    for (int k = 0; k < nsamp; ++k) {
      const double x = x0 + (xend - x0) * (k + 0.5) / nsamp;
      const BsplineEval1 e = eeos::bspline_eval1(view, x);
      max_f = std::max(max_f, std::fabs(e.f - std::sin(x)));
      max_fp = std::max(max_fp, std::fabs(e.fx - std::cos(x)));
      max_fpp = std::max(max_fpp, std::fabs(e.fxx + std::sin(x)));
    }
    err_f.push_back(max_f);
    err_fp.push_back(max_fp);
    err_fpp.push_back(max_fpp);
  }

  for (size_t i = 1; i < ns.size(); ++i) {
    const double order_f = std::log2(err_f[i - 1] / err_f[i]);
    const double order_fp = std::log2(err_fp[i - 1] / err_fp[i]);
    const double order_fpp = std::log2(err_fpp[i - 1] / err_fpp[i]);
    std::cout << "  bspline sin(x) convergence " << ns[i - 1] << "->" << ns[i] << ": order f="
              << order_f << " f'=" << order_fp << " f''=" << order_fpp << "\n";
    CHECK(order_f >= 3.7);
    CHECK(order_fp >= 2.7);
    CHECK(order_fpp >= 1.7);
  }
}

// ==========================================================================
// 5. C2 continuity at interior knots
// ==========================================================================

TEST_CASE("fit_bspline_1d: f'' is continuous across interior knots to roundoff") {
  std::mt19937 rng(3003u);
  std::uniform_real_distribution<double> val_dist(-50.0, 50.0);

  const int n = 30;
  std::vector<double> data(static_cast<size_t>(n));
  for (double &v : data) v = val_dist(rng);

  const std::vector<double> c = eeos::fit_bspline_1d(data);
  const double x0 = 0.0, h = 0.37;
  const BsplineView1 view{c.data(), n, x0, h};

  std::vector<double> left_fxx, right_fxx;
  double max_abs_fpp = 0.0;
  for (int i = 1; i <= n - 2; ++i) {
    const double xk = x0 + i * h;
    const BsplineEval1 el = eeos::bspline_eval1(view, xk - 1e-9);
    const BsplineEval1 er = eeos::bspline_eval1(view, xk + 1e-9);
    left_fxx.push_back(el.fxx);
    right_fxx.push_back(er.fxx);
    max_abs_fpp = std::max({max_abs_fpp, std::fabs(el.fxx), std::fabs(er.fxx)});
  }

  const double tol = 1e-6 * std::max(max_abs_fpp, 1e-12);
  for (size_t k = 0; k < left_fxx.size(); ++k) {
    CHECK(std::fabs(left_fxx[k] - right_fxx[k]) <= tol);
  }
}

// ==========================================================================
// 6. 3D separable exactness
// ==========================================================================

TEST_CASE("fit_bspline_3d/bspline_eval3: separable product of low-degree factors is exact") {
  auto A = [](double x) { return x * x * x - x; };
  auto Ap = [](double x) { return 3.0 * x * x - 1.0; };
  auto App = [](double x) { return 6.0 * x; };
  auto B = [](double u) { return 2.0 * u * u + u; };
  auto Bp = [](double u) { return 4.0 * u + 1.0; };
  auto Bpp = [](double /*u*/) { return 4.0; };
  auto C = [](double y) { return y * y * y + 1.0; };
  auto Cp = [](double y) { return 3.0 * y * y; };

  const int nx = 6, nu = 5, ny = 4;
  const double x0 = -1.0, hx = 0.4;
  const double u0 = 0.0, hu = 0.3;
  const double y0 = -0.5, hy = 0.3;
  const double xend = x0 + (nx - 1) * hx;
  const double uend = u0 + (nu - 1) * hu;
  const double yend = y0 + (ny - 1) * hy;

  std::vector<double> data(static_cast<size_t>(nx) * static_cast<size_t>(nu) * static_cast<size_t>(ny));
  for (int ky = 0; ky < ny; ++ky) {
    const double y = y0 + ky * hy;
    for (int ju = 0; ju < nu; ++ju) {
      const double u = u0 + ju * hu;
      for (int ix = 0; ix < nx; ++ix) {
        const double x = x0 + ix * hx;
        data[static_cast<size_t>(ix) + static_cast<size_t>(nx) *
                                            (static_cast<size_t>(ju) + static_cast<size_t>(nu) * static_cast<size_t>(ky))] =
            A(x) * B(u) * C(y);
      }
    }
  }

  const Bspline3 sp = eeos::fit_bspline_3d(nx, nu, ny, x0, hx, u0, hu, y0, hy, data);
  const BsplineView3 view = sp.view();

  std::mt19937 rng(4004u);
  std::uniform_real_distribution<double> xq(x0, xend), uq(u0, uend), yq(y0, yend);
  const int n_samp = static_cast<int>(eeos_n(200, 40));
  for (int k = 0; k < n_samp; ++k) {
    const double x = xq(rng), u = uq(rng), y = yq(rng);
    const BsplineEval3 e = eeos::bspline_eval3(view, x, u, y);

    // Mixed abs/rel bound (rel_err's `floor`), not a pure relative one: every
    // reference here is a PRODUCT with a factor that vanishes inside the
    // sampled box -- A(x)=x(x-1)(x+1) at x=-1,0,1; Ap(x)=3x^2-1 at
    // x=+-0.577; App(x)=6x at x=0; Cp(y)=3y^2 at y=0. Near such a point the
    // absolute error stays at roundoff (~1e-16) while the relative error
    // grows without bound as 1/|ref|, so a pure relative bound tests only
    // how close a random sample happened to land to a root. Measured on fy
    // (three vanishing factors, the worst case): ref=2.1e-4 -> 2.3e-12,
    // ref=8.0e-6 -> 3.5e-11, ref=1.8e-7 -> 1.2e-10 (arm64/clang), and
    // 1.08e-8 for that last sample on x86-64/g++ -- roundoff amplified, not
    // a fit error. The 1e-2 floor caps the bound at 1e-11 absolute in the
    // vanishing region, still ~4 orders above the roundoff floor and so
    // still a real exactness check, while away from the roots the original
    // 1e-9 relative bound applies unchanged.
    const double fl = 1e-2;
    CHECK(rel_err(e.f, A(x) * B(u) * C(y), fl) <= 1e-9);
    CHECK(rel_err(e.fx, Ap(x) * B(u) * C(y), fl) <= 1e-9);
    CHECK(rel_err(e.fu, A(x) * Bp(u) * C(y), fl) <= 1e-9);
    CHECK(rel_err(e.fy, A(x) * B(u) * Cp(y), fl) <= 1e-9);
    CHECK(rel_err(e.fxx, App(x) * B(u) * C(y), fl) <= 1e-9);
    CHECK(rel_err(e.fxu, Ap(x) * Bp(u) * C(y), fl) <= 1e-9);
    CHECK(rel_err(e.fuu, A(x) * Bpp(u) * C(y), fl) <= 1e-9);
  }
}

// ==========================================================================
// 7. 3D fit-order independence
// ==========================================================================

TEST_CASE("fit_bspline_3d: axis-pass order does not affect the fitted coefficients") {
  std::mt19937 rng(5005u);
  std::uniform_real_distribution<double> val_dist(-10.0, 10.0);

  const int nx = 8, nu = 7, ny = 6;
  std::vector<double> data(static_cast<size_t>(nx) * static_cast<size_t>(nu) * static_cast<size_t>(ny));
  for (double &v : data) v = val_dist(rng);

  const Bspline3 sp_xuy = eeos::fit_bspline_3d(nx, nu, ny, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, data);

  // Test-local reverse order: y -> u -> x (fit_bspline_3d itself does
  // x -> u -> y; see fit_along_axis()'s doc comment).
  const std::vector<double> stage_y = fit_along_axis(data, nx, nu, ny, /*axis=*/2);
  const std::vector<double> stage_yu = fit_along_axis(stage_y, nx, nu, ny + 2, /*axis=*/1);
  const std::vector<double> stage_yux = fit_along_axis(stage_yu, nx, nu + 2, ny + 2, /*axis=*/0);

  REQUIRE(stage_yux.size() == sp_xuy.coeffs().size());
  for (size_t i = 0; i < stage_yux.size(); ++i) {
    CHECK(rel_err(stage_yux[i], sp_xuy.coeffs()[i]) <= 1e-12);
  }
}

// ==========================================================================
// 8. 3D derivative FD cross-check
// ==========================================================================

TEST_CASE("bspline_eval3: analytic derivatives match central FD of the spline's own value") {
  const int nx = 30, nu = 30, ny = 10;
  const double x0 = 0.0, hx = 2.0 * kPi / (nx - 1);
  const double u0 = -1.0, hu = 2.0 / (nu - 1);
  const double y0 = 0.0, hy = kPi / (ny - 1);
  const double xend = x0 + (nx - 1) * hx;
  const double uend = u0 + (nu - 1) * hu;
  const double yend = y0 + (ny - 1) * hy;

  std::vector<double> data(static_cast<size_t>(nx) * static_cast<size_t>(nu) * static_cast<size_t>(ny));
  for (int ky = 0; ky < ny; ++ky) {
    const double y = y0 + ky * hy;
    for (int ju = 0; ju < nu; ++ju) {
      const double u = u0 + ju * hu;
      for (int ix = 0; ix < nx; ++ix) {
        const double x = x0 + ix * hx;
        data[static_cast<size_t>(ix) + static_cast<size_t>(nx) *
                                            (static_cast<size_t>(ju) + static_cast<size_t>(nu) * static_cast<size_t>(ky))] =
            std::sin(x) * std::exp(u / 3.0) * std::cos(y);
      }
    }
  }

  const Bspline3 sp = eeos::fit_bspline_3d(nx, nu, ny, x0, hx, u0, hu, y0, hy, data);
  const BsplineView3 view = sp.view();
  auto Sval = [&](double x, double u, double y) { return eeos::bspline_eval3(view, x, u, y).f; };

  // Plain 2nd-order-accurate central differences at step `hh`, used below
  // at two step sizes and Richardson-extrapolated (see the comment on `h`).
  auto d1 = [&](double x, double u, double y, int axis, double hh) {
    if (axis == 0) return (Sval(x + hh, u, y) - Sval(x - hh, u, y)) / (2.0 * hh);
    if (axis == 1) return (Sval(x, u + hh, y) - Sval(x, u - hh, y)) / (2.0 * hh);
    return (Sval(x, u, y + hh) - Sval(x, u, y - hh)) / (2.0 * hh);
  };
  auto d2 = [&](double x, double u, double y, int axis, double hh) {
    const double sc = Sval(x, u, y);
    if (axis == 0) return (Sval(x + hh, u, y) - 2.0 * sc + Sval(x - hh, u, y)) / (hh * hh);
    return (Sval(x, u + hh, y) - 2.0 * sc + Sval(x, u - hh, y)) / (hh * hh);
  };
  auto dxu = [&](double x, double u, double y, double hh) {
    return (Sval(x + hh, u + hh, y) - Sval(x + hh, u - hh, y) - Sval(x - hh, u + hh, y) +
            Sval(x - hh, u - hh, y)) /
           (4.0 * hh * hh);
  };
  // Richardson-extrapolates a pair of 2nd-order-accurate estimates (step
  // h and 2h) to 4th order by cancelling their leading O(h^2) truncation
  // term.
  auto richardson = [](double d_h, double d_2h) { return (4.0 * d_h - d_2h) / 3.0; };

  // A plain central difference for a 2nd derivative has two competing
  // error sources: truncation ~h^2 and roundoff ~eps*|S|/h^2 (the
  // numerator is an O(h^2)-sized combination of O(1)-sized double values,
  // so it loses precision as h shrinks). Their sum bottoms out around
  // h~1e-4, and even there the floor is a few times 1e-6 absolute -- too
  // coarse for "fuu", whose true magnitude here is only ~f/9 (the
  // exp(u/3) chain-rule factor), to reliably clear a tolerance *relative*
  // to that smaller scale. Richardson-extrapolating two such estimates
  // (step h and 2h, via `richardson` above) cancels the leading
  // truncation term, leaving ~h^4 truncation + eps/h^2 roundoff -- a much
  // more forgiving tradeoff. h=3e-4 sits well inside that basin (checked
  // empirically: absolute FD error stays at or below ~1e-7 there, two
  // orders of magnitude under the tolerance applied below), while
  // reamining a "central FD of the spline's own f" per the design.
  const double h = 3e-4;

  std::mt19937 rng(6006u);
  // Keep points comfortably inside the grid (well away from any boundary
  // cell) so the FD stencil never straddles an extrapolated region.
  std::uniform_real_distribution<double> xq(x0 + 0.05 * (xend - x0), xend - 0.05 * (xend - x0));
  std::uniform_real_distribution<double> uq(u0 + 0.05 * (uend - u0), uend - 0.05 * (uend - u0));
  std::uniform_real_distribution<double> yq(y0 + 0.05 * (yend - y0), yend - 0.05 * (yend - y0));

  // Each derivative's agreement is checked as an absolute difference
  // bounded by 1e-6 times that derivative's own max magnitude over the
  // sampled points -- exactly as test 5 above scales its C2-continuity
  // tolerance by max|f''| rather than a pointwise value, this is what
  // "1e-6 relative" has to mean for a quantity (like fxx, driven by
  // sin(x)*cos(y)) that legitimately passes through zero: a strictly
  // pointwise relative check would spuriously fail near such a zero
  // crossing even for a perfectly correct analytic derivative, since it
  // is the FD reference that loses precision there, not the code under
  // test.
  const int npts = 30;
  std::vector<double> ana_fx(npts), ana_fu(npts), ana_fy(npts), ana_fxx(npts), ana_fxu(npts), ana_fuu(npts);
  std::vector<double> fd_fx(npts), fd_fu(npts), fd_fy(npts), fd_fxx(npts), fd_fxu(npts), fd_fuu(npts);

  for (int k = 0; k < npts; ++k) {
    const double x = xq(rng), u = uq(rng), y = yq(rng);
    const BsplineEval3 e = eeos::bspline_eval3(view, x, u, y);
    const size_t i = static_cast<size_t>(k);

    ana_fx[i] = e.fx;
    ana_fu[i] = e.fu;
    ana_fy[i] = e.fy;
    ana_fxx[i] = e.fxx;
    ana_fxu[i] = e.fxu;
    ana_fuu[i] = e.fuu;

    fd_fx[i] = richardson(d1(x, u, y, 0, h), d1(x, u, y, 0, 2.0 * h));
    fd_fu[i] = richardson(d1(x, u, y, 1, h), d1(x, u, y, 1, 2.0 * h));
    fd_fy[i] = richardson(d1(x, u, y, 2, h), d1(x, u, y, 2, 2.0 * h));
    fd_fxx[i] = richardson(d2(x, u, y, 0, h), d2(x, u, y, 0, 2.0 * h));
    fd_fuu[i] = richardson(d2(x, u, y, 1, h), d2(x, u, y, 1, 2.0 * h));
    fd_fxu[i] = richardson(dxu(x, u, y, h), dxu(x, u, y, 2.0 * h));
  }

  auto max_abs = [](const std::vector<double> &v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::fabs(x));
    return std::max(m, 1e-12);
  };
  const double scale_x = max_abs(ana_fx), scale_u = max_abs(ana_fu), scale_y = max_abs(ana_fy);
  const double scale_xx = max_abs(ana_fxx), scale_xu = max_abs(ana_fxu), scale_uu = max_abs(ana_fuu);

  for (int k = 0; k < npts; ++k) {
    const size_t i = static_cast<size_t>(k);
    CHECK(std::fabs(ana_fx[i] - fd_fx[i]) <= 1e-6 * scale_x);
    CHECK(std::fabs(ana_fu[i] - fd_fu[i]) <= 1e-6 * scale_u);
    CHECK(std::fabs(ana_fy[i] - fd_fy[i]) <= 1e-6 * scale_y);
    CHECK(std::fabs(ana_fxx[i] - fd_fxx[i]) <= 1e-6 * scale_xx);
    CHECK(std::fabs(ana_fuu[i] - fd_fuu[i]) <= 1e-6 * scale_uu);
    CHECK(std::fabs(ana_fxu[i] - fd_fxu[i]) <= 1e-6 * scale_xu);
  }
}

// ==========================================================================
// 9. Extrapolation sanity
// ==========================================================================

TEST_CASE("bspline_eval1: smooth polynomial continuation outside the grid") {
  std::mt19937 rng(7007u);
  std::uniform_real_distribution<double> val_dist(-10.0, 10.0);

  const int n = 10;
  std::vector<double> data(static_cast<size_t>(n));
  for (double &v : data) v = val_dist(rng);

  const double x0 = 0.0, h = 0.5;
  const std::vector<double> c = eeos::fit_bspline_1d(data);
  const BsplineView1 view{c.data(), n, x0, h};
  const double xend = x0 + (n - 1) * h;

  // Slightly outside the low end: finite, and (by construction, since
  // there is no knot between x0-0.3h and x0) the same cubic as the
  // boundary cell.
  const BsplineEval1 e_low = eeos::bspline_eval1(view, x0 - 0.3 * h);
  CHECK(std::isfinite(e_low.f));
  CHECK(std::isfinite(e_low.fx));
  CHECK(std::isfinite(e_low.fxx));

  const double eps = 1e-6;
  const BsplineEval1 e0 = eeos::bspline_eval1(view, x0);
  const BsplineEval1 e_eps = eeos::bspline_eval1(view, x0 - eps);
  CHECK(std::fabs(e_eps.f - (e0.f - eps * e0.fx)) <= 1e-9);

  // Same check at the high end.
  const BsplineEval1 e_high = eeos::bspline_eval1(view, xend + 0.3 * h);
  CHECK(std::isfinite(e_high.f));
  CHECK(std::isfinite(e_high.fx));
  CHECK(std::isfinite(e_high.fxx));

  const BsplineEval1 eN = eeos::bspline_eval1(view, xend);
  const BsplineEval1 eN_eps = eeos::bspline_eval1(view, xend + eps);
  CHECK(std::fabs(eN_eps.f - (eN.f + eps * eN.fx)) <= 1e-9);
}
