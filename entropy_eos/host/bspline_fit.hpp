// entropy_eos/host/bspline_fit.hpp
//
// Host-only B-spline coefficient fitting: a banded LU solver (LAPACK
// gbtrf/gbtrs-style compact storage, factor once per axis length / solve
// many right-hand sides) plus the not-a-knot uniform cubic B-spline fit
// itself, 1D and tensor-product 3D. STL throughout; may throw (see CODE.md
// "Environment"). The evaluation side
// (entropy_eos/core/bspline_eval.hpp) documents the basis functions this
// module's linear system is built from; see that header for the exact math
// this fit inverts.

#pragma once

#include <cstddef>
#include <vector>

#include "entropy_eos/core/bspline_eval.hpp"

namespace eeos {

// Banded LU factorization with partial pivoting, lower/upper bandwidth 4
// (kl = ku = 4 -- the bandwidth of the not-a-knot cubic B-spline system,
// see bspline_eval.hpp's module comment and fit_bspline_1d() below), using
// LAPACK gbtrf-style compact storage extended with kl extra columns of
// per-row workspace to absorb the fill-in partial pivoting produces.
// Factor once for a given n x n matrix and solve() any number of times
// against different right-hand sides -- exactly the fit_bspline_1d/_3d
// access pattern: the system depends only on the axis length n, so one
// factorization serves every line along that axis (see fit_bspline_3d()).
//
// General-purpose (not spline-specific): exposed as public API mainly so
// it can be unit-tested directly against a dense reference solver
// (tests/test_bspline.cpp), independent of the spline fit built on top of
// it.
class BandedLU {
public:
  static constexpr int kl = 4; // lower bandwidth
  static constexpr int ku = 4; // upper bandwidth

  // An uninitialized (all entries implicitly 0) n x n banded system, n >= 1.
  // Throws std::invalid_argument if n < 1.
  explicit BandedLU(int n);

  int n() const { return n_; }

  // Mutable access to matrix entry (row, col), 0 <= row,col < n. Valid only
  // before factor() is called (afterwards the storage holds the L/U
  // factors, not the original matrix). Throws std::out_of_range if
  // row/col are out of [0,n) or |row-col| > kl (== ku): such an entry is
  // structurally zero and not stored at all, so it cannot be set to
  // anything else.
  double &at(int row, int col);

  // Factors the matrix in place via banded Gaussian elimination with
  // partial pivoting (row search restricted to the band, matching LAPACK
  // dgbtf2's unblocked algorithm). Throws std::runtime_error if the matrix
  // is numerically singular (a zero pivot survives pivoting), and
  // std::logic_error if called more than once.
  void factor();

  // Solves A x = rhs for the already-factored matrix, in place: `rhs`
  // holds b on entry and x on exit. Throws std::invalid_argument if
  // rhs.size() != n(), and std::logic_error if factor() has not been
  // called yet. May be called repeatedly (that is the point).
  void solve(std::vector<double> &rhs) const;

private:
  // Element (row, col) of the extended band storage: entries with
  // |row-col| <= kl+ku (the fill-in workspace beyond the true band) are
  // addressable; used internally by factor()/solve(), unlike the narrower
  // public at().
  double &elem(int row, int col);
  double elem(int row, int col) const;

  int n_;
  int row_stride_;         // 2*kl + ku + 1: entries stored per row
  std::vector<double> ab_; // banded storage, n_ rows x row_stride_ columns
  std::vector<int> pivot_; // pivot_[j] = row swapped into row j during factor()
  bool factored_ = false;
};

// Fits a uniform not-a-knot cubic B-spline to f (n values sampled at
// x0+i*h, i=0..n-1; n implied by f.size(), n >= 4), returning the n+2
// coefficients c_0..c_{n+1} such that S(x0+i*h) = f[i] for every i (see
// bspline_eval.hpp for S(x)). The (n+2)x(n+2) system solved, rows ordered
// [nak_left, interp_0..interp_{n-1}, nak_right] (banded, bandwidth 4):
//
//   interpolation, i = 0..n-1: (c_i + 4*c_{i+1} + c_{i+2}) / 6 = f_i
//   not-a-knot left:  c_0 - 4*c_1 + 6*c_2 - 4*c_3 + c_4 = 0
//   not-a-knot right: c_{n-3} - 4*c_{n-2} + 6*c_{n-1} - 4*c_n + c_{n+1} = 0
//
// (third-derivative continuity at the first/last interior knots -- this is
// what makes the fitted spline reproduce a global cubic exactly, see
// tests/test_bspline.cpp). Throws std::invalid_argument if n < 4.
std::vector<double> fit_bspline_1d(const std::vector<double> &f);

// A B-spline-fitted 3D scalar field: owns the tensor-product coefficient
// array (dims (nx+2, nu+2, ny+2), ix fastest -- same "fastest axis first"
// layout convention as RawTable, axes renamed rho/T/Ye -> x/u/y) plus the
// grid parameters, and hands out a core::BsplineView3 for
// entropy_eos/core/bspline_eval.hpp to evaluate. Produced by
// fit_bspline_3d(); the constructor is public mainly so tests can build a
// comparison instance directly. Throws std::invalid_argument if `coeffs`'s
// size does not match (nx+2)*(nu+2)*(ny+2).
class Bspline3 {
public:
  Bspline3(int nx, int nu, int ny, double x0, double hx, double u0, double hu, double y0, double hy,
           std::vector<double> coeffs);

  BsplineView3 view() const;

  const std::vector<double> &coeffs() const { return coeffs_; }
  int nx() const { return nx_; }
  int nu() const { return nu_; }
  int ny() const { return ny_; }

private:
  int nx_, nu_, ny_;
  double x0_, hx_, u0_, hu_, y0_, hy_;
  std::vector<double> coeffs_;
};

// Tensor-product not-a-knot fit: fits fit_bspline_1d() along x for every
// (u,y) line -> intermediate coefficients (nx+2, nu, ny); then along u for
// every (x,y) line -> (nx+2, nu+2, ny); then along y for every (x,u) line
// -> (nx+2, nu+2, ny+2). Each pass factors its axis's banded matrix once
// (BandedLU's factor-once/solve-many contract) and reuses it for every
// line of that pass. The three 1D fits are independent linear operators on
// separate axes, so the three passes commute -- any application order
// gives the same final coefficients (see tests/test_bspline.cpp).
//
// `data` is indexed [ky][ju][ix] (ix fastest, size nx*nu*ny) -- the same
// convention as RawTable::index(), axes renamed rho/T/Ye -> x/u/y. Grid
// uniformity is not checked here (nor is any relationship between x0/hx
// and an actual RawTable axis): the fit only ever sees x0/h per axis, and
// it is the caller's job (adapter build, M2b) to have validated the axis
// is actually uniform. Throws std::invalid_argument if nx, nu, or ny < 4,
// or if data.size() != nx*nu*ny.
Bspline3 fit_bspline_3d(int nx, int nu, int ny, double x0, double hx, double u0, double hu, double y0,
                        double hy, const std::vector<double> &data);

} // namespace eeos
