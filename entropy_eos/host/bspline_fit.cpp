#include "entropy_eos/host/bspline_fit.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace eeos {

// --- BandedLU ----------------------------------------------------------

BandedLU::BandedLU(int n) : n_(n), row_stride_(2 * kl + ku + 1) {
  if (n < 1) {
    throw std::invalid_argument("BandedLU: n must be >= 1 (got " + std::to_string(n) + ")");
  }
  ab_.assign(static_cast<size_t>(n_) * static_cast<size_t>(row_stride_), 0.0);
}

double &BandedLU::elem(int row, int col) {
  const int d = col - row + kl;
  return ab_[static_cast<size_t>(row) * static_cast<size_t>(row_stride_) + static_cast<size_t>(d)];
}

double BandedLU::elem(int row, int col) const {
  const int d = col - row + kl;
  return ab_[static_cast<size_t>(row) * static_cast<size_t>(row_stride_) + static_cast<size_t>(d)];
}

double &BandedLU::at(int row, int col) {
  if (factored_) {
    throw std::logic_error("BandedLU::at: matrix already factored");
  }
  if (row < 0 || row >= n_ || col < 0 || col >= n_) {
    throw std::out_of_range("BandedLU::at: index (" + std::to_string(row) + ", " + std::to_string(col) +
                             ") out of range for n=" + std::to_string(n_));
  }
  const int diff = row - col;
  if (diff > kl || -diff > ku) {
    throw std::out_of_range("BandedLU::at: entry (" + std::to_string(row) + ", " + std::to_string(col) +
                             ") lies outside the stored band (kl=ku=" + std::to_string(kl) + ")");
  }
  return elem(row, col);
}

// Banded Gaussian elimination with partial pivoting, restricted to the band
// (this is LAPACK dgbtf2's unblocked algorithm, transcribed 0-based): the
// pivot search for column j is limited to rows j..j+kl (the only rows that
// can still hold a nonzero there), and each row interchange/elimination
// step only touches columns up to `ju`, the rightmost column any row
// involved could reach -- `ju` grows monotonically as pivoting shuffles
// rows around, which is exactly the fill-in the extended (kl extra
// columns) storage in `elem()` exists to hold.
void BandedLU::factor() {
  if (factored_) {
    throw std::logic_error("BandedLU::factor: already factored");
  }
  pivot_.assign(static_cast<size_t>(n_), 0);
  int ju = -1; // rightmost column touched by any interchange/update so far

  for (int j = 0; j < n_; ++j) {
    const int km = std::min(kl, n_ - 1 - j); // subdiagonal rows still available: j+1..j+km

    int prow = j;
    double pval = std::fabs(elem(j, j));
    for (int i = 1; i <= km; ++i) {
      const double v = std::fabs(elem(j + i, j));
      if (v > pval) {
        pval = v;
        prow = j + i;
      }
    }
    if (elem(prow, j) == 0.0) {
      throw std::runtime_error("BandedLU::factor: singular matrix (zero pivot at column " +
                                std::to_string(j) + ")");
    }
    pivot_[static_cast<size_t>(j)] = prow;

    const int i0 = prow - j;
    ju = std::max(ju, std::min(j + ku + i0, n_ - 1));

    if (prow != j) {
      for (int c = j; c <= ju; ++c) {
        std::swap(elem(j, c), elem(prow, c));
      }
    }

    if (km > 0) {
      const double diag = elem(j, j);
      for (int i = 1; i <= km; ++i) {
        elem(j + i, j) /= diag;
      }
      for (int c = j + 1; c <= ju; ++c) {
        const double ujc = elem(j, c);
        if (ujc != 0.0) {
          for (int i = 1; i <= km; ++i) {
            elem(j + i, c) -= elem(j + i, j) * ujc;
          }
        }
      }
    }
  }

  factored_ = true;
}

void BandedLU::solve(std::vector<double> &rhs) const {
  if (!factored_) {
    throw std::logic_error("BandedLU::solve: factor() has not been called");
  }
  if (rhs.size() != static_cast<size_t>(n_)) {
    throw std::invalid_argument("BandedLU::solve: rhs size " + std::to_string(rhs.size()) +
                                 " != n=" + std::to_string(n_));
  }

  // Forward: apply the recorded row interchanges, then the unit-lower-
  // triangular multipliers (L y = P b).
  for (int j = 0; j < n_; ++j) {
    const int p = pivot_[static_cast<size_t>(j)];
    if (p != j) {
      std::swap(rhs[static_cast<size_t>(j)], rhs[static_cast<size_t>(p)]);
    }
    const int km = std::min(kl, n_ - 1 - j);
    for (int i = 1; i <= km; ++i) {
      rhs[static_cast<size_t>(j + i)] -= elem(j + i, j) * rhs[static_cast<size_t>(j)];
    }
  }

  // Back-substitution (U x = y); `hi` mirrors factor()'s `ju` bound.
  for (int j = n_ - 1; j >= 0; --j) {
    double sum = rhs[static_cast<size_t>(j)];
    const int hi = std::min(j + ku + kl, n_ - 1);
    for (int c = j + 1; c <= hi; ++c) {
      sum -= elem(j, c) * rhs[static_cast<size_t>(c)];
    }
    rhs[static_cast<size_t>(j)] = sum / elem(j, j);
  }
}

// --- not-a-knot cubic B-spline fit --------------------------------------

namespace {

// Builds and factors the (n+2)x(n+2) not-a-knot system of
// fit_bspline_1d()'s doc comment. Shared by fit_bspline_1d() (one line) and
// fit_bspline_3d() (factor once, reused for every line of a pass).
BandedLU build_notaknot_factored(int n) {
  const int m = n + 2;
  BandedLU lu(m);

  lu.at(0, 0) = 1.0;
  lu.at(0, 1) = -4.0;
  lu.at(0, 2) = 6.0;
  lu.at(0, 3) = -4.0;
  lu.at(0, 4) = 1.0;

  for (int i = 0; i < n; ++i) {
    const int row = i + 1;
    lu.at(row, i + 0) = 1.0 / 6.0;
    lu.at(row, i + 1) = 4.0 / 6.0;
    lu.at(row, i + 2) = 1.0 / 6.0;
  }

  const int rlast = n + 1;
  lu.at(rlast, n - 3) = 1.0;
  lu.at(rlast, n - 2) = -4.0;
  lu.at(rlast, n - 1) = 6.0;
  lu.at(rlast, n - 0) = -4.0;
  lu.at(rlast, n + 1) = 1.0;

  lu.factor();
  return lu;
}

// Solves the not-a-knot system already factored in `lu` (built for this
// same n = f.size()) against interpolation data `f`, returning n+2
// coefficients.
std::vector<double> notaknot_solve(const BandedLU &lu, const std::vector<double> &f) {
  const size_t n = f.size();
  std::vector<double> rhs(n + 2, 0.0);
  for (size_t i = 0; i < n; ++i) {
    rhs[i + 1] = f[i];
  }
  lu.solve(rhs);
  return rhs;
}

} // namespace

std::vector<double> fit_bspline_1d(const std::vector<double> &f) {
  const int n = static_cast<int>(f.size());
  if (n < 4) {
    throw std::invalid_argument("fit_bspline_1d: n must be >= 4 (got " + std::to_string(n) + ")");
  }
  const BandedLU lu = build_notaknot_factored(n);
  return notaknot_solve(lu, f);
}

// --- Bspline3 / fit_bspline_3d ------------------------------------------

Bspline3::Bspline3(int nx, int nu, int ny, double x0, double hx, double u0, double hu, double y0,
                    double hy, std::vector<double> coeffs)
    : nx_(nx), nu_(nu), ny_(ny), x0_(x0), hx_(hx), u0_(u0), hu_(hu), y0_(y0), hy_(hy),
      coeffs_(std::move(coeffs)) {
  const size_t expected =
      static_cast<size_t>(nx_ + 2) * static_cast<size_t>(nu_ + 2) * static_cast<size_t>(ny_ + 2);
  if (coeffs_.size() != expected) {
    throw std::invalid_argument("Bspline3: coeffs size " + std::to_string(coeffs_.size()) +
                                 " != (nx+2)*(nu+2)*(ny+2)=" + std::to_string(expected));
  }
}

BsplineView3 Bspline3::view() const {
  BsplineView3 v;
  v.c = coeffs_.data();
  v.nx = nx_;
  v.nu = nu_;
  v.ny = ny_;
  v.x0 = x0_;
  v.hx = hx_;
  v.u0 = u0_;
  v.hu = hu_;
  v.y0 = y0_;
  v.hy = hy_;
  return v;
}

namespace {

// Gathers `count` values from `data` starting at `base`, `stride` apart.
std::vector<double> gather_line(const std::vector<double> &data, size_t base, size_t stride, int count) {
  std::vector<double> line(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    line[static_cast<size_t>(i)] = data[base + static_cast<size_t>(i) * stride];
  }
  return line;
}

// Inverse of gather_line(): writes `line` back into `data` at the same
// base/stride.
void scatter_line(std::vector<double> &data, size_t base, size_t stride, const std::vector<double> &line) {
  for (size_t i = 0; i < line.size(); ++i) {
    data[base + i * stride] = line[i];
  }
}

} // namespace

Bspline3 fit_bspline_3d(int nx, int nu, int ny, double x0, double hx, double u0, double hu, double y0,
                        double hy, const std::vector<double> &data) {
  if (nx < 4 || nu < 4 || ny < 4) {
    throw std::invalid_argument("fit_bspline_3d: nx, nu, ny must all be >= 4 (got nx=" +
                                 std::to_string(nx) + ", nu=" + std::to_string(nu) +
                                 ", ny=" + std::to_string(ny) + ")");
  }
  const size_t expected = static_cast<size_t>(nx) * static_cast<size_t>(nu) * static_cast<size_t>(ny);
  if (data.size() != expected) {
    throw std::invalid_argument("fit_bspline_3d: data size " + std::to_string(data.size()) +
                                 " != nx*nu*ny=" + std::to_string(expected));
  }

  const int nxp = nx + 2;
  const int nup = nu + 2;
  const int nyp = ny + 2;

  // Pass 1: fit along x for every (u,y) line. `data` is [ky][ju][ix], ix
  // fastest, so a fixed-(ju,ky) line is already contiguous -- gather with
  // stride 1. Result: stage1, dims (nxp, nu, ny), ix fastest.
  const BandedLU lu_x = build_notaknot_factored(nx);
  std::vector<double> stage1(static_cast<size_t>(nxp) * static_cast<size_t>(nu) * static_cast<size_t>(ny));
  for (int ky = 0; ky < ny; ++ky) {
    for (int ju = 0; ju < nu; ++ju) {
      const size_t base_in = static_cast<size_t>(nx) *
                              (static_cast<size_t>(ju) + static_cast<size_t>(nu) * static_cast<size_t>(ky));
      const std::vector<double> line = gather_line(data, base_in, 1, nx);
      const std::vector<double> c = notaknot_solve(lu_x, line);
      const size_t base_out =
          static_cast<size_t>(nxp) *
          (static_cast<size_t>(ju) + static_cast<size_t>(nu) * static_cast<size_t>(ky));
      scatter_line(stage1, base_out, 1, c);
    }
  }

  // Pass 2: fit along u for every (x,y) line. stage1 is (nxp, nu, ny); a
  // fixed-(ix,ky) line has stride nxp. Result: stage2, dims (nxp, nup, ny).
  const BandedLU lu_u = build_notaknot_factored(nu);
  std::vector<double> stage2(static_cast<size_t>(nxp) * static_cast<size_t>(nup) *
                              static_cast<size_t>(ny));
  for (int ky = 0; ky < ny; ++ky) {
    for (int ix = 0; ix < nxp; ++ix) {
      const size_t base_in =
          static_cast<size_t>(ix) + static_cast<size_t>(nxp) * static_cast<size_t>(nu) * static_cast<size_t>(ky);
      const std::vector<double> line = gather_line(stage1, base_in, static_cast<size_t>(nxp), nu);
      const std::vector<double> c = notaknot_solve(lu_u, line);
      const size_t base_out =
          static_cast<size_t>(ix) + static_cast<size_t>(nxp) * static_cast<size_t>(nup) * static_cast<size_t>(ky);
      scatter_line(stage2, base_out, static_cast<size_t>(nxp), c);
    }
  }

  // Pass 3: fit along y for every (x,u) line. stage2 is (nxp, nup, ny); a
  // fixed-(ix,iu) line has stride nxp*nup. Result: coeffs, dims
  // (nxp, nup, nyp).
  const BandedLU lu_y = build_notaknot_factored(ny);
  std::vector<double> coeffs(static_cast<size_t>(nxp) * static_cast<size_t>(nup) *
                              static_cast<size_t>(nyp));
  const size_t y_stride = static_cast<size_t>(nxp) * static_cast<size_t>(nup);
  for (int iu = 0; iu < nup; ++iu) {
    for (int ix = 0; ix < nxp; ++ix) {
      const size_t base = static_cast<size_t>(ix) + static_cast<size_t>(nxp) * static_cast<size_t>(iu);
      const std::vector<double> line = gather_line(stage2, base, y_stride, ny);
      const std::vector<double> c = notaknot_solve(lu_y, line);
      scatter_line(coeffs, base, y_stride, c);
    }
  }

  return Bspline3(nx, nu, ny, x0, hx, u0, hu, y0, hy, std::move(coeffs));
}

} // namespace eeos
