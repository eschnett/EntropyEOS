#include "entropy_eos/host/adapter_build.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace eeos {

namespace {

// Number of OpenMP threads that will run the parallel scan below (1 when
// built without OpenMP; mirrors host/check.cpp's max_threads()/this_thread()
// pattern for a per-thread-accumulator-then-merge parallel reduction).
int max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

int this_thread() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

constexpr std::size_t kMaxWorst = 10;

// Per-thread accumulator for one of the Sigma_u/L_u monotonicity audits:
// running minimum, violation count, and a bounded (<=10) "most negative
// first" worst-location list, merged across threads after the parallel
// scan. Bounded memory regardless of how many points violate (relevant
// since build_entropy_eos() does not require a pre-repaired table).
struct AuditAccum {
  double min_value = std::numeric_limits<double>::infinity();
  std::size_t violation_count = 0;
  std::vector<AuditLoc> worst;

  void offer(const AuditLoc &loc) {
    if (worst.size() < kMaxWorst) {
      worst.push_back(loc);
      return;
    }
    std::size_t max_idx = 0;
    double max_val = worst[0].value;
    for (std::size_t i = 1; i < worst.size(); ++i) {
      if (worst[i].value > max_val) {
        max_val = worst[i].value;
        max_idx = i;
      }
    }
    if (loc.value < max_val) {
      worst[max_idx] = loc;
    }
  }

  void consider(double x, double u, double y, double value) {
    min_value = std::min(min_value, value);
    if (value <= 0.0) {
      ++violation_count;
      offer(AuditLoc{x, u, y, value});
    }
  }

  void merge_from(const AuditAccum &other) {
    min_value = std::min(min_value, other.min_value);
    violation_count += other.violation_count;
    for (const AuditLoc &loc : other.worst) {
      offer(loc);
    }
  }

  MonotonicityAudit finalize() const {
    MonotonicityAudit m;
    m.min_value = min_value;
    m.violation_count = violation_count;
    m.worst = worst;
    std::sort(m.worst.begin(), m.worst.end(),
              [](const AuditLoc &a, const AuditLoc &b) { return a.value < b.value; });
    return m;
  }
};

void check_uniform_axis(const std::vector<double> &axis, const char *name, double tol) {
  const std::size_t n = axis.size();
  const double span = axis.back() - axis.front();
  const double h = span / static_cast<double>(n - 1);
  if (!(h > 0.0)) {
    throw std::runtime_error(std::string("build_entropy_eos: axis '") + name +
                              "' is not strictly increasing");
  }
  for (std::size_t i = 1; i < n; ++i) {
    const double spacing = axis[i] - axis[i - 1];
    const double rel = std::fabs(spacing - h) / h;
    if (rel > tol) {
      throw std::runtime_error(std::string("build_entropy_eos: axis '") + name +
                                "' is not uniform (relative spacing deviation " + std::to_string(rel) +
                                " at index " + std::to_string(i) + " exceeds tolerance " +
                                std::to_string(tol) + ")");
    }
  }
}

void check_field_present_finite(const RawTable &table, const char *name) {
  if (!table.has_field(name)) {
    throw std::runtime_error(std::string("build_entropy_eos: table has no field '") + name + "'");
  }
  for (double v : table.field(name)) {
    if (!std::isfinite(v)) {
      throw std::runtime_error(std::string("build_entropy_eos: field '") + name +
                                "' has a non-finite value");
    }
  }
}

struct ScanResult {
  double eps_hat_min = std::numeric_limits<double>::infinity();
  AdapterAudit audit;
};

// M2d-2: min eps_hat over the *extended* box (opts.ext_cells grid cells
// beyond the physical box on every side of x and u), sampled at the same
// refinement as scan_refined_grid() below but through the designed-tail
// evaluator evaluate() itself uses (core/adapter_eval.hpp's
// detail::aeval_extended()) instead of a plain bspline_eval3() -- see
// build_entropy_eos()'s step 3 doc comment for why this must run (the L
// u-low tail can dip eps_hat below the table's own minimum) and why it is
// safe to run with the *unshifted* x0 (kappa is not yet known at this
// point in build_entropy_eos()). Only L is needed (eps_hat is purely a
// function of it); sigma does not participate in the eps floor directly,
// but M3g's causal slope clamp makes L's u-HIGH tail depend on sigma's own
// log-tail growth rate, so the sigma view is threaded in too and the scan
// keeps evaluating exactly what evaluate() would (the clamp is
// kappa-independent -- it only involves eps_hat and dln(sigma)/du -- so it
// is as safe to run before the kappa relabeling as the rest of this scan).
// M3i changed what this scan sees in the x-LOW band: L's tail there is now
// the plain generic (linear-in-L, i.e. power-law-in-rho) continuation rather
// than the old slope-to-zero override, so eps_hat is no longer frozen at its
// seam value below rho_min. It moves in the *safe* direction at a radiation
// seam (L_x < 0, so eps grows as rho drops) and by <~ a few percent at a
// matter seam (L_x ~ 0); this scan is what turns that into a measured
// statement about the eps floor rather than an argument. It does move kappa
// (by 2.6e-7 / 5.1e-8 relative on the two real tables, and not at all on the
// synthetic one) -- the extended-box eps minimum sits at the x-low x u-low
// corner, i.e. exactly where M3i works. See CODE.md's M3i findings, which
// censuses that drift as the ONLY in-box change M3i produces.
double scan_extended_eps_floor(const BsplineView3 &L_view, const BsplineView3 &sigma_view, double x0,
                                double hx, int nx, double u0, double hu, int nu, double y0, double hy,
                                int ny, int refine, int ext_cells, double slope_floor_sigma,
                                double slope_floor_L, double cs2_ext_cap, double shift_hat,
                                double inv_c2) {
  const double x_lo = x0, x_hi = x0 + static_cast<double>(nx - 1) * hx;
  const double u_lo = u0, u_hi = u0 + static_cast<double>(nu - 1) * hu;
  const double x_ext_lo = x_lo - static_cast<double>(ext_cells) * hx;
  const double u_ext_lo = u_lo - static_cast<double>(ext_cells) * hu;

  const int rx = (nx - 1 + 2 * ext_cells) * refine + 1;
  const int ru = (nu - 1 + 2 * ext_cells) * refine + 1;
  const int ry = (ny - 1) * refine + 1;
  const double dx = hx / static_cast<double>(refine);
  const double du = hu / static_cast<double>(refine);
  const double dy = hy / static_cast<double>(refine);

  const int nthreads = max_threads();
  std::vector<double> local_min(static_cast<std::size_t>(nthreads), std::numeric_limits<double>::infinity());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int ky = 0; ky < ry; ++ky) {
    const int tid = this_thread();
    const double y = y0 + static_cast<double>(ky) * dy;
    double &m = local_min[static_cast<std::size_t>(tid)];

    for (int ju = 0; ju < ru; ++ju) {
      const double u = u_ext_lo + static_cast<double>(ju) * du;
      const bool u_above = u > u_hi;
      for (int ix = 0; ix < rx; ++ix) {
        const double x = x_ext_lo + static_cast<double>(ix) * dx;

        // M3g causal clamp (u-high side only): b_cap = (1+cs2_ext_cap)*alpha
        // with alpha = sigma's log-tail growth rate at the seam.
        double b_cap = 0.0;
        if (u_above) {
          const double x_seam = std::min(std::max(x, x_lo), x_hi);
          const double alpha = detail::aeval_sigma_u_high_alpha(sigma_view, x_seam, u_hi, y,
                                                                 slope_floor_sigma);
          if (alpha > 0.0) b_cap = (1.0 + cs2_ext_cap) * alpha;
        }
        const detail::ExtSpec spec{x_lo,
                                   x_hi,
                                   u_lo,
                                   u_hi,
                                   x_ext_lo,
                                   slope_floor_L,
                                   /*x_low_log=*/false,  // sigma-only (M3i); this scan evaluates L
                                   /*u_high_log=*/false, // sigma-only (M3g); ditto
                                   b_cap,
                                   shift_hat,
                                   inv_c2};
        const BsplineEval3 Lv = detail::aeval_extended(L_view, x, u, y, spec);
        const double eps_hat = std::pow(10.0, Lv.f) * inv_c2 - shift_hat;
        if (eps_hat < m) m = eps_hat;
      }
    }
  }

  double result = std::numeric_limits<double>::infinity();
  for (int t = 0; t < nthreads; ++t) result = std::min(result, local_min[static_cast<std::size_t>(t)]);
  return result;
}

// One pass over a refine-times-refined grid spanning the whole (unshifted)
// physical box, evaluating both fitted splines at every sample point:
// tracks the running minimum of eps_hat (for the S5 kappa floor) and the
// Sigma_u/L_u monotonicity audits (S10), all from the *spline* values
// (never the raw data -- see build_entropy_eos()'s doc comment). Columns
// (here, ky slices) are independent -> OpenMP.
ScanResult scan_refined_grid(const BsplineView3 &sigma_view, const BsplineView3 &L_view, double x0,
                              double hx, int nx, double u0, double hu, int nu, double y0, double hy,
                              int ny, int refine, double shift_hat, double inv_c2) {
  const int rx = (nx - 1) * refine + 1;
  const int ru = (nu - 1) * refine + 1;
  const int ry = (ny - 1) * refine + 1;
  const double dx = hx / static_cast<double>(refine);
  const double du = hu / static_cast<double>(refine);
  const double dy = hy / static_cast<double>(refine);

  const int nthreads = max_threads();
  std::vector<double> local_eps_min(static_cast<std::size_t>(nthreads),
                                     std::numeric_limits<double>::infinity());
  std::vector<AuditAccum> local_sigma(static_cast<std::size_t>(nthreads));
  std::vector<AuditAccum> local_L(static_cast<std::size_t>(nthreads));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int ky = 0; ky < ry; ++ky) {
    const int tid = this_thread();
    const double y = y0 + static_cast<double>(ky) * dy;
    double &eps_min = local_eps_min[static_cast<std::size_t>(tid)];
    AuditAccum &sig_acc = local_sigma[static_cast<std::size_t>(tid)];
    AuditAccum &L_acc = local_L[static_cast<std::size_t>(tid)];

    for (int ju = 0; ju < ru; ++ju) {
      const double u = u0 + static_cast<double>(ju) * du;
      for (int ix = 0; ix < rx; ++ix) {
        const double x = x0 + static_cast<double>(ix) * dx;

        const BsplineEval3 sig = bspline_eval3(sigma_view, x, u, y);
        const BsplineEval3 Lv = bspline_eval3(L_view, x, u, y);

        sig_acc.consider(x, u, y, sig.fu);
        L_acc.consider(x, u, y, Lv.fu);

        const double eps_hat = std::pow(10.0, Lv.f) * inv_c2 - shift_hat;
        if (eps_hat < eps_min) {
          eps_min = eps_hat;
        }
      }
    }
  }

  ScanResult result;
  AuditAccum sigma_total, L_total;
  for (int t = 0; t < nthreads; ++t) {
    result.eps_hat_min = std::min(result.eps_hat_min, local_eps_min[static_cast<std::size_t>(t)]);
    sigma_total.merge_from(local_sigma[static_cast<std::size_t>(t)]);
    L_total.merge_from(local_L[static_cast<std::size_t>(t)]);
  }
  result.audit.sigma_u = sigma_total.finalize();
  result.audit.L_u = L_total.finalize();
  return result;
}

} // namespace

EntropyEOS::EntropyEOS(Bspline3 sigma, Bspline3 L, double kappa, double m_B_star_g, double m_B_table_g,
                        double shift_hat, double conv_t, double x_lo, double x_hi, double u_lo, double u_hi,
                        double y_lo, double y_hi, double x_ext_lo, double x_ext_hi, double u_ext_lo,
                        double u_ext_hi, double ext_slope_floor_sigma, double ext_slope_floor_L,
                        double cs2_ext_cap, AdapterAudit audit, int max_iter)
    : sigma_(std::move(sigma)), L_(std::move(L)), kappa_(kappa), m_B_star_g_(m_B_star_g),
      m_B_table_g_(m_B_table_g), shift_hat_(shift_hat), conv_t_(conv_t), x_lo_(x_lo), x_hi_(x_hi),
      u_lo_(u_lo), u_hi_(u_hi), y_lo_(y_lo), y_hi_(y_hi), x_ext_lo_(x_ext_lo), x_ext_hi_(x_ext_hi),
      u_ext_lo_(u_ext_lo), u_ext_hi_(u_ext_hi), ext_slope_floor_sigma_(ext_slope_floor_sigma),
      ext_slope_floor_L_(ext_slope_floor_L), cs2_ext_cap_(cs2_ext_cap), max_iter_(max_iter),
      audit_(std::move(audit)) {}

EntropyEOSView EntropyEOS::view() const {
  EntropyEOSView v;
  v.sigma = sigma_.view();
  v.L = L_.view();
  v.kappa = kappa_;
  v.shift_hat = shift_hat_;
  v.conv_t = conv_t_;
  v.inv_c2 = 1.0 / (c_light_cm_s * c_light_cm_s);
  v.x_lo = x_lo_;
  v.x_hi = x_hi_;
  v.u_lo = u_lo_;
  v.u_hi = u_hi_;
  v.y_lo = y_lo_;
  v.y_hi = y_hi_;
  v.x_ext_lo = x_ext_lo_;
  v.x_ext_hi = x_ext_hi_;
  v.u_ext_lo = u_ext_lo_;
  v.u_ext_hi = u_ext_hi_;
  v.ext_slope_floor_sigma = ext_slope_floor_sigma_;
  v.ext_slope_floor_L = ext_slope_floor_L_;
  v.cs2_ext_cap = cs2_ext_cap_;
  v.max_iter = max_iter_;
  return v;
}

EntropyEOS build_entropy_eos(const RawTable &table, const BuildOptions &opts) {
  // --- 1. Validate (does not repair) ---------------------------------------
  const std::size_t nrho = table.nrho();
  const std::size_t ntemp = table.ntemp();
  const std::size_t nye = table.nye();

  if (nrho < 4) {
    throw std::runtime_error("build_entropy_eos: rho axis has fewer than 4 points");
  }
  if (ntemp < 4) {
    throw std::runtime_error("build_entropy_eos: T axis has fewer than 4 points");
  }
  if (nye < 4) {
    throw std::runtime_error("build_entropy_eos: Ye axis has fewer than 4 points");
  }

  check_uniform_axis(table.logrho(), "logrho", opts.uniform_tol);
  check_uniform_axis(table.logtemp(), "logtemp", opts.uniform_tol);
  check_uniform_axis(table.ye(), "ye", opts.uniform_tol);

  check_field_present_finite(table, "logenergy");
  check_field_present_finite(table, "entropy");
  if (!table.has_attribute("energy_shift")) {
    throw std::runtime_error("build_entropy_eos: table has no 'energy_shift' attribute");
  }

  // --- 2. Fit Sigma-hat, L-hat on the table's native (x,u,y) grid ----------
  const double x0 = table.logrho().front();
  const double hx = (table.logrho().back() - table.logrho().front()) / static_cast<double>(nrho - 1);
  const double u0 = table.logtemp().front();
  const double hu = (table.logtemp().back() - table.logtemp().front()) / static_cast<double>(ntemp - 1);
  const double y0 = table.ye().front();
  const double hy = (table.ye().back() - table.ye().front()) / static_cast<double>(nye - 1);

  const int inx = static_cast<int>(nrho);
  const int inu = static_cast<int>(ntemp);
  const int iny = static_cast<int>(nye);

  Bspline3 sigma_raw =
      fit_bspline_3d(inx, inu, iny, x0, hx, u0, hu, y0, hy, table.field("entropy"));
  Bspline3 L_raw = fit_bspline_3d(inx, inu, iny, x0, hx, u0, hu, y0, hy, table.field("logenergy"));

  const double shift_cgs = table.energy_shift();
  const double inv_c2 = 1.0 / (c_light_cm_s * c_light_cm_s);
  const double shift_hat = shift_cgs * inv_c2;
  const double conv_t = MeV_to_erg / (opts.m_B_table_g * c_light_cm_s * c_light_cm_s);

  // --- 3. kappa re-zeroing + 4. monotonicity audit (one refined-grid pass) -
  const ScanResult scan = scan_refined_grid(sigma_raw.view(), L_raw.view(), x0, hx, inx, u0, hu, inu, y0,
                                             hy, iny, opts.refine, shift_hat, inv_c2);

  // M2d-2: the eps floor must also cover the extension zones (build_entropy_eos()'s
  // doc comment step 3) -- scan the extended box too, with the unshifted x0
  // (kappa-independent, see that comment), and fold its min in alongside
  // the physical-box scan's own.
  const double ext_eps_min = scan_extended_eps_floor(
      L_raw.view(), sigma_raw.view(), x0, hx, inx, u0, hu, inu, y0, hy, iny, opts.refine, opts.ext_cells,
      opts.ext_slope_floor_sigma, opts.ext_slope_floor_L, opts.cs2_ext_cap, shift_hat, inv_c2);
  const double eps_hat_min = std::min(scan.eps_hat_min, ext_eps_min);

  const double eps_floor =
      std::min(0.0, eps_hat_min - (opts.kappa_margin_abs + opts.kappa_margin_rel * std::fabs(eps_hat_min)));
  const double kappa = 1.0 + eps_floor;
  const double m_B_star_g = kappa * opts.m_B_table_g;
  const double x0_star = x0 + std::log10(kappa);

  // Relabel the fitted splines' x-origin by the density shift -- same
  // coefficients, no refit (eos-adapter-F-to-U.md S5's "Implementation is a
  // relabeling at build time").
  Bspline3 sigma_final(inx, inu, iny, x0_star, hx, u0, hu, y0, hy, sigma_raw.coeffs());
  Bspline3 L_final(inx, inu, iny, x0_star, hx, u0, hu, y0, hy, L_raw.coeffs());

  const double x_lo = x0_star;
  const double x_hi = x0_star + static_cast<double>(nrho - 1) * hx;
  const double u_lo = u0;
  const double u_hi = u0 + static_cast<double>(ntemp - 1) * hu;
  const double y_lo = y0;
  const double y_hi = y0 + static_cast<double>(nye - 1) * hy;

  // M2d-2 extended box: same log10(kappa) shift as x_lo/x_hi above (the
  // extension's blend width and extent are physical grid-cell counts, so
  // they ride along with the relabeling unchanged).
  const double x_ext_lo = x_lo - static_cast<double>(opts.ext_cells) * hx;
  const double x_ext_hi = x_hi + static_cast<double>(opts.ext_cells) * hx;
  const double u_ext_lo = u_lo - static_cast<double>(opts.ext_cells) * hu;
  const double u_ext_hi = u_hi + static_cast<double>(opts.ext_cells) * hu;

  return EntropyEOS(std::move(sigma_final), std::move(L_final), kappa, m_B_star_g, opts.m_B_table_g,
                     shift_hat, conv_t, x_lo, x_hi, u_lo, u_hi, y_lo, y_hi, x_ext_lo, x_ext_hi, u_ext_lo,
                     u_ext_hi, opts.ext_slope_floor_sigma, opts.ext_slope_floor_L, opts.cs2_ext_cap,
                     scan.audit, /*max_iter=*/50);
}

} // namespace eeos
