# EOS tables

This directory holds the tabulated EOS files used by the repair and test harnesses
(`eos_repair`, `eos_test`; see `../CODE.md`). The files are a few hundred MB each and
are **not committed** — everything here except this README and the `.gitignore` is
ignored by git. Download them as follows.

Both tables are in the stellarcollapse.org (O'Connor & Ott 2010) HDF5 format:
log10-spaced `logrho` [g/cm³], `logtemp` [MeV], linear `ye`; `logenergy` =
log10(ε + `energy_shift`) [erg/g], linear `entropy` [k_B/baryon], `logpress`
[dyn/cm²], plus further fields.

## LS220 (Lattimer & Swesty, K = 220 MeV; O'Connor–Ott tabulation)

```bash
curl -fL -o tables/LS220_234r_136t_50y_analmu_20091212_SVNr26.h5.bz2 \
  https://stellarcollapse.org/EOS/LS220_234r_136t_50y_analmu_20091212_SVNr26.h5.bz2
bunzip2 tables/LS220_234r_136t_50y_analmu_20091212_SVNr26.h5.bz2
```

Grid: 234 × 136 × 50 (ρ × T × Ye). Source page:
https://stellarcollapse.org/equationofstate.html
(A newer 240×140×50 variant, `LS220_240r_140t_50y_analmu_20120628_SVNr28.h5.bz2`,
is available on the same page.)

## SRO LS220 (Schneider, Roberts & Ott 2017 re-implementation)

```bash
curl -fL -o tables/LS220_3335_rho391_temp163_ye66.h5.bz2 \
  https://stellarcollapse.org/EOS/LS220_3335_rho391_temp163_ye66.h5.bz2
bunzip2 tables/LS220_3335_rho391_temp163_ye66.h5.bz2
```

Grid: 391 × 163 × 66 (ρ × T × Ye). Source page:
https://stellarcollapse.org/SROEOS.html

## Citations

- E. O'Connor & C. D. Ott, Class. Quantum Grav. 27, 114103 (2010) — table format.
- J. M. Lattimer & F. D. Swesty, Nucl. Phys. A 535, 331 (1991) — LS220 physics.
- A. S. Schneider, L. F. Roberts & C. D. Ott, Phys. Rev. C 96, 065802 (2017) — SRO.
