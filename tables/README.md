# EOS tables

This directory holds the tabulated EOS files used by the repair and test harnesses
(`eos_repair`, `eos_test`; see `../CODE.md`). The files are a few hundred MB each and
are **not committed** — everything here except this README and the `.gitignore` is
ignored by git. Download them as follows.

All four tables are in the stellarcollapse.org (O'Connor & Ott 2010) HDF5 format:
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

## DD2 (Typel et al. relativistic mean field; Hempel–Schaffner-Bielich tabulation)

```bash
curl -fL -o tables/Hempel_DD2EOS_rho234_temp180_ye60_version_1.1_20120817.h5.bz2 \
  https://stellarcollapse.org/~evanoc/Hempel_DD2EOS_rho234_temp180_ye60_version_1.1_20120817.h5.bz2
bunzip2 tables/Hempel_DD2EOS_rho234_temp180_ye60_version_1.1_20120817.h5.bz2
```

Grid: 234 × 180 × 60 (ρ × T × Ye), 444 MB uncompressed. Source page:
https://stellarcollapse.org/equationofstate.html

## SFHo (Steiner–Hempel–Fischer, astrophysically favoured; same tabulation)

```bash
curl -fL -o tables/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5.bz2 \
  https://stellarcollapse.org/~evanoc/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5.bz2
bunzip2 tables/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5.bz2
```

Grid: 222 × 180 × 60 (ρ × T × Ye), 421 MB uncompressed. Source page:
https://stellarcollapse.org/equationofstate.html

## Sanitizer fixture (optional, generated)

Sanitizer test runs (`make test SAN=1`) automatically substitute a small cropped
LS220 fixture when present, cutting the SAN wall time from ~9 min to ~2.7 min while
keeping real-data coverage (the crop contains LS220's three genuine non-finite
cs2/gamma points and the full temperature axis). Generate it once after downloading
LS220:

```bash
make san-fixture
```

which runs `tools/eos_crop` (index ranges irho 60–120 × jT 0–135 × kYe 10–25,
~19 MB). Without the fixture, sanitizer runs fall back to the full table
transparently. Plain (non-sanitizer) builds and `make integration` always use the
full tables.

## Citations

- E. O'Connor & C. D. Ott, Class. Quantum Grav. 27, 114103 (2010) — table format.
- J. M. Lattimer & F. D. Swesty, Nucl. Phys. A 535, 331 (1991) — LS220 physics.
- A. S. Schneider, L. F. Roberts & C. D. Ott, Phys. Rev. C 96, 065802 (2017) — SRO.
- M. Hempel & J. Schaffner-Bielich, Nucl. Phys. A 837, 210 (2010) — DD2/SFHo tabulation
  (nuclear statistical equilibrium model).
- S. Typel et al., Phys. Rev. C 81, 015803 (2010) — DD2 physics.
- A. W. Steiner, M. Hempel & T. Fischer, Astrophys. J. 774, 17 (2013) — SFHo physics.
