// entropy_eos/host/units.hpp
//
// Physical constants and unit conventions used by the host-side table and
// adapter code. Host-only (see CODE.md "Layout"): not needed by core/.

#pragma once

namespace eeos {

// MeV -> erg. CODATA 2018 exact value: 1 eV = 1.602176634e-19 J, so
// 1 MeV = 1.602176634e-13 J = 1.602176634e-6 erg (1 J = 1e7 erg).
constexpr double MeV_to_erg = 1.602176634e-6;

// Boltzmann constant, erg/K. CODATA 2018 exact value (SI redefinition):
// k_B = 1.380649e-23 J/K = 1.380649e-16 erg/K.
constexpr double k_B_erg_per_K = 1.380649e-16;

// Speed of light, cm/s. Exact SI value: c = 2.99792458e8 m/s = 2.99792458e10 cm/s.
constexpr double c_light_cm_s = 2.99792458e10;

// Atomic mass unit, g. CODATA 2018: 1 u = 1.66053906892e-27 kg = 1.66053906892e-24 g.
constexpr double m_amu_g = 1.66053906892e-24;

// Neutron mass (CODATA 2022). Empirically the baryon-mass convention of the
// SRO (Schneider-Roberts-Ott) tables: rebuilding the adapter with this m_B
// collapses the delta_T fidelity quantiles from a flat ~8.7e-3 (the m_n/m_u
// ratio) to ~1.6e-5 -- see CODE.md "M2 design notes" and the adapter doc's
// remark that the delta_T audit measures the table's true m_B.
constexpr double m_neutron_g = 1.67492749804e-24;

// Default baryon mass used to convert between per-baryon and per-gram
// quantities when a table does not declare its own convention.
//
// [decide] (CODE.md "Open decisions" #2): the m_B convention is per table
// family — some formats carry an explicit attribute, others rely on a
// documented constant tied to the format. This default (the atomic mass
// unit) is a placeholder callers must be able to override; it must never be
// silently assumed for a real table without checking the format's
// documented convention first.
constexpr double m_B_default_g = m_amu_g;

} // namespace eeos
