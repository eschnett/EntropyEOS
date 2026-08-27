// entropy_eos/host/io_stellarcollapse.hpp
//
// I/O backend for the stellarcollapse.org (O'Connor-Ott) HDF5 table layout
// shared by LS220 and SRO (Schneider-Roberts-Ott) tables (see CODE.md "Table
// formats"): log10-stored rho/T axes, linear Ye, linear entropy,
// "energy_shift" scalar attribute. This is the *only* header in the library
// whose corresponding .cpp includes <hdf5.h> (CODE.md "Environment": "HDF5
// is the only external dependency, confined to the io_* translation
// units") -- this header itself includes no HDF5 header and names no HDF5
// type, so consumers that don't need table I/O can use the rest of the
// library (adapter, con2prim) without linking HDF5 at all.
//
// Host-only: STL throughout, may throw (see table.hpp / CODE.md "Data
// model"). Every function throws std::runtime_error, naming the file and/or
// dataset involved, on any failure to open/read/write.

#pragma once

#include <string>

#include "entropy_eos/host/repair.hpp"
#include "entropy_eos/host/table.hpp"

namespace eeos {

// Reads a stellarcollapse-format file into a RawTable:
//   - axes come from "logrho"/"logtemp"/"ye" (all three must exist, each a
//     1-D dataset); if the scalar {1} datasets "pointsrho"/"pointstemp"/
//     "pointsye" are also present, their values are cross-checked against
//     the corresponding axis size and a mismatch throws;
//   - every 3-D dataset whose dims equal (nye, ntemp, nrho) -- the file's
//     native axis order, C-order, iRho fastest -- and whose native type is
//     a float type (readable as double; see the .cpp for why f4 and f8 need
//     no different code path) becomes a RawTable field, added in the order
//     datasets are encountered in the file (see the .cpp for what "file
//     order" means for a format that generally has no creation-order
//     index), so RawTable::field_names() reproduces it;
//   - the scalar {1} datasets "energy_shift" and "have_rel_cs2", if present,
//     become RawTable attributes (as double, regardless of their stored
//     type -- HDF5 converts on read);
//   - everything else (opaque provenance blobs, "points*", any dataset
//     whose dims or type don't match the rules above) is left out of the
//     RawTable entirely; write_stellarcollapse()'s (path_out, table,
//     path_in) overload is what makes such datasets round-trip, by copying
//     them straight from path_in.
// Throws std::runtime_error, naming `path` and (where applicable) the
// offending dataset, if the file can't be opened, an axis is missing or not
// 1-D, a points* cross-check fails, or a candidate field/axis dataset can't
// be read.
RawTable read_stellarcollapse(const std::string &path);

// Writes `table` to a new file at `path_out` (overwritten if it exists),
// using `path_in` as a template for every dataset the RawTable does not
// itself carry: axes ("logrho"/"logtemp"/"ye") and every name in
// table.field_names() are written fresh from the in-memory table (as
// double); every other dataset of `path_in` -- opaque provenance blobs,
// "points*", "energy_shift", "have_rel_cs2", anything read_stellarcollapse()
// left as passthrough -- is copied byte-for-byte via H5Ocopy, so it survives
// even though the RawTable never held it. `path_in`'s datasets are visited
// in the same "file order" read_stellarcollapse() uses; any table field not
// found by name in path_in is appended at the end of `path_out`.
//
// Documented consequence: "energy_shift" (and "have_rel_cs2", if present)
// are always passthrough-copied from path_in by this overload, because
// read_stellarcollapse() surfaces them as RawTable *attributes*, not fields
// -- so mutating table.attribute("energy_shift") between reading and
// calling this overload has no effect on the written file. Use the
// source-less overload below (or hand-roll the dataset) if a table's own
// energy_shift value must reach the file.
//
// Throws std::runtime_error, naming the file(s)/dataset involved, if
// path_in can't be opened, path_out can't be created, or any read/copy/write
// fails.
void write_stellarcollapse(const std::string &path_out, const RawTable &table,
                            const std::string &path_in);

// Writes `table` to a new file at `path_out` (overwritten if it exists) with
// no source file to draw passthrough datasets from: writes "logrho"/
// "logtemp"/"ye", "pointsrho"/"pointstemp"/"pointsye" (derived from the axis
// sizes, stored as the format's native int), every attribute in
// table.attribute_names() (as a scalar {1} double dataset -- this is what
// actually writes "energy_shift", required to be present), and every field
// in table.field_names() (as a 3-D double dataset, dims (nye, ntemp, nrho)).
// Intended for synthetic tables that have no stellarcollapse-format source
// file to pass through provenance blobs from.
//
// Throws std::runtime_error if `table` has no "energy_shift" attribute, if
// path_out can't be created, or if any write fails.
void write_stellarcollapse(const std::string &path_out, const RawTable &table);

// Appends a "/repair" group (must not already exist) to the existing file at
// `path`, recording the outcome of a repair_table() run for provenance:
//   - if result.entries is non-empty, five parallel datasets of that length
//     under /repair: "irho"/"jT"/"kYe" (unsigned), "old_value"/"new_value"
//     (double), plus "field" -- a fixed-length (32-byte, NUL-padded) string
//     dataset naming the field of each entry (chosen over an index into a
//     separate name-list dataset: with only ever "entropy"/"logenergy" in
//     practice, a plain string column is simpler to inspect with h5ls/h5dump
//     and to read back in tests, at a storage cost that's negligible next to
//     the two double columns it sits beside);
//   - if result.entries is empty, none of the above five datasets are
//     created at all -- just the group and its attributes, with
//     n_modified == 0;
//   - scalar attributes on the /repair group (always written): the two
//     min-slope options used ("min_slope_entropy"/"min_slope_logenergy",
//     double), the M3f causal-cap stage's parameters and headline outcome
//     ("causal_cap" 0/1, "cs2_max"/"cs2_cap" double, "causal_rounds_max"/
//     "trace_depth_max"/"anchor_pad" uint64, plus "causal_nodes_capped",
//     "causal_cs2_violations_before"/"_after" and "causal_reverted" 0/1 --
//     see RepairResult::CausalCapSummary), "tool_version"/"input_path"
//     (fixed-length string datasets -- sic, see the .cpp: HDF5 *attributes*
//     here, not top-level datasets), "input_fnv1a" (uint64_t) and
//     "n_modified" (uint64_t).
// Throws std::runtime_error if `path` can't be opened for writing, already
// has a "/repair" group, or any write fails.
void append_repair_group(const std::string &path, const RepairResult &result,
                          const RepairOptions &options, const std::string &input_path,
                          unsigned long long input_fnv1a, const std::string &tool_version);

// Streaming 64-bit FNV-1a hash of a file's raw bytes: a dependency-free
// provenance fingerprint recorded by append_repair_group() (via
// "input_fnv1a") to let a later run confirm which exact input file a
// repaired output was derived from. NOT a cryptographic hash -- collisions
// are easy to construct deliberately; it only guards against accidental
// mismatch (wrong file, re-run after an edit, ...).
// Throws std::runtime_error if `path` can't be opened for reading.
unsigned long long fnv1a_file(const std::string &path);

} // namespace eeos
