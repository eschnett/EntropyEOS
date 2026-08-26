// entropy_eos/entropy_eos.hpp
//
// Umbrella header for the entropy_eos library. Consumers write
// `#include <entropy_eos/entropy_eos.hpp>` (or the individual headers below)
// regardless of consumption mode (copied source vs. installed library); see
// CODE.md "Environment". This header currently pulls in the M1 modules;
// later milestones (bspline_fit/adapter_build/adapter_eval, prim2con/
// con2prim, io_*) extend it as they land.

#pragma once

// IWYU pragma: begin_exports
#include "entropy_eos/core/adapter_eval.hpp"
#include "entropy_eos/core/bspline_eval.hpp"
#include "entropy_eos/core/con2prim.hpp"
#include "entropy_eos/core/defs.hpp"
#include "entropy_eos/core/prim2con.hpp"
#include "entropy_eos/core/state_policy.hpp"

#include "entropy_eos/host/adapter_build.hpp"
#include "entropy_eos/host/bspline_fit.hpp"
#include "entropy_eos/host/check.hpp"
#include "entropy_eos/host/io_stellarcollapse.hpp"
#include "entropy_eos/host/repair.hpp"
#include "entropy_eos/host/synthetic.hpp"
#include "entropy_eos/host/table.hpp"
#include "entropy_eos/host/units.hpp"
// IWYU pragma: end_exports
