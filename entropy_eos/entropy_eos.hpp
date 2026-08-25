// entropy_eos/entropy_eos.hpp
//
// Umbrella header for the entropy_eos library. Consumers write
// `#include <entropy_eos/entropy_eos.hpp>` (or the individual headers below)
// regardless of consumption mode (copied source vs. installed library); see
// CODE.md "Environment". This header currently pulls in the M1 modules;
// later milestones (bspline_fit/adapter_build/adapter_eval, prim2con/
// con2prim, io_*) extend it as they land.

#pragma once

#include "entropy_eos/core/defs.hpp"

#include "entropy_eos/host/synthetic.hpp"
#include "entropy_eos/host/table.hpp"
#include "entropy_eos/host/units.hpp"
