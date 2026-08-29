#!/bin/bash
# tests/check_mirror_parallel.sh -- M4 (eos-device-interface.md S5): the HIP
# mirror ships compile-untested (no AMD hardware), so its trustworthiness
# argument is that it is the MECHANICAL s/cuda/hip/ rename of the tested
# mirror_cuda.hpp. This check makes that a property CI enforces rather than a
# promise: comments may differ (each header documents itself), but after
# stripping them every code line of mirror_hip.hpp must map 1:1 onto
# mirror_cuda.hpp under the rename. Pure text -- no GPU toolchain involved.
set -eu
cd "$(dirname "$0")/.."

cuda=entropy_eos/device/mirror_cuda.hpp
hip=entropy_eos/device/mirror_hip.hpp

# Drop comments (whole-line and trailing) and blank lines; neither header
# contains "//" inside a string literal, which keeps this sed exact.
strip_comments() {
  sed -e 's|//.*||' -e 's/[[:space:]]*$//' "$1" | grep -v '^$'
}

# The include path is the one non-uniform vendor spelling
# (<hip/hip_runtime.h> vs <cuda_runtime.h>): map it first, then apply the
# mechanical rename to everything else.
renamed_hip() {
  strip_comments "$hip" |
    sed -e 's|<hip/hip_runtime.h>|<cuda_runtime.h>|' \
        -e 's/hip/cuda/g' -e 's/Hip/Cuda/g' -e 's/HIP/CUDA/g'
}

if ! diff -u <(strip_comments "$cuda") <(renamed_hip); then
  echo "check_mirror_parallel: FAIL -- mirror_hip.hpp is not the mechanical" >&2
  echo "s/cuda/hip/ rename of mirror_cuda.hpp; edit mirror_cuda.hpp first," >&2
  echo "then re-derive mirror_hip.hpp (see the file headers)." >&2
  exit 1
fi
echo "check_mirror_parallel: OK (mirror_hip.hpp == rename of mirror_cuda.hpp)"
