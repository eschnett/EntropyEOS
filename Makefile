# Makefile for entropy_eos.
#
# Deliberately plain: no autodetection, no cmake/configure (see CODE.md
# "Environment"). Wildcards are used throughout so that later modules
# (bspline_fit, adapter_build, io_stellarcollapse, ...) need no edits here —
# dropping a new entropy_eos/host/*.cpp or tests/test_*.cpp file is enough.
#
# The documented manual compile line (no Makefile at all) must always work,
# e.g.:
#   c++ -O2 -std=c++17 -fopenmp -I. entropy_eos/host/*.cpp tools/eos_repair.cpp -lhdf5 -o eos_repair

CXX ?= c++
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra

# Empty by default: Apple clang (the default `c++` on macOS) has no bundled
# libomp, so OpenMP is opt-in. Set OPENMP=-fopenmp with gcc (e.g. in CI).
OPENMP ?=

# HDF5_DIR/HDF5_INC/HDF5_LIB wire up io_stellarcollapse's only external
# dependency (see CODE.md "Environment"). HDF5_DIR's default is MacPorts'
# install location on the dev machine; override it (or HDF5_INC/HDF5_LIB
# directly) per system -- e.g. Debian/Ubuntu's libhdf5-dev uses split
# include/lib paths that don't fit a single -dir variable, so CI passes
# HDF5_INC/HDF5_LIB explicitly instead (see .github/workflows/ci.yml).
HDF5_DIR ?= /opt/local
HDF5_INC ?= -I$(HDF5_DIR)/include
HDF5_LIB ?= -L$(HDF5_DIR)/lib -lhdf5

PREFIX ?= /usr/local

CPPFLAGS += -I. $(HDF5_INC)
ALL_CXXFLAGS = $(CXXFLAGS) $(OPENMP)
LDFLAGS ?=
LDFLAGS += $(HDF5_LIB)

# SAN=1 turns on ASan+UBSan for both compiling and linking.
ifeq ($(SAN),1)
ALL_CXXFLAGS += -fsanitize=address,undefined
LDFLAGS += -fsanitize=address,undefined
endif

LIB = libentropy_eos.a
HOST_SRCS = $(wildcard entropy_eos/host/*.cpp)
HOST_OBJS = $(HOST_SRCS:.cpp=.o)

TEST_SRCS = $(wildcard tests/test_*.cpp)
TEST_BINS = $(TEST_SRCS:.cpp=)

TOOL_SRCS = $(wildcard tools/*.cpp)
TOOL_BINS = $(TOOL_SRCS:.cpp=)

.PHONY: all lib test tools integration install clean san-fixture

all: lib

lib: $(LIB)

$(LIB): $(HOST_OBJS)
	$(AR) rcs $@ $^

%.o: %.cpp
	$(CXX) $(ALL_CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

# Each tests/test_*.cpp builds into its own self-contained binary (it defines
# DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN) linked against the static lib.
tests/test_%: tests/test_%.cpp $(LIB)
	$(CXX) $(ALL_CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

test: $(TEST_BINS)
	@set -e; for t in $(TEST_BINS); do echo "== $$t =="; ./$$t; done

# Each tools/*.cpp is a thin main() over the static lib (CODE.md: "no physics
# or table logic in the tools"), built the same way as a test binary.
tools/%: tools/%.cpp $(LIB)
	$(CXX) $(ALL_CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LIB) $(LDFLAGS)

tools: $(TOOL_BINS)

integration: tools
	./tests/integration.sh

# Regenerates the small (~20 MB) cropped LS220 fixture used by the LS220
# real-table tests under SAN=1 (see tests/test_scale.hpp's eeos_san_table()):
# a real-data box that still contains LS220's genuine Inf points in
# cs2/gamma (irho 69-83, jT 9-17, kYe 18) and keeps the full temperature
# axis, so the in-test repair paths still exercise genuine pathologies. Not
# run automatically by `test`/`integration` -- run by hand (or whenever the
# fixture needs regenerating) since it needs the full LS220 table under
# tables/ (see tables/README.md), which CI never has.
san-fixture: tools
	./tools/eos_crop tables/LS220_234r_136t_50y_analmu_20091212_SVNr26.h5 \
		tables/LS220_san_crop.h5 --irho 60 120 --jt 0 135 --kye 10 25

# --- M4 GPU device test (opt-in; `make`/`make test` never build or run this,
# see eos-device-interface.md S6) ------------------------------------------
# No autodetection: invoke explicitly, e.g. on a CUDA machine
#   make tests/test_device_cuda NVCC=/usr/local/cuda/bin/nvcc GPU_ARCH=sm_90
# and run under the scheduler (srun -p h200q --gpus=1 ./tests/test_device_cuda).
# The documented manual compile line (no Makefile at all) must always work:
#   nvcc -O2 -std=c++17 -arch=sm_90 --expt-relaxed-constexpr -I. \
#     tests/test_device_cuda.cu entropy_eos/host/table.cpp \
#     entropy_eos/host/synthetic.cpp entropy_eos/host/bspline_fit.cpp \
#     entropy_eos/host/adapter_build.cpp -o tests/test_device_cuda
NVCC ?= nvcc
GPU_ARCH ?= sm_90
NVCCFLAGS ?= -O2 -g -std=c++17 -arch=$(GPU_ARCH) --expt-relaxed-constexpr

# The HDF5-free host closure the synthetic-table GPU test needs (synthetic
# table -> built adapter), compiled directly so the target neither builds nor
# links $(LIB) and its -lhdf5. (No OpenMP flags: the pragmas in these TUs are
# _OPENMP-guarded, and the one-off serial fit costs seconds.)
DEVICE_TEST_HOST_SRCS = entropy_eos/host/table.cpp entropy_eos/host/synthetic.cpp \
	entropy_eos/host/bspline_fit.cpp entropy_eos/host/adapter_build.cpp

tests/test_device_cuda: tests/test_device_cuda.cu $(DEVICE_TEST_HOST_SRCS)
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $^

.PHONY: gpu-test
gpu-test: tests/test_device_cuda
	./tests/test_device_cuda

# Real-table variant: adds --table support (read + repair + build a
# stellarcollapse file) behind -DEEOS_GPU_TEST_HDF5; needs the three I/O and
# repair TUs plus HDF5 (same HDF5_INC/HDF5_LIB variables as the host build).
# E.g. on symmetry (HDF5 in /usr, Ubuntu split paths):
#   make tests/test_device_cuda_hdf5 NVCC=/usr/local/cuda/bin/nvcc \
#     HDF5_INC=-I/usr/include/hdf5/serial \
#     HDF5_LIB="-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5"
DEVICE_TEST_HDF5_SRCS = $(DEVICE_TEST_HOST_SRCS) entropy_eos/host/io_stellarcollapse.cpp \
	entropy_eos/host/check.cpp entropy_eos/host/repair.cpp

tests/test_device_cuda_hdf5: tests/test_device_cuda.cu $(DEVICE_TEST_HDF5_SRCS)
	$(NVCC) $(NVCCFLAGS) -DEEOS_GPU_TEST_HDF5 -I. $(HDF5_INC) -o $@ $^ $(HDF5_LIB)

install: lib
	mkdir -p $(PREFIX)/lib $(PREFIX)/include/entropy_eos
	(cd entropy_eos && find . -name '*.hpp' -print0 | tar --null -T - -cf -) | \
		(cd $(PREFIX)/include/entropy_eos && tar -xf -)
	cp $(LIB) $(PREFIX)/lib/

clean:
	rm -f $(HOST_OBJS) $(LIB) $(TEST_BINS) $(TOOL_BINS) tests/test_device_cuda
