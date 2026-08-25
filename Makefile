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

.PHONY: all lib test tools integration install clean

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

install: lib
	mkdir -p $(PREFIX)/lib $(PREFIX)/include/entropy_eos
	(cd entropy_eos && find . -name '*.hpp' -print0 | tar --null -T - -cf -) | \
		(cd $(PREFIX)/include/entropy_eos && tar -xf -)
	cp $(LIB) $(PREFIX)/lib/

clean:
	rm -f $(HOST_OBJS) $(LIB) $(TEST_BINS) $(TOOL_BINS)
