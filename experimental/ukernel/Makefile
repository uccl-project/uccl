# Compiler and environment
CUDA_PATH      ?= /usr/local/cuda
# Auto-detect CUDA if default path doesn't exist (e.g. /usr/local/cuda-13).
ifeq ($(wildcard $(CUDA_PATH)/bin/nvcc),)
  _CUDA_AUTO := $(firstword $(sort $(wildcard /usr/local/cuda-*)))
  ifneq ($(_CUDA_AUTO),)
    CUDA_PATH := $(_CUDA_AUTO)
  endif
endif
override CXX   := /usr/bin/g++
NVCC           := $(CUDA_PATH)/bin/nvcc
CONDA_LIB_HOME ?= /usr/lib
DEBUG          ?= 0

# skip ThunderKittens now
# TK_ROOT := $(abspath ../../thirdparty/ThunderKittens)

# GPU architecture + per-arch reduce-kernel defaults (ILP / TMA / smem).
include mk/arch_defaults.mk

# Base compilation flags
ifeq ($(DEBUG),1)
CXX_OPT_FLAGS  := -O0 -g
NVCC_OPT_FLAGS := -O0 -g
else
CXX_OPT_FLAGS  := -O3
NVCC_OPT_FLAGS := -O3
endif

CXXFLAGS   := $(CXX_OPT_FLAGS) -std=c++17 -Wall -Wno-unused-function -Wno-sign-compare -Wno-reorder -Wno-unused-variable -Wno-unused-label -Wno-unused-but-set-variable -Wno-stringop-overread -Wno-narrowing -pthread -fPIC \
              -I$(CONDA_LIB_HOME)/../include

NVCCFLAGS  := $(NVCC_OPT_FLAGS) -std=c++20 \
              --compiler-options "-fPIC -Wall -Wno-sign-compare -Wno-reorder -Wno-unused-variable -Wno-unused-label -pthread -I$(CONDA_LIB_HOME)/../include" \
              --expt-extended-lambda \
              --expt-relaxed-constexpr \
              -DKITTENS_HOPPER \
              $(NVCC_ARCH_FLAGS) \
              -ccbin $(CXX)
# Relocatable device code: the persistent kernels call the per-dtype
# reduce dispatch across TUs; link the .so with nvcc to run nvlink.
NVCCFLAGS += -rdc=true

# Rebuild when compile flags change (see mk/arch_defaults.mk).
$(shell printf '%s\n' '$(NVCCFLAGS) $(CXXFLAGS)' > $(BUILD_FLAGS_STAMP).tmp; \
        cmp -s $(BUILD_FLAGS_STAMP).tmp $(BUILD_FLAGS_STAMP) || \
        mv -f $(BUILD_FLAGS_STAMP).tmp $(BUILD_FLAGS_STAMP); \
        rm -f $(BUILD_FLAGS_STAMP).tmp)
#             -I$(TK_ROOT)/include \

LDFLAGS    := -L$(CONDA_LIB_HOME) \
              -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 \
              -Wl,-rpath,/usr/lib/$(GCC_MACHINE) \
              -Wl,-rpath-link,/lib/$(GCC_MACHINE):/usr/lib/$(GCC_MACHINE) \
              -lcudart -lcuda \
              -lz -libverbs -lnl-3 -lnl-route-3 -lpthread 

# Local includes must come first to avoid conflicts with UCCL headers
LOCAL_INCLUDES := -Iinclude -Isrc/transport -Isrc/device -Isrc/device/fifo -Isrc/ccl

# System and external library includes
GCC_MACHINE := $(shell $(CXX) -dumpmachine 2>/dev/null || echo x86_64-linux-gnu)
GDRCOPY_INCLUDEDIR ?= /usr/include
INCLUDES := -I$(CUDA_PATH)/include -I$(GDRCOPY_INCLUDEDIR) -I/usr/include \
            -I/usr/local/include -I/usr/include/$(GCC_MACHINE) \
            -I../../include
GDRCOPY_LIBDIR ?=
ifneq ($(strip $(GDRCOPY_LIBDIR)),)
GDRCOPY_LIBDIR_FLAGS := -L$(GDRCOPY_LIBDIR) -Wl,-rpath,$(GDRCOPY_LIBDIR)
else
GDRCOPY_LIBDIR_FLAGS :=
endif

# Redis support: when USE_REDIS_OOB=1
ifeq ($(USE_REDIS_OOB),1)
PKG_CONFIG := pkg-config
REDIS_CFLAGS := $(shell $(PKG_CONFIG) --cflags hiredis)
REDIS_LIBS   := $(shell $(PKG_CONFIG) --libs hiredis)

CXXFLAGS   += $(REDIS_CFLAGS) -DUSE_REDIS_OOB
NVCCFLAGS  += -DUSE_REDIS_OOB --compiler-options "$(REDIS_CFLAGS)"
LDFLAGS    += $(REDIS_LIBS)
endif

# Source files - exclude module-local tests and device-only helper trees that
# are built by their module Makefiles.
SRC_CPP := $(shell find src -name '*.cc' | grep -v 'benchmarks' | grep -v 'src/device/' | grep -v 'src/transport/test/' | grep -v 'src/ccl/test/' | grep -v 'uccl_adapter' | sort)
SRC_CU  := $(shell find src -name '*.cu' | grep -v 'benchmarks')

OBJ_CC   := $(SRC_CPP:.cc=.o)
OBJ_CU   := $(SRC_CU:.cu=.o)

# Benchmark files
BENCH_TRANSPORT_SRC := bench/bench_transport.cc
BENCH_TRANSPORT_OBJ := $(BENCH_TRANSPORT_SRC:.cc=.o)
BENCH_TRANSPORT_BIN := bench_transport
BENCH_GDRCOPY_SRC := bench/bench_gdrcopy.cc
BENCH_GDRCOPY_OBJ := $(BENCH_GDRCOPY_SRC:.cc=.o)
BENCH_GDRCOPY_BIN := bench/bench_gdrcopy
BENCH_GDRCOPY_CXXFLAGS := $(CXXFLAGS) -DENABLE_CUDA
BENCH_GDRCOPY_INCLUDES := -I$(CUDA_PATH)/include -I$(GDRCOPY_INCLUDEDIR) -I/usr/include -I/usr/include/$(GCC_MACHINE) -I/usr/local/include -I../../include
BENCH_GDRCOPY_LIBS := -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 \
                      -Wl,-rpath,/usr/lib/$(GCC_MACHINE) \
                      $(GDRCOPY_LIBDIR_FLAGS) \
                      -lcudart -lcuda -lgdrapi
BENCH_BINS := $(BENCH_TRANSPORT_BIN) $(BENCH_GDRCOPY_BIN)

# Targets
STATIC_LIB  := libukernel.a
SHARED_LIB  := libukernel.so

# Build the core ukernel artifacts plus top-level benchmarks.
all: $(STATIC_LIB) $(SHARED_LIB) bench

# Compilation rules - local includes must come before system/UCCL includes to avoid header conflicts
%.o: %.cc $(BUILD_FLAGS_STAMP)
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

%.o: %.cpp $(BUILD_FLAGS_STAMP)
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

%.o: %.cu $(BUILD_FLAGS_STAMP)
	$(NVCC) $(LOCAL_INCLUDES) $(NVCCFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(STATIC_LIB): $(OBJ_CC) $(OBJ_CU)
	ar rcs $@ $^

$(SHARED_LIB): $(OBJ_CC) $(OBJ_CU)
	$(NVCC) -shared $(NVCCFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

$(BENCH_TRANSPORT_BIN): $(BENCH_TRANSPORT_OBJ) $(STATIC_LIB)
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $(INCLUDES) $< $(STATIC_LIB) $(LDFLAGS) -o $@

$(BENCH_GDRCOPY_OBJ): $(BENCH_GDRCOPY_SRC)
	$(CXX) $(BENCH_GDRCOPY_CXXFLAGS) $(BENCH_GDRCOPY_INCLUDES) -MMD -MP -c $< -o $@

$(BENCH_GDRCOPY_BIN): $(BENCH_GDRCOPY_OBJ)
	$(CXX) $< -o $@ $(BENCH_GDRCOPY_LIBS)

bench: $(BENCH_BINS)

# NCCL compatibility shared library (drop-in replacement via LD_PRELOAD).
# Installs to a standard include/lib layout.
# Override: make nccl NCCL_PREFIX=/opt/ukernel_nccl
NCCL_PREFIX ?= $(CURDIR)/build/nccl
nccl:
	$(MAKE) -C src/ccl nccl_dev NCCL_PREFIX=$(NCCL_PREFIX) $(SUBMAKE_SM)

device_test:
	$(MAKE) -C src/device test $(SUBMAKE_SM)

ccl_test:
	$(MAKE) -C src/ccl test $(SUBMAKE_SM)

transport_test:
	$(MAKE) -C src/transport test

transport_suite:
	$(MAKE) -C src/transport suite

device_bench:
	$(MAKE) -C src/device bench $(SUBMAKE_SM)

# Clean
clean:
	rm -f $(OBJ_CC) $(OBJ_CU) $(BENCH_TRANSPORT_OBJ) $(BENCH_GDRCOPY_OBJ) \
	      $(STATIC_LIB) $(SHARED_LIB) $(BENCH_BINS) \
	      *.d src/*.d src/ccl/*.d src/transport/*.d src/transport/test/*.d bench/*.d
	find src -type f \( -name '*.o' -o -name '*.d' -o -name '.DS_Store' \) -delete
	rm -f src/ccl/test_components
	$(MAKE) -C src/ccl clean
	$(MAKE) -C src/device clean
	$(MAKE) -C src/transport clean

# Include dependencies
-include $(OBJ_CC:.o=.d) $(OBJ_CU:.o=.d) $(BENCH_TRANSPORT_OBJ:.o=.d) $(BENCH_GDRCOPY_OBJ:.o=.d)

# [TODO: Yihan] Build only TK GEMM operator if a dedicated binding returns later.
# tk_gemm: src/device/persistent_kernel_ops.o
# 	@echo "TK GEMM operator compiled: src/device/persistent_kernel_ops.o"

# src/device/persistent_kernel_ops.o: src/device/persistent_kernel_ops.cu
# 	$(NVCC) $(NVCCFLAGS) -Iinclude -Isrc/transport -Isrc/device -Isrc/device/fifo -MMD -MP -c $< -o $@

.PHONY: all clean bench transport_bench device_test ccl_test transport_test transport_suite device_bench

transport_bench: $(BENCH_TRANSPORT_BIN)
