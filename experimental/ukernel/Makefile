# Compiler and environment
CUDA_PATH      ?= /usr/local/cuda
override CXX   := /usr/bin/g++
NVCC           := $(CUDA_PATH)/bin/nvcc
CONDA_LIB_HOME ?= /usr/lib
DEBUG          ?= 0

# skip ThunderKittens now
# TK_ROOT := $(abspath ../../thirdparty/ThunderKittens)

# GPU architecture
# Default test builds target common datacenter GPUs used in this repo.
DEFAULT_SMS := 80 86 89
ifeq ($(origin SM), undefined)
NVCC_ARCH_FLAGS := $(foreach sm,$(DEFAULT_SMS),-gencode arch=compute_$(sm),code=sm_$(sm))
ENABLE_TMA := 0
SUBMAKE_SM :=
else
SM_NUM := $(shell printf '%s\n' "$(SM)" | sed 's/[^0-9].*//')
NVCC_ARCH_FLAGS := -arch=sm_$(SM)
ENABLE_TMA := $(shell if [ -n "$(SM_NUM)" ] && [ "$(SM_NUM)" -ge 90 ]; then echo 1; else echo 0; fi)
SUBMAKE_SM := SM=$(SM)
endif

# Base compilation flags
ifeq ($(DEBUG),1)
CXX_OPT_FLAGS  := -O0 -g
NVCC_OPT_FLAGS := -O0 -g
else
CXX_OPT_FLAGS  := -O3
NVCC_OPT_FLAGS := -O3
endif

CXXFLAGS   := $(CXX_OPT_FLAGS) -std=c++17 -Wall -Wno-unused-function -Wno-sign-compare -Wno-reorder -Wno-unused-variable -Wno-unused-label -Wno-unused-but-set-variable -Wno-stringop-overread -Wno-narrowing -pthread -fPIC \
              -I$(CONDA_LIB_HOME)/../include \
              -DUKERNEL_ENABLE_TMA=$(ENABLE_TMA)

NVCCFLAGS  := $(NVCC_OPT_FLAGS) -std=c++20 \
              --compiler-options "-fPIC -Wall -Wno-sign-compare -Wno-reorder -Wno-unused-variable -Wno-unused-label -pthread -I$(CONDA_LIB_HOME)/../include" \
              --expt-extended-lambda \
              --expt-relaxed-constexpr \
              -DKITTENS_HOPPER \
              -DUKERNEL_ENABLE_TMA=$(ENABLE_TMA) \
              $(NVCC_ARCH_FLAGS) \
              -ccbin $(CXX)
#             -I$(TK_ROOT)/include \

LDFLAGS    := -L$(CONDA_LIB_HOME) \
              -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 \
              -lcudart -lcuda \
              -lgflags -lz -libverbs -lnl-3 -lnl-route-3 -lpthread 

GDRCOPY_ROOT := $(abspath ../../thirdparty/gdrcopy)
GDRCOPY_LIBDIR := $(GDRCOPY_ROOT)/src

# Local includes must come first to avoid conflicts with UCCL headers
LOCAL_INCLUDES := -Iinclude -Isrc/transport -Isrc/device -Isrc/device/fifo -Isrc/ccl

# System and external library includes
INCLUDES := -I$(CUDA_PATH)/include -I/usr/include -I../../include -I$(GDRCOPY_ROOT)/include

# Redis support: when USE_REDIS_OOB=1
ifeq ($(USE_REDIS_OOB),1)
PKG_CONFIG := pkg-config
REDIS_CFLAGS := $(shell $(PKG_CONFIG) --cflags hiredis)
REDIS_LIBS   := $(shell $(PKG_CONFIG) --libs hiredis)

CXXFLAGS   += $(REDIS_CFLAGS) -DUSE_REDIS_OOB
NVCCFLAGS  += -DUSE_REDIS_OOB --compiler-options "$(REDIS_CFLAGS)"
LDFLAGS    += $(REDIS_LIBS)
endif

# UCCL collective support
UCCL_ROOT := $(abspath ../../)
UCCL_RDMA_ROOT := $(UCCL_ROOT)/collective/rdma
UCCL_RDMA_STATIC := $(UCCL_RDMA_ROOT)/librdma.a
UCCL_CFLAGS := -I$(UCCL_ROOT) -I$(UCCL_ROOT)/collective/rdma -I$(UCCL_ROOT)/include
UCCL_LDFLAGS := $(UCCL_RDMA_STATIC) -libverbs -lrdmacm

CXXFLAGS += $(UCCL_CFLAGS)
NVCCFLAGS += --compiler-options "$(UCCL_CFLAGS)"
LDFLAGS += $(UCCL_LDFLAGS)

# Source files - exclude module-local tests and device-only helper trees that
# are built by their module Makefiles.
SRC_CPP := $(shell find src -name '*.cc' | grep -v 'benchmarks' | grep -v 'src/device/' | grep -v 'src/transport/test/' | grep -v 'src/ccl/test/' | sort)
SRC_CU  := $(shell find src -name '*.cu' | grep -v 'benchmarks')

OBJ_CC   := $(SRC_CPP:.cc=.o)
OBJ_CU   := $(SRC_CU:.cu=.o)

# Benchmark files
BENCH_TRANSPORT_SRC := benchmarks/bench_transport.cc
BENCH_TRANSPORT_OBJ := $(BENCH_TRANSPORT_SRC:.cc=.o)
BENCH_TRANSPORT_BIN := bench_transport
BENCH_GDRCOPY_SRC := benchmarks/bench_gdrcopy.cc
BENCH_GDRCOPY_OBJ := $(BENCH_GDRCOPY_SRC:.cc=.o)
BENCH_GDRCOPY_BIN := benchmarks/bench_gdrcopy
BENCH_GDRCOPY_CXXFLAGS := $(CXXFLAGS) -DENABLE_CUDA
BENCH_GDRCOPY_INCLUDES := -I$(CUDA_PATH)/include -I/usr/include -I../../include -I$(GDRCOPY_ROOT)/include
BENCH_GDRCOPY_LIBS := -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 \
                      -L$(GDRCOPY_LIBDIR) -Wl,-rpath,$(GDRCOPY_LIBDIR) \
                      -lcudart -lcuda -lgdrapi
BENCH_BINS := $(BENCH_TRANSPORT_BIN) $(BENCH_GDRCOPY_BIN)

# Targets
STATIC_LIB  := libukernel.a
SHARED_LIB  := libukernel.so

# Build the core ukernel artifacts plus top-level benchmarks.
all: $(STATIC_LIB) $(SHARED_LIB) bench

# Compilation rules - local includes must come before system/UCCL includes to avoid header conflicts
%.o: %.cc
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

%.o: %.cpp
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

%.o: %.cu
	$(NVCC) $(LOCAL_INCLUDES) $(NVCCFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

ensure-uccl-rdma:
	@if [ ! -f "$(UCCL_RDMA_STATIC)" ]; then \
		echo "[ukernel] building collective/rdma..."; \
		$(MAKE) -C "$(UCCL_RDMA_ROOT)" build; \
	fi

ensure-gdrcopy:
	@if [ ! -f "$(GDRCOPY_ROOT)/include/gdrapi.h" ] || [ ! -f "$(GDRCOPY_ROOT)/Makefile" ]; then \
		echo "[ukernel] initializing submodules for gdrcopy..."; \
		git -C "$(UCCL_ROOT)" submodule update --init --recursive; \
	fi
	@if [ ! -f "$(GDRCOPY_ROOT)/include/gdrapi.h" ]; then \
		echo "[ukernel] missing $(GDRCOPY_ROOT)/include/gdrapi.h after submodule init"; \
		exit 1; \
	fi
	@if ! ls "$(GDRCOPY_LIBDIR)"/libgdrapi.so* >/dev/null 2>&1; then \
		echo "[ukernel] building gdrcopy..."; \
		$(MAKE) -C "$(GDRCOPY_ROOT)" all; \
	fi

# Static library
$(STATIC_LIB): $(OBJ_CC) $(OBJ_CU)
	ar rcs $@ $^

# Shared library
$(SHARED_LIB): $(OBJ_CC) $(OBJ_CU) | ensure-uccl-rdma
	$(CXX) -shared $(CXXFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

# Benchmark build rules
$(BENCH_TRANSPORT_OBJ): $(BENCH_TRANSPORT_SRC)
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(BENCH_TRANSPORT_BIN): $(BENCH_TRANSPORT_OBJ) $(STATIC_LIB) | ensure-uccl-rdma
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $(INCLUDES) $< $(STATIC_LIB) $(LDFLAGS) -o $@

$(BENCH_GDRCOPY_OBJ): $(BENCH_GDRCOPY_SRC)
	$(CXX) $(BENCH_GDRCOPY_CXXFLAGS) $(BENCH_GDRCOPY_INCLUDES) -MMD -MP -c $< -o $@

$(BENCH_GDRCOPY_BIN): $(BENCH_GDRCOPY_OBJ) | ensure-gdrcopy ensure-uccl-rdma
	$(CXX) $< -o $@ $(BENCH_GDRCOPY_LIBS)

bench: $(BENCH_BINS)

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
	      *.d src/*.d src/ccl/*.d src/transport/*.d src/transport/test/*.d benchmarks/*.d
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

.PHONY: all clean bench transport_bench device_test ccl_test transport_test transport_suite device_bench ensure-uccl-rdma ensure-gdrcopy

transport_bench: $(BENCH_TRANSPORT_BIN)
