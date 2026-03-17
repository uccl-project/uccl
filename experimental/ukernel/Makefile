# Compiler and environment
CUDA_PATH      ?= /usr/local/cuda
override CXX   := /usr/bin/g++
NVCC           := $(CUDA_PATH)/bin/nvcc
CONDA_LIB_HOME ?= /usr/lib

# skip ThunderKittens now
# TK_ROOT := $(abspath ../../thirdparty/ThunderKittens)

# GPU architecture - use sm_90a for TMA/wgmma support
SM ?= 90a
SM_NUM := $(shell printf '%s\n' "$(SM)" | sed 's/[^0-9].*//')
ENABLE_TMA := $(shell if [ -n "$(SM_NUM)" ] && [ "$(SM_NUM)" -ge 90 ]; then echo 1; else echo 0; fi)

# Base compilation flags
CXXFLAGS   := -O0 -std=c++17 -Wall -Wno-unused-function -Wno-sign-compare -Wno-reorder -Wno-unused-variable -Wno-unused-label -Wno-unused-but-set-variable -Wno-stringop-overread -Wno-narrowing -pthread -fPIC \
              -I$(CONDA_LIB_HOME)/../include \
              -DUKERNEL_ENABLE_TMA=$(ENABLE_TMA)

NVCCFLAGS  := -g -O0 -std=c++20 \
              --compiler-options "-fPIC -Wall -Wno-sign-compare -Wno-reorder -Wno-unused-variable -Wno-unused-label -pthread -I$(CONDA_LIB_HOME)/../include" \
              --expt-extended-lambda \
              --expt-relaxed-constexpr \
              -DKITTENS_HOPPER \
              -DUKERNEL_ENABLE_TMA=$(ENABLE_TMA) \
              -arch=sm_$(SM) \
              -ccbin $(CXX)
#             -I$(TK_ROOT)/include \

LDFLAGS    := -L$(CONDA_LIB_HOME) \
              -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 \
              -lcudart -lcuda \
              -lgflags -lz -libverbs -lnl-3 -lnl-route-3 -lpthread 

GDRCOPY_ROOT := $(abspath ../../thirdparty/gdrcopy)

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

# Source files - exclude benchmarks and most device directory pieces.
# Keep CCL selector and device PK selector because they are shared helpers.
SRC_CPP := $(shell find src -name '*.cc' | grep -v 'benchmarks' | grep -v 'src/device/' | grep -v 'src/transport/test/' | sort)
SRC_CPP += src/device/pk_selector.cc
SRC_CU  := $(shell find src -name '*.cu' | grep -v 'benchmarks')
TEST_DIR := src/ccl/test
SRC_TEST := $(TEST_DIR)/test_main.cc \
            $(TEST_DIR)/test_ccl_plan.cc \
            $(TEST_DIR)/test_ccl_executor.cc \
            $(TEST_DIR)/test_ccl_rdma_allgather.cc

OBJ_CC   := $(SRC_CPP:.cc=.o)
OBJ_CU   := $(SRC_CU:.cu=.o)
OBJ_TEST := $(SRC_TEST:.cc=.o)

# Benchmark files
BENCH_SRC := benchmarks/bench_transport.cc
BENCH_OBJ := $(BENCH_SRC:.cc=.o)
BENCH_BIN := bench_transport

# Targets
STATIC_LIB  := libukernel.a
SHARED_LIB  := libukernel.so
TARGET_TEST := test_main

# Build both static, shared, and test by default
all: $(STATIC_LIB) $(SHARED_LIB) $(TARGET_TEST)

# Compilation rules - local includes must come before system/UCCL includes to avoid header conflicts
%.o: %.cc
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

%.o: %.cu
	$(NVCC) $(LOCAL_INCLUDES) $(NVCCFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

ensure-uccl-rdma:
	@if [ ! -f "$(UCCL_RDMA_STATIC)" ]; then \
		echo "[ukernel] building collective/rdma..."; \
		$(MAKE) -C "$(UCCL_RDMA_ROOT)" build; \
	fi

# Static library
$(STATIC_LIB): $(OBJ_CC)
	ar rcs $@ $^

# Shared library
$(SHARED_LIB): $(OBJ_CC) $(OBJ_CU) | ensure-uccl-rdma
	$(CXX) -shared $(CXXFLAGS) $(INCLUDES) $^ $(LDFLAGS) -o $@

# Test executable
$(TARGET_TEST): $(OBJ_TEST) $(STATIC_LIB) | ensure-uccl-rdma
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJ_TEST) $(STATIC_LIB) $(LDFLAGS) -o $@

# Run tests
run_test: $(TARGET_TEST)
	./$(TARGET_TEST)

# Benchmark build rules
$(BENCH_OBJ): $(BENCH_SRC)
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(BENCH_BIN): $(BENCH_OBJ) $(STATIC_LIB) | ensure-uccl-rdma
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $(INCLUDES) $< $(STATIC_LIB) $(LDFLAGS) -o $@

bench: $(BENCH_BIN)

device_test:
	$(MAKE) -C src/device test SM=$(SM)

ccl_test:
	$(MAKE) -C src/ccl/test test SM=$(SM)

transport_test:
	$(MAKE) -C src/transport test

transport_suite:
	$(MAKE) -C src/transport suite

device_bench:
	$(MAKE) -C src/device bench SM=$(SM)

# Clean
clean:
	rm -f $(OBJ_CC) $(OBJ_CU) $(OBJ_TEST) $(BENCH_OBJ) \
	      $(TARGET_TEST) $(STATIC_LIB) $(SHARED_LIB) $(BENCH_BIN) \
	      *.d src/*.d src/ccl/test/*.d src/transport/*.d src/transport/test/*.d benchmarks/*.d

# Include dependencies
-include $(OBJ_CC:.o=.d) $(OBJ_CU:.o=.d) $(OBJ_TEST:.o=.d)

# [TODO: Yihan] Build only TK GEMM operator if a dedicated binding returns later.
# tk_gemm: src/device/persistent_kernel_ops.o
# 	@echo "TK GEMM operator compiled: src/device/persistent_kernel_ops.o"

# src/device/persistent_kernel_ops.o: src/device/persistent_kernel_ops.cu
# 	$(NVCC) $(NVCCFLAGS) -Iinclude -Isrc/transport -Isrc/device -Isrc/device/fifo -MMD -MP -c $< -o $@

.PHONY: all clean run_test bench device_test ccl_test transport_test transport_suite device_bench ensure-uccl-rdma
