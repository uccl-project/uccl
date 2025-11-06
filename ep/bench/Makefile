# ===============================================================================
# Unified Makefile for UCCL Benchmarks and EP Runtime
# Supports both GH200 (Grace-Hopper) and H200 with EFA environments
#
# Usage:
#   make all                - Build all tests and extensions
#   make benchmarks         - Build all benchmark tests
#   make ep                 - Build EP runtime and Python extension
#   make fc                 - Build flat combining test
#   make lock               - Build lock throughput test
#   make clean              - Remove all build artifacts
#   make run-fc             - Run flat combining test
#   make run-lock           - Run lock throughput test
#   make install            - Install Python extension
# ===============================================================================

# ===============================================================================
# Environment Detection and Configuration
# ===============================================================================

# CUDA configuration
CUDA_PATH ?= /usr/local/cuda
override CXX := /usr/bin/g++
NVCC := $(CUDA_PATH)/bin/nvcc

# Detect architecture
ARCH := $(shell uname -m)
GPU_NAME := $(shell nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 2>/dev/null || echo "Unknown")

# Auto-detect GPU architecture if not specified
GPU_ARCH ?= auto
SM ?= 90
ifeq ($(GPU_ARCH), auto)
    # For H200, use sm_90
    GPU_NAME_CHECK := $(shell nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 2>/dev/null)
    ifneq (,$(findstring H200,$(GPU_NAME_CHECK)))
        SM := 90
    else ifneq (,$(findstring H100,$(GPU_NAME_CHECK)))
        SM := 90
    else ifneq (,$(findstring GH200,$(GPU_NAME_CHECK)))
        SM := 90
    else
        # Fall back to detection or default to 90
        SM := $(shell $(NVCC) --list-gpu-code 2>/dev/null | grep 'sm_' | tail -1 | sed 's/.*sm_//' || echo "90")
    endif
endif

# Build configuration
BUILD_TYPE ?= release
BUILD_MODE ?= auto

# ===============================================================================
# Platform Detection
# ===============================================================================

# Detect EFA (Elastic Fabric Adapter) for H200 environment
EFA_HOME ?= /opt/amazon/efa
ifeq ($(wildcard $(EFA_HOME)),)
    $(info EFA not detected, building without EFA support)
    EFA_CFLAGS :=
    EFA_LDFLAGS :=
    HAS_EFA := 0
else
    $(info EFA detected at $(EFA_HOME), building with EFA support)
    EFA_CFLAGS := -DEFA -I$(EFA_HOME)/include
    EFA_LDFLAGS := -L$(EFA_HOME)/lib -lefa
    HAS_EFA := 1
endif

# Detect GH200 (Grace-Hopper)
CPU_IS_ARM64 := 0
GPU_IS_HOPPER := 0
ifeq ($(ARCH),aarch64)
    CPU_IS_ARM64 := 1
endif
ifneq (,$(findstring GH200,$(GPU_NAME)))
    GPU_IS_HOPPER := 1
endif
ifneq (,$(findstring H100,$(GPU_NAME)))
    GPU_IS_HOPPER := 1
endif
ifneq (,$(findstring H200,$(GPU_NAME)))
    GPU_IS_HOPPER := 1
endif

ifeq ($(and $(CPU_IS_ARM64),$(GPU_IS_HOPPER)),1)
    GH_CFLAGS := -DUSE_GRACE_HOPPER
    PLATFORM := gh200
else ifeq ($(HAS_EFA),1)
    $(info H200/EFA environment detected)
    GH_CFLAGS :=
    PLATFORM := h200_efa
else
    $(info Standard CUDA environment detected)
    GH_CFLAGS :=
    PLATFORM := standard
endif

# ===============================================================================
# Python Configuration (for EP runtime)
# ===============================================================================

PYTHON ?= python3
PYBIND11_AVAILABLE := $(shell $(PYTHON) -c "import pybind11" 2>/dev/null && echo "yes" || echo "no")

ifeq ($(PYBIND11_AVAILABLE),yes)
    PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
    EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig as s; print(s.get_config_var('EXT_SUFFIX') or '.so')")

    # PyTorch configuration
    TORCH_AVAILABLE := $(shell $(PYTHON) -c "import torch" 2>/dev/null && echo "yes" || echo "no")
    ifeq ($(TORCH_AVAILABLE),yes)
        TORCH_INCLUDE := $(shell $(PYTHON) -c "import torch, pathlib; p=pathlib.Path(torch.__file__).parent; print(p/'include')")
        TORCH_INCLUDE_API := $(shell $(PYTHON) -c "import torch, pathlib; p=pathlib.Path(torch.__file__).parent; print(p/'include/torch/csrc/api/include')")
        TORCH_LIBDIR := $(shell $(PYTHON) -c "import torch, pathlib; p=pathlib.Path(torch.__file__).parent; print(p/'lib')")
        TORCH_ABI := $(shell $(PYTHON) -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
        TORCH_INCS := -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE_API)
        PYTHON_SITE_PACKAGES := $(shell $(PYTHON) -c "import site; print(site.getsitepackages()[0])")
        INSTALL_DIR := $(PYTHON_SITE_PACKAGES)/uccl
    endif
endif

# ===============================================================================
# Compiler Flags
# ===============================================================================

# Base C++ flags
CXXFLAGS := -std=c++17 -Wall -pthread -fPIC -fvisibility=hidden
CXXFLAGS += $(EFA_CFLAGS) $(GH_CFLAGS)

# Base NVCC flags
NVCC_FLAGS := -std=c++17
NVCC_FLAGS += -gencode arch=compute_$(SM),code=sm_$(SM)
NVCC_FLAGS += -Xcompiler "-Wall -pthread -fPIC -fvisibility=hidden"
NVCC_FLAGS += --expt-relaxed-constexpr
NVCC_FLAGS += -ccbin /usr/bin/g++
NVCC_FLAGS += $(EFA_CFLAGS) $(GH_CFLAGS)

# Release vs Debug flags
ifeq ($(BUILD_TYPE), debug)
    CXXFLAGS += -g -O0 -DDEBUG
    NVCC_FLAGS += -g -G -O0 -DDEBUG
    BUILD_SUFFIX := _debug
    $(info Building DEBUG version for $(PLATFORM))
else
    CXXFLAGS += -O3 -DNDEBUG
    NVCC_FLAGS += -O3 -DNDEBUG --use_fast_math
    BUILD_SUFFIX :=
    $(info Building RELEASE version for $(PLATFORM) with GPU arch sm_$(SM))
endif

# Include paths
INCLUDES := -I../include -I../src -I../../include
INCLUDES += -Iinclude -I$(CUDA_PATH)/include -I/usr/include
INCLUDES += -I../thirdparty/DeepEP/csrc
INCLUDES += $(EFA_CFLAGS) $(GH_CFLAGS)

# Libraries
LIBS := -lpthread -lcuda -lcudart
LIBS += -L$(CUDA_PATH)/lib64 -Xlinker -rpath -Xlinker $(CUDA_PATH)/lib64

# Add EFA libs if available
ifeq ($(HAS_EFA),1)
    LIBS += $(EFA_LDFLAGS) -libverbs -lnl-3 -lnl-route-3
endif

# Add glog if available
GLOG_AVAILABLE := $(shell pkg-config --exists libglog 2>/dev/null && echo "yes" || echo "no")
ifeq ($(GLOG_AVAILABLE),yes)
    LIBS += -lglog
endif

# ===============================================================================
# Target Definitions
# ===============================================================================

# Benchmark test targets
FC_TARGET := test_fc_throughput$(BUILD_SUFFIX)
LOCK_TARGET := test_lock_throughput$(BUILD_SUFFIX)
MULTI_RING_TARGET := test_multi_ring_throughput$(BUILD_SUFFIX)
FC_LATENCY_TARGET := test_fc_latency$(BUILD_SUFFIX)
FC_LOAD_LATENCY_TARGET := test_fc_load_latency$(BUILD_SUFFIX)
CAS_TARGET := test_cas_throughput$(BUILD_SUFFIX)

# RDMA test targets (require EFA)
ifeq ($(HAS_EFA),1)
    ALLTOALL_RAIL_SHARED_QP_TARGET := test_alltoall_rail_shared_qp$(BUILD_SUFFIX)
    ALLTOALL_NONRAIL_SHARED_QP_TARGET := test_alltoall_nonrail_shared_qp$(BUILD_SUFFIX)
    DISPATCH_CONNECTIONLESS_TARGET := test_dispatch_connectionless$(BUILD_SUFFIX)
    DISPATCH_MULTIQP_TARGET := test_dispatch_multiqp$(BUILD_SUFFIX)
endif

# EP runtime targets (if EFA is available)
ifeq ($(HAS_EFA),1)
    TARGET_LOCAL := benchmark_local$(BUILD_SUFFIX)
    TARGET_REMOTE := benchmark_remote$(BUILD_SUFFIX)
    TARGET_DUAL := benchmark_dual$(BUILD_SUFFIX)
    EP_EXT := ep$(EXT_SUFFIX)
endif

# Header dependencies
HEADERS := $(wildcard ../include/*.h ../include/*.cuh ../include/*.hpp)
HEADERS += $(wildcard include/*.h include/*.cuh include/*.hpp)

# ===============================================================================
# Source Files (EP Runtime)
# ===============================================================================

ifeq ($(HAS_EFA),1)
    SRC_CPP := ../src/proxy.cpp ../src/rdma.cpp ../src/common.cpp
    SRC_CPP += ../src/peer_copy_worker.cpp ../src/uccl_proxy.cpp
    SRC_CPP += ../src/uccl_bench.cpp ../src/peer_copy_manager.cpp ../src/fifo.cpp

    SRC_CU := ../src/bench_kernel.cu ../src/peer_copy.cu
    SRC_CU += ../src/internode_ll.cu ../src/internode.cu ../src/layout.cu
    SRC_CU += ../src/intranode.cu ../src/ep_runtime.cu

    SRC_LOCAL := benchmark_local.cu
    SRC_REMOTE := benchmark_remote.cu
    SRC_DUAL := benchmark_dual.cu

    OBJ_CPP := $(SRC_CPP:.cpp=.o)
    OBJ_CU := $(SRC_CU:.cu=.o)

    OBJ_LOCAL := $(OBJ_CPP) $(OBJ_CU) $(SRC_LOCAL:.cu=.o)
    OBJ_REMOTE := $(OBJ_CPP) $(OBJ_CU) $(SRC_REMOTE:.cu=.o)
    OBJ_DUAL := $(OBJ_CPP) $(OBJ_CU) $(SRC_DUAL:.cu=.o)

    ifeq ($(PYBIND11_AVAILABLE),yes)
        SRC_BIND := ../src/uccl_ep.cc
        OBJ_BIND := $(SRC_BIND:.cc=.o)
    endif
endif

# ===============================================================================
# Build Rules
# ===============================================================================

.DEFAULT_GOAL := all

# Main targets
ifeq ($(HAS_EFA),1)
    ALL_TARGETS := benchmarks ep-runtime
else
    ALL_TARGETS := benchmarks
endif

all: $(ALL_TARGETS)
	@echo "All targets built successfully for $(PLATFORM)"

# Benchmark targets
benchmarks: $(FC_TARGET) $(LOCK_TARGET) $(MULTI_RING_TARGET) $(CAS_TARGET)
	@echo "Benchmark tests built successfully"

# RDMA test targets (only if EFA is available)
ifeq ($(HAS_EFA),1)
rdma-tests: $(ALLTOALL_RAIL_SHARED_QP_TARGET) $(ALLTOALL_NONRAIL_SHARED_QP_TARGET) $(DISPATCH_CONNECTIONLESS_TARGET) $(DISPATCH_MULTIQP_TARGET)
	@echo "RDMA tests built successfully"
endif

# EP Runtime targets (only if EFA is available)
ifeq ($(HAS_EFA),1)
ep-runtime: $(TARGET_LOCAL) $(TARGET_REMOTE) $(TARGET_DUAL) $(EP_EXT)
	@echo "EP runtime built successfully"

ep: ep-runtime
endif

# ===============================================================================
# Compilation Rules
# ===============================================================================

# C++ compilation rule with dependency generation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(TORCH_INCS) $(INCLUDES) $(PYBIND11_INCLUDES) -MMD -MP -c $< -o $@

# CUDA compilation rule with dependency generation
%.o: %.cu
	$(NVCC) -arch=sm_$(SM) $(NVCC_FLAGS) $(TORCH_INCS) $(INCLUDES) -MMD -MP -c $< -o $@

# C++ for binding
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(TORCH_INCS) $(INCLUDES) $(PYBIND11_INCLUDES) -MMD -MP -c $< -o $@

# ===============================================================================
# Benchmark Test Build Rules
# ===============================================================================

# Flat Combining throughput test
fc: $(FC_TARGET)

$(FC_TARGET): test_fc_throughput.cu $(HEADERS)
	@echo "Building Flat Combining throughput test..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)
	@echo "Build complete: $@"

# Lock throughput test
lock: $(LOCK_TARGET)

$(LOCK_TARGET): test_lock_throughput.cu $(HEADERS)
	@echo "Building Lock throughput test..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)
	@echo "Build complete: $@"

# Multi Ring throughput test
multi-ring: $(MULTI_RING_TARGET)

$(MULTI_RING_TARGET): test_multi_ring_throughput.cu $(HEADERS)
	@echo "Building Multi Ring throughput test..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)
	@echo "Build complete: $@"

# Flat Combining latency test
fc-latency: $(FC_LATENCY_TARGET)

$(FC_LATENCY_TARGET): test_fc_latency.cu $(HEADERS)
	@echo "Building Flat Combining latency test..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)
	@echo "Build complete: $@"

# Flat Combining load-latency test
fc-load-latency: $(FC_LOAD_LATENCY_TARGET)

$(FC_LOAD_LATENCY_TARGET): test_fc_load_latency.cu $(HEADERS)
	@echo "Building Flat Combining load-latency test..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)
	@echo "Build complete: $@"

# CAS throughput test
cas: $(CAS_TARGET)

$(CAS_TARGET): test_cas_throughput.cu $(HEADERS)
	@echo "Building CAS throughput test..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)
	@echo "Build complete: $@"

# ===============================================================================
# RDMA Test Build Rules (only if EFA is available)
# ===============================================================================

ifeq ($(HAS_EFA),1)
# AllToAll Rail Shared QP test
alltoall-rail-shared-qp: $(ALLTOALL_RAIL_SHARED_QP_TARGET)

$(ALLTOALL_RAIL_SHARED_QP_TARGET): test_alltoall_rail_shared_qp.cpp $(HEADERS)
	@echo "Building AllToAll Rail Shared QP test..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS) -libverbs
	@echo "Build complete: $@"

# AllToAll NonRail Shared QP test
alltoall-nonrail-shared-qp: $(ALLTOALL_NONRAIL_SHARED_QP_TARGET)

$(ALLTOALL_NONRAIL_SHARED_QP_TARGET): test_alltoall_nonrail_shared_qp.cpp $(HEADERS)
	@echo "Building AllToAll NonRail Shared QP test..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS) -libverbs
	@echo "Build complete: $@"

# Dispatch Connectionless test (MoE pattern)
dispatch-connectionless: $(DISPATCH_CONNECTIONLESS_TARGET)

$(DISPATCH_CONNECTIONLESS_TARGET): test_dispatch_connectionless.cpp $(HEADERS)
	@echo "Building Dispatch Connectionless test (MoE pattern)..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS) -libverbs
	@echo "Build complete: $@"

# Dispatch Multi-QP test (DeepEP one QP per expert)
dispatch-multiqp: $(DISPATCH_MULTIQP_TARGET)

$(DISPATCH_MULTIQP_TARGET): test_dispatch_multiqp.cpp $(HEADERS)
	@echo "Building Dispatch Multi-QP test (DeepEP one QP per expert)..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS) -libverbs
	@echo "Build complete: $@"
endif

# ===============================================================================
# EP Runtime Build Rules (only if EFA is available)
# ===============================================================================

ifeq ($(HAS_EFA),1)
$(TARGET_LOCAL): $(OBJ_LOCAL)
	$(NVCC) -arch=sm_$(SM) $(NVCC_FLAGS) $(INCLUDES) $(OBJ_LOCAL) $(LIBS) -o $@

$(TARGET_REMOTE): $(OBJ_REMOTE)
	$(NVCC) -arch=sm_$(SM) $(NVCC_FLAGS) $(INCLUDES) $(OBJ_REMOTE) $(LIBS) -o $@

$(TARGET_DUAL): $(OBJ_DUAL)
	$(NVCC) -arch=sm_$(SM) $(NVCC_FLAGS) $(INCLUDES) $(OBJ_DUAL) $(LIBS) -o $@

ifeq ($(PYBIND11_AVAILABLE),yes)
ifeq ($(TORCH_AVAILABLE),yes)
$(EP_EXT): $(OBJ_CPP) $(OBJ_CU) $(OBJ_BIND)
	$(CXX) $(CXXFLAGS) -shared -fPIC \
	    -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI) \
	    $(INCLUDES) $(PYBIND11_INCLUDES) $^ \
	    -L$(TORCH_LIBDIR) -Wl,-rpath,$(TORCH_LIBDIR) \
	    -ltorch_python -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
	    $(LIBS) -o $@

py: $(EP_EXT)
	@echo "Python extension built: $(EP_EXT)"

install: $(EP_EXT)
	@mkdir -p $(INSTALL_DIR)
	@cp $(EP_EXT) $(INSTALL_DIR)/
	@echo "Installation complete. Module installed to: $(INSTALL_DIR)/$(EP_EXT)"
else
$(EP_EXT):
	@echo "PyTorch not found. Skipping Python extension build."

py:
	@echo "PyTorch not found. Cannot build Python extension."

install:
	@echo "PyTorch not found. Cannot install Python extension."
endif
else
$(EP_EXT):
	@echo "pybind11 not found. Skipping Python extension build."

py:
	@echo "pybind11 not found. Cannot build Python extension."

install:
	@echo "pybind11 not found. Cannot install Python extension."
endif
endif

# ===============================================================================
# Run Rules
# ===============================================================================

# Run flat combining test
run-fc: $(FC_TARGET)
	@echo "Running Flat Combining throughput test..."
	@echo "================================="
	./$(FC_TARGET)

# Run lock throughput test
run-lock: $(LOCK_TARGET)
	@echo "Running Lock throughput test..."
	@echo "================================="
	./$(LOCK_TARGET)

# Run multi ring throughput test
run-multi-ring: $(MULTI_RING_TARGET)
	@echo "Running Multi Ring throughput test..."
	@echo "================================="
	./$(MULTI_RING_TARGET)

# Run flat combining latency test
run-fc-latency: $(FC_LATENCY_TARGET)
	@echo "Running Flat Combining latency test..."
	@echo "====================================="
	./$(FC_LATENCY_TARGET)

# Run flat combining load-latency test
run-fc-load-latency: $(FC_LOAD_LATENCY_TARGET)
	@echo "Running Flat Combining load-latency test..."
	@echo "=========================================="
	./$(FC_LOAD_LATENCY_TARGET)

# Run CAS throughput test
run-cas: $(CAS_TARGET)
	@echo "Running CAS throughput test..."
	@echo "=============================="
	./$(CAS_TARGET)

# Run EP benchmarks (if available)
ifeq ($(HAS_EFA),1)
run-ep-local: $(TARGET_LOCAL)
	@echo "Running EP local benchmark..."
	@echo "============================="
	./$(TARGET_LOCAL)

run-ep-remote: $(TARGET_REMOTE)
	@echo "Running EP remote benchmark..."
	@echo "=============================="
	./$(TARGET_REMOTE)

run-ep-dual: $(TARGET_DUAL)
	@echo "Running EP dual benchmark..."
	@echo "============================"
	./$(TARGET_DUAL)

run-ep: run-ep-local run-ep-remote run-ep-dual
endif

# Quick test - run main benchmarks with minimal config
quick: $(FC_TARGET) $(LOCK_TARGET)
	@echo "Quick test - Flat Combining:"
	./$(FC_TARGET) | head -20
	@echo ""
	@echo "Quick test - Lock Throughput:"
	./$(LOCK_TARGET) | head -30

# Benchmark suite
benchmark: benchmarks
	@echo "Running complete benchmark suite..."
	@echo "==================================="
	@echo "\n--- Platform: $(PLATFORM) ---"
	@echo "\n--- Flat Combining Throughput ---"
	./$(FC_TARGET)
	@echo "\n--- Lock Throughput ---"
	./$(LOCK_TARGET)
	@echo "\n--- Multi Ring Throughput ---"
	./$(MULTI_RING_TARGET)
ifeq ($(HAS_EFA),1)
	@echo "\n--- EP Local Benchmark ---"
	./$(TARGET_LOCAL)
	@echo "\n--- EP Remote Benchmark ---"
	./$(TARGET_REMOTE)
endif

# ===============================================================================
# Analysis and Profiling
# ===============================================================================

# Memory check
memcheck-fc: $(FC_TARGET)
	@echo "Running memory check on Flat Combining test..."
	@if command -v compute-sanitizer >/dev/null 2>&1; then \
		compute-sanitizer --tool memcheck ./$(FC_TARGET); \
	elif command -v cuda-memcheck >/dev/null 2>&1; then \
		cuda-memcheck --tool memcheck --leak-check full ./$(FC_TARGET); \
	else \
		echo "No CUDA memory checker found."; \
	fi

memcheck-lock: $(LOCK_TARGET)
	@echo "Running memory check on Lock test..."
	@if command -v compute-sanitizer >/dev/null 2>&1; then \
		compute-sanitizer --tool memcheck ./$(LOCK_TARGET); \
	elif command -v cuda-memcheck >/dev/null 2>&1; then \
		cuda-memcheck --tool memcheck --leak-check full ./$(LOCK_TARGET); \
	else \
		echo "No CUDA memory checker found."; \
	fi

# Profile with Nsight Compute
profile-fc: $(FC_TARGET)
	@echo "Profiling Flat Combining test..."
	@if command -v ncu >/dev/null 2>&1; then \
		ncu --target-processes all --force-overwrite --export profile_fc ./$(FC_TARGET); \
		echo "Profile saved to profile_fc.ncu-rep"; \
	else \
		echo "Nsight Compute (ncu) not found."; \
	fi

profile-lock: $(LOCK_TARGET)
	@echo "Profiling Lock test..."
	@if command -v ncu >/dev/null 2>&1; then \
		ncu --target-processes all --force-overwrite --export profile_lock ./$(LOCK_TARGET); \
		echo "Profile saved to profile_lock.ncu-rep"; \
	else \
		echo "Nsight Compute (ncu) not found."; \
	fi

# ===============================================================================
# Utility Rules
# ===============================================================================

# Print build information
info:
	@echo "Build Configuration:"
	@echo "  Platform: $(PLATFORM)"
	@echo "  CUDA_PATH: $(CUDA_PATH)"
	@echo "  GPU_ARCH: sm_$(SM)"
	@echo "  BUILD_TYPE: $(BUILD_TYPE)"
	@echo "  Features:"
	@echo "    EFA Support: $(HAS_EFA)"
	@echo "    Grace-Hopper: $(if $(findstring GRACE,$(GH_CFLAGS)),yes,no)"
	@echo "    Python Support: $(PYBIND11_AVAILABLE)"
	@echo "    PyTorch Support: $(TORCH_AVAILABLE)"
	@echo "  Benchmark Targets:"
	@echo "    Flat Combining: $(FC_TARGET)"
	@echo "    Lock Throughput: $(LOCK_TARGET)"
	@echo "    Multi Ring: $(MULTI_RING_TARGET)"
ifeq ($(HAS_EFA),1)
	@echo "  EP Runtime Targets:"
	@echo "    Local: $(TARGET_LOCAL)"
	@echo "    Remote: $(TARGET_REMOTE)"
	@echo "    Dual: $(TARGET_DUAL)"
	@echo "    Python Extension: $(EP_EXT)"
endif
	@echo "  NVCC_FLAGS: $(NVCC_FLAGS)"
	@echo "  CXXFLAGS: $(CXXFLAGS)"

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "NVCC not found at $(NVCC)"; exit 1; }
	@echo "NVCC version:"
	@$(NVCC) --version | head -4
	@echo ""
	@echo "GPU information:"
	@nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"
	@echo ""
	@echo "Platform detection:"
	@echo "  Architecture: $(ARCH)"
	@echo "  GPU Name: $(GPU_NAME)"
	@echo "  Platform: $(PLATFORM)"
	@echo ""
	@echo "Optional dependencies:"
	@echo -n "  EFA: "; if [ -d "$(EFA_HOME)" ]; then echo "Found at $(EFA_HOME)"; else echo "Not found"; fi
	@echo -n "  glog: "; if [ "$(GLOG_AVAILABLE)" = "yes" ]; then echo "Found"; else echo "Not found"; fi
	@echo -n "  Python: "; command -v $(PYTHON) >/dev/null 2>&1 && $(PYTHON) --version || echo "Not found"
	@echo -n "  pybind11: "; echo "$(PYBIND11_AVAILABLE)"
	@echo -n "  PyTorch: "; echo "$(TORCH_AVAILABLE)"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	# Clean benchmark test targets
	rm -f $(FC_TARGET) $(LOCK_TARGET) $(MULTI_RING_TARGET)
	rm -f $(FC_LATENCY_TARGET) $(FC_LOAD_LATENCY_TARGET) $(CAS_TARGET)
	rm -f test_fc_throughput test_fc_throughput_debug
	rm -f test_lock_throughput test_lock_throughput_debug
	rm -f test_multi_ring_throughput test_multi_ring_throughput_debug
	rm -f test_fc_latency test_fc_latency_debug
	rm -f test_fc_load_latency test_fc_load_latency_debug
	rm -f test_cas_throughput test_cas_throughput_debug
	# Clean RDMA test targets
ifeq ($(HAS_EFA),1)
	rm -f $(ALLTOALL_RAIL_SHARED_QP_TARGET)
	rm -f test_alltoall_rail_shared_qp test_alltoall_rail_shared_qp_debug
	rm -f $(ALLTOALL_NONRAIL_SHARED_QP_TARGET)
	rm -f test_alltoall_nonrail_shared_qp test_alltoall_nonrail_shared_qp_debug
	rm -f $(DISPATCH_CONNECTIONLESS_TARGET)
	rm -f test_dispatch_connectionless test_dispatch_connectionless_debug
	rm -f $(DISPATCH_MULTIQP_TARGET)
	rm -f test_dispatch_multiqp test_dispatch_multiqp_debug
endif
	# Clean EP runtime targets
ifeq ($(HAS_EFA),1)
	rm -f $(TARGET_LOCAL) $(TARGET_REMOTE) $(TARGET_DUAL)
	rm -f benchmark_local benchmark_local_debug
	rm -f benchmark_remote benchmark_remote_debug
	rm -f benchmark_dual benchmark_dual_debug
	rm -f $(EP_EXT)
	rm -f $(OBJ_CPP) $(OBJ_CU) $(OBJ_BIND)
	rm -f $(SRC_LOCAL:.cu=.o) $(SRC_REMOTE:.cu=.o) $(SRC_DUAL:.cu=.o)
	rm -f src/*.o src/*.d bench/*.o bench/*.d
endif
	# Clean profiling and debug files
	rm -f *.ncu-rep
	rm -f core.*
	rm -f *.d
	@echo "Clean complete."

# Help
help:
	@echo "Unified UCCL Makefile - Platform: $(PLATFORM)"
	@echo ""
	@echo "Available targets:"
	@echo "  all              - Build all available tests and extensions"
	@echo "  benchmarks       - Build all benchmark tests"
ifeq ($(HAS_EFA),1)
	@echo "  ep-runtime       - Build EP runtime and benchmarks (EFA required)"
	@echo "  py               - Build Python extension (requires pybind11 and PyTorch)"
endif
	@echo ""
	@echo "Individual benchmark targets:"
	@echo "  fc               - Build flat combining throughput test"
	@echo "  lock             - Build lock throughput test"
	@echo "  multi-ring       - Build multi-ring throughput test"
	@echo "  fc-latency       - Build flat combining latency test"
	@echo "  fc-load-latency  - Build flat combining load-latency test"
	@echo "  cas              - Build CAS throughput test"
ifeq ($(HAS_EFA),1)
	@echo ""
	@echo "RDMA test targets (EFA required):"
	@echo "  alltoall-rail-shared-qp    - Build AllToAll Rail test with shared QP"
	@echo "  alltoall-nonrail-shared-qp - Build AllToAll NonRail test with shared QP"
endif
	@echo ""
	@echo "Run targets:"
	@echo "  run-fc           - Run flat combining test"
	@echo "  run-lock         - Run lock throughput test"
	@echo "  run-multi-ring   - Run multi-ring test"
	@echo "  run-fc-latency   - Run flat combining latency test"
	@echo "  run-fc-load-latency - Run flat combining load-latency test"
	@echo "  run-cas          - Run CAS throughput test"
ifeq ($(HAS_EFA),1)
	@echo "  run-ep-local     - Run EP local benchmark"
	@echo "  run-ep-remote    - Run EP remote benchmark"
	@echo "  run-ep-dual      - Run EP dual benchmark"
	@echo "  run-ep           - Run all EP benchmarks"
endif
	@echo "  quick            - Quick test of main benchmarks"
	@echo "  benchmark        - Run complete benchmark suite"
	@echo ""
	@echo "Analysis targets:"
	@echo "  memcheck-fc      - Memory check flat combining test"
	@echo "  memcheck-lock    - Memory check lock test"
	@echo "  profile-fc       - Profile flat combining test"
	@echo "  profile-lock     - Profile lock test"
	@echo ""
	@echo "Utility targets:"
ifeq ($(HAS_EFA),1)
	@echo "  install          - Install Python extension"
endif
	@echo "  info             - Print build configuration"
	@echo "  check-deps       - Check build dependencies"
	@echo "  clean            - Remove build artifacts"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  GPU_ARCH/SM      - Target GPU architecture (default: auto-detect)"
	@echo "  BUILD_TYPE       - Build type: release or debug (default: release)"
	@echo "  CUDA_PATH        - Path to CUDA installation (default: /usr/local/cuda)"
	@echo "  EFA_HOME         - Path to EFA installation (default: /opt/amazon/efa)"
	@echo "  PYTHON           - Python interpreter (default: python3)"

# ===============================================================================
# Dependency Management
# ===============================================================================

# Automatically include dependency files if they exist
ifeq ($(HAS_EFA),1)
    DEPS := $(OBJ_CPP:.o=.d) $(OBJ_CU:.o=.d)
    DEPS += $(SRC_LOCAL:.cu=.d) $(SRC_REMOTE:.cu=.d) $(SRC_DUAL:.cu=.d)
    ifneq ($(PYBIND11_AVAILABLE),no)
        DEPS += $(OBJ_BIND:.o=.d)
    endif
    -include $(DEPS)
endif

# ===============================================================================
# Phony targets
# ===============================================================================
.PHONY: all benchmarks ep-runtime ep py
.PHONY: fc lock multi-ring fc-latency fc-load-latency cas
.PHONY: alltoall-rail-shared-qp alltoall-nonrail-shared-qp
.PHONY: run-fc run-lock run-multi-ring run-fc-latency run-fc-load-latency run-cas
.PHONY: run-ep-local run-ep-remote run-ep-dual run-ep
.PHONY: quick benchmark
.PHONY: memcheck-fc memcheck-lock profile-fc profile-lock
.PHONY: install info check-deps clean help