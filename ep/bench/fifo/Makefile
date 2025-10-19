# ===============================================================================
# Makefile for FIFO Performance Benchmark
#
# Usage:
#   make [release|debug]     - Build optimized or debug version
#   make test               - Run basic throughput test
#   make benchmark          - Run comprehensive benchmark suite
#   make latency            - Run latency tests
#   make clean              - Remove build artifacts
# ===============================================================================

# ===============================================================================
# Configuration
# ===============================================================================
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CXX := g++

# Auto-detect GPU architecture if not specified
GPU_ARCH ?= auto
ifeq ($(GPU_ARCH), auto)
    # Detect GPU compute capability from nvidia-smi
    GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | sed 's/\.//' || echo "90")
endif

# Build configuration
BUILD_TYPE ?= release
TARGET := benchmark_fifo

# ===============================================================================
# Compiler Flags
# ===============================================================================
# Base NVCC flags
NVCC_FLAGS := -std=c++17
NVCC_FLAGS += -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
NVCC_FLAGS += -Xcompiler -Wall
NVCC_FLAGS += -Xcompiler -Wextra
NVCC_FLAGS += --expt-relaxed-constexpr
NVCC_FLAGS += -DMSCCLPP_USE_CUDA

# Base CXX flags
CXX_FLAGS := -std=c++17 -Wall -Wextra -pthread
CXX_FLAGS += -DMSCCLPP_USE_CUDA

# Release vs Debug flags
ifeq ($(BUILD_TYPE), debug)
    NVCC_FLAGS += -g -G -O0 -DDEBUG
    CXX_FLAGS += -g -O0 -DDEBUG
    TARGET := $(TARGET)_debug
    $(info Building DEBUG version)
else
    NVCC_FLAGS += -O3 -g -DNDEBUG
    NVCC_FLAGS += --use_fast_math
    CXX_FLAGS += -O3 -g -DNDEBUG
    $(info Building RELEASE version for GPU arch $(GPU_ARCH))
endif

# Include paths
INCLUDES := -I../../include -I../../src
INCLUDES += -I$(CUDA_PATH)/include

# Libraries
LIBS := -lpthread -lcudart -lcuda -lnuma
LIBS += -L$(CUDA_PATH)/lib64

# Source files
SRC_CU := launch_kernel_shim.cu
SRC_CPP := benchmark_fifo.cpp ../../src/fifo.cc

# Additional sources from ep/src
SRC_CPP += ../../src/numa.cc
SRC_CPP += ../../src/gpu_utils.cc
SRC_CPP += ../../src/errors.cc

# Object files
OBJ_CU := $(SRC_CU:.cu=.o)
OBJ_CPP := $(SRC_CPP:.cc=.o)
OBJ_CPP := $(OBJ_CPP:.cpp=.o)

# ===============================================================================
# Build Rules
# ===============================================================================
.DEFAULT_GOAL := release

# Main targets
release: BUILD_TYPE := release
release: $(TARGET)

debug: BUILD_TYPE := debug
debug: $(TARGET)

$(TARGET): $(OBJ_CU) $(OBJ_CPP)
	@echo "Linking $@..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^ $(LIBS)
	@echo "Build complete: $@"

# CUDA compilation rule
%.o: %.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# C++ compilation rule for local cpp files
%.o: %.cpp
	@echo "Compiling C++: $<"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# C++ compilation rule for ep/src files
../../src/%.o: ../../src/%.cc
	@echo "Compiling C++: $<"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# ===============================================================================
# Test and Benchmark Rules
# ===============================================================================

# Basic throughput test
test: $(TARGET)
	@echo "Running basic throughput test..."
	@echo "================================="
	./$(TARGET)

# Quick performance test
quick: $(TARGET)
	@echo "Running quick performance test..."
	./$(TARGET)

# Comprehensive benchmark suite
benchmark: $(TARGET)
	@echo "Running comprehensive benchmark suite..."
	@echo "========================================"
	./$(TARGET)

# Latency tests
latency: $(TARGET)
	@echo "Running latency tests..."
	@echo "========================"
	./$(TARGET) -l

# Verbose output
verbose: $(TARGET)
	@echo "Running with verbose output..."
	./$(TARGET) -v

# Full test (throughput + latency)
full: $(TARGET)
	@echo "Running full benchmark suite..."
	@echo "==============================="
	./$(TARGET) -l -v

# Profile with Nsight Compute (if available)
profile: $(TARGET)
	@echo "Running performance profile..."
	@if command -v ncu >/dev/null 2>&1; then \
		ncu --target-processes all --force-overwrite --export profile_fifo ./$(TARGET); \
		echo "Profile saved to profile_fifo.ncu-rep"; \
	else \
		echo "Nsight Compute (ncu) not found. Skipping profile."; \
	fi

# ===============================================================================
# Utility Rules
# ===============================================================================

# Print build information
info:
	@echo "Build Configuration:"
	@echo "  CUDA_PATH: $(CUDA_PATH)"
	@echo "  GPU_ARCH: $(GPU_ARCH)"
	@echo "  BUILD_TYPE: $(BUILD_TYPE)"
	@echo "  TARGET: $(TARGET)"
	@echo "  NVCC_FLAGS: $(NVCC_FLAGS)"
	@echo "  CXX_FLAGS: $(CXX_FLAGS)"
	@echo "  INCLUDES: $(INCLUDES)"
	@echo "  LIBS: $(LIBS)"

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "NVCC not found at $(NVCC)"; exit 1; }
	@command -v $(CXX) >/dev/null 2>&1 || { echo "G++ not found"; exit 1; }
	@echo "NVCC version:"
	@$(NVCC) --version | head -4
	@echo "GPU information:"
	@nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET) benchmark_fifo benchmark_fifo_debug
	rm -f *.o ../../src/*.o
	rm -f *.ncu-rep
	rm -f core.*
	@echo "Clean complete."

# Help target
help:
	@echo "Available targets:"
	@echo "  release      - Build optimized version (default)"
	@echo "  debug        - Build debug version"
	@echo "  test         - Run basic throughput test"
	@echo "  quick        - Run quick performance test"
	@echo "  benchmark    - Run comprehensive benchmark suite"
	@echo "  latency      - Run latency tests"
	@echo "  verbose      - Run with verbose output"
	@echo "  full         - Run full benchmark suite"
	@echo "  profile      - Run performance profiling"
	@echo "  info         - Print build configuration"
	@echo "  check-deps   - Check build dependencies"
	@echo "  clean        - Remove build artifacts"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  GPU_ARCH     - Target GPU architecture (default: auto-detect)"
	@echo "  BUILD_TYPE   - Build type: release or debug (default: release)"
	@echo "  CUDA_PATH    - Path to CUDA installation (default: /usr/local/cuda)"

# ===============================================================================
# Phony targets
# ===============================================================================
.PHONY: all release debug test quick benchmark latency verbose full profile
.PHONY: info check-deps clean help

