# ===============================================================================
# Makefile for UCCL Benchmarks
# Includes: Flat Combining test and Lock Throughput test
#
# Usage:
#   make all                - Build all tests
#   make fc                 - Build flat combining test
#   make lock               - Build lock throughput test
#   make clean              - Remove all build artifacts
#   make run-fc             - Run flat combining test
#   make run-lock           - Run lock throughput test
# ===============================================================================

# Configuration
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

# Auto-detect GPU architecture if not specified
GPU_ARCH ?= auto
ifeq ($(GPU_ARCH), auto)
    GPU_ARCH := $(shell $(NVCC) --list-gpu-code | grep 'sm_' | tail -1 | sed 's/.*sm_//' || echo "90")
endif

# Build configuration
BUILD_TYPE ?= release

# Target executables
FC_TARGET := test_fc_throughput
LOCK_TARGET := test_lock_throughput
FC_LATENCY_TARGET := test_fc_latency
FC_LOAD_LATENCY_TARGET := test_fc_load_latency

# ===============================================================================
# Compiler Flags
# ===============================================================================
# Base flags
NVCC_FLAGS := -std=c++17
NVCC_FLAGS += -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
NVCC_FLAGS += -Xcompiler -Wall
NVCC_FLAGS += --expt-relaxed-constexpr

# Release vs Debug flags
ifeq ($(BUILD_TYPE), debug)
    NVCC_FLAGS += -g -G -O0
    NVCC_FLAGS += -DDEBUG
    FC_TARGET := $(FC_TARGET)_debug
    LOCK_TARGET := $(LOCK_TARGET)_debug
    FC_LATENCY_TARGET := $(FC_LATENCY_TARGET)_debug
    FC_LOAD_LATENCY_TARGET := $(FC_LOAD_LATENCY_TARGET)_debug
    $(info Building DEBUG version)
else
    NVCC_FLAGS += -O3 -g
    NVCC_FLAGS += -DNDEBUG
    NVCC_FLAGS += --use_fast_math
    $(info Building RELEASE version for GPU arch $(GPU_ARCH))
endif

# Include paths
INCLUDES := -I../include -I../src -I../../include

# Libraries
LIBS := -lpthread

# Header dependencies
HEADERS := $(wildcard ../include/*.h ../include/*.cuh ../include/*.hpp)

# ===============================================================================
# Build Rules
# ===============================================================================
.DEFAULT_GOAL := all

# Build all targets (only working tests for now)
all: $(FC_TARGET) $(LOCK_TARGET)
	@echo "All targets built successfully"

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

# Quick test - run both with minimal config
quick: $(FC_TARGET) $(LOCK_TARGET)
	@echo "Quick test - Flat Combining:"
	./$(FC_TARGET) | head -20
	@echo ""
	@echo "Quick test - Lock Throughput:"
	./$(LOCK_TARGET) | head -30

# Benchmark suite
benchmark: all
	@echo "Running complete benchmark suite..."
	@echo "==================================="
	@echo "\n--- Flat Combining Throughput ---"
	./$(FC_TARGET)
	@echo "\n--- Lock Throughput ---"
	./$(LOCK_TARGET)
	@echo "\n--- Flat Combining Latency ---"
	./$(FC_LATENCY_TARGET)
	@echo "\n--- Flat Combining Load-Latency ---"
	./$(FC_LOAD_LATENCY_TARGET)

# ===============================================================================
# Analysis and Profiling
# ===============================================================================

# Memory check
memcheck-fc: $(FC_TARGET)
	@echo "Running memory check on Flat Combining test..."
	@if command -v cuda-memcheck >/dev/null 2>&1; then \
		cuda-memcheck --tool memcheck --leak-check full ./$(FC_TARGET); \
	else \
		echo "cuda-memcheck not found."; \
	fi

memcheck-lock: $(LOCK_TARGET)
	@echo "Running memory check on Lock test..."
	@if command -v cuda-memcheck >/dev/null 2>&1; then \
		cuda-memcheck --tool memcheck --leak-check full ./$(LOCK_TARGET); \
	else \
		echo "cuda-memcheck not found."; \
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
	@echo "  CUDA_PATH: $(CUDA_PATH)"
	@echo "  GPU_ARCH: $(GPU_ARCH)"
	@echo "  BUILD_TYPE: $(BUILD_TYPE)"
	@echo "  Targets:"
	@echo "    Flat Combining: $(FC_TARGET)"
	@echo "    Lock Throughput: $(LOCK_TARGET)"
	@echo "    FC Latency: $(FC_LATENCY_TARGET)"
	@echo "    FC Load-Latency: $(FC_LOAD_LATENCY_TARGET)"
	@echo "  NVCC_FLAGS: $(NVCC_FLAGS)"

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "NVCC not found at $(NVCC)"; exit 1; }
	@echo "NVCC version:"
	@$(NVCC) --version | head -4
	@echo "GPU information:"
	@nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(FC_TARGET) $(LOCK_TARGET) $(FC_LATENCY_TARGET) $(FC_LOAD_LATENCY_TARGET)
	rm -f test_fc_throughput test_fc_throughput_debug
	rm -f test_lock_throughput test_lock_throughput_debug
	rm -f test_fc_latency test_fc_latency_debug
	rm -f test_fc_load_latency test_fc_load_latency_debug
	rm -f test_lock test_lock_debug
	rm -f *.ncu-rep
	rm -f core.*
	@echo "Clean complete."

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build all tests (default)"
	@echo "  fc               - Build flat combining throughput test"
	@echo "  lock             - Build lock throughput test"
	@echo "  fc-latency       - Build flat combining latency test"
	@echo "  fc-load-latency  - Build flat combining load-latency test"
	@echo "  run-fc           - Run flat combining test"
	@echo "  run-lock         - Run lock throughput test"
	@echo "  run-fc-latency   - Run flat combining latency test"
	@echo "  run-fc-load-latency - Run flat combining load-latency test"
	@echo "  quick            - Quick test of main benchmarks"
	@echo "  benchmark        - Run complete benchmark suite"
	@echo "  memcheck-fc      - Memory check flat combining test"
	@echo "  memcheck-lock    - Memory check lock test"
	@echo "  profile-fc       - Profile flat combining test"
	@echo "  profile-lock     - Profile lock test"
	@echo "  info             - Print build configuration"
	@echo "  check-deps       - Check build dependencies"
	@echo "  clean            - Remove build artifacts"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  GPU_ARCH     - Target GPU architecture (default: auto-detect)"
	@echo "  BUILD_TYPE   - Build type: release or debug (default: release)"
	@echo "  CUDA_PATH    - Path to CUDA installation (default: /usr/local/cuda)"

# ===============================================================================
# Phony targets
# ===============================================================================
.PHONY: all fc lock fc-latency fc-load-latency
.PHONY: run-fc run-lock run-fc-latency run-fc-load-latency
.PHONY: quick benchmark
.PHONY: memcheck-fc memcheck-lock profile-fc profile-lock
.PHONY: info check-deps clean help
