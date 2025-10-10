# TCPX Unpack Architecture Makefile

# Toolchain
CXX        := g++
CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc

# Flags
CXXFLAGS   := -std=c++17 -fPIC -O2 -Wall -Iinclude -I. -I$(CUDA_HOME)/include
NVCCFLAGS  := -std=c++17 -Xcompiler "-fPIC -O2 -Wall" -Iinclude -I. -I$(CUDA_HOME)/include
LIBS       := -ldl -lpthread
CUDA_LIBS  := -L$(CUDA_HOME)/lib64 -lcuda -lcudart

# Source files
# Note: rx_descriptor.h is now header-only, no .cc files needed
DEVICE_SRCS := device/unpack_kernels.cu device/unpack_launch.cu
CORE_SRCS   := src/sliding_window.cc src/bootstrap.cc src/channel_manager.cc

# Object files
DEVICE_OBJS := device/unpack_kernels.o device/unpack_launch.o
CORE_OBJS   := src/sliding_window.o src/bootstrap.o src/channel_manager.o

# Build object files
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

src/%.o: src/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

device/unpack_kernels.o: device/unpack_kernels.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

device/unpack_launch.o: device/unpack_launch.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Core components (device + multi-channel modules)
core: $(DEVICE_OBJS) $(CORE_OBJS)
	@echo "Core TCPX components built successfully!"
	@echo "Device objects: $(DEVICE_OBJS)"
	@echo "Core objects: $(CORE_OBJS)"

# Multi-channel performance test (main program)
test_tcpx_perf_multi: $(DEVICE_OBJS) $(CORE_OBJS)
	@echo "Building test_tcpx_perf_multi..."
	$(NVCC) $(NVCCFLAGS) -o tests/test_tcpx_perf_multi tests/test_tcpx_perf_multi.cc tcpx_impl.cc $(DEVICE_OBJS) $(CORE_OBJS) $(LIBS) $(CUDA_LIBS)

# Multi-channel transfer test
test_tcpx_transfer_multi: $(DEVICE_OBJS) $(CORE_OBJS)
	@echo "Building test_tcpx_transfer_multi..."
	$(NVCC) $(NVCCFLAGS) -o tests/test_tcpx_transfer_multi tests/test_tcpx_transfer_multi.cc tcpx_impl.cc $(DEVICE_OBJS) $(CORE_OBJS) $(LIBS) $(CUDA_LIBS)

# Module unit tests
test_modules: $(CORE_OBJS)
	@echo "Building test_modules..."
	$(CXX) $(CXXFLAGS) -o tests/test_modules tests/test_modules.cc tcpx_impl.cc $(CORE_OBJS) $(LIBS) $(CUDA_LIBS)

# Build targets
all: core test_tcpx_perf_multi test_tcpx_transfer_multi test_modules

# Quick test - just core components
quick: core
	@echo "Quick build completed!"

# Check what files we have
check:
	@echo "=== File Check ==="
	@echo "Public headers:"
	@ls -la include/*.h 2>/dev/null || echo "Missing header files"
	@echo ""
	@echo "Implementation files:"
	@ls -la tcpx_impl.cc 2>/dev/null || echo "Missing implementation files"
	@echo ""
	@echo "Device source files:"
	@ls -la device/*.cu device/*.h 2>/dev/null || echo "Missing device files"
	@echo ""
	@echo "Test files:"
	@ls -la tests/*.cc 2>/dev/null || echo "Missing test files"
	@echo ""
	@echo "Built objects:"
	@ls -la $(DEVICE_OBJS) 2>/dev/null || echo "No objects built yet"

# Clean built files
clean:
	@echo "Cleaning build artifacts..."
	rm -f tests/test_tcpx_perf_multi tests/test_tcpx_transfer_multi tests/test_modules
	rm -f $(DEVICE_OBJS) $(CORE_OBJS) *.o
	@echo "Clean completed."

.PHONY: all core quick check clean test_tcpx_perf_multi test_tcpx_transfer_multi test_modules
