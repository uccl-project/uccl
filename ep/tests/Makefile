# Paths
CUDA_HOME := /usr/local/cuda
EFA_HOME := /opt/amazon/efa

# Tools
NVCC := $(CUDA_HOME)/bin/nvcc
CXX := g++

# Flags
INCLUDES := -I../include -I../../include -I$(CUDA_HOME)/include -I$(EFA_HOME)/include
LDFLAGS := -L$(CUDA_HOME)/lib64 -L$(EFA_HOME)/lib -lcuda -lcudart -libverbs -lpthread
CXXFLAGS := -O3 -g -std=c++17 $(INCLUDES)
CUDAFLAGS := -arch=sm_90  # Use sm_80 for A100
HOST_TEST_INCLUDES ?= -Iinclude -I../include
HOST_TEST_CXXFLAGS := -std=c++17 -Wall -Wextra -Werror $(HOST_TEST_INCLUDES)

# Files
COMMON_SRC := ../src/common.cpp
COMMON_OBJ := ../src/common.o

# Binaries
CUDA_BIN := pcie_bench gpu_to_cpu_bench batched_gpu_to_cpu_bench
HOST_TEST_BIN := rdma_addressing_test

all: $(CUDA_BIN)

# Object file
$(COMMON_OBJ): $(COMMON_SRC) ../include/common.hpp
	$(CXX) -c $(CXXFLAGS) $(COMMON_SRC) -o $@

# Test benchmarks
pcie_bench: pcie_bench.cu $(COMMON_OBJ)
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS) pcie_bench.cu $(COMMON_OBJ) -o $@ $(LDFLAGS)

gpu_to_cpu_bench: gpu_to_cpu_bench.cu $(COMMON_OBJ)
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS) gpu_to_cpu_bench.cu $(COMMON_OBJ) -o $@ $(LDFLAGS)

batched_gpu_to_cpu_bench: batched_gpu_to_cpu_bench.cu $(COMMON_OBJ)
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS) batched_gpu_to_cpu_bench.cu $(COMMON_OBJ) -o $@ $(LDFLAGS)

$(HOST_TEST_BIN): rdma_addressing_test.cpp ../include/rdma_addressing.hpp
	$(CXX) $(HOST_TEST_CXXFLAGS) $< -o $@

test: $(HOST_TEST_BIN)
	./$(HOST_TEST_BIN)

clean:
	rm -f $(CUDA_BIN) $(HOST_TEST_BIN) $(COMMON_OBJ)

.PHONY: all clean test
