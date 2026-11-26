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
CUDA_BIN := pcie_bench gpu_to_cpu_bench batched_gpu_to_cpu_bench barrier_concurrent
HOST_TEST_BIN := rdma_addressing_test
INTRANODE_TEST_BIN := intranode_combine_test
FP4_TEST_BIN := fp4_cvt_test mx_cvt_test mxfp8_block_test ll_block_scaled_test
# The hardware ue8m0/e2m1 cvt needs an arch- or family-specific target.  Use
# -gencode: `-arch=sm_120a` also embeds compute_120 PTX, which ptxas rejects.
FP4_CUDAFLAGS ?= -gencode arch=compute_120a,code=sm_120a

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

# Round-pairing test for barrier_block (uccl-project/uccl#986), header-only.
barrier_concurrent: barrier_concurrent.cu $(wildcard ../include/*.cuh) $(wildcard ../include/*.hpp)
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS) $< -o $@

$(INTRANODE_TEST_BIN): intranode_combine_test.cu ../src/intranode.cu $(wildcard ../include/*.cuh) $(wildcard ../include/*.hpp) ../../include/util/gpu_rt.h
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS) --expt-relaxed-constexpr intranode_combine_test.cu ../src/intranode.cu -o $@ $(LDFLAGS)

# Inline FP4 quantizer check (hardware cvt vs software encoder).
fp4_cvt_test: fp4_cvt_test.cu ../include/fp4_utils.cuh ../include/mx_cvt.cuh
	$(NVCC) $(FP4_CUDAFLAGS) -O3 -std=c++17 -I../include $< -o $@

# Element/scale conversion check (mx_cvt.cuh vs CUDA's own conversions).
mx_cvt_test: mx_cvt_test.cu ../include/mx_cvt.cuh
	$(NVCC) $(FP4_CUDAFLAGS) -O3 -std=c++17 -I../include $< -o $@

mxfp8_block_test: mxfp8_block_test.cu ../include/mx_cvt.cuh
	$(NVCC) $(FP4_CUDAFLAGS) -O3 -std=c++17 -I../include $< -o $@

# Two visible GPUs with P2P access. Runs the production LL dispatch/combine kernels.
ll_block_scaled_test: ll_block_scaled_test.cu ../src/internode_ll.cu $(wildcard ../include/*.cuh) $(wildcard ../include/*.hpp)
	$(NVCC) $(FP4_CUDAFLAGS) $(CXXFLAGS) --expt-relaxed-constexpr \
		ll_block_scaled_test.cu ../src/internode_ll.cu -o $@

$(HOST_TEST_BIN): rdma_addressing_test.cpp ../include/rdma_addressing.hpp
	$(CXX) $(HOST_TEST_CXXFLAGS) $< -o $@

test: $(HOST_TEST_BIN)
	./$(HOST_TEST_BIN)

clean:
	rm -f $(CUDA_BIN) $(HOST_TEST_BIN) $(INTRANODE_TEST_BIN) $(FP4_TEST_BIN) $(COMMON_OBJ)

.PHONY: all clean test
