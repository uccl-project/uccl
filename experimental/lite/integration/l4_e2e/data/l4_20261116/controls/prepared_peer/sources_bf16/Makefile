# No Torch extension or vendored dependencies; ctypes uses this narrow C ABI.
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= sm_89
NCCL_HOME ?= $(HOME)/third_party/nccl/build
CUDA_LIB ?= $(CUDA_HOME)/lib64
LITE := ../..
BUILD := $(LITE)/build/l4_e2e
NVCC := $(CUDA_HOME)/bin/nvcc
MPI_HOME ?= /usr/mpi/gcc/openmpi-4.1.7rc1
FLAGS := -I$(MPI_HOME)/include -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler=-fPIC -MMD -MP -I$(LITE)/include -I$(NCCL_HOME)/include
.PHONY: all
all: $(BUILD)/libuccl_lite_peer_e2e.so
$(BUILD):
	mkdir -p $@
$(BUILD)/peer.o: $(LITE)/src/collectives/peer_collectives.cu | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@
$(BUILD)/bridge.o: collectives.cc | $(BUILD)
	$(NVCC) $(FLAGS) -x cu -c $< -o $@
$(LITE)/build/cpu_router/libuccl_lite_cpu_router.a:
	$(MAKE) -C $(LITE) build/cpu_router/libuccl_lite_cpu_router.a CUDA_HOME=$(CUDA_HOME) CUDA_ARCH=$(CUDA_ARCH)
$(BUILD)/libuccl_lite_peer_e2e.so: $(BUILD)/bridge.o $(BUILD)/peer.o $(LITE)/build/cpu_router/libuccl_lite_cpu_router.a
	$(NVCC) -shared -L$(MPI_HOME)/lib -lmpi -Xlinker=-rpath,$(MPI_HOME)/lib -arch=$(CUDA_ARCH) $^ -L$(NCCL_HOME)/lib -l:libnccl.so.2 -L$(CUDA_LIB) -lcudart -Xcompiler=-pthread -ldl -Xlinker=-rpath,$(NCCL_HOME)/lib -Xlinker=-rpath,$(CUDA_LIB) -o $@
-include $(BUILD)/*.d
