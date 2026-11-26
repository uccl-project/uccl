# No Torch extension or vendored dependencies; ctypes uses this narrow C ABI.
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= sm_120
NCCL_HOME ?= /usr
CUDA_LIB ?= $(CUDA_HOME)/lib64
LITE := ../..
BUILD := $(LITE)/build/peer_e2e
NVCC := $(CUDA_HOME)/bin/nvcc
FLAGS := -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler=-fPIC -MMD -MP -I$(LITE)/include -I$(NCCL_HOME)/include
.PHONY: all peer-runtime-refresh
all: $(BUILD)/libuccl_lite_peer_e2e.so
$(BUILD):
	mkdir -p $@
$(BUILD)/bridge.o: collectives.cc | $(BUILD)
	$(NVCC) $(FLAGS) -x cu -c $< -o $@
peer-runtime-refresh:
	$(MAKE) -C $(LITE) peer-runtime CUDA_HOME=$(CUDA_HOME) CUDA_ARCH=$(CUDA_ARCH)
$(LITE)/build/runtime/libuccl_lite_peer.a: peer-runtime-refresh
$(BUILD)/libuccl_lite_peer_e2e.so: $(BUILD)/bridge.o $(LITE)/build/runtime/libuccl_lite_peer.a
	$(NVCC) -shared -arch=$(CUDA_ARCH) $^ -L$(NCCL_HOME)/lib -l:libnccl.so.2 -L$(CUDA_LIB) -lcudart -Xcompiler=-pthread -ldl -Xlinker=-rpath,$(NCCL_HOME)/lib -Xlinker=-rpath,$(CUDA_LIB) -o $@
-include $(BUILD)/*.d
