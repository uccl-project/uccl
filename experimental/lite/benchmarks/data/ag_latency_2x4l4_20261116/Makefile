# Rebuild the exact measured source snapshot, not an unrelated later revision.
LITE := ../../..
BUILD ?= $(abspath $(LITE)/build/ag_latency_2x4l4_20261116)
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= $(HOME)/third_party/nccl/build
MPICXX ?= /usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx
NVCC := $(CUDA_HOME)/bin/nvcc
FLAGS := -O3 -arch=sm_89 -std=c++17 -MMD -MP -Imeasured_source/include -I$(NCCL_HOME)/include $(shell $(MPICXX) --showme:compile)
LIBS := -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib $(shell $(MPICXX) --showme:link | sed 's/-Wl,/-Xlinker /g') -lcudart -lpthread
all: $(BUILD)/uccl_lite_multinode_host_allgather $(BUILD)/uccl_lite_nccl_2x4l4_baseline
$(BUILD):
	mkdir -p $@
$(BUILD)/uccl_lite_multinode_host_allgather: measured_source/multinode_host_allgather.cu measured_source/include/uccl_lite/multinode_host_common.cuh | $(BUILD)
	$(NVCC) $(FLAGS) $< -o $@ $(LIBS)
$(BUILD)/uccl_lite_nccl_2x4l4_baseline: measured_source/nccl_2x4l4_baseline.cu | $(BUILD)
	$(NVCC) $(FLAGS) $< -o $@ $(LIBS)
-include $(BUILD)/*.d
