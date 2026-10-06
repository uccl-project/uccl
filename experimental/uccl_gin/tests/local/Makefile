CUDA_HOME ?= /usr/local/cuda
SM ?= 120
CXX ?= g++
NUMA_INCLUDE_DIR ?=
NUMA_LIBRARY_DIR ?=
BUILD_DIR ?= build/sm$(SM)

GIN := ../..
ROOT := ../../../..
NVCC := $(CUDA_HOME)/bin/nvcc
INCLUDES := -I$(GIN) -I$(ROOT)/include -I$(CUDA_HOME)/include $(if $(NUMA_INCLUDE_DIR),-I$(NUMA_INCLUDE_DIR))
HOSTFLAGS := -O3 -std=c++17 -pthread -Wall $(INCLUDES)
DEVFLAGS := -O3 -std=c++17 -arch=sm_$(SM) -ccbin=$(CXX) --expt-relaxed-constexpr $(INCLUDES) -Xcompiler=-pthread,-Wall
LIBS := -L$(CUDA_HOME)/lib64 -Xlinker=-rpath -Xlinker=$(CUDA_HOME)/lib64 $(if $(NUMA_LIBRARY_DIR),-L$(NUMA_LIBRARY_DIR) -Xlinker=-rpath -Xlinker=$(NUMA_LIBRARY_DIR)) -lnuma -lcuda

.PHONY: all clean
all: $(BUILD_DIR)/capabilities $(BUILD_DIR)/queue_smoke

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/fifo.o: $(GIN)/transport/fifo.cpp | $(BUILD_DIR)
	$(CXX) $(HOSTFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%: %.cu queue_fixture.hpp $(BUILD_DIR)/fifo.o
	$(NVCC) $(DEVFLAGS) -DUCCL_GIN_WITH_NCCL_GIN=0 -MMD -MP $< $(BUILD_DIR)/fifo.o $(LIBS) -o $@

clean:
	rm -rf build

-include $(BUILD_DIR)/*.d
