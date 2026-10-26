CUDA_HOME ?= /usr/local/cuda
SM ?= 90
NCCL_DIR ?=
NCCL_EP_USE_UCCL_GIN ?= OFF
BUILD_DIR ?= build/sm$(SM)-uccl$(NCCL_EP_USE_UCCL_GIN)
JOBS ?= 2
CMAKE_FLAGS ?=

.PHONY: all lib clean
all: lib
lib:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_CUDA_COMPILER=$(CUDA_HOME)/bin/nvcc \
	  -DCMAKE_CUDA_ARCHITECTURES=$(SM) -DNCCL_DIR=$(NCCL_DIR) \
	  -DNCCL_EP_USE_UCCL_GIN=$(NCCL_EP_USE_UCCL_GIN) $(CMAKE_FLAGS)
	cmake --build $(BUILD_DIR) --target nccl_ep_shared -j$(JOBS)
clean:
	rm -rf $(BUILD_DIR)
