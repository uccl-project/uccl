PYTHON ?= python3
SM ?= 89
TORCH_CUDA_ARCH_LIST ?= $(shell echo "$(SM)" | sed 's/^\([0-9]\)\([0-9]\)$$/\1.\2/')
DISABLE_SM90_FEATURES ?= $(shell test "$(SM)" -lt 90 && echo 1 || echo 0)
DEEP_EP_BUILD_LEGACY ?= 0
NCCL_ROOT ?= /home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/deps_nccl_2304/nvidia/nccl
TORCH_NVSHMEM_STUB ?= /home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/torch-stubs/libtorch_nvshmem.so
EP_JIT_CACHE_DIR ?= /home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/jit-cache

.PHONY: all install clean test-intra

all:
	TORCH_CUDA_ARCH_LIST=$(TORCH_CUDA_ARCH_LIST) \
	DISABLE_SM90_FEATURES=$(DISABLE_SM90_FEATURES) \
	DEEP_EP_BUILD_LEGACY=$(DEEP_EP_BUILD_LEGACY) \
	EP_NCCL_ROOT_DIR=$(NCCL_ROOT) \
	$(PYTHON) setup.py build_ext --inplace

install:
	TORCH_CUDA_ARCH_LIST=$(TORCH_CUDA_ARCH_LIST) \
	DISABLE_SM90_FEATURES=$(DISABLE_SM90_FEATURES) \
	DEEP_EP_BUILD_LEGACY=$(DEEP_EP_BUILD_LEGACY) \
	EP_NCCL_ROOT_DIR=$(NCCL_ROOT) \
	$(PYTHON) -m pip install --no-build-isolation -e .

test-intra:
	CUDA_VISIBLE_DEVICES=2,3 \
	PYTHONPATH=$$(pwd) \
	EP_NCCL_ROOT_DIR=$(NCCL_ROOT) \
	LD_LIBRARY_PATH=$(NCCL_ROOT)/lib:$${LD_LIBRARY_PATH:-} \
	LD_PRELOAD=$(NCCL_ROOT)/lib/libnccl.so.2:$(TORCH_NVSHMEM_STUB) \
	EP_SUPPRESS_NCCL_CHECK=1 \
	EP_FORCE_NO_NVLINK=1 \
	NCCL_NET_GDR_LEVEL=0 \
	NCCL_GIN_TYPE=2 \
	DISABLE_SM90_FEATURES=$(DISABLE_SM90_FEATURES) \
	EP_TEST_DISABLE_FP8=1 \
	EP_FORCE_PROCESS_EXIT=1 \
	EP_JIT_CACHE_DIR=$(EP_JIT_CACHE_DIR) \
	$(PYTHON) tests/elastic/test_ep.py --num-processes 2 --allow-hybrid-mode 0 \
	  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 --test-first-only

clean:
	rm -rf build deep_ep.egg-info deep_ep/_C*.so
