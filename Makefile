SHELL := /bin/bash

PYTHON ?= python3

# Resolve parallel build jobs from provided parameters.
BUILD_JOBS ?= 1
BUILD_JOBS := $(strip $(if $(BUILD_JOBS),$(BUILD_JOBS),1))
BUILD_JOBS := $(strip $(if $(BULD_JOBS),$(BULD_JOBS),$(BUILD_JOBS)))
MAKE_PARALLEL := -j$(BUILD_JOBS)

# Detect ROCm from provided parameters only.
ROCM_DETECTED ?= 0
ROCM_FLAG := $(strip $(ROCM_DETECTED))
ROCM_ENABLED := $(if $(filter 1 true yes,$(ROCM_FLAG)),1,0)

OUTPUT_DIR := uccl
LIB_DIR := $(OUTPUT_DIR)/lib
EP_PY_DIR := ep/python/uccl_ep

# Select per-module makefiles.
RDMA_MAKEFILE := $(if $(ROCM_ENABLED),$(if $(wildcard collective/rdma/Makefile.rocm),Makefile.rocm,Makefile),Makefile)
P2P_MAKEFILE := $(if $(ROCM_ENABLED),$(if $(wildcard p2p/Makefile.rocm),Makefile.rocm,Makefile),Makefile)

MODULE_TARGETS := ccl_rdma p2p ep

.PHONY: all dirs clean $(MODULE_TARGETS) ccl_efa

all: dirs $(MODULE_TARGETS)

dirs:
	mkdir -p $(LIB_DIR)
	mkdir -p $(EP_PY_DIR)

ccl_rdma: dirs
	$(MAKE) -C collective/rdma -f $(RDMA_MAKEFILE) clean || true
	$(MAKE) -C collective/rdma -f $(RDMA_MAKEFILE) $(MAKE_PARALLEL)
	@if [ -f collective/rdma/libnccl-net-uccl.so ]; then cp collective/rdma/libnccl-net-uccl.so $(LIB_DIR)/; fi
	@if [ -f collective/rdma/librccl-net-uccl.so ]; then cp collective/rdma/librccl-net-uccl.so $(LIB_DIR)/; fi

ccl_efa: dirs
	$(MAKE) -C collective/efa clean || true
	$(MAKE) -C collective/efa $(MAKE_PARALLEL)
	@if [ -f collective/efa/libnccl-net-efa.so ]; then cp collective/efa/libnccl-net-efa.so $(LIB_DIR)/; fi
	@if ls thirdparty/nccl-sg/build/lib/libnccl*.so >/dev/null 2>&1; then \
		cp thirdparty/nccl-sg/build/lib/libnccl*.so $(LIB_DIR)/; \
	fi

p2p: dirs
	$(MAKE) -C p2p -f $(P2P_MAKEFILE) clean || true
	$(MAKE) -C p2p -f $(P2P_MAKEFILE) $(MAKE_PARALLEL) PYTHON=$(PYTHON)
	@if [ -f p2p/libuccl_p2p.so ]; then cp p2p/libuccl_p2p.so $(LIB_DIR)/; fi
	@if ls p2p/p2p.*.so >/dev/null 2>&1; then cp p2p/p2p.*.so $(OUTPUT_DIR)/; fi
	@if [ -f p2p/collective.py ]; then cp p2p/collective.py $(OUTPUT_DIR)/; fi
	@if [ -f p2p/utils.py ]; then cp p2p/utils.py $(OUTPUT_DIR)/; fi

ifeq ($(ROCM_ENABLED),1)
ep: dirs
	$(MAKE) -C ep -f Makefile.rocm \
		PYTHON=$(PYTHON) \
		BUILD_JOBS=$(BUILD_JOBS) \
		BULD_JOBS=$(BUILD_JOBS) \
		ROCM_DETECTED=$(ROCM_FLAG) \
		$(if $(ROCM_ARCH_LIST),ROCM_ARCH_LIST=$(ROCM_ARCH_LIST),) \
		$(if $(TORCH_CUDA_ARCH_LIST),TORCH_CUDA_ARCH_LIST=$(TORCH_CUDA_ARCH_LIST),) \
		$(if $(HIP_HOME),HIP_HOME=$(HIP_HOME),) \
		$(if $(CONDA_LIB_HOME),CONDA_LIB_HOME=$(CONDA_LIB_HOME),) \
		$(if $(USE_INTEL_RDMA_NIC),USE_INTEL_RDMA_NIC=$(USE_INTEL_RDMA_NIC),) \
		$(if $(PER_EXPERT_BATCHING),PER_EXPERT_BATCHING=$(PER_EXPERT_BATCHING),) \
		$(if $(DISABLE_BUILTIN_SHLF_SYNC),DISABLE_BUILTIN_SHLF_SYNC=$(DISABLE_BUILTIN_SHLF_SYNC),) \
		$(if $(DISABLE_AGGRESSIVE_PTX_INSTRS),DISABLE_AGGRESSIVE_PTX_INSTRS=$(DISABLE_AGGRESSIVE_PTX_INSTRS),) \
		OUTPUT_DIR=$(abspath $(OUTPUT_DIR))
else
ep: dirs
	cd ep && $(PYTHON) setup.py build
	rm -f $(EP_PY_DIR)/*.so
	find ep/build -name '*.so' -exec cp {} $(EP_PY_DIR)/ \;
endif

clean:
	-$(MAKE) -C collective/rdma -f $(RDMA_MAKEFILE) clean
	-$(MAKE) -C collective/efa clean
	-$(MAKE) -C p2p -f $(P2P_MAKEFILE) clean
	rm -rf ep/build
	rm -f $(LIB_DIR)/*.so
	rm -f $(OUTPUT_DIR)/p2p.*.so
	rm -f $(OUTPUT_DIR)/collective.py $(OUTPUT_DIR)/utils.py
	rm -f $(EP_PY_DIR)/*.so
