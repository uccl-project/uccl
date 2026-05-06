SHELL := /bin/bash

PYTHON ?= python3

# Parallelism for sub-makes.
BUILD_JOBS ?= 32
MAKE_PARALLEL := -j$(BUILD_JOBS)

ARCH ?= $(shell uname -m)

# TARGET: cu* (CUDA, default) | roc6 / roc7 (ROCm) | therock (TheRock ROCm).
# USE_ROCM (0/1) overrides TARGET when explicitly set.
TARGET ?=
USE_ROCM ?=

ifeq ($(USE_ROCM),)
ifneq (,$(filter roc% therock,$(TARGET)))
USE_ROCM := 1
else
USE_ROCM := 0
endif
endif

# Predicates derived from TARGET / ARCH.
IS_THEROCK := $(if $(filter therock,$(TARGET)),1,0)
IS_ROC6    := $(if $(filter roc6,$(TARGET)),1,0)
IS_AARCH64 := $(if $(filter aarch64,$(ARCH)),1,0)

# IS_EFA=1 swaps ccl_rdma -> ccl_efa in the ``all`` target.
IS_EFA ?= 0

# Optional features.
USE_DIETGPU       ?= 0
USE_INTEL_RDMA_NIC ?= 0

# BUILD_LIB: wheel staging dir passed by setup.py. When set, artefacts go
# directly to BUILD_LIB/uccl/{,lib,ep}; otherwise they stay in-tree.
BUILD_LIB ?=
ifeq ($(BUILD_LIB),)
INSTALL_PREFIX := $(CURDIR)/uccl
else
INSTALL_PREFIX := $(abspath $(BUILD_LIB))/uccl
endif
LIB_DIR := $(INSTALL_PREFIX)/lib
EP_PY_DIR := ep/python/uccl_ep

# Pick the right per-module Makefile for the toolchain.
ifeq ($(TARGET),therock)
RDMA_MAKEFILE := $(if $(wildcard collective/rdma/Makefile.therock),Makefile.therock,Makefile.rocm)
P2P_MAKEFILE  := $(if $(wildcard p2p/Makefile.therock),Makefile.therock,Makefile.rocm)
else ifeq ($(USE_ROCM),1)
RDMA_MAKEFILE := $(if $(wildcard collective/rdma/Makefile.rocm),Makefile.rocm,Makefile)
P2P_MAKEFILE  := $(if $(wildcard p2p/Makefile.rocm),Makefile.rocm,Makefile)
else
RDMA_MAKEFILE := Makefile
P2P_MAKEFILE  := Makefile
endif

UKERNEL_MAKEFILE := $(if $(filter 1,$(USE_ROCM)),Makefile.rocm,Makefile)

ifeq ($(IS_EFA),1)
COLLECTIVE_TARGET := ccl_efa
else
COLLECTIVE_TARGET := ccl_rdma
endif

MODULE_TARGETS := $(COLLECTIVE_TARGET) p2p ep

# BUILD_TYPE picks what ``all`` builds:
#   all (default) -> $(MODULE_TARGETS)
#   p2p_ep        -> p2p + ep
#   anything else -> the value verbatim (e.g. ccl_rdma, p2p, ep, ukernel).
BUILD_TYPE ?= all
ifeq ($(BUILD_TYPE),all)
ALL_TARGETS := $(MODULE_TARGETS)
else ifeq ($(BUILD_TYPE),p2p_ep)
ALL_TARGETS := p2p ep
else
ALL_TARGETS := $(BUILD_TYPE)
endif

.PHONY: all dirs clean ccl_rdma ccl_efa p2p ep ukernel \
        _rccl_header _nccl_sg _dietgpu

all: dirs $(ALL_TARGETS)

dirs:
	mkdir -p $(LIB_DIR)
	@if [ -n "$(BUILD_LIB)" ]; then mkdir -p $(INSTALL_PREFIX)/ep; fi

############################################################
# Pre-build dependencies
############################################################

# Generate nccl.h via rccl cmake (ROCm only; no-op for CUDA).
RCCL_NCCL_HEADER := thirdparty/rccl/build/release/include/nccl.h

_rccl_header:
ifeq ($(USE_ROCM),1)
	@if [ ! -f "$(RCCL_NCCL_HEADER)" ]; then \
	    if [ "$(IS_THEROCK)" = "1" ]; then \
	        echo "[make] generating nccl.h via rccl cmake (TheRock toolchain)"; \
	        cd thirdparty/rccl && CXX=hipcc cmake -B build/release -S . \
	            -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
	            -DCMAKE_PREFIX_PATH=$$(rocm-sdk path --cmake) \
	            -DROCM_PATH=$$(rocm-sdk path --root) \
	            -DHIP_PLATFORM=amd >/dev/null 2>&1 || true; \
	    else \
	        echo "[make] generating nccl.h via rccl cmake (/opt/rocm hipcc)"; \
	        cd thirdparty/rccl && CXX=/opt/rocm/bin/hipcc cmake -B build/release -S . \
	            -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF >/dev/null 2>&1 || true; \
	    fi; \
	fi
else
	@true
endif

# Custom NCCL fork required by ccl_efa.
NCCL_SG_LIB := thirdparty/nccl-sg/build/lib/libnccl.so
NVCC_GENCODE ?= -gencode=arch=compute_90,code=sm_90

_nccl_sg:
ifeq ($(USE_ROCM),1)
	@echo "[make] USE_ROCM=1 -> nccl-sg skipped"
else ifeq ($(IS_AARCH64),1)
	@echo "[make] ARCH=aarch64 -> nccl-sg skipped"
else
	$(MAKE) -C thirdparty/nccl-sg src.build $(MAKE_PARALLEL) \
	    NVCC_GENCODE="$(NVCC_GENCODE)" \
	    USE_INTEL_RDMA_NIC=$(USE_INTEL_RDMA_NIC)
endif

# Optional dietgpu dependency for p2p. ROCm needs a Python build pass first
# to generate HIP shims; CUDA goes straight to Makefile.cuda.
DIETGPU_FLOAT_DIR := thirdparty/dietgpu/dietgpu/float
DIETGPU_LIB       := $(DIETGPU_FLOAT_DIR)/libdietgpu_float.so

# TORCH_CUDA_ARCH_LIST=9.0+PTX -> sm_90 for dietgpu CUDA build.
# Default to 9.0 when unset (mirrors build_inner.sh's ${TORCH_CUDA_ARCH_LIST:-9.0}).
DIETGPU_TORCH_ARCH := $(if $(strip $(TORCH_CUDA_ARCH_LIST)),$(TORCH_CUDA_ARCH_LIST),9.0)
DIETGPU_GPU_ARCH := sm_$(subst .,,$(firstword $(subst +PTX,,$(DIETGPU_TORCH_ARCH))))

_dietgpu:
ifeq ($(USE_DIETGPU),1)
ifeq ($(USE_ROCM),1)
	@echo "[make] building dietgpu (ROCm)"
	cd thirdparty/dietgpu && rm -rf build/ && $(PYTHON) setup.py build
	$(MAKE) -C $(DIETGPU_FLOAT_DIR) -f Makefile.rocm clean
	$(MAKE) -C $(DIETGPU_FLOAT_DIR) -f Makefile.rocm $(MAKE_PARALLEL) \
	    GPU_ARCH=$(TORCH_CUDA_ARCH_LIST)
else
	@echo "[make] building dietgpu (CUDA, GPU_ARCH=$(DIETGPU_GPU_ARCH))"
	$(MAKE) -C $(DIETGPU_FLOAT_DIR) -f Makefile.cuda clean
	$(MAKE) -C $(DIETGPU_FLOAT_DIR) -f Makefile.cuda $(MAKE_PARALLEL) \
	    GPU_ARCH=$(DIETGPU_GPU_ARCH)
endif
else
	@true
endif

############################################################
# Module targets
############################################################

ccl_rdma: dirs $(if $(filter 1,$(USE_ROCM)),_rccl_header,)
	@echo "[make] ccl_rdma TARGET=$(TARGET) ARCH=$(ARCH) IS_EFA=$(IS_EFA)"
ifeq ($(USE_ROCM),0)
	$(MAKE) -C collective/rdma -f $(RDMA_MAKEFILE) clean
	$(MAKE) -C collective/rdma -f $(RDMA_MAKEFILE) $(MAKE_PARALLEL) \
	    USE_INTEL_RDMA_NIC=$(USE_INTEL_RDMA_NIC)
	@if [ -f collective/rdma/libnccl-net-uccl.so ]; then \
	    cp collective/rdma/libnccl-net-uccl.so $(LIB_DIR)/; \
	fi
else ifeq ($(IS_AARCH64),1)
	@echo "[make] aarch64 + ROCm -> ccl_rdma skipped"
else ifeq ($(IS_THEROCK),1)
	$(MAKE) -C collective/rdma -f $(RDMA_MAKEFILE) clean
	$(MAKE) -C collective/rdma -f $(RDMA_MAKEFILE) $(MAKE_PARALLEL) \
	    HIP_HOME=$$(rocm-sdk path --root) \
	    CONDA_LIB_HOME=$$VIRTUAL_ENV/lib
	@if [ -f collective/rdma/librccl-net-uccl.so ]; then \
	    cp collective/rdma/librccl-net-uccl.so $(LIB_DIR)/; \
	fi
else
	$(MAKE) -C collective/rdma -f $(RDMA_MAKEFILE) clean
	$(MAKE) -C collective/rdma -f $(RDMA_MAKEFILE) $(MAKE_PARALLEL)
	@if [ -f collective/rdma/librccl-net-uccl.so ]; then \
	    cp collective/rdma/librccl-net-uccl.so $(LIB_DIR)/; \
	fi
endif

ccl_efa: dirs
	@echo "[make] ccl_efa TARGET=$(TARGET) ARCH=$(ARCH)"
ifeq ($(USE_ROCM),1)
	@echo "[make] ROCm -> ccl_efa skipped"
else ifeq ($(IS_AARCH64),1)
	@echo "[make] aarch64 -> ccl_efa skipped"
else
	$(MAKE) -C collective/efa clean
	$(MAKE) -C collective/efa $(MAKE_PARALLEL) \
	    USE_INTEL_RDMA_NIC=$(USE_INTEL_RDMA_NIC)
	$(MAKE) _nccl_sg
	@if [ -f collective/efa/libnccl-net-efa.so ]; then \
	    cp collective/efa/libnccl-net-efa.so $(LIB_DIR)/; \
	fi
	@if [ -f $(NCCL_SG_LIB) ]; then \
	    cp $(NCCL_SG_LIB) $(LIB_DIR)/libnccl-efa.so; \
	fi
endif

p2p: dirs $(if $(filter 1,$(USE_DIETGPU)),_dietgpu,)
	@echo "[make] p2p TARGET=$(TARGET) ARCH=$(ARCH) USE_DIETGPU=$(USE_DIETGPU)"
ifeq ($(IS_THEROCK),1)
	$(MAKE) -C p2p -f $(P2P_MAKEFILE) clean PYTHON=$(PYTHON)
	$(MAKE) -C p2p -f $(P2P_MAKEFILE) $(MAKE_PARALLEL) PYTHON=$(PYTHON) \
	    HIP_HOME=$$(rocm-sdk path --root) \
	    CONDA_LIB_HOME=$$VIRTUAL_ENV/lib
else
	$(MAKE) -C p2p -f $(P2P_MAKEFILE) clean PYTHON=$(PYTHON)
	$(MAKE) -C p2p -f $(P2P_MAKEFILE) $(MAKE_PARALLEL) PYTHON=$(PYTHON)
endif
	@if [ "$(USE_DIETGPU)" = "1" ] && [ -f $(DIETGPU_LIB) ]; then \
	    cp $(DIETGPU_LIB) $(LIB_DIR)/; \
	fi
	@if [ -f p2p/libuccl_p2p.so ]; then cp p2p/libuccl_p2p.so $(LIB_DIR)/; fi
	@if ls p2p/p2p.*.so >/dev/null 2>&1; then cp p2p/p2p.*.so $(INSTALL_PREFIX)/; fi
	@if [ -f p2p/collective.py ]; then cp p2p/collective.py $(INSTALL_PREFIX)/; fi
	@if [ -f p2p/utils.py ]; then cp p2p/utils.py $(INSTALL_PREFIX)/; fi

# EP artefacts go to ep/python/uccl_ep/ so uccl.ep stays a self-contained sub-package.
ep: dirs
	@echo "[make] ep TARGET=$(TARGET) ARCH=$(ARCH)"
ifeq ($(IS_ROC6),1)
	@echo "ERROR: EP requires roc7 (ROCm 7); roc6 is not supported." >&2
	@exit 1
else ifeq ($(IS_THEROCK),1)
	@echo "[make] therock -> EP skipped (no GPU-driven support yet)"
else
	cd ep && USE_INTEL_RDMA_NIC=$(USE_INTEL_RDMA_NIC) $(PYTHON) -u setup.py -v build
	rm -f $(EP_PY_DIR)/*.so
	mkdir -p $(EP_PY_DIR)
	find ep/build -name '*.so' -exec cp {} $(EP_PY_DIR)/ \;
	@if [ -n "$(BUILD_LIB)" ]; then \
	    mkdir -p $(INSTALL_PREFIX)/ep; \
	    cp $(EP_PY_DIR)/*.so $(INSTALL_PREFIX)/ep/; \
	fi
endif

ukernel: dirs
	@echo "[make] ukernel TARGET=$(TARGET) ARCH=$(ARCH)"
ifeq ($(IS_THEROCK),1)
	@echo "[make] therock -> ukernel skipped (matches build_inner.sh)"
else ifeq ($(USE_ROCM),1)
ifeq ($(IS_AARCH64),1)
	@echo "[make] aarch64 + ROCm -> ukernel skipped"
else
	$(MAKE) -C experimental/ukernel -f $(UKERNEL_MAKEFILE) clean
	$(MAKE) -C experimental/ukernel -f $(UKERNEL_MAKEFILE) $(MAKE_PARALLEL)
endif
else
	$(MAKE) -C experimental/ukernel -f $(UKERNEL_MAKEFILE) clean
	$(MAKE) -C experimental/ukernel -f $(UKERNEL_MAKEFILE) $(MAKE_PARALLEL)
endif
	@if ls experimental/ukernel/*ukernel*.so >/dev/null 2>&1; then \
	    cp experimental/ukernel/*ukernel*.so $(LIB_DIR)/; \
	fi

clean:
	-$(MAKE) -C collective/rdma -f $(RDMA_MAKEFILE) clean
	-$(MAKE) -C collective/efa clean
	-$(MAKE) -C p2p -f $(P2P_MAKEFILE) clean PYTHON=$(PYTHON)
	-$(MAKE) -C experimental/ukernel -f $(UKERNEL_MAKEFILE) clean 2>/dev/null
	rm -rf ep/build
	rm -f $(LIB_DIR)/*.so
	rm -f $(INSTALL_PREFIX)/p2p.*.so
	rm -f $(INSTALL_PREFIX)/collective.py $(INSTALL_PREFIX)/utils.py
	rm -f $(EP_PY_DIR)/*.so
