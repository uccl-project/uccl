SHELL := /bin/bash

PYTHON ?= python3

# Parallelism for sub-makes (e.g. `make -j16` style).
BUILD_JOBS ?= 32
MAKE_PARALLEL := -j$(BUILD_JOBS)

# Build platform: set USE_ROCM=1 to build the ROCm/HIP variants of
# collective/rdma and p2p. Anything else (the default) builds CUDA.
USE_ROCM ?= 0

# When invoked from setup.py the wheel staging directory is passed as
# BUILD_LIB. In that case build artefacts are dropped directly into the
# wheel layout (BUILD_LIB/uccl, BUILD_LIB/uccl/lib, BUILD_LIB/uccl/ep) so
# no extra Python post-processing is needed. Standalone `make` invocations
# leave artefacts in the in-tree uccl/ and ep/python/uccl_ep/ directories.
BUILD_LIB ?=
ifeq ($(BUILD_LIB),)
INSTALL_PREFIX := $(CURDIR)/uccl
else
INSTALL_PREFIX := $(abspath $(BUILD_LIB))/uccl
endif
LIB_DIR := $(INSTALL_PREFIX)/lib
EP_PY_DIR := ep/python/uccl_ep

# EP build environment variables (forwarded to ep/setup.py via the env).
USE_INTEL_RDMA_NIC ?= 0

# Pick CUDA or ROCm sub-makefiles for the native modules. ``USE_ROCM`` is a
# string ("0"/"1"), so use ``ifeq`` rather than ``$(if ...)`` (any non-empty
# string would be truthy and pick the ROCm path even when USE_ROCM=0).
ifeq ($(USE_ROCM),1)
RDMA_MAKEFILE := $(if $(wildcard collective/rdma/Makefile.rocm),Makefile.rocm,Makefile)
P2P_MAKEFILE := $(if $(wildcard p2p/Makefile.rocm),Makefile.rocm,Makefile)
else
RDMA_MAKEFILE := Makefile
P2P_MAKEFILE := Makefile
endif

MODULE_TARGETS := ccl_rdma p2p ep

.PHONY: all dirs clean $(MODULE_TARGETS) ccl_efa

all: dirs $(MODULE_TARGETS)

dirs:
	mkdir -p $(LIB_DIR)
	@if [ -n "$(BUILD_LIB)" ]; then mkdir -p $(INSTALL_PREFIX)/ep; fi

ccl_rdma: dirs
	$(MAKE) -C collective/rdma -f $(RDMA_MAKEFILE) $(MAKE_PARALLEL)
	@if [ -f collective/rdma/libnccl-net-uccl.so ]; then cp collective/rdma/libnccl-net-uccl.so $(LIB_DIR)/; fi
	@if [ -f collective/rdma/librccl-net-uccl.so ]; then cp collective/rdma/librccl-net-uccl.so $(LIB_DIR)/; fi

ccl_efa: dirs
	$(MAKE) -C collective/efa $(MAKE_PARALLEL)
	@if [ -f collective/efa/libnccl-net-efa.so ]; then cp collective/efa/libnccl-net-efa.so $(LIB_DIR)/; fi
	@if ls thirdparty/nccl-sg/build/lib/libnccl*.so >/dev/null 2>&1; then \
		cp thirdparty/nccl-sg/build/lib/libnccl*.so $(LIB_DIR)/; \
	fi

p2p: dirs
	$(MAKE) -C p2p -f $(P2P_MAKEFILE) $(MAKE_PARALLEL) PYTHON=$(PYTHON)
	@if [ -f p2p/libuccl_p2p.so ]; then cp p2p/libuccl_p2p.so $(LIB_DIR)/; fi
	@if ls p2p/p2p.*.so >/dev/null 2>&1; then cp p2p/p2p.*.so $(INSTALL_PREFIX)/; fi
	@if [ -f p2p/collective.py ]; then cp p2p/collective.py $(INSTALL_PREFIX)/; fi
	@if [ -f p2p/utils.py ]; then cp p2p/utils.py $(INSTALL_PREFIX)/; fi

ep: dirs
	cd ep && USE_INTEL_RDMA_NIC=$(USE_INTEL_RDMA_NIC) $(PYTHON) -u setup.py -v build
	rm -f $(EP_PY_DIR)/*.so
	mkdir -p $(EP_PY_DIR)
	find ep/build -name '*.so' -exec cp {} $(EP_PY_DIR)/ \;
	@if [ -n "$(BUILD_LIB)" ]; then \
		mkdir -p $(INSTALL_PREFIX)/ep; \
		cp $(EP_PY_DIR)/*.so $(INSTALL_PREFIX)/ep/; \
	fi

clean:
	-$(MAKE) -C collective/rdma -f $(RDMA_MAKEFILE) clean
	-$(MAKE) -C collective/efa clean
	-$(MAKE) -C p2p -f $(P2P_MAKEFILE) clean
	rm -rf ep/build
	rm -f $(LIB_DIR)/*.so
	rm -f $(INSTALL_PREFIX)/p2p.*.so
	rm -f $(INSTALL_PREFIX)/collective.py $(INSTALL_PREFIX)/utils.py
	rm -f $(EP_PY_DIR)/*.so
