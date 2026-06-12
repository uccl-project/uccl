# Standalone UCCL-GIN build.
#
# This intentionally builds its own copied transport objects from
# experimental/uccl_gin/src instead of linking another target's objects. The
# code is still a copied transport snapshot and will be trimmed further, but the
# dependency boundary is already local to experimental/uccl_gin.
#
# Two platforms:
#   PLATFORM=cuda (default) -> NVIDIA + EFA, nvcc, NCCL-GIN reference path on.
#   PLATFORM=rocm           -> AMD/ROCm + RoCE, hipcc, NCCL-GIN reference off.
# On ROCm the inter-node atomic primitives (red_add_rel/put_tail_add/put_value)
# are NOT supported yet (EFA-shaped software-seq atomics); only put + quiet work.
# See AMD_SUPPORT_PLAN.md.

PLATFORM ?= cuda
PYTHON    ?= python3
NUM_PROXY_THS ?= 4

BUILD_DIR := build
TEST_DIR  := tests

PYTHON_INC := $(shell $(PYTHON) -c "import sysconfig;print(sysconfig.get_paths()['include'])")
PY_EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")

TRANSPORT_SRCS := \
  transport/proxy.cpp \
  transport/rdma.cpp \
  transport/common.cpp \
  transport/uccl_proxy.cpp \
  transport/fifo.cpp \
  transport/adaptive_sleeper.cc

TRANSPORT_OBJS := $(patsubst transport/%.cpp,$(BUILD_DIR)/transport/%.o,$(filter %.cpp,$(TRANSPORT_SRCS))) \
                  $(patsubst transport/%.cc,$(BUILD_DIR)/transport/%.o,$(filter %.cc,$(TRANSPORT_SRCS)))
ROOT_OBJS := $(BUILD_DIR)/context.o $(BUILD_DIR)/bindings.o

PY_PKG_DIR := python/uccl_gin
PY_EXT := $(PY_PKG_DIR)/_uccl_gin$(PY_EXT_SUFFIX)

# ===========================================================================
ifeq ($(PLATFORM),rocm)
# --------------------------- AMD / ROCm / RoCE -----------------------------
ROCM_HOME ?= /opt/rocm
GFX       ?= gfx942
UCCL_GIN_WITH_NCCL_GIN := 0
HIPCC     ?= $(ROCM_HOME)/bin/hipcc

# hipcc compiles host (.cpp/.cc) and device (.cu) sources alike.
CC_HOST   := $(HIPCC)
CC_DEV    := $(HIPCC)

MPICXX ?= mpicxx
MPI_PREFIX := $(patsubst %/bin/mpicxx,%,$(shell which $(MPICXX) 2>/dev/null))
MPI_INC    := $(shell $(MPICXX) --showme:compile 2>/dev/null)
MPI_LIB    := $(shell $(MPICXX) --showme:link 2>/dev/null)

# No EFA, no SOFTWARE_ORDERING => the plain RC ibv_post_send WRITE path.
DEFS := -DUCCL_NUM_PROXY_THS=$(NUM_PROXY_THS) \
        -DUCCL_GIN_WITH_NCCL_GIN=$(UCCL_GIN_WITH_NCCL_GIN)
INCLUDES := -I. -Itransport -I../../include -I$(ROCM_HOME)/include -I$(PYTHON_INC) \
            $(MPI_INC)

HOSTFLAGS := -O3 -std=c++17 -fPIC -fvisibility=hidden $(DEFS) $(INCLUDES) \
             -Wno-unused-result -Wno-unused-variable
DEVFLAGS  := -O3 -std=c++17 --offload-arch=$(GFX) -fPIC -fvisibility=hidden \
             $(DEFS) $(INCLUDES) -Wno-unused-result -Wno-unused-variable

LDFLAGS := -L$(ROCM_HOME)/lib -lamdhip64 $(MPI_LIB) \
           -libverbs -lnl-3 -lnl-route-3 -lnuma -lpthread

COMPILE_HOST = $(CC_HOST) $(HOSTFLAGS) -MMD -MP -c $< -o $@
COMPILE_DEV  = $(CC_DEV)  $(DEVFLAGS)  -MMD -MP -c $< -o $@
LINK_EXE     = $(CC_DEV)  $(DEVFLAGS)
LINK_SO      = $(CC_HOST) -shared

else
# --------------------------- NVIDIA / CUDA / EFA ---------------------------
CUDA_HOME ?= /usr/local/cuda-13.0
EFA_HOME  ?= /opt/amazon/efa
SM ?= 90
UCCL_GIN_WITH_NCCL_GIN ?= 1

NVCC := $(CUDA_HOME)/bin/nvcc
CXX  ?= /usr/bin/g++

ifneq ($(wildcard /opt/amazon/openmpi/bin/mpicxx),)
MPICXX ?= /opt/amazon/openmpi/bin/mpicxx
else ifneq ($(wildcard /opt/amazon/openmpi5/bin/mpicxx),)
MPICXX ?= /opt/amazon/openmpi5/bin/mpicxx
else
MPICXX ?= mpicxx
endif

MPI_PREFIX := $(patsubst %/bin/mpicxx,%,$(MPICXX))
MPI_INC    := -I$(MPI_PREFIX)/include
MPI_LIB    := -L$(MPI_PREFIX)/lib -lmpi -Wl,-rpath,$(MPI_PREFIX)/lib

NCCL_INC := $(shell $(PYTHON) -c "import nvidia.nccl,pathlib;print(pathlib.Path(nvidia.nccl.__path__[0])/'include')")
NCCL_LIB := $(shell $(PYTHON) -c "import nvidia.nccl,pathlib;print(pathlib.Path(nvidia.nccl.__path__[0])/'lib')")
NCCL_SO  := $(notdir $(firstword $(wildcard $(NCCL_LIB)/libnccl.so $(NCCL_LIB)/libnccl.so.*)))

DEFS := -DEFA -DNCCL_CHECK_CUDACC=1 -DUCCL_NUM_PROXY_THS=$(NUM_PROXY_THS) \
        -DUCCL_GIN_WITH_NCCL_GIN=$(UCCL_GIN_WITH_NCCL_GIN)
INCLUDES := -I. -Itransport -I$(NCCL_INC) -I../../include -I$(CUDA_HOME)/include -I$(PYTHON_INC) \
            -I$(EFA_HOME)/include $(MPI_INC)

HOSTFLAGS := -O3 -std=c++17 -Wall -pthread -fPIC -fvisibility=hidden $(DEFS) $(INCLUDES)
DEVFLAGS  := -O3 -std=c++17 -arch=sm_$(SM) --expt-relaxed-constexpr $(DEFS) $(INCLUDES) \
             -Xcompiler "-Wall -pthread -fPIC -fvisibility=hidden"

NVCC_RPATH := -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64 \
              -Xlinker -rpath -Xlinker $(NCCL_LIB) \
              -Xlinker -rpath -Xlinker $(MPI_PREFIX)/lib

LDFLAGS := -L$(CUDA_HOME)/lib64 $(NVCC_RPATH) \
           -L$(EFA_HOME)/lib -L$(NCCL_LIB) \
           -L$(MPI_PREFIX)/lib -lmpi -l:$(NCCL_SO) -lefa -libverbs -lnl-3 -lnl-route-3 \
           -lnuma -lcudart -lcuda -lpthread

COMPILE_HOST = $(CXX) $(HOSTFLAGS) -MMD -MP -c $< -o $@
COMPILE_DEV  = $(NVCC) $(DEVFLAGS) -MMD -MP -c $< -o $@
LINK_EXE     = $(NVCC) $(DEVFLAGS)
LINK_SO      = $(CXX) -shared
endif
# ===========================================================================

all: $(BUILD_DIR)/uccl_gin_microbench $(PY_EXT)

# put/quiet-only smoke (no NCCL, no atomics) — the AMD/ROCm validation vehicle.
smoke: $(BUILD_DIR)/uccl_gin_put_quiet_smoke

$(BUILD_DIR):
	mkdir -p $@
	mkdir -p $(BUILD_DIR)/transport

$(BUILD_DIR)/transport/%.o: transport/%.cpp | $(BUILD_DIR)
	$(COMPILE_HOST)

$(BUILD_DIR)/transport/%.o: transport/%.cc | $(BUILD_DIR)
	$(COMPILE_HOST)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(COMPILE_HOST)

$(BUILD_DIR)/uccl_gin_microbench: $(TEST_DIR)/microbench.cu $(TRANSPORT_OBJS) $(BUILD_DIR)/context.o
	$(LINK_EXE) $< $(TRANSPORT_OBJS) $(BUILD_DIR)/context.o -o $@ $(LDFLAGS)

$(BUILD_DIR)/uccl_gin_put_quiet_smoke: $(TEST_DIR)/put_quiet_smoke.cu $(TRANSPORT_OBJS) $(BUILD_DIR)/context.o
	$(LINK_EXE) $< $(TRANSPORT_OBJS) $(BUILD_DIR)/context.o -o $@ $(LDFLAGS)

$(PY_EXT): $(BUILD_DIR)/bindings.o $(BUILD_DIR)/context.o $(TRANSPORT_OBJS) | $(BUILD_DIR)
	$(LINK_SO) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(PY_PKG_DIR)/_uccl_gin*.so

.PHONY: all clean smoke

-include $(TRANSPORT_OBJS:.o=.d) $(BUILD_DIR)/bindings.d $(BUILD_DIR)/context.d
