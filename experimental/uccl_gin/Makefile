# Standalone UCCL-GIN build.
#
# This intentionally builds its own copied transport objects from
# experimental/uccl_gin/src instead of linking another target's objects. The
# code is still a copied transport snapshot and will be trimmed further, but the
# dependency boundary is already local to experimental/uccl_gin.

CUDA_HOME ?= /usr/local/cuda-13.0
EFA_HOME  ?= /opt/amazon/efa
PYTHON    ?= python3
NUM_PROXY_THS ?= 4
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
PYTHON_INC := $(shell $(PYTHON) -c "import sysconfig;print(sysconfig.get_paths()['include'])")
PY_EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")

BUILD_DIR := build
SRC_DIR   := src
TEST_DIR  := tests/cpp
PY_PKG_DIR := python/uccl_gin

TRANSPORT_SRCS := \
  $(SRC_DIR)/proxy.cpp \
  $(SRC_DIR)/rdma.cpp \
  $(SRC_DIR)/common.cpp \
  $(SRC_DIR)/uccl_proxy.cpp \
  $(SRC_DIR)/context.cpp \
  $(SRC_DIR)/adaptive_sleeper.cc

TRANSPORT_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(filter %.cpp,$(TRANSPORT_SRCS))) \
                  $(patsubst $(SRC_DIR)/%.cc,$(BUILD_DIR)/%.o,$(filter %.cc,$(TRANSPORT_SRCS)))

DEFS := -DEFA -DNCCL_CHECK_CUDACC=1 -DUCCL_NUM_PROXY_THS=$(NUM_PROXY_THS) \
        -DUCCL_GIN_WITH_NCCL_GIN=$(UCCL_GIN_WITH_NCCL_GIN)
INCLUDES := -I$(NCCL_INC) -Iinclude -I../../include -I$(CUDA_HOME)/include -I$(PYTHON_INC) \
            -I$(EFA_HOME)/include $(MPI_INC)

CXXFLAGS := -O3 -std=c++17 -Wall -pthread -fPIC -fvisibility=hidden $(DEFS) $(INCLUDES)
NVCCFLAGS := -O3 -std=c++17 -arch=sm_$(SM) --expt-relaxed-constexpr $(DEFS) $(INCLUDES) \
             -Xcompiler "-Wall -pthread -fPIC -fvisibility=hidden"

NVCC_RPATH := -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64 \
              -Xlinker -rpath -Xlinker $(NCCL_LIB) \
              -Xlinker -rpath -Xlinker $(MPI_PREFIX)/lib

LDFLAGS := -L$(CUDA_HOME)/lib64 $(NVCC_RPATH) \
           -L$(EFA_HOME)/lib -L$(NCCL_LIB) \
           -L$(MPI_PREFIX)/lib -lmpi -l:$(NCCL_SO) -lefa -libverbs -lnl-3 -lnl-route-3 \
           -lnuma -lcudart -lcuda -lpthread

PY_EXT := $(PY_PKG_DIR)/_uccl_gin$(PY_EXT_SUFFIX)

all: $(BUILD_DIR)/uccl_gin_microbench $(PY_EXT)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/uccl_gin_microbench: $(TEST_DIR)/microbench.cu $(TRANSPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) $< $(TRANSPORT_OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/bindings.o: $(SRC_DIR)/bindings.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(PY_EXT): $(BUILD_DIR)/bindings.o $(TRANSPORT_OBJS) | $(BUILD_DIR)
	$(CXX) -shared $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(PY_PKG_DIR)/_uccl_gin*.so

.PHONY: all clean

-include $(TRANSPORT_OBJS:.o=.d) $(BUILD_DIR)/bindings.d
