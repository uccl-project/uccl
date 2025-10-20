CUDA_PATH ?= /usr/local/cuda
override CXX := /usr/bin/g++
NVCC      := $(CUDA_PATH)/bin/nvcc

EFA_HOME?=/opt/amazon/efa
ifeq ($(wildcard $(EFA_HOME)),)
  $(warning EFA not detected, building without EFA)
  EFA_CFLAGS :=
  EFA_LDFLAGS :=
else
  $(info EFA detected, building with EFA)
  EFA_CFLAGS := -DEFA -I$(EFA_HOME)/include
  EFA_LDFLAGS := -L$(EFA_HOME)/lib -lefa
endif

ifeq ($(MAKE_NORMAL_MODE),1)
  NORMAL_CFLAGS := -DUSE_NORMAL_MODE
  $(info Building with normal mode)
else
  NORMAL_CFLAGS :=
  $(info Building with low latency mode)
endif

ARCH := $(shell uname -m)
GPU_NAME := $(shell nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
CPU_IS_ARM64 := 0
GPU_IS_HOPPER := 0
ifeq ($(ARCH),aarch64)
  CPU_IS_ARM64 := 1
endif
ifneq (,$(findstring GH200,$(GPU_NAME)))
  GPU_IS_HOPPER := 1
endif
ifeq ($(and $(CPU_IS_ARM64),$(GPU_IS_HOPPER)),0)
  $(warning GH200 not detected, building without GH200)
  GH_CFLAGS :=
else
  $(info GH200 detected, building with GH200)
  GH_CFLAGS := -DUSE_GRACE_HOPPER
endif

PYTHON            ?= python3
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
EXT_SUFFIX        := $(shell $(PYTHON) -c "import sysconfig as s; print(s.get_config_var('EXT_SUFFIX') or '.so')")

TORCH_INCLUDE      := $(shell $(PYTHON) -c "import torch, pathlib; p=pathlib.Path(torch.__file__).parent; print(p/'include')")
TORCH_INCLUDE_API  := $(shell $(PYTHON) -c "import torch, pathlib; p=pathlib.Path(torch.__file__).parent; print(p/'include/torch/csrc/api/include')")
TORCH_LIBDIR       := $(shell $(PYTHON) -c "import torch, pathlib; p=pathlib.Path(torch.__file__).parent; print(p/'lib')")
TORCH_ABI          := $(shell $(PYTHON) -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
TORCH_INCS := -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE_API)

# Python installation path
PYTHON_SITE_PACKAGES := $(shell $(PYTHON) -c "import site; print(site.getsitepackages()[0])")
INSTALL_DIR          := $(PYTHON_SITE_PACKAGES)/uccl

# GPU arch (override with: make SM=90)
SM ?= 90

CXXFLAGS  := -O3 -std=c++17 -Wall -pthread -fPIC -fvisibility=hidden
LDFLAGS := -lpthread -lglog -libverbs -lnl-3 -lnl-route-3 -lnuma -Xlinker -rpath -Xlinker $(CUDA_PATH)/lib64
NVCCFLAGS := -O3 -std=c++17 -Xcompiler "-Wall -pthread -fPIC -fvisibility=hidden" -ccbin /usr/bin/g++ --expt-relaxed-constexpr
INCLUDES := -Iinclude -I$(CUDA_PATH)/include -I/usr/include -I../include 

CXXFLAGS  += $(EFA_CFLAGS) $(GH_CFLAGS) $(NORMAL_CFLAGS)
NVCCFLAGS += $(EFA_CFLAGS) $(GH_CFLAGS) $(NORMAL_CFLAGS)
LDFLAGS   += $(EFA_LDFLAGS)
INCLUDES  += $(EFA_CFLAGS) $(GH_CFLAGS) $(NORMAL_CFLAGS)

SRC_CPP := src/proxy.cpp src/rdma.cpp src/common.cpp src/peer_copy_worker.cpp src/uccl_proxy.cpp src/uccl_bench.cpp src/uccl_bench_fifo.cpp src/uccl_fifo_proxy.cpp src/peer_copy_manager.cpp src/fifo.cpp
SRC_CU  := src/gpu_kernel.cu src/gpu_kernel_fifo.cu src/peer_copy.cu src/py_cuda_shims.cu src/internode_ll.cu src/internode.cu src/layout.cu src/intranode.cu src/ep_runtime.cu

SRC_LOCAL  := bench/benchmark_local.cu
SRC_REMOTE := bench/benchmark_remote.cu
SRC_DUAL   := bench/benchmark_dual.cu

OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU  := $(SRC_CU:.cu=.o)

OBJ_LOCAL  := $(OBJ_CPP) $(OBJ_CU) $(SRC_LOCAL:.cu=.o)
OBJ_REMOTE := $(OBJ_CPP) $(OBJ_CU) $(SRC_REMOTE:.cu=.o)
OBJ_DUAL   := $(OBJ_CPP) $(OBJ_CU) $(SRC_DUAL:.cu=.o)

SRC_BIND := src/uccl_ep.cc
OBJ_BIND := $(SRC_BIND:.cc=.o)
EP_EXT  := ep$(EXT_SUFFIX)

TARGET_LOCAL  := benchmark_local
TARGET_REMOTE := benchmark_remote
TARGET_DUAL   := benchmark_dual
PYTARGET      := ep$(EXT_SUFFIX)

.PHONY: all py clean

all: $(TARGET_LOCAL) $(TARGET_REMOTE) $(TARGET_DUAL) $(EP_EXT)

# C++ compilation rule with dependency generation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(TORCH_INCS) $(INCLUDES) $(PYBIND11_INCLUDES) -MMD -MP -c $< -o $@
# CUDA compilation rule with dependency generation
%.o: %.cu
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) $(TORCH_INCS) $(INCLUDES) -MMD -MP -c $< -o $@

# Linking rules
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(TORCH_INCS) $(INCLUDES) $(PYBIND11_INCLUDES) -MMD -MP -c $< -o $@

$(TARGET_LOCAL): $(OBJ_LOCAL)
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) $(INCLUDES) $(OBJ_LOCAL) -lcuda -lcudart $(LDFLAGS) -o $@

$(TARGET_REMOTE): $(OBJ_REMOTE)
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) $(INCLUDES) $(OBJ_REMOTE) -lcuda -lcudart $(LDFLAGS) -o $@

$(TARGET_DUAL): $(OBJ_DUAL)
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) $(INCLUDES) $(OBJ_DUAL) -lcuda -lcudart $(LDFLAGS) -o $@

$(EP_EXT): $(OBJ_CPP) $(OBJ_CU) $(OBJ_BIND)
	$(CXX) $(CXXFLAGS) -shared -fPIC \
	    -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI) \
	    $(INCLUDES) $(PYBIND11_INCLUDES) $^ \
	    -L$(TORCH_LIBDIR) -Wl,-rpath,$(TORCH_LIBDIR) \
	    -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 \
	    -ltorch_python -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
	    -lcuda -lcudart $(LDFLAGS) -o $@

py: $(PYTARGET) $(EP_EXT)

# Install the module
install: $(EP_EXT)
	@mkdir -p $(INSTALL_DIR)
	@cp $(EP_EXT) $(INSTALL_DIR)/
	@echo "Installation complete. Modules installed as: $(INSTALL_DIR)/$(PYTARGET), $(INSTALL_DIR)/$(EP_EXT)"

# Clean all generated files
clean:
	rm -f $(OBJ_CPP) $(OBJ_CC) $(OBJ_CU) \
	      $(SRC_LOCAL:.cu=.o) $(SRC_REMOTE:.cu=.o) $(SRC_DUAL:.cu=.o) \
	      $(TARGET_LOCAL) $(TARGET_REMOTE) $(TARGET_DUAL) \
	      $(EP_EXT) \
	      *.d src/*.d bench/*.d src/*.o

# Automatically include dependency files if they exist
DEPS := $(OBJ_CPP:.o=.d) $(OBJ_CC:.o=.d) $(OBJ_CU:.o=.d) $(OBJ_BIND:.o=.d) \
        $(SRC_LOCAL:.cu=.d) $(SRC_REMOTE:.cu=.d) $(SRC_DUAL:.cu=.d) \
        $(EP_OBJ:.o=.d)
-include $(DEPS)