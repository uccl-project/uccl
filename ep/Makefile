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

ARCH := $(shell uname -m)
GPU_NAME ?= $(shell nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
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

# Python and nanobind configuration (stable ABI)
PYTHON            ?= python3
PYTHON_CONFIG      = $(PYTHON)-config
EXT_SUFFIX        := .abi3.so
NB_DIR            := $(shell $(PYTHON) -c "import nanobind as _nb, os; print(os.path.dirname(_nb.__file__))")
NB_INCLUDES       := -I$(NB_DIR)/include -I$(NB_DIR)/ext/robin_map/include -I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
NB_ABI_FLAGS      := -DPy_LIMITED_API=0x030C0000 -DNB_STABLE_ABI=1
NB_SRC_DIR        := $(NB_DIR)/src
NB_SRCS           := nb_internals.cpp nb_func.cpp nb_type.cpp nb_enum.cpp \
                     nb_ndarray.cpp nb_static_property.cpp common.cpp error.cpp \
                     trampoline.cpp implicit.cpp
NB_OBJECTS        := $(addprefix nb_,$(NB_SRCS:.cpp=.o))
PYTHON_LDFLAGS    := $(shell $(PYTHON_CONFIG) --ldflags)

TORCH_ROOT        := $(shell $(PYTHON) -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent)")
TORCH_INCLUDE     := $(TORCH_ROOT)/include
TORCH_INCLUDE_API := $(TORCH_ROOT)/include/torch/csrc/api/include
TORCH_LIBDIR      := $(TORCH_ROOT)/lib
TORCH_ABI         := $(shell $(PYTHON) -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
TORCH_INCS := -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE_API)

# Python installation path
PYTHON_SITE_PACKAGES := $(shell $(PYTHON) -c "import site; print(site.getsitepackages()[0])")
INSTALL_DIR          := $(PYTHON_SITE_PACKAGES)/uccl

# GPU arch (auto-detected, override with: make SM=90)
DETECTED_SM := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
SM ?= $(DETECTED_SM)

CXXFLAGS  := -O3 -std=c++17 -Wall -pthread -fPIC -fvisibility=hidden -Wno-interference-size -Wno-unused-function
LDFLAGS := -lpthread -lglog -libverbs -lnl-3 -lnl-route-3 -lnuma -Xlinker -rpath -Xlinker $(CUDA_PATH)/lib64
NVCCFLAGS := -O3 -std=c++17 -Xcompiler "-Wall -pthread -fPIC -fvisibility=hidden" -ccbin /usr/bin/g++ --expt-relaxed-constexpr
INCLUDES := -Iinclude -I$(CUDA_PATH)/include -I/usr/include -I../include

# Add Intel RDMA NIC support if USE_INTEL_RDMA_NIC=1
ifeq ($(USE_INTEL_RDMA_NIC),1)
  override CXXFLAGS += -DINTEL_RDMA_NIC
  override NVCCFLAGS += -DINTEL_RDMA_NIC
endif

# Enable per-expert batching path in dispatch-LL (default: disabled)
PER_EXPERT_BATCHING ?= 0
ifeq ($(PER_EXPERT_BATCHING),1)
  override CXXFLAGS += -DPER_EXPERT_BATCHING
  override NVCCFLAGS += -DPER_EXPERT_BATCHING
endif

CXXFLAGS  += $(EFA_CFLAGS) $(GH_CFLAGS) $(NORMAL_CFLAGS)
NVCCFLAGS += $(EFA_CFLAGS) $(GH_CFLAGS) $(NORMAL_CFLAGS)
LDFLAGS   += $(EFA_LDFLAGS)
INCLUDES  += $(EFA_CFLAGS) $(GH_CFLAGS) $(NORMAL_CFLAGS)

SRC_CPP := src/proxy.cpp src/rdma.cpp src/common.cpp src/peer_copy_worker.cpp src/uccl_proxy.cpp src/uccl_bench.cpp src/peer_copy_manager.cpp src/fifo.cpp
SRC_CU  := src/bench_kernel.cu src/peer_copy.cu src/internode_ll.cu src/internode.cu src/layout.cu src/intranode.cu src/ep_runtime.cu

OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU  := $(SRC_CU:.cu=.o)

SRC_BIND := src/uccl_ep.cc
OBJ_BIND := $(SRC_BIND:.cc=.o)
EP_EXT  := ep$(EXT_SUFFIX)
PYTARGET      := ep$(EXT_SUFFIX)

.PHONY: all py clean

all: $(EP_EXT)

# Compile nanobind source files for stable ABI
nb_%.o: $(NB_SRC_DIR)/%.cpp
	$(CXX) -O3 -std=c++17 -fPIC -fvisibility=hidden $(NB_ABI_FLAGS) $(NB_INCLUDES) -c $< -o $@

# C++ compilation rule with dependency generation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(TORCH_INCS) $(INCLUDES) $(NB_INCLUDES) $(NB_ABI_FLAGS) -MMD -MP -c $< -o $@
# CUDA compilation rule with dependency generation
%.o: %.cu
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) $(TORCH_INCS) $(INCLUDES) -MMD -MP -c $< -o $@

# Binding source compilation
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(TORCH_INCS) $(INCLUDES) $(NB_INCLUDES) $(NB_ABI_FLAGS) -MMD -MP -c $< -o $@

$(EP_EXT): $(OBJ_CPP) $(OBJ_CU) $(OBJ_BIND) $(NB_OBJECTS)
	$(CXX) $(CXXFLAGS) -shared -fPIC \
	    -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI) \
	    $(INCLUDES) $(NB_INCLUDES) $^ \
	    -L$(TORCH_LIBDIR) -Wl,-rpath,$(TORCH_LIBDIR) \
	    -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 \
	    -ltorch_python -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
	    -lcuda -lcudart $(LDFLAGS) $(PYTHON_LDFLAGS) -o $@

py: $(PYTARGET) $(EP_EXT)

# Install the module
install: $(EP_EXT)
	@mkdir -p $(INSTALL_DIR)
	@cp $(EP_EXT) $(INSTALL_DIR)/
	@echo "Installation complete. Modules installed as: $(INSTALL_DIR)/$(PYTARGET), $(INSTALL_DIR)/$(EP_EXT)"

# Clean all generated files
clean:
	rm -f $(OBJ_CPP) $(OBJ_CC) $(OBJ_CU) $(NB_OBJECTS) \
	      $(EP_EXT) \
	      *.d src/*.d bench/*.d src/*.o **/*.hip **/*_hip.*

# Automatically include dependency files if they exist
DEPS := $(OBJ_CPP:.o=.d) $(OBJ_CC:.o=.d) $(OBJ_CU:.o=.d) $(OBJ_BIND:.o=.d)
-include $(DEPS)
