CUDA_PATH ?= /usr/local/cuda
override CXX := /usr/bin/g++
NVCC      := $(CUDA_PATH)/bin/nvcc

PYTHON            ?= python3
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
EXT_SUFFIX        := $(shell $(PYTHON) -c "import sysconfig as s; print(s.get_config_var('EXT_SUFFIX') or '.so')")

TORCH_INCLUDE      := $(shell $(PYTHON) -c "import torch, pathlib; p=pathlib.Path(torch.__file__).parent; print(p/'include')")
TORCH_INCLUDE_API  := $(shell $(PYTHON) -c "import torch, pathlib; p=pathlib.Path(torch.__file__).parent; print(p/'include/torch/csrc/api/include')")
TORCH_LIBDIR       := $(shell $(PYTHON) -c "import torch, pathlib; p=pathlib.Path(torch.__file__).parent; print(p/'lib')")
TORCH_ABI          := $(shell $(PYTHON) -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

# GPU arch (override with: make SM=90)
SM ?= 90

CXXFLAGS  := -O3 -std=c++17 -Wall -pthread -fPIC -fvisibility=hidden
LDFLAGS := -lpthread -lglog -libverbs -lnl-3 -lnl-route-3 -Xlinker -rpath -Xlinker $(CUDA_PATH)/lib64
NVCCFLAGS := -O3 -std=c++17 -Xcompiler "-Wall -pthread -fPIC -fvisibility=hidden" -ccbin /usr/bin/g++

INCLUDES := -Iinclude -I$(CUDA_PATH)/include -I/usr/include -I../include

SRC_CPP := src/proxy.cpp src/rdma.cpp src/common.cpp src/peer_copy_worker.cpp src/uccl_proxy.cpp src/uccl_bench.cpp src/peer_copy_manager.cpp
SRC_CU  := src/gpu_kernel.cu src/peer_copy.cu src/py_cuda_shims.cu

SRC_LOCAL  := bench/benchmark_local.cu
SRC_REMOTE := bench/benchmark_remote.cu
SRC_DUAL   := bench/benchmark_dual.cu

PYBIND_SRC := src/pybind_proxy.cc
OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU  := $(SRC_CU:.cu=.o)

OBJ_LOCAL  := $(OBJ_CPP) $(OBJ_CU) $(SRC_LOCAL:.cu=.o)
OBJ_REMOTE := $(OBJ_CPP) $(OBJ_CU) $(SRC_REMOTE:.cu=.o)
OBJ_DUAL   := $(OBJ_CPP) $(OBJ_CU) $(SRC_DUAL:.cu=.o)

PYBIND_OBJ := $(PYBIND_SRC:.cc=.o)

UCCL_EP_SRC := src/uccl_ep.cc
UCCL_EP_OBJ := $(UCCL_EP_SRC:.cc=.o)
UCCL_EP_EXT := uccl_ep$(EXT_SUFFIX)

$(UCCL_EP_OBJ): $(UCCL_EP_SRC)
	$(CXX) $(CXXFLAGS) -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI) \
	    -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE_API) \
	    $(INCLUDES) $(PYBIND11_INCLUDES) -MMD -MP -c $< -o $@

$(UCCL_EP_EXT): $(UCCL_EP_OBJ)
	$(CXX) $(CXXFLAGS) -shared -fPIC \
	    -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI) \
	    $(INCLUDES) $(PYBIND11_INCLUDES) $^ \
	    -L$(TORCH_LIBDIR) -Wl,-rpath,$(TORCH_LIBDIR) \
	    -ltorch_python \
	    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
	    -o $@

TARGET_LOCAL  := benchmark_local
TARGET_REMOTE := benchmark_remote
TARGET_DUAL   := benchmark_dual
PYTARGET      := gpu_driven$(EXT_SUFFIX)

.PHONY: all py clean

all: $(TARGET_LOCAL) $(TARGET_REMOTE) $(TARGET_DUAL) $(PYTARGET) $(UCCL_EP_EXT)

# C++ compilation rule with dependency generation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PYBIND11_INCLUDES) -MMD -MP -c $< -o $@
# CUDA compilation rule with dependency generation
%.o: %.cu
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# Linking rules
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PYBIND11_INCLUDES) -MMD -MP -c $< -o $@

$(TARGET_LOCAL): $(OBJ_LOCAL)
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) $(INCLUDES) $(OBJ_LOCAL) -lcuda -lcudart $(LDFLAGS) -o $@

$(TARGET_REMOTE): $(OBJ_REMOTE)
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) $(INCLUDES) $(OBJ_REMOTE) -lcuda -lcudart $(LDFLAGS) -o $@

$(TARGET_DUAL): $(OBJ_DUAL)
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) $(INCLUDES) $(OBJ_DUAL) -lcuda -lcudart $(LDFLAGS) -o $@

SRC_CU  := src/gpu_kernel.cu src/peer_copy.cu src/py_cuda_shims.cu

$(PYTARGET): $(OBJ_CPP) $(OBJ_CU) src/pybind_proxy.o
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) -shared -Xcompiler "-fPIC" \
	  $(INCLUDES) $(PYBIND11_INCLUDES) $^ -lcuda -lcudart $(LDFLAGS) -o $@

# $(UCCL_EP_EXT): $(UCCL_EP_OBJ)
# 	$(CXX) $(CXXFLAGS) -shared -fPIC $(INCLUDES) $(PYBIND11_INCLUDES) \
# 	-DHAVE_UCCL $^ -o $@

py: $(PYTARGET) $(UCCL_EP_EXT)

# Clean all generated files
clean:
	rm -f $(OBJ_CPP) $(OBJ_CU) \
	      $(SRC_LOCAL:.cu=.o) $(SRC_REMOTE:.cu=.o) $(SRC_DUAL:.cu=.o) \
	      $(PYBIND_OBJ) \
	      $(TARGET_LOCAL) $(TARGET_REMOTE) $(TARGET_DUAL) \
	      $(PYTARGET) $(UCCL_EP_EXT) \
	      *.d src/*.d bench/*.d

# Automatically include dependency files if they exist
DEPS := $(OBJ_CPP:.o=.d) $(OBJ_CU:.o=.d) \
        $(SRC_LOCAL:.cu=.d) $(SRC_REMOTE:.cu=.d) $(SRC_DUAL:.cu=.d) \
        $(PYBIND_OBJ:.o=.d) $(UCCL_EP_OBJ:.o=.d)
-include $(DEPS)