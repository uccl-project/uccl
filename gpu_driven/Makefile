# === Toolchains ===
CUDA_PATH ?= /usr/local/cuda
override CXX := /usr/bin/g++
NVCC      := $(CUDA_PATH)/bin/nvcc

# === Python / pybind11 ===
PYTHON            ?= python3
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
EXT_SUFFIX        := $(shell $(PYTHON) -c "import sysconfig as s; print(s.get_config_var('EXT_SUFFIX') or '.so')")

# GPU arch (override with: make SM=90)
SM ?= 90

# === Flags ===
CXXFLAGS  := -O3 -std=c++17 -Wall -pthread -fPIC
NVCCFLAGS := -O3 -std=c++17 -Xcompiler "-Wall -pthread -fPIC" -ccbin /usr/bin/g++
LDFLAGS := -lpthread -libverbs -lnl-3 -lnl-route-3 -Xlinker -rpath -Xlinker $(CUDA_PATH)/lib64

INCLUDES := -Iinclude -I$(CUDA_PATH)/include -I/usr/include -I../include

# === Sources ===
SRC_CPP := src/proxy.cpp src/rdma.cpp src/common.cpp src/peer_copy_worker.cpp
# include the shim here (one definition only)
SRC_CU  := src/gpu_kernel.cu src/peer_copy.cu src/py_cuda_shims.cu

SRC_LOCAL  := bench/benchmark_local.cu
SRC_REMOTE := bench/benchmark_remote.cu
SRC_DUAL   := bench/benchmark_dual.cu

# pybind11 binding source
PYBIND_SRC := src/pybind_proxy.cc

# === Objects ===
OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU  := $(SRC_CU:.cu=.o)

OBJ_LOCAL  := $(OBJ_CPP) $(OBJ_CU) $(SRC_LOCAL:.cu=.o)
OBJ_REMOTE := $(OBJ_CPP) $(OBJ_CU) $(SRC_REMOTE:.cu=.o)
OBJ_DUAL   := $(OBJ_CPP) $(OBJ_CU) $(SRC_DUAL:.cu=.o)

PYBIND_OBJ := $(PYBIND_SRC:.cc=.o)

# === Targets ===
TARGET_LOCAL  := benchmark_local
TARGET_REMOTE := benchmark_remote
TARGET_DUAL   := benchmark_dual
PYTARGET      := pyproxy$(EXT_SUFFIX)

.PHONY: all py clean

all: $(TARGET_LOCAL) $(TARGET_REMOTE) $(TARGET_DUAL) $(PYTARGET)

# --- C++ compilation rule (with dep gen) ---
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# --- CUDA compilation rule (with dep gen) ---
%.o: %.cu
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# --- pybind11 C++ compilation (with dep gen) ---
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PYBIND11_INCLUDES) -MMD -MP -c $< -o $@

# --- Linking rules for benchmarks ---
$(TARGET_LOCAL): $(OBJ_LOCAL)
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) $(INCLUDES) $(OBJ_LOCAL) -lcuda -lcudart $(LDFLAGS) -o $@

$(TARGET_REMOTE): $(OBJ_REMOTE)
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) $(INCLUDES) $(OBJ_REMOTE) -lcuda -lcudart $(LDFLAGS) -o $@

$(TARGET_DUAL): $(OBJ_DUAL)
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) $(INCLUDES) $(OBJ_DUAL) -lcuda -lcudart $(LDFLAGS) -o $@

SRC_CU  := src/gpu_kernel.cu src/peer_copy.cu src/py_cuda_shims.cu

# --- Python extension (linked with NVCC to handle device objects) ---
$(PYTARGET): $(OBJ_CPP) $(OBJ_CU) src/pybind_proxy.o
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) -shared -Xcompiler "-fPIC" \
	  $(INCLUDES) $(PYBIND11_INCLUDES) $^ -lcuda -lcudart $(LDFLAGS) -o $@

py: $(PYTARGET)

# --- Clean ---
clean:
	rm -f $(OBJ_CPP) $(OBJ_CU) \
	      $(SRC_LOCAL:.cu=.o) $(SRC_REMOTE:.cu=.o) $(SRC_DUAL:.cu=.o) \
	      $(PYBIND_OBJ) \
	      $(TARGET_LOCAL) $(TARGET_REMOTE) $(TARGET_DUAL) \
	      $(PYTARGET) \
	      *.d src/*.d bench/*.d

# --- Auto-include dependency files ---
DEPS := $(OBJ_CPP:.o=.d) $(OBJ_CU:.o=.d) \
        $(SRC_LOCAL:.cu=.d) $(SRC_REMOTE:.cu=.d) $(SRC_DUAL:.cu=.d) \
        $(PYBIND_OBJ:.o=.d)
-include $(DEPS)