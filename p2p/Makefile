CUDA_PATH ?= /usr/local/cuda
override CXX := /usr/bin/g++
NVCC      := $(CUDA_PATH)/bin/nvcc

CXXFLAGS  := -O3 -std=c++17 -Wall -pthread
LDFLAGS   := -lpthread

CXXFLAGS  += -mclwb
LDFLAGS   += -libverbs -lnl-3 -lnl-route-3
NVCCFLAGS := -O3 -std=c++17 -Xcompiler "-Wall -pthread" -ccbin /usr/bin/g++

INCLUDES := -Iinclude -I$(CUDA_PATH)/include -I/usr/include

SRC_CPP := src/proxy.cpp src/rdma.cpp src/common.cpp
SRC_CU  := src/gpu_kernel.cu src/ring_buffer.cu
SRC_LOCAL  := src/benchmark_local.cu
SRC_REMOTE := src/benchmark_remote.cu

OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU := $(SRC_CU:.cu=.o)

OBJ_LOCAL  := $(OBJ_CPP) $(OBJ_CU) $(SRC_LOCAL:.cu=.o)
OBJ_REMOTE := $(OBJ_CPP) $(OBJ_CU) $(SRC_REMOTE:.cu=.o)

TARGET_LOCAL  := benchmark_local
TARGET_REMOTE := benchmark_remote

all: $(TARGET_LOCAL) $(TARGET_REMOTE)

# C++ compilation rule with dependency generation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# CUDA compilation rule with dependency generation
%.o: %.cu
	$(NVCC) -arch=sm_90 $(NVCCFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# Linking rules
$(TARGET_LOCAL): $(OBJ_LOCAL)
	$(NVCC) -arch=sm_90 $(NVCCFLAGS) $(INCLUDES) $(OBJ_LOCAL) -lcuda -lcudart $(LDFLAGS) -o $@

$(TARGET_REMOTE): $(OBJ_REMOTE)
	$(NVCC) -arch=sm_90 $(NVCCFLAGS) $(INCLUDES) $(OBJ_REMOTE) -lcuda -lcudart $(LDFLAGS) -o $@

# Clean all generated files
clean:
	rm -f $(OBJ_CPP) $(OBJ_CU) $(SRC_LOCAL:.cu=.o) $(SRC_REMOTE:.cu=.o) $(TARGET_LOCAL) $(TARGET_REMOTE) *.d src/*.d

.PHONY: all clean

# Automatically include dependency files if they exist
-include $(OBJ_LOCAL:.o=.d) $(OBJ_REMOTE:.o=.d)