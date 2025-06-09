CUDA_PATH ?= /usr/local/cuda
CXX       ?= g++
NVCC      := $(CUDA_PATH)/bin/nvcc

CXXFLAGS  := -O3 -std=c++17 -Wall -pthread
LDFLAGS   := -lpthread

ifdef NO_RDMA
CXXFLAGS  += -DNO_RDMA
NVCCFLAGS := -O3 -std=c++17 -Xcompiler "-Wall -pthread" -DNO_RDMA
else
CXXFLAGS  += -mclwb
LDFLAGS   += -libverbs
NVCCFLAGS := -O3 -std=c++17 -Xcompiler "-Wall -pthread"
endif

INC       := -Iinclude -I$(CUDA_PATH)/include

SRC_CPP   := src/proxy.cpp src/rdma.cpp
SRC_CU_LOCAL := src/benchmark_local.cu src/gpu_kernel.cu src/ring_buffer.cu
SRC_CU_REMOTE := src/benchmark_remote.cu src/gpu_kernel.cu src/ring_buffer.cu

OBJ_CPP   := $(SRC_CPP:.cpp=.o)
OBJ_CU_LOCAL := $(SRC_CU_LOCAL:.cu=.o)
OBJ_CU_REMOTE := $(SRC_CU_REMOTE:.cu=.o)

OBJ_LOCAL := $(OBJ_CPP) $(OBJ_CU_LOCAL)
OBJ_REMOTE := $(OBJ_CPP) $(OBJ_CU_REMOTE)

TARGET_LOCAL  := benchmark_local
TARGET_REMOTE := benchmark_remote

all: $(TARGET_LOCAL) $(TARGET_REMOTE)

# Compile C++ sources
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@ -Wno-deprecated-gpu-targets

# Compile CUDA sources
%.o: %.cu
	$(NVCC) -arch=sm_90 $(NVCCFLAGS) $(INC) -c $< -o $@ -Wno-deprecated-gpu-targets

# Linking
$(TARGET_LOCAL): $(OBJ_LOCAL)
	$(NVCC) -arch=sm_90 $(NVCCFLAGS) $(INC) $(OBJ_LOCAL) -lcuda -lcudart $(LDFLAGS) -o $@  -Wno-deprecated-gpu-targets

$(TARGET_REMOTE): $(OBJ_REMOTE)
	$(NVCC) -arch=sm_90 $(NVCCFLAGS) $(INC) $(OBJ_REMOTE) -lcuda -lcudart $(LDFLAGS) -o $@  -Wno-deprecated-gpu-targets

clean:
	rm -f $(OBJ_CPP) $(OBJ_CU_LOCAL) $(OBJ_CU_REMOTE) $(TARGET_LOCAL) $(TARGET_REMOTE)

.PHONY: all clean