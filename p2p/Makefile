CUDA_PATH ?= /usr/local/cuda
CXX       ?= g++
NVCC      := $(CUDA_PATH)/bin/nvcc

CXXFLAGS  := -O3 -std=c++17 -Wall -pthread
LDFLAGS   := -lpthread

ifdef NO_RDMA
CXXFLAGS  += -DNO_RDMA
NVCCFLAGS := -O3 -std=c++17 -Xcompiler "-Wall -pthread" -DNO_RDMA
else
CXXFLAGS  +=
LDFLAGS   += -libverbs
NVCCFLAGS := -O3 -std=c++17 -Xcompiler "-Wall -pthread"
endif

INC       := -Iinclude -I$(CUDA_PATH)/include

SRC_CPP   := src/proxy.cpp
SRC_CU    := src/benchmark.cu src/gpu_kernel.cu src/ring_buffer.cu

OBJ_CPP   := $(SRC_CPP:.cpp=.o)
OBJ_CU    := $(SRC_CU:.cu=.o)
OBJ       := $(OBJ_CPP) $(OBJ_CU)

TARGET    := benchmark

all: $(TARGET)

# C++ file compilation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@ -Wno-deprecated-gpu-targets

# CUDA file compilation
%.o: %.cu
	$(NVCC) -gencode arch=compute_90,code=sm_90 $(NVCCFLAGS) $(INC) -c $< -o $@ -Wno-deprecated-gpu-targets

# Linking
$(TARGET): $(OBJ)
	$(NVCC) -gencode arch=compute_90,code=sm_90 $(NVCCFLAGS) $(INC) $(OBJ) -lcuda -lcudart $(LDFLAGS) -o $@  -Wno-deprecated-gpu-targets

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean