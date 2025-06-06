CUDA_PATH ?= /usr/local/cuda
CXX       ?= g++
NVCC      := $(CUDA_PATH)/bin/nvcc
CXXFLAGS  := -O3 -std=c++17 -Wall -pthread
LDFLAGS   := -lpthread

ifdef NO_RDMA
CXXFLAGS  += -DNO_RDMA
NVCCFLAGS := -O3 -std=c++17 -Xcompiler '-Wall -pthread' -DNO_RDMA
else
CXXFLAGS  += -libverbs
LDFLAGS   += -libverbs
NVCCFLAGS := -O3 -std=c++17 -Xcompiler '-Wall -pthread'
endif

INC        = -Iinclude -I$(CUDA_PATH)/include

SRC_CPP   := src/proxy.cpp 
SRC_CU    := src/benchmark.cu src/gpu_kernel.cu src/ring_buffer.cu
OBJ       := $(SRC_CPP:.cpp=.o) $(SRC_CU:.cu=.o)

all: benchmark

%.o: %.cppls
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INC) -c $< -o $@ -Wno-deprecated-gpu-targets

benchmark: $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(INC) $(OBJ) -lcuda -lcudart $(LDFLAGS) -o $@

clean:
	rm -f $(OBJ) benchmark

.PHONY: all clean