# ===============================================================================
# Makefile for FIFO Performance Benchmark
# ===============================================================================

CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CXX := g++

# Auto-detect GPU architecture
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | sed 's/\.//' || echo "90")

TARGET := benchmark_fifo

# NVCC flags (release mode)
NVCC_FLAGS := -std=c++17
NVCC_FLAGS += -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
NVCC_FLAGS += -Xcompiler -Wall -Xcompiler -Wextra
NVCC_FLAGS += --expt-relaxed-constexpr
NVCC_FLAGS += -DMSCCLPP_USE_CUDA
NVCC_FLAGS += -O3 -DNDEBUG --use_fast_math

# CXX flags (release mode)
CXX_FLAGS := -std=c++17 -Wall -Wextra -pthread
CXX_FLAGS += -DMSCCLPP_USE_CUDA
CXX_FLAGS += -O3 -DNDEBUG

# Include paths
INCLUDES := -I../../include -I../../src
INCLUDES += -I$(CUDA_PATH)/include

# Libraries
LIBS := -lpthread -lcudart -lcuda -lnuma
LIBS += -L$(CUDA_PATH)/lib64

# Source files
SRC_CU := launch_kernel_shim.cu
SRC_CPP := benchmark_fifo.cpp ../../src/fifo.cpp

# Object files
OBJ_CU := $(SRC_CU:.cu=.o)
OBJ_CPP := $(SRC_CPP:.cpp=.o)

# ===============================================================================
# Build Rules
# ===============================================================================
.DEFAULT_GOAL := all

all: $(TARGET)

$(TARGET): $(OBJ_CU) $(OBJ_CPP)
	@echo "Linking $@..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^ $(LIBS)
	@echo "Build complete: $@"

# CUDA compilation rule
%.o: %.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# C++ compilation rule for local cpp files
%.o: %.cpp
	@echo "Compiling C++: $<"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# C++ compilation rule for ep/src files
../../src/%.o: ../../src/%.cpp
	@echo "Compiling C++: $<"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -f $(TARGET) *.o ../../src/*.o
	@echo "Done."

.PHONY: all clean

