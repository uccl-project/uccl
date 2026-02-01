# Makefile for building libdietgpu.so (standalone) and test binary
#
# Usage:
#   make              # build libdietgpu.so and test
#   make run          # build and run the test
#   make clean        # remove build artifacts
#   make SM=80        # override GPU architecture (default: 90)

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
CXX       := g++

# GPU architecture (override with: make SM=80)
SM ?= 90

# Source root (no need to copy files here)
DIETGPU   := ../dietgpu
BUILDDIR  := obj

# Output
LIB    := libdietgpu.so
TEST   := test_float_compress

# Source files (relative to DIETGPU)
CU_SRCS := \
	$(DIETGPU)/ans/GpuANSEncode.cu \
	$(DIETGPU)/ans/GpuANSDecode.cu \
	$(DIETGPU)/ans/GpuANSInfo.cu \
	$(DIETGPU)/float/GpuFloatCompress.cu \
	$(DIETGPU)/float/GpuFloatDecompress.cu \
	$(DIETGPU)/float/GpuFloatInfo.cu

CPP_SRCS := \
	$(DIETGPU)/utils/DeviceUtils.cpp \
	$(DIETGPU)/utils/StackDeviceMemory.cpp

# Object files (flatten into obj/)
CU_OBJS  := $(patsubst $(DIETGPU)/%.cu,$(BUILDDIR)/%.o,$(CU_SRCS))
CPP_OBJS := $(patsubst $(DIETGPU)/%.cpp,$(BUILDDIR)/%.o,$(CPP_SRCS))
LIB_OBJS := $(CU_OBJS) $(CPP_OBJS)

# Include paths: -I.. resolves #include "dietgpu/ans/..." to ../dietgpu/ans/...
INCLUDES := -I.. -I$(CUDA_HOME)/include

# Compiler flags
CXXFLAGS := -O3 -fPIC -std=c++17 \
	-Wno-deprecated-declarations \
	-Wno-unused-variable \
	-Wno-sign-compare \
	-Wno-reorder \
	-Wno-attributes \
	-Wno-unused-result \
	-Wno-unused-function

NVCCFLAGS := -O3 -std=c++17 --compiler-options '-fPIC' \
	-gencode=arch=compute_$(SM),code=sm_$(SM) \
	-gencode=arch=compute_$(SM),code=compute_$(SM) \
	-DDISABLE_AGGRESSIVE_PTX_INSTRS

# Libraries
LDLIBS := -lglog -L$(CUDA_HOME)/lib64 -lcudart

# ---- Targets ----

all: $(LIB) $(TEST)

# Build the shared library
$(LIB): $(LIB_OBJS)
	$(CXX) -shared -o $@ $^ $(LDLIBS)
	@echo "Built: $@"

# Build the test binary, linking against libdietgpu.so
$(TEST): $(BUILDDIR)/test_float_compress.o $(LIB)
	$(CXX) -o $@ $< -L. -ldietgpu $(LDLIBS)
	@echo "Built: $@"

# Compile CUDA library sources
$(BUILDDIR)/%.o: $(DIETGPU)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ library sources
$(BUILDDIR)/%.o: $(DIETGPU)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile the test (needs NVCC for cuda_fp16.h / __float2half)
$(BUILDDIR)/test_float_compress.o: test_float_compress.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(BUILDDIR) $(LIB) $(TEST)

# Run the test with the library path set
run: $(TEST)
	LD_LIBRARY_PATH=.:$(CUDA_HOME)/lib64:$$LD_LIBRARY_PATH ./$(TEST)

.PHONY: all clean run
