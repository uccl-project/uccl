# Makefile for the dietgpu float compression test
#
# Requires libdietgpu.so to be built first:
#   make -f Makefile.lib    (from the parent directory)
#
# Usage:
#   make              # build the test binary
#   make run          # build and run
#   make clean        # remove build artifacts
#   make SM=80        # override GPU architecture

CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
CXX       := g++

SM ?= 90

BUILDDIR := obj
TEST     := test_float_compress
LIB      := ../libdietgpu.so

# -I../.. resolves #include "dietgpu/..." to ../../dietgpu/...
# But we're inside thirdparty/dietgpu/test/, and sources use
# #include "dietgpu/float/...", so -I.. resolves to ../dietgpu/...
INCLUDES := -I.. -I$(CUDA_HOME)/include

NVCCFLAGS := -O3 -std=c++17 --compiler-options '-fPIC' \
	-gencode=arch=compute_$(SM),code=sm_$(SM) \
	-gencode=arch=compute_$(SM),code=compute_$(SM) \
	-DDISABLE_AGGRESSIVE_PTX_INSTRS

LDLIBS := -lglog -L$(CUDA_HOME)/lib64 -lcudart

# ---- Targets ----

all: $(TEST)

$(TEST): $(BUILDDIR)/test_float_compress.o $(LIB)
	$(CXX) -o $@ $< -L.. -ldietgpu $(LDLIBS)
	@echo "Built: $@"

$(BUILDDIR)/test_float_compress.o: test_float_compress.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(LIB):
	$(MAKE) -C .. -f Makefile.lib SM=$(SM)

clean:
	rm -rf $(BUILDDIR) $(TEST)

run: $(TEST)
	LD_LIBRARY_PATH=..:$(CUDA_HOME)/lib64:$$LD_LIBRARY_PATH ./$(TEST)

.PHONY: all clean run
