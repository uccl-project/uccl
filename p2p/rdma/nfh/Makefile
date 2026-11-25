# NFH BF16 codec for the P2P compressed-write path.
#
#   make                      # libnfh_codec.so for GPU_ARCH (default sm_90)
#   make GPU_ARCH=sm_100a
#   make GPU_ARCH_FLAGS="-gencode=arch=compute_89,code=sm_89 ..."   # several
#   make test                 # build and run the round-trip checks
#
# The P2P build links this library when built with USE_NFH=1.

CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
GPU_ARCH ?= sm_90
GPU_ARCH_FLAGS ?= -arch=$(GPU_ARCH)
NVCC_FLAGS ?= -O3 -std=c++17 $(GPU_ARCH_FLAGS) -Xcompiler -fPIC

LIB := libnfh_codec.so

.PHONY: all test clean
all: $(LIB)

$(LIB): nfh_codec.cu nfh_codec.h
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ nfh_codec.cu -lcuda

test_nfh_codec: test_nfh_codec.cu nfh_codec.h $(LIB)
	$(NVCC) $(NVCC_FLAGS) -o $@ test_nfh_codec.cu -L. -lnfh_codec -Xlinker -rpath,$(CURDIR)

test: test_nfh_codec
	./test_nfh_codec

clean:
	rm -f $(LIB) test_nfh_codec
