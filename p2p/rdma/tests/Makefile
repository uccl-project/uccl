# SPDX-License-Identifier: (GPL-2.0 OR BSD-2-Clause)

# !!! Remember to conda deactivate before running make !!!

# Platform selection: set USE_HIP=1 for ROCm/HIP, default is CUDA
USE_HIP ?= 0

CUDA_HOME := /usr/local/cuda
HIP_HOME := /opt/rocm
DIETGPU_ROOT := ../../../thirdparty/dietgpu
DIETGPU_FLOAT_DIR := $(DIETGPU_ROOT)/dietgpu/float

# Common flags
COMMON_FLAGS := -I /opt/amazon/efa/include -L /opt/amazon/efa/lib -libverbs -lefa -lpthread \
	-I ../../../include -I ../../../collective/rdma -I ../../include -I $(DIETGPU_ROOT) -std=c++17 -lglog -lgflags

ifeq ($(USE_HIP),1)
	# HIP/ROCm flags
	CXXFLAGS = -D__HIP_PLATFORM_AMD__ \
		-I$(HIP_HOME)/include -L$(HIP_HOME)/lib -lamdhip64 \
		-L$(DIETGPU_FLOAT_DIR) -ldietgpu_float \
		-Wl,-rpath,$(abspath $(DIETGPU_FLOAT_DIR)) \
		$(COMMON_FLAGS)
else
	# CUDA flags
	CXXFLAGS = -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64 -lcudart -lcuda \
		-L$(DIETGPU_FLOAT_DIR) -ldietgpu_float \
		-Wl,-rpath,$(abspath $(DIETGPU_FLOAT_DIR)) \
		$(COMMON_FLAGS)
endif

all: test_efa_endpoint test_compression

test_efa_endpoint: test_efa_endpoint.cpp ../define.h ../rdma_device.h ../rdma_context.h ../efa_channel.h ../efa_endpoint.h ../epoll_client.h ../epoll_server.h ../../../collective/rdma/param.cc
	g++ -O3 -g test_efa_endpoint.cpp ../../../collective/rdma/param.cc -o test_efa_endpoint $(CXXFLAGS)

test_compression: test_compression.cpp ../compression.h ../define.h ../memory_allocator.h
	g++ -O3 -g test_compression.cpp -o test_compression $(CXXFLAGS)

clean:
	rm -f test_efa_endpoint test_efa_endpoint.o test_compression test_compression.o
