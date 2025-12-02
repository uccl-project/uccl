# SPDX-License-Identifier: (GPL-2.0 OR BSD-2-Clause)

# !!! Remember to conda deactivate before running make !!!

CUDA_HOME:=/usr/local/cuda
CXXFLAGS = -I /opt/amazon/efa/include -L /opt/amazon/efa/lib -lfabric -libverbs -lefa -lpthread \
	-I$(CUDA_HOME)/include -L ${CUDA_HOME}/lib64 -lcudart -lcuda -I ../../include -I ../../collective/rdma -std=c++17 -lglog -lgflags  

all: efa_endpoint_example 

efa_endpoint_example: main.cpp define.h rdma_device.h rdma_context.h efa_channel.h efa_endpoint.h epoll_client.h epoll_server.h ../../collective/rdma/param.cc
	g++ -O3 -g main.cpp ../../collective/rdma/param.cc -o efa_endpoint_example $(CXXFLAGS)

clean:
	rm -f efa_endpoint_example main.o
