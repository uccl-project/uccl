# SPDX-License-Identifier: (GPL-2.0 OR BSD-2-Clause)

# !!! Remember to conda deactivate before running make !!!

CUDA_HOME:=/usr/local/cuda
CXXFLAGS = -I /opt/amazon/efa/include -L /opt/amazon/efa/lib -lfabric -libverbs -lefa -lpthread \
	-I$(CUDA_HOME)/include -L ${CUDA_HOME}/lib64 -lcudart -lcuda -I ../../../include -I ../../../collective/rdma -std=c++17 -lglog -lgflags
	
all: test_efa_endpoint 

test_efa_endpoint: test_efa_endpoint.cpp ../define.h ../rdma_device.h ../rdma_context.h ../efa_channel.h ../efa_endpoint.h ../epoll_client.h ../epoll_server.h ../../../collective/rdma/param.cc
	g++ -O3 -g test_efa_endpoint.cpp ../../../collective/rdma/param.cc -o test_efa_endpoint $(CXXFLAGS)

clean:
	rm -f test_efa_endpoint test_efa_endpoint.o
