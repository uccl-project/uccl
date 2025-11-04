# SPDX-License-Identifier: (GPL-2.0 OR BSD-2-Clause)

# !!! Remember to conda deactivate before running make !!!

CUDA_HOME:=/usr/local/cuda
CXXFLAGS = -I /opt/amazon/efa/include -L /opt/amazon/efa/lib -lfabric -libverbs -lefa -lpthread \
	-I$(CUDA_HOME)/include -L ${CUDA_HOME}/lib64 -lcudart -lcuda -I ../../include

all: efa_rdma_write_class

efa_rdma_write_class: efa_rdma_write_class.cc
	g++ -O3 -g efa_rdma_write_class.cc -o efa_rdma_write_class $(CXXFLAGS)

clean:
	rm -f efa_rdma_write_class
