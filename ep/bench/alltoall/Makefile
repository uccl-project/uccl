# Makefile for All-to-All RDMA Benchmarks

CXX := g++
CUDA_HOME ?= /usr/local/cuda

CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -march=native -pthread
INCLUDES := -I$(CUDA_HOME)/include
LDFLAGS  := -L$(CUDA_HOME)/lib64 -lcudart -libverbs -lefa

TARGETS := benchmark test_alltoall test_alltoall_rail

.PHONY: all clean
all: $(TARGETS)

benchmark: benchmark.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) -lpthread -lnuma

test_alltoall: test_alltoall.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

test_alltoall_rail: test_alltoall_rail.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o
