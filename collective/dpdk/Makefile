ABS_ROOT_PATH := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
ABS_REPO_ROOT := $(shell realpath $(ABS_ROOT_PATH)/../..)

CXX := g++-13
CXXFLAGS := -std=c++17 -Wall -Wextra
CXXFLAGS += $(shell pkg-config --cflags libdpdk)
CXXFLAGS += -I$(ABS_REPO_ROOT)/include
CXXFLAGS += -I$(ABS_ROOT_PATH)/core
CXXFLAGS += -I$(ABS_ROOT_PATH)/network
CXXFLAGS += -Wno-unused-function

LDFLAGS := $(shell pkg-config --libs libdpdk)
LDFLAGS += $(shell pkg-config --libs libglog)
LDFLAGS += -libverbs

all: transport_test uccl-util-dpdk-test

transport_test: transport_test.cc transport.cc pmd_port.cc packet_pool.cc
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgflags

uccl-util-dpdk-test: util_dpdk_test.cc pmd_port.cc packet_pool.cc
	$(CXX) $(CXXFLAGS) -o $@  $^ $(LDFLAGS)

uccl-dpdk: main.cc driver.h
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) 


clean:
	rm -f uccl-dpdk uccl-util-dpdk-test transport_test test