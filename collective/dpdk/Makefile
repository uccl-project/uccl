CXX := g++-13
CXXFLAGS := -std=c++17 -Wall -Wextra -Werror
CXXFLAGS += $(shell pkg-config --cflags libdpdk)
LDFLAGS := $(shell pkg-config --libs libdpdk)
LDFLAGS += $(shell pkg-config --libs libglog)

all: transport_test uccl-util-dpdk-test test

transport_test: transport_test.cc transport.cc pmd_port.cc packet_pool.cc
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgflags

uccl-util-dpdk-test: util_dpdk_test.cc pmd_port.cc packet_pool.cc
	$(CXX) $(CXXFLAGS) -o $@  $^ $(LDFLAGS)

uccl-dpdk: main.cc driver.h
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) 

test: test.cc pmd_port.cc packet_pool.cc
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)


clean:
	rm -f uccl-dpdk uccl-util-dpdk-test transport_test test