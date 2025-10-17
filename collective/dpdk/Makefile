CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Werror
CXXFLAGS += $(shell pkg-config --cflags libdpdk)
LDFLAGS := $(shell pkg-config --libs libdpdk)
LDFLAGS += $(shell pkg-config --libs libglog)


uccl-util-dpdk-test: util_dpdk_test.cc pmd_port.cc packet_pool.cc
	$(CXX) $(CXXFLAGS) -o $@  $^ $(LDFLAGS)

uccl-dpdk: main.cc driver.h
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) 



clean:
	rm -f uccl-dpdk uccl-util-dpdk-test