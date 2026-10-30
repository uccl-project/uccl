CXX ?= g++
CUDA_HOME ?= /usr/local/cuda
CPPFLAGS += -I../../include -I$(CUDA_HOME)/include
CXXFLAGS ?= -O2 -g
CXXFLAGS += -std=c++17
LDLIBS += -lgtest -pthread

BACKENDS := efa dpdk afxdp
TESTS := $(addprefix sack_bitmap_test_,$(BACKENDS))

.PHONY: all test clean
all: $(TESTS)

$(TESTS): sack_bitmap_test_%: sack_bitmap_test.cc
	$(CXX) $(CPPFLAGS) -I../$* $(CXXFLAGS) -MMD -MP $< $(LDFLAGS) $(LDLIBS) -o $@

test: $(TESTS)
	@set -e; for test in $(TESTS); do ./$$test; done

clean:
	rm -f $(TESTS) $(addsuffix .d,$(TESTS))

-include $(addsuffix .d,$(TESTS))
