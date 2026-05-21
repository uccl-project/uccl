CXX ?= g++
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wno-unused-variable
LDLIBS = -libverbs -lpthread

bench: bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f bench
