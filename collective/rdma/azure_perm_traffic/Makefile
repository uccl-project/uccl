CXX = g++

TARGET = permutation_traffic

SRC = permutation_traffic.cc

LDLIBS = -L/usr/lib/x86_64-linux-gnu/openmpi/lib -L/usr/local/cuda/lib64 -lmpi_cxx -lmpi -lcudart -lglog -lgflags -lgtest -lz -lelf -libverbs

CFLAGS = -I../ -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -O3 -g -std=c++17 -Wno-pointer-arith -Wno-interference-size -fPIC

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CFLAGS) permutation_traffic.cc -o permutation_traffic ../eqds.o ../transport.o ../util_lrpc.o ../util_rdma.o $(LDLIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean