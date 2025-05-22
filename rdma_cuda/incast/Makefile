CXX = g++

TARGET = incast

SRC = incast.cc

LDLIBS = -L/usr/lib/x86_64-linux-gnu/openmpi/lib -L/usr/local/cuda/lib64 -lmpi_cxx -lmpi -lcudart -lglog -lgflags -lgtest -lz -lelf -libverbs

CFLAGS = -I../ -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -O3 -g -std=c++17 -Wno-pointer-arith -Wno-interference-size -fPIC -I/usr/local/cuda/include

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CFLAGS) incast.cc -o incast ../eqds.o ../transport.o ../util_lrpc.o ../util_rdma.o $(LDLIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean