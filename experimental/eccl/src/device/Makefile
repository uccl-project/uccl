CXX = g++
CXXFLAGS = -std=c++17 -O3 -g -Wall
NVCC = nvcc
NVCCFLAGS = -O3 -g -G -Xcompiler -Wall -arch=sm_80

GDRCOPY_LIBDIR = $(abspath ../../../../thirdparty/gdrcopy/src)

INCLUDES = -I./ -I/usr/local/cuda/include
LIBS = -L/usr/local/cuda/lib -lcudart -lcuda -lnuma \
       -L$(GDRCOPY_LIBDIR) -lgdrapi

RPATH = -Xlinker -rpath -Xlinker $(GDRCOPY_LIBDIR)

TARGET = test_persistent

SRC_CPP := ./test_persistent.cc ./c2d_fifo.cc
SRC_CU  := ./operator.cu

OBJ_CC   := $(SRC_CPP:.cc=.o)
OBJ_CU   := $(SRC_CU:.cu=.o)

%.o: %.cc
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

%.o: %.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(TARGET): $(OBJ_CC) $(OBJ_CU)
	$(NVCC) $(OBJ_CC) $(OBJ_CU) -o $(TARGET) $(LIBS) $(RPATH) $(NVCCFLAGS)

clean:
	rm -f $(OBJ_CC) $(OBJ_CU) $(TARGET)
