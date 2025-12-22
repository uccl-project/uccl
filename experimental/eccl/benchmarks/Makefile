CXX = g++
CXXFLAGS = -std=c++17 -O3 -g -Wall -DENABLE_CUDA
NVCC = nvcc
NVCCFLAGS = -O3 -g -Xcompiler -Wall -arch=sm_80

INCLUDES = -I./ -I/usr/local/cuda/include

# gdrcopy build dir
GDRCOPY_LIBDIR = $(abspath ../../../thirdparty/gdrcopy/src)

LIBS = -L/usr/local/cuda/lib -lcudart -lcuda \
       -L$(GDRCOPY_LIBDIR) -lgdrapi \
	   -L/usr/lib/x86_64-linux-gnu

RPATH = -Xlinker -rpath -Xlinker $(GDRCOPY_LIBDIR) \
        -Xlinker -rpath -Xlinker /usr/lib/x86_64-linux-gnu

TARGET = bench_gdrcopy

SRC_CPP := ./bench_gdrcopy.cc
SRC_CU  :=

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
