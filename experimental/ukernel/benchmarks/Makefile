CXX = g++
CXXFLAGS = -std=c++17 -O3 -g -Wall -DENABLE_CUDA
NVCC = nvcc
NVCCFLAGS = -O3 -g -Xcompiler -Wall -arch=sm_80

UCCL_ROOT := $(abspath ../../..)
GDRCOPY_ROOT = $(UCCL_ROOT)/thirdparty/gdrcopy
GDRCOPY_INCLUDEDIR = $(GDRCOPY_ROOT)/include
GDRCOPY_HEADER = $(GDRCOPY_INCLUDEDIR)/gdrapi.h

INCLUDES = -I./ -I/usr/local/cuda/include -I$(GDRCOPY_INCLUDEDIR)

# gdrcopy build dir
GDRCOPY_LIBDIR = $(GDRCOPY_ROOT)/src

LIBS = -L/usr/local/cuda/lib -lcudart -lcuda \
       -L$(GDRCOPY_LIBDIR) -lgdrapi

RPATH = -Xlinker -rpath -Xlinker $(GDRCOPY_LIBDIR)

TARGET = bench_gdrcopy

SRC_CPP := ./bench_gdrcopy.cc
SRC_CU  :=

OBJ_CC   := $(SRC_CPP:.cc=.o)
OBJ_CU   := $(SRC_CU:.cu=.o)

.PHONY: ensure-gdrcopy

ensure-gdrcopy:
	@if [ ! -f "$(GDRCOPY_HEADER)" ] || [ ! -f "$(GDRCOPY_ROOT)/Makefile" ]; then \
		echo "[ukernel/benchmarks] initializing submodules for gdrcopy..."; \
		git -C "$(UCCL_ROOT)" submodule update --init --recursive; \
	fi
	@if [ ! -f "$(GDRCOPY_HEADER)" ]; then \
		echo "[ukernel/benchmarks] missing $(GDRCOPY_HEADER) after submodule init"; \
		exit 1; \
	fi
	@if ! ls "$(GDRCOPY_LIBDIR)"/libgdrapi.so* >/dev/null 2>&1; then \
		echo "[ukernel/benchmarks] building gdrcopy..."; \
		$(MAKE) -C "$(GDRCOPY_ROOT)" all; \
	fi

%.o: %.cc
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

%.o: %.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(OBJ_CC) $(OBJ_CU): | ensure-gdrcopy

$(TARGET): $(OBJ_CC) $(OBJ_CU)
	$(NVCC) $(OBJ_CC) $(OBJ_CU) -o $(TARGET) $(LIBS) $(RPATH) $(NVCCFLAGS)

clean:
	rm -f $(OBJ_CC) $(OBJ_CU) $(TARGET)
