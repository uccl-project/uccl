CXX      := g++
NVCC     := nvcc
CXXFLAGS := -std=c++17 -O3 -g -Wall
NVCCFLAGS:= -O3 -g -G -Xcompiler -Wall -arch=sm_80

UCCL_ROOT := $(abspath ../../../../)
CUDA_HOME ?= /usr/local/cuda
GDRCOPY_ROOT := $(UCCL_ROOT)/thirdparty/gdrcopy
GDRCOPY_INCLUDEDIR = $(GDRCOPY_ROOT)/include
GDRCOPY_LIBDIR = $(GDRCOPY_ROOT)/src
GDRCOPY_HEADER = $(GDRCOPY_INCLUDEDIR)/gdrapi.h

INCLUDES := -I./ -I./fifo -I../../include -I$(CUDA_HOME)/include -I$(GDRCOPY_INCLUDEDIR)
LIBS := -L$(CUDA_HOME)/lib -lcudart -lcuda -lnuma \
        -L$(GDRCOPY_LIBDIR) -lgdrapi

RPATH = -Xlinker -rpath -Xlinker $(GDRCOPY_LIBDIR)

COMMON_CPP := fifo/c2d_fifo.cc fifo/d2c_fifo.cpp fifo/sm_fifo.cc pk_selector.cc
COMMON_CU  := persistent_kernel_ops.cu

TEST_SRC   := test/test_device_runtime.cc
BENCH_SRC  := benchmarks/bench_device_fifo.cu
BENCH_FULL_SRC  := benchmarks/bench_device_full_fifo.cu
BENCH_SM_SRC  := benchmarks/bench_device_sm_fifo.cu

COMMON_OBJ := $(patsubst %.cc,%.o,$(patsubst %.cpp,%.o,$(COMMON_CPP))) $(COMMON_CU:.cu=.o)
TEST_OBJ   := $(TEST_SRC:.cc=.o)
BENCH_OBJ  := $(BENCH_SRC:.cu=.o)
BENCH_FULL_OBJ  := $(BENCH_FULL_SRC:.cu=.o)
BENCH_SM_OBJ := $(BENCH_SM_SRC:.cu=.o)

TEST_BIN  := test_device_runtime
BENCH_BIN := benchmarks/bench_device_fifo
BENCH_FULL_BIN := benchmarks/bench_device_full_fifo
BENCH_SM_BIN := benchmarks/bench_device_sm_fifo

ALL_BIN   := $(TEST_BIN) $(BENCH_BIN) $(BENCH_FULL_BIN) ${BENCH_SM_BIN}

.PHONY: all test bench clean ensure-gdrcopy

all: $(ALL_BIN)

test: $(TEST_BIN)

bench: $(BENCH_BIN) $(BENCH_FULL_BIN) ${BENCH_SM_BIN}

ensure-gdrcopy:
	@if [ ! -f "$(GDRCOPY_HEADER)" ] || [ ! -f "$(GDRCOPY_ROOT)/Makefile" ]; then \
		echo "[ukernel/device] initializing submodules for gdrcopy..."; \
		git -C "$(UCCL_ROOT)" submodule update --init --recursive; \
	fi
	@if [ ! -f "$(GDRCOPY_HEADER)" ]; then \
		echo "[ukernel/device] missing $(GDRCOPY_HEADER) after submodule init"; \
		exit 1; \
	fi
	@if ! ls "$(GDRCOPY_LIBDIR)"/libgdrapi.so* >/dev/null 2>&1; then \
		echo "[ukernel/device] building gdrcopy..."; \
		$(MAKE) -C "$(GDRCOPY_ROOT)" all; \
	fi

%.o: %.cc
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

%.o: %.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(COMMON_OBJ) $(TEST_OBJ) $(BENCH_OBJ) $(BENCH_FULL_OBJ) $(BENCH_SM_OBJ): | ensure-gdrcopy
$(TEST_OBJ): $(TEST_SRC)
	$(NVCC) -x cu -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(TEST_BIN): $(COMMON_OBJ) $(TEST_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

$(BENCH_BIN): $(COMMON_OBJ) $(BENCH_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

$(BENCH_FULL_BIN): $(COMMON_OBJ) $(BENCH_FULL_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

${BENCH_SM_BIN}: $(COMMON_OBJ) $(BENCH_SM_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

clean:
	rm -f *.o ./benchmarks/*.o ./fifo/*.o ./test/*.o *.d $(ALL_BIN)
