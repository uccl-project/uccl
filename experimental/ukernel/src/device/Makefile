CXX      := g++
NVCC     := nvcc
SM       ?= 80
SM_NUM   := $(shell printf '%s\n' "$(SM)" | sed 's/[^0-9].*//')
ENABLE_TMA := $(shell if [ -n "$(SM_NUM)" ] && [ "$(SM_NUM)" -ge 90 ]; then echo 1; else echo 0; fi)
CXXFLAGS := -std=c++17 -O3 -g -Wall
NVCCFLAGS:= -O3 -g -G -Xcompiler -Wall -arch=sm_$(SM)
CXXFLAGS += -DUKERNEL_ENABLE_TMA=$(ENABLE_TMA)
NVCCFLAGS += -DUKERNEL_ENABLE_TMA=$(ENABLE_TMA)

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

COMMON_CPP := fifo/c2d_fifo.cc fifo/d2c_fifo.cpp fifo/sm_fifo.cc
COMMON_CU  := persistent_kernel_ops.cu
WORKER_CPP := worker.cc

TEST_SRC   := test/test_device_runtime.cc
# TEST_MATCHING_SRC := test/test_fifo_matching.cc
TEST_PREWARM_SRC := test/test_prewarm.cc
TEST_MULTI_BLOCK_SRC := test/test_multi_block.cc
TEST_WORKER_POOL_SRC := test/test_worker_pool.cc
BENCH_SRC  := benchmarks/bench_device_fifo.cu
BENCH_FULL_SRC  := benchmarks/bench_device_full_fifo.cu
BENCH_SM_SRC  := benchmarks/bench_device_sm_fifo.cu

COMMON_OBJ := $(patsubst %.cc,%.o,$(patsubst %.cpp,%.o,$(COMMON_CPP))) $(COMMON_CU:.cu=.o) $(WORKER_CPP:.cc=.o)
TEST_OBJ   := $(TEST_SRC:.cc=.o)
# TEST_MATCHING_OBJ := $(TEST_MATCHING_SRC:.cc=.o)
TEST_PREWARM_OBJ := $(TEST_PREWARM_SRC:.cc=.o)
TEST_MULTI_BLOCK_OBJ := $(TEST_MULTI_BLOCK_SRC:.cc=.o)
TEST_WORKER_POOL_OBJ := $(TEST_WORKER_POOL_SRC:.cc=.o)
BENCH_OBJ  := $(BENCH_SRC:.cu=.o)
BENCH_FULL_OBJ  := $(BENCH_FULL_SRC:.cu=.o)
BENCH_SM_OBJ := $(BENCH_SM_SRC:.cu=.o)

TEST_BIN  := test_device_runtime
# TEST_MATCHING_BIN := test_fifo_matching
TEST_PREWARM_BIN := test_prewarm
TEST_MULTI_BLOCK_BIN := test_multi_block
TEST_WORKER_POOL_BIN := test_worker_pool
BENCH_BIN := benchmarks/bench_device_fifo
BENCH_FULL_BIN := benchmarks/bench_device_full_fifo
BENCH_SM_BIN := benchmarks/bench_device_sm_fifo

ALL_BIN   := $(TEST_BIN) $(TEST_PREWARM_BIN) $(TEST_MULTI_BLOCK_BIN) $(TEST_WORKER_POOL_BIN) $(BENCH_BIN) $(BENCH_FULL_BIN) ${BENCH_SM_BIN}

.PHONY: all test test_matching test_prewarm test_multi_block test_worker_pool bench clean ensure-gdrcopy

all: $(ALL_BIN)

test: $(TEST_BIN)

test_matching: $(TEST_MATCHING_BIN)

test_prewarm: $(TEST_PREWARM_BIN)

test_multi_block: $(TEST_MULTI_BLOCK_BIN)

test_worker_pool: $(TEST_WORKER_POOL_BIN)

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

$(COMMON_OBJ) $(TEST_OBJ) $(TEST_PREWARM_OBJ) $(TEST_MULTI_BLOCK_OBJ) $(TEST_WORKER_POOL_OBJ): | ensure-gdrcopy
$(TEST_OBJ): $(TEST_SRC)
	$(NVCC) -x cu -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(TEST_PREWARM_OBJ): $(TEST_PREWARM_SRC)
	$(NVCC) -x cu -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(TEST_MULTI_BLOCK_OBJ): $(TEST_MULTI_BLOCK_SRC)
	$(NVCC) -x cu -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(TEST_WORKER_POOL_OBJ): $(TEST_WORKER_POOL_SRC)
	$(NVCC) -x cu -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(TEST_BIN): $(COMMON_OBJ) $(TEST_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

$(TEST_PREWARM_BIN): $(COMMON_OBJ) $(TEST_PREWARM_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

$(TEST_MULTI_BLOCK_BIN): $(COMMON_OBJ) $(TEST_MULTI_BLOCK_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

$(TEST_WORKER_POOL_BIN): $(COMMON_OBJ) $(TEST_WORKER_POOL_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

$(BENCH_OBJ): $(BENCH_SRC)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(BENCH_BIN): $(COMMON_OBJ) $(BENCH_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

$(BENCH_FULL_OBJ): $(BENCH_FULL_SRC)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(BENCH_FULL_BIN): $(COMMON_OBJ) $(BENCH_FULL_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

$(BENCH_SM_OBJ): $(BENCH_SM_SRC)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(BENCH_SM_BIN): $(COMMON_OBJ) $(BENCH_SM_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

clean:
	rm -f *.o ./benchmarks/*.o ./fifo/*.o ./test/*.o *.d $(ALL_BIN)
