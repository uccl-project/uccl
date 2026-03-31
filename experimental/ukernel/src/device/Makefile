CXX      := g++
NVCC     := nvcc
DEFAULT_SMS := 80 86 89
ifeq ($(origin SM), undefined)
NVCC_ARCH_FLAGS := $(foreach sm,$(DEFAULT_SMS),-gencode arch=compute_$(sm),code=sm_$(sm))
ENABLE_TMA := 0
else
SM_NUM := $(shell printf '%s\n' "$(SM)" | sed 's/[^0-9].*//')
NVCC_ARCH_FLAGS := -arch=sm_$(SM)
ENABLE_TMA := $(shell if [ -n "$(SM_NUM)" ] && [ "$(SM_NUM)" -ge 90 ]; then echo 1; else echo 0; fi)
endif
CXXFLAGS := -std=c++17 -O3 -g -Wall
NVCCFLAGS:= -O3 -g -Xcompiler -Wall $(NVCC_ARCH_FLAGS)
ifeq ($(DEBUG),1)
NVCCFLAGS += -G
endif
CXXFLAGS += -DUKERNEL_ENABLE_TMA=$(ENABLE_TMA)
NVCCFLAGS += -DUKERNEL_ENABLE_TMA=$(ENABLE_TMA)

UCCL_ROOT := $(abspath ../../../../)
CUDA_HOME ?= /usr/local/cuda
GDRCOPY_ROOT := $(UCCL_ROOT)/thirdparty/gdrcopy
GDRCOPY_INCLUDEDIR = $(GDRCOPY_ROOT)/include
GDRCOPY_LIBDIR = $(GDRCOPY_ROOT)/src
GDRCOPY_HEADER = $(GDRCOPY_INCLUDEDIR)/gdrapi.h

INCLUDES := -I./ -I./fifo -I./ops -I./test -I./test/common -I./test/unit -I./test/integration \
	-I../../include -I$(CUDA_HOME)/include -I$(GDRCOPY_INCLUDEDIR)
LIBS := -L$(CUDA_HOME)/lib -lcudart -lcuda -lnuma \
        -L$(GDRCOPY_LIBDIR) -lgdrapi

RPATH = -Xlinker -rpath -Xlinker $(GDRCOPY_LIBDIR)

COMMON_CPP := fifo/c2d_fifo.cc fifo/d2c_fifo.cpp fifo/sm_fifo.cc
COMMON_CU  := persistent_kernel_ops.cu
WORKER_CPP := worker.cc

TEST_ALL_SRC := test/unit/test_all.cc
SIMPLE_COPY_TEST_SRC := test/integration/simple_copy_test.cc
BENCH_SRC  := benchmarks/bench_device_fifo.cu
BENCH_FULL_SRC  := benchmarks/bench_device_full_fifo.cu
BENCH_SM_SRC  := benchmarks/bench_device_sm_fifo.cu
BENCH_VS_SRC := benchmarks/bench_device_launch_vs_worker.cu

COMMON_OBJ := $(patsubst %.cc,%.o,$(patsubst %.cpp,%.o,$(COMMON_CPP))) $(COMMON_CU:.cu=.o) $(WORKER_CPP:.cc=.o)
TEST_ALL_OBJ := $(TEST_ALL_SRC:.cc=.o)
SIMPLE_COPY_TEST_OBJ := $(SIMPLE_COPY_TEST_SRC:.cc=.o)
BENCH_OBJ  := $(BENCH_SRC:.cu=.o)
BENCH_FULL_OBJ  := $(BENCH_FULL_SRC:.cu=.o)
BENCH_SM_OBJ := $(BENCH_SM_SRC:.cu=.o)
BENCH_VS_OBJ := $(BENCH_VS_SRC:.cu=.o)

TEST_ALL_BIN := test_all
SIMPLE_COPY_TEST_BIN := simple_copy_test
BENCH_BIN := benchmarks/bench_device_fifo
BENCH_FULL_BIN := benchmarks/bench_device_full_fifo
BENCH_SM_BIN := benchmarks/bench_device_sm_fifo
BENCH_VS_BIN := benchmarks/bench_device_launch_vs_worker

ALL_BIN   := $(TEST_ALL_BIN) $(SIMPLE_COPY_TEST_BIN) $(BENCH_BIN) $(BENCH_FULL_BIN) ${BENCH_SM_BIN} $(BENCH_VS_BIN)

.PHONY: all test test-unit test-integration unit integration bench clean ensure-gdrcopy

all: $(ALL_BIN)

test: test-unit test-integration

test-unit: $(TEST_ALL_BIN)
	./$(TEST_ALL_BIN)

test-integration: $(SIMPLE_COPY_TEST_BIN)
	./$(SIMPLE_COPY_TEST_BIN)

unit: test-unit

integration: test-integration

simple: test-integration

bench: $(BENCH_BIN) $(BENCH_FULL_BIN) ${BENCH_SM_BIN} $(BENCH_VS_BIN)

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
		$(MAKE) -C "$(GDRCOPY_ROOT)"; \
	fi

%.o: %.cc
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

%.o: %.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(COMMON_OBJ) $(TEST_ALL_OBJ) $(SIMPLE_COPY_TEST_OBJ): | ensure-gdrcopy
$(TEST_ALL_OBJ): $(TEST_ALL_SRC)
	$(NVCC) -x cu -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(TEST_ALL_BIN): $(COMMON_OBJ) $(TEST_ALL_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

$(SIMPLE_COPY_TEST_OBJ): $(SIMPLE_COPY_TEST_SRC)
	$(NVCC) -x cu -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(SIMPLE_COPY_TEST_BIN): $(COMMON_OBJ) $(SIMPLE_COPY_TEST_OBJ)
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

$(BENCH_VS_OBJ): $(BENCH_VS_SRC)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(BENCH_VS_BIN): $(COMMON_OBJ) $(BENCH_VS_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

clean:
	rm -f *.o ./benchmarks/*.o ./fifo/*.o ./ops/*.o \
		./test/*.o ./test/*.d \
		./test/unit/*.o ./test/unit/*.d \
		./test/integration/*.o ./test/integration/*.d \
		*.d $(ALL_BIN)
