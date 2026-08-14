CXX      := /usr/bin/g++
CUDA_HOME ?= /usr/local/cuda
# Auto-detect CUDA if default path doesn't exist (e.g. /usr/local/cuda-13).
ifeq ($(wildcard $(CUDA_HOME)/bin/nvcc),)
  _CUDA_AUTO := $(firstword $(sort $(wildcard /usr/local/cuda-*)))
  ifneq ($(_CUDA_AUTO),)
    CUDA_HOME := $(_CUDA_AUTO)
  endif
endif
NVCC_BIN ?= $(CUDA_HOME)/bin/nvcc
ifeq ($(wildcard $(NVCC_BIN)),)
  # Fall back to PATH only if the CUDA_HOME toolkit is missing — never pick
  # up a conda/pyenv nvcc when a real toolkit exists (conda CUDA 12.x cannot
  # target sm_103 and drags the wrong host toolchain).
  NVCC_BIN := $(shell which nvcc 2>/dev/null)
endif
NVCC = $(NVCC_BIN)
# GPU architecture + per-arch reduce-kernel defaults (ILP / TMA / smem).
include ../../mk/arch_defaults.mk
ifeq ($(DEBUG),1)
DEV_OPT_FLAGS := -O0 -g
else
DEV_OPT_FLAGS := -O3
endif

CXXFLAGS := -std=c++17 $(DEV_OPT_FLAGS) -Wall
NVCCFLAGS:= $(DEV_OPT_FLAGS) -Xcompiler -Wall $(NVCC_ARCH_FLAGS)
ifeq ($(DEBUG),1)
NVCCFLAGS += -G
endif
NVCCFLAGS += -DUK_REDUCE_ILP=$(REDUCE_ILP) -DUK_REDUCE_SMEM_KB=$(REDUCE_SMEM_KB) -DUK_TMA_REDUCE=$(TMA_REDUCE) -DUK_TMA_WARPSPEC=$(TMA_WARPSPEC)
CXXFLAGS += -DUK_REDUCE_SMEM_KB=$(REDUCE_SMEM_KB)
# Rebuild when compile flags change (see mk/arch_defaults.mk).
$(shell printf '%s\n' '$(NVCCFLAGS) $(CXXFLAGS)' > $(BUILD_FLAGS_STAMP).tmp; \
        cmp -s $(BUILD_FLAGS_STAMP).tmp $(BUILD_FLAGS_STAMP) || \
        mv -f $(BUILD_FLAGS_STAMP).tmp $(BUILD_FLAGS_STAMP); \
        rm -f $(BUILD_FLAGS_STAMP).tmp)

UCCL_ROOT := $(abspath ../../../../)
GCC_MACHINE := $(shell $(CXX) -dumpmachine 2>/dev/null || echo x86_64-linux-gnu)
GDRCOPY_INCLUDEDIR ?= /usr/include
GDRCOPY_LIBDIR ?=
GDRCOPY_LIBDIR_FLAGS := $(if $(strip $(GDRCOPY_LIBDIR)),-L$(GDRCOPY_LIBDIR),)
ifneq ($(strip $(GDRCOPY_LIBDIR)),)
GDRCOPY_RPATH := -Wl,-rpath,$(GDRCOPY_LIBDIR)
else
GDRCOPY_RPATH :=
endif

INCLUDES := -I./ -I./fifo -I./ops -I./test -I./test/common -I./test/unit -I./test/integration \
	-I../../include -I$(CUDA_HOME)/include -I$(GDRCOPY_INCLUDEDIR) -I/usr/include \
	-I/usr/local/include -I/usr/include/$(GCC_MACHINE)
LIBS := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lcudart -lcuda -lnuma \
        $(GDRCOPY_LIBDIR_FLAGS) -lgdrapi
CXX_RPATH = -Wl,-rpath,$(CUDA_HOME)/lib64 \
            -Wl,-rpath,/usr/lib/$(GCC_MACHINE) \
            -Wl,-rpath-link,/lib/$(GCC_MACHINE):/usr/lib/$(GCC_MACHINE) \
            $(GDRCOPY_RPATH)

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

.PHONY: all test test-unit test-integration unit integration bench clean

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

%.o: %.cc $(BUILD_FLAGS_STAMP)
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

%.o: %.cpp $(BUILD_FLAGS_STAMP)
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

%.o: %.cu $(BUILD_FLAGS_STAMP)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(TEST_ALL_OBJ): $(TEST_ALL_SRC) $(BUILD_FLAGS_STAMP)
	$(NVCC) -x cu -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(TEST_ALL_BIN): $(COMMON_OBJ) $(TEST_ALL_OBJ)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(LIBS) $(CXX_RPATH)

$(SIMPLE_COPY_TEST_OBJ): $(SIMPLE_COPY_TEST_SRC)
	$(NVCC) -x cu -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(SIMPLE_COPY_TEST_BIN): $(COMMON_OBJ) $(SIMPLE_COPY_TEST_OBJ)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(LIBS) $(CXX_RPATH)

$(BENCH_OBJ): $(BENCH_SRC)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(BENCH_BIN): $(COMMON_OBJ) $(BENCH_OBJ)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(LIBS) $(CXX_RPATH)

$(BENCH_FULL_OBJ): $(BENCH_FULL_SRC)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(BENCH_FULL_BIN): $(COMMON_OBJ) $(BENCH_FULL_OBJ)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(LIBS) $(CXX_RPATH)

$(BENCH_SM_OBJ): $(BENCH_SM_SRC)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(BENCH_SM_BIN): $(COMMON_OBJ) $(BENCH_SM_OBJ)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(LIBS) $(CXX_RPATH)

$(BENCH_VS_OBJ): $(BENCH_VS_SRC)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(BENCH_VS_BIN): $(COMMON_OBJ) $(BENCH_VS_OBJ)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(LIBS) $(CXX_RPATH)

clean:
	rm -f *.o ./benchmarks/*.o ./fifo/*.o ./ops/*.o \
		./test/*.o ./test/*.d \
		./test/unit/*.o ./test/unit/*.d \
		./test/integration/*.o ./test/integration/*.d \
		*.d $(ALL_BIN)
