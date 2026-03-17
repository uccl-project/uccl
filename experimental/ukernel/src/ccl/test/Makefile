CXX      := g++
NVCC     := nvcc
SM       ?= 80
SM_NUM   := $(shell printf '%s\n' "$(SM)" | sed 's/[^0-9].*//')
ENABLE_TMA := $(shell if [ -n "$(SM_NUM)" ] && [ "$(SM_NUM)" -ge 90 ]; then echo 1; else echo 0; fi)
CXXFLAGS := -std=c++17 -O3 -g -Wall
NVCCFLAGS:= -O3 -g -G -Xcompiler -Wall -arch=sm_$(SM)
CXXFLAGS += -DUKERNEL_ENABLE_TMA=$(ENABLE_TMA)
NVCCFLAGS += -DUKERNEL_ENABLE_TMA=$(ENABLE_TMA)

UCCL_ROOT := $(abspath ../../../../../)
CUDA_HOME ?= /usr/local/cuda
GDRCOPY_ROOT := $(UCCL_ROOT)/thirdparty/gdrcopy
GDRCOPY_INCLUDEDIR = $(GDRCOPY_ROOT)/include
GDRCOPY_LIBDIR = $(GDRCOPY_ROOT)/src
GDRCOPY_HEADER = $(GDRCOPY_INCLUDEDIR)/gdrapi.h

INCLUDES := -I./ -I../ -I../../include -I../../device -I../../device/fifo -I$(CUDA_HOME)/include -I$(GDRCOPY_INCLUDEDIR)
LIBS := -L$(CUDA_HOME)/lib -lcudart -lcuda -lnuma \
        -L$(GDRCOPY_LIBDIR) -lgdrapi

RPATH = -Xlinker -rpath -Xlinker $(GDRCOPY_LIBDIR)

DEVICE_COMMON_CPP := ../../device/fifo/c2d_fifo.cc ../../device/fifo/d2c_fifo.cpp ../../device/fifo/sm_fifo.cc ../../device/pk_selector.cc
DEVICE_COMMON_CU  := ../../device/persistent_kernel_ops.cu
CCL_COMMON_CPP := ../topology.cc ../plan.cc ../selector.cc ../executor.cc ../device_backend.cc

TEST_PERSISTENT_SRC := test_ccl_persistent_backend.cc
TEST_CE_SRC := test_ccl_copy_engine_backend.cc

DEVICE_COMMON_OBJ := $(notdir $(patsubst %.cc,%.o,$(patsubst %.cpp,%.o,$(DEVICE_COMMON_CPP)))) $(notdir $(DEVICE_COMMON_CU:.cu=.o))
CCL_COMMON_OBJ := $(notdir $(CCL_COMMON_CPP:.cc=.o))
TEST_PERSISTENT_OBJ := $(TEST_PERSISTENT_SRC:.cc=.o)
TEST_CE_OBJ := $(TEST_CE_SRC:.cc=.o)

TEST_PERSISTENT_BIN := test_ccl_persistent_backend
TEST_CE_BIN := test_ccl_copy_engine_backend

ALL_BIN := $(TEST_PERSISTENT_BIN) $(TEST_CE_BIN)

.PHONY: all test clean ensure-gdrcopy

all: $(ALL_BIN)

test: $(ALL_BIN)

ensure-gdrcopy:
	@if [ ! -f "$(GDRCOPY_HEADER)" ] || [ ! -f "$(GDRCOPY_ROOT)/Makefile" ]; then \
		echo "[ukernel/ccl] initializing submodules for gdrcopy..."; \
		git -C "$(UCCL_ROOT)" submodule update --init --recursive; \
	fi
	@if [ ! -f "$(GDRCOPY_HEADER)" ]; then \
		echo "[ukernel/ccl] missing $(GDRCOPY_HEADER) after submodule init"; \
		exit 1; \
	fi
	@if ! ls "$(GDRCOPY_LIBDIR)"/libgdrapi.so* >/dev/null 2>&1; then \
		echo "[ukernel/ccl] building gdrcopy..."; \
		$(MAKE) -C "$(GDRCOPY_ROOT)" all; \
	fi

%.o: %.cc
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

%.o: %.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

c2d_fifo.o: ../../device/fifo/c2d_fifo.cc | ensure-gdrcopy
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

d2c_fifo.o: ../../device/fifo/d2c_fifo.cpp | ensure-gdrcopy
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

sm_fifo.o: ../../device/fifo/sm_fifo.cc | ensure-gdrcopy
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

pk_selector.o: ../../device/pk_selector.cc | ensure-gdrcopy
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

persistent_kernel_ops.o: ../../device/persistent_kernel_ops.cu | ensure-gdrcopy
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

topology.o: ../topology.cc | ensure-gdrcopy
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

plan.o: ../plan.cc | ensure-gdrcopy
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

selector.o: ../selector.cc | ensure-gdrcopy
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

executor.o: ../executor.cc | ensure-gdrcopy
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

device_backend.o: ../device_backend.cc | ensure-gdrcopy
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

$(TEST_PERSISTENT_OBJ): $(TEST_PERSISTENT_SRC) | ensure-gdrcopy
	$(NVCC) -x cu -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(TEST_CE_OBJ): $(TEST_CE_SRC) | ensure-gdrcopy
	$(NVCC) -x cu -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(TEST_PERSISTENT_BIN): $(DEVICE_COMMON_OBJ) $(CCL_COMMON_OBJ) $(TEST_PERSISTENT_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

$(TEST_CE_BIN): $(DEVICE_COMMON_OBJ) $(CCL_COMMON_OBJ) $(TEST_CE_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

clean:
	rm -f *.o *.d $(ALL_BIN)
