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
UCCL_RDMA_ROOT := $(UCCL_ROOT)/collective/rdma
UCCL_RDMA_STATIC := $(UCCL_RDMA_ROOT)/librdma.a

VPATH := .:test:../transport:../device:../device/fifo

INCLUDES := -I. -I./test -I../transport -I../device -I../device/fifo -I../../include -I$(CUDA_HOME)/include -I$(GDRCOPY_INCLUDEDIR)
LIBS := -L$(CUDA_HOME)/lib -lcudart -lcuda -lnuma \
        -L$(GDRCOPY_LIBDIR) -lgdrapi \
        $(UCCL_RDMA_STATIC) -libverbs -lrdmacm

RPATH = -Xlinker -rpath -Xlinker $(GDRCOPY_LIBDIR)

CCL_SRC := topology.cc plan.cc selector.cc executor.cc device_backend.cc transport_backend.cc
TRANSPORT_SRC := communicator.cc ipc_channel.cc memory_registry.cc oob_redis.cc oob_shm.cc oob_socket.cc request.cc transport_engine.cc uccl_transport_adapter.cc utils.cc
DEVICE_CPP_SRC := c2d_fifo.cc d2c_fifo.cpp sm_fifo.cc pk_selector.cc
DEVICE_CU_SRC := persistent_kernel_ops.cu

TEST_MAIN_SRC := test_main.cc test_ccl_plan.cc test_ccl_executor.cc test_ccl_rdma_allgather.cc
TEST_PERSISTENT_SRC := test_ccl_persistent_backend.cc
TEST_COPY_ENGINE_SRC := test_ccl_copy_engine_backend.cc

CCL_OBJ := $(CCL_SRC:.cc=.o)
TRANSPORT_OBJ := $(TRANSPORT_SRC:.cc=.o)
DEVICE_CPP_OBJ := $(patsubst %.cc,%.o,$(filter %.cc,$(DEVICE_CPP_SRC))) $(patsubst %.cpp,%.o,$(filter %.cpp,$(DEVICE_CPP_SRC)))
DEVICE_CU_OBJ := $(DEVICE_CU_SRC:.cu=.o)
COMMON_OBJ := $(CCL_OBJ) $(TRANSPORT_OBJ) $(DEVICE_CPP_OBJ) $(DEVICE_CU_OBJ)

TEST_MAIN_OBJ := $(TEST_MAIN_SRC:.cc=.o)
TEST_PERSISTENT_OBJ := $(TEST_PERSISTENT_SRC:.cc=.o)
TEST_COPY_ENGINE_OBJ := $(TEST_COPY_ENGINE_SRC:.cc=.o)

TEST_MAIN_BIN := test_ccl_main
TEST_PERSISTENT_BIN := test_ccl_persistent_backend
TEST_COPY_ENGINE_BIN := test_ccl_copy_engine_backend
ALL_BIN := $(TEST_MAIN_BIN) $(TEST_PERSISTENT_BIN) $(TEST_COPY_ENGINE_BIN)

.PHONY: all test clean ensure-gdrcopy ensure-uccl-rdma

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

ensure-uccl-rdma:
	@if [ ! -f "$(UCCL_RDMA_STATIC)" ]; then \
		echo "[ukernel/ccl] building collective/rdma..."; \
		$(MAKE) -C "$(UCCL_RDMA_ROOT)" build; \
	fi

%.o: %.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

%.o: %.cpp
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

%.o: %.cu
	$(NVCC) -MMD -MP -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(COMMON_OBJ) $(TEST_MAIN_OBJ) $(TEST_PERSISTENT_OBJ) $(TEST_COPY_ENGINE_OBJ): | ensure-gdrcopy ensure-uccl-rdma

$(TEST_MAIN_BIN): $(COMMON_OBJ) $(TEST_MAIN_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

$(TEST_PERSISTENT_BIN): $(COMMON_OBJ) $(TEST_PERSISTENT_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

$(TEST_COPY_ENGINE_BIN): $(COMMON_OBJ) $(TEST_COPY_ENGINE_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

clean:
	rm -f *.o *.d test/*.d $(ALL_BIN)

-include $(COMMON_OBJ:.o=.d) $(TEST_MAIN_OBJ:.o=.d) $(TEST_PERSISTENT_OBJ:.o=.d) $(TEST_COPY_ENGINE_OBJ:.o=.d)
