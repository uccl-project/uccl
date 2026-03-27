CXX := g++
NVCC := nvcc
DEFAULT_SMS := 80 86 89
CUDA_HOME ?= /usr/local/cuda
CONDA_LIB_HOME ?= /usr/lib
UCCL_ROOT := $(abspath ../../../..)
GDRCOPY_ROOT := $(UCCL_ROOT)/thirdparty/gdrcopy
GDRCOPY_INCLUDEDIR := $(GDRCOPY_ROOT)/include
GDRCOPY_LIBDIR := $(GDRCOPY_ROOT)/src
UCCL_RDMA_ROOT := $(UCCL_ROOT)/collective/rdma
UCCL_RDMA_STATIC := $(UCCL_RDMA_ROOT)/librdma.a

ifeq ($(origin SM), undefined)
NVCC_ARCH_FLAGS := $(foreach sm,$(DEFAULT_SMS),-gencode arch=compute_$(sm),code=sm_$(sm))
ENABLE_TMA := 0
else
SM_NUM := $(shell printf '%s\n' "$(SM)" | sed 's/[^0-9].*//')
NVCC_ARCH_FLAGS := -arch=sm_$(SM)
ENABLE_TMA := $(shell if [ -n "$(SM_NUM)" ] && [ "$(SM_NUM)" -ge 90 ]; then echo 1; else echo 0; fi)
endif

CXXFLAGS := -std=c++17 -O3 -g -Wall
CXXFLAGS += -DUKERNEL_ENABLE_TMA=$(ENABLE_TMA)
NVCCFLAGS := -std=c++17 -O3 -g -Xcompiler -Wall $(NVCC_ARCH_FLAGS)
NVCCFLAGS += -DUKERNEL_ENABLE_TMA=$(ENABLE_TMA)

VPATH := .:test/common:test/unit:test/integration:../transport:../device:../device/fifo
INCLUDES := -I. -I./test -I./test/common -I./test/unit -I./test/integration \
            -I../transport -I../device -I../device/fifo \
            -I../../include -I$(UCCL_ROOT) -I$(UCCL_ROOT)/include \
            -I$(CUDA_HOME)/include -I$(GDRCOPY_INCLUDEDIR)
CXX_RPATH_FLAGS := -Wl,-rpath,$(CUDA_HOME)/lib64 \
                   -Wl,-rpath,$(GDRCOPY_LIBDIR)
NVCC_RPATH_FLAGS := -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64 \
                    -Xlinker -rpath -Xlinker $(GDRCOPY_LIBDIR)
TRANSPORT_LIBS := -L$(CONDA_LIB_HOME) -L$(CUDA_HOME)/lib64 \
                  -lcudart -lcuda -lgflags -lz -libverbs \
                  -lnl-3 -lnl-route-3 -lrdmacm -lpthread \
                  $(UCCL_RDMA_STATIC)
DEVICE_LIBS := -L$(GDRCOPY_LIBDIR) -lgdrapi -lnuma

CCL_CORE_SRC := topology.cc plan.cc selector.cc executor.cc
CCL_CORE_OBJ := $(CCL_CORE_SRC:.cc=.o)
REAL_EXECUTOR_SRC := executor_real.cc
REAL_EXECUTOR_OBJ := $(REAL_EXECUTOR_SRC:.cc=.o)
REAL_BACKEND_SRC := backend/device_backend.cc backend/transport_backend.cc
REAL_BACKEND_OBJ := $(REAL_BACKEND_SRC:.cc=.o)
TRANSPORT_SRC := ../transport/communicator.cc ../transport/host_bounce_pool.cc \
                 ../transport/ipc_cache.cc \
                 ../transport/ipc_channel.cc ../transport/memory_registry.cc \
                 ../transport/oob.cc ../transport/oob_shm.cc \
                 ../transport/oob_socket.cc ../transport/peer_session.cc \
                 ../transport/request.cc ../transport/tcp_transport_adapter.cc \
                 ../transport/uccl_transport_adapter.cc \
                 ../transport/utils.cc
TRANSPORT_OBJ := $(patsubst ../transport/%.cc,transport_%.o,$(TRANSPORT_SRC))
DEVICE_CPP_OBJ := device_c2d_fifo.o device_d2c_fifo.o device_sm_fifo.o \
                  device_worker.o
DEVICE_CU_OBJ := device_persistent_kernel_ops.o
DEVICE_OBJ := $(DEVICE_CPP_OBJ) $(DEVICE_CU_OBJ)

UNIT_TEST_BINS := test_components
INTEGRATION_TEST_BINS := test_device_backend test_transport_backend test_multiprocess_collective
TEST_BINS := $(UNIT_TEST_BINS) $(INTEGRATION_TEST_BINS)

.PHONY: all test test-unit test-integration unit integration real_backends clean

all: $(TEST_BINS) real_backends

test: test-unit test-integration
	@true

test-unit: $(UNIT_TEST_BINS)
	./test_components

test-integration: $(INTEGRATION_TEST_BINS)
	./test_device_backend
	bash ./test/integration/run_transport_backend_suite.sh ./test_transport_backend
	bash ./test/integration/run_ccl_multiprocess_suite.sh ./test_multiprocess_collective

unit: test-unit

integration: test-integration

real_backends: $(REAL_BACKEND_OBJ)

test_components: $(CCL_CORE_OBJ) test/unit/test_components.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) -lpthread

$(UCCL_RDMA_STATIC):
	$(MAKE) -C "$(UCCL_RDMA_ROOT)" librdma.a

test_multiprocess_collective: $(CCL_CORE_OBJ) backend/transport_backend.o \
		backend/device_backend.o $(REAL_EXECUTOR_OBJ) $(TRANSPORT_OBJ) $(DEVICE_OBJ) \
		test/integration/test_multiprocess_collective.o $(UCCL_RDMA_STATIC)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(INCLUDES) $(NVCC_RPATH_FLAGS) \
		$(TRANSPORT_LIBS) $(DEVICE_LIBS)

test_device_backend: backend/device_backend.o $(DEVICE_OBJ) \
		test/integration/test_device_backend.o
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(INCLUDES) $(NVCC_RPATH_FLAGS) \
		-L$(CUDA_HOME)/lib64 -lcudart -lcuda $(DEVICE_LIBS) -lpthread

test_transport_backend: backend/transport_backend.o $(TRANSPORT_OBJ) \
		test/integration/test_transport_backend.o $(UCCL_RDMA_STATIC)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(CXX_RPATH_FLAGS) \
		$(TRANSPORT_LIBS)

%.o: %.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

transport_%.o: ../transport/%.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

device_c2d_fifo.o: ../device/fifo/c2d_fifo.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

device_d2c_fifo.o: ../device/fifo/d2c_fifo.cpp
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

device_sm_fifo.o: ../device/fifo/sm_fifo.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

device_worker.o: ../device/worker.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

backend/device_backend.o: backend/device_backend.cc
	$(NVCC) -MMD -MP -x cu -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

device_persistent_kernel_ops.o: ../device/persistent_kernel_ops.cu
	$(NVCC) -MMD -MP -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

clean:
	rm -f *.o *.d backend/*.o backend/*.d \
		test/*.o test/*.d \
		test/common/*.o test/common/*.d \
		test/unit/*.o test/unit/*.d \
		test/integration/*.o test/integration/*.d \
		transport_*.o transport_*.d device_*.o device_*.d $(TEST_BINS)

-include $(CCL_CORE_OBJ:.o=.d) $(REAL_BACKEND_OBJ:.o=.d) \
	$(REAL_EXECUTOR_OBJ:.o=.d) \
	$(TRANSPORT_OBJ:.o=.d) $(DEVICE_OBJ:.o=.d) \
	test/unit/test_components.d \
	test/integration/test_device_backend.d \
	test/integration/test_transport_backend.d \
	test/integration/test_multiprocess_collective.d
