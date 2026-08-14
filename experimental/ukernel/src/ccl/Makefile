CXX := /usr/bin/g++
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
CONDA_LIB_HOME ?= /usr/lib
UCCL_ROOT := $(abspath ../../../..)
GDRCOPY_INCLUDEDIR ?= /usr/include
GDRCOPY_LIBDIR ?=
GDRCOPY_LIBDIR_FLAGS := $(if $(strip $(GDRCOPY_LIBDIR)),-L$(GDRCOPY_LIBDIR),)

# GPU architecture + per-arch reduce-kernel defaults (ILP / TMA / smem).
include ../../mk/arch_defaults.mk

ifeq ($(DEBUG),1)
CCL_OPT_FLAGS := -O0 -g
else
CCL_OPT_FLAGS := -O3
endif

CXXFLAGS := -std=c++17 $(CCL_OPT_FLAGS) -Wall -fPIC
NVCCFLAGS := -std=c++17 $(CCL_OPT_FLAGS) -Xcompiler -Wall -Xcompiler -fPIC $(NVCC_ARCH_FLAGS)
NVCCFLAGS += -DUK_REDUCE_ILP=$(REDUCE_ILP) -DUK_REDUCE_SMEM_KB=$(REDUCE_SMEM_KB) -DUK_TMA_REDUCE=$(TMA_REDUCE) -DUK_TMA_WARPSPEC=$(TMA_WARPSPEC)
CXXFLAGS += -DUK_REDUCE_SMEM_KB=$(REDUCE_SMEM_KB)

VPATH := .:test/common:test/unit:test/integration:../transport:../device:../device/fifo
GCC_MACHINE := $(shell $(CXX) -dumpmachine 2>/dev/null || echo x86_64-linux-gnu)
INCLUDES := -I. -I./test -I./test/common -I./test/unit -I./test/integration \
            -I../transport -I../device -I../device/fifo \
            -I../../include -I$(UCCL_ROOT) -I$(UCCL_ROOT)/include \
            -I$(CUDA_HOME)/include -I$(GDRCOPY_INCLUDEDIR) -I/usr/include \
            -I/usr/local/include -I/usr/include/$(GCC_MACHINE)
CXX_RPATH_FLAGS := -Wl,-rpath,$(CUDA_HOME)/lib64 \
                   -Wl,-rpath,/usr/lib/$(GCC_MACHINE) \
                   -Wl,-rpath-link,/lib/$(GCC_MACHINE):/usr/lib/$(GCC_MACHINE)
ifneq ($(strip $(GDRCOPY_LIBDIR)),)
CXX_RPATH_FLAGS += -Wl,-rpath,$(GDRCOPY_LIBDIR)
endif
TRANSPORT_LIBS := -L$(CONDA_LIB_HOME) -L$(CUDA_HOME)/lib64 \
                  -lcudart -lcuda -lz -libverbs \
                  -lnl-3 -lnl-route-3 -lpthread
DEVICE_LIBS := $(GDRCOPY_LIBDIR_FLAGS) -lgdrapi -lnuma

CCL_CORE_SRC := coll_algo.cc lower.cc
CCL_CORE_OBJ := $(CCL_CORE_SRC:.cc=.o)
REAL_BACKEND_SRC := backend/device_backend.cc backend/transport_backend.cc backend/signal_backend.cc backend/backend.cc executor_factory.cc
REAL_BACKEND_OBJ := $(REAL_BACKEND_SRC:.cc=.o)
TRANSPORT_SRC := ../transport/communicator.cc \
                 ../transport/request_tracker.cc \
                 ../transport/bounce_buffer_pool.cc \
                 ../transport/memory/mr_manager.cc \
                 ../transport/memory/ipc_manager.cc \
                 ../transport/memory/shm_manager.cc \
                 ../transport/oob/oob.cc \
                 ../transport/oob/oob_socket.cc \
                 ../transport/adapter/shmring_exchanger.cc \
                  ../transport/adapter/tcp_adapter.cc \
                  ../transport/adapter/ipc_adapter.cc \
                  ../transport/adapter/rdma_adapter.cc \
                  ../transport/util/utils.cc
TRANSPORT_OBJ := transport_communicator.o \
                 transport_request_tracker.o \
                 transport_bounce_buffer_pool.o \
                 transport_memory_mr_manager.o \
                 transport_memory_ipc_manager.o \
                 transport_memory_shm_manager.o \
                 transport_oob_oob.o transport_oob_oob_socket.o \
                  transport_adapter_shmring_exchanger.o \
                   transport_adapter_tcp_adapter.o \
                  transport_adapter_ipc_adapter.o \
                  transport_adapter_rdma_adapter.o \
                  transport_util_utils.o
DEVICE_CPP_OBJ := device_c2d_fifo.o device_d2c_fifo.o device_sm_fifo.o \
                  device_worker.o
DEVICE_CU_OBJ := device_persistent_kernel_ops.o
DEVICE_OBJ := $(DEVICE_CPP_OBJ) $(DEVICE_CU_OBJ)

UNIT_TEST_BINS := test_modules test_async test_spray_executor
INTEGRATION_TEST_BINS := test_perf_p2p_copy test_transport_backend_e2e test_device_backend_e2e test_signal_backend_e2e test_put_signal_e2e test_spray_executor_e2e test_rdma_l2_flush test_perf_spray_allreduce
TEST_BINS := $(UNIT_TEST_BINS) $(INTEGRATION_TEST_BINS)

.PHONY: all test test-unit test-integration unit integration real_backends clean

all: real_backends

test: test-unit test-integration
	@true

test-unit: $(UNIT_TEST_BINS)
	./test_modules
	./test_async
	./test_spray_executor

test-integration: $(INTEGRATION_TEST_BINS)
	@echo "Built test_perf_p2p_copy. Run it manually in two terminals (server + client); see src/ccl/README.md."

unit: test-unit

integration: test-integration

real_backends: $(REAL_BACKEND_OBJ)

test_modules: $(CCL_CORE_OBJ) test/unit/test_modules.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) -lpthread

test_async: $(CCL_CORE_OBJ) executor.o backend/backend.o backend/signal_backend.o \
		$(TRANSPORT_OBJ) test/unit/test_async.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(CXX_RPATH_FLAGS) \
		-L$(CUDA_HOME)/lib64 -lcudart -lcuda $(TRANSPORT_LIBS) -lpthread

test_spray_executor: $(CCL_CORE_OBJ) executor.o backend/backend.o backend/signal_backend.o \
		$(TRANSPORT_OBJ) test/unit/test_spray_executor.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(CXX_RPATH_FLAGS) \
		-L$(CUDA_HOME)/lib64 -lcudart -lcuda $(TRANSPORT_LIBS) -lpthread

test_perf_p2p_copy: $(CCL_CORE_OBJ) executor.o backend/backend.o backend/device_backend.o backend/transport_backend.o \
		$(DEVICE_OBJ) $(TRANSPORT_OBJ) test/integration/test_perf_p2p_copy.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(CXX_RPATH_FLAGS) \
		-L$(CUDA_HOME)/lib64 -lcudart -lcuda $(TRANSPORT_LIBS) $(DEVICE_LIBS) -lpthread

test/integration/test_perf_p2p_copy.o: test/integration/test_perf_p2p_copy.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

test_transport_backend_e2e: $(CCL_CORE_OBJ) executor.o backend/backend.o backend/transport_backend.o \
		$(TRANSPORT_OBJ) test/unit/test_transport_backend_e2e.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(CXX_RPATH_FLAGS) \
		-L$(CUDA_HOME)/lib64 -lcudart -lcuda $(TRANSPORT_LIBS) -lpthread

test/unit/test_transport_backend_e2e.o: test/unit/test_transport_backend_e2e.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

test_device_backend_e2e: $(CCL_CORE_OBJ) executor.o backend/backend.o backend/device_backend.o \
		$(DEVICE_OBJ) $(TRANSPORT_OBJ) test/unit/test_device_backend_e2e.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(CXX_RPATH_FLAGS) \
		-L$(CUDA_HOME)/lib64 -lcudart -lcuda $(TRANSPORT_LIBS) $(DEVICE_LIBS) -lpthread

test/unit/test_device_backend_e2e.o: test/unit/test_device_backend_e2e.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

test_signal_backend_e2e: $(CCL_CORE_OBJ) executor.o backend/backend.o backend/signal_backend.o \
		$(TRANSPORT_OBJ) test/unit/test_signal_backend_e2e.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(CXX_RPATH_FLAGS) \
		-L$(CUDA_HOME)/lib64 -lcudart -lcuda $(TRANSPORT_LIBS) -lpthread

test/unit/test_signal_backend_e2e.o: test/unit/test_signal_backend_e2e.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

test_put_signal_e2e: $(CCL_CORE_OBJ) $(TRANSPORT_OBJ) test/integration/test_put_signal_e2e.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(CXX_RPATH_FLAGS) \
		-L$(CUDA_HOME)/lib64 -lcudart -lcuda $(TRANSPORT_LIBS) -lpthread

test/integration/test_put_signal_e2e.o: test/integration/test_put_signal_e2e.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

test_spray_executor_e2e: $(CCL_CORE_OBJ) executor.o executor_factory.o backend/backend.o backend/device_backend.o backend/transport_backend.o backend/signal_backend.o \
		$(DEVICE_OBJ) $(TRANSPORT_OBJ) test/integration/test_spray_executor_e2e.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(CXX_RPATH_FLAGS) \
		-L$(CUDA_HOME)/lib64 -lcudart -lcuda $(TRANSPORT_LIBS) $(DEVICE_LIBS) -lpthread

test/integration/test_spray_executor_e2e.o: test/integration/test_spray_executor_e2e.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

test_rdma_l2_flush: $(CCL_CORE_OBJ) executor.o executor_factory.o backend/backend.o backend/device_backend.o backend/transport_backend.o backend/signal_backend.o \
		$(DEVICE_OBJ) $(TRANSPORT_OBJ) test/integration/test_rdma_l2_flush.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(CXX_RPATH_FLAGS) \
		-L$(CUDA_HOME)/lib64 -lcudart -lcuda $(TRANSPORT_LIBS) $(DEVICE_LIBS) -lpthread

test/integration/test_rdma_l2_flush.o: test/integration/test_rdma_l2_flush.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

test_perf_spray_allreduce: $(CCL_CORE_OBJ) executor.o executor_factory.o backend/backend.o backend/device_backend.o backend/transport_backend.o backend/signal_backend.o \
		$(DEVICE_OBJ) $(TRANSPORT_OBJ) test/integration/test_perf_spray_allreduce.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(CXX_RPATH_FLAGS) \
		-L$(CUDA_HOME)/lib64 -lcudart -lcuda $(TRANSPORT_LIBS) $(DEVICE_LIBS) -lpthread

test/integration/test_perf_spray_allreduce.o: test/integration/test_perf_spray_allreduce.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

%.o: %.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

transport_%.o: ../transport/%.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

transport_memory_%.o: ../transport/memory/%.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

transport_oob_%.o: ../transport/oob/%.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

transport_adapter_%.o: ../transport/adapter/%.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

transport_util_%.o: ../transport/util/%.cc
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

# NCCL compatibility — driven by top-level `make nccl`.
NCCL_PREFIX ?= $(CURDIR)/build/nccl
NCCL_SO := libnccl.so
NCCL_OBJ := nccl.o

.PHONY: nccl_dev nccl
nccl_dev: $(NCCL_SO)
	@mkdir -p $(NCCL_PREFIX)/lib $(NCCL_PREFIX)/include
	# ABI-compatible drop-in: native libnccl's SONAME is libnccl.so.2, and
	# perftests/engines built against native NCCL record that name — install
	# the shim under it plus the dev symlink so LD_LIBRARY_PATH swaps work.
	cp $(NCCL_SO) $(NCCL_PREFIX)/lib/libnccl.so.2
	ln -sf libnccl.so.2 $(NCCL_PREFIX)/lib/libnccl.so
	cp ../../include/nccl.h $(NCCL_PREFIX)/include/
	cp ../../include/gpu_rt.h $(NCCL_PREFIX)/include/
	@echo "NCCL compat installed to $(NCCL_PREFIX)/"

$(NCCL_SO): $(CCL_CORE_OBJ) executor.o executor_factory.o backend/backend.o \
		backend/device_backend.o backend/transport_backend.o backend/signal_backend.o \
		$(DEVICE_OBJ) $(TRANSPORT_OBJ) $(NCCL_OBJ)
	$(CXX) -shared -o $@ $^ $(CXXFLAGS) $(INCLUDES) \
		-L$(CUDA_HOME)/lib64 -lcudart -lcuda $(TRANSPORT_LIBS) $(DEVICE_LIBS) -lpthread \
		-Wl,-soname,libnccl.so.2

$(NCCL_OBJ): nccl.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

clean:
	rm -f *.o *.d backend/*.o backend/*.d \
		test/*.o test/*.d \
		test/common/*.o test/common/*.d \
		test/unit/*.o test/unit/*.d \
		test/integration/*.o test/integration/*.d \
		transport_*.o transport_*.d device_*.o device_*.d $(TEST_BINS)
	find . -type f \( -name '*.o' -o -name '*.d' -o -name '.DS_Store' \) -delete

# Cover every generated .d (wildcard): the previous explicit list missed
# several test objects, which then went stale on header changes (ODR).
-include $(wildcard *.d backend/*.d test/common/*.d test/unit/*.d \
		test/integration/*.d)
