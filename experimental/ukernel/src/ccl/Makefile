CXX := g++
SM ?= 80
CUDA_HOME ?= /usr/local/cuda
CONDA_LIB_HOME ?= /usr/lib
UCCL_ROOT := $(abspath ../../../..)
GDRCOPY_ROOT := $(UCCL_ROOT)/thirdparty/gdrcopy
GDRCOPY_INCLUDEDIR := $(GDRCOPY_ROOT)/include
UCCL_RDMA_ROOT := $(UCCL_ROOT)/collective/rdma
UCCL_RDMA_STATIC := $(UCCL_RDMA_ROOT)/librdma.a

SM_NUM := $(shell printf '%s\n' "$(SM)" | sed 's/[^0-9].*//')
ENABLE_TMA := $(shell if [ -n "$(SM_NUM)" ] && [ "$(SM_NUM)" -ge 90 ]; then echo 1; else echo 0; fi)

CXXFLAGS := -std=c++17 -O3 -g -Wall
CXXFLAGS += -DUKERNEL_ENABLE_TMA=$(ENABLE_TMA)

VPATH := .:test:../transport:../device:../device/fifo
INCLUDES := -I. -I./test -I../transport -I../device -I../device/fifo \
            -I../../include -I$(UCCL_ROOT) -I$(UCCL_ROOT)/include \
            -I$(CUDA_HOME)/include -I$(GDRCOPY_INCLUDEDIR)
TRANSPORT_LDFLAGS := -L$(CONDA_LIB_HOME) -L$(CUDA_HOME)/lib64 \
                     -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lcuda \
                     -lgflags -lz -libverbs -lnl-3 -lnl-route-3 -lrdmacm \
                     -lpthread $(UCCL_RDMA_STATIC)

CCL_CORE_SRC := topology.cc plan.cc selector.cc executor.cc
CCL_CORE_OBJ := $(CCL_CORE_SRC:.cc=.o)
REAL_BACKEND_SRC := backend/device_backend.cc backend/transport_backend.cc
REAL_BACKEND_OBJ := $(REAL_BACKEND_SRC:.cc=.o)
TRANSPORT_SRC := ../transport/communicator.cc ../transport/ipc_cache.cc \
                 ../transport/ipc_channel.cc ../transport/memory_registry.cc \
                 ../transport/oob.cc ../transport/oob_shm.cc \
                 ../transport/oob_socket.cc ../transport/peer_session.cc \
                 ../transport/request.cc ../transport/uccl_transport_adapter.cc \
                 ../transport/utils.cc
TRANSPORT_OBJ := $(patsubst ../transport/%.cc,transport_%.o,$(TRANSPORT_SRC))

TEST_BINS := test_components test_multiprocess_collective

.PHONY: all test real_backends clean

all: $(TEST_BINS) real_backends

test: all
	./test_components
	bash ./test/run_ccl_multiprocess_suite.sh ./test_multiprocess_collective

real_backends: $(REAL_BACKEND_OBJ)

test_components: $(CCL_CORE_OBJ) test/test_components.o
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) -lpthread

$(UCCL_RDMA_STATIC):
	$(MAKE) -C "$(UCCL_RDMA_ROOT)" librdma.a

test_multiprocess_collective: $(CCL_CORE_OBJ) backend/transport_backend.o \
		$(TRANSPORT_OBJ) test/test_multiprocess_collective.o $(UCCL_RDMA_STATIC)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(INCLUDES) $(TRANSPORT_LDFLAGS)

%.o: %.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

transport_%.o: ../transport/%.cc
	$(CXX) -MMD -MP -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

clean:
	rm -f *.o *.d backend/*.o backend/*.d test/*.o test/*.d \
		transport_*.o transport_*.d $(TEST_BINS)

-include $(CCL_CORE_OBJ:.o=.d) $(REAL_BACKEND_OBJ:.o=.d) \
	$(TRANSPORT_OBJ:.o=.d) test/test_components.d test/test_multiprocess_collective.d
