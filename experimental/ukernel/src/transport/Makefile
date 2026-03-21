CXX            := /usr/bin/g++
CUDA_PATH      ?= /usr/local/cuda
CONDA_LIB_HOME ?= /usr/lib

CXXFLAGS := -O0 -std=c++17 -Wall -Wno-unused-function -Wno-sign-compare -Wno-reorder -Wno-unused-variable -Wno-unused-label -Wno-unused-but-set-variable -Wno-stringop-overread -Wno-narrowing -pthread -fPIC -I$(CONDA_LIB_HOME)/../include
LDFLAGS  := -L$(CONDA_LIB_HOME) -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 -lcudart -lcuda -lgflags -lz -libverbs -lnl-3 -lnl-route-3 -lpthread

UCCL_ROOT := $(abspath ../../../..)
UKERNEL_ROOT := $(abspath ../..)
UCCL_RDMA_ROOT := $(UCCL_ROOT)/collective/rdma
UCCL_RDMA_STATIC := $(UCCL_RDMA_ROOT)/librdma.a
UCCL_CFLAGS := -I$(UCCL_ROOT) -I$(UCCL_ROOT)/collective/rdma -I$(UCCL_ROOT)/include
UCCL_LDFLAGS := $(UCCL_RDMA_STATIC) -libverbs -lrdmacm

CXXFLAGS += $(UCCL_CFLAGS)
LDFLAGS += $(UCCL_LDFLAGS)

LOCAL_INCLUDES := -I$(UCCL_ROOT)/experimental/ukernel/include -I$(UCCL_ROOT)/include -I. -Itest -I../../include -I$(CUDA_HOME)/include

SRC_CPP := communicator.cc ipc_cache.cc ipc_channel.cc memory_registry.cc oob.cc oob_shm.cc oob_socket.cc peer_session.cc request.cc uccl_transport_adapter.cc utils.cc
TEST_CPP := test/test_main.cc test/test_transport_core.cc test/test_transport_communicator.cc test/test_transport_oob_socket.cc test/test_transport_oob_shm.cc test/test_transport_utils.cc

OBJ_CPP := $(SRC_CPP:.cc=.o)
OBJ_TEST := $(TEST_CPP:.cc=.o)
TEST_BIN := test_transport_main
UCCL_RDMA_DEPS := $(wildcard $(UCCL_RDMA_ROOT)/*.cc $(UCCL_RDMA_ROOT)/*.h)

.PHONY: all test suite clean

all: $(TEST_BIN)
test: $(TEST_BIN)
suite: $(TEST_BIN)
	./test/run_transport_suite.sh ./$(TEST_BIN)

$(UCCL_RDMA_STATIC): $(UCCL_RDMA_DEPS)
	@echo "[ukernel/transport] rebuilding collective/rdma static library..."
	$(MAKE) -C "$(UCCL_RDMA_ROOT)" librdma.a

%.o: %.cc
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(TEST_BIN): $(OBJ_CPP) $(OBJ_TEST) $(UCCL_RDMA_STATIC)
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(OBJ_CPP) $(OBJ_TEST) $(TEST_BIN) *.d test/*.d

-include $(OBJ_CPP:.o=.d) $(OBJ_TEST:.o=.d)
