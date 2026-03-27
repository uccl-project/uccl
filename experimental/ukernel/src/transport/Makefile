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

LOCAL_INCLUDES := -I$(UCCL_ROOT)/experimental/ukernel/include -I$(UCCL_ROOT)/include \
	-I. -Itest -Itest/common -Itest/unit -Itest/integration \
	-I../../include -I$(CUDA_HOME)/include

SRC_CPP := communicator.cc host_bounce_pool.cc ipc_cache.cc ipc_channel.cc memory_registry.cc oob.cc oob_shm.cc oob_socket.cc peer_session.cc request.cc tcp_transport_adapter.cc uccl_transport_adapter.cc utils.cc
TEST_UNIT_CPP := test/unit/test_transport_unit_main.cc \
	test/unit/test_transport_core.cc \
	test/unit/test_transport_oob_socket.cc \
	test/unit/test_transport_oob_shm.cc \
	test/unit/test_transport_oob_exchangeable.cc \
	test/unit/test_transport_peer_session.cc \
	test/unit/test_transport_host_bounce_pool.cc \
	test/unit/test_transport_tcp_adapter.cc \
	test/unit/test_transport_utils.cc
TEST_INTEGRATION_CPP := test/integration/test_transport_integration_main.cc \
	test/integration/test_transport_communicator.cc

OBJ_CPP := $(SRC_CPP:.cc=.o)
OBJ_TEST_UNIT := $(TEST_UNIT_CPP:.cc=.o)
OBJ_TEST_INTEGRATION := $(TEST_INTEGRATION_CPP:.cc=.o)
TEST_UNIT_BIN := test_transport_unit
TEST_INTEGRATION_BIN := test_transport_integration
UCCL_RDMA_DEPS := $(wildcard $(UCCL_RDMA_ROOT)/*.cc $(UCCL_RDMA_ROOT)/*.h)

.PHONY: all test test-unit test-integration unit integration suite clean

all: $(TEST_UNIT_BIN) $(TEST_INTEGRATION_BIN)

test: test-unit test-integration

test-unit: $(TEST_UNIT_BIN)
	./$(TEST_UNIT_BIN) core
	./$(TEST_UNIT_BIN) oob-socket
	./$(TEST_UNIT_BIN) oob-socket-meta --world-size 4
	./$(TEST_UNIT_BIN) oob-serde
	./$(TEST_UNIT_BIN) peer-session
	./$(TEST_UNIT_BIN) host-bounce
	./$(TEST_UNIT_BIN) tcp-adapter
	./$(TEST_UNIT_BIN) utils-host-id

test-integration: $(TEST_INTEGRATION_BIN)
	./$(TEST_INTEGRATION_BIN) communicator-local

unit: test-unit

integration: test-integration

suite: $(TEST_INTEGRATION_BIN)
	./test/integration/run_transport_suite.sh ./$(TEST_INTEGRATION_BIN)

$(UCCL_RDMA_STATIC): $(UCCL_RDMA_DEPS)
	@echo "[ukernel/transport] rebuilding collective/rdma static library..."
	$(MAKE) -C "$(UCCL_RDMA_ROOT)" librdma.a

%.o: %.cc
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(TEST_UNIT_BIN): $(OBJ_CPP) $(OBJ_TEST_UNIT) $(UCCL_RDMA_STATIC)
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

$(TEST_INTEGRATION_BIN): $(OBJ_CPP) $(OBJ_TEST_INTEGRATION) $(UCCL_RDMA_STATIC)
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(OBJ_CPP) $(OBJ_TEST_UNIT) $(OBJ_TEST_INTEGRATION) \
		$(TEST_UNIT_BIN) $(TEST_INTEGRATION_BIN) *.d \
		test/*.o test/*.d \
		test/common/*.d test/unit/*.o test/unit/*.d \
		test/integration/*.o test/integration/*.d

-include $(OBJ_CPP:.o=.d) $(OBJ_TEST_UNIT:.o=.d) $(OBJ_TEST_INTEGRATION:.o=.d)
