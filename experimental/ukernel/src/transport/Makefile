CXX            := /usr/bin/g++
CUDA_PATH      ?= /usr/local/cuda
# Auto-detect CUDA if default path doesn't exist (e.g. /usr/local/cuda-13).
ifeq ($(wildcard $(CUDA_PATH)/bin/nvcc),)
  _CUDA_AUTO := $(firstword $(sort $(wildcard /usr/local/cuda-*)))
  ifneq ($(_CUDA_AUTO),)
    CUDA_PATH := $(_CUDA_AUTO)
  endif
endif
# Alias for sub-makefiles that expect CUDA_HOME.
CUDA_HOME      ?= $(CUDA_PATH)
GCC_MACHINE    := $(shell $(CXX) -dumpmachine 2>/dev/null || echo x86_64-linux-gnu)
CONDA_LIB_HOME ?= /usr/lib
DEBUG          ?= 0

ifeq ($(DEBUG),1)
CXX_OPT_FLAGS := -O0 -g
else
CXX_OPT_FLAGS := -O3
endif

CXXFLAGS := $(CXX_OPT_FLAGS) -std=c++17 -Wall -Wno-unused-function -Wno-sign-compare -Wno-reorder -Wno-unused-variable -Wno-unused-label -Wno-unused-but-set-variable -Wno-stringop-overread -Wno-narrowing -pthread -fPIC -I$(CONDA_LIB_HOME)/../include
LDFLAGS  := -L$(CONDA_LIB_HOME) -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 \
            -Wl,-rpath,/usr/lib/$(GCC_MACHINE) \
            -Wl,-rpath-link,/lib/$(GCC_MACHINE):/usr/lib/$(GCC_MACHINE) \
            -lcudart -lcuda -lz -libverbs -lnl-3 -lnl-route-3 -lpthread


UCCL_ROOT := $(abspath ../../../..)
UKERNEL_ROOT := $(abspath ../..)

LOCAL_INCLUDES := -I$(UCCL_ROOT)/experimental/ukernel/include -I$(UCCL_ROOT)/include \
	-I. -Itest -Itest/common -Itest/unit -Itest/integration \
	-I../../include -I$(CUDA_HOME)/include

SRC_CPP := communicator.cc request_tracker.cc bounce_buffer_pool.cc adapter/ipc_adapter.cc adapter/tcp_adapter.cc adapter/rdma_adapter.cc adapter/shmring_exchanger.cc memory/mr_manager.cc memory/ipc_manager.cc memory/shm_manager.cc oob/oob.cc oob/oob_socket.cc util/utils.cc
TEST_UNIT_CPP := test/unit/test_transport_unit_main.cc \
	test/unit/test_transport_core.cc \
	test/unit/test_transport_oob_socket.cc \
	test/unit/test_transport_oob_shm.cc \
	test/unit/test_transport_oob_exchangeable.cc \
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

.PHONY: all test test-unit test-integration unit integration suite clean

all: $(TEST_UNIT_BIN) $(TEST_INTEGRATION_BIN)

test: test-unit test-integration

test-unit: $(TEST_UNIT_BIN)
	./$(TEST_UNIT_BIN) core
	./$(TEST_UNIT_BIN) oob-socket
	./$(TEST_UNIT_BIN) oob-socket-meta --world-size 4
	./$(TEST_UNIT_BIN) oob-serde
	./$(TEST_UNIT_BIN) host-bounce
	./$(TEST_UNIT_BIN) tcp-adapter
	./$(TEST_UNIT_BIN) utils-host-id

test-integration: $(TEST_INTEGRATION_BIN)
	./$(TEST_INTEGRATION_BIN) communicator-local

unit: test-unit

integration: test-integration

suite: $(TEST_INTEGRATION_BIN)
	./test/integration/run_transport_suite.sh ./$(TEST_INTEGRATION_BIN)

%.o: %.cc
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(TEST_UNIT_BIN): $(OBJ_CPP) $(OBJ_TEST_UNIT)
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

$(TEST_INTEGRATION_BIN): $(OBJ_CPP) $(OBJ_TEST_INTEGRATION)
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(OBJ_CPP) $(OBJ_TEST_UNIT) $(OBJ_TEST_INTEGRATION) \
		$(TEST_UNIT_BIN) $(TEST_INTEGRATION_BIN) *.d \
		test/*.o test/*.d \
		test/common/*.d test/unit/*.o test/unit/*.d \
		test/integration/*.o test/integration/*.d
	find . -type f \( -name '*.o' -o -name '*.d' -o -name '.DS_Store' \) -delete

-include $(OBJ_CPP:.o=.d) $(OBJ_TEST_UNIT:.o=.d) $(OBJ_TEST_INTEGRATION:.o=.d)
