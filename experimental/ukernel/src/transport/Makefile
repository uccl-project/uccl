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

ifeq ($(USE_REDIS_OOB),1)
PKG_CONFIG := pkg-config
REDIS_CFLAGS := $(shell $(PKG_CONFIG) --cflags hiredis)
REDIS_LIBS   := $(shell $(PKG_CONFIG) --libs hiredis)
CXXFLAGS += $(REDIS_CFLAGS) -DUSE_REDIS_OOB
LDFLAGS += $(REDIS_LIBS)
endif

LOCAL_INCLUDES := -I$(UCCL_ROOT)/experimental/ukernel/include -I$(UCCL_ROOT)/include -I. -Itest -I../../include -I$(CUDA_HOME)/include

SRC_CPP := communicator.cc ipc_channel.cc memory_registry.cc oob_redis.cc oob_socket.cc oob_uds.cc request.cc transport_engine.cc uccl_transport_adapter.cc utils.cc
TEST_CPP := test/test_main.cc test/test_transport_core.cc test/test_transport_communicator.cc test/test_transport_oob_socket.cc test/test_transport_oob_redis.cc test/test_transport_oob_uds.cc test/test_transport_utils.cc

OBJ_CPP := $(SRC_CPP:.cc=.o)
OBJ_TEST := $(TEST_CPP:.cc=.o)
TEST_BIN := test_transport_main

.PHONY: all test suite clean ensure-uccl-rdma

all: $(TEST_BIN)
test: $(TEST_BIN)
suite: $(TEST_BIN)
	./test/run_transport_suite.sh ./$(TEST_BIN)

ensure-uccl-rdma:
	@if [ ! -f "$(UCCL_RDMA_STATIC)" ]; then \
		echo "[ukernel/transport] building collective/rdma..."; \
		$(MAKE) -C "$(UCCL_RDMA_ROOT)" build; \
	fi

%.o: %.cc
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(TEST_BIN): $(OBJ_CPP) $(OBJ_TEST) | ensure-uccl-rdma
	$(CXX) $(LOCAL_INCLUDES) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(OBJ_CPP) $(OBJ_TEST) $(TEST_BIN) *.d test/*.d

-include $(OBJ_CPP:.o=.d) $(OBJ_TEST:.o=.d)
