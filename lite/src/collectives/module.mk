# Algorithm objects used by the common runtime. Historical NCCL experiment
# targets have separate objects/policies; their knobs cannot change these.
include src/collectives/peer_packet_flags.mk

HOST_PACKET_BYTES ?= 1048576
uccl_lite_host_packet_device_flags = -DUCCL_LITE_MP_PACKET_BYTES=$(1) -DUCCL_LITE_MP_AG_BLOCKS=64 -DUCCL_LITE_MP_AG_SMALL_BLOCKS=1 -DUCCL_LITE_MP_AG_SINGLE_PRODUCERS=32 -DUCCL_LITE_MP_WAIT_LANES=8 -DUCCL_LITE_MP_RS_BLOCKS=64 -DUCCL_LITE_MP_RS_SINGLE_TILE_PRODUCERS=56
uccl_lite_host_packet_control_flags = -DUCCL_LITE_MP_PACKET_BYTES=$(1) -DUCCL_LITE_MP_AG_PIPELINE_MIN_BYTES=4194304 -DUCCL_LITE_MP_PARALLEL_RS_SINGLE_TILE -DUCCL_LITE_MP_POLL_WORKERS -DUCCL_LITE_MP_GLOBAL_WORKER_ISOLATION
HOST_PACKET_DEVICE_FLAGS = $(call uccl_lite_host_packet_device_flags,$(HOST_PACKET_BYTES))
HOST_PACKET_CONTROL_FLAGS = $(call uccl_lite_host_packet_control_flags,$(HOST_PACKET_BYTES))
PEER_PACKET_REPRO_FLAGS = --frandom-seed=uccl_lite_peer_packet --keep --keep-dir $(BUILD)/collectives/peer_packet-intermediates

.PHONY: collective-build-flags
collective-build-flags:

$(BUILD)/collectives/peer_packet.flags: collective-build-flags
	mkdir -p $(@D)
	@printf '%s\n' '$(NVCC) $(NVCC_FLAGS) $(PEER_PACKET_RETAINED_FLAGS) $(PEER_PACKET_REPRO_FLAGS)' '$(IB_INC)' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

$(BUILD)/collectives/host_packet.flags: collective-build-flags
	mkdir -p $(@D)
	@printf '%s\n' '$(NVCC) $(NVCC_FLAGS) $(HOST_PACKET_DEVICE_FLAGS)' '$(CXX) $(HOST_PACKET_CONTROL_FLAGS)' '$(CUDA_HOME)' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

$(BUILD)/collectives/peer_packet.o: src/collectives/peer_packet.cu $(BUILD)/collectives/peer_packet.flags src/collectives/module.mk src/collectives/peer_packet_flags.mk
	mkdir -p $(@D)/peer_packet-intermediates
	$(NVCC) $(NVCC_FLAGS) $(PEER_PACKET_RETAINED_FLAGS) $(PEER_PACKET_REPRO_FLAGS) -std=c++17 -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden $(IB_INC) -c $< -o $@

$(BUILD)/collectives/host_packet_device.o: src/collectives/multiprocess_packet_ag.cu $(BUILD)/collectives/host_packet.flags src/collectives/module.mk
	$(NVCC) $(NVCC_FLAGS) $(HOST_PACKET_DEVICE_FLAGS) -std=c++17 -Xcompiler=-fPIC -Iinclude -c $< -o $@

$(BUILD)/collectives/host_packet_control.o: src/collectives/multiprocess_collectives.cc $(BUILD)/collectives/host_packet.flags src/collectives/module.mk
	$(CXX) -O3 -std=c++17 -fPIC -pthread -MMD -MP $(HOST_PACKET_CONTROL_FLAGS) -Iinclude -I$(CUDA_HOME)/include -c $< -o $@

-include $(wildcard $(BUILD)/collectives/*.d)

.PHONY: direct-exchange direct-exchange-check
direct-exchange: $(BUILD)/collectives/libuccl_lite_direct_exchange.a
direct-exchange-check: $(BUILD)/uccl_lite_direct_exchange_check
$(BUILD)/collectives/direct_exchange.o: src/collectives/direct_exchange.cu include/uccl_lite/collectives/direct_exchange.hpp
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Xcompiler=-fPIC $(IB_INC) -c $< -o $@
$(BUILD)/collectives/libuccl_lite_direct_exchange.a: $(BUILD)/collectives/direct_exchange.o $(BUILD)/runtime/direct_exchange_backend.o $(BUILD)/runtime/backend_registry_direct.o
	rm -f $@.tmp
	$(AR) rcs $@.tmp $^
	mv $@.tmp $@
$(BUILD)/uccl_lite_direct_exchange_check: tests/collectives/direct_exchange_check.cu $(BUILD)/collectives/libuccl_lite_direct_exchange.a $(PLANNER_STATIC) $(TOPOLOGY_STATIC)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(IB_INC) $(MPI_INC) $< $(BUILD)/collectives/libuccl_lite_direct_exchange.a $(PLANNER_STATIC) $(TOPOLOGY_STATIC) $(IB_LIB) $(MPI_LIB) -ldl -Xcompiler=-pthread -o $@

$(BUILD)/runtime/direct_exchange_backend.o: src/runtime/direct_exchange_backend.cc
	mkdir -p $(@D)
	$(CXX) -O2 -std=c++17 -fPIC -MMD -MP -Iinclude -I$(CUDA_HOME)/include -c $< -o $@

$(BUILD)/runtime/backend_registry_direct.o: src/runtime/backend_registry.cc
	mkdir -p $(@D)
	$(CXX) -O2 -std=c++17 -fPIC -MMD -MP -DUCCL_LITE_REGISTER_DIRECT_EXCHANGE -Iinclude -I$(CUDA_HOME)/include -c $< -o $@
