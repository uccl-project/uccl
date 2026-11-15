# Use with: make -f Makefile -f benchmarks/data/peer_common_refactor_20261115/paired.mk peer-common-paired
# Export the recorded baseline source/header into build/ first (see README).
.PHONY: peer-common-paired
peer-common-paired: $(BUILD)/uccl_lite_peer_common_paired
$(BUILD)/peer_common_baseline.o: $(BUILD)/peer_common_baseline.cu $(BUILD)/peer_common_baseline.hpp
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -Iinclude/uccl_lite -I$(BUILD) -DJointCollectives=BaselineJointCollectives -DJointMode=BaselineJointMode -c $< -o $@
$(BUILD)/uccl_lite_peer_common_paired: benchmarks/data/peer_common_refactor_20261115/paired_in_process.cu $(BUILD)/peer_common_baseline.o $(JOINT_OBJECTS) $(ROUTER_STATIC)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -Iinclude/uccl_lite -I$(BUILD) -I$(NCCL_HOME)/include $< $(BUILD)/peer_common_baseline.o $(JOINT_OBJECTS) $(ROUTER_STATIC) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl -o $@
