# Network algorithm objects for the common runtime. Historical integration
# policies compile their own objects and do not configure this implementation.
include src/multinode/network_hierarchy_flags.mk

.PHONY: network-build-flags network-runtime-check
network-build-flags:
network-runtime-check: $(BUILD)/uccl_lite_network_prepare_check

$(BUILD)/multinode/network_hierarchy.flags: network-build-flags
	mkdir -p $(@D)
	@printf '%s\n' '$(NVCC) $(NVCC_FLAGS) $(NETWORK_HIERARCHY_RETAINED_FLAGS)' '$(IB_INC)' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

$(BUILD)/multinode/network_hierarchy.o: src/multinode/mscclpp_backend.cu $(BUILD)/multinode/network_hierarchy.flags src/multinode/module.mk src/multinode/network_hierarchy_flags.mk
	$(NVCC) $(NVCC_FLAGS) $(NETWORK_HIERARCHY_RETAINED_FLAGS) -std=c++20 -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden $(IB_INC) -c $< -o $@

-include $(wildcard $(BUILD)/multinode/*.d)
