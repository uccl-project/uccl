PLANNER_OBJECTS := $(BUILD)/planner/planner.o $(BUILD)/planner/peer_plan.o $(BUILD)/planner/hardware_profile.o $(BUILD)/planner/network_plan.o $(BUILD)/planner/peer_profile.o
PLANNER_STATIC := $(BUILD)/planner/libuccl_lite_planner.a
PLANNER_OBJECTS += $(BUILD)/planner/local_plan.o
PLANNER_OBJECTS += $(BUILD)/planner/network_profile.o
PLANNER_OBJECTS += $(BUILD)/planner/measured_network_plan.o
PLANNER_OBJECTS += $(BUILD)/planner/network_cache.o
PLANNER_OBJECTS += $(BUILD)/planner/overlap_model.o
PLANNER_OBJECTS += $(BUILD)/planner/direct_plan.o

.PHONY: planner planner-check
planner: $(PLANNER_STATIC)
planner-check: $(BUILD)/uccl_lite_planner_check $(BUILD)/uccl_lite_peer_plan_check $(BUILD)/uccl_lite_hardware_profile_check

$(BUILD)/planner/%.o: src/planner/%.cc
	mkdir -p $(@D)
	$(CXX) -O2 -std=c++17 -fPIC -MMD -MP -Iinclude -c $< -o $@

$(PLANNER_STATIC): $(PLANNER_OBJECTS) src/planner/module.mk
	rm -f $@.tmp
	$(AR) rcs $@.tmp $(PLANNER_OBJECTS)
	mv $@.tmp $@

$(BUILD)/uccl_lite_planner_check: tests/planner/planner_check.cc $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $^ -o $@

-include $(wildcard $(BUILD)/planner/*.d)

$(BUILD)/uccl_lite_peer_plan_check: tests/planner/peer_plan_check.cc tests/planner/peer_fixture.hpp $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(PLANNER_STATIC) -o $@

$(BUILD)/uccl_lite_hardware_profile_check: tests/planner/hardware_profile_check.cc $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(PLANNER_STATIC) -o $@

planner-check: $(BUILD)/uccl_lite_network_plan_check
$(BUILD)/uccl_lite_network_plan_check: tests/planner/network_plan_check.cc $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(PLANNER_STATIC) -o $@

planner-check: $(BUILD)/uccl_lite_peer_profile_check
planner-check: $(BUILD)/uccl_lite_local_plan_check
planner-check: $(BUILD)/uccl_lite_network_profile_check
planner-check: $(BUILD)/uccl_lite_measured_network_plan_check
planner-check: $(BUILD)/uccl_lite_network_group_check
planner-check: $(BUILD)/uccl_lite_overlap_model_check
$(BUILD)/uccl_lite_overlap_model_check: tests/planner/overlap_model_check.cc $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(PLANNER_STATIC) -o $@
planner-check: $(BUILD)/uccl_lite_network_cache_check
$(BUILD)/uccl_lite_network_cache_check: tests/planner/network_cache_check.cc $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(PLANNER_STATIC) -o $@
planner-check: $(BUILD)/uccl_lite_peer_tree_plan_check
$(BUILD)/uccl_lite_peer_tree_plan_check: tests/planner/peer_tree_plan_check.cc $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(PLANNER_STATIC) -o $@
$(BUILD)/uccl_lite_network_group_check: tests/planner/network_group_check.cc $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(PLANNER_STATIC) -o $@
$(BUILD)/uccl_lite_measured_network_plan_check: tests/planner/measured_network_plan_check.cc $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(PLANNER_STATIC) -o $@
$(BUILD)/uccl_lite_network_profile_check: tests/planner/network_profile_check.cc $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(PLANNER_STATIC) -o $@
$(BUILD)/uccl_lite_local_plan_check: tests/planner/local_plan_check.cc tests/planner/peer_fixture.hpp $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(PLANNER_STATIC) -o $@
$(BUILD)/uccl_lite_peer_profile_check: tests/planner/peer_profile_check.cc $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(PLANNER_STATIC) -o $@

planner-check: $(BUILD)/uccl_lite_direct_plan_check
$(BUILD)/uccl_lite_direct_plan_check: tests/planner/direct_plan_check.cc $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(PLANNER_STATIC) -o $@
