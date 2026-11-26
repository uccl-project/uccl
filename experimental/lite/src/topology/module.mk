TOPOLOGY_OBJECTS := $(BUILD)/topology/discovery.o $(BUILD)/topology/hardware.o
TOPOLOGY_STATIC := $(BUILD)/topology/libuccl_lite_topology.a

.PHONY: topology topology-check
topology: $(TOPOLOGY_STATIC) $(BUILD)/uccl_lite_topology
topology-check: $(BUILD)/uccl_lite_topology_check

$(BUILD)/topology/%.o: src/topology/%.cc
	mkdir -p $(@D)
	$(CXX) -O2 -std=c++17 -fPIC -MMD -MP -Iinclude -I$(CUDA_HOME)/include -c $< -o $@

$(TOPOLOGY_STATIC): $(TOPOLOGY_OBJECTS) src/topology/module.mk
	rm -f $@.tmp
	$(AR) rcs $@.tmp $(TOPOLOGY_OBJECTS)
	mv $@.tmp $@

$(BUILD)/uccl_lite_topology_check: tests/topology/discovery_check.cc $(BUILD)/topology/hardware.o
	$(CXX) -O2 -std=c++17 -Iinclude $^ -o $@

$(BUILD)/uccl_lite_topology: benchmarks/probes/topology.cc $(TOPOLOGY_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $^ -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@

-include $(wildcard $(BUILD)/topology/*.d)

.PHONY: calibrate-hardware
calibrate-hardware: $(BUILD)/uccl_lite_calibrate_hardware
$(BUILD)/topology/calibration.o: src/topology/calibration.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Xcompiler=-fPIC -Iinclude -c $< -o $@
$(BUILD)/uccl_lite_calibrate_hardware: benchmarks/probes/calibrate_hardware.cc $(BUILD)/topology/calibration.o $(TOPOLOGY_STATIC) $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(BUILD)/topology/calibration.o $(TOPOLOGY_STATIC) $(PLANNER_STATIC) -L$(CUDA_HOME)/lib64 -lcudart -ldl -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@

.PHONY: calibration-check
calibration-check: $(BUILD)/uccl_lite_calibration_check
$(BUILD)/uccl_lite_calibration_check: tests/topology/calibration_check.cu $(BUILD)/topology/calibration.o $(TOPOLOGY_STATIC) $(PLANNER_STATIC)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude $< $(BUILD)/topology/calibration.o $(TOPOLOGY_STATIC) $(PLANNER_STATIC) -ldl -o $@

.PHONY: calibrate-host
calibrate-host: $(BUILD)/uccl_lite_calibrate_host
$(BUILD)/uccl_lite_calibrate_host: benchmarks/probes/calibrate_host.cc $(BUILD)/topology/host_calibration.o $(TOPOLOGY_STATIC) $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(BUILD)/topology/host_calibration.o $(TOPOLOGY_STATIC) $(PLANNER_STATIC) -L$(CUDA_HOME)/lib64 -lcudart -pthread -ldl -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@

.PHONY: host-calibration-check
host-calibration-check: $(BUILD)/uccl_lite_host_calibration_check
$(BUILD)/uccl_lite_host_calibration_check: tests/topology/host_calibration_check.cc $(BUILD)/topology/host_calibration.o $(TOPOLOGY_STATIC) $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(BUILD)/topology/host_calibration.o $(TOPOLOGY_STATIC) $(PLANNER_STATIC) -L$(CUDA_HOME)/lib64 -lcudart -pthread -ldl -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@
