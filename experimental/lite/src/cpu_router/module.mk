# Optional module: the existing single-node/fusion targets do not link this library.
ROUTER_BUILD := $(BUILD)/cpu_router
ROUTER_INC := -Iinclude -Iinclude/uccl_lite/cpu_router -I$(CUDA_HOME)/include
ROUTER_HOST := cpu_executor cpu_router_c memory router_core signal backend_policy collective_dispatch
ROUTER_CUDA := channel packet_pipeline
ROUTER_OBJECTS := $(addprefix $(ROUTER_BUILD)/,$(addsuffix .o,$(ROUTER_HOST) $(ROUTER_CUDA)))
ROUTER_STATIC := $(ROUTER_BUILD)/libuccl_lite_cpu_router.a
ROUTER_SHARED := $(ROUTER_BUILD)/libuccl_lite_cpu_router.so
ROUTER_LIB := $(ROUTER_STATIC) -L$(CUDA_HOME)/lib64 -lcudart -lpthread -ldl -Wl,-rpath,$(CUDA_HOME)/lib64
LOCAL_STATIC := $(BUILD)/libuccl_lite_local_collectives.a
ROUTER_CUDA_TESTS := primitives_check primitive_example router_check router_graph_check gpu_reduction_check dispatcher_cuda_check host_ring_check
ROUTER_HOST_TESTS := cpu_bf16_check router_core_check dispatcher_check

.PHONY: cpu-router local-collectives adaptive-collectives local-collectives-check router-checks router-policy-check nccl-identity-check ddio-reduce-probe
ddio-reduce-probe: $(BUILD)/uccl_lite_ddio_reduce_probe

$(BUILD)/uccl_lite_ddio_reduce_probe: benchmarks/cpu_router/ddio_reduce_probe.cu $(ROUTER_STATIC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) $< $(ROUTER_STATIC) -Xcompiler=-pthread -ldl -o $@

cpu-router: $(ROUTER_STATIC) $(ROUTER_SHARED)
local-collectives: $(LOCAL_STATIC)
local-collectives-check: $(BUILD)/uccl_lite_local_collectives_check
adaptive-collectives: $(BUILD)/uccl_lite_collectives
router-policy-check: $(BUILD)/uccl_lite_router_policy_check
router-checks: cpu-router router-policy-check nccl-identity-check $(addprefix $(BUILD)/uccl_lite_,$(ROUTER_CUDA_TESTS) $(ROUTER_HOST_TESTS))
nccl-identity-check: $(BUILD)/uccl_lite_nccl_identity_check

$(BUILD)/uccl_lite_nccl_identity_check: tests/cpu_router/nccl_identity_check.cc benchmarks/cpu_router/nccl_identity.hpp | $(BUILD)
	$(CXX) -O2 -std=c++17 $< -o $@

$(ROUTER_BUILD):
	mkdir -p $@

$(addprefix $(ROUTER_BUILD)/,$(addsuffix .o,$(ROUTER_HOST))): $(ROUTER_BUILD)/%.o: src/cpu_router/%.cc | $(ROUTER_BUILD)
	$(CXX) -O3 -std=c++17 -fPIC -MMD -MP $(ROUTER_INC) -c $< -o $@

$(addprefix $(ROUTER_BUILD)/,$(addsuffix .o,$(ROUTER_CUDA))): $(ROUTER_BUILD)/%.o: src/cpu_router/%.cu | $(ROUTER_BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Xcompiler=-fPIC $(FUSION_FLAGS) $(ROUTER_INC) -I$(CUTLASS_HOME)/include -c $< -o $@

$(ROUTER_STATIC): $(ROUTER_OBJECTS)
	$(AR) rcs $@ $^

$(ROUTER_SHARED): $(ROUTER_OBJECTS)
	$(CXX) -shared $^ -L$(CUDA_HOME)/lib64 -lcudart -lpthread -ldl -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@

$(BUILD)/local_collectives.o: src/collectives/local_collectives.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude -c $< -o $@

$(LOCAL_STATIC): $(BUILD)/local_collectives.o
	$(AR) rcs $@ $^

$(BUILD)/uccl_lite_local_collectives_check: tests/collectives/local_collectives_check.cu $(LOCAL_STATIC) $(ROUTER_STATIC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude $< $(LOCAL_STATIC) $(ROUTER_STATIC) -Xcompiler=-pthread -ldl -o $@

$(addprefix $(BUILD)/uccl_lite_,$(ROUTER_HOST_TESTS)): $(BUILD)/uccl_lite_%: tests/cpu_router/%.cc $(ROUTER_STATIC) | $(BUILD)
	$(CXX) -O3 -std=c++17 -MMD -MP $(ROUTER_INC) $< $(ROUTER_LIB) -o $@

$(addprefix $(BUILD)/uccl_lite_,$(ROUTER_CUDA_TESTS)): $(BUILD)/uccl_lite_%: tests/cpu_router/%.cu $(ROUTER_STATIC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) $< $(ROUTER_STATIC) -Xcompiler=-pthread -ldl -o $@

$(BUILD)/uccl_lite_router_policy_check: tests/cpu_router/backend_policy_check.cc src/cpu_router/backend_policy.cc include/uccl_lite/backend_policy.hpp | $(BUILD)
	$(CXX) -O2 -std=c++17 -Iinclude tests/cpu_router/backend_policy_check.cc src/cpu_router/backend_policy.cc -o $@

$(BUILD)/uccl_lite_collectives: benchmarks/cpu_router/collectives_bench.cu $(LOCAL_STATIC) $(ROUTER_STATIC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(LOCAL_STATIC) $(ROUTER_STATIC) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl -o $@

-include $(wildcard $(ROUTER_BUILD)/*.d)
