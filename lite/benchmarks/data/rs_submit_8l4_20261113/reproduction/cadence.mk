include Makefile
$(BUILD)/collectives_cadence_probe: $(BUILD)/collectives_cadence_probe.cu $(LOCAL_STATIC) $(ROUTER_STATIC)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -Ibenchmarks/cpu_router -I$(NCCL_HOME)/include $< $(LOCAL_STATIC) $(ROUTER_STATIC) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl -o $@
