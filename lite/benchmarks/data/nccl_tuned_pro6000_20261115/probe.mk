include Makefile
.PHONY: nccl-tuning-reference
nccl-tuning-reference: build/nccl_tuned_20261115/nccl_reference
build/nccl_tuned_20261115/nccl_reference: build/nccl_tuned_20261115/nccl_reference.cu
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DNCCL_REFERENCE_ONLY -Ibenchmarks/cpu_router -I$(NCCL_HOME)/include $< -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -o $@
