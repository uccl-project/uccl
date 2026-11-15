include Makefile
.PHONY: nccl-four-tuning
nccl-four-tuning: build/nccl_tuned_4l4_20261115/peer_nccl build/uccl_lite_nccl_4l4_mpi_baseline
build/nccl_tuned_4l4_20261115/peer_nccl: build/nccl_tuned_4l4_20261115/peer_nccl.cu
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MSCCLPP_INC) -I$(NCCL_HOME)/include $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib
