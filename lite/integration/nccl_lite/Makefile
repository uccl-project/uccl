CXX ?= g++
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
NCCL_HOME ?= /usr
REAL_NCCL ?= libnccl.so.2
TORCHRUN ?= torchrun
BUILD := build
LITE := ../..

.PHONY: all check check-torch check-shared-packet check-sum2 check-ag-gemm check-ready-rs bench-sum2 clean
all: $(BUILD)/libuccl_lite_nccl_compat.so $(BUILD)/libuccl_lite_nccl.so

.PHONY: dropin
dropin: $(BUILD)/libuccl_lite_nccl.so

$(BUILD)/libuccl_lite_nccl.so: dropin.cc $(LITE)/src/collectives/multiprocess_collectives.cc $(LITE)/include/uccl_lite/multiprocess_collectives.h $(LITE)/src/cpu_router/cpu_executor.cc $(BUILD)/multiprocess_packet_ag.o | $(BUILD)
	$(CXX) -O3 -std=c++17 -fPIC -shared -pthread -I$(LITE)/include -I$(LITE)/include/uccl_lite/cpu_router -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include dropin.cc $(LITE)/src/collectives/multiprocess_collectives.cc $(LITE)/src/cpu_router/cpu_executor.cc $(BUILD)/multiprocess_packet_ag.o -L$(CUDA_HOME)/lib64 -lcudart -lcublas -ldl -L$(NCCL_HOME)/lib -Wl,--no-as-needed -l:libnccl.so.2 -Wl,--as-needed -Wl,-rpath,$(CUDA_HOME)/lib64 -Wl,-rpath,$(abspath $(NCCL_HOME)/lib) -o $@

check: $(BUILD)/test_compat
	UCCL_LITE_REAL_NCCL=$(REAL_NCCL) UCCL_LITE_NCCL_ENABLE=1 NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 ./$(BUILD)/test_compat

check-torch: $(BUILD)/libuccl_lite_nccl_compat.so
	LD_PRELOAD=$(abspath $(BUILD)/libuccl_lite_nccl_compat.so) UCCL_LITE_REAL_NCCL=$(REAL_NCCL) UCCL_LITE_NCCL_ENABLE=1 NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 $(TORCHRUN) --standalone --nproc-per-node=8 test_torch.py

check-shared-packet: $(BUILD)/test_shared_packet
	./$(BUILD)/test_shared_packet

check-sum2: $(BUILD)/test_rs_sum2
	UCCL_LITE_REAL_NCCL=$(REAL_NCCL) ./run_packet_8l4.sh ./$(BUILD)/test_rs_sum2

check-ag-gemm: $(BUILD)/test_ag_gemm
	UCCL_LITE_REAL_NCCL=$(REAL_NCCL) ./run_packet_8l4.sh ./$(BUILD)/test_ag_gemm

check-ready-rs: $(BUILD)/test_rs_ready
	UCCL_LITE_REAL_NCCL=$(REAL_NCCL) ./run_packet_8l4.sh timeout 30 ./$(BUILD)/test_rs_ready

bench-sum2: $(BUILD)/bench_rs_sum2
	UCCL_LITE_REAL_NCCL=$(REAL_NCCL) ./run_packet_8l4.sh ./$(BUILD)/bench_rs_sum2

$(BUILD):
	mkdir -p $@

$(BUILD)/test_shared_packet: test_shared_packet.cu $(LITE)/include/uccl_lite/cpu_router/channel_device.cuh | $(BUILD)
	$(NVCC) -O2 -std=c++17 -gencode=arch=compute_89,code=sm_89 -I$(LITE)/include $< -o $@

$(BUILD)/multiprocess_packet_ag.o: $(LITE)/src/collectives/multiprocess_packet_ag.cu $(LITE)/include/uccl_lite/multiprocess_packet_ag.h | $(BUILD)
	$(NVCC) -O3 -std=c++17 -Xcompiler=-fPIC -gencode=arch=compute_89,code=sm_89 -I$(LITE)/include -c $< -o $@

$(BUILD)/libuccl_lite_nccl_compat.so: nccl_compat.cc $(LITE)/src/collectives/multiprocess_collectives.cc $(LITE)/include/uccl_lite/multiprocess_collectives.h $(LITE)/src/cpu_router/cpu_executor.cc $(LITE)/include/uccl_lite/cpu_router/cpu_executor.hpp $(BUILD)/multiprocess_packet_ag.o | $(BUILD)
	$(CXX) -O3 -std=c++17 -fPIC -shared -pthread -I$(LITE)/include -I$(LITE)/include/uccl_lite/cpu_router -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include nccl_compat.cc $(LITE)/src/collectives/multiprocess_collectives.cc $(LITE)/src/cpu_router/cpu_executor.cc $(BUILD)/multiprocess_packet_ag.o -L$(CUDA_HOME)/lib64 -lcudart -lcublas -ldl -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@

$(BUILD)/test_compat: test_compat.cc $(BUILD)/libuccl_lite_nccl_compat.so
	$(CXX) -O2 -std=c++17 -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include $< -L$(BUILD) -luccl_lite_nccl_compat -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(abspath $(BUILD)) -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@

$(BUILD)/test_rs_sum2: test_rs_sum2.cc nccl_fusion.h $(BUILD)/libuccl_lite_nccl_compat.so
	$(CXX) -O2 -std=c++17 -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include $< -L$(BUILD) -luccl_lite_nccl_compat -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(abspath $(BUILD)) -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@

$(BUILD)/test_ag_gemm: test_ag_gemm.cc nccl_fusion.h $(BUILD)/libuccl_lite_nccl_compat.so
	$(CXX) -O2 -std=c++17 -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include $< -L$(BUILD) -luccl_lite_nccl_compat -L$(CUDA_HOME)/lib64 -lcudart -lcublas -Wl,-rpath,$(abspath $(BUILD)) -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@

$(BUILD)/bench_rs_sum2: bench_rs_sum2.cu nccl_fusion.h $(BUILD)/libuccl_lite_nccl_compat.so
	$(NVCC) -O2 -std=c++17 -gencode=arch=compute_89,code=sm_89 -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include $< -L$(BUILD) -luccl_lite_nccl_compat -L$(CUDA_HOME)/lib64 -lcudart -Xlinker=-rpath,$(abspath $(BUILD)) -o $@

$(BUILD)/test_rs_ready: test_rs_ready.cu $(LITE)/include/uccl_lite/multiprocess_collectives.h $(BUILD)/libuccl_lite_nccl_compat.so
	$(NVCC) -O2 -std=c++17 -gencode=arch=compute_89,code=sm_89 -I$(LITE)/include -I$(CUDA_HOME)/include $< -L$(BUILD) -luccl_lite_nccl_compat -L$(CUDA_HOME)/lib64 -lcudart -Xlinker=-rpath,$(abspath $(BUILD)) -o $@

clean:
	rm -rf $(BUILD)
