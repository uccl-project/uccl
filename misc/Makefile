# SPDX-License-Identifier: (GPL-2.0 OR BSD-2-Clause)

# !!! Remember to conda deactivate before running make !!!

XDP_TARGETS  := af_xdp_kern af_xdp_kern_efa
USER_TARGETS := af_xdp_user af_xdp_user_efa
CUDA_HOME:=/usr/local/cuda
CXXFLAGS = -I../include -I /opt/amazon/efa/include -L /opt/amazon/efa/lib -lfabric -libverbs -lefa -lpthread
CUFLAGS = -I $(CUDA_HOME)/include -L ${CUDA_HOME}/lib64 -lcudart -lpthread -lcuda
ARCH_FLAGS = -arch=sm_90

all: fi efa cuda

FI_BIN = fi_udpgen fi_efagen fi_efasink
fi: $(FI_BIN)

fi_udpgen: fi_udpgen.c
	gcc -O3 fi_udpgen.c -o fi_udpgen $(CXXFLAGS)

fi_efagen: fi_efagen.c
	gcc -O3 -g fi_efagen.c -o fi_efagen $(CXXFLAGS)

fi_efasink: fi_efasink.c
	gcc -O3 -g fi_efasink.c -o fi_efasink $(CXXFLAGS)

EFA_BIN = efa_shared_cp_libfabric efa_shared_cp_ibverbs efa_sge_mr efa_load_latency
efa: $(EFA_BIN)

efa_shared_cp_libfabric: efa_shared_cp_libfabric.cc
	g++ -O3 -g efa_shared_cp_libfabric.cc -o efa_shared_cp_libfabric $(CXXFLAGS)

efa_shared_cp_ibverbs: efa_shared_cp_ibverbs.cc
	g++ -O3 -g efa_shared_cp_ibverbs.cc -o efa_shared_cp_ibverbs $(CXXFLAGS) -fstack-protector-all

efa_sge_mr: efa_sge_mr.cc
	g++ -O3 -g efa_sge_mr.cc -o efa_sge_mr $(CXXFLAGS) $(CUFLAGS)

efa_load_latency: efa_load_latency.cc
	g++ -O3 -g efa_load_latency.cc -o efa_load_latency $(CXXFLAGS) $(CUFLAGS)

CUDA_BIN = cuda_event cuda_memcpy cuda_memcpy_async cuda_persist_kernel cuda_memcpy_gpu cuda_concurrent cuda_greenctx
cuda: $(CUDA_BIN)

cuda_event: cuda_event.cu
	nvcc -O3 -g cuda_event.cu -o cuda_event $(CUFLAGS) $(ARCH_FLAGS) -std=c++17

cuda_memcpy: cuda_memcpy.cu
	nvcc -O3 -g cuda_memcpy.cu -o cuda_memcpy $(CUFLAGS) $(ARCH_FLAGS) -std=c++17

cuda_memcpy_async: cuda_memcpy_async.cu
	nvcc -O3 -g cuda_memcpy_async.cu -o cuda_memcpy_async $(CUFLAGS) $(ARCH_FLAGS) -std=c++17

cuda_persist_kernel: cuda_persist_kernel.cu
	nvcc -O3 -g cuda_persist_kernel.cu -o cuda_persist_kernel $(CUFLAGS) $(ARCH_FLAGS) -std=c++17

cuda_memcpy_gpu: cuda_memcpy_gpu.cu
	nvcc -O3 -g cuda_memcpy_gpu.cu -o cuda_memcpy_gpu $(CUFLAGS) $(ARCH_FLAGS) -std=c++17

cuda_concurrent: cuda_concurrent.cu
	nvcc -O3 -g cuda_concurrent.cu -o cuda_concurrent $(CUFLAGS) $(ARCH_FLAGS) -std=c++17

cuda_greenctx: cuda_greenctx.cu
	nvcc -O3 -g cuda_greenctx.cu -o cuda_greenctx $(CUFLAGS) $(ARCH_FLAGS) -std=c++17

clean:
	rm $(FI_BIN) $(EFA_BIN) $(CUDA_BIN)