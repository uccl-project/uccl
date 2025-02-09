# SPDX-License-Identifier: (GPL-2.0 OR BSD-2-Clause)

# !!! Remember to conda deactivate before running make !!!

XDP_TARGETS  := af_xdp_kern af_xdp_kern_efa
USER_TARGETS := af_xdp_user af_xdp_user_efa
FI_FLAGS = -I /opt/amazon/efa/include -L /opt/amazon/efa/lib -lfabric -libverbs
LDLIBS += -lpthread $(FI_FLAGS)

all: fi efa cuda

FI_BIN = fi_pingpong fi_info fi_udpgen fi_efagen fi_efasink
fi: $(FI_BIN)

fi_pingpong: fi_pingpong.c
	gcc -O3 fi_pingpong.c -o fi_pingpong $(FI_FLAGS)

fi_info: fi_info.c
	gcc -O3 fi_info.c -o fi_info $(FI_FLAGS)

fi_udpgen: fi_udpgen.c
	gcc -O3 fi_udpgen.c -o fi_udpgen $(FI_FLAGS)

fi_efagen: fi_efagen.c
	gcc -O3 -g fi_efagen.c -o fi_efagen $(FI_FLAGS)

fi_efasink: fi_efasink.c
	gcc -O3 -g fi_efasink.c -o fi_efasink $(FI_FLAGS)

EFA_BIN = efa_shared_cp efa_sge_mr
efa: $(EFA_BIN)

efa_shared_cp: efa_shared_cp.cc
	g++ -O3 -g efa_shared_cp.cc -o efa_shared_cp $(FI_FLAGS)

efa_sge_mr: efa_sge_mr.cu
	nvcc -O3 -g efa_sge_mr.cu -o efa_sge_mr $(FI_FLAGS)

CUDA_BIN = cuda_event cuda_memcpy cuda_memcpy_async cuda_persist_kernel
cuda: $(CUDA_BIN)

cuda_event: cuda_event.cu
	nvcc -O3 -g cuda_event.cu -o cuda_event $(FI_FLAGS) -std=c++11

cuda_memcpy: cuda_memcpy.cu
	nvcc -O3 -g cuda_memcpy.cu -o cuda_memcpy $(FI_FLAGS) -std=c++11

cuda_memcpy_async: cuda_memcpy_async.cu
	nvcc -O3 -g cuda_memcpy_async.cu -o cuda_memcpy_async $(FI_FLAGS) -std=c++11

cuda_persist_kernel: cuda_persist_kernel.cu
	nvcc -O3 -g cuda_persist_kernel.cu -o cuda_persist_kernel $(FI_FLAGS) -std=c++11

clean:
	rm $(FI_BIN) $(EFA_BIN) $(CUDA_BIN)