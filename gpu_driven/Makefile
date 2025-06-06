CUDA_HOME:=/usr/local/cuda
EFA_HOME:=/opt/amazon/efa
CXXFLAGS = -I${EFA_HOME}/include -L${EFA_HOME}/lib -I$(CUDA_HOME)/include -L ${CUDA_HOME}/lib64 -libverbs -lefa -lcudart -lcuda -lpthread -std=c++17
CUDAFLAGS = -arch=sm_80 # for A100
# CUDAFLAGS = -arch=sm_90 # for H100

NVCC=${CUDA_HOME}/bin/nvcc
CUDA_BIN = pcie_bench gpu_to_cpu_bench batched_gpu_to_cpu_bench
all: $(CUDA_BIN)

pcie_bench: pcie_bench.cu
	${NVCC} -O3 -g pcie_bench.cu -o pcie_bench $(CXXFLAGS) $(CUDAFLAGS)

gpu_to_cpu_bench: gpu_to_cpu_bench.cu
	${NVCC} -O3 -g gpu_to_cpu_bench.cu -o gpu_to_cpu_bench $(CXXFLAGS) $(CUDAFLAGS)

batched_gpu_to_cpu_bench: batched_gpu_to_cpu_bench.cu
	${NVCC} -O3 -g batched_gpu_to_cpu_bench.cu -o batched_gpu_to_cpu_bench $(CXXFLAGS) $(CUDAFLAGS)

clean:
	rm $(CUDA_BIN)
