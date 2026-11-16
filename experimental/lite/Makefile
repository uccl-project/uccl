# UCCL Lite standalone benchmarks. External dependencies are never vendored.
.DEFAULT_GOAL := single-node

CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= sm_89
MSCCLPP_HOME ?= $(HOME)/third_party/mscclpp
MSCCLPP_BUILD ?= $(MSCCLPP_HOME)/build
MSCCLPP_IB_BUILD ?= $(MSCCLPP_HOME)/build-ib
NCCL_HOME ?= $(HOME)/third_party/nccl/build
CUTLASS_HOME ?= $(HOME)/third_party/cutlass
MPICXX ?= mpicxx
BUILD := build
EXTRA_NVCC_FLAGS ?=

NVCC_FLAGS := -O3 -arch=$(CUDA_ARCH) -MMD -MP $(EXTRA_NVCC_FLAGS)
MSCCLPP_INC = -Iinclude -I$(MSCCLPP_HOME)/include -I$(MSCCLPP_BUILD)/include
MSCCLPP_LIB = -L$(MSCCLPP_BUILD)/lib -lmscclpp -Xlinker -rpath=$(MSCCLPP_BUILD)/lib
IB_INC = -Iinclude -I$(MSCCLPP_HOME)/include -I$(MSCCLPP_IB_BUILD)/include
IB_LIB = -L$(MSCCLPP_IB_BUILD)/lib -lmscclpp -Xlinker -rpath=$(MSCCLPP_IB_BUILD)/lib
MPI_INC = $(shell $(MPICXX) --showme:compile)
# Open MPI emits host-compiler -Wl options; nvcc needs -Xlinker instead.
MPI_LIB = $(shell $(MPICXX) --showme:link | sed 's/-Wl,/-Xlinker /g')
FUSION_INC = $(MSCCLPP_INC) -Iinclude/uccl_lite -I$(CUTLASS_HOME)/include -I$(NCCL_HOME)/include
FUSION_FLAGS := --expt-relaxed-constexpr -diag-suppress 20012,20014
FUSION_LIB = $(MSCCLPP_LIB) -L$(NCCL_HOME)/lib -lnccl -lcublas -Xlinker -rpath=$(NCCL_HOME)/lib

COLLECTIVES := allreduce alltoall
PEER_COLLECTIVES := peer_ag_rs
NCCL_CONTROLS := nccl_4l4_mpi_baseline nccl_2x4l4_baseline
MULTINODE := multinode_allreduce multinode_cpu_router host_hierarchical_allreduce
HOST_MULTINODE := multinode_host_allgather multinode_host_rdma_allgather
FUSION := dense_fusion fused_moe fused_moe_layer fused_moe_layer_qwen router_fusion_8l4
PROBES := gemm_probe mma_gemm_probe mma_peak_probe moe_overlap_probe cutlass_gemm_probe hardware_profiler
ALL_NAMES := $(COLLECTIVES) $(PEER_COLLECTIVES) $(NCCL_CONTROLS) $(MULTINODE) $(HOST_MULTINODE) host_allreduce $(FUSION) $(PROBES)
BIN = $(addprefix $(BUILD)/uccl_lite_,$(1))

.PHONY: all single-node peer-collectives peer-ag-rs-packet eight-gpu-alltoall duplex-allgather-8l4 host-memory-bw-triad fusion probes multinode headers help $(ALL_NAMES)
all: single-node peer-collectives fusion probes headers
single-node: $(call BIN,$(COLLECTIVES))
peer-collectives: $(call BIN,$(PEER_COLLECTIVES))
peer-ag-rs-packet: $(BUILD)/uccl_lite_peer_ag_rs_packet
eight-gpu-alltoall: $(BUILD)/uccl_lite_alltoall_8l4
duplex-allgather-8l4: $(BUILD)/uccl_lite_duplex_allgather_8l4
host-memory-bw-triad: $(BUILD)/host_memory_bw_4l4_20261111/stream_triad
fusion: $(call BIN,$(FUSION))
probes: $(call BIN,$(PROBES))
multinode: $(call BIN,$(MULTINODE) host_allreduce)
headers: $(BUILD)/header_check.o
$(ALL_NAMES): %: $(BUILD)/uccl_lite_%

$(BUILD):
	mkdir -p $@

$(call BIN,allreduce alltoall): $(BUILD)/uccl_lite_%: src/collectives/%.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(MSCCLPP_INC) $< -o $@ $(MSCCLPP_LIB) -lcudart -lpthread

$(BUILD)/uccl_lite_peer_ag_rs: src/collectives/peer_ag_rs.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MSCCLPP_INC) -I$(NCCL_HOME)/include $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib

$(BUILD)/uccl_lite_peer_ag_rs_packet: src/collectives/peer_ag_rs.cu $(BUILD)/cpu_router/libuccl_lite_cpu_router.a | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_PACKET_ROUTER $(MSCCLPP_INC) $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(ROUTER_STATIC) -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl

$(BUILD)/uccl_lite_nccl_4l4_mpi_baseline: benchmarks/collectives/nccl_4l4_mpi_baseline.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -I$(NCCL_HOME)/include $(MPI_INC) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib $(MPI_LIB) -lcudart

$(BUILD)/uccl_lite_nccl_2x4l4_baseline: benchmarks/collectives/nccl_2x4l4_baseline.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -I$(NCCL_HOME)/include $(MPI_INC) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib $(MPI_LIB) -lcudart

$(BUILD)/uccl_lite_alltoall_8l4: benchmarks/collectives/alltoall_8l4_bench.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -I$(NCCL_HOME)/include $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread

$(BUILD)/uccl_lite_duplex_allgather_8l4: benchmarks/cpu_router/duplex_allgather_8l4.cu $(BUILD)/cpu_router/libuccl_lite_cpu_router.a | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(ROUTER_STATIC) -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl

$(call BIN,$(MULTINODE)): $(BUILD)/uccl_lite_%: src/collectives/%.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(IB_INC) $(MPI_INC) $< -o $@ $(IB_LIB) $(MPI_LIB) -lcudart -lpthread

$(BUILD)/uccl_lite_multinode_cpu_router: src/collectives/multinode_allreduce.cu

$(BUILD)/uccl_lite_multinode_host_allgather: src/collectives/multinode_host_allgather.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -lcudart -lpthread

# Optional pre-registered host RDMA data plane; the MPI-only target stays independent.
$(BUILD)/uccl_lite_multinode_host_rdma_allgather: src/collectives/multinode_host_allgather.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 -DUCCL_LITE_HOST_RDMA $(IB_INC) $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(IB_LIB) $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -lcudart -lpthread

# This fallback deliberately has no MSCCL++ dependency or GPU-memory registration.
$(BUILD)/uccl_lite_host_allreduce: src/collectives/host_allreduce.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(MPI_INC) $< -o $@ $(MPI_LIB) -lcudart -lpthread

$(call BIN,dense_fusion fused_moe fused_moe_layer): $(BUILD)/uccl_lite_%: benchmarks/fusion/%.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(FUSION_FLAGS) $(FUSION_INC) $< -o $@ $(FUSION_LIB) -lcudart -lpthread

$(BUILD)/uccl_lite_router_fusion_8l4: benchmarks/fusion/router_fusion_8l4.cu $(BUILD)/cpu_router/libuccl_lite_cpu_router.a | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(FUSION_FLAGS) $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(ROUTER_STATIC) -o $@ -L$(NCCL_HOME)/lib -lnccl -lcublas -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl

$(BUILD)/uccl_lite_fused_moe_layer_qwen: benchmarks/fusion/fused_moe_layer.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DMOE_QWEN36 $(FUSION_FLAGS) $(FUSION_INC) $< -o $@ $(FUSION_LIB) -lcudart -lpthread

$(call BIN,gemm_probe mma_gemm_probe mma_peak_probe moe_overlap_probe): $(BUILD)/uccl_lite_%: benchmarks/probes/%.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $< -o $@ -lcublas

$(BUILD)/uccl_lite_cutlass_gemm_probe: benchmarks/probes/cutlass_gemm_probe.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(FUSION_FLAGS) -Iinclude/uccl_lite -I$(CUTLASS_HOME)/include $< -o $@ -lcublas

$(BUILD)/uccl_lite_hardware_profiler: benchmarks/probes/hardware_profiler.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $< -o $@ -Xcompiler=-pthread

$(BUILD)/host_memory_bw_4l4_20261111/stream_triad: benchmarks/data/host_memory_bw_4l4_20261111/stream_triad.cc | $(BUILD)
	mkdir -p $(@D)
	$(CXX) -O3 -march=native -fopenmp -std=c++17 $< -o $@

$(BUILD)/header_check.o: benchmarks/header_check.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(FUSION_FLAGS) $(FUSION_INC) -c $< -o $@

help:
	@printf '%s\n' 'UCCL Lite targets:' '  make single-node  (default): AllReduce and All-to-All' '  make peer-collectives: four-GPU AllGather/ReduceScatter + NCCL comparison' '  make fusion: dense and expert-parallel MoE benchmarks' '  make probes: hardware topology/bandwidth and GEMM/overlap diagnostics' '  make headers: compile-check reusable MoE headers' '  make all: all of the above' '  make multinode: two-node benchmarks (Open MPI required)' '  make <name>: build one executable, e.g. multinode_allreduce' 'See docs/build.md for dependency paths and launch commands.'
	@printf '%s\n' '  make cpu-router: optional static/shared CPU-router module' '  make local-collectives: two-NUMA BF16 fallback library' '  make adaptive-collectives: BF16 dispatcher + local/router calibration (optional NCCL control)' '  make router-checks local-collectives-check: core, policy, eager, and graph correctness executables'
	@printf '%s\n' '  make eight-gpu-alltoall: BF16 host/peer paths against NCCL on two four-GPU NUMA groups'
	@printf '%s\n' '  make nccl_4l4_mpi_baseline nccl_2x4l4_baseline: checked FP32 AllReduce and AR/AG/RS controls'
	@printf '%s\n' '  make multinode_host_allgather: experimental two-copy host-RDMA AllGather + NCCL control'
	@printf '%s\n' '  make host-memory-bw-triad: NUMA-local large-working-set host DRAM probe'
	@printf '%s\n' '  make peer-ag-rs-packet: FP32 AllGather peer paths plus optional host-packet router'

	@printf '%s\n' '  make peer-ring-check: signed-input asynchronous ring RS check' '  make peer-hybrid-allreduce peer-hybrid-check: experimental in-place ring/packet AR and independent check'

	@printf '%s\n' '  make peer-small-medium-check: phased LL and host-gather checks' '  make resident-collectives resident-check: experimental prepared collectives and boundary checks'
	@printf '%s\n' '  make compact-collectives: experimental BF16 mapped-host AG and resident AR/AG/RS, with router/NCCL controls'

include src/cpu_router/module.mk

.PHONY: compact-collectives
compact-collectives: $(BUILD)/uccl_lite_compact_collectives
$(BUILD)/uccl_lite_compact_collectives: benchmarks/cpu_router/compact_collectives.cu benchmarks/cpu_router/compact_transport.cuh benchmarks/cpu_router/host_ring.cuh $(ROUTER_STATIC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(ROUTER_STATIC) -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl

-include $(wildcard $(BUILD)/*.d)

# Historical serving data verification and README-style SVGs; no GPU required.
PYTHON ?= python3
.PHONY: vllm-pcie-figures
vllm-pcie-figures:
	$(PYTHON) integration/vllm_pcie/plot.py --preview

.PHONY: vllm-pcie-collective-figures
vllm-pcie-collective-figures:
	$(PYTHON) integration/vllm_pcie/plot_collectives.py --preview

# Host-only Python binding to the same out-of-place NCCL AllReduce API.
PYBIND11_INC ?= $(shell $(PYTHON) -c 'import torch; print(torch.__path__[0] + "/include")')
PYTHON_INC = $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("include"))')
.PHONY: vllm-pcie-nccl-reference
vllm-pcie-nccl-reference: $(BUILD)/_uccl_lite_nccl_reference.so
$(BUILD)/_uccl_lite_nccl_reference.so: integration/vllm_pcie/nccl_reference.cpp | $(BUILD)
	$(CXX) -O3 -std=c++17 -shared -fPIC -I$(PYBIND11_INC) -I$(PYTHON_INC) -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include $< -o $@ -L$(NCCL_HOME)/lib -l:libnccl.so.2 -Wl,-rpath,$(NCCL_HOME)/lib

.PHONY: peer-ring-check
peer-ring-check: $(BUILD)/uccl_lite_peer_ring_check
$(BUILD)/uccl_lite_peer_ring_check: tests/collectives/peer_ring_check.cu src/collectives/peer_ag_rs.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MSCCLPP_INC) -I$(NCCL_HOME)/include $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib

.PHONY: peer-hybrid-allreduce
peer-hybrid-allreduce: $(BUILD)/uccl_lite_peer_hybrid_allreduce
$(BUILD)/uccl_lite_peer_hybrid_allreduce: src/collectives/peer_hybrid_allreduce.cu src/collectives/peer_ag_rs.cu $(BUILD)/cpu_router/libuccl_lite_cpu_router.a | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_PACKET_ROUTER $(MSCCLPP_INC) $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(ROUTER_STATIC) -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl

.PHONY: peer-hybrid-check
peer-hybrid-check: $(BUILD)/uccl_lite_peer_hybrid_check
$(BUILD)/uccl_lite_peer_hybrid_check: tests/collectives/peer_hybrid_check.cu src/collectives/peer_ag_rs.cu $(BUILD)/cpu_router/libuccl_lite_cpu_router.a | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_PACKET_ROUTER $(MSCCLPP_INC) $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(ROUTER_STATIC) -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl

.PHONY: peer-small-medium-check
peer-small-medium-check: $(BUILD)/uccl_lite_peer_small_medium_check
$(BUILD)/uccl_lite_peer_small_medium_check: tests/collectives/peer_small_medium_check.cu src/collectives/peer_ag_rs.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MSCCLPP_INC) -I$(NCCL_HOME)/include $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib

.PHONY: resident-collectives
resident-collectives: $(BUILD)/uccl_lite_resident_collectives
$(BUILD)/uccl_lite_resident_collectives: src/collectives/resident_collectives.cu src/collectives/peer_ag_rs.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MSCCLPP_INC) -I$(NCCL_HOME)/include $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib

.PHONY: resident-check
resident-check: $(BUILD)/uccl_lite_resident_check
$(BUILD)/uccl_lite_resident_check: tests/collectives/resident_check.cu src/collectives/resident_collectives.cu src/collectives/peer_ag_rs.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MSCCLPP_INC) -I$(NCCL_HOME)/include $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib

# Joint peer/router experiment: reference excludes every mutable solution object.
.PHONY: joint-bench joint-nccl-reference
JOINT_OBJECTS := $(BUILD)/joint_collectives.o $(BUILD)/peer_collectives.o
joint-bench: $(BUILD)/uccl_lite_joint_bench
joint-nccl-reference: $(BUILD)/uccl_lite_joint_nccl_reference
$(BUILD)/joint_collectives.o: src/collectives/joint_collectives.cu include/uccl_lite/joint_collectives.hpp | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -c $< -o $@
$(BUILD)/peer_collectives.o: src/collectives/peer_collectives.cu include/uccl_lite/collectives/peer_collectives.hpp include/uccl_lite/collectives/device_primitives.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -c $< -o $@
$(BUILD)/uccl_lite_joint_bench: benchmarks/joint/joint_bench.cu $(JOINT_OBJECTS) $(ROUTER_STATIC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(JOINT_OBJECTS) $(ROUTER_STATIC) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl -o $@
$(BUILD)/uccl_lite_joint_nccl_reference: benchmarks/joint/joint_bench.cu benchmarks/cpu_router/nccl_identity.hpp | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DNCCL_REFERENCE_ONLY -I$(NCCL_HOME)/include $< -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -o $@

# Frozen four/eight-rank router fixtures from the retained SHM experiment.
.PHONY: joint-router-four joint-router-eight
joint-router-four: $(BUILD)/uccl_lite_router_four_bench $(BUILD)/uccl_lite_router_four_nccl_reference $(BUILD)/uccl_lite_router_four_mixed
joint-router-eight: $(BUILD)/uccl_lite_router_eight_bench $(BUILD)/uccl_lite_router_eight_nccl_reference $(BUILD)/uccl_lite_router_eight_mixed
$(BUILD)/uccl_lite_router_%_bench: benchmarks/joint/focused/%_aligned.cu $(JOINT_OBJECTS) $(ROUTER_STATIC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -Ibenchmarks/joint -I$(NCCL_HOME)/include $< $(JOINT_OBJECTS) $(ROUTER_STATIC) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl -o $@
$(BUILD)/uccl_lite_router_%_mixed: benchmarks/joint/focused/%_mixed_alignment.cu $(JOINT_OBJECTS) $(ROUTER_STATIC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -Ibenchmarks/joint -I$(NCCL_HOME)/include $< $(JOINT_OBJECTS) $(ROUTER_STATIC) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl -o $@
$(BUILD)/uccl_lite_router_%_nccl_reference: benchmarks/joint/focused/%_aligned.cu benchmarks/cpu_router/nccl_identity.hpp | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DNCCL_REFERENCE_ONLY -Ibenchmarks/joint -I$(NCCL_HOME)/include $< -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -o $@

.PHONY: joint-peer-check
joint-peer-check: $(BUILD)/uccl_lite_joint_peer_check
$(BUILD)/uccl_lite_joint_peer_check: tests/collectives/joint_peer_check.cu benchmarks/joint/joint_bench.cu $(JOINT_OBJECTS) $(ROUTER_STATIC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(JOINT_OBJECTS) $(ROUTER_STATIC) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl -o $@
