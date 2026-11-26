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

$(BUILD)/host_memory_bw_4l4_20261111/stream_triad: benchmarks/probes/host_memory_triad.cc | $(BUILD)
	mkdir -p $(@D)
	$(CXX) -O3 -march=native -fopenmp -std=c++17 $< -o $@

$(BUILD)/header_check.o: benchmarks/header_check.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(FUSION_FLAGS) $(FUSION_INC) -c $< -o $@

help:
	@printf '%s\n' 'Topology-driven runtime targets:' '  make prepared-runtime prepared-runtime-check: unified discovery/selection API and checks' '  make topology calibrate-hardware planner: discovery, calibration and resource planning' '  make peer-runtime: standalone peer library' '  make prepared-local-runtime: multiprocess selection core' '  make prepared-network-runtime prepared-network-check: distributed caller-baseline selection and checks' '  make nccl-runtime: common strict NCCL adapter' '  make native-nccl-fallback: optional caller-owned native adapter' '  make nccl-prepared-api nccl-prepared-api-check: native communicator C extension and checks' '  make nccl-prepared-abi nccl-auto-init-check: native NCCL ABI with opt-in automatic preparation' '  make prepared-local-check prepared-scalar-check prepared-graph-check: build correctness clients' '  make network-runtime-check: build the two-host registry checker' 'See docs/build.md and docs/topology-runtime-refactor.md for supported capabilities.'
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

include src/planner/module.mk
include src/topology/module.mk
CORE_CALL_OBJECT := $(BUILD)/runtime/collective_call.o
include src/cpu_router/module.mk
include src/collectives/module.mk
include src/multinode/module.mk

.PHONY: compact-collectives
compact-collectives: $(BUILD)/uccl_lite_compact_collectives
$(BUILD)/uccl_lite_compact_collectives: benchmarks/cpu_router/compact_collectives.cu benchmarks/cpu_router/compact_transport.cuh benchmarks/cpu_router/host_ring.cuh $(ROUTER_STATIC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(ROUTER_STATIC) -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl

-include $(wildcard $(BUILD)/*.d)

# Optional NCCL ABI adapter: MSCCL++ algorithm runtime + retained UCCL kernels.
include integration/nccl_lite/policy.mk

.PHONY: nccl-dropin-figures
nccl-dropin-figures:
	MPLCONFIGDIR=$(abspath $(BUILD)/mplconfig) python3 -B scripts/plot_nccl_dropin.py --preview

.PHONY: nccl-local8-figures
nccl-local8-figures:
	MPLCONFIGDIR=$(abspath $(BUILD)/mplconfig) python3 -B scripts/plot_nccl_local8.py --preview

# Track named policy changes even when an existing BUILD directory is reused.
# Experimental EXTRA_NVCC_FLAGS changes still require a fresh BUILD directory.
.PHONY: uccl_lite_nccl_policy_force
uccl_lite_nccl_policy_force:

$(BUILD)/nccl_lite/build.policy: uccl_lite_nccl_policy_force
	mkdir -p $(@D)
	@printf '%s\n' '$(UCCL_LITE_NCCL_POLICY)' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

.PHONY: nccl-mscclpp nccl-mscclpp-strict
nccl-mscclpp: nccl-mscclpp-strict
nccl-mscclpp-strict: $(BUILD)/nccl_lite/strict/libnccl.so.2 $(BUILD)/nccl_lite/libuccl_lite_nccl_mscclpp.so

# Eight-L4 BF16 transport, sharing the strict ABI and CPU control protocol.
# This target has no native NCCL or MSCCL++ link dependency.
.PHONY: nccl-local8-strict uccl_lite_local8_flags_force
nccl-local8-strict: $(BUILD)/nccl_local8/strict/libnccl.so.2

LOCAL8_PACKET_BYTES ?= 1048576
LOCAL8_PACKET_FLAGS = $(call uccl_lite_host_packet_device_flags,$(LOCAL8_PACKET_BYTES))
LOCAL8_HOST_FLAGS = $(call uccl_lite_host_packet_control_flags,$(LOCAL8_PACKET_BYTES))
LOCAL8_OBJECTS = $(BUILD)/nccl_runtime/frontend.o $(TOPOLOGY_OBJECTS) $(BUILD)/runtime/backend_registry_host.o $(BUILD)/runtime/host_packet_backend.o $(BUILD)/nccl_local8/multiprocess.o $(BUILD)/nccl_local8/packet.o $(BUILD)/cpu_router/cpu_executor.o

uccl_lite_local8_flags_force:

$(BUILD)/nccl_local8/build.flags: uccl_lite_local8_flags_force
	mkdir -p $(@D)
	@printf '%s\n' '$(NVCC) $(NVCC_FLAGS) $(LOCAL8_PACKET_FLAGS)' '$(CXX) $(LOCAL8_HOST_FLAGS)' '$(CUDA_HOME) $(NCCL_HOME)' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

$(BUILD)/nccl_local8/packet.o: src/collectives/multiprocess_packet_ag.cu $(BUILD)/nccl_local8/build.flags
	$(NVCC) $(NVCC_FLAGS) $(LOCAL8_PACKET_FLAGS) -std=c++17 -Xcompiler=-fPIC -Iinclude -c $< -o $@

$(BUILD)/nccl_local8/multiprocess.o: src/collectives/multiprocess_collectives.cc $(BUILD)/nccl_local8/build.flags
	$(CXX) -O3 -std=c++17 -fPIC -pthread -MMD -MP $(LOCAL8_HOST_FLAGS) -Iinclude -I$(CUDA_HOME)/include -c $< -o $@

$(BUILD)/nccl_local8/libuccl_lite_nccl_local8.so: $(LOCAL8_OBJECTS) integration/nccl_lite/mscclpp_exports.version
	$(CXX) -shared -pthread $(LOCAL8_OBJECTS) -Wl,--version-script=integration/nccl_lite/mscclpp_exports.version -Wl,-soname,libnccl.so.2 -Wl,-z,defs -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@

$(BUILD)/nccl_local8/strict/libnccl.so.2: $(BUILD)/nccl_local8/libuccl_lite_nccl_local8.so
	mkdir -p $(@D)
	ln -sfn ../libuccl_lite_nccl_local8.so $@

-include $(wildcard $(BUILD)/nccl_local8/*.d)

$(BUILD)/nccl_local8/strict_abi_check: tests/nccl_mscclpp/strict_abi.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_ABI_LOCAL8 -DUCCL_LITE_ABI_EXTENDED -DUCCL_LITE_ABI_BOUNDARY512 -DUCCL_LITE_ABI_SERIAL_PACKET -DUCCL_LITE_ABI_MEDIUM64 -DUCCL_LITE_ABI_MEDIUM128 $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(abspath $(NCCL_HOME)/lib)

$(BUILD)/nccl_local8/reference_timing_check: tests/nccl_mscclpp/local8_reference.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(abspath $(NCCL_HOME)/lib)

$(BUILD)/nccl_local8/queue_check: tests/nccl_mscclpp/local8_queue.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(abspath $(NCCL_HOME)/lib)

$(BUILD)/nccl_lite/libuccl_lite_nccl_strict.so: $(BUILD)/nccl_runtime/frontend.o $(BUILD)/runtime/backend_registry_network.o $(BUILD)/runtime/network_backend.o $(BUILD)/runtime/network_probe.o $(BUILD)/runtime/rank_permutation.o $(BUILD)/planner/network_plan.o $(BUILD)/planner/network_profile.o $(BUILD)/planner/measured_network_plan.o $(TOPOLOGY_OBJECTS) $(BUILD)/nccl_lite/mscclpp_backend.o $(PLANNER_STATIC) include/uccl_lite/collectives.h integration/nccl_lite/host_control.h integration/nccl_lite/mscclpp_exports.version
	$(CXX) -O3 -std=c++17 -fPIC -shared -pthread -Iinclude -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include $(BUILD)/nccl_runtime/frontend.o $(BUILD)/runtime/backend_registry_network.o $(BUILD)/runtime/network_backend.o $(BUILD)/runtime/network_probe.o $(BUILD)/runtime/rank_permutation.o $(BUILD)/planner/network_plan.o $(BUILD)/planner/network_profile.o $(BUILD)/planner/measured_network_plan.o $(TOPOLOGY_OBJECTS) $(BUILD)/nccl_lite/mscclpp_backend.o $(PLANNER_STATIC) -Wl,--version-script=integration/nccl_lite/mscclpp_exports.version -Wl,-soname,libnccl.so.2 -L$(MSCCLPP_IB_BUILD)/lib -lmscclpp -Wl,-rpath,$(MSCCLPP_IB_BUILD)/lib -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64 $(UCCL_LITE_NCCL_POLICY_LIBS) $(NCCL_STRICT_EXTRA_LIBS) -ldl -o $@

$(BUILD)/nccl_lite/strict/libnccl.so.2: $(BUILD)/nccl_lite/libuccl_lite_nccl_strict.so
	mkdir -p $(@D)
	ln -sfn ../libuccl_lite_nccl_strict.so $@
$(BUILD)/nccl_lite/mscclpp_backend.o: src/multinode/mscclpp_backend.cu src/multinode/device_collectives.cuh src/multinode/algorithm_transport.h include/uccl_lite/collectives.h
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) $(UCCL_LITE_NCCL_POLICY_FLAGS) -std=c++20 -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden $(IB_INC) -c $< -o $@

$(BUILD)/nccl_lite/mscclpp_backend.o: $(BUILD)/nccl_lite/build.policy integration/nccl_lite/policy.mk

$(BUILD)/nccl_lite/network_sizes_check: tests/multinode/network_sizes.cu $(BUILD)/nccl_lite/mscclpp_backend.o
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(IB_INC) $(MPI_INC) $< $(BUILD)/nccl_lite/mscclpp_backend.o -o $@ $(IB_LIB) $(MPI_LIB) -lcudart -lpthread $(UCCL_LITE_NCCL_POLICY_LIBS) $(NCCL_STRICT_EXTRA_LIBS)

$(BUILD)/nccl_lite/network_queue_check: tests/multinode/network_queue.cu $(BUILD)/nccl_lite/mscclpp_backend.o
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(IB_INC) $(MPI_INC) $< $(BUILD)/nccl_lite/mscclpp_backend.o -o $@ $(IB_LIB) $(MPI_LIB) -lcudart -lpthread $(UCCL_LITE_NCCL_POLICY_LIBS) $(NCCL_STRICT_EXTRA_LIBS)

$(BUILD)/nccl_lite/shared_host_packets_check: tests/multinode/shared_host_packets.cu src/multinode/shared_host_packets.h
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(IB_INC) $(MPI_INC) $< -o $@ $(IB_LIB) $(MPI_LIB) -lcudart -lpthread

# The previous mixed-adapter filename now resolves to the strict library too.
$(BUILD)/nccl_lite/libuccl_lite_nccl_mscclpp.so: $(BUILD)/nccl_lite/libuccl_lite_nccl_strict.so
	ln -sfn libuccl_lite_nccl_strict.so $@

-include $(wildcard $(BUILD)/nccl_lite/*.d)

$(BUILD)/nccl_lite/mscclpp_abi_check: tests/nccl_mscclpp/abi.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib

$(BUILD)/nccl_lite/mscclpp_strict_abi_check: tests/nccl_mscclpp/strict_abi.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib

$(BUILD)/nccl_lite/mscclpp_strict_abi_extended_check: tests/nccl_mscclpp/strict_abi.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_ABI_EXTENDED $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib

$(BUILD)/nccl_lite/mscclpp_strict_abi_boundary_check: tests/nccl_mscclpp/strict_abi.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_ABI_EXTENDED -DUCCL_LITE_ABI_BOUNDARY512 $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib

$(BUILD)/nccl_lite/mscclpp_strict_abi_serial_check: tests/nccl_mscclpp/strict_abi.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_ABI_EXTENDED -DUCCL_LITE_ABI_BOUNDARY512 -DUCCL_LITE_ABI_SERIAL_PACKET $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib

$(BUILD)/nccl_lite/mscclpp_strict_abi_medium64_check: tests/nccl_mscclpp/strict_abi.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_ABI_EXTENDED -DUCCL_LITE_ABI_BOUNDARY512 -DUCCL_LITE_ABI_SERIAL_PACKET -DUCCL_LITE_ABI_MEDIUM64 $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib

$(BUILD)/nccl_lite/mscclpp_strict_abi_medium128_check: tests/nccl_mscclpp/strict_abi.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_ABI_EXTENDED -DUCCL_LITE_ABI_BOUNDARY512 -DUCCL_LITE_ABI_SERIAL_PACKET -DUCCL_LITE_ABI_MEDIUM64 -DUCCL_LITE_ABI_MEDIUM128 $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib

$(BUILD)/nccl_lite/mscclpp_control_check: tests/nccl_dropin/host_control.cc integration/nccl_lite/host_control.h include/uccl_lite/bootstrap/control.hpp
	mkdir -p $(@D)
	$(CXX) -O2 -std=c++17 -pthread $< -o $@

# Historical serving data verification and README-style SVGs; no GPU required.
PYTHON ?= python3
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
JOINT_OBJECTS := $(CORE_CALL_OBJECT) $(BUILD)/joint_collectives.o $(BUILD)/peer_collectives.o $(PLANNER_OBJECTS) $(BUILD)/runtime/peer_calibration.o $(BUILD)/topology/calibration.o
joint-bench: $(BUILD)/uccl_lite_joint_bench
joint-nccl-reference: $(BUILD)/uccl_lite_joint_nccl_reference
$(BUILD)/joint_collectives.o: src/collectives/joint_collectives.cu include/uccl_lite/joint_collectives.hpp | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -c $< -o $@
$(BUILD)/peer_collectives.o: src/collectives/peer_collectives.cu include/uccl_lite/collectives/peer_collectives.hpp include/uccl_lite/collectives/device_primitives.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Xcompiler=-fPIC -Iinclude -c $< -o $@
$(BUILD)/uccl_lite_joint_bench: benchmarks/joint/joint_bench.cu $(JOINT_OBJECTS) $(ROUTER_STATIC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(JOINT_OBJECTS) $(ROUTER_STATIC) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl -o $@
$(BUILD)/uccl_lite_joint_nccl_reference: benchmarks/joint/joint_bench.cu benchmarks/cpu_router/nccl_identity.hpp | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DNCCL_REFERENCE_ONLY -I$(NCCL_HOME)/include $< -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -o $@

.PHONY: joint-peer-check
joint-peer-check: $(BUILD)/uccl_lite_joint_peer_check
$(BUILD)/uccl_lite_joint_peer_check: tests/collectives/joint_peer_check.cu tests/planner/peer_fixture.hpp benchmarks/joint/joint_bench.cu $(JOINT_OBJECTS) $(ROUTER_STATIC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(JOINT_OBJECTS) $(ROUTER_STATIC) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl -o $@

# Matched FP32 in-place C++ AllReduce; the UCCL target does not link NCCL.
.PHONY: pro6000-ag-rs pro6000-ag-rs-reference pro6000-ag-rs-evidence-check
pro6000-ag-rs-evidence-check:
	$(PYTHON) -B -m unittest discover -s tests -p test_pro6000_ag_rs_evidence.py -v
pro6000-ag-rs: $(BUILD)/uccl_lite_pro6000_ag_rs
pro6000-ag-rs-reference: $(BUILD)/uccl_lite_pro6000_ag_rs_nccl
$(BUILD)/uccl_lite_pro6000_ag_rs: src/collectives/pro6000_ag_rs.cu include/uccl_lite/pro6000_ag_rs.cuh include/uccl_lite/pro6000_allreduce.cuh include/uccl_lite/allreduce_validation.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 -Iinclude -I$(MSCCLPP_HOME)/include $< -o $@ -L$(MSCCLPP_HOME)/lib -lmscclpp -lcudart -lpthread -Xlinker -rpath=$(MSCCLPP_HOME)/lib
$(BUILD)/uccl_lite_pro6000_ag_rs_nccl: src/collectives/pro6000_ag_rs.cu include/uccl_lite/allreduce_validation.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 -DUCCL_LITE_NCCL_REFERENCE -Iinclude -I$(MSCCLPP_HOME)/include -I$(NCCL_HOME)/include $< -o $@ -L$(MSCCLPP_HOME)/lib -lmscclpp -L$(NCCL_HOME)/lib -l:libnccl.so.2 -lcudart -lpthread -Xlinker -rpath=$(MSCCLPP_HOME)/lib -Xlinker -rpath=$(NCCL_HOME)/lib

.PHONY: pro6000-ar pro6000-ar-reference
pro6000-ar: $(BUILD)/uccl_lite_pro6000_ar
pro6000-ar-reference: $(BUILD)/uccl_lite_pro6000_ar_nccl
$(BUILD)/uccl_lite_pro6000_ar: src/collectives/pro6000_allreduce.cu include/uccl_lite/pro6000_allreduce.cuh include/uccl_lite/allreduce_validation.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 -Iinclude -I$(MSCCLPP_HOME)/include $< -o $@ -L$(MSCCLPP_HOME)/lib -lmscclpp -lcudart -lpthread -Xlinker -rpath=$(MSCCLPP_HOME)/lib
$(BUILD)/uccl_lite_pro6000_ar_nccl: src/collectives/pro6000_allreduce.cu include/uccl_lite/allreduce_validation.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 -DUCCL_LITE_NCCL_REFERENCE -Iinclude -I$(MSCCLPP_HOME)/include -I$(NCCL_HOME)/include $< -o $@ -L$(MSCCLPP_HOME)/lib -lmscclpp -L$(NCCL_HOME)/lib -l:libnccl.so.2 -lcudart -lpthread -Xlinker -rpath=$(MSCCLPP_HOME)/lib -Xlinker -rpath=$(NCCL_HOME)/lib

.PHONY: pro6000-ar-evidence-check
pro6000-ar-evidence-check:
	$(PYTHON) -B -m unittest discover -s tests -p test_pro6000_allreduce_evidence.py -v
$(BUILD)/nccl_lite/host_packet_transport_check: tests/multinode/host_packet_transport.cu src/multinode/host_packet_transport.h src/multinode/shared_host_packets.h
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(IB_INC) $(MPI_INC) $< -o $@ $(IB_LIB) $(MPI_LIB) -lcudart -lpthread

$(BUILD)/nccl_lite/host_packet_callbacks_check: tests/multinode/host_packet_callbacks.cu src/multinode/host_packet_transport.h src/multinode/shared_host_packets.h
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(IB_INC) $(MPI_INC) $< -o $@ $(IB_LIB) $(MPI_LIB) -lcudart -lpthread

$(BUILD)/nccl_lite/async_host_gather_check: tests/multinode/async_host_gather.cu src/multinode/async_host_gather.cuh src/multinode/host_packet_transport.h src/multinode/shared_host_packets.h
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) $(UCCL_LITE_NCCL_POLICY_FLAGS) -std=c++20 $(IB_INC) $(MPI_INC) $< -o $@ $(IB_LIB) $(MPI_LIB) -lcudart -lpthread

$(BUILD)/nccl_lite/async_host_gather_check: $(BUILD)/nccl_lite/build.policy integration/nccl_lite/policy.mk

# Asynchronous graph calls with fresh data and allocation reuse; no native link.
$(BUILD)/nccl_lite/sync_device_check: tests/nccl_mscclpp/sync_device.cu $(BUILD)/nccl_lite/strict/libnccl.so.2
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(BUILD)/nccl_lite -l:libuccl_lite_nccl_strict.so -lcuda -Xlinker -rpath=$(abspath $(BUILD)/nccl_lite/strict)

$(BUILD)/nccl_lite/graph_queue_check: tests/nccl_mscclpp/graph_queue.cu $(BUILD)/nccl_lite/strict/libnccl.so.2
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(BUILD)/nccl_lite -l:libuccl_lite_nccl_strict.so -Xlinker -rpath=$(abspath $(BUILD)/nccl_lite/strict)

$(BUILD)/nccl_lite/layout_queue_check: tests/nccl_mscclpp/layout_queue.cu $(BUILD)/nccl_lite/strict/libnccl.so.2
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(BUILD)/nccl_lite -l:libuccl_lite_nccl_strict.so -Xlinker -rpath=$(abspath $(BUILD)/nccl_lite/strict)

# Explicit per-call comparison including internal retirement; no native link.
$(BUILD)/nccl_lite/reference_timing_check: tests/nccl_mscclpp/reference_timing.cu $(BUILD)/nccl_lite/strict/libnccl.so.2
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(BUILD)/nccl_lite -l:libuccl_lite_nccl_strict.so -Xlinker -rpath=$(abspath $(BUILD)/nccl_lite/strict)

# Four-L4 candidate reuses the strict ABI and CPU identity consensus.
include integration/nccl_lite/local_policy.mk

.PHONY: uccl_lite_nccl_local_policy_force
uccl_lite_nccl_local_policy_force:

$(BUILD)/nccl_local:
	mkdir -p $@

$(BUILD)/nccl_local/build.policy: export UCCL_LOCAL_BUILD_POLICY = $(UCCL_LITE_NCCL_LOCAL_POLICY)
$(BUILD)/nccl_local/build.policy: export UCCL_LOCAL_BUILD_NVCC = $(NVCC)
$(BUILD)/nccl_local/build.policy: export UCCL_LOCAL_BUILD_CXX = $(CXX)
$(BUILD)/nccl_local/build.policy: export UCCL_LOCAL_BUILD_NVCC_FLAGS = $(NVCC_FLAGS) $(UCCL_LITE_NCCL_LOCAL_FLAGS) $(UCCL_LITE_NCCL_LOCAL_REPRO_FLAGS)
$(BUILD)/nccl_local/build.policy: export UCCL_LOCAL_BUILD_INCLUDES = $(MSCCLPP_INC) -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include
$(BUILD)/nccl_local/build.policy: export UCCL_LOCAL_BUILD_LIBS = $(MSCCLPP_BUILD)/lib $(CUDA_HOME)/lib64 $(UCCL_LITE_NCCL_LOCAL_LIBS) $(NCCL_LOCAL_EXTRA_LIBS)
$(BUILD)/nccl_local/build.policy: uccl_lite_nccl_local_policy_force integration/nccl_lite/write_build_stamp.py | $(BUILD)/nccl_local
	$(PYTHON) -B integration/nccl_lite/write_build_stamp.py $@

.PHONY: nccl-local-strict
nccl-local-strict: $(BUILD)/nccl_local/strict/libnccl.so.2
$(BUILD)/nccl_local/local_strict.o: src/collectives/peer_packet.cu include/uccl_lite/collectives/peer_packet.hpp $(BUILD)/nccl_local/build.policy integration/nccl_lite/local_policy.mk
	mkdir -p $(@D) $(@D)/cuda-intermediates
	$(NVCC) $(NVCC_FLAGS) $(UCCL_LITE_NCCL_LOCAL_FLAGS) $(UCCL_LITE_NCCL_LOCAL_REPRO_FLAGS) -std=c++17 -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden $(MSCCLPP_INC) -c $< -o $@
$(BUILD)/nccl_local/libuccl_lite_nccl_local_strict.so: $(BUILD)/nccl_runtime/frontend.o $(BUILD)/runtime/backend_registry_peer.o $(BUILD)/runtime/peer_packet_backend.o $(TOPOLOGY_OBJECTS) $(BUILD)/nccl_local/local_strict.o $(PLANNER_STATIC) integration/nccl_lite/host_control.h integration/nccl_lite/mscclpp_exports.version $(BUILD)/nccl_local/build.policy $(UCCL_LITE_NCCL_LOCAL_DEPS)
	$(CXX) -O3 -std=c++17 -MMD -MP -MF $(BUILD)/nccl_local/mscclpp_strict.d -MT $@ -fPIC -shared -pthread -Iinclude -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include $(BUILD)/nccl_runtime/frontend.o $(BUILD)/runtime/backend_registry_peer.o $(BUILD)/runtime/peer_packet_backend.o $(TOPOLOGY_OBJECTS) $(BUILD)/nccl_local/local_strict.o $(PLANNER_STATIC) -Wl,--version-script=integration/nccl_lite/mscclpp_exports.version -Wl,-soname,libnccl.so.2 -L$(MSCCLPP_BUILD)/lib -lmscclpp -Wl,-rpath,$(MSCCLPP_BUILD)/lib -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64 $(UCCL_LITE_NCCL_LOCAL_LIBS) $(NCCL_LOCAL_EXTRA_LIBS) -o $@
$(BUILD)/nccl_local/strict/libnccl.so.2: $(BUILD)/nccl_local/libuccl_lite_nccl_local_strict.so
	mkdir -p $(@D)
	ln -sfn ../libuccl_lite_nccl_local_strict.so $@
-include $(BUILD)/nccl_local/local_strict.d $(BUILD)/nccl_local/mscclpp_strict.d
$(BUILD)/nccl_local/strict_abi_check: tests/nccl_mscclpp/strict_abi.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_ABI_LOCAL -DUCCL_LITE_ABI_EXTENDED -DUCCL_LITE_ABI_BOUNDARY512 -DUCCL_LITE_ABI_SERIAL_PACKET -I$(NCCL_HOME)/include $(MPI_INC) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib $(MPI_LIB) -lcudart
$(BUILD)/nccl_local/reference_timing_check: tests/nccl_mscclpp/reference_timing.cu $(BUILD)/nccl_local/strict/libnccl.so.2
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_REFERENCE_LOCAL $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(BUILD)/nccl_local -l:libuccl_lite_nccl_local_strict.so -Xlinker -rpath=$(abspath $(BUILD)/nccl_local/strict)
$(BUILD)/nccl_local/queue_check: tests/nccl_mscclpp/local_queue.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -I$(NCCL_HOME)/include $(MPI_INC) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib $(MPI_LIB) -lcudart
# Local event-timing client for the original peer AG/RS timing scope.
$(BUILD)/nccl_local/reference_event_check: tests/nccl_mscclpp/reference_timing.cu $(BUILD)/nccl_local/strict/libnccl.so.2
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_REFERENCE_LOCAL $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(BUILD)/nccl_local -l:libuccl_lite_nccl_local_strict.so -Xlinker -rpath=$(abspath $(BUILD)/nccl_local/strict)
$(BUILD)/nccl_local/ring_queue_check: tests/nccl_mscclpp/local_queue.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_LOCAL_RING_QUEUE -I$(NCCL_HOME)/include $(MPI_INC) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib $(MPI_LIB) -lcudart
$(BUILD)/nccl_local/strict_abi_diagnostic_check: tests/nccl_mscclpp/strict_abi.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_ABI_LOCAL -DUCCL_LITE_ABI_EXTENDED -DUCCL_LITE_ABI_BOUNDARY512 -DUCCL_LITE_ABI_SERIAL_PACKET -I$(NCCL_HOME)/include $(MPI_INC) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib $(MPI_LIB) -lcudart

$(BUILD)/nccl_local/sm_queue_check: tests/nccl_mscclpp/local_queue.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_LOCAL_SM_QUEUE -I$(NCCL_HOME)/include $(MPI_INC) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib $(MPI_LIB) -lcudart

$(BUILD)/nccl_local/hierarchy_queue_check: tests/nccl_mscclpp/local_queue.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_LOCAL_HIER_QUEUE -I$(NCCL_HOME)/include $(MPI_INC) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib $(MPI_LIB) -lcudart

$(BUILD)/nccl_local/gather_queue_check: tests/nccl_mscclpp/local_queue.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_LOCAL_AG_GRAPH_QUEUE -I$(NCCL_HOME)/include $(MPI_INC) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib $(MPI_LIB) -lcudart

# Read-only diagnostic against the frozen original; never an admission gate.
LOCAL_REFERENCE ?= build/local_reference/source
$(BUILD)/nccl_local/standalone_completion_probe: tests/nccl_mscclpp/standalone_completion_probe.cu $(LOCAL_REFERENCE)/src/collectives/peer_ag_rs.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MSCCLPP_INC) -I$(LOCAL_REFERENCE)/src/collectives -I$(NCCL_HOME)/include $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread

$(BUILD)/nccl_local/ring4_queue_check: tests/nccl_mscclpp/local_queue.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_LOCAL_RING_QUEUE -DUCCL_LITE_LOCAL_RING4_QUEUE -I$(NCCL_HOME)/include $(MPI_INC) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib $(MPI_LIB) -lcudart

# Medium-AG epoch range: changing 16/32/64/128/256-KiB calls and streams.
$(BUILD)/nccl_local/gather_range_queue_check: tests/nccl_mscclpp/local_queue.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_LOCAL_AG_GRAPH_QUEUE -DUCCL_LITE_LOCAL_AG_GRAPH_RANGE_QUEUE -I$(NCCL_HOME)/include $(MPI_INC) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib $(MPI_LIB) -lcudart

# Three legacy allocations and one stream-ordered allocation; same queue checks.
$(BUILD)/nccl_local/mixed_allocation_queue_check: tests/nccl_mscclpp/local_queue.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_LOCAL_HIER_QUEUE -DUCCL_LITE_LOCAL_MIXED_ALLOC_QUEUE -I$(NCCL_HOME)/include $(MPI_INC) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib $(MPI_LIB) -lcudart -lcuda

.PHONY: joint-policy-check
joint-policy-check: $(BUILD)/uccl_lite_joint_policy_check
$(BUILD)/uccl_lite_joint_policy_check: tests/collectives/joint_policy_check.cc $(JOINT_OBJECTS) $(ROUTER_STATIC)
	$(CXX) -O2 -std=c++17 $(ROUTER_INC) $< $(JOINT_OBJECTS) $(ROUTER_LIB) -o $@

$(BUILD)/runtime/peer_calibration.o: src/runtime/peer_calibration.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Xcompiler=-fPIC -Iinclude -c $< -o $@
-include $(wildcard $(BUILD)/runtime/*.d)

.PHONY: peer-qualification-check
peer-qualification-check: $(BUILD)/uccl_lite_peer_qualification_check
$(BUILD)/uccl_lite_peer_qualification_check: tests/collectives/peer_qualification_check.cu $(BUILD)/runtime/peer_calibration.o $(TOPOLOGY_STATIC) $(PLANNER_STATIC)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude $< $(BUILD)/runtime/peer_calibration.o $(TOPOLOGY_STATIC) $(PLANNER_STATIC) -ldl -o $@

# Shared ABI frontend; providers describe runtime capabilities instead of
# compiling different dtype/topology constants into the NCCL entry points.
$(BUILD)/nccl_runtime/frontend.o: integration/nccl_lite/mscclpp_strict.cc
	mkdir -p $(@D)
	$(CXX) -O3 -std=c++17 -fPIC -pthread -MMD -MP -Iinclude -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -c $< -o $@
$(BUILD)/runtime/%.o: src/runtime/%.cc
	mkdir -p $(@D)
	$(CXX) -O3 -std=c++17 -fPIC -pthread -MMD -MP -Iinclude -I$(CUDA_HOME)/include -c $< -o $@
$(BUILD)/runtime/peer_packet_backend.o: src/runtime/peer_packet_backend.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Xcompiler=-fPIC -Iinclude -c src/runtime/peer_packet_backend.cu -o $@
$(BUILD)/runtime/backend_registry_peer.o: src/runtime/backend_registry.cc
	mkdir -p $(@D)
	$(CXX) -O3 -std=c++17 -fPIC -MMD -MP -DUCCL_LITE_REGISTER_PEER_PACKET -Iinclude -I$(CUDA_HOME)/include -c $< -o $@
$(BUILD)/runtime/backend_registry_host.o: src/runtime/backend_registry.cc
	mkdir -p $(@D)
	$(CXX) -O3 -std=c++17 -fPIC -MMD -MP -DUCCL_LITE_REGISTER_HOST_PACKET -Iinclude -I$(CUDA_HOME)/include -c $< -o $@
$(BUILD)/runtime/backend_registry_network.o: src/runtime/backend_registry.cc
	mkdir -p $(@D)
	$(CXX) -O3 -std=c++17 -fPIC -MMD -MP -DUCCL_LITE_REGISTER_NETWORK_HIERARCHY -Iinclude -I$(CUDA_HOME)/include -c $< -o $@
$(BUILD)/runtime/backend_registry_all.o: src/runtime/backend_registry.cc
	mkdir -p $(@D)
	$(CXX) -O3 -std=c++17 -fPIC -MMD -MP -DUCCL_LITE_REGISTER_PEER_PACKET -DUCCL_LITE_REGISTER_HOST_PACKET -DUCCL_LITE_REGISTER_PEER_STAGED -DUCCL_LITE_REGISTER_PREPARED_LOCAL -DUCCL_LITE_REGISTER_NETWORK_HIERARCHY -DUCCL_LITE_REGISTER_PREPARED_DISTRIBUTED -Iinclude -I$(CUDA_HOME)/include -c $< -o $@

$(BUILD)/collectives/peer_staged.o: src/collectives/peer_staged.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Xcompiler=-fPIC -Iinclude -I$(MSCCLPP_HOME)/include -I$(MSCCLPP_IB_BUILD)/include -MMD -MP -c $< -o $@
-include $(wildcard $(BUILD)/collectives/*.d)
.PHONY: peer-graph-check
peer-graph-check: $(BUILD)/uccl_lite_peer_graph_check
.PHONY: peer-tree-check
peer-tree-check: $(BUILD)/uccl_lite_peer_tree_check
$(BUILD)/uccl_lite_peer_tree_check: tests/collectives/peer_tree_check.cu $(BUILD)/collectives/peer_staged.o
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude $(MPI_INC) $< $(BUILD)/collectives/peer_staged.o $(MPI_LIB) $(IB_LIB) -o $@
$(BUILD)/uccl_lite_peer_graph_check: tests/collectives/peer_graph_check.cu $(BUILD)/collectives/peer_staged.o
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude $(MPI_INC) $< $(BUILD)/collectives/peer_staged.o $(MPI_LIB) $(IB_LIB) -o $@

.PHONY: nccl-runtime
nccl-runtime: $(BUILD)/nccl_runtime/strict/libnccl.so.2
RUNTIME_BACKEND_OBJECTS = $(BUILD)/runtime/backend_registry_all.o $(BUILD)/runtime/peer_packet_backend.o $(BUILD)/runtime/host_packet_backend.o $(BUILD)/runtime/network_backend.o $(BUILD)/runtime/network_probe.o $(BUILD)/runtime/rank_permutation.o $(BUILD)/planner/network_plan.o $(BUILD)/collectives/peer_packet.o $(BUILD)/multinode/network_hierarchy.o $(BUILD)/collectives/host_packet_control.o $(BUILD)/collectives/host_packet_device.o
RUNTIME_BACKEND_OBJECTS += $(BUILD)/planner/network_profile.o $(BUILD)/planner/measured_network_plan.o
RUNTIME_BACKEND_OBJECTS += $(BUILD)/runtime/peer_staged_backend.o $(BUILD)/collectives/peer_staged.o
RUNTIME_BACKEND_OBJECTS += $(BUILD)/runtime/prepared_local_backend.o $(BUILD)/runtime/prepared_local.o $(BUILD)/planner/planner.o $(BUILD)/planner/peer_plan.o $(BUILD)/planner/local_plan.o $(BUILD)/planner/hardware_profile.o $(BUILD)/topology/calibration.o
RUNTIME_BACKEND_OBJECTS += $(BUILD)/planner/overlap_model.o $(BUILD)/planner/direct_plan.o
RUNTIME_BACKEND_OBJECTS += $(BUILD)/runtime/prepared_distributed_backend.o $(BUILD)/runtime/prepared_network.o $(BUILD)/runtime/network_selection.o $(BUILD)/runtime/direct_exchange_backend.o $(BUILD)/collectives/direct_exchange.o $(BUILD)/planner/network_cache.o

# Collective preparation/qualification shared by local multiprocess callers.
.PHONY: calibrate-network network-profile-report
calibrate-network: $(BUILD)/uccl_lite_calibrate_network
network-profile-report: $(BUILD)/uccl_lite_network_profile_report
$(BUILD)/uccl_lite_network_profile_report: benchmarks/probes/network_profile_report.cc $(PLANNER_STATIC)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(PLANNER_STATIC) -o $@
$(BUILD)/uccl_lite_calibrate_network: benchmarks/probes/calibrate_network.cu $(TOPOLOGY_STATIC) $(PLANNER_STATIC)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(IB_INC) $(MPI_INC) $< $(TOPOLOGY_STATIC) $(PLANNER_STATIC) $(MPI_LIB) $(IB_LIB) -ldl -o $@

.PHONY: prepared-local-runtime
.PHONY: native-nccl-fallback
native-nccl-fallback: $(BUILD)/integration/libuccl_lite_native_fallback.so
.PHONY: nccl-prepared-api nccl-prepared-api-check
.PHONY: nccl-prepared-abi
nccl-prepared-abi: $(BUILD)/integration/libuccl_lite_nccl_prepared_abi.so
.PHONY: nccl-auto-init-check
nccl-auto-init-check: $(BUILD)/uccl_lite_auto_manifest_check $(BUILD)/uccl_lite_auto_init_check nccl-prepared-abi-check
	$(BUILD)/uccl_lite_auto_manifest_check
$(BUILD)/uccl_lite_auto_manifest_check: tests/nccl_mscclpp/auto_manifest.cc $(BUILD)/integration/auto_manifest.o
	$(CXX) -O2 -std=c++17 -Iintegration/nccl_lite -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include $^ -o $@
$(BUILD)/uccl_lite_auto_init_check: tests/nccl_mscclpp/auto_init.cu $(BUILD)/integration/libuccl_lite_nccl_prepared_abi.so
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iintegration/nccl_lite -I$(NCCL_HOME)/include $(MPI_INC) $< -L$(BUILD)/integration -luccl_lite_nccl_prepared_abi -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(abspath $(BUILD)/integration) -Xlinker=-rpath=$(NCCL_HOME)/lib $(MPI_LIB) -o $@
.PHONY: nccl-prepared-abi-check
nccl-prepared-abi-check: $(BUILD)/uccl_lite_prepared_abi_check $(BUILD)/integration/libuccl_lite_delayed_native_test.so $(BUILD)/integration/prepared_abi_c_check.o
nccl-prepared-abi-check: $(BUILD)/uccl_lite_prepared_abi_bf16_check
$(BUILD)/uccl_lite_prepared_abi_bf16_check: tests/nccl_mscclpp/prepared_abi_bf16.cu $(BUILD)/integration/libuccl_lite_nccl_prepared_abi.so
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iintegration/nccl_lite -I$(NCCL_HOME)/include $(MPI_INC) $< -L$(BUILD)/integration -luccl_lite_nccl_prepared_abi -luccl_lite_nccl_prepared -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(abspath $(BUILD)/integration) -Xlinker=-rpath=$(NCCL_HOME)/lib $(MPI_LIB) -o $@
$(BUILD)/integration/prepared_abi_c_check.o: tests/nccl_mscclpp/prepared_abi_c.c integration/nccl_lite/prepared_abi.h integration/nccl_lite/prepared_api.h
	mkdir -p $(@D)
	$(CC) -std=c11 -Wall -Wextra -Werror -Iintegration/nccl_lite -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -c $< -o $@
.PHONY: native-group-reference
native-group-reference: $(BUILD)/uccl_lite_native_group_reference
$(BUILD)/uccl_lite_native_group_reference: tests/nccl_mscclpp/native_group_reference.cu
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -I$(NCCL_HOME)/include $(MPI_INC) $< $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib -o $@
$(BUILD)/integration/libuccl_lite_delayed_native_test.so: tests/nccl_mscclpp/delayed_native.cc
	mkdir -p $(@D)
	$(CXX) -O2 -std=c++17 -fPIC -shared -pthread -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include $< -Wl,--no-as-needed -L$(NCCL_HOME)/lib -lnccl -Wl,-rpath,$(NCCL_HOME)/lib -ldl -o $@
$(BUILD)/uccl_lite_prepared_abi_check: tests/nccl_mscclpp/prepared_abi.cu $(BUILD)/integration/libuccl_lite_nccl_prepared_abi.so
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iintegration/nccl_lite -I$(NCCL_HOME)/include $(MPI_INC) $< -L$(BUILD)/integration -luccl_lite_nccl_prepared_abi -luccl_lite_nccl_prepared -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(abspath $(BUILD)/integration) -Xlinker=-rpath=$(NCCL_HOME)/lib $(MPI_LIB) -o $@
$(BUILD)/integration/auto_%.o: integration/nccl_lite/auto_%.cc integration/nccl_lite/auto_init.hpp integration/nccl_lite/auto_manifest.hpp integration/nccl_lite/prepared_abi.h integration/nccl_lite/prepared_api.h
	mkdir -p $(@D)
	$(CXX) -O2 -std=c++17 -fPIC -pthread -MMD -MP -Iintegration/nccl_lite -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -c $< -o $@
$(BUILD)/integration/prepared_abi.o: integration/nccl_lite/prepared_abi.cu integration/nccl_lite/prepared_abi.h integration/nccl_lite/auto_init.hpp src/runtime/qualification.cuh
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Xcompiler=-fPIC -Iinclude -Isrc/runtime -Iintegration/nccl_lite -I$(NCCL_HOME)/include -I$(MSCCLPP_HOME)/include -I$(MSCCLPP_IB_BUILD)/include -c $< -o $@
$(BUILD)/integration/libuccl_lite_nccl_prepared_abi.so: $(BUILD)/integration/prepared_abi.o $(BUILD)/integration/auto_init.o $(BUILD)/integration/auto_manifest.o $(BUILD)/integration/native_fallback.o $(BUILD)/runtime/libuccl_lite_prepared.so $(BUILD)/integration/libuccl_lite_nccl_prepared.so integration/nccl_lite/prepared_abi.map
	$(CXX) -shared -pthread $(BUILD)/integration/prepared_abi.o $(BUILD)/integration/auto_init.o $(BUILD)/integration/auto_manifest.o $(BUILD)/integration/native_fallback.o -Wl,-z,defs -Wl,--version-script=integration/nccl_lite/prepared_abi.map -L$(BUILD)/runtime -luccl_lite_prepared -L$(BUILD)/integration -Wl,--no-as-needed -luccl_lite_nccl_prepared -L$(NCCL_HOME)/lib -lnccl -Wl,--as-needed -Wl,-rpath,'$$ORIGIN:$$ORIGIN/../runtime' -Wl,-rpath,$(NCCL_HOME)/lib -L$(MSCCLPP_IB_BUILD)/lib -lmscclpp -Wl,-rpath,$(MSCCLPP_IB_BUILD)/lib -L$(CUDA_HOME)/lib64 -lcudart -lcuda -Wl,-rpath,$(CUDA_HOME)/lib64 -ldl -o $@
nccl-prepared-api: $(BUILD)/integration/libuccl_lite_nccl_prepared.so
nccl-prepared-api-check: $(BUILD)/uccl_lite_nccl_prepared_api_check $(BUILD)/integration/prepared_api_c_check.o
$(BUILD)/integration/prepared_api.o: integration/nccl_lite/prepared_api.cc integration/nccl_lite/prepared_api.h
	mkdir -p $(@D)
	$(CXX) -O2 -std=c++17 -fPIC -pthread -MMD -MP -Iinclude -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -c $< -o $@
$(BUILD)/integration/libuccl_lite_nccl_prepared.so: $(BUILD)/integration/prepared_api.o $(BUILD)/integration/native_fallback.o $(BUILD)/runtime/libuccl_lite_prepared.so integration/nccl_lite/prepared_api.map
	$(CXX) -shared -pthread $(BUILD)/integration/prepared_api.o $(BUILD)/integration/native_fallback.o -Wl,-z,defs -Wl,--version-script=integration/nccl_lite/prepared_api.map -L$(BUILD)/runtime -luccl_lite_prepared -Wl,-rpath,'$$ORIGIN/../runtime' -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64 -ldl -o $@
$(BUILD)/integration/prepared_api_c_check.o: tests/nccl_mscclpp/prepared_api_c.c integration/nccl_lite/prepared_api.h
	mkdir -p $(@D)
	$(CC) -std=c11 -Wall -Wextra -Werror -Iintegration/nccl_lite -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -c $< -o $@
$(BUILD)/uccl_lite_nccl_prepared_api_check: tests/nccl_mscclpp/prepared_api.cu tests/collectives/scalar_oracle.cuh $(BUILD)/integration/libuccl_lite_nccl_prepared.so
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iintegration/nccl_lite -I$(NCCL_HOME)/include $(MPI_INC) $< -L$(BUILD)/integration -luccl_lite_nccl_prepared -Xlinker=-rpath=$(abspath $(BUILD)/integration) $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib -o $@
.PHONY: native-fallback-check
native-fallback-check: $(BUILD)/uccl_lite_native_fallback_check
.PHONY: prepared-scalar-check
prepared-scalar-check: $(BUILD)/uccl_lite_prepared_scalar_check $(BUILD)/nccl_runtime/scalar_rank_abi_check
.PHONY: peer-packet-mapping-check
peer-packet-mapping-check: $(BUILD)/uccl_lite_peer_packet_mapping_check
$(BUILD)/uccl_lite_peer_packet_mapping_check: tests/collectives/peer_packet_mapping_check.cu tests/collectives/scalar_oracle.cuh $(BUILD)/runtime/libuccl_lite_prepared_local.a
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude $(MPI_INC) $< $(BUILD)/runtime/libuccl_lite_prepared_local.a $(MPI_LIB) $(IB_LIB) -lcuda -Xcompiler=-pthread -ldl -o $@
$(BUILD)/uccl_lite_prepared_scalar_check: tests/collectives/prepared_scalar_check.cu tests/collectives/scalar_oracle.cuh $(BUILD)/runtime/libuccl_lite_prepared_local.a $(BUILD)/integration/native_fallback.o
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude -Iintegration/nccl_lite -I$(NCCL_HOME)/include $(MPI_INC) $< $(BUILD)/runtime/libuccl_lite_prepared_local.a $(BUILD)/integration/native_fallback.o $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib $(IB_LIB) -lcuda -Xcompiler=-pthread -ldl -o $@
$(BUILD)/nccl_runtime/scalar_rank_abi_check: tests/nccl_mscclpp/scalar_rank_abi.cu tests/collectives/scalar_oracle.cuh
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MPI_INC) -I$(NCCL_HOME)/include $< $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib -o $@
$(BUILD)/uccl_lite_native_fallback_check: tests/collectives/native_fallback_check.cu tests/collectives/prepared_local_oracle.cuh $(BUILD)/runtime/libuccl_lite_prepared_local.a $(BUILD)/integration/native_fallback.o
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude -Iintegration/nccl_lite -I$(NCCL_HOME)/include $(MPI_INC) $< $(BUILD)/runtime/libuccl_lite_prepared_local.a $(BUILD)/integration/native_fallback.o $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib $(IB_LIB) -lcuda -Xcompiler=-pthread -ldl -o $@
$(BUILD)/integration/native_fallback.o: integration/nccl_lite/native_fallback.cc integration/nccl_lite/native_fallback.hpp
	mkdir -p $(@D)
	$(CXX) -O2 -std=c++17 -fPIC -MMD -MP -Iinclude -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -c $< -o $@
$(BUILD)/integration/libuccl_lite_native_fallback.so: $(BUILD)/integration/native_fallback.o
	$(CXX) -shared $< -Wl,-z,defs -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64 -ldl -o $@
-include $(wildcard $(BUILD)/integration/*.d)
prepared-local-runtime: $(BUILD)/runtime/libuccl_lite_prepared_local.a
.PHONY: prepared-network-runtime
prepared-network-runtime: $(BUILD)/runtime/libuccl_lite_prepared_network.a $(BUILD)/runtime/libuccl_lite_prepared_network.so
.PHONY: prepared-network-check
prepared-network-check: $(BUILD)/uccl_lite_prepared_network_check
.PHONY: automatic-network-check
automatic-network-check: $(BUILD)/uccl_lite_automatic_network_check
.PHONY: network-cache-runtime-check
network-cache-runtime-check: $(BUILD)/uccl_lite_network_cache_runtime_check
$(BUILD)/uccl_lite_network_cache_runtime_check: tests/collectives/network_cache_runtime_check.cu tests/collectives/scalar_oracle.cuh $(BUILD)/runtime/libuccl_lite_prepared_network.a $(BUILD)/integration/native_fallback.o
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude -Iintegration/nccl_lite -I$(NCCL_HOME)/include $(MPI_INC) $< $(BUILD)/runtime/libuccl_lite_prepared_network.a $(BUILD)/integration/native_fallback.o $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib $(IB_LIB) -lcuda -Xcompiler=-pthread -ldl -o $@
$(BUILD)/uccl_lite_automatic_network_check: tests/collectives/automatic_network_check.cu tests/collectives/scalar_oracle.cuh $(BUILD)/runtime/libuccl_lite_prepared_network.a $(BUILD)/integration/native_fallback.o
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude -Iintegration/nccl_lite -I$(NCCL_HOME)/include $(MPI_INC) $< $(BUILD)/runtime/libuccl_lite_prepared_network.a $(BUILD)/integration/native_fallback.o $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib $(IB_LIB) -lcuda -Xcompiler=-pthread -ldl -o $@
$(BUILD)/uccl_lite_prepared_network_check: tests/collectives/prepared_network_check.cu tests/collectives/scalar_oracle.cuh $(BUILD)/runtime/libuccl_lite_prepared_network.a $(BUILD)/integration/native_fallback.o
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude -Iintegration/nccl_lite -I$(NCCL_HOME)/include $(MPI_INC) $< $(BUILD)/runtime/libuccl_lite_prepared_network.a $(BUILD)/integration/native_fallback.o $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib $(IB_LIB) -lcuda -Xcompiler=-pthread -ldl -o $@
PREPARED_NETWORK_OBJECTS = $(BUILD)/runtime/network_selection.o $(BUILD)/runtime/direct_exchange_backend.o $(BUILD)/collectives/direct_exchange.o $(BUILD)/runtime/prepared_network.o $(BUILD)/runtime/network_backend.o $(BUILD)/runtime/network_probe.o $(BUILD)/runtime/rank_permutation.o $(BUILD)/multinode/network_hierarchy.o $(PLANNER_OBJECTS) $(TOPOLOGY_OBJECTS)
$(BUILD)/runtime/prepared_network.o: src/runtime/prepared_network.cu src/runtime/qualification.cuh src/runtime/network_preflight.hpp
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Xcompiler=-fPIC -Iinclude -I$(MSCCLPP_HOME)/include -I$(MSCCLPP_IB_BUILD)/include -c $< -o $@
$(BUILD)/runtime/libuccl_lite_prepared_network.a: $(PREPARED_NETWORK_OBJECTS) Makefile
	@if test -e $@.tmp; then rm $@.tmp; fi
	$(AR) rcs $@.tmp $(PREPARED_NETWORK_OBJECTS)
	mv $@.tmp $@
$(BUILD)/runtime/libuccl_lite_prepared_network.so: $(PREPARED_NETWORK_OBJECTS) Makefile
	$(CXX) -shared -pthread $(PREPARED_NETWORK_OBJECTS) -Wl,-z,defs -L$(MSCCLPP_IB_BUILD)/lib -lmscclpp -Wl,-rpath,$(MSCCLPP_IB_BUILD)/lib -L$(CUDA_HOME)/lib64 -lcudart -lcuda -ldl -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@
PREPARED_LOCAL_OBJECTS = $(BUILD)/runtime/prepared_local.o $(BUILD)/runtime/peer_staged_backend.o $(BUILD)/collectives/peer_staged.o $(BUILD)/runtime/host_packet_backend.o $(BUILD)/collectives/host_packet_control.o $(BUILD)/collectives/host_packet_device.o $(BUILD)/cpu_router/cpu_executor.o $(PLANNER_OBJECTS) $(TOPOLOGY_OBJECTS) $(BUILD)/topology/calibration.o
PREPARED_LOCAL_OBJECTS += $(BUILD)/runtime/peer_packet_backend.o $(BUILD)/collectives/peer_packet.o

.PHONY: prepared-runtime
prepared-runtime: $(BUILD)/runtime/libuccl_lite_prepared.a $(BUILD)/runtime/libuccl_lite_prepared.so
.PHONY: prepared-runtime-check
prepared-runtime-check: $(BUILD)/uccl_lite_prepared_runtime_check
prepared-runtime-check: $(BUILD)/uccl_lite_prepared_runtime_shared_check
$(BUILD)/uccl_lite_prepared_runtime_shared_check: tests/collectives/prepared_runtime_check.cu tests/collectives/scalar_oracle.cuh $(BUILD)/runtime/libuccl_lite_prepared.so $(BUILD)/integration/native_fallback.o
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude -Iintegration/nccl_lite -I$(NCCL_HOME)/include $(MPI_INC) $< $(BUILD)/integration/native_fallback.o -L$(BUILD)/runtime -luccl_lite_prepared -Xlinker=-rpath=$(abspath $(BUILD)/runtime) $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl -o $@
$(BUILD)/uccl_lite_prepared_runtime_check: tests/collectives/prepared_runtime_check.cu tests/collectives/scalar_oracle.cuh $(BUILD)/runtime/libuccl_lite_prepared.a $(BUILD)/integration/native_fallback.o
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude -Iintegration/nccl_lite -I$(NCCL_HOME)/include $(MPI_INC) $< $(BUILD)/runtime/libuccl_lite_prepared.a $(BUILD)/integration/native_fallback.o $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib $(IB_LIB) -lcuda -Xcompiler=-pthread -ldl -o $@
PREPARED_OBJECTS = $(sort $(PREPARED_LOCAL_OBJECTS) $(PREPARED_NETWORK_OBJECTS) $(BUILD)/runtime/prepared.o)
$(BUILD)/runtime/prepared.o: src/runtime/prepared.cc
	mkdir -p $(@D)
	$(CXX) -O2 -std=c++17 -fPIC -pthread -MMD -MP -Iinclude -I$(CUDA_HOME)/include -I$(MSCCLPP_HOME)/include -I$(MSCCLPP_IB_BUILD)/include -c $< -o $@
$(BUILD)/runtime/libuccl_lite_prepared.a: $(PREPARED_OBJECTS) Makefile
	@if test -e $@.tmp; then rm $@.tmp; fi
	$(AR) rcs $@.tmp $(PREPARED_OBJECTS)
	mv $@.tmp $@
$(BUILD)/runtime/libuccl_lite_prepared.so: $(PREPARED_OBJECTS) Makefile
	$(CXX) -shared -pthread $(PREPARED_OBJECTS) -Wl,-z,defs -L$(MSCCLPP_IB_BUILD)/lib -lmscclpp -Wl,-rpath,$(MSCCLPP_IB_BUILD)/lib -L$(CUDA_HOME)/lib64 -lcudart -lcuda -ldl -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@

$(BUILD)/runtime/prepared_local.o: src/runtime/prepared_local.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Xcompiler=-fPIC -Iinclude -c $< -o $@
$(BUILD)/runtime/libuccl_lite_prepared_local.a: $(PREPARED_LOCAL_OBJECTS) Makefile
	rm -f $@.tmp
	$(AR) rcs $@.tmp $(PREPARED_LOCAL_OBJECTS)
	mv $@.tmp $@

.PHONY: prepared-graph-check
prepared-graph-check: $(BUILD)/uccl_lite_prepared_graph_check
$(BUILD)/uccl_lite_prepared_graph_check: tests/collectives/prepared_graph_check.cu tests/collectives/scalar_oracle.cuh $(BUILD)/runtime/libuccl_lite_prepared_local.a $(BUILD)/integration/native_fallback.o
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude -Iintegration/nccl_lite -I$(NCCL_HOME)/include $(MPI_INC) $< $(BUILD)/runtime/libuccl_lite_prepared_local.a $(BUILD)/integration/native_fallback.o $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib $(IB_LIB) -lcuda -Xcompiler=-pthread -ldl -o $@
.PHONY: prepared-local-check
prepared-local-check: $(BUILD)/uccl_lite_prepared_local_check $(BUILD)/uccl_lite_local_plan_check
$(BUILD)/uccl_lite_prepared_local_check: tests/collectives/prepared_local_check.cu $(BUILD)/runtime/libuccl_lite_prepared_local.a
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude $(MPI_INC) $< $(BUILD)/runtime/libuccl_lite_prepared_local.a $(MPI_LIB) -lcuda -Xcompiler=-pthread -ldl -L$(MSCCLPP_IB_BUILD)/lib -lmscclpp -Xlinker=-rpath=$(MSCCLPP_IB_BUILD)/lib -o $@
$(BUILD)/nccl_runtime/libuccl_lite_nccl_runtime.so: $(BUILD)/nccl_runtime/frontend.o $(RUNTIME_BACKEND_OBJECTS) $(ROUTER_STATIC) integration/nccl_lite/mscclpp_exports.version
	$(CXX) -shared -pthread $(BUILD)/nccl_runtime/frontend.o $(RUNTIME_BACKEND_OBJECTS) $(ROUTER_STATIC) -Wl,--version-script=integration/nccl_lite/mscclpp_exports.version -Wl,-soname,libnccl.so.2 -Wl,-z,defs -L$(MSCCLPP_IB_BUILD)/lib -lmscclpp -Wl,-rpath,$(MSCCLPP_IB_BUILD)/lib -L$(CUDA_HOME)/lib64 -lcudart -lcuda -ldl -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@
$(BUILD)/nccl_runtime/strict/libnccl.so.2: $(BUILD)/nccl_runtime/libuccl_lite_nccl_runtime.so
	mkdir -p $(@D)
	ln -sfn ../libuccl_lite_nccl_runtime.so $@
-include $(wildcard $(BUILD)/nccl_runtime/*.d)

.PHONY: runtime-registry-check
runtime-registry-check: $(BUILD)/uccl_lite_backend_spec_check
$(BUILD)/uccl_lite_backend_spec_check: tests/nccl_mscclpp/backend_spec_check.cc src/runtime/backend_registry.cc include/uccl_lite/runtime/collective_backend.hpp
	$(CXX) -O2 -std=c++17 -Iinclude -I$(CUDA_HOME)/include tests/nccl_mscclpp/backend_spec_check.cc src/runtime/backend_registry.cc -o $@

$(BUILD)/runtime/rank_permutation.o: src/runtime/rank_permutation.cu src/runtime/rank_permutation.hpp
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Xcompiler=-fPIC -Iinclude -c $< -o $@

.PHONY: rank-permutation-check
rank-permutation-check: $(BUILD)/uccl_lite_rank_permutation_check
$(BUILD)/uccl_lite_rank_permutation_check: tests/collectives/rank_permutation_check.cu $(BUILD)/runtime/rank_permutation.o
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Isrc/runtime $^ -o $@

$(BUILD)/runtime/network_backend.o $(BUILD)/runtime/network_probe.o $(BUILD)/runtime/network_selection.o: $(BUILD)/runtime/%.o: src/runtime/%.cc
	mkdir -p $(@D)
	$(CXX) -O3 -std=c++17 -fPIC -pthread -MMD -MP -Iinclude -I$(CUDA_HOME)/include -I$(MSCCLPP_HOME)/include -I$(MSCCLPP_IB_BUILD)/include -c $< -o $@

# Standalone peer/runtime package: no CPU-router, NCCL or MSCCL++ link dependency.
PEER_RUNTIME_OBJECTS = $(CORE_CALL_OBJECT) $(BUILD)/peer_collectives.o $(PLANNER_OBJECTS) $(TOPOLOGY_OBJECTS) $(BUILD)/runtime/peer_calibration.o $(BUILD)/topology/calibration.o
.PHONY: peer-runtime peer-runtime-flags
peer-runtime: $(BUILD)/runtime/libuccl_lite_peer.a $(BUILD)/runtime/libuccl_lite_peer.so
peer-runtime-flags:
$(BUILD)/runtime/peer.flags: peer-runtime-flags
	mkdir -p $(@D)
	@printf '%s\n' '$(NVCC) $(NVCC_FLAGS) $(CUDA_HOME) -fPIC' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp
$(BUILD)/peer_collectives.o $(BUILD)/runtime/peer_calibration.o $(BUILD)/topology/calibration.o: $(BUILD)/runtime/peer.flags
$(BUILD)/runtime/libuccl_lite_peer.a: $(PEER_RUNTIME_OBJECTS) Makefile
	rm -f $@.tmp
	$(AR) rcs $@.tmp $(PEER_RUNTIME_OBJECTS)
	mv $@.tmp $@
$(BUILD)/runtime/libuccl_lite_peer.so: $(PEER_RUNTIME_OBJECTS) Makefile
	$(CXX) -shared -pthread $(PEER_RUNTIME_OBJECTS) -Wl,-z,defs -L$(CUDA_HOME)/lib64 -lcudart -ldl -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@

.PHONY: peer-runtime-check
peer-runtime-check: $(BUILD)/uccl_lite_peer_runtime_check $(BUILD)/uccl_lite_peer_runtime_static_check
$(BUILD)/uccl_lite_peer_runtime_check: tests/collectives/peer_runtime_check.cu $(BUILD)/runtime/libuccl_lite_peer.so
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude $< -L$(BUILD)/runtime -luccl_lite_peer -Xlinker=-rpath=$(abspath $(BUILD)/runtime) -o $@
$(BUILD)/uccl_lite_peer_runtime_static_check: tests/collectives/peer_runtime_check.cu $(BUILD)/runtime/libuccl_lite_peer.a
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude $< $(BUILD)/runtime/libuccl_lite_peer.a -Xcompiler=-pthread -ldl -o $@

$(BUILD)/nccl_runtime/host_rank_abi_check: tests/nccl_mscclpp/host_rank_abi.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MPI_INC) -I$(NCCL_HOME)/include $< -o $@ $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib

$(BUILD)/nccl_runtime/network_profile_fixture: tests/nccl_mscclpp/network_profile_fixture.cc $(PLANNER_STATIC)
	mkdir -p $(@D)
	$(CXX) -O2 -std=c++17 -Iinclude $< $(PLANNER_STATIC) -o $@

$(BUILD)/nccl_runtime/network_prepare_rejection: tests/nccl_mscclpp/network_prepare_rejection.cc
	mkdir -p $(@D)
	$(MPICXX) -O2 -std=c++17 -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include $< -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib -lnccl -Wl,-rpath,$(NCCL_HOME)/lib -o $@

$(BUILD)/uccl_lite_network_prepare_check: tests/collectives/network_prepare_check.cu $(BUILD)/runtime/network_backend.o $(BUILD)/runtime/network_probe.o $(BUILD)/runtime/rank_permutation.o $(BUILD)/multinode/network_hierarchy.o $(PLANNER_STATIC) $(TOPOLOGY_STATIC)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude $(MPI_INC) $< $(BUILD)/runtime/network_backend.o $(BUILD)/runtime/network_probe.o $(BUILD)/runtime/rank_permutation.o $(BUILD)/multinode/network_hierarchy.o $(PLANNER_STATIC) $(TOPOLOGY_STATIC) $(MPI_LIB) $(IB_LIB) -lcuda -ldl -Xcompiler=-pthread -o $@

.PHONY: router-worker-calibrate
router-worker-calibrate: $(BUILD)/uccl_lite_router_worker_calibrate
$(BUILD)/uccl_lite_router_worker_calibrate: benchmarks/cpu_router/worker_plans.cu benchmarks/joint/joint_bench.cu $(JOINT_OBJECTS) $(ROUTER_STATIC)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(JOINT_OBJECTS) $(ROUTER_STATIC) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl -o $@

.PHONY: network-candidates-check
network-candidates-check: $(BUILD)/uccl_lite_network_selection_check $(BUILD)/uccl_lite_network_candidates_check
	$(BUILD)/uccl_lite_network_selection_check
$(BUILD)/uccl_lite_network_selection_check: tests/planner/network_selection_check.cc $(BUILD)/runtime/network_selection.o
	$(CXX) -O2 -std=c++17 -Iinclude -I$(CUDA_HOME)/include $^ -o $@
$(BUILD)/uccl_lite_network_candidates_check: tests/collectives/network_candidates_check.cu $(BUILD)/runtime/libuccl_lite_prepared_network.a $(BUILD)/integration/native_fallback.o
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(IB_INC) $(MPI_INC) -Iintegration/nccl_lite -I$(NCCL_HOME)/include $< $(BUILD)/runtime/libuccl_lite_prepared_network.a $(BUILD)/integration/native_fallback.o $(IB_LIB) $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib -ldl -lcuda -Xcompiler=-pthread -o $@

network-candidates-check: $(BUILD)/uccl_lite_network_pool_demotion_check
$(BUILD)/uccl_lite_network_pool_demotion_check: tests/collectives/network_pool_demotion_check.cu $(BUILD)/runtime/libuccl_lite_prepared.so $(BUILD)/integration/native_fallback.o
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude -Iintegration/nccl_lite -I$(NCCL_HOME)/include $(MPI_INC) $< $(BUILD)/integration/native_fallback.o -L$(BUILD)/runtime -luccl_lite_prepared -Xlinker=-rpath=$(abspath $(BUILD)/runtime) $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib -ldl -Xcompiler=-pthread -o $@

$(BUILD)/runtime/prepared_distributed_backend.o: src/runtime/prepared_distributed_backend.cu src/runtime/qualification.cuh
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Xcompiler=-fPIC -Iinclude -Isrc/runtime -I$(MSCCLPP_HOME)/include -I$(MSCCLPP_IB_BUILD)/include -c $< -o $@

.PHONY: strict-distributed-check
strict-distributed-check: $(BUILD)/nccl_runtime/distributed_rank_abi_check
$(BUILD)/nccl_runtime/distributed_rank_abi_check: tests/nccl_mscclpp/distributed_rank_abi.cu $(BUILD)/nccl_runtime/strict/libnccl.so.2
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude -I$(NCCL_HOME)/include $(MPI_INC) $< $(MPI_LIB) $(BUILD)/nccl_runtime/libuccl_lite_nccl_runtime.so -Xlinker=-rpath=$(abspath $(BUILD)/nccl_runtime/strict) -o $@

.PHONY: direct-cache-check
direct-cache-check: $(BUILD)/uccl_lite_direct_cache_runtime_check $(BUILD)/uccl_lite_network_cache_check
	$(BUILD)/uccl_lite_network_cache_check
$(BUILD)/uccl_lite_direct_cache_runtime_check: tests/collectives/direct_cache_runtime_check.cu $(BUILD)/runtime/libuccl_lite_prepared_network.a $(BUILD)/integration/native_fallback.o
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(IB_INC) $(MPI_INC) -Iintegration/nccl_lite -I$(NCCL_HOME)/include $< $(BUILD)/runtime/libuccl_lite_prepared_network.a $(BUILD)/integration/native_fallback.o $(IB_LIB) $(MPI_LIB) -L$(NCCL_HOME)/lib -lnccl -Xlinker=-rpath=$(NCCL_HOME)/lib -ldl -lcuda -Xcompiler=-pthread -o $@

.PHONY: direct-calibration-check
direct-calibration-check: $(BUILD)/uccl_lite_direct_calibration_check
$(BUILD)/uccl_lite_direct_calibration_check: tests/collectives/direct_calibration_check.cu src/runtime/qualification.cuh $(BUILD)/collectives/libuccl_lite_direct_exchange.a $(PLANNER_STATIC) $(TOPOLOGY_STATIC)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(IB_INC) $(MPI_INC) -Isrc/runtime $< $(BUILD)/collectives/libuccl_lite_direct_exchange.a $(PLANNER_STATIC) $(TOPOLOGY_STATIC) $(IB_LIB) $(MPI_LIB) -ldl -Xcompiler=-pthread -o $@
