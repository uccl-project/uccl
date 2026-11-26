include src/multinode/network_hierarchy_flags.mk

# NCCL ABI policies. graph512 passed the recorded two-host/four-L4-per-host
# FP32 1-KiB–1-GiB qualification; the exact runtime profile is still required.
# Historical EXTRA_NVCC_FLAGS commands use UCCL_LITE_NCCL_POLICY=base.
UCCL_LITE_NCCL_POLICY ?= graph512
UCCL_LITE_NCCL_POLICY_FLAGS :=
UCCL_LITE_NCCL_POLICY_LIBS :=
UCCL_LITE_NCCL_POLICY_BASE := $(UCCL_LITE_NCCL_POLICY)
ifeq ($(UCCL_LITE_NCCL_POLICY),graph512)
UCCL_LITE_NCCL_POLICY_BASE := medium24
endif

ifeq ($(UCCL_LITE_NCCL_POLICY_BASE),medium24)
UCCL_LITE_NCCL_POLICY_FLAGS := $(NETWORK_HIERARCHY_BASE_FLAGS)
UCCL_LITE_NCCL_POLICY_LIBS := -lcuda
ifeq ($(UCCL_LITE_NCCL_POLICY),graph512)
UCCL_LITE_NCCL_POLICY_FLAGS += -DUCCL_LITE_RS16M_GRAPH_BLOCKS=512
endif
else ifneq ($(UCCL_LITE_NCCL_POLICY),base)
$(error Unsupported UCCL_LITE_NCCL_POLICY '$(UCCL_LITE_NCCL_POLICY)'; choose base, medium24 or graph512)
endif
