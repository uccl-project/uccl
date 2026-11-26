include src/collectives/peer_packet_flags.mk

# Local NCCL ABI configurations. four-l4-fp32 passed the recorded four-L4
# FP32 1-KiB–1-GiB gates with core2 placement; exact runtime identity is required.
# Runtime topology/software/shape verification remains mandatory.
UCCL_LITE_NCCL_LOCAL_POLICY ?= four-l4-fp32
UCCL_LITE_NCCL_LOCAL_FLAGS :=
UCCL_LITE_NCCL_LOCAL_LIBS :=
UCCL_LITE_NCCL_LOCAL_DEPS :=
UCCL_LITE_NCCL_LOCAL_REPRO_FLAGS :=
ifeq ($(UCCL_LITE_NCCL_LOCAL_POLICY),four-l4-fp32)
UCCL_LITE_NCCL_LOCAL_REPRO_FLAGS := --frandom-seed=uccl_lite_local_strict --keep --keep-dir $(BUILD)/nccl_local/cuda-intermediates
UCCL_LITE_NCCL_LOCAL_FLAGS := $(PEER_PACKET_RETAINED_FLAGS)
UCCL_LITE_NCCL_LOCAL_LIBS := -lcuda $(ROUTER_STATIC)
UCCL_LITE_NCCL_LOCAL_DEPS := $(ROUTER_STATIC)
else ifneq ($(UCCL_LITE_NCCL_LOCAL_POLICY),base)
$(error Unsupported UCCL_LITE_NCCL_LOCAL_POLICY '$(UCCL_LITE_NCCL_LOCAL_POLICY)'; choose base or four-l4-fp32)
endif
