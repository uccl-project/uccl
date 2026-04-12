
#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE=docker.io/rocm/primus:v25.9_gfx942
# ---------------------------------------------------------------------------
# Training Config
# ---------------------------------------------------------------------------
export MODEL_NAME=deepseek_v2_lite
export EP=16

# ---------------------------------------------------------------------------
# Cluster Config
# ---------------------------------------------------------------------------
export NNODES=2
export NODE_LISTS=

# rebuild bnxt user library if you want to train with broadcom nic
export USING_BNXT=${USING_BNXT:-0}
export PATH_TO_BNXT_TAR_PACKAGE=${PATH_TO_BNXT_TAR_PACKAGE:-}
######################## NCCL/UCCL Settings ###############################
export NCCL_IB_HCA=${NCCL_IB_HCA:-^mlx5_1,mlx5_6}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}

export USE_UCCL=1
export UCCL_REF=main

bash "${SCRIPT_DIR}"/primus_slurm_pretrain_cli.sh --moe_token_dispatcher_type="flex" \
												  --moe_enable_deepep="True" \
												  --moe_shared_expert_overlap="False" \
												  --moe_router_dtype="fp32"