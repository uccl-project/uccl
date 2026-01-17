
#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DOCKER_IMAGE="docker.io/rocm/primus:v25.11"
# ---------------------------------------------------------------------------
# Training Config
# ---------------------------------------------------------------------------
export MODEL_NAME=deepseek_v2_lite
export EP=8

# ---------------------------------------------------------------------------
# Cluster Config
# ---------------------------------------------------------------------------
export NNODES=1
export NODE_LISTS=

bash "${SCRIPT_DIR}"/primus_run_pretrain_cli.sh --moe_token_dispatcher_type "flex" \
												--moe_enable_deepep "True" \
												--moe_shared_expert_overlap "False" \
												--moe_router_dtype "fp32" 
