
#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_NAME=deepseek_v3
export EP=8
export PP=1
export NNODES=1

bash "${SCRIPT_DIR}"/run_pretrain_cli.sh --moe_token_dispatcher_type "flex" \
										 --moe_enable_deepep "True" \
										 --moe_shared_expert_overlap "False" \
										 --moe_router_dtype "fp32" 
