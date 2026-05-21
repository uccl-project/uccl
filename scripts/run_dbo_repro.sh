#!/bin/bash
# Two-node minimal-repro launcher for the DBO "dispatch twice on buffer" issue.
# Uses the ziming uv env at /home/ubuntu/efs/zm/uccl/ziming.
# Usage: ./run_dbo_repro.sh <node_rank>      # 0 = master(local), 1 = remote(p5en_1)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Activate ziming uv env
source "$REPO_DIR/ziming/bin/activate"

# ---- LD_LIBRARY_PATH (EFA + cuda)
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/amazon/ofi-nccl/lib:${LD_LIBRARY_PATH:-}

# NCCL / UCCL bootstrap
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=enp71s0
export UCCL_SOCKET_IFNAME=enp71s0
export NCCL_IB_GID_INDEX=0
export UCCL_IB_GID_INDEX=0

# EFA
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1

# Cluster
NODE_RANK="${1:?node_rank required (0=local, 1=remote)}"
NNODES=2
NPROC_PER_NODE=8
MASTER_ADDR="172.31.73.10"   # local node's enp71s0 IP
MASTER_PORT="12355"

cd "$REPO_DIR/ep"

echo ">>> torchrun node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT"
exec torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    bench/test_dbo_dispatch_twice.py \
    --num-tokens 4096 --hidden 7168 --num-experts 256 --num-topk 8 \
    --iters 1 --moe-layers 48 --all-empty-b --only-empty-pattern --fp8
