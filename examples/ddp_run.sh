#!/bin/bash
# Unified DDP training script for NVIDIA and AMD GPUs
#
# Usage:
#   ./ddp_run.sh                    # Single node with all GPUs
#   ./ddp_run.sh 4                  # Single node with 4 GPUs
#   ./ddp_run.sh 8 2 0 10.0.0.1     # Multi-node: 8 GPUs/node, 2 nodes, rank 0

set -e

NGPUS=${1:-$(python3 -c "import torch; print(torch.cuda.device_count())")}
NNODES=${2:-1}
NODE_RANK=${3:-0}
MASTER_ADDR=${4:-"localhost"}
MASTER_PORT=${5:-12355}

# Detect platform and set UCCL plugin
if python3 -c "import torch; exit(0 if hasattr(torch.version, 'hip') and torch.version.hip else 1)" 2>/dev/null; then
    echo "Detected AMD ROCm platform"
    export NCCL_NET_PLUGIN=$(python3 -c "import uccl; print(uccl.rccl_plugin_path())")
else
    echo "Detected NVIDIA CUDA platform"
    export NCCL_NET_PLUGIN=$(python3 -c "import uccl; print(uccl.nccl_plugin_path())")
fi

echo "Using UCCL plugin: $NCCL_NET_PLUGIN"
echo "Configuration: ${NGPUS} GPUs/node, ${NNODES} nodes, rank ${NODE_RANK}"

# Run training
if [ "$NNODES" -eq 1 ]; then
    torchrun --standalone --nproc_per_node=$NGPUS ddp_train.py "$@"
else
    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NGPUS \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        ddp_train.py "${@:6}"
fi
