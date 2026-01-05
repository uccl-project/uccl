#!/bin/bash
set -euo pipefail

#!/bin/bash

#!/bin/bash

WORK_DIR=$UCCL_HOME/p2p/benchmarks
CONDA_PATH="${CONDA_PATH:-conda}"
ENV_NAME="${ENV_NAME:-shuangma_env}"

# Validate environment variables
if [ -z "$WORK_DIR" ] || [ -z "$CONDA_PATH" ] || [ -z "$ENV_NAME" ]; then
    echo "Error: Please set the following environment variables:"
    echo "  WORK_DIR, CONDA_PATH, ENV_NAME"
    exit 1
fi

# Check if directory exists
if [ ! -d "$WORK_DIR" ]; then
    echo "Error: Directory does not exist: $WORK_DIR"
    exit 1
fi

# Change to working directory
cd "$WORK_DIR" || { echo "Error: Cannot change to directory $WORK_DIR"; exit 1; }

# Check if conda command is available
if ! command -v "$CONDA_PATH" &> /dev/null; then
    echo "Error: Conda command not found: $CONDA_PATH"
    exit 1
fi

# Activate conda environment using conda activate command
eval "$($CONDA_PATH shell.bash hook)" 2>/dev/null || {
    echo "Error: Cannot initialize conda shell hook"
    exit 1
}

conda activate "$ENV_NAME" || { 
    echo "Error: Cannot activate conda environment: $ENV_NAME"
    echo "Available environments:"
    "$CONDA_PATH" env list || echo "Cannot list conda environments"
    exit 1
}

echo "Successfully activated environment: $ENV_NAME"
echo "Current directory: $(pwd)"
echo "Python path: $(which python)"

export PATH=/home/yuankach@amd.com/anaconda3/envs/shuangma_env/bin:$PATH

# NCCL envs
export NCCL_IB_QPS_PER_CONNECTION=16
export NCCL_NCHANNELS_PER_NET_PEER=16
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_P2P_NET_CHUNKSIZE=131072
export NCCL_BUFFSIZE=1048576
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_PCI_RELAXED_ORDERING=1

# Parameters
MASTER_ADDR=${MASTER_ADDR:?}
MASTER_PORT=${MASTER_PORT:-19999}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:?}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

NUM_QO_HEADS=${NUM_QO_HEADS:-32}
GQA_GROUP_SIZE=${GQA_GROUP_SIZE:-4}
HEAD_DIM=${HEAD_DIM:-128}
NUM_ITERS=${NUM_ITERS:-100}
BENCHMARK_SCRIPT=${BENCHMARK_SCRIPT:-"benchmark_nccl_alltoall.py"}
# BLOCK_SIZES=${BLOCK_SIZES:-"1 4 16 64 256 1024 4096 16384 65536 264114"}
BLOCK_SIZES=${BLOCK_SIZES:-"256 1024 2048 4096 8192 16384"}
# 1M 4M 8M 16M 32M
echo "[$(hostname)] Starting worker rank=$NODE_RANK / $((NNODES-1))"

if [ "$NNODES" -eq 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
        $BENCHMARK_SCRIPT \
        --block-sizes $BLOCK_SIZES \
        --num-qo-heads $NUM_QO_HEADS \
        --gqa-group-size $GQA_GROUP_SIZE \
        --head-dim $HEAD_DIM \
        --num-iters $NUM_ITERS
else
    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $BENCHMARK_SCRIPT \
        --block-sizes $BLOCK_SIZES \
        --num-qo-heads $NUM_QO_HEADS \
        --gqa-group-size $GQA_GROUP_SIZE \
        --head-dim $HEAD_DIM \
        --num-iters $NUM_ITERS
fi
