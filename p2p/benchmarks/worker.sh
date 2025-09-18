#!/bin/bash
set -euo pipefail

cd /home/yuankach@amd.com/yang/shuangma/uccl/p2p/benchmarks
source /home/yuankach@amd.com/anaconda3/bin/activate shuangma_env
export PATH=/home/yuankach@amd.com/anaconda3/envs/shuangma_env/bin:$PATH

# NCCL envs
export NCCL_IB_QPS_PER_CONNECTION=16
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_P2P_NET_CHUNKSIZE=6024288
export NCCL_BUFFSIZE=256388608
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export UCCL_ENTROPY=16
export UCCL_CHUNK_SIZE_KB=16
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
BLOCK_SIZES=${BLOCK_SIZES:-"1 4 16 64 256 1024 4096 16384 65536 264114"}

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
