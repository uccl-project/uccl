#!/bin/bash

# usage:
# NCCL_IB_GID_INDEX=3 MASTER_ADDR="10.2.96.68" NNODES=2 NODE_RANK=0 NPROC_PER_NODE=1 bash run_benchmark_nccl_alltoall.sh
# NCCL_IB_GID_INDEX=3 MASTER_ADDR="10.2.96.68" NNODES=2 NODE_RANK=1 NPROC_PER_NODE=1 bash run_benchmark_nccl_alltoall.sh
# Set environment variables for optimal performance
export OMP_NUM_THREADS=1
export NCCL_MAX_NCHANNELS=32
export NCCL_MIN_NCHANNELS=32

# Configuration parameters (can be overridden by environment variables)
MASTER_ADDR=${MASTER_ADDR:-"10.2.96.68"}  # Master node address, defaults to localhost
MASTER_PORT=${MASTER_PORT:-"29500"}      # Master node port, defaults to 29500
NNODES=${NNODES:-1}                      # Number of nodes, defaults to 1 (single node)
NODE_RANK=${NODE_RANK:-0}                # Current node rank, defaults to 0
NPROC_PER_NODE=${NPROC_PER_NODE:-8}      # Processes per node, defaults to 8

# Block sizes to test (can be overridden by environment variable)
BLOCK_SIZES=${BLOCK_SIZES:-"256 1024 4096 16384 65536 264114"} # 264114

# Script parameters with defaults
NUM_QO_HEADS=${NUM_QO_HEADS:-32}
GQA_GROUP_SIZE=${GQA_GROUP_SIZE:-4}
HEAD_DIM=${HEAD_DIM:-128}
NUM_ITERS=${NUM_ITERS:-100}

echo "================================================"
echo "NCCL AlltoAll Benchmark Configuration"
echo "================================================"
echo "Master Address:   $MASTER_ADDR"
echo "Master Port:      $MASTER_PORT"
echo "Number of Nodes:  $NNODES"
echo "Node Rank:        $NODE_RANK"
echo "GPUs per Node:    $NPROC_PER_NODE"
echo "Block Sizes:      $BLOCK_SIZES"
echo "QO Heads:         $NUM_QO_HEADS"
echo "GQA Group Size:   $GQA_GROUP_SIZE"
echo "Head Dimension:   $HEAD_DIM"
echo "Iterations:       $NUM_ITERS"
echo "================================================"

# Check if torchrun command exists
if ! command -v torchrun &> /dev/null; then
    echo "Error: torchrun command not found. Please ensure PyTorch is properly installed."
    exit 1
fi

# Check if CUDA is available (optional check)
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "False"; then
    echo "Warning: CUDA is not available. Performance may be degraded."
fi

# Main benchmark loop
for block_size in $BLOCK_SIZES
do
    echo ""
    echo "🚀 Starting benchmark with block_size: $block_size"
    echo "================================================"
    
    if [ "$NNODES" -eq 1 ]; then
        # Single-node multi-GPU mode (standalone)
        echo "Mode: Single-node multi-GPU (--standalone)"
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
            benchmark_nccl_alltoall.py \
            --block-size $block_size \
            --num-qo-heads $NUM_QO_HEADS \
            --gqa-group-size $GQA_GROUP_SIZE \
            --head-dim $HEAD_DIM \
            --num-iters $NUM_ITERS
    else
        # Multi-node distributed mode
        echo "Mode: Multi-node distributed (Node $NODE_RANK/$((NNODES-1)))"
        torchrun \
            --nnodes=$NNODES \
            --nproc_per_node=$NPROC_PER_NODE \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            benchmark_nccl_alltoall.py \
            --block-size $block_size \
            --num-qo-heads $NUM_QO_HEADS \
            --gqa-group-size $GQA_GROUP_SIZE \
            --head-dim $HEAD_DIM \
            --num-iters $NUM_ITERS
    fi
    
    # Check exit status of the previous command
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Benchmark completed successfully: block_size=$block_size"
    else
        echo "❌ Benchmark failed with exit code $exit_code: block_size=$block_size"
        # Continue with other tests despite failure (comment out next line to stop on error)
        # exit $exit_code
    fi
done

echo ""
echo "================================================"
echo "All benchmark tests completed!"
echo "================================================"