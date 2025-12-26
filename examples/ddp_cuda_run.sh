#!/bin/bash

# PyTorch Distributed Training Script for NVIDIA GPUs
# Supports both single-node multi-GPU and multi-node multi-GPU training

# Parse command line arguments (BACKEND first then MODE)
BACKEND=${1:-nccl}  # nccl or uccl
MODE=${2:-single}   # single or multi
BATCH_SIZE=${3:-128}
EPOCHS=${4:-10}

# Configuration
PROG="ddp_cuda_test.py"
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}

# ---------------- Common environment ----------------
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_BUFFSIZE=8388608
export NCCL_MIN_NCHANNELS=32
export NCCL_MAX_NCHANNELS=32
export NCCL_NCHANNELS_PER_NET_PEER=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_QPS=1

# Optionally set CUDA_VISIBLE_DEVICES if not already set
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Optionally set IB HCA devices for multi-NIC systems
# export NCCL_IB_HCA="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"

# Optionally set socket interface for multi-node
# export NCCL_SOCKET_IFNAME="eth0"

# ---------------- UCCL-specific ----------------
if [ "$BACKEND" = "uccl" ]; then
    if [[ -z "${UCCL_HOME}" ]]; then
        echo "Error: UCCL_HOME is not set"
        echo "Please set UCCL_HOME to the root of your UCCL checkout"
        exit 1
    else
        echo "UCCL_HOME = ${UCCL_HOME}"
    fi

    # Check if UCCL plugin exists
    UCCL_PLUGIN="${UCCL_HOME}/collective/rdma/libnccl-net-uccl.so"
    if [ ! -f "$UCCL_PLUGIN" ]; then
        echo "Error: UCCL plugin not found at $UCCL_PLUGIN"
        echo "Please build UCCL first: cd ${UCCL_HOME}/collective/rdma && make -j"
        exit 1
    fi

    export GLOG_v=0
    export NCCL_NET_PLUGIN="${UCCL_PLUGIN}"
    export UCCL_NUM_ENGINES=4
    export UCCL_PORT_ENTROPY=8
    export UCCL_CHUNK_SIZE_KB=128

    # Optionally add glog library path if needed
    if [[ -n "${CONDA_LIB_HOME}" ]]; then
        export LD_LIBRARY_PATH="${CONDA_LIB_HOME}:${LD_LIBRARY_PATH}"
    fi
fi

# Function to print usage
print_usage() {
    echo "Usage: $0 [BACKEND] [MODE] [BATCH_SIZE] [EPOCHS]"
    echo ""
    echo "Parameters:"
    echo "  BACKEND     : nccl or uccl [default: nccl]"
    echo "  MODE        : single (single-node) or multi (multi-node) [default: single]"
    echo "  BATCH_SIZE  : batch size per GPU [default: 128]"
    echo "  EPOCHS      : number of training epochs [default: 10]"
    echo ""
    echo "Environment Variables:"
    echo "  UCCL_HOME           : Path to UCCL checkout (required for uccl backend)"
    echo "  NUM_GPUS_PER_NODE   : Number of GPUs per node [default: 8]"
    echo "  CUDA_VISIBLE_DEVICES: GPUs to use (e.g., '0,1,2,3')"
    echo "  NCCL_IB_HCA         : IB HCA devices (e.g., 'mlx5_0:1,mlx5_1:1')"
    echo "  NCCL_SOCKET_IFNAME  : Network interface for multi-node (e.g., 'eth0')"
    echo ""
    echo "Examples:"
    echo "  Single-node training with NCCL:"
    echo "    $0 nccl single 128 10"
    echo ""
    echo "  Single-node training with UCCL:"
    echo "    UCCL_HOME=/path/to/uccl $0 uccl single 128 10"
    echo ""
    echo "  Multi-node training with UCCL (run on each node):"
    echo "    Node 0: MASTER_ADDR=10.0.0.1 MASTER_PORT=12355 NODE_RANK=0 WORLD_SIZE=2 \\"
    echo "            UCCL_HOME=/path/to/uccl $0 uccl multi 128 10"
    echo "    Node 1: MASTER_ADDR=10.0.0.1 MASTER_PORT=12355 NODE_RANK=1 WORLD_SIZE=2 \\"
    echo "            UCCL_HOME=/path/to/uccl $0 uccl multi 128 10"
}

# Main execution logic
main() {
    echo "=== PyTorch Distributed Training for NVIDIA GPUs ==="
    echo "Backend: $BACKEND"
    echo "Mode: $MODE"
    echo "Batch Size: $BATCH_SIZE"
    echo "Epochs: $EPOCHS"
    echo "GPUs per node: $NUM_GPUS_PER_NODE"
    echo ""

    # Get the directory where this script is located
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

    if [ "$MODE" = "single" ]; then
        echo "=== Starting Single-Node Multi-GPU Training ==="

        # Single-node training using torchrun
        torchrun \
            --nproc_per_node=${NUM_GPUS_PER_NODE} \
            --nnodes=1 \
            --node_rank=0 \
            --master_addr=localhost \
            --master_port=12355 \
            ${SCRIPT_DIR}/${PROG} \
            --batch_size ${BATCH_SIZE} \
            --epochs ${EPOCHS}

    elif [ "$MODE" = "multi" ]; then
        echo "=== Starting Multi-Node Multi-GPU Training ==="

        # Check required environment variables for multi-node
        if [ -z "$MASTER_ADDR" ] || [ -z "$MASTER_PORT" ] || [ -z "$NODE_RANK" ] || [ -z "$WORLD_SIZE" ]; then
            echo "Error: Multi-node training requires the following environment variables:"
            echo "  MASTER_ADDR  : IP address of the master node"
            echo "  MASTER_PORT  : Port number for communication (e.g., 12355)"
            echo "  NODE_RANK    : Rank of this node (0 for master, 1,2,... for workers)"
            echo "  WORLD_SIZE   : Total number of nodes"
            echo ""
            echo "Example setup for 2 nodes:"
            echo "  Node 0 (Master): MASTER_ADDR=192.168.1.100 MASTER_PORT=12355 NODE_RANK=0 WORLD_SIZE=2 $0 uccl multi"
            echo "  Node 1 (Worker): MASTER_ADDR=192.168.1.100 MASTER_PORT=12355 NODE_RANK=1 WORLD_SIZE=2 $0 uccl multi"
            exit 1
        fi

        echo "Multi-node configuration:"
        echo "  Master Address: $MASTER_ADDR"
        echo "  Master Port: $MASTER_PORT"
        echo "  Node Rank: $NODE_RANK"
        echo "  World Size: $WORLD_SIZE"
        echo ""

        # Multi-node training using torchrun
        torchrun \
            --nproc_per_node=${NUM_GPUS_PER_NODE} \
            --nnodes=${WORLD_SIZE} \
            --node_rank=${NODE_RANK} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            ${SCRIPT_DIR}/${PROG} \
            --batch_size ${BATCH_SIZE} \
            --epochs ${EPOCHS}

    else
        echo "Error: Invalid mode '$MODE'. Use 'single' or 'multi'."
        print_usage
        exit 1
    fi
}

# Handle help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    print_usage
    exit 0
fi

# Run main function
main

echo ""
echo "=== Training Completed ==="
