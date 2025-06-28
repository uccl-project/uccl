#!/bin/bash

# Wrapper script for multi_pg_test.py
# Mirrors the environment-setup style used by ddp_amd.sh so that both demos
# behave consistently.

# Positional arguments: BACKEND then MODE for consistency with other scripts
BACKEND=${1:-nccl}  # nccl or uccl
MODE=${2:-single}   # single or multi
NUM_GPUS_PER_NODE=${3:-4}
ITERS=${4:-100}
TENSOR_SIZE=${5:-1024}

PROG="multi_pg_test.py"
DEVICES="0,1,2,5"  # Only these GPUs have corresponding NICs on the AMD cluster

# Common for both NCCL and UCCL on the AMD cluster
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_BUFFSIZE=8388608
export NCCL_MIN_NCHANNELS=32
export NCCL_MAX_NCHANNELS=32
export NCCL_NCHANNELS_PER_NET_PEER=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export HIP_VISIBLE_DEVICES=${DEVICES}
export NCCL_IB_HCA="rdma0:1,rdma2:1,rdma3:1,rdma4:1"
export NCCL_SOCKET_IFNAME="cni0"

# UCCL specific environment variables
if [ "$BACKEND" = "uccl" ]; then
    if [[ -z "${UCCL_HOME}" || -z "${CONDA_LIB_HOME}" ]]; then
        echo "UCCL_HOME or CONDA_LIB_HOME is not set or is empty"
        exit 1
    else
        echo "UCCL_HOME is set to: ${UCCL_HOME}"
        echo "CONDA_LIB_HOME is set to: ${CONDA_LIB_HOME}"
    fi

    export GLOG_v=0
    export LD_LIBRARY_PATH="${CONDA_LIB_HOME}:${LD_LIBRARY_PATH}"
    export NCCL_NET_PLUGIN="${UCCL_HOME}/rdma/librccl-net-uccl.so"
    export UCCL_NUM_ENGINES=4
    export UCCL_PORT_ENTROPY=8
    export UCCL_CHUNK_SIZE_KB=128
fi

# Function to print usage
print_usage() {
    echo "Usage: $0 [BACKEND] [MODE] [NUM_GPUS_PER_NODE] [ITERS] [TENSOR_SIZE]"
    echo ""
    echo "Parameters:"
    echo "  BACKEND     : nccl or uccl [default: nccl]"
    echo "  MODE        : single (single-node) or multi (multi-node) [default: single]"
    echo "  NUM_GPUS_PER_NODE : GPUs per node [default: 4]"
    echo "  ITERS       : iterations per collective [default: 100]"
    echo "  TENSOR_SIZE : float elements per tensor [default: 1024]"
    echo ""
    echo "Examples:"
    echo "  Single-node run: $0 nccl single 4 200 4096"
    echo "  Multi-node run:  MASTER_ADDR=10.0.0.1 MASTER_PORT=12355 NODE_RANK=0 WORLD_SIZE=2 $0 uccl multi 4 200 4096"
}

# Main execution logic
main() {
    echo "=== Concurrent Collectives Demo ==="
    echo "Backend: $BACKEND"
    echo "Mode: $MODE"
    echo "GPUs per node: $NUM_GPUS_PER_NODE"
    echo "Iterations: $ITERS"
    echo "Tensor size: $TENSOR_SIZE"
    echo ""

    if [ "$MODE" = "single" ]; then
        echo "=== Starting Single-Node Run ==="

        torchrun \
            --nproc_per_node=${NUM_GPUS_PER_NODE} \
            --nnodes=1 \
            --node_rank=0 \
            --master_addr=localhost \
            --master_port=12355 \
            ${PROG} \
            --iters ${ITERS} \
            --tensor_size ${TENSOR_SIZE}

    elif [ "$MODE" = "multi" ]; then
        echo "=== Starting Multi-Node Run ==="

        if [ -z "$MASTER_ADDR" ] || [ -z "$MASTER_PORT" ] || [ -z "$NODE_RANK" ] || [ -z "$WORLD_SIZE" ]; then
            echo "Error: Multi-node execution requires MASTER_ADDR, MASTER_PORT, NODE_RANK and WORLD_SIZE."
            print_usage
            exit 1
        fi

        echo "  Master Address: $MASTER_ADDR"
        echo "  Master Port: $MASTER_PORT"
        echo "  Node Rank: $NODE_RANK"
        echo "  World Size: $WORLD_SIZE"
        echo ""

        torchrun \
            --nproc_per_node=${NUM_GPUS_PER_NODE} \
            --nnodes=${WORLD_SIZE} \
            --node_rank=${NODE_RANK} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            ${PROG} \
            --iters ${ITERS} \
            --tensor_size ${TENSOR_SIZE}
    else
        echo "Error: Invalid mode '$MODE'. Use 'single' or 'multi'."
        print_usage
        exit 1
    fi
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    print_usage
    exit 0
fi

main

echo ""
echo "=== Concurrent Collectives Completed ===" 