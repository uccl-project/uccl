#!/bin/bash
#
# Launch Low Latency Test Script (Distributed)
#
# Launches test_low_latency.py across multiple nodes via SSH
# Supports both Docker and Conda execution modes
#
# Usage:
#   ./launch_distributed_ll.sh [--docker | --conda]   # Launch the test (default: conda)
#   ./launch_distributed_ll.sh kill                   # Kill running processes
#
# Options:
#   --docker    Run inside Docker container (default container: lam_rocm)
#   --conda     Run inside conda environment (default: /home/ubuntu/lam/conda_lam_local)
#
# Environment variables:
#   MASTER_PORT   - Master port (default: random 29500-30499)
#   CONTAINER     - Docker container name (default: lam_rocm, only for --docker)
#   CONDA_ENV     - Conda environment path (default: /home/ubuntu/lam/conda_lam_local, only for --conda)
#   NUM_TOKENS    - Number of tokens (default: 128)
#   HIDDEN        - Hidden size (default: 7168)
#   NUM_TOPK      - Top-k value (default: 8)
#   NUM_EXPERTS   - Number of experts (default: 288)
#
# This script passes --stop-after-first to test_low_latency.py so only the first
# experiment (return_recv_hook=False, dispatch_use_fp8=False, ...) is run.
#
# Examples:
#   ./launch_distributed_ll.sh --conda
#   ./launch_distributed_ll.sh --docker
#   ./launch_distributed_ll.sh --conda NUM_TOKENS=256
#   ./launch_distributed_ll.sh --docker CONTAINER=my_container kill
#

# Parse command line options
RUN_MODE="conda"  # default mode
while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            RUN_MODE="docker"
            shift
            ;;
        --conda)
            RUN_MODE="conda"
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Node IPs (in order of node_rank)
NODES=(
    "172.31.24.178"
    "172.31.20.184"
)

# Configuration - MASTER_ADDR automatically set to first node
MASTER_ADDR="${NODES[0]}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"  # Random port 29500-30499 if not specified
NNODES=2
NPROC_PER_NODE=8

# Docker container name
CONTAINER="${CONTAINER:-lam_rocm}"

# Conda environment path
CONDA_ENV="${CONDA_ENV:-/home/ubuntu/lam/conda_lam_local}"

# Kill command - kills processes based on mode
if [ "$1" == "kill" ]; then
    echo "Scanning for processes on all nodes (mode: ${RUN_MODE})..."

    # Declare associative array to store PIDs per node
    declare -A NODE_PIDS
    TOTAL_PIDS=0

    # Stage 1: Collect PIDs from all nodes
    for node_ip in "${NODES[@]}"; do
        echo "  Checking ${node_ip}..."
        if [ "${RUN_MODE}" == "docker" ]; then
            pids=$(ssh -o StrictHostKeyChecking=no "${node_ip}" "docker exec ${CONTAINER} pgrep -f 'python.*torch'" 2>/dev/null | tr '\n' ' ')
        else
            pids=$(ssh -o StrictHostKeyChecking=no "${node_ip}" "pgrep -f 'python.*test_low_latency'" 2>/dev/null | tr '\n' ' ')
        fi
        pids=$(echo "$pids" | xargs)  # trim whitespace
        if [ -n "$pids" ]; then
            NODE_PIDS["$node_ip"]="$pids"
            pid_count=$(echo "$pids" | wc -w)
            TOTAL_PIDS=$((TOTAL_PIDS + pid_count))
            echo "    Found PIDs: $pids"
        else
            echo "    No processes found"
        fi
    done

    # Check if any PIDs found
    if [ "$TOTAL_PIDS" -eq 0 ]; then
        echo ""
        echo "No processes found on any node."
        exit 0
    fi

    # Stage 2: Ask for confirmation
    echo ""
    echo "=========================================="
    echo "Summary: $TOTAL_PIDS process(es) to kill"
    echo "=========================================="
    for node_ip in "${!NODE_PIDS[@]}"; do
        echo "  ${node_ip}: ${NODE_PIDS[$node_ip]}"
    done
    echo ""
    read -p "Kill these processes? [y/N]: " confirm

    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Aborted."
        exit 0
    fi

    # Stage 3: Kill confirmed PIDs
    echo ""
    echo "Killing processes..."
    for node_ip in "${!NODE_PIDS[@]}"; do
        pids="${NODE_PIDS[$node_ip]}"
        echo "  Killing on ${node_ip}: $pids"
        if [ "${RUN_MODE}" == "docker" ]; then
            ssh -o StrictHostKeyChecking=no "${node_ip}" "for pid in $pids; do docker exec ${CONTAINER} kill -9 \$pid 2>/dev/null; done"
        else
            ssh -o StrictHostKeyChecking=no "${node_ip}" "for pid in $pids; do kill -9 \$pid 2>/dev/null; done"
        fi
    done
    echo "Done."
    exit 0
fi

# Default benchmark parameters
NUM_TOKENS="${NUM_TOKENS:-128}"
HIDDEN="${HIDDEN:-7168}"
NUM_TOPK="${NUM_TOPK:-8}"
NUM_EXPERTS="${NUM_EXPERTS:-288}"

# Log directory (shared filesystem)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
RUN_LOG_DIR="${LOG_DIR}/latest"

# Create log directory
mkdir -p "${RUN_LOG_DIR}"

# Clean old logs before starting
echo "Cleaning old logs in ${RUN_LOG_DIR}..."
rm -rf "${RUN_LOG_DIR:?}"/*

echo "=============================================="
echo "Launching distributed test_low_latency.py"
echo "=============================================="
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Nodes: ${NNODES}, GPUs per node: ${NPROC_PER_NODE}"
echo "Total GPUs: $((NNODES * NPROC_PER_NODE))"
echo "Mode: ${RUN_MODE}"
if [ "${RUN_MODE}" == "docker" ]; then
    echo "Container: ${CONTAINER}"
else
    echo "Conda env: ${CONDA_ENV}"
fi
echo "Parameters: tokens=${NUM_TOKENS}, hidden=${HIDDEN}, topk=${NUM_TOPK}, experts=${NUM_EXPERTS}"
echo "Logs: ${RUN_LOG_DIR}/"
echo "  rank{0..$((NNODES * NPROC_PER_NODE - 1))}.log + node{0..$((NNODES - 1))}.log"
echo "=============================================="

# Function to launch on a node
launch_node() {
    local node_rank=$1
    local node_ip=${NODES[$node_rank]}

    echo "[$(date '+%H:%M:%S')] Launching on node ${node_rank} (${node_ip})..."

    # Build command based on mode
    # Use </dev/null and redirect to file to ensure SSH returns immediately
    # UCCL_LOG_DIR tells the Python script where to write rank{N}.log files
    if [ "${RUN_MODE}" == "docker" ]; then
        # Docker mode - include network environment variables
        local env_vars="export NCCL_SOCKET_IFNAME=enp193s0f1np1 && \
export GLOO_SOCKET_IFNAME=enp193s0f1np1 && \
export UCCL_IB_GID_INDEX=1 && \
export NCCL_IB_GID_INDEX=1 && \
export DEBUG_DISPATCH=1 && \
export UCCL_LOG_DIR=${RUN_LOG_DIR}"
        local torchrun_cmd="${CONDA_ENV}/bin/python -m torch.distributed.run \
            --nnodes=${NNODES} \
            --nproc_per_node=${NPROC_PER_NODE} \
            --node_rank=${node_rank} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            ${SCRIPT_DIR}/test_low_latency.py \
            --num-tokens=${NUM_TOKENS} \
            --hidden=${HIDDEN} \
            --num-topk=${NUM_TOPK} \
            --num-experts=${NUM_EXPERTS} \
            --stop-after-first"
        local exec_cmd="docker exec ${CONTAINER} bash -c '${env_vars} && cd ${SCRIPT_DIR} && ${torchrun_cmd}'"
        ssh -o StrictHostKeyChecking=no \
            -o ServerAliveInterval=30 \
            -o ServerAliveCountMax=10 \
            "${node_ip}" "nohup ${exec_cmd} </dev/null >/dev/null 2>&1 &"
    else
        # Conda mode
        local conda_cmd="export UCCL_LOG_DIR=${RUN_LOG_DIR} && \
            cd ${SCRIPT_DIR} && ${CONDA_ENV}/bin/python -m torch.distributed.run \
            --nnodes=${NNODES} \
            --nproc_per_node=${NPROC_PER_NODE} \
            --node_rank=${node_rank} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            ${SCRIPT_DIR}/test_low_latency.py \
            --num-tokens=${NUM_TOKENS} \
            --hidden=${HIDDEN} \
            --num-topk=${NUM_TOPK} \
            --num-experts=${NUM_EXPERTS} \
            --stop-after-first"
        ssh -o StrictHostKeyChecking=no \
            -o ServerAliveInterval=30 \
            -o ServerAliveCountMax=10 \
            "${node_ip}" "nohup bash -c '${conda_cmd}' </dev/null >/dev/null 2>&1 &"
    fi

    echo "[$(date '+%H:%M:%S')] Node ${node_rank} launched"
}

# Launch all nodes
for rank in $(seq 0 $((NNODES - 1))); do
    launch_node ${rank}
done

echo ""
echo "All nodes launched."
echo "Waiting for completion (checking marker files)..."
echo ""

TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))

# Poll until all ranks are done (check for .done.{rank} marker files)
POLL_INTERVAL=3
while true; do
    done_count=0
    for rank in $(seq 0 $((TOTAL_GPUS - 1))); do
        if [ -f "${RUN_LOG_DIR}/.done.${rank}" ]; then
            done_count=$((done_count + 1))
        fi
    done
    
    if [ "$done_count" -eq "$TOTAL_GPUS" ]; then
        break
    fi
    
    echo "[$(date '+%H:%M:%S')] Progress: ${done_count}/${TOTAL_GPUS} ranks done"
    sleep ${POLL_INTERVAL}
done

# Clean up marker files
rm -f "${RUN_LOG_DIR}"/.done.*

# Print summary
echo ""
echo "=============================================="
echo "Execution completed"
echo "=============================================="
for i in $(seq 0 $((NNODES - 1))); do
    echo "Node ${i} (${NODES[$i]}): DONE"
done

# Create per-node combined logs
echo ""
echo "Creating per-node combined logs..."
for node_rank in $(seq 0 $((NNODES - 1))); do
    node_log="${RUN_LOG_DIR}/node${node_rank}.log"
    > "${node_log}"  # Clear/create file
    for local_rank in $(seq 0 $((NPROC_PER_NODE - 1))); do
        global_rank=$((node_rank * NPROC_PER_NODE + local_rank))
        rank_log="${RUN_LOG_DIR}/rank${global_rank}.log"
        if [ -f "${rank_log}" ]; then
            echo "=== rank${global_rank} ===" >> "${node_log}"
            cat "${rank_log}" >> "${node_log}"
            echo "" >> "${node_log}"
        fi
    done
done

echo ""
echo "Logs saved to: ${RUN_LOG_DIR}"
echo "  Per-rank: rank0.log ... rank$((TOTAL_GPUS - 1)).log (${TOTAL_GPUS} files)"
echo "  Per-node: node0.log ... node$((NNODES - 1)).log (${NNODES} files)"
echo "  Total: $((TOTAL_GPUS + NNODES)) files"
