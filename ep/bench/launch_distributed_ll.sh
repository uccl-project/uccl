#!/bin/bash
#
# Launch Low Latency Test Script (Distributed)
#
# Launches test_low_latency.py across multiple nodes via SSH
#
# Usage:
#   ./launch_distributed.sh          # Launch the test
#   ./launch_distributed.sh kill     # Kill lam conda env GPU processes only
#
# Environment variables:
#   MASTER_PORT   - Master port (default: random 29500-30499)
#   CONDA_ENV     - Conda environment to use (default: lam)
#   NUM_TOKENS    - Number of tokens (default: 128)
#   HIDDEN        - Hidden size (default: 7168)
#   NUM_TOPK      - Top-k value (default: 8)
#   NUM_EXPERTS   - Number of experts (default: 288)
#   PER_RANK_LOG  - Enable per-GPU/rank logging (default: 1, set to 0 to disable)
#
# Example:
#   CONDA_ENV=lam NUM_TOKENS=256 ./launch_distributed.sh
#   PER_RANK_LOG=0 ./launch_distributed.sh   # disable per-GPU logs (node-level only)
#

# Configuration
MASTER_ADDR="10.162.224.129"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"  # Random port 29500-30499 if not specified
NNODES=4
NPROC_PER_NODE=8

# Node IPs (in order of node_rank)
NODES=(
    "10.162.224.129"
    "10.162.224.132"
    "10.162.224.133"
    "10.162.224.134"
)

# Kill command - only kills processes running under the lam conda environment
# Two-stage: first shows PIDs, then asks for confirmation
if [ "$1" == "kill" ]; then
    echo "Scanning for lam conda env GPU processes on all nodes..."
    LAM_PYTHON_PATH="/home/yangzhou/miniconda3/envs/lam/bin/python"
    
    # Declare associative array to store PIDs per node
    declare -A NODE_PIDS
    TOTAL_PIDS=0
    
    # Stage 1: Collect PIDs from all nodes
    for node_ip in "${NODES[@]}"; do
        echo "  Checking ${node_ip}..."
        if [ "${node_ip}" == "${MASTER_ADDR}" ]; then
            pids=$(sudo amd-smi process 2>/dev/null | grep -A1 "NAME: ${LAM_PYTHON_PATH}" | grep 'PID:' | awk '{print $2}' | sort -u | tr '\n' ' ')
        else
            pids=$(ssh -o StrictHostKeyChecking=no "${node_ip}" "sudo amd-smi process 2>/dev/null | grep -A1 'NAME: ${LAM_PYTHON_PATH}' | grep 'PID:' | awk '{print \$2}' | sort -u | tr '\n' ' '" 2>/dev/null)
        fi
        pids=$(echo "$pids" | xargs)  # trim whitespace
        if [ -n "$pids" ]; then
            NODE_PIDS["$node_ip"]="$pids"
            pid_count=$(echo "$pids" | wc -w)
            TOTAL_PIDS=$((TOTAL_PIDS + pid_count))
            echo "    Found PIDs: $pids"
        else
            echo "    No lam processes found"
        fi
    done
    
    # Check if any PIDs found
    if [ "$TOTAL_PIDS" -eq 0 ]; then
        echo ""
        echo "No lam processes found on any node."
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
        if [ "${node_ip}" == "${MASTER_ADDR}" ]; then
            echo "$pids" | xargs -r kill -9 2>/dev/null
        else
            ssh -o StrictHostKeyChecking=no "${node_ip}" "echo '$pids' | xargs -r kill -9" 2>/dev/null
        fi
    done
    echo "Done."
    exit 0
fi

# Conda configuration
CONDA_PATH="/home/yangzhou/miniconda3"
CONDA_ENV="${CONDA_ENV:-lam}"  # Can override with CONDA_ENV=myenv

# Default benchmark parameters
NUM_TOKENS="${NUM_TOKENS:-128}"
HIDDEN="${HIDDEN:-7168}"
NUM_TOPK="${NUM_TOPK:-8}"
NUM_EXPERTS="${NUM_EXPERTS:-288}"

# Log directory (shared filesystem)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"

# Create log directory
mkdir -p "${RUN_LOG_DIR}"

echo "=============================================="
echo "Launching distributed test_low_latency.py"
echo "=============================================="
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Nodes: ${NNODES}, GPUs per node: ${NPROC_PER_NODE}"
echo "Conda env: ${CONDA_ENV}"
echo "Parameters: tokens=${NUM_TOKENS}, hidden=${HIDDEN}, topk=${NUM_TOPK}, experts=${NUM_EXPERTS}"
echo "Logs: ${RUN_LOG_DIR}"
if [ "${PER_RANK_LOG:-1}" == "1" ]; then
    echo "Per-rank logging: ENABLED (logs in node_X/local_rank_Y/)"
else
    echo "Per-rank logging: disabled"
fi
echo "=============================================="

# Function to launch on a node
launch_node() {
    local node_rank=$1
    local node_ip=${NODES[$node_rank]}
    local log_file="${RUN_LOG_DIR}/node_${node_rank}_${node_ip}.log"
    
    echo "[$(date '+%H:%M:%S')] Launching on node ${node_rank} (${node_ip})..."
    
    # Build the command using explicit Python path (more reliable than conda activate)
    local PYTHON_BIN="${CONDA_PATH}/envs/${CONDA_ENV}/bin/python"
    
    # Per-rank logging: use torchrun's --log-dir and --tee options
    local LOG_OPTS=""
    if [ "${PER_RANK_LOG:-1}" == "1" ]; then
        # Create per-rank log directory for this node
        local RANK_LOG_DIR="${RUN_LOG_DIR}/node_${node_rank}"
        mkdir -p "${RANK_LOG_DIR}"
        # --tee 3 = tee both stdout(1) and stderr(2) to log files
        # --log-dir = directory where per-rank logs will be written
        LOG_OPTS="--log-dir ${RANK_LOG_DIR} --tee 3"
    fi
    
    local torchrun_cmd="${PYTHON_BIN} -m torch.distributed.run \
        --nnodes=${NNODES} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --node_rank=${node_rank} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        ${LOG_OPTS} \
        bench/test_low_latency.py \
        --num-tokens=${NUM_TOKENS} \
        --hidden=${HIDDEN} \
        --num-topk=${NUM_TOPK} \
        --num-experts=${NUM_EXPERTS}" 
    
    # Set required environment variables for network interfaces
    local env_vars="export NCCL_SOCKET_IFNAME=enp49s0f1np1 && \
export GLOO_SOCKET_IFNAME=enp49s0f1np1 && \
export UCCL_IB_GID_INDEX=3 && \
export NCCL_IB_GID_INDEX=3 && \
export DEBUG_DISPATCH=1"
    local cmd="${env_vars} && cd ${SCRIPT_DIR}/.. && ${torchrun_cmd}"
    
    # Write command to log file first
    {
        echo "=============================================="
        echo "Command: ${torchrun_cmd}"
        echo "Node: ${node_rank} (${node_ip})"
        echo "Time: $(date)"
        echo "Environment:"
        echo "  NCCL_SOCKET_IFNAME=enp49s0f1np1"
        echo "  GLOO_SOCKET_IFNAME=enp49s0f1np1"
        echo "  UCCL_IB_GID_INDEX=3"
        echo "  NCCL_IB_GID_INDEX=3"
        echo "=============================================="
        echo ""
    } > "${log_file}"
    
    if [ "${node_ip}" == "${MASTER_ADDR}" ] && [ "${node_rank}" -eq 0 ]; then
        # Local execution for master node
        nohup bash -c "${cmd}" >> "${log_file}" 2>&1 &
    else
        # SSH to remote node
        # Write output to shared filesystem log file on remote side
        # Use nohup so process survives SSH disconnect
        ssh -o StrictHostKeyChecking=no \
            -o ServerAliveInterval=30 \
            -o ServerAliveCountMax=10 \
            "${node_ip}" "nohup bash -c '${cmd}' >> ${log_file} 2>&1 &" &
    fi
    
    echo "[$(date '+%H:%M:%S')] Node ${node_rank} launched, PID: $!, log: ${log_file}"
}

# Launch all nodes
PIDS=()
for rank in $(seq 0 $((NNODES - 1))); do
    launch_node ${rank}
    PIDS+=($!)
done

echo ""
echo "All nodes launched. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."
echo ""

# Wait for all processes and collect exit codes
EXIT_CODES=()
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    EXIT_CODES+=($?)
done

# Print summary
echo ""
echo "=============================================="
echo "Execution completed"
echo "=============================================="
for i in "${!EXIT_CODES[@]}"; do
    status="SUCCESS"
    if [ ${EXIT_CODES[$i]} -ne 0 ]; then
        status="FAILED (exit code: ${EXIT_CODES[$i]})"
    fi
    echo "Node ${i} (${NODES[$i]}): ${status}"
done

# Create a combined log file
COMBINED_LOG="${RUN_LOG_DIR}/combined.log"
echo "Creating combined log: ${COMBINED_LOG}"
for rank in $(seq 0 $((NNODES - 1))); do
    echo "" >> "${COMBINED_LOG}"
    echo "========== Node ${rank} (${NODES[$rank]}) ==========" >> "${COMBINED_LOG}"
    echo "" >> "${COMBINED_LOG}"
    cat "${RUN_LOG_DIR}/node_${rank}_${NODES[$rank]}.log" >> "${COMBINED_LOG}"
done

echo ""
echo "Logs saved to: ${RUN_LOG_DIR}"
echo "  - Individual logs: node_X_<ip>.log"
echo "  - Combined log: combined.log"
