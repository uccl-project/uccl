#!/usr/bin/env bash
# Usage: CONDA_PATH=/home/yuankach@amd.com/anaconda3/condabin/conda UCCL_HOME=/home/yuankach@amd.com/yang/shuangma/uccl BENCHMARK_SCRIPT="benchmark_uccl_alltoall.py" IDEAL_NODES=4 ENV_NAME=shuangma_env bash luncher_alltoall.sh
set -euo pipefail

# UCCL_HOME=/home/yuankach@amd.com/yang/shuangma/uccl
UCCL_HOME=${UCCL_HOME:-"$HOME/yang/shuangma/uccl"}
SCRIPT_DIR="$UCCL_HOME/scripts"
NODE_SCRIPT="$SCRIPT_DIR/slurm_monitor_select_nodes.py"
HOSTFILE="$SCRIPT_DIR/node_ips/best_nodes.txt"


if [ ! -d "$UCCL_HOME" ]; then
    echo "❌ UCCL_HOME directory does not exist: $UCCL_HOME"
    exit 1
fi

mkdir -p "$(dirname "$HOSTFILE")"

# ================= Check if HOSTFILE exists and is valid =================
if [ -f "$HOSTFILE" ]; then
    echo "✅ HOSTFILE exists: $HOSTFILE"
    if [ ! -s "$HOSTFILE" ]; then
        echo "⚠️  HOSTFILE exists but is empty. Regenerating..."
        rm "$HOSTFILE"
    elif ! grep -q -E "^[a-zA-Z0-9]" "$HOSTFILE"; then
        echo "⚠️  HOSTFILE exists but contains invalid data. Regenerating..."
        rm "$HOSTFILE"
    else
        echo "📋 Using existing nodes:"
        cat "$HOSTFILE"
        echo "ℹ️  To regenerate, delete: $HOSTFILE"
    fi
fi

# ================= Generate HOSTFILE if needed =================
if [ ! -f "$HOSTFILE" ]; then
    if [ ! -f "$NODE_SCRIPT" ]; then
        echo "❌ Error: node selection script not found ($NODE_SCRIPT)"
        exit 1
    fi
        echo "🔍 Generating new node list with $IDEAL_NODES nodes..."
    
    # ================= Run node selection =================
    echo "🎯 Running node selection script..."
    python "$NODE_SCRIPT" --num_nodes "$IDEAL_NODES" --verbose &> "$HOSTFILE"
    
    # Check if node selection was successful
    if [ $? -ne 0 ]; then
        echo "❌ Error: Node selection script failed with exit code $?"
        [ -f "$HOSTFILE" ] && echo "Last output:" && tail -20 "$HOSTFILE"
        [ -f "$HOSTFILE" ] && rm "$HOSTFILE"
        exit 1
    fi
    
    if [ ! -s "$HOSTFILE" ]; then
        echo "❌ Error: Generated HOSTFILE is empty"
        [ -f "$HOSTFILE" ] && rm "$HOSTFILE"
        exit 1
    fi
    
    # Extract just node names (in case verbose output included other info)
    # This keeps only lines that look like node names
    grep -E '^[a-zA-Z0-9_-]+$' "$HOSTFILE" > "${HOSTFILE}.tmp" && mv "${HOSTFILE}.tmp" "$HOSTFILE"
    
    echo "✅ Node selection completed. Generated HOSTFILE: $HOSTFILE"
    echo "📋 Selected nodes:"
    cat "$HOSTFILE"
    
    # Count actual nodes selected
    NODE_COUNT=$(wc -l < "$HOSTFILE")
    echo "🔢 Selected $NODE_COUNT nodes (requested: $IDEAL_NODES)"
    
    if [ "$NODE_COUNT" -eq 0 ]; then
        echo "❌ Error: No nodes were selected"
        rm "$HOSTFILE"
        exit 1
    fi
fi

BENCHMARK_SCRIPT=${BENCHMARK_SCRIPT:-"benchmark_nccl_alltoall.py"}
IDEAL_NODES=${IDEAL_NODES:-3}
echo $BENCHMARK_SCRIPT
# ================= Run node selection =================
python "$NODE_SCRIPT" --num_nodes "$IDEAL_NODES" &> "$HOSTFILE"

if [ ! -f "$HOSTFILE" ]; then
    echo "❌ Error: hostfile was not generated ($HOSTFILE)"
    exit 1
fi

HOSTS=($(awk '{print $1}' "$HOSTFILE"))
MASTER_HOST="${HOSTS[0]}"
MASTER_ADDR=$(grep -w $MASTER_HOST /etc/hosts | awk '{print $1}' | head -n1)

if [ -z "$MASTER_ADDR" ]; then
    echo "❌ Failed to determine MASTER_ADDR for $MASTER_HOST from /etc/hosts"
    exit 1
fi
echo "✅ MASTER_ADDR resolved to $MASTER_ADDR"

MASTER_PORT=19999
NNODES=${#HOSTS[@]}
NPROC_PER_NODE=1

LOG_DIR=$UCCL_HOME/logs
mkdir -p "$LOG_DIR"

ENV_VARS=$(env | grep -E '^(UCCL_HOME|BENCHMARK_SCRIPT|MASTER_ADDR|MASTER_PORT|CONDA_PATH|ENV_NAME)=' | xargs)
ENV_VARS="$ENV_VARS MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE"
echo "Environment variables for $MASTER_HOST:"
echo "$ENV_VARS"

if [ "$(hostname)" != "$MASTER_HOST" ]; then
    echo ">>> Jumping to $MASTER_HOST ..."
    exec ssh "$MASTER_HOST" \
        "$ENV_VARS bash -s" \
        < "$0" "$@"
fi


# From here on, we are running on the master
echo "✅ Running on master node: $MASTER_ADDR"
echo "✅ Hostname: $(hostname)"
echo "Total nodes: $NNODES"
echo "Log dir: $LOG_DIR"

# ===== Trap Ctrl+C for cleanup =====
PIDS=()
cleanup() {
    echo "⚠️ cleaning up..."
    for host in "${HOSTS[@]}"; do
        echo "🧹 Killing torchrun on $host ..."
        ssh "$host" "pkill -f torchrun || true"
    done
    exit 1
}
trap cleanup INT TERM EXIT SIGINT SIGTERM


PIDS=()
# Launch all non-master nodes first
for i in "${!HOSTS[@]}"; do
    host=${HOSTS[$i]}
    log_file="$LOG_DIR/${host}.out"

    # Skip rank 0 (master)
    if [ "$i" -eq 0 ]; then
        continue
    fi

    echo "🚀 Launching on $host (NODE_RANK=$i), logging to $log_file"
    ssh -n "$host" \
        "NODE_RANK=$i \
         $ENV_VARS bash $UCCL_HOME/p2p/benchmarks/worker.sh" \
        >"$log_file" 2>&1 &
    PIDS+=($!)
done

# Wait a short moment for other nodes to initialize (optional)
sleep 2

# Launch master node (rank 0) in foreground
master_host=${HOSTS[0]}
if [ "$(hostname)" == "$master_host" ]; then
    echo "🚀 Launching MASTER node in foreground ($master_host, NODE_RANK=0)"
    BENCHMARK_SCRIPT=$BENCHMARK_SCRIPT MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NNODES=$NNODES NODE_RANK=0 NPROC_PER_NODE=$NPROC_PER_NODE \
        bash $UCCL_HOME/p2p/benchmarks/worker.sh
fi

# ===== Wait for background workers =====
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait $pid; then
        FAIL=1
        cleanup
    fi
done

if [ $FAIL -eq 0 ]; then
    echo "✅ All workers finished successfully"
    echo "📂 Logs are saved under $LOG_DIR/"
fi
