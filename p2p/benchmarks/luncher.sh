#!/usr/bin/env bash

set -euo pipefail


trap cleanup EXIT SIGINT SIGTERM

# ===== Cluster hosts =====

# ================= Configuration =================
HOME_UCCL=/home/yuankach@amd.com/yang/shuangma/uccl
SCRIPT_DIR="$HOME_UCCL/scripts"
NODE_SCRIPT="$SCRIPT_DIR/slurm_monitor_select_nodes.py"
HOSTFILE="$SCRIPT_DIR/node_ips/best_nodes.txt"

if [ ! -f "$NODE_SCRIPT" ]; then
    echo "❌ Error: node selection script not found ($NODE_SCRIPT)"
    exit 1
fi

IDEAL_NODES=${IDEAL_NODES:-3}
echo $BENCHMARK_SCRIPT
# ================= Run node selection =================
python "$NODE_SCRIPT" AIG_Models "$IDEAL_NODES" &> "$HOSTFILE"

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

LOG_DIR=$HOME_UCCL
mkdir -p "$LOG_DIR"

if [ "$(hostname)" != "$MASTER_HOST" ]; then
    echo ">>> Jumping to $MASTER_HOST ..."

    ENV_VARS=$(env | grep -E '^(BENCHMARK_|IDEAL_)' | xargs -I {} echo "export {};" | tr '\n' ' ')

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
        "BENCHMARK_SCRIPT=$BENCHMARK_SCRIPT MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NNODES=$NNODES NODE_RANK=$i NPROC_PER_NODE=$NPROC_PER_NODE \
         bash $HOME_UCCL/p2p/benchmarks/worker.sh" \
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
        bash $HOME_UCCL/p2p/benchmarks/worker.sh
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
