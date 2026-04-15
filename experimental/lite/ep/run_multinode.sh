#!/bin/bash
# DeepEP-Lite Multi-Node Launch Script
#
# Launches EP test across 2 nodes (l40 + l41) with 4 GPUs each.
# Run this from l40 (4.14.153.89).
#
# Prerequisites:
#   1. SSH access from l40 to l41 (use env -u SSH_AUTH_SOCK ssh l41)
#   2. ep.abi3.so built and installed on both nodes
#   3. l40: conda env 'uccl' (python 3.12)
#      l41: virtualenv ~/zhongjie/zj_py (python 3.13)
#
# Usage:
#   bash run_multinode.sh                               # default test
#   bash run_multinode.sh --test bench/test_low_latency.py \
#       --test-args "--num-tokens=128 --hidden=2048 --num-topk=4 --num-experts=8 --disable-nvlink"
#   bash run_multinode.sh --gpus-per-node 1             # 1 GPU per node

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EP_DIR="$SCRIPT_DIR"

# Defaults
GPUS_PER_NODE=4
TEST_SCRIPT="bench/test_low_latency.py"
TEST_ARGS=""
MASTER_ADDR="4.14.153.89"
MASTER_PORT=12356
L41_HOST="l41"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
        --test) TEST_SCRIPT="$2"; shift 2 ;;
        --test-args) TEST_ARGS="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

GPU_LIST=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
TOTAL_GPUS=$((GPUS_PER_NODE * 2))

echo "=== DeepEP-Lite Multi-Node Test ==="
echo "Nodes: l40 ($MASTER_ADDR) + l41"
echo "GPUs per node: $GPUS_PER_NODE (total: $TOTAL_GPUS)"
echo "Test: $TEST_SCRIPT $TEST_ARGS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo ""

# SSH to l41 without stale vscode agent
ssh_l41() {
    env -u SSH_AUTH_SOCK ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        -o IdentitiesOnly=yes -i ~/.ssh/uccl-dev "$L41_HOST" "$@"
}

echo "Checking SSH to l41..."
if ! ssh_l41 "echo ok" &>/dev/null; then
    echo "ERROR: Cannot SSH to l41."
    echo "Fix: Add this key to l41's ~/.ssh/authorized_keys:"
    cat ~/.ssh/uccl-dev.pub
    exit 1
fi
echo "SSH to l41: OK"

EP_SO="$EP_DIR/ep.abi3.so"
if [ ! -f "$EP_SO" ]; then
    echo "ERROR: ep.abi3.so not found. Build first: cd $EP_DIR && conda run -n uccl make -j SM=89"
    exit 1
fi

cp "$EP_SO" /home/yangz/nfs/miniconda3/envs/uccl/lib/python3.12/site-packages/uccl/ep.abi3.so
ssh_l41 "cp /home/yangz/nfs/zhongjie/uccl/experimental/lite/ep/ep.abi3.so ~/zhongjie/zj_py/lib/python3.13/site-packages/uccl/ep.abi3.so"
echo "ep.abi3.so installed on both nodes."

L41_LOG="/home/yangz/nfs/zhongjie/l41_latest.log"

echo ""
echo "Starting node 1 (l41) in background..."
ssh_l41 "nohup bash -c '
    source ~/zhongjie/zj_py/bin/activate
    cd /home/yangz/nfs/zhongjie/uccl/experimental/lite/ep
    NCCL_IB_HCA=mlx5_0 CUDA_VISIBLE_DEVICES=$GPU_LIST \
    torchrun --nnodes=2 --nproc_per_node=$GPUS_PER_NODE --node_rank=1 \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        $TEST_SCRIPT $TEST_ARGS
' > $L41_LOG 2>&1 &"

sleep 4

echo "Starting node 0 (l40)..."
cd "$EP_DIR"
conda run -n uccl env \
    NCCL_IB_HCA=mlx5_0 CUDA_VISIBLE_DEVICES=$GPU_LIST \
    torchrun --nnodes=2 --nproc_per_node=$GPUS_PER_NODE --node_rank=0 \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        $TEST_SCRIPT $TEST_ARGS 2>&1

echo ""
echo "=== l40 (node 0) complete ==="
sleep 3
echo ""
echo "=== l41 output ==="
cat "$L41_LOG"
