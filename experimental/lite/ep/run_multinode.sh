#!/bin/bash
# DeepEPv2 NCCL GIN multi-node launcher for l40 + l41.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EP_DIR="$SCRIPT_DIR"

GPUS_PER_NODE=2
GPU_LIST="2,3"
TEST_SCRIPT="tests/elastic/test_ep.py"
TEST_ARGS="--allow-hybrid-mode 0 --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64"
MASTER_ADDR="4.14.153.89"
MASTER_PORT=12356
L41_HOST="l41"
L41_ENV_CMD='true'
PYTHON_BIN="/home/yangz/nfs/miniconda3/envs/uccl/bin/python"
NCCL_ROOT="/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/deps_nccl_2304/nvidia/nccl"
TORCH_NVSHMEM_STUB="/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/torch-stubs/libtorch_nvshmem.so"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
        --gpu-list) GPU_LIST="$2"; shift 2 ;;
        --test) TEST_SCRIPT="$2"; shift 2 ;;
        --test-args) TEST_ARGS="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --l41-env-cmd) L41_ENV_CMD="$2"; shift 2 ;;
        --python-bin) PYTHON_BIN="$2"; shift 2 ;;
        --nccl-root) NCCL_ROOT="$2"; shift 2 ;;
        --torch-nvshmem-stub) TORCH_NVSHMEM_STUB="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

TOTAL_GPUS=$((GPUS_PER_NODE * 2))
L41_LOG="/home/yangz/nfs/zhongjie/l41_latest.log"

echo "=== DeepEPv2 NCCL GIN Multi-Node Test ==="
echo "Nodes: l40 ($MASTER_ADDR) + l41"
echo "GPUs per node: $GPUS_PER_NODE (total: $TOTAL_GPUS)"
echo "Physical GPU list per node: $GPU_LIST"
echo "Test: $TEST_SCRIPT $TEST_ARGS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo ""

ssh_l41() {
    env -u SSH_AUTH_SOCK ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        -o IdentitiesOnly=yes -i ~/.ssh/uccl-dev "$L41_HOST" "$@"
}

echo "Checking SSH to l41..."
if ! ssh_l41 "echo ok" &>/dev/null; then
    echo "ERROR: Cannot SSH to l41."
    exit 1
fi
echo "SSH to l41: OK"

COMMON_ENV="WORLD_SIZE=2 MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT CUDA_VISIBLE_DEVICES=$GPU_LIST PYTHONPATH=$EP_DIR EP_NCCL_ROOT_DIR=$NCCL_ROOT LD_LIBRARY_PATH=$NCCL_ROOT/lib:\${LD_LIBRARY_PATH:-} LD_PRELOAD=$NCCL_ROOT/lib/libnccl.so.2:$TORCH_NVSHMEM_STUB EP_SUPPRESS_NCCL_CHECK=1 EP_FORCE_NO_NVLINK=1 EP_FORCE_HOST_WINDOW=\${EP_FORCE_HOST_WINDOW:-0} NCCL_IB_HCA=\${NCCL_IB_HCA:-mlx5_0} NCCL_NET_GDR_LEVEL=\${NCCL_NET_GDR_LEVEL:-0} NCCL_GIN_TYPE=\${NCCL_GIN_TYPE:-2} DISABLE_SM90_FEATURES=1 EP_TEST_DISABLE_FP8=1 EP_FORCE_PROCESS_EXIT=1 EP_JIT_CACHE_DIR=/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/jit-cache"
if [[ -n "${NCCL_GIN_PLUGIN:-}" ]]; then
  COMMON_ENV="$COMMON_ENV NCCL_GIN_PLUGIN=$NCCL_GIN_PLUGIN"
fi
if [[ -n "${NCCL_GIN_ENABLE:-}" ]]; then
  COMMON_ENV="$COMMON_ENV NCCL_GIN_ENABLE=$NCCL_GIN_ENABLE"
fi
if [[ -n "${NCCL_GIN_GDAKI_NIC_HANDLER:-}" ]]; then
  COMMON_ENV="$COMMON_ENV NCCL_GIN_GDAKI_NIC_HANDLER=$NCCL_GIN_GDAKI_NIC_HANDLER"
fi

echo ""
echo "Starting node 1 (l41) in background..."
ssh_l41 "nohup bash -lc '
    $L41_ENV_CMD
    cd $EP_DIR
    $COMMON_ENV RANK=1 $PYTHON_BIN $TEST_SCRIPT --num-processes $GPUS_PER_NODE $TEST_ARGS
' > $L41_LOG 2>&1 &"

sleep 4

echo "Starting node 0 (l40)..."
cd "$EP_DIR"
eval "$COMMON_ENV RANK=0 $PYTHON_BIN $TEST_SCRIPT --num-processes $GPUS_PER_NODE $TEST_ARGS"

echo ""
echo "=== l40 (node 0) complete ==="
sleep 3
echo ""
echo "=== l41 output ==="
cat "$L41_LOG"
