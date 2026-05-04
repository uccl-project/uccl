#!/bin/bash
# DeepEPv2 UCCL proxy multi-node launcher.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EP_DIR="$SCRIPT_DIR"

GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
GPU_LIST="${GPU_LIST:-0,1}"
TEST_SCRIPT="${TEST_SCRIPT:-tests/elastic/test_ep.py}"
TEST_ARGS="${TEST_ARGS:---allow-hybrid-mode 0 --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 --do-handle-copy-modes=1 --expert-alignment-modes=128 --fp8-dispatch-modes=0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-12356}"
NODE1_HOST="${NODE1_HOST:-node1}"
NODE1_ENV_CMD="${NODE1_ENV_CMD:-true}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NCCL_ROOT="${NCCL_ROOT:-${EP_NCCL_ROOT_DIR:-}}"
TORCH_NVSHMEM_STUB="${TORCH_NVSHMEM_STUB:-}"
EP_JIT_CACHE_DIR="${EP_JIT_CACHE_DIR:-$EP_DIR/.jit-cache}"
NODE1_LOG="${NODE1_LOG:-/tmp/uccl_ep_node1.log}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
        --gpu-list) GPU_LIST="$2"; shift 2 ;;
        --test) TEST_SCRIPT="$2"; shift 2 ;;
        --test-args) TEST_ARGS="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --node1-host) NODE1_HOST="$2"; shift 2 ;;
        --node1-env-cmd) NODE1_ENV_CMD="$2"; shift 2 ;;
        --python-bin) PYTHON_BIN="$2"; shift 2 ;;
        --nccl-root) NCCL_ROOT="$2"; shift 2 ;;
        --torch-nvshmem-stub) TORCH_NVSHMEM_STUB="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$NCCL_ROOT" ]]; then
    echo "ERROR: Set NCCL_ROOT or EP_NCCL_ROOT_DIR, or pass --nccl-root."
    exit 1
fi
LD_PRELOAD_VALUE="$NCCL_ROOT/lib/libnccl.so.2"
if [[ -n "$TORCH_NVSHMEM_STUB" ]]; then
    LD_PRELOAD_VALUE="$LD_PRELOAD_VALUE:$TORCH_NVSHMEM_STUB"
fi

TOTAL_GPUS=$((GPUS_PER_NODE * 2))

echo "=== DeepEPv2 UCCL Proxy Multi-Node Test ==="
echo "Nodes: local ($MASTER_ADDR) + $NODE1_HOST"
echo "GPUs per node: $GPUS_PER_NODE (total: $TOTAL_GPUS)"
echo "Physical GPU list per node: $GPU_LIST"
echo "Test: $TEST_SCRIPT $TEST_ARGS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo ""

ssh_node1() {
    env -u SSH_AUTH_SOCK ssh -o ConnectTimeout=10 "$NODE1_HOST" "$@"
}

echo "Checking SSH to $NODE1_HOST..."
if ! ssh_node1 "echo ok" &>/dev/null; then
    echo "ERROR: Cannot SSH to $NODE1_HOST."
    exit 1
fi
echo "SSH to $NODE1_HOST: OK"

DEFAULT_NCCL_NET_GDR_LEVEL=0
if [[ "${UCCL_FORCE_NO_GDR:-1}" == "0" ]]; then
  DEFAULT_NCCL_NET_GDR_LEVEL=5
fi

COMMON_ENV="WORLD_SIZE=2 LOCAL_WORLD_SIZE=$GPUS_PER_NODE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT CUDA_VISIBLE_DEVICES=$GPU_LIST PYTHONPATH=$EP_DIR EP_NCCL_ROOT_DIR=$NCCL_ROOT LD_LIBRARY_PATH=$NCCL_ROOT/lib:\${LD_LIBRARY_PATH:-} LD_PRELOAD=$LD_PRELOAD_VALUE EP_SUPPRESS_NCCL_CHECK=1 EP_USE_UCCL_PROXY=\${EP_USE_UCCL_PROXY:-1} UCCL_FORCE_NO_GDR=\${UCCL_FORCE_NO_GDR:-1} EP_FORCE_NO_NVLINK=1 EP_FORCE_HOST_WINDOW=\${EP_FORCE_HOST_WINDOW:-0} NCCL_IB_HCA=\${NCCL_IB_HCA:-mlx5_0} NCCL_NET_GDR_LEVEL=\${NCCL_NET_GDR_LEVEL:-$DEFAULT_NCCL_NET_GDR_LEVEL} NCCL_GIN_TYPE=\${NCCL_GIN_TYPE:-2} DISABLE_SM90_FEATURES=1 EP_FORCE_PROCESS_EXIT=1 EP_JIT_CACHE_DIR=$EP_JIT_CACHE_DIR"
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
echo "Starting node 1 ($NODE1_HOST) in background..."
ssh_node1 "nohup bash -lc '
    $NODE1_ENV_CMD
    cd $EP_DIR
    $COMMON_ENV RANK=1 $PYTHON_BIN $TEST_SCRIPT --num-processes $GPUS_PER_NODE $TEST_ARGS
' > $NODE1_LOG 2>&1 &"

sleep 4

echo "Starting local node (rank 0)..."
cd "$EP_DIR"
eval "$COMMON_ENV RANK=0 $PYTHON_BIN $TEST_SCRIPT --num-processes $GPUS_PER_NODE $TEST_ARGS"

echo ""
echo "=== local node (rank 0) complete ==="
sleep 3
echo ""
echo "=== $NODE1_HOST output ==="
cat "$NODE1_LOG"
