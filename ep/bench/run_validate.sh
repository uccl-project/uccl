#!/bin/bash
# Validation launcher for test_internode.py on the 2-node EFA H200 setup.
# Usage: run_validate.sh <node_rank> <nproc_per_node> [num_experts]
set -u

NODE_RANK=${1:-0}
NPROC=${2:-8}
NUM_EXPERTS=${3:-256}
MASTER_ADDR=172.31.85.63
MASTER_PORT=12355
IFNAME=enp71s0

PY=/home/ubuntu/efs/ziming/conda/envs/ziming/bin/python

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# NCCL only used for small bootstrap collectives -> force TCP sockets over enp71s0.
export NCCL_SOCKET_IFNAME=$IFNAME
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# UCCL EP out-of-band / RDMA selection.
export UCCL_SOCKET_IFNAME=$IFNAME
export GLOO_SOCKET_IFNAME=$IFNAME
export OMP_NUM_THREADS=6

echo "[run_validate] node_rank=$NODE_RANK nproc=$NPROC experts=$NUM_EXPERTS master=$MASTER_ADDR ifname=$IFNAME"
cd /home/ubuntu/efs/zm/uccl/ep/bench

exec "$PY" -m torch.distributed.run \
    --nnodes=2 --nproc_per_node=$NPROC --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    test_internode.py --num-tokens=4096 \
    --hidden=7168 --num-topk=8 --num-experts=$NUM_EXPERTS --test-ll-compatibility
