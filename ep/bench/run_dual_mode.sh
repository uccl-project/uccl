#!/bin/bash
# Launcher for test_dual_mode.py (single dual-mode buffer: HT + LL phases).
#
# Intranode (single node, 8 ranks):
#   ./run_dual_mode.sh intra
#
# Internode (2 nodes x 8 ranks), run on each node:
#   ./run_dual_mode.sh 0     # on master (172.31.77.96)
#   ./run_dual_mode.sh 1     # on second node (172.31.76.80)
set -u

MODE=${1:-intra}
NPROC=${2:-8}
NUM_EXPERTS=${3:-256}
MASTER_ADDR=172.31.77.96
MASTER_PORT=12366
IFNAME=enp71s0

PY=/home/ubuntu/efs/ziming/conda/envs/ziming/bin/python

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# NCCL only used for small bootstrap collectives -> force TCP sockets.
export NCCL_SOCKET_IFNAME=$IFNAME
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

export UCCL_SOCKET_IFNAME=$IFNAME
export GLOO_SOCKET_IFNAME=$IFNAME
export OMP_NUM_THREADS=6

cd /home/ubuntu/efs/zm/uccl/ep/bench

if [ "$MODE" = "intra" ]; then
    exec "$PY" -m torch.distributed.run --standalone --nproc_per_node=$NPROC \
        test_dual_mode.py --num-tokens 128 --hidden 7168 \
        --num-topk 8 --num-experts $NUM_EXPERTS
else
    exec "$PY" -m torch.distributed.run \
        --nnodes=2 --nproc_per_node=$NPROC --node_rank=$MODE \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        test_dual_mode.py --num-tokens 128 --hidden 7168 \
        --num-topk 8 --num-experts $NUM_EXPERTS
fi
