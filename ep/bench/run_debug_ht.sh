#!/bin/bash
# ./run_debug_ht.sh <node_rank> <ht|dual>
set -u
NODE_RANK=${1:?}
BUF=${2:-ht}
PY=/home/ubuntu/efs/ziming/conda/envs/ziming/bin/python
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export NCCL_SOCKET_IFNAME=enp71s0 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN
export UCCL_SOCKET_IFNAME=enp71s0 GLOO_SOCKET_IFNAME=enp71s0 OMP_NUM_THREADS=6
cd /home/ubuntu/efs/zm/uccl/ep/bench
exec "$PY" -m torch.distributed.run \
    --nnodes=2 --nproc_per_node=8 --node_rank=$NODE_RANK \
    --master_addr=172.31.77.96 --master_port=12367 \
    debug_dual_ht.py --buffer $BUF
