# /usr/bin/bash

source ../scripts/shared.sh

# Usage ./run_nccl_test.sh [UCCL] [# of Nodes] [# of GPUs per node] [allreduce/alltoall: 0/1]

UCCL=${1:-1}
NUM_PROCS=${2:-2}
NUM_GPUS_PER_NODE=${3:-8}
PROG_OPTION=${4:-0}
PROCS_PER_NODE=${5:-1}
HOSTNAME=${6:-"hosts_single_process"}

# IP of Nodes.
NODES="192.168.102.190,192.168.102.191,192.168.102.192,192.168.102.193,192.168.102.194,192.168.102.195"
# NODES="192.168.102.190,192.168.102.195"
# Names of HCAs."

# NCCL uses the following GPU-NIC mapping can achieve 47GB/s:
#  0-7, 1-6, 2-5, 3-4, 4-3, 5-2, 6-1, 7-0
# If mismatched, 30GB/s is expected.

# The topology detected by UCCL is the same as NCCL. But UCCL using the above mapping only achieves ~5GB/s.
# if mismatched, 30GB/s is expected. Occasionally, segmentation fault occurs.

HCA_NAMES="mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1"
# Name of Control NIC.
CTRL_NIC="enp164s0"
# Path of NCCL
NCCL_PATH="${UCCL_HOME}/thirdparty/nccl/build/lib"
# Path of UCCL
PLUGIN_LIB="${UCCL_HOME}/rdma/libnccl-net-uccl.so"

# Number of chunnels.
NUM_CHUNNEL=8
# Chunk size.
# 131072, 262144, 524288
P2P_NET_CHUNKSIZE=524288
# Buffer size.
BUFFSIZE=8388608
# Number of chunnels per NET peer.
# CHANNELS_NET_PEER=-1
CHANNELS_NET_PEER=8
# Algorithm
# TREE, RING
ALGO=-1

NCCL_PROTO=-1

# Multi-QP for NCCL.
NUM_QPS_PER_CONNECTION=4
SPLIT_DATA_ON_QPS=1

# all_gather_perf  all_reduce_perf  alltoall_perf  broadcast_perf  gather_perf
# hypercube_perf  reduce_perf  reduce_scatter_perf  scatter_perf  sendrecv_perf

if [ "$PROG_OPTION" -eq 0 ]; then
    PROG_NAME=all_reduce_perf
    # We force allreduce to use Simple protocol to avoid outlier for both UCCL and NCCL.
    NCCL_PROTO="Simple"
elif [ "$PROG_OPTION" -eq 1 ]; then
    PROG_NAME=alltoall_perf
else
    PROG_NAME=sendrecv_perf
fi

if [ "$UCCL" -ne 1 ]; then
    PLUGIN_LIB=""
fi

P2P_DISABLE=1
SHM_DISABLE=1
PXN_DISABLE=1

echo "Running test: ${PROG_NAME}, $([ "${UCCL}" -eq 1 ] && echo "UCCL" || echo "NCCL"), ${NUM_PROCS} nodes, ${NUM_GPUS_PER_NODE} GPUs per node, $((NUM_PROCS * NUM_GPUS_PER_NODE)) GPUs in total."

echo -e "Details: NCCL_NCHANNELS=${NUM_CHUNNEL} \n\t NCCL_P2P_NET_CHUNKSIZE=${P2P_NET_CHUNKSIZE} \n\t NCCL_BUFFSIZE=${BUFFSIZE} \n\t NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \n\t NCCL_ALGO=${ALGO} \n\t NCCL_IB_QPS_PER_CONNECTION=${NUM_QPS_PER_CONNECTION} \n\t NCCL_IB_SPLIT_DATA_ON_QPS=${SPLIT_DATA_ON_QPS} \n\t NCCL_PXN_DISABLE=${PXN_DISABLE} \n\t NCCL_P2P_DISABLE=${P2P_DISABLE} \n\t NCCL_SHM_DISABLE=${SHM_DISABLE} \n\t NCCL_IB_HCA=${HCA_NAMES}"

echo $NUM_PROCS
echo $PROCS_PER_NODE
mpirun --allow-run-as-root -np ${NUM_PROCS} -N ${PROCS_PER_NODE} \
    -x NCCL_IB_DISABLE=0  \
    -hostfile ${HOSTNAME} \
    -x NCCL_DEBUG=WARN \
    -x NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 \
    -x NCCL_IB_QPS_PER_CONNECTION=${NUM_QPS_PER_CONNECTION} -x NCCL_IB_SPLIT_DATA_ON_QPS=${SPLIT_DATA_ON_QPS} \
    -x NCCL_IB_PCI_RELAXED_ORDERING=1 \
    -x NCCL_IB_GID_INDEX=3 \
    -x NCCL_ALGO=Ring \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -x NCCL_SOCKET_IFNAME=enp164s0 \
    -x NCCL_IGNORE_CPU_AFFINITY=1 \
    -x NCCL_SOCKET_NTHREADS=2 \
    -x NCCL_CROSS_NIC=0 \
    -x NCCL_P2P_NET_CHUNKSIZE=524288 \
    -x NCCL_BUFFSIZE=8388608 \
    -x NCCL_DMABUF_ENABLE=0 \
    -x LD_LIBRARY_PATH=${NCCL_PATH}:${LD_LIBRARY_PATH} \
    -x NCCL_IB_MERGE_NICS=0 \
    -x NCCL_NVLS_ENABLE=0 \
    -x NCCL_NET_PLUGIN=$PLUGIN_LIB \
    --mca btl tcp,self \
    --mca btl_tcp_if_include enp164s0 \
    ${UCCL_HOME}/thirdparty/nccl-tests/build/${PROG_NAME} -c 0 \
    -b 1K -e 1G \
    -f 2 -w 50 -n 50 \
    -g 1 -t ${NUM_GPUS_PER_NODE}
