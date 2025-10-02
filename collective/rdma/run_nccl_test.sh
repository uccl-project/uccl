# !/bin/bash

source ../../scripts/shared.sh

# Usage ./run_nccl_test.sh [UCCL] [# of Nodes] [# of GPUs per process] [allreduce/alltoall: 0/1]

TEST=${1:-1}
NUM_PROCS=${2:-2}
NUM_GPUS_PER_PROC=${3:-8}
PROG_OPTION=${4:-0}
PROCS_PER_NODE=${5:-1}
HOSTFILE="${UCCL_HOME}/scripts/node_ips/h100_6.txt"

# Names of HCAs.
HCA_NAMES="mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1"
# Name of Control NIC.
CTRL_NIC="enp164s0"
# Path of NCCL
NCCL_PATH="${UCCL_HOME}/thirdparty/nccl/build/lib"

# Number of channels.
NUM_CHANNELS=8
# Chunk size.
# 131072, 262144, 524288
P2P_NET_CHUNKSIZE=524288
# Buffer size.
BUFFSIZE=8388608
# Number of channels per NET peer.
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


if [ "$TEST" = "nccl" ]; then
    echo "Running NCCL test"
    PLUGIN_PATH=""
elif [ "$TEST" = "uccl" ]; then
    echo "Running UCCL test"
    # PLUGIN_PATH="${UCCL_HOME}/collective/rdma/libnccl-net-uccl.so"
    PLUGIN_PATH=`python -c "import uccl; print(uccl.nccl_plugin_path())"`
    echo "PLUGIN_PATH: ${PLUGIN_PATH}"
else
    echo "Unsupport benchmark type."
    exit 1
fi

NVLINK_ON=1
NVLINK_OFF=$((1 - NVLINK_ON))

echo "Running test: ${PROG_NAME}, ${TEST}, ${NUM_PROCS} nodes, ${NUM_GPUS_PER_PROC} GPUs per process, $((NUM_PROCS * NUM_GPUS_PER_PROC)) GPUs in total."

echo -e "Details: NCCL_NCHANNELS=${NUM_CHANNELS} \n\t NCCL_P2P_NET_CHUNKSIZE=${P2P_NET_CHUNKSIZE} \n\t NCCL_BUFFSIZE=${BUFFSIZE} \n\t NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \n\t NCCL_ALGO=${ALGO} \n\t NCCL_IB_QPS_PER_CONNECTION=${NUM_QPS_PER_CONNECTION} \n\t NCCL_IB_SPLIT_DATA_ON_QPS=${SPLIT_DATA_ON_QPS} \n\t NVLINK_ON=${NVLINK_ON} \n\t NCCL_IB_HCA=${HCA_NAMES}"

echo $NUM_PROCS
echo $PROCS_PER_NODE
mpirun --allow-run-as-root -np ${NUM_PROCS} -N ${PROCS_PER_NODE} \
    -x NCCL_IB_DISABLE=0  \
    -hostfile ${HOSTFILE} --map-by ppr:${PROCS_PER_NODE}:node \
    -x NCCL_DEBUG=WARN \
    -x NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 \
    -x NCCL_IB_QPS_PER_CONNECTION=${NUM_QPS_PER_CONNECTION} -x NCCL_IB_SPLIT_DATA_ON_QPS=${SPLIT_DATA_ON_QPS} \
    -x NCCL_IB_PCI_RELAXED_ORDERING=1 \
    -x NCCL_P2P_DISABLE=${NVLINK_OFF} \
    -x NCCL_SHM_DISABLE=${NVLINK_OFF} \
    -x NCCL_IB_GID_INDEX=3 \
    -x NCCL_ALGO=Ring \
    -x NCCL_MAX_NCHANNELS=${NUM_CHANNELS} \
    -x NCCL_MIN_NCHANNELS=${NUM_CHANNELS} \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -x NCCL_SOCKET_IFNAME=${CTRL_NIC} \
    -x NCCL_IGNORE_CPU_AFFINITY=1 \
    -x NCCL_SOCKET_NTHREADS=2 \
    -x NCCL_CROSS_NIC=0 \
    -x NCCL_P2P_NET_CHUNKSIZE=524288 \
    -x NCCL_BUFFSIZE=8388608 \
    -x NCCL_DMABUF_ENABLE=0 \
    -x LD_LIBRARY_PATH=${NCCL_PATH}:${LD_LIBRARY_PATH} \
    -x NCCL_IB_MERGE_NICS=0 \
    -x NCCL_NVLS_ENABLE=0 \
    -x NCCL_NET_PLUGIN=${PLUGIN_PATH} \
    --mca btl tcp,self \
    --mca btl_tcp_if_include ${CTRL_NIC} \
    ${UCCL_HOME}/thirdparty/nccl-tests/build/${PROG_NAME} -c 0 \
    -b 1K -e 1G \
    -f 2 -w 50 -n 50 \
    -g 1 -t ${NUM_GPUS_PER_PROC}
