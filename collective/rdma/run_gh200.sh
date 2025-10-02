# !/bin/bash

source ../../scripts/shared.sh

# Usage ./run_nccl_test.sh [UCCL] [# of Nodes] [# of GPUs per process] [allreduce/alltoall: 0/1]

TEST=${1:-nccl}
NUM_PROCS=${2:-2}
NUM_GPUS_PER_PROC=${3:-1}
PROG_OPTION=${4:-0}
PROCS_PER_NODE=${5:-1}
HOSTFILE="${UCCL_HOME}/scripts/node_ips/gh200.txt"

NODES=""
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  host=$(echo "$line" | awk '{print $1}')
  NODES="${NODES}${host},"
done < "$HOSTFILE"

# Trim trailing comma
NODES=${NODES%,}
echo "Parsed NODES: $NODES"

IFS=',' read -ra ADDR <<< "$NODES"
for node in "${ADDR[@]}"; do
    echo "Checking GPU usage on $node..."
    gpu_procs=$(ssh "$node" "nvidia-smi --query-compute-apps=pid --format=csv,noheader")

    if [[ -n "$gpu_procs" ]]; then
        echo "GPU in use on $node by PIDs: $gpu_procs"
        exit 1
    else
        echo "No GPU processes found on $node."
    fi
done

# Names of HCAs."
HCA_NAMES="mlx5_2:1"
# Name of Control NIC.
CTRL_NIC="bond0"
# Path of NCCL
NCCL_PATH="${UCCL_HOME}/thirdparty/nccl/build/lib"

# Number of chunnels.
NUM_CHUNNEL=8
# Chunk size.
# 131072, 262144, 524288
P2P_NET_CHUNKSIZE=524288
# Buffer size.
BUFFSIZE=8388608
# Number of chunnels per NET peer.
CHANNELS_NET_PEER=4
# Algorithm
# TREE, RING
ALGO=-1

NCCL_PROTO=-1

# Multi-QP for NCCL.
NUM_QPS_PER_CONNECTION=1
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
    plugin_path=""
elif [ "$TEST" = "uccl" ]; then
    echo "Running UCCL test"
    # plugin_path="${UCCL_HOME}/collective/rdma/libnccl-net-uccl.so"
    plugin_path=`python -c "import uccl; print(uccl.nccl_plugin_path())"`
    echo "plugin_path: ${plugin_path}"
else
    echo "Unsupport benchmark type."
    exit 1
fi

NVLINK_OFF=1

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

echo "Running test: ${PROG_NAME}, ${TEST}, ${NUM_PROCS} nodes, ${NUM_GPUS_PER_PROC} GPUs per process, $((NUM_PROCS * NUM_GPUS_PER_PROC)) GPUs in total."

echo -e "Details: NCCL_NCHANNELS=${NUM_CHUNNEL} \n\t NCCL_P2P_NET_CHUNKSIZE=${P2P_NET_CHUNKSIZE} \n\t NCCL_BUFFSIZE=${BUFFSIZE} \n\t NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \n\t NCCL_ALGO=${ALGO} \n\t NCCL_IB_QPS_PER_CONNECTION=${NUM_QPS_PER_CONNECTION} \n\t NCCL_IB_SPLIT_DATA_ON_QPS=${SPLIT_DATA_ON_QPS} \n\t NCCL_PXN_DISABLE=${NVLINK_OFF} \n\t NCCL_P2P_DISABLE=${NVLINK_OFF} \n\t NCCL_SHM_DISABLE=${NVLINK_OFF} \n\t NCCL_IB_HCA=${HCA_NAMES}"

/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun --prefix /usr/mpi/gcc/openmpi-4.1.7rc1 --bind-to none -np ${NUM_PROCS} -N ${PROCS_PER_NODE} \
    -hostfile ${HOSTFILE} --map-by ppr:${PROCS_PER_NODE}:node \
    --mca btl_tcp_if_include ${CTRL_NIC} \
    --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
    --mca orte_base_help_aggregate 0 \
    -x LD_LIBRARY_PATH=${NCCL_PATH}:${LD_LIBRARY_PATH} \
    -x NCCL_NET_PLUGIN=${plugin_path} \
    -x NCCL_SOCKET_IFNAME=${CTRL_NIC} \
    -x GLOG_logtostderr=1 \
    -x GLOG_v=0 \
    -x NCCL_DEBUG=WARN \
    -x NCCL_DEBUG_SUBSYS=NET \
    -x NCCL_PROTO=${NCCL_PROTO} \
    -x NCCL_PXN_DISABLE=${NVLINK_OFF} \
    -x NCCL_P2P_DISABLE=${NVLINK_OFF} \
    -x NCCL_SHM_DISABLE=${NVLINK_OFF} \
    -x NCCL_NET_DISABLE=0 \
    -x NCCL_ALGO=${ALGO} \
    -x NCCL_MAX_NCHANNELS=${NUM_CHUNNEL} \
    -x NCCL_MIN_NCHANNELS=${NUM_CHUNNEL} \
    -x NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \
    -x NCCL_P2P_NET_CHUNKSIZE=${P2P_NET_CHUNKSIZE} \
    -x NCCL_BUFFSIZE=${BUFFSIZE} \
    -x NCCL_IB_QPS_PER_CONNECTION=${NUM_QPS_PER_CONNECTION} \
    -x NCCL_IB_SPLIT_DATA_ON_QPS=${SPLIT_DATA_ON_QPS} \
    -x NCCL_IB_HCA=${HCA_NAMES} \
    -x NCCL_IB_PCI_RELAXED_ORDERING=1 \
    -x NCCL_IGNORE_CPU_AFFINITY=1 \
    -x NCCL_CROSS_NIC=0 \
    ${UCCL_HOME}/thirdparty/nccl-tests/build/${PROG_NAME} \
    -f 2 \
    -c 0 \
    --minbytes 1K --maxbytes 1G \
    --warmup_iters 20 --iters 20 \
    -g 1 -t ${NUM_GPUS_PER_PROC}