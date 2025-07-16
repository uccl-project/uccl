# !/bin/bash

source ../scripts/shared.sh

# Run nccl-tests with multiple processes
# Usage: ./run_nccl_test_mp.sh [srd|ud] [Total Processes/Ranks/GPUs] [Benchtype, 0: alltoall, 1: allgather, 2: multi-allreduce]

# Visible GPUs to application.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Disable NVLink.
NV_LINK_DISABLE=1
MULTI_GROUP=0
NIC=ens32
# Processes/Ranks/GPUs per node.
PROCS_PER_NODE=8

TEST=${1:-srd}
NUM_PROCS=${2:-32}
PROG_NAME=${3:-0}

# all_gather_perf  all_reduce_perf  alltoall_perf  broadcast_perf  gather_perf
# hypercube_perf  reduce_perf  reduce_scatter_perf  scatter_perf  sendrecv_perf
if [ "$PROG_NAME" -eq 0 ]; then
    PROG_NAME="alltoall_perf"
elif [ "$PROG_NAME" -eq 1 ]; then
    PROG_NAME="all_gather_perf"
elif [ "$PROG_NAME" -eq 2 ]; then
    PROG_NAME="all_reduce_perf"
    MULTI_GROUP=0x7
else
    echo "Invalid program name: ${PROG_NAME}"
    exit 1
fi

CHANNELS=8
CHANNELS_NET_PEER=4

# UCCL optimal parameters. Yang: for allreduce with nvlink, we need to use larger buffer to catch up with NCCL with larger buffers, and avoid outliers.
CHUNK_SIZE=131072
BUFFSIZE=1048576

if [ "$TEST" = "srd" ]; then
    # SRD optimal parameters.
    CHUNK_SIZE=524288
    BUFFSIZE=8388608
fi

NODES=$(get_nodes "../scripts/node_ips/p4d.txt")
echo "Running test: ${TEST}, ${PROG_NAME}, ${NUM_PROCS} processes, NIC ${NIC}, uccl_quite ${UCCL_QUITE}, ${NODES}, ${CHANNELS} channels."

if [ "$TEST" = "srd" ]; then

    for ((rank = 0; rank < NUM_PROCS; rank++)); do
        >"nccl_test_outputs/output_rank_$rank.log"
    done

    LIBNCCL_PATH="${UCCL_HOME}/thirdparty/nccl/build/lib/libnccl.so"
    PLUGIN_PATH="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so"

    mpirun --bind-to none -np ${NUM_PROCS} -N ${PROCS_PER_NODE} --hostfile hosts \
        --tag-output --merge-stderr-to-stdout \
        --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
        --mca orte_base_help_aggregate 0 \
        --mca btl_tcp_if_include ${NIC} \
        -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
        -x NCCL_PROTO=Simple \
        -x NCCL_P2P_DISABLE=${NV_LINK_DISABLE} \
        -x NCCL_SHM_DISABLE=${NV_LINK_DISABLE} \
        -x NCCL_NET_DISABLE=0 \
        -x NCCL_MAX_NCHANNELS=${CHANNELS} \
        -x NCCL_MIN_NCHANNELS=${CHANNELS} \
        -x CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        -x NCCL_TESTS_SPLIT_MASK=${MULTI_GROUP} \
        -x NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \
        -x NCCL_P2P_NET_CHUNKSIZE=${CHUNK_SIZE} \
        -x NCCL_BUFFSIZE=${BUFFSIZE} \
        ${UCCL_HOME}/thirdparty/nccl-tests/build/${PROG_NAME} \
        -b 1K -e 1G -f 2 -c 1 -w 5 -n 10 -t 1 -g 1 \
        2>&1 | while read -r line; do
        if [[ "$line" =~ ^\[[0-9]+,([0-9]+)\](.+) ]]; then
            RANK=${BASH_REMATCH[1]}
            CONTENT=${BASH_REMATCH[2]}
            echo "Rank $RANK: $CONTENT"
            echo "$CONTENT" >>"nccl_test_outputs/output_rank_$RANK.log"
        else
            echo "$line"
        fi
    done

elif [ "$TEST" = "ud" ]; then

    for ((rank = 0; rank < NUM_PROCS; rank++)); do
        >"nccl_test_outputs/output_rank_$rank.log"
    done

    LIBNCCL_PATH="${UCCL_HOME}/thirdparty/nccl-sg/build/lib/libnccl.so"
    PLUGIN_PATH="${UCCL_HOME}/efa/libnccl-net-efa.so"
    # LIBNCCL_PATH=`python -c "import uccl; print(uccl.efa_nccl_path())"`
    # PLUGIN_PATH=`python -c "import uccl; print(uccl.efa_plugin_path())"`
    echo "LIBNCCL_PATH: ${LIBNCCL_PATH}"
    echo "PLUGIN_PATH: ${PLUGIN_PATH}"

    mpirun --bind-to none -np ${NUM_PROCS} -N ${PROCS_PER_NODE} --hostfile hosts \
        --tag-output --merge-stderr-to-stdout \
        --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
        --mca orte_base_help_aggregate 0 \
        --mca btl_tcp_if_include ${NIC} \
        -x NCCL_NET_PLUGIN=${PLUGIN_PATH} \
        -x LD_PRELOAD="${LIBNCCL_PATH}" \
        -x NCCL_PROTO=Simple \
        -x NCCL_P2P_DISABLE=${NV_LINK_DISABLE} \
        -x NCCL_SHM_DISABLE=${NV_LINK_DISABLE} \
        -x NCCL_NET_DISABLE=0 \
        -x GLOG_logtostderr=0 \
        -x CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        -x NCCL_TESTS_SPLIT_MASK=${MULTI_GROUP} \
        -x NCCL_MAX_NCHANNELS=${CHANNELS} \
        -x NCCL_MIN_NCHANNELS=${CHANNELS} \
        -x NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \
        -x NCCL_NET_GDR_LEVEL=SYS \
        -x NCCL_P2P_NET_CHUNKSIZE=${CHUNK_SIZE} \
        -x NCCL_BUFFSIZE=${BUFFSIZE} \
        -x NCCL_GDRCOPY_ENABLE=1 \
        -x NCCL_GDRCOPY_FLUSH_ENABLE=1 \
        -x NCCL_GDRCOPY_SYNC_ENABLE=0 \
        -x NCCL_GDRCOPY_FIFO_ENABLE=0 \
        -x CUDA_MODULE_LOADING=EAGER \
        -x NCCL_TOPO_FILE=${UCCL_HOME}/efa/p4d-24xl-topo.xml \
        -x NCCL_PXN_DISABLE=1 \
        -x UCCL_ENGINE_QUIET=1 \
        ${UCCL_HOME}/thirdparty/nccl-tests/build/${PROG_NAME} \
        -b 1K -e 1G -f 2 -c 1 -w 5 -n 10 -t 1 -g 1 \
        2>&1 | while read -r line; do
        if [[ "$line" =~ ^\[[0-9]+,([0-9]+)\](.+) ]]; then
            RANK=${BASH_REMATCH[1]}
            CONTENT=${BASH_REMATCH[2]}
            echo "Rank $RANK: $CONTENT"
            echo "$CONTENT" >>"nccl_test_outputs/output_rank_$RANK.log"
        else
            echo "$line"
        fi
    done
else
    echo "Invalid test: ${TEST}"
    exit 1
fi
