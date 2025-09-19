# !/bin/bash

source ../shared.sh

# Usage: ./run_nccl_test.sh [tcp|afxdp] [num of processes] [ens6|enp199s0]

TEST=${1:-tcp}
LIBNCCL_PATH="${UCCL_HOME}/thirdparty/nccl/build/lib/libnccl.so"
# all_gather_perf  all_reduce_perf  alltoall_perf  broadcast_perf  gather_perf
# hypercube_perf  reduce_perf  reduce_scatter_perf  scatter_perf  sendrecv_perf
PROG_NAME=all_reduce_perf
NUM_PROCS=${2:-4}
NIC=${3:-ens6} # enp199s0 for g4.metal
NODES=$(get_nodes "../nodes.txt")

echo "Running test: ${TEST}, ${PROG_NAME}, ${NUM_PROCS} processes, NIC ${NIC}, ${NODES}"

if [ "$TEST" = "tcp" ]; then

    # PLUGIN_PATH="${UCCL_HOME}/thirdparty/nccl/ext-net/google-fastsocket/libnccl-net.so"
    PLUGIN_PATH="/opt/aws-ofi-nccl/lib/libnccl-net.so"

    mpirun --bind-to none -np ${NUM_PROCS} -N 1 --host ${NODES} \
        --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
        --mca orte_base_help_aggregate 0 \
        --mca btl_tcp_if_include ${NIC} \
        -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
        -x NCCL_DEBUG=INFO \
        -x NCCL_SOCKET_NTHREADS=4 \
        -x NCCL_NSOCKS_PERTHREAD=2 \
        -x NCCL_MAX_NCHANNELS=8 \
        -x NCCL_MIN_NCHANNELS=8 \
        ${UCCL_HOME}/thirdparty/nccl-tests/build/${PROG_NAME} \
        -b 1K -e 1G -f 2 -g 1 -w 100 -n 100 -t 1

        # -x NCCL_SOCKET_IFNAME=${NIC} \

        # Does not help and causes perf degradation for large sizes. 
        # -x NCCL_SOCKET_NTHREADS=16 \
        # -x NCCL_NSOCKS_PERTHREAD=4 \
        # -x NCCL_MAX_NCHANNELS=16 \
        # -x NCCL_MIN_NCHANNELS=16 \

        # -x NCCL_P2P_DISABLE=1 \
        # -x NCCL_SHM_DISABLE=1 \
        # -x NCCL_NET_DISABLE=0 \

elif [ "$TEST" = "afxdp" ]; then

    # Clear existing files for all ranks
    for ((rank = 0; rank < NUM_PROCS; rank++)); do
        >"output_rank_$rank.log" # Truncate or create empty file
    done

    PLUGIN_PATH="${UCCL_HOME}/collective/afxdp/libnccl-net.so"

    mpirun --bind-to none -np ${NUM_PROCS} -N 1 --host ${NODES} \
        --tag-output --merge-stderr-to-stdout \
        --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
        --mca orte_base_help_aggregate 0 \
        --mca btl_tcp_if_include ${NIC} \
        -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
        -x NCCL_DEBUG=INFO \
        -x UCCL_ENGINE_QUIET=1 \
        -x GLOG_logtostderr=1 \
        -x NCCL_SOCKET_NTHREADS=4 \
        -x NCCL_NSOCKS_PERTHREAD=2 \
        -x NCCL_MAX_NCHANNELS=8 \
        -x NCCL_MIN_NCHANNELS=8 \
        ${UCCL_HOME}/thirdparty/nccl-tests/build/${PROG_NAME} \
        -b 1K -e 1G -f 2 -g 1 -w 1 -n 100 -t 1 \
        2>&1 | while read -r line; do
        # Extract rank from the format [1,2]
        if [[ "$line" =~ ^\[[0-9]+,([0-9]+)\](.+) ]]; then
            RANK=${BASH_REMATCH[1]}                   # Extract second number as rank
            CONTENT=${BASH_REMATCH[2]}                # Extract the remaining content
            echo "Rank $RANK: $CONTENT"               # Print to terminal
            echo "$CONTENT" >>"output_rank_$RANK.log" # Append to rank-specific file
        else
            echo "$line" # Print untagged output to the terminal
        fi

        # gdb -ex run --args \
    done
else
    echo "Invalid test: ${TEST}"
    exit 1
fi
