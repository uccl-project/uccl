# !/bin/bash

source ../shared.sh

# Usage: ./run_nccl_test.sh [tcp|afxdp] [num of processes] [ens6|enp199s0]

TEST=${1:-tcp}
UCCL_HOME="/opt/uccl"
LIBNCCL_PATH="${UCCL_HOME}/nccl/build/lib/libnccl.so"
# all_gather_perf  all_reduce_perf  alltoall_perf  broadcast_perf  gather_perf
# hypercube_perf  reduce_perf  reduce_scatter_perf  scatter_perf  sendrecv_perf
PROG_NAME=all_reduce_perf
NUM_PROCS=${2:-4}
NIC=${3:-ens6} # enp199s0 for g4.metal
NODES=$(get_nodes "../nodes.txt")

echo "Running test: ${TEST}, ${PROG_NAME}, ${NUM_PROCS} processes, NIC ${NIC}, ${NODES}"

if [ "$TEST" = "tcp" ]; then

    # PLUGIN_PATH="${UCCL_HOME}/nccl/ext-net/google-fastsocket/libnccl-net.so"
    PLUGIN_PATH="/opt/aws-ofi-nccl/lib/libnccl-net.so"

    mpirun -np ${NUM_PROCS} -N 1 --host ${NODES} \
        --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
        --mca orte_base_help_aggregate 0 \
        --mca btl_tcp_if_include ${NIC} \
        -x NCCL_SOCKET_NTHREADS=16 \
        -x NCCL_NSOCKS_PERTHREAD=4 \
        -x NCCL_IGNORE_CPU_AFFINITY=1 \
        -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
        -x NCCL_DEBUG=INFO \
        ${UCCL_HOME}/nccl-tests/build/${PROG_NAME} \
        -b 1K -e 16M -f 2 -g 1 -w 100 -n 100 -t 1

elif [ "$TEST" = "afxdp" ]; then

    # Clear existing files for all ranks
    for ((rank = 0; rank < NUM_PROCS; rank++)); do
        >"output_rank_$rank.log" # Truncate or create empty file
    done

    PLUGIN_PATH="${UCCL_HOME}/afxdp/libnccl-net.so"

    mpirun -np ${NUM_PROCS} -N 1 --tag-output --merge-stderr-to-stdout --host ${NODES} \
        --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
        --mca orte_base_help_aggregate 0 \
        --mca btl_tcp_if_include ${NIC} \
        -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
        -x NCCL_DEBUG=INFO \
        -x UCCL_ENGINE_QUIET=1 \
        -x GLOG_logtostderr=1 \
        -x NCCL_SOCKET_NTHREADS=1 \
        -x NCCL_NSOCKS_PERTHREAD=1 \
        -x NCCL_MAX_NCHANNELS=1 \
        -x NCCL_IGNORE_CPU_AFFINITY=1 \
        ${UCCL_HOME}/nccl-tests/build/${PROG_NAME} \
        -b 1K -e 16M -f 2 -g 1 -w 100 -n 100 -t 1 \
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
    done
else
    echo "Invalid test: ${TEST}"
    exit 1
fi
