# !/bin/bash

UCCL_HOME="/opt/uccl"

LIBNCCL_PATH="${UCCL_HOME}/nccl/build/lib/libnccl.so"

# all_gather_perf  all_reduce_perf  alltoall_perf  broadcast_perf  gather_perf
# hypercube_perf  reduce_perf  reduce_scatter_perf  scatter_perf  sendrecv_perf
PROG_NAME=all_reduce_perf

NODES=""
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^# ]] && continue
    NODES+="$line,"
done < ../nodes.txt

# PLUGIN_PATH="${UCCL_HOME}/nccl/ext-net/google-fastsocket/libnccl-net.so"
# PLUGIN_PATH="/opt/aws-ofi-nccl/lib/libnccl-net.so"

# mpirun -np 4 -N 1 --host ${NODES} \
# --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
# --mca orte_base_help_aggregate 0 \
# --mca btl_tcp_if_include ens5 \
# -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
# -x NCCL_DEBUG=INFO \
# ${UCCL_HOME}/nccl-tests/build/${PROG_NAME} \
# -b 1024 -e 1048576 -f 2 -g 1 -w 100 -n 100

PLUGIN_PATH="${UCCL_HOME}/afxdp/libnccl-net.so"

mpirun -np 4 -N 1 --host ${NODES} \
--mca plm_rsh_args "-o StrictHostKeyChecking=no" \
--mca orte_base_help_aggregate 0 \
--mca btl_tcp_if_include ens5 \
-x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
-x NCCL_DEBUG=INFO \
-x UCCL_ENGINE_QUIET=1 \
-x GLOG_logtostderr=1 \
-x NCCL_SOCKET_NTHREADS=1 \
-x NCCL_NSOCKS_PERTHREAD=1 \
-x NCCL_MAX_NCHANNELS=1 \
${UCCL_HOME}/nccl-tests/build/${PROG_NAME} \
-b 1024 -e 1048576 -f 2 -g 1 -w 100 -n 100
