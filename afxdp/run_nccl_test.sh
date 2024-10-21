# !/bin/bash

UCCL_HOME="/home/ubuntu/uccl"

LIBNCCL_PATH="${UCCL_HOME}/nccl/build/lib/libnccl.so"

# PLUGIN_PATH="${UCCL_HOME}/nccl/ext-net/google-fastsocket/libnccl-net.so"

# PLUGIN_PATH="/opt/aws-ofi-nccl/lib/libnccl-net.so"

# mpirun -np 2 -N 1 --host 172.31.16.220,172.31.30.246 \
#     --mca orte_base_help_aggregate 0 \
#     --mca btl_tcp_if_include ens5 \
#     -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
#     -x NCCL_DEBUG=INFO \
#     ${UCCL_HOME}/nccl-tests/build/all_reduce_perf \
#     -b 1024000 -e 1024000 -f 2 -g 1 -w 100

PLUGIN_PATH="${UCCL_HOME}/afxdp/libnccl-net.so"

mpirun -np 2 -N 1 --host 172.31.16.220,172.31.30.246 \
    --mca orte_base_help_aggregate 0 \
    --mca btl_tcp_if_include ens5 \
    -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
    -x NCCL_DEBUG=INFO \
    -x GLOG_logtostderr=1 \
    -x NCCL_SOCKET_NTHREADS=1 \
    -x NCCL_NSOCKS_PERTHREAD=1 \
    -x NCCL_MAX_NCHANNELS=1 \
    ${UCCL_HOME}/nccl-tests/build/all_reduce_perf \
    -b 1024000 -e 1024000 -f 2 -g 1 -w 100


#    -x NCCL_P2P_NET_CHUNKSIZE=16777216 \
#    -x NCCL_BUFFSIZE=16777216 \
