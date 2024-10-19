# !/bin/bash

UCCL_HOME="/home/ubuntu/uccl"

LIBNCCL_PATH="${UCCL_HOME}/nccl/build/lib/libnccl.so"

PLUGIN_PATH="${UCCL_HOME}/afxdp/libnccl-net.so"

# Note that LD_PRELOAD does not work with sudo, so we have to register the nccl and plugin share libraries in /etc/ld.so.conf.d/nccl.conf
sudo mpirun --allow-run-as-root -np 2 -N 1 --host 172.31.18.199,172.31.30.246 \
    --mca orte_base_help_aggregate 0 \
    --mca btl_tcp_if_include ens6 \
    -x NCCL_DEBUG=INFO \
    sudo -E ${UCCL_HOME}/nccl-tests/build/all_reduce_perf \
    -b 1024 -e 1M -f 2 -g 1


# PLUGIN_PATH="/opt/aws-ofi-nccl/lib/libnccl-net.so"
# PLUGIN_PATH="${UCCL_HOME}/nccl/ext-net/google-fastsocket/libnccl-net.so"

# mpirun -np 2 -N 1 --host 172.31.18.199,172.31.30.246 \
#     --mca orte_base_help_aggregate 0 \
#     --mca btl_tcp_if_include ens6 \
#     -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
#     -x NCCL_DEBUG=INFO \
#     ${UCCL_HOME}/nccl-tests/build/all_reduce_perf \
#     -b 1024 -e 1M -f 2 -g 1
