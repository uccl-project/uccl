# !/bin/bash

PLUGIN_PATH="/opt/aws-ofi-nccl/lib/libnccl-net.so" # for p4d
# PLUGIN_PATH="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so" # for p3d

# mpirun --bind-to none -np 4 -N 1 --host 172.31.37.118,172.31.45.5,172.31.43.3,172.31.46.175 \
#     --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
#     --mca orte_base_help_aggregate 0 \
#     -x LD_PRELOAD="/opt/uccl/nccl/build/lib/libnccl.so" \
#     -x NCCL_NET_PLUGIN="${PLUGIN_PATH}" \
#     -x NCCL_DEBUG=INFO \
#     -x NCCL_P2P_DISABLE=1 \
#     -x NCCL_SHM_DISABLE=1 \
#     -x NCCL_NET_DISABLE=0 \
#     -x NCCL_P2P_NET_CHUNKSIZE=524288 \
#     -x NCCL_BUFFSIZE=8388608 \
#     /opt/uccl/nccl-tests/build/alltoall_perf \
#     -b 1K -e 1G -f 2 -g 1 -t 8
# -w 100 -n 100
# -x NCCL_SOCKET_NTHREADS=4 \
# -x NCCL_NSOCKS_PERTHREAD=2 \
# -x NCCL_MAX_NCHANNELS=8 \
# -x NCCL_MIN_NCHANNELS=8 \
# -x NCCL_IB_QPS_PER_CONNECTION=1 \
# -x CUDA_VISIBLE_DEVICES=0,2,4,6 \
# all_reduce_perf, alltoall_perf \
# -x NCCL_DEBUG=INFO \
# -x NCCL_TESTS_SPLIT_MASK=0x7 \
# -x OFI_NCCL_PROTOCOL=RDMA \
# -R 1

for i in 1 2 3 4; do
    echo "Run alltoall across $i p4d with NVLink"
    mpirun --bind-to none -np $i -N 1 --host 172.31.37.118,172.31.45.5,172.31.43.3,172.31.46.175 \
        --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
        --mca orte_base_help_aggregate 0 \
        -x LD_PRELOAD="/opt/uccl/nccl/build/lib/libnccl.so" \
        -x NCCL_NET_PLUGIN="${PLUGIN_PATH}" \
        -x NCCL_P2P_NET_CHUNKSIZE=524288 \
        -x NCCL_BUFFSIZE=8388608 \
        /opt/uccl/nccl-tests/build/alltoall_perf \
        -b 1K -e 1G -f 2 -g 1 -t 8
done

for i in 1 2 3 4; do
    echo "Run alltoall across $i p4d without NVLink"
    mpirun --bind-to none -np $i -N 1 --host 172.31.37.118,172.31.45.5,172.31.43.3,172.31.46.175 \
        --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
        --mca orte_base_help_aggregate 0 \
        -x LD_PRELOAD="/opt/uccl/nccl/build/lib/libnccl.so" \
        -x NCCL_NET_PLUGIN="${PLUGIN_PATH}" \
        -x NCCL_P2P_DISABLE=1 \
        -x NCCL_SHM_DISABLE=1 \
        -x NCCL_NET_DISABLE=0 \
        -x NCCL_P2P_NET_CHUNKSIZE=524288 \
        -x NCCL_BUFFSIZE=8388608 \
        /opt/uccl/nccl-tests/build/alltoall_perf \
        -b 1K -e 1G -f 2 -g 1 -t 8
done

# for i in 1 2 3 4; do
#     echo "Run alltoall across $i p4d without NVLink but with SHM (PCIe)"
#     mpirun --bind-to none -np $i -N 1 --host 172.31.37.118,172.31.45.5,172.31.43.3,172.31.46.175 \
#         --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
#         --mca orte_base_help_aggregate 0 \
#         -x LD_PRELOAD="/opt/uccl/nccl/build/lib/libnccl.so" \
#         -x NCCL_NET_PLUGIN="${PLUGIN_PATH}" \
#         -x NCCL_P2P_DISABLE=1 \
#         -x NCCL_SHM_DISABLE=0 \
#         -x NCCL_NET_DISABLE=0 \
#         -x NCCL_P2P_NET_CHUNKSIZE=524288 \
#         -x NCCL_BUFFSIZE=8388608 \
#         /opt/uccl/nccl-tests/build/alltoall_perf \
#         -b 1K -e 1G -f 2 -g 1 -t 8
# done

# # Test if the tput drop is caused by two GPUs sharing the same NIC---No.
# for i in 1 2 3 4; do
#     echo "Run alltoall across $i p4d without NVLink"
#     mpirun --bind-to none -np $i -N 1 --host 172.31.37.118,172.31.45.5,172.31.43.3,172.31.46.175 \
#         --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
#         --mca orte_base_help_aggregate 0 \
#         -x LD_PRELOAD="/opt/uccl/nccl/build/lib/libnccl.so" \
#         -x NCCL_NET_PLUGIN="${PLUGIN_PATH}" \
#         -x NCCL_P2P_DISABLE=1 \
#         -x NCCL_SHM_DISABLE=1 \
#         -x NCCL_NET_DISABLE=0 \
#         -x NCCL_P2P_NET_CHUNKSIZE=524288 \
#         -x NCCL_BUFFSIZE=8388608 \
#         -x CUDA_VISIBLE_DEVICES=0,2,4,6 \
#         /opt/uccl/nccl-tests/build/alltoall_perf \
#         -b 1K -e 1G -f 2 -g 1 -t 4
# done
