# !/bin/bash

mpirun --bind-to none --output-filename output.log -np 2 -N 1 --host 172.31.37.118,172.31.45.5,172.31.43.3,172.31.46.175 \
    --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
    --mca orte_base_help_aggregate 0 \
    -x LD_PRELOAD="/opt/uccl/nccl/build/lib/libnccl.so" \
    -x NCCL_NET_PLUGIN="/opt/aws-ofi-nccl/lib/libnccl-net.so" \
    -x NCCL_DEBUG=INFO \
    /opt/uccl/nccl-tests/build/alltoall_perf \
    -b 1K -e 1G -f 2 -g 1 -t 8

    # -w 100 -n 100
    # -x NCCL_P2P_DISABLE=1 \
    # -x NCCL_SHM_DISABLE=1 \
    # -x NCCL_NET_DISABLE=0 \
    # -x NCCL_SOCKET_NTHREADS=4 \
    # -x NCCL_NSOCKS_PERTHREAD=2 \
    # -x NCCL_MAX_NCHANNELS=8 \
    # -x NCCL_MIN_NCHANNELS=8 \
    # -x CUDA_VISIBLE_DEVICES=0,2,4,6 \
    # all_reduce_perf, alltoall_perf \

# for i in 1 2 3 4; do
#     echo "Run alltoall across $i p4d with NVLink"
#     mpirun --bind-to none -np $i -N 1 --host 172.31.37.118,172.31.45.5,172.31.43.3,172.31.46.175 \
#         --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
#         --mca orte_base_help_aggregate 0 \
#         -x LD_PRELOAD="/opt/uccl/nccl/build/lib/libnccl.so" \
#         -x NCCL_NET_PLUGIN="/opt/aws-ofi-nccl/lib/libnccl-net.so" \
#         /opt/uccl/nccl-tests/build/alltoall_perf \
#         -b 1K -e 1G -f 2 -g 1 -t 8
# done

# for i in 1 2 3 4; do
#     echo "Run alltoall across $i p4d without NVLink"
#     mpirun --bind-to none -np $i -N 1 --host 172.31.37.118,172.31.45.5,172.31.43.3,172.31.46.175 \
#         --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
#         --mca orte_base_help_aggregate 0 \
#         -x LD_PRELOAD="/opt/uccl/nccl/build/lib/libnccl.so" \
#         -x NCCL_NET_PLUGIN="/opt/aws-ofi-nccl/lib/libnccl-net.so" \
#         -x NCCL_P2P_DISABLE=1 \
#         -x NCCL_SHM_DISABLE=1 \
#         -x NCCL_NET_DISABLE=0 \
#         /opt/uccl/nccl-tests/build/alltoall_perf \
#         -b 1K -e 1G -f 2 -g 1 -t 8
# done

# for i in 1 2 3 4; do
#     echo "Run alltoall across $i p4d without NVLink but with SHM (PCIe)"
#     mpirun --bind-to none -np $i -N 1 --host 172.31.37.118,172.31.45.5,172.31.43.3,172.31.46.175 \
#         --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
#         --mca orte_base_help_aggregate 0 \
#         -x LD_PRELOAD="/opt/uccl/nccl/build/lib/libnccl.so" \
#         -x NCCL_NET_PLUGIN="/opt/aws-ofi-nccl/lib/libnccl-net.so" \
#         -x NCCL_P2P_DISABLE=1 \
#         -x NCCL_SHM_DISABLE=0 \
#         -x NCCL_NET_DISABLE=0 \
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
#         -x NCCL_NET_PLUGIN="/opt/aws-ofi-nccl/lib/libnccl-net.so" \
#         -x NCCL_P2P_DISABLE=1 \
#         -x NCCL_SHM_DISABLE=1 \
#         -x NCCL_NET_DISABLE=0 \
#         -x CUDA_VISIBLE_DEVICES=0,2,4,6 \
#         /opt/uccl/nccl-tests/build/alltoall_perf \
#         -b 1K -e 1G -f 2 -g 1 -t 4
# done
