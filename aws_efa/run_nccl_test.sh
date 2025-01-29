# !/bin/bash

mpirun --bind-to none -np 4 -N 1 --host 172.31.37.118,172.31.45.5,172.31.43.3,172.31.46.175 \
        --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
        --mca orte_base_help_aggregate 0 \
        -x LD_PRELOAD="/opt/uccl/nccl/build/lib/libnccl.so" \
        -x NCCL_NET_PLUGIN="/opt/aws-ofi-nccl/lib/libnccl-net.so" \
        -x NCCL_DEBUG=INFO \
        -x NCCL_P2P_DISABLE=1 \
        -x NCCL_SHM_DISABLE=1 \
        -x NCCL_NET_DISABLE=0 \
        /opt/uccl/nccl-tests/build/alltoall_perf \
        -b 1K -e 1G -f 2 -g 1 -t 4
        # -w 100 -n 100

        # -x NCCL_SOCKET_NTHREADS=4 \
        # -x NCCL_NSOCKS_PERTHREAD=2 \
        # -x NCCL_MAX_NCHANNELS=8 \
        # -x NCCL_MIN_NCHANNELS=8 \
        # all_reduce_perf, alltoall_perf \
