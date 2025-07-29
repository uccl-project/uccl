#!/bin/bash

export OMP_NUM_THREADS=1
export NCCL_MAX_NCHANNELS=32
export NCCL_MIN_NCHANNELS=32

# for block_size in 1024 2048 4096 8192 16384 32768
for block_size in 262144
do
    torchrun --standalone --nproc_per_node=8 \
        benchmark_nccl_alltoall.py \
        --block-size $block_size \
        --num-qo-heads 32 \
        --gqa-group-size 4 \
        --head-dim 128 \
        --num-iters 100
done