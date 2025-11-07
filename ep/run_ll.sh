#!/bin/bash

node_rank=$1

echo "Running low latency benchmark on node rank: $node_rank"

torchrun --nnodes=4 --nproc_per_node=8 --node_rank=$node_rank \
    --master_addr=172.31.36.62 --master_port=12355 \
    bench/test_low_latency.py --num-tokens=128 \
    --hidden=7168 --num-topk=1 --num-experts=256