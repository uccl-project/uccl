# !/bin/bash

# torchrun --nnodes=$1 --nproc_per_node=8 --node_rank=$2 \
#     --master_addr=172.31.36.62 --master_port=12355 \
#     test_low_latency.py --num-tokens=128 \
#     --hidden=7168 --num-topk=8 --num-experts=288

torchrun --nnodes=$1 --nproc_per_node=8 --node_rank=$2 \
    --master_addr=172.31.36.62 --master_port=12355 \
    test_internode.py  --num-tokens=4096 \
    --hidden=7168 --num-topk=8 --num-experts=288 --test-ll-compatibility
