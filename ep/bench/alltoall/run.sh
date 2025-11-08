#!/bin/bash

total_nodes=$1
node_rank=$2

torchrun --nnodes=$total_nodes --nproc_per_node=8 --node_rank=$node_rank --master_addr=172.31.36.62 --master_port=29500 launch.py