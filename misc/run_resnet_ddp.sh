#!/bin/bash

# Set strict error handling
set -e
set -x

# Configuration variables
UCCL_HOME="/opt/uccl_rdma"
NV_LINK_DISABLE=1
CHANNELS=4
CHANNELS_NET_PEER=4
CHUNK_SIZE=524288
BUFFSIZE=8388608

# Default parameters for training
BATCH_SIZE=128
EPOCHS=10
LEARNING_RATE=0.1

# Environment variables for NCCL
export NCCL_NET_PLUGIN="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so"
export NCCL_DEBUG=INFO
export NCCL_PROTO=Simple
export NCCL_P2P_DISABLE=${NV_LINK_DISABLE}
export NCCL_SHM_DISABLE=${NV_LINK_DISABLE}
export NCCL_NET_DISABLE=0
export NCCL_MAX_NCHANNELS=${CHANNELS}
export NCCL_MIN_NCHANNELS=${CHANNELS}
export CUDA_VISIBLE_DEVICES=0,2,4,6
export NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER}
export NCCL_P2P_NET_CHUNKSIZE=${CHUNK_SIZE}
export NCCL_BUFFSIZE=${BUFFSIZE}

# Memory settings for EFA
sudo sysctl -w vm.max_map_count=1048576
sudo sysctl -w vm.nr_hugepages=2048

# Run the training script with environment variable
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    --node_rank=0 \
    --nnodes=1 \
    resnet_ddp.py \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE}