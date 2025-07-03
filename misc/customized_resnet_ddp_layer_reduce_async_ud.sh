#!/bin/bash

# Set strict error handling
set -e
set -x

# Configuration variables
UCCL_HOME="/opt/uccl_rdma_mp"
NV_LINK_DISABLE=1
CHANNELS=4
CHANNELS_NET_PEER=4
CHUNK_SIZE=131072
BUFFSIZE=1048576

# Default parameters for training
BATCH_SIZE=128
EPOCHS=10
LEARNING_RATE=0.1

# Memory settings for EFA
sudo sysctl -w vm.max_map_count=1048576
sudo sysctl -w vm.nr_hugepages=2048

# Clean up any existing RDMA resources
sudo rmmod ib_uverbs || true
sudo modprobe ib_uverbs

# Environment variables for UCCL
export LD_PRELOAD="${UCCL_HOME}/nccl/build/lib/libnccl.so"
export NCCL_NET_PLUGIN="${UCCL_HOME}/efa/libnccl-net-efa.so"
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
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_TOPO_FILE=${UCCL_HOME}/efa/p4d-24xl-topo.xml
export UCCL_ENGINE_QUIET=1
export GLOG_logtostderr=0

# Additional settings for async communication
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_LAUNCH_MODE=PARALLEL

# Run the training script
echo "Starting training with async layer-wise reduction..."
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    --node_rank=0 \
    --nnodes=1 \
    customized_resnet_ddp_layer_reduce_async.py \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE}

echo "Training completed!" 