# !/bin/bash

source ../scripts/shared.sh

TEST=${1:-srd}

NV_LINK_DISABLE=1
UCCL_QUITE=1
PROG=ddp.py
DEVICES=0,1,2,3,4,5,6,7
NUM_DEVS=8

CHANNELS=4
CHANNELS_NET_PEER=4
CHUNK_SIZE=131072
BUFFSIZE=1048576
if [ "$TEST" = "srd" ]; then
    CHANNELS=4
    CHANNELS_NET_PEER=4
    CHUNK_SIZE=524288
    BUFFSIZE=8388608
fi

if [ "$TEST" = "srd" ]; then
    LD_PRELOAD="${UCCL_HOME}/thirdparty/nccl/build/lib/libnccl.so" \
        NCCL_NET_PLUGIN="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so" \
        NCCL_PROTO=Simple \
        NCCL_P2P_DISABLE=${NV_LINK_DISABLE} \
        NCCL_SHM_DISABLE=${NV_LINK_DISABLE} \
        NCCL_NET_DISABLE=0 \
        CUDA_VISIBLE_DEVICES=${DEVICES} \
        NCCL_MAX_NCHANNELS=${CHANNELS} \
        NCCL_MIN_NCHANNELS=${CHANNELS} \
        NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \
        NCCL_P2P_NET_CHUNKSIZE=${CHUNK_SIZE} \
        NCCL_BUFFSIZE=${BUFFSIZE} \
        torchrun --nproc_per_node=${NUM_DEVS} ${PROG} --batch_size 128 --epochs 10

elif [ "$TEST" = "ud" ]; then
    LD_PRELOAD="${UCCL_HOME}/thirdparty/nccl-sg/build/lib/libnccl.so" \
        NCCL_NET_PLUGIN="${UCCL_HOME}/efa/libnccl-net-efa.so" \
        NCCL_PROTO=Simple \
        NCCL_P2P_DISABLE=${NV_LINK_DISABLE} \
        NCCL_SHM_DISABLE=${NV_LINK_DISABLE} \
        NCCL_NET_DISABLE=0 \
        CUDA_VISIBLE_DEVICES=${DEVICES} \
        NCCL_MAX_NCHANNELS=${CHANNELS} \
        NCCL_MIN_NCHANNELS=${CHANNELS} \
        NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \
        NCCL_P2P_NET_CHUNKSIZE=${CHUNK_SIZE} \
        NCCL_BUFFSIZE=${BUFFSIZE} \
        NCCL_NET_GDR_LEVEL=SYS \
        CUDA_MODULE_LOADING=EAGER \
        NCCL_TOPO_FILE=${UCCL_HOME}/efa/p4d-24xl-topo.xml \
        NCCL_PXN_DISABLE=1 \
        UCCL_ENGINE_QUIET=${UCCL_QUITE} \
        GLOG_logtostderr=0 \
        torchrun --nproc_per_node=${NUM_DEVS} ${PROG} --batch_size 128 --epochs 10
fi