# !/bin/bash

source ../scripts/shared.sh

if [[ -z "${CONDA_LIB_HOME}" ]]; then
  echo "CONDA_LIB_HOME is not set or is empty"
  exit 1
else
  echo "CONDA_LIB_HOME is set to: ${CONDA_LIB_HOME}"
fi

NODEFILE=${UCCL_HOME}/scripts/node_ips/amd.txt

TEST=${1:-uccl}

if [ "$TEST" = "rccl" ]; then
    echo "Running RCCL test"
    plugin_path=""
elif [ "$TEST" = "uccl" ]; then
    echo "Running UCCL test"
    plugin_path="${UCCL_HOME}/rdma/librccl-net-uccl.so"
else
    echo "Unsupport benchmark type."
    exit 1
fi

NVLINK_ON=0

NVLINK_OFF=$((1 - NVLINK_ON))

mpirun --prefix /usr/local/bin/ompi --bind-to none -np 2 -N 1 --hostfile $NODEFILE --map-by ppr:1:node \
    -x LD_LIBRARY_PATH=${UCCL_HOME}/thirdparty/rccl/build/release:${CONDA_LIB_HOME}:/opt/rocm-6.3.1/lib:${LD_LIBRARY_PATH} \
    -x NCCL_NET_PLUGIN=${plugin_path} \
    -x GLOG_v=0 \
    -x NCCL_P2P_DISABLE=${NVLINK_OFF} \
    -x NCCL_SHM_DISABLE=${NVLINK_OFF} \
    -x NCCL_IB_PCI_RELAXED_ORDERING=1 \
    -x NCCL_P2P_NET_CHUNKSIZE=524288 \
    -x NCCL_BUFFSIZE=8388608 \
    -x NCCL_MIN_NCHANNELS=32 \
    -x NCCL_MAX_NCHANNELS=32 \
    -x NCCL_NCHANNELS_PER_NET_PEER=1 \
    -x NCCL_IB_QPS_PER_CONNECTION=4 \
    -x NCCL_IB_SPLIT_DATA_ON_QPS=1 \
    -x HIP_VISIBLE_DEVICES=1,2,0,5 \
    -x NCCL_IB_HCA="rdma0:1,rdma2:1,rdma3:1,rdma4:1" \
    -x NCCL_SOCKET_IFNAME="cni0" \
    -x UCCL_NUM_ENGINES=4 \
    -x UCCL_PORT_ENTROPY=8 \
    -x UCCL_CHUNK_SIZE_KB=128 \
    ${UCCL_HOME}/thirdparty/rccl-tests/build/alltoall_perf \
    -b 1K -e 1G -f 2 -w 5 -n 20 -c 1 -g 1 -t 4 |&
    tee alltoall_debug_${TEST}.log

# alltoall_perf, all_reduce_perf
# -x NCCL_DMABUF_ENABLE=1 \
# -x NCCL_NET_GDR_LEVEL=SYS \
# -x NCCL_DEBUG_SUBSYS=NET \
# -x NCCL_DEBUG=INFO \
