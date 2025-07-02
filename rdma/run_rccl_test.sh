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
    # plugin_path="${UCCL_HOME}/rdma/librccl-net-uccl.so"
    plugin_path=`python -c "import uccl; print(uccl.rccl_plugin_path())"`
    echo "plugin_path: ${plugin_path}"
else
    echo "Unsupport benchmark type."
    exit 1
fi

NVLINK_ON=1

NVLINK_OFF=$((1 - NVLINK_ON))

HCA_NAMES="rdma0:1,rdma1:1,rdma2:1,rdma3:1,rdma4:1,rdma5:1,rdma6:1,rdma7:1"
CTRL_NIC="cni0"

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
    -x NCCL_IB_QPS_PER_CONNECTION=1 \
    -x NCCL_IB_SPLIT_DATA_ON_QPS=1 \
    -x HIP_VISIBLE_DEVICES=0,1,3,4,5,6,7 \
    -x NCCL_IB_HCA=${HCA_NAMES} \
    -x NCCL_SOCKET_IFNAME=${CTRL_NIC} \
    -x UCCL_NUM_ENGINES=4 \
    -x UCCL_PORT_ENTROPY=1 \
    -x UCCL_CHUNK_SIZE_KB=512 \
    ${UCCL_HOME}/thirdparty/rccl-tests/build/all_reduce_perf \
    -b 1K -e 1G -f 2 -w 5 -n 20 -c 1 -g 1 -t 7 |&
    tee alltoall_debug_${TEST}.log

# alltoall_perf, all_reduce_perf
# -x NCCL_DMABUF_ENABLE=1 \
# -x NCCL_NET_GDR_LEVEL=SYS \
# -x NCCL_DEBUG_SUBSYS=NET \
# -x NCCL_DEBUG=INFO \
