# !/bin/bash

source ../../scripts/shared.sh

if [[ -z "${CONDA_LIB_HOME}" ]]; then
  echo "CONDA_LIB_HOME is not set or is empty"
  exit 1
else
  echo "CONDA_LIB_HOME is set to: ${CONDA_LIB_HOME}"
fi

NODEFILE=${UCCL_HOME}/scripts/node_ips/chi.txt

TEST=${1:-uccl}

if [ "$TEST" = "rccl" ]; then
    echo "Running RCCL test"
    plugin_path=""
elif [ "$TEST" = "uccl" ]; then
    echo "Running UCCL test"
    # plugin_path="${UCCL_HOME}/collective/rdma/librccl-net-uccl.so"
    plugin_path=`python -c "import uccl; print(uccl.rccl_plugin_path())"`
    echo "plugin_path: ${plugin_path}"
else
    echo "Unsupport benchmark type."
    exit 1
fi

NVLINK_ON=1

NVLINK_OFF=$((1 - NVLINK_ON))

# alltoall_perf, all_reduce_perf
TEST_BIN=alltoall_perf
QP_SCALING=4
UCCL_ENTROPY=2
UCCL_CHUNK_SIZE_KB=64

mpirun --bind-to none -np 3 -N 1 --hostfile $NODEFILE --map-by ppr:1:node \
    -x LD_LIBRARY_PATH=${UCCL_HOME}/thirdparty/rccl/build/release:${CONDA_LIB_HOME}:/opt/rocm/lib:${LD_LIBRARY_PATH} \
    --mca btl tcp,self \
    --mca btl_tcp_if_include ens51f1np1 \
    -x NCCL_NET_PLUGIN=${plugin_path} \
    -x NCCL_P2P_DISABLE=${NVLINK_OFF} \
    -x NCCL_SHM_DISABLE=${NVLINK_OFF} \
    -x NCCL_IB_PCI_RELAXED_ORDERING=1 \
    -x NCCL_P2P_NET_CHUNKSIZE=524288 \
    -x NCCL_BUFFSIZE=8388608 \
    -x NCCL_MIN_NCHANNELS=32 \
    -x NCCL_MAX_NCHANNELS=32 \
    -x NCCL_NCHANNELS_PER_NET_PEER=1 \
    -x NCCL_IB_QPS_PER_CONNECTION=${QP_SCALING} \
    -x NCCL_IB_SPLIT_DATA_ON_QPS=0 \
    -x NCCL_IB_GID_INDEX=3 \
    -x HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -x NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7 \
    -x GLOG_v=0 \
    -x UCCL_NUM_ENGINES=4 \
    -x UCCL_PORT_ENTROPY=${UCCL_ENTROPY} \
    -x UCCL_CHUNK_SIZE_KB=${UCCL_CHUNK_SIZE_KB} \
    -x UCCL_RCMODE=1 \
    ${UCCL_HOME}/thirdparty/rccl-tests/build/${TEST_BIN} \
    -b 1K -e 1G -f 2 -w 5 -n 20 -c 1 -g 1 -t 8 |& 
    tee ${TEST_BIN}_$([ "$TEST" = "rccl" ] && echo "rccl_qp${QP_SCALING}" || echo "uccl_qp${UCCL_ENTROPY}").log

# -x NCCL_DMABUF_ENABLE=1 \
# -x NCCL_NET_GDR_LEVEL=SYS \
# -x NCCL_DEBUG_SUBSYS=NET \
# -x NCCL_DEBUG=INFO \

# CTRL_NIC="cni0"
# HCA_NAMES="rdma0:1,rdma1:1,rdma2:1,rdma3:1,rdma4:1,rdma5:1,rdma6:1,rdma7:1"
# -x NCCL_SOCKET_IFNAME=${CTRL_NIC} \
# -x NCCL_IB_HCA=${HCA_NAMES} \
