# !/bin/bash

source ../../scripts/shared.sh

if [[ -z "${CONDA_LIB_HOME}" ]]; then
  echo "CONDA_LIB_HOME is not set or is empty"
  exit 1
else
  echo "CONDA_LIB_HOME is set to: ${CONDA_LIB_HOME}"
fi

# sinfo
# squeue --me
# salloc -N 2 -n 2 -p mi2104x -t 00:30:00

# mi2508x has better PCIe switch connection to achieve 200G between GPUs and NICs.
#   GPU 0,1 <-> mlx5_0
#   GPU 6,7 <-> mlx5_2
# salloc -N 2 -n 2 -p mi2508x -t 00:30:00

NODEFILE=nodes.txt
scontrol show hostnames $SLURM_JOB_NODELIST >$NODEFILE

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

mpirun --bind-to none -np 2 -N 1 --hostfile $NODEFILE --map-by ppr:1:node \
    -x LD_LIBRARY_PATH=${UCCL_HOME}/thirdparty/rccl/build/release:${CONDA_LIB_HOME}:/opt/rocm-6.3.1/lib:${LD_LIBRARY_PATH} \
    -x NCCL_NET_PLUGIN=${plugin_path} \
    -x GLOG_v=0 \
    -x NCCL_DMABUF_ENABLE=1 \
    -x NCCL_P2P_DISABLE=1 \
    -x NCCL_SHM_DISABLE=1 \
    -x NCCL_NET_DISABLE=0 \
    -x NCCL_NET_GDR_LEVEL=SYS \
    -x NCCL_IB_QPS_PER_CONNECTION=4 \
    -x HIP_VISIBLE_DEVICES=0 \
    -x NCCL_IB_HCA="mlx5_0:1" \
    -x NCCL_SOCKET_IFNAME="eth0" \
    -x UCCL_NUM_ENGINES=1 \
    -x UCCL_PORT_ENTROPY=256 \
    -x UCCL_CHUNK_SIZE_KB=128 \
    ${UCCL_HOME}/thirdparty/rccl-tests/build/all_reduce_perf \
    -b 1K -e 1G -f 2 -w 5 -n 20 -c 1 -g 1 -t 1 |&
    tee alltoall_debug.log

# -x NCCL_DEBUG=INFO \

# On mi2104x
# -x NCCL_NET_GDR_LEVEL=SYS \
# -x HIP_VISIBLE_DEVICES=0 \
# -x NCCL_IB_HCA="mlx5_0:1" \

# On mi2508x
# -x HIP_VISIBLE_DEVICES=0,1,6,7 \
# -x NCCL_IB_HCA="mlx5_0:1,mlx5_2:1" \

# Setting to 4 will significantly degrade alltoall perf with 32 channels.
# -x NCCL_IB_QPS_PER_CONNECTION=1 \

# Default has 4 channels and 1 channel per net peer.
# -x NCCL_MAX_NCHANNELS=32 \
# -x NCCL_MIN_NCHANNELS=32 \
# -x NCCL_NCHANNELS_PER_NET_PEER=1 \

# -x NCCL_IB_SPLIT_DATA_ON_QPS=1 \
# -x RCCL_MSCCL_FORCE_ENABLE=1 \
# -x RCCL_MSCCLPP_ENABLE=1 \
# -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
# -x NCCL_PROTO=Simple \
# -x NCCL_P2P_NET_CHUNKSIZE=524288 \
# -x NCCL_BUFFSIZE=8388608 \
# all_reduce_perf, alltoall_perf, sendrecv_perf