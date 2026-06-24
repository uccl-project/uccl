# !/bin/bash

source ../../scripts/shared.sh

# Usage ./run_nccl_test.sh [nccl] [# of Nodes] [# of GPUs per process] [allreduce/alltoall: 0/1]

TEST=${1:-nccl}
NUM_PROCS=${2:-2}
NUM_GPUS_PER_PROC=${3:-8}
PROG_OPTION=${4:-0}
PROCS_PER_NODE=${5:-1}
HOSTFILE="${HOSTFILE:-${UCCL_HOME}/scripts/node_ips/tcpx.txt}"
TCPX_RUNTIME_LIBDIR="${TCPX_RUNTIME_LIBDIR:-/usr/local/tcpx/lib64}"
TCPX_SRC_LIBDIR="${TCPX_SRC_LIBDIR:-/var/lib/tcpx/lib64}"
SSH_CMD="${TCPX_SSH_CMD:-ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null}"
PLM_RSH_AGENT="${PLM_RSH_AGENT:-${SSH_CMD}}"

stage_tcpx_runtime() {
    local src_dir="$1"
    local dst_dir="$2"
    local required_lib

    if [[ ! -d "${src_dir}" ]]; then
        echo "TCPX source directory not found: ${src_dir}" >&2
        return 1
    fi

    for required_lib in libnccl-net.so libnccl.so.2; do
        if [[ ! -e "${src_dir}/${required_lib}" ]]; then
            echo "Missing ${src_dir}/${required_lib}" >&2
            return 1
        fi
    done

    echo "Staging TCPX runtime from ${src_dir} to ${dst_dir}"
    sudo install -d -m 755 "${dst_dir}"
    sudo cp -af "${src_dir}"/libnccl.so* "${src_dir}/libnccl-net.so" "${dst_dir}/"
    echo "TCPX runtime ready in ${dst_dir}"
}

# all_gather_perf  all_reduce_perf  alltoall_perf  broadcast_perf  gather_perf
# hypercube_perf  reduce_perf  reduce_scatter_perf  scatter_perf  sendrecv_perf

if [ "$PROG_OPTION" -eq 0 ]; then
    PROG_NAME=all_reduce_perf
elif [ "$PROG_OPTION" -eq 1 ]; then
    PROG_NAME=alltoall_perf
else
    PROG_NAME=sendrecv_perf
fi

if [ "$TEST" = "nccl" ]; then
    echo "Running NCCL test"
else
    echo "Unsupport benchmark type."
    exit 1
fi

echo "Running test: ${PROG_NAME}, ${TEST}, ${NUM_PROCS} nodes, ${NUM_GPUS_PER_PROC} GPUs per process, $((NUM_PROCS * NUM_GPUS_PER_PROC)) GPUs in total."

echo $NUM_PROCS
echo $PROCS_PER_NODE

stage_tcpx_runtime "${TCPX_SRC_LIBDIR}" "${TCPX_RUNTIME_LIBDIR}"
LOCAL_HOST_IP="$(hostname -I | awk '{print $1}')"
while read -r host _; do
    [[ -z "${host}" || "${host}" =~ ^# ]] && continue
    if [[ "${host}" == "${LOCAL_HOST_IP}" || "${host}" == "$(hostname)" ]]; then
        continue
    fi
    ${SSH_CMD} "${host}" "TCPX_SRC_LIBDIR='${TCPX_SRC_LIBDIR}' TCPX_RUNTIME_LIBDIR='${TCPX_RUNTIME_LIBDIR}' bash -lc '
        set -euo pipefail
        src_dir=\"\${TCPX_SRC_LIBDIR:-/var/lib/tcpx/lib64}\"
        dst_dir=\"\${TCPX_RUNTIME_LIBDIR:-/usr/local/tcpx/lib64}\"
        for required_lib in libnccl-net.so libnccl.so.2; do
            [[ -e \"\${src_dir}/\${required_lib}\" ]] || { echo \"Missing \${src_dir}/\${required_lib}\" >&2; exit 1; }
        done
        sudo install -d -m 755 \"\${dst_dir}\"
        sudo cp -af \"\${src_dir}\"/libnccl.so* \"\${src_dir}/libnccl-net.so\" \"\${dst_dir}/\"
    '"
done < "${HOSTFILE}"

# adapted from https://github.com/skypilot-org/skypilot/blob/master/examples/gcp_gpu_direct_tcpx/nccl_tcpx_gcpvm_h100.yaml
mpirun --allow-run-as-root -np ${NUM_PROCS} -N ${PROCS_PER_NODE} \
    -hostfile ${HOSTFILE} --map-by ppr:${PROCS_PER_NODE}:node \
    --mca btl tcp,self \
    --mca btl_tcp_if_include eth0 \
    --mca plm_rsh_agent "${PLM_RSH_AGENT}" \
    -x PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:$PATH" \
    -x LD_LIBRARY_PATH="${TCPX_RUNTIME_LIBDIR}:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/lib:$LD_LIBRARY_PATH" \
    -x NCCL_IGNORE_CPU_AFFINITY=1 \
    -x NCCL_ALGO=Ring \
    -x NCCL_PROTO=Simple \
    -x NCCL_MAX_NCHANNELS=8 \
    -x NCCL_MIN_NCHANNELS=8 \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x NCCL_CROSS_NIC=0 \
    -x NCCL_NSOCKS_PERTHREAD=4 \
    -x NCCL_SOCKET_NTHREADS=1 \
    -x NCCL_DYNAMIC_CHUNK_SIZE=524288 \
    -x NCCL_P2P_NET_CHUNKSIZE=524288 \
    -x NCCL_P2P_PCI_CHUNKSIZE=524288 \
    -x NCCL_P2P_NVL_CHUNKSIZE=1048576 \
    -x NCCL_BUFFSIZE=8388608 \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -x CUDA_MODULE_LOADING=EAGER \
    -x NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4 \
    -x NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0 \
    -x NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177" \
    -x NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191" \
    -x NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=50000 \
    -x NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX="/run/tcpx" \
    -x NCCL_GPUDIRECTTCPX_FORCE_ACK=0 \
    -x NCCL_NET_GDR_LEVEL=PIX \
    -x NCCL_P2P_PXN_LEVEL=0 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_DEBUG_SUBSYS=ENV \
    ${UCCL_HOME}/thirdparty/nccl-tests/build/${PROG_NAME} -c 0 \
    -b 1K -e 1G \
    -f 2 -w 50 -n 50 \
    -g 1 -t ${NUM_GPUS_PER_PROC}
