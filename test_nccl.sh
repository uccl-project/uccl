#!/usr/bin/env bash
set -euo pipefail

INSTALL_PKGS=0
NODE_IPS=("172.31.5.88" "172.31.6.215")
NIC="enp71s0"
PROCS_PER_NODE=8
NUM_PROCS=$(( ${#NODE_IPS[@]} * PROCS_PER_NODE ))

export UCCL_HOME=/home/ubuntu/efs/ziming/uccl

cd "$UCCL_HOME/thirdparty/nccl-tests"

# host list
nodes=""
for ip in "${NODE_IPS[@]}"; do
  nodes+="${ip}:${PROCS_PER_NODE},"
done
nodes="${nodes%,}"

# Make sure OpenMPI isn't inheriting conflicting MCA vars
unset OMPI_MCA_btl_tcp_if_include OMPI_MCA_btl_tcp_if_exclude
unset OMPI_MCA_btl OMPI_MCA_pml OMPI_MCA_mtl OMPI_MCA_btl_base_verbose

# Force the NCCL you built
export LD_LIBRARY_PATH="$UCCL_HOME/thirdparty/nccl/build/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for proto in "Simple" "LL128"; do
for algo in "TREE" "RING" "COLLNET_DIRECT" "COLLNET_CHAIN" "NVLS" "NVLS_TREE" "PAT"; do
for num_gpus_per_node in 1 2 4 8; do
    date
    if [ "$proto" = "Simple" ]; then
        MAX_BYTES=4G
    else
        MAX_BYTES=4M
    fi

    echo "Running with algo=$algo, num_gpus_per_node=$num_gpus_per_node, proto=$proto"
    NUM_PROCS=$(( ${#NODE_IPS[@]} * $num_gpus_per_node ))
    /opt/amazon/openmpi/bin/mpirun \
    --allow-run-as-root \
    --tag-output \
    -H "$nodes" \
    -np "$NUM_PROCS" -N "$num_gpus_per_node" \
    --bind-to none \
    -x LD_LIBRARY_PATH \
    -x CUDA_VISIBLE_DEVICES \
    -x NCCL_DEBUG=INFO \
    -x NCCL_ALGO=$algo \
    -x NCCL_PROTO=$proto \
    -x NCCL_SOCKET_IFNAME="$NIC" \
    -x NCCL_NVLS_ENABLE=0 \
    --mca pml ob1 \
    --mca btl tcp,self,vader \
    --mca btl_tcp_if_include "$NIC" \
    "$UCCL_HOME/thirdparty/nccl-tests/build/all_reduce_perf" \
        -b 8 -e "$MAX_BYTES" -f 2 -g 1 -c 5 -w 5 -n 20
done
done
done
