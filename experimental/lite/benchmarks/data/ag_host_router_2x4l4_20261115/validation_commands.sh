#!/usr/bin/env bash
set -euo pipefail
root="$PWD"
out="$root/benchmarks/data/ag_host_router_2x4l4_20261115/checks"
mkdir -p "$out"
for workers in 0 1; do
 for progress in steps ready; do
  for network in multicast rdma; do
   binary="$root/build/ag_host_router_20261115/uccl_lite_multinode_host_allgather"
   if [[ $network == rdma ]]; then binary="$root/build/ag_host_router_20261115/uccl_lite_multinode_host_rdma_allgather"; fi
   timeout 180 /usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun --host 10.10.55.2:2,10.10.55.1:2 -np 4 --map-by ppr:2:node --bind-to none --mca pml ucx --mca coll '^hcoll' \
    -x MSCCLPP_SOCKET_IFNAME=ibp55s0f0 -x MSCCLPP_HCA_DEVICES=mlx5_0,mlx5_1 \
    "$binary" -b 1572864 -e 12582912 -k 131200 -w 5 -n 10 -r 1 --download sm --local-input self --local-exchange shared --early-consume 1 --progress "$progress" --gpu-workers "$workers" --network "$network" --delay-consumer-rank 1 --nccl 0 > "$out/${network}_w${workers}_${progress}_delayed_tail.log" 2>&1
  done
 done
done
# Exercise the optional GPU-pair shortcut, SM producer and DMA consumer together.
timeout 180 /usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun --host 10.10.55.2:2,10.10.55.1:2 -np 4 --map-by ppr:2:node --bind-to none --mca pml ucx --mca coll '^hcoll' \
 "$root/build/ag_host_router_20261115/uccl_lite_multinode_host_allgather" -b 1572864 -e 12582912 -k 131200 -w 5 -n 10 -r 1 --download dma --upload sm --local-input pair --local-exchange shared --early-consume 1 --progress ready --gpu-workers 1 --delay-consumer-rank 6 --nccl 0 > "$out/pair_sm_upload_dma_download.log" 2>&1
for name in router_policy_check dispatcher_check dispatcher_cuda_check; do
 timeout 180 "$root/build/uccl_lite_${name}" > "$out/${name}.log" 2>&1
done
timeout 180 "$root/build/uccl_lite_router_check" --gpus 4 --workers-per-node 2 > "$out/router_eager.log" 2>&1
timeout 180 "$root/build/uccl_lite_router_graph_check" --gpus 4 --workers-per-node 2 > "$out/router_graph.log" 2>&1
