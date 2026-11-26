#!/usr/bin/env bash
set -euo pipefail
export HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0 MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun
out="$PWD/build/hybrid_ar_20261115"
for round in 1 2 3; do
 modes=(old hybrid nccl)
 if [[ $round == 2 ]]; then modes=(nccl old hybrid); fi
 if [[ $round == 3 ]]; then modes=(hybrid nccl old); fi
 for mode in "${modes[@]}"; do
  log="$out/final_${mode}_${round}.log"; [[ ! -e "$log" ]] || exit 2
  if [[ $mode == hybrid ]]; then
   timeout 600 env BUILD_DIR="$out" bash scripts/run_multinode_hybrid.sh --op allreduce -b 1048576 -e 1073741824 -w 20 -n 10 --check-each 1 -c 1 > "$log" 2>&1
  elif [[ $mode == old ]]; then
   timeout 600 env BUILD_DIR="$PWD/build/multinode_trial" bash scripts/run_multinode.sh multinode_allreduce --op allreduce --ar-auto 1 -b 1048576 -e 1073741824 -w 20 -n 10 --check-each 1 -c 1 > "$log" 2>&1
  else
   timeout 600 "$MPIRUN" --host "$HOSTS" -np 8 --map-by ppr:4:node --bind-to none \
    --mca pml ob1 --mca btl self,vader,tcp --mca btl_tcp_if_include "$SOCKET_IFNAME" \
    -x NCCL_SOCKET_IFNAME=ibp55s0f0 -x NCCL_IB_HCA=mlx5_0,mlx5_1 -x NCCL_ALGO=Ring -x NCCL_PROTO=Simple \
    "$out/uccl_lite_nccl_2x4l4_baseline" --op allreduce -b 1048576 -e 1073741824 -w 20 -n 10 > "$log" 2>&1
  fi
 done
done
