#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${E2E_PYTHON:=/home/zhongjie/flux/.venv/bin/python}"
: "${E2E_CHECKPOINT:=/home/zhongjie/.cache/uccl-lite-gpt2-large}"
: "${E2E_NCCL:=$HOME/third_party/nccl/build/lib/libnccl.so.2}"
: "${E2E_OUT:=build/l4_e2e/paired}"
mkdir -p "$E2E_OUT"
export OMPI_MCA_hwloc_base_binding_policy=none
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_PRELOAD="$E2E_NCCL"
launcher=()
workload=()
if [[ ${E2E_NODES:-1} == 2 ]]; then
  workload=(--cases 16:32:16 16:128:16 32:256:16 --train-cases 16:32 16:64 32:128)
  export NCCL_SOCKET_IFNAME=ibp55s0f0 NCCL_IB_HCA=mlx5_0,mlx5_1
  launcher=(/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun --host 10.10.55.2:1,10.10.55.1:1 -np 2 --map-by ppr:1:node --bind-to none --mca pml ob1 --mca btl self,tcp --mca btl_tcp_if_include ibp55s0f0 -x LD_PRELOAD -x PYTHONDONTWRITEBYTECODE -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA)
fi
index=0
for backend in ${E2E_ORDER:-nccl peer peer nccl}; do
  for kind in serving training; do
    prefix=$(printf '%02d_%s_%s' "$index" "$backend" "$kind")
    "${launcher[@]}" "$E2E_PYTHON" integration/l4_e2e/benchmark.py \
      --checkpoint "$E2E_CHECKPOINT" --backend "$backend" --kind "$kind" \
      --out "$E2E_OUT/$prefix.json" "${workload[@]}" "$@" > "$E2E_OUT/$prefix.log" 2>&1
  done
  index=$((index+1))
done
