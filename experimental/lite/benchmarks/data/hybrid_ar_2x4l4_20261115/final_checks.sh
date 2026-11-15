#!/usr/bin/env bash
set -euo pipefail
out="$PWD/build/hybrid_ar_20261115"
make router-policy-check build/uccl_lite_dispatcher_check build/uccl_lite_dispatcher_cuda_check build/uccl_lite_router_check build/uccl_lite_router_graph_check > "$out/router_build.log" 2>&1
build/uccl_lite_router_policy_check > "$out/router_policy.log" 2>&1
build/uccl_lite_dispatcher_check > "$out/dispatcher_host.log" 2>&1
timeout 180 build/uccl_lite_dispatcher_cuda_check > "$out/dispatcher_cuda.log" 2>&1
timeout 180 build/uccl_lite_router_check --gpus 4 --workers-per-node 3 > "$out/router_eager.log" 2>&1
timeout 180 build/uccl_lite_router_graph_check --gpus 4 --workers-per-node 3 > "$out/router_graph.log" 2>&1
export HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0 MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun BUILD_DIR="$out"
timeout 180 bash scripts/run_multinode_hybrid.sh --op allreduce --host-repeat-check 3 -b 25165824 -e 25165824 -w 2 -n 3 --check-each 1 -c 1 > "$out/flat_gap.log" 2>&1
timeout 180 bash scripts/run_multinode_hybrid.sh --op allgather --host-piece-bytes 16777216 -b 134217728 -e 134217728 -w 2 -n 3 --check-each 1 -c 1 > "$out/max_piece.log" 2>&1
