#!/usr/bin/env bash
set -euo pipefail
export HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0
export MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun
export BUILD_DIR="$PWD/build/ar_fix"
record="$PWD/benchmarks/data/rs_full_screen_2x4l4_20261115"
run() {
  name="$1"; begin="$2"; end="$3"; shift 3
  log="$record/raw/$name.log"; [[ ! -e "$log" ]]
  timeout 300 bash scripts/run_multinode.sh multinode_allreduce --op reducescatter \
    -b "$begin" -e "$end" -w 5 -n 5 --check-each 1 -c 1 "$@" > "$log" 2>&1
  echo "Completed $name"
}
run baseline 1024 1073741824 --rs-auto 1 --rs-piece-bytes 8388608 -B 96 -T 512
run serial 1024 8388608 --rs-serial-local 1 -B 96 -T 512
run direct_medium 131072 16777216 --rs-ring 1 --rs-direct-ring 1 --rs-pieces 1 -B 96 -T 512
run launch_96_256 1024 1073741824 --rs-auto 1 --rs-piece-bytes 8388608 -B 96 -T 256
run launch_192_256 1024 1073741824 --rs-auto 1 --rs-piece-bytes 8388608 -B 192 -T 256
run piece16 33554432 1073741824 --rs-auto 1 --rs-piece-bytes 16777216 -B 96 -T 512
run piece32 33554432 1073741824 --rs-auto 1 --rs-piece-bytes 33554432 -B 96 -T 512
# Existing opt-in fused forwarding experiment, exact source archived in the prior record.
export BUILD_DIR=/home/zhongjie/uccl-zhongjie-dev/experimental/lite/build/ar_latency_2x4l4_20261115
run fused4 1024 1073741824 --rs-fused-ring 1 --rs-fused-piece-bytes 4194304 -B 96 -T 512
run fused8 33554432 1073741824 --rs-fused-ring 1 --rs-fused-piece-bytes 8388608 -B 96 -T 512
