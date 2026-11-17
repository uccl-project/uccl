# Building and running UCCL Lite

Run all commands from `experimental/lite/`. The Makefile builds independent
benchmarks and writes only to `build/`; it does not install libraries or modify
the rest of UCCL. Use `make help` for the target list.

The [optional CPU-router module](cpu-router.md) has independent targets:
`make cpu-router` builds its static/shared libraries, `make local-collectives`
builds the project-owned two-NUMA BF16 fallback, `make adaptive-collectives`
builds the AR/AG/RS calibration example. `make router-checks` and
`make local-collectives-check` build their checks. These are not dependencies of the
existing single-node/fusion targets. The router and local fallback use CUDA
without NCCL; the calibration binary links NCCL for an optional external
comparison. Its default fallback is UCCL Lite local on eight supported GPUs.
`make eight-gpu-alltoall` builds the separate BF16 two-NUMA host/peer benchmark
and its grouped NCCL send/recv control.

## Dependencies

| Dependency | Used by | Tested configuration |
| --- | --- | --- |
| CUDA toolkit | All CUDA sources | CUDA 12.9, L4 / `sm_89` |
| MSCCL++ | Collectives and fusion | Commit `52f659d` for the IB experiments; see compatibility note below |
| NCCL | Fusion and peer reference baselines; optional adaptive-example control | 2.28.7 on four L4s; 2.28.9 on the eight-L4 control |
| cuBLAS | GEMM baselines and dense fusion | Toolkit library, including grouped GEMM API |
| CUTLASS | Grouped fusion kernels and optional CPU router | 4.5.0, using its 2.x grouped-GEMM interfaces |
| Open MPI + UCX | Two-node benchmarks | Open MPI 4.1.7rc1 on the measured host |
| Python + Matplotlib + NumPy | Figure regeneration only | No Python dependency for CUDA executables |

The single-node reference library was an earlier local MSCCL++ build; its exact
revision was not recorded. Do not interpret the IB revision as a fully pinned
single-node environment. Build and validate against the headers and library from
the same MSCCL++ checkout. Generated MSCCL++ headers are under its build tree.

The ordinary kernels compile as C++20. CUTLASS-heavy targets compile as C++17
because the measured CUDA 12.9 compiler had a frontend failure with C++20.

## Build targets

```bash
# Required paths can be supplied on every make invocation.
make -j2 single-node MSCCLPP_HOME=/path/to/mscclpp
make peer_ag_rs MSCCLPP_HOME=/path/to/mscclpp NCCL_HOME=/path/to/nccl/build
make peer-ag-rs-packet MSCCLPP_HOME=/path/to/mscclpp \
  NCCL_HOME=/path/to/nccl/build CUTLASS_HOME=/path/to/cutlass
make nccl_4l4_mpi_baseline NCCL_HOME=/path/to/nccl/build MPICXX=/path/to/mpicxx
make nccl_2x4l4_baseline NCCL_HOME=/path/to/nccl/build MPICXX=/path/to/mpicxx
make multinode_host_allgather NCCL_HOME=/path/to/nccl/build MPICXX=/path/to/mpicxx
make eight-gpu-alltoall NCCL_HOME=/path/to/nccl/build
make -j2 fusion MSCCLPP_HOME=/path/to/mscclpp \
  CUTLASS_HOME=/path/to/cutlass NCCL_HOME=/path/to/nccl/build
make -j2 probes CUTLASS_HOME=/path/to/cutlass
make hardware_profiler  # standalone CUDA topology and bandwidth probe
make headers MSCCLPP_HOME=/path/to/mscclpp CUTLASS_HOME=/path/to/cutlass
```

`make all` builds single-node and peer collectives, fusion, probes, and the reusable-header
compile check. Multi-node targets are separate. Individual target names are
`allreduce`, `alltoall`, `peer_ag_rs`, optional `peer-ag-rs-packet`, `nccl_4l4_mpi_baseline`,
`nccl_2x4l4_baseline`, `multinode_host_allgather`, `dense_fusion`, `fused_moe`, `fused_moe_layer`,
`fused_moe_layer_qwen`, `gemm_probe`, `mma_gemm_probe`, `mma_peak_probe`,
`moe_overlap_probe`, `cutlass_gemm_probe`, and `hardware_profiler`. Executables have an `uccl_lite_`
prefix, for example `build/uccl_lite_dense_fusion`.

`peer_ag_rs` uses the MSCCL++ LL16 packet header but links no MSCCL++ library.
`peer-ag-rs-packet` links the optional CPU-router library and adds the FP32
AllGather host-packet candidate; the ordinary `peer_ag_rs` target remains
independent of that library.
`--packet-router-per-source` makes its GPU consumer copy each source tile
immediately after that source publishes readiness; `--packet-router-gpu-progress`
retains the older all-sources-ready GPU control for paired ablations.

Override `CUDA_HOME`, `NVCC`, `CUDA_ARCH`, `MSCCLPP_BUILD`, and
`EXTRA_NVCC_FLAGS` as needed. Changing compiler flags or dependency paths does
not invalidate existing Make outputs automatically; use `make -B <target>`
when changing those settings. Start with `-j2`: CUTLASS template compilation
uses substantial host memory.

## Single-node correctness runs

Use an idle four-GPU host, verify NUMA order with `nvidia-smi topo -m`, and
ensure CUDA peer access is available. The native AllReduce and All-to-All
executables fork four ranks before CUDA initialization; **do not launch them
with mpirun or torchrun**. The `nccl_4l4_mpi_baseline` control instead
requires four MPI ranks.

```bash
./build/uccl_lite_allreduce -b 1024 -e 16777216 -w 2 -n 3
./build/uccl_lite_alltoall -b 1024 -e 1048576 -w 2 -n 3
./build/uccl_lite_peer_ag_rs -b 1008 -e 1008 -w 2 -n 10 -r 3

./build/uccl_lite_dense_fusion \
  --ag-m 128 --ag-n 1024 --ag-k 512 \
  --rs-m 128 --rs-n 512 --rs-k 1024 -w 2 -n 3

./build/uccl_lite_fused_moe -b 8 -e 8 -w 1 -n 1 -r 1 -S 0
./build/uccl_lite_fused_moe_layer -b 8 -e 8 -w 1 -n 1 -r 1 -S 0 -c 2
```

For collectives, `-b/-e` are bytes. For MoE, they are tokens per rank. Common
flags are `-w` warmups, `-n` iterations, and `-c` correctness checking. MoE's
`-r` is measurement rounds and `-S` is settling time in milliseconds. Use
`-c 2` for the full-layer independent reference check. Keep checking enabled
when porting; some harnesses print an error metric without returning failure,
so inspect `maxrelerr`, `badrow`, and `bad` in the output as well as exit codes.

On the current eight-L4 host, the out-of-place BF16 All-to-All benchmark checks
every output element on all ranks and accepts a non-power-of-two tail size:

```bash
./build/uccl_lite_alltoall_8l4 --begin 1048704 --end 1048704 \
  --warmup 1 --iterations 1 --repeats 3
```

## Two-node build

The tested layout has four L4 GPUs and two HCAs per node. The same executable
and shared-library paths must exist on both hosts.

Build MSCCL++ with IB and GDRCopy enabled for GDR paths. The tested adapter
requires `host-no-atomic` signaling and a loaded `gdrdrv`; RDMA-atomic signaling
returned remote-access errors on that host. Host-staged global-ring mode also
needs [the mapped-host fix](../patches/mscclpp-host-memory-registration.patch).
Review/apply it to the external dependency before rebuilding that library.

```bash
cmake -S /path/to/mscclpp -B /path/to/mscclpp/build-ib \
  -DMSCCLPP_USE_IB=ON -DMSCCLPP_USE_GDRCOPY=ON \
  -DGDRCOPY_ROOT_DIR=/path/to/gdrcopy/install \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build /path/to/mscclpp/build-ib -j2

make -j2 multinode MSCCLPP_HOME=/path/to/mscclpp \
  MSCCLPP_IB_BUILD=/path/to/mscclpp/build-ib MPICXX=/path/to/mpicxx
```

`host_allreduce` depends only on CUDA and MPI and can be built separately without
MSCCL++. The default `multinode_cpu_router` requires local MSCCL++ CUDA IPC but
does not register GPU memory with the NIC; its optional direct-receive mode does.

```bash
export HOSTS=node0:4,node1:4
export MPIRUN=/path/to/mpirun
export SOCKET_IFNAME=your_network_interface
export HCA_DEVICES=mlx5_0,mlx5_1

bash scripts/run_multinode.sh multinode_allreduce -b 1024 -e 67108864 -c 1
bash scripts/run_multinode.sh multinode_allreduce --op allgather \
  --ag-ring 1 --gpu-timing 1 -b 1024 -e 67108864 -c 1
bash scripts/run_multinode.sh multinode_allreduce --op reducescatter \
  --rs-ring 1 --rs-direct-ring 1 --rs-piece-bytes 8388608 -B 96 -T 512 \
  --gpu-timing 1 -b 1024 -e 67108864 -c 1
bash scripts/run_multinode.sh multinode_cpu_router -k 1048576 -c 1
bash scripts/run_multinode.sh host_allreduce -k 4194304 -c 1
```

The launcher uses eight MPI ranks and four ranks per node. It uses TCP/shared
memory for MPI bootstrap when MSCCL++ owns the IB data path, and UCX for the
host-router paths. Set `BUILD_DIR` if executables live outside the default
directory. On hosts with multiple IP addresses on one interface, use an
interface with one address for `SOCKET_IFNAME`; the recorded two-node AG/RS
comparison used `ibp55s0f0`. The same executable and MSCCL++ library must be
present on both hosts. See [multi-node design](multinode.md) for tuning,
matched NCCL commands and constraints.

## November 10 refresh commands

The paired four-L4 AllReduce script runs the native UCCL Lite executable and
the four-rank NCCL control in alternating order. Both use host
enqueue+completion time, FP32 in-place buffers, and check every warmup and
timed call. The peer AG/RS harness contains both FP32 paths and reports
paired per-call CUDA-event samples.

```bash
OUTPUT_DIR=build/single-node-refresh bash scripts/run_single_node_refresh.sh
./build/uccl_lite_peer_ag_rs -b 1024 -e 1073741824 -w 5 -n 10 -r 3
```

For two nodes, first build and copy the same `multinode_allreduce` and
`nccl_2x4l4_baseline` executables to both hosts.
The checked AR/AG/RS control uses one NCCL rank per GPU and the same host
timing scope as UCCL Lite. `CHECK_EACH=1` verifies each UCCL call; the custom
NCCL control verifies each of its calls.

```bash
export HOSTS=node0:4,node1:4 SOCKET_IFNAME=ibp55s0f0
export HCA_DEVICES=mlx5_0,mlx5_1
export NCCL_LIB=/path/to/nccl/build/lib
export NCCL_BASELINE_BIN=/same/path/on/both/hosts/uccl_lite_nccl_2x4l4_baseline
TIMING_MODE=host CHECK_EACH=1 OPERATIONS='allreduce allgather reducescatter' \
  OUTPUT_DIR=build/multinode-refresh bash scripts/run_multinode_peer_compare.sh
```

For the measured follow-up schedule, add `AUTO_VARIANTS=1` to that paired
two-node command. The four-L4 peer AG/RS executable selects destination-side
peer reads through 32 KiB and source-side peer writes above that extent;
its input fill and full-output verification are outside timing for both it
and NCCL. Keep the same warmup, paired-repeat, and checking settings.

The fusion refresh uses the shipped complete-operation controls and retains
the L4 power-stability method for MoE:

```bash
OUTPUT_DIR=build/dense-refresh bash scripts/run_dense_refresh.sh
OUTPUT_LOG=build/moe-refresh.log bash scripts/run_moe_refresh.sh
```

Keep output from each run and the exact binary/library hashes with the
summary CSVs. The [refresh record](../benchmarks/data/refresh_20261110/README.md)
contains the commands, raw logs, and timing boundaries used for the README.

## November 13 four-L4 candidates

```bash
make peer_ag_rs peer-ring-check peer-hybrid-allreduce peer-hybrid-check
./build/uccl_lite_peer_ring_check
./build/uccl_lite_peer_hybrid_check
./build/uccl_lite_peer_ag_rs --rs-only --rs-ring -b 1024 -e 1073741824 -w 10 -n 10 -r 3
./build/uccl_lite_peer_hybrid_allreduce -b 16777216 -e 1073741824 -w 10 -n 10 -r 3
./build/uccl_lite_peer_ag_rs --graph -b 1024 -e 1048576 -w 10 -n 10 -r 3
```

The ring RS and hybrid AR are opt-in standalone experiments. Hybrid AR is
FP32 in-place, one process, GPU peer RS followed by mapped-host packet AG;
it requires the optional CPU-router library. It does not support graph capture.
`--rs-ring-chunk-bytes` sets RS piece size. `--packet-bytes` and `--packet-blocks`
expose packet AG geometry without changing the router's default policy.
The `--graph` peer harness supports SM pull/push-serial and NCCL through 1 MiB;
CE peer copies and host-generation LL/fence candidates are excluded. Its timing
is graph-versus-graph, separate from the eager README curves.
Native AR `-a 8` selects only the existing ring for diagnostics; omitting `-a`
retains the original candidate sweep. See the [additional optimization results](../benchmarks/data/optimization_4l4_20261113/README.md#reducescatter-granularity).

## Full-size and small/medium follow-up

```bash
make allreduce peer_ag_rs peer-ag-rs-packet peer-small-medium-check resident-collectives resident-check
./build/uccl_lite_peer_small_medium_check
./build/uccl_lite_resident_check
./build/uccl_lite_allreduce --sm-variants --fast-sweep -b 1024 -e 1073741824 -w 5 -n 10
./build/uccl_lite_peer_ag_rs_packet --packet-router-ag --packet-router-per-source \
  --mini-host-max-bytes 16777216 --push-grid-blocks 16 -b 16 -e 1073741824 -w 20 -n 10 -r 3
./build/uccl_lite_peer_ag_rs --rs-only --rs-ring --rs-hier-ll-max-bytes 32768 \
  -b 16 -e 1073741824 -w 20 -n 10 -r 3
./build/uccl_lite_resident_collectives -b 16 -e 1048576 -w 20 -n 20 -r 3
```

Native AR now waits for its input copy on the collective stream before timing;
its reference changes generation every call and rejects nonfinite outputs.
`--sm-variants` adds explicit 1-/8-CTA variants; `--fast-sweep` retains all
eligible candidates through 4 MiB and only the established ring above it.
`--sm-blocks` and `--allocation-bytes` are diagnostic overrides.

Mini-host AG remains eager-only. Its prepared capacity must be a multiple of
16 bytes and its block count a multiple of four. The phased LL RS candidate
requires integral float2 shards and all-rank completion before scratch reuse.
No automatic production routing policy was added.

Resident mode is a separate input-ready, host-blocking experiment. It reserves
one CTA per GPU; stop the workers before device-wide synchronization or cleanup.
`--packets 0` is an ablation. `--nccl-graph 1` measures a global NCCL graph
control; `--nccl-graph 2` uses per-device graphs. These options change only the
reference control, not the resident execution contract. See the
[full study and complete tables](../benchmarks/data/small_medium_4l4_20261113/README.md).
