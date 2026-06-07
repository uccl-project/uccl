# L40/L41 P2P send/recv benchmark runbook

This runbook records the commands used to compare the `lite-collective` NCCL
compatibility layer against native NCCL `sendrecv_perf` on the L40/L41 two-node
testbed.

The benchmark exercises `ncclSend`/`ncclRecv` only.  In the `mscclpp` backend,
`sendrecv_perf` loads `lite-collective/nccl/build/libmscclpp_nccl.so`; in the
`nccl` backend, the same MPI `sendrecv_perf` binary loads the native NCCL
shared library from `/home/yangz/nfs/zhongjie/nccl`.

## Native collective support status

The NCCL compatibility layer currently exposes native paths for the target
collectives below.  Native benchmark runs must keep `MSCCLPP_NCCL_LIB_PATH`
unset, set `MSCCLPP_FORCE_NCCL_FALLBACK_OPERATION=`, and use
`MSCCLPP_NCCL_LOCAL_P2P_FALLBACK=0` so the table does not hide missing support
behind NCCL fallback.

| Collective | `1nx2g` native | `2nx1g` native | `2nx4g` native | Current performance gate |
| --- | --- | --- | --- | --- |
| `allreduce` | Supported | Supported | Supported | `2nx4g/1MiB` native RS+AG composition beats NCCL no-GDR out-of-place and in-place (`232.54/234.32 us` vs `1281.14/783.76 us`). |
| `allgather` | Supported | Supported | Supported | `2nx4g/1MiB` NUMA-split host-slab path uses both NICs and beats NCCL no-GDR out-of-place and in-place (`84.57/83.64 us` vs `119.07/117.32 us`). Full `128B-1GiB` sweep is correct. `<128KiB` now uses a direct-QP ring-slot path that is close to NCCL and wins some rows. |
| `reducescatter` | Supported | Supported | Supported | `2nx4g/1MiB` NUMA-pair local fan-in + pairwise host-staged RDMA path is correct and improved (`123.29/123.35 us`, down from `143.53/143.06 us`) but still slower than NCCL no-GDR (`107.65/107.59 us`). |
| `alltoall` | Supported | Supported | Supported | `1nx2g/1MiB` and `2nx1g/1MiB` beat NCCL; `2nx4g/1MiB` remains the blocker (`107.44 us` native vs `96.85 us` NCCL out-of-place; `101.25 us` native vs `95.29 us` NCCL in-place). |

All measurements above are on L40/L41 with `NCCL_SOCKET_IFNAME=ibp55s0f0`,
`NCCL_IB_HCA=mlx5_0,mlx5_1`, `MSCCLPP_SOCKET_IFNAME=ibp55s0f0`, and
`MSCCLPP_HCA_DEVICES=mlx5_0,mlx5_1`.

### 1MiB collective comparison snapshot

These rows use either `scripts/benchmark-collectives.sh` or the isolated
`scripts/run-nccl-tests.sh` commands listed below, with native fallback
disabled. Times are out-of-place `nccl-tests` latency in microseconds; `#wrong`
was `0` for every native row listed.

| Collective | Topology | NCCL time (us) | Native time (us) | Speedup | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `allgather` | `1nx2g` | 62.60 | 41.26 | 1.52x | Pass |
| `reducescatter` | `1nx2g` | 60.42 | 46.31 | 1.30x | Pass |
| `allreduce` | `1nx2g` | 97.81 | 77.56 | 1.26x | Pass |
| `alltoall` | `1nx2g` | 59.52 | 41.04 | 1.45x | Pass |
| `alltoall` | `2nx1g` | 74.32 | 73.93 | 1.01x | Pass |
| `alltoall` | `2nx4g` | 96.85 | 107.44 | 0.90x | Fail; native correct but still slower |
| `allgather` | `2nx1g` | N/A | 117.07 | N/A | Native correct; NCCL baseline segfaulted |
| `reducescatter` | `2nx1g` | N/A | 139.13 | N/A | Native correct; NCCL baseline blocked |
| `allreduce` | `2nx1g` | N/A | 227.28 | N/A | Native correct; NCCL baseline blocked |
| `allgather` | `2nx4g` | 119.07 | 84.57 | 1.41x | Pass against NCCL no-GDR; full `128B-1GiB` sweep correct |
| `reducescatter` | `2nx4g` | 107.65 | 123.29 | 0.87x | Correct and improved, but still slower than NCCL no-GDR |
| `allreduce` | `2nx4g` | 1281.14 | 232.54 | 5.51x | Pass against NCCL no-GDR |

The `2nx4g` rows above use isolated native-only runs with
`--iters 20 --warmup-iters 5 --check-iters 1`. NCCL no-GDR baselines set
`NCCL_NET_GDR_LEVEL=0`. Raw logs:

| Collective | Native log | NCCL no-GDR log |
| --- | --- | --- |
| `allgather` | `.tmp/collective-benchmarks/allgather-numa-fix-128B-1G-20260607-092852/all_gather_lite_128B_1G.log` | `.tmp/collective-benchmarks/allgather-numa-fix-128B-1G-20260607-092852/all_gather_nccl_nogdr_128B_1G.log` |
| `reducescatter` | `.tmp/collective-benchmarks/rs-final-add-retained-20260606-093009/reduce_scatter_mscclpp.log` | `.tmp/collective-benchmarks/rs-numa-retained-20260606-081603/reduce_scatter_nccl_nogdr.log` |
| `allreduce` | `.tmp/collective-benchmarks/final-native-current-20260606-071108/all_reduce_mscclpp.log` | `.tmp/collective-benchmarks/ag-ar-nccl-nogdr-current-20260606-060420/all_reduce_nccl_nogdr.log` |

Detailed `2nx4g/1MiB` no-GDR comparison:

| Collective | NCCL out (us) | Native out (us) | Out speedup | NCCL in (us) | Native in (us) | In speedup | `#wrong` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `allgather` | 119.07 | 84.57 | 1.41x | 117.32 | 83.64 | 1.40x | 0 |
| `reducescatter` | 107.65 | 123.29 | 0.87x | 107.59 | 123.35 | 0.87x | 0 |
| `allreduce` | 1281.14 | 232.54 | 5.51x | 783.76 | 234.32 | 3.34x | 0 |

### ReduceScatter NCCL algorithm comparison

NCCL `2nx4g/1MiB` ReduceScatter selects `Algo RING proto SIMPLE`
(`channel{Lo..Hi}={0..1}` in
`.tmp/collective-benchmarks/nccl-rs-proto-compare-20260606-072914/reduce_scatter_default_debug.log`).
The benchmark wrapper passes through `NCCL_ALGO`, `NCCL_PROTO`,
`NCCL_MIN_NCHANNELS`, and `NCCL_MAX_NCHANNELS` for reproducible forced-algorithm
experiments.

| Backend / algorithm | Out (us) | In (us) | Correct | Notes |
| --- | ---: | ---: | --- | --- |
| NCCL default | 108.05 | 106.02 | Yes | Default no-GDR selection. |
| NCCL forced `RING` | 107.29 | 107.15 | Yes | Same effective path as default. |
| NCCL forced `RING,SIMPLE` | 112.03 | 107.05 | Yes | Simple ring protocol. |
| NCCL forced `RING,LL128` | 125.96 | 480.95 | Yes | Out-of-place slower; in-place much slower. |
| NCCL forced `RING,LL` | 1920.11 | 1020.73 | Yes | Not competitive on this testbed. |
| NCCL forced `PAT,SIMPLE` | N/A | N/A | No | NCCL returned invalid usage. |
| Native retained NUMA-pair local fan-in + dedicated final add | 123.29 | 123.35 | Yes | Retained improvement: reduce within GPU pairs 0/1 and 2/3, exchange one cross-NUMA partial, then use pairwise host-staged RDMA and a specialized float-sum final add kernel. Best observed clean run with this code was `118.62/117.91 us`, but the latest retained log is shown here. |
| Native lite P2P ring experiment | 2241.31 | 1974.25 | Yes | Correct only with per-step receiver-consumed barrier; not retained because it is far slower than the packed local + pairwise RDMA path. |

Rejected ReduceScatter follow-up experiments after the NUMA-pair improvement:
host-remote partial reduction (`194.22/194.08 us`), direct GPU kernel writes to
peer IPC scratch (`369.21/375.46 us`), host-mapped final remote partial reads
(`123.45/124.27 us`; a later mapped-host read with the dedicated final add was
`131.53/131.50 us`), direct local partial output to `recvbuff`
(`147.64/141.32 us`), fast-path QP writes replacing `Connection::write`
(`147.99/136.97 us`), CUDA IPC event local handoff (`132.20/130.83 us`),
CudaIpc input-pull local fan-in (`973.09/941.72 us`), D2H/previous-ack overlap
(`126.16/125.08 us`), direct-copy NUMA packing (`140.33/139.65 us`), H2D
directly to output plus in-place add (`125.40/124.20 us`), single-sync final
H2D+add (`126.29/125.92 us`), partner-only NUMA packing
(`145.56/138.94 us`), split-stream own/cross pair reduction
(`130.21/129.86 us`), and narrower scratch-reuse barriers
(`120.27/119.58 us`).

The original native NCCL `2nx4g/1MiB` comparison failure was caused by
`scripts/run-nccl-tests.sh` forcing `NCCL_BUFFSIZE=1073741824`. With NCCL
2.29.7 this made net transport setup crash before any timing row; the faulting
offsets symbolize to `sendProxyConnect` and `recvProxyConnect` in
`/home/yangz/nfs/zhongjie/nccl/src/transport/net.cc`, dereferencing
`resources->sendMem`/`recvMem` at low unmapped addresses such as `0x530000`.
Keeping `NCCL_BUFFSIZE` at `4194304` avoids the invalid connect-map state and
the same runs pass. Fixed NCCL logs are under
`.tmp/nccl-buff4m-wrapper-2nx4g-20iter/`.

## Testbed assumptions

Current L40/L41 mapping:

| Node | IB IP | GPU | HCA | Interface |
| --- | --- | --- | --- | --- |
| L40 | `10.10.55.1` | `0` | `mlx5_0` | `ibp55s0f0` |
| L41 | `10.10.55.2` | `0` | `mlx5_0` | `ibp55s0f0` |

For GPU `0`, `nvidia-smi topo -m` should show `GPU0` close to `NIC0`, where
`NIC0` is `mlx5_0`.  If using GPUs `2,3`, switch to `mlx5_1` on this testbed.

Do not use the public `4.14.x.x` addresses for this benchmark; use the IB
private addresses above.

## One-time build

Run from the repository root for this subproject:

```bash
cd /home/yangz/nfs/zhongjie/uccl/experimental/lite/lite-collective

export ROOT=$PWD
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.7rc1
export CUDA_HOME=/usr/local/cuda
export NCCL_TESTS_DIR=/home/yangz/nfs/zhongjie/nccl-tests
export EXTERNAL_NCCL_DIR=/home/yangz/nfs/zhongjie/nccl

# Use absolute paths.  Relative SDK_DIR/BUILD_DIR values are interpreted from
# nccl-tests/src during `make -C`, which can make nccl.h disappear.
export SDK_DIR=$ROOT/.tmp/mint-nccl-sdk
export BUILD_DIR=$ROOT/.tmp/mint-nccl-tests-mpi-build
export RUNTIME_ROOT=$ROOT/.tmp/mint-nccl-tests-runtime

make
make -C nccl

# Build the MPI-mode nccl-tests sendrecv binary and prepare runtime symlinks.
# A short 8-byte smoke is enough; later sections run the real benchmark.
NCCL_SOCKET_IFNAME=ibp55s0f0 \
NCCL_IB_HCA=mlx5_0 \
MSCCLPP_SOCKET_IFNAME=ibp55s0f0 \
MSCCLPP_HCA_DEVICES=mlx5_0 \
bash scripts/run-nccl-tests.sh \
  --test sendrecv --backend nccl --topology inter \
  --hosts 10.10.55.1,10.10.55.2 --gpus 0 \
  --min-bytes 8 --max-bytes 8 --iters 1 --warmup-iters 1
```

The binary should exist after this step:

```bash
ls -l $BUILD_DIR/sendrecv_perf_mpi
```

## Common environment

Use the direct `mpirun` form below for reproducible runs.  It avoids accidental
changes to `LD_LIBRARY_PATH` and makes the active NCCL library explicit.

```bash
cd /home/yangz/nfs/zhongjie/uccl/experimental/lite/lite-collective

export ROOT=$PWD
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.7rc1
export CUDA_HOME=/usr/local/cuda
export BIN=$ROOT/.tmp/mint-nccl-tests-mpi-build/sendrecv_perf_mpi
export HOST_SPEC=10.10.55.1:1,10.10.55.2:1
export GPU_LIST=0
export IFNAME=ibp55s0f0
export HCA=mlx5_0
```

Sanity checks:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 10.10.55.2 true
nvidia-smi topo -m
ssh 10.10.55.2 'nvidia-smi topo -m'
ibv_devices
ssh 10.10.55.2 'ibv_devices'
```

Make sure no old MPI benchmark is still running:

```bash
ps -u "$USER" -o pid,cmd | grep -E 'sendrecv_perf|mpirun' | grep -v grep || true
ssh 10.10.55.2 \
  'ps -u "$USER" -o pid,cmd | grep -E "sendrecv_perf|mpirun" | grep -v grep || true'
```

## Run lite-collective P2P

Create a runtime directory whose `libnccl.so.2` points at the
`lite-collective` shim:

```bash
export LITE_RT=$ROOT/.tmp/mint-nccl-tests-runtime/mscclpp
mkdir -p $LITE_RT
ln -sfn $ROOT/nccl/build/libmscclpp_nccl.so $LITE_RT/libnccl.so
ln -sfn $ROOT/nccl/build/libmscclpp_nccl.so $LITE_RT/libnccl.so.2

export LITE_LD_LIBRARY_PATH=$LITE_RT:$MPI_HOME/lib:$CUDA_HOME/lib64:$ROOT/build:$ROOT/nccl/build:${LD_LIBRARY_PATH:-}
```

Small-message latency run:

```bash
$MPI_HOME/bin/mpirun -np 2 --host $HOST_SPEC --bind-to none \
  -x CUDA_VISIBLE_DEVICES=$GPU_LIST \
  -x LD_LIBRARY_PATH=$LITE_LD_LIBRARY_PATH \
  -x MSCCLPP_SOCKET_IFNAME=$IFNAME \
  -x MSCCLPP_HCA_DEVICES=$HCA \
  -x MSCCLPP_DEBUG=WARN \
  -x NCCL_DEBUG=WARN \
  $BIN -g 1 -b 8 -e 64K -f 2 -w 100 -n 1000
```

Large-message throughput run:

```bash
$MPI_HOME/bin/mpirun -np 2 --host $HOST_SPEC --bind-to none \
  -x CUDA_VISIBLE_DEVICES=$GPU_LIST \
  -x LD_LIBRARY_PATH=$LITE_LD_LIBRARY_PATH \
  -x MSCCLPP_SOCKET_IFNAME=$IFNAME \
  -x MSCCLPP_HCA_DEVICES=$HCA \
  -x MSCCLPP_DEBUG=WARN \
  -x NCCL_DEBUG=WARN \
  $BIN -g 1 -b 256M -e 256M -f 2 -w 20 -n 50
```

Expected 256 MiB result on L40/L41 is about `12.3 ms`, or `21.7 GB/s`.

## Run native NCCL P2P baseline

Create a runtime directory whose `libnccl.so.2` points at native NCCL:

```bash
export NCCL_RT=$ROOT/.tmp/mint-nccl-tests-runtime/nccl
mkdir -p $NCCL_RT
ln -sfn /home/yangz/nfs/zhongjie/nccl/build/lib/libnccl.so.2.29.7 $NCCL_RT/libnccl.so
ln -sfn /home/yangz/nfs/zhongjie/nccl/build/lib/libnccl.so.2.29.7 $NCCL_RT/libnccl.so.2

export NCCL_LD_LIBRARY_PATH=$NCCL_RT:$MPI_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
```

Small-message latency run:

```bash
$MPI_HOME/bin/mpirun -np 2 --host $HOST_SPEC --bind-to none \
  -x CUDA_VISIBLE_DEVICES=$GPU_LIST \
  -x LD_LIBRARY_PATH=$NCCL_LD_LIBRARY_PATH \
  -x NCCL_SOCKET_IFNAME=$IFNAME \
  -x NCCL_IB_HCA=$HCA \
  -x NCCL_P2P_NET_CHUNKSIZE=2097152 \
  -x NCCL_DEBUG=WARN \
  $BIN -g 1 -b 8 -e 64K -f 2 -w 100 -n 1000
```

Large-message throughput run:

```bash
$MPI_HOME/bin/mpirun -np 2 --host $HOST_SPEC --bind-to none \
  -x CUDA_VISIBLE_DEVICES=$GPU_LIST \
  -x LD_LIBRARY_PATH=$NCCL_LD_LIBRARY_PATH \
  -x NCCL_SOCKET_IFNAME=$IFNAME \
  -x NCCL_IB_HCA=$HCA \
  -x NCCL_P2P_NET_CHUNKSIZE=2097152 \
  -x NCCL_DEBUG=WARN \
  $BIN -g 1 -b 256M -e 256M -f 2 -w 20 -n 50
```

Expected 256 MiB result on L40/L41 is about `16.4 ms`, or `16.3 GB/s`.

NCCL `2.29.3` and `2.29.7` have similar performance on this testbed for this
case.  To compare with `2.29.3`, change both symlinks to
`/home/yangz/nfs/zhongjie/nccl/build/lib/libnccl.so.2.29.3`.

## Verify NCCL is using IB

For a one-point network-path check, run native NCCL with `INFO` logging:

```bash
$MPI_HOME/bin/mpirun -np 2 --host $HOST_SPEC --bind-to none \
  -x CUDA_VISIBLE_DEVICES=$GPU_LIST \
  -x LD_LIBRARY_PATH=$NCCL_LD_LIBRARY_PATH \
  -x NCCL_SOCKET_IFNAME=$IFNAME \
  -x NCCL_IB_HCA=$HCA \
  -x NCCL_P2P_NET_CHUNKSIZE=2097152 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_DEBUG_SUBSYS=INIT,NET \
  $BIN -g 1 -b 256M -e 256M -f 2 -w 20 -n 10 2>&1 | \
    tee /tmp/nccl_sendrecv_netcheck.log
```

Expected log fragments:

```text
NCCL INFO NET/IB: [0] mlx5_0:uverbs0:1/IB provider=Mlx5 speed=400000
NCCL INFO Using network IB
Channel ... via NET/IB/0/Shared
```

`GPU Direct RDMA Disabled ... distance 8 > 6` is expected for the no-GDR
host-staging path on this setup.

## Benchmark hygiene

- Use larger warmup counts than the `nccl-tests` defaults.  Low warmup can
  include lazy NCCL setup, staging-buffer setup, and pipeline fill time in the
  measured iterations.
- Recommended small-message setting: `-w 100 -n 1000`.
- Recommended large-message setting: `-w 20 -n 50`.
- Avoid drawing conclusions from `-w 10 -n 20`: it can make small-message lite
  latency look like `~1 ms`, and can make native NCCL 256 MiB throughput look
  like `7-8 GB/s` instead of the stable `~16 GB/s`.
- For NCCL large-message baseline, keep
  `NCCL_P2P_NET_CHUNKSIZE=2097152`; the default 128 KiB chunk size is slower
  for host-staged IB.
- Keep runtime `libnccl.so` symlinks in the shared NFS workspace, not `/tmp`.
  MPI ranks on the remote node cannot load local-node-only `/tmp` symlinks.
- After failed runs, check both nodes for stale `sendrecv_perf` or `mpirun`
  processes before re-running.

## Recent reference results

On L40/L41, GPU `0`, `mlx5_0`, `ibp55s0f0`:

| Backend | Command shape | 256 MiB time | 256 MiB BW |
| --- | --- | ---: | ---: |
| lite-collective | `-w 20 -n 50` | `12.37 ms` | `21.7 GB/s` |
| native NCCL 2.29.7 | `-w 20 -n 50`, `NCCL_P2P_NET_CHUNKSIZE=2097152` | `16.43 ms` | `16.3 GB/s` |

For small messages with `-w 100 -n 1000`, typical one-way P2P latency is about
`13-16 us` for lite-collective and `20-27 us` for native NCCL over the tested
8 B to 64 KiB range.
