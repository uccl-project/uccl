# L40/L41 P2P send/recv benchmark runbook

This runbook records the commands used to compare the `lite-collective` NCCL
compatibility layer against native NCCL `sendrecv_perf` on the L40/L41 two-node
testbed.

The benchmark exercises `ncclSend`/`ncclRecv` only.  In the `mscclpp` backend,
`sendrecv_perf` loads `lite-collective/nccl/build/libmscclpp_nccl.so`; in the
`nccl` backend, the same MPI `sendrecv_perf` binary loads the native NCCL
shared library from `/home/yangz/nfs/zhongjie/nccl`.

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