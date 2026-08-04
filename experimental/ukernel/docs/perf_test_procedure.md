# Perf test procedure (shim vs native NCCL, spray)

How to reproduce the ukernel CCL performance numbers on a multi-GPU
machine. Verified on an A40 pair (2 ranks); the shim's ring algorithms
are generic, so `-np 4/8` works for multi-card runs too.

## 1. Environment

```bash
# OpenMPI (needed by nccl-tests)
mpirun --version || sudo apt-get install -y openmpi-bin libopenmpi-dev

# pick a P2P-capable GPU pair
nvidia-smi topo -m
```

## 2. Build the shim

Default config is baked in: 256 threads/block, 8 blocks/worker, 1MB
tile sweet spot (see `coll_config.h`, `executor.h`).

```bash
cd <repo>
git checkout uk-300
cd experimental/ukernel
make clean -f Makefile            # AMD machine: make clean -f Makefile.rocm
make -j$(nproc) -f Makefile nccl  # AMD: make -j$(nproc) -f Makefile.rocm nccl
# artifacts: build/nccl/lib/libnccl.so + build/nccl/include/nccl.h
```

## 3. Get and build nccl-tests

Pull **only** the nccl-tests submodule (the other `thirdparty/`
submodules stay untouched):

```bash
cd <repo>
git submodule update --init --depth 1 thirdparty/nccl-tests
# without --depth 1 if you need the full submodule history
```

Build it against a **normal NCCL install** (needs CUDA + OpenMPI).
nccl-tests only needs NCCL's headers and library at build time; the
binary resolves `libnccl.so` at runtime via `LD_LIBRARY_PATH`, so the
same build serves both the shim and the native comparison — we never
compile perftests against the ukernel shim itself.

```bash
cd <repo>/thirdparty/nccl-tests
export NCCL_HOME=<native-nccl-prefix>                     # e.g. /usr/lib/x86_64-linux-gnu/nccl or a source build
export MPI_HOME=<your-openmpi-prefix>                     # e.g. /usr/lib/x86_64-linux-gnu/openmpi
make -j$(nproc)
# binaries land in build/: all_reduce_perf, all_gather_perf, reduce_scatter_perf, ...
```

## 4. nccl-tests sweeps (shim)

```bash
cd <repo>/thirdparty/nccl-tests/build
export LD_LIBRARY_PATH=<repo>/experimental/ukernel/build/nccl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1

MPI="mpirun --mca plm_rsh_agent sh --mca hwloc_base_binding_policy none \
  -np 2 -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES"

for t in all_reduce_perf all_gather_perf reduce_scatter_perf; do
  echo "== $t"
  $MPI ./$t -b 256K -e 256M -f 2 -g 1 -c 1 -n 20 -w 5
done

# key numbers to check (A40-class pair, 2 ranks)
$MPI ./all_reduce_perf     -b 256M -e 256M -f 2 -g 1 -c 1 -n 20 -w 5   # ~5.5ms
$MPI ./all_gather_perf     -b 256M -e 256M -f 2 -g 1 -c 1 -n 20 -w 5   # ~2.8ms
$MPI ./reduce_scatter_perf -b 256M -e 256M -f 2 -g 1 -c 1 -n 20 -w 5   # ~3.1ms
```

Reference (A40 pair, our runs): AR 256KB 82us / 16MB 419us / 256MB
5.5ms; AG 256MB 2.8ms; RS 256MB 3.1ms; all 0 wrong. Native NCCL 2.29.7
baselines: AR 256MB 6.3ms, AG 3.8ms, RS 3.8ms.

## 5. Native NCCL comparison

```bash
export LD_LIBRARY_PATH=<path-to-native-nccl-lib>:$LD_LIBRARY_PATH
ldd ./all_reduce_perf | grep nccl    # confirm it points at native
# re-run the same sweeps above
```

## 6. Spray (native ukernel executor) benchmark

```bash
cd experimental/ukernel/src/ccl && make test_perf_spray_allreduce

# two terminals (or background the server)
CUDA_VISIBLE_DEVICES=0,1 ./test_perf_spray_allreduce --role=server --gpu 0 --dev-blocks=8 &
CUDA_VISIBLE_DEVICES=0,1 ./test_perf_spray_allreduce --role=client --gpu 1 --dev-blocks=8

# AllToAll: add --kind=alltoall   (256MB ~2.9ms, beats native Send/Recv-group)
# Multi-block: --dev-blocks=1|4|8|64 to sweep SM usage
```

## 7. Troubleshooting

- **mpirun must use `--mca hwloc_base_binding_policy none`** — CPU
  binding inflates small-message latency tens of times.
- **Exchanger hang after a crashed run**: stale shared memory from a
  killed process makes both ranks wait for a leader that never comes:
  ```bash
  rm -f /dev/shm/uk_oob_kv_* /dev/shm/uk_cmpl_*
  ```
- **Cross-node**: set `NCCL_SOCKET_IFNAME` to the right NIC (the
  exchanger uses TCP; data path between nodes is RDMA).
- **Multi-card**: raise `-np` and add the GPUs to
  `CUDA_VISIBLE_DEVICES`; pick P2P-connected pairs per the topology.
