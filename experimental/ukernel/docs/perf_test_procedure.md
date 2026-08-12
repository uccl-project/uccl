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

```bash
cd <repo>
git checkout uk-300
cd experimental/ukernel
# Build ONLY the target arch: SM=<compute capability> (B300=103, A40=86,
# L40S=89). The default 4-arch build is slow and sm_103 is not included.
make clean -f Makefile            # AMD machine: make clean -f Makefile.rocm
# Perf build (TMA bulk reduce + 224KB smem; ~15-25 min on B300):
make SM=<arch> REDUCE_ILP=4 REDUCE_SMEM_KB=224 TMA_REDUCE=1 \
    TMA_WARPSPEC=0 -j8 -f Makefile nccl   # AMD: ... -f Makefile.rocm nccl
# Fast validation build (C++-only checks, ~1 min relink): drop the TMA
# paths with VALIDATE=1:
make SM=<arch> VALIDATE=1 -j8 nccl
# artifacts: build/nccl/lib/libnccl.so + build/nccl/include/nccl.h
```

The fused reduce+copy work runs on the vector LD/ST path, so validation
builds can drop TMA; keep `TMA_REDUCE=1 REDUCE_SMEM_KB=224` only for
final perf builds. See [reduce_kernel.md](reduce_kernel.md) for the
kernel tuning knobs.

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

Find the native NCCL prefix first (if `nccl.h` is already on the default
CUDA include path, `NCCL_HOME` can be omitted — nccl-tests falls back to
the default search paths):

```bash
ldconfig -p | grep libnccl                    # runtime .so location
dpkg -L libnccl-dev 2>/dev/null | grep -E 'nccl\.h|libnccl\.so'   # Ubuntu/Debian
```

`NVCC_GENCODE` is generated from the GPUs actually present, so there is
no hardcoded arch list (each unique compute capability gets an entry):

```bash
export NVCC_GENCODE="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
  | sort -u | awk -F. '{printf " -gencode=arch=compute_%d%d,code=sm_%d%d", $1, $2, $1, $2}')"
echo "NVCC_GENCODE=$NVCC_GENCODE"
# AMD/ROCm: no nvidia-smi — pass the target archs explicitly, e.g.
# NVCC_GENCODE="--offload-arch=gfx90a" (see rocm_agent_enumerator).
```

Locate MPI too (OpenMPI; nccl-tests links `-lmpi`). `MPI_HOME` must
contain `include/` and `lib/` (or `lib64/`) directly:

```bash
readlink -f "$(which mpirun)"                 # real binary -> install prefix
find /usr /opt -name mpi.h 2>/dev/null | head -3
# Ubuntu/Debian OpenMPI: MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
# conda OpenMPI:          MPI_HOME=$CONDA_PREFIX
```

```bash
cd <repo>/thirdparty/nccl-tests
export NCCL_HOME=<native-nccl-prefix>                     # optional if nccl.h is on the default path
export MPI_HOME=<openmpi-prefix>                          # see "Locate MPI" above
make NVCC_GENCODE="$NVCC_GENCODE" -j$(nproc)
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

## 6. ROCm: build and run rccl-tests (AMD)

On AMD machines the perftests are the vendored **rccl-tests** (ROCm's
nccl-tests fork, same `nccl*` API). Build them against the normal RCCL
install; the binary is swapped to the shim at runtime via
`LD_LIBRARY_PATH` — the ROCm shim install ships as `librccl.so.1`
(RCCL's SONAME) alongside `libnccl.so.2`.

Pull **only** the rccl-tests submodule:

```bash
cd <repo>
git submodule update --init --depth 1 thirdparty/rccl-tests
```

Find the RCCL install (usually `/opt/rocm`; `NCCL_HOME` can be left
unset if RCCL is under `ROCM_PATH`):

```bash
find /opt/rocm /usr -name 'librccl.so*' 2>/dev/null | head
```

Generate `GPU_TARGETS` from the GPUs actually present instead of
hardcoding a gfx list:

```bash
export GPU_TARGETS="$(rocm_agent_enumerator | grep '^gfx' | sort -u | tr '\n' ',' | sed 's/,$//')"
echo "GPU_TARGETS=$GPU_TARGETS"
```

Build and run:

```bash
cd <repo>/thirdparty/rccl-tests
# rccl-tests looks for <MPI_HOME>/openmpi/{include,lib} — on Ubuntu/Debian
# that is MPI_HOME=/usr/lib/x86_64-linux-gnu (NOT the .../openmpi prefix
# that nccl-tests uses). See §3 for how to locate MPI.
make MPI=1 ROCM_PATH=/opt/rocm MPI_HOME=<mpi-prefix> \
    GPU_TARGETS="$GPU_TARGETS" -j$(nproc)
# binaries land in build/: all_reduce_perf, all_gather_perf, reduce_scatter_perf, ...

cd <repo>/experimental/ukernel
export LD_LIBRARY_PATH=$(pwd)/build/nccl/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0,1
mpirun --mca hwloc_base_binding_policy none -np 2 \
    -x LD_LIBRARY_PATH -x HIP_VISIBLE_DEVICES \
    ../../thirdparty/rccl-tests/build/all_reduce_perf -b 1M -e 256M -g 1
```

The native-RCCL comparison is the same run without the shim on
`LD_LIBRARY_PATH` (and without `LD_LIBRARY_PATH` pointing at
`build/nccl/lib`).

## 7. Project-owned benchmarks

Transport, GDRCopy, device-reduce, shim-sweep, AllToAll, and CE
contention benchmarks are documented in
[benchmarks.md](benchmarks.md).

## 8. Troubleshooting

- **mpirun must use `--mca hwloc_base_binding_policy none`** — CPU
  binding inflates small-message latency tens of times.
- **Propagate env with `-x`.** Setting `UK_CCL_DEBUG=2` (or
  `UK_CCL_PUT_PATH=...`) in the shell does NOT reach the rank processes;
  add `-x UK_CCL_DEBUG` / `-x UK_CCL_PUT_PATH` to mpirun. Debug levels:
  1 = executor, 2 = +transport, 3 = all. The per-op path is logged as
  `[pick rN] op[x] peer=y -> path=z` (z: 0=device, 1=IPC, 2=RDMA) —
  same-host ranks should show 1; anything else means transport selection
  went wrong.
- **A/B put-path knobs**: `UK_CCL_PUT_PATH=device|ipc|rdma` forces the
  same-host data path (unset = IPC for same-host, RDMA for remote peers).
  Use it to tell apart transport selection vs executor/device-kernel
  slowdowns.
- **Exchanger hang after a crashed run**: stale shared memory from a
  killed process makes both ranks wait for a leader that never comes:
  ```bash
  rm -f /dev/shm/uk_oob_kv_* /dev/shm/uk_cmpl_*
  ```
- **Cross-node**: set `NCCL_SOCKET_IFNAME` to the right NIC (the
  exchanger uses TCP; data path between nodes is RDMA).
- **Multi-card**: raise `-np` and add the GPUs to
  `CUDA_VISIBLE_DEVICES`; pick P2P-connected pairs per the topology.
