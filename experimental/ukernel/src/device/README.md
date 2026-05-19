# Device

## Build

```bash
cd experimental/ukernel/src/device
make clean
make -j$(nproc)
```

Common override:

```bash
make -j$(nproc) SM=80
```

Default builds are release-style. To enable device debug compilation, add
`DEBUG=1`:

```bash
make -j$(nproc) DEBUG=1
```

## Test

Run unit tests:

```bash
make test-unit
```

Run integration tests:

```bash
make test-integration
```

Run everything:

```bash
make test
```

## Benchmarks

```bash
make bench
./benchmarks/bench_device_fifo
./benchmarks/bench_device_full_fifo
./benchmarks/bench_device_sm_fifo
./benchmarks/bench_device_launch_vs_worker
```

`bench_device_full_fifo` launches one persistent worker kernel per FIFO. On
large GPUs, using every SM as an independent persistent-kernel worker can
exceed the hardware concurrent-kernel residency limit and leave some FIFOs
forever undrained. The benchmark now defaults to `64` workers and lets you
override it explicitly:

```bash
./benchmarks/bench_device_full_fifo 32
```

To compare `100` direct kernel launches vs `1` persistent worker processing
`100` tasks for `nop`, `copy`, and `reduce`:

```bash
./benchmarks/bench_device_launch_vs_worker 100 1000 100 4096
```

You can also sweep persistent-worker launch parameters:

```bash
./benchmarks/bench_device_launch_vs_worker 100 1000 100 4096 1 64 0
./benchmarks/bench_device_launch_vs_worker 100 1000 100 4096 1 128 0
./benchmarks/bench_device_launch_vs_worker 100 1000 100 4096 1 64 16384
./benchmarks/bench_device_launch_vs_worker 100 1000 100 4096 4 256 0
```

The benchmark reports three paths:

- `Launch kernels (single stream)`
- `Persistent worker (single enqueue)`
- `Persistent worker (batch enqueue)`

Arguments are:

```bash
./benchmarks/bench_device_launch_vs_worker [tasks_per_batch] [rounds] [warmup] [bytes] [num_blocks] [threads_per_block] [smem_size]
```

## Simple AR Multi-GPU Benchmarks

`bench_device_allreduce_p2p` uses MPI + CUDA IPC to exchange GPU buffer
handles across processes and exercises persistent-kernel allreduce / alltoall
with real cross-GPU P2P reads.

### Build

```bash
make bench_device_allreduce_p2p
```

MPI include/library paths are auto-detected via `mpic++ --showme`.  If your
MPI is not found add the env override:

```bash
MPI_CXX=/path/to/mpicxx make bench_device_allreduce_p2p
```

### Usage

```bash
mpirun -np <N> ./bench_device_allreduce_p2p \
    --bench=<allreduce|alltoall> \
    --workers=<W> --bytes=<B> --blocks=<NB> \
    [--sm] [--warmup=N] [--iters=N]
```

| Flag | Meaning | Default |
|------|---------|---------|
| `--bench=S` | `allreduce` or `alltoall` | `all` (both) |
| `--workers=W` | Persistent workers per rank | 1 |
| `--bytes=B` | Data size per rank (bytes) | 1024 |
| `--blocks=NB` | CUDA blocks per worker | 1 |
| `--sm` | Measure SM occupancy (idle+poll/sync/compute/tail) | off |
| `--warmup=N` | Warmup iterations | 5 |
| `--iters=N` | Timed iterations | 20 |

### Examples

```bash
# 2-GPU allreduce, 1 worker per rank, 1 MB data, SM on
mpirun -np 2 ./bench_device_allreduce_p2p \
    --bench=allreduce --workers=1 --bytes=1048576 --sm

# 4-GPU alltoall, 2 workers per rank, 4 blocks per worker
mpirun -np 4 ./bench_device_allreduce_p2p \
    --bench=alltoall --workers=2 --bytes=4194304 --blocks=4 --sm

# 8-GPU allreduce, sweep larger data
mpirun -np 8 ./bench_device_allreduce_p2p \
    --bench=allreduce --workers=1 --bytes=16777216 --sm
```

### Semantics

**SM occupancy** — When `--sm` is passed each worker records clock64
timestamps per dispatched task:

| Phase | Formula | What it captures |
|-------|---------|-----------------|
| idle+poll | T1 − T0 | FIFO-empty spinning + active poll to find a task |
| sync | T2 − T1 | `__syncthreads` overhead |
| compute | T3 − T2 | `dispatch_task` (copy / reduce) |
| tail | T4 − T3 | Tail publish + multi-block completion wait |

### Single-GPU (single-process) Bench

`bench_device_allreduce` is a single-process, single-GPU variant useful for
quick kernel-microbenchmark rounds without MPI:

```bash
make bench_device_allreduce
./bench_device_allreduce --nodes=2 --workers=4 --blocks=1 --bytes=1048576 --sm
```

| Flag | Meaning | Default |
|------|---------|---------|
| `--nodes=K` | Number of logical data sources | 2 |
| `--workers=M` | Workers per node | 1 |
| `--blocks=N` | CUDA blocks per worker | 1 |
| `--bytes=B` | Data per node (bytes) | 1024 |
| `--sm` | Measure SM occupancy | off |

## Notes

- `test-unit` covers task encoding, task manager behavior, worker lifecycle, enqueue semantics, dtype copy, and multi-fifo behavior.
- `test-integration` covers copy/reduce execution, same-flow reduce pipeline, and multi-block reduce.
- This module requires system-installed GDRCopy (`gdrapi.h` + `libgdrapi`).
- If `libgdrapi.so` is outside default linker paths, pass `GDRCOPY_LIBDIR`:

```bash
make GDRCOPY_LIBDIR=/usr/local/lib
```
