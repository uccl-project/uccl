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
./benchmarks/bench_device_launch_vs_worker 100 1000 100 4096 64 0
./benchmarks/bench_device_launch_vs_worker 100 1000 100 4096 128 0
./benchmarks/bench_device_launch_vs_worker 100 1000 100 4096 64 16384
```

The benchmark reports three paths:

- `Launch kernels (single stream)`
- `Persistent worker (single enqueue)`
- `Persistent worker (batch enqueue)`

Arguments are:

```bash
./benchmarks/bench_device_launch_vs_worker [tasks_per_batch] [rounds] [warmup] [bytes] [threads_per_block] [smem_size]
```

## Notes

- `test-unit` covers task encoding, task manager behavior, worker lifecycle, enqueue semantics, dtype copy, and multi-fifo behavior.
- `test-integration` covers copy/reduce execution, same-flow reduce pipeline, and multi-block reduce.
- If `gdrcopy` is missing, initialize submodules first:

```bash
git submodule update --init --recursive
```
