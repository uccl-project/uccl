# Device

`device` contains GPU-side worker, FIFO, and task execution infrastructure.

## Build

```bash
cd experimental/ukernel/src/device
make clean
make -j$(nproc)
```

Common overrides:

```bash
make -j$(nproc) SM=80
make -j$(nproc) DEBUG=1
```

## Test

```bash
make test-unit
make test-integration
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

`bench_device_full_fifo` defaults to 64 workers to avoid oversubscribing persistent-kernel residency on large GPUs.

Override worker count:

```bash
./benchmarks/bench_device_full_fifo 32
```

Launch-vs-worker benchmark arguments:

```bash
./benchmarks/bench_device_launch_vs_worker [tasks_per_batch] [rounds] [warmup] [bytes] [num_blocks] [threads_per_block] [smem_size]
```

## Notes

- Requires system-installed GDRCopy (`gdrapi.h` + `libgdrapi`).
- If `libgdrapi.so` is outside default linker paths, pass `GDRCOPY_LIBDIR=/path/to/lib`.
