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
```

## Notes

- `test-unit` covers task encoding, task manager behavior, worker lifecycle, enqueue semantics, dtype copy, and multi-fifo behavior.
- `test-integration` covers copy/reduce execution, same-flow reduce pipeline, and multi-block reduce.
- If `gdrcopy` is missing, initialize submodules first:

```bash
git submodule update --init --recursive
```
