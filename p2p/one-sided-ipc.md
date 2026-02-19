# Plan
## Add benchmarks for read_ipc and write_ipc to benchmark_uccl.py

Add `--write-ipc` and `--read-ipc` modes to `benchmark_uccl.py`.

### write_ipc benchmark
- Server (rank 1): allocate GPU buffer, `advertise_ipc`, send blob to client
- Client (rank 0): receive blob, `write_ipc` in a loop, measure throughput
- Per size: server re-advertises, client re-writes

### read_ipc benchmark
- Server (rank 1): allocate GPU buffer with data, `advertise_ipc`, send blob to client
- Client (rank 0): receive blob, `read_ipc` in a loop, measure throughput

### Running
```bash
torchrun --nproc_per_node=2 p2p/benchmarks/benchmark_uccl.py --write-ipc
torchrun --nproc_per_node=2 p2p/benchmarks/benchmark_uccl.py --read-ipc
```

### Note
`--async-api` is not yet wired for write/read_ipc (sync only for now).
`write_ipc_async` pybind has a bug: returns only `success`, not `(success, transfer_id)`.
