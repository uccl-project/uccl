# Local device validation

Build native SM120 device code and exercise the production MSCCLPP FIFO without
an RDMA NIC, EFA, MPI, PyTorch, or NCCL. CUDA, a C++17 compiler, pthread, and
libnuma development files are the only dependencies.

```sh
make -C experimental/uccl_gin/tests/local SM=120 CUDA_HOME=/usr/local/cuda -j4
LOCAL_BIN=experimental/uccl_gin/tests/local/build/sm120
CUDA_VISIBLE_DEVICES=0 timeout 90s "$LOCAL_BIN/capabilities" --device 0
CUDA_VISIBLE_DEVICES=0 timeout 90s "$LOCAL_BIN/queue_smoke" --device 0 \
  --producers 32 --capacity 512 --commands 100000 --rounds 3
```

Repeat on the second physical GPU, then run both instances concurrently with
separate logs and exit statuses. Device numbers are relative to each process's
`CUDA_VISIBLE_DEVICES`. Each fixture owns its FIFO and GPU pointers; peer access
is reported as a capability and is never enabled or required.

`capabilities` checks a real mapped-host GPU write and production FIFO allocation.
`queue_smoke` compares every decoded field, rejects duplicate/missing commands,
checks signed atomic values and the FIFO reserved-bit transformation, reuses the
FIFO across rounds, and tests scalar completion. Producers are limited to FIFO
capacity; vary `--producers` and `--capacity` for backpressure and wraparound.
The consumer starts before the producer kernel and has a 60-second deadline.
Use the outer timeout to bound CUDA initialization and teardown as well.

For a rootless libnuma prefix, pass `NUMA_INCLUDE_DIR` and `NUMA_LIBRARY_DIR`.
An SM90 compile comparison uses `make SM=90`; architecture-specific directories
prevent stale object reuse. Use `cuobjdump --list-elf` to inspect native code and
`ldd` to verify that no EFA/MPI/NCCL runtime is linked.

These tests cover SM120 compilation and the real GPU-to-host command/completion
path. The consumer supplies test completions. EFA source consumption, receiver
visibility, and distributed NCCL-EP dispatch/combine require the network testbed.
