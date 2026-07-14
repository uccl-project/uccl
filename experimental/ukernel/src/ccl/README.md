# CCL

Collective communication layer: planner, lower, SprayExecutor (multi-path
parallel execution engine for collectives), and three backends — DeviceBackend
(SM copy/reduce), TransportBackend (IPC/RDMA put), SignalBackend (peer
coordination).

## Build

```bash
cd experimental/ukernel/src/ccl
make clean
make -j$(nproc)
```

Common override:

```bash
make -j$(nproc) CUDA_HOME=/usr/local/cuda CONDA_LIB_HOME=/usr/lib SM=80
```

## Test

### Unit tests

```bash
make test-unit
```

| Binary | Coverage |
|---|---|
| `test_modules` | planner, lower, tile scheduling, DAG construction |
| `test_async` | SprayExecutor lifecycle with mock backends (allreduce / alltoall / concurrent) |
| `test_spray_executor` | multi-path dispatch priority, deferred re-queue, SignalBackend routing |

Unit tests run single-process with mock backends — no GPU or Communicator needed.

### Backend e2e tests

Each backend has a standalone two-process test using a real Communicator.

Build:

```bash
make test_transport_backend_e2e test_device_backend_e2e test_signal_backend_e2e SM=80
```

Run (two terminals, pick a GPU pair with P2P support):

**TransportBackend** — supports `--transport=ipc|rdma`:

```bash
# server
CUDA_VISIBLE_DEVICES=6,7 ./test_transport_backend_e2e --role=server --gpu=0
# client (IPC)
CUDA_VISIBLE_DEVICES=6,7 ./test_transport_backend_e2e --role=client --gpu=1
# client (RDMA)
CUDA_VISIBLE_DEVICES=6,7 ./test_transport_backend_e2e --role=client --gpu=1 --transport=rdma
```

**DeviceBackend**:

```bash
CUDA_VISIBLE_DEVICES=6,7 ./test_device_backend_e2e --role=server --gpu=0
CUDA_VISIBLE_DEVICES=6,7 ./test_device_backend_e2e --role=client --gpu=1
```

**SignalBackend**:

```bash
CUDA_VISIBLE_DEVICES=6,7 ./test_signal_backend_e2e --role=server --gpu=0
CUDA_VISIBLE_DEVICES=6,7 ./test_signal_backend_e2e --role=client --gpu=1
```

### SprayExecutor e2e test

Full-pipeline integration: DeviceBackend + TransportBackend + SignalBackend with
a real Communicator, exercising the complete AllReduce DAG.

Build:

```bash
make test_spray_executor_e2e SM=80
```

Run:

```bash
# server
CUDA_VISIBLE_DEVICES=6,7 ./test_spray_executor_e2e --role=server --gpu=0
# client
CUDA_VISIBLE_DEVICES=6,7 ./test_spray_executor_e2e --role=client --gpu=1
```

The test submits a 4 MB AllReduce correctness check (in=1.0, out=3.0 for rank 0;
in=2.0, out=3.0 for rank 1) and exits cleanly.

Troubleshooting:

- `Failed to connect to Exchanger`: a stale run is holding the port.

  ```bash
  pkill -f test_spray_executor_e2e
  ```

- Start the server first, then the client within ~3s (leader-ready timeout is
  3000 ms; raise it with `UHM_OOB_LEADER_READY_TIMEOUT_MS=30000`).

### P2P copy performance benchmark

Benchmarks three same-node P2P copy paths: ukernel `DeviceBackend` (several
`blocks_per_worker`), CUDA `cudaMemcpyPeerAsync`, and
`Communicator::send_put_async` (IPC put).

Build:

```bash
make test_perf_p2p_copy SM=80
```

Pick a GPU pair with P2P support:

```bash
nvidia-smi topo -p2p r
```

Run:

```bash
# server
CUDA_VISIBLE_DEVICES=6,7 ./test_perf_p2p_copy --role=server --gpu=0 --exchanger-port=6979
# client
CUDA_VISIBLE_DEVICES=6,7 ./test_perf_p2p_copy --role=client --gpu=1 --exchanger-ip=127.0.0.1 --exchanger-port=6979
```

The server terminal prints latency (µs) and throughput (GB/s) tables over sizes
from 1 KB to 1 GB.

Troubleshooting:

- `Peer access NOT supported` / `Cannot resolve remote IPC`: no P2P path between
  the GPUs — pick a pair shown as `OK` in `nvidia-smi topo -p2p r`.
- Stale port: `pkill -f test_perf_p2p_copy`, then retry with a fresh port.

### RDMA L2 cache flush test

Verifies GPU L2 cache coherence after RDMA write. Rank 0 writes a known
float pattern via RDMA into rank 1's GPU buffer. Rank 1 waits for the
signal, then reads the data through a selected backend path and validates
correctness on the host. Three test cases isolate different read paths:

| Case | Read path |
|---|---|
| `gpuMemcpy` | `cudaMemcpy` D2D (baseline) |
| `CollCopy` | DeviceBackend SM CollCopy kernel |
| `Reduce` | DeviceBackend SM Reduce kernel (sum with local data) |

IPC (same-host) should pass. RDMA may fail on pre-Hopper GPUs due to
stale L2 cache lines after the NIC writes directly to GPU DRAM.

Build:

```bash
make test_rdma_l2_flush SM=80
```

Run:

```bash
# server (rank 0)
CUDA_VISIBLE_DEVICES=6,7 ./test_rdma_l2_flush --role=server --gpu=0 --case=gpuMemcpy
# client (rank 1)
CUDA_VISIBLE_DEVICES=6,7 ./test_rdma_l2_flush --role=client --gpu=1 --case=gpuMemcpy
```

Substitute `--case=CollCopy` or `--case=Reduce` to test SM kernel paths.

### SprayExecutor AllReduce performance benchmark

Benchmarks AllReduce throughput for sizes 256 KB through 256 MB using the
full SprayExecutor pipeline.

Build:

```bash
make test_perf_spray_allreduce SM=80
```

Run:

```bash

pkill -f test_perf_spray_allreduce
ls /dev/shm/uk_cmpl_* 2>/dev/null && rm -f /dev/shm/uk_cmpl_*
nvidia-smi | grep test_perf

# server
CUDA_VISIBLE_DEVICES=6,7 ./test_perf_spray_allreduce --role=server --gpu=0
# client
CUDA_VISIBLE_DEVICES=6,7 ./test_perf_spray_allreduce --role=client --gpu=1
```

### Run everything

```bash
make test
```

## Notes

- `test-unit` covers planner, lowering, executor lifecycle, and multi-path dispatch.
- Each backend has a standalone e2e test; run manually in two terminals (see above).
- `test-integration` builds the p2p copy performance test.
- All e2e tests require two GPUs with P2P support on the same node.
- Use `SM=80` (or your GPU's compute capability) when building CUDA tests.
