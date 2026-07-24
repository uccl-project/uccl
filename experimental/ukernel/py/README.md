# Python Bindings

`ukernel_ccl` and `ukernel_p2p` are `torch`-based extensions built with nanobind
over the `transport + ccl + device` stack in `experimental/ukernel`.

## Build

```bash
cd experimental/ukernel/py
pip install nanobind
python setup.py build_ext --inplace
```

## Modules

- **`ukernel_ccl`** — collectives behind a persistent `ProcessGroup`:
  `allreduce` (ring), `alltoall` (equal-split, in-place), `barrier`, plus an
  async API (`allreduce_submit` / `alltoall_submit` → handle, with `poll` /
  `wait` / `status` / `error_message` / `release`).

- **`ukernel_p2p`** — point-to-point `Communicator`: peer connect/accept,
  `reg_ipc` / `reg_rdma` buffer registration, `send` / `signal` /
  `wait_data` blocking helpers, and async `send_put_async` /
  `send_signal_async` / `wait_signal_async` + `poll`.

---

## Benchmarks

All benchmarks live under `benchmarks/`. They run with `torchrun` (2 GPUs).

### `bench_collective.py`

Compares ukernel_ccl AllReduce and AllToAll throughput against NCCL.

| What it measures | AllReduce + AllToAll bandwidth across sizes 4 KB – 256 MB |
|---|---|
| Warmup / iters | 3 warmup, 20 timed iterations |
| Tile size | 512 KB |
| Output | Table with per-size per-backend latency (ms) |

```bash
cd experimental/ukernel/py
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 benchmarks/bench_collective.py
```

### `bench_p2p.py`

Compares ukernel_p2p, UCCL p2p (optional), and NCCL send/recv ping-pong
bandwidth.

| What it measures | Bidirectional ping-pong throughput across sizes 4 KB – 1 GB |
|---|---|
| Warmup / iters | 3 warmup, 20 timed iterations |
| Phases | ukernel → UCCL (if installed) → NCCL, run sequentially |
| Output | Table with per-backend latency (ms) and bandwidth (GB/s) |

```bash
cd experimental/ukernel/py

# ukernel + NCCL only (default)
CUDA_VISIBLE_DEVICES=1,7 torchrun --nproc_per_node=2 benchmarks/bench_p2p.py

# Include UCCL p2p (IPC mode)
UK_P2P_TRANSPORT=ipc CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 benchmarks/bench_p2p.py

# Force a specific transport
UK_P2P_TRANSPORT=rdma CUDA_VISIBLE_DEVICES=1,7 torchrun --nproc_per_node=2 benchmarks/bench_p2p.py
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `EXCHANGER_PORT` | 29610 | Bootstrap exchanger port |
| `UK_P2P_TRANSPORT` | `auto` | Force transport: `auto`, `ipc`, `rdma`, `tcp` |

---

## Tests

All tests live under `tests/`. They run with `torchrun` (2 GPUs, unless noted
otherwise). Conventions: same-node tests require a GPU pair with P2P support
(`nvidia-smi topo -p2p r`); stale state from previous runs should be cleaned
with `pkill -f python` before re-running.

### `test_collective.py`

Smoke test for the ukernel_ccl ProcessGroup.

| Tests | AllReduce correctness, equal-split AllToAll correctness, async API smoke (submit/poll/wait/status/release) |
|---|---|
| Data | Float32 tensors, element counts divisible by world_size |
| Exit | Prints `all tests passed` on success |

```bash
cd experimental/ukernel/py
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 tests/test_collective.py
```

### `test_p2p.py`

End-to-end send/recv through ukernel_p2p.

| Tests | Peer connect/accept, buffer registration (RDMA + IPC), send/signal/wait cycle, data validation |
|---|---|
| Validation | Checks received tensor values match what was sent |
| Exit | `P2P server/client test passed!` |

```bash
cd experimental/ukernel/py
CUDA_VISIBLE_DEVICES=6,7 RANK=0 python tests/test_p2p.py &
CUDA_VISIBLE_DEVICES=6,7 RANK=1 python tests/test_p2p.py &
```

Or via torchrun:

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 tests/test_p2p.py
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `EXCHANGER_PORT` | 29610 | Bootstrap exchanger port |
| `UK_P2P_TRANSPORT` | `auto` | Force transport |

### `test_tile_signal.py`

Verifies out-of-order tag delivery via `wait_signal_async` + `poll` API.
The sender emits tags [30, 10, 20], the receiver collects them out of order
and confirms all three arrived.

| Tests | Out-of-order signal matching, multi-tag async wait + poll |
|---|---|
| Timeout | 30 seconds |

```bash
cd experimental/ukernel/py
CUDA_VISIBLE_DEVICES=6,7 RANK=0 python tests/test_tile_signal.py &
CUDA_VISIBLE_DEVICES=6,7 RANK=1 python tests/test_tile_signal.py &
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `EXCHANGER_PORT` | 29611 | Bootstrap exchanger port |
| `UK_P2P_TRANSPORT` | `auto` | Force transport |

### `test_transport_paths.py`

Correctness suite for offset-based P2P transfers via `isend`/`irecv` +
`wait_finish`. Exercises:

| Case | Description |
|---|---|
| `oneway_full_rank0_to_rank1` | Full-buffer send from rank 0, zero offsets |
| `oneway_offset_rank1_to_rank0` | Non-zero send and recv offsets from rank 1 |
| `oneway_offset_rank0_to_rank1` | Non-zero send and recv offsets from rank 0 |
| `bidir_offset` | Simultaneous bidirectional send/recv with `wait_finish_multi` |

Each case validates that the payload arrives at the correct offset and that
guard regions on either side are untouched.

```bash
cd experimental/ukernel/py
TRANSPORT=ipc CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 tests/test_transport_paths.py
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `EXCHANGER_PORT` | 29620 | Bootstrap exchanger port |
| `TRANSPORT` | `auto` | Force transport: `auto`, `ipc`, `tcp`, `uccl` |

### `test_bidirectional_p2p.py`

Bidirectional one-sided PUT (`send_put_async` + `poll`) test. Both ranks
issue a PUT into each other's receive buffer, then validate the received
data on the CPU side.

| Tests | One-sided PUT, async completion via poll, bidirectional data validation |
|---|---|
| Data | 256K float32 elements per direction |
| Transport | IPC or RDMA only (TCP is rejected — requires signal/wait) |

```bash
cd experimental/ukernel/py
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 tests/test_bidirectional_p2p.py
```

### `run_transport_paths_suite.sh`

Runs `test_transport_paths.py` across all supported transports (ipc, tcp,
uccl) in sequence. Each transport gets its own exchanger port to avoid
collisions.

```bash
cd experimental/ukernel/py
GPU_IDS=6,7 tests/run_transport_paths_suite.sh

# Run a subset of transports
GPU_IDS=6,7 tests/run_transport_paths_suite.sh ipc tcp

# Cross-node: set MASTER_ADDR on both nodes
MASTER_ADDR=10.0.0.1 GPU_IDS=0 tests/run_transport_paths_suite.sh ipc rdma
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `GPU_IDS` | `6,7` | `CUDA_VISIBLE_DEVICES` value |
| `NPROC_PER_NODE` | `2` | Processes per node |
| `MASTER_ADDR` | `127.0.0.1` | torchrun master address |
| `MASTER_PORT` | `29790` | torchrun master port |
| `EXCHANGER_PORT_BASE` | `29800` | Base exchanger port (each transport gets `base + i`) |

---

## Quick Smoke

Run all same-node tests after a fresh build:

```bash
cd experimental/ukernel/py

# Collective
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 tests/test_collective.py

# P2P
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 tests/test_p2p.py

# Transport paths (all transports)
GPU_IDS=6,7 tests/run_transport_paths_suite.sh

# Tile signal
CUDA_VISIBLE_DEVICES=6,7 RANK=0 python tests/test_tile_signal.py &
CUDA_VISIBLE_DEVICES=6,7 RANK=1 python tests/test_tile_signal.py &

# Bidirectional PUT
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 tests/test_bidirectional_p2p.py
```
