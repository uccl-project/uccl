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

Common overrides:

```bash
make -j$(nproc) CUDA_HOME=/usr/local/cuda CONDA_LIB_HOME=/usr/lib SM=80 \
    GDRCOPY_INCLUDEDIR=/home/user/gdrcopy/include GDRCOPY_LIBDIR=/home/user/gdrcopy/lib
```

DeviceBackend tests (e.g. `test_device_backend_e2e`, `test_spray_executor_e2e`)
additionally require GDRCopy. See the [ukernel README](../../README.md#install-gdrcopy)
for installation and custom path setup.

## Test

Unit tests are single-process (mock backends, no GPU needed). Every e2e
test is two processes — `--role=server` (rank 0) and `--role=client`
(rank 1) — and takes `--gpu`, `--exchanger-ip`, `--exchanger-port`
(all tests default to exchanger port **16998**; override
`--exchanger-port` on BOTH ends only when running several tests
concurrently). Conventions for all e2e sections below:

- Same-node: pick a GPU pair with P2P support (`nvidia-smi topo -p2p r`).
- Cross-node: same build on both nodes, client gets
  `--exchanger-ip=<SERVER_IP>`, and the port must pass the firewall —
  verify with `nc -vz <SERVER_IP> <PORT>` from the client and
  `ss -tlnp | grep <PORT>` on the server. Cross-node traffic is RDMA
  (data and signals), so an active mlx5 port must exist on both nodes;
  same-node runs work even with RDMA down.
- Node leadership: the exchanger is hierarchical — each node needs a
  local leader, and every test passes its `--gpu` index as `local_id`.
  Same-node: use two different `--gpu` values (0 = leader, 1 =
  follower). Cross-node: use `--gpu=0` on both nodes (each node gets
  its own leader). A client started with a nonzero `local_id` on a
  leaderless node fails with `[oob] non-leader ... shm store not found`.
- Stale state: `pkill -f <test>` and `rm -f /dev/shm/uk_cmpl_*` before
  reruns.

### Unit tests

```bash
make test-unit
```

| Binary | Coverage |
|---|---|
| `test_modules` | planner, lower, tile scheduling, DAG construction |
| `test_async` | SprayExecutor lifecycle with mock backends (allreduce / alltoall / concurrent) |
| `test_spray_executor` | multi-path dispatch priority, deferred re-queue, SignalBackend routing |

### test_transport_backend_e2e

Plain Put throughput/latency through TransportBackend over IPC and RDMA.

```bash
make test_transport_backend_e2e

# same-node (IPC is the default)
CUDA_VISIBLE_DEVICES=6,7 ./test_transport_backend_e2e --role=server --gpu=0
CUDA_VISIBLE_DEVICES=6,7 ./test_transport_backend_e2e --role=client --gpu=1

# same-node over RDMA
CUDA_VISIBLE_DEVICES=6,7 ./test_transport_backend_e2e --role=server --gpu=0 --transport=rdma
CUDA_VISIBLE_DEVICES=6,7 ./test_transport_backend_e2e --role=client --gpu=1 --transport=rdma

# cross-node (RDMA)
./test_transport_backend_e2e --role=server --gpu=0 --transport=rdma
./test_transport_backend_e2e --role=client --gpu=0 --transport=rdma --exchanger-ip=<SERVER_IP>
```

### test_device_backend_e2e

DeviceBackend SM copy tasks through a real Communicator. Same-node only —
the device path needs same-host P2P.

```bash
make test_device_backend_e2e

CUDA_VISIBLE_DEVICES=6,7 ./test_device_backend_e2e --role=server --gpu=0
CUDA_VISIBLE_DEVICES=6,7 ./test_device_backend_e2e --role=client --gpu=1
```

### test_signal_backend_e2e

Signal/WaitSignal matching latency through SignalBackend: same-node over
the shm signal ring, cross-node over the RDMA signal QP.

```bash
make test_signal_backend_e2e

# same-node
CUDA_VISIBLE_DEVICES=6,7 ./test_signal_backend_e2e --role=server --gpu=0
CUDA_VISIBLE_DEVICES=6,7 ./test_signal_backend_e2e --role=client --gpu=1

# cross-node
./test_signal_backend_e2e --role=server --gpu=0
./test_signal_backend_e2e --role=client --gpu=0 --exchanger-ip=<SERVER_IP>
```

### test_put_signal_e2e

Fused put+signal primitive: a single op delivers both the data and the
peer signal — IPC: the send worker writes the tag into the peer's shm
signal ring right after the copy; RDMA: the last chunk is a
write-with-imm carrying the tag. Verifies the core semantic: the peer
observes the signal only after the data has landed.

```bash
make test_put_signal_e2e

# same-node (IPC is the default)
CUDA_VISIBLE_DEVICES=6,7 ./test_put_signal_e2e --role=server --gpu=0
CUDA_VISIBLE_DEVICES=6,7 ./test_put_signal_e2e --role=client --gpu=1

# same-node over RDMA
CUDA_VISIBLE_DEVICES=6,7 ./test_put_signal_e2e --role=server --gpu=0 --transport=rdma
CUDA_VISIBLE_DEVICES=6,7 ./test_put_signal_e2e --role=client --gpu=1 --transport=rdma

# cross-node (RDMA; both sides must run this version — data QPs
# pre-post receive WQEs for write-with-imm)
./test_put_signal_e2e --role=server --gpu=0 --transport=rdma
./test_put_signal_e2e --role=client --gpu=0 --transport=rdma --exchanger-ip=<SERVER_IP>
```

Pass criteria: `can_fuse_put_signal=1`, `data-after-signal: verified`,
`[PASS]` on both sides.

Note: same-host IPC mapping opens are serialized across processes
(flock) and explicitly enable peer access before `cudaIpcOpenMemHandle`.
On A40 pairs, lazy-only enablement can return a mapping whose writes
never reach the owner's pages; the explicit enable + serialized open
avoids that driver behavior.

### test_spray_executor_e2e

Full-pipeline integration: DeviceBackend + TransportBackend +
SignalBackend with a real Communicator, exercising the complete
AllReduce DAG. Submits a 4 MB AllReduce correctness check (in=1.0,
out=3.0 for rank 0; in=2.0, out=3.0 for rank 1) and exits.

```bash
make test_spray_executor_e2e

# same-node
CUDA_VISIBLE_DEVICES=6,7 ./test_spray_executor_e2e --role=server --gpu=0
CUDA_VISIBLE_DEVICES=6,7 ./test_spray_executor_e2e --role=client --gpu=1

# cross-node
./test_spray_executor_e2e --role=server --gpu=0
./test_spray_executor_e2e --role=client --gpu=0 --exchanger-ip=<SERVER_IP>
```

Troubleshooting:

- `Failed to connect to Exchanger`: a stale run is holding the port —
  `pkill -f test_spray_executor_e2e`.
- Start the server first, then the client within ~3s (leader-ready
  timeout is 3000 ms; raise it with `UHM_OOB_LEADER_READY_TIMEOUT_MS=30000`).

### test_perf_p2p_copy

Benchmarks three same-node P2P copy paths: ukernel `DeviceBackend`
(several `blocks_per_worker`), CUDA `cudaMemcpyPeerAsync`, and
`Communicator::send_put_async` (IPC put), plus an RDMA section. The
server terminal prints latency (µs) and throughput (GB/s) tables over
sizes from 1 KB to 1 GB.

```bash
make test_perf_p2p_copy

# same-node (device/ipc sections)
CUDA_VISIBLE_DEVICES=6,7 ./test_perf_p2p_copy --role=server --gpu=0
CUDA_VISIBLE_DEVICES=6,7 ./test_perf_p2p_copy --role=client --gpu=1 --exchanger-ip=127.0.0.1

# cross-node (RDMA section only; device/ipc copies are same-node)
./test_perf_p2p_copy --role=server --gpu=0
./test_perf_p2p_copy --role=client --gpu=0 --exchanger-ip=<SERVER_IP>
```

Troubleshooting:

- `Peer access NOT supported` / `Cannot resolve remote IPC`: no P2P path
  between the GPUs — pick a pair shown as `OK` in `nvidia-smi topo -p2p r`.

### test_rdma_l2_flush

Verifies GPU L2 cache coherence after RDMA write. Rank 0 writes a known
float pattern via RDMA into rank 1's GPU buffer; rank 1 waits for the
signal, reads the data through a selected path and validates on the
host. Three cases isolate different read paths:

| Case | Read path |
|---|---|
| `gpuMemcpy` | `cudaMemcpy` D2D (baseline) |
| `CollCopy` | DeviceBackend SM CollCopy kernel |
| `Reduce` | DeviceBackend SM Reduce kernel (sum with local data) |

IPC (same-host) should pass. RDMA may fail on pre-Hopper GPUs due to
stale L2 cache lines after the NIC writes directly to GPU DRAM.

```bash
make test_rdma_l2_flush

# same-node
CUDA_VISIBLE_DEVICES=6,7 ./test_rdma_l2_flush --role=server --gpu=0 --case=gpuMemcpy
CUDA_VISIBLE_DEVICES=6,7 ./test_rdma_l2_flush --role=client --gpu=1 --case=gpuMemcpy

# cross-node
./test_rdma_l2_flush --role=server --gpu=0 --case=gpuMemcpy --transport rdma
./test_rdma_l2_flush --role=client --gpu=0 --case=gpuMemcpy --transport rdma --exchanger-ip=<SERVER_IP>
```

Substitute `--case=CollCopy` or `--case=Reduce` to test SM kernel paths.

### test_perf_spray_allreduce

Main performance vehicle: AllReduce/AllToAll throughput for sizes
256 KB through 512 MB using the full SprayExecutor pipeline (fused
PutSignal active).

```bash
make test_perf_spray_allreduce

# same-node
UK_CCL_DEBUG=1 CUDA_VISIBLE_DEVICES=6,7 ./test_perf_spray_allreduce --role=server --gpu=0 --kind=alltoall
UK_CCL_DEBUG=1 CUDA_VISIBLE_DEVICES=6,7 ./test_perf_spray_allreduce --role=client --gpu=1 --kind=alltoall

# cross-node
UK_CCL_DEBUG=1 ./test_perf_spray_allreduce --role=server --gpu=0 --kind=alltoall
UK_CCL_DEBUG=1 ./test_perf_spray_allreduce --role=client --gpu=0 --kind=alltoall --exchanger-ip=<SERVER_IP>
```

CLI reference:

| Flag | Default | Meaning |
|---|---|---|
| `--role=server\|client` | (required) | server = rank 0, client = rank 1 |
| `--gpu=<n>` | rank | index into `CUDA_VISIBLE_DEVICES` |
| `--kind=allreduce\|alltoall` | `allreduce` | collective kind |
| `--exchanger-ip=<ip>` | `0.0.0.0` / `127.0.0.1` | bootstrap exchanger address |
| `--exchanger-port=<n>` | `16998` | bootstrap exchanger port |
| `--sig-group=<G>` | `1` | one Signal/WaitSignal per G tiles per chunk pair |
| `--dev-fifos=<n>` | `1` | number of DeviceBackend workers (persistent kernels), one fifo each |
| `--dev-blocks=<n>` | `1` | `blocks_per_worker`: grid size of each worker kernel; one copy task is partitioned across its blocks |

`--sig-group G` aggregates one signal per G tiles per chunk pair
(default 1 = per tile). Sweep `1/2/4/8` to find the sweet spot. How
signals are emitted on each path (fused PutSignal):

| Path | G=1 | G>1 |
|---|---|---|
| same-node Device | device kernel writes the tag into the peer's shm signal ring after the P2P copy | every put fuses the same way; the wait counts G arrivals |
| same-node IPC | send worker writes the tag after the copy | every put fuses the same way (a put that cannot fuse is rerouted to IPC); the wait counts G arrivals |
| RDMA | write-with-imm on the put | every put carries an imm; the wait counts G arrivals |

Same-host groups always fully fuse (IPC is the guaranteed fallback);
remote groups fuse iff RDMA supports write-with-imm, else the group
falls back to one standalone signal. All fused channels feed the same
tag-matching layer, so per-op paths may mix within a group.

### Verifying the fused PutSignal changes

Run this checklist (both nodes on the same build) after touching the
signal/fusion paths:

1. `make test-unit` — planner/lower/executor regressions.
2. `test_put_signal_e2e`, IPC and RDMA — data-before-signal semantics.
3. `test_spray_executor_e2e` — numeric AllReduce correctness (3.0).
4. `test_perf_spray_allreduce` same-node and cross-node, plus
   `--sig-group 1/2/4/8` on both — every size must complete without
   hanging; compare small-size latency against the previous commit.
5. `test_signal_backend_e2e`, `test_transport_backend_e2e` (ipc and
   `--transport=rdma`), `test_device_backend_e2e` — unfused paths
   unaffected.

Known gap: `--sig-group` correctness is covered by completion/perf
only — the perf benchmark does not validate results numerically.

### Run everything

```bash
make test
```

## Debugging

Runtime debug output is gated by `UK_CCL_DEBUG` (see
`experimental/ukernel/include/util/uk_debug.h`). It is compiled in always
and costs nothing when unset.

```bash
UK_CCL_DEBUG=1 ./test_perf_spray_allreduce ...   # executor events (submit/enqueue/drain)
UK_CCL_DEBUG=2 ./test_perf_spray_allreduce ...   # + transport layer (signal matching, rings) + signal logs
UK_CCL_DEBUG=3 ./test_perf_spray_allreduce ...   # everything, incl. per-op trace + host profile
```

Other runtime switches:

- `UK_CCL_DEBUG=1` also counts Put ops per path (Device/IPC/RDMA); the
  perf benchmark prints the counters at the end.
- `UK_CCL_PUT_PATH=device|ipc|rdma` — force every same-host Put onto one
  path for A/B benchmarking (remote peers are always RDMA). Combine with
  `UK_CCL_DEBUG=1` to verify the forced distribution, and compare against
  the automatic multi-path LB (unset).
- `UK_CCL_DEV_FIFOS=<n>` / `UK_CCL_DEV_BLOCKS=<n>` /
  `UK_CCL_DEV_THREADS=<n>` — override DeviceBackend parallelism at
  executor creation (win over `SprayExecutorConfig` values): FIFOS is
  the number of workers (one persistent kernel per fifo), BLOCKS is
  `blocks_per_worker` (grid size of each worker kernel, tasks are
  partitioned across its blocks), THREADS is threads per block.
  When BLOCKS is unset, a per-GPU default is picked from the device's
  compute capability (A40-class 8, Hopper 16, Blackwell 32); THREADS
  defaults to 256 (the ILP reduce's sweet spot).
  `test_perf_spray_allreduce` also takes `--dev-fifos=<n>` /
  `--dev-blocks=<n>`.
- `UK_BAR1_WINDOW_MB=<n>` — fall back to IPC for remote device-put
  accesses beyond the BAR1 window (consumer GPUs with 256 MiB BAR1).

## Notes

- `test-unit` covers planner, lowering, executor lifecycle, and multi-path dispatch.
- Each backend has a standalone e2e test; run manually in two terminals (see above).
- `test-integration` builds the p2p copy performance test.
- Same-node e2e tests require two GPUs with P2P support.
- Use `SM=80` (or your GPU's compute capability) when building CUDA tests.
