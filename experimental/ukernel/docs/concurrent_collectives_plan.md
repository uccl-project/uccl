# Concurrent collectives across streams — problem statement and solution route

Date: 2026-09-05. Status: analysis/plan (no measurements yet).

## 1. Motivation

Single-op shim-vs-native comparisons on L40S are mixed but the hard cells
are real:

- same-node AllReduce 16M+ wins at S2/S4 (+9-19%) and S8 is parity, but
  1-4M is host-floor-bound (native 1.5-3x ahead, e.g. S4 4M 13.0 vs
  21.3 GB/s);
- cross-node X16 AllReduce is 0.65-0.84 of native at 16M+ and ~1/3 at
  1M; X16 AllToAll remains 2.0 vs 4.4-4.7 GB/s.

Real workloads, however, do not issue one collective at a time: FSDP
issues AllGather/ReduceScatter for several parameter groups, pipeline
and TP/EP layers overlap their collectives with compute, and user code
commonly keeps two or more CUDA streams busy. NCCL also receives these
calls concurrently, but whether two collective operations on different
streams *actually run in parallel* inside NCCL is implementation
behavior, not a guarantee.

Hypothesis: our single-op gap is a *host* critical path (per-op
dispatch, per-hop signal/transport/device drain) rather than a GPU
capacity limit. If several collectives run concurrently, each stream's
CPU-GPU orchestration can overlap with the others' device work — a
software-pipelined / CCL-level fused regime — and the *aggregate*
throughput may match or beat native even where each individual op loses.

## 2. Current architecture facts relevant to concurrency

- One `SprayExecutor` per communicator. Multiple submitted runs live in
  `comm->pending`; a single enqueue thread dispatches all ready runs and
  single drain threads per backend (device/transport/signal) complete
  them.
- Each run is stream-ordered on its user stream: an input event is
  recorded on the user stream and polled by the enqueue loop; completion
  publishes a monotonic done flag that the user stream waits on with
  `cudaStreamWaitValue32`. So several runs on different user streams can
  be in flight at once without blocking each other's *streams*.
- The device side is one persistent multi-block worker per executor:
  concurrent runs feed the same task FIFO and share the `B` worker
  blocks. There is no per-stream device worker, and there must not be
  one: the SM budget rule caps total resident blocks at native channels.
- AllToAll and standalone AllGather are pure CE paths (0 worker SMs);
  their copies are externalized on the user stream or ride the IPC
  engine. The IPC adapter already has a per-peer stream pool
  (`UK_CCL_IPC_STREAMS_PER_PEER`).
- HostProf (B300 1M, 2026-09-04) shows dispatch is ~0.2-0.6us per
  collective while signal/device drain grows to 50-110us per collective
  at 8 ranks — the cost that concurrency could hide.

## 3. Two distinct interpretations of "CCL-level fusion"

### 3a. Parallel orchestration (software pipelining)

Each stream keeps its own submission path; host costs of op A overlap
with device work of op B. No plan merging. This is the cheapest and the
one the user's note describes ("cpu-gpu 开销只在各自 stream 的
executor").

### 3b. Plan-level fusion of concurrent ops

When several collectives on different streams become ready at about the
same time, merge them into one executor run/DAG: one prepare, one
dispatch batch, pooled signal/flag slots and scratch, task streams
interleaved in the same FIFO; completion is still published per original
op so each user stream only waits on its own slice. This amortizes the
per-op host fixed cost further and lets CE/IPC transfers from different
ops go back-to-back.

Both interpretations keep the invariant: total resident SM blocks
`<=` native channels for the placement; AllToAll/AllGather stay 0-SM CE.

## 4. Where the win should come from

1. **Host-overhead hiding.** Per-op dispatch/drain is one stream's host
   thread; with c concurrent ops the GPU/CE pipeline can stay fed even
   if no single op saturates it.
2. **CE/PCIe utilization.** IPC copies from multiple ops fill the
   per-peer stream pool and copy engine; worker reduce tasks of one op
   can overlap CE copies of another (SM + CE overlap).
3. **Aggregate vs NCCL.** If NCCL serializes concurrent stream calls on
   its communicator/progress machinery, native aggregate throughput
   scales weakly with c, and shim's parallel regime overtakes even where
   the single-op ratio is below 1.
4. **Fixed host budget.** Enqueue is O(ops), drain is O(ops x hops); for
   small messages this fixed cost dominates. Parallel dispatch converts
   it from critical path to background.

## 5. Where it could fail (risks to measure first)

- **Native may already scale.** NCCL's progress thread may genuinely
   overlap two collectives from different streams; then the comparison
   is aggregate-vs-aggregate and our per-op overheads still matter.
- **Single host enqueue thread.** Today all runs of a communicator are
   dispatched by one thread; concurrency across streams inside one comm
   may not hide host cost until dispatch is per-run/per-stream.
- **Shared hardware ceilings.** PCIe bus and one CE engine bound
   same-node aggregate; NVLink aggregate on B300 is likewise finite.
   Gains must be measured relative to native under the same c.
- **Cross-node proxy.** The RDMA fused proxy is likely host-serialized;
   X16 concurrency may not help until per-QP/per-proxy parallelism or
   multi-QP striping exists (ties into the existing X16 open gap).
- **Resource caps and correctness.** Flag/signal slot limits, scratch
   growth, plan-cache size, idle-exit coordination across runs, and
   same-communicator thread-safety/group semantics all need stress
   (existing D3/D4 items).
- **SM budget.** Per-stream executors each with B blocks would violate
   the rule; the device worker must remain shared with `B` total blocks.

## 6. Experiment plan (validation before implementation)

Harness: `c` CUDA streams each run a repeated collective on its own
stream; measure wall aggregate busbw, per-op latency (p50/p99), and
overlap efficiency (sum of per-op device-busy time / wall). Same total
work is also run sequentially for reference. Everything validated 0
wrong.

Dimensions:

- `c` in {1, 2, 4, 8}; L40S S2/S4/S8 and X16 (X16 is the key win
  target); sizes {1M, 16M, 256M};
- same op (AllReduce only) and mixed ops (AllReduce + AllGather +
  AllToAll, exercising SM + CE overlap);
- same communicator vs separate communicators per stream;
- staggered starts to separate "overlap quality" from "burst queueing";
- identical harness against native NCCL (also tells us whether native
  overlaps streams at all).

Decision gates:

- G1: does *current* shim (one comm, c streams) aggregate beat
  sequential shim and native-concurrent? If yes → tune parallel dispatch
  and ship; if no → identify the serialization point (enqueue thread,
  FIFO, signal/flag caps, drain threads).
- G2: does plan-level fusion (3b) add a further win beyond parallel
  dispatch at c=2/4? If yes at small/mid sizes → implement fusion for
  the common FSDP/TP patterns.
- G3: does concurrency help X16? If not → parallel proxy/QP route.

## 7. Implementation route

### Phase A — measurement on current code (no semantic change)

- Build the stream-concurrency harness; run G1 matrix on L40S
  (same-node + X16), record aggregate and per-op latency.
- Stress correctness: 2-8 streams, sizes 1M-256M, 0 wrong; find any
  deadlock/resource-cap issue in concurrent runs.

### Phase B — parallel dispatch (if G1 is negative or partial)

- Decouple per-run submission from the single enqueue loop: a dispatcher
  per active stream, or a small dispatch thread pool, feeding one shared
  device FIFO and the shared transport/signal backends.
- Keep the device worker shared and its total blocks `<= B` (the SM
  budget). This is the architectural version of "each stream has its own
  executor on the CPU, one GPU worker".
- Keep per-run stream gates and completion flags unchanged.

### Phase C — plan-level fusion of concurrent ops

- When `k` ops from different streams are ready within a short window,
  build one merged plan (disjoint buffers, shared ring/topology and
  signal namespace); dispatch once; publish one completion flag per
  original op (or per stream) so each user stream's WaitValue fires
  independently.
- Target first: 2-4 concurrent same-shape AllReduces (FSDP), then
  mixed AR + AG + A2A to overlap SM reduce with CE copies.
- Keep 0-SM CE collectives out of the worker: fuse only among ops that
  share the data path engine.

### Phase D — cross-node and proxy concurrency

- Parallel proxy workers or per-stream QP assignment for X4/X8/X16;
  revisit multi-QP striping for the RDMA path. Gate G3 decides depth.

## 8. Metrics and acceptance

- Aggregate busbw under c streams, shim vs native (same harness).
- Per-op p50/p99 latency under concurrency (should not regress vs
  sequential shim beyond scheduling noise).
- Host CPU occupancy (enqueue/drain threads) to prove the overhead moved
  off the critical path.
- SM accounting: total resident blocks <= native channels; AllToAll and
  standalone AllGather remain 0-SM CE.
- All cells 0 wrong; same-communicator concurrent stress and
  cudaGraph/group semantics covered by D3/D4 checks.

## 9. Open questions for the author

1. Target concurrency level for the real workload story: 2 streams
   (FSDP) or up to 8 (pipeline/EP)?
2. Should concurrent collectives share one communicator (drop-in
   semantics) or may workloads use separate communicators per stream
   (easier parallelism, less drop-in fidelity)?
3. Which real workload should the e2e validation use once the
   microbenchmark gates pass?
