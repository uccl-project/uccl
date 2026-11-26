# Direct-exchange traffic and dependency model

`include/uccl_lite/direct_plan.hpp` and `src/planner/direct_plan.cc` define the
direct sender schedule and its CPU-only cost model. The executor consumes the
same `directExchangeSteps` schedule: AR/AG stage their input once per chunk;
RS stages the corresponding shard before each destination transfer. Self
transfers are explicit local copies. The receive reduction remains in logical
source order.

The shared schedule is materialized at prepare, not at enqueue. The model does
not create GPU storage, transports or workers. It uses the common `Topology`,
`PlanCandidate`, `ResourceDemand` and task-graph estimator rather than a separate
bandwidth solver.

## Exact logical traffic

Let `N` be the participating rank count and `E` the bytes of one rank's AG
contribution, one destination's RS shard, or the full AR tensor. The operation
always stages input and consumes its private receive slots, including in-place
calls. Logical GPU reads plus writes across all ranks are:

| Operation | GPU memory bytes |
| --- | --- |
| AllGather | `(4*N*N + 2*N) * E` |
| AllReduce | `(3*N*N + 3*N) * E` |
| ReduceScatter | `(5*N*N + N) * E` |

Every remote directed edge carries `E` bytes. Actual IPC versus IB routes split
that payload into peer and network counters; an IB loopback route remains
network traffic even when its endpoints are on the same host. Tail chunks
contribute their actual length. Counters also expose staging/self/output copies,
reduction launches, caller-stream synchronizations, transport flushes and
agreement rounds.

The control-payload count uses the same status/call structs as the executor.
It counts the ring all-gather payload for three entry agreements and two
agreements per chunk. It excludes TCP/IP framing and other library-internal
traffic. GPU byte counts are logical accesses, not measured DRAM transactions:
cache hits, transaction granularity and hardware amplification are not inferred.

`DirectExchangeCollectives::traffic` and the common backend traffic query return
these counters for the actual prepared routes. Native-backed direct candidate
reports include them in the `traffic` object. A cached caller decision that does
not construct direct does not invent a fresh traffic report.

## Cost graph

`directExchangeModel` produces a bounded, bulk-synchronous advisory graph for
an exact workload and chunk size. A composed topology uses unique logical
device tokens and globally scoped resource IDs; CUDA ordinals are never assumed
to identify devices across hosts.

Each rank's staging, self copies and outbound transfers form a serial chain.
Chains on different ranks may overlap. A transfer demands source/destination
GPU memory and its declared operational path plus shared physical PCIe, NUMA or
NIC bounds. GPU memory is counted by the generator and cannot be duplicated in
the supplied path. Publication joins the send chains; each rank then performs
visibility handling and AG copies or source-ordered reduction. Retirement joins
receiver completion before the next chunk.

This is a conservative global-drain abstraction. It does not model individual
TCP progress, overlap of control-message tails with the next chunk, GPU cache
behavior or SM arbitration. Caller-supplied extra overheads must have a scope
compatible with the resource latencies, avoiding double counting. They default
to unknown. The graph is limited to 65,536 tasks and explicitly rejects larger
materializations; the cheap traffic counters remain available independently.

The shared estimator requires usable bandwidth/latency and overhead evidence.
Missing GPU memory, missing paths or unknown control costs yield no estimate.
Advertised link capacities can constrain measured operations, but cannot by
themselves provide an operational prediction or admit an algorithm. The runtime
continues to use exact guarded qualification for actual selection; a populated
direct cost graph is now [automatically calibrated](direct-model-calibration.md)
from scoped primitive measurements when resource probing is enabled.

## Checks

`make planner-check` builds `build/uccl_lite_direct_plan_check`. Its independent
byte oracle checks FP32/BF16, all three collectives, 2/3/4/8/32 logical ranks,
tails, sender order and every generated dependency. Synthetic calibrated
topologies demonstrate a memory-bandwidth-driven choice reversal between peer
direct and staged direct models, NIC shared-resource load, and chunk control
overhead. Unknown data, malformed paths and oversized graphs reject.

These analytical fixtures are not hardware performance measurements. CUDA
checks separately verify that the shared execution schedule preserves actual
results and that reported route traffic matches the prepared transports.

Final evidence is in `build/direct-model-verified/`, `build/direct-model-runtime/`
and `build/direct-model-strict/`: 17 runtime cases and 16 CPU checker executables
pass. The actual two-host L4 tests cover heterogeneous layouts, isolated GPU
views, FP32/BF16, full buffers/guards, source order, aliases, chunk tails and two
lifetimes. Two four-rank memchecks report zero device errors; only known
capability-probe API reports are suppressed. All 36 common C traffic reports
match independent formulas. Final staged/current identities are recorded in
`build/direct-model-artifacts.sha256`.

Hardware is the recorded dual-NUMA Xeon Silver 4410Y/L4 setup. Synthetic model
rates and delayed-native branch coverage are not hardware performance evidence.
No speedup, best-variant curve, or validation beyond two physical hosts/eight GPUs
is claimed by this stage.
