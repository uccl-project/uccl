# Dependency and shared-bandwidth cost model

`ExecutionPlan` now supports either serial `PlanStage` groups or an explicit
`PlanTask` dependency graph. Each task declares resource traffic, predecessor
indices and a known startup overhead. Empty-demand tasks represent dependency
events or explicitly supplied delays. This is a CPU-only advisory model; it
does not schedule CUDA work or independently admit an algorithm.

Ready tasks can start without waiting for unrelated tasks. Startup is the
maximum declared operational-resource latency plus task overhead, and occupies
no bandwidth. During data movement, the model progressively allocates max-min
fair fractions of each active task's isolated progress rate, constrained by
every resource it uses. Allocation is recomputed when tasks finish or become
ready. This defines the model's arbitration assumption; it is not a claim that
GPU memory controllers, copy engines or NICs implement that policy.

Shared resource IDs accumulate traffic, while independent directed PCIe paths,
GPU/host memories and NIC resources can overlap. Physical capacity bounds
constrain measured operations without supplying invented latency. A task made
only of physical bounds cannot produce an estimate. Missing rates/latencies,
unknown task overhead, invalid dependencies, cycles, duplicate IDs, arithmetic
overflow and excessive model size produce an unavailable estimate. A plan
cannot mix serial stages and task graphs.

The result includes the modeled finish time and task start/finish times, plus
three comparison quantities:

- `serialUs`: sum of isolated task costs, as if every task ran serially.
- `dependencyOnlyUs`: critical path with isolated task costs and no contention.
- `resourceLoadUs`: largest aggregate bandwidth service demand on one resource.

These are model quantities, not measured hardware timings or guaranteed bounds
on the full application. They expose why overlap is available or limited. The
C interface reports them as `resource_serial_us`, `resource_dependency_us` and
`resource_load_us`, alongside `resource_estimate_us`. Unavailable diagnostics
remain `-1`.

## Executable peer-tree integration

The first runtime consumer is the [source-ordered peer tree](peer-tree-runtime.md).
Its model now describes each rank's input copy, accumulator initialization,
forwarding depths, local consumption and completion. CUDA-stream work depends
on that rank's preceding operation. After a neighbor handshake, work depends
on the preceding work of the rank and its actual neighbors. It does not insert
a global barrier that the executor does not have. AllGather includes explicit
final reader retirement; AR/RS include their final conversion/copy.

Preparation compares the measured-edge spanning tree with deduplicated BFS
trees rooted at each participating GPU. The graph estimate selects a prior
among these bounded alternatives at the runtime's fixed-capacity AllReduce
basis. The selected tree remains fixed for the runtime lifetime. If none has
an estimate, preparation retains the original capability/locality prior.
The chosen tree still undergoes complete-buffer correctness and exact-request
paired qualification against the caller's native baseline. Graph and eager
admission remain independent.

This changes actual tree construction, not just a diagnostic number. A CPU
fixture gives neighboring edges higher bandwidth, making the edge-only prior
choose a chain; dependency/overlap scoring chooses a different tree with lower
modeled completion time. Unknown measurements preserve the original prior.

## Analytical checks

For three chunks crossing three 100-µs stages, with FIFO dependencies within
each stage, the test fixture produces:

| Resource assumption | Serialized work | Dependency path | Largest resource load | Modeled finish |
| --- | ---: | ---: | ---: | ---: |
| Independent GPU-memory / PCIe / NIC stages | 900 µs | 500 µs | 300 µs | 500 µs |
| PCIe stage also competes for input GPU memory | 900 µs | 500 µs | 600 µs | 700 µs |
| All stages share one resource | 900 µs | 500 µs | 900 µs | 900 µs |

These are synthetic analytical fixtures, not machine measurements. Additional
tests cover full duplex, uneven transfers, delayed arrivals, coupled resources,
dependency-only nodes, control delays, invalid graphs and unknown capacities.
Three hundred deterministic random DAGs check predecessor ordering, isolated
throughput limits, aggregate load, critical paths, serialized comparisons and
invariance under reversing task storage order. Tree fixtures independently
check generated dependencies against the path's neighbor retirement rules for
2/3/4/8/32 logical ranks.

## Scope and evidence

The tree generator supplies a movement-only prior with zero explicit launch and
polling overhead, as its earlier model did. Control/semaphore costs, SM
occupancy, arbitration details and unmeasured shared cuts are not newly claimed
as known. The distributed hierarchy does not yet have a complete task graph;
its measured path/cohort rail planning remains separate. These limitations are
why exact guarded timing still controls promotion.

`build/overlap-model-release/` contains final four-L4 tree/native transition, real
automatic-selection and forced-fallback regressions over two lifetimes,
including owned graphs and cross-stream ordering. All 36 real automatic tree
trials are correct and have finite graph diagnostics. None wins on this setup:
35 choices retain NCCL and 1 selects peer packet. The model update is not a
claimed speedup over NCCL. The transition fixture's labeled native delay is
used only for branch correctness.

Timing evidence uses the existing three ABBA rounds, two warmups and three
timed calls/block, maximum-rank host enqueue-through-completion. Hardware is
the documented dual-NUMA Xeon Silver 4410Y/four-L4 system; baseline is NCCL
2.28.7. No best-variant curve is used. Distributed regressions and artifact
hashes are recorded in the migration ledger.

`build/overlap-model-release/` also passes two four-L4 hosts through the shared C
interface after the common model/diagnostic changes, including automatic
network resource calibration and owned native graphs. Final source and binary
identities are in `build/overlap-model-artifacts.sha256`.

Earlier runs remain in `build/overlap-model-final/`
and `build/overlap-model-network/`; their separate timing observations are not
combined into a best-variant result. Unavailable graph estimates now clear all
partial comparison quantities as well as their task timeline.

The [direct-exchange graph](direct-exchange-model.md) adds a consumer of this
estimator with explicit sender chains, logical GPU/path traffic and conservative
publication/retirement boundaries. Unknown overheads remain unavailable.
