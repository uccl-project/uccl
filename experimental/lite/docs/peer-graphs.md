# Owned graphs in the staged peer executor

`PeerStagedCollectives` can prepare and replay an owned CUDA graph containing a
fixed sequence of BF16 or FP32 collectives. This is a low-level execution API.
It does not itself qualify graph performance. The separate
[prepared graph selector](prepared-graphs.md) consumes this API for independent
graph admission; capture/group calls remain unsupported in the strict adapter.

The executor still requires a complete same-host CUDA peer-access graph. Rank
order follows the caller and need not match physical GPU or NUMA order. Its
declared storage capacity is 2–32 ranks; the validation below covers 2–4 L4s.

## Lifetime and ordering

Call `prepare()` first, then call `prepareGraph` collectively while the executor
is idle and outside capture. Every rank must provide the same operation/count
sequence and dtype. The preparation exchanges metadata before constructing any
graph and exchanges construction success before returning a handle. Each rank
may use different pointers, alignment and aliases, subject to scalar alignment
and capacity checks. Calls bind pointers permanently; caller-owned allocations
must remain valid until the graph is released or the executor is destroyed.

```cpp
auto graph = peer.prepareGraph({
    {NetworkOperation::AllGather, input, gathered, count},
    {NetworkOperation::ReduceScatter, gathered, output, count},
});
peer.launchGraph(graph, stream);
peer.synchronize();
peer.releaseGraph(graph);
```

Preparation captures a private stream and instantiates the graph without
executing the collectives. Replay performs no graph construction, CPU-worker
initialization or allocation in the executor. Host calls to one executor must
be serialized. An outer completion event orders graph and eager operations
across caller streams, including repeated launches of the same graph. GPU-side
publication/reader-retirement counters advance during execution, so capture
does not freeze a host-side communication epoch.

`launchGraph` and eager `enqueue` reject external stream capture. Applications
must also keep graph preparation and release outside capture. Graph handles are
instance-local and never recycled; launching a released handle rejects. Release
requires idle execution. Destruction waits for the recorded completion event,
destroys retained graph executables, and then frees their peer resources.

## Validation

Build with `make peer-graph-check` and the repository's MSCCL++/MPI configuration.
The checker uses one process per GPU and accepts `bf16|fp32 ITERATIONS`:

```sh
CUDA_VISIBLE_DEVICES=2,0,3,1 mpirun --bind-to none -np 4 \
  build/uccl_lite_peer_graph_check bf16 1000
```

The checker covers AG/RS/AR graph chains, in-place source/destination regions,
scalar tails, producer/consumer ordering, graph/eager transitions, repeated
cross-stream launches of the same executable, a delayed rank, rejected
operation/count disagreement, released handles and external capture rejection.
Every replay checks all input/output allocations and guards against an
independent CPU formula. It repeats construction/execution/destruction over two
communicator lifetimes. Integer reduction inputs remain exactly representable
in BF16 on the tested 2–4-rank layouts; this does not replace the existing scalar
oracle's general FP32 tolerance and bitwise AG edge-value checks.

Current local logs are `build/peer-graph-{bf16,fp32,three,two}.log` and
`build/peer-graph-memcheck.log`. Hardware is the four-L4, dual-NUMA Xeon Silver
4410Y host, with shuffled visible-device order. Four-rank runs cover both dtypes,
three-rank BF16 and two-rank FP32 cover non-four-rank layouts. Each uninstrumented
case runs 1,000 iterations per lifetime. Memcheck covers four-rank FP32 with 100
iterations per lifetime and checks device memory errors; known MSCCL++ CUDA
capability-query API errors are suppressed with `--report-api-errors no`.
These are correctness checks, with no measured baseline or performance claim.

The [prepared graph selector](prepared-graphs.md) now compares peer and native
NCCL graph execution. Transparent external capture and broader network graph
execution remain separate work.
