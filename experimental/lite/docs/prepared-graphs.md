# Graph-aware prepared local selection

`PreparedLocalRuntime::prepareGraph(request, input, output)` selects and prepares
one collective as an owned CUDA graph with fixed caller buffers. It uses the
same topology, dtype, exact count, per-rank layout and software-identity checks
as eager preparation, with a separate graph-mode consensus and qualification
cache. It never imports an eager winner or eager timing samples.

The generic staged peer backend supports owned BF16/FP32 graphs. A caller-owned
native NCCL adapter supports graph baselines when the loaded NCCL is at least
2.9. Other caller backends must explicitly implement the owned graph interface;
an eager-only fallback cannot qualify a graph request. Host packet and retained
peer packet currently have no owned graph implementation and are excluded.
This changes the core prepared interface; the strict NCCL ABI still rejects
external capture and does not provide transparent native fallback.

```cpp
auto graph = runtime.prepareGraph(request, input, output); // collective, idle
auto const& decision = runtime.graphSelection(graph);    // decision.graph == true
runtime.launchGraph(graph, stream);
runtime.synchronize();
runtime.releaseGraph(graph);                            // collective, idle
```

Each graph launch is collective across the prepared participants. One process
owns each GPU; host API calls to a runtime are serialized. The input/output
allocations must remain valid until collective release or runtime destruction.
All API calls occur outside caller capture and native group scopes. Ordinary
native calls outside the runtime require caller-managed synchronization, as in
the eager fallback contract. These restrictions follow the NCCL requirement for
[uniform capture and collective graph launches](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html).

## Independent qualification

With a caller fallback, preparation compares staged peer graph replay against
native graph replay on private guarded buffers at the exact requested layout.
Three ABBA rounds each use two warmups and three timed calls per block, retaining
18 candidate and 18 paired baseline samples. Timing is maximum-rank host
launch-to-completion in microseconds, including backend completion tracking.
Graph construction/instantiation, input generation, rank rendezvous and full
buffer verification are outside timing. Every warmup and timed call is checked.
The existing >3% median improvement and nonoverlapping-range rule governs
selection. A failing native baseline rejects preparation. Without a fallback,
the peer graph is the baseline and receives three checked blocks, nine timed
samples. No other graph algorithm is currently admitted.

The resource model remains an advisory data-movement estimate; graph launch
costs are not inferred from eager measurements. All qualification samples stay
in `LocalSelection`, with `graph=true`. Unavailable graph alternatives are
explicitly ineligible. Fallback-only preparation neither probes hardware nor
creates peer/host/packet resources when none are retained by earlier plans.

## Retained bindings and mixed execution

Before another qualification, idle preparation suspends previously bound graph
executables. It can then destroy UCCL candidate buffers and workers around
baseline blocks, preserving the original baseline-isolation contract. After
selection, it recreates executors required by old plans and rebuilds active
graphs from their fixed bindings. A live cache hit also performs identity
agreement and restores suspended graphs. Failed preparation disables execution
until a successful collective preparation; released graph handles are never
recycled or rebuilt.

Graph and eager calls share the runtime's completion event, while each backend
also preserves its own ordering. This chains different streams and backends
without CPU rendezvous, allocation or routing decisions in launch. Destruction
waits for execution, releases remaining graphs, and then destroys executor
resources. The caller retains ownership of a supplied native communicator.

## Validation

`make prepared-graph-check` builds a one-process-per-GPU client accepting
`bf16|fp32 peer|native|fallback|transitions NATIVE_LIBRARY`. It covers AR/AG/RS,
64/257/4096 elements, mixed in-place layouts and alignments, independent complete
buffer/guard checks, bitwise AG edge values, three input generations, retained
graph rebuilding, graph/eager cache separation, mode/binding disagreement,
failure recovery, capture/release rejection and two communicator lifetimes.
It also queues different retained graphs across streams before synchronization,
and checks ordinary native use after runtime teardown.

`transitions` adds a clearly identified **test-only 3-ms delay** to native graph
launches to force a peer winner. It verifies both peer→native and native→peer
producer/consumer chains across streams with full guards. Its times are not
performance evidence against unmodified NCCL. Memory instrumentation is also
correctness-only and must not establish a performance winner.

Current validation uses the documented four-L4 dual-NUMA Xeon Silver 4410Y host
and NCCL 2.28.7. Logs and source hashes live under `build/`; no best-variant curve
is reported. General external capture, graph-program fusion/selection across
multiple collectives, multi-host graphs and transparent ABI routing remain open.

`build/prepared-graph-native-final.log` retained native NCCL for all 18 FP32
operation/count/layout requests in both lifetimes. All paired samples are kept;
this is a negative result for the staged peer alternative on these requests,
not evidence about other algorithms or sizes. `prepared-graph-transitions.log`
passes the deliberately delayed BF16 mixed-backend fixture, `three.log` passes
three-rank FP32 peer execution, and `fallback.log` passes two-rank BF16 native-only
execution (each filename has the `prepared-graph-` prefix).

`build/prepared-graph-memcheck.log` passes the four-rank FP32 native comparison
and reports zero memory errors on every rank. Known MSCCL++ capability-probe
CUDA API errors are suppressed with `--report-api-errors no`; device memory
checking and an error exit code remain enabled. The native/eager regression logs
also pass after relinking. `prepared-graph-artifacts.sha256` and
`prepared-graph-hardware.csv` record current source/checker hashes and hardware.
