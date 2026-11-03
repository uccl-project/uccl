# Cooperative P2P proxy scheduling

The P2P send and receive proxies normally run one synchronous vector transfer
to completion before dequeuing another request. An independent small transfer
can therefore wait behind an entire large vector. This opt-in implementation
lets eligible RDMA vectors share the proxy's submission budget and make progress
before an earlier independent vector completes.

Set `UCCL_P2P_COOPERATIVE_PROXY=1` before creating an `Endpoint`. Unset it, or
set it to `0`, to use the existing blocking scheduler. The shared MR ownership
change also applies when the switch is off. Hardware correctness and
performance measurements are still required before enabling it by default.

## Supported work and ordering

The cooperative path handles network `WRITEV` and `READV` tasks that already
qualify for `SendConnection::post_write_or_read_batch`: every IOV must fit one
transport chunk, and the connection must permit raw submission. Congestion
control, compression, multi-chunk IOVs and other ineligible work use the original
blocking methods. Each fallback acts as a barrier in its proxy: older tasks drain
first, and later tasks wait for it. Staged-network backends use the original proxy.
Scalar requests are not split by this change; small direct-submit scalar requests
also bypass these proxies as before.

Requests on the same connection retain FIFO order. Different connections can
overlap when their write destinations do not conflict. READ compares destination
ranges in the local process; WRITE conservatively compares numeric remote ranges.
The latter can serialize two independent remote processes that happen to use the
same virtual addresses. It does not infer a distinct address space merely from
a distinct connection. This limitation is checked explicitly by the mixed-load
WRITE benchmark. Source buffers shared only for reading need not be serialized.

The send and receive proxies have separate schedulers. This change does not add
ordering or a shared resource budget between those proxies, synchronous calls,
direct-submit operations, or different Endpoints. Callers must keep buffers
valid until completion and must coordinate dependencies across these domains.

## Scheduling and ownership

`p2p/util/proxy_scheduler.h` contains the production scheduler, independent of
GPU and verbs headers. `Endpoint::CooperativeProxyBackend` in `p2p/engine.cc`
adapts it to the existing raw submission, progress and completion operations.
The scheduler maintains admission order for dependency checks and a rotating
service cursor for eligible requests. Polling an empty CQ does not rotate the
cursor; only an accepted submission does. This avoids starving a request when
completion timing repeatedly aligns with the same polling phase.

Each proxy accepts at most 32 scheduler-owned requests. The existing
`max_one_sided_inflight_ops()` limit applies to their **combined** in-flight IOV
count. Each request may hold at most 64 credits, capped by the proxy limit. One
turn submits at most 16 IOVs and the adapter bounds a batch to 1 MiB (or one IOV
if a build permits larger individual IOVs). These are initial policy constants,
not hardware-tuned defaults. A request can still wait for an occupied credit or
an ordering dependency; the scheduler does not preempt already posted work.

Completion credits count IOVs, not CQEs or completion records. IB can report one
QP-tail wait covering several IOVs; EFA can report individual waits. Out-of-order
completion returns exactly the represented credits. A request publishes its
completion once, after all accepted work is terminal. Even a zero-IOV request
must respect earlier ordering dependencies.

Each active raw task retains its task metadata and shared ownership of the MR
registrations it uses. Registry locks are held only while looking up these
owners, not while DMA is in flight. Removing a registration from the registry
prevents new acquisitions; its transport registration is released when the last
owner finishes. These pins own registrations, **not the caller's data buffers**.
They do not make freeing an in-flight buffer valid. A registration removed while
a task is queued is detected before its first raw submission.

On shutdown, admission stops and requests already owned by the scheduler are
drained, including their unsent IOVs. Requests still in the input ring are outside
that guarantee, as in the existing implementation. Pending work is actively
polled because CQ completions do not wake the task eventfd.

## Failure boundaries

The generic scheduler accepts transient backpressure, partial submissions and
terminal failed completions from its backend, and drains every accepted wait
before releasing a failed request. A failed completion must mean that the
covered buffers are no longer in use; an arbitrary transport error cannot be
used as a safe completion.

The production adapter retains the existing transport's failure policy. Raw
provider failures after tracker allocation can still terminate the process, and
completion errors have not gained a new recovery protocol. Fake-backend failure
tests establish scheduler accounting, not recovery from a real QP failure.

## CPU verification

On Linux, these commands need only a C++17 compiler, Make and Python 3:

```bash
make -C p2p/tests test
make -C p2p/tests sanitize
```

The first test compiles the actual scheduler header. Its fake backend controls
submission acceptance and completion rounds; it does not choose requests,
allocate scheduling credits or decide when a task finishes. Assertions cover
small-request progress while a large request remains unfinished, FIFO and range
dependencies, grouped completions, limits, retries, partial submission failure,
failure draining, fallback barriers, single-credit fairness, shutdown admission
and empty-request ordering. Assertions use event order, not simulated latency.

The adapter test compiles the actual adapter and proxy-loop bodies extracted
from `engine.cc` with transport/device doubles. It exercises production wiring,
MR ownership, byte limits, IB/EFA-shaped waits, fallback eligibility and the opt-in
entry points. It also runs the production send loop with the switch off and on:
the former must submit the small request after the large request completes,
whereas the latter must submit it earlier. The blocking transport operation and
hardware completions use a deterministic fake clock; its ticks are not latency
measurements. Separate cases check that one request's MR can be released while
another stays active and that send/receive pins keep a shared MR alive.

This is intentionally separate from compiling the full engine:
stub declarations cannot prove compatibility with real transport headers. Build
the normal P2P targets with the relevant SDKs as an additional integration check.
The P2P Host Tests workflow runs the CPU checks without GPU runners.

## Hardware verification still needed

Use [the mixed-size benchmark](../p2p/benchmarks/benchmark_proxy_fairness.md) on
approved machines to compare fresh processes with the switch off and on. It
uses one initiating Endpoint and independent peer connections, validates payloads
outside the timed region, and reports small-request p50/p99, large and aggregate
throughput, and whole-process CPU usage. The script does not supply performance
evidence until it is run against real hardware.

Also validate single-connection workloads, read/write traffic, provider-specific
completion grouping, queue pressure and fallback transitions on the relevant
IB/RoCE or EFA setup. CPU tests do not establish real QP ordering, device failure
recovery, wire throughput, or a reduction in application latency.
