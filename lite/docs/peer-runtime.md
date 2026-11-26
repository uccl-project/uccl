# Standalone peer runtime

The peer runtime can be built without the optional CPU-router, NCCL or MSCCL++
implementation. CUDA remains required:

```bash
make peer-runtime peer-runtime-check
./build/uccl_lite_peer_runtime_check
./build/uccl_lite_peer_runtime_static_check
```

Artifacts are `build/runtime/libuccl_lite_peer.so` and `.a`. The checks link
only this package and CUDA/system libraries, perform automatic preparation,
and verify AR/AG/RS with shuffled devices and scalar tails on four L4s.

Use `collective_call.hpp`, `collectives/peer_collectives.hpp` and `topology.hpp`
for the common call, executor and discovery interfaces. `PeerCollectives`
requires capability-supported BF16 out-of-place calls and an explicit idle
prepare boundary for each exact workload. Enqueue uses cached plans. Its
current device-argument capacity is four; larger groups need another registered
implementation.

`DevicePort` is plain shared call data. The old `cpu_router::PacketPort` type
spelling is retained for C++ ABI compatibility, but its definition now lives in
the common header and does not pull in router implementation headers.
`CollectiveCall::workload()` is implemented in the core runtime. The optional
router library includes that common object for existing consumers.

`JointCollectives` and `CollectiveDispatcher` remain optional router integration
layers. The standalone peer library contains no CpuRouter, PacketPipeline,
CpuExecutor, BackendPolicy or CollectiveDispatcher implementation. Dynamic
library dependency checks confirm no NCCL, MSCCL++ or CPU-router dependency.
The two-GPU application bridge now uses the standalone package; the separate
four-GPU application integration still links its explicitly requested packet
router path.

## Reusing exact qualified plans

Set `PeerTuning.planCache`, or `UCCL_LITE_PEER_PLAN_CACHE`, to a local plan file.
Automatic prepare first checks for matching topology/placement, executor ELF
build identity, collective, dtype, exact byte extent, layout and alignment.
It re-evaluates the same baseline-margin and non-overlapping-range rule used by
fresh qualification. Every eligible family must have the complete three-sample
record. Stale records miss; malformed or incomplete records reject.

A hit skips resource probing and candidate timing but still prepares required
execution resources. Device pointers, streams, events and CUDA graphs are never
serialized. A changed executable/shared library requires fresh qualification;
using the shared peer library permits reuse across processes with the same
executor artifact and exact discovery identity.

Updates merge under a file lock and replace the file atomically. Concurrent
writers retain distinct workload entries. Negative results are retained as
well: if no candidate clears admission, the saved record selects the direct
baseline. `PeerSelection.fromCache` reports whether a persisted record was used.
The cache's fixed timing-scope tag prevents mixing records from different
measurement protocols.
