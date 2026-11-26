# Native communicator preparation extension

`make nccl-prepared-api nccl-prepared-api-check` builds
`build/integration/libuccl_lite_nccl_prepared.so` and its MPI correctness client.
The public C header is `integration/nccl_lite/prepared_api.h`. This extension
borrows a ready native NCCL communicator and exposes the
[unified preparation runtime](prepared-runtime.md) without C++ types. It exports
only versioned `ucclLite*` symbols; it does not interpose `nccl*` calls.

The application supplies the communicator, an explicit native library path,
its common native initialization/configuration identity, capacity and dtype.
Rank, device and size are read from that native communicator. Host ownership,
GPU UUIDs, visibility, NUMA groups and candidate eligibility are discovered by
the core. Applications do not select a machine-specific library or provider.
The default resource-probing flag also enables [automatic calibration of an
absent network profile](automatic-network-calibration.md) during eligible prepare.
Candidate diagnostics expose its identity, timing scopes and path/cohort counts.
`UCCL_LITE_NETWORK_PLAN_CACHE` optionally enables [persistent network plans](network-plan-cache.md).
Reports separate `from_cache` and the original qualification samples from fresh
`validation_samples_us`/`validation_baseline_samples_us`; graph admission stays
independent of eager cache records.

## Lifecycle

1. Initialize a blocking native NCCL communicator normally. Complete prior work
   and close native groups and external capture scopes. Select that
   communicator's CUDA device before preparing or executing a plan.
2. Call `ucclLiteGetUniqueId` on one rank and distribute its 128-byte value using
   the application's control plane. This is separate from `ncclUniqueId`.
3. Initialize `ucclLiteOptions` with `ucclLiteOptionsInit`, fill its required
   fields, then call `ucclLiteAttach` on each rank. Attachment copies strings
   and creates no accelerated transport resources. One session/process and
   one process/participating GPU are supported, for 1..32 ranks.
4. Collectively call `ucclLitePrepare` for each exact operation/count/layout.
   Layout arrays contain every logical rank's legal alias and exact minimum
   input/output alignment capped at 256 bytes. Counts follow NCCL semantics;
   reductions are SUM. Requests may explicitly retain native NCCL.
5. Execute the returned handle with `ucclLiteEnqueue`. It does no discovery,
   timing, allocation or independent fallback selection. A native winner is a
   prepared native plan, using the caller's communicator. Calls are host
   serialized; the runtime orders its own eager and graph work across streams.
6. Use `ucclLiteSynchronize` before direct calls to native NCCL. Finish direct
   native work before returning to this API. This extension cannot observe
   native calls made outside it.
7. Collectively finish and detach with `ucclLiteDetach` before destroying the
   native communicator. Detach retires submitted work and destroys only
   extension-owned state. Detached session handles never become valid again.

```c
ucclLiteOptions options;
ucclLiteOptionsInit(&options);  /* Check every result in real code. */
options.nativeLibrary = "/path/to/native/libnccl.so.2";
options.configurationIdentity = "application-native-init-v1;blocking-defaults";
options.capacityBytes = 1u << 20;
options.dtype = ncclFloat32;
ucclLiteSession session;
ucclLiteAttach(native_comm, &shared_uccl_id, &options, &session);
/* layouts[] has nranks entries, identical in logical order on every rank. */
ucclLiteRequest request = {
    ucclLiteAllReduce, count, layouts, nranks, 0};
ucclLitePlan plan;
ucclLitePrepare(session, &request, &plan);
ucclLiteEnqueue(session, plan, input, output, stream);
ucclLiteSynchronize(session);
ucclLiteDetach(session);
```

Owned fixed-binding graphs use `ucclLitePrepareGraph`, `ucclLiteLaunchGraph`
and `ucclLiteReleaseGraph`. Graph admission is independent of eager evidence;
distributed graphs currently retain native NCCL. Buffers remain alive through
release or detach. Graph and eager handles cannot be interchanged. Released
graphs cannot execute, but their immutable selection reports remain readable.

Errors return `ncclResult_t` and a thread-local `ucclLiteLastError` diagnostic;
exceptions do not cross the C boundary. Malformed request content goes through
the core's collective rejection envelope, including a null request on one rank.
Graph release also sends invalid tokens through collective rejection, so one
rank passing an eager token cannot leave peers waiting. Failed collective
prepare requires a subsequent successful prepare before any
execution. Invalid local session/output pointers are caller errors and cannot
participate in consensus. Do not have peers enter a collective after a local
attachment failure. Open NCCL groups, external capture and nonblocking native
communicators are outside this extension's contract; there is no claim of
transparent ABI forwarding or automatic group tracking.

## Selection evidence

`ucclLitePlanReport(session, plan, NULL, &bytes)` queries the required JSON buffer
size, including NUL. A short buffer is untouched and returns the required size.
The report includes the exact request, dtype, rank layouts, participants,
selected backend, decision reason, topology/executor/baseline identities and
all paired candidate and native samples. Diagnostics do not initialize GPU
resources. Domain codes are local=0, distributed=1, caller-only=2; backend codes
are caller=0, peer=1, host=2, peer-packet=3, network-hierarchy=4,
peer-tree=5 and direct-exchange=6. Distributed reports include both candidates,
per-rank direct configuration identities and optional final-pool timing arrays;
see [distributed candidate selection](network-candidate-selection.md).
The [source-ordered peer tree](peer-tree-runtime.md) uses backend code 5 and
reports `peer_tree_parents`; the broad peer flag enables its qualification.
Its [graph-model diagnostics](overlap-model.md) also expose serialized,
dependency-only and aggregate-resource costs; these are advisory quantities.

`resource_estimate_us=-1` is unknown, not zero cost. Eligibility and correctness
are distinct from winning. Samples use the core's host enqueue-through-backend
completion timing, maximum across ranks, with three ABBA rounds, two warmups
and three timed calls/block. The >3% median and nonoverlapping-range rule still
controls promotion. No measurements from historical machine-specific profiles
are promoted merely because GPU counts match.

## Validation scope

The C-only translation unit checks header compatibility. The CUDA/MPI client
checks full-buffer independent scalar references, preserved inputs and guards,
mixed per-rank aliases, minimally aligned pointers, cache/rejection recovery,
owned graph/eager stream dependencies, session/handle lifetime and continued
native communicator use after detach. It runs two attachment lifetimes. See
the migration record for the completed runs and artifact hashes.

Final runs are under `build/nccl-prepared-api-release/`: `local.log` and
`distributed.log` cover one and two four-L4 hosts; `six-ranks.log`, `hidden.log`,
`single.log`, `half.log` and `fallback.log` cover the restricted/native paths.
FP16 checks are eager only. `summary.json` records actual choices; these checks
do not establish a general performance gain. `build/nccl-prepared-api-artifacts.sha256`
binds the source, client and shared libraries to those runs.

This is the native-communicator C extension stage. Standard NCCL ABI
interposition, open-group completion semantics, external-capture handoff and
broader accelerated topology contracts remain separate unfinished work.
