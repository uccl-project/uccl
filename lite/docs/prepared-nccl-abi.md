# Standard NCCL calls with prepared topology selection

`make nccl-prepared-abi nccl-prepared-abi-check` builds
`build/integration/libuccl_lite_nccl_prepared_abi.so`. This native-enabled
interposer exports the ordinary host collective/group/lifecycle entry points
and their `pnccl*` aliases. It is separate from the strict native-free runtime
and the handle-based C extension.

Initialize a native communicator normally, distribute an ID from
`ucclLiteGetUniqueId`, and collectively call `ucclLiteNcclPrepare` with the
native library/configuration, capacity, dtype and complete exact-request list.
Options and request structs are shared with `prepared_api.h`. The new header is
`integration/nccl_lite/prepared_abi.h` and is valid C. Set
`UCCL_LITE_REAL_NCCL` before loading the interposer; its implementation must match
`options.nativeLibrary`. Do not load it against the strict `libnccl.so.2` alias.

After preparation, ordinary `ncclAllReduce`, `ncclAllGather` and
`ncclReduceScatter` may use qualified UCCL executors. The communicator remains
the actual native `ncclComm_t`; unprepared communicators forward to native.
Preparation currently requires a ready blocking communicator, one process/GPU,
1–32 ranks and host-serialized calls. The opt-in
[automatic initialization mode](automatic-nccl-init.md) prepares a declared
workload from ordinary communicator initialization. Do not mix this integration with
`ucclLiteAttach` on the same communicator.

## Selection includes ABI overhead

The unified core discovers the participants and supplies candidates. The ABI
then separately compares each selected UCCL candidate against direct calls to
the original native library. Candidate timing invokes the exported wrapper,
including its CPU agreement, buffer matching, executor and completion record.
Native timing has no routing agreement or retained alternative executor.
Private guarded buffers and three ABBA rounds (two warmups, three timed calls
per block) retain 18 native and 18 dispatch samples. Setup, rendezvous and full
verification are outside timing; the statistic is maximum-rank host completion.
The existing >3% median/nonoverlapping-range rule applies without relaxation.

Only admitted executors are recreated for application calls. Until native
ownership is agreed, each global collective exchanges a fixed-size CPU envelope
covering sequence, operation/count/dtype, actual alias/alignment, current device
and mode. The full rank-layout vector must match a prepared request. A single
rank cannot independently fall back while peers enter an accelerator. This
agreement has a real cost and can cause a core winner to lose at the ABI level.
Reports expose both identities, the timing scope, core choice, attempted/correct/
admitted flags, all samples and call/resource counters.

## Native ownership and completion

An unadmitted request, unsupported reduction/dtype, grouped collective, capture
or local P2P use transfers future execution to native NCCL. Ownership is terminal
for that communicator. P2P and zero-count calls do not require nonparticipating
ranks to rendezvous; they mark a local native request, propagated by the next
global call. Zero-count calls add no CUDA dependency.

Before native work, the caller stream waits for the final accelerated completion
event. Grouped native calls may remain deferred until `ncclGroupEnd`, so the
adapter does not record completion after each deferred call. Captured native
calls import the frozen event with `cudaEventWaitExternal`. Its lifetime extends
through native communicator destruction, including after UCCL resources are
released. This preserves replay of caller-owned native graphs.

`ucclLiteNcclRelease` is collective, idle/outside capture and with groups closed.
It retires UCCL storage/workers while preserving the native communicator and
completion lease. It is safe to call again collectively. Destruction retires
UCCL work first and keeps the event alive until native destruction completes.
The adapter bypasses its own dispatch during internal preparation/native calls,
preventing recursive interposition or false group-depth accounting.

The initial ownership transition requires matching grouped/captured status
across ranks. A mismatch rejects collectively before submitting native work;
callers may close their groups and retry natively. Subsequent native execution
follows native ordering requirements. NVIDIA documents consistent operation
order within groups and uniform capture status among participating ranks.
([NCCL groups](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html),
[NCCL CUDA graphs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html))

## Limits

This remains a prepared-communicator integration, not automatic
application-wide replacement. Undeclared workload discovery, nonblocking
prepared communicators, concurrent calls on an attached communicator,
device-side/RMA communication integration and resuming acceleration after native
ownership are not implemented. Directly bypassing the interposer's native
entry points is outside its ordering contract.

Revoke/shrink require prior release of prepared resources. Native abort cannot
cancel already submitted UCCL kernels: active prepared resources are retained
for process-lifetime cleanup and acceleration cannot be re-enabled in that
process. Ordinary unprepared native communicators retain native behavior. CUDA
event errors from submitted UCCL work are also surfaced by the async-error
wrapper. This is not a claim of recoverable UCCL transport fault handling.

## Validation records

The checker verifies complete buffers and guards over two lifetimes, actual
cross-stream producer/consumer handoff, standard AR/AG/RS and profiler aliases,
groups, capture/replay after UCCL retirement, subset P2P, per-rank layout
mismatch, zero-count and non-SUM fallback. A separate native shim adds a fixed
20-ms host delay solely to force transition coverage; its identity is distinct
and these observations are not performance evidence against stock NCCL.

Two initial findings are preserved:

- The first grouped checker made AG consume an AR result in the same group.
  The installed native-only reference also observed the old input (`AR=40`,
  `AG=-99`). The formal grouped checker now uses independent group inputs,
  each dependent on the preceding accelerated call.
- Mixed group boundaries stalled both the prepared run and an independent
  native-only run. The latter reached its 30-second timeout. The prepared
  transition now rejects this mismatch before native submission.

The initial 3-ms delay fixture also recorded a 4-ms dispatch outlier and was
correctly rejected by the unchanged admission rule. The 20-ms fixture replaces
that value for branch coverage; it does not weaken the rule. Original logs are
retained under `build/prepared-abi-group-first.log`,
`prepared-abi-group-diagnostic.log`, `prepared-abi-native-group-reference.log`,
`prepared-abi-native-mixed-reference.log` and `prepared-abi-protocol/`.
Final validation scope and artifacts are tracked in the migration ledger.

Final records are in `build/prepared-abi-verified/` and
`build/prepared-abi-network/`. Four-rank FP32 covers accelerated AR/AG/RS,
profiler aliases, nested groups, mixed-mode rejection/recovery, capture with
changed replay inputs after resource retirement, subset P2P, per-rank layout
changes, zero-count calls, non-SUM fallback and group simulation. BF16 covers
all three collectives with 257-element tails, two-byte alignment, full guards
and native graph replay. Capture and P2P each pass four-rank memcheck with zero
memory errors; capability-probe API reports alone are suppressed.

Two real four-L4 hosts pass eager/group/capture/P2P/layout handoff and real
native/fallback checks. The real 64-element FP32 request retains native NCCL on
both one and two hosts. Untouched `all_reduce_perf` passes via `LD_PRELOAD` on
one process/four GPUs over 8 bytes–1 MiB, with validation enabled and zero
out-of-bounds values. The first preload invocation lacked the native loader
path; its failure and the initial independent single-GPU MPI launches are
retained separately. No speedup is inferred from these compatibility checks.

The final capture audit also found that native NCCL's internal ready-check
entered the async-error wrapper during capture. Internal queries now stay
native; only explicit user queries additionally inspect UCCL's event. The
failing diagnostic and successful correction are retained in
`build/prepared-abi-capture-query-diagnostic.log` and
`build/prepared-abi-capture-fixed.log`.

Source/client/library identities are recorded in
`build/prepared-abi-artifacts.sha256`. Hardware is the documented dual-NUMA
Xeon Silver 4410Y/L4 setup; native NCCL is 2.28.7. The delayed library is a
test-only implementation with a separate artifact identity.

The ABI now also dispatches qualified [general direct exchange](network-candidate-selection.md).
When multiple executor keys survive initial ABI qualification, their actual
simultaneous pool receives another full ABBA qualification. A shrinking pool is
rechecked; no native baseline block retains alternative resources. Reports keep
isolated and pooled samples separate. GPU inventory is mapped by UUID for each
process, including multi-host one-GPU visibility.
