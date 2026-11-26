# NCCL drop-in validation status

The strict two-host FP32 replacement is now qualified on the recorded two
four-L4 hosts: all 126 size/layout points pass the original-UCCL zero-regression
comparison, and native NCCL fallback is forbidden. See the
[qualification report](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/qualified_release/README.md)
and [current usage](../integration/nccl_lite/MSCCLPP.md).

The notes below are the historical snapshot of the separate mixed-routing
integration. Its native-fallback behavior and old admission status do not describe
the strict replacement above.

## Historical handoff (2026-11-16)

The unified binary supports the four-local-L4 BF16 packet path and the two
four-L4 FP32 network path, with real NCCL fallback. The count-zero AR/AG/RS
fast path and empty/nonempty saved-graph ownership tests are implemented.
Current correctness evidence is archived in
`benchmarks/data/dropin_empty_graph_4l4_2x4l4_20261116/`: real two-host ABI
stress, local/network nccl-tests Graph replay, four-GPU peer/staged cases,
router policy/dispatcher/graph checks, and retained CUDA target builds pass.
Both interposers export exactly 19 NCCL wrappers.

Performance is **not certified**. The pre-empty-fix local admission selected
AR counts 32/256 (minimum 1.113x/1.147x), but independent 8 B–1 GiB deployment
failed 9/160 strict rows. Network mixed-curve regressions also remain open.
Neither the retired nor disabled-keep diagnostic became production policy.

The changed binary's independent 256 KiB mixed deployment screen failed
**19/88** strict 5% rows (correctness passes), so session 25535 terminated
without launching the 1 GiB repeat. Both local calibration session 30657 /
PID 2709911 and session 25535 are terminal. Fresh admission selects only AR
counts 32/256, minimum speedups 1.104x/1.151x, but is not a deployment
certificate. Raw admission/deployment are archived with the graph checks.

## CPU control implementation update

The independent CPU control channel is implemented in private integration
`host_control.h`, selected explicitly by the common rank-0 IPv4 address
`UCCL_LITE_DROPIN_CONTROL_ROOT`. Its public-NCCL-ID session handshake never
parses private NCCL fields. It validates rank/session/framing/order, bounds
I/O and propagates transport errors as NCCL initialization errors. Exact
profile/host/PID/device/UUID voting is retained; stale/missing-profile ranks
participate. The network lifetime ID uses the same CPU channel at first use.
Without the setting, the previous NCCL metadata path is retained. The root
and protocol are fingerprinted and MPI launchers explicitly preserve/export
the setting. Local identities now also hash linked UCCL/MSCCL++ dependencies
when selecting the unified interposer. No MPI dependency was added to libraries.

CPU 4-/8-process tests and ASan+UBSan pass (changing payloads, three lifetimes,
late root, wrong session, disconnected peer). Both interposers build and keep
19 exports. Actual two-host ABI stress, local/network nccl-tests Graph replay,
local single-process four-GPU native fallback, single-host stale/missing
profile and stale NCCL_ALGO=Tree AR checks pass. An invalid control root returns
an initialization error before route preparation. Evidence before the later
stream-cache cleanup is archived in
`benchmarks/data/dropin_cpu_control_4l4_2x4l4_20261116/`.

A controlled local candidate screen has 7/88 strict failures, but every native
128/256 KiB row is within 1%–4.3% of controls, eliminating the repeated steady
penalty in that range. The network screen fails 9/66 rows: pure-native AG is
near controls, selected AR/RS remain faster, but AR fallback after network
activation still regresses substantially (128 KiB up to 160.88 us vs Native
90.65/93.22). RS64 KiB also fails around 6%. These are candidate screens, not
deployment certificates. Do not start full calibration until the remaining
steady failure is addressed.

Short Nsight traces starting at 16 KiB put custom preparation before the
first application native collective. They do NOT reproduce the original
128 B start/late-prepare order and cannot explain the full-curve failure.
Both output checks pass, but the UCCL diagnostic exits 139 afterward in the
MSCCL++ global temporary-stream cache destructor after nccl-tests calls
cudaDeviceReset. Idle cached streams are now released at the explicit network
prepare boundary using the public gpuStreamPool()->clear() API. The rebuilt
network library is b25ba65fc30d2c9fc11cc89693d88f32107ce9d1b96e4d3f090ae9d70ce55aad;
post-cleanup actual ABI stress passes in build/network_cpu_control_stream_abi.
The changed library/helper identity invalidates older profiles.

**Current implementation and live work (latest):** CPU control protocol is
now version 2. A real hashed-port collision occurred in
build/network_worker_diagnostic/1_keep.log and terminated that job at init.
Protocol 2 tries 64 session ports in 20000–29999 (below this host's ephemeral
32768–60999 range), rejects self-connects with a distinct server ACK, and
keeps bounded I/O. CPU tests now include an occupied listening first port;
normal and ASan+UBSan checks pass. Both .so files were rebuilt and staged.
The setting/profile fingerprint records protocol 2. Docs use the new range.

The full-order Nsight experiment (session 14531, terminal) starts at 128 B,
reproduces late preparation, passes output and exits normally after idle
stream-cache cleanup. Unlike the short trace, the *same* NCCL RING_LL kernel
with identical geometry takes longer on GPU: 128 KiB in-place median about
104.5 us versus native 69.6 us. Submission gaps are a smaller component.
Files/analysis: build/network_cpu_control_trace_full/. The direct candidate
identity override used to survive profiling is diagnostic only.

Preparation-only diagnostics keep all GPU resource initialization but run
every operation on NCCL. They still slow the large fallback curve, so actual
custom kernel execution is not required (build/network_prepare_only_diagnostic).
Early peer-permission priming fails to restore performance and was NOT
promoted (build/network_prime_peer_diagnostic). Worker hooks in
build/network_worker_diagnostic_v2 disable the MSCCL port proxy, receive
poller or both, only with the all-native preparation-only build. Each hook
ran on all eight ranks, outputs pass, but none consistently restores large
fallback. One keep-workers control is anomalously fast; preserve all rows.
All these diagnostic variants are unsafe for production and were not promoted.

The optional network library now registers/sends CUDA-IPC descriptors only to
local peers and IB-only descriptors across hosts. Previously cross-host memory
with both flags invited a foreign CUDA IPC import and benign-error fallback.
The retained benchmark defaults remain unchanged via the private constructor's
optional transportScopedMemory flag; the .so opts in. This removes the
handled invalid-resource-handle warnings. Current actual eight-rank ABI stress
passes in build/network_scoped_memory_abi (two routed lifetimes, third native
graph lifetime, placement/guards/streams/layouts). Current network library SHA:
eac3253b5ec73580ac65b2a7804577ec504aca3dca2ea228180aee7f8c3e6a6b.
Current unified interposer SHA:
f5075c4b337723e0d1d67eb6e8027247951fab2716a43a856e08af87d7fc603e.

**No live GPU work.** Sessions 14616 (scoped memory screen), 69081
(NUMA binding comparison) and 98241 (allocation-isolation comparison) are
terminal. All current ABI/transport builds and datasets are retained.
The scoped-registration screen still fails 9/66 strict 5% rows; it removes
foreign IPC errors but not the large-AR fallback problem. Correctness passes.

A verified GPU-local NUMA CPU+memory binding experiment applies the same
binding to Native and UCCL. It does not fix the penalty: bound 128 KiB AR
in-place Native 78.83/75.52 us, UCCL 115.18/112.50. GPU0/1 and mlx5_0 are
NUMA0, GPU2/3 and mlx5_1 NUMA1 on the actual topology. NUMA policy is not a
production fix. A build-only allocation hook rounds UCCL-network/MSCCL++
allocations to 2 MiB (160 hits/eight-rank job), also without improvement:
isolated 128 KiB in-place 121.67/120.66 versus Native 78.56/86.44. Neither
binding nor allocation-hook profiles are admission. Both use a deliberately
supplied candidate identity for controlled comparisons. Do not promote hooks.
Archive: benchmarks/data/dropin_cpu_control_4l4_2x4l4_20261116/.

**Next available action:** isolate initialization components with the
preparation-only all-native diagnostic (never run custom kernels with missing
transport pieces). Add build-only transport modes for: buffer/allocation and
CPU bootstrap only; IB registration only; local CUDA-IPC registration/import
without semaphores; then local semaphores and IB/GDR semaphore setup. Keep
capacity 131072, full 128 B start, 20/100 repetitions, both layouts, max rank,
all rank profile/placement votes, and compare fresh paired Native controls.
This distinguishes which allocation/registration stage changes an already
active native NCCL path. Actual custom kernel execution is not required,
worker-only and early-peer-only explanations failed, and buffer rounding
alone has not helped. Preserve all failed/negative rows and anomalies.

Production files now include CPU control protocol2, checked root-port retries,
exact dependency/control identities, early idle-stream-cache release and
transport-scoped memory descriptors. Current .so numerics/graphs are covered
by actual mixed ABI stress, but no current production profile is performance
certified. After the remaining cause is fixed, finish final post-v2
Graph/stale/missing/Tree/wrong-placement checks, retained CUDA builds,
19-symbol export checks, fresh local/network admission and independent mixed
validation over all alignments and large sizes. Do not replace that scope
with an easier no-acceleration or constant-size-only result.

## Implementation

- `build/nccl_lite/libuccl_lite_nccl.so`: local BF16 calibrated interposer.
- `build/nccl_lite/libuccl_lite_nccl_multinode.so`: same local paths plus the
  optional two-host/four-rank FP32 AR/AG/RS backend; one artifact can serve both
  placements using the corresponding local or network profile.
- Private network kernels retain the original registered-buffer variant and
  add caller-I/O fusion. Double banks protect reuse across size/family changes;
  the integration borrows the wrapper completion event to avoid duplicates.
- Native P2P metadata votes preserve collective fallback on stale/missing
  profiles and avoid `NCCL_ALGO` filters on metadata AllGather/Broadcast.
- First nonempty capture, including an unsupported shape before any UCCL use,
  permanently leaves that communicator native. Tests cover a saved graph
  replayed after later eager operations, mixed streams/in-place buffers,
  caller CUDA-device restoration and two accelerated communicator lifetimes.
- Calibration reserves an explicit capacity in candidate and admitted profiles,
  preserving the measured registration/bank layout after rejecting rows.
- `dropin.map` exports only the 19 NCCL wrappers, hiding private core/CUDA/STL
  definitions. `tests/nccl_dropin/symbols.py` enforces this surface.

## Evidence and limits

`build/network_dropin_capacity_admission/` completed 96 independent nccl-tests
jobs: two ABBA rounds at offsets 0/4/8/16 bytes, 20 warmups and 100 measured
calls, maximum-rank host enqueue/completion batch means, both layouts.
The pre-visibility-fix build admitted FP32 AR count 4096 and RS shard count
4096, minimum speedups 1.098x and 1.108x. Seven other shapes were rejected.

`build/network_dropin_deployed_align0/` then failed the strict independent
mixed-curve gate (42/66 combinations), including pure native fallback. That
profile is **not certified**. Nsight traces in
`build/network_fallback_diagnostic_v2/` show the same native NCCL GPU kernels
and similar typical kernel durations with no UCCL routing, motivating a loader/
symbol audit rather than changing mathematical kernels to explain fallback.

Restricting exports from 339 to 19 restored the controlled no-profile fallback
comparison to approximately 35 us, within 2.5% of its native controls in
`build/network_fallback_visibility/`. This is a bounded diagnostic, not yet a
full-profile or full-range performance certification. All failed/negative runs
are retained; older measurements do not validate the final linked binary.

The visibility-limited calibration in
`build/network_dropin_hidden_admission/` finished; its independent validation
failed, as recorded below. Do not modify binaries, launch helpers or the timing
driver while a calibration/validation process is live.

## Required remaining checks

1. Finish the current calibration and inspect every rejected/admitted shape.
2. Independently validate the resulting production profile with
   `tests/nccl_dropin/network.py --profile ... --abba` for offsets 0/1/2/4
   elements. Include native fallback rows in the 5% regression gate.
3. Run final routed ABI stress, graph replay, stale/missing profile cases and
   a stale-profile `NCCL_ALGO=Tree` AR-only case. Confirm route counters on every
   rank for eager admitted calls, and no router preparation on fallback cases.
4. Extend the deployment curve through large native fallback sizes; earlier
   128 KiB curves cannot establish behavior up to 1 GiB.
5. Refresh four-local-L4 calibration/checks using the unified binary and the
   actual nccl-tests timing scope; older Torch/CUDA-event calibration alone
   does not certify nccl-tests batch throughput. Local launch/calibration tools
   accept `--library` to select the unified binary.
6. Build/test affected retained benchmark variants and inspect the final symbol
   surface, exact software identities, correctness and performance reports.

Normal abort retains unfinished backend resources until process exit, avoiding
waits on missing peers. In-process network fault recovery is not established.

## Continuation update: application-communicator metadata regression

The visibility-limited calibration finished: AR/RS count 4096 were admitted
(minimum speedups 1.102x/1.098x), but independent deployment still failed 36/66
mixed rows in `build/network_hidden_deploy_align0/`. Do not deploy that profile.

Controlled native / no-profile / valid / invalid / launcher comparisons are in
`build/network_fallback_hidden_states/`; metadata-AllGather experiments are in
`build/network_metadata_comparison/`. Even invalid profiles affected later
native timings after inserting control traffic on the application comm.
`build/network_shadow_comparison/` instead uses a non-sharing temporary NCCL
child for the vote and restores the native fallback comparison near controls.

The implementation now uses `ControlComm` (`blocking=1`, `splitShare=0`) for
both identity voting and first-use network-ID exchange, leaving application
communicator GPU traffic untouched. `build/network_control_child_stress/`
passes the full 90-call/rank ABI stress, two routed lifetimes, saved unsupported
graph replay and current-CUDA-device restoration. Both interposers still export
only the 19 intended NCCL wrappers.

The calibration in `build/network_control_child_admission/` finished; its
independent mixed-curve failure is recorded below. Sessions 5882 and 39161
are terminal. Neither profile is certified for the current binary.

## Continuation update: caller placement leak

Session 39161 is terminal. Calibration admitted AR/RS 4096 (1.111x/1.099x),
but deployment `build/network_control_child_deploy0/` still failed 24/66 rows.
All network GPU jobs/diagnostics in this section finished. The later local
calibration is recorded below.

The serial-local backend no longer starts the unused local DMA proxy (the
retained benchmark transport default still does). More importantly, MSCCL++
Fifo construction changes caller affinity/NUMA policy via numaBind().
`CallerPlacement` in the network prepare boundary now restores both. Raw
90-call/rank tests assert exact placement before/after prepare; ABI tests
assert it around every call and pass (`build/network_placement_abi_assert/`).

Performance is still unresolved: `build/network_placement_subset/` has 36/66
strict 5% failures. Matched/differing-iteration event diagnostics are retained,
but the unsafe no-event diagnostic was NOT promoted. The current library
binary differs from the earlier admitted profile; do not reuse that certificate.
See benchmarks/data/dropin_placement_2x4l4_20261116/README.md. Continue from the
remaining mixed fallback performance issue and actual local nccl-tests batch
checks; do not launch another full admission sweep before a focused fix is
shown to address it.

## Local nccl-tests admission completed; deployment in progress

The unified binary completed actual four-L4 nccl-tests calibration in
`build/local4_nccl_batch_admission/` using `calibrate_local_nccl.py`.
The driver uses BF16/SUM, offsets 0/1/2/4/8 elements (0/2/4/8/16 bytes),
two independent ABBA rounds per offset, both layouts, 20 warmups and 100
measured calls, maximum-rank host enqueue/completion batch means. It retains
one fixed implementation and the original candidate registration capacity.
Admission requires the slowest UCCL job to beat the fastest NCCL control by
at least 5% for every alignment/layout. Rejected shapes are retained.

The initial aligned screen (`build/local4_nccl_batch_candidate/`) found gains
for AR counts 32/256; larger AR and all AG/RS candidates regressed and must
remain native. This is a screen, not independent deployment certification.
PID 2695598 is terminal. Calibration admits only AR counts 32/256, with
minimum speedups 1.113x/1.147x across all ten alignment/layout combinations.
Independent mixed-curve deployment through 1 GiB completed in
`build/local4_nccl_deploy_1g_align0/` with correctness intact but 9/160 strict
5% failures. Session 47433 is terminal. No local deployment certificate is
claimed. This record and admission predate the empty-collective fast path.

A build-only network diagnostic in `build/network_retire_diagnostic/` compares
Native/current/retired/retired/current/Native at identical 20 warmups and 100
measured calls, FP32 AR 16–128 KiB, both layouts, maximum rank. The retired
variant drains and releases the transport at the first unselected call and
permanently uses NCCL. nccl-tests warms *all sizes* before timing, so this
variant's timed curve is all native, including the nominal admitted size.
It is only a resource-isolation experiment, not an accelerated candidate or
deployment profile. Typical fallback times improve; the remaining variability
prevents attribution to a particular retained resource. No production code
was changed to retire routes. The network mixed-curve regression is open.

The same all-size warmup means graph-mode nccl-tests can prepare and use an
eager route before the first graph capture. Graph checks must allow those
warmup counters and prove that capture/remaining calls stay native, rather
than claiming no preparation ever occurred.


Independent local AR completed through 1 GiB with correctness intact but two
strict 5% failures among 56 alignment-0 layout/size rows (8 B out-of-place,
1.063x; 128 KiB out-of-place, 1.130x). AG/RS still run; neither the local
admission nor this partial curve is a deployment certificate.

Exec session 74539 waits for local validation PID 2704424 to finish, then
builds/runs `build/network_disabled_keep_diagnostic/`: Native/current/
disabled-keep/retired/retired/disabled-keep/current/Native. The disabled-keep
build permanently disables routing at the first unsupported call but keeps
all prepared resources. This separates wrapper steady-state completion
bookkeeping from retained-resource effects. These are build-only diagnostic
variants; no production routing policy changed. Do not start concurrent GPU
jobs while either session 47433 or 74539 is live.

The `*.map` ignore rule now has a precise exception for the interposer version
script. Both current interposers pass the 19-export symbol check. Measurement
directories also explicitly retain their JSON/log/CSV evidence rather than
silently dropping it under the repository's generic artifact ignore rules.


## Empty-collective fast path and graph-check correction

Sessions 47433/74539 are terminal. The independent local 1 GiB curve failed
9/160 strict rows: four empty AG/RS layout rows and five positive-count rows.
All raw evidence is archived under
`benchmarks/data/dropin_nccl_batch_4l4_20261116/`. The resource/bookkeeping
diagnostics are archived under
`benchmarks/data/dropin_completion_diagnostic_2x4l4_20261116/`; neither
retired nor disabled-keep was promoted.

NCCL's `taskAppend` discards count-zero collectives without GPU work, while
zero-byte Send/Recv still append matching tasks. The AR/AG/RS wrapper now
forwards count-zero calls straight to real NCCL argument validation. It does
not change route ownership, completion events or sequence counters. This
also allows an empty saved graph to precede later eligible eager calls.
The two-host real-ABI stress now captures all three zero-count collectives
before each routed lifetime and replays that empty graph afterward; it passes
in `build/network_empty_graph_abi/`, with the expected two routed lifetime
counter records per rank and the original nonempty unsupported graph test
still native. Both interposers rebuild and pass the 19-export symbol check.

nccl-tests warms all sizes eagerly and issues a synchronization call before
the first capture. Graph checkers now require exactly the applicable eager
calls, rather than incorrectly requiring zero route preparation. The first
nonempty capture still permanently selects native NCCL. Network nccl-tests
Graph validation passed in `build/network_empty_graph_nccltests/`
(session 46347 terminal). Local BF16 Graph validation also passed in
`build/local4_empty_graph_nccltests/` (session 5175 terminal). Fresh local
admission for the changed binary is the only live work, identified above.
No old profile certifies this new binary. The network mixed regression and
five positive-count local regressions remain unresolved.


The completed eager/Graph checks and retained-target builds are archived in
`benchmarks/data/dropin_empty_graph_4l4_2x4l4_20261116/`, including exact binary
and source hashes. Core checks: 26 dispatch, 164 peer and 93 staged four-GPU
cases pass. Router checks: 35 policy checks, CPU/CUDA dispatch and 1000 mixed
captured collectives with five replays/ring wrap pass. Both intended retained
network CUDA binaries build; the host and host-RDMA AG targets are up to date.
These are correctness/build checks, not renewed benchmark claims.

Fresh calibration's first completed offsets show empty AG/RS at 0.16–0.17 us
versus matching Native controls around 0.20–0.33 us; the former approximately
2x zero-length wrapper penalty is removed. Full admission is still live.
