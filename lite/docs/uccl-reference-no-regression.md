# Active goal: preserve UCCL Lite performance

The user's acceptance target is **zero regression against README UCCL Lite**
for AG, SUM AR and SUM RS over 1 KiB–1 GiB, including both NCCL layouts.
The candidate must remain a replacement shared library for unchanged nccl-tests
and must never load/execute native NCCL as a fallback. The prior 5%-versus-NCCL
validation does not meet this goal. No 5% allowance applies to this requirement.

## Authoritative references

- AG: `ag_combined_latest_2x4l4_20261117`, combined CE/SM preset, 256-KiB packets,
  four MPI processes/two GPUs each. Original binary hash verified unchanged.
- AR: `ar_small_fix_2x4l4_20261115`, source commit `39ce6ad9`, exact recorded
  source/header hashes recovered. Small GPU AR; GPU RS plus host-packet gather
  at larger sizes. Eight MPI ranks/one GPU each.
- RS: `rs_direct_medium_2x4l4_20261115`, source commit `833e4ac6`, exact recorded
  source/header hashes recovered. Original fixed tuned standalone size policy.

Frozen source builds, identity records, the detached reference controller and
fresh logs live under `build/uccl_readme_reference/`. The controller PID is
recorded in `controller_pid.json`; verify process liveness and `fresh/exit.json`
before resuming work. Never restart merely because a tool handle disappeared.

The [historical gap diagnosis](../benchmarks/data/uccl_reference_gap_20261118/README.md)
retains all 126 candidate rows and reference provenance. It currently finds
61 slower rows. Different process/layout/timing conventions make these diagnostic
ratios, not paired admission. Original measurements remain a performance target;
a slower substitute baseline must not be used to qualify the adapter.

## Implementation direction

The current strict library uses GPU-local ring/CE schedules and paired IB links.
It omits the README bulk AG's shared-host CE schedule and AR's host-packet gather.
Recovering those paths is the principal task, rather than tuning only against
native NCCL.

Reuse the recorded algorithms and MSCCL++ bootstrap/registration/IB components.
The eight-rank one-GPU-per-process ABI needs shared NUMA-local packet storage,
leader network exchange and reader acknowledgements corresponding to the
original two-GPU groups. Preserve original payload reduction (own shard stays
on GPU), packet-slot ownership and three-slot overlap.

CUDA copies and completion dependencies must be visible to CUDA scheduling.
An external worker that issues CUDA copies while the caller waits in a pageable
copy previously deadlocked. A port must not reintroduce that cycle or replace
stream semantics with a synchronous benchmark-only implementation. Network-only
host progress/callbacks must perform no CUDA API calls. Dynamic caller pointers,
in-place layouts, allocation recycling and two communicator lifetimes remain
required correctness cases.

## Required evidence before completion

1. Reproduce the exact original fixed UCCL implementations on the two four-L4
   hosts; retain configuration, hashes, hardware and every negative result.
2. Establish paired measurements with explicit, comparable timing and buffer
   reuse rules. Preserve the README fast implementation; do not make the baseline
   slower by adding avoidable adapter work or changing its algorithm.
3. Run unchanged nccl-tests over all 21 powers of two, AG/AR/RS, both layouts,
   with the strict library. Validate every rank/size and zero native fallback.
4. Check every row against UCCL with a zero-slowdown criterion. Characterize
   uncertainty with independent repetitions; do not silently loosen the threshold,
   discard slow points, or select a best-per-size envelope across runs.
5. Rebuild affected CUDA targets and repeat appropriate numerical, stream,
   aliasing, lifetime and queue/packet checks for changed paths. A standalone
   reference reproduction or a partial sweep cannot establish completion.

The goal remains active and unmet.

## Current verified progress

The detached reference controller completed all nine jobs successfully. Fresh
three-round 1-GiB medians are AG 41.922 ms, AR 96.445 ms and RS 53.693 ms.
Exact commands, raw checked samples and source identities are archived under
`benchmarks/data/uccl_reference_gap_20261118/fresh_reference/`.

`src/multinode/shared_host_packets.h` implements the first porting component:
two NUMA-bound POSIX shared regions/node, three packet slots, packed own/remote
rows, CUDA registration/mappings, shared control storage, and unlink-after-open
lifetime handling. It has no worker and is not selected by the current `.so`.
The Makefile builds `build/nccl_lite/shared_host_packets_check`. On two four-L4
hosts it passes 48 cases/rank over two lifetimes: all four local GPU producers,
CE upload/download, mapped GPU reads, three slots, full and short packets,
and shared CPU flag visibility. This tests storage/local access, not network
RDMA or collective performance. Artifacts are in the audit's `shared_packets/`.

Next: reuse MSCCL++ host-mode IB registration/connections to exchange leader
packets between both remote NUMA groups. Preserve receive credits and reader
ACKs before slot reuse. Then connect CUDA-visible uploads/downloads and progress
to the NCCL caller stream without CUDA calls from progress callbacks/workers.
The README AG/AR fast paths still need integration; the target remains unmet.

## Host transport implementation (next checkpoint)

`host_packet_transport.h` now uses MSCCL++ CPU IB endpoints to multicast each
NUMA leader's two packed rows to both remote NUMA groups. Registered shared
storage includes independent receive-credit and arrival counters, four local
reader ACKs, two remote-row reader ACKs, and an outgoing-retirement marker.
Reuse requires the preceding epoch's readers and outgoing writes to finish.
Transport operations after preparation use host atomics/verbs and no CUDA APIs.

Two-node checks pass 48 cases/rank over two lifetimes for (a) full eight-source
transport, mapped GPU/CE reads, tails, delayed consumer and rejecting premature
credit/reuse, and (b) eager/captured CUDA host callbacks followed immediately by
pageable D2H. Build with the Makefile targets
`build/nccl_lite/host_packet_transport_check` and
`build/nccl_lite/host_packet_callbacks_check`. Exact logs/hashes are in the
reference audit's `host_transport/`. These are correctness/progress checks;
neither component is selected by the production `.so` yet.

Next implementation constraint: the installed CUDA API explicitly prohibits
updating multidimensional graph memcpy operands. Do not ignore nccl-tests'
rotating pointers or recycle a stale pointer-keyed graph. Preserve direct,
dynamic 2D H2D destinations. A feasible integration is bounded per-slot CUDA
streams with upload/consume flag kernels and explicit caller-stream joins,
while a dedicated network-only CPU progress actor advances all three slots.
The actor must not issue CUDA calls. Each slot permits at most one waiting GPU
stage so future waits cannot exhaust SM resources. Stream waits/signals and
packet credits must make reuse safe across calls and operation transitions.
Avoid blocking per-packet host callbacks that can occupy the sole callback
executor while waiting for a later ACK callback in the same process.

Required next evidence: complete asynchronous three-slot AG over actual caller
buffers, in-place/rotating/recycled allocations and delayed readers, then full
paired comparison against the reproduced UCCL reference. Hybrid AR must reuse
that fast gather path. The zero-regression goal remains active and unmet.

## Asynchronous AG prototype checkpoint

`async_host_gather.cuh` now implements three CUDA lanes joined to the caller
stream and a leader-only CPU progress thread. CUDA kernels publish/poll mapped
slot flags; the worker uses only host atomics and MSCCL++ RDMA. Direct 2D copies
accept each invocation's actual caller pointers. There is no pointer-keyed graph
cache and no synchronous substitute for enqueue. The experimental integration
is guarded by `UCCL_LITE_HOST_GATHER_EXPERIMENT` in the network backend, routes
AG at/above 8 MiB, preserves outstanding GPU-transport retirement state, and
synchronizes/destroys the host runtime before its resources. Default builds do
not select it. Optional `UCCL_LITE_HOST_GATHER_CHUNK` controls prepared capacity.

Correctness: the initial prototype passed 36 calls/rank over two lifetimes;
the expanded 1-MiB-capacity version passed 42, including a 128-MiB+128-byte
output tail, in-place/misaligned pointers, alternating streams, delayed rank,
pageable D2H and allocation recycling. Each tested experimental `.so` passed
the 126-call/rank mixed NCCL ABI at 1-GiB capacity, with native library absent
and zero fallback. Standalone transport/callback checks remain archived.

Three-run nccl-tests AG diagnostics (20 warmups/30 iterations, 1 KiB–1 GiB)
compare against the fresh reference only diagnostically: harness/process and
buffer-cycle differences are still explicit, so these are not paired admission.
- Initial late-consume/256-KiB path: 1-GiB out/in 45.072/44.138 ms versus
  41.922 ms reference; 8-MiB out 482.15 us versus 355.37 us reference.
- Early local consumption: 8-MiB out 447.10 us, but 1-GiB out 46.821 ms.
- Current adaptive experiment: early consumption through 32-MiB full output,
  late above; 256-KiB pieces through 32 MiB and 1-MiB pieces above. 1-GiB
  out/in 45.069/45.331 ms; 8-MiB out 448.91 us. It remains slower and unadmitted.

Artifacts are archived under the reference audit's `async_host_ag/`. Active
builds are `build/uccl_host_ag_candidate`, `..._early`, and `..._adaptive`;
all their jobs are terminal. Source/runtime is correct on these checks but
performance does not satisfy the goal. Next: profile per-packet GPU signaling,
copy scheduling and CPU progress placement; compare with the recorded pipeline.
Hybrid AR has not yet been connected to the new gather. Keep the original UCCL
baseline and zero-slowdown target intact.

## Hybrid integration checkpoint

`UCCL_LITE_HOST_HYBRID_EXPERIMENT` (with the host-gather experiment macro) now
uses the existing GPU ReduceScatter into a prepared private shard, followed by
the asynchronous host gather on the caller stream, for AR at/above 8 MiB.
It preserves caller input lifetime for in-place AR and synchronizes the host
runtime before freeing its private shard. This remains a separate experimental
build; there is no native fallback and no default admission.

`build/uccl_host_hybrid_candidate/abi` passed all 126 guarded calls/rank over
two lifetimes at 1-GiB capacity, including mixed operations, alignment, stream
changes and pointer recycling. Three unchanged nccl-tests AR sweeps (20 warmups,
30 iterations, full 1-KiB–1-GiB allocation/coverage) passed numerical/strict route
checks. 1-GiB out/in medians are 100.957/100.857 ms versus the freshly reproduced
UCCL reference's 96.445 ms. 8-MiB out/in are 948.29/936.96 us versus 890.40 us.
These are diagnostics across differing harness conventions, not paired admission.
They improve on the old strict GPU-ring/graph adapter but do not meet the user
requirement. Artifacts are in the audit's `host_hybrid/`.

All recorded jobs are terminal. Current implementation work should profile
host packet control/CPU placement and copy scheduling before further tuning.
The original baseline pins its routing CPU and retains shared/self packet IO;
that behavior and the 0% regression criterion must remain explicit. Do not
reinterpret the earlier native-NCCL 5% result as success. AG still needs roughly
7.5% improvement at 1 GiB and more at medium extents; hybrid AR remains roughly
4.7% slower at 1 GiB. RS still needs its own zero-regression paired audit.


## Host-control profiling and rejected variants

[Recorded experiments](../benchmarks/data/uccl_reference_gap_20261118/host_control_experiments/README.md)
preserve the full positive and negative curves. CUDA stream writes instead of
publication kernels were substantially slower; CPU pinning did not help.
Fusing publication with the next within-call readiness/reuse wait passed the
extended ABI and reduced 1-GiB AR to 100.278/100.316 ms and AG to
44.688/44.492 ms (out/in), still above the original UCCL targets. Deferring
network flush until reader completion also passed ABI but did not improve AG
(44.845/44.644 ms); no AR sweep was spent on that rejected AG change.

Next hypotheses isolate two measured costs: batch independent ready H2D
copies in one CUDA submission, and read independent host flags in parallel
warp lanes instead of serial PCIe loads. Each remains separately gated and
requires correctness before measurement. No experimental path is admitted as
the zero-regression replacement; RS and matched full-curve admission remain.


## Cooperative medium path and fresh RS diagnostic

Independent fixed-policy screens rejected batched H2D submissions, CPU-ready
aggregation and six-slot bulk pipelining. Warp flag reads help medium CE; two
slots help 8-MiB and in-place bulk, but no CE policy yet beats the UCCL reference
throughout the curve. All negative data are preserved in the host-control record.

`UCCL_LITE_HOST_COOPERATIVE_SM` adds one resident cooperative grid for medium
host-packet AG. It performs input upload, ordered local/remote waits and SM
multicast in the caller stream. Grid-wide completion and system fences protect
host publications. Prepared occupancy bounds guarantee grid residency.
`UCCL_LITE_HOST_COOPERATIVE_ONE_PACKET` prepares 512-KiB capacity and packs each
medium source shard into one packet, leaving bulk CE piece size unchanged.
Both versions pass extended ABI and the expanded 54-call/rank direct check,
including the two-packet/single-packet 4-MiB-minus-128-byte tail, alignment,
per-rank layout differences, caller-stream changes and allocation recycling.

The one-packet fixed build measures 512-KiB/1-MiB/2-MiB/4-MiB AG out-of-place
at 42.10/62.97/105.06/188.61 us, versus fresh UCCL 54.24/73.87/114.32/197.17 us.
Its in-place medians also improve diagnostically. These are cross-harness
results, not the required matched admission. 8-MiB and bulk AG still regress.

Fresh RS measurement (unchanged algorithm) finds 12/42 slower rows, including
1-GiB out/in 54.032/53.870 ms vs 53.693 ms and 8-MiB 508.37/506.65 us vs
494.40 us. The next RS experiment doubles its graph target piece from 8 to
16 MiB, reducing large-message graph stages; correctness and full curves are
required before retention. The default strict build remains unchanged in
selection. No native NCCL fallback is permitted or used.


RS piece-size screens are now terminal: 4- and 16-MiB targets both regress at
1 GiB, so retain the original graph target of 8 MiB. The existing streamed RS
kernel improves 8 MiB to about 490 us, but extending it through 16 MiB regresses
there. A combined experiment is being built with graph threshold 16 MiB, so
only 8-MiB RS changes. It also extends cooperative AG down to 256 KiB and
keeps the validated one-packet policy through 4 MiB. The pending full comparison
must use that single binary, not a combination of old experiment minima.


## Single-library full curve and composition regression

The complete [combined candidate](../benchmarks/data/uccl_reference_gap_20261118/cooperative_rs8_candidate/README.md)
(SHA 723d9941c04dba57fd78b3ec5369f149cefd1733507419bcc02c633c74d37cc3)
passes extended ABI and nine unchanged nccl-tests jobs, all operations, all
21 extents and both layouts, with three sweeps each. All candidate processes
report native NCCL absent and zero fallback. Comparison against fresh original
UCCL remains diagnostic: 51/126 slower rows (AG 16, AR 22, RS 13). The full
curve and negatives are preserved; the target remains unmet.

This whole-library measurement exposes a composition regression: 8-MiB public
RS improves to 489.63/489.80 us, while hybrid 8-MiB AR worsens to
951.89/946.27 us. The earlier graph-based shard producer was faster in this
composition. `build/uccl_hybrid_graph_rs8` therefore restores graph RS for the
internal host-AllReduce producer, with an explicit `hostAllReducePart` context
argument, while retaining streamed public 8-MiB RS. Its ABI and AR curve must
validate that decision; do not infer them from the separate RS win.

Remaining high-value investigations: the AR 512-KiB path is the largest AR
relative gap (about 1.28x UCCL); the existing direct-input medium AR path may
be worth testing at that boundary. Bulk AG still pays per-packet CUDA operation
and PCIe flag costs. A future cooperative producer/consumer pipeline must keep
all actors resident, preserve both reader ACKs and outgoing retirement, and
avoid CUDA calls from the CPU progress worker. The currently implemented
cooperative kernel covers only the medium one/two-packet case. Matched timing
and full zero-regression admission remain required after the data paths improve.


The [hybrid producer follow-up](../benchmarks/data/uccl_reference_gap_20261118/hybrid_graph_rs8/README.md)
is complete: SHA 468d1a723c004428abd67f3189f861999f5b2d3a4354497e659dfb7e2cc92c47; extended mixed ABI and all three AR sweeps pass.
8-MiB AR returns to 922.15/915.80 us, improving the combined candidate's
951.89/946.27 us but still exceeding 890.40 us UCCL. 1-GiB remains
100.689/100.678 ms versus 96.445 ms. AG/RS timings were not repeated for this
follow-up, so it has no new full 126-row performance admission. All recorded
controllers/jobs are terminal. Current source matches this follow-up snapshot;
the default build still does not select these host-gather experiments.

Next implementation priority: test the existing direct-input AR path at exactly
512 KiB, retaining its single-bank reuse/drain protocol initially. The current
medium-bank predicate intentionally excludes AR below 8 MiB; extending the
size branch must not silently change that policy. Add an offset/in-place ABI
case for the newly selected 512-KiB branch before measuring it. Preserve all
existing 1-KiB–1-GiB coverage and zero native fallback.


## Direct AR boundary and persistent AG checkpoint

[Experiment record](../benchmarks/data/uccl_reference_gap_20261118/ar512_persistent/README.md).
The 512-KiB direct-input AR branch retains its original single-bank reuse/drain
protocol and requires preparation of large scratch. New optional ABI boundary
coverage adds 512 KiB without removing any prior case: 144 calls/rank over two
lifetimes. This candidate passes and measures 110.79/109.86 us versus UCCL
115.06 us. Extending it to 256 KiB is slower and is rejected; retain 512 KiB only.

The persistent host AG experiment uses one cooperative grid, with bounded
slot reuse, all actors resident, mapped host uploads, existing MSCCL++ CPU IB
transport, reader ACKs and outgoing retirement. No worker issues CUDA calls.
The first 4-uploader/28-consumer prototype passes correctness but is much slower.
FlashInfer per-block timelines show phase skew and repeated late consumers;
64 vector tiles are unevenly divided across 28 blocks. A device-scope consumer
fence alone does not fix performance. A separate publication block improves it,
but remains too slow. All failed results are preserved.

The fixed 4-uploader/16-consumer/1-publisher variant passes 54 component calls,
144 mixed ABI calls/rank, and all nine unchanged full nccl-tests jobs. Its one
binary measures 1-GiB AG 38.149/38.235 ms and AR 92.180/92.247 ms, faster than
fresh UCCL 41.922/96.445 ms. However, the full diagnostic still has 22/126 slower
rows: AG 5 (8/16 MiB and a tiny 32-MiB out-of-place excess), AR 6 (256 KiB,
4 MiB and 8 MiB), RS 11. No positive ratio is rounded into a pass.

The 64-KiB packet experiment through 32-MiB complete output passes correctness
but substantially regresses medium AG; reject it and retain 256-KiB packets.
All recorded jobs are terminal. The best complete experimental artifact is
build/uccl_persistent_consumers16/nccl_lite/libuccl_lite_nccl_strict.so,
SHA 093552a3b930858b5294935a89eeb1700747c62c405ce7ff52e213c171331e33.
Neither the default build nor a zero-regression admission has been promoted.

Next: investigate medium AG startup without adding more packet handshakes.
One bounded hypothesis is extending the existing whole-grid cooperative path
to 8–16 MiB using at most two larger source packets (prepared capacity currently
must remain <=1 MiB); compare local-first scheduling across those packets.
Another is increasing uploader parallelism while retaining 16 consumers.
Do not repeat the rejected small-packet screen. RS gaps and matched timing/
buffer-reuse admission still require work. The original full scope and zero
native fallback requirement remain unchanged.


## Medium follow-ups completed

[Recorded full curves](../benchmarks/data/uccl_reference_gap_20261118/medium_followups.md).
Extending the cooperative path through 16 MiB (one/two 1-MiB source packets)
passes 60 component and 144 mixed ABI calls/rank. Three AG sweeps give 8-MiB
357.13/356.48 us, close to but above UCCL 355.37 us; 16-MiB remains
730.66/731.19 us versus 672.00 us. Increasing the grid to 64 blocks and changing
to local-first consumption across the two packets do not solve the 16-MiB gap.
A combined warp load preserves all readiness and abort checks but loses bulk
performance; it is not retained globally. The old packet-major loop remains
literal under the default experimental policy, avoiding incidental changes to
previously measured configurations.

The separate RS medium-bank extension to 16 MiB passes mixed ABI and all three
full RS sweeps. Its 16-MiB median is 908.51/910.18 us versus UCCL 913.60 us.
It preserves two-bank reuse protection and only expands RS's bank-selection
range. Other RS sizes, including 1 GiB, still require work. This positive
RS-only screen is not a new full 126-row admission.

All recorded jobs are terminal. Best complete matrix remains
build/uccl_persistent_consumers16/full_diagnostic.json (22 slower diagnostic
rows); no default release has been promoted. Next bounded hypotheses: increase
persistent upload parallelism without changing the 16 consumers; eliminate
redundant same-stream waits in AsyncHostGather while preserving exact
cross-stream completion boundaries; consider extending RS alternating banks
to 32 MiB with the existing mixed ABI coverage. Matched timing and the original
zero-regression gate remain required; do not infer completion from near ties.


## Fused combined checkpoint: seven diagnostic gaps remain

[Complete fixed-library record](../benchmarks/data/uccl_reference_gap_20261118/ar512_persistent/uccl_fused_combined/README.md).
SHA 5063907450acd9fa2b1611213708cb6680806cba6619e7c4e400e96dd3381479. The same binary passes 60 component calls/rank, 144 mixed ABI
calls/rank over two lifetimes, and nine unchanged nccl-tests jobs (three full
sweeps of each operation, 1 KiB–1 GiB, both layouts). Native NCCL is absent and
all per-size/per-rank route and numerical checks pass.

The fixed policy combines cooperative AG through 8 MiB, the 4-upload/16-consume/
1-publisher persistent path above, RS alternating banks through 32 MiB, a fused
final RS addition/output, and hybrid AR from 1 MiB. The fused RS stage retains
all forwarding and remote-send storage. It waits for the remote sum before
reading the final own contribution and writes each output element once;
(own + local partial) + remote arithmetic order and in-place lifetime are
preserved. Future pieces are disjoint. The 512-KiB direct-input AR path remains.

The complete cross-harness diagnostic now has 7/126 slower rows: AG 5, AR 2,
RS 0. No positive ratio is rounded away. All 42 RS medians are no higher than
the fresh original reference, but this is not matched admission. At 1 GiB,
AG is 38.114/38.139 ms, AR 91.643/91.608 ms and RS 53.491/53.374 ms (out/in),
versus reference 41.922/96.445/53.693 ms. The user goal remains active and unmet.

Failed hypotheses and intermediate matrices are preserved in
[synchronization/RS follow-ups](../benchmarks/data/uccl_reference_gap_20261118/sync_rs_followups.md).
Eight uploaders, stream-ID-based redundant-wait removal and omitting the final
post-ACK grid barrier did not solve the target. Global combined flag loads,
64-KiB medium packets, and cooperative expansion to 16 MiB remain rejected.
Do not reintroduce those as an unmeasured improvement. The optional experiments
remain guarded; the default build has not been promoted as a passing release.

All recorded jobs are terminal; current source hashes match this build's
source checkpoint under build/uccl_fused_combined/source/. Next: inspect the
remaining AG 8/16/32-MiB and AR 256-KiB rows in the saved CSVs, retain the current
RS gains, and establish matched timing/pointer-reuse controls before any final
zero-regression claim. No new native-relative tolerance or narrower scope applies.


## AR boundary solved diagnostically; six AG rows remain

[Fixed-binary matrix](../benchmarks/data/uccl_reference_gap_20261118/ar512_persistent/uccl_ar256_threads384/README.md).
SHA a077ca8d461269b7836df3896ff45b7db4cafe389c129a210701228b80f70ec8. The 384-thread setting applies only to exactly 256-KiB AR; it
keeps the existing algorithm, banks, retirement and synchronization rules.
144 mixed ABI calls/rank over two lifetimes and all nine unchanged full
nccl-tests sweeps pass. AR 256-KiB out/in are 69.14/65.93 us versus UCCL
74.55 us. Both AR and RS now have zero slower rows in this binary's diagnostic.

Six AG rows remain slower (8, 16 and 32 MiB, both layouts). Do not combine old
AG timings with the new AR/RS curve: the new full AG measurements are retained,
including their run-to-run variation. This is not matched admission and the
full zero-regression goal remains unmet. No native NCCL is loaded or executed.

[Launch/parameter screens](../benchmarks/data/uccl_reference_gap_20261118/launch_parameter_followups.md)
record the negative 1024-thread case, the mixed 256-thread result, rejected
128-KiB medium packets, eight medium consumers and parallel cooperative ACKs.
A grid-constant/const-reference experiment removes all 141 static local-memory
instructions and the persistent kernel's 160-byte stack frame, but does not
improve end-to-end performance; do not promote it on compilation metrics.
All recorded jobs are terminal. Best complete experimental artifact is
build/uccl_ar256_threads384/nccl_lite/libuccl_lite_nccl_strict.so.

Next: profile the current 16-MiB persistent path, including publisher and
producer/consumer readiness, rather than reusing the old 28-consumer timeline.
Retain current AR/RS gains. Matched reference timing must account for the original
AG send-retirement boundary and per-call buffer preparation; GPU-only completion
must not be silently compared against a stronger reference completion scope.
The existing nccl-tests sweep and zero-slowdown criterion remain required.


## Current 16-MiB timeline and warp publisher

[Timeline record](../benchmarks/data/uccl_reference_gap_20261118/timeline16/README.md).
A fresh instrumented build records only 16 MiB while running the complete
1-KiB–1-GiB nccl-tests sweep, retaining its allocation and pointer-rotation
pattern. It passes mixed ABI and all sweep checks. Publisher phases are now
recorded separately. Upload publication averages 6–9 us/packet and the three
reader-ACK stores 12–14 us; these are instrumented spans, not scored timings.

`UCCL_LITE_HOST_WARP_PUBLISHER` keeps the same one control block and replicates
its state across one warp. Only lane 0 reads completion counters; warp broadcasts
keep decisions uniform. Counter reset precedes a warp synchronization, followed
by independent system-release ACK stores from three lanes. Another warp sync
precedes advancing the slot. Every reader count and outgoing-retirement condition
remains intact. No CPU progress worker issues CUDA calls.

The [new full fixed-binary matrix](../benchmarks/data/uccl_reference_gap_20261118/ar512_persistent/uccl_warp_publisher/README.md)
uses SHA ac4c463e187f9b0fae6f0d818ee07453259eadb3bcc683d98af880f9907dc708. Its 60-call component, 144-call mixed ABI over two lifetimes,
and all nine unchanged full nccl-tests jobs pass. The diagnostic has four slower
rows, all AG at 8 and 16 MiB (both layouts); AR and RS have zero slower rows.
32-MiB AG is 1307.64/1306.73 us versus 1308.87 us reference, a small margin
that is not treated as matched admission. 16-MiB remains 715.85/713.42 us versus
672.00 us. The original zero-regression goal is still active and unmet.

All recorded jobs are terminal. Current source hashes match this build's
checkpoint under build/uccl_warp_publisher/source/. Default production selection
has not been promoted. Next work must resolve the 8/16-MiB AG gaps and establish
matched reference timing, preparation and retirement scope; do not reduce scope
or substitute native NCCL. The new timeline is the current evidence, replacing
the old 28-consumer profile for diagnosis of this implementation.


## GPU gates: full diagnostic passes; stronger comparison underway

The GPU-gate implementation retains the original host readiness conditions,
reader ACKs and send retirement. A publisher warp uses system-acquire loads,
then warp synchronization and device-release stores to publish per-slot GPU
tokens. Actor blocks acquire those tokens before their block barriers. Slot
reuse, local data and remote data have distinct epoch tokens. The CPU worker
still performs only host atomics/verbs. All actors remain resident.

[Fixed-binary result](../benchmarks/data/uccl_reference_gap_20261118/ar512_persistent/uccl_gpu_gates8m/README.md).
Lowering the cooperative cutoff to 4 MiB lets 8 MiB use this new persistent
path without changing prepared capacity. Its 60-call component, 144-call mixed
ABI and all nine unchanged full nccl-tests jobs pass. All 126 diagnostic
medians are <= the fresh original UCCL reference, with zero native NCCL loaded
or fallback. This does NOT establish final admission: original preparation,
per-call timing and internal retirement must still be compared fairly.

The new tests/nccl_mscclpp/reference_timing.cu client reproduces per-call data
preparation, fixed pointers, max-rank timing and full-output verification.
uccl_lite_nccl_synchronize is a non-finalizing diagnostic extension that drains
GPU and internal transport work. Ordinary nccl-tests does not use it. The client
passes six smoke jobs (all operations/layouts through 1 MiB); its first staging
attempt lacked executable permissions, and the corrected copy2 retry is saved.

Fresh original reference binaries are SHA-verified. A 27-job, three-round
comparison is running under build/uccl_reference_protocol/paired_retry/ with
original algorithm process layouts retained. Candidate data preparation and
warmup/iteration counts match each reference operation; both candidate layouts
are compared against the original recorded reference layout. The first paired
attempt used reference-only environment variables for candidates; strict identity
checks rejected it. The retry separates environments, without bypassing checks.
Early aligned AG results expose slower small-message per-call latency when full
retirement is included. These failures must not be hidden by the passing batch
throughput diagnostic. The goal remains active and unmet.

An inline-packet-retirement candidate has been built but not measured yet:
build/uccl_inline_retirement/. It reuses the existing packetDrain path to avoid
a host round trip and a second drain-kernel launch in per-call synchronization.
Do not infer a win from that build. Wait for the active paired job before new GPU
work, then validate any candidate with both timing protocols and unchanged scope.


## Per-call comparison completed; not admitted

[Aligned protocol record](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/README.md).
All 27 jobs complete. Full output and candidate native-absence/route checks pass,
but 54/126 latency rows are slower: AG 21, AR 11, RS 22. Retain the passing batch
throughput result as a distinct measurement, not proof of this stronger scope.
Original algorithms, layouts and process counts were preserved; candidate pointers,
preparation and full-retirement completion were aligned. Reference original layouts
remain the target for both candidate layouts, and process-layout differences are
explicit. Final quiet paired qualification remains necessary.

Part of the difference is diagnostic I/O inside NetworkCollectives::synchronize.
The quiet variant removes only its drain printf and passes ABI. The inline-packet
retirement variant passes ABI and all nine unchanged full nccl-tests sweeps with
zero batch diagnostic regressions. An independent 18-job small-AG per-call screen
with 1-GiB allocations shows 1-KiB out/in improving from 33.212/31.873 us to
21.052/20.656 us, but 32–128 KiB still miss the reference. Both the logger cost
and the host round trip/second drain kernel matter; no failed data were discarded.

All recorded controllers are terminal. Next: overlap a correctly ordered MSCCL++
TriggerSync request with local work and wait on its FIFO-position completion ticket,
or enqueue the existing drain on a CUDA stream dependency before the host waits.
Keep all retirement guarantees, unchanged nccl-tests coverage, and zero fallback.
The current best batch result is not a final release or completed goal.


## Ordered retirement and early-ticket checkpoint

[Screen record](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/ordered_early_screen/README.md).
Both new builds pass 144 mixed ABI calls/rank. Ordered synchronization queues
its drain behind the exact completed-collective event before host waiting; it
avoids dependencies on later application work. Early AG retirement posts the
existing MSCCL++ TriggerSync after the final IB put and waits on that exact FIFO
position after local work. No completion or reader condition was removed.

The 18-job small-AG screen retains fixed 1-GiB allocations and full retirement.
Early tickets give 1-KiB out/in 19.340/19.160 us, 32-KiB 26.321/25.762 us,
64-KiB 29.643/29.278 us and 128-KiB 36.594/36.312 us. Ordered synchronization
is slower than inline retirement and is not retained. The early-ticket build
also passes all nine unchanged full nccl-tests jobs with zero batch diagnostic
regressions. Per-call 64–128-KiB gaps and other full aligned cases remain.

All recorded jobs are terminal. Useful next evidence: profile the remaining
small AG fan-out/output path and validate retirement overlap for other packet
collectives only after identifying their final IB write. Do not treat the
focused AG screen or passing batch gate as full paired qualification. Preserve
the 54-row original aligned failure record and repeat final paired tests quietly
once a complete candidate is ready. Scope and zero native fallback are unchanged.


## Small AG hypothesis screen and full follow-up

[Fan-out/geometry negative results](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/ag_fanout_screen/README.md)
and [extended local packets negative results](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/ag_local128_screen/README.md)
reject three alternatives. All pass mixed ABI/full-output/route/native-absence checks,
but none improves 64–128 KiB. Preserve these results and leave their flags unselected.

[Output-copy overlap screen](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/output_ordered_screen/README.md)
shows 128-KiB out/in 36.091/36.312 us versus contemporaneous early-ticket control
36.692/36.787 us. This modest difference does not close the original reference gap.
The candidate also uses ordered synchronization for outstanding nonpacket sends;
that path is not exercised by this small-AG screen. The exact final IB ticket still
must retire before kernel completion. No original benchmark, tolerance, route guard,
or native-fallback prohibition changed.

A full fixed-library campaign is active under build/uccl_output_ordered/qualify.py:
nine unchanged nccl-tests jobs followed by 27 full-range original-reference paired
jobs. Consult qualification_state.json and the process/session before starting any
other GPU work. Source snapshots and build command are saved under that build.
No release promotion or final zero-regression admission is justified yet.

The output-overlap/ordered candidate subsequently passed all nine unchanged
full nccl-tests sweeps. All 126 batch medians are no slower than the original
UCCL reference (worst ratios AG 0.95523, AR 0.96651, RS 0.99689). Full-output,
exact route and zero-native-fallback checks pass. The 27-job paired protocol
has started; its eventual result is required before any broader claim. Active
controller PID 3505379 / exec session 68718 is recorded in the checkpoint.


## Serial AG diagnostic queued behind the full paired run

The first paired round is provisional: AG 128 KiB remains about 10–12% slower,
256 KiB about 5%; AR out-of-place 16–64 KiB now passes, but 256/512 KiB remain
about 3%/1% slower. Do not replace the required three-round result with these
partial numbers. The same qualification process remains live.

Diagnostic-only FlashInfer probes were added behind UCCL_LITE_PACKET_TIMELINE
for serial AG input staging, packet send/flush submission, own fan-out, receive,
remote fan-out, local readiness/barrier, output copy and final flush wait. Their
buffer is per context and records up to 128 calls without a per-call reset or
host transfer. Profiling uses the existing fixed 1-GiB allocation/full-retirement
client, 20 warmups and 30 calls, 64/128 KiB and both layouts, and checks all output,
route, native-absence and trace records. Internal helper barriers are included
in their phase durations. Instrumented timings are diagnostic only.

build/serial_timeline_after_pair.py waits for qualification to finish successfully
before building or running any diagnostic GPU work. Source hashes are frozen for
that future build. The current unprofiled qualification library and its source
snapshot remain unchanged. The new probes have not yet been compiled or run;
consult the queued controller state and exec session 96148 before continuing.


## Static operation specialization prepared; not yet measured

[Resource audit](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/serial_resource_audit/README.md)
finds stack/local-memory instructions in the generic small-collective kernel,
while the frozen original RS specialized kernel has no stack. Added guarded
UCCL_LITE_STATIC_COLLECTIVE instantiations with compile-time operation selection;
the same collective bodies, arguments, synchronization and packet/bank rules are
used. Without the flag the generic operation path remains. No speedup is assumed.

The queued diagnostic build will keep generic selection and its FlashInfer probes;
the expected future source hashes were explicitly refreshed after adding the
unselected specialization. The currently running qualification .so is unchanged.
After qualification and all four diagnostic profiles finish successfully,
build/static_collective_after_profile.py will build a noninstrumented specialized
copy, run the 144-call mixed ABI check, then compare against the frozen generic
candidate in 36 jobs (three operations, both layouts, three repetitions, 32–512 KiB,
fixed 1-GiB allocations, full retirement; 20 warmups and AG30/AR10/RS10 iterations).
This screen is not final full-range admission. Queue/session information is saved
in build/uccl_reference_checkpoint.json. No native fallback is allowed.


## Output-overlap full paired result: 25 remaining slower rows

[Full paired report](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/output_ordered_full/README.md).
All 27 jobs complete with full-output/route/native-absence checks passing. The
aligned per-call result has 25/126 slower rows: AG8, AR6, RS11, down from the earlier
54-row failure but still not zero. AR64 KiB remains slower after three rounds,
despite the provisional first round. Tiny positive ratios (AG64 KiB in-place,
AR128 KiB in-place, RS256 MiB out-of-place) remain failures without rounding away.
RS512 KiB is the largest relative gap at16.82%/19.19%; AG128 KiB is10.04%/9.67%.
Nine unchanged nccl-tests sweeps still pass126/126 batch diagnostics. Both protocols
remain required and no default release is promoted.

Qualification controller/session68718 is now terminal. The dependent diagnostic
controller/session96148 has started building; static-specialization/session27217
still waits for diagnostic success. Their states remain authoritative for live work.


## Serial diagnosis and specialization comparison completed

[Phase diagnosis](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/serial_timeline/README.md)
passes mixed ABI and all four full-output/route/native-absence/trace checks.
128-KiB AG phase medians identify local ready/barrier (roughly7–9 us), packet
send/flush submission (about6 us), receive (about5 us), and output copy (about3.2 us).
Fan-out is about1 us per local/remote phase; final flush waiting is under about1 us.
These are instrumented diagnostic times, with visible timer quantization on ranks0–3,
not scored gains or proof of a performance improvement.

[Specialization screen](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/static_collective_screen/README.md)
passes144 mixed ABI calls/rank and all36 screen jobs. Removing RS stack in resource
metadata does not solve RS512 KiB. AG improvement is inconsistent. AR32/64 KiB
improves both layouts, while larger AR shapes do not; the whole static variant
is not selected. Exact commands, repeated timings, negative results, source snapshots
and raw trace hashes are preserved.

All recorded controllers are terminal. The active zero-regression goal remains unmet
at25/126 slower full paired rows. Next useful isolated experiments: AG per-bank
single-writer ready flags with system release/acquire and dedicated zeroed storage
(preserve wrap reset and reader/bank lifetimes), or eliminate dynamic channel-array
stack indexing without changing semaphores; extend streamed RS to exactly512 KiB
with its allocated alternating banks; narrow static AR code generation only where
validated. None is admitted before mixed correctness and both timing protocols.


## Independent readiness and RS-boundary screens

[Screen report](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/ready_rs_screen/README.md).
Both independent candidates pass the strengthened 198-call/rank ABI, including
32/64/128 KiB and a delayed AG rank, without removing any prior case. A separate
near-wrap AG build passes and records the expected 16 resets across eight ranks
and two lifetimes. All 24 focused screen jobs pass full-output/route/native checks.
AG dedicated ready flags improve some 32/64-KiB numbers but do not close 128 KiB;
the next base does not select this experiment. RS512 KiB streamed improves out/in
from 79.921/79.837 us to 58.840/58.469 us against the contemporaneous fixed control.
The original 25-row full-paired failure count remains until a complete new candidate
is qualified. No per-size measurement envelope or release promotion is implied.

Next isolated packet experiments reuse MSCCL++ publication and flush components:
combine TriggerData and TriggerSync into one FIFO request for final AG sends,
keeping all existing fences; separately rely on the CTA producer barrier followed
by MSCCL++ FIFO system-release publication and CPU acquire before posting RDMA,
avoiding an additional per-lane system fence. These are separate guarded builds,
with the exact outgoing flush ticket still awaited before completion. Relevant
local code is fifo_device.hpp::push, fifo.cc::poll and port_channel.cc's data-before-
sync handler. The ordering rationale uses the transitive synchronization relations
in the NVIDIA PTX memory model:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model
No performance or correctness claim is made for those new builds before testing.


## Packet publication comparison complete

[Full focused record](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/packet_publish_screen/README.md).
Both independent candidates and the FIFO near-wrap build pass198 mixed ABI calls/rank;
the latter records16 resets across eight ranks and two lifetimes. All54 screen jobs
pass full-output/route/native checks. FIFO publication gives AG128 KiB out/in
34.209/35.016 us against contemporaneous control36.359/36.705 us; AR64 KiB gives
43.736/43.446 us against45.108/45.393 us. It does not yet close AG128 KiB versus
original UCCL near33 us. Coalescing AG data+flush alone gives inconsistent/modest
improvement (128-KiB out/in36.398/35.422 us); preserve this result rather than
assembling a synthetic per-size winner curve.

All recorded jobs are terminal. Useful next work is a fixed combination of FIFO
publication and the separately validated streamed RS512 path, then isolated AG128
launch geometry/coalesced flush experiments on that base. The last full paired
count remains25/126 slower rows. Any combination needs strengthened ABI, unchanged
full nccl-tests and final quiet original-reference paired qualification. Default
release promotion remains undone and native fallback remains prohibited.


## AR retirement/bank and shared-event candidates combined

[AR screen](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/ar_retirement_banks_screen/README.md)
passes 198-call mixed ABI on both independent changes and their near-wrap combination,
plus all 54 comparison jobs. Serial early retirement gives AR128-KiB out/in
52.692/54.446 us versus 56.009/56.343 us control; the 256-KiB out-of-place gap remains.
AR512 alternating medium banks give 100.679/100.758 us versus 117.315/115.959 us,
while preserving in-kernel outgoing retirement rather than adding a drain launch.

[Host completion screen](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/shared_host_screen/README.md)
passes the strengthened standalone component (60 calls/rank, alternating owned and
borrowed completion events with later owner re-records), mixed ABI198, and all36
comparison jobs. Sharing the owner's CUDA completion event removes duplicate record/
wait work while retaining stream dependencies and the CPU transport drain. It has a
modest benefit. Moving the host-gather cutover down to128 KiB is slower and is rejected.

The retained combined library is build/uccl_shared_host_event/nccl_lite/
libuccl_lite_nccl_strict.so. All nine unchanged full nccl-tests sweeps complete with
126/126 batch medians no slower than original UCCL, and full-output/route/native checks
pass. This is still a batch diagnostic: the last full aligned paired failure count is
25/126 for the older output-overlap candidate. No zero-regression admission or default
release promotion is claimed. All recorded controllers are terminal.

Further isolated work should target AG128-KiB local fan-out/ready cost without changing
completion, AR256-KiB remaining out-of-place latency, and public RS8–64-MiB per-call
costs. A two-phase logical-pair exchange can reduce cross-NUMA payload at the cost of
another readiness phase; any prototype must preserve private-bank reader safety and
both timing protocols. Graph setup optimization must not confuse pointer equality
with allocation identity for CUDA memcpy nodes; recycled allocation tests remain.


## Targeted gap experiments and graph setup checks

[Three-gap screen](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/three_gaps_screen/README.md)
passes five 198-call ABIs and all 48 jobs. AG128 pair exchange and both AR256 thread
changes are slower and remain unselected. RS8 MiB with16 blocks improves only slightly;
32 blocks is slower. These results do not replace the required full paired curve.

[Graph/RS screen](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/graph_rs_screen/README.md)
passes three ABIs and all48 jobs. Replacing graph input CE copies with GPU copies is
slower and rejected. RS8-MiB remote-first halves improve out/in from502.392/501.429 us
to495.875/487.423 us while retaining reduction order and send retirement. Descriptor
caching has modest/layout-dependent benefits. A separate NCCL-only queued client
validates48 calls/rank with new data, guards, offset changes and allocation recycling;
every rank exercises16 reuse and27 pending-writer cases, with one observed VA reuse.

[Allocation-aware bindings/banks screen](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/binding_banks_screen/README.md)
passes216-call ABIs (old cases preserved,64 MiB and delayed rank added), queued48-call
checks on all three candidates, and all48 comparisons. CUDA allocation IDs plus
legacy-IPC capability gate memcpy binding reuse; pointer equality alone never does.
Both binding candidates exercise40 hits,8 updates and2 recycled-allocation invalidations
per rank. Unknown allocation kinds keep the original UCCL rebinding path. Expanding
alternating banks through64 MiB preserves the existing32-bit offset bound check and
reduces several per-call costs, but the combination still misses some original rows.

Two single-piece graph variants are now running under build/check_onepiece_graph.py:
keep one piece through32 MiB or64 MiB, instead of forcing at least two above16 MiB.
This differs from changing target piece bytes while retaining the forced-two minimum.
Both builds have passed216-call ABI and queued48-call checks. First-round RS32/64 MiB
improvements are promising but provisional; consult the live controller and checkpoint.
Do not launch other GPU work or compile while its36-job comparison runs. The full
zero-regression goal remains unmet and no release has been promoted.


## Single-piece graph result and new full qualification

[Three-round screen](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/onepiece_graph_screen/README.md).
Both32/64-MiB ceilings pass216-call mixed ABI and queued48-call tests, including
actual VA reuse with allocation-ID invalidation and both descriptor cache branches.
All36 comparison jobs pass. The64-MiB ceiling improves RS32-MiB out/in from
1839.816/1826.909 us to1730.327/1727.248 us and RS64-MiB from3454.406/3450.241 us
to3356.355/3356.737 us. Other rows and all negative evidence remain preserved.

The selected fixed library is build/uccl_graph_single64/nccl_lite/
libuccl_lite_nccl_strict.so. A serial controller is running nine unchanged full
nccl-tests jobs followed by27 original-reference paired jobs, all operations,
all21 sizes1 KiB–1 GiB and both layouts. It does not redefine the goal around
focused gains. Check live session71767 and its saved qualification state before
starting any other compilation/GPU work. Full zero regression remains unproven.


## Single64 full batch passed; paired run and queued AG-tail work

All nine unchanged full nccl-tests jobs pass on the single64 fixed library with
126/126 batch medians <= original UCCL; native-absence, route and full-output checks
pass. The full 27-job paired run is live. First-round AG128 KiB remains about35 us
versus33 us reference; some host-path rows have large outliers, retained in the logs.
Only the completed three-round result can replace the prior full paired failure count.

Two guarded AG experiments are prepared but not yet compiled. Compact barriers keep
one CTA barrier after all writes to distinct peers in each fan-out, and avoid the
second CTA barrier immediately following packetReceive's existing barrier. Sources
remain immutable and the barrier before ready publication remains. Early own-output
copying at128 KiB moves the two locally owned shards into caller output before the
existing last fan-out barrier, while copying the other six shards only after peer
readiness. Input staging is already complete, standard in-place offsets remain safe,
and the same outgoing send ticket is still awaited. Independent and combined variants
will include prior coalesced-flush/ready-flag experiments only as explicitly fixed builds.

build/ag_tail_after_qualification.py (exec88537) waits for the current full qualification
before compilation or GPU jobs, then runs strengthened ABI216 and60 focused AG/AR
comparisons. Source hashes are frozen for those future builds. The currently measured
single64 .so and its source snapshot remain unchanged. No final admission or release
promotion is implied by the batch pass or the queued experiments.


## Single64 paired completion and remaining12 rows

[Full paired result](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/single64_full/README.md).
All27 jobs complete with full-output/route/native checks passing. Slower rows fall
from25 to12: AG7, AR1, RS4. AG128 KiB remains a stable gap; host-path outliers remain
in the mean-based comparison rather than being trimmed. First-round RS8-MiB out
contains one1095.723-us sample among roughly478–504-us samples; AG1-MiB in and4-MiB
out similarly have millisecond samples. All are retained, and any positive median
ratio remains a failure. No final zero-regression or release claim is made.

[AG-tail screen](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/ag_tail_screen/README.md)
passes all216-call ABIs and60 jobs, but coalescing CTA peer-copy barriers greatly
slows AG. Early own-output copying alone gives only a modest gain. Those combinations
remain unselected.

[Small/progress screen](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/small_spin_screen/README.md)
passes five216-call ABIs, two60-call host components and all66 jobs. Fusing owned
output with existing loads does not close AG128 KiB. AR barrier coalescing is slower;
MSCCL++ DeviceSyncer already uses a CTA-only path when blockNum==1, so a separate
replacement of that primitive is not a justified gain. Unpinned host progress polling
has positive evidence: AG256-KiB out/in39.282/39.752 us versus41.512/40.620 us control;
4-MiB out/in197.201/195.688 us versus197.750/196.243 us. None of its180 four-MiB samples
exceeds500 us, while control and pinned polling each have one. This is diagnostic
noise evidence, not proof of an absolute tail bound. Two busy CPUs/host are an explicit
resource cost. The policy preserves queue locks, stop/error handling and all GPU/
transport completion conditions. It is now running nine full unchanged nccl-tests
sweeps in session81439; consult the checkpoint before other GPU or compile work.


## Host polling full batch and next context/geometry experiments

The unpinned host polling library passes all nine unchanged full nccl-tests jobs,
all126 batch medians <= original UCCL; full-output, route and native-absence checks
pass. The latest full paired result is still12/126 slower rows for single64, not a
completed acceptance of the newer library.

Four independent candidates atop host polling have been built. AG128 KiB tests384
and768 threads after256/1024 were slower; other sizes keep their launch geometry.
A single graph piece through128 MiB sends a16-MiB shard, within MSCCL++'s32-bit
trigger-size field, preserving synchronization and final retirement. The exact-
primary-context fastpath caches a CUcontext only after cudaSetDevice has bound the
owner's primary context. Later completion may omit redundant runtime device calls
only when cuCtxGetCurrent matches that exact handle; ordinal equality alone is never
used. Foreign contexts retain the original switching/restoration behavior.

The ABI checker preserves every prior case and adds128 MiB plus a delayed rank,
for234 calls/rank. A dedicated completion client covers same-primary calls, another
GPU's primary context, a custom context on the same GPU, and returning to the primary,
with full result and context/device checks. Session69188 runs these correctness gates
before66 focused performance jobs. Source snapshots and commands are retained. No
new performance/admission claim is justified before those measurements complete.


## Context/geometry screen live; raw AG128 protocol queued

All four new candidates pass234 mixed ABI calls/rank, including the added128-MiB
boundary. The exact-primary completion variant also passes same-primary,
foreign-device, custom-context-on-the-same-GPU and restored-primary cases with
full results and exact context/device checks. Performance session69188 is live;
its first round is provisional and does not replace full paired results.

A guarded raw128-KiB AG path is prepared but not yet compiled. It reuses the
existing nonpacket serial gather with MSCCL++ Data|Flag|Sync: the one outgoing
raw shard writes its standard remote-output offset, the incoming signal still
precedes consumption, all local peer readiness/barriers remain, and the exact
send ticket retires after caller-output copying. It retains alternating private
banks. Other sizes/protocols remain in their current paths; LL sequence counts
advance only for LL calls, consistently on all ranks.

build/ag_direct_after_context.py (exec79965) waits for the current screen to
finish before building separate raw-AG and raw-AG+primary-context candidates,
then runningABI234 and18 focused comparisons. Expected source hashes are saved.
No queued experiment is a correctness/performance claim or a release promotion.


## Context/geometry and raw AG screens complete

[Context/geometry record](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/context_geometry_screen/README.md).
All234-call ABIs and66 jobs pass. The context fastpath gives a modest AR256 out/in
improvement (74.258/72.128 us versus75.277/72.876 us), while AG128 gains are small.
384/768 AG threads show no consistent advantage; single-piece RS128 is slower and
is not selected. Cross-device and same-GPU custom-context checks pass.

[Raw AG record](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/ag_direct_screen/README.md).
Both candidates passABI234 and all18 jobs. Raw AG128 out/in is32.367/32.134 us versus
35.319/35.604 us LL control. Adding primary-context reuse gives32.430/32.581 us.
The original reference was near33 us; only fresh full paired qualification can
establish the final claim. No native NCCL is loaded/executed.

RS128 three/four-piece variants are now running; they alter only that size and keep
current allocation-aware graph bindings, reduction order and retirement. After that
screen succeeds, build/ar256_after_rs128.py will test AR256 using the already allocated
alternating medium banks, retaining in-kernel send retirement, at2/4/8 blocks. This is
different from the previously rejected256-KiB direct path with single-bank reuse.
All old validation cases remain. Neither queued work nor focused gains constitute
full zero-regression admission or release promotion.


## AR256 bank result and RS128 prefix-bank validation

[RS piece-count record](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/rs128_pieces_screen/README.md)
preserves234-call correctness and36 performance jobs: three/four pieces at128 MiB
are slower, so the existing two-piece policy remains.

[AR256 bank record](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/ar256_banks_screen/README.md)
passes three234-call ABIs and all24 jobs. Eight blocks gives71.221/70.501 us out/in
versus74.136/72.139 us control. Four/two blocks are less effective. The new reuse
protocol differs from the old rejected single-bank256-KiB experiment. The candidate
contains the already measured raw128-KiB AG path; no curve is stitched from variants.

The RS128 prefix-bank experiment allocates no new GPU storage. Two452-MiB layouts
fit inside the existing1-GiB large data region when capacity permits. Entry after an
ordinary large-layout call retains the existing reuseLarge coordination; subsequent
128-MiB calls alternate disjoint banks. Any ordinary large-layout operation invalidates
that mode and retains its own original reuse guard. CUDA graph keys include extent
and bank offset; fixed communicator capacity prevents a normal/prefix128-key collision.
Other medium banks and small-packet storage are disjoint from this region. The existing
32-bit offset bound remains unchanged.

The first prefix candidate passesABI234 and a new NCCL-only layout queue test:
48 calls/rank alternate128/256/128-MiB RS and AR, with fresh GPU-produced input,
full output/guards, offset changes and allocation recycling. All ranks observe exactly
3 entries,29 same-layout reuses and2 departures, proving those transition branches
were exercised. The primary-context combination is still validating before36 focused
jobs. Consult live session27949 and build/check_prefix128_state.json; do not overlap
compilation or GPU tests. These results do not establish full zero-regression admission.


## Prefix-bank comparison complete; combined full qualification live

[Prefix128 record](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/prefix128_screen/README.md).
Both234-call ABI runs and48-call queued layout-transition checks pass. Every rank
exercises3 entries,29 reuses and2 departures with full output/guards and actual
allocation reuse. All36 performance jobs pass. RS128 out/in is6720.965/6717.986 us
versus6740.765/6770.617 us control; the separate primary-context candidate measures
6708.605/6713.237 us. No per-size mixing of these libraries is used.

The fixed non-primary prefix candidate combines the measured raw AG128 path,
AR256 alternating-bank direct ring, busy host progress, allocation-aware graph
bindings and previous selected graph/ring policies. It now passes all nine unchanged
full nccl-tests sweeps with126/126 batch medians <= original UCCL; numerical, exact
route and native-absence checks pass. Its27-job full original-reference paired run
is live in session18375. The earlier12-row full failure remains authoritative until
all three new rounds complete. No final zero-regression or release claim is made.

The usage document now separates strict correctness commands from historical
NCCL-relative5% measurements and explicitly states that default build promotion is
pending. Current experiments use exact command manifests and external dependency
identities; they are not silently enabled in the ordinary Makefile target.


## Independent raw-log audit added during the running prefix comparison

`tests/nccl_mscclpp/audit_uccl_pair.py` reads completed logs without rerunning GPU
work. It verifies all27 unique runs, original reference binary hashes and process
layouts, matching warmups/iterations and1-KiB–1-GiB coverage, every sample index,
finite positive timings, unchanged numerical tolerance, native absence and exact
candidate route counts. It recomputes means from every sample without trimming,
then medians and all126 ratios, and checks the existing manifest/CSV/result.
The earlier single64 result independently reproduces12 slower rows and exits2;
it is not converted into a pass. The current prefix run remains live and is not
audited as complete. Batch qualification and default build/release promotion are
separate requirements even if a future paired audit passes.


## Prefix full audit: four rows remain; completion observation diagnosis

[Full paired report](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/prefix_full/README.md)
completes27 jobs with all correctness/route/native checks passing. Slower rows fall
from12 to4: AG8 MiB out and16 MiB in, RS16 MiB in (+0.424%) and128 MiB out (+0.0877%).
Independent raw-log auditing reproduces all126 means/medians and failures, exits2,
and rejects nonfinite timing, a missing route, a duplicate sample and a modified mean.
All raw outliers remain; no percentile or tolerance replaces the mean-based gate.

[Completion diagnosis](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/completion_trace/README.md)
uses separate instrumented binaries, each passingABI234 and six full AG runs. Two
captured8-MiB operations have GPU intervals341/343 us but host GPU-wait phases1509/1232 us.
This establishes late host observation in those samples, not its ultimate OS/driver cause.
The16-MiB CPU-only spikes were not reproduced with timing events. Diagnostic overhead
is explicit and none of these traces is used as a scored performance claim.

[Exact-event polling screen](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/query_completion_screen/README.md)
passesABI234,48 queued layout calls and all12 performance jobs. It polls the same event,
preserves all errors and transport retirement, and changes no collective data path.
Observed8/16-MiB tail counts improve without excluding samples. A new fixed query library
is running nine full nccl-tests jobs then27 full original-reference pairs in session53167.
No final admission or default build promotion is justified until that full audit completes.


## Explicit completion query full audit: two RS rows remain

[Full paired report](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/query_full/README.md)
completes all 27 original-reference paired jobs and all nine unchanged nccl-tests
sweeps. Batch medians pass 126/126; the matched per-call audit passes 124/126.
AG and AR have no slower rows. RS 256 KiB out is 49.234 versus 48.951 us
(+0.578%); RS 128 MiB in is 6746.988 versus 6741.384 us (+0.0831%). These
remain failures under the unchanged zero-slowdown rule. All correctness, route,
native-absence and four corrupted-log rejection cases pass. The original source
and raw samples are retained; no outliers are discarded. The conditional medium
AG consumer experiment was skipped because all targeted AG rows passed.

The next isolated candidate adds only the previously screened exact-primary-context
reuse flag to the query build. It may remove redundant runtime device switching;
it does not change collective arithmetic, communication or completion boundaries.
It must repeat 234 ABI calls/rank, four device/context cases, 48 queued layout
transitions and a 36-job comparison covering RS 128–512 KiB, RS 64–256 MiB and
AG 4–16 MiB. This focused comparison is not full admission. The zero-regression
goal and default build promotion remain pending.


The [primary-context focused screen](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/query_primary_screen/README.md)
has now completed all 36 jobs and the required correctness/context/layout checks.
RS 256 KiB improves from 49.388/49.855 us to 48.685/48.619 us (out/in).
RS 128 MiB changes from 6716.959/6693.635 to 6709.118/6706.086 us, so the
in-place result is slower than this control. Several neighboring sizes and AG
points are also slower; every point is preserved in the report. This motivates
full comparison against the original baseline, not a local acceptance claim.
The fixed combined candidate is running nine full nccl-tests sweeps then 27
original-reference pairs in session 52152. Session 7096 waits on that controller
before raw-log auditing, rejection fixtures and archival. No source or measured
binary is being changed while qualification runs.


## Independent batch-log and executable audit

The primary-context candidate completes all nine unchanged nccl-tests sweeps,
with all 126 batch medians no slower than the recorded UCCL reference. The full
matched per-call campaign remains live. `audit_nccl_sweeps.py` independently
checks the three staged test executables against the frozen binaries, the staged
replacement hash, all 21 extents and both layouts, finite positive timings,
correctness summaries, all eight ranks, exact 84 calls per rank/size in this
frozen protocol, native absence, and medians reconstructed from all three runs.
It reproduces the primary and prior query batch results. Four rejection fixtures
(NaN, missing route, duplicate timing row, altered stored timing) all fail as
intended. This strengthens evidence for the unchanged nccl-tests requirement;
it does not replace the matched per-call performance gate or change its rules.


## Residual profiling sessions discovered during primary qualification

Read-only host checks found no CFS quota/throttling for the active job. On the
remote host, nsys sessions `profile-2618669`, `profile-2618668` and `profile-2618671`
remain in Collection state roughly 20 hours after launch. Agent PIDs are
2618754/2618782/2618803. Their recorded target PIDs 2618825/2618833/2618837 are absent;
the retained stderr identifies strict UCCL Lite ranks 5/4/7 with native absence.
These are residual sessions from earlier UCCL profiling, not the current job.
Their counters still accrue CPU time and they hold GPU device descriptors. This
is a potential measurement interference source, not a proven explanation of the
observed timing tails. The current qualification continues with its environment
unchanged and all negative samples retained.

Cleanup session 87198 waits for qualification and raw-log audit to terminate.
It checks exact agent identity/start time, same UID, absent target and UCCL stderr,
backs up session/trace files, then stops only those named sessions and uses
`nsys shutdown --kill=none` if needed. No application signals are sent. All backups
are copied into build/nsys_residual_cleanup. Further GPU work must wait for cleanup
to finish. A fresh complete original-reference comparison under the clean
measurement environment is required before promotion; the baseline, timing rule
and sample retention policy remain unchanged.


## Primary full result and verified profiling cleanup

[Primary full report](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/query_primary_full/README.md)
completes 9 nccl-tests sweeps and 27 original-reference pairs. Both independent
raw-log auditors and eight negative fixtures pass. Batch timings have 0/126 slower
rows; the paired result still has two RS failures: 256 KiB in-place +1.3855% and
512 MiB in-place +0.0524%. AG and AR have no slower rows. This run's residual
profiling environment is explicitly recorded, and all samples remain intact.

All three identified orphaned nsys collections were stopped after the run. The
first cleanup controller conservatively failed because shutdown had returned while
two agents still briefly appeared in Shutdown state. No clean GPU run started.
Independent subsequent verification confirms all three agent PIDs absent and both
hosts' nsys session lists empty. The initial failure and verification are both
retained in build/nsys_residual_cleanup; 267 MiB of session/trace backups were
copied locally. No application signals were sent.

Session 7172 now runs the same SHA-verified binary in build/uccl_query_primary_clean,
with no rebuild or policy change, through all nine batch and 27 paired jobs.
The original reference binaries, warmups, iterations and zero-slowdown comparison
are unchanged. Before/after profiler-session observations, both raw-log audits,
negative fixtures and a separate report are part of this run. Final admission
and default build promotion remain pending.


## Clean primary full audit: one AG point remains

[Clean full report](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/query_primary_clean_full/README.md)
completes nine unchanged nccl-tests sweeps and 27 original-reference paired jobs.
Both raw auditors and all rejection fixtures pass. Batch timings pass126/126;
paired timings pass125/126. AR and RS have no slower rows. AG16MiB in-place is
700.144 us versus671.947 us (+4.1963%). The original reference, binary SHA and
protocol are unchanged. Both hosts have empty nsys session lists before and after.
This does not prove that the earlier residual profiling caused any specific tail;
all previous measurements remain intact and the zero-regression goal remains unmet.

The conditional RS256KiB streamed experiment was skipped without changing source,
because RS256KiB passed. The conditional medium AG experiment now proceeds in
session55247. It tests24/32 consumers at8–32MiB AG while retaining16 consumers
for bulk and keeping producer/control roles and retirement intact. Component60,
ABI234 and36 focused AG/AR jobs precede any selection. Header launch capacity is
bounded by cooperative occupancy; the optional timeline buffer uses the maximum
configured grid. No default-policy promotion is implied.


## Medium consumer screen complete; fixed24-consumer qualification live

[Focused report](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/medium_consumers_screen/README.md)
preserves all36 jobs for the primary control and24/32 medium-consumer candidates.
Both candidates pass component60 and ABI234; every performance job passes output,
route and native-absence checks. The24-consumer variant improves AG16MiB in-place
from662.533 to649.498us, while out-of-place slows from659.997 to664.469us (+0.678%).
Both32MiB layouts and the measured AR points improve; the32-consumer alternative
is less consistent. Selection is of one fixed24-consumer library, not per-size
best results. Bulk keeps16 consumers, and the original control retains its source
and binary identity. Session26541 now runs all9 nccl-tests and27 paired jobs,
with independent raw audits and no active profiler sessions. Zero regression and
default build promotion remain unproven until full qualification completes.


## Medium24 full audit and explicit policy build verification

[Full report](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/medium24_full/README.md)
passes all9 nccl-tests sweeps and125/126 aligned per-call points. AG and AR have
no slower rows. RS16MiB out-of-place is912.103 versus911.773us (+0.03616%) and
remains a failure without tolerance changes. Both raw auditors and8 rejection
fixtures pass; profiler-session lists are empty before and after the campaign.

[Build verification](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/policy_medium24_build/README.md)
confirms the explicit UCCL_LITE_NCCL_POLICY=medium24 preset, affected-target builds,
unchanged-policy no-op, policy-switch invalidation and rebuilt-library ABI234.
The loaded ELF code/data identities match the measured binaries; raw hashes and
the precisely excluded NVCC symbol/build-ID metadata are recorded separately.
The default remains base and no zero-regression promotion is made.

The next candidate reduces repeated graph-binding attribute queries. It queries
BUFFER_ID first only for an already eligible binding with the same input pointer;
a matching live ID retains the previously checked legacy eligibility. Changed,
unknown or ineligible bindings retain complete attribute validation and rebinding;
unexpected driver errors still propagate. NVIDIA documents allocation IDs as
process-lifetime unique and not reused after freeing:
[CUDA pointer attributes](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html).
No input data or numerical result is cached. Session74217 runs ABI234,48 queued
fresh-input/guard/address-recycling checks, then24 focused RS/AR jobs. Full
qualification remains required after any candidate selection.


## Allocation-ID-first experiment rejected; driver completion experiment

[Focused ID-query report](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/binding_id_fast_screen/README.md)
passesABI234 and graphQueue48 with actual VA reuse and40/8/2 binding hit/update/
invalidation counts on every rank. All24 jobs pass correctness and native absence,
but16-MiB RS does not consistently improve. The flag is not selected; all negative
results remain. The medium24 full RS16MiB +0.03616% failure is still authoritative.

The next isolated flag uses cuEventQuery on the exact existing completion event,
retaining device/context guards, GPU algorithms and transport retirement. NVIDIA
explicitly permits runtime and driver event handles to interoperate:
[CUDA runtime/driver interaction](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DRIVER.html).
Session45752 builds on the medium24 control with GRAPH_BINDING_ID_FAST disabled,
then runsABI234, four device/context cases, graphQueue48 and36 focused RS/AR/AG
jobs. Driver errors remain errors; this introduces no native NCCL path. No final
performance or default-policy promotion is implied.


## Driver-query experiment rejected; public16MiB streamed RS

[Driver-query report](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/driver_completion_screen/README.md)
passesABI234,context4 and graphQueue48, but16MiB RS slows in both layouts. It is
not selected, and all36 jobs remain archived. Both query experiments are disabled
in the next candidates; the medium24 control remains the selected base.

Session2058 now tests the existing remote-first streamed RS kernel at public16MiB,
with8 or16 CTAs, using the existing alternating medium banks. The hostAllReducePart
boundary explicitly keeps hybrid AR's private graph producer. No new GPU storage,
packing, reduction operation or native path is introduced. Cooperative occupancy
limits include the remote-first kernel. The234-call ABI retains all cases and
adds a delayed peer at the existing16MiB RS case; full-output/guard graphQueue48
then tests queued mixed RS/AR calls and address reuse. A36-job RS/AR8–32MiB screen
follows before any full qualification or policy selection.


## Streamed16MiB RS rejected; immutable dispatch and graph-launch overhead

[Streamed RS report](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/rs16_stream_screen/README.md)
preserves all36 jobs. Both8/16-CTA candidates passABI234 with the delayed16MiB
peer and graphQueue48, but are slower at16MiB (971–978 versus908–910us). They are
not selected. All three recent rejected features remain disabled in new candidates.

Source inspection of the retained MSCCL++ runtime shows that this private collection
always selects one immutable registered algorithm per operation. Session78739 tests
caching those same objects to avoid repeated request/string/map work while retaining
NativeAlgorithm::execute, its initialization/context cache and all per-call guards.
An independent variant calls cuGraphLaunch on the existing graph executable and
stream. The installed CUDA headers define cudaGraphExec_t and CUgraphExec with the
same CUgraphExec_st pointer type; runtime correctness remains explicitly tested.
Both candidates must passABI234 and graphQueue48, then54 focused RS/AR/small-AG
jobs. Neither is combined or promoted before measurement. The original full-matrix
RS16MiB +0.03616% failure and all negative evidence remain unchanged.


## Host dispatch screen not selected;16MiB graph reduction geometry

[Host dispatch report](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/host_dispatch_screen/README.md)
preserves all54 jobs. Both candidates passABI234 and graphQueue48, but neither
improves16MiB RS out-of-place:913.349/909.489us versus907.807us. They are not
selected for the remaining per-point gate; gains and slower neighboring points
remain available. NativeAlgorithm reuse is retained in the selected control.

Session28215 now tests64 or128 CTAs instead of256 for only the reduction kernels
in16MiB scatter graphs. Copy engines, graph dependencies, signals, per-element
addition order, output ownership and retirement are unchanged. The same geometry
also applies to hybrid AR's private16MiB scatter graph, so both RS and AR are
measured. ABI234 (including delayed16MiB peer) and graphQueue48 precede36 focused
jobs. All earlier rejected flags are disabled. The original full medium24
RS16MiB +0.03616% failure remains a failure; no default promotion is made.


## Smaller graph grids rejected;512-CTA16MiB experiment

[64/128-CTA report](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/rs16_graph_screen/README.md)
preserves all36 focused jobs and correctness results. Both grids have identical
CUDA fatbins, passABI234 and graphQueue48, but do not improve16MiB RS. They are
not selected. Session32990 tests512 CTAs with the same256 threads, which assigns
one float4 per thread for the2MiB shard at this extent. The independent graph
reduction kernels do not require whole-grid residency. The original CE copies,
graph order, signals, arithmetic and retirement remain. ABI234 and graphQueue48
precede24 RS/AR jobs. The full medium24 +0.03616% failure is still authoritative;
no tolerance adjustment or release promotion is made.


## 512-CTA screen complete; full qualification live

[Focused report](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/rs16_graph512_screen/README.md)
passesABI234,graphQueue48 and all24 focused jobs. RS16MiB out-of-place improves
912.745->911.646us (0.120%); in-place slows910.335->911.871us (0.169%). RS8MiB
out-of-place is13.9% slower in this screen despite an unchanged kernel path.
All such samples and neighboring results remain; no cause or harmlessness is
assumed. Session67995 now runs one fixed library through9 unchanged nccl-tests
sweeps and27 original-reference pairs, environment observations, both raw audits
and8 rejection fixtures. This is a new candidate qualification, not acceptance
of the mixed focused results. The latest completed full medium24 failure is still
RS16MiB out-of-place +0.03616%, and default build promotion remains pending.


## Final qualification and default promotion — complete

The fixed graph512 library passes all 9 unchanged nccl-tests sweeps and all
27 original-reference paired jobs: **126/126 points, zero slower rows**. Independent
raw-log auditors reproduce every row and reject all 8 corruption fixtures.
Worst candidate/original per-call latency ratios are AG 0.996521338,
AR 0.963631707 and RS 0.999463772. Hardware, dtype, range, layouts, repetitions,
original process layouts and full-retirement timing scope are unchanged.

The default Makefile policy is now graph512. A build without a policy override
produces the canonical strict library and libnccl.so.2 aliases. All loaded ELF
code/data and CUDA fatbins match the measured library, with raw hashes retained
and only documented non-runtime/build-ID metadata excluded from that identity.
Both hosts' hardware/dependency records match, apart from the new raw library hash.
The canonical default passes its own ABI234 and another stock nccl-tests sweep
for each operation over the full range, with complete output/route/native checks.
Old default artifacts were backed up, and all prior negative results remain.

[Final qualification and usage](../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/qualified_release/README.md)
includes the completion audit, exact artifacts, build/source provenance, commands
and limits. This completes the requested no-regression/no-native-fallback work
for the recorded FP32, eager, one-GPU-per-process, two-four-L4 configuration and
21 power-of-two extents from 1 KiB through 1 GiB, both layouts. It does not claim
validation of other configurations or unsupported NCCL API modes.
