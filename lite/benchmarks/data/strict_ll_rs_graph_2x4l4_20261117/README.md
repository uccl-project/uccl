> Historical optimization record. The final independent result passes 126/126
> rows under the 5% gate: [final report](../strict_release_2x4l4_20261117/README.md).
> Earlier incomplete/failed statements below describe their recorded revisions.

# Strict RS graphs and LL packets, November 17

The full no-regression objective is still unmet. These are exploratory screens,
not final admission. Native NCCL is absent from every candidate process; it runs
only in independent baseline jobs. No best-per-size envelope is reported.

## Measurement contract

Two four-NVIDIA-L4 hosts, 10.10.55.2 and 10.10.55.1, eight MPI processes,
FP32 SUM AllReduce/ReduceScatter and FP32 AllGather, both layouts. Unmodified
nccl-tests 2.19.7 and separate NCCL 2.28.7 controls. All full screens cover
21 powers of two, 1 KiB–1 GiB, native/candidate/candidate/native, 5 warmups and
20 timed iterations. Timing is maximum-rank host enqueue/completion batch means
(-a 3), including per-call graph updates and completion but excluding preparation.
CSV points are medians of two independent job means/backend. Exact topology,
versions, commands and identities are in each manifest and host JSON.
The unchanged gate allows at most 5% slower in every one of 126 rows; final
independent two-round confirmation is still required.

## Experiments

| Directory | Observation / decision |
| --- | --- |
| goal_rs_ce_graph | RS CE graph, direct caller input, fused output. 47/126 fail; all RS 16 MiB–1 GiB pass. Retained. |
| goal_small_ll | IB LL16Packet protocol with every-call flush. Correct but slower; 50/126 fail. |
| goal_small_ll_batch | Initial batching attempt accidentally retained per-call drain and doubled the packet sequence. 49/126 fail. This is not evidence for the corrected batching protocol. |
| goal_small_ll_batched | Correct bounded batch drain, wrap reset, teardown drain. 40/126 fail. Forced-wrap 601-iteration small-message validation passes. |
| goal_small_pull | Local peer pull, including one-shot tiny AR. Correct; only tiny AG benefits. 37/126 fail. Rejected in favor of local packet protocol. |
| goal_local_ll | Dedicated local LL packet region, AG and RS. 27/126 fail; RS 1–32 KiB passes. |
| goal_ar_local_ll | AR local LL scatter/gather. 26/126 fail; retain only up to 8 KiB because larger AR regresses. |
| goal_packet_ag_tiles | Four-CTA AG packet tiles at 32–128 KiB. 25/126 fail; severe AG regression. Rejected. |
| goal_ag_single_ready | One CTA; AG local LL up to 16 KiB, raw local fan-out above with only the final ready notification. Result recorded separately below. |

## Protocol obligations

Local packet storage is separate from network send/receive packet slots and raw
scratch. Two alternating banks use monotonically increasing nonzero 32-bit flags.
Wrap quiesces all ranks, drains IB, clears both packet regions and synchronizes
before reusing flags. Send completion is drained every 32 calls, on transition
to raw protocols, and before teardown; bounded outstanding WRs remain below CQ
capacity. GPU output readiness alone does not permit releasing registered memory.

Extended ABI checks cover 126 calls/rank, two communicator lifetimes, delayed
ranks, per-rank alignment/layout differences, mixed sizes/operations, rotated
inputs and freed/reallocated caller buffers. All archived completed ABI runs
pass. No sanitizer or fault-recovery claim is made.

The single-ready attempt named full used a mistyped second hostname, never
launched MPI, and was terminated. Its partial staging is not measurement evidence.
The corrected full_correct_hosts directory is the authoritative sweep.

## Subsequent medium-message screens

`goal_ag_single_ready/full_correct_hosts` passes numerical/routing checks but
fails 20/126 performance rows. `goal_medium_ce_graph/full` fails 27/126; simply
lowering the graph threshold to 1 MiB is rejected.

`goal_serial512/small` uses 10 warmups and 100 timed iterations, 1–512 KiB,
and fails 2/60 rows. Parallel local ready notifications in
`goal_peer_ready_parallel/small` reduce the 64-KiB AG/AR latency; 1/60 still
fails due to a retained 32-KiB in-place AG job outlier (29.8 vs 18.74 us).

`goal_medium_double_bank/abi` first failed at 1-MiB AR because the existing
local gather assumed zero data offset. The corrected helper takes the actual
bank offset; `abi_offsets` passes all 126 guarded calls/rank over two lifetimes.
The corrected full sweep fails 15/126; all RS rows pass, but medium AR gets
slower. Dedicated medium double buffering is retained for AG/RS only.

`goal_ar_medium_piece1/medium` and `goal_ag_medium8/medium` cover 1 KiB–16 MiB,
10 warmups/50 timed iterations. They fail 5/90 and 6/90 respectively. One-piece
AR worsens the GPU path at 4 MiB and is rejected there; the 8-MiB graph retains
one piece. Eight-CTA AG improves 4 MiB but still misses 8 MiB.
`goal_ar_medium_single_bank/medium` restores AR's original single bank and
GPU two-piece policy, failing 4/90 rows. These partial sweeps do not substitute
for full-range independent final confirmation.

## Final streamed paths and validation

Later verdicts and failed rows are preserved in `later_screen_results.json`
and their respective directories. Vectorized aligned AG staging recovered its
medium curve. Streaming AR reductions/output plus aligned 8-MiB banks produced
a first full 126-row screening pass. Streaming RS subsequently reduced 4-MiB
latency to about 254–255 us versus about 270 us native. Four- and sixteen-CTA
AR variants were slower than eight CTAs; negative curves remain recorded.

The retained release adds aligned AR IO, bounded AR/LL completion retirement,
blocking finalize draining, and explicit synchronization. The new finalize
assertion initially expected the wrong error enum; the adapter had rejected the
call with its existing ncclInvalidArgument. That failure and its explanation
remain in `goal_ar_batched_completion/`; numerical references and performance
criteria were not relaxed. Queue stress validates 604 calls/rank over two
lifetimes, including releasing caller allocations before transport retirement.

Final independent data, controller-recovery provenance, source snapshots and
all supporting checks are in the linked release report. Unmeasured build-only
variants do not contribute to any admitted curve.
