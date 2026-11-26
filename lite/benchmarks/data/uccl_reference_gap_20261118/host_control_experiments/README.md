# Host-control experiments, 2026-11-18

These are experimental UCCL-only builds on two hosts with four NVIDIA L4s
per host, FP32 SUM/copy. Every completed build below passed the extended NCCL ABI check
(126 guarded calls/rank, two communicator lifetimes, 1-GiB prepared capacity).
Candidate processes loaded no native NCCL and executed no fallback.

Each performance curve is a fixed build: three unchanged nccl-tests sweeps,
21 powers of two from 1 KiB to 1 GiB, both layouts, 20 warmups and 30 timed
iterations, maximum-rank host enqueue/completion batch means. Entries below
are medians across the three sweeps; no best-per-size selection is used.
The fresh original UCCL reference uses different harness/process conventions,
so ratios are diagnostic and cannot establish paired zero-regression admission.

| Fixed experiment | AG 1 GiB out/in (ms) | AR 1 GiB out/in (ms) | Decision |
| --- | --- | --- | --- |
| CUDA stream write publication | 62.778 / 61.152 | 119.161 / 119.248 | Reject; slower |
| Pin network progress CPUs | 44.900 / 44.185 | 100.951 / 100.850 | No material improvement |
| Fuse publication and next wait | 44.688 / 44.492 | 100.278 / 100.316 | Small gain, still fails target |
| Flush after reader completion | 44.845 / 44.644 | Not run | Reject AG change; no gain |
| Batched ready H2D copies | 44.872 / 44.517 | Not run | No gain |
| Parallel warp flag loads | 44.382 / 44.463 | 100.298 / 100.376 | Helps 8 MiB; bulk still fails |
| CPU-aggregated ready flags | 44.683 / 44.667 | Not run | Reject; 8 MiB worsens |
| Six slots + warp flags | 45.955 / 46.185 | Not run | Reject; bulk worsens |
| Two slots + warp flags | 44.494 / 43.782 | Not run | Helps 8 MiB/in-place; still fails |
| Cooperative medium, two packets | 44.813 / 44.393 | Not run | Medium gain only; bulk CE unchanged |
| Cooperative medium, one packet | 44.592 / 44.400 | Not run | Medium gain only; bulk still fails |
| Fresh original reference | 41.922 (out only) | 96.445 (out only) | Unchanged target |

Fused publication only waits for an already-submitted successor in the same
invocation; the final packet never waits for a future invocation. Late flush
retains both reader ACKs and outgoing-write retirement before slot reuse.
Its correctness pass does not imply a performance benefit. Native-NCCL
baseline wording inside the ABI staging manifests is inherited metadata;
their actual command/results lists contain only the ABI job, no timings or
native control. The performance manifests identify the diagnostic scope.

The Nsight summary comes from the earlier hybrid prototype, profiling node 0
while node 1 ran unprofiled, with 1-GiB AG and 2 warmups/3 timed iterations.
It shows thousands of per-packet wait/publication kernels and 256-KiB/512-KiB/
1-MiB H2D copies. These instrumented durations are not scored timings.
Raw captures remain under build/uccl_host_hybrid_candidate/profile_ag1g/.
Build commands, library hashes, correctness logs, full curves and negative
results are retained in each subdirectory. Generated binaries remain in build/.

## Cooperative medium-path evidence

The new cooperative kernel keeps the same shared host rows, MSCCL++ CPU IB
transport, receive credits, source uploads and reader acknowledgements. A single
resident grid uploads one or two medium packets, waits for local readiness,
copies local rows (retaining its own shard on GPU), waits for remote rows, then
copies remote rows. Each producer issues a system fence before the grid leader
publishes upload; all grid readers finish before the leader publishes ACKs.
Preparation checks cooperative-launch support and the whole-grid residency
bound. The kernel uses the caller stream and actual input/output pointers.
Unaligned buffers use scalar copies; there is no native fallback or pointer cache.

The first cooperative experiment uses the original 256-KiB source packet cap.
The second prepares 512-KiB capacity and uses one whole source shard only for
512-KiB–4-MiB complete output; bulk CE still uses 256-KiB source pieces. This
is one measured fixed policy, not an envelope of independently selected points.
Both pass the extended ABI. Both also pass 54 direct calls/rank over two
lifetimes: pointer offsets, mixed per-rank in-place layouts, alternating streams,
reallocation, delayed rank, pageable D2H, and a 4-MiB-minus-128-byte tail, plus
larger noncooperative cases through 128 MiB + 128 B. No sanitizer claim is made.

One-packet cooperative build, median of three full-range sweeps:

| Complete AG output | Candidate out/in (us) | Fresh UCCL reference (us) |
| --- | --- | --- |
| 512 KiB | 42.10 / 42.13 | 54.24 |
| 1 MiB | 62.97 / 63.01 | 73.87 |
| 2 MiB | 105.06 / 104.86 | 114.32 |
| 4 MiB | 188.61 / 188.32 | 197.17 |

These diagnostic gains do not establish matched admission. In the same fixed
build, 8-MiB AG is 422.21/416.41 us versus 355.37 us and 1-GiB AG remains
44.592/44.400 ms versus 41.922 ms. The full-scope goal remains unfulfilled.

A fresh three-sweep RS curve with the warp-flag build leaves the RS algorithm
unchanged: 1-GiB out/in are 54.032/53.870 ms versus 53.693 ms reference;
8-MiB is 508.37/506.65 us versus 494.40 us. Twelve of 42 RS rows are slower
in this cross-harness comparison. Small margins are not rounded into passes.
This motivates measuring larger graph pieces; it does not relax the zero gate.

The ABI-only metadata in future runs now correctly says no baseline/no timing;
previous manifests remain immutable, with their actual command/results lists
as evidence. Correctness rules and performance thresholds were not changed.


## RS follow-up decisions

All three RS follow-up builds passed the extended ABI and three full-range
nccl-tests sweeps. Neither graph-piece change improves bulk performance:
16-MiB pieces give 55.889/55.907 ms at 1 GiB and 4-MiB pieces give
54.900/54.887 ms, versus the 8-MiB-piece control's 54.032/53.870 ms.
Both are rejected. The reference target remains 53.693 ms, with zero tolerance.

Extending the existing streamed RS kernel through 16 MiB gives 8-MiB
out/in 490.06/490.10 us versus the reference's 494.40 us, but regresses at
16 MiB (about 990 us versus 914 us reference). The ensuing combined candidate
therefore uses streamed RS only at 8 MiB and retains 8-MiB graph pieces above.
That combined policy must be remeasured as one library across all operations;
these separate experiment minima are not its acceptance curve.
