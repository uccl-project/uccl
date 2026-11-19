# Output-copy overlap and ordered send retirement

The full zero-regression goal remains unmet. Two hosts with four NVIDIA L4s each,
FP32 AllGather, fixed 1-GiB allocations, 1–128 KiB, both layouts, 20 warmups and
30 timed calls, three repetitions with counterbalanced variant order. Timing
includes GPU and send retirement; medians of maximum-rank per-call means,
no best-per-size envelope. Baseline is the early-flush-ticket candidate.

The candidate moves the existing packet-AG flush-ticket wait after the final
caller-output copy, whose stores cannot modify the IB send packets. The kernel
still cannot complete before the same ticket is retired. Local-reader barriers
and alternating bank lifetimes are preserved. For outstanding nonpacket sends,
it also queues the existing drain behind the exact completed-collective event
before the host waits. This latter change is inactive on this small-AG screen
because inline packet retirement leaves no outstanding send.

The candidate passes 144 mixed ABI calls/rank. All 12 screen jobs pass full-output,
exact route and native-library-absent checks.

| Variant | 32 KiB out/in (us) | 64 KiB out/in (us) | 128 KiB out/in (us) |
| --- | --- | --- | --- |
| Early-ticket control | 26.217 / 26.286 | 29.390 / 29.440 | 36.692 / 36.787 |
| Output overlap | 26.046 / 26.320 | 29.387 / 29.148 | 36.091 / 36.312 |

This is a modest screen result, not final admission. The 128-KiB values still
exceed the previous original-reference result near 33 us. One repeat had a
40.734-us in-place 128-KiB mean; it is retained in the raw record. A full fixed-
library comparison is running under build/uccl_output_ordered/: nine unchanged
nccl-tests sweeps followed by the 27-job original-reference paired protocol,
all three collectives, 1 KiB–1 GiB, both layouts. The final campaign runs without
concurrent compilation or candidate GPU tests. Native NCCL fallback is forbidden.

The nine unchanged full nccl-tests jobs have now completed, all correctness,
route and native-absence checks passing. All 126 batch median rows are <= the
original reference. The full paired comparison is still running; this batch
result does not establish zero regression under the aligned per-call protocol.
