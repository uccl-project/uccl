# Ordered synchronization and early flush tickets

The full zero-regression goal remains unmet. This screen uses two four-L4
hosts, FP32 AllGather, fixed 1-GiB allocations, full internal retirement,
20 warmups and 30 iterations per size, both layouts, three repetitions with
counterbalanced variant order. Sizes are 1–128 KiB; this is a focused screen,
not full-range admission. Each value is the median of three per-call maximum-rank
means. No per-size minima are selected and no retirement requirement is removed.

The ordered synchronization variant queues a CUDA event wait and drain on the
private builder stream before waiting on the CPU. It does not depend on later
application work in the caller stream. The early-ticket variant uses MSCCL++'s
existing FIFO TriggerSync and flushDonePos semantics: after the AG's final IB
put, it posts a flush request, overlaps local fan-out/output, and waits for that
exact FIFO position before returning. It applies only where no later IB writes
belong to the same operation. Local reader barriers remain unchanged.

Both variants pass 144 mixed ABI calls/rank over two lifetimes. All 18 screen
jobs pass full-output, exact route and native-library-absent checks. Median
out/in latency in microseconds:

| Variant | 1 KiB | 32 KiB | 64 KiB | 128 KiB |
| --- | --- | --- | --- | --- |
| Inline retirement control | 20.781 / 21.102 | 27.790 / 27.595 | 30.999 / 30.960 | 37.794 / 37.871 |
| Ordered synchronization | 23.685 / 23.490 | 30.474 / 30.722 | 33.811 / 33.562 | 40.453 / 40.579 |
| Early flush ticket | 19.340 / 19.160 | 26.321 / 25.762 | 29.643 / 29.278 | 36.594 / 36.312 |

Early-ticket retirement subsequently passes all nine unchanged full nccl-tests
sweeps (all operations, 1 KiB–1 GiB, both layouts), with all 126 batch diagnostic
medians <= the original reference. That batch result does not qualify the
remaining per-call 64–128-KiB gaps or replace the full aligned comparison.
The original 54-row aligned failure record remains intact. No default release
has been promoted and native NCCL fallback remains forbidden.
