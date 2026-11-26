# Rejected small AG geometry and fan-out experiments

Two hosts with four NVIDIA L4 GPUs each, FP32 AllGather, fixed 1-GiB allocations,
1–128 KiB, both layouts, 20 warmups and 30 timed calls, three repetitions with
counterbalanced variant order. Timing includes full GPU and internal send retirement;
rows are medians of maximum-rank per-call means, never a best-per-size envelope.
Baseline is the fixed early-flush-ticket candidate, not native NCCL.

Both candidates pass 144 mixed ABI calls/rank and all 18 screen jobs pass full-output,
exact route and native-library-absent checks. Neither candidate is retained.

| Variant | 32 KiB out/in (us) | 64 KiB out/in (us) | 128 KiB out/in (us) |
| --- | --- | --- | --- |
| Early-ticket control | 26.246 / 26.235 | 29.380 / 29.801 | 37.119 / 36.532 |
| Parallel fan-out | 26.094 / 25.879 | 30.185 / 30.203 | 37.462 / 37.079 |
| 256 threads | 26.184 / 25.977 | 30.202 / 29.710 | 38.671 / 38.426 |

The parallel fan-out experiment distributes distinct peer/quad writes across CTA
threads and retains synchronization after all peer writes. The thread experiment
changes only packet AG 32–128 KiB launch geometry. Both are compile-time experiments
and remain unselected. These negative results do not alter the original acceptance
scope: all three collectives, 1 KiB–1 GiB, both layouts, zero native NCCL fallback.
