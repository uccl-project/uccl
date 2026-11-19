# Independent AG readiness and RS 512-KiB experiments

Two four-L4 hosts, FP32 AllGather and FP32 SUM ReduceScatter, fixed 1-GiB
allocations, both layouts, three repetitions with counterbalanced variant order.
AG covers 1–128 KiB with 20 warmups/30 timed calls; RS covers 256 KiB–1 MiB with
20 warmups/10 timed calls. Times are medians of maximum-rank per-call means with
full GPU and internal send retirement. Baseline is the frozen output-overlap/
ordered candidate. No best-per-size envelope and no native NCCL fallback.

Each candidate passes the strengthened 198-call/rank mixed ABI check, adding
32/64/128 KiB to all prior cases with nonaligned pointers, in-place layouts,
stream alternation, allocation recycling, two communicator lifetimes and a delayed
128-KiB AG rank. The separate AG near-wrap build starts packet epochs at
UINT32_MAX-10, passes the same check, and confirms one reset per rank per lifetime.
Both banks, including new dedicated readiness storage, are reset. All 24 screen
jobs pass full-output, exact route and native-absence checks.

AG uses dedicated per-bank flags with one writer per sender, system release/acquire,
and the original CTA barriers. User payload never aliases flags. It improves some
32/64-KiB timings modestly but does not close the 128-KiB reference gap. It is not
selected as the next base; its positive and negative evidence remain available.
RS moves exactly 512 KiB to the existing streamed kernel only when its medium
banks were allocated. Existing alternating-bank and retirement rules are preserved.
Its screen result is promising; full-range fixed-library qualification remains.

| Variant | Operation / bytes | out (us) | in (us) |
| --- | --- | ---: | ---: |
| Control | AG 32 KiB | 26.464 | 26.235 |
| Ready flags | AG 32 KiB | 25.647 | 25.070 |
| Control | AG 64 KiB | 29.326 | 29.790 |
| Ready flags | AG 64 KiB | 29.038 | 28.662 |
| Control | AG 128 KiB | 36.437 | 36.251 |
| Ready flags | AG 128 KiB | 36.061 | 35.819 |
| Control | RS 512 KiB | 79.921 | 79.837 |
| Streamed | RS 512 KiB | 58.840 | 58.469 |

The full zero-regression goal remains unmet. The last full paired candidate still
has 25/126 slower rows; these screens do not replace that required comparison.
