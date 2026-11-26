# AG pair exchange, AR thread geometry and RS block count

Two four-L4 hosts, FP32 AG and SUM AR/RS, fixed 1-GiB allocations, both layouts,
20 warmups and AG 30 / AR 10 / RS 10 timed calls, three counterbalanced repetitions.
AG covers 64–256 KiB; AR 128–512 KiB; RS 4–16 MiB. Values are medians of maximum-rank
per-call means with full retirement. The fixed shared-host-event library is the
control. No per-size winner curve or native NCCL fallback is used.

All five candidates pass 198 mixed ABI calls/rank; all 48 focused jobs pass full-output,
exact route and native-absence checks. AG's two-phase logical-pair exchange halves
cross-NUMA payload but adds a readiness phase and is slower. Both AR 256 thread-count
changes are slower; retain 384 threads. RS 16 blocks gives a small improvement; 32 blocks
is slower. These outcomes do not establish full-range admission. An existing older
AR 256 experiment directory was preserved and the new 256-thread experiment used a
new directory; no old binary or measurement was overwritten.

| Variant | Operation / bytes | Layout | Median(us) |
| --- | --- | --- | ---: |
| uccl_shared_host_event | ag 131072 | out | 34.944 |
| uccl_shared_host_event | ag 131072 | in | 34.751 |
| uccl_ag128_pairs | ag 131072 | out | 35.771 |
| uccl_ag128_pairs | ag 131072 | in | 35.672 |
| uccl_shared_host_event | ar 262144 | out | 74.813 |
| uccl_shared_host_event | ar 262144 | in | 72.609 |
| uccl_ar256_threads512 | ar 262144 | out | 80.377 |
| uccl_ar256_threads512 | ar 262144 | in | 82.377 |
| uccl_ar256_after_flush256 | ar 262144 | out | 78.576 |
| uccl_ar256_after_flush256 | ar 262144 | in | 76.793 |
| uccl_shared_host_event | rs 8388608 | out | 503.304 |
| uccl_shared_host_event | rs 8388608 | in | 503.400 |
| uccl_rs8m_blocks16 | rs 8388608 | out | 502.151 |
| uccl_rs8m_blocks16 | rs 8388608 | in | 499.432 |
| uccl_rs8m_blocks32 | rs 8388608 | out | 506.935 |
| uccl_rs8m_blocks32 | rs 8388608 | in | 506.779 |
