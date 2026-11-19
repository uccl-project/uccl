# Exact primary-context reuse with event-query completion

Two hosts with four NVIDIA L4s each, FP32 copy for AG and FP32 SUM for RS,
one rank/GPU, both layouts, fixed 1-GiB allocations, three counterbalanced rounds.
Warmups/timed calls are RS 10/10 and AG 20/30. Each number is the median of
three maximum-rank per-call means including GPU and internal send retirement.
The control is the fixed event-query candidate; this is a focused screen, not
comparison against original README UCCL and not full admission. No per-size
winner selection, sample trimming, or native NCCL fallback is used.

The candidate adds only UCCL_LITE_SYNC_PRIMARY_CONTEXT. It skips redundant
runtime device switching only when the exact current context equals the primary
context observed after cudaSetDevice. Algorithms and retirement are unchanged.
234 mixed ABI calls/rank, four device/context cases and 48 queued layout calls/rank
pass, as do all output, route and native-absence checks in 36 performance jobs.
All slower points remain in the table and original logs. The candidate improves
both 256-KiB RS layouts, while 128-MiB RS and several AG points have mixed results.
A complete original-reference qualification is required before selection.

Candidate SHA-256: `0adc90877552af4d3585b08591b950aaedc38bb8849835649e7be57be9213f53`.

| Operation | Bytes | Layout | Control (us) | Candidate (us) | Ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| rs | 131072 | out | 36.830 | 36.730 | 0.997307 |
| rs | 131072 | in | 37.448 | 35.591 | 0.950406 |
| rs | 262144 | out | 49.388 | 48.685 | 0.985756 |
| rs | 262144 | in | 49.855 | 48.619 | 0.975198 |
| rs | 524288 | out | 58.591 | 58.906 | 1.005378 |
| rs | 524288 | in | 59.122 | 58.097 | 0.982661 |
| rs | 67108864 | out | 3384.342 | 3370.495 | 0.995909 |
| rs | 67108864 | in | 3375.180 | 3370.536 | 0.998624 |
| rs | 134217728 | out | 6716.959 | 6709.118 | 0.998833 |
| rs | 134217728 | in | 6693.635 | 6706.086 | 1.001860 |
| rs | 268435456 | out | 13428.297 | 13458.245 | 1.002230 |
| rs | 268435456 | in | 13417.067 | 13396.180 | 0.998443 |
| ag | 4194304 | out | 195.708 | 198.396 | 1.013736 |
| ag | 4194304 | in | 196.719 | 198.488 | 1.008993 |
| ag | 8388608 | out | 347.859 | 349.332 | 1.004232 |
| ag | 8388608 | in | 348.477 | 349.797 | 1.003788 |
| ag | 16777216 | out | 662.796 | 659.817 | 0.995506 |
| ag | 16777216 | in | 675.937 | 660.007 | 0.976432 |
