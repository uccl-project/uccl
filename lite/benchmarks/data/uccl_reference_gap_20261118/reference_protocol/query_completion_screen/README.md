# Polling the exact completion event

Two four-L4 hosts, FP32 AllGather, fixed 1-GiB allocations, 4–32 MiB, both layouts,
20 warmups/30 timed calls, three counterbalanced repetitions. Values are medians of
maximum-rank per-call means with full retirement. The fixed prefix128 library is the
control. No per-size winner curve or native fallback is used.

The candidate repeatedly calls cudaEventQuery on the same recorded completion event,
with CPU pause between not-ready results. Success is required; other CUDA errors
propagate. Device/context handling, event boundary, stream dependencies, all GPU work
and CPU transport retirement remain unchanged. No later caller work is included.
It is an uninstrumented build. The caller thread remains runnable while waiting.

ABI 234 and the 48-call queued 128/256-MiB layout transition test pass, including full
output/guards, input variation, allocation reuse and exact 3/29/2 entry/reuse/leave
counts per rank. All 12 focused jobs pass numerical, route and native-absence checks.
In 180 samples per point (both layouts/all repeats), 8-MiB samples above 700 us decrease
from 2 to 0, and 16-MiB samples above 1000 us from 3 to 1. These are descriptive tail counts,
not an acceptance filter or proof of zero jitter; every sample still enters its mean.

| Variant | Bytes | Layout | Median(us) |
| --- | ---: | --- | ---: |
| uccl_rs128_prefix | 4194304 | out | 197.834 |
| uccl_rs128_prefix | 4194304 | in | 197.782 |
| uccl_rs128_prefix | 8388608 | out | 350.176 |
| uccl_rs128_prefix | 8388608 | in | 351.018 |
| uccl_rs128_prefix | 16777216 | out | 658.740 |
| uccl_rs128_prefix | 16777216 | in | 677.100 |
| uccl_rs128_prefix | 33554432 | out | 1283.691 |
| uccl_rs128_prefix | 33554432 | in | 1284.080 |
| uccl_query_completion | 4194304 | out | 195.814 |
| uccl_query_completion | 4194304 | in | 195.938 |
| uccl_query_completion | 8388608 | out | 348.083 |
| uccl_query_completion | 8388608 | in | 347.545 |
| uccl_query_completion | 16777216 | out | 659.613 |
| uccl_query_completion | 16777216 | in | 657.924 |
| uccl_query_completion | 33554432 | out | 1285.869 |
| uccl_query_completion | 33554432 | in | 1296.599 |

The fixed query-completion library has started nine unchanged full nccl-tests jobs
followed by 27 original-reference paired jobs, all operations, 21 sizes 1 KiB–1 GiB and
both layouts. The last completed full audit remains 4/126 slower rows for the prefix
control. No threshold relaxation, sample removal or release promotion is implied.
