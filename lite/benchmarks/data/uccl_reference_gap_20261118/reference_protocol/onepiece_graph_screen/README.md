# Single graph piece through 32/64 MiB

Two four-L4 hosts, FP32 SUM AR/RS, fixed 1-GiB allocations, 16–64 MiB,
both layouts, 20 warmups/10 timed calls, three counterbalanced repetitions.
Values are medians of maximum-rank per-call means including GPU and internal
send retirement. The fixed control combines descriptor/binding caches, 64-MiB
alternating banks and 8-MiB remote-first RS. No per-size winner curve or native fallback.

The variants raise the single-piece graph ceiling from 16 MiB to 32 or 64 MiB.
This removes the forced-two-piece minimum at those sizes; local synchronization,
reduction order, actual caller pointers and final outgoing retirement remain.
Both variants pass 216 mixed ABI calls/rank and 48 queued calls/rank with fresh input,
guards, offset variation and allocation recycling. Descriptor reuse/pending cases
and allocation-ID invalidation on actual VA reuse are observed on every rank.
All 36 performance jobs pass full-output, exact route and native-absence checks.

The 64-MiB ceiling gives RS 32-MiB out/in 1.730/1.727 ms versus 1.840/1.827 ms control,
and RS 64-MiB 3.356/3.357 ms versus 3.454/3.450 ms. Raw repetitions and AR results are
retained below and in the CSV; this is a fixed-library comparison, not a synthesized
best-per-size curve. The goal is still unmet until a full original-reference paired
comparison passes. The selected candidate is now running nine unchanged full
nccl-tests sweeps followed by 27 full-range paired jobs. No release promotion.

| Variant | Operation | Bytes | Layout | Median(us) |
| --- | --- | ---: | --- | ---: |
| uccl_graph_banks_combo | rs | 16777216 | out | 918.155 |
| uccl_graph_banks_combo | rs | 16777216 | in | 916.836 |
| uccl_graph_banks_combo | rs | 33554432 | out | 1839.816 |
| uccl_graph_banks_combo | rs | 33554432 | in | 1826.909 |
| uccl_graph_banks_combo | rs | 67108864 | out | 3454.406 |
| uccl_graph_banks_combo | rs | 67108864 | in | 3450.241 |
| uccl_graph_single32 | rs | 16777216 | out | 918.944 |
| uccl_graph_single32 | rs | 16777216 | in | 947.239 |
| uccl_graph_single32 | rs | 33554432 | out | 1739.188 |
| uccl_graph_single32 | rs | 33554432 | in | 1730.036 |
| uccl_graph_single32 | rs | 67108864 | out | 3451.520 |
| uccl_graph_single32 | rs | 67108864 | in | 3454.497 |
| uccl_graph_single64 | rs | 16777216 | out | 913.405 |
| uccl_graph_single64 | rs | 16777216 | in | 916.868 |
| uccl_graph_single64 | rs | 33554432 | out | 1730.327 |
| uccl_graph_single64 | rs | 33554432 | in | 1727.248 |
| uccl_graph_single64 | rs | 67108864 | out | 3356.355 |
| uccl_graph_single64 | rs | 67108864 | in | 3356.737 |
| uccl_graph_banks_combo | ar | 16777216 | out | 1571.799 |
| uccl_graph_banks_combo | ar | 16777216 | in | 1575.515 |
| uccl_graph_banks_combo | ar | 33554432 | out | 3105.492 |
| uccl_graph_banks_combo | ar | 33554432 | in | 3094.098 |
| uccl_graph_banks_combo | ar | 67108864 | out | 5936.955 |
| uccl_graph_banks_combo | ar | 67108864 | in | 5925.291 |
| uccl_graph_single32 | ar | 16777216 | out | 1567.051 |
| uccl_graph_single32 | ar | 16777216 | in | 1568.287 |
| uccl_graph_single32 | ar | 33554432 | out | 3009.871 |
| uccl_graph_single32 | ar | 33554432 | in | 2992.604 |
| uccl_graph_single32 | ar | 67108864 | out | 5938.081 |
| uccl_graph_single32 | ar | 67108864 | in | 5936.781 |
| uccl_graph_single64 | ar | 16777216 | out | 1574.375 |
| uccl_graph_single64 | ar | 16777216 | in | 1564.035 |
| uccl_graph_single64 | ar | 33554432 | out | 3007.659 |
| uccl_graph_single64 | ar | 33554432 | in | 2998.285 |
| uccl_graph_single64 | ar | 67108864 | out | 5859.003 |
| uccl_graph_single64 | ar | 67108864 | in | 5866.064 |
