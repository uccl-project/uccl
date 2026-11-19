# AG barrier and early-owned-output experiments

Two four-L4 hosts, FP32 AG and SUM AR, fixed 1-GiB allocations, 32–256 KiB,
both layouts, 20 warmups and AG 30 / AR 10 timed calls, three counterbalanced repetitions.
Times are medians of maximum-rank per-call means with full retirement. The fixed
single64 library is the control. All four candidates pass 216 mixed ABI calls/rank
and all 60 focused jobs pass full-output, exact route and zero-native-fallback checks.
No per-size best variant curve is used.

Coalescing the peer-copy CTA barriers substantially worsens AG 128 KiB and is rejected,
as are the combinations carrying that change. Early owned-output copying alone gives
only a modest improvement and still misses the original reference. All negative data
are preserved. Next tests fuse owned-output stores with existing loads, without
coalescing AG barriers; neither those tests nor this screen establish full admission.

| Variant | Operation / bytes | Layout | Median(us) |
| --- | --- | --- | ---: |
| uccl_graph_single64 | AG 128 KiB | out | 35.389 |
| uccl_graph_single64 | AG 128 KiB | in | 35.144 |
| uccl_ag_compact_barriers | AG 128 KiB | out | 54.625 |
| uccl_ag_compact_barriers | AG 128 KiB | in | 54.908 |
| uccl_ag_early_own | AG 128 KiB | out | 34.708 |
| uccl_ag_early_own | AG 128 KiB | in | 34.701 |
| uccl_ag_tail_combo | AG 128 KiB | out | 53.346 |
| uccl_ag_tail_combo | AG 128 KiB | in | 52.396 |
| uccl_ag_tail_ready | AG 128 KiB | out | 53.649 |
| uccl_ag_tail_ready | AG 128 KiB | in | 55.283 |
