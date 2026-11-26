# Smaller16MiB graph reduction grids: not selected

Two four-L4 hosts, FP32 SUM RS/AR, fixed1-GiB allocations, both layouts,
three counterbalanced repetitions. Warmups/timed calls are RS10/10 and AR20/10.
Each value is the median of three maximum-rank per-call means including GPU and
internal send retirement. The fixed medium24 candidate is the control. No native
fallback, sample trimming or per-size winner envelope is used.

Only reduction grids in16MiB scatter graphs change from256 to64 or128 CTAs;
threads/CTA remain256. Copy engines, graph order, signals, per-element arithmetic,
output ownership and retirement remain. Hybrid AR's private16MiB scatter graph
uses the same change. Both candidates have identical CUDA fatbins and passABI234
with the delayed16MiB peer, plus graphQueue48 with full outputs/guards and actual
address reuse. All36 focused performance jobs pass correctness and route checks.

Neither smaller grid improves16MiB RS: out-of-place is915.028/915.120us versus
912.777us for control; in-place is915.146/911.088us versus907.906us. They are not
selected. All neighboring-size and AR measurements are preserved below and in logs.
The original full medium24 +0.03616% failure remains a failure.

| Operation | Bytes | Layout | Control (us) | 64 CTAs (us) | 128 CTAs (us) |
| --- | ---: | --- | ---: | ---: | ---: |
| rs | 8388608 | out | 500.509 | 505.165 | 497.407 |
| rs | 8388608 | in | 501.155 | 518.552 | 487.766 |
| rs | 16777216 | out | 912.777 | 915.028 | 915.120 |
| rs | 16777216 | in | 907.906 | 915.146 | 911.088 |
| rs | 33554432 | out | 1722.051 | 1746.443 | 1725.899 |
| rs | 33554432 | in | 1730.798 | 1732.541 | 1724.164 |
| rs | 67108864 | out | 3371.607 | 3366.955 | 3363.729 |
| rs | 67108864 | in | 3363.295 | 3372.826 | 3367.222 |
| ar | 8388608 | out | 845.837 | 842.053 | 843.460 |
| ar | 8388608 | in | 848.109 | 847.562 | 840.142 |
| ar | 16777216 | out | 1550.680 | 1563.662 | 1558.185 |
| ar | 16777216 | in | 1548.696 | 1547.127 | 1545.464 |
| ar | 33554432 | out | 2982.743 | 2982.343 | 2973.608 |
| ar | 33554432 | in | 2974.888 | 2967.601 | 2977.002 |
