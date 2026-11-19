# Public16MiB remote-first streamed RS: not selected

Two four-L4 hosts, FP32 SUM RS/AR, fixed1-GiB allocations, both layouts,
three counterbalanced repetitions. Warmups/timed calls are RS10/10 and AR20/10.
Each value is the median of three maximum-rank per-call means including GPU and
internal send retirement. The fixed medium24 candidate is the control. No native
fallback, sample trimming or per-size winner envelope is used.

At public16MiB only, candidates use the existing remote-first streamed ring and
alternating medium banks, with8 or16 CTAs. Hybrid AR's private RS graph remains
unchanged. No additional GPU allocation or new reduction arithmetic is introduced.
Both candidates pass234 mixed ABI calls/rank, now including a delayed peer at the
existing16MiB RS case, and48 queued mixed RS/AR full-output/guard calls with address
reuse. All36 performance jobs pass output, exact route and native-absence checks.

Both streamed variants are slower at16MiB:971–978us versus908–910us for the graph
control. They are rejected and disabled in subsequent candidates. All neighboring
size results and negative measurements remain below and in raw logs. The original
full medium24 RS16MiB +0.03616% failure is not converted into a pass.

| Operation | Bytes | Layout | Control (us) | 8 CTAs (us) | 16 CTAs (us) |
| --- | ---: | --- | ---: | ---: | ---: |
| rs | 8388608 | out | 484.702 | 550.895 | 493.667 |
| rs | 8388608 | in | 511.642 | 507.403 | 496.848 |
| rs | 16777216 | out | 909.831 | 971.933 | 971.115 |
| rs | 16777216 | in | 908.431 | 977.889 | 974.537 |
| rs | 33554432 | out | 1725.494 | 1729.270 | 1721.424 |
| rs | 33554432 | in | 1725.324 | 1723.441 | 1738.485 |
| ar | 8388608 | out | 842.105 | 845.368 | 845.121 |
| ar | 8388608 | in | 848.153 | 839.700 | 845.124 |
| ar | 16777216 | out | 1558.318 | 1552.574 | 1557.600 |
| ar | 16777216 | in | 1547.872 | 1556.037 | 1551.937 |
| ar | 33554432 | out | 2988.523 | 2995.162 | 2980.178 |
| ar | 33554432 | in | 2977.681 | 2971.175 | 2972.087 |
