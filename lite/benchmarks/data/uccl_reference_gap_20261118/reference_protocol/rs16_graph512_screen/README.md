# 512-CTA16MiB graph reduction screen

Two four-L4 hosts, FP32 SUM RS/AR, fixed1-GiB allocations, both layouts,
three counterbalanced repetitions. Warmups/timed calls are RS10/10 and AR20/10.
Each value is the median of three maximum-rank per-call means including GPU and
internal send retirement. The fixed medium24 candidate is the control. No native
fallback, sample trimming or per-size winner envelope is used.

Only16MiB scatter-graph reduction grids change to512 CTAs with256 threads. At the
2MiB shard, each thread handles one float4. CE copies, graph order, signals,
per-element arithmetic and retirement remain. ABI234, including the delayed16MiB
peer, and graphQueue48 with full outputs/guards and actual address reuse pass.
All24 focused performance jobs pass correctness and route/native checks.

The targeted RS16MiB out-of-place result improves from912.745 to911.646us (0.120%),
while in-place slows from910.335 to911.871us (0.169%). RS8MiB out-of-place is13.9%
slower in this screen despite its kernel path being unchanged; this observation
is retained without assigning a cause or removing samples. Results are mixed.
One fixed512-CTA candidate proceeds to full original-reference qualification to
judge every size/layout. No performance admission or default promotion is made.

| Operation | Bytes | Layout | Control (us) | Candidate (us) | Ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| rs | 8388608 | out | 492.276 | 560.633 | 1.138858 |
| rs | 8388608 | in | 489.543 | 501.352 | 1.024121 |
| rs | 16777216 | out | 912.745 | 911.646 | 0.998796 |
| rs | 16777216 | in | 910.335 | 911.871 | 1.001687 |
| rs | 33554432 | out | 1721.643 | 1732.385 | 1.006240 |
| rs | 33554432 | in | 1731.201 | 1727.121 | 0.997643 |
| rs | 67108864 | out | 3385.958 | 3365.017 | 0.993815 |
| rs | 67108864 | in | 3385.761 | 3381.821 | 0.998836 |
| ar | 8388608 | out | 849.464 | 846.440 | 0.996441 |
| ar | 8388608 | in | 846.094 | 842.585 | 0.995853 |
| ar | 16777216 | out | 1554.945 | 1555.391 | 1.000287 |
| ar | 16777216 | in | 1547.399 | 1553.451 | 1.003911 |
| ar | 33554432 | out | 2979.762 | 2986.011 | 1.002097 |
| ar | 33554432 | in | 2970.523 | 2973.927 | 1.001146 |
