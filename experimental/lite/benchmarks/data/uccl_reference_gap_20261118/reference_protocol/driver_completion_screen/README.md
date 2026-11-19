# Driver event-query screen: not selected

Two four-L4 hosts, FP32 copy for AG and FP32 SUM for RS/AR, fixed1-GiB allocations,
both layouts, three counterbalanced repetitions. Warmups/timed calls are AG20/30,
RS10/10 and AR20/10. Each number is the median of three maximum-rank per-call
means including GPU and internal send retirement. The fixed medium24 candidate is
the control; no per-size winner envelope, sample trimming or native fallback.

The candidate calls cuEventQuery on the exact event previously queried through
the runtime API, retaining context guards and transport retirement. Runtime and
driver event handles are compatible according to
[NVIDIA's interoperability documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DRIVER.html).
ABI234, four same/foreign-device/custom-context restoration cases, and48 queued
graph calls with actual address reuse all pass. All36 performance jobs pass
output, route and native-absence checks. Nevertheless,16-MiB RS is slower in both
layouts:914.365/917.692 versus907.919/910.647us (out/in). This flag is not selected;
all results, including gains at other points, remain below and in raw logs.

| Operation | Bytes | Layout | Control (us) | Candidate (us) | Ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| rs | 8388608 | out | 490.661 | 489.417 | 0.997463 |
| rs | 8388608 | in | 496.693 | 488.914 | 0.984340 |
| rs | 16777216 | out | 907.919 | 914.365 | 1.007100 |
| rs | 16777216 | in | 910.647 | 917.692 | 1.007736 |
| rs | 33554432 | out | 1722.286 | 1723.605 | 1.000766 |
| rs | 33554432 | in | 1726.240 | 1728.976 | 1.001585 |
| rs | 67108864 | out | 3371.338 | 3368.146 | 0.999053 |
| rs | 67108864 | in | 3369.281 | 3370.587 | 1.000388 |
| ar | 8388608 | out | 853.876 | 849.140 | 0.994453 |
| ar | 8388608 | in | 847.585 | 842.515 | 0.994019 |
| ar | 16777216 | out | 1557.600 | 1553.536 | 0.997391 |
| ar | 16777216 | in | 1545.499 | 1543.401 | 0.998643 |
| ar | 33554432 | out | 2990.258 | 2984.508 | 0.998077 |
| ar | 33554432 | in | 2963.454 | 2970.332 | 1.002321 |
| ag | 8388608 | out | 346.772 | 350.767 | 1.011521 |
| ag | 8388608 | in | 346.235 | 347.026 | 1.002283 |
| ag | 16777216 | out | 654.839 | 650.582 | 0.993500 |
| ag | 16777216 | in | 677.974 | 687.865 | 1.014589 |
