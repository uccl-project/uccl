# Allocation-ID-first graph binding screen: not selected

Two four-L4 hosts, FP32 SUM RS/AR, fixed1-GiB allocations, both layouts,
three counterbalanced repetitions. Warmups/timed calls are RS10/10 and AR20/10.
Each value is the median of three maximum-rank per-call means including GPU and
internal send retirement. The fixed medium24 candidate is the control; this is
not a full original-README comparison. All samples and slower points are retained.

The candidate queries only the allocation ID for an already eligible graph binding
with the same input pointer. A matching ID retains the eligibility previously
checked for that allocation; changed, unknown or ineligible bindings retain full
attribute validation and rebinding. Unexpected driver errors still propagate.
NVIDIA documents IDs as process-lifetime unique and not reassigned after freeing:
[CUDA pointer attributes](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html).
No input data or numerical results are cached.

ABI234 and48 queued graph calls/rank pass. Every rank actually recycles a virtual
address and records40 binding hits,8 updates and2 recycled-allocation invalidations,
with full output/guards and native absence. All24 focused jobs pass correctness.
The candidate does not consistently improve the remaining16-MiB RS gap: out-of-place
is908.650 versus908.092us; in-place is912.991 versus908.512us. It is not selected
for full qualification or the named/default policy. The guard remains experimental
and disabled in subsequent candidates unless explicitly measured and selected.

| Operation | Bytes | Layout | Control (us) | Candidate (us) | Ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| rs | 8388608 | out | 490.611 | 503.512 | 1.026296 |
| rs | 8388608 | in | 514.798 | 486.825 | 0.945663 |
| rs | 16777216 | out | 908.092 | 908.650 | 1.000615 |
| rs | 16777216 | in | 908.512 | 912.991 | 1.004929 |
| rs | 33554432 | out | 1728.290 | 1728.852 | 1.000325 |
| rs | 33554432 | in | 1727.621 | 1718.910 | 0.994957 |
| rs | 67108864 | out | 3364.341 | 3372.238 | 1.002347 |
| rs | 67108864 | in | 3370.827 | 3370.411 | 0.999877 |
| ar | 8388608 | out | 850.382 | 847.911 | 0.997094 |
| ar | 8388608 | in | 845.490 | 845.726 | 1.000280 |
| ar | 16777216 | out | 1552.446 | 1561.615 | 1.005906 |
| ar | 16777216 | in | 1554.320 | 1544.600 | 0.993746 |
| ar | 33554432 | out | 2978.784 | 2977.566 | 0.999591 |
| ar | 33554432 | in | 2970.419 | 2975.892 | 1.001843 |
