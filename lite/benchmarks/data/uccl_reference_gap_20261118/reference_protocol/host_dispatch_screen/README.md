# Fixed algorithm selection and driver graph launch: not selected

Two four-L4 hosts, FP32 copy for AG and FP32 SUM for RS/AR, fixed1-GiB allocations,
both layouts, three counterbalanced repetitions. Warmups/timed calls are AG20/30,
RS10/10 and AR20/10. Each value is the median of three maximum-rank per-call means
including GPU and internal send retirement. The fixed medium24 library is the
control; no native fallback, sample trimming or per-size winner envelope.

The fixed-dispatch candidate retains the same immutable registered MSCCL++
NativeAlgorithm objects, initialization, context cache and execute method, while
avoiding per-call request/string/map selection. All call guards remain. The other
candidate uses cuGraphLaunch on the existing graph executable and stream; installed
CUDA headers define both graph-executable types with CUgraphExec_st pointers.
Both candidates passABI234 and graphQueue48 with fresh data, full guards and actual
address reuse. All54 performance jobs pass output, exact route and native absence.

Neither candidate improves the targeted16MiB RS out-of-place result:913.349us for
fixed dispatch and909.489us for driver graph launch versus907.807us for control.
Both improve its in-place result and some other points, but are not selected for
this remaining per-point gate. All gains and slower points are retained below.
The full medium24 +0.03616% RS16MiB failure remains unchanged.

| Operation | Bytes | Layout | Control (us) | Fixed selection (us) | Driver graph (us) |
| --- | ---: | --- | ---: | ---: | ---: |
| rs | 8388608 | out | 529.065 | 513.402 | 521.430 |
| rs | 8388608 | in | 497.280 | 491.110 | 501.818 |
| rs | 16777216 | out | 907.807 | 913.349 | 909.489 |
| rs | 16777216 | in | 911.605 | 909.714 | 907.966 |
| rs | 33554432 | out | 1794.360 | 1733.337 | 1722.010 |
| rs | 33554432 | in | 1728.410 | 1729.517 | 1733.465 |
| rs | 67108864 | out | 3383.633 | 3370.171 | 3372.138 |
| rs | 67108864 | in | 3367.875 | 3366.138 | 3386.421 |
| ar | 8388608 | out | 846.720 | 842.319 | 842.676 |
| ar | 8388608 | in | 845.230 | 843.796 | 843.460 |
| ar | 16777216 | out | 1554.803 | 1557.676 | 1557.601 |
| ar | 16777216 | in | 1551.749 | 1546.765 | 1559.072 |
| ar | 33554432 | out | 2982.410 | 2987.328 | 2977.968 |
| ar | 33554432 | in | 2972.157 | 2970.215 | 2969.799 |
| ag | 1024 | out | 17.730 | 17.548 | 17.601 |
| ag | 1024 | in | 18.333 | 17.659 | 17.982 |
| ag | 2048 | out | 18.314 | 17.607 | 17.570 |
| ag | 2048 | in | 17.940 | 17.776 | 17.831 |
| ag | 4096 | out | 18.264 | 17.962 | 17.968 |
| ag | 4096 | in | 18.125 | 17.944 | 17.803 |
| ag | 8192 | out | 19.847 | 19.855 | 19.797 |
| ag | 8192 | in | 19.637 | 19.589 | 19.572 |
| ag | 16384 | out | 23.835 | 22.917 | 23.003 |
| ag | 16384 | in | 23.104 | 23.435 | 23.010 |
| ag | 32768 | out | 25.500 | 24.505 | 25.324 |
| ag | 32768 | in | 24.993 | 25.143 | 24.996 |
| ag | 65536 | out | 28.033 | 27.562 | 28.705 |
| ag | 65536 | in | 27.730 | 27.788 | 27.815 |
| ag | 131072 | out | 32.006 | 31.549 | 32.278 |
| ag | 131072 | in | 31.995 | 32.837 | 32.198 |
| ag | 262144 | out | 39.316 | 39.515 | 39.207 |
| ag | 262144 | in | 39.340 | 39.186 | 39.066 |
