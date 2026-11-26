# AR 256-KiB direct ring with alternating banks

Two four-L4 hosts, FP32 SUM AllReduce, fixed 1-GiB allocations, 128 KiB–1 MiB,
both layouts, 20 warmups/10 timed calls, three counterbalanced repetitions. Values
are medians of maximum-rank per-call means including GPU and internal send retirement.
The control includes the measured raw 128-KiB AG path and host polling. No per-size
winner curve or native fallback is used.

This retest differs from the earlier rejected direct-input AR 256 experiment: it
uses already allocated alternating medium banks rather than single-bank reuse.
The existing direct-input reduction is retained, with in-kernel outgoing retirement.
Independent variants use 2/4/8 blocks only at 256 KiB. All three pass 234 mixed ABI
calls/rank; all 24 performance jobs pass full-output, exact route and native-absence checks.
Eight blocks gives the best repeated result and is a candidate for combination,
not full-range admission. All other rows/repetitions remain preserved.

| Variant | AR 256 KiB out(us) | in(us) |
| --- | ---: | ---: |
| uccl_ag_direct128 | 74.136 | 72.139 |
| uccl_ar256_banks8 | 71.221 | 70.501 |
| uccl_ar256_banks4 | 71.819 | 73.161 |
| uccl_ar256_banks2 | 73.560 | 73.329 |
