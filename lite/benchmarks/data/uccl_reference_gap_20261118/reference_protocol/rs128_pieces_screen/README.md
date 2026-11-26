# RS 128-MiB graph piece count: rejected

Two four-L4 hosts, FP32 SUM AR/RS, fixed 1-GiB allocations, both layouts,
20 warmups/10 timed calls, three counterbalanced repetitions. RS covers 64–256 MiB;
AR covers 64–128 MiB. Medians use maximum-rank per-call means with full retirement.
The fixed host-polling library is the control. Both candidates pass 234 mixed ABI
calls/rank; all 36 jobs pass full-output, exact route and zero-native-fallback checks.
Only 128 MiB changes piece count, from two to three or four. The existing tail handling,
arithmetic order and final retirement remain. Both alternatives are slower and rejected.
No per-size winner curve or final admission is claimed.

| Variant | Operation | Bytes | Layout | Median(us) |
| --- | --- | ---: | --- | ---: |
| uccl_host_spin | rs | 134217728 | out | 6734.517 |
| uccl_host_spin | rs | 134217728 | in | 6778.544 |
| uccl_rs128_three | rs | 134217728 | out | 6884.655 |
| uccl_rs128_three | rs | 134217728 | in | 6879.947 |
| uccl_rs128_four | rs | 134217728 | out | 6930.884 |
| uccl_rs128_four | rs | 134217728 | in | 6906.615 |
| uccl_host_spin | ar | 134217728 | out | 11679.759 |
| uccl_host_spin | ar | 134217728 | in | 11737.818 |
| uccl_rs128_three | ar | 134217728 | out | 11815.084 |
| uccl_rs128_three | ar | 134217728 | in | 11834.823 |
| uccl_rs128_four | ar | 134217728 | out | 11849.513 |
| uccl_rs128_four | ar | 134217728 | in | 11853.212 |
