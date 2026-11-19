# RS 128-MiB alternating banks within existing storage

Two four-L4 hosts, FP32 SUM AR/RS, fixed 1-GiB allocations, both layouts,
20 warmups/10 timed calls, three counterbalanced repetitions. RS covers 64–256 MiB;
AR covers 64–128 MiB. Values are medians of maximum-rank per-call means including
GPU and internal send retirement. The fixed raw-AG128/AR256-bank/host-polling library
is the control. Its two busy progress threads/host remain part of the configuration.
No per-size best variant curve or native NCCL fallback is used.

Two 452-MiB layouts fit in the already allocated 1-GiB large data region. Eligible
128-MiB RS calls alternate banks without enlarging the allocation or relaxing 32-bit
offset bounds. Entry after an ordinary large-layout operation uses the original
reuseLarge coordination. Normal large-layout paths invalidate this mode and retain
their original guards. The first entry therefore still has a transition cost; these
steady-state measurements exclude warmups as before. Capacity below 904 MiB retains
the original UCCL layout. Communicator capacity is fixed, so the 128-MiB graph key
cannot alternate between normal and prefix layouts within that communicator.

Both candidates pass 234 mixed ABI calls/rank. The baseline and candidates also pass
48 queued 128/256-MiB RS/AR calls/rank with new GPU inputs, full output/guards, offset
changes and allocation recycling. Every candidate rank executes exactly 3 entries,
29 reuses and 2 departures, demonstrating layout transitions rather than only steady
calls. The primary-context variant additionally retains the previously checked
context-switching behavior. All 36 performance jobs pass full-output, exact route and
native-absence checks. Full combined original-reference qualification is still required.

| Variant | Operation | Bytes | Layout | Median(us) |
| --- | --- | ---: | --- | ---: |
| uccl_ar256_banks8 | rs | 134217728 | out | 6740.765 |
| uccl_ar256_banks8 | rs | 134217728 | in | 6770.617 |
| uccl_rs128_prefix | rs | 134217728 | out | 6720.965 |
| uccl_rs128_prefix | rs | 134217728 | in | 6717.986 |
| uccl_rs128_prefix_primary | rs | 134217728 | out | 6708.605 |
| uccl_rs128_prefix_primary | rs | 134217728 | in | 6713.237 |
| uccl_ar256_banks8 | ar | 134217728 | out | 11681.415 |
| uccl_ar256_banks8 | ar | 134217728 | in | 11665.614 |
| uccl_rs128_prefix | ar | 134217728 | out | 11663.456 |
| uccl_rs128_prefix | ar | 134217728 | in | 11642.179 |
| uccl_rs128_prefix_primary | ar | 134217728 | out | 11645.841 |
| uccl_rs128_prefix_primary | ar | 134217728 | in | 11658.881 |

The full zero-regression goal remains unfulfilled until a fresh full paired matrix passes. No release promotion is implied.

The selected non-primary candidate subsequently passes all nine unchanged full
nccl-tests sweeps, all 126 batch medians <= original UCCL. Full-output, exact route
and native-absence checks pass. A fresh 27-job full paired comparison is running;
this batch diagnostic and focused screen do not establish final admission.
