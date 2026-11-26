# Medium gather consumer count screen

Two hosts with four NVIDIA L4s each, FP32 copy for AG and FP32 SUM for AR,
one rank/GPU, both layouts, fixed1-GiB allocations, three counterbalanced rounds.
Warmups/timed calls are AG20/30 and AR20/10. Each value is the median of three
maximum-rank per-call means including GPU and internal send retirement. The
control is the fixed primary-context/event-query library. This screen compares
candidates to that control, not the original README baseline. All raw samples,
including tails and slower points, remain. No per-size winner envelope is used.

Only consumer CTA count at8–32MiB host gather changes to24 or32. Bulk keeps16;
producer/control roles, GPU readiness gates, ACKs and transport retirement are
unchanged. Cooperative occupancy bounds both grids; optional trace storage covers
the larger one. Both candidates pass60 standalone component calls/rank over two
lifetimes and234 mixed ABI calls/rank. All36 performance jobs pass full-output,
exact route and native-absence checks. The ordinary Makefile policy is unchanged.

The24-consumer candidate is selected for full original-reference qualification.
It improves16-MiB AG in-place from662.533 to649.498us and both32-MiB layouts.
Its16-MiB out-of-place result is slower:664.469 versus659.997us (+0.678%).
The32-consumer alternative has less consistent medium-AG results. This selection
is a hypothesis for full validation, not a zero-regression or release claim.

| Operation | Bytes | Layout | Control (us) | 24 consumers (us) | 32 consumers (us) |
| --- | ---: | --- | ---: | ---: | ---: |
| ag | 4194304 | out | 198.057 | 197.218 | 197.568 |
| ag | 4194304 | in | 196.886 | 197.331 | 196.341 |
| ag | 8388608 | out | 368.128 | 345.566 | 348.290 |
| ag | 8388608 | in | 375.400 | 346.696 | 379.948 |
| ag | 16777216 | out | 659.997 | 664.469 | 649.489 |
| ag | 16777216 | in | 662.533 | 649.498 | 673.262 |
| ag | 33554432 | out | 1289.695 | 1254.537 | 1283.666 |
| ag | 33554432 | in | 1291.679 | 1253.744 | 1286.362 |
| ag | 67108864 | out | 2487.247 | 2487.111 | 2488.009 |
| ag | 67108864 | in | 2521.896 | 2515.665 | 2514.323 |
| ar | 4194304 | out | 460.518 | 455.870 | 458.493 |
| ar | 4194304 | in | 464.390 | 462.433 | 458.928 |
| ar | 8388608 | out | 845.840 | 845.650 | 848.325 |
| ar | 8388608 | in | 844.744 | 844.463 | 844.334 |
| ar | 16777216 | out | 1565.552 | 1550.127 | 1555.123 |
| ar | 16777216 | in | 1557.711 | 1544.901 | 1545.443 |
| ar | 33554432 | out | 3015.426 | 2969.858 | 2974.623 |
| ar | 33554432 | in | 2991.649 | 2959.620 | 2954.819 |
