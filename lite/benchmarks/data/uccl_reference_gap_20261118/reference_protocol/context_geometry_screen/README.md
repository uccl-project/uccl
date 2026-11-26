# Primary-context reuse, AG geometry and 128-MiB graph cutover

Two four-L4 hosts, FP32 AG and SUM AR/RS, fixed 1-GiB allocations, both layouts,
20 warmups and AG 30 / AR 10 / RS 10 timed calls, three counterbalanced repetitions.
The fixed unpinned host-polling library is the control, retaining its two busy
progress CPU threads/host. Independent cases cover AG 64–256 KiB, AR 32–256 KiB,
RS 64–256 MiB and AR 64–128 MiB. Values are medians of maximum-rank per-call means
including GPU and internal send retirement. No per-size winner curve or native fallback.

All four builds pass 234 mixed ABI calls/rank, retaining old cases and adding 128 MiB
with a delayed rank. The primary-context variant additionally passes full output,
exact device and context checks for same-primary, another GPU's primary, a custom
context on the same GPU, and returning to the primary. It only skips runtime device
switching when cuCtxGetCurrent equals the exact primary handle previously observed
after cudaSetDevice, never on ordinal equality alone. All 66 performance jobs pass
full-output, exact route and native-absence checks.

Thread-count variants change only 128-KiB AG. The single-piece variant changes only
the 128-MiB graph cutover relative to the existing 64-MiB ceiling; its 16-MiB IB payload
fits MSCCL++'s 32-bit size field. All controls, repetitions and slower results remain
in the saved CSV/logs. This is a focused screen, not final full paired admission.

| Case | Variant | Bytes | Layout | Median(us) |
| --- | --- | ---: | --- | ---: |
| context_ag | uccl_host_spin | 131072 | out | 34.992 |
| context_ag | uccl_host_spin | 131072 | in | 35.303 |
| context_ag | uccl_sync_primary | 131072 | out | 34.894 |
| context_ag | uccl_sync_primary | 131072 | in | 35.073 |
| context_ar | uccl_host_spin | 262144 | out | 75.277 |
| context_ar | uccl_host_spin | 262144 | in | 72.876 |
| context_ar | uccl_sync_primary | 262144 | out | 74.258 |
| context_ar | uccl_sync_primary | 262144 | in | 72.128 |
| geometry_ag | uccl_host_spin | 131072 | out | 35.672 |
| geometry_ag | uccl_host_spin | 131072 | in | 36.581 |
| geometry_ag | uccl_ag128_threads384 | 131072 | out | 36.699 |
| geometry_ag | uccl_ag128_threads384 | 131072 | in | 36.428 |
| geometry_ag | uccl_ag128_threads768 | 131072 | out | 35.545 |
| geometry_ag | uccl_ag128_threads768 | 131072 | in | 35.823 |
| single_rs | uccl_host_spin | 134217728 | out | 6784.514 |
| single_rs | uccl_host_spin | 134217728 | in | 6747.059 |
| single_rs | uccl_graph_single128 | 134217728 | out | 6872.447 |
| single_rs | uccl_graph_single128 | 134217728 | in | 6836.590 |
| single_ar | uccl_host_spin | 134217728 | out | 11666.371 |
| single_ar | uccl_host_spin | 134217728 | in | 11670.448 |
| single_ar | uccl_graph_single128 | 134217728 | out | 11795.340 |
| single_ar | uccl_graph_single128 | 134217728 | in | 11795.950 |
