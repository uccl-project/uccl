# Small-message fusion and CPU progress waiting policy

Two four-L4 hosts, FP32 AG and SUM AR, fixed 1-GiB allocations, both layouts,
20 warmups and AG 30 / AR 10 timed calls, three counterbalanced repetitions.
Independent cases cover AG 64–256 KiB, AR 16–256 KiB, AG 256 KiB–4 MiB and AR 1–4 MiB.
Values are medians of maximum-rank per-call means with full retirement. The fixed
single64 library is the control; no per-size best variant curve or native fallback.

All five candidates pass 216 mixed ABI calls/rank. The two CPU polling variants also
pass 60 direct host-gather component calls/rank with owned/borrowed completion events.
All 66 focused jobs pass full-output, exact route and native-absence checks.

Fusing owned-output stores with existing fan-out loads gives no consistent gain.
Its combination with coalesced flush and ready flags is only modestly better and
still misses AG 128 KiB. AR barrier coalescing is much slower and is rejected.
Inspection also confirms MSCCL++ DeviceSyncer already has a one-CTA fast path, so
substituting a CTA barrier alone is not expected to remove a grid synchronization.

The host polling policy replaces idle condition-variable sleep with an atomic queue
doorbell and CPU pause polling, and uses pause rather than yield while progressing
work. Queue/active/failure state retains its original mutex; shutdown publishes an
atomic stop request and joins the worker. All GPU flags, reader acknowledgements,
transport retirement and the caller's completion wait remain. The policy consumes
two busy progress CPU threads per host in addition to existing proxies. An independent
variant pins only those threads to CPUs 0/1; caller affinity is unchanged.

Unpinned polling improves several host-path medians. At 4-MiB AG, all 180 timed samples
(both layouts and all repetitions) have maximum 223.095 us; control has one sample
above 500 us (max 1160.386), pinned polling also has one (max 911.084). These are empirical
tail diagnostics, not proof that all jitter is removed. No sample is excluded and
admission still uses the original median-of-run-means rule. The unpinned version is
running nine unchanged full nccl-tests sweeps next; final full paired admission remains.

| Case | Variant | Bytes | Layout | Median(us) |
| --- | --- | ---: | --- | ---: |
| ag_small | uccl_graph_single64 | 131072 | out | 34.774 |
| ag_small | uccl_graph_single64 | 131072 | in | 34.874 |
| ag_small | uccl_ag_fused_own | 131072 | out | 34.852 |
| ag_small | uccl_ag_fused_own | 131072 | in | 35.114 |
| ag_small | uccl_ag_fused_ready | 131072 | out | 34.134 |
| ag_small | uccl_ag_fused_ready | 131072 | in | 34.676 |
| ar_small | uccl_graph_single64 | 262144 | out | 74.821 |
| ar_small | uccl_graph_single64 | 262144 | in | 72.183 |
| ar_small | uccl_ar_compact_barriers | 262144 | out | 99.556 |
| ar_small | uccl_ar_compact_barriers | 262144 | in | 97.143 |
| ag_host | uccl_graph_single64 | 262144 | out | 41.512 |
| ag_host | uccl_graph_single64 | 262144 | in | 40.620 |
| ag_host | uccl_graph_single64 | 4194304 | out | 197.750 |
| ag_host | uccl_graph_single64 | 4194304 | in | 196.243 |
| ag_host | uccl_host_spin | 262144 | out | 39.282 |
| ag_host | uccl_host_spin | 262144 | in | 39.752 |
| ag_host | uccl_host_spin | 4194304 | out | 197.201 |
| ag_host | uccl_host_spin | 4194304 | in | 195.688 |
| ag_host | uccl_host_spin_pin | 262144 | out | 41.856 |
| ag_host | uccl_host_spin_pin | 262144 | in | 39.873 |
| ag_host | uccl_host_spin_pin | 4194304 | out | 196.791 |
| ag_host | uccl_host_spin_pin | 4194304 | in | 196.843 |
