# Raw-data AG at 128 KiB

Two four-L4 hosts, FP32 AllGather, fixed 1-GiB allocations, 32–256 KiB, both layouts,
20 warmups/30 timed calls, three counterbalanced repetitions. Values are medians of
maximum-rank per-call means including GPU and internal send retirement. The fixed
unpinned host-polling library is the control; no per-size best variant curve.

At exactly 128 KiB, the candidate reuses UCCL's nonpacket serial gather with MSCCL++
Data|Flag|Sync. The raw shard writes its standard remote-output offset. Incoming
signal acquisition and all local peer readiness/barriers remain; final completion
still waits the exact outgoing ticket after output copying. Alternating private
banks are retained. The second candidate also uses exact-primary context reuse.

Both candidates pass 234 mixed ABI calls/rank, including delayed ranks, 128-MiB
boundary coverage, nonalignment, in-place layouts, stream alternation, allocation
recycling and two communicator lifetimes. All 18 focused jobs pass full-output,
exact route and zero-native-fallback checks.

| Variant | AG 128 KiB out(us) | in(us) |
| --- | ---: | ---: |
| LL control |35.319|35.604|
| Raw data/signal |32.367|32.134|
| Raw + primary-context reuse |32.430|32.581|

The raw path is promising against the previously measured original reference near
33 us, but it has not yet been combined with all pending changes or completed full
paired qualification. The last full paired candidate still has 12/126 slower rows.
All sizes/repetitions, including unaffected controls, are retained in the CSV/manifest.
