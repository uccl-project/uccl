# Shared host-gather completion event and 128-KiB cutover

Two four-L4 hosts, FP32 AllGather and FP32 SUM AllReduce, fixed 1-GiB allocations,
AG 128 KiB–4 MiB and AR 512 KiB–4 MiB, both layouts, 20 warmups, AG 30 / AR 10 timed calls,
three counterbalanced repetitions. Values are medians of maximum-rank per-call means
including full GPU and internal send retirement. The fixed control combines FIFO
publication, streamed RS 512, serial AR retirement overlap and AR 512 alternating banks.
No native fallback and no per-size best variant curve.

The shared-event candidate records the network owner's completion event directly
inside host-gather, removing a duplicate event record. Network synchronization first
waits its latest ordered GPU boundary, then drains CPU transport work without waiting
the same GPU boundary twice. All stream waits remain. The borrowed event outlives the
host runtime; only its owner destroys it. The standalone component alternates owned
and borrowed events, re-records the borrowed event after later caller-stream work,
uses both synchronization paths, and checks all output/guards over 60 calls/rank and
two lifetimes. All three builds pass this component and 198 mixed ABI calls/rank.
All 36 screen jobs pass full-output, exact route and native-absence checks.

The shared-event change has a modest benefit. Lowering the cooperative host AG cutover
to 128 KiB worsens that size and is rejected. All variations and negative results remain
in the manifest/CSV; host AR 4 MiB has visible run-to-run variation, not a basis for a
precise claimed speedup. Full original-reference qualification is still outstanding.

| Variant | AG 128 KiB out/in(us) | AG 256 KiB out/in(us) | AG 2 MiB out/in(us) | AG 4 MiB out/in(us) |
| --- | --- | --- | --- | --- |
| uccl_retirement_combo | 35.485 / 35.054 | 41.842 / 41.954 | 114.987 / 114.164 | 198.621 / 197.549 |
| uccl_shared_host_event | 35.045 / 34.839 | 41.146 / 40.942 | 114.738 / 113.358 | 198.628 / 196.536 |
| uccl_shared_host128 | 36.928 / 36.881 | 40.484 / 41.002 | 113.976 / 113.132 | 198.679 / 197.572 |

The retained shared-event library is now running nine unchanged full nccl-tests
sweeps over all three operations, 1 KiB–1 GiB, both layouts. That batch diagnostic
will not replace the required final per-call paired comparison. No release promotion.

The retained shared-event library subsequently passed all nine unchanged full
nccl-tests sweeps, with all 126 batch medians <= the original UCCL reference.
Full-output, exact route and zero-native-fallback checks pass. The full aligned
per-call gate is still unmet; no default release is promoted.
