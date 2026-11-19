# Rejected extension of local LL packet AG

Two four-L4 hosts, FP32 AllGather, fixed 1-GiB allocations, 1–128 KiB,
both layouts, 20 warmups and 30 timed calls, three repetitions with
counterbalanced variant order. Timing includes GPU and send retirement;
medians of per-call maximum-rank means, no best-per-size envelope.
Baseline is the fixed early-flush-ticket candidate, not native NCCL.

The candidate extends the existing local LL16 packet AG path from 16 KiB
through 128 KiB. Compile-time checks require the doubled packet payload to fit
existing local packet storage. It passes 144 mixed ABI calls/rank and all
12 screen jobs, including exact routes and native-library-absent checks.
Performance is substantially worse, so this experiment is rejected.

| Variant | 32 KiB out/in (us) | 64 KiB out/in (us) | 128 KiB out/in (us) |
| --- | --- | --- | --- |
| Early-ticket control | 26.403 / 26.154 | 29.863 / 28.870 | 36.435 / 36.285 |
| Extended local packets | 34.058 / 33.326 | 78.699 / 77.952 | 146.508 / 147.509 |

The full zero-regression goal remains unmet. No native NCCL fallback is permitted.
