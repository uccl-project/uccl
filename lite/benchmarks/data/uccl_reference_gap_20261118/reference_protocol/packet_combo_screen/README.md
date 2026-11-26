# FIFO publication + streamed RS combination and AG follow-ups

Two four-L4 hosts, FP32 AG and SUM AR/RS, fixed 1-GiB allocations,32–512 KiB,
both layouts,20 warmups and AG30/AR10/RS10 timed calls,three counterbalanced
repetitions. Values are medians of maximum-rank per-call means with full retirement.
Each candidate is one fixed library; no per-size best variant curve is used.
All three pass198 mixed ABI calls/rank and all54 screen jobs pass full-output,
exact route and native-absence checks. Full-range admission remains required.

The base combines FIFO publication and exactly512-KiB streamed RS. The geometry
variant changes only128-KiB AG to1024 threads. It worsens AG and is rejected.
The coalesced variant combines final AG data+flush into one MSCCL++ trigger; it
modestly improves AG128 KiB but remains above the original UCCL reference near33 us.
The new RS512 KiB improvement survives both combinations. Other rows and all
repetitions remain in the saved CSV/manifest, including AR variations.

| Variant | AG128 KiB out/in(us) | AR64 KiB out/in(us) | RS512 KiB out/in(us) |
| --- | --- | --- | --- |
| Combined base |35.043/35.227|43.728/43.233|58.403/58.850|
| AG128 threads1024 |38.215/38.502|43.719/43.295|58.507/58.707|
| Coalesced AG flush |34.126/34.148|43.248/43.983|58.031/58.923|

The latest full original-reference paired count remains25/126 slower rows until
another complete library is qualified. No release promotion or native fallback.
