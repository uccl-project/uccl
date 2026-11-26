# Serial AR retirement overlap and 512-KiB alternating banks

Two four-L4 hosts, FP32 AG and SUM AR/RS, fixed 1-GiB allocations, 32–512 KiB,
both layouts, 20 warmups and AG 30 / AR 10 / RS 10 timed calls, three counterbalanced
repetitions. Values are medians of maximum-rank per-call means with full retirement.
The control is one fixed FIFO-publication + streamed-RS 512 library. No best-per-size
curve or native NCCL fallback is used.

Serial AR's only outgoing IB write requests retirement early using the existing
MSCCL++ Data|Flag|Sync trigger (or packet Data|Sync), then waits for that exact ticket
after local fan-out and output copying. Input/reduction order and all reader barriers
remain unchanged. The separate 512-KiB variant uses allocated alternating medium banks,
omitting single-bank reuse coordination while retaining in-kernel send retirement.
Both independent candidates and their near-wrap combination pass 198 mixed ABI calls/rank;
the combined near-wrap check confirms 16 resets over eight ranks and two lifetimes.
All 54 screen jobs pass full-output, exact route and native-absence checks.

| Variant | AR 128 KiB out/in(us) | AR 256 KiB out/in(us) | AR 512 KiB out/in(us) |
| --- | --- | --- | --- |
| uccl_packet_rs_combo | 56.009 / 56.343 | 74.913 / 73.308 | 117.315 / 115.959 |
| uccl_ar_overlap_combo | 52.692 / 54.446 | 74.922 / 72.285 | 117.259 / 116.790 |
| uccl_ar512_banks | 56.258 / 55.729 | 76.185 / 72.530 | 100.679 / 100.758 |

The serial overlap improves 128 KiB but does not close the 256-KiB out-of-place gap.
Alternating banks substantially improve 512 KiB. Full-range performance of their
combined library is still required; prior full paired failures remain authoritative.
