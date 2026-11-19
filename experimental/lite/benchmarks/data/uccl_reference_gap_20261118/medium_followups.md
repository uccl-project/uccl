# Medium-size follow-ups

The zero-regression goal remains unmet. The best complete single-library
matrix remains the 16-consumer candidate with 22/126 slower diagnostic rows.
The following experiments are independent fixed builds, not a combined curve.
All use two four-L4 hosts, FP32, unchanged nccl-tests, three full-range sweeps
per tested operation, both layouts, 20 warmups and 30 timed iterations. Timing
is maximum-rank host enqueue/completion batch means, with three-run medians.
The original UCCL harness has different pointer/process/timing conventions;
these comparisons remain diagnostic and do not establish matched admission.

| Fixed experiment | 8-MiB AG out/in (us) | 16-MiB AG out/in (us) | 1-GiB AG out/in (ms) |
| --- | --- | --- | --- |
| Cooperative through 16 MiB, 32 blocks | 357.13 / 356.48 | 730.66 / 731.19 | 38.120 / 38.130 |
| Combined warp flag load | 356.03 / 355.11 | 724.45 / 722.77 | 40.927 / 41.558 |
| Cooperative 64 blocks | 356.11 / 355.52 | 731.41 / 729.64 | 38.091 / 38.094 |
| Local-first consumption across packets | 358.12 / 357.09 | 725.70 / 724.28 | 38.147 / 38.109 |
| Fresh original UCCL | 355.37 | 672.00 | 41.922 |

The extended cooperative path prepares 1-MiB packet capacity and handles at
most two source packets for complete output up to 16 MiB. Larger extents retain
the measured persistent path. The local-first experiment keeps every packet
immutable until both local and remote reads finish; it does not release ACKs
after local reads alone. Its alternate loop is guarded, preserving the original
packet-major loop in builds without that experiment.

Each cooperative build passes 60 component calls/rank over two lifetimes,
including a new 16-MiB-minus-128-byte tail, offsets, mixed per-rank layouts,
streams and reallocation. Each also passes the 144-call/rank mixed ABI and all
three complete AG sweeps. Every rank/size routes UCCL; native NCCL is absent
and fallback counts are zero. No variant meets the complete AG target, and
none is admitted. Combining the flag loads globally also slows bulk AG versus
the 38-ms persistent control, so it is not retained as a global optimization.

The independent RS experiment extends the existing alternating medium banks
from 8 to 16 MiB for RS only. All other operations keep their prior bank
selection. It preserves the same two-generation reuse protection; it does not
omit reader synchronization. Its 144-call mixed ABI and three full RS sweeps
pass. At 16 MiB it measures 908.51/910.18 us versus 913.60 us reference, improving
the prior approximately 925-us graph path. At 1 GiB it remains slower:
54.218/53.960 ms versus 53.693 ms reference. This RS-only screen is not a new
full-matrix admission and does not resolve other RS rows.

Raw logs, full curves, build commands and binary hashes are retained under
ar512_persistent/uccl_coop16m*, ar512_persistent/uccl_rs_medium16/ and their
matching build/ directories. No negative measurements were removed.
