# Per-call preparation and full-retirement comparison

**The goal is not complete.** The 27-job comparison finishes successfully but
54/126 latency rows are slower (AG 21, AR 11, RS 22). This is separate from the
passing unchanged nccl-tests batch-throughput diagnostic. No failed row is
removed. All original UCCL algorithms/binaries are preserved and SHA-verified.

Two four-L4 hosts, FP32 SUM/copy. For every size from 1 KiB to 1 GiB, input data
are regenerated, pointers remain fixed, preparation and full-output verification
are outside timing, and per-call maximum-rank latency is averaged. Counts match
the original reference commands: AG 20 warmups/30 iterations, AR 20/10, RS 10/10.
Three rounds counterbalance original-reference and candidate out/in order.
Reference layouts and process counts remain those originally recorded (AG four
processes/two GPUs each; AR/RS eight processes/one GPU each). Both candidate
layouts are compared with the recorded reference layout; that limitation remains
explicit. This is not a best-per-size envelope or a final performance admission.

The candidate uses NCCL entry points, followed by uccl_lite_nccl_synchronize,
a non-finalizing diagnostic extension that waits for GPU work and internal send
retirement. Ordinary nccl-tests does not use this extension. The reference
client retains full verification, including the original generation pattern.
Six small smoke jobs and the 144-call mixed ABI pass. Candidate runs require
native-library absence and exact per-rank/per-size route counts.

The initial smoke staging failed because executable permissions were not copied;
copy2 fixed it. The first paired candidate initialization rejected reference-only
environment variables as a stale identity. The retry separates reference and
candidate environments without weakening identity checks. Both failed attempts
are retained. A later client-only rebuild accidentally tried to rebuild the
backend without its external include configuration and failed; the corrected
Make invocation marked measured library prerequisites old and verified that
library SHA-256 values did not change.

## Completion-path overhead screen

The full-retirement helper's existing synchronization path prints a drain line
inside timing. The quiet experiment removes only that line; correctness, route
and native-library audit output remains intact. The inline experiment instead
uses the existing packetDrain path inside the collective kernel, avoiding a host
round trip and a second drain-kernel launch. Neither removes send retirement.
The inline variant passes the 144-call ABI and all nine unchanged full nccl-tests
jobs with zero slower batch diagnostic medians.

An 18-job small-AG screen uses the same fixed 1-GiB allocation, full retirement,
20 warmups and 30 iterations per size, three repetitions, counterbalanced variant
order, both layouts, and unchanged full-output checks. This screen alone is not
full-range admission. Median out/in latencies in microseconds:

| Variant | 1 KiB | 32 KiB | 128 KiB |
| --- | --- | --- | --- |
| Original synchronization helper | 33.212 / 31.873 | 38.235 / 37.690 | 47.982 / 48.024 |
| Quiet drain | 28.957 / 29.044 | 35.781 / 35.620 | 45.205 / 45.713 |
| Inline packet retirement | 21.052 / 20.656 | 28.108 / 27.762 | 37.930 / 37.714 |

Logging explains part, but not all, of the original discrepancy. Inline retirement
improves small-message latency, yet 32–128 KiB still needs work. Next candidates
must retain full retirement and pass both protocols. A useful bounded hypothesis
is posting the existing MSCCL++ FIFO TriggerSync immediately after the final IB
put, overlapping its completion with local fan-out/output, then waiting on the
same flushDonePos/ticket before return. It must not flush ahead of later writes.
No native NCCL fallback is permitted. Some exploratory compilation overlapped
this campaign; final qualification must use quiet, repeated paired runs.
