# Warp-publisher candidate

**Zero-regression remains unmet: 4/126 diagnostic rows are slower.**
All operations use one binary: `ac4c463e187f9b0fae6f0d818ee07453259eadb3bcc683d98af880f9907dc708`.

Two four-L4 hosts, FP32 SUM AR/RS and bitwise AG, 21 powers of two from
1 KiB to 1 GiB, both layouts, three unchanged nccl-tests sweeps per operation,
20 warmups and 30 timed iterations. Timing includes host enqueue/completion
and uses the maximum rank. Each row is the three-run median. This is one fixed
policy, not an envelope of per-size minima. Original UCCL harness process,
pointer-reuse and timing conventions differ, so this is not matched admission.

| Operation | Slower rows / 42 | 1-GiB candidate out/in (ms) | UCCL (ms) |
| --- | ---: | --- | --- |
| AG | 4 | 38.175 / 38.186 | 41.922 |
| AR | 0 | 91.603 / 91.612 | 96.445 |
| RS | 0 | 53.481 / 53.420 | 53.693 |

![Diagnostic ratios](diagnostic_ratios.png)

The fixed policy uses one-packet cooperative AG through 8 MiB, the persistent
four-uploader/16-consumer/one-warp-publisher path above it, alternating RS banks
through 32 MiB, and hybrid AR from 1 MiB. The 512-KiB direct-input AR path is
retained. Exactly 256-KiB AR uses 384 threads. RS waits for the remote sum before its final own-half addition, then
writes the result once. Arithmetic order remains (own + local partial) + remote.
All forwarding has finished before that wait, and only the opposite half is
sent over IB. In-place input is read before overwrite; future pieces are disjoint.

The mixed ABI passes 144 calls/rank over
two lifetimes. All nine timing jobs pass numerical, exact per-size/per-rank
routing, native-library-absent and zero-fallback checks. No default release
has been promoted. No positive diagnostic ratio is rounded into a pass.

Build commands, raw logs, complete curves and this generator are retained.
Negative experiments are preserved in the surrounding records.
