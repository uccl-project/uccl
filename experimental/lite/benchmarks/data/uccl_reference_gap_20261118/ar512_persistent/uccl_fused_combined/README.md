# Fused combined candidate

**Zero-regression remains unmet: 7/126 diagnostic rows are slower.**
All operations use one binary: `5063907450acd9fa2b1611213708cb6680806cba6619e7c4e400e96dd3381479`.

Two four-L4 hosts, FP32 SUM AR/RS and bitwise AG, 21 powers of two from
1 KiB to 1 GiB, both layouts, three unchanged nccl-tests sweeps per operation,
20 warmups and 30 timed iterations. Timing includes host enqueue/completion
and uses the maximum rank. Each row is the three-run median. This is one fixed
policy, not an envelope of per-size minima. Original UCCL harness process,
pointer-reuse and timing conventions differ, so this is not matched admission.

| Operation | Slower rows / 42 | 1-GiB candidate out/in (ms) | UCCL (ms) |
| --- | ---: | --- | --- |
| AG | 5 | 38.114 / 38.139 | 41.922 |
| AR | 2 | 91.643 / 91.608 | 96.445 |
| RS | 0 | 53.491 / 53.374 | 53.693 |

![Diagnostic ratios](diagnostic_ratios.png)

The fixed policy uses one-packet cooperative AG through 8 MiB, the persistent
four-uploader/16-consumer/one-publisher path above it, alternating RS banks
through 32 MiB, and hybrid AR from 1 MiB. The 512-KiB direct-input AR path is
retained. RS waits for the remote sum before its final own-half addition, then
writes the result once. Arithmetic order remains (own + local partial) + remote.
All forwarding has finished before that wait, and only the opposite half is
sent over IB. In-place input is read before overwrite; future pieces are disjoint.

The component passes 60 calls/rank and the mixed ABI passes 144 calls/rank over
two lifetimes. All nine timing jobs pass numerical, exact per-size/per-rank
routing, native-library-absent and zero-fallback checks. No default release
has been promoted. No positive diagnostic ratio is rounded into a pass.

Exact source snapshots remain under build/uccl_fused_combined/source/; their
hashes, build commands, raw logs, complete curves and this generator are retained.
Negative experiments are preserved in the surrounding records.
