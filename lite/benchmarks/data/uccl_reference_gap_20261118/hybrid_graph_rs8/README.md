# Preserve graph RS inside hybrid AllReduce

**The full zero-regression goal remains unmet.** This fixed follow-up binary is
`468d1a723c004428abd67f3189f861999f5b2d3a4354497e659dfb7e2cc92c47`.

The preceding complete candidate made public 8-MiB RS faster, but using that
same streamed path to produce hybrid AR's private shard made 8-MiB AR slower.
The internal Context::run call now carries explicit hostAllReducePart context.
It retains graph RS for the internal AR producer while keeping the streamed
public RS threshold at 8 MiB. No pointer-identity heuristic or native fallback
is involved. Other experimental host-gather policies are unchanged.

Two four-L4 hosts, FP32 SUM, unchanged nccl-tests, full 1-KiB–1-GiB AR curve,
both layouts, three repetitions, 20 warmups and 30 timed iterations. Timing is
maximum-rank host enqueue/completion batch means, with medians across sweeps.
This is one fixed policy, not a best-per-size curve.

| AR extent | Previous combined out/in | This follow-up out/in | Fresh UCCL |
| --- | --- | --- | --- |
| 8 MiB | 951.89 / 946.27 us | 922.15 / 915.80 us | 890.40 us |
| 1 GiB | 100.581 / 100.542 ms | 100.689 / 100.678 ms | 96.445 ms |

The composition regression is recovered; the UCCL target is still missed.
The fresh original UCCL reference uses different harness conventions, so this
is diagnostic and not matched admission. No regression allowance was added.

The extended mixed ABI passes 126 calls/rank over two lifetimes. All three
AR jobs pass numerical checks and per-rank/per-size strict route and native
library absence checks. AG/RS timings were not rerun for this binary; do not
claim the preceding combined matrix proves a complete gate for this revision.
Exact source snapshots remain in build/uccl_hybrid_graph_rs8/source/, with
hashes, build commands and logs retained here. The first build log records an
unmodified build following a failed edit assertion; it was not measured. The
recorded measured binary was rebuilt with make -B after applying the edit.
