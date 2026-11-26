# ReduceScatter full-range exploratory screen, November 15

The screen retains negative results and motivates the separate
[repeated direct-medium candidate](../rs_direct_medium_2x4l4_20261115/README.md).
No measured best-variant-per-size envelope is promoted from this screen.

Two four-L4 hosts, mibura-sky-test-02 and mibura-sky-test-01; GPUs 0/1 and 2/3
are NUMA pairs with mlx5_0/mlx5_1 IB rails. Eight MPI ranks, FP32 sum;
N=input bytes/GPU, output N/8. CUDA 12.9, Open MPI 4.1.7rc1, MSCCL++
build-ib-pcieccl. Timing is slowest-rank host enqueue-to-completion, excluding
input fill and complete-output checking. One exploratory round, five warmups
and five timed checked calls per size. `summary.csv` reports means.

The baseline uses the actual tuned README configuration: `--rs-auto 1
--rs-piece-bytes 8388608 -B 96 -T 512`, not the slower default launch. All
cases use the AllReduce-fixed executable except the two opt-in fused cases,
which use the prior current-source experiment binary. Its complete source
patch is archived in
[the preceding current-source record](../ar_latency_2x4l4_20261115/README.md),
against the revision in that record's `environment.txt`. `hashes.txt` identifies
both exact binaries. No NCCL measurements are run.

| Input/GPU | Tuned baseline (µs) | Direct, one piece (µs) |
| ---: | ---: | ---: |
| 8 MiB | 510.69 | 492.63 |
| 16 MiB | 981.86 | 897.59 |

The direct case avoids the input permutation kernel and the extra packed
buffer traffic. It transfers standard-layout immutable input directly with
one piece. The effect is a measured end-to-end result; no isolated pack-time
attribution is claimed. Below this regime, direct ring is slower than the
existing serial/packed path. Extending serial through 8 MiB is also slower.
Serial and direct-medium diagnostics use smaller maximum allocations than
the full-range baseline, so their small-message latency is not an isolated
algorithm comparison; the accepted candidate requires the repeated full-range
comparison with identical maximum extents.

At 1 GiB the tuned baseline is 53.676 ms. Changing to 96×256 or 192×256 gives
53.764/53.550 ms, insufficient exploratory evidence for a launch change.
Increasing the target to 16/32 MiB gives 55.442/60.853 ms, both worse.
Fused GPU forwarding with 4-MiB pieces gives 76.607 ms, also worse; its
8-MiB variant takes 73.273 ms and does not close the gap. These experiments retain explicit
experimental status and are not enabled by this work. The original source's
fused experiment is left untouched.

All 620 timed calls and their warmups pass full-output checks. This is eager
execution only; no graph replay or sanitizer result is claimed. Reproduce
with `bash screen.sh` from lite after staging the binaries at the recorded
paths; the script refuses to overwrite logs. `python3 summarize.py` checks
completeness and regenerates the CSV. Preserve these logs before reruns.
