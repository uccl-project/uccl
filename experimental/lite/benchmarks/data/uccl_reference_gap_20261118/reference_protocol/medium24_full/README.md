# Medium24 consumer full comparison

This experimental result does not by itself promote the default build. This fixed
library completes all nine unchanged
nccl-tests sweeps with 126/126 batch medians no slower than the original UCCL reference.
The aligned per-call campaign completes all 27 jobs with full output verification and
candidate route/native-absence checks. **1/126 rows are slower**:
AG 0, AR 0, RS 1.

Hardware: two hosts, four NVIDIA L4 GPUs each, mlx5_0/mlx5_1. Dtype: FP32 copy for AG,
FP32 SUM for AR/RS. Baseline: the SHA-verified original README UCCL binaries and original
process layouts (AG four processes/two GPUs each; AR/RS eight processes). Candidate:
eight processes/one GPU each, both out-of-place and in-place. Each candidate layout
is compared to the original recorded reference layout. No native NCCL implementation
is loaded or executed by candidates.

All 21 powers of two from 1 KiB through 1 GiB are included. Three paired rounds alternate
reference/out/in ordering. Warmups/timed calls are AG20/30, AR20/10, RS10/10. Each point
is the median of three maximum-rank per-call means, with fixed pointers, matching data
preparation, and full GPU plus internal send retirement. Full-output checks and MPI
barriers are outside timing. No per-size winner envelope or tolerance relaxation is used.
The diagnostic synchronize extension is used for this protocol; ordinary nccl-tests
remains unchanged. The campaign had no concurrent compilation or candidate GPU tests. Both hosts had no active nsys sessions before and after the campaign. Earlier
negative results and all samples remain available. No timing criterion was changed.

Candidate SHA-256: `b0ba7b498fddcd64d5f1788da56c3e581dfe1e3b5b90b4f9f0c8ec3239639398`. Original failure data (54/126, 25/126 and12/126 rows
for the earlier candidate) remain intact. This result is neither a release promotion
nor final zero-regression admission. The exact tested source snapshots are included here. The candidate adds24 consumer CTAs for8–32MiB host gather, keeping16 for bulk.
Producer/control roles, ACKs and transport retirement are retained, together with
the previously tested exact-primary-context reuse and event-query completion. The
raw-log auditor independently reconstructed every result and rejected four corrupted
log/manifest fixtures.

![Full paired latency ratios](paired_ratios.png)

| Operation | Bytes | Layout | Original (us) | Candidate (us) | Ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| rs | 16777216 | out | 911.773 | 912.103 | 1.000362 |
