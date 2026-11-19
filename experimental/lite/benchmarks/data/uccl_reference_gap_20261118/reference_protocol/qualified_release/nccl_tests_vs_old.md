# Stock nccl-tests measurements versus old UCCL Lite

The new-library values below are actual, unchanged `all_gather_perf`,
`all_reduce_perf` and `reduce_scatter_perf` results, using the qualified UCCL Lite
replacement `.so`; native NCCL is not loaded or used as fallback. The old values
come from the frozen original UCCL Lite standalone benchmarks, not nccl-tests.
Therefore these are cross-harness measurements, not directly comparable speedups.
In particular, nccl-tests reports batch timing, while the old harness uses
per-call timing with full retirement. The separate aligned per-call campaign,
not this cross-harness table alone, supports the earlier126/126 no-regression claim.

Hardware: two hosts with four NVIDIA L4s each, mlx5_0/mlx5_1. FP32 copy for AG,
FP32 SUM for AR/RS. New: eight MPI ranks, one GPU/rank,20 warmups and30 timed
iterations, median of three sweeps. Old: original process layouts (AG four ranks,
two GPUs/rank; AR/RS eight ranks), original warmup/timing counts and three-run
medians. Every size is a power of two from1KiB through1GiB. Size denotes total AG
output, total RS input or AR buffer. All latencies are microseconds; lower is better.
The same old recorded reference layout is shown against both new layouts.

Measured `.so` SHA-256: `8ad6c2da1808eb85983902e9383138ac6e009d0c228e89406a675d08c6da11a5`.
The rebuilt default has identical loaded code/data; raw metadata hashes are recorded
in the [qualification report](README.md). No data was rerun, trimmed or selected
from different variants to produce these tables. Display values are rounded only;
[CSV](nccl_tests_vs_old.csv) retains full precision.

## AG

| Size | Old standalone (us) | New nccl-tests out (us) | New nccl-tests in (us) |
| --- | ---: | ---: | ---: |
| 1 KiB | 24.31 | 11.26 | 10.58 |
| 2 KiB | 23.97 | 10.83 | 10.64 |
| 4 KiB | 25.36 | 11.03 | 10.88 |
| 8 KiB | 26.79 | 12.43 | 12.36 |
| 16 KiB | 30.82 | 15.88 | 15.69 |
| 32 KiB | 26.85 | 18.47 | 18.18 |
| 64 KiB | 28.78 | 21.11 | 20.95 |
| 128 KiB | 33.19 | 25.62 | 25.25 |
| 256 KiB | 40.73 | 31.05 | 31.12 |
| 512 KiB | 54.24 | 41.98 | 42.24 |
| 1 MiB | 73.87 | 63.19 | 63.08 |
| 2 MiB | 114.32 | 104.98 | 104.75 |
| 4 MiB | 197.17 | 188.70 | 188.47 |
| 8 MiB | 355.37 | 330.60 | 330.93 |
| 16 MiB | 672.00 | 630.25 | 629.92 |
| 32 MiB | 1,308.87 | 1,231.19 | 1,229.19 |
| 64 MiB | 2,689.11 | 2,476.83 | 2,474.51 |
| 128 MiB | 5,236.00 | 4,927.24 | 4,921.01 |
| 256 MiB | 10,474.48 | 9,833.37 | 9,821.03 |
| 512 MiB | 20,950.54 | 19,657.20 | 19,642.40 |
| 1 GiB | 41,921.76 | 39,229.90 | 39,254.90 |

## AR

| Size | Old standalone (us) | New nccl-tests out (us) | New nccl-tests in (us) |
| --- | ---: | ---: | ---: |
| 1 KiB | 40.18 | 16.43 | 15.25 |
| 2 KiB | 38.95 | 16.04 | 15.70 |
| 4 KiB | 39.00 | 17.38 | 16.89 |
| 8 KiB | 38.61 | 24.08 | 25.12 |
| 16 KiB | 39.01 | 24.57 | 24.14 |
| 32 KiB | 41.18 | 28.67 | 27.82 |
| 64 KiB | 46.07 | 37.42 | 36.28 |
| 128 KiB | 57.55 | 45.67 | 45.18 |
| 256 KiB | 74.55 | 64.23 | 63.96 |
| 512 KiB | 115.06 | 97.67 | 97.30 |
| 1 MiB | 180.02 | 142.51 | 142.74 |
| 2 MiB | 306.73 | 242.85 | 242.98 |
| 4 MiB | 495.50 | 442.57 | 441.86 |
| 8 MiB | 890.40 | 842.33 | 841.98 |
| 16 MiB | 1,683.37 | 1,544.30 | 1,543.69 |
| 32 MiB | 3,189.44 | 2,957.67 | 2,956.07 |
| 64 MiB | 6,162.14 | 5,843.74 | 5,837.31 |
| 128 MiB | 12,138.95 | 11,615.40 | 11,608.90 |
| 256 MiB | 24,177.45 | 23,211.50 | 23,223.30 |
| 512 MiB | 48,304.72 | 46,374.90 | 46,388.90 |
| 1 GiB | 96,444.89 | 92,649.20 | 92,650.80 |

## RS

| Size | Old standalone (us) | New nccl-tests out (us) | New nccl-tests in (us) |
| --- | ---: | ---: | ---: |
| 1 KiB | 31.04 | 17.49 | 15.42 |
| 2 KiB | 30.79 | 15.23 | 15.20 |
| 4 KiB | 30.88 | 15.28 | 15.25 |
| 8 KiB | 30.48 | 15.80 | 15.58 |
| 16 KiB | 30.83 | 16.57 | 16.26 |
| 32 KiB | 31.87 | 18.88 | 18.57 |
| 64 KiB | 34.72 | 24.43 | 24.29 |
| 128 KiB | 39.27 | 29.71 | 29.41 |
| 256 KiB | 49.18 | 40.58 | 40.43 |
| 512 KiB | 66.66 | 50.99 | 50.90 |
| 1 MiB | 111.72 | 79.23 | 79.27 |
| 2 MiB | 184.10 | 138.72 | 138.53 |
| 4 MiB | 291.30 | 254.82 | 254.58 |
| 8 MiB | 494.40 | 478.37 | 478.64 |
| 16 MiB | 913.60 | 904.23 | 904.27 |
| 32 MiB | 1,824.85 | 1,712.68 | 1,708.75 |
| 64 MiB | 3,453.23 | 3,336.15 | 3,334.35 |
| 128 MiB | 6,751.64 | 6,680.32 | 6,676.93 |
| 256 MiB | 13,471.18 | 13,377.60 | 13,354.90 |
| 512 MiB | 26,947.02 | 26,717.60 | 26,710.70 |
| 1 GiB | 53,692.54 | 53,445.70 | 53,334.30 |

## Source records

- New raw stock-test logs/manifests: `build/uccl_rs16_graph512/{ag,ar,rs}_measure/`.
- Old reference records: `build/uccl_readme_reference/fresh/`.
- Independent batch auditor: `build/uccl_rs16_graph512/batch_raw_log_audit.json`.
- [Aligned per-call comparison](../graph512_full/comparison.csv) and
  [protocol/report](../graph512_full/README.md).
