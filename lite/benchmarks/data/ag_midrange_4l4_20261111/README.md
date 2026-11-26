# Single-node 4×L4 AllGather midrange optimization (2026-11-11)

Both `mibura-sky-test-02` (owner) and `l40` (`mibura-sky-test-01`, test node) ran the exact same source SHA and NCCL 2.28.7 library SHA; their driver versions differ and their results are kept separate. The archived FP32 figure in `docs/peer-ag-rs.md` uses only the l40 paired sweep. The current main AllGather figure uses a fresh owner-host FP32 sweep that retains this small/medium implementation and selects the per-source host-packet router from 16 MiB. See [that record](../fp32_packet_ag_per_source_4l4_20261111/README.md) and `hardware_software.txt` for topology, versions and hashes.

FP32 out-of-place AllGather, four peer-accessible L4 GPUs in a 2+2 NUMA layout. Each source GPU launches one CTA. That CTA copies its input shard into each rank-ordered output slice, one destination at a time; a CTA barrier separates destinations. The destination order rotates by source rank to spread writes. This avoids the earlier concurrent multi-peer fan-out collapse and removes 16 peer-copy submissions and cross-device completion waits from the small-message path. The default candidate applies through 2 MiB complete output bytes per GPU; above that the measured direct peer-copy path is selected. The host-staged CPU router is retained only as a detailed candidate.

Command on both hosts: `peer_ag_rs --router-ag -b 1024 -e 1073741824 -w 20 -n 10 -r 3`. The same harness rotates UCCL candidates and NCCL; every warmup and timed call checks all output elements. The selected CSV chooses the fastest correct non-CPU-router UCCL median per size without consulting NCCL. The table and archived FP32 SVG use slowest-GPU per-call CUDA-event latency, median of three run means. Host-clock launch-to-completion samples are separately retained in `*_samples.csv` and are not mixed with event values.

| Full output bytes/GPU | l40 UCCL path | l40 UCCL (µs) | l40 NCCL (µs) |
| ---: | --- | ---: | ---: |
| 1 KiB | push-serial | 13.95 | 26.49 |
| 2 KiB | push-serial | 14.12 | 26.12 |
| 4 KiB | push-serial | 13.99 | 26.88 |
| 8 KiB | push-serial | 13.85 | 27.04 |
| 16 KiB | push-serial | 14.33 | 28.17 |
| 32 KiB | push-serial | 14.80 | 29.47 |
| 64 KiB | push-serial | 15.85 | 32.36 |
| 128 KiB | push-serial | 23.70 | 38.90 |
| 256 KiB | push-serial | 28.96 | 52.73 |
| 512 KiB | push-serial | 42.12 | 76.67 |
| 1 MiB | push-serial | 72.88 | 99.92 |
| 2 MiB | push-serial | 130.18 | 144.32 |
| 4 MiB | peer | 239.43 | 234.39 |
| 8 MiB | peer | 404.88 | 415.49 |
| 16 MiB | peer | 742.05 | 786.88 |
| 32 MiB | peer | 1471.62 | 1534.22 |
| 64 MiB | peer | 2891.27 | 3021.04 |
| 128 MiB | peer | 5709.77 | 5984.66 |
| 256 MiB | peer | 11397.38 | 11938.06 |
| 512 MiB | peer | 22755.99 | 24085.51 |
| 1 GiB | peer | 45487.59 | 48215.26 |

On l40, the selected UCCL path wins at 20 of 21 sizes. Its only loss is 4 MiB: 239.43 vs 234.39 µs. At 64 KiB, sequential push is 15.85 vs 32.36 µs; at 1 MiB, 72.88 vs 99.92 µs. The owner host independently reports 16.56 vs 33.54 µs and 72.36 vs 100.21 µs at those sizes. See `owner_selected.csv` and `l40_selected.csv` for every point.

## Bottleneck and negative candidates

At 64 KiB, Nsight Systems shows the old source-side path issues three P2P copies per GPU per call. Across the profiled calls, those copies average 1.43 µs each, while the paired path is about 54 µs; the submission and completion schedule dominates this size. The single-CTA push kernel averages 6.2 µs in the profile. This is diagnostic evidence, not a decomposition of the precise end-to-end critical path. Raw reports are `profile_before.nsys-rep` and `profile_push.nsys-rep`.

Extending the direct peer-read kernel above 32 KiB is a negative result (`pull_extended.log`). Vectorized peer reads improve 16 KiB but regress above it (`pull_vec_trial.log`). The prior concurrent fan-out kernel was also negative (see the earlier optimization record). Sequential push loses at 4 MiB and larger (`push_serial_crossover.log`), so the default cutoff is 2 MiB. The 128/512/1024-thread trials are in `threads_*.log`; 256 threads are fastest across most of the target range, with 128 threads used at exactly 64 KiB.

## Artifacts and validation

- `owner_full.log` and `l40_full.log`: raw paired sweeps, including all UCCL candidates, NCCL and per-call checks. `*_samples.csv`: 2,430 timed calls per host; `*_candidates.csv` and `*_selected.csv`: all candidate and selected medians.
- `owner_tail_1008.log` and `l40_tail_1008.log`: non-power-of-two shard tail passes; `owner_rs_regression.log` and `l40_rs_regression.log`: default ReduceScatter path passes.
- `final_build.log`, `l40_build.log`, `hardware_software.txt`, and the two Nsight reports record build and measurement identity.
- `scripts/summarize_ag_midrange.py` validates every size, repetition, call count and selected path. `scripts/plot_optimized_collectives.py` regenerates the archived FP32 SVG in the established eight-L4 style.
