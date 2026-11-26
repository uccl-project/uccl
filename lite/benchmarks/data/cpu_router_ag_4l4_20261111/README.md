# Single-node 4×L4 host-staged AllGather (2026-11-11)

Four NVIDIA L4 GPUs on `mibura-sky-test-02`, 2+2 NUMA; FP32, out-of-place.
The `cpu-router` candidate stages every source shard in pinned host memory and every GPU reads all four shards back, including its own. Two slots per source and 1 MiB chunks overlap GPU-to-host uploads with host-to-GPU fan-out. CUDA events guard slot reuse; the input-ready boundary is outside the timed call.

Run: `./build/uccl_lite_peer_ag_rs --router-ag -b 1024 -e 1073741824 -w 20 -n 10 -r 3`. The same harness alternates router, direct peer, small-message peer pull, and NCCL 2.28.7 with no `NCCL_*` environment overrides. Every warmup and timed call checks all output elements. The table is the median of three run means of host-clock launch-to-completion latency; fill and verification are outside the timed call. `summary.csv` also includes CUDA-event values; do not compare one timer with the other.
The legacy `# selected` log marker covers direct peer/pull only; this record
compares the router separately.

| Full output bytes/rank | CPU router (µs) | Best direct UCCL Lite (µs) | NCCL (µs) |
| ---: | ---: | ---: | ---: |
| 1 KiB | 71.96 | 20.79 | 31.62 |
| 2 KiB | 70.97 | 20.65 | 32.14 |
| 4 KiB | 72.12 | 20.54 | 32.06 |
| 8 KiB | 71.80 | 20.92 | 32.69 |
| 16 KiB | 71.97 | 32.70 | 33.46 |
| 32 KiB | 73.57 | 46.65 | 34.65 |
| 64 KiB | 75.96 | 60.24 | 37.92 |
| 128 KiB | 69.94 | 60.71 | 44.18 |
| 256 KiB | 69.93 | 61.71 | 57.64 |
| 512 KiB | 79.61 | 89.58 | 81.98 |
| 1 MiB | 102.49 | 123.57 | 105.03 |
| 2 MiB | 145.42 | 161.20 | 148.87 |
| 4 MiB | 237.52 | 246.92 | 240.05 |
| 8 MiB | 497.64 | 412.55 | 421.29 |
| 16 MiB | 988.84 | 745.42 | 786.81 |
| 32 MiB | 1900.18 | 1480.14 | 1545.99 |
| 64 MiB | 3697.45 | 2902.21 | 3054.93 |
| 128 MiB | 7294.83 | 5735.18 | 6032.13 |
| 256 MiB | 14481.01 | 11448.24 | 11989.78 |
| 512 MiB | 28839.58 | 22823.24 | 23877.50 |
| 1 GiB | 57622.29 | 45572.66 | 49241.24 |

Router is faster than NCCL by median at 512 KiB–4 MiB (4 of 21 sizes), though the 1 and 4 MiB three-run ranges overlap. At 1 GiB, router is 57.62 ms versus direct peer 45.57 ms and NCCL 49.24 ms. Small-message router latency is about 72 µs; the direct peer-read path is about 21 µs at 1 KiB.

Files: `paired_1m_w20.log` (accepted full raw log), `samples.csv` (all 2,070 timed paired samples), `summary.csv` (three-run medians and ranges), `hardware_software.txt`, `final_build.log`. Other `paired_*` and `chunk_*` logs are the 5-warmup sweep and chunk-size ablations. The 1 KiB point in the first 5-warmup 1 MiB sweep was cold and is not used in the accepted summary; `warmup_1k.log` reproduces the stable point with 20 warmups.

Final-source build, default peer AG/RS smoke, router policy, dispatcher, eager,
and graph checks passed; logs are in this directory. `tail_3chunks.log`
checks a non-power-of-two 3-chunk shard with a 1008-byte tail.

At 1 GiB, the 256 KiB / 1 MiB / 4 MiB / 8 MiB / 16 MiB chunk trials gave router medians of approximately 75.5 / 57.6 / 64.8 / 66.9 / 67.7 ms. The candidate is experimental and is not auto-selected by the existing dispatcher.
