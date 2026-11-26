# Single-node 4×L4 duplex host-staged AllGather (2026-11-11)

Four L4 GPUs on `mibura-sky-test-02`, two per NUMA node. FP32 out-of-place AllGather; full output bytes per GPU. Each source uploads chunks into two pinned host slots bound to its PCI NUMA node. Every GPU downloads all four source chunks, including its own. CUDA events release each slot only after all four downloads finish. Separate upload and download streams allow D2H of a later chunk to overlap H2D of an earlier chunk.

Accepted run: `./build/uccl_lite_peer_ag_rs --router-ag --router-host-placement local -b 1024 -e 1073741824 -w 20 -n 10 -r 3`. The default chunk is 4 MiB. The same harness rotates CPU router, direct peer (or direct peer read through 32 KiB), and NCCL 2.28.7 with no `NCCL_*` overrides. Every warmup and timed call checks the complete output. Numbers below are host-clock launch-to-completion microseconds, median of three run means. Input fill and output check run outside the timer. `summary.csv` also records separate CUDA-event measurements.

| Full output bytes/GPU | CPU router (µs) | Best direct UCCL Lite (µs) | NCCL (µs) |
| ---: | ---: | ---: | ---: |
| 1 KiB | 72.95 | 20.93 | 31.93 |
| 2 KiB | 71.97 | 20.88 | 32.27 |
| 4 KiB | 73.02 | 20.92 | 31.89 |
| 8 KiB | 72.48 | 21.40 | 33.12 |
| 16 KiB | 73.34 | 32.60 | 33.84 |
| 32 KiB | 74.23 | 47.03 | 34.90 |
| 64 KiB | 77.56 | 59.59 | 38.47 |
| 128 KiB | 70.28 | 61.14 | 44.02 |
| 256 KiB | 71.33 | 61.44 | 57.40 |
| 512 KiB | 79.67 | 89.15 | 81.84 |
| 1 MiB | 102.12 | 122.71 | 105.39 |
| 2 MiB | 145.44 | 162.23 | 148.90 |
| 4 MiB | 239.52 | 247.34 | 240.22 |
| 8 MiB | 445.46 | 413.55 | 419.83 |
| 16 MiB | 860.53 | 749.57 | 788.89 |
| 32 MiB | 1690.99 | 1477.25 | 1535.08 |
| 64 MiB | 3256.68 | 2899.19 | 3035.26 |
| 128 MiB | 6310.70 | 5730.12 | 5998.36 |
| 256 MiB | 11974.58 | 11435.35 | 11925.45 |
| 512 MiB | 23276.17 | 22806.03 | 23773.01 |
| 1 GiB | 45704.93 | 45545.28 | 49137.56 |

At 1 GiB, the final paired result is 45.70 ms router, 45.55 ms direct peer, and 49.14 ms NCCL. Router is faster than NCCL by median at 6 of 21 sizes: 512 KiB–4 MiB and 512 MiB–1 GiB. Three-run ranges overlap at 1 MiB and 4 MiB. Small messages remain slower: 1 KiB router 72.95 µs, NCCL 31.93 µs, direct peer read 20.93 µs.

## Duplex diagnosis

`duplex_overlap.csv` analyzes the CUDA memcpy intervals in three raw Nsight Systems reports. The old CUDA-allocated host slots and local NUMA slots use 1 MiB chunks; the final local slots use 4 MiB chunks. Nsight traces include seven 1 GiB router calls plus tiny correctness transfers. The fraction below is D2H active time during which H2D is also active on the same GPU:

| Placement and chunk | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
| --- | ---: | ---: | ---: | ---: |
| CUDA allocation, 1 MiB | 2.5% | 3.2% | 93.7% | 98.9% |
| Local NUMA, 1 MiB | 93.2% | 95.7% | 94.4% | 94.6% |
| Local NUMA, 4 MiB | 97.0% | 97.2% | 96.9% | 96.7% |

In the final trace, each GPU's memcpy-active rate is 22.0–22.3 GB/s D2H
and 23.5–23.7 GB/s H2D. These are rates during active copies, not PCIe
hardware-counter measurements or end-to-end throughput.

The original CUDA-allocated slots produced asymmetric overlap; NUMA-bound,
registered slots removed it. At the same 1 MiB chunk, the 1 GiB router median
fell from 57.87 to 46.60 ms. The allocation API also changed, so this A/B
does not isolate NUMA placement alone. Tuning the chunk to 4 MiB reduced the
final full-sweep median to 45.70 ms. These are separate runs on the same host;
the final table uses only its paired controls.

`chunk_tuning.csv` records the 1 GiB 3-run chunk ablation. Its best local-NUMA median is 45.48 ms at 4 MiB; 256 KiB chunks regress to 76.53 ms because copy/event overhead dominates.

## Artifacts and checks

- `paired_local_4m_w20.log`: complete raw final run; `samples.csv`: all 2,070 timed paired samples; `summary.csv`: run-level medians and ranges.
- `one_gib_cuda.log`, `one_gib_local.log`, and `chunk_*.log`: placement and chunk ablations, including losses.
- `profile_*.nsys-rep`: raw CUDA timelines; `duplex_overlap.csv`: derived overlap; `scripts/summarize_router_duplex_profile.py` regenerates it from Nsight SQLite exports.
- `hardware_software.txt` and `build.log`: topology, versions, source/binary identity, and build.
- `tail_3chunks.log`: non-power-of-two 3-chunk tail check. `default_peer_smoke.log`, `router_policy.log`, `dispatcher.log`, `router_eager.log`, and `router_graph.log` all pass.

This is an earlier host-staged FP32 candidate. Its direct-peer curves remain in the detailed archive; the main README's current FP32 AllGather uses the per-source packet router from 16 MiB.
