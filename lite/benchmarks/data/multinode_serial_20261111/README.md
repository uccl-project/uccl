# Two-node serial local collectives (2026-11-11)

Hosts: `mibura-sky-test-02` (repository owner) and `l40` / `mibura-sky-test-01` (test node), four L4 GPUs and two active IB rails per host. GPU 0/1 and `mlx5_0` are on NUMA 0; GPU 2/3 and `mlx5_1` are on NUMA 1. See `hardware_software.txt` for the fresh GPU/NIC topology, driver, CUDA, MPI/UCX, and matching executable/library hashes. Only the owner changed repository files and pushed the branch.

FP32, eight ranks, complete bytes per rank. AllReduce is in place; AllGather and ReduceScatter are out of place. UCCL Lite and NCCL 2.28.7 use the slowest-rank host enqueue+completion time, 20 warmups, three alternating paired rounds of ten timed calls, and complete output checks on every invocation. Input fill and checking are outside timing. NCCL uses `NCCL_SOCKET_IFNAME=ibp55s0f0` and `NCCL_IB_HCA=mlx5_0,mlx5_1`; algorithm and protocol are unset (library defaults). UCCL uses `MSCCLPP_IBV_MODE=host-no-atomic` and the same rail map.

From `experimental/lite`, with the matching staged binary at `BUILD_DIR` on both hosts:

```bash
HOSTS='10.10.55.2:4,10.10.55.1:4' SOCKET_IFNAME=ibp55s0f0 \
HCA_DEVICES=mlx5_0,mlx5_1 IBV_MODE=host-no-atomic \
MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun \
BUILD_DIR=/home/zhongjie/uccl-zhongjie-dev/experimental/lite/build/multinode_trial \
NCCL_BASELINE_BIN=/home/zhongjie/uccl-zhongjie-dev/experimental/lite/build/uccl_lite_nccl_2x4l4_baseline \
NCCL_LIB=/home/zhongjie/third_party/nccl/build/lib \
OUTPUT_DIR=/home/zhongjie/uccl-zhongjie-dev/experimental/lite/benchmarks/data/multinode_serial_20261111/paired_w20 \
AUTO_VARIANTS=1 CHECK_EACH=1 TIMING_MODE=host \
OPERATIONS='allreduce allgather reducescatter' MIN_BYTES=1024 MAX_BYTES=1073741824 \
WARMUPS=20 ITERATIONS=10 REPEATS=3 TIMEOUT_SECONDS=600 \
bash scripts/run_multinode_peer_compare.sh
```

The runner refuses to overwrite existing logs; use a new `OUTPUT_DIR` for a rerun.

## Selected paths

| Operation | Complete bytes/rank | UCCL path |
| --- | --- | --- |
| AllReduce | ≤1 MiB | one-CTA local direct ReduceScatter → IB exchange → local direct AllGather |
| AllReduce | ≥2 MiB | prior local ring ReduceScatter/AllGather |
| AllGather | ≤2 MiB | one-CTA direct local peer writes; IB exchange overlaps the first local broadcast |
| AllGather | ≥4 MiB | prior local ring |
| ReduceScatter | ≤1 MiB | one-CTA direct local peer writes and reduction → IB exchange |
| ReduceScatter | 2–16 MiB | prior GPU-packed local ring |
| ReduceScatter | ≥32 MiB | prior direct local ring |

Each direct path uses separate CUDA-IPC memory channels and GPU semaphores for the three local peers. The IB PortChannel and its proxy remain independent. AllReduce keeps its outbound partial immutable until `ib.flush()`; an early prototype overwrote it and failed correctness (`ar_serial_64k_smoke.log`). The corrected path passed the full sweep.

## Paired results

The following values are milliseconds, median of three per-round means. The full 21-size tables and selected path for each operation are in `summary.csv`; `samples.csv` has all 3,780 timed calls.

| Full bytes/rank | AR UCCL / NCCL | AG UCCL / NCCL | RS UCCL / NCCL |
| ---: | ---: | ---: | ---: |
| 1 KiB | 0.038 / 0.033 | 0.033 / 0.032 | 0.031 / 0.033 |
| 2 KiB | 0.039 / 0.034 | 0.034 / 0.034 | 0.031 / 0.033 |
| 4 KiB | 0.038 / 0.054 | 0.033 / 0.034 | 0.032 / 0.035 |
| 16 KiB | 0.040 / 0.053 | 0.034 / 0.037 | 0.032 / 0.038 |
| 64 KiB | 0.047 / 0.070 | 0.034 / 0.044 | 0.035 / 0.043 |
| 128 KiB | 0.057 / 0.101 | 0.037 / 0.064 | 0.039 / 0.100 |
| 256 KiB | 0.075 / 1.008 | 0.042 / 0.390 | 0.048 / 0.721 |
| 512 KiB | 0.115 / 2.646 | 0.057 / 2.254 | 0.066 / 2.439 |
| 1 MiB | 0.203 / 0.692 | 0.089 / 0.132 | 0.112 / 0.117 |
| 2 MiB | 0.327 / 0.321 | 0.174 / 0.186 | 0.184 / 0.177 |
| 4 MiB | 0.571 / 0.510 | 0.290 / 0.268 | 0.291 / 0.279 |
| 64 MiB | 7.062 / 6.902 | 3.296 / 3.582 | 3.445 / 3.795 |
| 1 GiB | 106.053 / 112.735 | 53.957 / 56.596 | 53.673 / 59.212 |

At 64 KiB, AR/AG/RS take 47.08/34.34/34.59 µs versus NCCL 69.57/44.15/42.61 µs. At 1 KiB, AR and AG still lose (38.40/33.09 versus 32.67/32.41 µs); RS takes 31.06 versus 32.90 µs. Numerically the selected paths are lower at 13/17/17 of 21 sizes against this exact default-NCCL control; that count includes the spike sizes below and is not a stable-gain count. At 1 GiB the retained large-message paths take 106.053/53.957/53.673 ms versus NCCL 112.735/56.596/59.212 ms.

The default NCCL control has repeatable 256–512 KiB spikes and an AllReduce 1-MiB spike. These are recorded, not treated as stable UCCL gains. `control_probe/` tests Ring, LL, LL128, Simple, and Ring+Simple as separate diagnostic NCCL configurations; Simple removes most spikes but is much slower at small sizes. Those probes are one-run, 5-warmup diagnostics and are excluded from `summary.csv`. The later [tuned NCCL control](../nccl_tuned_2x4l4_20261111/README.md) freshly pairs the affected points and supplies the current main figures. This archive remains the complete default-configuration result.

## Raw evidence and checks

- `paired_w20/`: accepted 18 raw paired logs; `samples.csv` and `summary.csv`: validated per-call samples, three-round medians, selected paths and negative points. `paired/`, `samples_default_nccl.csv`, and `summary_default_nccl.csv`: earlier 5-warmup diagnostics.
- `small_recheck/`, `small_samples.csv`, and `small_summary.csv`: separate 1–64 KiB, 20-warmup paired recheck. The accepted full sweep also uses 20 warmups, so no small-size values are spliced from this recheck into the main figures.
- `control_probe/` and `control_probe_summary.csv`: checked NCCL protocol/algorithm ablations; they do not replace the default control.
- `ar_*_b*.log`, `allgather_*_b*.log`, and `rs_*_flatb*.log`: one-run launch-geometry probes. One CTA reduces the proxy-path floor at small sizes but can regress at 256 KiB and above; direct local kernels are selected only through their measured cutoffs.
- `ag_serial_*.log`, `rs_serial_*.log`, `ar_serial_*.log`: checked opt-in direct-path crossover probes. `ar_serial_64k_smoke.log` retains the failed immutable-source prototype; `ar_serial_64k_smoke_fixed.log` and the accepted sweeps verify the correction.
- `tail_checks/`: all three automatic paths pass complete checks at the non-power-of-two 1536-byte and 1003520-byte shapes on both nodes.
- `build_auto_serial.log` records the owner build. Both hosts used the same staged UCCL and NCCL binaries and matching MSCCL++/NCCL libraries; hashes are in `hardware_software.txt`.

Generate summaries with `python3 scripts/summarize_multinode_checked.py benchmarks/data/multinode_serial_20261111/paired_w20 benchmarks/data/multinode_serial_20261111/samples.csv benchmarks/data/multinode_serial_20261111/summary.csv --auto` from `experimental/lite`. Generate the figures with `python3 scripts/plot_optimized_collectives.py`.
