# Tuned NCCL control for two 4×L4 nodes (2026-11-11)

This record updates only the NCCL control used in the three main two-node
figures. The UCCL Lite algorithms and their size cutoffs are unchanged.
Eight FP32 ranks run on `mibura-sky-test-02` and `l40` with four GPUs and two
IB rails per node. AllReduce is in place; AllGather and ReduceScatter are
out of place. Both sides use slowest-rank host enqueue+completion timing,
20 warmups, three alternating paired rounds of ten calls, and complete
output checks on every invocation. [hardware and software identity](../multinode_serial_20261111/hardware_software.txt) records the
matching executable/library hashes, topology, and software versions.

## Explicit NCCL protocol policy

All sizes use NCCL 2.28.7 with
`NCCL_SOCKET_IFNAME=ibp55s0f0` and `NCCL_IB_HCA=mlx5_0,mlx5_1`.
`NCCL_ALGO` and `NCCL_PROTO` are unset except at the following complete
bytes **per rank**, where the control sets `NCCL_ALGO=Ring` and
`NCCL_PROTO=Simple`:

| Collective | RING+SIMPLE sizes |
| --- | --- |
| AllReduce | 256 KiB, 512 KiB, 1 MiB |
| AllGather | 256 KiB, 512 KiB |
| ReduceScatter | 128 KiB, 256 KiB, 512 KiB |

The seven original spike points and one additional unstable ReduceScatter
point were freshly rerun with the same UCCL binary and NCCL control binary.
At every targeted size, the new UCCL and NCCL values come from the **same
three alternating paired rounds**. All other 55 points come from the
[complete 20-warmup paired sweep](../multinode_serial_20261111/README.md)
on the same hosts and binaries. `tuned_summary.csv` has 63 rows and records
the NCCL configuration and raw-log source for every point. It is the only
CSV used for the refreshed figures. The default-NCCL raw sweep remains
unchanged in the linked archive.
This is the explicit RING+SIMPLE fix for the observed LL spike, not a claim
that it is the globally fastest NCCL protocol at every shape. One-run
LL128 diagnostics in the original archive are not promoted into this
fully checked paired baseline.

## Targeted paired results

Values are microseconds, median of three run means. The UCCL path for each
size is recorded in `targeted_summary.csv`.

| Full bytes/rank | Collective | UCCL Lite | Tuned NCCL |
| ---: | --- | ---: | ---: |
| 256 KiB | AllReduce | 74.33 | 135.80 |
| 512 KiB | AllReduce | 114.76 | 155.51 |
| 1 MiB | AllReduce | 206.24 | 247.63 |
| 256 KiB | AllGather | 43.38 | 89.68 |
| 512 KiB | AllGather | 59.79 | 100.96 |
| 128 KiB | ReduceScatter | 36.81 | 77.95 |
| 256 KiB | ReduceScatter | 47.53 | 81.97 |
| 512 KiB | ReduceScatter | 66.86 | 92.23 |

All eight rechecks passed complete output validation. The AllReduce 1-MiB
three-run ranges overlap (UCCL 194.7–220.8 µs; NCCL 204.8–452.7 µs),
so its median difference is not a stable gain claim. `paired_tuned/`
contains all 48 raw logs; `targeted_samples.csv` contains the 480 timed
calls. `tuning/` confirms `RING + SIMPLE`, channels `0..1`, at every
targeted size in separate debug runs. Debug-run times are not used in the
figure.

Run the targeted rechecks from `experimental/lite` after staging the same
binaries on both nodes, with a new `OUTPUT_DIR` for each rerun:

```bash
HOSTS='10.10.55.2:4,10.10.55.1:4' SOCKET_IFNAME=ibp55s0f0 \
HCA_DEVICES=mlx5_0,mlx5_1 IBV_MODE=host-no-atomic \
MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun \
BUILD_DIR=/home/zhongjie/uccl-zhongjie-dev/experimental/lite/build/multinode_trial \
NCCL_BASELINE_BIN=/home/zhongjie/uccl-zhongjie-dev/experimental/lite/build/uccl_lite_nccl_2x4l4_baseline \
NCCL_LIB=/home/zhongjie/third_party/nccl/build/lib \
OUTPUT_DIR=/home/zhongjie/uccl-zhongjie-dev/experimental/lite/benchmarks/data/nccl_tuned_2x4l4_20261111/paired_tuned \
bash scripts/run_multinode_tuned_nccl.sh
```

Rebuild `targeted_samples.csv`, `targeted_summary.csv`, and
`tuned_summary.csv` with:

```bash
python3 scripts/summarize_multinode_tuned_nccl.py \
  benchmarks/data/multinode_serial_20261111/summary.csv \
  benchmarks/data/nccl_tuned_2x4l4_20261111/paired_tuned \
  benchmarks/data/nccl_tuned_2x4l4_20261111
python3 scripts/plot_optimized_collectives.py
```

The figures retain the existing eight-L4 blue/orange, two-panel style and
show exactly one selected UCCL Lite path against this documented tuned NCCL
policy. The earlier [protocol diagnosis](../nccl_spike_2x4l4_20261111/README.md)
and the full default-NCCL sweep remain available for comparison.
