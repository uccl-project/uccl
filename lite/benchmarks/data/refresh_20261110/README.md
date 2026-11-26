# November 10 UCCL Lite refresh

This refresh starts from `uccl-lite` commit `61686dc338ca092774d3f4a783f23e74fd94514a`.
`mibura-sky-test-02` is the sole source owner; `l40` resolves to
`mibura-sky-test-01` and is a test node. The same built executables and
libraries were copied to l40 and checked by SHA-256. l40 did not commit or
push. The existing eight-L4 results are separate and unchanged.

## Hardware and software

[Owner topology, peer access, CUDA, MPI/UCX and HCAs](hardware_local.txt) ·
[l40 topology and versions](hardware_l40.txt) ·
[owner NIC port and UCX check](network_local.txt) ·
[l40 NIC port and UCX check](network_l40.txt) ·
[source and local binary/library identities](software_artifacts.txt) ·
[l40 binary/library identities](software_artifacts_l40.txt).

Both hosts have four NVIDIA L4s. GPUs 0/1 are on NUMA node 0, GPUs 2/3 are
on NUMA node 1; `mlx5_0` and `mlx5_1` are respectively local to those pairs.
All 12 directed peer-read checks report access. The owner uses driver
595.58.03, l40 uses 580.178.04; both use CUDA 12.9, Open MPI 4.1.7rc1 and
UCX 1.19.0. The NCCL library is 2.28.7. Rebuilds used the Makefile and
CUDA 12.9 `nvcc`. Initialization, allocation, input fill and output checking
are excluded from all timed call latencies.
The refreshed collective figures follow the eight-L4 layout: latency through
1 MiB and decimal bus throughput from 1 MiB to 1 GiB. Throughput is derived
from complete bytes per rank divided by median call latency, multiplied by
`2*(r-1)/r` for AllReduce and `(r-1)/r` for AllGather/ReduceScatter. It is a
collective useful-byte metric, not physical wire bandwidth.
The [scoped CUDA build log](build_scoped.log) includes the refreshed AR/AG/RS,
fusion, and NCCL control targets. The
[four-L4 NCCL control smoke](single_ar_current_nccl_smoke_local.log),
[peer non-power-of-two tail check](single_peer_tail_1008.log), and
[two-node checked smoke](multinode_smoke_checked/) passed before the full
sweeps.

## Single-node collectives

AllReduce runs on l40. The native UCCL Lite executable forks
four processes before CUDA initialization; the custom NCCL 2.28.7 control
uses four MPI ranks, one GPU per rank. Each reports the slowest-rank host
enqueue+completion time per call. AllReduce is FP32 in place.
Every warmup and timed call's complete output is checked. Three paired runs
alternate backend order; each has five warmups and ten timed calls. The main
summary chooses the lowest correct three-run median UCCL Lite path at each
size, independently of NCCL. The other UCCL paths, including regressions,
remain in the candidate CSV and raw logs.
The NCCL control uses its default P2P and shared-memory selection; no
`NCCL_P2P_LEVEL`, `NCCL_P2P_DISABLE`, or CUDA-graph override was applied.

[AR paired logs](single_native_l40/) ·
[5,580 timed call samples](single_native_samples.csv) ·
[all measured UCCL candidates](single_native_candidates.csv) ·
[selected path and NCCL summary](single_native_summary.csv).
The sample CSV includes 30 `A0-null` launch-floor diagnostics, which are
excluded from path selection.

The direct-peer AllGather and ReduceScatter run on the owner host. Both are
FP32 out of place, with complete bytes per rank as the size. Their single
harness alternates peer and NCCL 2.28.7 over three repetitions, five warmups
and ten timed calls. Each call uses slowest-rank CUDA-event time and checks
every output element. These event figures are kept separate from the
host-clock AR figure.

[AG/RS raw log](single_peer_ag_rs_raw.log) ·
[2,520 timed call samples](single_peer_samples.csv) ·
[complete paired summary](single_peer_summary.csv) ·
[1008-byte tail check](single_peer_tail_1008.log).

## Two-node collectives

Each node runs four GPU ranks. AllReduce is FP32 in place; AllGather and
ReduceScatter are FP32 out of place. Each size is the complete
input/output extent per rank: AllGather input and ReduceScatter output are
one eighth of it. The AR/AG/RS control is a custom eight-rank NCCL 2.28.7
program; both it and the UCCL Lite programs time slowest-rank host
enqueue+completion per call and check every output. All
comparisons use five warmups, ten timed calls, three alternating paired
repetitions, `mlx5_0,mlx5_1`, and the same eight-GPU placement. UCCL Lite
AR/AG/RS use the documented fixed ring/direct-ring paths.
The NCCL controls set `NCCL_SOCKET_IFNAME=ibp55s0f0` and
`NCCL_IB_HCA=mlx5_0,mlx5_1`; P2P and GPUDirect use NCCL defaults. UCCL Lite
MSCCL++ uses `host-no-atomic` IB signaling.

[AR/AG/RS paired logs](multinode_checked/) ·
[3,780 timed call samples](multinode_paired_samples.csv) ·
[all-size paired summary](multinode_paired_summary.csv).

At 1 MiB, NCCL AllReduce has a high and variable delay in the full sweep.
The [three additional paired runs](multinode_ar_1m_recheck/) and their
[run summary](multinode_ar_1m_recheck_summary.csv) reproduce the
anomaly, with NCCL run means of 2.213, 1.190 and 2.279 ms. The main summary
keeps the original full-sweep values, and no stable 1-MiB speedup is inferred
from this point. Some other small NCCL sizes also have wide ranges; the
per-call samples expose them.

The earlier [CUDA-event versus host-clock diagnostic sweep](multinode_preliminary/)
is **not** used in the refreshed charts or speedup claims. The old
`multinode_peer_20261110` archive remains separately available with a timing
correction notice.

## Fusion

Dense FP16 uses l40 and the existing in-executable cuBLAS + NCCL controls.
Eight shapes run three independent times, with the shape order reversed on
the second run. Each invocation uses five warmups, twenty iterations, and
correctness checking. [All 24 logs](fusion_dense_l40/),
[all variant run samples](fusion_dense_samples.csv), and the
[selected-path summary](fusion_dense_summary.csv) retain small regressions
and all diagnostic paths. AG takes its recorded dispatch path; RS and AR use
ready-ring fusion. The chart compares complete-operation latency per shape.

DeepSeek-shaped BF16 full-layer MoE uses l40. The retained variants are
NCCL + grouped cuBLAS, UCCL Lite + cuBLAS, and fused UCCL Lite. Each token
count uses a 2.5-second settling period followed by twelve interleaved rounds
of four calls per variant. The minimum slowest-rank round time per variant is
reported, as in the shipped harness. The [full log](fusion_moe_l40.log),
[per-round samples](fusion_moe_samples.csv), and
[summary](fusion_moe_summary.csv) retain every small-batch loss. Routing-plan
construction and framework work are outside timing.

## Reproduction

From `experimental/lite`, use
[`run_single_node_refresh.sh`](../../../scripts/run_single_node_refresh.sh),
[`run_multinode_peer_compare.sh`](../../../scripts/run_multinode_peer_compare.sh),
[`run_dense_refresh.sh`](../../../scripts/run_dense_refresh.sh), and
[`run_moe_refresh.sh`](../../../scripts/run_moe_refresh.sh). The exact build
variables and two-node launch commands are in [build.md](../../../docs/build.md).
Summaries are validated by the corresponding `summarize_*refresh.py` or
`summarize_multinode_checked.py` scripts; charts are regenerated with
`plot_refresh_collectives.py` and `plot_refresh_fusion.py`. The raw output
files, not the plotted medians, are the primary measurement record.
