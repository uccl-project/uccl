# Two-node host multicast AllGather prototype (2026-11-11)

An opt-in FP32 AllGather for two nodes with four L4 GPUs each. Two MPI processes per node each own one 2-GPU NUMA group. GPUs 0/1 upload shards into NUMA 0 pinned host packets; GPUs 2/3 upload into NUMA 1 packets. Each group sends the same two-shard packet to **both** remote NUMA groups over its local HCA and sends one copy to the other local group. The two destination groups therefore each hold a complete, NUMA-local set of eight source shards. CPU routing launches GPU SM consumers that read the same mapped host packet and write rank-ordered output; it does not make one CPU payload copy per GPU.

Three immutable slots overlap D2H upload of chunk j, UCX/IB transfer of chunk j-1, and GPU host-packet consumption of chunk j-2. A slot is reused only after both GPU consumers and all three MPI sends have completed. The code is [multinode_host_allgather.cu](../../../src/collectives/multinode_host_allgather.cu); the optional target is `make multinode_host_allgather`. The complete output is checked on every invocation, including the non-power-of-two and three-chunk tails in `tail_checks/`.
This is a standalone candidate; it is not graph-capturable or wired into the
automatic collective dispatcher. Its later 1–256-KiB figure selection is
recorded [separately](../host_multicast_selected_2x4l4_20261111/README.md).

For complete output B bytes per GPU, each source GPU contributes B/8. Each node sends B/2 bytes to the other node **twice**, one copy to each destination NUMA pool: B inter-node bytes per node. The intra-node packet exchange copies another B/2 bytes per node. Every GPU consumes B output bytes from its NUMA-local host packet set. `UCX_NET_DEVICES` binds the two group processes to `mlx5_0:1` and `mlx5_1:1`; `hardware_software.txt` records the GPU/NIC topology, versions and matching binary hashes on both hosts.

## Small/medium comparison

`external_paired/` alternates three UCCL candidates across three rounds: existing direct GPU/IB AllGather (8 MPI ranks), host multicast with UCX automatic protocol (4 MPI ranks), and host multicast with `UCX_RNDV_SCHEME=put_zcopy,UCX_RNDV_THRESH=0` (4 MPI ranks). Each round has 20 warmups and ten timed calls at every 1 KiB–2 MiB size. All use FP32, eight GPUs, complete bytes/shape, full output checks and slowest-rank host enqueue+completion timing. **The process count differs by design**, so these are paired end-to-end implementation comparisons, not a single-kernel or identical-MPI-rank microbenchmark. `external_paired/samples.csv` has 1,080 timed calls and `external_paired/summary.csv` keeps every size, path and three-run range.

| Full output bytes/GPU | Direct UCCL (µs) | Host/UCX auto (µs) | Host/forced RDMA WRITE (µs) |
| ---: | ---: | ---: | ---: |
| 1 KiB | 30.89 | 24.72 | 32.53 |
| 2 KiB | 30.79 | 25.32 | 31.88 |
| 4 KiB | 30.73 | 25.35 | 33.13 |
| 8 KiB | 31.00 | 28.13 | 32.58 |
| 16 KiB | 31.92 | 31.48 | 32.83 |
| 32 KiB | 32.05 | 28.09 | 33.79 |
| 64 KiB | 32.98 | 29.98 | 36.15 |
| 128 KiB | 35.68 | 34.90 | 40.55 |
| 256 KiB | 41.76 | 46.55 | 50.43 |
| 512 KiB | 56.81 | 64.73 | 67.33 |
| 1 MiB | 89.16 | 100.07 | 100.55 |
| 2 MiB | 190.79 | 171.83 | 168.68 |

At 1–128 KiB, UCX auto often wins, but `ucx_auto_protocol_1k.log` shows **eager short** over `rc_mlx5` at 1 KiB. That is a host multicast benefit, not an RDMA WRITE win. The forced RDMA WRITE path loses or ties the direct path from 1 KiB through 1 MiB. At 2 MiB, 128 KiB packets give the best tested forced-write result; the larger 4 MiB point is a tie with direct UCCL and still loses to NCCL.

## Same-window three-way rechecks

`threeway_summary.csv` and `threeway_samples.csv` validate 450 timed calls. Each case alternates the chosen host variant, direct UCCL, and the checked NCCL 2.28.7 control over three rounds with 20 warmups and ten calls each. Values below are medians of run means:

| Shape and host variant | Host multicast | Direct UCCL | NCCL 2.28.7 |
| --- | ---: | ---: | ---: |
| 1 KiB, UCX auto | 24.97 µs | 30.66 µs | 32.53 µs |
| 64 KiB, UCX auto | 30.20 µs | 34.53 µs | 41.92 µs |
| 2 MiB, forced WRITE, 128 KiB packet | 160.63 µs | 195.35 µs | 200.94 µs |
| 4 MiB, forced WRITE, 128 KiB packet | 294.40 µs | 294.02 µs | 267.53 µs |

The 2-MiB forced-write host path remains below the lowest of the three direct UCCL run means (host max 161.64 versus direct min 190.66 µs) in that alternating recheck. The NCCL 2-MiB control varies substantially across rounds; its raw calls are retained. The complete [tuned NCCL control](../nccl_tuned_2x4l4_20261111/README.md) is a separate 8-rank baseline and is not spliced into this table.

`ucx_put_protocol_2m.log` reports `rendezvous zero-copy flushed write to remote` over the NUMA-local `rc_mlx5` rail; `ucx_auto_protocol_1k.log` reports eager transport. Cross-HCA `ib_write_bw` probes in `rdma_cross/` succeeded in both directions at about 45,000 MiB/s for 1 MiB writes. These are one-stream link probes, not an end-to-end AllGather result.

## Larger sizes and negative variants

`paired_auto_ucx.log` and `paired_put_zcopy.log` are separate complete 1 KiB–1 GiB checked sweeps against an NCCL control **inside the 4-MPI-process harness**; `auto/` and `put_zcopy/` retain 1,260 timed samples and 21-size summaries each. These controls have a different process layout from the main 8-rank NCCL figure and are not used in that figure. At 1 GiB, host multicast takes 66.35 ms with UCX auto and 76.61 ms with forced RDMA WRITE, compared with 53.96 ms for the existing UCCL direct path in its separately recorded 8-rank sweep. These are different harnesses, so the numbers are diagnostic and are not presented as a fresh matched speedup ratio. The large-message host route is not auto-selected.

`chunk_summary.csv` and `chunk_probe/` retain packet-size and DMA-multicast losses. At 2 MiB, a 128 KiB packet takes 160.90 µs in the single-candidate probe; 64 KiB packets regress to 191.67 µs and DMA multicast takes 197.06 µs at 128 KiB. At 4 MiB, 128 KiB packets take 284.79 µs versus 315.86 µs with a 1 MiB packet, showing that multi-chunk overlap helps before per-packet overhead dominates. `pilots/` keeps early configuration checks, including the original UCX auxiliary-transport failure; these are excluded from accepted summaries.

From `experimental/lite`, regenerate the summaries with
`scripts/summarize_multinode_host_allgather.py` for each full log,
`scripts/summarize_multinode_host_multicast_compare.py` for this record's
`external_paired/` directory, and `scripts/summarize_multinode_host_threeway.py`
for this record's root directory. The launch commands and order are in
`external_paired/commands.txt` and [the runner](../../../scripts/run_multinode_host_multicast_compare.py).

From `experimental/lite`, stage the same candidate and direct UCCL binaries
on both nodes, then run with a fresh output directory:

```bash
HOSTS='10.10.55.2:4,10.10.55.1:4' SOCKET_IFNAME=ibp55s0f0 \
HCA_DEVICES=mlx5_0,mlx5_1 IBV_MODE=host-no-atomic \
MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun \
BUILD_DIR=/home/zhongjie/uccl-zhongjie-dev/experimental/lite/build/multinode_trial \
HOST_MULTICAST_BIN=/home/zhongjie/uccl-zhongjie-dev/experimental/lite/build/uccl_lite_multinode_host_allgather \
MIN_BYTES=1024 MAX_BYTES=2097152 OUTPUT_DIR=build/host-multicast-rerun \
python3 scripts/run_multinode_host_multicast_compare.py
```
