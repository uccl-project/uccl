# Two-node host AllGather: forced UCX eager versus NCCL, 1–256 KiB

This is the experimental host multicast AllGather from
[`multinode_host_allgather.cu`](../../../src/collectives/multinode_host_allgather.cu),
not the direct-GPU UCCL Lite collective. Two L4 nodes each run four GPUs.
Each of four MPI processes controls one two-GPU NUMA group; the full output
per GPU is FP32 and 1–256 KiB. The cross-node packet is one quarter of that
output size, at most 64 KiB. `UCX_RNDV_THRESH=1048576` forces these packets
below the rendezvous threshold. [`protocol_256k_eager.log`](protocol_256k_eager.log)
shows the cross-node `rc_mlx5` eager range extending to 1 MiB. The
automatic-UCX runs are retained separately; their protocol may vary by size.

Both hosts ran the **same executable hash** and the same NCCL 2.28.7 library
hash; see [`hardware_software.txt`](hardware_software.txt). NCCL and the host
implementation ran inside the same four-process harness, using the same two
CUDA streams per process and eight GPUs. Each size used 20 warmups, three
backend-alternating rounds of ten timed calls, complete output checks on
every call, and slowest-process host enqueue-to-completion latency. Numbers
are the median of the three run means. `summary.csv` includes min/max run
means; `samples.csv` has all 1,080 timed calls. Run
`python3 summarize.py` to validate and regenerate them from the four raw logs.

NCCL uses the default algorithm/protocol at 1–64 KiB and explicit
`NCCL_ALGO=Ring,NCCL_PROTO=Simple` at 128/256 KiB. The latter avoids the
default-NCCL LL latency spike. The NCCL column below comes from the **same
paired run as forced eager** at each size. The automatic-UCX column is from
separate paired runs and is diagnostic, not used to compute eager/NCCL ratios.

| Complete output/GPU | Forced eager (µs) | Paired NCCL (µs) | NCCL/eager | UCX auto (µs) |
| ---: | ---: | ---: | ---: | ---: |
| 1 KiB | 25.10 | 37.45 | 1.49× | 25.54 |
| 2 KiB | 25.51 | 38.80 | 1.52× | 25.62 |
| 4 KiB | 26.06 | 39.02 | 1.50× | 25.42 |
| 8 KiB | 28.87 | 41.24 | 1.43× | 28.50 |
| 16 KiB | 30.94 | 44.61 | 1.44× | 31.14 |
| 32 KiB | 33.51 | 48.81 | 1.46× | 28.02 |
| 64 KiB | 40.08 | 59.67 | 1.49× | 30.58 |
| 128 KiB | 71.38 | 91.56 | 1.28× | 34.78 |
| 256 KiB | 72.39 | 96.70 | 1.34× | 47.42 |

The 128-KiB forced-eager result is **unstable**: its three run means span
53.20–177.63 µs and cross NCCL's 91.24–91.67 µs. Its median is not a
repeatable win. The 1-KiB eager run also has a 51.80-µs outlier run, although
the other two are about 25 µs. Forced eager is slower than UCX auto at
32–256 KiB, especially 64–256 KiB. The 256-KiB forced-eager/NCCL run-mean
ranges are disjoint (72.21–79.98 versus 95.04–97.91 µs), but UCX auto is
faster still. The separate [`protocol_256k_eager.log`](protocol_256k_eager.log)
is for protocol confirmation and is excluded from summary timings.

The test ran initially with mismatched owner/l40 executable hashes; those
diagnostic runs were **discarded**. The four accepted raw logs were rerun
after matching the executable and NCCL library hashes. The main eight-rank
NCCL figures use a different MPI process layout and are not updated by this
record. This comparison does not imply that changing UCX settings affects
the direct-GPU MSCCL++ IB collectives.
