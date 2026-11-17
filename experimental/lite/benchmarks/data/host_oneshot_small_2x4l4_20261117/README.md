# Experimental host-staged one-shot AllReduce, two four-L4 hosts

November 17, 2026. **Host staging improves this one-shot implementation and
beats the current small-message kernel at the smallest tested sizes.** At
1 KiB, median latency falls from 36.22 to 27.32 us (24.6% lower, 1.33x speedup).
At 8 KiB it falls from 35.99 to 32.43 us. The 16-KiB median gain is only 4.2%
in speedup; 32 KiB and larger lose to the current kernel. This is a quick eager
experiment, not a promoted routing policy or proof of a universal cutoff.

## Implementation

Explicit flags: `--host-gather 1 --host-oneshot 1`.
The existing host-packet transport now has an optional full-input reduction
consumer; the ordinary gather consumer remains the default.

1. Copy engines upload each GPU's entire raw input into pinned shared-host
   packets, paired by the existing two-GPU NUMA groups.
2. Each group leader sends its paired packet to both remote NUMA-group leaders
   using MPI/UCX over InfiniBand. Local peers share the host packet directly.
3. Once all eight inputs are ready, each GPU launches a fused mapped-host-read
   and FP32 sum kernel, writing the full result directly into GPU output.
   Its own input is read from its own GPU; the other seven are host reads.
   There is no separate H2D memcpy and no CPU arithmetic.
4. Existing upload-ready, GPU-completion, network-completion and local ACK rules
   protect all sources and receive slots before reuse. No output pass runs for
   local-only readiness. The complete input for a packet is uploaded before any
   GPU overwrites that packet's in-place input region.

The measured host path performs **no GPU P2P payload accesses**. The inherited
benchmark still constructs GPU transport metadata during setup, but this path
never uses its channels and does not start its MSCCL++ proxy workers. Therefore
this experiment is not yet a standalone harness for hardware lacking P2P/GDR
setup support. The payload itself uses host memory only between GPUs/nodes.

The host consumer uses 256 threads/block and ceil(packet_bytes/4096) blocks,
capped at 128; the default packet size is 256 KiB. The prior P2P candidate uses
one block. Thus the comparison is of the complete implementations, not an
isolated substitution of host reads for peer reads at identical launch geometry.
All three variants use one reduction block at 1/2/4 KiB.

With N input bytes/GPU, host one-shot uploads 4N bytes/node, sends **8N bytes/node**
because packets go to both remote NUMA groups, and each GPU reads 7N bytes from
host memory. The P2P one-shot sends 4N bytes/node; the current hierarchy sends N.
The CPU/NIC progress and transfer replication tradeoffs are included in timing.
MPI/UCX selects its small-message protocol; this record does not claim that every
small message is a zero-copy RDMA write directly into the final shared packet.

## Comparison and measurement contract

Worktree: `/home/zhongjie/uccl-lite-oneshot-small`, branch
`experiment/oneshot-small-ar-20261117`. Initially created from freshly fetched
report branch `origin/uccl-lite` at `f9665eae`; this follow-up builds on the P2P
experiment and records at `14f00333`.

- Two hosts: mibura-sky-test-02 and mibura-sky-test-01, four NVIDIA L4s each.
  PCIe NODE/SYS local paths; two IB rails/host. GPUs 0/1 use mlx5_0, 2/3 mlx5_1.
- FP32, in-place sum, eight MPI ranks. Size is full input/output bytes per GPU.
- CUDA 12.9, driver 595.58.03, Open MPI 4.1.7rc1, existing
  MSCCL++ build-ib-pcieccl. Hardware/software capture: `environment.txt`.
- All three variants freshly measured with the same MPI/UCX hybrid launcher
  and executable. Current = existing single-CTA hierarchical GPU AllReduce;
  P2P = preceding full-input exchange and peer-read one-shot;
  host = new host-packet full-input sum.
- Slowest-rank host enqueue-to-completion latency, including CPU progress,
  D2H, network transfers, GPU reduction, and completion ACKs. Setup, input fill,
  full-output checking and external MPI barriers are excluded.
- Three rounds, 20 warmups plus 30 checked timed calls per size. Rotated order:
  current/P2P/host, host/current/P2P, P2P/host/current. Clocks were not locked.
- Table uses the median of three round means. `summary.csv` retains every
  variant's min/max round mean. No best-variant-per-size selection; no NCCL run.
- Host 1-KiB round means range 25.41–32.60 us; current 34.83–36.87 us.
  At 16 KiB the improvement is small enough to require more evidence before
  choosing a production threshold. No automatic dispatch has been changed.

## Results

| Input/GPU | Current µs | P2P one-shot µs | Host one-shot µs | Current / host |
|---|---:|---:|---:|---:|
| 1 KiB | 36.22 | 43.23 | 27.32 | 1.33× |
| 2 KiB | 35.69 | 47.14 | 27.47 | 1.30× |
| 4 KiB | 35.61 | 56.08 | 29.82 | 1.19× |
| 8 KiB | 35.99 | 78.57 | 32.43 | 1.11× |
| 16 KiB | 37.44 | 124.59 | 35.93 | 1.04× |
| 32 KiB | 39.63 | 214.10 | 42.11 | 0.94× |
| 64 KiB | 45.31 | 395.86 | 55.13 | 0.82× |
| 128 KiB | 57.03 | 751.12 | 81.81 | 0.70× |
| 256 KiB | 73.00 | 1467.40 | 134.50 | 0.54× |

## Validation

All **810 timed calls per variant** passed full-output checking, as did their
540 warmups and initial checks. Maximum relative error was zero for the tested
rank/generation/position-dependent patterns; original tolerance 1e-5 unchanged.

Host one-shot additionally passes eight consecutive in-place calls without
refill at 1 KiB, 1536 bytes, 256 KiB, and 1 MiB. The 1-MiB check uses
131,200-byte packets, exercising non-power-of-two packet length, a final tail,
and repeated reuse of all three slots. Existing hybrid AllGather and AllReduce
also pass checked 1/2/4-MiB regressions with the default gather consumer.

Affected CUDA targets `multinode_allreduce` and `multinode_cpu_router` build
with the Makefile. Runtime validation above is on both four-GPU hosts.
No graph replay, Compute Sanitizer, sub-1-KiB input, or random/cancellation
stress results are claimed. This remains an explicit experiment.

## Reproduction

Run from this worktree's `experimental/lite` directory:

```bash
make multinode_allreduce multinode_cpu_router \
  MSCCLPP_IB_BUILD=/home/zhongjie/third_party/mscclpp/build-ib-pcieccl \
  MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx
```

Stage the AllReduce binary at the identical worktree path on the other host.
`run.sh` records the full sweep and checks and refuses to overwrite existing
logs; use a new record directory for another run. `summarize.py` regenerates
the table and CSV. Build logs, binary/source hashes, and all raw measurements
are retained here. The earlier P2P-only record remains unchanged.
