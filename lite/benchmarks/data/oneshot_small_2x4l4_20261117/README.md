# Experimental one-shot AllReduce on two four-L4 hosts

Quick negative result, November 17, 2026. This candidate is slower than the
published small-message hierarchical GPU kernel at every size by the median
of three round means. It is not promoted or enabled by a default policy.
This does not establish that every one-shot design loses on this topology.

## Source and algorithm

Freshly fetched report branch `origin/uccl-lite`, base
`f9665eaead1eb13180628bf3861d004eebbda63e`, from
https://github.com/zhongjiechen/uccl-zhongjie-dev.git.
The original dirty development checkout was not copied into this worktree.
Branch: `experiment/oneshot-small-ar-20261117`.

Candidate: `--ar-oneshot 1`, in `src/multinode/oneshot_allreduce.cuh`.
It launches one CTA of 256 threads per GPU. Each GPU snapshots its full input
using SM stores, sends that input by GPU-memory IB RDMA to its equal-local-rank
partner, then synchronizes with its three local GPU peers. Each GPU reads all
eight original inputs from its node's four GPU allocations and sums the complete
output locally. There is no pre-network reduction and no final result gather.
Local peer payload reads and reduction use SMs, not copy engines. No explicit
host payload staging is used. Network progress uses the existing MSCCL++ proxy.

This is one-shot *reduction*, not one total synchronization event: the network
exchange, local ready barrier, and final local reader-completion barrier remain.
Two send/receive banks prevent reuse while the previous call is being read.
Local completion protects peer readers; cross-node call progression prevents
reuse of bank i before the other node has completed call i. The snapshot also
protects in-place input from overwrite by the output.

With N input bytes/GPU, the candidate sends 4N bytes/node over IB, versus N for
the baseline. Each GPU also reads 6N bytes from peer GPUs, plus its two locally
resident N-byte inputs. Both nodes independently calculate all output elements.
The single CTA and sequential peer-read loop are deliberately simple and have
not been tuned. The size-dependent slowdown is consistent with expensive PCIe
peer reads with limited concurrency; no profiler was run to isolate the cause.

Baseline: `--ar-serial-local 1`, the unmodified `hierarchicalAllReduceSerial`
kernel from the report branch. This is the kernel selected by the explicit
hybrid AllReduce path through 256 KiB. It locally reduces quarters, exchanges
node partials, adds, and pushes completed quarters to local peers. Both modes
use the same executable, MPI launcher, checker, transport, and timing harness.
A known handled CUDA IPC probe status is cleared after transport setup for both
modes; subsequent CUDA errors are checked. The initial failed probe-status
smoke run is retained in `raw/smoke.log`.

## Measurement contract

- Hosts: `mibura-sky-test-02` / `mibura-sky-test-01`, four NVIDIA L4s each.
- Two IB rails/host: local ranks 0/1 use mlx5_0, ranks 2/3 use mlx5_1.
  Local GPU paths are PCIe NODE/SYS, not NVLink; see `environment.txt`.
- CUDA 12.9, driver 595.58.03, Open MPI 4.1.7rc1, existing
  MSCCL++ `build-ib-pcieccl`, `MSCCLPP_IBV_MODE=host-no-atomic`.
- Eight MPI ranks, FP32 in-place sum. Size is full input/output bytes per GPU.
- Eager host enqueue-to-stream-completion latency; slowest of eight ranks per
  call. Input filling, full-output checking, and external MPI barriers excluded.
- Each round: 20 warmups and 30 individually checked timed calls per size.
  Three rounds ordered baseline/candidate, candidate/baseline, baseline/candidate.
- Table: median of three round means. `summary.csv` retains the min/max means.
  Fixed variants, not a best-variant-per-size envelope. No NCCL run here.
- Clocks were not locked. The third baseline round contains four 8–9 ms
  outliers at 2/4 KiB; they are retained, not discarded. Those round means are
  565.50/598.88 us. The other two rounds determine the reported medians there.
- Candidate max allocation is 1.5 MiB/GPU for the 256-KiB sweep, versus about
  648 KiB for the baseline, because candidate banks hold full inputs.

## Results

| Input/GPU | Baseline µs | One-shot µs | One-shot / baseline |
|---|---:|---:|---:|
| 1 KiB | 37.18 | 44.54 | 1.20× |
| 2 KiB | 37.62 | 49.06 | 1.30× |
| 4 KiB | 37.23 | 57.13 | 1.53× |
| 8 KiB | 36.02 | 79.40 | 2.20× |
| 16 KiB | 37.51 | 122.65 | 3.27× |
| 32 KiB | 39.33 | 213.03 | 5.42× |
| 64 KiB | 44.60 | 389.17 | 8.73× |
| 128 KiB | 56.74 | 748.36 | 13.19× |
| 256 KiB | 72.88 | 1461.87 | 20.06× |

## Correctness and limitations

All 810 timed calls per mode passed full-output checking, as did their 540
warmups and initial checks. The measured patterns use varying rank and call
generation values with position-dependent factors; maximum relative error was
zero for these patterns. The original tolerance (1e-5) and checker are unchanged.

Both modes also passed eight consecutive in-place calls without refill at
1 KiB, 1536 bytes, and 256 KiB, exercising buffer reuse and a non-power-of-two
size. The candidate separately passed checked calls at 512 KiB and 1 MiB,
its declared upper bound. These latter sizes are outside the latency comparison.

The Makefile builds the affected target on the two-host deployment path. This
is not compilation-only validation. No graph replay, Compute Sanitizer,
random/cancellation stress suite, or sizes below 1 KiB are claimed. No dispatcher
or application integration is changed.

## Reproduction

Run from this worktree's `experimental/lite` directory:

```bash
make multinode_allreduce \
  MSCCLPP_IB_BUILD=/home/zhongjie/third_party/mscclpp/build-ib-pcieccl \
  MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx
```

Stage the resulting binary at the identical worktree path on the second host.
`run.sh` records the exact sweep and repeated-call commands; it refuses to
overwrite existing logs. Use a new record directory for another run. Run
`python3 benchmarks/data/oneshot_small_2x4l4_20261117/summarize.py` to regenerate
the table and CSV. Binary and source hashes are retained alongside the logs.
