# Remote-first hierarchical ReduceScatter, November 15

Experimental, opt-in FP32 GPU path on `mibura-sky-test-02` and `l40`
(`mibura-sky-test-01`), four NVIDIA L4 GPUs and two IB HCAs per node.
GPU 0/1 + mlx5_0 belong to NUMA 0; GPU 2/3 + mlx5_1 belong to NUMA 1.
Topology, driver versions, and binary/library hashes are recorded alongside
this file. CUDA 12.9, Open MPI 4.1.7rc1, NCCL 2.28.7. The same staged executable
and MSCCL++/NCCL libraries are used on both hosts.

The new option reduces remote-destination partials first, submits IB, and reduces
local destinations while network progress runs. The push variant retains CUDA
IPC writes; the pull variant reads peers directly and is retained as a negative
experiment. See [design and lifetime invariants](../../../docs/multinode-hierarchy.md).
CPU-router integration is a proposed design, not an implemented or measured backend.

## Method

`validated/` is the final two-receive-slot implementation. Eight MPI processes,
one per GPU, with out-of-place ReduceScatter. Bytes denote the **complete input
per GPU**, not its one-eighth-size output. Each backend uses slowest-rank host
enqueue-to-completion time; initialization and checking are outside timing.
Every warmup and timed invocation checks the complete output against the existing
reference with generation-dependent input. Twenty warmups, twenty timed calls,
three rounds, sizes 1 KiB–4 MiB in powers of two. Summary values are medians of
three per-round means; the CSV also retains minimum and maximum round means.
There is no best-variant-per-size curve.

Controls are the explicit serial UCCL local path, existing `--rs-auto` policy,
NCCL defaults, and NCCL Ring+Simple. The latter exposes default-protocol anomalies
without silently selecting the slower control. The automatic UCCL control uses
`--rs-piece-bytes 8388608 -B 96 -T 512`. The experimental push and serial use one
CTA and 256 threads. Execution order is serial/auto/overlap/NCCL-default/
NCCL-simple in rounds 1 and 3 and reversed in round 2. No simultaneous GPU
benchmarks were launched by this experiment.

`*_pilot.log`, `final/`, `final_v2/`, and checks outside `validated_checks/`
are preliminary versions, not correctness or performance evidence for the final
source. Pilots used five warmups and ten timed calls and tested 1/4/16 CTAs.
They rejected direct PCIe reads and larger grids. `final/` used one receive slot;
`final_v2/` restored the original peer-copy schedule and separated the own-input
load. Both precede the two-slot lifetime correction. In `final_v2`, the adaptive
control was run after the other backends, rather than alternated. All these
negative and preliminary results are retained, but only `validated/` supports
the final comparison.

## Final results

All values below are microseconds, median of three round means. All final
comparison samples had zero reference error. See
[`validated/summary.csv`](validated/summary.csv) for every size and round range.

| Complete input/GPU | Serial UCCL | Adaptive UCCL | New overlap | NCCL default | NCCL Ring+Simple |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 KiB | 28.15 | 30.19 | 31.60 | 52.71 | 82.27 |
| 64 KiB | 31.20 | 31.24 | 34.22 | 42.18 | 76.34 |
| 256 KiB | 44.94 | 44.57 | 47.33 | 900.00 | 82.39 |
| 512 KiB | 65.64 | 64.28 | 66.73 | 2509.69 | 91.32 |
| 1 MiB | 105.96 | 105.54 | 104.17 | 119.45 | 115.96 |
| 2 MiB | 195.94 | 181.99 | 188.25 | 178.34 | 176.11 |
| 4 MiB | 358.41 | 290.24 | 343.39 | 282.02 | 281.15 |

At 1 MiB the candidate is **1.7% lower latency than serial UCCL**, with
non-overlapping round-mean ranges (103.78–104.97 versus 105.32–106.30 µs).
Its 1.3% advantage over adaptive UCCL has overlapping ranges and is not a
robust policy-selection win. It is 10.2% lower latency than the Ring+Simple
control. At 2/4 MiB it improves over serial but loses to adaptive UCCL and NCCL;
below 1 MiB it regresses against existing UCCL. Thus this is a narrow schedule
experiment, **not a general small/intermediate-message improvement**.

NCCL default has large protocol-sensitive delays at 256/512 KiB, and several
NCCL controls have large outlier rounds. Those observations are retained; they
do not establish corresponding gains from the new schedule. No automatic
routing threshold is changed. More substantial improvement needs the proposed
NUMA packet experiment and measurements of its staging/progress costs.

Final binary/source identities are in `validated_hashes.txt` and
`validated_remote_hash.txt`; `hashes.txt` refers to the earlier `final_v2` build.

## Reproduction

Build with the Makefile (the local IB-enabled MSCCL++ build has a nondefault name):

```bash
make multinode_allreduce multinode_cpu_router nccl_2x4l4_baseline \
  MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx \
  MSCCLPP_IB_BUILD=/home/zhongjie/third_party/mscclpp/build-ib-pcieccl
```

Stage identical `uccl_lite_multinode_allreduce` and
`uccl_lite_nccl_2x4l4_baseline` binaries at the same build directory on both hosts.
From `experimental/lite`, use a new output directory:

```bash
HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0 \
MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun \
BUILD_DIR="$PWD/build/multinode_pull_20261115" \
OUTPUT_DIR="$PWD/build/multinode-overlap-recheck" \
  bash scripts/run_multinode_overlap.sh
python3 scripts/summarize_multinode_overlap.py build/multinode-overlap-recheck
```

## Validation scope

The affected GPU executable and CPU-router wrapper both compile; the latter
includes the changed translation unit. No CPU-router core or policy changed.
The final GPU variants are checked on **both four-GPU hosts**, at one and four
CTAs, with non-power-of-two extents 1536 through 1572864 bytes, changing inputs,
and 50 consecutive launches without per-call host barriers at 1 KiB–4 MiB.
Consecutive-launch checks verify the final result and a fresh generation after
the batch; per-call checks use host barriers. Existing AR/AG automatic paths
receive a two-host checked smoke sweep through 1 MiB. This is eager validation;
CUDA graph replay, arbitrary floating-point distributions, and Compute Sanitizer
are not covered. No policy calibration or automatic route promotion is made.

Shared archived records: [local software identity](../ag_1m_1g_2x4l4_20261115/software_local.txt), [local GPU topology](../ag_1m_1g_2x4l4_20261115/topology_local.txt), [remote GPU topology](../ag_1m_1g_2x4l4_20261115/topology_local.txt).
These files are byte-identical to this experiment's recorded copies; fresh
runs must capture their own environment and software identities.
