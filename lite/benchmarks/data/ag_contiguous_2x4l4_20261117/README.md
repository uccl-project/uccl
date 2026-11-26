# Small-message contiguous AllGather validation, 2026-11-17

Historical experiment from the earlier refactor checkout. Its commands and source
paths describe that checkout; exact measured sources are archived here. The
[latest-branch port and fresh validation](../ag_combined_latest_2x4l4_20261117/README.md)
supersede this record for the current implementation.


A direct-receive, contiguous-host-buffer prototype reduces 1-KiB full-output
AllGather latency from 31.59 to 24.06 us (23.8%) versus the current shared-ready
SM pipeline. This is a quick two-host comparison, not automatic policy admission.
The 16-KiB point regresses by 3.2%; 256 KiB is effectively tied. Retain both results.

## Implementation

`--download contiguous-dma` enables an explicit one-piece experiment in
`src/multinode/multinode_host_allgather.cu`. Each NUMA process allocates a pinned,
NUMA-local buffer for the complete rank-ordered output. Both local GPUs upload
directly into their respective intervals. Three MPI receives place the other
local group's and both remote groups' packets into their final intervals.
No explicit CPU concatenation or pack pass is performed (MPI may internally copy).
After receives complete, each GPU receives the entire buffer with one H2D memcpy.
All GPU consumers and sends are drained before the buffer can be reused.

`--download contiguous-sm` keeps that exact communication/layout/synchronization
schedule but uses one SM copy kernel per GPU. This isolates the consumer choice
within the prototype. Compared with the existing packet path, the prototype also
changes shared-local exchange to a local MPI payload transfer, removes split
consumption and the ready-progress loop, and copies the GPU's own shard back from
host. H2D volume is N instead of 7N/8. Consequently, the full gain over the existing
path cannot be attributed to CE alone. At 1 KiB contiguous SM is 25.83 us; changing
that consumer to CE saves another 1.77 us. The same-layout CE disadvantage at
16 KiB remains unexplained; no profiler-based causal claim is made.

The prototype rejects multi-piece extents and incompatible modes. Existing modes
remain available with their original defaults. No core router, public communicator,
serving policy or admission profile is changed. Graph capture is not supported by
this MPI harness and no graph support is claimed.

## Method

- Hardware: mibura-sky-test-02 / 10.10.55.2 and mibura-sky-test-01 / 10.10.55.1;
  four NVIDIA L4 GPUs and two IB NICs per host; two NUMA groups per host.
- Four MPI processes, two GPUs/process; NUMA/UCX rail binding uses existing helpers.
- FP32, out-of-place; sizes mean complete output bytes per GPU, input is N/8.
- Build: CUDA 12.9, sm_89, Open MPI 4.1.7rc1, existing NCCL shared-library linkage.
  NCCL itself is not a measured baseline in this experiment.
- Baseline: current shared-ready host pipeline, DMA upload, SM download,
  256-KiB per-GPU chunks, self-input retention, shared local exchange, early consume.
- Additional control: the same packet configuration with DMA download (7 H2D +
  1 D2D submissions per GPU per piece).
- Timing: slowest-process host submission through GPU completion and send retirement.
  Input generation, MPI pre-call barrier, timing reduction and full-output verification
  are outside timing. Setup/allocation/registration are excluded.
- Nine power-of-two sizes, 1--256 KiB; three separate jobs per configuration;
  50 warmups + 100 timed invocations per size/job. Configuration order reverses in
  round 2. Values below are medians of three run means. CSV includes min/max.
- Each mode is a fixed configuration across all sizes; no per-size best selection.
- All 10,800 main timed invocations and 5,400 warmups passed full rank/index/generation
  output checks on both hosts. Tail/delayed-consumer checks separately cover seven
  non-power-of-two extents (1152--73728 bytes), with GPU rank 3 delayed: 280 timed
  calls and 140 warmups, all passing. No sanitizer claim.
- Both affected CUDA targets (MPI-only and optional host-RDMA) compiled. Runtime
  measurements exercise the MPI/UCX path, not the optional direct-RDMA backend.

## Results

| Output/GPU | Existing SM | Existing 8-copy CE | Contiguous SM | Contiguous CE | CE latency reduction vs existing SM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 KiB | 31.59 µs | 55.15 µs | 25.83 µs | 24.06 µs | 23.8% |
| 2 KiB | 30.93 µs | 55.13 µs | 25.43 µs | 23.78 µs | 23.1% |
| 4 KiB | 31.20 µs | 55.18 µs | 25.73 µs | 24.40 µs | 21.8% |
| 8 KiB | 31.51 µs | 55.79 µs | 27.32 µs | 25.98 µs | 17.6% |
| 16 KiB | 32.03 µs | 57.38 µs | 30.16 µs | 33.06 µs | -3.2% |
| 32 KiB | 35.00 µs | 62.31 µs | 28.14 µs | 26.40 µs | 24.6% |
| 64 KiB | 36.28 µs | 65.66 µs | 30.42 µs | 28.64 µs | 21.1% |
| 128 KiB | 39.09 µs | 69.55 µs | 34.62 µs | 32.68 µs | 16.4% |
| 256 KiB | 44.01 µs | 65.48 µs | 46.69 µs | 43.98 µs | 0.1% |

At 1 KiB the existing SM run means span 31.44--31.78 us, versus 23.95--24.17 us
for contiguous CE. At 16 KiB the CE regression appears in all three job means:
33.01--33.07 us versus 31.72--32.13 us. At 256 KiB the ranges overlap.
This supports a small-message specialization, not a universal CE replacement.

## Reproduction and artifacts

From the Lite root:

```bash
make multinode_host_allgather BUILD=build/ag_contiguous_20261117 \
  MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx
# Copy the executable to the identical build path on the second host.
python3 scripts/validate_multinode_ag_contiguous.py \
  --binary build/ag_contiguous_20261117/uccl_lite_multinode_host_allgather \
  --out build/ag_contiguous_repeat --rounds 3 --warmups 50 --iterations 100
```

The runner refuses to overwrite an existing output directory. `manifest.json`
records exact commands, UTC timestamps and the binary hash; `source_hashes.json`,
`measured_source/` and `prototype.patch` preserve the measured implementation and
only this experiment's changes relative to the pre-existing working tree.
Raw per-call logs, `run_means.csv`, `summary.csv`, both hosts' GPU/topology records,
and `tail_delay_checks/` are retained. Historical results do not validate later edits.
