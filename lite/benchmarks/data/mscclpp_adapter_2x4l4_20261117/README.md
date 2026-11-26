# MSCCL++ algorithm-runtime adapter, November 17

An unchanged MPI nccl-tests executable loads
`libuccl_lite_nccl_mscclpp.so` and reaches the retained UCCL Lite GPU hierarchy
through MSCCL++ NativeAlgorithm, channels and IB proxies. This is a new
experimental integration, not deployment admission or a validation of every
historical UCCL Lite curve.

## Configuration and timing

- Hosts: mibura-sky-test-02 / 10.10.55.2 and mibura-sky-test-01 / 10.10.55.1;
  four NVIDIA L4s and two IB rails per host, GPU pairs 0/1 and 2/3. Eight MPI
  ranks, one GPU per rank, consecutive ranks per host, rail map 0/0/1/1.
- FP32 SUM AR/RS and FP32 AG; both NCCL in-place and out-of-place layouts.
  Complete extent is input bytes/GPU for AR/RS and output bytes/GPU for AG.
- nccl-tests 2.19.7, NCCL 2.28.7, CUDA 12.9 build and linked runtime.
  MSCCL++ external checkout 52f659d, IB-enabled build. `host0.json` and
  `host1.json` retain GPU UUIDs, driver, topology, CPU affinity, environment,
  and actual library hashes; the external checkout already had a local
  gpu_utils.cc modification and was not changed by this integration.
- Native NCCL controls and the adapter use the same test executables and
  NCCL configuration. NCCL owns the data plane for fallback; no forced
  Ring/Simple tuning is applied. Explicit interface ibp55s0f0 and rails
  mlx5_0,mlx5_1 apply to both. MSCCL++ uses host-no-atomic IB mode.
- Maximum-rank host enqueue/completion batch means (`-a 3`), 5 warmups and
  20 timed calls, 1 KiB–1 MiB powers of two. Initialization and preparation
  occur before steady-state timing; per-call staging, scratch reuse handshake
  and completion bookkeeping are inside it.
- One native/candidate/candidate/native sequence per operation: two independent
  jobs per backend, medians of their means. Both individual results are
  retained. There is no per-size best-variant selection. The candidate uses
  one fixed GPU hierarchy for exact powers of two from 1–256 KiB; larger
  sizes use NCCL in the same communicator lifetime.

## Correctness

All 20 nccl-tests jobs passed both layouts and the final zero-error summary:
12 native/candidate controls, three element-offset-one misalignment jobs,
three graph replay jobs, one unsupported-product reduction job and one
missing-profile job. Candidate tests require positive UCCL route counters
on **all eight ranks**. The unsupported-product and missing-profile checks
require zero UCCL calls. An additional one-host-only missing-profile test
also passes, confirming that a negative vote on one host keeps all eight
ranks on NCCL without entering the UCCL prepare boundary.

Graph-mode nccl-tests eagerly warms all sizes before capture. The archived
counts correctly allow those eager UCCL calls; nonempty capture permanently
transfers the communicator to native NCCL. These are native graph fallback
checks, not evidence that the UCCL network kernels support graph replay.

`abi.log` adds 72 fully checked calls per rank across two communicator
lifetimes: signed/index-dependent full outputs, output guards, changing
collective families and extents, alternating streams, per-rank mixed in-place
layouts and pointer alignment, and interspersed 384-byte NCCL fallback calls.
Each lifetime records exactly AG=9, RS=9, AR=9, fallback=9 on every rank.
Saved empty graphs are replayed after activation; grouped calls and later
nonempty graph replay complete through NCCL. There is no sanitizer or
network-fault-recovery claim.

Standalone AR/AG/RS regression sweeps from 1 KiB to 16 MiB passed every output
check after extracting their shared device code. Each used one warmup and two
checked calls per size, `--ar-auto`, `--ag-auto` or `--rs-auto`, 96 blocks and
512 threads where the selected kernel uses a grid. The adapter, standalone
GPU target, CPU-router standalone target and NCCL-only ABI client compiled.
An old build dependency file from the previous branch referenced a missing
source path; a fresh `build/mscclpp_compile` directory resolved that build
artifact issue (`standalone_build.log`).

## Performance and retained negative results

The adapter is **not generally faster than NCCL**. Representative out-of-place
medians below are microseconds; all rows and both layouts are in `manifest.json`.

| Operation | Extent | Native NCCL | Adapter |
| --- | ---: | ---: | ---: |
| AllGather | 1 KiB | 18.775 | 42.325 |
| ReduceScatter | 1 KiB | 20.330 | 40.830 |
| AllReduce | 1 KiB | 28.685 | 47.155 |
| AllGather | 256 KiB | 190.685 | 57.195 |
| ReduceScatter | 256 KiB | 179.080 | 58.400 |
| AllReduce | 256 KiB | 586.530 | 87.500 |

Native controls have substantial size-dependent spikes and run variation
(e.g. 256-KiB AG 225.07/156.30 us and RS 132.82/225.34 us). The apparent gains
there do not establish robust admission against a tuned or stable NCCL
baseline. Small-message regressions are real negative evidence for this
adapter. No shape is promoted to production policy. `candidate_over_nccl.json`
retains ratios for every measured size, including fallback sizes above 256 KiB.

Early bring-up exposed grouped initialization skipping registration and a
CUDA 13 unversioned runtime probe mismatching the actually linked CUDA 12
runtime on node1. Those initial fallback-only passes were rejected as UCCL
validation. The final driver asserts routes and probes the linked runtime.
An initial MPI TCP interface mismatch was fixed by explicitly selecting the
documented interface; it is unrelated to collective correctness.

Large rings and MPI/UCX host-packet, combined CE, and host-one-shot algorithms
remain standalone. Their performance is not represented by this adapter's
candidate curve. See [integration instructions](../../../integration/nccl_lite/MSCCLPP.md)
and `extra_commands.json` for the additional checks. Source and adapter binary
hashes identify this measured implementation; subsequent changes need fresh
validation.
