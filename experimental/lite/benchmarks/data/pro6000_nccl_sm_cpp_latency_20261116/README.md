# Matched NCCL SM FP32/C++ batch latency on eight PRO6000 GPUs

Fresh measurements on `wxz-pro6000`, 2026-11-16. This experiment measures
NCCL SM with the timing contract of the
[original PRO5000 C++ PCIeCCL harness](../pro6000_original_cpp_latency_20261116/README.md).
It also freshly reruns that unchanged original PCIeCCL binary during the
independent validation window. This does not measure EAT c03 or the earlier
BF16/Python per-call interface.

[Latency comparison](results.md) · [all five-run values and ranges](summary.csv)

At 1 KiB, the independently validated median is 18.01 µs for tuned NCCL SM
versus 9.86 µs for original PCIeCCL. At 64 KiB the values are 19.97 versus
32.08 µs; at 1 MiB, 70.42 versus 73.40 µs. The original implementation wins
the measured 1–32-KiB points and 512 KiB; the selected NCCL references win
64–256 KiB and 1 MiB. Both wins and losses remain in the table.

All 61 calibration/control/validation runs completed. The 25 independent
validation runs contain 415 size/mode pre-timing correctness checks, all with
zero maximum relative error. The raw per-rank logs acknowledge the requested
algorithm/protocol settings and show non-CE transport.

## Matching the original timing contract

- Hardware: all eight NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs,
  dual-socket AMD EPYC 9B45, four GPUs per NUMA node. XOR-1 peers are PIX,
  XOR-2 NODE and XOR-4 SYS. Hardware identities and topology are in `raw/`.
- Software: NCCL 2.28.9, CUDA compiler 13.3.73, SM120, the same installed
  MSCCL++ dependency and `-O3 -std=c++20` toolchain as the original rerun.
  The NCCL baseline uses MSCCL++ only for untimed bootstrap/barriers and rank
  timing/error aggregation; it creates no MSCCL++ GPU channels or proxy.
- Eight forked C++ processes, one GPU per rank, FP32 in-place SUM. Every power
  of two from 1 KiB to 1 MiB is measured in ascending order. Buffer capacity
  for the complete tensor is 1 MiB throughout each sweep.
- NCCL directly calls `ncclAllReduce(buffer, buffer, ..., ncclFloat, ncclSum,
  comm, stream)`. There is no Python enqueue, caller staging copy or graph
  replay. Both backends poll `cudaGetLastError` after enqueue.
- The original CPU pinning function is preserved for both backends, including
  its hardcoded 0–83 / 84–167 ranges. It is not retuned for this host's
  96-core NUMA nodes. No clock or system-wide setting is changed.
- Each size receives 10 warmups followed by **20 continuous calls**. The C++
  `steady_clock` encloses those launches and the final stream synchronization;
  elapsed time is divided by 20, and the maximum per-rank average is reported.
  Allocation, communicator setup, input generation, rank barrier, correctness
  check and CPU output copy are outside timing. This is host-observed completed
  batch-average latency, not pure kernel time or individual-call p50 latency.
- Input/checking matches the original pattern: `(rank+1) * (1+j%97)`, with
  expected sum `36 * (1+j%97)`. Full-output correctness is checked once before
  timing each size. Timed calls repeatedly reduce the buffer without resetting
  input or checking each invocation, matching the original method.

## Independent tuning and validation

All jobs clear inherited `NCCL_*`/`VLLM_*` settings. Common settings include
`NCCL_P2P_USE_CUDA_MEMCPY=0`, `NCCL_IB_DISABLE=1`, `NCCL_NET=Socket`, and
visible GPUs 0–7. Every accepted NCCL rank startup log is checked for non-CE
transport; default transport may include SHM. The shared
`/home/wxzheng/uccl-eval.lock` is held throughout calibration and validation,
with a GPU process-idle check before each job.

The twelve SM candidates cover:

- Default NCCL; forced system-wide P2P; P2P with 128 threads / 1-MiB buffers.
- Ring and Tree with LL, LL128 or Simple protocol.
- Ring LL with two channels; Ring Simple with two channels; Ring Simple with
  eight channels / 1-MiB buffers.

Exact environment dictionaries are in [run.py](run.py) and the raw manifest.
Each candidate gets **three independent full-grid calibration runs**, using
10 warmups and 20 timed calls per size. Candidate order is deterministically
shuffled between calibration rounds. A failed candidate is excluded rather
than replaced or silently retried.

Each size selects its lowest median calibration latency, then the schedule is
frozen in `raw/frozen-selection.json` **before** validation. Five independent
full-grid runs validate every selected fixed configuration plus default NCCL.
The original unmodified PCIeCCL `-m all` sweep is interleaved once per validation
round, and backend order reverses in alternate rounds. All size orders remain
ascending, as in the original harness.

The tuned table uses the frozen configuration's **independent validation**
values, including losses. It does not choose the best validation run or the
best validation configuration. Reported centers are medians of five run means;
ranges are their observed minimum/maximum, not confidence intervals. The
composite reference uses per-size calibration choices, not a single unchanged
NCCL communicator policy for the whole curve.

The fresh original PCIeCCL column is the pre-EAT `auto` path: LL through
32 KiB and PUT above that in this size range. Source/binary SHA-256 identities
are checked. These results cannot quantify EAT improvement or isolate the
effects of Python enqueue, dtype, graph capture or other timing-scope changes.

## Records and reproduction

- `nccl_sm_latency.cu`: direct NCCL reference, built through the main Makefile.
- `run.py`: calibration, frozen selection, independent validation, shared-lock
  checks, process supervision and original-PCIeCCL control jobs.
- `raw/`: every calibration/validation output, all rank transport logs (losslessly
  packaged in `debug-logs.tar.gz`), hardware
  identity, build log, source/library/binary hashes, commands and policy records.
- `summary.csv`: all selected/default/original validation values and ranges.
- `results.md`: complete 1-KiB–1-MiB latency comparison.
- `provenance.json`: packaged raw-data and measured-source hashes.
- `summarize.py`: dependency-free verification of all raw records, selection,
  source identities, size coverage, zero-error checks and non-CE transport logs.

Build in `experimental/lite` with the pinned environment:

```bash
make nccl-sm-cpp-latency CUDA_ARCH=sm_120 \
  CUDA_HOME=/path/to/cuda13.3 MSCCLPP_HOME=/path/to/mscclpp/install \
  NCCL_HOME=/path/to/nvidia/nccl
python3 benchmarks/data/pro6000_nccl_sm_cpp_latency_20261116/run.py \
  --experiment /path/to/eat-pcieccl-vllm-pro6000-20261115-074910 \
  --original-root /path/to/uccl-lite-original-cpp-latency-20261116
```

The runner expects the recorded original binary and source, the experiment's
isolated toolkit layout, and a fresh `build/nccl-sm-cpp-latency/` output folder.
Binaries, dependency files and generated runtime output remain in `build/`.

Verify the packaged evidence and regenerate the CSV/table without GPUs:

```bash
make nccl-sm-cpp-latency-summary
```
