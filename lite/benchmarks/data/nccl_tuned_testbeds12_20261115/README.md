# Tuned NCCL baselines for Testbeds 1 and 2

> This directory retains only inputs and metadata needed by current figures or checks. [Complete original experiment](https://github.com/zhongjiechen/uccl-zhongjie-dev/tree/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/nccl_tuned_testbeds12_20261115) includes raw runs, removed scripts and source snapshots. Historical commands below require that archive.


These measurements add **NCCL CE (tuned)** and **NCCL SM (tuned)** to the six
8×L4 AR/AG/RS figures. **UCCL Lite** and **NCCL (default)** retain their original
[overview measurements](../overview_8l4_20261114/README.md). Figure layout,
colors and legend order follow the RTX PRO 6000 ×2 figures.

SM (tuned) means tuning the default NCCL implementation, not a new collective
backend. CE (tuned) enables NCCL's existing copy-engine-assisted transports.
NCCL is the reference implementation throughout; no NCCL source or UCCL
collective implementation is modified for this study.

## Hardware and timing

Both testbeds run on `wxz-l4`: eight NVIDIA L4 GPUs, four per NUMA node, a
96-logical-CPU Intel Xeon 2.20-GHz VM, CUDA 12.9 compilation (`sm_89`), and the
same NCCL **2.28.9** library used by the original figures.

- Testbed 1 preserves physical CUDA P2P capability within each four-GPU group.
  The screen includes both default NCCL transport policy and explicit P2P
  permission; the selected configuration and actual channel logs determine
  which transport is used.
- Testbed 2 disables CUDA peer access and sets `NCCL_P2P_DISABLE=1` and
  `NCCL_SHM_DISABLE=0` throughout. Its measurements are independent of Testbed 1.
- Both disable IB and use the Socket network fallback. The selected traces
  must contain only SHM/P2P local channels, not network fallback.

The existing `compact_collectives.cu` harness is built through the Makefile
and run with `--variants nccl`. It uses **BF16, in-place AR, out-of-place AG/RS**.
Bytes mean full AR/RS input or AG output per rank. The submitting thread is
CPU 47, as in the original figures. All producer streams complete before the
maximum-rank CUDA-event timer starts. Allocation, communicator initialization,
input generation and correctness checking are excluded. Every warmup and timed
call checks every output element and its guard.

Both panels retain the original ranges: 1 KiB–1 MiB latency and 1 MiB–1 GiB
bus throughput. Throughput is decimal GB/s: `bytes / microseconds / 1000`,
multiplied by `14/8` for AR or `7/8` for AG/RS.

## Tuning and independent selection

The initial screen compares channel counts, buffer sizes, protocol selection,
SHM memcpy direction, CPU affinity handling, and (Testbed 1 only) P2P policy.
The follow-up checks larger buffers with four/eight channels using each
operation's promising configurations. The CE setting with a 16-MiB buffer
and eight channels stalled during ascending AR near 128 KiB; its timeout is
retained, and identical settings are conservatively excluded from the remaining
workloads. One interrupted controller-restart attempt is preserved separately
and is not included in calibration. All settings and parent configurations
are retained in `candidates.json`.

- **SM:** `NCCL_SHM_USE_CUDA_MEMCPY=0` and
  `NCCL_P2P_USE_CUDA_MEMCPY=0`; tune the ordinary NCCL path.
- **CE:** both memcpy knobs are `1`, with `NCCL_PROTO=Simple`. NCCL's SHM
  memcpy path applies to Simple; enabling its knob alone would not establish
  that an LL transfer used CE. `NCCL_SHM_MEMCPY_MODE=1/2/3` controls sender,
  receiver or both sides. A one-sided mode still uses SM access on the other
  side, and reduction/progress kernels still use SMs. Thus CE means assisted
  transport, not a zero-SM collective. See the pinned
  [NCCL SHM implementation](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/transport/shm.cc).
- `NCCL_IGNORE_CPU_AFFINITY=1` is included where beneficial. The original
  single-CPU submission affinity can otherwise constrain NCCL initialization
  and proxy placement. This knob lets NCCL use the GPU's NUMA CPU affinity;
  the harness still submits on CPU 47. See
  [NCCL topology affinity handling](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/graph/topo.cc).

Calibration uses 3 repetitions, each with 3 warmups and 3 checked timed calls,
over every power of two from 1 KiB through 1 GiB. A configuration must complete
all 21 sizes for its operation to be eligible. Selection minimizes summed log
calibration latency, with a `log(1.03)` cost per configuration switch to avoid
noise-only fragments. The resulting **per-size calibrated schedule** is frozen
before validation; this is not a claim that one process can change NCCL
communicator settings at every call, nor a claim of global optimality.

Each selected setting is independently measured twice with **3 repetitions
of 5 warmups + 10 timed calls per size**. The second pass reverses configuration
order. Every process starts at **1 KiB and increases size**, even if only its
larger sizes were selected. Unselected prefix samples are preserved and do not
select or replace plotted values. A plotted point is the median of the three
round medians in each run, then the median of the two run medians. Run bounds
are descriptive, not confidence intervals. Validation data never selects the
configuration.

The UCCL/default curves remain the historical paired measurements from the
original overview. These new tuned references use the same host, library,
layout, correctness check and timing scope, but do not revalidate historical
UCCL performance against the current source revision. The tuned curves do not
universally beat the retained default curve: in particular, large-message AR
is slower in this validation. Those differences are preserved; the historical
default is not a contemporaneous control for attributing each difference to a knob.

## Evidence and reproduction

`summary.csv` contains all 252 selected CE/SM points (2 testbeds × 3 operations
× 21 sizes × 2 baseline classes), including both independent run medians and
bus bandwidth. `schedule.json` maps every point to its settings;
`transport.json` audits actual channel types and collective protocols from
INFO logs. Slower configurations and any failed attempts remain in the raw
measurement archive. No default or UCCL points are replaced with tuned values.

From `experimental/lite`, provide the dependency paths recorded in the build
log and build into a separate directory:

```bash
make -j2 BUILD=build/nccl_tuned_testbeds12_20261115 compact-collectives \
  CUDA_HOME=/usr/local/cuda-12.9 CUDA_ARCH=sm_89 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 NCCL_HOME=build/nccl-2.28.9
python3 benchmarks/data/nccl_tuned_testbeds12_20261115/run.py pilot
python3 benchmarks/data/nccl_tuned_testbeds12_20261115/run.py pilot2
python3 benchmarks/data/nccl_tuned_testbeds12_20261115/run.py calibrate
python3 benchmarks/data/nccl_tuned_testbeds12_20261115/select.py refine
python3 benchmarks/data/nccl_tuned_testbeds12_20261115/run.py refine
python3 benchmarks/data/nccl_tuned_testbeds12_20261115/select.py schedule
python3 benchmarks/data/nccl_tuned_testbeds12_20261115/run.py evaluate
python3 benchmarks/data/nccl_tuned_testbeds12_20261115/run.py trace
python3 benchmarks/data/nccl_tuned_testbeds12_20261115/analyze.py
python3 benchmarks/data/overview_8l4_20261114/plot.py --preview
```

The runner holds `/tmp/uccl-lite-integration-wxz-l4.lock`, checks for external
GPU work before/during/after runs and stops only its own process if interrupted.
Completed jobs can be resumed without overwriting their evidence. A partial
job must be preserved separately before retrying it. Binary/library hashes,
source identity, environments and exact commands accompany the measurements.

## Validated 1 GiB bus bandwidth

Decimal GB/s; UCCL/default retain the original measurements, and CE/SM use the independent validation runs.

| Testbed | Collective | UCCL Lite | NCCL CE (tuned) | NCCL SM (tuned) | NCCL (default) |
|---|---|---:|---:|---:|---:|
| 1 | AllReduce | 6.76 | 4.61 | 4.71 | 5.04 |
| 1 | AllGather | 5.80 | 4.59 | 5.06 | 4.80 |
| 1 | ReduceScatter | 5.43 | 4.57 | 4.49 | 4.12 |
| 2 | AllReduce | 6.76 | 4.60 | 4.71 | 5.20 |
| 2 | AllGather | 5.80 | 4.59 | 5.05 | 4.99 |
| 2 | ReduceScatter | 5.43 | 4.57 | 4.50 | 4.10 |

25620 timed calls and 12810 warmup calls passed in independent validation, including unplotted ascending-prefix samples. All selected configurations passed the INFO channel/protocol audit.

Raw samples, setup probes, slower configurations, exact commands and INFO logs are retained in [measurements.tar.gz](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/nccl_tuned_testbeds12_20261115/measurements.tar.gz); [raw_checksums.json](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/nccl_tuned_testbeds12_20261115/raw_checksums.json) authenticates every member. Extract into the recorded build directory to rerun `analyze.py` without remeasuring.
