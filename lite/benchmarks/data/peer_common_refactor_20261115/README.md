# Shared peer components: refactor validation, 2026-11-15

Baseline: `b968a8011fb6979398a12ee2b5273bcf6693b94f`, versus the shared-component
refactor identified by [source hashes](source_manifest.json). The existing
PRO6000 baseline binary came from `b0ac7b4a`; its joint implementation, public
header and frozen benchmark are identical to `b968a801`. Both sides were
measured afresh. No best-per-size selection, retuning, or historical timing
substitution is used here.

The change extracts one prepared BF16 peer executor from joint routing and
shares inline copy/ordered-reduction operations with the existing four-L4 FP32
peer and eight-L4 BF16 local paths. It preserves transport selection, message
thresholds, kernel launches, event dependencies, source order and rounding.
See [component ownership](../../../docs/retained-optimizations.md#shared-collective-components).

Raw logs, samples, host descriptions and command manifests are preserved
byte-for-byte in [measurements.tar.gz](measurements.tar.gz), with member hashes
in [evidence_manifest.json](evidence_manifest.json). Directory names below
refer to archive members. No raw output was trimmed to satisfy formatting.

## Performance result

No stable material regression was observed in these tests. This is a measured
scope, not a guarantee for other hardware, compiler versions or applications.
All percentages below are **new latency / old latency − 1**; negative is faster.

| Paired measurement | AllReduce | AllGather | ReduceScatter |
|---|---:|---:|---:|
| PRO6000, 1 KiB–1 GiB, separate-process ABBA | −0.68% | −0.36% | −0.08% |
| PRO6000, 1 KiB–1 MiB, adjacent calls in one process | +0.31% | −0.04% | −0.04% |
| Four L4, BF16 peer, 1 KiB–64 MiB | +0.10% | −0.15% | −0.50% |
| Eight L4, BF16 local, 1 KiB–64 MiB, submission CPU 0 | −1.49% | −1.54% | −1.00% |

These are equal-weight geometric means over each grid. Four-L4 FP32 shared
`pull` AG/RS changed by −0.05%/−0.12%; AG `pull-vec` changed by −0.08%.
Every affected FP32 variant's worst point was within +1.45%; every four-L4
BF16 point within +2.66%; every pinned eight-L4 point within +1.50%.
Native MSCCL++ AllReduce, fusion and multi-node implementations were not changed
or remeasured. The slow four-L4 BF16 peer schedule remains experimental.

On PRO6000, all **1 MiB–1 GiB** points were within +0.72% latency. At 1 GiB:

| Collective | Before bus GB/s | After bus GB/s |
|---|---:|---:|
| AllReduce | 48.98 | 48.98 |
| AllGather | 48.20 | 48.20 |
| ReduceScatter | 46.63 | 46.64 |

[All sizes and variants](cases.csv) include both old/new latency and bus
bandwidth. Sizes mean full AR/RS input or AG output per rank. Bus bandwidth is
`bytes / latency × (ranks−1)/ranks`, doubled for AR, in decimal GB/s.

### Small-message uncertainty and retained negative results

Separate-process PRO6000 runs had individual small points up to +10.91%
(under 1 µs). A CPU-0-pinned small-grid repeat still had points up to +5.85%.
These results are retained in `pro6000/` and `pro6000-small-pinned/`.

The diagnostic [in-process harness](paired_in_process.cu) links the old and new
implementations under distinct class names. It uses the frozen benchmark's
same buffers, input generation, correctness oracle and max-rank CUDA-event
timer, alternating old/new and new/old invocation order. Across three runs,
each of 33 cases has 1,350 adjacent pairs. The geometric mean of pair ratios
was at most +1.23% at any size; median pair ratios were within +0.15%.
Raw round-median estimates can still move by several percent as the order-
dependent, quantized timing distributions change. Both raw estimators remain
available; [paired_summary.json](paired_summary.json) explicitly reports the
adjacent-pair estimator rather than silently replacing the original runs.

The initial unpinned eight-L4 run had +3.77% aggregate AR and up to +8.41% at a
point. The longer, fixed-submission-CPU rerun above did not reproduce that
slowdown. Both runs remain recorded. In addition, the local implementation's
host `.text` and its GPU kernels are byte-identical, providing independent
evidence that this refactor did not change its execution instructions.

## Hardware and timing

| Host | Hardware/topology | Compiler | Measurement |
|---|---|---|---|
| `aws-pro6000` | 2 RTX PRO 6000 Blackwell Server Edition; Xeon Platinum 8559C; one NUMA node, PIX peers | CUDA 13.0, `sm_120` | BF16 out-of-place AR/AG/RS; forced peer |
| `l41` | 4 L4; Xeon Silver 4410Y; 2+2 NUMA, NODE/SYS peers | CUDA 12.9, `sm_89` | BF16 peer and FP32 peer/pull/pull-vec; NCCL 2.30.7 control for FP32 |
| local `wxz-l4` | 8 L4; Xeon @ 2.20 GHz VM; 4+4 NUMA, PHB/SYS | CUDA 12.9, `sm_89` | BF16 out-of-place local fallback, router off |

All timings are eager maximum-rank CUDA events around the collective,
including stream dependencies and submission gaps visible to those events.
Preparation, input generation and correctness checking are outside timing;
every warmup/timed invocation is checked. Local fallback includes its existing
host-blocking CPU/staging work. All sweeps run **small to large** while holding
the host's shared evaluation lock and rejecting external GPU work.

- PRO6000 full grid: 100 warmups, 5 rounds × 20 calls, old/new/new/old processes.
  Each process reports median of round medians; comparisons average the two
  old process medians and the two new process medians.
- PRO6000 pinned small grid: same ordering, 200 warmups, 9 × 50 calls,
  `taskset -c 0` on the runner and children.
- PRO6000 in-process: three runs pinned to CPU 0, 200 warmups per implementation
  per case, 9 × 50 pairs. No samples are trimmed or selected for speed.
- Four-L4 BF16: 20 warmups, 5 × 20 calls, old/new/new/old; 51 cases.
  FP32: 20 warmups, 5 × 20 calls; median of round means; 77 variant/size cases
  over 1 KiB–1 MiB. Explicit pull/pull-vec ceilings exercise the shared kernels.
- Eight-L4: 10 warmups, 5 × 10 calls initially; 40 warmups, 5 × 20 calls with
  `--main-cpu 0` on the repeat; 51 cases per process, old/new/new/old.

## Correctness and generated code

All affected CUDA targets build through the Makefile; command logs are in
`build-logs/` in the raw archive. Validation includes:

- Two/four-GPU joint selftests: independent BF16 oracle, odd tails, bitwise AG,
  producer/input reuse, profile rejection and graph-capture rejection/recovery.
- New `joint-peer-check`: 225 pairs of collective/size transitions per topology,
  with disjoint caller stream sets, delayed producers, scalar tails, guards,
  source preservation, device restoration and destruction while work is pending.
- Four-L4 peer small/medium, ring and resident checks on the documented 2+2 host.
- Eight-L4 local fallback: 61 cases, including tails, in-place/out-of-place,
  BF16 rounding and capture rejection.
- Router policy (35 checks), host dispatcher (74), CUDA dispatcher, and router
  eager (732 cases) / graph (1,000 mixed collectives) on the relevant hosts.

The first l41 staging attempt omitted the policy test's CSV fixture; the next
omitted its `build/` output directory. These two setup failures are preserved
in `l41-initial-checks/` and `l41-setup-retry/`. After staging both, all remaining
checks and measurements passed in `l41/`. No collective correctness failure
occurred.

[compare_sass.py](compare_sass.py) compares both 64-bit halves of each machine
instruction, including scheduling control bits, ignoring symbol names only:

- All six BF16 peer kernels: identical on **sm_89 and sm_120**.
- Three changed FP32 kernels: identical on sm_89.
- Both local BF16 kernels: identical on sm_89; local host `.text` SHA-256 is
  identical too (see source manifest).

Disassembly is retained in [sass.tar.gz](sass.tar.gz); hashes/results are in
`sass_*.json`. No binary or dependency tree is stored here.

## Reproduction

Build baseline and candidate in separate checkouts with identical toolchain
options, preserving the old binaries. Follow the main Makefile build
instructions, then run `run_paired.py --help`. Exact host commands, binary
hashes, GPU/CPU descriptions and test manifests are in each result directory.
For `run_checks.py`, create the repository's `build/` directory and retain
`benchmarks/data/primitives_8l4.csv`, then pass `JOBS_JSON NEW_OUTPUT_DIR LOCK`.

For the optional same-process diagnostic, from `experimental/lite` export the
baseline into `build/` (no generated source in production directories):

```bash
python3 - <<'PY'
from pathlib import Path
import subprocess
revision = 'b968a8011fb6979398a12ee2b5273bcf6693b94f'
Path('build').mkdir(exist_ok=True)
for source, target in [
    ('src/collectives/joint_collectives.cu', 'peer_common_baseline.cu'),
    ('include/uccl_lite/joint_collectives.hpp', 'peer_common_baseline.hpp'),
]:
    text = subprocess.check_output(
        ['git', 'show', revision + ':experimental/lite/' + source], text=True)
    if target.endswith('.cu'):
        text = text.replace('"uccl_lite/joint_collectives.hpp"',
                            '"peer_common_baseline.hpp"')
    Path('build', target).write_text(text)
PY
make -f Makefile -f benchmarks/data/peer_common_refactor_20261115/paired.mk \
  peer-common-paired CUDA_ARCH=sm_120 \
  CUDA_HOME=/path/to/cuda CUTLASS_HOME=/path/to/cutlass NCCL_HOME=/path/to/nccl
# Under the host evaluation lock, with idle GPUs:
taskset -c 0 ./build/uccl_lite_peer_common_paired 2 build/new-pairs.csv
```

`python3 benchmarks/data/peer_common_refactor_20261115/summarize.py` regenerates
all summary tables from the retained logs/samples. Historical README figures
remain measurements of their original configurations; this refactor's data is
separate. Fresh calibrated routing profiles must use the new binary identity.

## PRO6000 bus bandwidth, 1 MiB–1 GiB

Values are before → after, decimal GB/s, using the paired full-grid run above.

| Full bytes per rank | AllReduce | AllGather | ReduceScatter |
|---|---:|---:|---:|
| 1 MiB | 21.94 → 22.01 | 16.33 → 16.52 | 16.30 → 16.45 |
| 2 MiB | 26.24 → 26.20 | 22.02 → 21.97 | 21.82 → 22.05 |
| 4 MiB | 35.88 → 35.89 | 33.36 → 33.14 | 33.47 → 33.23 |
| 8 MiB | 42.96 → 43.15 | 41.14 → 41.49 | 39.90 → 39.86 |
| 16 MiB | 46.26 → 46.21 | 45.84 → 45.57 | 42.89 → 42.74 |
| 32 MiB | 49.22 → 49.33 | 48.70 → 48.62 | 46.99 → 47.06 |
| 64 MiB | 49.33 → 49.40 | 50.35 → 50.25 | 48.52 → 48.48 |
| 128 MiB | 48.99 → 49.02 | 48.58 → 48.54 | 46.49 → 46.53 |
| 256 MiB | 49.07 → 49.11 | 48.44 → 48.45 | 46.73 → 46.75 |
| 512 MiB | 49.02 → 49.04 | 48.28 → 48.29 | 46.69 → 46.74 |
| 1024 MiB | 48.98 → 48.98 | 48.20 → 48.20 | 46.63 → 46.64 |
