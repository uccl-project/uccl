# Calibrated NCCL drop-in on eight L4 GPUs

`libuccl_lite_nccl.so` interposes the NCCL ABI. An existing NCCL application
keeps its collective calls, process groups, communication hooks and streams.
The launcher selects the library with `LD_PRELOAD`; SGLang's explicit `dlopen`
uses `SGLANG_NCCL_SO_PATH`. The real NCCL remains installed and supplies
initialization, unsupported operations and fallback. No Torch/SGLang source
patch or fusion hook is required.

This is separate from the historical `libuccl_lite_nccl_compat.so` experiment.
Testbed 1's historical serving/DDP results used that proxy; its FSDP result
required a separate reduce-scatter process group and hook. Testbed 4's serving
and training results used an explicit parameter-sharded application adapter.
Neither figure establishes performance for this implementation.

The [full validation report](data/dropin_8l4_20261115/README.md) includes fresh
serving/training results, raw repetitions, correctness checks and limitations.

## Fresh eight-L4 serving results

GLM-4.5-Air-FP8, SGLang TP8/DP8/EP8, identical eager server flags and prompts,
two fresh servers per backend, three timed repetitions per case/server:

| Requests × input × output tokens | NCCL batch (s) | Drop-in batch (s) | Latency change |
| --- | ---: | ---: | ---: |
| 8 × 1024 × 1 | 3.10662 | 3.10852 | +0.06% |
| 32 × 2048 × 1 | 23.60588 | 19.07899 | −19.18% |
| 8 × 64 × 32 | 4.99765 | 5.01441 | +0.34% |

All measured whole-batch, TTFT and TPOT metrics pass the 5% regression gate.
The prefill output texts match across backends. Decode has repeated-run output
variation in both backends, including different additional text hashes, so
these results do not establish identical decode text or model-quality
equivalence. See the report for the numerical checks and complete scope.

## Fresh eight-L4 training results

On `wxz-l4`, 2026-11-15, real NCCL 2.28.9 versus the calibrated drop-in,
PyTorch 2.9.1+cu128, BF16 parameters/gradients, the same generated-token
637,847,552-parameter transformer used in the earlier training experiment:

| Stock training | NCCL (ms/step) | Drop-in (ms/step) | Latency reduction |
| --- | ---: | ---: | ---: |
| DDP | 470.30 | 400.76 | 14.8% |
| FSDP, default hook and one process group | 699.43 | 624.83 | 10.7% |

NCCL → UCCL → UCCL → NCCL, fresh processes, three warmup steps and 20 measured
full steps per job: **40 timed steps per backend/mode**. Values are medians of
the two job medians, using maximum-rank wall time per step. Independent DDP
job medians are 471.45/469.15 ms (NCCL), 395.58/405.94 ms (UCCL); FSDP's are
700.09/698.78 ms and 624.56/625.09 ms. No best-job selection is used.
All recorded parameter samples match across backends; the maximum complete
loss-history difference is 0.00104. These checks are not a convergence study
or a comparison of every trained parameter.

[Raw training data and summary](data/dropin_8l4_20261115/training/summary.json)
and [local calibration](data/dropin_8l4_20261115/calibration/summary.json) retain
the underlying measurements. The profile accepts **26 of 36** exact shapes;
the remaining ten use NCCL. These are fresh local measurements, separate from
the historical figures.

## Build and launch

From `lite`, with the desired real NCCL headers/library:

```bash
make -C integration/nccl_lite dropin NCCL_HOME=../../build/nccl-2.28.9

# Default: all calls use NCCL, with no packet allocation or worker creation.
python3 integration/nccl_lite/dropin_profile.py \
  --real-nccl build/nccl-2.28.9/lib/libnccl.so.2 -- \
  torchrun --standalone --nproc-per-node=8 train.py

# Use a freshly validated profile to accelerate eligible shapes.
python3 integration/nccl_lite/dropin_profile.py \
  --real-nccl build/nccl-2.28.9/lib/libnccl.so.2 \
  --profile build/my-calibration/profile.tsv -- \
  torchrun --standalone --nproc-per-node=8 train.py
```

The ready profile for this exact host/build is
[`data/dropin_8l4_20261115/profile.tsv`](data/dropin_8l4_20261115/profile.tsv).
Use the recorded `NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_PROFILER_PLUGIN=none`
environment and `/data/wxzheng/kt-venv/bin/python` for the measured setup.

`--real-nccl` must be the library to use in the application as well as fallback;
the launcher preloads that exact file. It does not tune NCCL. On this testbed,
the inherited `NCCL_NET=gIB` environment targets a different machine and fails
even native NCCL initialization. The measurements explicitly set
`NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_PROFILER_PLUGIN=none` for **both** backends, leaving algorithm,
protocol, channels, cuMem and NVLS at their existing defaults.
The profiler override excludes the system's automatically loaded GPUViz plugin;
native NCCL itself aborted when that plugin was initialized again after
communicator teardown. The failed diagnostic remains in the local run logs.

## Selection and compatibility

The profile includes exact operation and NCCL element count, BF16/SUM,
both in-place and out-of-place layouts, and 2/4/8/16-byte pointer alignment.
Calibration requires both independent packet jobs to beat both NCCL controls
by at least 5% for every layout/alignment. No per-size kernel variant sweep is
used. Unsupported sizes are not rounded or interpolated.

The launcher checks GPU UUIDs and PCI topology, driver, CUDA runtime and NCCL
binaries, library binary, CPU topology, launch affinity, packet configuration
and NCCL/CUDA environment. A stale identity means NCCL fallback. Communicator
initialization additionally agrees the profile, host and ordered rank/device
mapping through real NCCL. Acceleration requires eight distinct local
processes on devices 0–7. GPU count alone never enables a packet route.
The validated clients use the same library/profile and matching collective,
group and capture sequences on every rank.

| Call pattern | Behavior |
| --- | --- |
| Calibrated eager BF16 AG, SUM RS/AR, blocking eight-rank communicator | Packet route |
| Multiple caller streams on that communicator | Execute on the caller stream; insert a completion-event dependency when the stream changes |
| Legal in-place AG/RS | Stage the overlapping input before packet traffic |
| BF16 pointers below 16-byte alignment | Stage input/output through aligned buffers; copy cost is included in calibration |
| Other dtype, reduction, extent or communicator size | Real NCCL |
| Missing/mismatched profile; nonblocking or split communicator | Real NCCL |
| Grouped calls before any packet use | Real NCCL; no router preparation |
| Group or graph capture after packet use | Drain prior packet work once; keep this communicator on NCCL thereafter |
| CUDA graph replay | Native NCCL graph semantics; no captured packet counters/workers |

Packet buffers, scratch, workers and events are prepared only when the first
eligible eager call arrives. Preparation completes prior CUDA work and is
outside steady-state measurements. Fallback-only communicators never prepare
the packet transport. Normal teardown releases its resources. An abort with
unfinished packet work retains those resources until process exit to avoid
waiting for a missing peer; in-process fault recovery is not established.
The real NCCL is a dynamic dependency so symbols not interposed here remain
available to both linked callers and `dlopen` consumers.

The multiprocess API keeps its original single-stream entry point and adds
`uccl_lite_mp_enqueue_ordered` for integrations that supply these dependencies.
Worker isolation also preserves the explicitly pinned AR helper thread;
initialization previously moved it back onto the application's scheduler CPUs.

## Reproduce calibration and stock training

Use the same Python/environment, CPU affinity and NCCL settings for calibration
and deployment. Keep checkpoints, extensions and compiled artifacts outside
the source directory. Each output directory must be new.

```bash
export NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_PROFILER_PLUGIN=none
PYTHON=/data/wxzheng/kt-venv/bin/python
$PYTHON integration/nccl_lite/calibrate_dropin.py \
  --real-nccl build/nccl-2.28.9/lib/libnccl.so.2 \
  --out-dir build/my-calibration

$PYTHON integration/nccl_lite/run_dropin_training.py \
  --real-nccl build/nccl-2.28.9/lib/libnccl.so.2 \
  --profile build/my-calibration/profile.tsv \
  --out-dir build/my-training -- --warmup 3 --steps 20
```

`calibrate_dropin.py --cases cases.json` accepts exact `[operation,count]` pairs
(`1=AG`, `2=RS`, `3=AR`). Its candidate profile is explicitly marked and cannot
be used by the deployment launcher. The training runner uses ordinary DDP and
stock FSDP: one communication group, default communication hook, normal bound
process-group initialization and normal `summon_full_params`. Historical
adapter modes remain available only in the older benchmark's explicit options.

Calibration and training use fresh NCCL → UCCL → UCCL → NCCL processes. The
training runner records every full step, rank loss history and parameter
samples and fails a comparison with more than 5% median end-to-end regression.
Serving uses `run_e2e_comparison.py --dropin-profile ... --backends nccl dropin
dropin nccl`, with identical server flags and a pure real-NCCL control.

```bash
$PYTHON integration/nccl_lite/run_e2e_comparison.py \
  --real-nccl build/nccl-2.28.9/lib/libnccl.so.2 \
  --dropin-profile build/my-calibration/profile.tsv \
  --out-dir build/my-serving --backends nccl dropin dropin nccl \
  --cases 8:1024:1 32:2048:1 8:64:32 --repeats 3 --case-warmups 1 -- \
  --mem-fraction-static 0.87 --random-seed 12345 --chunked-prefill-size 8192

$PYTHON integration/nccl_lite/summarize_dropin.py \
  --training-dir build/my-training --serving-dir build/my-serving \
  --out build/my-summary.json
```

Fresh validation and numerical/performance limits are recorded alongside the
local run data; historical figures remain unchanged.
