# Original UCCL Lite C++ benchmark rerun on PRO6000

Measured on `wxz-pro6000`, 2026-11-16, using the **byte-identical original
standalone CUDA source** from the PRO5000 experiment. This is the pre-EAT
standalone implementation, not the c03 vLLM extension. New figures and derived
tables label this implementation **UCCL Lite**; historical source and raw logs
retain the recorded `PCIeCCL` name.

[Complete auto-path latency table](results.md) · [all-mode summary CSV](summary.csv)

## Method preserved from the PRO5000 run

- Eight ranks, one forked C++ process per GPU, in-place FP32 SUM on the internal
  registered communication buffer. No Python enqueue, caller input/output
  staging copies or CUDA Graph replay enter the timed loop.
- Original launch settings: LL 1×1024 threads, PUT 16×256, CE 16×512. The
  `auto` policy uses LL through 32 KiB, PUT above that and CE from 4 MiB.
  Consequently the requested 1-KiB–1-MiB auto sweep uses LL and PUT.
- Every power of two from 1 KiB to 1 MiB, inclusive. All four original modes
  are requested with `-m all`; explicit LL skips sizes above its 32-KiB cutoff.
  This gives 39 size/mode rows per run, including forced CE measurements.
- Ten warmups, then 20 consecutive launches. The original C++ host
  `steady_clock` encloses those launches and the final stream synchronization;
  elapsed time is divided by 20. The reported value is the maximum of the eight
  per-rank averages. Input initialization, registration, setup and verification
  are outside timing. This is a batch-average completed-call measurement,
  not a distribution of individually timed call latencies.
- Five independent complete ascending sweeps, each in a fresh benchmark
  process. Summary values are the median of those five reported run averages;
  the minimum/maximum is the observed run range, not a confidence interval.
  No fastest-run or best-variant-per-size selection is used for the auto table.
- The original checker fills position/rank-dependent FP32 input, performs one
  correctness call and checks the full output before timing each size/mode.
  All **195 size/mode checks** (1,560 rank checks) report zero maximum relative
  error. The original timed loop repeatedly reduces its buffer without
  resetting or verifying each timed invocation; that behavior is preserved.

The requested byte-range parameters differ from the historical full sweep:
`-b 1024 -e 1048576` instead of `-b 128 -e 1073741824`. The original source
sizes its internal buffer regions from `-e`, so this run uses 1-MiB region
capacity. The source's CPU pinning ranges (0–83 and 84–167) are preserved
unchanged; they were not retuned for this host's 96-core NUMA nodes.

## Hardware, software and identities

- Eight NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, dual-socket AMD
  EPYC 9B45, four GPUs per NUMA node. XOR-1 peers are PIX, XOR-2 NODE and
  XOR-4 SYS. Actual hardware identity and topology are in `raw/`.
- CUDA compiler 13.3.73, SM120, `-O3 -std=c++20`, matching the original
  build script's flags. The existing isolated MSCCL++ install is reused;
  its installed `version.hpp` reports 0.9.0 / commit `6231b4f38ddc`, while
  the dependency source checkout is tagged `v0.10.0` with local patches.
  The installed header contents, header hashes and actual shared-library hash
  are retained in [manifest.json](raw/manifest.json), rather than inferred
  from a directory or tag name.
- Unchanged CUDA source SHA-256:
  `80d7cb3d121121468744d872e7cc149ffcf83b7856a19b8059f3370bda0db3bb`.
  Source/binary identities are checked before and after the measurements.
- All jobs hold `/home/wxzheng/uccl-eval.lock` and check for other GPU compute
  processes before each run. No clock or system-wide configuration is changed.
  CUDA devices are exposed in physical order 0–7.

The original program **hardcodes “PRO5000” in its stdout header**. That line
is retained verbatim; `raw/gpu-identity.txt` identifies the actual PRO6000 GPUs.
The first launch failed before CUDA execution because the isolated toolkit's
`libcudart.so.13` was not on the loader path. Its log is preserved separately
as `raw/setup-loader-failure.txt`. The accepted five runs set the recorded
CUDA `LD_LIBRARY_PATH`; no benchmark source or algorithm was changed.

The historical PRO5000 numbers are included as a separate reference. Hardware,
dependency identity, measurement window and maximum buffer capacity differ;
this comparison cannot isolate a hardware effect or establish EAT speedup.
Likewise, these numbers must not be directly divided by the later BF16/Python
per-call c03 measurements to infer an EAT regression.

## Files and reproduction

- `measured_source/mscclpp_allreduce.cu`: original CUDA source, unchanged.
- `measured_source/build.sh`, `run_full.sh`: original build/full-sweep scripts.
- `raw/run-1.txt` through `run-5.txt`: every measured mode and correctness field.
- `raw/manifest.json`, build/loader logs and hardware records: actual commands,
  environment overrides, compiler, linked libraries, identities and run hashes.
- `pro5000_historical.txt`: unchanged historical PRO5000 data, separate from the
  fresh PRO6000 measurements.
- `provenance.json`: SHA-256 hashes for packaged source and measurement records.
- `run.py`: isolated runner using the original CLI and original checker.
- `summarize.py`: dependency-free archive/coverage/hash verification and summary.

From `experimental/lite`, with the pinned dependency/toolchain installed:

```bash
make pcieccl-original-latency CUDA_ARCH=sm_120 \
  CUDA_HOME=/path/to/cuda13.3 MSCCLPP_HOME=/path/to/mscclpp/install
python3 benchmarks/data/pro6000_original_cpp_latency_20261116/run.py \
  --experiment /path/to/eat-pcieccl-vllm-pro6000-20261115-074910
```

The runner expects the recorded experiment's isolated toolchain layout and a
fresh `build/original-cpp-latency/` output directory. Generated binaries and
measurement output stay under `build/`. Source dependencies are not vendored.

Verify the packaged measurements and regenerate the CSV/table without GPUs:

```bash
make pcieccl-original-latency-summary
```
