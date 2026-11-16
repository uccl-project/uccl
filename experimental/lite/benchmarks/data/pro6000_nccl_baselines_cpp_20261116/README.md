# FP32 collectives with default and tuned SM/CE NCCL on eight RTX PRO 6000 GPUs

All three figures compare **UCCL Lite**, **NCCL (tuned SM/CE)**, and
**NCCL (default)** using fresh five-round measurements. AllReduce's NCCL
baseline is newly calibrated across SM and CE configurations. The earlier
[CE-only AllReduce result](../pro6000_allreduce_ce_ring_cpp_20261116/README.md)
and [AG/RS result](../pro6000_ag_rs_cpp_20261116/README.md) remain historical
records; their timings do not enter these figures.

[Every size](results.md) · [AllReduce samples](allreduce.csv) ·
[AllGather samples](allgather.csv) · [ReduceScatter samples](reducescatter.csv) ·
[frozen policy](policy.json) · [configuration catalog](configurations.json) ·
[raw evidence](measurements.tar.gz) · [provenance](provenance.json)

| Operation | Lite median wins vs tuned | Lite median wins vs default | Sizes where tuned NCCL matches or beats Lite |
| --- | ---: | ---: | --- |
| AllReduce | 18/21 | 21/21 | 2 MiB, 4 MiB, 8 MiB |
| AllGather | 21/21 | 21/21 | none |
| ReduceScatter | 21/21 | 21/21 | none |

At 1 GiB, bus throughput in GB/s is:

| Operation | UCCL Lite | NCCL tuned SM/CE | NCCL default |
| --- | ---: | ---: | ---: |
| AllReduce | 51.85 | 50.68 | 21.63 |
| AllGather | 51.85 | 50.05 | 24.61 |
| ReduceScatter | 51.80 | 49.72 | 18.37 |

These are experimental standalone measurements. All points, including losses,
remain in the report. Points are medians; only UCCL Lite has observed-range
shading. Both NCCL series are unshaded. Full ranges remain in the CSVs.

Tuned NCCL ReduceScatter at 4 MiB varies from 104.52 to 195.95 µs across the
five runs, with a 157.80-µs median. That measured spread produces the visible
throughput dip; all five samples are retained.

## Measurement

Hardware: eight NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, dual-socket
AMD EPYC 9B45, 384 logical CPUs, four GPUs per NUMA node, no NVLink. XOR-1 peers
share a PCIe switch; XOR-2 connects switches on a socket; XOR-4 crosses sockets.
Driver 580.178.04, CUDA compiler 13.3.73/SM120, NCCL 2.28.9, and the same MSCCL++
installation as the prior runs. [Platform details](platform.json) record GPU
UUIDs, CPU/topology identity, environment, and library hashes.

The matched C++ harnesses call NCCL directly; they are not nccl-tests.
AllReduce is FP32 in-place SUM. AllGather and ReduceScatter are FP32
out-of-place, with bytes denoting the full output/input per rank respectively.
All 21 powers of two from 1 KiB to 1 GiB are reported. Bus factors are 7/4 for
AllReduce and 7/8 for AllGather/ReduceScatter.

Every fresh eight-process job allocates full 1-GiB tensor capacity per rank.
Each size uses ten warmups and twenty continuously queued operations.
`steady_clock` encloses enqueue through final stream completion, and the
slowest rank supplies the sample. Required staging, communication, and stream
joins are inside timing. Allocation, registration, initialization, checking,
CPU-affinity enforcement, and barriers are outside. Every live thread is
restricted to the same four physical CPU cores per rank after warmup and
checked after timing. Nine changing-input/reuse checks plus a complete check
after each timed batch validate every output and guard; AG/RS also check that
inputs remain unchanged. Timed intermediate results are not copied out.

## Calibration and independent validation

AllReduce NCCL calibration uses 48 configurations in three deterministically
shuffled rounds: the prior 26 CE configurations, 14 SM automatic/Ring
configurations, seven SM Tree configurations, and default. SM tuning includes
LL/LL128/Simple, channels, and thread counts; CE tuning includes channels,
transfer buffers, and algorithms. The lowest three-round median selects each
exact size among SM/CE candidates. Default stays a separate untuned curve.

UCCL Lite's three per-size policies and the tuned AG/RS policies are reused
from the source-matched earlier calibration; they are not retuned here.
All policies are frozen before independent validation. Each selected fixed
configuration sweeps from its lowest to highest selected size, including
intermediate powers of two, with full 1-GiB capacity. Only frozen
size/configuration pairs form the curves. Five fresh-process rounds alternate
operation and backend order. No validation point selects a configuration.

NCCL default has no tuning or network overrides: inherited `NCCL_*` variables
are removed, and both `/etc/nccl.conf` and `~/.nccl.conf` are absent. Only debug
logging controls are set. All timed runs, including calibration, use
initialization-only `INIT,ENV,GRAPH` logging. Separate diagnostic sweeps of
every selected NCCL/default configuration add per-operation `TUNING` logs to
verify actual per-size algorithm/protocol dispatch. Every timed job retains
its own eight-rank transport evidence. Forced CE must use actual P2P/CE edges;
forced SM must not use CE; forced P2P must not silently fall back to SHM.
The default diagnostics show both `P2P/CUMEM` and `SHM/direct/direct` transport
for all three collectives; the tuned candidates explicitly enable system-wide
P2P and disable SHM.

The audit covers **144 calibration jobs**,
**28 diagnostic jobs**, and **305 validation jobs**,
with **337,600 complete per-rank checks** and
**2,496 NCCL rank logs**. All checks passed.

All four CUDA targets were rebuilt through the Makefile. CUDA sources match
the earlier implementations and their recorded four-L4 correctness checks;
this rerun changes no collective kernel. It adds no graph, multi-node,
production-dispatch, or integration claim.

## Reproduction

Build the four `pro6000-ar`, `pro6000-ar-reference`, `pro6000-ag-rs`, and
`pro6000-ag-rs-reference` Makefile targets with `CUDA_ARCH=sm_120` and the
recorded MSCCL++/NCCL installations. Source the isolated testbed runtime
environment and export `LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64"`.
The archive includes the exact compiler commands and version.

```sh
python3 -B scripts/run_pro6000_nccl_baselines.py run \
  --output build/new-pro6000-baselines \
  --ar-policy benchmarks/data/pro6000_allreduce_ce_ring_cpp_20261116/frozen-policy.json \
  --ag-rs-policy benchmarks/data/pro6000_ag_rs_cpp_20261116/policy.json
```

The runner holds the shared GPU lock and refuses to overwrite an experiment.
For offline replay, use fresh directories from `experimental/lite`:

```sh
mkdir -p build/pro6000-baselines-replay
tar -xzf benchmarks/data/pro6000_nccl_baselines_cpp_20261116/measurements.tar.gz \
  -C build/pro6000-baselines-replay
python3 -B scripts/run_pro6000_nccl_baselines.py summarize \
  --input build/pro6000-baselines-replay/epoch \
  --output build/pro6000-baselines-report
python3 -B scripts/plot_pro6000_nccl_baselines.py \
  --summary build/pro6000-baselines-report --preview
```

The archive contains all raw logs, exact commands/environments, source
snapshots, build evidence, frozen policies, and hashes. Compiled binaries and
dependency trees are excluded.

The published auditor normalizes dispatch records to JSON lists for replay.
The measured runner snapshot and this serialization-only audit correction
are both archived; timing, policy selection, and CUDA code are unchanged.
