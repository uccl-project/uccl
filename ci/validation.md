# Migration validation — updated 2026-11-09

PR: [#1131](https://github.com/uccl-project/uccl/pull/1131), branch
`codex/structured-ci`. The three `UCCL_CI_*` secrets work from GitHub on L4,
GB10 (including its jump host) and both restored GH200 nodes. Hardware runs use
isolated validation branches and direct SSH; the new workflow awaits merging.

## PR #1131 follow-up

- At `9d7aada6`, all 27 host suites, including all 35 CI harness tests, and
  aggregate `CI status` passed in
  [GitHub CI](https://github.com/uccl-project/uccl/actions/runs/42715424010), along
  with [both formatting checks](https://github.com/uccl-project/uccl/actions/runs/42715423574).
  Automatic PR runs intentionally skip hardware.
- All 62 example/experimental entries remain disabled; GIN CI is removed.
- CI now records suite timings and checks measured performance against a 90% floor.
- Removed six unused workflow helpers; the five remaining workflows handle CI,
  formatting, manual dispatch, Docker publication and releases.
- Busy-node protection passed live checks: L4 skipped before source upload for
  another user's jobs; GB10 admitted both idle nodes and passed
  `nixl.internode.nccl` on both ranks in **5.54 seconds**, excluding shared setup.
  L4 reported two skipped ranks, zero passes and no performance sample/candidate.
- Stale-CI cleanup preserved an unrelated process in Linux testing. Local checks
  passed with the Linux-only case skipped; registry, formatter and workflow
  lint checks passed. AMD detection uses fixtures; AMD hardware remains offline.

These targeted checks do not replace the full sweep below. A busy-node skip is
visible in artifacts/JUnit but does not fail the aggregate check; it is not
evidence that the selected tests passed. Unreachable SSH remains a failure.

## Busy-node detector correction — 2026-11-09

The old GH200/GB10 workflows checked `nvidia-smi --query-compute-apps=pid`.
The new detector retained GPU context checks but incorrectly treated every
Python/Jupyter process as a running job. GH200_0 had no compute contexts, 0% GPU
utilization and 0 MiB GPU memory; its sleeping Python process used about 1.5%
of one CPU core. That skip was a false positive.

Idle interpreters no longer block CI. GPU contexts (including at 0% utilization),
active CPU/I/O work and explicit distributed launchers still do. CPU sampling now
excludes time spent querying GPU status. All **38 harness tests passed on Linux**,
including idle Python, active/root CPU jobs, waiting launchers and slow GPU queries.
Live checks admitted GH200_0 and L4. L4 passed PCIe, then yielded when another
user's CPU activity started; no unrelated process was killed.

## GH200 pair restoration — 2026-11-09

Both nodes now accept the dedicated CI key, run CUDA 13.2 with driver 595.58.03,
and report active 400G InfiniBand ports. GH200_0 is provisioned; its CI root
points to local `/ssd1/uccl-ci` to avoid the nearly full root disk. Both reused
`uccl` accounts have nonzero UID and unlimited memlock; existing group privileges
were preserved. No host-key or GitHub-secret changes were needed.

[Initial restored-node run](https://github.com/uccl-project/uccl/actions/runs/42709281256)
reached both nodes and passed idle checks, then failed before tests on a
truncated cuBLAS download. Retrying with unchanged dependency hashes succeeded;
GH200_0's `ep.barrier_concurrent` smoke suite passed in 3.86 seconds.
A second attempt skipped all 47 suites when Ubuntu's `landscape-sysinfo` status
helper triggered the workload guard; the helper is now explicitly ignored,
with regression coverage preserving real root/user workload detection.

Across the restored-node sweep and targeted reruns, all **47 GH200 suites** were
attempted: **37 valid command passes, 9 failures and 1 invalid zero-throughput
result**. Performance gate failures are separate. The
[30-suite continuation](https://github.com/uccl-project/uccl/actions/runs/42713846845)
took **15m 13s**; the
[extended calibration](https://github.com/uccl-project/uccl/actions/runs/42718246637)
took **10m 55s**. Earlier runs stopped or skipped for other users' activity;
they are not counted as passes.

The ring include-path fix exposed three source compile failures and a lock
benchmark with no completed CPU operations. The batched benchmark timed out.
Packing's later setup failure came from shared non-PIC build objects; cleaning
EP objects before extension setup fixed it in the
[follow-up](https://github.com/uccl-project/uccl/actions/runs/42719991618).

[Pair calibration](https://github.com/uccl-project/uccl/actions/runs/42713607053)
passed all three benchmarks with three repetitions each in **2m 45s**. Commands
and benchmark workloads were unchanged; medians and source commit `7208846e`
are recorded in `config/baselines.json`.

| GH200 metric | Baseline | Required |
|---|---:|---:|
| EP dispatch + combine | 220.585 µs | <=245.094 µs |
| Pinned-host P2P | 191.58 Gbps | >=172.422 Gbps |
| GPU-buffer P2P | 274.75 Gbps | >=247.275 Gbps |

P2P samples ranged from 147.94–209.16 Gbps (host) and 169.00–281.48 Gbps (GPU),
including values below the new floors. This variability remains visible;
thresholds are not relaxed. The existing L4/GB10 baselines are unchanged.

[Independent verification](https://github.com/uccl-project/uccl/actions/runs/42714441992)
passed all three gates in **1m 46s**: EP **223.16 µs**, host P2P **282.11 Gbps**,
GPU P2P **255.51 Gbps**. Normal single invocations were used; suite times were
14.94s, 6.62s and 6.44s respectively, excluding shared setup.

The extended runs calibrated eight more metrics: GPU copies, GPU-to-CPU latency,
NCCL, Ray read/write, compression, and Triton decode/prefill packing. All use
three unchanged benchmark invocations. Ring and failed FIFO/batched benchmarks
have no valid baseline. Ring metrics select the fixed 128-warp/4-proxy case.

[Independent extended checks](https://github.com/uccl-project/uccl/actions/runs/42719991618)
passed four of six gates; Ray write/read fell below their floors:

| GH200 metric | Measured | Required |
|---|---:|---:|
| Ray write bandwidth | 145.10 Gbps | >=239.580 Gbps |
| Ray read bandwidth | 166.90 Gbps | >=180.504 Gbps |

Benchmark code was unchanged. These are recorded threshold failures; the
measurements vary substantially and no tolerance has been relaxed.

[Packing verification](https://github.com/uccl-project/uccl/actions/runs/42720574548)
passed decode/prefill at **2.908/23.179 ms** in
**1m 06s** including setup. Overall, independent GH200 checks passed
**9/11 metrics**; only the two Ray gates failed.

[New suite times](reports/timings-2026-11-09.csv),
[GitHub job times](reports/runs-2026-11-09.csv) and
[performance samples](reports/performance-2026-11-09.csv) retain each attempt,
including busy-node skips. Local SSH follow-ups are identified separately.

## Confirmed behavior

[November 8 sweep](https://github.com/uccl-project/uccl/actions/runs/42522891573):

| Target | Selected | Initial result | Job time |
|---|---:|---|---:|
| L4 | 46 | 42 passed, 4 failed | 22m 37s |
| GB10 | 26 | 18 valid passes, 6 failed, 2 false passes | 11m 16s |
| GH200 | 47 | Blocked before tests | 1m 11s |

GB10's false passes came from ignored pytest failures and a benchmark that
caught compiler errors. Both are fixed: the Triton unit suite now passes 25
cases with two optional skips, and packing produces valid metrics. Three CUDA
13 benchmark build failures also passed retesting. Across the sweep and fixes,
23/26 GB10 suite commands passed; three fixture failures remain. L4 still has
four fixture failures. Threshold failures are counted separately below.

[Suite times](reports/timings-2026-11-08.csv),
[job times](reports/runs-2026-11-08.csv) and
[performance samples](reports/performance-2026-11-08.csv) record every run and
source commit. Suite time is the slowest participating rank, including builds
inside its command; job time also includes shared setup and collection, but
excludes queue waits. `INVALID` means a zero exit status concealed failure.

[Baselines](config/baselines.json) contain 28 metrics, each calibrated from three
benchmark repetitions, with sample ranges and source commits. Throughput must
be at least baseline × 0.9; latency must be at most baseline / 0.9.

| Metric | Baseline | Required |
|---|---:|---:|
| L4 pinned P2P | 380.44 Gbps | >=342.396 Gbps |
| L4 GPU-buffer P2P | 164.85 Gbps | >=148.365 Gbps |
| GB10 pinned P2P | 127.72 Gbps | >=114.948 Gbps |
| GB10 EP dispatch + combine | 233.945 µs | <=259.939 µs |

P2P uses 1 MiB transfers, 128 iterations. EP uses 32 tokens, hidden size 7168,
top-k 8 and 16 experts. `pcie_bench` on GB10/GH200 measures GPU copies on unified
memory; its GB/s result is not network/link throughput.

Independent checks use normal single benchmark invocations. The November 8
L4/GB10 checks passed **13/17 metrics**; four crossed their thresholds on
unchanged benchmark code:

| Metric | Measured | Required |
|---|---:|---:|
| GB10 batched GPU-to-CPU latency | 11.511 µs | <=8.44 µs |
| GB10 Triton decode packing | 8.006 ms | <=7.987 ms |
| L4 batched GPU-to-CPU latency | 6.505 µs | <=4.439 µs |
| L4 NCCL bandwidth | 100.27 Gbps | >=105.507 Gbps |

These expose same-code variability, not demonstrated code regressions. The
90% thresholds remain enforced. [L4 verification](https://github.com/uccl-project/uccl/actions/runs/42525401610)
and [GB10 verification](https://github.com/uccl-project/uccl/actions/runs/42525804181)
contain the measured failures; the CSV includes earlier checks too.

## Hardware/configuration exclusions established by testing

- GB10 native GPU registration fails; host-staged EP/P2P/NIXL stays enabled.
- L4 lacks peer atomics/NVLink; full native EP needs SM90.
- GDRCopy needs unavailable `/dev/gdrdrv`; userspace libraries are insufficient.
- AMD, EFA, MUSA/Hygon, privileged AF_XDP/DPDK and external-library fixtures
  remain excluded. Exact reasons live in `config/tests.json`.

## Known failures exposed by broader coverage

| Suite | Target | Failure |
|---|---|---|
| `collective.rdma.transport` | L4, GB10, GH200 | Transfers finish; shutdown times out |
| `p2p.compression_unit` | L4, GB10, GH200 | Removed/private fixture APIs |
| `ep.internode_simple` | GB10, GH200 | Outdated address/Buffer APIs |
| `ep.bench.pcie_bench` | L4 | Illegal GPU memory access |
| `nixl.internode.nccl` | L4 | Illegal GPU memory access, then timeout |
| `ep.bench.fifo` | GH200 | Removed CUDA 13 `cudaDeviceProp.clockRate` field |
| `ep.bench.rdma_fifo` | GH200 | Outdated `get_cpu_proxies_meta` call |
| `ep.bench.batched_gpu_to_cpu_bench` | GH200 | Times out after 300s |
| `ep.rb.cas`, `ep.rb.multi_ring` | GH200 | Removed `TransferCmd.is_combine` field |
| `ep.rb.fc` | GH200 | Invalid `record->payload_ size` expression |
| `ep.rb.lock` | GH200 | Zero throughput and GPU/CPU count mismatch; metric rejected |

These stay enabled and visible. The first five failure families have fixes in
separate PR [#1135](https://github.com/uccl-project/uccl/pull/1135), not integrated
here. After integration, update the compression invocation, register its new
NCCL test and rerun the combined suite. GH200's additional FIFO/ring failures
and batched timeout still need investigation. Failed, invalid or skipped
benchmarks receive no baseline.

## Changes made to test harnesses

- Propagate process failures; bound rendezvous, disconnect and descendant cleanup.
- Update IPC/Ray fixtures, architecture handling and pinned NIXL setup.
- Support CUDA 13's clock-rate query; use GB10's CUDA 13 assembler for Triton.
- Invoke Triton tests through pytest with JUnit; reject swallowed benchmark errors.
- Extract metrics, enforce thresholds and report per-suite times. Registry
  validation accounts for 184 suites.
- Supply the root include path for GH200 ring builds and launch FIFO fixtures
  with their distributed environment. Existing benchmark workloads, iteration
  counts and statistics are preserved.
- Clean EP build objects before extension setup: earlier C++ tests write a
  shared `common.o` without the extension's PIC/ABI flags.

## Deployment state and remaining administrator actions

L4/GB10 run as `uccl` with local storage and no sudo, Docker or NFS. GH200's
existing account retains administrator/Docker membership. The restricted jump
account uses the same dedicated CI key. No further secret upload is needed.

1. Set branch protection to `CI status` plus formatting; remove retired checks
   and the legacy jump secret after the new workflow is active.
2. Resolve the visible fixture failures and performance variability before
   treating the hardware/performance checks as reliably green.

See [README.md](README.md#server-provisioning-and-credentials) for setup.
