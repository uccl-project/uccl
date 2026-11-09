# Migration validation — 2026-11-08

PR: [#1131](https://github.com/uccl-project/uccl/pull/1131), branch
`codex/structured-ci`. The three `UCCL_CI_*` secrets work from GitHub on L4 and
GB10, including the jump host. Hardware runs used the isolated
`codex/ci-validation-20261108` branch while the new workflow awaits merging.

## PR #1131 follow-up

- All 27 host suites and both formatting checks passed after installing
  `libibverbs-dev` and aligning Black with version 26.5.1.
- All 62 example/experimental entries remain disabled; GIN CI is removed.
- CI now records suite timings and checks measured performance against a 90% floor.
- Removed six unused workflow helpers; the five remaining workflows handle CI,
  formatting, manual dispatch, Docker publication and releases.
- Busy-node protection passed live checks: L4 skipped before source upload for
  another user's jobs; GB10 admitted both idle nodes and passed
  `nixl.internode.nccl`. Skips produce no performance sample or baseline candidate.
- All 34 CI harness tests passed on GB10 Linux, including recorded stale-CI
  cleanup while preserving an unrelated process. AMD detection uses fixtures;
  no new GH200/AMD hardware run was possible.

## Planned GH200 pair restoration

Both GH200 nodes are unreachable. The GitHub job lost SSH to `gh200_1` during
source upload after 71 seconds; none of its 47 selected suites ran.
`gh200_0` still needs provisioning. CUDA 13.2 and the 400G RDMA link remain
configured assumptions; there is no GH200 pair performance baseline.
Earlier single-node results do not establish current availability.

## Confirmed behavior

[Full GitHub sweep](https://github.com/uccl-project/uccl/actions/runs/42522891573):

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

[Baselines](config/baselines.json) contain 17 metrics, each calibrated from three
benchmark repetitions, with sample ranges and source commits. Throughput must
be at least baseline × 0.9; latency must be at most baseline / 0.9.

| Metric | Baseline | Required |
|---|---:|---:|
| L4 pinned P2P | 380.44 Gbps | >=342.396 Gbps |
| L4 GPU-buffer P2P | 164.85 Gbps | >=148.365 Gbps |
| GB10 pinned P2P | 127.72 Gbps | >=114.948 Gbps |
| GB10 EP dispatch + combine | 233.945 µs | <=259.939 µs |

P2P uses 1 MiB transfers, 128 iterations. EP uses 32 tokens, hidden size 7168,
top-k 8 and 16 experts. GB10's `pcie_bench` is a GPU-copy microbenchmark on
unified memory; its GB/s result is not network/link throughput.

Independent checks use normal single benchmark invocations. The latest checks
passed **13/17 metrics**; four crossed their thresholds on unchanged benchmark
code:

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
| `collective.rdma.transport` | L4, GB10 | Transfers finish; shutdown times out |
| `p2p.compression_unit` | L4, GB10 | Removed/private fixture APIs |
| `ep.internode_simple` | GB10 | Outdated address/Buffer APIs |
| `ep.bench.pcie_bench` | L4 | Illegal GPU memory access |
| `nixl.internode.nccl` | L4 | Illegal GPU memory access, then timeout |

These stay enabled and visible. Fixes are in separate PR [#1135](https://github.com/uccl-project/uccl/pull/1135);
the combined suite still needs retesting after integration.
Broken benchmarks have no fabricated baseline;
GH200-only benchmarks remain uncalibrated. Performance-only verification runs
excluded known crashing fixtures after the full sweep had already tested them.

## Changes made to test harnesses

- Propagate process failures; bound rendezvous, disconnect and descendant cleanup.
- Update IPC/Ray fixtures, architecture handling and pinned NIXL setup.
- Support CUDA 13's clock-rate query; use GB10's CUDA 13 assembler for Triton.
- Invoke Triton tests through pytest with JUnit; reject swallowed benchmark errors.
- Extract metrics, enforce thresholds and report per-suite times. Registry
  validation accounts for 184 suites.

## Deployment state and remaining administrator actions

L4/GB10 run as `uccl` with local storage and no sudo, Docker or NFS. GH200's
existing account retains administrator/Docker membership. The restricted jump
account uses the same dedicated CI key. No further secret upload is needed.

1. Restore both GH200 nodes, provision `gh200_0`, then validate the pair.
2. Set branch protection to `CI status` plus formatting; remove retired checks
   and the legacy jump secret after the new workflow is active.
3. Resolve the visible fixture failures and performance variability before
   treating the hardware/performance checks as reliably green.

See [README.md](README.md#server-provisioning-and-credentials) for setup.
