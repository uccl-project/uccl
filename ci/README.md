# UCCL CI

`config/tests.json` registers suites, `config/targets.json` defines machines,
`catalog.py` selects tests, and `run.py` executes them. GitHub-hosted Ubuntu runs
CPU tests and orchestrates GPU jobs over SSH as `uccl`, using local
`/var/lib/uccl-ci` storage on each server.

## Where jobs run

| Target | Machines | Resources | Role |
|---|---|---|---|
| `cpu` | GitHub `ubuntu-24.04` | CPU | Python/C++ CPU tests and CI self-tests |
| `l4` | `l40`, `l41` | 4 L4 GPUs/node, IB, CUDA 12.9 | P2P, IPC, compression, GPU RDMA, NIXL |
| `gh200` | `gh200_1`, `gh200_0` | 1 GH200/node, 400G InfiniBand, CUDA 13.2 | EP, GPU RDMA, NIXL, EP/P2P benchmarks |
| `gb10` | `gb10_0`, `gb10_1` | 1 GB10/node, RoCE, CUDA 13.0 | Host-staged EP/P2P and NIXL |
| `amd` | Offline | ROCm | Disabled; worker support needs validation |

CPU setup adds kernel.org, when reachable, as the preferred entry in the runner's
`/etc/apt/apt-mirrors.txt`, retaining its default fallback mirrors. APT uses
bounded retries/timeouts and an eight-minute step limit. Setup output is saved
in the job log and the `ci-cpu` artifact as `cpu-setup.log`.

Both GH200 nodes are provisioned and reachable as of 2026-11-09; both RDMA ports
report active 400G links. Single-node suites use `gh200_1`; two-node suites
require both. Link rate is not application throughput. Per-host interfaces and
RDMA devices are defined in `config/targets.json`.

GPU limits: GB10 native GPU-buffer registration fails; L4 lacks native
peer atomics/NVLink; no available node has `/dev/gdrdrv`. Affected tests are
excluded explicitly. Host-staged GB10 EP/NIXL remains enabled.

## Current coverage

**185 registered suites: 90 enabled, 95 disabled.** Enabled entries comprise
74 default-tier suites and 16 opt-in performance benchmarks. These are suite
counts, not individual test cases; enabled does not mean passing.

| Enabled family | Entries | What it checks |
|---|---:|---|
| `ci.*` | 1 | Selection, inventory, process cleanup and performance comparisons |
| `p2p.*` | 42 | CPU APIs, registration, IPC, RDMA, compression, notifications and benchmarks |
| `ep.*` | 22 | Compatibility, synchronization, communication, Triton packing and benchmarks |
| `mccl.cpu.*` | 4 | MUSA plugin CPU fixtures; no GPU execution |
| `collective.*` | 14 | Protocols, congestion control, RDMA transport and CUDA utilities |
| `nixl.*` | 4 | Local IPC and two-node RDMA/NCCL transfers |
| `build.apptainer_flags` | 1 | GPU-option handling with mocks |
| `util.*` | 2 | GPU topology and runtime utilities |

Full selection before changed-file filtering:

| Target | Default (`cpu,build,integration`) | Opt-in `performance` |
|---|---:|---:|
| `cpu` | 34 | 0 |
| `l4` | 32 | 8 |
| `gh200` | 32 | 16 |
| `gb10` | 21 | 6 |

Target counts overlap. All 34 CPU suites run on GitHub. Seven protocol suites
install pinned CUDA headers/runtime for compilation, without requiring a GPU or
driver. No standalone build-tier entries are enabled.

All 62 entries under `examples/` and `experimental/` are disabled, including
GIN fixtures built through the experimental tree. The remaining 33 exclusions
cover unavailable devices, unsupported configurations and opted-out fixtures.
Every exclusion has a reason. Disabled tests cannot be forced with `--tests`.
Vendored tests are not automatically enrolled; experimental C++/CUDA formatting
remains checked.

Use `catalog.py list` for the inventory or `catalog.py plan --full` for the
current default selection. [Latest results](reports/latest.csv) include suite timings and performance.

## Selection and explicit registration

```bash
python3 ci/catalog.py validate
python3 ci/catalog.py list
python3 ci/catalog.py plan --base "$(git merge-base main HEAD)" --head HEAD --output /tmp/ci-plan.json
python3 ci/catalog.py plan --full --targets l4,gb10 --tests 'p2p.*,ep.low_latency' --exclude 'p2p.bench.*'
python3 ci/catalog.py plan --full --tiers performance --targets gh200
```

Register each test in `config/tests.json` with an ID, sources, enabled state,
tier, affected paths, targets/capabilities, resource needs, setup, timeout and
command. Disabled entries require a reason; `support` identifies non-runnable
helpers. Validation rejects unregistered first-party test sources and stale
source references.

Selection matches changed paths, tiers, targets and capabilities. Shared
headers/build/CI changes, unknown non-documentation changes or unavailable diff
bases select all eligible suites. PRs compare against their merge base; pushes
compare against the previous SHA. Use `--full` for uncommitted changes.

`--tests` overrides changed-file filtering; `--exclude` wins. Neither enables
disabled entries. Invalid or empty explicit selections fail. `plan.json` records
inclusion and exclusion reasons.

## GitHub entry points

Five workflows remain in `.github/workflows`; six unused helpers were removed:

| Workflow | Purpose |
|---|---|
| `ci.yml` | Plan and run suites; publish aggregate **CI status** |
| `run-benchmark.yml` | Dispatch `/run-ci` and `/run-benchmark` comments |
| `clang-format-check.yml` | Check C++ and Python formatting |
| `build-docker-images.yml` | Publish build images for releases, Dockerfile updates or manual runs |
| `release.yml` | Build wheels/source archives and publish packages |

| Trigger | CPU tests | GPU tests | Performance benchmarks |
|---|---|---|---|
| PR opened, updated or reopened | Affected CPU suites | Skipped | Skipped |
| Merge or push to `main` | Affected CPU suites | Affected suites | Skipped |
| `/run-ci all` on a PR | All CPU suites | All eligible default-tier suites | Skipped |
| `/run-ci <target> [IDs/globs]` | If target is `cpu` | Selected GPU target | Skipped |
| `/run-benchmark <target> [IDs/globs]` | None registered | Selected benchmarks | Enabled |

Manual runs ignore changed-file filtering. Comment commands require `write`,
`maintain` or `admin` permission and run the PR's exact head SHA at dispatch.
Supported targets: `all`, `cpu`, `l4`, `gb10`, `gh200`.

GPU jobs queue per target (up to 100 waiting runs), so newer requests do not
cancel waiting runs. `CI status` reports job outcomes; cancelled work is incomplete.

Each authorized command creates one bot comment, updated from queued to running
and then completed, failed, cancelled or skipped, with a run/log link. Forked
PRs use the same permission check and exact commit; only separate jobs running
trusted workflow code can update comments. Dispatch failures update the reply too.

```text
/run-ci all
/run-ci l4 p2p.*
/run-ci gh200 ep.*,p2p.rdma_native.*,nixl.*
/run-benchmark gh200 ep.bench.low_latency,p2p.bandwidth,p2p.bandwidth_gpu
```

Use `/run-benchmark` (singular). Actions UI inputs are `targets`, `tests`,
`exclude`, `tiers`, `sha` and `head_repo`; benchmarks require the `performance`
tier. Results and logs appear in workflow summaries/artifacts. Comment commands
use the default-branch workflow, so changes take effect after merge.

Automatic PR jobs receive no lab credentials. GPU checks on `main` happen
after merging; use `/run-ci` for pre-merge GPU validation. There is no
nightly or label trigger. Benchmarks do not run correctness suites first.

- CPU suites run on GitHub; GPU/RDMA suites run on lab nodes.
- GPU runs serialize per target. Busy GPUs, held locks or unverifiable GPU
  activity skip the target; CPU-only work does not block it.
- CI rechecks GPU activity during execution and stops its own work if another job
  starts. Only verified stale CI processes may be cleaned up.
- Unreachable lab SSH skips the target with a reason; recorded test failures still
  fail CI. Skips are reported, not counted as passes.
- Require `CI status` and formatting in branch protection; remove retired checks.

## Server provisioning and credentials

Prepare Python 3.12+, GPU drivers and the CUDA version in `config/targets.json`,
then run once as administrator on each test server:

```bash
sudo bash ci/provision.sh "$(cat /path/to/uccl-ci.pub)"
```

This creates `uccl`, installs native dependencies, grants GPU access and
unlimited memlock, and creates local storage. CI jobs use no sudo, Docker or
NFS. Existing privileges are preserved: GH200's reused `uccl` account still has
administrator/Docker access and is not an isolation boundary.

GH200_0's `/var/lib/uccl-ci` points to `/ssd1/uccl-ci` on its local SSD because
the root disk is nearly full. Existing host keys and the three CI secrets work
for both nodes. GH200 EP keeps `UCCL_EP_DISABLE_GH200=1` until the native
managed-memory RDMA path is validated.

Provision the GB10 jump host with the same public key:

```bash
sudo bash ci/provision-jump.sh "$(cat /path/to/uccl-ci.pub)"
```

This creates a separate `uccl` account for proxying to `localhost:57676` and
`localhost:57677`, with shell commands, PTYs and agent forwarding disabled.
In `github-config`, `p4_gate` uses `uccl` and `~/.ssh/uccl-ci`; GB10 aliases use
`ProxyJump p4_gate`. Host checking remains strict.

Keep the credential bundle outside the repository. Upload these three repository
Actions secrets, manually or with an authorized `gh` token:

| Secret | Bundle file |
|---|---|
| `UCCL_CI_SSH_KEY` | `id_ed25519` — same private key for test and jump hosts |
| `UCCL_CI_SSH_CONFIG` | `github-config` — SSH aliases and routing |
| `UCCL_CI_KNOWN_HOSTS` | `known_hosts` — verified server keys |

```bash
bash ci/install-secrets.sh /path/to/private-bundle uccl-project/uccl
```

Use file contents as secret values. No server password is needed. Re-upload the
SSH config for the jump-host migration; remove `UCCL_DEV_SSH_KEY` after the new
workflow is active.

## Local execution and results

Controllers need Python 3.12+ and SSH/rsync on macOS/Linux. Local CPU execution
requires Linux and Ubuntu packages `build-essential`, `python3-venv`,
`libgtest-dev`, `libibverbs-dev`, `libgflags-dev`, `zlib1g-dev` and `libelf-dev`.
SSH aliases for GPU execution must log in as `uccl`.

```bash
python3 ci/run.py --local --target cpu --output ci-results/cpu-run
python3 ci/run.py --target l4 --ssh-config /path/to/config --tests 'p2p.rdma_native.*' --output ci-results/rdma-run
python3 ci/run.py --target gh200 --ssh-config /path/to/config --tiers performance --tests p2p.bandwidth --output ci-results/perf-run
python3 ci/performance.py ci-results/perf-run --candidate ci-results/perf-run/baseline-candidate.json
```

Results include the selection plan, logs, JUnit, source/machine metadata and
suite timings. Dependency locks live in `requirements/`; source pins are in
`config/dependencies.json`.

Both ranks must pass. Test/build failures, timeouts and all-skipped pytest suites
fail CI. Unreachable SSH skips lab coverage. Workspaces are cleaned after collection;
use `--keep-remote` to debug. Caches live in `/var/lib/uccl-ci/cache`.

To find failure logs on a GitHub run page:

1. Open the **CPU** or **GPU** job → **Summarize results** for suite outcomes,
   timings and skip reasons. The same table appears on the run **Summary** page.
2. Expand **Show test logs**, then a suite's log group, for its last 200 lines.
3. For complete files, download `ci-cpu` or `ci-gpu-<target>` under **Summary → Artifacts**.
   `<suite-id>.log` holds test output; `environment.log` / `build-*.log` hold setup
   output; `node-*.ssh.log` holds worker/SSH output. `performance.json` explains gates.

## Performance policy and current validation

`ep.bench.low_latency` measures EP dispatch/combine latency on GH200/GB10.
`p2p.bandwidth` uses pinned host buffers; `p2p.bandwidth_gpu` uses GPU buffers on
L4/GH200. Only performance registrations extract and compare metrics.

[`config/baseline.json`](config/baseline.json) stores 28 measured baselines,
their provenance and per-metric `gate` policy:

| Gate | Below the 90% performance bar |
|---|---|
| `hard` | Fails CI; used for stable metrics |
| `soft` | Reports a warning; requires a reason for expected variability |

Throughput must be >= baseline × 0.9; latency must be <= baseline / 0.9.
The 14 soft gates cover short/noisy probes and observed unstable calibrations;
14 stable gates remain hard. Benchmark errors and missing/invalid measurements
always fail, including for soft gates. Uncalibrated metrics are `unbaselined`.

[Latest results](reports/latest.csv) contain targeted suite results and available
metric measurements, with dates and run/source references. Candidates never replace
active baselines automatically; new candidates start soft until reviewed.
Benchmark workloads and baseline values are unchanged.

Developer checks:

```bash
python3 -m unittest discover -s ci/tests -v
python3 ci/catalog.py validate
python3 -m compileall -q ci
# Also run actionlint on .github/workflows/ci.yml and run-benchmark.yml.
```
