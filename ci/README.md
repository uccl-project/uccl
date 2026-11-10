# UCCL CI

`config/tests.json` registers suites, `config/targets.json` defines machines,
`catalog.py` selects tests, and `run.py` executes them. GitHub-hosted Ubuntu runs
host tests and orchestrates GPU jobs over SSH as `uccl`, using local
`/var/lib/uccl-ci` storage on each server.

## Where jobs run

| Target | Machines | Hardware | Role |
|---|---|---|---|
| `cpu` | GitHub `ubuntu-24.04` | CPU | Python/C++ host tests and CI self-tests |
| `l4` | `l40`, `l41` | 4 L4 GPUs/node, IB, CUDA 12.9 | P2P, IPC, compression, GPU RDMA, NIXL |
| `gh200` | `gh200_1`, `gh200_0` | 1 GH200/node, 400G InfiniBand, CUDA 13.2 | EP, GPU RDMA, NIXL, EP/P2P benchmarks |
| `gb10` | `gb10_0`, `gb10_1` | 1 GB10/node, RoCE, CUDA 13.0 | Host-staged EP/P2P and NIXL |
| `amd` | Offline | ROCm | Disabled; worker support needs validation |

Both GH200 nodes are provisioned and reachable as of 2026-11-09; both RDMA ports
report active 400G links. Single-node suites use `gh200_1`; two-node suites
require both. Link rate is not application throughput. Per-host interfaces and
RDMA devices are defined in `config/targets.json`.

Hardware limits: GB10 native GPU-buffer registration fails; L4 lacks native
peer atomics/NVLink; no available node has `/dev/gdrdrv`. Affected tests are
excluded explicitly. Host-staged GB10 EP/NIXL remains enabled.

## Current coverage

**184 registered suites: 89 enabled, 95 disabled.** Enabled entries comprise
73 default-tier suites and 16 opt-in performance benchmarks. These are suite
counts, not individual test cases; enabled does not mean passing.

| Enabled family | Entries | What it checks |
|---|---:|---|
| `ci.*` | 1 | Selection, inventory, process cleanup and performance comparisons |
| `p2p.*` | 41 | Host APIs, registration, IPC, RDMA, compression, notifications and benchmarks |
| `ep.*` | 22 | Compatibility, synchronization, communication, Triton packing and benchmarks |
| `mccl.host.*` | 4 | MUSA plugin host fixtures; no GPU execution |
| `collective.*` | 14 | Protocols, congestion control, RDMA transport and CUDA utilities |
| `nixl.*` | 4 | Local IPC and two-node RDMA/NCCL transfers |
| `build.apptainer_flags` | 1 | GPU-option handling with mocks |
| `util.*` | 2 | GPU topology and runtime utilities |

Full selection before changed-file filtering:

| Target | Default (`host,build,integration`) | Opt-in `performance` |
|---|---:|---:|
| `cpu` | 27 | 0 |
| `l4` | 38 | 8 |
| `gh200` | 31 | 16 |
| `gb10` | 20 | 6 |

Target counts overlap. Seven host-tier protocol suites use L4's build
environment. No standalone build-tier entries are enabled.

All 62 entries under `examples/` and `experimental/` are disabled, including
GIN fixtures built through the experimental tree. The remaining 33 exclusions
cover unavailable hardware, unsupported configurations and opted-out fixtures.
Every exclusion has a reason. Disabled tests cannot be forced with `--tests`.
Vendored tests are not automatically enrolled; experimental C++/CUDA formatting
remains checked.

Use `catalog.py list` for the inventory or `catalog.py plan --full` for the
current default selection. See [validation.md](validation.md) for results.

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

| Trigger | Host tests | Hardware tests | Performance benchmarks |
|---|---|---|---|
| PR opened, updated or reopened | Affected CPU suites | Skipped | Skipped |
| Merge or push to `main` | Affected CPU suites | Affected suites | Skipped |
| `/run-ci all` on a PR | All CPU suites | All eligible default-tier suites | Skipped |
| `/run-ci <target> [IDs/globs]` | If target is `cpu` | Selected hardware target | Skipped |
| `/run-benchmark <target> [IDs/globs]` | None registered | Selected benchmarks | Enabled |

Manual runs ignore changed-file filtering. Comment commands require `write`,
`maintain` or `admin` permission and run the PR's exact head SHA at dispatch.
Supported targets: `all`, `cpu`, `l4`, `gb10`, `gh200`.

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

Automatic PR jobs receive no lab credentials. Hardware checks on `main` happen
after merging; use `/run-ci` for pre-merge hardware validation. There is no
nightly or label trigger. Benchmarks do not run correctness suites first.

Host and hardware jobs can run in parallel. Hardware runs serialize per target
and take a machine-local lock. Before source upload, CI checks every required
node for NVIDIA/AMD compute contexts, GPU activity, waiting compute launchers,
and CPU processes using at least 20% of one core. Idle shells, editors, Python
interpreters and notebook servers are ignored unless they own a GPU context;
explicit distributed launchers still reserve the node while waiting.
If any node is busy, another CI run holds its lock, or activity cannot be
verified, the target is **skipped**. SSH connection failures remain failures.

Workers recheck before setup and each suite, and every five seconds while running.
If a real job starts, CI stops its own processes and skips the remaining target.
This is cooperative protection, not a scheduler reservation. Only recorded,
same-user stale CI process groups may be cleaned after acquiring the node lock
and verifying PID identity. Unrecorded jobs are not automatically killed.
Artifacts/JUnit show skip reasons, and skipped benchmarks produce no performance
measurement or baseline candidate. Skips do not fail `CI status`; check the
summary's skipped counts before treating hardware coverage as validated.

Replace retired workflow checks in branch protection with `CI status`, retain
formatting checks, and remove the GIN check.

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
`libgtest-dev`, `libibverbs-dev`. SSH aliases must log in as `uccl`.

```bash
python3 ci/run.py --local --target cpu --output ci-results/host-run
python3 ci/run.py --target l4 --ssh-config /path/to/config --tests 'p2p.rdma_native.*' --output ci-results/rdma-run
python3 ci/run.py --target gh200 --ssh-config /path/to/config --tiers performance --tests p2p.bandwidth --output ci-results/perf-run
python3 ci/performance.py ci-results/perf-run --candidate ci-results/perf-run/baseline-candidate.json
```

Each run uses a fresh source workspace and output directory. Python dependency
inputs and hashed locks live in `requirements/`; external source pins live in
`config/dependencies.json`. Artifacts include the selection plan, results, JUnit,
per-rank logs, source identity and machine/toolchain inventories.

Both ranks must succeed unless the target is skipped for node activity.
Setup failures, timeouts, disconnects and all-skipped pytest suites fail the job;
cancellation stops only owned processes. Workspaces
are removed after artifact collection, even for failed tests. Use `--keep-remote`
to debug; collection failures preserve the workspace automatically. Reusable
caches live in `/var/lib/uccl-ci/cache`; avoid filling L4's limited disk.

## Performance policy and current validation

`ep.bench.low_latency` measures EP dispatch/combine latency on GH200/GB10.
`p2p.bandwidth` uses pinned host buffers; `p2p.bandwidth_gpu` uses GPU buffers on
L4/GH200. Only performance registrations extract and compare metrics.

`config/baselines.json` records measured baselines. A 0.1 tolerance requires 90%
of baseline performance: throughput >= baseline × 0.9; latency <= baseline / 0.9.
Missing, nonfinite or nonpositive measurements and threshold violations fail;
absent baselines are reported as `unbaselined`.
Benchmarks without configured metrics retain logs and exit status. Candidate
baselines are artifacts and never replace active thresholds automatically.
Workflow summaries show each suite's execution time, excluding shared setup.

All 27 GitHub host suites and formatting passed at `16c25c12`. The updated
busy detector passed 38 Linux harness tests and admits idle Python sessions.
Both GH200 nodes run CI; EP and host/GPU P2P passed independent baseline checks.
AMD remains unavailable.

[validation.md](validation.md) records hardware results, timings, baseline
provenance and remaining failures. There are 28 calibrated metrics, including
11 on GH200; independent checks passed 9/11, with Ray read/write below their floors. The November 8
L4/GB10 checks passed 13/17
metrics; four remain noisy. Fixture fixes in #1135 still need integration and
a combined sweep. Enabled suites may fail; disabling one requires a registry
reason. Historical example/experimental results are excluded from coverage.

Developer checks:

```bash
python3 -m unittest discover -s ci/tests -v
python3 ci/catalog.py validate
python3 -m compileall -q ci
# Also run actionlint on .github/workflows/ci.yml and run-benchmark.yml.
```
