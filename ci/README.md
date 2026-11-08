# UCCL CI

CI has one inventory (`tests.json`), one hardware inventory (`targets.json`), one
change selector (`catalog.py`), and one runner (`run.py`). GitHub-hosted Ubuntu
machines orchestrate jobs. GPU tests execute over SSH on the lab servers as
`uccl`, in separate local copies under `/var/lib/uccl-ci`.

## Where jobs run

| Target | Machines | Hardware | Role |
|---|---|---|---|
| `cpu` | GitHub `ubuntu-24.04` | CPU | Python regressions, C++ host tests, CI self-tests |
| `l4` | `l40`, `l41` | 4 L4 GPUs per node, IB, CUDA 12.9 | Native P2P, IPC, compression, two-node GPU RDMA, collectives |
| `gh200` | `gh200_1` | 1 GH200, IB, CUDA 13.2 | Single-GPU tests, EP barriers, CUDA utilities |
| `gb10` | `gb10_0`, `gb10_1` | 1 GB10 per node, RoCE, CUDA 13.0 | Two-node EP, host-staged communication, NIXL |
| `amd` | Offline | ROCm | Explicitly disabled |

`gh200_0` is not in any enabled target. L4 has GPU peer copies but no native
peer atomics/NVLink. GB10's native GPU memory registration failed both ordinary
verbs and DMA-BUF validation; native GPU RDMA tests therefore select L4. This
does not exclude GB10 host-staged EP or NIXL tests. GH200 has GDRCopy userspace
libraries but no `/dev/gdrdrv`; tests requiring that kernel driver are disabled
with reasons. These capabilities are separate from the presence of an RDMA NIC.

## Test families and tiers

| Family | What it checks |
|---|---|
| `ci.*` | Selection, inventory auditing, subprocess cleanup, cancellation, exit codes, baseline comparisons |
| `p2p.host.*` | Backends, Python wrappers, memory registration and IPC metadata, signals, endpoint/control logic |
| `p2p.*` | Native reads/writes, IPC, shutdown, device binding, compressed transfers, notifications and bandwidth |
| `ep.host.*`, `ep.*` | Configuration and compatibility, event ordering, barriers, dispatch/combine, low-latency and dual-mode communication |
| `mccl.host.*` | MUSA plugin ABI and behavior using host-side fixtures; no MUSA GPU claim |
| `collective.*` | RDMA protocol utilities, transport, NCCL plugin process groups, congestion-control algorithms |
| `gin.local.*` | Device API, compile-fail contracts, queues, cooperative operations, NCCL-EP adapter, routing and model fixtures |
| `ukernel.*` | Transport, device and collective components, Python bindings and communication paths |
| `lite.*` | UCCL-Lite EP and collective builds, correctness fixtures and staged-transfer benchmarks |
| `misc.*` | PCIe, NUMA and peer-copy measurements |

Tiers are `host`, `build`, `integration`, and `performance`. The first three are
the default. Compile-only success is reported in `build`; it is not runtime
coverage. Performance jobs require explicit selection of that tier. No nightly
schedule is enabled by this change. The inventory includes disabled hardware,
legacy fixtures, and opted-out comparisons, each with an explanation.

Tests and benchmarks under `examples/` and `experimental/` are explicitly
disabled, including GIN fixtures that build through the experimental tree.
They remain in the registry for auditing, but neither automatic selection nor
an explicit `--tests` request runs them. The separate GIN CUDA workflow is
removed as part of this exclusion. Formatting still checks experimental C++/CUDA.

`python3 ci/catalog.py list` is the complete executable inventory. Tests are
registered at suite granularity; a suite can contain many pytest or C++ cases.
Vendored upstream test trees are not implicitly enrolled. Specific NCCL-EP
fixtures are registered where used by UCCL's GIN adapter.

## Selection and explicit registration

```bash
python3 ci/catalog.py validate
python3 ci/catalog.py list
python3 ci/catalog.py plan --base "$(git merge-base main HEAD)" --head HEAD --output /tmp/ci-plan.json
python3 ci/catalog.py plan --full --targets l4,gb10 --tests 'p2p.*,ep.low_latency' --exclude 'p2p.bench.*'
python3 ci/catalog.py plan --full --tiers performance --targets l4
```

Each `tests.json` entry declares its ID, source files, enabled state, tier,
affected paths, compatible targets, required capabilities, nodes, GPUs per
node, setup components, timeout, and command. A disabled entry must explain why.
`support` accounts for helpers that are not independently runnable. The audit
fails when a new first-party test/benchmark source lacks an explicit decision,
or a registered source is removed without updating its entry.

Selection intersects changed paths, tier, enabled state, requested targets and
capabilities. Shared headers/build/CI changes select all eligible suites.
Unknown non-documentation changes and missing/unavailable diff bases fail open
to full coverage. Local revision-based plans compare commits; use `--full` when validating uncommitted edits.
GitHub uses the PR merge base for PR selection, and the previous
push SHA for pushes. `plan.json` records the reason for every included or
excluded test, including offline targets. A misspelled test selector or an
explicit selection with no eligible tests is an error. `--tests` overrides the
changed-file filter; `--exclude` wins. Neither override enables a disabled test:
change its registry entry deliberately to do that.

## GitHub entry points

- `ci.yml`: inventory/selection, host suites, one matrix job per hardware target,
  and the stable aggregate check **CI status**. PRs run host suites automatically.
  Main pushes run affected host and hardware suites.
- A maintainer with write/maintain/admin permission may comment
  `/run-ci l4`, `/run-ci all p2p.*`, or `/run-benchmark l4 p2p.bandwidth` on an open
  PR. The dispatcher runs the exact PR head SHA using `workflow_dispatch`.
  Dispatch summaries contain the SHA; results are in the UCCL CI run artifacts.
- The Actions UI exposes `targets`, `tests`, `exclude`, `tiers`, `sha`, and
  `head_repo` for manual dispatch. Empty selectors mean all eligible entries.
- Formatting remains an independent fast gate.
  Docker image publication and release workflows remain separate from tests.

Lab credentials are never supplied to automatic PR jobs. Maintainer dispatch
explicitly authorizes execution of the selected code on persistent lab hosts.
GitHub concurrency serializes jobs per target; a machine-local `flock` also
prevents overlapping local/manual CI runs. A busy GPU fails preflight. The
runner never terminates unrelated processes.

The old L4/GH200/GB10/AMD/AMD-zip workflows and duplicate EP host workflow are
replaced. Update branch protection rules referring to removed workflow check
names to `CI status`, keeping the formatting checks and removing the GIN check.
Existing AMD-specific shell helpers remain for future ROCm re-enablement;
they are not called by an enabled job. The runner currently implements Linux
CPU and NVIDIA CUDA environments. Add and validate a ROCm worker environment
when the AMD servers return before enabling that target.

## Server provisioning and credentials

One-time administrator setup, on each machine:

```bash
sudo bash ci/provision.sh "$(cat /path/to/uccl-ci.pub)"
```

The provisioner creates `uccl` when absent, grants GPU device groups and unlimited
memlock, installs native build dependencies, and creates local directories.
New accounts receive no sudo/docker membership. Existing account memberships
are preserved: the pre-existing GH200 `uccl` account still has its administrator
credentials and Docker membership. Running as that nonzero UID is **not** an
isolation boundary against a malicious approved PR. Removing those existing
privileges or moving to a freshly isolated machine is a separate administrator
migration. CI itself does not use sudo, Docker, a password, a personal conda
environment, agent forwarding, or NFS.

Install three repository Actions secrets using an Actions-secrets-capable token:

```bash
bash ci/install-secrets.sh /path/to/private-bundle uccl-project/uccl
```

The bundle contains `id_ed25519`, `github-config`, and `known_hosts`. It stays
outside the repository. `UCCL_CI_SSH_KEY` authenticates **only** the CI account;
`UCCL_CI_SSH_CONFIG` maps the inventory aliases; `UCCL_CI_KNOWN_HOSTS` pins already
verified server keys. The existing `UCCL_DEV_SSH_KEY` is used only for the GB10
jump host. Host checking is strict and agent forwarding is disabled. Do not
store the GH200 password in GitHub secrets for this workflow.

## Local execution and results

Use Python 3.12+ on Linux for `--local`, or Python 3.12+ with SSH/rsync as the
controller on macOS/Linux. The SSH config must log in as `uccl` and expose the
aliases above. Remote servers require Python 3.12 and their declared CUDA path.
Local CPU runs need `build-essential`, `python3-venv`, `libgtest-dev` and
`libibverbs-dev` on Ubuntu; several host tests include RDMA headers without
requiring an RDMA device.

```bash
python3 ci/run.py --local --target cpu --output ci-results/host-run
python3 ci/run.py --target l4 --ssh-config /path/to/config --tests 'p2p.rdma_native.*' --output ci-results/rdma-run
python3 ci/run.py --target gb10 --ssh-config /path/to/config --tests 'ep.low_latency,nixl.*' --output ci-results/gb10-run
python3 ci/run.py --target l4 --ssh-config /path/to/config --tiers performance --tests p2p.bandwidth --output ci-results/perf-run
python3 ci/performance.py ci-results/perf-run --candidate ci-results/perf-run/baseline-candidate.json
```

Each node receives an independent source archive in a unique workspace. The
controller records the Git SHA, dirty state and archive SHA256. The worker
records hostname, UID, filesystem, GPU UUID/model/driver, compiler/CUDA/system
packages and exact Python packages. Python environments are keyed by a hashed
lockfile and Python version; pip checks all package hashes. Native outputs are
rebuilt in the source workspace. External NIXL/Abseil/nccl-tests/GDRCopy userspace sources use
full commit pins in `dependencies.json`. System drivers/CUDA/apt packages are
recorded rather than claimed to be hermetically pinned.

Artifacts include `plan.json`, aggregate `results.json`, `junit.xml`, per-rank
logs, build logs, machine/toolchain/package inventories, and pytest JUnit case
counts. Both ranks must succeed. Timeouts, signals, disconnects, setup failures,
all-skipped pytest suites, and configured legacy failure markers fail the job.
On peer failure the controller cancels only the current suite's owned process
group. EOF/cancellation also terminates owned descendants. Source workspaces are
removed after successful artifact collection, including failed test runs; use
`--keep-remote` for debugging. Collection failure preserves the remote tree.
Persistent dependency caches live only in `/var/lib/uccl-ci/cache`.

Use a fresh output directory for each run. Empty runs and reused output trees
are rejected to prevent false passes and stale measurements. L4 disk space is
limited: avoid retaining workspaces or multiple large environments indefinitely.

## Performance policy and current validation

`baselines.json` contains only explicitly reviewed numeric baselines. A metric
needs a parser, direction, unit and tolerance. Missing/nonfinite values or a
threshold violation fail; missing baselines are labeled `unbaselined`, not a
performance pass. Other benchmark entry points retain their raw logs and exit
status until a numeric metric is explicitly configured. Baseline candidates are
artifacts; CI never updates the active baseline automatically. Benchmark success
does not substitute for correctness suites.

See [validation.md](validation.md) for measured coverage and known failures from
the migration. Registration is not a claim that every legacy test currently
passes. Failures remain visible; disabling a flaky or broken test requires an
explicit registry change with a reason.

Developer checks:

```bash
python3 -m unittest discover -s ci/tests -v
python3 ci/catalog.py validate
python3 -m compileall -q ci
# Also run actionlint on .github/workflows/ci.yml and run-benchmark.yml.
```
