# UCCL CI

CI has one inventory (`tests.json`), one hardware inventory (`targets.json`), one
change selector (`catalog.py`), and one runner (`run.py`). GitHub-hosted Ubuntu
machines orchestrate jobs. GPU tests execute over SSH on the lab servers as
`uccl`, in separate local copies under `/var/lib/uccl-ci`.

## Where jobs run

| Target | Machines | Hardware | Role |
|---|---|---|---|
| `cpu` | GitHub `ubuntu-24.04` | CPU | Python regressions, C++ host tests, CI self-tests |
| `l4` | `l40`, `l41` | 4 L4 GPUs per node, IB, CUDA 12.9 | Native P2P, IPC, compression, two-node GPU RDMA, protocol tests, NIXL |
| `gh200` | `gh200_1`, `gh200_0` | 1 GH200 per node, assumed 400G RDMA link, CUDA 13.2 | Two-node EP, native GPU RDMA, compression, NIXL and EP/P2P benchmarks |
| `gb10` | `gb10_0`, `gb10_1` | 1 GB10 per node, RoCE, CUDA 13.0 | Two-node EP, host-staged communication, NIXL |
| `amd` | Offline | ROCm | Explicitly disabled |

The GH200 pair is configured in anticipation of `gh200_0` returning, with the
same GPU/toolchain capabilities as `gh200_1` and a 400 Gbps RDMA link as specified
by the pool owner. These are configuration assumptions, not fresh validation.
Single-node GH200 suites use `gh200_1`, the first host in the inventory.
Two-node suites require both hosts online and provisioned.

GH200 control interfaces differ: `gh200_1` uses `enxc8a36290119d` and `gh200_0`
uses `enx3c18a0d51943`, as in the former GH200 workflow. The `interfaces` mapping
selects by SSH alias, including when `--hosts` changes rank order. Both hosts
use `mlx5_0` for RDMA; 400 Gbps is recorded as nominal link metadata, not a
measured throughput or regression threshold.

L4 has GPU peer copies but no native peer atomics/NVLink. GB10's native GPU
memory registration failed both ordinary verbs and DMA-BUF validation;
two-node native GPU RDMA suites select L4/GH200. This does not exclude GB10
host-staged EP or NIXL tests. GH200 has GDRCopy userspace
libraries but no `/dev/gdrdrv`; tests requiring that kernel driver are disabled
with reasons. These capabilities are separate from the presence of an RDMA NIC.

## Current coverage

The registry currently contains **184 entries: 89 enabled and 95 disabled**.
Of the enabled entries, 73 are in the default tiers and 16 are opt-in performance
benchmarks. Entries represent suites, not individual pytest or C++ test cases.
Enabled means eligible for selection, not necessarily passing; see
[validation.md](validation.md) for execution evidence and known failures.

| Enabled family | Entries | What it checks |
|---|---:|---|
| `ci.*` | 1 | Selection, inventory auditing, subprocess cleanup, cancellation, exit codes and baseline comparisons |
| `p2p.*` | 41 | Host backend/wrapper tests, memory registration, IPC, native reads/writes, device binding, compression, notifications and host/GPU-buffer benchmarks |
| `ep.*` | 22 | Host compatibility, event ordering, barriers, low-latency/deduplicated/dual-mode communication, Triton packing and benchmarks |
| `mccl.host.*` | 4 | MUSA plugin ABI and behavior using host fixtures; no MUSA GPU execution |
| `collective.*` | 14 | Protocol utilities, congestion-control algorithms, RDMA transport and CUDA utilities; no example-driven process-group test |
| `nixl.*` | 4 | Local transfers with the IPC option on/off, and two-node RDMA/NCCL smoke tests |
| `build.apptainer_flags` | 1 | Apptainer GPU-option handling using mocks |
| `util.*` | 2 | GPU topology and runtime utilities |

Full selection on each enabled target, before changed-file filtering:

| Target | Default (`host,build,integration`) | Opt-in `performance` |
|---|---:|---:|
| `cpu` | 27 | 0 |
| `l4` | 38 | 8 |
| `gh200` | 31 | 16 |
| `gb10` | 20 | 6 |

Some suites run on multiple targets, so target counts overlap. The `host` tier
describes the test type: 27 host suites use the GitHub CPU target, while seven
protocol suites use L4's build environment. There are currently no enabled
standalone `build`-tier entries, although integration suites still build their
dependencies. Performance jobs require explicit selection of that tier.

**Excluded coverage:** 62 entries under `examples/` and `experimental/` are
disabled by scope. This includes all `gin.local.*`, `ukernel.*`, `lite.*` and
`misc.*` suites, `collective.multi_process_groups`, and GIN fixtures sourced
from `thirdparty/nccl-ep` but built through the experimental tree. The other
33 disabled entries record unavailable hardware, unsupported configurations,
legacy fixtures or opted-out comparisons. Every disabled entry has a reason.

Disabled entries remain registered for auditing; neither automatic selection
nor an explicit `--tests` request runs them. The separate GIN CUDA workflow is
removed. Formatting still checks experimental C++/CUDA. Vendored upstream
test trees are not implicitly enrolled.

`tests.json` and `targets.json` are the source of truth for these counts and
assignments. Run `python3 ci/catalog.py list` to see enabled and disabled entries,
or `python3 ci/catalog.py plan --full` to inspect the current default selection.

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

`ci.yml` plans the run, starts a host job and one job per selected hardware
target, and publishes the aggregate check **CI status**.

| Trigger | Host tests | Hardware tests | Performance benchmarks |
|---|---|---|---|
| PR opened, updated or reopened | Affected CPU suites | Skipped | Skipped |
| Merge or direct push to `main` | Affected CPU suites | Affected suites on compatible targets | Skipped |
| `/run-ci all` on an open PR | All enabled CPU suites | All eligible non-performance suites | Skipped |
| `/run-ci <target> [test IDs/globs]` | Only when targeting `cpu` | Only when targeting `l4`, `gb10` or `gh200` | Skipped |
| `/run-benchmark <target> [test IDs/globs]` | No CPU performance suites are currently registered | Selected performance suites only | Enabled |

Automatic runs use changed-file selection. Shared headers/build/CI changes,
unknown source changes or an unavailable diff base select all eligible suites.
Manual runs do not apply changed-file filtering. Disabled tests, including all
example/experimental entries, remain excluded from every mode.

Only collaborators with `write`, `maintain` or `admin` permission may use the
comment commands. They run the PR's exact head SHA at dispatch time; later PR
commits require another command. The command is `/run-benchmark` (singular).
Supported targets are `all`, `cpu`, `l4`, `gb10`, and `gh200`:

```text
/run-ci all
/run-ci cpu
/run-ci l4 p2p.*
/run-ci gh200 ep.*,p2p.rdma_native.*,nixl.*
/run-benchmark l4 p2p.bandwidth
/run-benchmark gh200 ep.bench.low_latency,p2p.bandwidth,p2p.bandwidth_gpu
/run-benchmark all
```

The Actions UI also exposes `targets`, `tests`, `exclude`, `tiers`, `sha`, and
`head_repo`. By default it runs all eligible host/build/integration suites;
select the `performance` tier explicitly for benchmarks. Benchmarks do not
implicitly run correctness suites first. There is no label trigger or nightly
schedule. Main-push hardware checks happen after merging; the automatic PR gate
does not require hardware results. Use `/run-ci` before merging when needed.

Results, logs and JUnit reports are in the UCCL CI workflow summary/artifacts.
The dispatcher summary identifies the PR SHA; it does not post follow-up PR
comments. The dispatcher needs `contents: read` and `actions: write`; test jobs
need only `contents: read`. Comment commands use the workflow on the default
branch, so dispatcher changes take effect after merging to `main`.

Formatting remains an independent fast gate. Docker publication and release
workflows remain separate from tests.

Lab credentials are never supplied to automatic PR jobs. Maintainer dispatch
explicitly authorizes execution of the selected code on persistent lab hosts.
Host and hardware jobs can run in parallel after planning. GitHub concurrency
serializes hardware jobs per target; a machine-local `flock` also
prevents overlapping local/manual CI runs. A busy GPU fails preflight. The
runner never terminates unrelated processes.

The old L4/GH200/GB10/AMD/AMD-zip workflows and duplicate EP host workflow are
replaced. Update branch protection rules referring to removed workflow check
names to `CI status`, keeping the formatting checks and removing the GIN check.
Existing AMD-specific shell helpers remain for future ROCm re-enablement;
they are not called by an enabled job. The runner currently implements Linux
CPU and NVIDIA CUDA environments. Add and validate a ROCm worker environment
when the AMD servers return before enabling that target.

For the restored GH200 pair, `ep.bench.low_latency` records dispatch/combine
latency using the existing two-node EP fixture. `p2p.bandwidth` measures pinned
host buffers; `p2p.bandwidth_gpu` measures GPU-buffer RDMA writes. The latter is
eligible on L4/GH200; EP latency is eligible on GH200/GB10. The EP fixture also
checks correctness and prints timing output in its integration registration;
only the performance registration extracts metrics and compares baselines.

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

`gh200_0` still needs this provisioning when it returns, including the dedicated
CI public key, local storage, Python 3.12 and CUDA 13.2. The private SSH bundle
has its alias and previously trusted host keys; verify the keys and interface
names if the server is rebuilt, then refresh the repository CI secrets. No
connection or account changes on the offline node are claimed. GH200 EP keeps
the generic Hopper/RDMA build (`UCCL_EP_DISABLE_GH200=1`) because the managed
atomic-buffer path requires RDMA capabilities not established on this pool.

Install three repository Actions secrets using an Actions-secrets-capable token:

```bash
bash ci/install-secrets.sh /path/to/private-bundle uccl-project/uccl
```

The bundle contains `id_ed25519`, `github-config`, and `known_hosts`. It stays
outside the repository. `UCCL_CI_SSH_KEY` authenticates the `uccl` account on
both the test servers and the GB10 jump host;
`UCCL_CI_SSH_CONFIG` maps the inventory aliases; `UCCL_CI_KNOWN_HOSTS` pins already
verified server keys. These are the only repository secrets used by UCCL CI.
Host checking is strict and agent forwarding is disabled. Do not store the
GH200 password in GitHub secrets for this workflow.

On the GB10 jump host, use the lightweight provisioner instead of the test
server provisioner:

```bash
sudo bash ci/provision-jump.sh "$(cat /path/to/uccl-ci.pub)"
```

This creates `uccl` without adding sudo or Docker access and authorizes the same
CI key. The key denies shell commands, PTYs and agent forwarding, while allowing
SSH proxy connections to `localhost:57676` and `localhost:57677`. No build tools
or CI workspaces are installed on the jump host. In `github-config`, `p4_gate`
uses `User uccl` and `IdentityFile ~/.ssh/uccl-ci`; both GB10 aliases use
`ProxyJump p4_gate`. Re-upload `UCCL_CI_SSH_CONFIG` when migrating from the old
jump-host key. After the new workflow is active, `UCCL_DEV_SSH_KEY` is no longer
needed by this CI system.

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
the migration. Its historical example/experimental results are not current CI
coverage. Registration is not a claim that every legacy test currently passes.
Failures remain visible; disabling a flaky or broken test requires an explicit
registry change with a reason.

Developer checks:

```bash
python3 -m unittest discover -s ci/tests -v
python3 ci/catalog.py validate
python3 -m compileall -q ci
# Also run actionlint on .github/workflows/ci.yml and run-benchmark.yml.
```
