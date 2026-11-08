# Migration validation — 2026-11-08

Branch: `codex/structured-ci`, based on `78adb5dc`; PR: #1131. Local evidence is
in ignored `ci-results/` directories, with source archive hashes in `source.json`.
GitHub host/format checks have passed. Hardware dispatch still needs secrets;
the upload attempt returned HTTP 403.

## PR #1131 follow-up

- Four host suites initially failed on missing `infiniband/verbs.h`; installing
  `libibverbs-dev` fixed them. All 27 host suites subsequently passed on GitHub.
- Python formatting now uses Black 26.5.1, matching `uv.lock` and the formatting
  branch. Both format checks passed.
- All 62 example/experimental entries are disabled, including GIN fixtures;
  the separate GIN CUDA workflow is removed.

## Planned GH200 pair restoration

`gh200_0` is configured to match `gh200_1` with CUDA 13.2 and an assumed 400G
RDMA link. Single-node tests use `gh200_1`; compatible two-node EP/P2P/NIXL and
transport suites now include GH200. EP latency and host/GPU-buffer bandwidth
benchmarks are registered, but the pair and link throughput remain untested.

The SSH bundle includes the offline node's previously trusted keys. Its account,
local storage and toolchain still need provisioning when it returns.

All 16 harness tests pass, including host-interface selection and metric parsing.
On `gh200_1`, `ep.triton_pack_unpack` passed 25 cases with two optional skips;
artifacts and cleanup succeeded (`gh200-pair-worker`). No GH200 pair baseline
has been installed.

## Confirmed behavior

| Target | Validated coverage |
|---|---|
| Host | All 27 suites; Apptainer mock corrected for ARM hosts |
| L4 | Native P2P/GPU RDMA, IPC, compression, notifications, protocol utilities and bandwidth |
| GH200 | Native P2P/shutdown, EP ordering/barrier, CUDA utilities and Triton packing |
| GB10 | EP low-latency/dedup/dual-mode, all four NIXL suites, host P2P utilities and staged bandwidth |

Machine locks rejected concurrent runs. Harness tests cover peer failure,
timeout, disconnect and descendant cleanup. Workflow YAML passed actionlint.

Initial `p2p.bandwidth` samples: **385.68 Gbps on L4**, **124.59 Gbps on GB10**
(1 MiB host-pinned buffers, 128 iterations). These single observations are
baseline candidates only; `config/baselines.json` remains empty.

## Hardware/configuration exclusions established by testing

- **GB10:** native GPU registration fails with verbs and DMA-BUF; host-staged
  EP/NIXL remains enabled.
- **L4:** lacks native peer atomics/NVLink; full native EP builds require SM90.
  Barrier and Triton fixtures remain eligible.
- **GDRCopy:** no available `/dev/gdrdrv`; userspace libraries alone are insufficient.
- **Legacy RDMA fixtures:** EQDS transfer needs a different CCA build; the raw
  RDMA fixture hardcodes an incompatible NIC/GPU-memory configuration.
- **Other exclusions:** AMD, EFA, MUSA/Hygon, privileged AF_XDP/DPDK, external
  NVSHMEM/PPLX and eight-GPU comparisons. Exact reasons are in `config/tests.json`.

## Known failures exposed by broader coverage

Enabled failures remain visible. Rows marked **excluded** are historical
example/experimental results, outside current CI scope. Evidence paths below
are relative to `ci-results/`.

| Suite/family | Observed failure | Evidence directory |
|---|---|---|
| `collective.rdma.transport` | Initialization failure on L4/GB10 | `l4-extra`, `gb10-extra` |
| `collective.multi_process_groups` — excluded | NCCL RDMA plugin SIGSEGV | `l4-retry`, `gb10-retry` |
| `ukernel.transport.unit`, `ukernel.python.transport_paths.tcp` — excluded | L4 TCP payload mismatch | `l4`, `l4-local`, `gb10-libraries` |
| `ukernel.ccl.test_transport_backend_e2e` — excluded | Native transport failure | `l4-bindings` |
| `lite.ep.*` — excluded | Missing `pybind11` declarations | `l4-extra` |
| `p2p.compression_unit` | Fixture uses removed/private APIs | `l4-extra` |
| `ep.internode_simple` | Fixture uses outdated address/Buffer APIs | `gb10-extra` |
| `lite.collective.sendrecv` — excluded | Communicator initialization error | `l4-bindings` |
| `p2p.notifications` on GH200 | Receiver timeout; L4 passes | `gh200-notify`, `l4-final` |
| `ukernel.python.transport_paths` — excluded | L4 barrier failure; GB10 GPU registration unsupported | `l4-ukernel`, `gb10-identity` |

Not every target/suite combination has run. Registration describes intended
coverage; artifacts establish what passed. Hardware coverage is not all green.

## Changes made to test harnesses

- Propagate multiprocess failures and retry RDMA rendezvous during startup.
- Update IPC/Ray-style fixtures to current APIs and valid GPU selection.
- Make the Apptainer mock architecture-independent.
- Pin NIXL sources and use ASIO coordination with bounded connection retries.

## Deployment state and remaining administrator actions

L40/L41, GH200_1 and both GB10 servers have working `uccl` accounts and local
storage. GH200's existing account retains administrator/Docker privileges;
new CI accounts have neither. A separate `uccl` jump account reaches both GB10
servers with the CI key; shell commands and unrelated-port forwarding were
denied in verification.

Before deployment:

1. Upload the three `UCCL_CI_*` secrets, including the updated jump-host config,
   with a token that can manage Actions secrets:

   ```bash
   bash ci/install-secrets.sh /Users/yangzhou/.ssh/uccl-ci-bootstrap uccl-project/uccl
   ```

2. Provision `gh200_0` when it returns and validate the pair.
3. Update branch protection to `CI status` plus formatting checks; remove retired
   checks. Remove the legacy jump-host secret after the new workflow is active.
4. Review the enabled failures above before merging PR #1131.

See [README.md](README.md#server-provisioning-and-credentials) for setup details.
