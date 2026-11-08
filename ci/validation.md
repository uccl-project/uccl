# Migration validation — 2026-11-08

Work is on `codex/structured-ci`, based on `78adb5dce345c1ba5c52c77c6de8e09bcd4fe4cb`.
Runs use the working-tree source archive; their `source.json` records its SHA256.
Raw results are in the ignored local `ci-results/` directory. The new GitHub
workflow has run on GitHub in PR #1131. Hardware dispatch is not activated:
the connected GitHub token returned HTTP 403 for Actions secret administration.

## PR #1131 follow-up

The initial GitHub host run passed 23 suites and failed four C++ host suites
(`ep.host.rdma_addressing`, `p2p.host.notify_serde`, `p2p.host.epoll_client`,
`p2p.host.epoll_server`). Every failure was a missing `infiniband/verbs.h` header;
the workflow now installs `libibverbs-dev` alongside the compiler and GTest.
The lab machines already had those headers, which explains the earlier passes.

Python formatting failed on `p2p/tests/test_dmabuf_export.py` with Black 25.11.0.
The formatter is now pinned to 26.5.1, matching `uv.lock` and the independent
format-on-commit branch. Both formatting jobs use Ubuntu 24.04.

All 62 registry entries sourced from `examples/` or `experimental/`, including
the GIN routing/probability fixtures built through that tree, are now explicitly
disabled by CI scope. The separate GIN CUDA workflow is removed. Historical
results below include exploratory runs of those now-excluded suites.

## Planned GH200 pair restoration

The pool owner reports that `gh200_0` will return with a 400 Gbps RDMA connection
to `gh200_1`, and that the pair previously ran EP/P2P performance tests. The
inventory now assumes matching GH200 capabilities and CUDA 13.2 on both nodes.
`gh200_1` remains first for single-node runs. Per-host control interfaces come
from the prior GH200 workflow; RDMA remains pinned to `mlx5_0`.

Existing compatible two-node EP/P2P/NIXL and transport entries now include
GH200. New performance entries expose EP dispatch/combine latency and GPU-buffer
P2P bandwidth; the existing host-pinned bandwidth benchmark also includes GH200.
The 400 Gbps link rate is nominal metadata. No measured GH200 pair baseline has
been installed, and none of these additions claims a fresh two-node pass.

The dedicated SSH bundle includes `gh200_0` with its previously trusted host
keys. Its CI account/key, local directories and toolchain must be provisioned
and checked when it returns; no connection to that offline node was attempted.

Validation of this change: all 16 CI harness tests pass, covering interface
selection by host alias, reordered ranks, inventory validation, GH200 selection
and the new metric parsers. The updated worker ran `ep.triton_pack_unpack` on
`gh200_1` only: 25 cases passed and two optional CUDA-reference cases skipped.
Artifacts in `ci-results/gh200-pair-worker` record the expected SSH alias and
control interface; collection and cleanup completed successfully. This does
not validate the restored pair or its link throughput.

## Confirmed behavior

- All 27 host suites have passed, including 12 CI harness regression tests.
  The original 26-suite run passed on L41. The final host run on GH200 exposed
  an x86-only assumption in the Apptainer mock; it was fixed and passed on GB10.
- L4: native read/write, memory cache/device selection, GPU IPC, cross-node
  GPU RDMA, compressed write/arena, Ray-style endpoint transfers, large
  notifications, protocol utilities, ukernel transport integration and CCL
  unit tests, native signal backend, Python P2P/bidirectional/tile-signal/IPC-path tests,
  UCCL-Lite build, and bandwidth passed.
- GH200: native read/write and shutdown, EP event ordering and barrier,
  CUDA utility/flush/PCIe/scattered-copy tests, Triton packing, local GIN
  runtime/adapter/model/routing fixtures, ukernel transport and CCL unit tests
  passed.
- GB10: EP low-latency, deduplicated and dual-mode tests, all four NIXL smoke
  suites (local IPC option on/off, cross-node RDMA/NCCL), local GIN fixtures,
  P2P host-cache/shutdown/device tests, protocol utilities, ukernel host/transport
  suites, UCCL-Lite build, and host-staged bandwidth passed.
- Concurrent CI invocation was rejected by the machine lease before executing
  any test. Peer failure cancellation, timeout, disconnect and descendant
  cleanup are covered by the harness tests. New workflow YAML passes actionlint.

The initial performance samples were **385.68 Gbps on L4** and **124.59 Gbps on
GB10** for `p2p.bandwidth` (1 MiB host-pinned buffers, 128 iterations). These are
single observations, saved in baseline-candidate artifacts. They are not yet
accepted regression thresholds; `baselines.json` remains empty deliberately.

## Hardware/configuration exclusions established by testing

- GB10 native GPU-buffer registration fails with `ibv_reg_mr_iova2`/Bad address;
  the DMA-BUF attempt also failed. GPU-buffer read/write, compression arena and
  the current native notification fixture therefore select L4/GH200 as their
  other requirements permit. Ukernel fixtures requiring GPU RDMA are likewise
  restricted to L4. GB10 host-staged NIXL and EP remain enabled.
- L4 has peer copies but lacks native peer atomics. Native multi-GPU EP paths
  requiring those atomics/NVLink are explicitly disabled. Standalone barrier
  and Triton fixtures run on L4. Full native EP builds currently require SM90
  features, so event-ordering tests select GH200/GB10. Lite fixtures remain registered.
- No available node has the GDRCopy kernel device `/dev/gdrdrv`. Tests that
  require it are explicitly disabled. A pinned GDRCopy **userspace** library
  is built locally for ukernel bindings; this does not install a kernel driver.
- `collective.rdma.eqds_transfer` requests an EQDS CCA build and exits under the
  repository default. It remains explicitly disabled until that build variant
  is supplied. The EQDS algorithm unit tests pass.
- The legacy raw RDMA fixture hardcodes NIC index 2 and GPU-memory registration;
  that combination matches neither the two-NIC L4 pool nor GB10's memory path.
- AMD, EFA, MUSA/Hygon, privileged AF_XDP/DPDK, external NVSHMEM/PPLX, and
  eight-GPU comparisons have explicit exclusion reasons in the inventory.

## Known failures exposed by broader coverage

Failures in enabled suites remain failures, not converted into success.
The examples and experimental suites below are now excluded by the requested
directory scope, with explicit reasons in the registry.
Several involve stale fixture APIs or failures within the library. The evidence
records this pinned environment; it does not establish whether a different
legacy environment would pass. Do not promote this branch on the assumption that all
hardware gates are already green.

| Suite/family | Observed failure | Evidence directory |
|---|---|---|
| `collective.rdma.transport` | Native transport exits abnormally during initialization on L4/GB10 | `l4-extra`, `gb10-extra` |
| `collective.multi_process_groups` | Native NCCL RDMA plugin SIGSEGV on both L4 and GB10 | `l4-retry`, `gb10-retry` |
| `ukernel.transport.unit` and `ukernel.python.transport_paths.tcp` on L4 | TCP payload mismatch; host unit suite passes on GH200/GB10 | `l4`, `l4-local`, `gb10-libraries` |
| `ukernel.ccl.test_transport_backend_e2e` | Native transport backend failure | `l4-bindings` |
| `lite.ep.*` | Build uses `pybind11` types in `csrc/kernels/backend/api.cuh` without a visible declaration | `l4-extra` |
| `p2p.compression_unit` | Legacy C++ fixture references removed macros/API, `float_type` fields and private Compressor constructor | `l4-extra` |
| `ep.internode_simple` | Legacy fixture passes a tensor instead of an address and uses the old Buffer construction/lifetime API | `gb10-extra` |
| `lite.collective.sendrecv` | Native shim reports an internal error during communicator initialization | `l4-bindings` |
| `p2p.notifications` on GH200 | Client sends all messages, receiver does not finish before timeout; L4 passes | `gh200-notify`, `l4-final` |
| `ukernel.python.transport_paths` | Cross-node bidirectional barrier failure on L4; GB10 retry confirms GPU MR registration is unsupported | `l4-ukernel`, `gb10-identity` |

Not every target/suite combination or performance entry has been executed during
this migration. The registry is the intended coverage, while result artifacts
are the evidence of executed coverage. Build-only, disabled, unexecuted,
failed and passing cases must not be conflated.

## Changes made to test harnesses

- Multiprocess read/write tests now fail when either child exits unsuccessfully.
- Native RDMA rendezvous retries connection refusal during startup.
- Local IPC and Ray-style API fixtures use current metadata, transfer and
  polling APIs and valid GPU selection.
- The Apptainer CLI mock supplies a fixed x86 architecture so ROCm flag checks
  can execute on ARM without invoking a real ROCm build.
- NIXL uses pinned sources and ASIO coordination, with a bounded retry only for
  a pre-handshake connection refusal. Neither etcd nor shared filesystem state
  is needed.

## Deployment state and remaining administrator actions

The five previously provisioned servers have working `uccl` logins and local CI
storage; the sixth machine, `gh200_0`, is pending its return and provisioning.
New accounts on L40/L41/GB10 have no sudo or Docker membership. GH200 reuses its
existing non-root `uccl` account, which still has Docker membership and existing
administrator credentials. Those existing privileges were not removed because
`gh200_1_root` uses that same account. This limits isolation; preserve a separate
administrator login before hardening the reused account.

The private key, local CI SSH config, GitHub SSH config and verified host keys
are saved outside the repository at:

```
/Users/yangzhou/.ssh/uccl-ci-bootstrap/
```

With a token allowed to write this repository's Actions secrets, run:

```bash
bash ci/install-secrets.sh /Users/yangzhou/.ssh/uccl-ci-bootstrap uccl-project/uccl
```

This installs the three CI secrets, including the same dedicated key for the
GB10 jump host. The jump host uses a separate `uccl` account provisioned with
`ci/provision-jump.sh`; the legacy jump-host key secret is no longer consumed.
Both GB10 servers were reached as `uccl` through this jump account using only
the dedicated CI key, with strict host checking and no agent forwarding. The
jump account has no supplementary groups; shell commands and forwarding to
the unrelated `localhost:22` destination were verified to fail.
Review the known failures and required check-name migration before
merging. The branch is published as PR #1131; no merge has been performed.
