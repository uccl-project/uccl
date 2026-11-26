# Automatic preparation at native NCCL initialization

The native-enabled ABI can prepare a declared workload when ordinary
`ncclCommInitRank` or `ncclCommInitRankConfig` completes. Applications do not
call `ucclLiteNcclPrepare` in this mode. The configuration describes operations
and buffer contracts; it contains no GPU model, algorithm name, rank grouping,
PCIe speed or NIC selection policy. The common runtime discovers participants,
builds topology-derived candidates, measures missing resources and qualifies
the exact dispatch against the original native NCCL implementation.

This remains experimental and opt-in. Build with:

```sh
make nccl-prepared-abi nccl-auto-init-check
```

Use the same interposer and enable automatic initialization on **every** rank
of a participating communicator before initialization:

```sh
export UCCL_LITE_REAL_NCCL=/path/to/native/libnccl.so.2
export UCCL_LITE_AUTO_INIT=1
export UCCL_LITE_AUTO_MANIFEST=/path/to/workload.txt
export LD_PRELOAD=/path/to/build/integration/libuccl_lite_nccl_prepared_abi.so
```

Do not mix enabled and disabled ranks or interposer versions. This is a
collective deployment contract, just as explicit preparation requires every
rank to participate. The adapter does not discover arbitrary future call sizes
or start tuning from a collective enqueue. Initialization is the preparation
boundary; its cost includes correctness checks and calibration. Run it outside
CUDA capture, with prior application collective work completed.

## Workload file

For example, the following file declares three FP32 SUM operations, each with
NCCL count 64 and separate buffers aligned to 256 bytes:

```text
uccl-lite-auto-prepare-v1
dtype float32
capacity 1048576
flags 31
requests 3
allreduce 64 out 256
allgather 64 out 256
reducescatter 64 out 256
```

`capacity` is the maximum per-rank allocation extent in bytes, bounded to 1 GiB.
`flags` uses the public `prepared_api.h` bits: peer 1, host 2, peer packet 4,
network 8, resource probes 16 and fallback-only 32. The file is bounded to 1 MiB
and 1,024 requests. Supported scalar names are `float32`, `bfloat16` and
`float16`; FP16 currently retains native execution.

`out ALIGNMENT` and `in ALIGNMENT` expand a uniform layout to the actual
communicator size. Alignment is a power of two from the scalar size to 256,
and must match the runtime's capped pointer alignment. Mixed layouts use
`ranks N INPLACE ALIGNMENT ...`, in logical rank order, with exactly N pairs.
For example, `allreduce 64 ranks 2 0 256 1 4` describes two rank layouts,
not an algorithm specialized for a two-GPU machine.

Requests whose explicit layout count, capacity or in-place shard alignment
cannot match the actual communicator are omitted. Duplicate materialized
signatures and malformed syntax reject preparation. If no request fits, the
communicator stays native. The full file contents must match across ranks;
paths may differ, but even whitespace differences fail agreement. Runtime
signature mismatch still causes terminal native ownership as described in
[the prepared ABI contract](prepared-nccl-abi.md).

## Initialization and fallback

After native blocking initialization completes, a small native all-gather
checks workload and interposer identities and discovers process/GPU ownership.
Its temporary CUDA buffer and stream are released before qualification. A
separate CPU rendezvous then confirms that every rank has finished the private native
exchange, including on configuration-error paths. An early caller abort must
not tear down native communication while another initializer is still using it.
The completion ID is generated before parsing the manifest; local participants
join concurrently so grouped multi-rank initialization cannot wait serially.
This uses the independent `bootstrap::HostControl` channel, without touching
MSCCL++'s shared Unix socket service. Its IPv4 address comes from
`MSCCLPP_SOCKET_IFNAME`, or the OS default IPv4 route when no interface is set;
the root advertises that address in the native metadata exchange.
Native config fields, including string values rather than pointer addresses,
contribute to the baseline identity. Algorithm selection remains in the core;
the adapter does not choose a backend from the number of ranks.

Automatic preparation currently supports one process per GPU, 1–32 ranks,
serialized host calls and ready **blocking** native communicators. Effective
native blocking configuration must be uniform. Explicit nonblocking config or
`NCCL_COMM_BLOCKING=0` forwards to native. There is no public query of the
effective blocking setting: hidden native configuration-file overrides and
changes to cached NCCL configuration after first use are outside this mode's
contract.

Grouped initialization remains native. At the outer successful group end,
metadata exchanges for all local initializations are submitted together before
waiting, so multiple local ranks do not wait on one another serially. A
communicator mixing grouped and ungrouped initialization also stays native.
`ncclCommInitAll` is forwarded directly. An already active prepared executor,
duplicate process/GPU ownership or unsupported rank extent retains native
execution. These cases do not create UCCL accelerator buffers or workers.

With no file requested by any rank, native is retained. For a supported
automatic context, one missing/unreadable/malformed or different file causes
all participants to report an initialization error before UCCL preparation.
The native handle may already exist: callers must handle the initialization
failure and abort it. This is healthy-rank configuration consensus, not recovery
from process failure, transport failure or allocation failure during exchange.

`ucclLiteNcclReport` adds `automatic_init` and `init_status`. Native-only
automatic decisions also have reports, without accelerator resources or plans.
Destroy/abort removes those diagnostics. Opt-out calls retain native behavior.

The algorithm support limits remain those of the core: local peer trees can
represent sparse access graphs, while the retained hierarchy is
specialized to two hosts with four GPUs each. The
[distributed portfolio](network-candidate-selection.md) now also qualifies general
direct exchange for other supported layouts and BF16; hierarchy's own contract
is unchanged. Host/NUMA measurement coverage, control/SM costs and
broader distributed algorithms are still tracked in
[the migration record](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/docs/topology-runtime-refactor.md).

## Running checks

The CPU manifest check runs as part of the Make target. With the environment
above and a shared valid manifest, the real-native four-GPU check is:

```sh
mpirun --bind-to none -np 4 build/uccl_lite_prepared_abi_check \
  "$UCCL_LITE_REAL_NCCL" ordinary auto
```

This test has no explicit prepare call in `auto` mode. It checks full buffers,
guards, native fallback and communicator recreation. The `auto_init` checker
covers malformed/mismatched files and native-only initialization modes; its
arguments are `MODE MANIFEST ALTERNATE_MANIFEST`. The `eager`, `group` and
`capture` modes of the prepared ABI checker require a qualified accelerated
path. The delayed-native library in the test build is only a deterministic
branch-coverage fixture; timings against it are not performance evidence.

The final local/two-host evidence is in `build/auto-init-release/`, with source
and artifact hashes in `build/auto-init-artifacts.sha256`. It passes 26 runtime
cases and 16 CPU checker executables, including 2/3/4-L4 local dispatch and two
four-L4 hosts with interleaved logical ranks/reversed GPU ownership. Actual
native NCCL 2.28.7 retains the tested 64-element FP32 requests on one and two
hosts. Transition coverage uses the labeled delayed-native fixture; there is no
performance improvement claim. Qualification uses maximum-rank host completion,
three ABBA rounds, two warmups and three timed calls/block (18/18 samples), with
setup/verification excluded; hardware is the dual-NUMA Xeon Silver 4410Y/L4
configuration. No best-variant curve is reported. The migration record retains
the two initial completion/cleanup failures and their corrected regressions.
