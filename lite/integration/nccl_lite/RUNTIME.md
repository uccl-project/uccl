# Unified experimental NCCL runtime

`make nccl-runtime` builds one NCCL ABI frontend with registered collective
providers. Dtype, host layout, transport capabilities and exact experiment
profiles are interpreted at runtime rather than compiled into separate ABI
entry points. The implementation remains experimental and retains strict
profile admission. BF16 single-host workloads can now use collective automatic
peer/host qualification during preparation; broader planner integration remains
in progress. See [prepared local selection](../../docs/prepared-local-runtime.md)
for the exact layout contract, calibration scope and remaining limitations.

```bash
make nccl-runtime MSCCLPP_IB_BUILD=/path/to/mscclpp/build-ib
python3 -B integration/nccl_lite/runtime_profile.py catalog \
  --library build/nccl_runtime/libuccl_lite_nccl_runtime.so
```

The MSCCL++ build must support IB for the network provider. The local validation
used `/home/zhongjie/third_party/mscclpp/build-ib-pcieccl`. The loader alias is
`build/nccl_runtime/strict/libnccl.so.2`. No native NCCL implementation is linked
or used as fallback by this strict runtime.

A separate [native fallback adapter](../../docs/native-fallback.md) now lets
prepared core callers compare peer/host candidates against their own native
NCCL communicator. It does not interpose ordinary NCCL calls or change this
strict experiment library's admission contract.

The [strict distributed selector](../../docs/strict-distributed-selection.md)
now compares eligible hierarchy requests against a validated direct-exchange
reference, without loading native NCCL. Its providers support heterogeneous
host sizes and scalar tails. Catalog bounds and byte granularity are per
implementation; V3 rank maps retain explicit host/device bindings. New
[V4 rank profiles](../../docs/strict-rank-profiles.md) bind UUID ownership and
per-process visibility for the distributed providers.

The currently retained implementations have these contracts:

| Provider | Dtype | Participating layout | Additional requirements |
| --- | --- | --- | --- |
| `peer-packet` | FP32 | One host, four ranks | All-pairs CUDA peer access, mapped host memory, two physical NUMA pairs; topology-derived rank permutation |
| `peer-staged` | BF16 | One host, 2–32 ranks (capacity bound) | All-pairs CUDA peer access; MSCCL++ CUDA IPC |
| `peer-staged-fp32` | FP32 | One host, 2–32 ranks (capacity bound) | Generic staged peer, actual rank/device order |
| `prepared-local` | BF16 | One host, 2–32 ranks (capacity bound) | Complete peer graph; V2 prepared layouts; host additionally needs mapped memory, NUMA and AVX2 |
| `prepared-local-fp32` | FP32 | One host, 2–32 ranks (capacity bound) | V2 layouts; generic peer baseline and capability-gated retained peer candidate |
| `host-packet` | BF16 | One host, 2–32 ranks (capacity bound) | Mapped host memory and the packet runtime's placement checks |
| `prepared-distributed-fp32` | FP32 | 2–32 ranks over multiple hosts | Exact layouts; direct-reference validation and topology-gated hierarchy comparison |
| `prepared-distributed` | BF16 | 2–32 ranks over multiple hosts | Exact layouts; validated direct reference |
| `network-hierarchy` | FP32 | Two hosts, four ranks each | Peer access, mapped host memory, two NUMA pairs per host, IB; GPU–NIC paths with physical bounds or complete matching measurements |

These are implementation constraints, not rules saying GPU count alone enables
routing. Each rank must also match the exact hardware/software identity and
profile; dtype, reduction and count must be explicitly supported. Providers
require one participating GPU per process. V1/V2 retain canonical device order;
V3 supplies explicit rank/device maps, and V4 distributed profiles additionally
support differing per-process CUDA visibility.
New providers can be registered without changing the NCCL frontend's dtype or
host-count code. Runtime capability introspection is CPU-only and constructs no
transport.

## Candidate workflow

`runtime_profile.py` offers `catalog`, `probe`, `candidate` and `launch` modes.
Probe each participating host with the same library and launch environment.
Records use CUDA-visible ordinal/UUID order, including CUDA_VISIBLE_DEVICES
filtering and reordering; the NVML physical inventory remains diagnostic data:

```bash
python3 -B integration/nccl_lite/runtime_profile.py probe \
  --library build/nccl_runtime/libuccl_lite_nccl_runtime.so \
  --real-nccl /path/to/reference/libnccl.so.2 --output build/host.json
```

For a multi-host run, set `UCCL_LITE_MSCCLPP_CONTROL_ROOT` to the shared root IPv4
address before both probing and launching. Single-host runs default to
`127.0.0.1`. `--real-nccl` records the reference identity, not a fallback backend.

New candidates use `UCCL_LITE_RUNTIME_PROFILE_V1`, followed by the provider
name and explicit ranks-per-host layout. Legacy profile headers remain readable.
The host transport uses the actual communicator rank count; its old eight-rank
packet optimizations remain conditional specializations. Generic host staging
has been checked on 2/3/4 ranks; that does not validate every rank count up to its
storage capacity or the distinct eight-rank optimized modes.

The shapes file is a JSON list of `[operation, NCCL element count]`, with
AllGather=1, ReduceScatter=2, AllReduce=3. Supply exact application counts, not
full-tensor byte counts. Candidate creation matches dtype and observed host
layout against the library's catalog and rejects unsupported or ambiguous
matches. BF16 defaults to `prepared-local`; single-host FP32 defaults to
`prepared-local-fp32`. These compare their eligible candidates during
communicator preparation. FP32 no longer needs a fixed four-rank entry point;
see [dtype-aware selection](../../docs/scalar-runtime.md). Use
`--provider peer-staged`, `--provider peer-staged-fp32`, or
`--provider host-packet` for an explicit experiment.
Automatic requests use V2 with per-rank layouts; omitted layouts mean
out-of-place 256-byte alignment. See the prepared-runtime guide before using
offset pointers or aliases. Selection is never a first-match rule:

```bash
python3 -B integration/nccl_lite/runtime_profile.py candidate \
  --library build/nccl_runtime/libuccl_lite_nccl_runtime.so \
  --records build/host.json --shapes build/shapes.json --dtype float32 \
  --profile build/runtime.profile
```

Run `runtime_profile.py launch` on each MPI rank with `--library`,
`--real-nccl`, `--profile`, then `--` and the application command. It rechecks
identity before setting the strict loader environment. This is explicit
candidate execution; V2 performs fresh automatic qualification at initialization,
while V1 selects an explicit provider. Neither mode deploys a native fallback.
Historical timing records do not validate this changed implementation.

## Explicit logical rank placement (V3)

V1/V2 and legacy profiles keep their original host-contiguous rank/device
binding. For other assignments, pass `--rank-map build/ranks.json` to candidate
creation. The JSON contains one `[host-index, CUDA-device-ordinal]` pair for
each logical rank, using the ordered host records and each host's CUDA-visible
device view. Every listed host/device must appear exactly once. For example,
two hosts with interleaved logical ranks use:

```json
[[0,0],[1,0],[0,1],[1,1],[0,2],[1,2],[0,3],[1,3]]
```

The generated format is:

```text
UCCL_LITE_RUNTIME_PROFILE_V3
network-hierarchy 2 4 4
ranks 8 0 0 1 0 0 1 1 1 0 2 1 2 0 3 1 3
<host records, in inventory order>
<operation/count rows>
```

Automatic providers retain per-logical-rank buffer-layout columns in V3.
The catalog advertises supported `profile_versions`; tools reject V3 for older
frontends. The C++ frontend validates the map, actual host/device/UUID ownership,
per-rank capability and common profile digest before preparing a backend. A
changed actual device or a duplicate binding rejects collectively. V3 changes
placement representation, not performance admission or capture support.

Distributed backends use a stable local device inventory and discover logical
ownership collectively, so changing application rank assignment does not change
the physical resource-calibration view. Single-host providers retain their
logical-rank device order. Set the control-root IPv4 address to the **actual
logical rank zero's host** when probing and launching.

The network validator offers `--rank-by node` and `--reverse-rank-devices` for
ABI-only placement tests and automatically writes V3 bindings for those cases.
See [global rank validation](../../docs/global-rank-placement.md).

## Organization and initialization

- `integration/nccl_lite/mscclpp_strict.cc` contains NCCL ABI translation,
  communicator lifecycle and profile checks.
- `include/uccl_lite/runtime/collective_backend.hpp` and `src/runtime/` define the
  backend contract, registry and provider adapters.
- `src/collectives/peer_packet.cu` owns the FP32 peer-packet implementation.
  `local_strict.cu` is a compatibility source wrapper for retained commands.
- `src/collectives/peer_staged.cu` owns the rank-independent BF16 peer-staging
  implementation; its registry adapter lives in `src/runtime/`.
- `src/runtime/host_packet_backend.cc` owns the BF16 packet adapter formerly
  implemented in `integration/nccl_lite/local_strict.h`.
- `include/uccl_lite/bootstrap/control.hpp` supplies independent CPU-only
  bootstrap. The old integration header forwards to it for compatibility.

NCCL unique IDs are backend-independent random session identifiers. Once ranks
agree on profile and capabilities, bootstrap distributes the selected
provider's private ID. Transport preparation completes during
`ncclCommInitRank`; the frontend does not construct a transport on the first
collective. Retained providers still have internal algorithm/layout restrictions
and caches that require further migration into explicit plans.

Legacy `nccl-local-strict`, `nccl-local8-strict`, and `nccl-mscclpp-strict` targets
use the same frontend object with a restricted registry. Their names remain for
recorded experiments; the common entry point is `nccl-runtime`.

## Current validation

The new peer-staged provider passes the public NCCL ABI oracle on two, three
and four L4s, including visible-device permutations. Each rank executes 168
calls across two communicator lifetimes: all three collectives, legal in-place
aliases, scalar-aligned pointers, full input/output guards, queued producers,
cross-stream chaining and bitwise AllGather including NaNs and signed zeros.
Capture is rejected before enqueue and recovery is checked; this is not graph
execution support. Source rank order is preserved for BF16 accumulation.

The peer staging protocol publishes data and retires readers in separate
kernels. MSCCL++ supplies registration, peer mappings, semaphore storage and
acquire-wait. Each pair's single writer publishes a sequence with a system
release store because the installed MSCCL++ `signal()` uses a remote atomic
unsupported on the local PCIe links. Compute Sanitizer caught that issue in the
first implementation despite passing numerical tests. The corrected four-GPU
memcheck reports zero memory errors on every rank. API-error reporting alone
is disabled for MSCCL++'s expected VMM/fabric capability probes; device memory
checks and API memory-access checking remain enabled. The original failing log
is retained in `build/peer-staged-validation/memcheck.log`, with the corrected
run in `memcheck-store.log`. This provides a comparable BF16 peer candidate;
automatic primary-entry selection still requires measured admission.

The unified library and all three restricted libraries compile, and their
catalogs and dynamic dependencies pass checks. CPU bootstrap passes payload,
reconnection, failure and occupied-port tests with 1/3/4/5/8/9 participants.
On the documented local four-L4 topology, both the retained launcher and the
new catalog-driven launcher pass the unchanged strict ABI numerical/guard
oracle: 198 calls per rank, two communicator lifetimes, FP32, message sizes
through 32 MiB, legal aliases and unaligned pointers. The final run including
backend-independent unique IDs is in `build/runtime-canonical-id-abi/abi.log`.
This is correctness validation; no new performance claim is made.

The network provider uses the common PCIe/NIC rail planner. It resolves
MSCCL++ transport slots by device name (including configured HCA reordering),
then assigns participating GPUs using bandwidth bounds and shared-resource
load. It no longer derives rail indices from CUDA device numbers or requires
matching rail indices at both ends of a connection. Multiple active ports on
one HCA and GPU–NIC paths with missing bandwidth information remain explicit
unsupported cases in this adapter; GDR registration is validated by MSCCL++.

With `UCCL_LITE_NETWORK_PROFILE` set on every rank, preparation exchanges live
endpoint identities and directed capacity constraints, then jointly selects
both ends of each declared peer pair. The largest common measured payload size
supplies an isolated-path prior; shared physical bounds constrain projected
concurrent traffic. An exact measured cohort baseline allows comparison with
complete measured alternatives; subsets and different per-flow extents cannot
reuse that evidence. Missing cohort data retains the advisory prior, while stale
or incomplete path evidence retains the physical plan. Corrupt or inconsistent
profile configuration rejects collectively. These resource measurements do not
qualify GPU-driven collective performance or enable another algorithm. See
[concurrent network measurements](../../docs/network-concurrent-model.md) and
[network calibration and planning](../../docs/network-calibration.md)
for the exact scope and profile requirements.

Two real four-L4 hosts have passed strict ABI checks with reversed HCA ordering
and with different HCA ordering on the two hosts, plus a full 1 KiB–1 GiB FP32
nccl-tests correctness sweep. See the migration record for exact artifacts and
timing scope. The single-host eight-GPU BF16 provider still has compilation and
registry checks only; a two-host run does not validate that different layout. The primary runtime's automatic multi-provider planning, general rank/rail
layouts, CPU/NIC resource calibration and remaining migration work are tracked
in [the runtime migration record](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/docs/topology-runtime-refactor.md).
