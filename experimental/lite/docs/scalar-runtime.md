# Dtype-aware prepared collectives

`PreparedLocalRuntime` now has an immutable `LocalPreparationOptions::dtype`.
BF16 and FP32 use the generic staged peer executor; FP16 currently uses a
caller-supplied native fallback. Dtype participates in collective preparation
agreement, byte extents, pointer alignment, alias offsets and candidate
admission. A mismatched fallback dtype rejects before backend initialization.

The generic peer executor uses actual participating rank count, with a declared
2–32-rank storage capacity. It preserves logical source order under device
permutation, accumulates reductions in FP32, and copies AG payload bits without
floating-point conversion. FP32 AG checks include signed zeros, infinities and
NaN payloads. Numerical reduction checks use an independent double-precision
reference and a stated FP32 tolerance, while input preservation and guards remain
bitwise checks.

## Candidates and selection

| Dtype | Generic peer | Other accelerated candidate | Caller baseline |
| --- | --- | --- | --- |
| BF16 | Staged peer | Generic host packet | Optional native NCCL |
| FP32 | Staged peer | Retained peer packet, when its full contract matches | Optional native NCCL |
| FP16 | None qualified here | None qualified here | Native fallback required |

The retained FP32 implementation is a capability specialization, not a rule
saying that four GPUs enable it. Admission requires FP32, a complete local peer
graph, mapped-host support, four participating ranks in two distinct physical
NUMA pairs, complete byte extent divisible by 128, and scalar alignment.
`pairedNumaRankOrder` packs each discovered pair into adjacent execution ranks;
application rank order can be interleaved or reversed. Physical NUMA IDs are
passed separately to host-memory placement and CPU-worker binding. Unsupported
dtype, scalar tails, unknown nodes or groups other than two pairs exclude it.
It is then qualified on the exact prepared request, including permutation costs.

For a nonidentity mapping, the adapter permutes AG output back into logical
source-rank order and RS input into execution-rank shard order using private
buffers allocated at prepare. Legal caller aliases remain in logical rank
space. AR retains FP32 sum semantics with the retained implementation's reduction
tree; floating-point reassociation is checked with the scalar oracle's tolerance.
The identity mapping adds no permutation buffers or launches. The selected
candidate's execution-to-logical mapping is exposed as
`LocalSelection::peerPacketRankOrder`. See [packet mapping](peer-packet-mapping.md).

Without a caller fallback, the generic peer implementation is the correctness
baseline and the eligible alternative must show a robust measured gain. With
a caller fallback, generic peer, host and retained peer packet are compared
against their own paired baseline blocks. No new size crossover is hardcoded.
The resource model remains advisory; measured correctness and the existing
>3%/nonoverlapping-range rule govern promotion. Old admitted handles retain their
required executors after later preparations.

`LocalSelection::samples` is indexed by `LocalBackend`: generic peer=0,
host=1, caller fallback=2, retained peer packet=3. Paired baseline samples and
correctness flags use `kLocalAlternatives` order `{Peer, Host, PeerPacket}`.

## Entry points

The unified strict NCCL library registers `prepared-local-fp32` and
`peer-staged-fp32`. The generic profile tool selects the prepared provider for
single-host FP32 experiments, including two and three ranks. The explicit
`peer-packet` provider and legacy reproduction target remain available.
Network providers retain their separate declared contracts.

FP32 preparation profiles use four-byte scalar size for extent and in-place
offset validation. For example, an AG shard of 64 FP32 elements advances by
256 bytes, not 128. Two-byte FP32 alignment is rejected. The public strict ABI
still requires complete extents divisible by 128; the core API separately
supports scalar tails.

```cpp
LocalPreparationOptions options;
options.dtype = ScalarType::Float32;
options.fallback = integration::makeNativeNcclFallback(
    context, nativeComm, nativeLibrary, configurationIdentity,
    ScalarType::Float32);
PreparedLocalRuntime runtime(context, options);
```

One runtime has one dtype. Callers must manage dependencies when switching
between separate runtime instances or ordinary native calls. Capture/group
limitations of the prepared interface remain unchanged.

The underlying staged peer executor exposes an [owned graph API](peer-graphs.md).
The core's [graph-aware prepared selector](prepared-graphs.md) independently
qualifies BF16/FP32 graph replay against the available graph baseline. The strict
NCCL frontend still rejects external capture.

## Validation and recorded decisions

The documented hardware is four NVIDIA L4s across two NUMA nodes on the dual
Xeon Silver 4410Y host. The generic FP32 core passes two/three/four-rank cases,
including shuffled GPUs and scalar tails. The retained candidate is exercised
on the qualifying four-rank NUMA layout. FP16 native-only execution also passes.

`build/scalar-abi-final/{two,three,four-permuted,four-packet}/` uses a public
NCCL/MPI/CUDA client with two communicator lifetimes, full guards, mixed aliases
and alignment, bitwise AG, numerical reductions and capture rejection/recovery.
Its 64-element FP32 in-place cases exercise 256-byte shard alignment. The earlier
explicit generic-provider check is in `scalar-abi-validation/explicit-generic/`.

Core samples are retained in `build/scalar-packet-final.log` and
`build/scalar-native-final.log`. In the recorded native-backed run, four of 18
operation/count/layout requests selected the retained peer packet; 14 retained
native NCCL 2.28.7. These are local preparation decisions, not a blanket speedup
claim. Timing is maximum-rank host completion including adapter tracking,
three ABBA rounds, two warmups and three checked timed calls per block. Every
sample is retained; no best-variant curve is reported.

`build/scalar-memcheck.log` checks the generic and retained FP32 paths and reports
zero memory errors on all four ranks. Instrumented timings are not performance
evidence. The declared 32-rank capacity, other GPU models and arbitrary retained
kernel layouts are not validated by these local tests.
