# Caller-owned native fallback

`PreparedLocalRuntime` can now compare its BF16 peer/host alternatives against
a caller-supplied baseline. `integration/nccl_lite/native_fallback.hpp` adapts an
existing native NCCL communicator to that interface. Core topology/planning and
CPU-router code still have no NCCL dependency. The prepared interface now
supports [BF16/FP32 candidates and FP16 native-only fallback](scalar-runtime.md).
The adapter also supplies native graph baselines to the core's
[owned graph selector](prepared-graphs.md), with independent graph qualification.
It now also supplies the baseline for
[distributed network admission](prepared-network-runtime.md), which compares
whole network calls during collective preparation and retains native execution
for unsupported layouts or candidates without a robust measured gain.

This is an explicit prepared interface, not a transparent NCCL interposer.
The strict `nccl-runtime` library remains a separate experiment frontend that
does not load native NCCL. Existing native calls, groups and graphs outside this
interface retain their original NCCL implementation.

## Ownership and preparation

The caller supplies a ready **blocking** native communicator from the named
native library and a common identity for its initialization/configuration.
The adapter checks communicator size, rank and CUDA device through public NCCL
APIs. It uses symbols resolved from that explicit library, checks that they
belong to one implementation, and rejects the strict UCCL ABI as a fallback.
It neither creates nor destroys the caller's communicator.

The fallback identity includes native/adapter build IDs, NCCL version, dtype,
caller configuration identity and per-rank NCCL environment. Each prepare,
including cache hits, revalidates the native environment and communicator.
After failed preparation, old handles cannot enqueue until another successful
collective prepare revalidates the runtime.

Preparation and all prepared calls must occur outside native NCCL group scopes
and CUDA capture. Callers must keep the communicator alive and manage ordering
when using it directly outside the prepared runtime; synchronize the runtime
before handing work back to ordinary native calls. This interface deliberately
does not make per-rank opportunistic group/capture fallback decisions.

```cpp
using namespace uccl_lite;
using namespace uccl_lite::runtime;

// context contains a shared session ID, logical rank, CUDA device,
// ordered local device list and maximum byte capacity.
auto native = integration::makeNativeNcclFallback(
    context, nativeComm, "/path/to/native/libnccl.so.2",
    "application/default-nccl-communicator");
LocalPreparationOptions options;
options.fallback = native;
PreparedLocalRuntime runtime(context, options);

LocalRequest request{NetworkOperation::AllReduce, count, rankLayouts};
auto plan = runtime.prepare(request);  // collective, idle, groups closed
runtime.enqueue(plan, input, output, stream);
runtime.synchronize();
```

`rankLayouts` is the same complete per-rank array on every participant. Each
entry declares in-place placement and exact minimum pointer alignment. One
process owns each GPU. The typed prepared interface supports a one-rank native
baseline and the existing 2–32-rank storage capacity; current hardware validation
does not cover every capacity. FP32 has generic and retained peer candidates;
FP16 remains native-only in this interface.

## Selection and resource lifetime

With a fallback present, peer capability is no longer a prerequisite for the
whole runtime. The caller baseline remains usable when peer/host alternatives
are unavailable. Available alternatives are compared separately against paired
baseline blocks: three ABBA rounds, two warmups and three checked timed calls per
block, max-rank host completion. Each alternative retains 18 samples and its
own 18 baseline samples. The native wrapper's completion tracking is included;
these are not raw NCCL-only enqueue timings. All calls use private guarded
buffers and complete input/output validation.

Baseline blocks contain no UCCL peer or host instance. A candidate must pass
correctness, improve its paired baseline median by more than 3%, and have a
nonoverlapping sample range. The lowest admitted candidate median wins;
otherwise the caller baseline is retained. The root broadcasts the result.
Unselected instances are destroyed. Resources required by earlier admitted
handles are restored so those handles remain usable.

`LocalPreparationOptions::fallbackOnly`, or `LocalRequest::fallbackOnly` for a
particular handle, requests the caller baseline collectively. An initial
fallback-only runtime performs no GPU-resource probing, candidate timing, peer
allocation or host/router initialization. `allowPeer` and `allowHost` provide
collective candidate constraints. Different per-rank policies reject before
execution; enqueue never infers a local fallback from pointer properties.

Cross-backend events order prepared calls across caller streams. Existing
native-only calls are not intercepted, so the caller still owns dependencies
between those calls and this interface.

## Build and evidence

```bash
make native-nccl-fallback native-fallback-check \
  MSCCLPP_IB_BUILD=/home/zhongjie/third_party/mscclpp/build-ib-pcieccl \
  MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx
```

The adapter builds as `build/integration/libuccl_lite_native_fallback.so` and
does not link CPU-router, MSCCL++ or native NCCL as DT_NEEDED dependencies; the
chosen native library is opened explicitly. It is not linked into the strict
NCCL replacement.

Current four-L4, dual-NUMA measurements retained native NCCL 2.28.7 for all nine
tested operation/count combinations (AR/AG/RS at 64/257/65536 elements).
The tested peer-staged and generic host alternatives did not qualify. Raw
paired samples, including these negative outcomes, are kept in the validation
logs. This does not claim that every UCCL algorithm loses on every workload.

Checks cover fallback-only initialization, native communicator ownership and
continued use after adapter destruction, one/three/four-rank cases, mixed
aliases/alignment, scalar tails, old-plan reuse, per-rank metadata disagreement,
and native environment changes on a cached plan. An explicitly labeled test
baseline with an added AG delay forces a peer-to-native producer/consumer chain
across two streams; that fixture is not performance evidence against NCCL.

Memory instrumentation is run separately without requiring a particular winner:
instrumentation changes timing. The instrumented identity is marked
`memcheck-correctness-only`, and no such samples are performance-admission
evidence. The earlier attempt to require the delayed fixture's timing winner
under instrumentation is preserved as a failed test assumption, not a GPU
memory error.

Transparent standard-ABI routing, external grouped/captured execution, broader
accelerated dtypes/layouts and multi-host native-backed selection remain open.
