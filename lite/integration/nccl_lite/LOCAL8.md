# Strict NCCL ABI replacement on eight L4 GPUs

This is the single-host BF16 port of the [strict multi-node adapter](MSCCLPP.md).
It shares `mscclpp_strict.cc`, `host_control.h`, the identity launcher, the
version script, the ABI stress client and the nccl-tests route auditor. The
local backend uses UCCL Lite's existing multiprocess packet transport.

`make nccl-local8-strict` builds
`build/nccl_local8/libuccl_lite_nccl_local8.so`, with SONAME `libnccl.so.2` and
the loader alias `build/nccl_local8/strict/libnccl.so.2`. It has no native NCCL
dependency or forwarding implementation. The NCCL headers supply API types;
unsupported calls return errors. Initialization rejects a loaded native NCCL
library. The historical [mixed-routing adapter](DROPIN.md) is a separate target.

## Scope and preparation

The explicit candidate profile requires this host's eight ordered L4 UUIDs,
one GPU per process, devices 0–7, exact CPU/GPU topology, launch affinity,
CUDA/driver versions, library and launcher hashes, and packet configuration.
GPU count alone never enables the transport. Every rank participates in the
same CPU-only profile/session consensus used by the multi-node implementation.

Supported collectives are BF16 AllGather and SUM AllReduce/ReduceScatter at
profiled exact extents. The validation range is every power of two from 1 KiB
through 1 GiB, where bytes mean complete AG output or AR/RS input per rank.
Both legal layouts and element-aligned pointers are supported. Calls are
serialized per communicator; a completion event orders changes of stream.

Communicator construction creates no packet buffers or workers. The first
eligible collective prepares the transport explicitly, before steady timing.
Legal in-place AG preserves its own input segment. Overlapping RS stages the
smaller output and copies it back after all kernel input reads complete.
Misaligned buffers use aligned staging; these costs remain inside the ABI call.

After their first reduction job, the dedicated CPU worker and its activated
helper poll for later jobs until teardown. This avoids repeated sleep/wake
handoffs in sparse calls. The eight-rank configuration can occupy sixteen
pinned CPU threads: workers on 5/11/17/23/29/35/41/47 and helpers on
4/10/16/22/28/34/40/46. AG-only communicators leave these reduction workers
asleep. Application and CUDA threads in every rank exclude all of these
physical cores, including their SMT siblings. Full sweeps audit the actual
thread affinity through `/proc`. GPU-flag waits use busy polling on the reserved
cores. This CPU configuration is included in the identity and build evidence.

One live communicator per process, blocking initialization, and eager calls
are supported. Multiple communicator lifetimes and empty captured graphs are
tested. Nonempty capture, grouped collective submission, nonblocking init,
split, registration, unsupported dtypes/reductions and unprofiled extents return
errors; they cannot select NCCL. Grouped one-rank initialization is supported.

## Build and unchanged nccl-tests

Use CUDA 12.9 and the matching NCCL 2.28.9 headers on this testbed:

```bash
make nccl-local8-strict build/nccl_local8/strict_abi_check \
  build/nccl_local8/reference_timing_check build/nccl_local8/queue_check \
  CUDA_HOME=/usr/local/cuda-12.9 NCCL_HOME=build/nccl-2.28.9

# Keep external source and all generated files under build/.
git clone https://github.com/NVIDIA/nccl-tests.git build/nccl-tests-local8-src
git -C build/nccl-tests-local8-src checkout afd59abab774a6a4b492eb344bd0c845a0b7bb4d
make -C build/nccl-tests-local8-src MPI=1 \
  MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda-12.9 \
  NCCL_HOME="$PWD/build/nccl-2.28.9" \
  NVCC_GENCODE='-gencode=arch=compute_89,code=sm_89' -j8

python3 -B tests/nccl_mscclpp/run_local8.py \
  --abi build/nccl_local8/strict_abi_check \
  --queue build/nccl_local8/queue_check --checks \
  --out build/my-local8-nccl-tests
```

The runner freezes the library, launcher and unmodified test executables into
a new directory, records hashes and host identity, and acquires the existing
testbed lock. Candidate processes load only the replacement. Native controls
run in separate processes in NCCL/UCCL/UCCL/NCCL order. Default nccl-tests
options use one GPU/process, BF16/SUM, five warmups, twenty iterations, both
layouts, correctness checks, and no registration or application capture.

Every rank and size must have the expected route count and zero fallback.
Additional loader checks remove `LD_PRELOAD` and run all three programs through
the `libnccl.so.2` alias, from 1 KiB through 1 MiB.
Negative tests check unsupported dtype/reduction/group/capture and a missing
profile. The shared ABI client checks 234 calls/rank over two lifetimes,
including misalignment, per-rank layout variation, recycled allocations,
changing streams, delayed ranks, small/large transitions, full buffers and
guards. These checks do not claim in-process recovery after a failed peer.
The queue client submits 128 mixed calls/rank with changing sizes, per-rank
layouts and alternating streams, including a delayed rank, without host
synchronization between calls. It checks every output and adjacent guard.

## Performance acceptance

Correctness does not establish performance parity. Native NCCL is reported as
a separate control. The original README UCCL Lite binaries are the reference
for the zero-regression gate, following the multi-node port's methodology:

```bash
python3 -B tests/nccl_mscclpp/compare_local8_reference.py \
  --candidate build/my-local8-nccl-tests --out build/my-local8-reference
```

The driver verifies the exact original AR/AG full-refresh and RS recovery
binary hashes before remeasuring them. Both sides use input-ready per-call
CUDA events, maximum across GPUs, excluding preparation and checking full
outputs. The original launches eight GPUs from one process; the replacement
uses eight processes, as required by its ABI scope. Original AR is in place;
AG/RS are out of place. Both replacement layouts are compared against those
same original paths, giving 126 points. This process/layout difference is
recorded rather than hidden by comparing nccl-tests batch time to old figures.

The full protocol uses two independent jobs per backend, reverses their order,
and retains three repetitions of five warmups plus ten checked timed calls per
size/job. Medians retain every sample; no trimming or best variant per size is
allowed. Every point must have candidate/reference latency at most 1.0.
Partial or shortened screens never qualify the release.

Recompute the complete qualification from raw logs and frozen binary hashes:

```bash
python3 -B tests/nccl_mscclpp/audit_local8.py \
  --tests build/my-local8-nccl-tests --paired build/my-local8-reference \
  --library build/nccl_local8/libuccl_lite_nccl_local8.so
```

## Qualification

The [recorded eight-L4 qualification](../../benchmarks/data/nccl_local8_strict_20261119/README.md)
passes **126/126** size/layout points against freshly rerun original README
UCCL binaries. The exact library passed all nccl-tests, ABI, queue, unsupported
call, loader-alias and runtime-affinity audits. Native NCCL is faster at 25/126
points; the result establishes zero measured regression against original UCCL,
not native-NCCL parity.

The report preserves the failed candidates and worker-starvation diagnosis.
Its raw archive contains source/build identities and all accepted samples.
Four-/eight-GPU core eager/graph, policy and dispatcher checks pass. The shared
network frontend and FP32 ABI client compiled; a two-host rerun remains
unverified because authentication to the second host was denied.

Run the qualified replacement through the loader alias on this exact host:

```bash
python3 -B benchmarks/data/nccl_local8_strict_20261119/reproduce.py
```
