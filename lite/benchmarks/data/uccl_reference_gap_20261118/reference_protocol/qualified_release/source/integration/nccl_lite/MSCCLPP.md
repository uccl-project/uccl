> Qualified on the recorded **two hosts × four L4 GPUs**, FP32, eager,
> one GPU/process, 1 KiB–1 GiB powers of two, both layouts: **126/126 aligned
> comparison points have no measured regression versus the original README
> UCCL Lite binaries**. All nine unchanged nccl-tests sweeps also pass.
> The default policy is now `graph512`; native NCCL fallback is forbidden.
> [Qualification and default-build evidence](../../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/qualified_release/README.md).
> Earlier failures remain in the [progress record](../../docs/uccl-reference-no-regression.md).

# Strict UCCL Lite NCCL ABI replacement

The multi-node adapter now forbids native NCCL fallback. `make nccl-mscclpp`
(or `make nccl-mscclpp-strict`) builds `build/nccl_lite/libuccl_lite_nccl_strict.so`
with SONAME `libnccl.so.2` and a loader symlink at
`build/nccl_lite/strict/libnccl.so.2`. The former
`libuccl_lite_nccl_mscclpp.so` filename is also an alias to this strict library.
The superseded mixed-routing source is retained only in historical measurement
snapshots. The separate historical single-node BF16 integration is not used by
this target.

The strict library has **no native NCCL dependency, dynamic symbol forwarding,
or fallback implementation**. It owns communicator initialization, unique IDs,
CUDA allocations and supported collective calls. The replacement process must
not load the original `libnccl.so`; initialization checks `/proc/self/maps` and
rejects that configuration. Unsupported calls return errors rather than forward
to another implementation. `ncclGetVersion` reports the compatible header/API
version, not the presence of NVIDIA's NCCL implementation.

## Supported scope

- Two ordered hosts, four distinct NVIDIA L4s each, one GPU/process, rail map
  0/0/1/1. Exact GPU UUIDs, topology, CPU affinity, software binaries and relevant
  environment must match the recorded identity profile on every rank.
- Eager FP32 AllGather and SUM AllReduce/ReduceScatter at every profiled power
  of two from **1 KiB through 1 GiB**, both legal in-place and out-of-place
  layouts, including independently float-aligned pointers on different ranks.
- One live communicator per process; serialized host calls and caller streams
  ordered by completion events. Multiple communicator lifetimes are supported.
- Grouped initialization of one rank/process is supported. Grouped collective
  submission, nonempty CUDA graph capture, nonblocking initialization, split
  communicators, other reductions/dtypes and unprofiled extents return errors.
  Zero-count collectives are no-ops and can appear in an empty saved graph.
- Client registration/custom-reduction APIs return unsupported errors. Default
  nccl-tests execution uses `-g 1 -d float -o sum -m 1 -G 0 -R 0`.

A successful no-fallback result does **not** imply performance parity with NCCL.
The earlier mixed-routing 5% pass does not apply to this strict implementation.
The driver evaluates correctness, routing, and the performance gate separately.

## Reused components and data path

The [qualified default policy](POLICY.md) records the actual size cutovers and
resource use. Its one fixed configuration covers all measured points; the
compatibility base and rejected experiments remain available for reproduction.

MSCCL++ supplies `NativeAlgorithm`, `AlgorithmCollection`, stable context
caching, `TcpBootstrap`, `Communicator`, registration, `MemoryChannel`,
`PortChannel`, semaphores, `LL16Packet` and the IB proxy. UCCL Lite supplies the GPU collective
algorithms from `src/multinode/device_collectives.cuh` and the NCCL-compatible
API in `mscclpp_strict.cc`. No MSCCL++ dependency source is vendored or modified.
The tested external source revision is 52f659d, with an existing local
`gpu_utils.cc` modification recorded by the library hashes.

Construction creates no collective GPU buffers or workers. The CPU-only
HostControl channel performs exact rank/profile consensus, using the public
opaque unique ID as a session key. Every rank supplies the same reachable
rank-0 IPv4 address in `UCCL_LITE_MSCCLPP_CONTROL_ROOT`. No private NCCL fields
are read and no NCCL communicator is constructed. The first eligible collective
prepares the MSCCL++ transport outside steady-state timing.

<details>
<summary>Historical compatibility-base mechanics</summary>

Small calls through 512 KiB fuse client IO with the retained single-CTA
hierarchy and alternate two registered scratch banks. Tiny calls use separate
local/network LL16Packet slots, nonzero generation flags, a quiescent wrap reset,
bounded completion-queue draining and explicit teardown drain. No caller buffer
is remotely read. Local ready notifications and 512-thread serial transfers
reduce overhead at larger serial sizes.

Medium calls use UCCL GPU rings with direct client IO. AG uses four CTAs and
vectorized initial staging. AR forwards reduction results from registers into
peer scratch, and writes client output while gathering. RS streams both
standard-layout destination halves without packing a complete input tensor.
Naturally 16-byte-aligned AR pointers select a vector specialization; independently
4-byte-aligned pointers use the same UCCL algorithm with alignment-safe accesses.

AG/RS medium calls and 8-MiB AR use dedicated 2-MiB-aligned alternating banks;
1–4-MiB AR retains its original reuse handshake. All registered offsets remain
below the PortChannel 32-bit limit at the 1-GiB maximum extent. AG/AR use CE
graphs at 16 MiB and above; RS uses them at 8 MiB and above. Explicit CUDA
copy/compute dependencies preserve progress with blocking/pageable caller
copies. There is no local CPU CUDA-copy worker. IB uses MSCCL++'s proxy/verbs
path; the standalone benchmark retains its original PortChannel specialization.

Tiny LL calls and 8-MiB AR retire IB completion records in bounded batches of
32 calls. An AR call emits at most four WRs; the pending LL/AR combination plus
the next call stays below the configured 1024-entry send CQ. Alternating banks
and next-collective progress protect internal DMA sources. Caller input/output
are reusable after stream completion. Finalize, explicit `synchronize()` and
destruction drain outstanding transport work before resources are released.
This defers bookkeeping, never the numerical result or caller-buffer safety.

AR/RS graph first-hop sources are updated for each caller input, and GPU reduction
nodes read a stream-ordered input descriptor. The source graphs remain alive
for legal executable-node updates. Caller input is not registered or remotely
read. RS likewise reads client input locally and writes its final output
directly. Any remaining staging, graph updates/launch, synchronization and
completion events are included in the measured ABI call. Internal graphs do
not imply support for application capture: nonempty caller capture still
returns an explicit unsupported error.

</details>

## Build and validate strict correctness

Use the Makefile and external dependencies at identical paths on both hosts.
The default `UCCL_LITE_NCCL_POLICY=graph512` in [policy.mk](policy.mk) selects the
qualified configuration. The ordinary target builds the strict replacement and
its `libnccl.so.2` loader alias; no extra optimization flags are required.

Named policy changes invalidate the backend and component-check build. `base`
and `medium24` remain explicit reproduction options, without a zero-regression
claim. Historical commands with `EXTRA_NVCC_FLAGS` use
`UCCL_LITE_NCCL_POLICY=base`; changes to free-form flags require a fresh `BUILD`
directory. Preserve measured binaries when trying a different configuration.

```bash
make nccl-mscclpp-strict build/nccl_lite/mscclpp_strict_abi_medium128_check \
  MSCCLPP_IB_BUILD=/home/zhongjie/third_party/mscclpp/build-ib-pcieccl \
  MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx

python3 -B tests/nccl_mscclpp/run.py \
  --hosts node0,node1 --iface ibp55s0f0 \
  --mpirun /path/to/mpirun --tests /path/to/mpi-nccl-tests/build \
  --real-nccl /path/to/original/libnccl.so.2 \
  --library build/nccl_lite/libuccl_lite_nccl_strict.so \
  --max-bytes 1073741824 --abi-only --abi-extended --abi-boundary512 \
  --abi-serial-packet --abi-medium64 --abi-medium128 \
  --abi build/nccl_lite/mscclpp_strict_abi_medium128_check \
  --out build/my-new-strict-abi-run
```

Run the controller on node0 and use a new output directory. This ABI-only mode
stages the replacement library, symlink, client and launcher on node1 and runs
234 checked calls/rank over two communicator lifetimes. Cases include changing
streams, per-rank in-place/nonaligned pointers, rotated/recycled allocations,
delayed ranks and small/large transitions. Native collective controls are not run
in this mode. `--real-nccl` is retained for API/provenance identification and native
library absence checks; the candidate process must not load it.

The active performance gate is separate: unchanged nccl-tests AG/AR/RS over all
21 powers of two from 1 KiB to 1 GiB, both layouts, plus paired measurements against
the frozen original README UCCL binaries. The paired client uses fixed pointers,
reference-matched preparation and full internal send retirement. Every row must
meet the zero-slowdown criterion. Do not use the historical `--max-regression 0.05`
NCCL-relative gate as acceptance for this task, discard slower samples, or combine
best points from different libraries. Exact commands, hashes, repetitions, timing
scope and negative results are retained in the linked progress reports. The
recorded `graph512` campaign passes both gates; the default build has matching
loaded code/data and passes its own 234-call ABI run. This qualification applies
to the recorded placement/profile and supported API scope above.

The canonical default library was also staged and smoke-tested with all three
stock nccl-tests programs on the recorded hosts. Repeat those exact commands with
`bash build/uccl_default_release/reproduce_nccl_tests.sh`. The script uses the
matching identity profile under `build/uccl_default_release/abi`; these records
are specific to the measured placement, software and library. For another setup,
create and validate its own profile using the driver above.

Completed paired runs can be independently audited from their raw logs:

```bash
python3 -B tests/nccl_mscclpp/audit_uccl_pair.py build/CANDIDATE/paired
```

This checks all 27 runs and 126 rows, original binary identities, original
warmup/iteration counts, every sample index, finite positive timings, full-output
errors, exact rank/size routes and native-library absence. It recomputes untrimmed
means and medians and cross-checks the stored CSV. Exit code 2 means slower rows
remain; this paired-only audit does not replace the batch/build/release gates.

The separate batch-log audit verifies all nine frozen nccl-tests sweeps, including
test-executable and replacement-library hashes, finite timings, both layouts,
exact route counts, native absence and recomputed medians:

```bash
python3 -B tests/nccl_mscclpp/audit_nccl_sweeps.py build/CANDIDATE \
  --tests build/nccl-tests-mpi --reference build/uccl_readme_reference/fresh
```

Its comparison uses the recorded cross-harness reference and is diagnostic only;
passing it does not replace the matched per-call audit above. Both auditors retain
all reported samples and return 2 when slower rows remain.

## Historical NCCL-relative validation

The older driver mode without `--abi-only` runs separate native baseline processes
and uses `--max-regression` relative to NCCL. Those historical measurements are
retained below for provenance; they do not prove the active UCCL-relative target.

The [strict 1-KiB–1-GiB evaluation](../../benchmarks/data/mscclpp_strict_2x4l4_20261117/README.md)
passed correctness and per-size no-fallback checks in all 24 jobs. It failed
the performance gate in 91/126 rows; the worst slowdown was 31.33%. No
performance-parity claim or automatic performance admission is made.

Subsequent working changes and their measurements are tracked in
[strict optimization progress](../../docs/strict-optimization-progress.md).
The [final independent validation](../../benchmarks/data/strict_release_2x4l4_20261117/README.md)
passes **126/126 rows** under the established 5% per-row latency limit. Worst
candidate/native ratios are AG 1.04125, RS 1.03403 and AR 1.04712. All numerical,
per-rank/per-size route and native-library assertions pass. This is a scoped
5%-tolerance result, not a claim of zero absolute slowdown or full NCCL API coverage.
The report records the complete second AR round repeated after a controller
interruption, the unchanged binary hash, and all validation/provenance artifacts.
[Optimization history](../../benchmarks/data/strict_ll_rs_graph_2x4l4_20261117/README.md)
retains failed variants and screening outliers.

Additional core checks build through the Makefile:
`build/nccl_lite/network_sizes_check` covers 168 guarded calls/rank over two
lifetimes, including non-power-of-two extents and recycled caller allocations.
`build/nccl_lite/network_queue_check` checks 604 queued 8-MiB AR calls/rank,
mixed alignment/in-place layouts, a delayed producer, freeing caller buffers
before transport retirement, explicit synchronization and destruction draining.
Both link only the UCCL backend, MSCCL++, MPI and CUDA. These checks do not replace
the unchanged nccl-tests performance/reference checks.

The scoped ABI and error checks are not a guarantee of full NCCL API coverage
or in-process recovery after a missing peer/GPU fault. Abort frees an idle
transport; active/faulted transport state is retained until process exit to
avoid waiting indefinitely on an absent rank.

## Earlier records

[The initial adapter](../../benchmarks/data/mscclpp_adapter_2x4l4_20261117/README.md)
and [the mixed-routing 1-GiB pass](../../benchmarks/data/mscclpp_1g_2x4l4_20261117/README.md)
are historical measurements of implementations that allowed native fallback.
They do not satisfy the strict requirement and do not validate this changed
implementation.
