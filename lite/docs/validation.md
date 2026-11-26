# Validation and limitations

## Eight-L4 four-collective comparison — November 10, 2026

The current single-node README compares the fastest measured UCCL Lite path
with NCCL 2.28.9 for BF16 AR/AG/RS/A2A, 1 MiB–1 GiB. CPU-router/NCCL
calibration completed 1,980 checked timed calls per placement. The separate
out-of-place All-to-All experiment completed 990 checked timed calls across
host, peer+host, and NCCL paths, checked every output element on all eight
GPUs, and passed a 1 MiB + 128 B tail case. Host staging is the faster UCCL
Lite A2A path at all 11 sizes, but NCCL wins all 11; this negative result is
retained in the [full record](../benchmarks/data/alltoall_8l4_20261110/README.md).
The [paired router/NCCL record](../benchmarks/data/router_vs_nccl_8l4_20261110/README.md)
also retains the 1 MiB ReduceScatter regressions.

## Eight-L4 local fallback and adaptive rerun — November 10, 2026

The standalone BF16 example defaults to a UCCL Lite two-NUMA fallback.
Its host-staged AllGather, small host reduction, and large peer-group reduction
paths have no NCCL dependency. The final dedicated eight-GPU correctness
check passed **61 cases** across AR/AG/RS, both placements, five sizes
including tails, two input generations, BF16 rounding and nonfinite values;
it also checked capture rejection.

Both 1-MiB–1-GiB local/router calibrations completed 1,980 checked timed
calls per placement. Fresh-process automatic replays selected the router at
**32/33** keys in each placement and retained local AllGather at 1 GiB under
the 3% selection margin. The [full v3 record](../benchmarks/data/local_fallback_8l4_20261110_v3/README.md)
includes paired samples, run-level ranges, profiles, packet-size diagnostics,
and selection logs; the [v2 record](../benchmarks/data/local_fallback_8l4_20261110/README.md)
preserves the earlier negative result. At 1 MiB, router AllReduce took
**0.488/0.529 ms** in/out of place with 128 KiB packets, versus
**0.588/0.579 ms** in the separate NCCL control. Larger local AR/RS paths
remain slower than the router, and their negative results remain in the data.

This validation is single-node, eager, BF16, and specific to two four-GPU
NUMA groups with intra-group peer access. It does not establish application
throughput or a win on every measured size.

## Eight-L4 NCCL control rerun — November 10, 2026

The hardware probe and NCCL-backed BF16 adaptive benchmark ran on eight L4s
with four GPUs per NUMA node, CUDA 12.9, driver 580.178.04, and NCCL 2.28.9.
Both in-place and out-of-place 1-MiB–1-GiB doubling sweeps completed all
checked AR/AG/RS calls: 33 workload keys × two backends × three repetitions ×
ten timed calls per placement. Their NCCL-backed automatic replays selected the
router at 24/33 in-place and 20/33 out-of-place keys. A separate 1-GiB
in-place AllGather fallback replay did not initialize router workers. The
[full local record](../benchmarks/data/cpu_router_8l4_20261110/README.md)
includes probe output, run-level ranges, every timed sample, profiles, and
selection logs. This eager benchmark is not an application throughput test.

`make router-checks` built with CUDA 12.9. The host policy (31), dispatcher
(28), NCCL identity (8), BF16 oracle (524,312), router core, and primitive
channel checks passed. The eight-GPU eager check passed 732 mixed operations;
the graph check passed 1,000 captured mixed collectives across five replays.

## README figure refresh — November 10, 2026

The README uses compact line and dot plots for collectives, with latency for
small messages and bandwidth for large ones, and paired latency bars for
fusion. The historical eight-GPU
primitive figures and labeled dense shape plot live in their detailed design
pages. Plot scripts read the same committed benchmark data; no kernel or timing
result changed for this presentation update. The then-current nine SVGs regenerated
deterministically, PNG previews were inspected, and local Markdown links were
checked. The full sweeps and negative results remain in the data files.

## Hardware profiler and local-only router policy — November 10, 2026

`make hardware_profiler router-checks adaptive-collectives` builds on the
four-L4 host. The new [hardware profiler](hardware-profiler.md) completed
three independent runs of topology, directed and concurrent peer PCIe,
concurrent host PCIe, CPU copy, and BF16 reduction probes, each with three
repetitions. Complete outputs and measurement scope are archived separately
from collective timings. Its output checks the first and last elements of
each GPU copy and each CPU worker's result.

Automatic router selection now accepts only exact local calibration. The
historical eight-L4 CSV remains readable for offline analysis but cannot
enable a router on a different host. Current policy checks pass **31** cases,
including rejection of archived AG/RS/AR wins and acceptance of a synthetic
exact local eight-GPU record. The rebuilt dispatcher passes **28** host checks;
the NCCL identity checker passes **8**. Four-GPU eager routing passes **732**
mixed operations, the graph path passes **1,000** captured operations, and
the opt-in GPU reduction check passes **26,569,136** scalar/guard checks.

The rebuilt BF16 adaptive example, with no loaded local profile, checked
AR/AG/RS at 1 MiB and retained NCCL for all three calls without creating
router workers. This was a short integration check, not a performance result.
No eight-GPU or two-node rerun was available in that validation pass; the
eight-L4 local rerun is recorded above.

## Direct peer AG/RS — November 10, 2026

The new four-L4 executable builds against NCCL 2.28.7 and checks every output
element of each peer and NCCL result for three changed input generations per
size. The 1-KiB–64-MiB sweep, plus a separate 1008-byte tail check, passed for
both operations. Thirty repeated peer/NCCL launches per backend and size were
timed for the three-run sweep; the archived min/median/max ranges and negative
results are in [peer AG/RS](peer-ag-rs.md). This validation covers only the
documented out-of-place float32 single-node layout. It does not establish
multi-node behavior or safe reuse under arbitrary concurrent application
streams.

## Two-node AG/RS and current AllReduce regression — November 10, 2026

Two hosts with four L4 GPUs each ran checked 1-KiB–1-GiB AG and RS sweeps.
Each new path verified three changed input generations per size, including a
post-timing generation; numerical error was zero on all ranks. The current
AllReduce kernel also passed a checked 1-KiB–64-MiB regression sweep after
the shared local-ring helper changed. An oversized cooperative grid was
rejected by the new occupancy guard instead of hanging.

The paired MPI-enabled NCCL comparison uses the same eight-rank CUDA-event
timing scope for AG, RS, and a fresh current-source AR run through 1 GiB.
AllGather shows a repeatable large-message gain;
the direct standard-input RS ring wins by about 15–17% at 32 MiB–1 GiB.
The fixed AR configuration wins by 4–9% at 32 MiB–1 GiB. AG and RS lose on
small messages. The first RS prototype received prepacked input
outside timing; its logs are archived but excluded from the accepted
[multi-node results](multinode.md). This validation does not establish
portability to other networks, arbitrary streams, or application workloads.

## Migration validation — November 5, 2026

The UCCL Lite reorganization was checked on four NVIDIA L4 GPUs, driver
595.58.03, CUDA 12.9, with NUMA pairs `(0,1)` and `(2,3)`. Source changes consist
of relocation, project-owned naming, include/build plumbing, formatting, and
documentation. The collective and fusion algorithms retain their existing
numerical behavior.

| Check | Result |
| --- | --- |
| `make all` | All single-node, fusion, and probe executables build; reusable MoE headers compile |
| `make multinode` | All five two-node executables build with Open MPI and IB-enabled MSCCL++ |
| AllReduce, 1 KiB–16 MiB, doubling sizes | All reported correctness errors are zero across eligible variants |
| All-to-All, 1 KiB–1 MiB, doubling sizes | Zero mismatches, including the host-staging baseline |
| Dense AG/GEMM, GEMM/RS, GEMM/AR small-shape smoke | All 11 reported checks have `bad=0`; maximum absolute difference 0.0078125 |
| MoE dispatch + GEMM, 8 tokens/rank | Checked variants have zero misrouted rows and zero relative error |
| DeepSeek-shaped full MoE layer, 2,048 tokens/rank, `-c 2` | Zero misrouted rows; normalized RMS error 0.00225 against the FP32 reference |
| Qwen-shaped full MoE layer, 8 tokens/rank, `-c 2` | Zero misrouted rows; normalized RMS error 0.00460 against the FP32 reference |

Builds retain unused-variable warnings in diagnostic benchmark variants. These
are not CUDA compilation errors. Smoke timings are not performance claims: they
use short runs with minimal warmup and no thermal settling.

The checked commands are in [build.md](build.md). The DeepSeek prefill check is:

```bash
./build/uccl_lite_fused_moe_layer \
  -b 2048 -e 2048 -w 1 -n 1 -r 1 -S 0 -c 2
```

The new source follows the parent repository's clang-format 14 configuration.
The launcher is checked with `bash -n`; figures are regenerated from the
committed data. Local Markdown links and the staged file inventory are checked
before publication. Generated binaries and validation logs stay in ignored
`build/` storage.

## README figures

At migration time, the README included six blue bar figures generated from committed
CSV/text measurements. Generation checks for missing series, non-finite or
nonpositive values, and invalid error ranges. The dense figure selects the
current comparison block. Historical TP-MoE, direct-decode, and older two-node
figures were removed; their source records and kernel implementations are
retained. The new two-node AG/RS figure was added and validated separately
above.
All bar axes start at zero and the plotted small-batch regressions are retained.
PNG previews were visually checked for layout; SVG and PDF exports use the
same plots. This is a
documentation/visualization update; no kernels changed and no GPU benchmarks
were rerun for it.

Standalone AllGather and ReduceScatter figures additionally use the complete
matched eight-L4 BF16 CPU-router summary from source revision `9d088c9`.
The CSV checksum, dimensions, bandwidth normalization, and both placements were
checked. The subsequent runtime port is described in
[CPU-router module](cpu-router.md); its historical eight-GPU measurements remain
separate from the current host's validation.

## CPU-router module integration

The optional static/shared library, adaptive BF16 benchmark, and all router test
executables build on the current four-L4 host (CUDA 12.9, driver 595.58.03).
The module's shared-library dependencies contain neither NCCL nor MSCCL++.
Only the benchmark links NCCL 2.28.7 as its existing-backend callback.

| Check | Result |
| --- | --- |
| Pure host backend policy | Historical test: 31 checks passed, including matched eight-L4 selection. Current policy requires exact local calibration; its updated checks are recorded below. |
| Dispatcher boundary | 21 checks pass without GPU initialization; fallback callback, no router allocation, explicit prepare, dtype/capacity semantics, placement and device-order validation |
| CPU BF16 reference | 524,312 scalar-oracle checks pass |
| Router core operations | Host `forward/process/multicast` checks pass |
| Primitive channels and ownership | SM/DMA channels, peer transfer, CPU tasks, 300 multi-reader generations, and owned bindings across queue wrap pass on four GPUs |
| Direct AR/AG/RS example | 36 cases pass across SM/DMA, tails, and both placements on two GPUs |
| Four-GPU eager pipeline | 732 mixed producer/collective/consumer cases pass |
| Four-GPU SM CUDA Graph pipeline | 1,000 captured mixed collectives across five replays and 128-entry queue wrap pass through both C++ and count-aware C APIs |
| Adaptive example | AR/AG/RS pass with auto fallback and with forced router, including 1 MiB in-place calls |

The imported primitive test originally assumed eight visible GPUs. It now uses
the available rank count (up to eight), discovers NUMA nodes/CPU affinity, and
uses host routing when the direct channel peer-access test is unavailable.

### Local benefit gate

The paired calibration uses BF16, four GPUs, out-of-place AR/AG/RS at
1/2/4/8 MiB, five warmups, ten timed iterations per repetition, and three
repetitions with alternating backend order. Each invocation is independently
checked outside timing. Min/median/max are over run-level medians. CPU-router
and fallback instances coexist during calibration; initialization is excluded.
The recorded `dispatcher_router_initialized` flag describes each dispatcher,
not whether another calibration candidate exists in the process.

All 12 workloads retain NCCL in automatic mode: no measured router median
beats the fallback. Loading the saved profile in a fresh process selects the
fallback for all 12 and initializes **no CPU-router instance**. Both the raw
[calibration summary](../benchmarks/data/cpu_router_4l4_calibration.csv) and
[selection audit](../benchmarks/data/cpu_router_4l4_selection.csv) are retained.
The large AllReduce timing outlier is preserved in the range rather than removed.

At 8 MiB, the paired median latencies (fallback / router) are 725.5 / 1670.1 µs
for AR, 422.9 / 491.0 µs for AG, and 400.4 / 1083.4 µs for RS. These eager
integration measurements are not replacements for the historical collective
plots or a comparison against the separate float32 MSCCL++ implementation.

At the time of this four-GPU port validation, no eight-GPU machine was used.
The archived eight-L4 measurements no longer enable automatic routing. The
later current-source eight-L4 rerun is recorded at the top of this page.

## What this validation does not establish

- The older reorganized multi-node AllReduce reference table remains a
  historical measurement; the current code's new AG/RS paths have separate
  two-host validation above. Other multi-node variants have not been rerun.
- TP-MoE device headers are compile-checked, not numerically exercised by an
  included host driver. Their former application-side harness is out of scope.
- One smoke pass does not establish race freedom, arbitrary-stream safety,
  behavior across repeated process lifetimes, or portability to other GPUs.
- FP16/BF16 accumulation and reduction order differ from some reference paths.
  Passing tolerance checks is not a claim of bitwise model-output equivalence.
- The opt-in persistent host-hierarchical prototype has known correctness
  failures and remains excluded from the validated transport paths.

## Extending the checks

For communication changes, test small and large sizes, uneven tails where the
algorithm supports them, repeated invocations, and delayed ranks. Inspect data
and flag visibility across GPU, host, and NIC domains separately. Validate both
the default path and any fallback before publishing a performance curve.

For fusion changes, compare complete measured work against an efficient unfused
control, preserve the dtype and rounding boundaries, and include small shapes
that expose launch and scheduling overhead. Use new seeds and representative
routing before making a dispatch policy. Always state whether routing-plan
construction and framework work are inside the timed interval.
