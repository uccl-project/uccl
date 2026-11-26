# Four-L4 strict drop-in work in progress

<!-- latest-local-status -->
**The large12/core2 fixed-library campaign passes all 126 points:** AG 42/42,
AR 42/42, RS 42/42. Independent audits confirm every gate. All ten correctness
suites pass, including stock nccl-tests, forced resets, mixed allocation classes,
stream reuse and preparation boundaries. Native NCCL is absent and fallback is zero.

The default Makefile policy is now `four-l4-fp32`. A fresh default build is
byte-identical to the measured library (SHA256
`9b5d25d4a701f11c7c6dd1642c37f9b7d06365aee0c2450218bd61e606e953d7`), and its
stock nccl-tests confirmation passes. Evidence and project source inputs are
archived under `benchmarks/data/nccl_local_strict_4l4_20261119`; usage and limits
are in `docs/nccl-local-dropin.md`. The final default-launch and evidence audits pass.

These are FP32 measurements on four L4s at `mibura-sky-test-02`, against frozen
original UCCL source `0fc3714b`. AR uses maximum-rank per-call host timing;
AG/RS use maximum-GPU per-call events. Each curve is one fixed library and
placement, both layouts, three repetitions; the original comparison selects
the fastest valid variant per size. The exact acceptance rule is unchanged.
<!-- /latest-local-status -->

The goal remains a single-node replacement with no native NCCL fallback,
unchanged nccl-tests execution, and no regression against the retained UCCL
Lite single-node algorithms. The final qualification is summarized above; earlier candidate states are retained below.
The two-host qualification does not validate this backend.

`make nccl-local-strict` builds
`build/nccl_local/libuccl_lite_nccl_local_strict.so`, with SONAME `libnccl.so.2`
and the loader alias `build/nccl_local/strict/libnccl.so.2`. It reuses the
multi-node strict ABI source, HostControl consensus, version script, identity
fingerprinting and profile launcher. A distinct local profile magic prevents
accidentally accepting a two-host profile. The separate backend uses MSCCL++
TcpBootstrap and CUDA IPC. There is no native NCCL dependency or forwarding.

The current experimental scope is four distinct L4s, one GPU per process,
FP32 SUM AR/RS and FP32 AG, exact profiled extents, eager execution. All other
strict ABI limitations still apply. Shape admission is explicit candidate
selection, never automatic based on GPU count. It is not a deployment profile.

Reproduce the check from `experimental/lite` (use a new output directory):

```bash
make nccl-local-strict
python3 -B tests/nccl_mscclpp/local.py \
  --library build/nccl_local/libuccl_lite_nccl_local_strict.so \
  --real-nccl /home/zhongjie/third_party/nccl/build/lib/libnccl.so.2 \
  --tests build/nccl-tests-mpi \
  --mpirun /usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun \
  --out build/local_strict_checked --max-bytes 67108864
```

The driver records software/topology identities, commands and executable
hashes, requires numerical success and every rank/size route, and checks
native-library absence and zero fallback. Timings are stock nccl-tests batch
host enqueue/completion, maximum rank, five warmups and ten iterations,
both layouts, FP32, one sweep per operation, no best-variant selection.
These are correctness screens without a paired performance baseline.

Initial logs in `build/local_strict_smoke/` pass all three operations over
1 KiB–64 MiB with zero errors. The `_push.log` runs use per-sender flags
written to peer memory and polled locally. The initial remote-polling
AllReduce experiment also completed; its long execution was initially
mistaken for a stall. Neither version is performance-qualified. The current
naive remote-reading data path is very slow at large sizes (the initial
64-MiB AR screen is approximately 298 ms). Preserve these negative results.

Next work: port the retained single-node algorithms and measured size cutovers
into the backend; add guarded changing-pointer/stream/lifetime ABI stress,
misalignment, negative-profile/API checks, and 1-GiB coverage; then run matched
paired comparisons against frozen single-node UCCL binaries. Do not use the
correctness smoke or historical two-host result as a no-regression certificate.

The repeatable driver completed successfully against the clang-format-14 build
in `build/local_strict_checked/` (see `result.json`, identity and all three
logs). This confirms the current source passes the same 1-KiB–64-MiB smoke;
it does not change the unqualified performance status.

## Serial push, copy engines, and small packets (2026-11-19)

The current backend replaces direct peer reads with the retained peer AG/RS
serial-write schedule and local reduction. Above 512 KiB it uses CUDA copy
engines for the same staged reduce-scatter/all-gather schedule. System-scope
publication plus all-reader completion barriers protect scratch reuse, including
in-place calls and transitions between operations and streams. Small calls use
MSCCL++ LL16Packet data/readiness, adapted from the retained RS packet path;
scalar client accesses support independently float-aligned pointers. Packet
flags have an explicit quiescent wrap reset.

The candidate packet cutovers are AR <=4 KiB, AG <=8 KiB, RS <=16 KiB, chosen
from the screens below. The attempted common 32-KiB cutoff was rejected:
32-KiB AR took about 144 us versus serial push's 33 us. This is still an
explicit candidate policy, not automatic performance admission.

Evidence (all on mibura-sky-test-02, four L4s, FP32):

- `build/local_strict_push_ce/`: unchanged nccl-tests AG/RS/AR, 1 KiB–1 GiB,
  both layouts, all 126 size/layout rows pass; per-rank route count 29/size,
  native absent and fallback zero. Source and binary saved in `source/`.
- `build/local_strict_push_ce_abi/`: **198 calls/rank, two lifetimes**, shared
  multi-node strict ABI harness compiled for four ranks. Full-buffer/guard
  checks, per-rank misalignment and in-place layouts, recycled/changing
  allocations, changing streams, delayed ranks, empty graph replay and
  unsupported-operation checks pass.
- `build/local_strict_packet_abi/`: same 198-call stress passes with packets.
- `build/local_strict_packet_1g_unalign/`: all 126 stock nccl-tests rows pass
  with one-float misalignment; wider packet cutoff retained as a negative
  result, with exact source and binary snapshot.
- `build/local_strict_cutover_1g/`: **current build**, all 126 aligned stock
  nccl-tests rows pass after restricting packet cutovers. The driver now also
  audits exact rank and route counts, all timing rows, finite positive timings,
  dtype and both layouts' numerical errors, using the multi-node audit pattern.

All nccl-tests screens use five warmups, ten timed calls, one sweep per
operation, maximum-rank batch timing; there is no best-variant curve selection.
These are not matched original-UCCL performance qualifications. Relative to
the initial adapter, the serial/CE screen reduced 64-MiB AR from 297 ms to
6.4 ms, AG from 75 ms to 3.4 ms, and RS from 74 ms to 3.1 ms. The packet
screen gives 1-KiB timings around 11–15 us. Larger messages and several
small/medium points still need work against the original UCCL reference.

The shared ABI harness still defaults to eight ranks without its local build
macro; its ordinary eight-rank target compiles. This turn did not rerun it on
two hosts. All collective backend changes are isolated to the local target.

For the next matched campaign, the original single-node reference source is
available at git revision `0fc3714b`: both `src/collectives/allreduce.cu` and
`src/collectives/peer_ag_rs.cu` exactly match the SHA256 entries in
`benchmarks/data/small_medium_4l4_20261113/identity.json`. Current build-directory
reference binaries have different hashes and must not be presented as the
historical binaries. Freeze/rebuild the matching source and record new binary
identities; preserve original warmups, iterations, timing scope, full-output
checks, and candidate envelope. AG/RS reference uses one process/four GPUs;
AR uses four processes. Reuse the multi-node paired protocol tooling while
accounting explicitly for those timing/preparation differences.

## Matched reference campaign started

The frozen source tree at `build/local_reference/source/` was extracted from
`0fc3714b` (including its original Makefile, headers and CPU-router module).
Its original AR, peer AG/RS and packet AG targets all build through that
Makefile. Their new binary identities are recorded separately from historical
binaries; the original source hashes are unchanged.

`tests/nccl_mscclpp/local_pair.py` reuses the strict identity launcher and
multi-node per-call client. Its first gate is AR only: original `--sm-variants
--fast-sweep`, 5 warmups, 10 timed calls, three rounds with reversed middle
order, compared with both candidate layouts. Each value is maximum-rank
host enqueue plus stream completion; input preparation and full-output checking
are outside timing. The reference selects the fastest correct three-run median
per size across the frozen original variants, as its published envelope did.
Candidate selection uses one binary, with no sample trimming. Every sample,
error, route count and run command is retained. The local client uses direct
caller-stream synchronization to match the original AR timing boundary.

The full 1-KiB–1-GiB run is `build/local_pair_ar_v1/`, controller log
`build/local_pair_ar_v1.log`. Do not infer completion from this entry: inspect
the live tool handle or `result.json` and all nine process exit records. AR
alone cannot certify AG/RS, whose original harness uses single-process,
maximum-GPU CUDA-event timing and different input preparation.

At the last live poll, controller PID 3756987 (exec session 32755) was running
`1_out`: round 0 and round-1 in-place had exited successfully. The first
round's preliminary AR comparison shows roughly 1.6–2.6x regressions at
small sizes and about 27% at 512 MiB. These are provisional single-round
observations, not the final median gate. Resume polling this same handle;
do not restart the campaign merely because observation expires.

After the controller finishes, run:

```bash
python3 -B tests/nccl_mscclpp/audit_local_pair.py build/local_pair_ar_v1
```

Exit 2 means measured regressions remain; exceptions indicate invalid or
incomplete evidence. The independent audit additionally checks the full
original variant matrix and reference warmup/timed correctness summaries.
Next optimization candidates are the retained pairwise hierarchy for small
and medium AR, and avoiding redundant event waits when consecutive calls
use the same stream. Preserve this measured binary during those experiments.

## AR paired gate and hierarchy experiments

The complete `build/local_pair_ar_v1` campaign is terminal (exit 2 for slower
rows). The independent `audit_local_pair.py` also exits 2 and reports **40/42
AR rows slower**, worst ratio **2.56043x**. All numerical, complete-variant,
per-sample, identity and per-rank route checks pass. This definitively fails
AR no-regression for the initial local strict adapter; no goal completion is
claimed. AG/RS still require their matched paired gates.

Three separate build directories preserve subsequent experiments:

- `build/local_hier_ll`: `UCCL_LITE_LOCAL_HIER_LL`, a scalar-client adaptation
  of the original `arHierLL` pair/cross/pair sequence through 64 KiB. The
  198-call ABI stress passes. Three paired 1-KiB–64-KiB rounds improve many
  small-message points relative to the initial adapter, but fail 12/14 rows
  against fresh original controls. Its 64-KiB LL range regresses badly.
- `build/local_hier_stream`: adds `UCCL_LITE_LOCAL_STREAM_ID`. CUDA's unique
  stream ID prevents mistaking a recycled handle for the prior stream; same-ID
  calls omit redundant event waits. The 198-call ABI and 600-call queue stress
  pass, but the paired screen fails 14/14 rows and does not show a repeatable
  gain over hierarchy alone. This is not promoted.
- `build/local_hier_credit`: hierarchy plus `UCCL_LITE_LOCAL_HIER_CREDIT`,
  without the stream-ID experiment. Two packet banks remove the all-rank
  completion barrier inside a stable hierarchy run. The phase dependency
  limits skew to one call: to finish call n+1 a rank requires its cross peer's
  n+1 publication, which follows consumption of n. Operation/size transitions
  rendezvous before reinterpreting scratch; explicit wrap reset and teardown
  still quiesce all ranks. The range is bounded to 32 KiB so both banks fit.
  The strengthened 600-call queue test and 198-call guarded ABI test pass.
  Paired-screen completion must be read from its result and controller log.

The new `tests/nccl_mscclpp/local_queue.cu` uses the real NCCL ABI, queues
600 calls/rank over two lifetimes without per-call synchronization, varies
input generations/pointers, mixes operations and sizes, delays one rank,
alternates streams, and destroys/recreates streams while prior work may still
be pending. Every output and guard is checked. Build with
`make build/nccl_local/queue_check MPICXX=/path/to/mpicxx`; run via the local
launcher's `--abi build/nccl_local/queue_check` option. The earlier stream-ID
run used the first queue version; the credit run uses the stronger version
with changing inputs and consecutive AR runs.

These are compile-time experiments in fresh BUILD directories, not new default
policy. Small screens use the same five warmups, ten timed calls and three
paired rounds as the full AR gate, but stop at 64 KiB and consequently allocate
smaller workspaces. They cannot certify the full-range production allocation.

The credit candidate's paired small screen is terminal, exit 2: **8/14 slower
rows** (versus 12/14 for hierarchy with the all-rank completion barrier).
Most in-place points through 16 KiB pass; out-of-place points at 2–16 KiB
still regress by approximately 5–13%, and 32/64 KiB remain much slower than
the original SM hierarchy. The 1-KiB reference is unusually slow in these
small-allocation runs; preserve it but do not extrapolate its apparent win
to the full-allocation gate. All three candidate processes are terminal and
no GPU campaign remains live at this handoff. No new default is promoted.

Next: port vectorized pairwise SM hierarchy for medium AR, and improve the
small hierarchy's out-of-place path. Then recheck actual stock nccl-tests,
queue/ABI stress and the full-allocation paired gate. Frozen original and
candidate binaries remain available for independent re-audit.

## Vector SM hierarchy and local event timing

`UCCL_LITE_LOCAL_HIER_SM` ports the original `arHierPut` schedule: vectorized
pair RS, cross-pair AR and pair AG, with MSCCL++ DeviceSyncer grid boundaries
between remote relationships. Scalar client loads/stores retain arbitrary
float-aligned pointers. It uses one CTA through 128 KiB and eight through
2 MiB. The initial port includes an additional per-writer system fence.
`build/local_hier_sm` passes the 198-call ABI and expanded 600-call queue
checks, but its three-round 1-KiB–2-MiB AR screen fails **14/24** rows.

`UCCL_LITE_LOCAL_HIER_REFERENCE_FENCE` uses the original arHierPut publication
sequence: the release/acquire grid boundary followed by the leader's system
release, without an additional system fence on every writer. In the separate
`build/local_hier_ref_fence` build, the same queue and ABI checks pass, and the
paired medium screen fails **7/24** rows. `audit_local_pair.py --max-bytes
2097152` independently confirms all samples, original variants, numerical
checks, binary identities and stored ratios. Both layouts pass at 128 KiB
and 512 KiB–2 MiB; 64 KiB and 256 KiB remain slightly slower. These are scoped
screens with a 2-MiB allocation, not full-allocation qualification.

The queue test now covers AR through 256 KiB and AG/RS through 1 MiB, retaining
600 calls/rank, changing inputs, asynchronous stream recycling, delayed ranks,
layout guards and two lifetimes. The expanded test passes both SM builds.

The shared reference-timing client now has a local AG/RS mode matching the
original peer harness's maximum-GPU CUDA-event timing, GPU input preparation,
output/reference buffer clearing and full-output verification outside timing.
The original remains one process/four GPUs; the adapter is four processes, and
this layout difference is explicitly reported. AR's host timing stays intact.
The new Makefile target `build/nccl_local/reference_event_check` leaves the
already-measured AR client binary unchanged. Actual event-client checks pass
AG/RS, both layouts, 1 KiB–1 MiB, two warmups/three timed calls, under the strict
library with all four native-absence/fallback-zero markers. Reproduction is
`build/local_hier_ref_fence/check_events.sh`. These short runs verify the
measurement client, not performance qualification or the AG/RS paired gate.

A separate `build/local_hier_cut16` experiment changes only the LL cutoff to
16 KiB (`UCCL_LITE_LOCAL_HIER_LL_BYTES=16384`) so 32 KiB uses the vector SM path.
Read its result/exit record before treating the screen as complete. No new
configuration has been promoted to the default local target.

The 16-KiB-cutoff experiment is terminal for queue, ABI, and paired-medium
screens: queue/ABI pass, while the independent paired audit reports **6/24
slower rows**. Failures are 8 KiB out-of-place (~5.3%), 32 KiB both (~8.9%),
64 KiB both (~2%), and 128 KiB in-place (~0.3%). Most medium points now beat
the original reference, but zero-regression is still unmet. All negative
rows remain in `build/local_hier_cut16/paired_medium/{comparison.csv,audit.json}`.
A stock nccl-tests 1-KiB–1-GiB, both-layout, one-float-misalignment screen was
then launched at `build/local_hier_cut16/nccl_1g_unalign`; inspect its live
handle/result before claiming completion. The frozen default local library
and the full AR baseline campaign have not been overwritten.

That full nccl-tests screen has now completed successfully: all **126
size/layout points** pass with one-float misalignment, exact route counts,
native absence and zero fallback. The event-client checks, queue/ABI tests,
paired screens and full nccl-tests run are all terminal. No GPU job remains
live. Remaining work includes the six medium-screen regressions, large AR
performance, the matched AG/RS comparison, and independent full-allocation
qualification of one final fixed library/policy.

## Bracketed AG/RS event comparisons

`tests/nccl_mscclpp/local_peer_pair.py` now runs and audits the original peer
AG/RS comparison. Two complete original sweeps bracket three candidate runs
per layout. Original sweeps retain the published 16-B start, 20 warmups,
10 measured calls, three rotated rounds, exact reference options and all
variants. Candidate comparison starts at the supported 1-KiB extent. Both
use maximum-GPU per-call CUDA events; original is one process/four GPUs,
candidate is four processes. The gate uses the fastest original variant's
**faster bracket median** versus the candidate's three-run median. This is
conservative reference selection, not sample trimming; all samples remain.

The driver audits source/binary identity, exact original variant coverage,
all sample indices and finite timings, candidate output checks, exact 30
calls/rank/size, native absence and fallback zero. `--audit-only` recomputes
stored comparisons without launching jobs; `--summarize-only` regenerates
results from terminal raw logs. An initial RS coverage checker incorrectly
expected LL variants at 16 B; original code excludes them because its shard
contains one float. The checker was corrected to require the exact original
matrix, then the existing completed logs were re-audited without rerunning or
discarding measurements.

The 1-KiB–2-MiB driver screens are terminal and audited:
`build/local_peer_ag_screen` fails **24/24** rows;
`build/local_peer_rs_screen` fails **10/24**. They pass correctness and
zero-fallback. AG's largest small/medium gaps are near 512 KiB (~2.32x);
RS gaps cluster at 32–512 KiB, while its tiny and 1/2-MiB points pass.

The full 1-KiB–1-GiB bracketed campaigns run sequentially under
`build/run_local_peer_full.sh` (exec session 94874, controller PID 3767059).
Directories are `build/local_peer_ag_full` and `build/local_peer_rs_full`;
controller logs and terminal `.exit` records are beside them. Inspect the
live handle before treating either campaign as stopped. Exit 2 means slower
rows, not a process/correctness failure. The measured library remains the
unchanged `build/local_hier_cut16/nccl_local/libuccl_lite_nccl_local_strict.so`.

A not-yet-built `UCCL_LITE_LOCAL_PACKET_CREDIT` prototype extends two-bank
packet reuse to tiny AG/RS and protects operation/size transitions with the
shared rendezvous. It is isolated behind a build flag and has no current
correctness/performance claim. The queue checker now includes consecutive AG
and RS groups with changing input, preparing a meaningful reuse test.

For larger AG, inspect `src/multinode/shared_host_packets.h` before inventing
new host allocation/bootstrap machinery. Its two NUMA-region owners (local
ranks 0 and 2), MSCCL++ bootstrap, registration, unlink lifecycle and three
packet slots appear reusable with a four-rank bootstrap; the eight-element
identity array is storage, not a hardcoded allGather participant count. A
local path must still be implemented and validated, including capacity,
ownership credits and timing. Existing multiprocess BF16 packet AG is gated
to eight ranks and cannot simply be selected on this four-L4 host.

The full AG campaign is now terminal and re-audited: **42/42 slower rows**,
worst ratio **2.38595x**. At 1 GiB the original packet-router bracket is
37,519 us, versus candidate medians 54,910 us out-of-place and 54,880 us
in-place. Correctness/native-absence/route audits pass. The full RS campaign
was still live at the last poll (same exec session 94874), with its before
reference and first candidate pair complete. Preserve both full campaigns;
neither the AG screen nor full AG result is a performance qualification.

Both full bracketed campaigns are now terminal and re-audited. Full RS fails
**34/42** rows, worst ratio **1.50148x**. In addition to its 32–512-KiB gaps,
the full allocation regresses at tiny sizes and at 16 MiB–1 GiB; its 1–8-MiB
points pass. This differs materially from the 2-MiB screen. Full AG remains
42/42 slower. The script's final `.exit` values are both 2 (performance), with
all eight process exits per operation zero and all correctness/routing audits
passing. No measurement process from session 94874 remains live.

The `UCCL_LITE_LOCAL_PACKET_CREDIT` prototype is now built separately at
`build/local_packet_credit`, combined with the previously tested cut16/SM
flags. Two parity banks replace the tiny AG/RS end barrier. To reach n+2,
a rank must have received n+1 from every source, and each source publishes
n+1 only after consuming n. The common transition rendezvous protects the
packet layout when operation or count changes. The updated queue test includes
consecutive AG and RS runs, not merely alternating operations.

This candidate passes 600 asynchronous queue calls/rank and 198 guarded ABI
calls/rank over two lifetimes. Its audited AG 2-MiB-allocation screen fails
**16/24** rows, down from 24/24: all 1–8-KiB layout points pass, with event
timings roughly 8.6–11.8 us versus original 14.3–14.4 us. Larger AG points
are unchanged algorithmically and still regress. The corresponding RS screen
was launched next at `build/local_packet_credit/rs_screen`; inspect its
result before claiming completion. No default policy is promoted and this
small-allocation result is not full-range qualification.

The packet-credit RS screen is terminal and re-audited: **10/24** slower
rows remain (the same medium points as the previous small-allocation screen),
with improved small-message headroom. All comparison/queue/ABI processes are
terminal at this handoff. Next priorities: vectorize serial AG/RS transfers,
reuse the multi-node shared-host packet storage for medium/large AG, and
recheck full-allocation curves. Full AR still needs large-message and residual
small/medium optimization; no operation has final full-range qualification.

## Vector transfers and reuse of the multi-node shared-host infrastructure

`build/local_serial_vector` adds `UCCL_LITE_LOCAL_SERIAL_VECTOR`, porting the
original 16-byte serial transfer width with safe scalar accesses for
float-aligned client pointers. Queue and ABI tests pass. Audited 2-MiB screens
still fail 16/24 AG and 10/24 RS rows, but AG at 512 KiB improves from about
99.6 us to 71.9 us (still slower than the original 42.5-us path). This is a
measured improvement, not qualification.

`UCCL_LITE_LOCAL_HOST_AG` now reuses **unchanged**
`src/multinode/shared_host_packets.h` for local host gather: MSCCL++ bootstrap,
two NUMA-bound shared regions, CUDA registration, mapping identity exchange,
and unlink/teardown are shared with the multi-node port. A local four-rank
MSCCL++ communicator suffices; no network data plane or CPU-router workers are
created. The GPU algorithm adapts original miniHostGather: upload each source
once, gather independently ready sources, then publish all-reader credits.
A monotonic tile generation and slot credits protect changing sizes and
queued calls. Preparation stays outside capture and steady-state timing.

`build/local_host_ag` uses the existing three 1-MiB source slots. It passes
queue/ABI checks and an audited 1-KiB–16-MiB AG screen, with 16/30 slower
points. The host path beats original mini-host at 1/2/4 MiB, but its extra tile
boundaries regress at 8/16 MiB by 3–7%.

`build/local_host_wide` adds `UCCL_LITE_LOCAL_HOST_AG_WIDE`: no remote host
exists on this placement, so the existing region's unused network-row space
holds **two banks of two local source rows, up to 4 MiB each**. This uses at
most 16 MiB of each existing 18-MiB payload region; a runtime bound checks
smaller profiles too. The multi-node header, constructor limits, allocation,
registration, and control layout are unchanged. The local kernel uses two
credit slots. Queue and ABI tests pass; the independent screen audit reports
12/30 slower rows. All 1–16-MiB layout points pass, although the 16-MiB margin
is only 0.1–0.3% and requires full-allocation confirmation.

`build/local_host_small` lowers the host cutoff to 16 KiB using
`UCCL_LITE_LOCAL_HOST_AG_MIN_BYTES=16384`. Queue/ABI checks pass; the audited
screen fails 10/30 points. The host path improves 512 KiB but remains slower
at 16–256 KiB. `build/local_host_four` additionally selects four host CTAs
through 512 KiB via `UCCL_LITE_LOCAL_HOST_AG_SMALL_BLOCKS=4`: correctness
passes, but it worsens 256/512 KiB and fails 12/30 rows. It is not selected.
All screens retain 20 warmups, 10 timed calls, original rotated three-round
reference sweeps before/after three candidate repetitions per layout, FP32,
maximum-GPU event timing. No samples or failed variants are discarded.

The fixed next candidate is `build/local_host_cut512`: wide host AG from
512 KiB through 16 MiB, sixteen host CTAs, vector peer transfers below it,
and the earlier AR hierarchy/small packet paths. A full stock nccl-tests
1-KiB–1-GiB, both-layout, one-float-misalignment check is running at
`build/local_host_cut512/nccl_1g_unalign` (exec session 2512). Check the live
handle/result before treating it as finished. No default local target or
performance-qualified profile has been promoted.

The cut512 full stock nccl-tests run is now terminal and passes **126/126
size/layout points** with one-float misalignment, exact route counts, native
absence and fallback zero. All GPU jobs described in this section are
terminal. Current improvements are scoped correctness and performance screens,
not full qualification. Remaining priorities are 16–256-KiB AG/medium RS,
AG above 16 MiB (where the original CPU packet-router wins), residual AR gaps,
and independent full-allocation qualification of one final fixed binary.

## Serial credits, small AG, and graph output experiments

All of the following use fresh BUILD directories and keep prior binaries and
raw logs. Each completed candidate passes the 600-call queue test and the
198-call guarded ABI test before measurement. Screens use the bracketed,
20-warmup/10-iteration/three-repetition event protocol through 2 MiB; they do
not establish full-allocation qualification.

- `local_serial_credit`: two scratch banks remove the steady AG/RS end barrier;
  the existing publication dependency bounds skew and the common rendezvous
  protects changes in operation/extent. AG fails 12/24 rows; RS improves to
  4/24 failures.
- `local_serial_threads`: original serial geometry (256 threads; 128 for
  64-KiB AG) reduces RS to **0/24 failures**. AG still fails 10/24.
- `local_serial_fence`: original arHierPut-style block publication followed by
  the leader's system release removes an extra per-writer system fence. RS
  remains 0/24, with further headroom; AG remains 10/24.
- `local_hier_ag`: pair gather followed by cross-pair gather, with two packet
  banks and a CTA phase boundary, passes stress tests and improves 16-KiB AG.
  The tested 16–64-KiB range fails 8/24 screen rows; 32/64 KiB remain slower.
- `local_batch_signal`: one system release fence precedes the peer flag stores
  (or both host-credit stores); receivers still acquire each flag. Stress
  checks pass and larger serial/host points improve. This first build retained
  the 64-KiB AG hierarchy condition; later builds actually honor
  `UCCL_LITE_LOCAL_HIER_AG_MAX_BYTES=16384`. Treat the recorded comparisons
  accordingly. Compile-time packet-capacity bounds now guard both hierarchy
  cutoffs.
- `local_serial_self`: direct self contributions avoid staging the caller's
  own row; AG hierarchy is restricted to 16 KiB. It passes stress checks and
  fails 7/24 AG rows; RS remains 0/24.
- `local_segment_copy`: copying AG output by source row removes a runtime
  divide/modulo per vector. It passes stress checks and fails only **4/24 AG
  rows**, at 64/256 KiB; RS remains 0/24.
- `local_shared_input`: shared-memory caching at those two sizes is negative:
  64 KiB is unchanged and 256 KiB worsens. Its 5/24 AG failures and successful
  correctness checks are retained; it is not selected.

Nsight Systems evidence is in `build/local_ag_profile64/`. Both profiled
clients validate output. The 64-KiB uncached candidate (`local_segment_copy`)
has a median serial kernel duration of 14.480 us over 30 calls on profiled
rank 0. The frozen original's push kernel at that size (identified by its
128-thread geometry) has median 6.272 us over 372 instances/four GPUs. This
includes the reference's correctness/warmup calls. Only candidate rank 0 is
profiled, so these are diagnostic GPU timings, not a paired admission result.
They identify substantial GPU-side overhead beyond submission costs.

`local_copy_output` splits the 64/256-KiB staged output into a stream-ordered
CUDA copy, keeping the parity bank protected until copy completion. It passes
queue/ABI checks and the **full 126-point, 1-GiB, both-layout, misaligned stock
nccl-tests sweep**. The second trace shows the 64-KiB sender kernel reduced to
9.408 us, followed by median 1.152 us Device-to-Device copies. Event timing
still includes extra host submission gaps; the small AG screen fails 5/24
rows, while RS remains 0/24.

`local_graph_output` caches the kernel+copy graph by extent and bank. Kernel
parameters are rebound every call. Copy binding reuse follows the multi-node
backend's legacy allocation-ID rule: recycled addresses cannot reuse stale
allocation metadata, and unknown kinds are rebound. Graphs are released
before scratch/IPC teardown. This optional target links CUDA driver APIs via
`NCCL_LOCAL_EXTRA_LIBS=-lcuda`; there is still no NCCL dependency or forwarding.
Queue and ABI tests pass. Audited AG failures fall to 4/24 (64/128 KiB), with
both 256-KiB points now passing; RS remains 0/24.

`local_batch_acquire` additionally observes peer flags with system-relaxed
loads then executes one system acquire fence before the CTA/grid join. This
pairs with the release fence preceding publication; self-signals are omitted.
Host reuse credits use the same fence ordering. Queue/ABI tests and independent
screen audits pass, with **4/24 AG failures and 0/24 RS failures**. At 64 KiB,
out-of-place is about 1.7% slower and in-place about 6.9%; 128-KiB in-place
also regresses. A full stock nccl-tests run is active at
`build/local_batch_acquire/nccl_1g_unalign` (exec session 48982); inspect its
handle/result before treating it as finished. No default policy is promoted.

The `local_batch_acquire` full misaligned stock nccl-tests sweep is now terminal
and passes **126/126** size/layout points through 1 GiB. Static inspection also
confirms no NCCL DT_NEEDED dependency and no undefined NCCL symbols. All jobs
in this section are terminal. RS's 0/24 small-screen result is a scoped advance,
not full RS qualification; large RS/AR and AG above 16 MiB still need work.
The graph candidate's remaining small AG gaps are at 64/128 KiB. Next inspect
the multi-node graph binding fast path / fixed-binding replay overhead, and
port the original ring schedules for the large reduction curves. Preserve the
negative shared-memory experiment and all preceding full-range failures.

## Large reduction ring and interprocess event port

`UCCL_LITE_LOCAL_RING` ports the original `0 -> 1 -> 3 -> 2 -> 0` ring to the
strict adapter for AR/RS from 8 MiB. It uses 8-MiB pieces, separate transfer
and reduction streams, distinct incoming slots for the three RS steps, and
caller-output fusion in the final reduction. AR follows with three ring AG
steps into a separate bank and copies received pieces to caller output on the
auxiliary stream. Caller completion joins all auxiliary work; the final
all-rank barrier protects scratch across operations and lifetimes.

`build/local_ring` passes the 198-call guarded ABI test and all 126 stock
nccl-tests points through 1 GiB with one-float misalignment. A new Makefile
target, `build/nccl_local/ring_queue_check`, queues 48 changing-input calls/rank
over two lifetimes, mixing tiny operations with 8/16-MiB AR and 32/64-MiB
AG/RS, alternating/recycling streams and delaying one rank. Full outputs and
guards pass. The ordinary queue source now also delays/recycles earlier when
rebuilt; old binary-identified evidence remains unchanged.

The full matched ring campaigns are terminal and independently audited:
`build/local_ring/ar_pair` fails **23/42** points (earlier full AR: 40/42),
while `build/local_ring/rs_pair` fails **15/42** (earlier full RS: 34/42).
Large AR now wins at 8/16/64 MiB in both layouts and is about 0.3–2% slower at
128 MiB–1 GiB. Large RS has roughly 4% residual regressions. Stock batch
1-GiB timings improved AR from about 104 ms to 82 ms and RS from 51 ms to
41 ms. Those batch comparisons are diagnostic; the paired gates determine
qualification.

`UCCL_LITE_LOCAL_RING_IPC_EVENTS` then replaces the ring's per-piece GPU
signal/wait kernels with the original CUDA-visible event dependencies.
Each rank exports arrival events through the existing MSCCL++ bootstrap.
A small CPU publication/acknowledgement region is carved from the unused
trailing network payload of the existing shared-host allocation, with an
explicit non-overlap bound. No multi-node allocator/header changes are made.
Before a producer re-records an event, its consumer acknowledges that it
has enqueued the corresponding wait. This prevents a fast CPU rank from
rebinding a slow rank's wait to a future generation. Publication is batched
per stage; local reduced events still pipeline transfers and reductions.
The event epoch is independent of the GPU packet-generation wrap. Imported
events are destroyed on all ranks before owned events are released.

`build/local_ring_ipc` passes guarded ABI, large queued stress, and the full
126-point misaligned stock sweep. Its 1-GiB batch timings are about 79.5 ms AR
and 39.7 ms RS, versus the GPU-flag ring's 81.9/41.0 ms. The two full matched
campaigns are currently running sequentially under
`build/local_ring_ipc/run_pairs.sh`, exec session **33545**, controller PID
**4021127**. Outputs are `ar_pair` and `rs_pair` beneath that directory, with
separate driver logs and `.exit` records. Poll that handle and inspect live
processes before inferring termination; do not restart an observed-timeout
job. No default policy or full performance qualification is claimed.

At the latest poll, the IPC-event AR campaign is terminal and independently
audited: **14/42** rows remain slower (GPU-flag ring: 23/42). The RS campaign
is still live in the same exec session 33545/controller PID 4021127. Continue
that existing run; do not start a duplicate. Consult the raw comparison CSV
for the exact passing/failing sizes before choosing further cutovers.

The audited IPC-event AR result passes **every 8-MiB–1-GiB point in both
layouts**. Its 14 failures are below 8 MiB. The large-size candidate/reference
ratios range from about 0.950 to 0.996; 1 GiB is about 0.992 out-of-place and
0.988 in-place. This is measured large-AR parity on the recorded placement,
not completion of the full three-collective goal. The RS campaign remains the
next live handle to poll.

The IPC-event RS campaign has now completed and re-audits successfully:
**13/42** points remain slower (GPU-flag ring: 15/42; earlier flat path: 34/42).
Five failures are tiny in-place cases at 1–16 KiB (about 8–18%); eight are
128 MiB–1 GiB in both layouts, with only about **0.03–0.52%** slowdown.
All intermediate sizes through 64 MiB pass. These full-allocation failures
remain authoritative even where earlier small-allocation screens passed.
Both IPC-event campaigns are terminal (performance exit 2), with no live GPU
job at this handoff. No default policy is promoted. Remaining work includes
small AR, tiny in-place RS, the residual large RS margin, and AG qualification
including the large host-packet path. The current IPC-event binary is preserved
under `build/local_ring_ipc/nccl_local/`.

## Large AG NUMA replicas

`UCCL_LITE_LOCAL_HOST_REPLICA` reuses the multi-node shared-host allocation
and the unchanged, independently linked CPU-router `CpuExecutor`. GPUs upload
local source packets; four pinned CPU lanes per NUMA group copy the two remote
sources into local replicas; GPU consumers read local-NUMA data. Workers start
only at the first AG above 16 MiB. No NCCL fallback, CPU-router core changes,
or application hooks are added. The local adapter owns cancellation and
bounded CPU jobs. Source/replica reuse is protected by GPU reader credits.
A cooperative kernel separates eight upload blocks, thirty-two source-consumer
blocks, and one publication block so upload, CPU transfer and gather can overlap.

Required module checks pass on the four L4s: 35 policy checks, 74 CPU dispatcher
checks, CUDA dispatcher capture/eager recovery checks, 732 eager mixed router
cases, and 1000 captured mixed collectives with five replays/ring wrap. Logs:
`build/local_host_replica/router_checks/`. These are correctness checks, not
new calibration of the historical eight-GPU results.

`build/local_host_replica` (two banks, 4-MiB source packets, four CPU lanes per
NUMA group) passes guarded ABI, large queue stress, and all 126 full-range
misaligned stock nccl-tests points. Its audited full AG campaign fails
**19/42** rows, improved from the earlier 42/42 failure; 1-GiB batch latency
falls from roughly 55 ms to 44 ms but remains slower than the original.

`build/local_host_replica3` uses three banks in the existing allocation's
unused network rows. Ring IPC publication metadata moves to reserved control
space before the payload, with an explicit overlap check. All correctness
checks pass; full AG still fails **19/42** points, but the large-size gap
shrinks: 1-GiB batch latency is roughly 39 ms, and the matched 1-GiB slowdown
is about 5–6%. Small large-message cases (32/64 MiB) remain much slower.

`build/local_host_pause` changes only the CPU flag polling to the original
packet pipeline's PAUSE instruction, retaining cancellation. An initial build
hit GCC/MSCCL++ BF16 typedef conflicts from an intrinsics header; using the
PAUSE instruction directly avoids touching either dependency. The failed
build/probe is retained separately. The rebuilt candidate passes correctness
and full nccl-tests; its bracketed full AG campaign is the live session
**84732** until its result says otherwise. Batch improvement is modest.

The original packet-capacity code confirms that its 4-MiB concatenated packet
contains 1 MiB per source. A source change for
`UCCL_LITE_LOCAL_HOST_REPLICA_BYTES` now gives large replicas their own view
while keeping medium direct-host packets wide. A common all-rank rendezvous
protects changes between these storage interpretations. This packet-size
candidate has not been built/measured yet; do not infer performance from it.

The PAUSE-polling full AG campaign is terminal and re-audited: **18/42**
slower rows remain. It does not remove the large-message gap.

`build/local_host_packet1m` retains three banks and PAUSE polling, and uses
1-MiB source packets only for the CPU-replicated large path. The medium direct
host view remains at 4 MiB/source. Explicit mode/size rendezvous protects
reinterpretation of their shared payload storage. Guarded ABI, large queue
stress, and all 126 full-range misaligned stock nccl-tests points pass. Its
full matched AG campaign is running under exec session **22415**, at
`build/local_host_packet1m/ag_full` with `ag_full_driver.log` alongside it.
Poll this same handle before restarting or interpreting a timeout. No default
policy is promoted. The remaining experiment to consider, if the packet-size
change is insufficient, is the original pipeline's per-source reader-credit
protocol; do not treat aggregate all-source credits as equivalent evidence.

## Per-source ownership prototype

The 1-MiB source-packet campaign is terminal and re-audited: **19/42** AG
points remain slower. Smaller packets alone are insufficient.

`UCCL_LITE_LOCAL_HOST_PER_SOURCE` splits GPU consumption counters and reader
credits by source. A source may recycle its row after its own readers finish,
without waiting for unrelated sources. Replicas use a separate control plane
and generation counter; the existing mode rendezvous joins readers before
switching between direct and replicated host layouts. The shared allocator
and CPU executor remain unchanged.

The first `build/local_host_source` candidate failed correctness on its first
32-MiB AG call. Added diagnostic reporting identified stale data in source-2
output rows. Investigation found the new readiness controls overlapped the
old ring IPC-event control placement. That failed binary/source/logs are
preserved in `failure_source/` and `diagnostic/`; no performance campaign ran.
The diagnostic checker has unchanged acceptance checks and prints the precise
rank, lifetime, generation, size, operation, index, and observed/expected value.

`build/local_host_source_v2` moves ring IPC metadata to the final 2 KiB of the
control area, checks the actual private-control bounds, and passes the
198-call guarded ABI test and large queued stress. Its normal sequence next
runs full misaligned nccl-tests and the matched full AG campaign. The active
exec session is **62987**; outputs are under that build directory. Inspect the
live handle/result before calling it complete or restarting it. No default
policy or performance qualification has been promoted.

The corrected per-source candidate's full AG campaign is terminal and
re-audited: **16/42** points remain slower (aggregate-credit 1-MiB candidate:
19/42). Correctness and route/native-absence checks pass. The per-source
protocol is an improvement, not full qualification.

`build/local_host_workers8` raises the CPU copy subdivision to four parts per
remote source (eight workers per NUMA group), matching the original worker
count. Worker CPUs are selected from the allowed NUMA CPU list and repeated
when necessary; the exact bindings are logged, and caller affinity is not
changed. The CPU-ready field range remains within the checked control region.
Guarded ABI, large queued stress, and all 126 full-range misaligned stock
nccl-tests points pass. Its full matched AG campaign is active in exec session
**79331**, with output under `build/local_host_workers8/ag_full`. Poll the
same handle before treating it as terminal or restarting it.

An optional `UCCL_LITE_LOCAL_HOST_BATCH_PUBLISH` source change is prepared but
has not yet been built or measured. It acquires each completed producer/reader
counter, resets that counter, then publishes only those ready conditions after
one system release fence per slot. It does not merge source credits or grant
early reuse. Keep the currently measured eight-worker binary unchanged until
its campaign finishes. No default policy is promoted.

## Batched host publication and self-copy follow-up

The eight-worker full AG campaign is terminal and re-audited: **21/42**
points remain slower. It improves some large out-of-place timings, but does
not qualify the curve.

`build/local_host_batch` enables the prepared batched-publication change.
Guarded ABI, large queue stress, and all 126 full-range misaligned stock
nccl-tests points pass. Its full matched AG campaign is terminal and audited:
**18/42** regressions remain. The large out-of-place tail is roughly 1–2%
slower, and the large in-place tail roughly 4% slower. All failed rows and
original bracket controls remain in the record; no default policy is promoted.

`build/local_host_self` is the next isolated candidate. It performs the
own-source copy in-place as well, after that source's upload completes, to
test whether the previous skip changed pipeline pacing. The source copy
remains ordered after producer reads; all existing per-source reader-credit
conditions remain. Its correctness-then-performance sequence is active under
exec session **3442**, with logs and results in that directory. Poll that
same handle before inferring completion or restarting the run.

## Correction: the original large AG path uses GPU progress

Inspection of the **actual frozen reference source and logs** corrects an
earlier interpretation in this report. `--packet-router-per-source` sets
`packetRouterGpuProgress=true`; the reference log explicitly reports
`packet_progress=gpu routing_cpu_workers=0`. Its configured `workersPerNode=8`
is inactive in this mode. Its consumers read the original mapped source rows,
not CPU-created replicas. Its per-source feature concerns readiness; reader
credits are published after the consumer crew finishes. Earlier wording about
matching the original CPU workers or original per-source reader-credit protocol
was incorrect. The CPU-replica experiments remain valid measured alternatives,
but they are not reproductions of the accepted reference algorithm.

The own-source-copy experiment (`build/local_host_self`) completed with
**17/42** AG regressions and is retained; it did not solve qualification.

`build/local_host_gpu_progress` now tests the direct GPU-progress path in the
adapter, using the existing shared-host source rows and no CPU copy jobs.
It uses 32 upload and 96 consumer CTAs plus the control CTA, with an explicit
cooperative-residency check. (The prior replica kernel uses 80 registers/thread,
so copying the reference's 248-CTA cap blindly would exceed residency.) The
GPU per-source credit protocol and transition bounds remain intact. This
candidate's correctness-then-full-comparison sequence is active in exec session
**36201**; inspect its live handle and outputs before claiming completion.
No default policy is promoted and all previous negative results are preserved.

The initial direct GPU-progress port is terminal and re-audited with **22/42
AG regressions**. Its 1-GiB stock batch latency worsens substantially. Further
source comparison identifies an important difference: the reference lets one
leader poll host readiness/credits and relays readiness through GPU-local
flags, whereas the first port let every CTA poll the host control words.

`UCCL_LITE_LOCAL_HOST_GPU_GATES` adds that GPU-local relay: one producer CTA
checks the four reader credits, and one consumer CTA per source checks source
readiness. Other CTAs acquire GPU-local gates. Epochs and per-source ownership
are preserved. This candidate is built at `build/local_host_gpu_gates`; its
correctness/full-sweep/matched-comparison sequence is active in exec session
**74556**. No result or qualification should be inferred before the handle and
result files establish completion. Prior failed configurations remain intact.

The GPU-local relay candidate now passes guarded ABI, large queued stress,
and all 126 full-range misaligned stock nccl-tests points. Its 1-GiB AG batch
time is approximately **35.6 ms**, compared with roughly 87–93 ms for the
first direct port. The matched full AG campaign is still active under exec
session 74556. This is promising diagnostic evidence, not yet qualification.
The candidate executes no CPU copy jobs; the original GPU-progress reference
also reports zero routing CPU workers. Earlier CPU-worker interpretations are
superseded by the explicit correction above.

The `local_host_gpu_gates` full AG campaign is terminal and independently
re-audited: **8/42** points remain slower, all at 512 KiB or below. Every
**1-MiB–1-GiB point passes in both layouts**. Large packet-path ratios are
approximately 0.905–0.960 of the original reference; 1 GiB is about 0.960
out-of-place and 0.959 in-place. The remaining failures are 8 KiB out, 32 KiB
out, 64/128 KiB both, and 256/512 KiB out. Full no-regression qualification
still requires these gaps and remaining AR/RS gaps to be resolved on one
final fixed library. All jobs in this section are terminal; no default policy
is promoted.

## Tiny AR tree and pending launch/layout experiments

`UCCL_LITE_LOCAL_TREE_LL` adapts the retained arTreeLL schedule to direct
caller IO and parity packet banks for AR through 8 KiB. Local pair leaders
exchange partial sums, then publish to their leaves. The existing transition
rendezvous protects changes between this layout and the hierarchy layout.
`build/local_tree_ll` passes guarded ABI, queued stress and all 126 full-range
misaligned stock nccl-tests points. Its full-allocation matched AR campaign is
active under exec session **58207**, at `build/local_tree_ll/ar_pair`.

Two separate source experiments are prepared but not yet built or measured:
`UCCL_LITE_LOCAL_SMALL_GRAPH` caches a small kernel plus completion-event
record node, rebinding kernel parameters each call; and
`UCCL_LITE_LOCAL_FIXED_PACKET_BANKS` uses a fixed 128-KiB stride within the
existing 256-KiB packet reserve. The latter is motivated by repeatable odd/even
latency differences in the first tree run. Supported algorithms use at most
96 KiB per bank. Neither experiment has a correctness/performance claim yet.
Keep their builds separate, preserve the current measured tree binary, and
poll the existing campaign handle before treating it as finished.

The tree campaign is terminal and independently re-audited: **11/42 AR
points remain slower**, versus the prior 14/42 full result. Tiny rows still
fail and the 8-KiB tree is not consistently better; this is not a promoted
policy. Exact ratios remain in `build/local_tree_ll/ar_pair/comparison.csv`.

`build/local_fixed_banks` adds only the fixed 128-KiB packet-bank stride to
that tree configuration. It has built and passed guarded ABI checks; its
queue/full-nccl-tests/full-AR sequence is active in exec session **92811**.
The small-graph source experiment is still unbuilt/unmeasured. Keep these
claims separate and poll the existing handle before inferring completion.

The fixed-bank full AR campaign is terminal and independently re-audited:
**14/42** points remain slower, worse than the tree-only 11/42 result. The
first-run tiny-size improvement did not establish a repeatable full-curve win;
fixed banks are not promoted.

`local_pair.py --allocation-bytes` now supports subset screening without
shrinking either the original AR workspace or candidate profile capacity.
The manifest records both measured maximum and allocation size, and the
independent audit checks the exact allocation arguments. Defaults preserve
the full-range gate. Screens remain scoped evidence and cannot complete the
goal in place of that gate.

`build/local_small_graph` adds only the prepared kernel-plus-completion-event
graph to the fixed-bank configuration. Its correctness sequence and 1-KiB–64-KiB
AR screen with **1-GiB allocations** are active in exec session **97318**.
The screen is `ar_screen` (driver log `ar_pair_driver.log`). Compare it with the
fixed-bank control, not an easier configuration, and poll the existing handle
before interpreting completion. No default policy is promoted.

The small-graph candidate passes guarded/queued checks and all 126 full-range
misaligned stock nccl-tests points, but its independently audited, full-allocation
small AR screen fails **12/14** points. It is not promoted.

`build/local_driver_tree` caches the CUDA tree function handle during prepare
and uses `cuLaunchKernel` for the tree, retaining the same kernel and completion
ordering. Correctness checks pass; its audited full-allocation small AR screen
still fails **11/14** points. No full performance claim follows from that screen.

Disassembly of the tree kernel shows a 32-byte thread-local stack with packet
pointer stores/reloads. `UCCL_LITE_LOCAL_DIRECT_TREE_POINTERS` replaces that
array with direct own/intra/cross pointers, preserving the original comparison
branch. `build/local_tree_registers` is the isolated candidate on the driver
launch configuration. Its correctness/full-sweep/full-allocation small-screen
sequence is active in exec session **32485**; inspect that handle/results
before treating it as finished. No default policy is promoted.

The direct-pointer tree candidate completed correctness checks but still failed
**11/14** full-allocation screen points. Resource inspection showed why the
expected stack removal had not occurred: the compiler instead copied the
64-byte `Peers` argument to local memory.

`UCCL_LITE_LOCAL_TREE_GRID_PARAM` applies the multi-node port's immutable
`__grid_constant__` parameter pattern. `build/local_tree_grid` reports **zero
stack bytes** for the tree kernel and passes guarded ABI, queued stress and
all 126 full-range misaligned stock nccl-tests points. Its independent
1-KiB–64-KiB screen with 1-GiB allocations fails **8/14** points, improved from
11/14 but still unqualified. The 1/2-KiB out-of-place points pass; their
in-place counterparts and 4/8-KiB cases still regress. All jobs in this section
are terminal. The fixed-bank and small-graph experiments remain negative
results, not promoted defaults. The whole goal remains incomplete pending
one fixed library passing every original performance gate.

## Hierarchy parameter and packet-geometry follow-up

An initial edit for `local_hier_grid` failed before compilation; its test
sequence was stopped and is not evidence for the intended change. The corrected
fresh `build/local_hier_grid_v2` uses direct, grid-constant packet pointers for
the hierarchy and restricts the tree to 1/2 KiB. It passes guarded/queued and
full-range misaligned stock correctness. Its independently audited 1-GiB-
allocation small AR screen fails **4/14** points (previous tree-grid: 8/14).

`build/local_packet_threads` sizes the tree/hierarchy blocks to their packet
unit count, capped at 1024 threads. It passes correctness and improves 16 KiB,
but the full-allocation screen fails **5/14** points. Do not mix its best rows
with other configurations to claim a passing curve.

`build/local_grid_peers` applies immutable grid parameters to the SM hierarchy
and const-reference peer helpers. Resource inspection confirms zero SM-hierarchy
stack bytes (previously 64). Correctness passes, but the independent screen
fails **7/14** points. This configuration is not promoted. All jobs described
here are terminal. Preserve the earlier promising packet hierarchy, and require
full-allocation comparisons on one fixed candidate before selecting a policy.

## Device-epoch graphs and banked SM hierarchy

`build/local_device_epoch` keeps packet graph parameters stable when caller
pointers are unchanged by generating sequence tags on the GPU. Its private
packet arena occupies the upper 128 KiB of the existing reservation; explicit
wrap reset clears both tags and packets. It passes 198-call guarded ABI,
600-call queued stress, and all 126 full-range misaligned stock points.
The independently audited 1-KiB–64-KiB AR screen with 1-GiB allocations fails
**5/14** points. In-place 1/2 KiB and some SM points still regress; this
configuration is not promoted.

`build/local_sm_banks` instead starts from the earlier hierarchy-grid v2
configuration and alternates SM hierarchy scratch banks. It removes the final
all-rank rendezvous only when both banks fit; pair/cross/pair dependencies
protect each bank until consumption. Size/algorithm transitions rendezvous,
and profiles with insufficient capacity retain the original completion fence.
Validation is in progress (controller session 23207). A separate SM queued
stress target covers repeated 32-KiB, 128-KiB and 2-MiB AR runs plus transitions,
changing pointers, delayed rank, recycled streams, guards, and two lifetimes.
No performance qualification is claimed before the recorded checks finish.

The SM-bank candidate passes the guarded ABI, 600-call queue, 126 full-range
misaligned stock points, the new 96-call medium queue, and a 64-KiB-capacity
stock sweep exercising the unbanked fallback. Its independent small AR audit
fails **8/14**, all at 1–8 KiB; the changed 32/64-KiB paths pass with ratios
0.835–0.929. These are scoped measurements, not full qualification.

Nsight Systems rank-0 diagnostics at 1-KiB AR show median traced kernel times
about 4 us and driver-launch API durations about 4–6 us after warmup. Event
record/wait and synchronization add host overhead. Traces are diagnostic only;
profiling overhead and rank skew preclude treating these as scored timing.
Artifacts: `build/local_sm_banks/trace_{out,in}_*.nsys-rep`.

`build/local_packet_specialize` retains SM banking and specializes the small
flat-packet kernel by operation, using grid-constant peer pointers without a
thread-local packet-pointer array. Validation controller: session 45780.
A full AR/RS/AG paired campaign is prepared in `run_pairs.sh` and must run only
after correctness passes. The source snapshot and hashes are under `source/`.

Packet specialization passes the 198-call guarded ABI, 600-call queue, and all
126 full-range misaligned stock correctness points. `readelf` confirms SONAME
`libnccl.so.2` with MSCCL++/CUDA dependencies and no native NCCL dependency.
Full paired measurement for this fixed library is now running, AR then RS
then AG, in session **11398**. The parent script preserves each gate's exit
status and stops on execution errors; exit 2 means valid measured regressions.
No candidate or policy has been promoted.

The full packet-specialized AR campaign completed, and independent raw audit
reports **7/42 failures**, all at 1–8 KiB (ratios 1.023–1.143). Every 16-KiB–1-GiB
point passes in both layouts on this one fixed library. Artifacts:
`build/local_packet_specialize/ar_pair/`. This improves the complete AR curve,
but remains a failed no-regression gate. RS and AG continue in session 11398.

The full packet-specialized RS campaign and raw audit report **8/42 failures**.
All 1-KiB–64-MiB points pass in both layouts, including all previous tiny
in-place failures. Remaining failures are both layouts at 128 MiB, 256 MiB,
512 MiB and 1 GiB, ratios 1.00324–1.00526. These small regressions still fail
the strict gate. Artifacts: `build/local_packet_specialize/rs_pair/`.
AG is running in the same controller session 11398.

Packet-specialized AG also completed: independent audit reports **5/42
failures**, at 32-KiB in-place and both layouts of 64/128 KiB. Every
256-KiB–1-GiB point passes. The complete fixed-library campaign therefore
passes numerical/runtime checks but fails **20/126 performance points**
(AR 7, RS 8, AG 5). Session 11398 is terminal (exit 2); all raw evidence and
three gate exit files are preserved. No default policy is promoted.

`build/local_serial_specialize` specializes medium serial AG/RS kernels by
operation and uses immutable grid parameters with offset-based parity banks.
The shared peer barrier receives a const reference, avoiding another copy of
those parameters. CUDA resource inspection shows zero stack bytes for both
specialized kernels, versus 128 bytes in the preceding generic serial kernel.
The guarded ABI passes; queued/full stock correctness and a full AG paired
gate are running in controller session **22192**. Snapshot under `source/`.

The serial specialization passes correctness but its full AG audit fails
**9/42** points, versus 5/42 for the preceding candidate. Zero stack storage
did not establish a performance improvement. This configuration is not
promoted. Session 22192 is terminal (exit 2); retain its raw negative result.

`build/local_ring_vector` returns to the packet-specialized full-campaign
configuration and vectorizes only the ring's local two-input reduction.
Aligned float4 accesses retain scalar handling for misaligned pointers and
partial tails, preserving in-place incoming/output aliasing. Guarded ABI,
600-call queue, 48-call ring queue, and 126 full-range misaligned stock checks
pass. The full RS pair is active in session **42741**. Source is snapshotted.
The current source also compiles through the default `nccl-local-strict`
target without experimental flags (`build/local_default_compile.log`); that
compile does not qualify the default's performance.

The vector ring passes correctness but full RS audit fails **9/42** points:
64-MiB out-of-place and both layouts from 128 MiB through 1 GiB. Large ratios
worsen to 1.025–1.036, so vectorization is not promoted. Session 42741 is
terminal (exit 2). The scalar ring remains the comparison configuration.

`build/local_ring_chunk16` keeps scalar reduction and changes only RS ring
pieces from 8 to 16 MiB. AR retains 8-MiB pieces. Guarded ABI, 600-call queue,
48-call ring queue and 126 full-range misaligned stock checks pass. Full RS
measurement is active in session **65843**; source snapshot is under `source/`.

Unimplemented follow-up: the original `pushSerialGather` writes directly into
peer caller outputs. The adapter instead stages and copies, and must supply
completion ordering absent from that isolated original kernel. Direct-output
IPC could remove staging, but requires lifetime IDs, per-call descriptor
exchange, import retirement on allocation changes, and entry/exit GPU ordering.
Pointer equality is insufficient; ranks must agree on eligibility and retain
UCCL staging for non-exportable allocations. No such path is enabled yet.
For tiny AR, the earlier device-epoch graph passed most small points; a next
experiment can combine it with the now-tested SM banks and evaluate unique
stream-ID wait elision and hierarchy at 2 KiB. Earlier stream-ID and graph
results were mixed; neither can be assumed to qualify this combination.

The 16-MiB RS chunk experiment passes correctness but fails **10/42** full RS
points. Both 64-MiB layouts regress about 8%; 128-MiB–1-GiB points regress
about 0.85–1.37%. It is not promoted. Session 65843 is terminal (exit 2).
All controllers from this iteration are terminal. Best complete fixed-library
evidence remains `build/local_packet_specialize`: AR 7/42, RS 8/42, AG 5/42
failures, or **20/126** total. Every correctness/runtime check passed, but the
user's no-regression goal remains incomplete and no default is qualified.

Another unimplemented RS hypothesis: replace the balanced ring with two
hierarchical stages using the existing IPC-event publication protocol. Each
NUMA pair first reduces interleaved destination shards (rank parity p owns
shards p and p+2), then cross-pair peers exchange only the shard needed by the
other NUMA group and reduce their own output. This moves 2 shards per rank
within NUMA and 1 across NUMA, versus the ring's 3 shards on two cross-NUMA
senders. It may help if the cross-NUMA link limits the ring; this is a bound,
not measured evidence. Local partials need separate scratch slots, imported
partner events, binding publication/acknowledgement, and final all-rank scratch
retirement. Preserve scalar 8-MiB ring as control; do not combine this with
vectorization or larger chunks that already failed.

## Hierarchical large ReduceScatter experiment

`build/local_hier_rs` implements the interleaved-shard hierarchical RS from
128 MiB upward, on the packet-specialized full-campaign configuration. Two
local shard transfers/reductions precede one cross-NUMA partial exchange and
final reduction. Distinct slots retain all partials until the existing final
retirement barrier. The existing IPC publication/acknowledgement mechanism
now selects the local partner for stages 0/1 and cross partner for stage 2;
additional imported events alias existing imports where the partner matches.
AR and smaller RS retain the original ring. Scalar 8-MiB pieces are unchanged.

New `hierarchy_queue_check` covers changing 128/256-MiB AG/RS and ring AR,
then six consecutive changing 256-MiB RS inputs; output guards, delayed rank,
recycled streams, and two lifetimes remain checked (24 calls/rank total).
The build and validation/full RS sequence is active in session **6174**.
Source/test/Makefile hashes and snapshot are under `source/`. No result is
claimed until the controller and independent audit finish.

Hierarchical RS passes every correctness/queue test but its independent full
RS gate fails **8/42**, both layouts from 128 MiB through 1 GiB, ratios
1.094–1.101. Fewer cross-NUMA bytes did not yield lower latency. Session 6174
is terminal (exit 2); this path is not promoted.

`build/local_epoch_stream` returns to the packet-specialized full-campaign
base, adds the earlier device-epoch small graph, chooses hierarchy from 2 KiB,
and uses CUDA's unique stream ID to elide same-stream completion waits. Its
correctness/full-stock/full-allocation small AR screen is being launched.
This is a new fixed combination, not a combination of best measured rows.

The epoch/stream-ID tiny AR combination passes guarded/queued/full-stock
correctness. Its independently audited 1-GiB-allocation small AR screen fails
**2/14** points: 2-KiB in-place (1.0252x) and 8-KiB in-place (1.0520x).
All out-of-place screen points pass. Session 39466 is terminal (exit 2).
Full-range performance is not qualified by this subset.

`build/local_ag_epoch` returns to the packet-specialized full-campaign base
and uses a prepared publish/copy/completion graph at 64/128-KiB AG. The sender
maintains a private device epoch (flag word 129), using barrier words 16–19;
a parallel kernel copies the selected parity bank to the caller. Kernel
parameters are rebound on pointer changes; no CUDA memcpy allocation binding
is cached. Normal operation/size transition rendezvous and global wrap reset
remain in force. A new 192-call `gather_queue_check` stresses repeated and
changing 16/64/128/256-KiB AG, other operations, pointers and streams.
The correctness/full-stock/queue/full AG sequence is being launched.

The AG epoch graph's guarded ABI check passes. Resource inspection reports
zero stack bytes for both publish and copy kernels. Controller session:
**87690**; source/test/Makefile snapshot and hashes are under `source/`.
The tiny AR epoch/stream build's source snapshot is copied from the preceding
hierarchical experiment: there were no source edits between those builds;
their compile-time flags are distinct and retained in their build logs.

The AG epoch graph passes guarded ABI, 600-call queue, 192-call gather queue,
and all 126 full-range misaligned stock correctness points. Its independent
full AG audit fails **13/42** points (8–512 KiB, with most outside the two
changed sizes). At the changed 64/128-KiB sizes it still fails both layouts,
ratios 1.023–1.103. It is not promoted. Session 87690 is terminal (exit 2).

Nsight Systems diagnostics are now running in session **15591**, under
`build/local_ag_epoch/trace64/`: candidate 64-KiB out/in with 1-GiB allocations,
20 warmups/10 calls, then the original 64-KiB–1-GiB sweep retaining 1-GiB
allocations (5 warmups/10 calls/one repeat). CUDA event completion tracing is
disabled to avoid profiler-induced event dependency artifacts. These traces
are diagnostic only and cannot replace the unprofiled paired gates.

Node-level AG traces completed successfully. At 64 KiB, median publish kernel
durations are 9.6–13.8 us across candidate ranks/layouts, versus 5.9–6.4 us for
the original 128-thread push kernels. The candidate's parallel copy takes
1.248 us, with a 0.192-us node gap. See `trace64/summary.json` and raw CSVs.
Tracing perturbs launch skew, so these locate costs rather than certify speed.
All trace controllers are terminal.

`build/local_ag_packet_grid` extends specialized flat AG packets through
128 KiB and distributes packet units across up to 16 CTAs. It retains each
packet's owner, rank-order reduction logic, parity reuse credit and transition
rendezvous. Two flat AG banks require up to 512 KiB; the independent epoch
packet family sits above them with its own 128-KiB region. Allocation and
wrap clearing use the enlarged bound. Other operations retain their geometry.
The candidate starts from the packet-specialized full-campaign configuration,
not the unsuccessful AG epoch graph. Correctness, the 192-call gather queue,
and full AG measurement are being launched.

The extended packet build's guarded ABI check passes. Controller session:
**11041**. Source/test/Makefile snapshot and hashes are under `source/`.

The multi-block packet extension passes guarded ABI, 600-call queue,
192-call gather queue and 126 full-range misaligned stock correctness.
Its full AG audit fails **6/42** points; the changed 32–128-KiB path
regresses substantially (first out-of-place means approximately 28/55/91 us).
It is not promoted. Session 11041 is terminal (exit 2). Preserve the expanded
arena bounds in this optional experiment; do not silently enable its cutoff.

## Current diagnosis and next checks

The 128-MiB RS diagnostic is complete under
`build/local_packet_specialize/rs_trace128/`. Both candidate and original
retain 1-GiB allocations; candidate measures 128 MiB only (5 warmups/10 calls),
original starts at 128 MiB and retains its full 1-GiB maximum (5/10/one repeat).
For the last ten ring calls at 128 MiB, each rank contributes 120 8-MiB copies
and reductions. Copy medians match closely: original 405.3–405.8 us, candidate
405.4–406.2 us. Reduction kernels are about 40 us in both. Within-call copy
gaps are 5.1–5.2 us in the original and 6.6–7.2 us in the candidate; the
candidate's final retirement barrier adds 7.1–8.4 us. Raw traces, CSVs and
`summary.json` retain the evidence. These are profiled diagnostics, not gates.

The next ring hypothesis should target handoff/retirement. One possible
experiment is a stream-ordered system-fenced value write after each CE copy,
with the destination's flag wait fused into its reduction kernel. Probe the
actual pointer/capability first and justify scheduler progress for the two
streams. Existing `ce_allgather.cu` has separate historical memop experiments:
those small eager barriers lost, and do not validate this ring protocol.
Do not remove the retirement barrier without proving receive-slot lifetime;
alternating full RS scratch banks may provide the required reuse credit but
must also rendezvous at operation/size changes and pass delayed queued tests.

The direct caller-output IPC idea has an additional constraint: the installed
CUDA runtime header (`cuda_runtime_api.h`, cudaIpcOpenMemHandle documentation)
states that cudaFree before importer close is undefined. Querying BUFFER_ID
on the next collective cannot repair that earlier lifetime violation. A
persistent legacy-IPC caller-buffer cache therefore requires a valid lifetime
mechanism; do not enable it merely because recycled-pointer tests pass.

All benchmark and profiling controllers from this iteration are terminal.
Best complete fixed-library evidence remains `local_packet_specialize` with
20/126 failures. `local_epoch_stream` improves the scoped AR screen to 2/14
failures but has not passed the full three-collective gate. The goal is active;
no default routing policy or candidate has been promoted.

## Stream-value ring handoff

`build/local_rs_memop` starts from the packet-specialized full-campaign base.
RS at 128 MiB and above keeps scalar 8-MiB copies and the three ring phases,
but replaces each exported arrival event with a default-fenced 64-bit stream
value write to the receiving GPU's private flag page. The reduction kernel
fuses a CTA-leader system-acquire flag wait with the unchanged scalar sum.
Local native events still order the two streams. Phase-zero CE copies have
no reduction dependency; all previous-phase reductions precede next-phase
flag waits, so blocked reduction CTAs cannot prevent their own needed CE work.
The final all-rank retirement barrier remains. AR and smaller RS are unchanged.

Preparation checks the device capability and executes a remote IPC-address
write/synchronize/readback probe, then clears its reserved word. Flag word 448
is separate from ring data flags 256–447 and other collective controls. The
experiment uses no NO_MEMORY_BARRIER or unsupported remote-flush option.
Controller **84355** runs guarded ABI, normal/ring/large changing-input queues,
full misaligned stock correctness, then full RS comparison. Source is frozen
under `source/`; no performance claim precedes completion and audit.

The stream-value handoff passes the probe and all correctness/queue checks,
but its full RS audit fails **10/42** points. Raw results are under
`build/local_rs_memop/rs_pair/`; it has not qualified no-regression. Session
84355 is terminal (exit 2).

`build/local_rs_banks` instead keeps the original IPC-event handoff and
alternates two complete receive banks for RS at 128 MiB and above. Six shards
occupy 1.5 times the input extent, within the existing twice-capacity allocation.
Before a predecessor can begin call n+2, its completed n+1 three-step reduction
must have received this rank's phase-zero n+1 contribution; that contribution
follows this rank's completion of n. Thus the predecessor cannot overwrite
the bank used by n until its consumer has retired. The final global barrier
is omitted only for these banked calls. A distinct transition mode and count
check rendezvous before an operation or size changes the interpretation.
Local stream completion and IPC record-publication acknowledgements remain.

Controller **70191** runs the full correctness/three-queue/full RS sequence.
Source/test/Makefile snapshot and hashes are under `source/`. The memop and
hierarchical RS experiments are explicitly incompatible with this initial
banking flag; they are not silently mixed with its reuse argument.

The banked IPC ring passes every correctness/queue test but its full RS audit
still fails **8/42**, both layouts from 128 MiB through 1 GiB. Ratios range
1.00064–1.00568. The 128-MiB gap is smaller, but this remains unqualified.
Session 70191 is terminal (exit 2). The next isolated geometry experiment
uses 12-MiB chunks only for these large banked RS calls; smaller sizes keep
8 MiB. The existing 16-MiB whole-range experiment remains a negative result.

`build/local_rs_banks12` limits 12-MiB chunks to RS inputs >=128 MiB and retains
8-MiB chunks below that boundary. It retains the proven two-bank reuse protocol.
Guarded ABI, all queues, and the 126-point full misaligned stock sweep pass,
including partial final chunks. Full RS measurement is active in session
**85254**. Source/test/Makefile snapshot and hashes are under `source/`.

The 12-MiB banked ring passes correctness but full RS audit fails **7/42**,
with the worst remaining ratio about 1.006. One 128-MiB in-place result is
only barely above 1, but still fails the strict gate. Session 85254 is terminal
(exit 2). No qualification follows from the one-row reduction in failures.

`build/local_ag_segmented` starts from the packet-specialized full-campaign
base. At 32–128-KiB AG, one cooperative kernel has up to four 512-thread CTAs,
each owning a disjoint vector slice. Each CTA caches its input vector, sends
to the three peers, publishes its own system-release readiness flag after
its writes, acquires the three corresponding local flags and copies only its
slice to caller output. The own contribution goes directly to caller output.
Ready cache lines occupy words 160–211, outside other active controls.
Two-bank serial reuse credits and operation/size rendezvous are unchanged;
cooperative residency is checked before use so peer block waits can progress.

Guarded ABI, queues (including the 192-call gather queue) and full misaligned
stock correctness pass. The full AG comparison is active in session **1319**;
source/test/Makefile snapshot and hashes are under `source/`.
The RS trace's memcpy API medians are slightly lower for the adapter than for
the original, so a lower-level copy API is not the measured bottleneck.

The segmented AG path passes correctness but full AG audit fails **8/42**,
with the changed 64/128-KiB points roughly 24–33% slower. Session 1319 is
terminal (exit 2); this path is not promoted.

## Completion-contract diagnostic

`standalone_completion_probe` includes the frozen original peer source with
only its main symbol renamed; the source bytes match git commit `0fc3714b`.
It warms the original 64-KiB push-serial AG, then delays rank 3's source stream
and queues another unchanged launch. Ranks 0/1/2 each finish their own stream
with 4096 incorrect/unavailable output elements. Rank 3 is still pending at
observation. After the original all-GPU synchronization, every output validates.
Artifacts: `build/local_completion_probe/result.{log,json}` and build log.
This diagnoses semantics, not performance, and does not change any gate.

The original timing records each source's end event before all-GPU
synchronization. The adapter promises each caller stream's output-ready
completion. A clarification is pending about which completion contract should
define no-regression. Until an explicit answer arrives, retain the original
standalone gate and its failures; do not silently substitute another baseline.

`build/local_rs_banks31` retains the 12-MiB schedule at 128 MiB and tries
31-MiB chunks for inputs >=256 MiB. This reduces IPC handoffs while retaining
partial final chunks and the two-bank lifetime protocol. All correctness and
queue tests pass; the full RS comparison is active in session **90200**.
Source/test/Makefile snapshot and hashes are under `source/`.

A pending small-AR hypothesis is paired float stores: the retained original
LL kernels use float2 stores, while the adapter's direct and device-epoch
kernels issue two scalar stores to adjacent floats. Test aligned float2 stores
with a scalar fallback for 4-byte-only alignment on the epoch/stream candidate;
keep reduction order and all synchronization unchanged. This is unimplemented
and unmeasured, not a performance claim.

The 31-MiB candidate's first completed out-of-place run was about 10% slower
at large sizes. Its controller was deliberately terminated (process group
4085186, session 90200 exited 143) to reject it without completing more
repeats. `early_rejection.json` and partial raw logs are preserved. This is
not a completed paired gate and must not be reported as one.

`build/local_ar_pair_store` adds aligned float2 output stores to the
`local_epoch_stream` candidate's tree/hierarchy AR kernels, including device-
epoch graph entries. Four-byte-only alignment retains scalar stores; reduction
order and synchronization are unchanged. SASS confirms paired stores plus
the scalar fallback (`pair_store_sass.txt`). Guarded ABI passes. Controller
**55604** is running queue/full-stock correctness and the 1-GiB-allocation
small AR screen. Source/test/Makefile snapshot and hashes are under `source/`.

The paired-store candidate passes queue/full-stock correctness and its
independent small AR audit passes **14/14** points (1 KiB–64 KiB, both layouts,
1-GiB allocations). Ratios range 0.806–0.996; the 16-KiB in-place point has
little margin. Session 55604 is terminal (exit 0). This remains scoped evidence.
The full 1-KiB–1-GiB AR gate is now being launched on the exact same library,
with results under `build/local_ar_pair_store/ar_pair/` and controller output
in `ar_pair_full_driver.log`. Do not infer completion before that gate and
the remaining collective gates pass on one fixed final library.

The full paired-store AR gate is active in session **46640**. Two candidate
rounds have completed correctly; continue polling that controller before
starting GPU work or interpreting the full curve.

A next RS hypothesis, not implemented: move the exported readiness event off
the transfer stream by making the destination CE pull a prepared private
source slot. Stage zero first copies the caller's outgoing shard into private
IPC memory on the reduction stream; later source slots are the preceding
partial reductions. Exported readiness events follow those producers on the
reduction stream. The receiving transfer stream waits for the imported source
ready event, pulls into its own receive slot and records a native local copy
completion event. Its reduction stream joins that local event, reduces, and
publishes the next stage. This may hide the more costly IPC record behind CE
work; the initial private copy and pull bandwidth must be measured, not assumed.
Existing IPC publication/acknowledgement can protect event generations.
No caller allocation is imported. A four-shard private layout per bank fits
the existing twice-capacity allocation with two banks, but reuse/transition
retirement needs a fresh proof and delayed changing-input queue checks.

**The paired-store library passes the full AR gate: 42/42 points**, 1 KiB–1 GiB,
both layouts, original three-round variant envelope, full 1-GiB allocations.
Independent `audit_local_pair.py` reports zero failures. The closest ratio is
0.99430 (32-MiB out-of-place). All numerical, route, native-library absence,
and fallback-zero checks pass. Session 46640 is terminal (exit 0).
Artifacts: `build/local_ar_pair_store/ar_pair/`. This qualifies the measured
AR curve only, not the overall drop-in objective.

Full RS then AG comparisons are now being launched on this exact library via
`build/local_ar_pair_store/run_peer_pairs.sh`. Preserve this fixed binary and
its source snapshot. No default policy is promoted, and the existing original
standalone gate remains in effect while the optional completion-contract
clarification is unanswered.

Peer-comparison controller handle: session **43429** (RS then AG), confirmed
running after launch. Read `rs_pair_driver.log` / `ag_pair_driver.log` and the
same handle before restarting or interpreting completion. The full AR result
above is final for this frozen library; peer gates are still pending.

The paired-store library's full RS audit completed with **8/42 failures**,
both layouts at 128 MiB–1 GiB (ratios 1.00060–1.00433). Numerical/runtime
checks pass. Full AG continues in controller session 43429; full AR remains
42/42 on this exact binary. Do not mix the separate banked-RS experiments
into this result without measuring a combined library.

The paired-store library's full AG audit reports **4/42 failures**, both
layouts at 64/128 KiB (ratios 1.02868–1.14626). Controller 43429 is terminal
(exit 2). The complete fixed-library result is now **12/126 failures**:
AR 0, RS 8, AG 4. All numerical/runtime checks pass. This is improved evidence
on one library, but still does not satisfy the full no-regression goal.

## Private staged pull AllGather

`build/local_ag_pull` starts from the AR-qualified paired-store library. At
64/128-KiB AG a cooperative 16-CTA kernel stages each rank's input only in its
own private buffer, publishes readiness, then vector-reads peer private rows
into caller output. Producer system fences and two grid joins bracket readiness
exchange (flag words 28–31). Peer data uses volatile vector loads after the
acquire boundary. Two-bank reuse and operation/size rendezvous are retained;
cooperative residency is checked. No caller buffer is imported or retained.

Guarded ABI, 600-call queue, 192-call gather queue, and all 126 full-range
misaligned stock points pass. Resource inspection shows zero stack bytes.
Full AG measurement is running in session **39373**. Snapshot under `source/`.
The preceding fixed library remains the only complete three-collective result;
this new library has not inherited its AR/RS performance qualification.

The private staged-pull AG passes correctness but its full audit fails the
four changed 64/128-KiB points by a wide margin. Session 39373 is terminal
(exit 2). The next isolated experiment uses `ld.global.cg` payload loads,
retaining the producer system fence, acquire readiness and cooperative joins.
The PTX cache-operator documentation describes .cg as L2-and-below caching,
bypassing L1; cache operators do not change consistency semantics. This is
not permission to remove the readiness protocol. Validate changing inputs
and all layouts before measuring the new access policy.

`build/local_ag_pull_cg` passes guarded ABI, normal/gather queues, and all 126
full-range misaligned stock checks. Controller **24171** is running full AG.
Source/test/Makefile snapshot and hashes are under `source/`; `pull_sass.txt`
records the load instructions. The payload load is ordered by the existing
readiness acquire and grid join; cache hints alone are not synchronization.

The .cg peer-read AG variant passes correctness but its full audit fails
**6/42**, with the changed 64/128-KiB path still far slower. Session 24171
is terminal (exit 2). Neither peer-read AG variant is promoted.

## Copy-engine pull ReduceScatter

`build/local_rs_pull` starts from the complete paired-store library. Large RS
stages its initial outgoing shard into a fourth private slot. IPC readiness
records follow that staging and subsequent partial reductions on the producer
stream. The receiver transfer stream waits for the imported ready event,
pulls into its own slot and records a native local copy-completion event;
the reduction stream joins that event before summing. The existing IPC
publication/acknowledgement mechanism protects each event generation. Local
native event slots 0–2 mark copies; slot group 3 marks the final reduction.
The final all-rank retirement barrier remains; no receive-bank elision is
mixed into this first implementation. No caller buffer is imported.

Guarded ABI and the full 126-point misaligned stock sweep pass. Controller
**81850** runs all queues then full RS comparison. Source/test/Makefile
snapshot and hashes are under `source/`. Performance remains unqualified.

The CE-pull ring passes all correctness/queue checks but is decisively slower.
Stock 128-MiB/1-GiB RS measures approximately 98/780 ms (about 1 GiB/s bus rate),
versus approximately 5/40 ms for the retained push ring. The first paired
128-MiB out-of-place mean is approximately 97 ms. Controller 81850/process
group 4094215 was deliberately terminated to reject the candidate without
finishing the remaining repeats. `early_rejection.json` and partial logs are
preserved. This is not a completed full paired gate. Receiver-initiated CE
transfers are not viable on this testbed; retain sender-initiated copies.

The first `local_rs_memop32` edit failed before compilation; its controller
was stopped (session 67918/process group 4095576, exit 143). The binary still
used 64-bit operations and is explicitly invalid as a 32-bit experiment.

The corrected `build/local_rs_memop32_v2` uses 32-bit release writes and
32-bit acquire flag loads in the fused reduction. Global wrap reset bounds
generation values before they can exceed 32 bits. `nm` confirms the binary
imports cuStreamWriteValue32_v2. Probe, guarded ABI, all queues and the full
126-point misaligned stock sweep pass. Full RS measurement is active in
session **20646**; first large timings remain slower. Snapshot under `source/`.

The corrected 32-bit stream-value candidate passes correctness but its full
RS audit fails **7/42** points, including large ratios up to 1.00871. Session
20646 is terminal (exit 2); it is not a no-regression-qualified replacement.

`build/local_ag_epoch512` changes only the earlier AG epoch graph's producer
block size to 512 threads at its 64/128-KiB path. Data format, system fences,
readiness flags, copy node and reuse protocol are unchanged. Guarded ABI,
normal/gather queues, and the full misaligned stock sweep pass. Full AG
measurement is active in session **17114**. Snapshot under `source/`.

The 512-thread AG epoch producer passes correctness but full AG audit fails
**7/42** points, including both layouts at the changed 64/128-KiB sizes.
Session 17114 is terminal (exit 2); it is not selected.

`build/local_ag_hier_sm` starts from the complete paired-store library. At
64/128-KiB AG a cooperative four-CTA kernel stages the local pair's two rows,
then pushes both rows to the cross-pair counterpart. Two pairwise handshakes
use the established DeviceSyncer release/acquire publication pattern and
system-scope leader flags (words 96–103). Output copies are distributed across
the CTAs. Two parity banks plus local/cross dependency credit protect reuse;
size/operation changes retain the common rendezvous. No remote payload reads
or per-element flag packets are used. Cooperative residency is checked.

Guarded ABI, normal/gather queues, and all 126 full-range misaligned stock
checks pass. Resource inspection reports zero stack bytes. Full AG timing
is running in session **22551**. Snapshot under `source/`.

The four-CTA SM hierarchy passes both changed 128-KiB AG points, but 64-KiB
points still regress about 19–21%. Full AG audit has **6/42** failures,
including small (<=0.74%) differences at unchanged 32/256-KiB points. Session
22551 is terminal (exit 2). The next isolated geometry change retains four
256-thread CTAs at 128 KiB and uses one 1024-thread CTA at 64 KiB. DeviceSyncer
explicitly reduces to a CTA barrier when its block count is one.

The first single-CTA build failed compilation because its host launch variables
were missing; no tests or measurement ran. Session 63271 is terminal (exit 1).
The corrected fresh build is `build/local_ag_hier_one64_v2`; its controller
uses the same intended one-CTA/1024-thread 64-KiB geometry and retains four
256-thread CTAs at 128 KiB. Preserve the failed build log as an implementation
failure, not performance evidence.

The corrected single-CTA 64-KiB hierarchy passes guarded ABI, normal/gather
queues and all 126 full-range misaligned stock points. Controller **92383**
is running full AG measurement. The new block size's cooperative residency
is checked explicitly. Snapshot under `source/`.

The single-CTA 64-KiB hierarchy passes correctness but full AG audit still
fails **4/42**: both 64-KiB layouts, plus unchanged 32/256-KiB out-of-place
points. The 128-KiB hierarchy remains passing. Session 92383 is terminal
(exit 2). It is not a complete AG-qualified replacement.

## Host readiness with GPU-resident AG payload

`build/local_ag_host_ready` starts from the complete paired-store library and
uses the prepared AG epoch graph at 64/128 KiB. Payload remains GPU-resident.
Each producer publishes one system-release 32-bit flag into a separate pinned
host control line; the receiver reads all four flags together and acquires
before its output copy. A second small SharedHostPackets allocation supplies
the private, lifetime-managed mapped control region. No worker is started.
The existing grid/CTA publication, parity reuse and prepare boundary remain.

The GPU epoch counter cannot exceed the global generation between resets.
At global wrap, all ranks quiesce, rank 0 clears the host control line, and
GPU counters/packet flags reset before the closing bootstrap barrier. A
separate test-only build, `local_ag_host_ready_wrap`, sets the generation
limit to 31 to exercise this branch repeatedly; it is not measured for speed.

Guarded ABI, normal/gather queues, full misaligned stock correctness, and the
forced-wrap 192-call gather queue pass. Full AG timing is active in session
**58401**; first 64-KiB out-of-place mean is approximately 16.4 us. Source
snapshots and hashes are retained for both builds. Do not infer qualification
before the completed audit. Default generation limit remains 0xffffffff.

**Host-readiness AG passes the full gate: 42/42**, both layouts from 1 KiB to
1 GiB. Independent `local_peer_pair.py --audit-only` reports zero failures.
The closest ratio is 0.997659 at 64-KiB out-of-place; retain that small margin
as a limitation until the final configuration is confirmed. Session 58401 is
terminal (exit 0). All correctness/runtime and forced-wrap checks pass.

Full AR then RS comparisons are now being launched on this exact library via
`build/local_ag_host_ready/run_remaining_pairs.sh`. Earlier AR-qualified
results belong to another hash and are not substituted for this new run.
No default policy is promoted; the original timing gate remains unchanged.

Remaining-pair controller handle: session **70953** (AR then RS), confirmed
running after launch. Read `ar_pair_driver.log` / `rs_pair_driver.log` and
that handle before restarting or interpreting completion. The full AG pass
is final for this frozen library; its other performance gates remain pending.

The host-readiness library's full AR audit completed with **2/42 failures**:
2-KiB in-place (1.00750x) and 4-MiB in-place (1.00200x). All output/runtime
checks pass. These results belong to the exact AG-qualified library, so the
previous AR 42/42 result is not substituted. RS continues in session 70953.

The host-readiness library's completed full campaign has **10/126 failures**:
AG 0/42, AR 2/42, RS 8/42. The RS failures remain both layouts at 128 MiB–1 GiB
(ratios 1.00011–1.01064). Controller 70953 is terminal (exit 2).

`build/local_ar_load_ring4` adds aligned float2 input loads (scalar fallback)
to the tiny AR kernels and uses the existing IPC ring for 4-MiB AR. Event
preparation now starts at the configured AR ring minimum, including a 4-MiB
capacity profile. RS retains its 8-MiB threshold. A new ring4 queue variant
covers changing 4-MiB AR calls and transitions; a small-capacity stock sweep
checks the new preparation boundary. Full correctness then full AR measurement
are being launched. No result is inherited from another library hash.

The paired-input/4-MiB-ring candidate passes guarded ABI, normal queue,
48-call ring4 queue, all 126 full-range misaligned stock points, and the
4-MiB-capacity stock sweep. Full AR measurement is active in session **45978**.
Source/test/Makefile snapshot and hashes are under `source/`.

The paired-input/4-MiB-ring full AR audit reports **1/42 failure**, 1-KiB
out-of-place (1.0695x). Both previously failing points now pass, but the new
library is not AR-qualified. Session 45978 is terminal (exit 2).

## Explicit NUMA-local CPU placement

A separate local-only launcher, `integration/nccl_lite/local_numa_profile.py`,
reuses the existing hardware/software recorder and strict profile format. It
derives each GPU's NUMA node from its PCI device, intersects that node's CPUs
with the parent's allowed set, fingerprints the complete rank-mask map and
launcher, and checks the applied mask before exec. Unknown topology and empty
intersections are refused. The existing shared/multi-node launcher is unchanged.
The local drivers have an optional `--cpu-placement numa`; unbound remains the
default. Audits verify all four binding markers, profile identity, topology
snapshot and frozen launcher copies. No numerical/timing acceptance rule or
original reference command changed. Five placement admission tests pass, and
previous unbound AR/AG audits remain valid.

On the unchanged `local_ag_host_ready` library, NUMA-local ABI, queue, and full
misaligned stock correctness pass. The independent small AR screen passes
**14/14** with 1-GiB allocations (worst ratio 0.94473). Session 64721 is terminal
(exit 0). This is a new placement, not a new binary or a full qualification.
Full RS, AR and AG gates are now being launched under
`build/local_numa_placement/`, using one fixed binary and placement across all
sizes. The original baseline remains unmodified and uses its original launch.

NUMA-local full-campaign controller: session **93689** (RS, AR, AG).
A live `/proc`/sched_getaffinity observation during the RS candidate run
confirms all four measured client processes retain exactly the recorded
rank masks after MPI/CUDA initialization; see `observed_affinity.json`.
Do not restart this controller or alter the measured launcher while it runs.

NUMA-local full RS audit completed with **8/42 failures**, both layouts from
128 MiB through 1 GiB (ratios 1.00247–1.00590). All correctness/runtime/placement
checks pass. CPU locality did not remove the large RS handoff gap. AR and AG
continue in controller session 93689; the placement is still unqualified.

The completed NUMA-local campaign is unqualified: **AR 3/42, RS 8/42, AG 3/42
failures** (14/126 total). Session 93689 is terminal (exit 2). The unbound
configuration remains the better complete result. The optional launcher and
its admission checks remain separate; no default CPU binding was introduced.

## Thin host flags for the large RS handoff

`build/local_rs_host_ready` returns to the unbound AG-qualified library.
Sender-initiated 8-MiB copies remain. Each copy publishes a default-fenced
32-bit stream value in a private host control region local to the source NUMA
node. One leader in the cooperative reduction grid acquires that flag and
releases a device-local gate; other CTAs poll only the local gate before summing.
The grid is capped by measured cooperative residency. Native local events
still order streams/phases, and the final all-rank retirement barrier remains.
An actual mapped-host write/readback probe precedes use.

AG uses control words 0–3 in region 0. RS source rows start at word 64, with
128-word strides per local source; their three 32-word stage groups and a
reserved probe word fit below 4 KiB in each region. GPU gate word 449 is
separate from other controls. Global wrap quiesces all ranks and clears both
host control pages plus the GPU flag page. The separate wrap build uses a
five-call limit to exercise this branch during the large changing-input queue;
it is not benchmarked for speed.

Controller **68043** runs correctness, queues, forced-wrap large stress and
full RS measurement. Snapshots and hashes are under each build's `source/`.
No qualification is inherited from the earlier library.

The thin-host-flag RS candidate passes guarded ABI, normal/ring/large queues,
all 126 full-range misaligned stock points, and the five-call-limit forced-wrap
large queue. Full RS measurement has started in session 68043. Keep this
separate from the completed unbound/NUMA-local results until its audit finishes.

The thin-host-flag RS audit completed with **6/42 failures**, both layouts
at 128/256/512 MiB (ratios 1.00170–1.00466). One-GiB points pass in this run,
but the path remains unqualified. Session 68043 is terminal (exit 2).

The raw CE routing probe passes every full-output and untouched-quarter check.
Three-round median means are 39.366 ms for the existing gray-forward ring,
39.799 ms gray-reverse, 39.652 ms linear-forward, and 39.835 ms linear-reverse.
The alternatives are 0.73–1.19% slower. This is a raw-copy diagnostic with no
reduction/IPC handoff, not an admission benchmark. Artifacts are under
`build/local_topology_probe/`; session 89480 is terminal (exit 0).

The next geometry test retains the current route and tries 4-MiB RS chunks
at >=128 MiB. Three 64-slot event groups fit the existing 192 event objects;
AR retains six 32-slot groups. On a grouping change, the host waits for all
previous binding acknowledgements before re-recording any reinterpreted event
index. GPU completion still orders execution and buffer reuse. Correctness
must cover transitions between these groupings before measurement.

`build/local_rs_chunk4` passes guarded ABI, normal/ring/large queues, and all
126 full-range misaligned stock correctness points. Full RS measurement is
active in session **7525**. Snapshot under `source/`. The large queue exercises
AR/RS transitions between the six-by-32 and three-by-64 event groupings.

The four-MiB RS chunk candidate passes correctness but full RS audit fails
**8/42**, both layouts from 128 MiB through 1 GiB, ratios 1.0060–1.0133.
Session 7525 is terminal (exit 2). The smaller copy geometry is not selected.
The next experiment retains 8-MiB copy commands but records one IPC-ready event
and runs one reduction for each pair of adjacent copies. This reduces event
traffic without changing individual copy sizes; the longer final reduction
is included in the measured completion time.

`build/local_rs_group2` starts from the AG-qualified library. At >=128-MiB RS,
each group sends two ordinary 8-MiB copies, then publishes one IPC event and
reduces the combined region. Native previous-phase waits and final all-rank
retirement remain. Event indices keep the same logical stage grouping, and
existing acknowledgements protect old bindings when group counts change.
Guarded ABI passes. Controller **97657** runs queues/full-stock correctness
then full RS comparison. Source/test/Makefile snapshot is under `source/`.

The group-of-two candidate passes correctness but full RS audit fails **8/42**,
with large regressions up to 1.01716x. Session 97657 is terminal (exit 2).
A 128-MiB diagnostic trace (`build/local_rs_group2/trace128/summary.json`)
shows 8-MiB copy medians around 405–407 us, and grouped 16-MiB reductions at
87–88 us versus about 40 us for the original 8-MiB kernel. Within-group copy
gaps remain 5–7 us. Trace session 97843 is terminal; these are diagnostics.

The next candidate groups readiness for up to four 8-MiB copies, keeps the
reduction kernels at 8 MiB, and reserves the final copy as a one-copy group.
This preserves the original final reduction length. Event publication and
acknowledgements still cover each complete group; offsets and partial tails
must be validated before any performance conclusion.

`build/local_rs_group4_tail` implements that grouped-readiness/small-reduction
schedule. The last copy is a separate group; preceding groups may contain
fewer than four pieces. Source and destination use identical offset/length
helpers, and all pieces are covered by the group's recorded readiness event.
Guarded ABI passes; controller **34006** runs queues/full-stock correctness
then full RS measurement. Snapshot under `source/`.

The group-four/separate-tail candidate passes all correctness/queue tests,
but full RS audit fails **8/42**, ratios 1.0178–1.1131. Session 34006 is
terminal (exit 2). Keeping reduction kernels small did not recover the lost
performance. Grouping readiness is not selected.

All benchmark/profiling controllers are terminal. Best completed fixed-library
result remains `local_ag_host_ready`: AG 42/42, AR 40/42, RS 34/42. The original
gate is unchanged; no default policy is promoted and the goal remains active.
Unsuccessful variants, raw traces, CPU-placement records and the immutable
standalone completion diagnostic are preserved.

All controllers are terminal at this checkpoint. The best complete result is
unchanged (10/126 failures), and no default is promoted.

Next unimplemented synchronization hypothesis: preserve 8-MiB copies and
reductions but use separate sender and receiver CUDA graphs per ring phase.
The sender graph records the existing IPC-ready events after each copy; only
after the existing CPU publication handshake may the receiver graph bind those
events, reduce, and record native reduced events. The next sender phase then
binds the preceding local reduced events. Do not combine both sides into one
cross-process graph launch, which could bind a peer's previous event generation.
A source memcpy binding cache must validate allocation IDs and rebind after
pointer/allocation changes; kernel arguments alone do not retain allocation
bindings. Imported/private memory and events must outlive the graphs. Check
actual IPC event-node support and changing-input/lifetime queues before timing.
Graph launch behavior is unmeasured; do not assume it eliminates device gaps.

## Separate ring phase graphs

`build/local_rs_graph` preserves scalar 8-MiB transfers/reductions and the
original three ring phases. Each phase has a sender graph (local dependency,
copy, IPC-ready record) and a receiver graph (imported wait, sum, native
reduced record). Existing CPU publication/acknowledgement separates the graph
launches so waits bind the current event generation. The final retirement
barrier remains. Source memcpy bindings require a matching pointer and live
allocation ID; changed bindings are updated. Kernel pointer changes update
the reduction nodes. Unknown/non-legacy source allocation kinds retain the
eager UCCL ring. Graph objects retire before imported memory/events.

The smaller-cutoff correctness-only build `local_rs_graph_small` passes the
198-call/two-lifetime ABI test, with nine actual graph calls and zero eager
calls per rank per lifetime. Thus IPC event nodes and allocation/pointer
rebinding are exercised, not merely compiled. Controller **68153** continues
normal/queued/full-stock/large-queue validation and then the full RS gate at
128 MiB and above. Both source snapshots are under `source/`.

The separate phase graphs pass all correctness/queue checks, but independent
full RS audit reports **8/42 failures**, ratios 1.00151–1.00705. All six timed
candidate runs log 120 graph calls/rank and zero eager calls; the graph path
was actually measured. Session 68153 is terminal (exit 2).

The next graph variant makes IPC record nodes side branches of their copies.
Copy i+1 may overlap record i, while copy i+2 depends on record i. This bounds
publication lag to one chunk and preserves per-chunk receive waits, payload
size and reduction size. It changes no event-generation handshake or binding
lifetime rule. The complete graph still retires before the next phase.

The bounded-overlap graph builds are `local_rs_graph_overlap` and the
correctness-only lower-cutoff `local_rs_graph_overlap_small`. Controller
**18282** runs ABI/pointer/lifetime tests, all queues, full misaligned stock
correctness, and then full RS comparison. Snapshots are under `source/`.

The bounded-overlap graph passes correctness, but its first paired 1-GiB
out-of-place mean is 40.862 ms versus original approximately 39.5 ms, with
120 actual graph calls/rank. The controller was deliberately stopped
(session 18282/process group 4128582, exit 143); partial logs and
`early_rejection.json` are retained. This is not a completed paired gate.

The next experiment moves IPC publication explicitly to the reduction stream.
A native local event records each outgoing copy; the auxiliary stream waits
for it, publishes the IPC event, then waits for the peer event and reduces.
Per-tile CPU publication/acknowledgement binds the correct generation before
the receive wait. The native local event is re-recorded after reduction only
after its copy wait has been installed. Protocol transitions acknowledge all
old bindings before event reuse. Payload sizes and final retirement remain.

`local_rs_tile_publication` passes full misaligned stock correctness, ABI,
normal/ring/large changing-buffer queues. The lower-cutoff build additionally
passes the ABI test with forced five-generation resets. Controller process
group 4130238 is running the full RS paired gate. Source snapshots are saved
under each build directory. No performance verdict yet.

The next isolated candidate, `local_rs_split_tail`, retains all eight-MiB
copies except the last copy of the third RS phase, split into seven and one
MiB. Both pieces wait on the same phase-two reduction and write disjoint
outputs. Its final reduction reads one MiB, shortening the pipeline drain.
The existing 192 event objects support a 64-event phase stride for large RS;
all old stage bindings are acknowledged before stride transitions. Other
operations and smaller sizes keep their schedules. The first compilation
caught a missing conditional field declaration; its failed log is preserved.
The corrected build succeeds and is snapshotted. Its controller waits for
process group 4130238 before running correctness and a full paired RS gate.

The tile-publication full RS audit is **37/42 passing**, with five failures:
256-MiB out-of-place, both 512-MiB layouts, and both one-GiB layouts.
Failing ratios are 1.00096–1.00374. All six timed runs exercised 120 calls
per rank through the new path. Independent audit reproduces the result;
process group 4130238 is terminal. This is not a qualified RS replacement.
Split-tail controller session **91860** is now running serially after it.

A separate compiled candidate, `local_rs_peer_copy`, changes only the large-RS
sender copy API to `cudaMemcpyPeerAsync` with the validated source/destination
device ordinals. The original uses that API; the adapter had used D2D copies
through imported pointers. Prior CPU API timing did not favor the original,
so the hypothesis is specifically GPU command/handoff behavior, not lower
host call cost. Source snapshots are saved. Its controller waits for the
split-tail controller (process group 4132661) before running tests. IPC memory
compatibility must pass real correctness before this path is measured.

The split-tail full RS audit is **36/42 passing**. Both layouts at 128 MiB
pass; both layouts at 256/512 MiB and one GiB fail, ratios 1.00050–1.00345.
Independent audit reproduces all six failures. Controller 91860 is terminal
(exit 2). The reduced drain does not remove the transfer-count-dependent gap.
The explicit peer-copy controller is session **1283**, now in correctness
validation. No candidate is promoted.

Fresh 128-MiB/one-GiB-allocation timelines are queued after the explicit-copy
controller: tile publication, explicit peer copies, and the immutable original.
Controller **59618**, `build/local_rs_handoff_trace/run.py`, records every
command and exit code and exports GPU trace CSVs. Five warmups/ten measured
calls; original sweep retains the one-GiB maximum. These captures diagnose
copy gaps/reduction/retirement and do not replace the paired performance gate.

The explicit peer-copy full RS audit is **35/42 passing**, with seven failures:
128-MiB out-of-place and both layouts from 256 MiB through one GiB, ratios
1.00078–1.00508. Correctness passed; independent audit reproduces the failures.
Controller 1283 is terminal (exit 2). The copy API is not promoted.

The first new profiler captures warn that default CUDA event tracing can add
false cross-stream dependencies. They are retained under
`build/local_rs_handoff_trace`, but are not used to establish a handoff cause.
A fresh serialized capture set explicitly disables that feature with
`--cuda-event-trace=false`, under `build/local_rs_handoff_trace_noevents`.
Its controller waits for process group 4136369 before using the GPUs. Earlier
profiled copy-gap estimates also need this control before treating them as
intrinsic adapter costs; the unprofiled performance gates remain unchanged.

The controlled timelines finish successfully (`local_rs_handoff_trace_noevents/summary.json`).
At 128 MiB with one-GiB allocations, over 120 copies/reductions per rank,
original median gaps are 5.02–5.28 us; tile publication 4.66–5.31 us; explicit
peer-copy 6.50–7.44 us. Copy medians are respectively 405.60–406.03,
405.95–406.75, and 405.68–406.46 us; all reduction medians are approximately
40 us. Tile publication removes most of the gap but retains its 6.8–8.3-us
retirement barrier and slightly longer copy times in this capture. Native
peer-copy syntax does not remove the handoff gap. These are diagnostics, not
new qualification results. Both profiling controllers are terminal (exit 0).

`local_rs_gpu_ready` next changes the earlier host-readiness protocol's
control address to a producer-local GPU word. A default-fenced 32-bit stream
write publishes after each outgoing CE copy; one resident receiving CTA polls
the producer's imported flag and relays readiness through the existing local
GPU gate. Remaining CTAs read only that local gate. The source copy stream
has no IPC event or host-memory notification. Default system fencing, native
prior-reduction dependencies, scalar eight-MiB reductions and final all-rank
retirement remain. The active GPU words are 256–351, probe 352, relay 449;
all reset at the existing quiescent wrap. Preparation probes the actual local
GPU stream-write address class. The lower-cutoff correctness build exercises
the protocol from eight MiB with five-generation resets. Controller **90853**
builds both, checks them, then runs the full RS paired gate.

Both GPU-readiness builds succeed. The lower-cutoff guarded ABI passes
198 calls/rank and two lifetimes, logging nine actual GPU-readiness calls
per rank per lifetime with forced generation resets. Production ABI, normal
queue, full misaligned nccl-tests, ring queue and changing large-buffer queue
all pass. Controller 90853 is now running the full RS paired gate.
The current source also builds through the unchanged default target in
`build/local_default_compile_current3` (session 66519, exit 0); compilation
alone does not qualify that default configuration.

The first GPU-readiness paired samples are slower by roughly two percent,
consistent with a possible cost from polling a peer throughout each transfer;
this is a hypothesis, not a measured traffic attribution. A conditional next
controller, **44022** (`build/local_rs_gpu_native/prepare.py`), waits for 90853
and runs only if its full RS gate fails. It verifies the source hash before
editing. The planned variant records a native local event after each outgoing
copy, binds the auxiliary wait to that copy record, then launches one resident
kernel whose leader publishes its producer-local GPU flag before polling the
predecessor and relaying readiness. The native event is re-recorded only after
reduction, preserving subsequent-phase dependencies. All ranks publish before
waiting; no IPC event or CPU binding handshake is needed on this path. This
should avoid remote polling during the full copy interval while keeping the
copy-stream native-event schedule. It retains full retirement and wrap reset.
Correctness and forced-wrap tests precede its full paired gate. This controller
is prepared/queued, not yet implementation or measurement evidence.

The stream-write GPU-readiness full RS audit fails **8/42**, both layouts from
128 MiB through one GiB, ratios 1.01979–1.02378. All 24 timed rank/run records
confirm 120 GPU-readiness calls; independent audit reproduces the result.
Controller 90853 is terminal (exit 2). The native-event successor's source
hash check passed, the planned edit was applied, and its production build
succeeds. The resulting library has no `cuStreamWriteValue` import, confirming
that its publisher is the kernel after the native copy wait. Lower-cutoff
build and correctness/performance controller 44022 continue; no result yet.

The native-event/GPU-readiness lower-cutoff ABI passes, including two lifetimes,
nine path calls/rank/lifetime, and forced five-generation resets. Production
ABI, queues and full misaligned nccl-tests also pass. Its full paired RS
comparison is running. Default compilation succeeds at
`build/local_default_compile_current4` (session 93206, exit 0).

Conditional controller **16448**, `build/local_rs_gpu_native_tail/prepare.py`,
waits for native controller process group 4140516 and only continues on a
completed failing RS gate. It verifies the source hash before edits. It adds
the previously checked seven-plus-one-MiB final-phase split to the native
GPU-readiness path. Sixty-four native events per phase fit the existing 192;
producer flag indices through 416 remain below relay word 449. Relay tokens
use the full 192-entry stride, preserving monotonicity between calls. Both
final pieces wait on the same prior full reduction. No retirement elision is
included. A lower-cutoff/forced-wrap ABI and all correctness queues precede
its own full RS comparison. This successor is still queued, not measured.

The native-event/GPU-readiness full gate fails **6/42**, both layouts from
256 MiB through one GiB, ratios 1.00114–1.00214. Independent audit and all
24 rank/run path counts are verified. Controller 44022 is terminal (wrapper
exit 1 for the paired gate's expected exit 2). The native split-tail source
was applied after its hash check; both builds succeed and the forced-wrap ABI
passes. Controller 16448 continues production correctness then its paired gate.

A conditional one-GiB diagnostic is queued in
`build/local_rs_native_trace1g/run.py`, after process group 4142966 and only
if that complete gate fails. It compares native readiness, native split-tail
and original at the actual largest failing extent, explicitly disabling CUDA
event tracing. It retains all 15 calls and summarizes the last ten (960 or
970 copies/rank), including the final reduction separately. No samples are
removed from a performance gate.

The native split-tail production ABI, normal/ring/large changing-buffer queues
and full 126-point misaligned stock correctness all pass. Its full paired RS
gate is now active. Stock batch timings improve at the four changed large
sizes, but are not substituted for the repeated matched gate.

The native split-tail full RS gate is **39/42 passing**. Failures are
512-MiB out-of-place (1.000607), one-GiB out-of-place (1.000670), and one-GiB
in-place (1.000166). Independent audit and all 24 rank/run counts are verified.
Controller 16448 is terminal (wrapper exit 1 for gate exit 2). The one-GiB
profiling controller 68868 finishes successfully (exit 0).

Profiler-analysis correction: the immutable original performs two untimed
correctness calls before its 5+10 measurement calls and one after. The initial
128-MiB extraction inadvertently retained two warmups instead of the last two
timed calls. Both new 128-MiB analyses now skip those checks and select exactly
the ten timed calls; original summaries are retained as
`summary_initial_call_window.json`. The one-GiB analysis explicitly verifies
18 original versus 15 candidate calls and uses only the ten timed calls.
Performance-gate sampling was already correct and is unchanged. The corrected
controlled copy gaps remain about 5 us for the original/tile publication and
6.5–7.4 us for explicit peer copies.

The one-GiB trace shows the final reduction shrink from about 49–50 us in the
original to 14–48 us in the split-tail candidate, including variable readiness
waits. Profiling perturbs the candidate substantially: its traced copy span
is roughly 39.83–39.86 ms, unlike the unprofiled approximately 39.48-ms total.
Do not use that trace to attribute the unprofiled 6–26-us residual precisely.

`local_rs_native_graph` next represents the validated native-readiness/tail
pipeline as one cached CUDA graph. Copy-chain edges, prior-phase reduction
edges and the auxiliary kernel chain replace local event records/waits.
Cooperative reduction nodes still publish before polling; their residency is
checked. One initial node publishes the current host generation to private
GPU word 450, and final retirement reads it before recording the communicator
completion event. Payload geometry, scalar arithmetic, all-rank retirement,
and wrap reset remain. Input memcpy bindings validate allocation IDs and
legacy eligibility; unsupported allocation kinds retain eager UCCL execution.
Changed input/output bindings update future launches. Graphs retire before
imported memory. Controller **87848** builds production and lowered-cutoff/
five-generation-wrap versions, then checks them before a full RS comparison.

The native graph lower-cutoff ABI passes with nine graph calls/rank/lifetime
and zero eager calls; full stock and all queue correctness also pass. Its
full RS audit fails **4/42**, both layouts at 512 MiB/one GiB, ratios
1.000007–1.001546. Independent audit and all 24 rank/run counts confirm
120 graph calls and zero eager calls. Controller 87848 is terminal (exit 2).
One failing difference is only 0.14 us, but the original exact gate is retained.

`local_rs_native_graph10` changes only native-readiness transfer geometry to
ten MiB, retaining the final one-MiB reduction. At one GiB it sends 79 copies
instead of 97; at 512 MiB, 40 instead of 49. The observed roughly five-us copy
gaps bound the possible savings at about 90/45 us respectively, before any
bandwidth penalty. Larger 16/31-MiB earlier experiments lost, so no gain is
assumed for ten MiB. Tail splitting now checks whether the existing remainder
exceeds one MiB, preventing underflow for a shorter remainder. Other operations
and cutovers stay unchanged. Controller **74005** builds production and
lower-cutoff/five-generation-reset versions; the latter also runs the large
changing-buffer queue so non-power-of-two chunks are exercised across resets.
All correctness checks precede its own full RS comparison.

Ten-MiB native graphs pass all correctness, including the forced-reset large
queue, but full RS audit fails **2/42**, both 512-MiB layouts, ratios
1.000154–1.000827. Nine-MiB native graphs likewise pass correctness but fail
**3/42**: both 256-MiB layouts and one-GiB out-of-place, ratios
1.000347–1.001254. Independent audits and all rank/run path counts confirm
120 graph calls and zero eager calls. Controllers 74005 and 63737 are terminal
(exit 2). AR/AG were not run on these RS-failing libraries. Current default
compilation succeeded at `local_default_compile_current5` (41178, exit 0).

`local_rs_native_banks9` now adds two receive banks to the nine-MiB graph.
Six shards fit the existing twice-capacity allocation. The same three-hop
credit proof used by the earlier banked IPC ring applies: a predecessor's
completed n+1 result requires this rank's phase-zero n+1 contribution, which
follows this rank's completion of n; it therefore cannot overwrite bank n
in n+2 early. Both graph and eager allocation-compatibility paths use the
same parity bank and omit the final global retirement. Graph caches are keyed
by size and bank. A distinct transition mode (13) rendezvouses on operation
or count changes; wrap still joins all GPUs before reset. Local completion
still covers every source read and the complete local output.

Controller **6035** builds production and lowered-cutoff/five-generation-reset
versions. Lowered ABI and large changing-buffer queue pass; production ABI,
normal/ring/large queues and all misaligned stock points pass. Its full RS
comparison is now running. No performance qualification is inferred yet.

The two-bank/nine-MiB native graph passes every correctness/forced-reset test
but fails **2/42** RS points, both 512-MiB layouts, ratios 1.000087–1.001257.
Independent audit and path counts (120 graph calls, zero eager) are verified.
Controller 6035 is terminal (exit 2). Banking does not qualify this curve;
its stock batch latency also worsened at large sizes and is retained.

`local_rs_native_mixed` returns to the single-bank ten-MiB graph and changes
only the final/output phase to eight-MiB chunks, retaining the final one-MiB
reduction. Its range dependency waits on the last overlapping prior-phase
reduction; ordered prior kernels cover any earlier crossed piece. Graph and
eager paths share `NativeRingLayout`, so their offsets, flags and dependencies
agree even with different input allocation classes. Geometry bounds and partial
tails are checked. The hypothesis is to keep fewer transfers in the two
in-place partial phases while reducing the output phase's active data footprint;
cache attribution is not established by measurement yet.

Controller **3305** builds production and lowered-cutoff/five-generation-reset
versions. ABI, forced-reset large queue, production queues and full misaligned
stock correctness pass, including copies crossing ten-MiB reduction boundaries.
The full RS comparison is running. A separate conditional remaining-pairs
controller waits for process group 4154239 and runs AR then AG on this exact
binary only if RS passes; see `build/local_remaining_pairs.py`.

The ten/eight-MiB mixed-phase graph passes all correctness but its full RS
gate is **41/42 passing**. Only one-GiB out-of-place fails: 39513.661 us
versus 39503.149 us, ratio 1.000266. Independent audit and all rank/run path
counts are verified (120 graph calls, zero eager). Controllers 3305 and 99759
are terminal (exit 2); remaining operations were not launched.

`local_rs_native_mixed11` changes the first two phase chunks from ten to
eleven MiB, retaining eight MiB in the output phase and the one-MiB final
reduction. At one GiB, it removes four more transfers (81 versus 85), giving
a roughly 20-us upper estimate from observed copy gaps before bandwidth
penalties. No protocol, kernel arithmetic, or timing rule changes. Both builds,
forced-reset ABI/large queue, production queues and full misaligned stock
correctness pass. Controller **93349** now runs full RS and, only on success,
full AR then AG on that same library. No policy is promoted.

**The 11/8-MiB mixed native graph passes the full RS gate: 42/42**, both layouts
from one KiB through one GiB. Independent audit reproduces zero failures;
all 24 rank/run logs show 120 actual graph calls and zero eager calls.
Controller 93349 has advanced to AR, then AG, on this exact frozen library.
Earlier AR/AG wins belong to other hashes and are not substituted. The full
goal remains active and no default policy is promoted yet.

The same-library campaign on `local_rs_native_mixed11` is now complete:
**RS 42/42, AR 36/42, AG 38/42**, ten total regressions. Independent AR and
AG audits reproduce the failures. AR fails 1-KiB in, 4-KiB in, both 8-KiB
layouts, 16-KiB in, and 4-MiB out. AG fails 32-KiB in, 64-KiB in, 128-KiB out,
and 256-KiB in. Every output/runtime check passes. Controller 93349 is terminal
(exit 2). The RS minimum margin is 4.8 us (0.0123%) at one GiB out-of-place;
this is a narrow measured pass, not a robust universal margin.

The packet-shape AR screen passes correctness/forced resets and **13/14**
full-workspace small points; only eight-KiB in-place on the tree fails
(1.1477x). `local_ar_packet_ring4` restricts the tree to four KiB, retains
one packet unit per thread and aligned pair loads, and enables the existing
four-MiB AR ring. It passes forced-wrap ABI, full stock, ring4 queue and the
four-MiB-capacity check; full AR audit is **41/42**, with only four-KiB
out-of-place failing (1.00770x). Controllers 34949 and 17212 are terminal.

`local_ag_epoch_range` extends the existing host-ready epoch graph to
32–256 KiB. New `gather_range_queue_check` covers changing 16/32/64/128/256-KiB
calls (240/rank, two lifetimes), stream reuse, and delayed ranks. The first
client build lacked `mpicxx` on PATH and stopped; its failed log is preserved.
Using the explicit MPI compiler path fixes the build. ABI, all stock/queue
checks and forced five-generation resets pass. Full AG audit is **41/42**,
with only 128-KiB out-of-place failing by 0.0864 us (1.00370x). Controllers
47825 (failed client build) and 80130 (complete gate) are terminal.

`local_combined_cached` restricts the tree to two KiB and caches each thread's
eight input vectors in registers for 128-KiB AG before its four destination
writes. Store order, readiness and reuse credits are unchanged. Resource
inspection reports 48 registers and zero stack/local memory for the cached
producer. All correctness/forced-reset tests pass. AG improves the changed
128-KiB points to 0.912/0.945x, but the unchanged 64-KiB in-place point fails
(1.02143x); all six timed runs confirm 30 cached calls/rank. **Full AR now
passes 42/42**, independently audited. RS confirmation continues in controller
**64044** on this exact library.

Controller **23423** waits for process group 4166927, verifies the source hash,
then extends that same eight-vector cached producer to 64 KiB (128 threads)
as `local_combined_cached64`. It preserves 256 threads at 128 KiB and runs
all correctness plus AG/AR/RS gates on the new fixed binary. It is still queued.
A branch-reduction draft generator exists under `build/local_consolidate`;
its outputs are review-only, uncompiled and unqualified. No source cleanup or
default policy promotion has occurred.

The `local_combined_cached` full campaign finishes at **AR 42/42, AG 41/42,
RS 38/42**. Large RS fails both layouts at 512 MiB/one GiB, ratios
1.00150–1.00451. Disassembly hashes of `ringGraphWaitSum`, `ringGraphEpoch`
and `ringGraphRetire` and all RS geometry flags match `local_rs_native_mixed11`;
this does not validate the new binary's timing. Controller 64044 is terminal.

The `local_combined_cached64` full campaign reaches **125/126 passing**:
**AR 42/42, RS 42/42, AG 41/42**. Only 64-KiB out-of-place AG fails,
16.7296 versus 16.1504 us (1.035863x). Independent audits reproduce every
result. All timed AG logs confirm 60 cached-producer calls/rank; all RS logs
confirm 120 graph calls and zero eager calls. The closest RS pass is one-GiB
in-place at 0.999638x (14.3 us). Controller 23423 is terminal (exit 2).
A GPU process/clock snapshot during the subsequent reference phase shows
only the intended benchmark process: all four L4s at P0, 2040-MHz SM and
6251-MHz memory, 35–36 C. It does not characterize earlier timing samples.

`local_combined_prefetch` changes only the 64-KiB output-copy kernel: each
block's first thread prefetches the first/last address of its actual output
slice before reading the epoch and staged data. All addressed floats are
within that block's output range, including float-aligned pointers. This is
a test of overlapping output translation with existing work; a TLB bottleneck
is a hypothesis, not a measured attribution. No payload, mathematical or
synchronization change is made. Controller **23601** builds both production
and five-generation-reset versions and runs full correctness and all three
performance gates on the new fixed library.

The unrestricted output-prefetch candidate passes correctness but fails two
AG points: unchanged 32-KiB in-place (1.001055x) and 64-KiB in-place
(1.058963x). Its AR gate passes 42/42. The remaining RS run was deliberately
stopped (process group 4174973, controller 23601 exit 143); partial logs and
`early_rejection.json` are retained, not a complete RS verdict.

`local_combined_prefetch_out` applies the 64-KiB output prefetch only when
`input != output + rank * count`, the canonical out-of-place relationship.
The copy-node boolean is rebound whenever caller pointers change, including
switching layouts. All original payload/reuse ordering remains. Its five-reset
ABI, 240-call gather queues, ring4/small-capacity checks, and full misaligned
stock correctness pass. Controller **82113** runs AG then AR/RS. Watcher
**5212** stops remaining comparisons after a completed failing AG or AR gate;
partial logs are retained with an explicit early-rejection record.

Installed nvcc documents `--frandom-seed` for reproducible PTX/object files.
Current binaries contain PID-derived `tmpxft` FILE symbols, so reproducible
build integration needs attention before final promotion. No deterministic
build flag, source consolidation, or default-policy change has yet been made.

**Out-of-place-only prefetch passes full AG: 42/42**, independently audited.
Controller 82113 has advanced to AR, with RS afterward. The fixed library is
`build/local_combined_prefetch_out/nccl_local/libuccl_lite_nccl_local_strict.so`.
No earlier AR/RS result is substituted for its pending measurements.

**The out-of-place-only prefetch library also passes full AR: 42/42**,
independently audited. RS is in its final repetitions under controller 82113.
The watcher leaves RS running because both preceding gates passed.

The out-of-place-only prefetch campaign finishes at **125/126 passing**:
AG and AR 42/42, RS 41/42. Only one-GiB out-of-place RS fails by 18.26 us
(1.000463x). Controller 82113 and watcher 5212 are terminal; every completed
gate is independently audited. This is not a qualified overall replacement.

`local_combined_device_epoch` removes the native-RS graph's per-call host
parameter update. Its initial node increments private GPU word 450. Stable
bindings launch the existing executable directly; pointer/allocation changes
still refresh the required copy/kernel nodes. Native RS retirement now uses
private words 464–467 (phase 116), avoiding stale larger global-generation
values from other algorithms. The existing all-rank retirement stays in place.
Every eager allocation-compatibility call advances the same GPU counter and
uses the same readiness and retirement kernels, so mixed execution modes stay
in lockstep. Global wrap quiesces all ranks and clears both counter domains.

New `mixed_allocation_queue_check` uses ordinary CUDA allocations on ranks
0–2 and stream-ordered allocations on rank 3. It verifies the actual legacy
IPC capability (1/1/1/0), changing large buffers/streams and mixed operations.
The lowered-cutoff/five-reset build passes 24 calls/rank over two lifetimes:
ranks 0–2 each log seven graph calls and zero eager calls per lifetime; rank 3
logs zero graph and seven eager calls, with full outputs/guards correct.
Production ABI/queues, 240-call gather queue, ring4/small-capacity and all
misaligned stock points also pass. Controller **89893** is running all three
performance gates on the new fixed library. No result is inferred yet.

The GPU-owned RS epoch full campaign completes at **AG 40/42, AR 39/42,
RS 41/42**. RS misses only one-GiB out-of-place by 0.399 us (1.000010x).
AG fails 64/128-KiB out-of-place; AR fails 2/4/8-KiB in-place. The cached AG
publisher and output-copy SASS hashes match the preceding AG-qualified binary
(`ag_sass_comparison.json`), but these timing results remain failures.
Controller 89893 is terminal (exit 2).

An optional `core` placement uses `local_core_profile.py`, reusing the NUMA
launcher's binding/exec boundary. It assigns one distinct physical core per
rank on the GPU's NUMA node, retaining that core's allowed SMT siblings and
reserving the first two local cores when possible. Unknown or insufficient
core topology is refused. Exact masks, CPU topology, algorithm version and
all three launcher snapshots are fingerprinted and audited. Existing unbound
and NUMA launchers are unchanged; `local_cpu_profile.py` dispatches modes.
Only the accepted placement choices/import changed in the drivers; no timing,
source, reference command, sample rule or acceptance threshold changed.

All twelve old/new placement unit checks pass. Historical unbound and NUMA
audits still reproduce their prior verdicts. The current mask map is
rank 0 [4,28], rank 1 [6,30], rank 2 [5,29], rank 3 [7,31]. A live process
observation after MPI/CUDA initialization matches all four masks; evidence is
`build/local_core_device_epoch/observed_affinity.json`. Controller **83220**
runs correctness and full AG/AR/RS on the unchanged
`local_combined_device_epoch` library under this new placement. It is not yet
qualified and is not the default.

**Core placement passes AG and AR, 42/42 each**, independently audited.
RS continues under controller 83220 on the same library and masks. The named
`four-l4-fp32` candidate is now available in `integration/nccl_lite/local_policy.mk`;
`base` remains the default. Local compilation/link inputs are tracked by
`write_build_stamp.py`, with unchanged timestamps for identical inputs and
rebuilds when flags change. C++ header dependency tracking is included.
The named candidate uses nvcc's documented deterministic random-seed option.
Controller **5705** waits for the GPU campaign to finish, then builds two fresh
named-policy libraries and checks byte identity. Neither seeded build has
performance evidence yet; default promotion remains pending.

The one-core campaign completes at **AG 42/42, AR 42/42, RS 39/42**. Failures
are 512-MiB out-of-place and both one-GiB RS layouts, ratios
1.000481–1.002528. Controller 83220 is terminal (exit 2). The independent AR
auditor initially lacked the new placement dispatcher; updating only that
import allows the core audit, and historical unbound audits still pass.

The first two seeded builds differed only in ELF `.strtab` temporary FILE
names; their executable/data/CUDA sections matched. `--keep` with a stable
intermediate basename removes that remaining nondeterminism. Two fresh named
builds, `local_policy_keep_a` and `local_policy_keep_b`, are byte-identical:
`daab5506e5f9ce8530e7afc4aca7d0d2a397feb134876f05e85ebb7342c2055d`.
Evidence is under `build/local_policy_repro`. Controllers 5705 (initial
metadata mismatch) and 4506 (successful stable-intermediate rebuilds) are
terminal. Intermediate files stay under each BUILD directory.

Optional `core2` placement assigns two disjoint physical cores/rank, including
allowed SMT siblings. It reuses the same launcher and fingerprint validation;
core width and the helper snapshot are included. Historical one-core records
remain auditable. All fourteen placement unit checks pass. Current masks are
[4,6,28,30], [8,10,32,34], [5,7,29,31], [9,11,33,35]. The reason for this trial
is to provide CPU capacity for CUDA helper threads while preserving locality;
helper contention is a hypothesis, not a measured attribution.

Controller **61006**, process group **4190754**, validates the reproducible
`local_policy_keep_a` library with core2 placement. Production ABI/normal queue,
full misaligned stock, forced-reset ABI, mixed allocation queue, 240-call gather
queues, ring4/large queues and four-MiB capacity all pass. Full AG/AR/RS timing
follows on the same binary and placement. Default promotion and final evidence
packaging remain pending.

The reproducible core2 build passes full AG **42/42**, independently audited.
Live candidate processes after MPI/CUDA initialization retain the exact four
recorded core2 masks; `observed_affinity.json` stores that observation. AR and
RS remain pending under controller 61006. Candidate usage and explicit limits
are now documented in `docs/nccl-local-dropin.md`; its status remains unqualified.

**The reproducible core2 build passes full AR 42/42**, independently audited.
AG also passes 42/42. RS is the last pending gate. A link/export audit records
SONAME `libnccl.so.2`, the four requested collective/init exports, and no native
NCCL dependency in `build/local_core2_release/link_audit.json`. Ten correctness
suites have also been cross-checked against current binaries, client hashes,
placement snapshots and raw native/fallback markers (`validation_summary.json`).

The reproducible core2 campaign completes at **125/126 passing**: AG and AR
42/42, RS 41/42. Only one-GiB out-of-place RS fails by 26.94 us (1.000681x).
Independent audits confirm the results; controller 61006 is terminal (exit 2).
The validated library SHA remains `daab5506e5f9ce8530e7afc4aca7d0d2a397feb134876f05e85ebb7342c2055d`;
this is not full no-regression qualification and is not promoted.

`local_policy_large12` changes only the first two phase chunks at one GiB to
12 MiB, removing four transfers relative to 11 MiB. The output phase stays
eight MiB with the final one-MiB reduction; smaller sizes remain unchanged.
The lowered-cutoff/forced-reset build applies the larger chunks from 128 MiB
so the existing changing large-buffer and mixed-allocation queues exercise
partial boundaries across resets. Controller **48743** builds through the named
Makefile policy, runs the complete correctness suite under core2 placement,
and measures RS first, then AG/AR only after a completed passing RS gate.
The new output directory is `build/local_core2_large12`.

The large12 production and lowered-cutoff/five-reset builds succeed. All ten
correctness suites pass under core2 placement, including the actual mixed
legacy/stream-ordered allocation queue. Controller 48743 (process group 1030,
after the host PID counter wrapped) is running full RS comparison. AG/AR remain
conditional on a passing RS gate; previous operation wins are not substituted.

**The targeted large12 library passes full RS: 42/42**, independently audited.
All six timed runs show 120 graph calls/rank and zero eager calls. Controller
48743 continues with AG then AR on the same core2 placement and fixed binary.
Default promotion and final evidence packaging remain pending.

**The large12 library also passes full AG: 42/42**, independently audited.
AR is the final pending gate under controller 48743, with the same binary and
core2 placement. Neither the Makefile default nor performance status is promoted
before that result and the final requirement audit.

**Full large12/core2 qualification is complete: AG 42/42, AR 42/42, RS 42/42.**
All three gates independently audit as passing and use the same library hash
and core2 masks (`fixed_configuration.json`). Controller 48743 is terminal
(exit 0). The named policy now includes the measured one-GiB 12-MiB chunk
selection and is the Makefile default. A fresh no-policy-override build at
`build/local_default_final` is byte-identical to the measured library; controller
13632 exits 0. Its full misaligned stock confirmation also passes (75301, exit 0).
Raw timing/correctness records, identities, build logs, source dependencies and
the comparison table are archived under `benchmarks/data/nccl_local_strict_4l4_20261119`.
Earlier failures remain above; they are not erased or reclassified as wins.
