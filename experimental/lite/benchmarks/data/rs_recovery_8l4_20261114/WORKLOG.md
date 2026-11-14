# RS recovery experiment (in progress)

Target: fix 1, 2, 4, 8 MiB BF16 ReduceScatter regression on this 8-L4
host, in normal and explicit SHM modes. Baseline source: `5b9a7f86`.
The prior full refresh is immutable. This directory contains screening evidence,
including unsuccessful candidates. No candidate is a production selection yet.

Host: wxz-l4, 8 NVIDIA L4, rank 0–3 NUMA 0 and 4–7 NUMA 1;
CUDA 12.9, NCCL 2.28.9, driver 580.178.04. Full hardware/configuration
record is in `../full_refresh_8l4_20261114`.

`raw/*.json` records each command, environment, executable hash, status and GPU
ownership. `raw/*.log` retains every sample and correctness result. Screenings
use eager enqueue-to-all-streams-complete latency with changing inputs and
full output checks outside the timed region. They are not final performance
claims: NCCL latency varies noticeably between windows. Reversed matched
full-range confirmations are required.

Findings so far:

- CPU worker/packet/interleave/direct sweeps: stripes with 512 KiB packets
  and 8 workers help 4–8 MiB but hurt 1 MiB. No single validated fix.
- GPU direct reduction screens: reading all mapped-host source buffers is slow.
- CPU BF16 loop unrolling passes expanded scalar-oracle tests but does not
  improve end-to-end latency reliably. Reverted; trial patch retained.
- Nsight traces suggest upload/remote partial waiting is more important than
  CPU arithmetic here. Tracing perturbs launch costs; not benchmark evidence.
- Host-mediated ring prototype: one cooperative kernel per GPU, two pinned
  banks, adjacent-rank partial exchange. BF16 rounding per hop. Single progress
  counter and packed adds improve some sizes, but 1 MiB is not a robust win.
  Intermediate implementations are retained under `reproduction/`.
- Recursive-halving prototype under test: XOR partners 1, 2, then 4; only last
  exchange crosses NUMA. GPU scratch retains local partials; no GPU P2P needed.

Build and large trace artifacts are under `build/rs_recovery_8l4_20261114`.
Driver: `scripts/run_rs_recovery.py`. GPU jobs run serially, with no compilation
concurrent with measurement. Do not overwrite existing raw run names.

Caution: `ring_parallel_b8` and `ring_parallel_b16` accidentally reran the
preceding serial-launch binary after a failed build (missing `<algorithm>`).
Their executable hashes identify the old implementation. They must NOT be
interpreted as parallel-launch measurements. Corrected runs use different names.

Further progress:

- Halving and hierarchical GPU-local reduction did not beat ring. Retained raw
  comparisons and source snapshots. Hierarchical upload elision and vectorized
  final adds still did not produce a win.
- Removing per-thread system fences before CTA/grid barriers plus the leader's
  system-release publication improved ring latency. This follows the CUDA/PTX
  transitive synchronization model; CUDA 12.9 cooperative grid sync uses GPU
  release/acquire atomics (local `cooperative_groups/details/sync.h`).
- `release_b16`: CTA ring 254.98/445.44/807.94/1548.29 us at 1/2/4/8 MiB,
  matched NCCL 301.06/482.30/853.95/1602.56 us. Screening only, not final claim.
- Added explicit `PacketPipelineOptions::hostRingReduceScatter` (default false),
  with fixed resident CTA geometry, per-CTA credits, GPU reduction/progress,
  full-rank out-of-place aligned BF16 shards bounded by packet capacity.
  Ring uses the regular dynamic command queue and graph completion path.
  Dispatcher identity distinguishes ring; unsupported/unmeasured Auto falls
  back before allocation. The benchmark exposes `router-ring`; `current` is
  still unchanged pending full confirmation.
- `core_ring_check_4` passed 1,048,576 independent scalar BF16 checks plus 270
  changing-extent async calls and 3 x 140 captured calls, guards and rejection
  recovery. Devices 0,1,4,5 are a subset of this eight-L4 host, not a recheck of
  the separate historical four-L4 machine.
- `core_ring_check_8` was automatically terminated when another user's GPU jobs
  appeared. Exclude this interrupted run. Wait for free GPUs before retrying.
- The per-CTA prototype is only screened at a fixed aligned extent; production
  rejects unaligned shards/buffers, so lane ownership stays invariant across
  changing extents. Do not infer generic tail/alias safety of old prototypes.
- After the initial core check, changed `Transport` to union the ring pointer
  table with the unused generic peer-flag table, preserving generic payload
  layout. Added dispatcher fallback/identity tests and early fused-GEMM rejection.
  These revisions need fresh correctness and performance checks.
- Benchmark `current` now selects the explicit ring at 1–8 MiB only;
  `previous` retains the `5b9a7f86` selection for paired confirmation. This
  benchmark selection is provisional until final full sweeps pass; no default
  library policy is enabled. Other collective/size choices are unchanged.
- Final revision passed `final_ring_check_4`, `final_ring_check_8`, and
  `final_ring_check_shm`, with 1,048,576 / 2,097,152 scalar checks respectively,
  plus each run's async/graph/guard/rejection checks. Dispatcher CUDA and generic
  eager/graph checks passed in both GPU and SHM modes.
- Host checks passed: CPU BF16 860,132; policy 35; dispatcher 74; NCCL identity 8;
  router core passed. `review_router_policy_check` used the wrong working
  directory and failed to find its reference CSV; `review2_router_policy_check`
  reran from the Lite root and passed. This was a runner mistake, not a policy
  regression.
- `core_gpu` paired revised core: current 277.504/456.192/822.272/1562.112 us,
  NCCL 296.448/476.160/849.408/1593.344 us at 1/2/4/8 MiB. This short sweep
  passed all points; still require full-range confirmations.
- `core_shm` was interrupted by another GPU evaluation phase. Exclude it.
  The pending driver now waits for the known evaluation parent to exit, plus
  30 seconds without any GPU owners, before `core_shm_retry` and four full RS
  sweeps. No measurement or figure claim uses interrupted runs.
- `source_hashes.json` freezes the final verification source state. Subsequent
  source edits require rebuilding and rerunning affected verification.
- Final API review moved the new option to the end of `PacketPipelineOptions`,
  preserving existing positional aggregate initialization of earlier fields.
  The idle benchmark waiter was stopped before rebuilding (no GPU measurements
  were in flight). `source_hashes_pre_append.json` records the preceding state.
  Rebuilt checks and final performance sweeps must use the appended-field build.

## Continuation checkpoint

The append-only options revision built successfully (`build_ring_options_append.log`).
No GPU measurement is currently running. The prior idle waiter was terminated
before this rebuild. Its exec session has completed. GPU cards are occupied by
a third phase of the same external evaluation (parent PID 3100546); do not kill
or modify that job. Short idle gaps between phases are insufficient.

Next steps:

1. Wait for the external evaluation to finish and all GPU contexts to drain.
2. Run rebuilt host checks and GPU checks under NEW names (e.g. `post_append_*`).
   Existing successful `final_*` logs and `core_gpu` predate the option-field
   relocation; preserve them and interrupted `core_shm` / `core_ring_check_8`.
3. Recheck current vs previous vs NCCL at 1–8 MiB in both modes, then run four
   full sweeps using `scripts/run_rs_recovery.py full` (`confirm_*` names are
   not yet used). The final current selection is ring at 1–8 MiB, previous
   elsewhere. No library default auto-routing was enabled.
4. Run `scripts/summarize_rs_recovery.py`, inspect all 16 target/sweep results
   and unchanged-size regressions. Confirm AR/AG with `controls`; check kernel
   with compute-sanitizer in an exclusive window if feasible. Do not claim
   success solely from the short screening win.
5. Update RS overview source and figures, README/tables/provenance, checksums,
   commit and push only this task's changes after evidence is satisfactory.
   AR/AG historical curves remain explicitly tied to their prior source unless
   freshly replaced. Preserve the user's unrelated Blackwell docs/data edits.
6. Mark the active goal complete only after all required verification and push.

The last user-facing update reported that the final API-compatible build was
complete and full-range points would not be updated from incomplete runs.

Next goal continuation (same GPU-availability blocker, second goal turn):

- Authoritative check still finds the external evaluation parent PID 3100546
  live and eight GPU contexts active, with additional trainer processes being
  started. No own GPU benchmark or idle waiter is running.
- All 39 final source hashes match. Rebuilt `post_append_*` host checks passed:
  CPU BF16 860,132; policy 35; dispatcher 74; NCCL identity 8; router core.
- A minimal C++17 program using the earlier positional initializer
  `PacketPipelineOptions{TransportBackend::GpuKernel}` compiled and ran with
  `hostRingReduceScatter == false`. An initial attempt to static_assert the
  default failed because the options type contains vectors and is non-literal;
  the corrected runtime check above passed. No library change was needed.
- Post-append GPU checks, full GPU/SHM measurements, figures and push remain
  pending the same external-resource blocker. The previous goal turn made
  implementation and validation progress; this turn adds final host evidence.

Blocked audit, third consecutive goal turn:

The same external evaluation parent (PID 3100546) remains live. NVIDIA reports
GPU contexts for PIDs 3671752, 3671748, 3671596, 3671742, 3671744, 3671747,
3671743 and 3671751. No own GPU benchmark or waiter is running; no `confirm_*`
full-sweep results exist. The preceding turn completed final host checks and
verified the resource wait. All independent implementation, build and host
validation work is done; remaining completion evidence requires an exclusive
eight-GPU test window. Mark the goal blocked rather than claim completion or
publish unverified performance. Resume from the continuation steps above when
GPU resources are available. Do not overwrite previous successful/interrupted
runs, and preserve unrelated Blackwell edits.

## Resumed verification, 2026-11-14 17:27 UTC onward

The external eight-GPU evaluation has ended. Verified the recorded 39 source
hashes before resuming; no stale benchmark was used. The appended-options build
passed fresh GPU/SHM eager, graph and dispatcher checks and paired 1–8 MiB
checks. Four full 1 KiB–1 GiB sweeps then passed 7,560 timed calls and 3,780
warmups. All 16 target size/mode/sweep combinations beat both the paired
previous selection and NCCL; aggregate latency reductions versus previous are
6.3–15.0%. No unchanged RS point has an aggregate slowdown above 5%.

The only subsequent C++ source change makes the test select native devices
0,1,2,3 on an actual four-GPU host (while retaining 0,1,4,5 on this eight-GPU
host). The test was rebuilt through Make and rerun locally. After a transient
SSH stall cleared, the same executable and generic eager/graph checks all
passed on the separate documented l41 four-L4 host. Exact binary hashes and
remote logs are under `four_l4/`; `final_source_hashes.json` records the final
source state. The measured library and benchmark remain unchanged.

Full eight-GPU Compute Sanitizer instrumentation timed out at 180 seconds. It
is preserved as an incomplete check, not a pass. A bounded memcheck of the
first four ring dispatches on l41 passed with zero errors; remaining cases
completed natively. Native tests cover the full asynchronous/graph workload.

Both revised RS overview plots use all 42 newly measured points and paired
NCCL controls. AR/AG figures remain byte-identical to their earlier revision.
New probes verify 16 SHM/direct/direct NCCL edges in each testbed.

Final confirmation completed: all four AR/AG controls passed another 7,560
timed calls plus 3,780 warmups. Current/previous selector implementations are
identical at 84 keys, with no aggregate slowdown above 5%; these are controls
within the revised binary, not old/new generic-kernel comparisons.

An additional short cross-binary RS check uses the exact archived pre-change
executable (SHA-256 3d1b584e...) versus the new executable, with paired NCCL
controls and reversed binary order between testbeds. All eight keys improve,
by 6.0–15.2%. The 960 timed calls and 480 warmups are independent of the full
sweeps. `original_binary_control.csv` and `original_binary_validation.json`
retain this direct original-code evidence.

The bounded eight-GPU memcheck also passed (first eight ring dispatches, zero
errors, complete native test afterward). The full instrumentation timeout is
still retained as a limitation. Make reports all affected CUDA, host-check and
adaptive targets current; clang-format 14 and source/documentation whitespace
checks pass. Raw hardware/NCCL logs preserve their native trailing spaces.
The implementation, 42 RS figure points, remote four-L4 checks, control runs,
and source/binary identity have been revalidated before the final task commit.
Unrelated Blackwell documentation/data edits are excluded from that commit.
