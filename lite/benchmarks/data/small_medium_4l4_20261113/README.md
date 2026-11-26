# Four-L4 full-size and small/medium optimization study

> This directory retains only inputs and metadata needed by current figures or checks. [Complete original experiment](https://github.com/zhongjiechen/uccl-zhongjie-dev/tree/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/small_medium_4l4_20261113) includes raw runs, removed scripts and source snapshots. Historical commands below require that archive.


This follow-up explicitly covers small, medium and large messages. Hardware is
`mibura-sky-test-02`: four NVIDIA L4s, PCIe Gen4 x16, NUMA pairs (0,1)/(2,3),
dual Xeon Silver 4410Y, CUDA toolkit/runtime 12.9, driver 595.58.03, NCCL 2.28.7.
All payloads are FP32. Byte counts mean complete AR/RS input or AG output per
GPU; AG inputs and RS outputs are one quarter of that extent.

## Accepted ordinary-call results

- **AR:** 21 powers of two, 1 KiB–1 GiB, four processes, in place, maximum-rank
  host enqueue+completion timing; 5 warmups and 3 alternating rounds × 10 calls.
  The final source explicitly completes input H2D copies before timing and uses
  changed input generations. `--sm-variants` includes 1- and 8-CTA hierarchy
  candidates. `--fast-sweep` measures all eligible candidates through 4 MiB and
  the established ring above it; this is a measured shortlist, not an exhaustive
  geometry oracle. Complete output is checked on every warmup/timed invocation.
- **AG/RS:** 27 powers of two, 16 B–1 GiB, one process, out of place,
  maximum-GPU per-call CUDA events including enqueue gaps; 20 warmups and
  3 rotated rounds × 10 calls. Every warmup and timed output is checked.
- Curves select the lowest correct UCCL median at each size, independently of
  NCCL. They are benchmark envelopes, not a portable runtime routing policy.
  Min/max run means, every candidate and all negative results remain in CSV/logs.

[AR selected](ar_selected.csv) · [AR candidates](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/small_medium_4l4_20261113/ar_candidates.csv) ·
[AG selected](ag_selected.csv) · [AG candidates](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/small_medium_4l4_20261113/ag_candidates.csv) ·
[RS selected](rs_selected.csv) · [RS candidates](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/small_medium_4l4_20261113/rs_candidates.csv).
These CSVs retain paired UCCL/NCCL timings in microseconds, including every
slower UCCL median; differences are descriptive, not significance tests.

The new ordinary-path gains are concentrated where the previous report was
weak: fused host AG at 1–16 MiB, phased LL RS at 16–32 KiB, and reduced-block
native AR at 32 KiB–2 MiB. Representative full-sweep values:

| Operation | Full bytes/GPU | Selected path | UCCL / paired NCCL |
| --- | ---: | --- | ---: |
| AR | 32 KiB | hierarchy, 1 CTA | 24.20 / 23.61 µs (remaining loss) |
| AR | 64 KiB | hierarchy, 1 CTA | 27.52 / 29.92 µs |
| AR | 1 MiB | hierarchy, 8 CTAs | 122.96 / 140.73 µs |
| AG | 1 MiB | mini-host | 66.43 / 99.86 µs |
| AG | 4 MiB | mini-host | 188.79 / 233.67 µs |
| AG | 16 MiB | mini-host | 677.11 / 778.43 µs |
| RS | 32 KiB | phased hierarchy LL | 19.47 / 30.43 µs |

In the same AG sweep, prior UCCL candidates took 72.82, 241.38 and 691.76 µs
at 1, 4 and 16 MiB respectively. The 4-MiB AG loss is closed. Ordinary native
AR still has nominal losses at 32 KiB and 4 MiB; they are retained in the table.
The independently monitored final-source AG recheck is `mini_final_recheck.log`;
its monitor records exit 0 and no foreign GPU process. It confirms 16 MiB at
676.20 µs. Its NCCL control belongs to that window, not the main full sweep.

## Implemented mechanisms

`miniHostGather` uses one cooperative grid per GPU to upload each source shard
once into NUMA-local registered host memory, then gather independently ready
sources. The local shard stays on the GPU. Per-source readiness and all-reader
credits protect the bank across asynchronous consecutive calls. The measured
geometry is 16 blocks × 256 threads. Preparation is outside timing. The
standalone prototype reuses the existing two-slot allocation helper while
using one bank; it has no active CPU routing workers. It remains eager-only.

`scatterHierLL` reduces within each local pair, then exchanges the reduced
rank-owned shards across sockets. Two grid barriers keep blocks in the same
phase; the unphased version collapsed at larger sizes. LL packet overhead
makes this useful only in a narrow small-message range. The accepted sweep
uses eight blocks and makes it eligible through 32 KiB. Like the prior LL
candidate, it requires all-rank completion before scratch reuse.

The native AR 1-/8-CTA variants reuse the existing hierarchy algorithm. No
publication fence was weakened. Block-count tuning is represented as explicit
candidates rather than replacing every size with a small grid.

## Native AR input-readiness defect and correction

The initial broad sweep exposed intermittent 1-MiB errors in the native SM
path. Debugging reproduced wrong results on all ranks and showed stale input
contributions. The benchmark filled pageable host input using `cudaMemcpy` on
the default stream, then launched the collective on a nonblocking stream.
A pageable H2D call can return before its DMA completes; the nonblocking
collective stream did not depend on that transfer. This is documented in
[NVIDIA's API synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).

The fix enqueues H2D on the collective stream and explicitly synchronizes it
before the bootstrap barrier and timer. It also changes input generation on
every call and rejects nonfinite outputs in the reference check. The original
SM kernel, without extra fences, then passed 200 timed changed-input stress
calls with a 1-GiB workspace. The final paired `ar_ready_full/` sweep passed its
complete checks. `ar_full/`, `ar_unready_summary/`, and pre-fix geometry screens
are diagnostic records, not accepted final performance measurements.

A fence-only trial did not fix the input race. Its source and logs are retained;
those extra fences were removed. The defect was in input readiness, not proof
that MSCCL++'s existing signal/barrier protocol was invalid.

## Prepared resident mode — separate execution contract

`resident_collectives.cu` prepares one persistent CTA per GPU. A packed 64-bit
release/acquire command carries epoch, operation and extent, replacing three
separate host-memory reads. The workers copy/reduce actual inputs. Tiny RS/AR
use LL packets (through 8 KiB / 2 KiB); larger AR uses fused RS+AG. No precision
conversion or result precomputation occurs.

This mode is **one-process, input-ready, host-blocking**, with FP32 AR in place
and AG/RS out of place. Both resident and NCCL start after all producer streams
complete. The host timer includes command/enqueue and global completion, but
excludes preparation, input fill and checks. Resident workers are stopped for
NCCL windows. It reserves a CTA per GPU and uses CPU polling. Do not call
`cudaDeviceSynchronize` while the workers are active; stop them first or
synchronize only caller streams. Concurrent model compute, multiprocess use,
and integration into application graphs are not validated.

`resident_doorbell_final.log` covers 16 B–1 MiB, 17 sizes × 3 operations,
20 warmups and 3 alternating rounds × 20 calls. All configured warmups and
all **6,120 timed invocations** check changed signed-random inputs against an
independent GPU sum/reference. [Full resident data](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/small_medium_4l4_20261113/resident_selected.csv) are separate from eager curves.
At 1 KiB, AR / AG / RS take **7.65 / 6.03 / 7.42 µs**, versus their paired eager
NCCL controls **33.80 / 30.15 / 29.95 µs**.

A stronger control captures NCCL into a single multi-device fork/join graph,
without per-GPU timing nodes. `resident_graph_final.log` contains another
6,120 checked timed calls with the same input-ready host timer; graph
preparation is excluded. Its [own paired data](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/small_medium_4l4_20261113/resident_graph_selected.csv)
must be used for ratios. A separate per-device-graph probe at 1 KiB passed but
was slower than the global-graph control; its raw results are retained.
These observations do not imply that graphs never help NCCL or that a resident
collective is a drop-in replacement for a stream-enqueued library call.

## Validation and experimental status

- `peer_small_medium_check` checks signed-random data and delayed ranks at
  1056, 262176 and 8388640 bytes. Mini-host includes four asynchronously queued
  generations at a time; LL/fence candidates use their eager boundary.
- `resident_check` checks 83 mixed operations with an independent host reference:
  packet/raw boundaries, scalar tails, input sizes up to 1 MiB, repeated
  start/stop, and safe epoch exhaustion. This checker does not initialize NCCL.
- Compute Sanitizer memcheck with 32-byte allocation padding reports **zero
  errors** for both isolated checks. The NCCL-initializing sanitizer attempt
  reported library initialization errors and timed out; it is not counted as
  a successful sanitizer run.
- An early resident prototype had mismatched cutoff/allocation constants.
  Its `resident_rsag_invalid_bounds*` timings are excluded even though its
  output checker passed. The accepted implementation derives allocation,
  cutoffs and metadata from shared constants and passes the boundary tests.
- Larger serial CTAs, ordinary grid-push AG/RS and extended native LL cutoffs
  did not provide a general gain. All screening logs/source snapshots remain.
- A graph-capable mini-host prototype passed output checks, but its timing
  probe overlapped a later external workload. It is archived as
  `peer_mini_graph_experiment.cu`; the validated eager implementation remains
  active. `mini_graph_smoke.log` is excluded from accepted timings.
- No CPU-router core or dispatcher policy changed. No backend is enabled
  automatically from GPU count. Exact workload/topology/software calibration
  is still required for any production routing decision.

## Reproduction

Run from `experimental/lite` on an idle four-L4 node:

```bash
make allreduce peer_ag_rs peer-ag-rs-packet peer-small-medium-check resident-collectives resident-check
./build/uccl_lite_peer_small_medium_check
./build/uccl_lite_resident_check
./build/uccl_lite_peer_ag_rs_packet --packet-router-ag --packet-router-per-source \
  --mini-host-max-bytes 16777216 --push-grid-blocks 16 \
  -b 16 -e 1073741824 -w 20 -n 10 -r 3
./build/uccl_lite_peer_ag_rs --rs-only --rs-ring --rs-hier-ll-max-bytes 32768 \
  -b 16 -e 1073741824 -w 20 -n 10 -r 3
./build/uccl_lite_resident_collectives -b 16 -e 1048576 -w 20 -n 20 -r 3
./build/uccl_lite_resident_collectives --nccl-graph 1 \
  -b 16 -e 1048576 -w 20 -n 20 -r 3
```

For native AR, alternate three runs of the following two commands, reversing
order in run 2. Save the raw logs using the names in `ar_ready_full/`.

```bash
./build/uccl_lite_allreduce --sm-variants --fast-sweep \
  -b 1024 -e 1073741824 -w 5 -n 10
mpirun --host localhost:4 -np 4 --map-by ppr:4:node --bind-to none \
  --mca pml ob1 --mca btl self,vader,tcp ./build/uccl_lite_nccl_4l4_mpi_baseline \
  --op allreduce -b 1024 -e 1073741824 -w 5 -n 10
```

Regenerate the accepted summaries and figures with
`python3 scripts/summarize_all_sizes_4l4.py` after validating native AR with
`scripts/summarize_single_node_refresh.py`. `identity.json` records final
source/binary/library hashes. `run_idle_gpu_benchmark.py` samples GPU process
ownership and aborts only its own child if a foreign process appears; it is a
monitor, not an exclusive reservation.

## Figure publication

The 1 KiB–1 MiB README figures were regenerated on November 14 from the
recorded samples above; no new timings were substituted. Each focused figure
shows all eleven power-of-two sizes. Ordinary and resident contracts remain
separate, and the full-range figures and tables are retained.

Publication also incorporates the upstream eight-GPU changes through
`f62ff03f`. Those later CPU-router changes do not constitute a remeasurement of
this four-L4 study. Use the recorded source/library hashes for its performance
provenance; compilation and integration checks of the combined revision are
separate from these timing results.


The README presentation now uses the 8×L4 overview style: exactly two panels
per ordinary collective, latency over 1 KiB–1 MiB and normalized bus throughput
over 1 MiB–1 GiB, including 1 MiB in both. Each series has eleven points.
Four-rank normalization is 1.5 for AR and 0.75 for AG/RS, with decimal GB/s.
The former combined focused figures and resident-only plots remain archived
and linked separately; resident data are not extrapolated to 1 GiB. Replotting
does not change any recorded timing or per-size selection.
