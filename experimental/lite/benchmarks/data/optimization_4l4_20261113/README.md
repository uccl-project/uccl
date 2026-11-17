# Four-L4 pipelined ReduceScatter, November 13, 2026

On `mibura-sky-test-02`, the new opt-in `--rs-ring` candidate overlaps peer
copy-engine transfers with GPU reduction of 8 MiB pieces on a separate stream.
Three ring steps use order 0→1→3→2→0, immutable receive slots and per-tile CUDA
events. Each public stream joins its reduction stream before completion;
all-rank entry events protect scratch across asynchronous calls. FP32 inputs
remain immutable during each call and the result is out of place.

The full sweep's 1 GiB result is **39.446 ms**, versus the original peer path
**48.595 ms** and paired NCCL **48.972 ms**: **1.232×** and **1.242×** speedups.
Bus bandwidth is **20.416 GB/s**, using 0.75 × full input bytes / seconds.
The first shorter sweep measured 39.436 / 48.627 / 47.495 ms; NCCL varies across
windows, so use each run's own paired control. Do not compare against an older
NCCL window or claim an identical NCCL ratio across runs.

Hardware: four NVIDIA L4s, PCIe Gen4 x16, NUMA pairs (0,1)/(2,3), dual Xeon
Silver 4410Y, CUDA toolkit/runtime 12.9, driver 595.58.03, NCCL 2.28.7 default.
The full run has 10 warmups and three rotated rounds of 10 timed calls per
candidate/size. Maximum-GPU per-call CUDA events include host enqueue gaps;
allocation, fill and verification are outside timing. Every warmup and timed
output is checked. `selected.csv` and the SVG choose the fastest measured UCCL
candidate at each size, independently of NCCL. This is a benchmark envelope,
not a portable runtime routing policy. The ring remains explicitly opt-in.

| Full input/GPU | Selected ring | Original peer | Paired NCCL |
| ---: | ---: | ---: | ---: |
| 16 MiB | 725.54 µs | 799.33 µs | 905.55 µs |
| 64 MiB | 2.554 ms | 3.057 ms | 3.238 ms |
| 256 MiB | 9.939 ms | 12.239 ms | 12.596 ms |
| 1 GiB | 39.446 ms | 48.595 ms | 48.972 ms |

Reproduce from `experimental/lite`:

```bash
make peer_ag_rs peer-ring-check
./build/uccl_lite_peer_ring_check
./build/uccl_lite_peer_ag_rs --rs-only --rs-ring \
  -b 1024 -e 1073741824 -w 10 -n 10 -r 3
python3 scripts/summarize_rs_pipeline.py
```

The summarizer consumes `rs_full.log`, validates 2,940 timed samples and writes
all 21 selected rows plus the README figure. `rs_final_1g.log` is a separate
final-build confirmation (5 warmups, 3×10 calls): ring 39.664 ms, peer
48.676 ms, NCCL 46.425 ms, or 1.227× over peer and 1.170× over NCCL. The full sweep preceded only
clang-format-14 changes; the final build/check revalidates the formatted source.
`identity.json` records final source/binary/library hashes, not a purported hash
of earlier experimental binaries.

The additional independent check uses signed pseudo-random rank/position/
generation-dependent values and a double-precision host sum, with absolute plus
relative tolerance 1e-6. It tests 1008, 262160, 8388624 and 67108880 bytes, rotates
a delayed rank and enqueues four changed generations without host completion,
checking every fourth result (12 generations per size). The standard benchmark
checks every invocation. Both tests are eager; graph capture is not claimed.

Negative controls remain: whole-shard hierarchy `--rs-hier` reached 53.727 ms
at 1 GiB; the unpipelined ring reached 50.000 ms. Cutting socket traffic or
changing to a ring without copy/reduction overlap did not improve that size.
The original and unpipelined source snapshots and all raw logs are retained.
The hierarchy is experimental and is not included in the published envelope.

No CPU-router core or selection policy changed. Additional AR, AG, geometry
and graph experiments, including negative results, follow below.

All measurements below use the same four-L4 owner host described in README.md.
No rejected tuning is promoted just because it reduces traffic or block count.
Raw logs are in `tuning/`; all original controls remain available.

## ReduceScatter granularity

The validated default is 8 MiB per ring piece. A subsequent 1 GiB sweep used
3 warmups and 3 rotated rounds of 5 checked calls, FP32 out of place,
maximum-GPU CUDA-event timing. Ring times:

| Piece | Time |
| --- | ---: |
| 1 MiB | 40.891 ms |
| 4 MiB | 39.746 ms |
| 8 MiB (previous accepted window) | 39.446 ms |
| 16 MiB | 39.888 ms |
| 32 MiB | 44.920 ms |

4/8/16 MiB are close; 8 MiB remains the established choice. There is no
per-size retuning in the published ring candidate. The small differences
across windows do not prove a universal optimum.

## AllGather packet geometry

Six candidates varied the packet cap and transport-block cap with unchanged
per-source GPU progress. Each 1 GiB window paired packet AG, direct peer and
NCCL with 3 warmups and 3×5 checked calls. FP32 bits are copied without dtype
conversion; results are full output/GPU and maximum-GPU CUDA-event latency.

| Block cap | Packet cap | Packet AG |
| ---: | ---: | ---: |
| 64 | 1 MiB | 39.635 ms |
| 64 | 8 MiB | 38.959 ms |
| 128 | 1 MiB | 38.069 ms |
| 128 | 8 MiB | 37.895 ms |
| 248 | 1 MiB | 38.527 ms |
| 248 | 8 MiB | 37.403 ms |

The established 248-block/4-MiB path measured 36.429 ms in the initial
confirmation window. None of these candidates justifies changing that
geometry. The final same-geometry sweep is recorded under
[Final AllGather confirmation](#final-allgather-confirmation).
The harness now exposes `--packet-bytes` and `--packet-blocks`; these do not
modify the CPU-router core or its calibration policies.

## Graph measurement contract

The initial attempt to capture the direct peer-copy path failed at
`cudaMemcpyPeerAsync` on CUDA 12.9 (see `graph_diagnostic.log`). Supported graph
experiments are restricted to SM pull/push-serial candidates and NCCL through
1 MiB. LL/fused flags that embed a host generation are excluded: replaying a
captured constant generation would not be a correct protocol.

One fork/join graph spans the four GPUs. Capture/instantiation and changed-input
fill are excluded from timing. Each replay records per-GPU external timing
events and joins all GPU ends to the origin stream before validation. Every
warmup and timed replay checks complete output. It does not divide a graph
candidate by eager NCCL. The earlier capture with ordinary internal event nodes
failed `cudaEventElapsedTime`; the corrected graph uses external record nodes
for actual replay timestamps. Failed captures are diagnostics, not timings.

The completed supported graph sweep (`graph_full.log`) has 10 warmups and
3×10 checked calls, 11 sizes (1 KiB–1 MiB), two collectives: **1,680 timed
samples**. The 1008-byte scalar tail also passes. Paired graph results:

| Operation | Full bytes/GPU | Selected SM candidate | UCCL / NCCL (µs) |
| --- | ---: | --- | ---: |
| AG | 1 KiB | pull | 12.218 / 20.502 |
| AG | 32 KiB | push-serial | 13.088 / 23.350 |
| AG | 1 MiB | push-serial | 69.178 / 93.690 |
| RS | 1 KiB | pull | 11.837 / 20.512 |
| RS | 32 KiB | push-serial | 37.712 / 23.600 |
| RS | 1 MiB | push-serial | 83.318 / 84.112 |

This selects per-size UCCL variants within the supported graph set, not against
NCCL. In particular, graph RS at 32 KiB loses and is worse than the earlier
eager fused-fence candidate. It is not a reason to default all callers to graph
mode. The protocol events are re-recorded after graph destruction before any
subsequent eager calls: internal captured event dependencies are not live eager
event records. Failed lifecycle/capture diagnostics are retained separately.

Reproduce: `./build/uccl_lite_peer_ag_rs --graph -b 1024 -e 1048576 -w 10 -n 10 -r 3`.
The ordinary eager path and its selected envelopes remain separate.

## AllReduce launch geometry and ring granularity

The native AR harness adds only a diagnostic `-a INDEX` algorithm filter; index
8 selects the existing A3 ring. No publication fence was weakened. Six 1 GiB
configurations used FP32 in-place, four processes, slowest-rank host call timing,
2 warmups and 5 checked calls each. There is no paired NCCL in these tuning
windows and no per-size best-geometry curve.

| Blocks × threads | Piece target | Mean latency |
| --- | ---: | ---: |
| 8 × 512 | 2 MiB | 80.132 ms |
| 8 × 512 | 8 MiB | 80.114 ms |
| 24 × 512 | 2 MiB | 80.292 ms |
| 24 × 512 | 8 MiB | 80.230 ms |
| 48 × 512 | 2 MiB | 80.956 ms |
| 48 × 512 | 8 MiB | 80.962 ms |

The code caps pieces at 16 per quarter, so at 1 GiB both requested piece targets
produce 16 pieces and an actual 16 MiB extent. This is a geometry/cap check,
not evidence that actual 2 MiB and 8 MiB transfers perform identically.
The ~0.1% difference between 8 and 24 blocks is insufficient to promote a
new default from these single windows; 48 blocks regresses. The original
24×512 and 8 MiB requested target remain unchanged.

At 64 MiB, where the piece targets are not capped to the same value, 2 MiB
pieces took 5.569 ms versus 5.253 ms for 8 MiB (3 warmups, 10 checked calls).
This also supports retaining the established target.

The final native 1 GiB AR confirmation alternates three UCCL/NCCL windows,
5 warmups and 10 timed calls in each, four processes, FP32 in-place,
slowest-rank host enqueue+completion timing. Median run means are **80.243 ms**
for A3 ring and **90.364 ms** for NCCL. All checked errors are zero.
`ar_uccl_paired_*.log` and `ar_nccl_paired_*.log` contain 60 timed samples.
The earlier attempted NCCL command with 3 warmups/5 calls was rejected by the
baseline's minimum-count validation; it supplied no measurement.

## Experimental hybrid AllReduce

`make peer-hybrid-allreduce` builds `uccl_lite_peer_hybrid_allreduce`, an explicit
one-process FP32 in-place candidate. It composes the new ring ReduceScatter
with the existing per-source mapped-host AllGather. RS writes a final immutable
shard to temporary output; AG reads that shard and overwrites the original
full input with the reduced result. No extra full-array output copy is needed.
Per-stream completion and packet ownership protect the two stages.

`hybrid_final.log`: 16 MiB–1 GiB, fixed 8 MiB RS pieces and 4 MiB AG packets,
248-block AG cap, 10 warmups and three alternating rounds of 10 calls per
backend. Every warmup and timed call checks all elements on all GPUs. The timer
is maximum-GPU eager CUDA-event duration, including submission gaps. The paired
NCCL 2.28.7 control is also one-process FP32 in-place with the identical timer.
This curve uses one fixed candidate, **no per-size variant selection**.

| Full input/GPU | Hybrid AR | Paired NCCL |
| ---: | ---: | ---: |
| 16 MiB | 1.376 ms | 1.485 ms |
| 64 MiB | 4.939 ms | 5.514 ms |
| 256 MiB | 19.179 ms | 21.753 ms |
| 1 GiB | **75.978 ms** | **91.165 ms** |

At 1 GiB the matched NCCL speedup is **1.200×**. The first shorter sweep
measured 76.111 versus 89.869 ms. The candidate is experimental and opt-in;
it is not a replacement for the multiprocess native AR API. The native AR
80.243 ms is a different process/timing contract and is not the denominator
of a claimed hybrid speedup. At a 1008-byte tail, hybrid takes 127.3 µs versus
NCCL 32.4 µs, a retained negative result. Do not select it solely by GPU count.

`make peer-hybrid-check` builds an independent signed-input host-reference
check for the composition. It passes 1008, 262160, 8388624 and 67108880 bytes,
12 changing generations per size, rotating delayed producers, with groups of
four calls enqueued asynchronously and every fourth output fully checked.
Both stage-specific RS checks and the composition check use a double host sum
and absolute+relative tolerance 1e-6. Hybrid graph capture is not supported or
claimed. The existing router graph regression is separate.

Reproduce the fixed large-message curve:

```bash
make peer-hybrid-allreduce peer-hybrid-check
./build/uccl_lite_peer_hybrid_check
./build/uccl_lite_peer_hybrid_allreduce -b 16777216 -e 1073741824 \
  -w 10 -n 10 -r 3
python3 scripts/summarize_latest_4l4.py
```

The summarizer checks all **420 timed samples**, writes `ar_hybrid_summary.csv`
and the hybrid SVG. Final regression logs also retain backend policy,
dispatcher, four-GPU eager GPU-progress reduction (26,569,136 scalar/guard
checks), and four-GPU per-source graph checks (1,000 captured mixed collectives,
five replays, producer/consumer dependencies and ring wrap). The CPU-router
core and dispatch policy source were unchanged.

## Final AllGather confirmation

`ag_final_full.log` confirms the unchanged default per-source packet path:
**36.463 ms** at 1 GiB versus paired peer **45.507 ms** and NCCL **48.656 ms**.
It contains **2,070 checked timed calls**, 21 sizes, 10 warmups and 3×10 calls,
with the same FP32 out-of-place event timing as the earlier geometry windows.
`ag_selected.csv` and the refreshed SVG select the lowest UCCL median per size
without consulting NCCL. The earlier packet sweep and figure remain archived.

The final binary's independent 1 GiB hybrid recheck (5 warmups, 3×10 calls)
measured **75.883 ms** versus NCCL **86.570 ms** (**1.141×**). This confirms
candidate timing while exposing baseline variability; do not report the
full-sweep 1.200× ratio as invariant. The final source also explicitly rejects
`--graph` for hybrid AR rather than silently ignoring that option.

## Small native AllReduce geometry/proxy trial

The final tuning trial retained 2,400 fully checked timed calls: hierarchy LL
and tree LL at 1–16 KiB, four block/thread/proxy configurations, three windows,
10 warmups and 20 calls per size. It uses the native four-process FP32 in-place
host timer, without a paired NCCL run. `tuning/ar_small_summary.csv` preserves
all median/min/max run means; the raw logs include every call and zero errors.

Tree LL median examples (µs):

| Configuration | 1 KiB | 16 KiB |
| --- | ---: | ---: |
| Default 24×1024, proxy enabled | 15.802 | 24.641 |
| 1×256, proxy enabled | 18.064 | 38.043 |
| 1×256, proxy disabled | 17.462 | 37.691 |
| 4×256, proxy disabled | 13.648 | 27.772 |

The tiny-message candidate trades a lower 1-KiB median for a 16-KiB regression;
window variation is substantial, and proxy enablement and block geometry both
change in the last row. This is not sufficient evidence for a new global
default or a new advertised speedup against NCCL. The existing native defaults
remain intact, and no small-AR timing from this trial is spliced into another
host's curve.
