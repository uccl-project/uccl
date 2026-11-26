# Four-L4 and two-node collective optimization

This experiment follows the checked
[November 10 baseline](../refresh_20261110/README.md). It uses the same
four-L4 2+2 NUMA nodes, NCCL 2.28.7 library, FP32 dtype, complete byte
extents, placements, and per-operation timing scopes. UCCL path selection
uses only correct UCCL candidate medians; NCCL timings do not select a path.

## Bottleneck evidence

The original single-node peer AG/RS path submits 12 cross-GPU and four local
copies, records source completion events, and makes destination streams wait
for source completion. RS then launches a separate reduction kernel and
waits for destination ownership before the next call. The
[1-KiB Nsight Systems summary](ablation/peer_1k_nsys_stats.txt) records
hundreds of peer copies and over a thousand event waits across the profiled
warmups, checks, and calls, while individual tiny GPU peer copies take about
one microsecond. The summary includes setup and verification, so its totals
are not a per-call critical-path decomposition. The measured improvement
from removing these submissions is the primary causal test.

The peer-read prototype uses one kernel on each destination GPU to read the
other ranks' immutable input directly. An unfenced version passed tiny sizes
but [failed RS correctness at 1 MiB](ablation/peer_pull_no_fence_failure.log).
Cross-device event publication and waits fixed correctness but left roughly
30-µs small-message latency
([event-control log](ablation/peer_pull_event_sync.log)). The accepted eager
benchmark synchronizes every device's input fill before timing **for all
backends**, then issues only the four peer-read kernels. This passed the
checked [small-size sweep](ablation/peer_pull_device_sync.log). Remote peer
reads lose bandwidth quickly, so the small-message policy returns to the
peer-write path above 32 KiB. This input-ready contract has been validated
for the standalone eager harness; arbitrary-stream integration is outside
this result.

For RS, the earlier peer-write path also copied each rank's own input shard
to local scratch before launching the destination reduction. Two-host A/B
checks show that reading the own shard directly in the reduction is faster
at 256 KiB, 1–8 MiB, and at least 32 MiB. The
[l40 comparison](ablation/peer_no_self_l40.log) and
[owner-host comparison](ablation/peer_no_self_owner.log) retain the full
candidate sweeps. At 1 GiB the median UCCL latency falls from about
50.87 to 48.62 ms on l40 and from 50.85 to 48.56 ms on the owner host.
The 128-KiB, 512-KiB, and 16-MiB cases regress on at least one host; the
policy keeps the local copy at those extents.

On l40, the [original](ablation/peer_control_l40.log) and
[compact-event](ablation/peer_compact_l40.log) binaries ran the same checked
1-KiB–8-MiB workload. Collecting all source completion events on one leader
and publishing one completion event reduces AG latency at 32–256 KiB and RS
latency at 32–64 KiB. At and above 1 MiB the compact route is slightly
slower; the final policy gates it to those small extents.

The two-node ring AG/RS kernels impose dependent local phases before and
after the one-rank-per-rail IB exchange. The checked
[flat AG](ablation/multinode_ag_flat.log) and
[flat RS](ablation/multinode_rs_flat.log) probes show that avoiding local
ring phases cuts 2-KiB–512-KiB AG and 2-KiB–512-KiB RS latency. For RS at
1–16 MiB, the existing
[ring with one GPU-packed piece](ablation/multinode_rs_pack1.log) is faster
than both flat and direct ring; the [four-piece variant](ablation/multinode_rs_pack4.log)
is a negative result. At and above 32 MiB, direct ring wins. The
[flat local-AG AllReduce](ablation/multinode_ar_flat.log) probe shows a
smaller small-message gain. Reducing the AllReduce grid to 8×256
([negative control](ablation/multinode_ar_b8t256.log)) did not help.
The checked [three-mode RS smoke](ablation/multinode_rs_3mode_smoke.log)
measured 1/2/4/8-MiB calls at about 130/185/292/512 µs; the earlier
[two-mode summary](ablation/multinode_2mode_summary.csv) was about
188/280/412/592 µs at those sizes. These diagnostic observations are
confirmed or rejected by the full paired sweep below.
After removing the extra local ring phases, small AG/RS still have an
approximately 75–90-µs floor. IB proxy progress and the remaining dependent
network phase are plausible contributors; this sweep does not isolate their
shares, so no finer bottleneck attribution is claimed.

The single-node AllReduce 32-KiB regression did not improve by extending
the LL cutoff ([LL probe](ablation/allreduce_32k_ll.log)) or by the tested
[SM launch geometries](ablation/). These negative results remain in the
archive; that kernel is unchanged by this optimization.

## Measurement and artifacts

The accepted comparisons use at least five warmups, three alternating paired
runs, ten timed calls per run, and complete output checks after every call.
Single-node peer AG/RS use slowest-rank CUDA-event time in one harness.
Two-node AR/AG/RS use slowest-rank host enqueue+completion time with one GPU
per MPI rank and a checked NCCL control. Input fill and verification are
outside each timed call. Raw logs, per-call samples, candidate paths, and
summary CSVs are retained here. The earlier baseline files remain unchanged.

### Single-node four-L4 result

The accepted l40 sweep contains the [complete log](single_peer_l40.log),
[2,880 timed call samples](single_peer_samples.csv),
[all peer-copy/peer-read/NCCL candidates](single_peer_candidates.csv), and
[selected-path summary](single_peer_summary.csv). The same final binary
passed a [1-KiB–8-MiB checked sweep on the owner host](ablation/peer_owner_small_checked.log),
whose [summary](ablation/peer_owner_small_summary.csv) is kept separately
because that host has a different driver. At 2 KiB on l40, AG/RS take
14.07/13.73 µs with UCCL Lite versus 26.57/26.34 µs with NCCL. At 1 GiB,
AG wins at 45.489 versus 47.599 ms; RS loses at 48.579 versus 46.567 ms.
Within this same checked sweep, the peer-write candidate alone beats NCCL
at 8 AG and 2 RS sizes; the selected peer-read/peer-write policy raises
those counts to 13 AG and 7 RS sizes. This is a within-harness comparison,
not an inference from the different-driver historical sweep.
The [source-side fan-out prototype](ablation/peer_push_negative_l40.log)
passed correctness but collapsed at medium sizes and was excluded.
The [pre-compact owner sweep](ablation/peer_precompact_partial.log) is a
partial diagnostic stopped after the complete final l40 sweep was accepted;
it is not used in any selected-path summary.
An earlier full peer sweep used a GPU comparison that would not count a NaN
output as an error. Its [raw log](ablation/peer_before_nancheck_l40.log)
and [summary](ablation/peer_before_nancheck_summary.csv) are retained but
excluded. The accepted sweep uses a NaN/Inf-aware full-output checker and
was rerun on l40; its small sizes and 1008-byte tail were also rechecked
on the owner host.
The subsequent [self-copy control](ablation/peer_self_copy_l40.log) already
used the NaN-aware checker; it is retained as the direct-self reduction
ablation, while the accepted log above uses the measured size-specific
choice.

### Two-node result

The accepted three-run
[paired AR/AG/RS logs](multinode_paired/),
[3,780 timed call samples](multinode_samples.csv), and
[complete selected-path summary](multinode_summary.csv) retain all 21 sizes,
run ranges, path identities, and losses. At 1 MiB, the three-mode RS path
takes 135.1 µs versus 276.3 µs for the earlier UCCL path, while the paired
NCCL control takes 117.6 µs. At 1 GiB, AR/AG/RS take about
106.108/53.960/53.670 ms versus 109.581/61.209/58.722 ms for NCCL.
The [peer build](peer_build.log), [two-node build](multinode_build.log), and
[source/binary/library identities](artifacts.txt) are retained here. The
first complete auto-policy scan used different NCCL control executable
hashes on the two nodes; its [raw logs](ablation/multinode_unmatched_binary/)
and [summary](ablation/multinode_unmatched_summary.csv) are preserved but
excluded from accepted figures. The final paired scan uses identical
UCCL, NCCL control, MSCCL++, and NCCL library hashes on both nodes.
The final scoped rebuild passed
([build log](final_build.log)); a fresh
[1008-byte peer tail check](ablation/peer_tail_1008_final_build.log) also
passed. The measured executable hashes are preserved in `artifacts.txt`.

## Reproduce

Run from `experimental/lite/` after building the affected Makefile targets
and staging identical two-node executables and libraries:

```bash
./build/uccl_lite_peer_ag_rs -b 1024 -e 1073741824 \
  -w 5 -n 10 -r 3 > build/peer-optimized.log

HOSTS=localhost:4,l40:4 SOCKET_IFNAME=ibp55s0f0 \
HCA_DEVICES=mlx5_0,mlx5_1 \
MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun \
NCCL_BASELINE_BIN="$PWD/build/uccl_lite_nccl_2x4l4_baseline" \
NCCL_LIB=/home/zhongjie/third_party/nccl/build/lib \
OPERATIONS='allreduce allgather reducescatter' \
MIN_BYTES=1024 MAX_BYTES=1073741824 WARMUPS=5 ITERATIONS=10 \
REPEATS=3 TIMING_MODE=host CHECK_EACH=1 AUTO_VARIANTS=1 \
OUTPUT_DIR="$PWD/build/multinode-optimized" \
bash scripts/run_multinode_peer_compare.sh
```

`scripts/summarize_peer_pull.py` and
`scripts/summarize_multinode_checked.py --auto` validate the checked samples
and selected paths. `scripts/plot_optimized_collectives.py --preview`
regenerates the five optimized SVGs and PNG/PDF previews in the current
eight-L4 figure style. The [baseline record](../refresh_20261110/README.md)
contains the hardware topology, software versions, and earlier raw sweeps.
