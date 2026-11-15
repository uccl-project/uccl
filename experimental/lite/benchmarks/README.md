# Reference measurements

The older files retain a compact measurement record for the kernels,
including historical TP-MoE layer measurements whose host harness is not
shipped here. Those files were collected before the UCCL Lite reorganization;
their labels were normalized, but numerical values were not changed. The
November 10 refresh adds new raw logs and per-call samples separately.
Migration-time validation is in [validation.md](../docs/validation.md).

## Testbed and interpretation

The following four-GPU setup describes the earlier records; eight-GPU
records give their own topology and measurement protocol.

- Four NVIDIA L4 GPUs per node, PCIe Gen4, no NVLink; dual Xeon Silver 4410Y.
- NUMA pairs `(0,1)` and `(2,3)`; two-node tests use two local network rails.
- Current main four-L4 and two-node AR/AG/RS: FP32; archived four-L4
  CPU-router AR/AG/RS: BF16; dense fusion: FP16; MoE: BF16.
- Speedup is baseline latency divided by compared latency. Above 1× is faster.
- AllReduce bus bandwidth and All-to-All aggregate bandwidth are different
  metrics; do not compare their numerical values as if they were the same.
- A zero-error collective check uses deterministic, position-sensitive inputs.
  It does not prove correctness for all inputs or every floating-point order.

## Included data

TXT logs not needed by the main README figures have been pruned. Retained
CSVs, figures and study notes preserve historical measurements and negative
results; references to pruned raw logs describe the original archive, available
in Git history. Restore those logs before rerunning historical summarizers.

| File | Scope and method |
| --- | --- |
| [nccl_tuned_4l4_20261115](data/nccl_tuned_4l4_20261115/README.md) | Testbed 3 FP32 AR/AG/RS: separately calibrated NCCL CE and SM references, three independent validation runs, complete output checks and verified transport paths. |
| [nccl_ascending_pro6000_20261115](data/nccl_ascending_pro6000_20261115/README.md) | Two-PRO6000 BF16 AR/AG/RS memcpy tuning with complete 1 KiB→1 GiB ascending sweeps; independent calibration and validation, with a separate purple reference curve. |
| [nccl_tuned_pro6000_20261115](data/nccl_tuned_pro6000_20261115/README.md) | Two-PRO6000 BF16 AR/AG/RS: independent per-size NCCL calibration, forward/reverse validation, transport evidence and retained stalled candidates; tuned reference added to the README figures. |
| [overview_8l4_20261114](data/overview_8l4_20261114/README.md) | Six GPU/SHM README figures using fresh measurements at all 126 keys from 1 KiB–1 GiB, with no historical tails. |
| [full_refresh_8l4_20261114](data/full_refresh_8l4_20261114/README.md) | Complete paired BF16 AR/AG/RS rerun on GPU-default and explicit SHM testbeds: two reversed sweeps, 22,680 checked timed calls, full-output guards, base-router ablation and actual NCCL transport evidence. |
| [rs_recovery_8l4_20261114](data/rs_recovery_8l4_20261114/README.md) | Opt-in BF16 host-ring ReduceScatter at 1–8 MiB: 6–15% lower latency than the paired previous schedule, four full GPU/SHM sweeps, 7,560 checked timed calls, and retained failed candidates. |
| [shm_refresh_8l4_20261114](data/shm_refresh_8l4_20261114/README.md) | Fresh eight-L4 BF16 AR/AG/RS versus NCCL SHM, fixed optimized configurations through 4 MiB, reversed-order sweeps, 15,600 checked timed calls and verified SHM channels. |
| [rs_submit_8l4_20261113](data/rs_submit_8l4_20261113/provenance.json) | Eight-L4 BF16 ReduceScatter follow-up: small-message GPU progress, optional parallel host submission, fixed-configuration paired NCCL confirmation, launch-skew traces, dense/sparse call-spacing probes, negative results, and eager/graph/capture-guard checks. |
| [ar_ag_transfer_8l4_20261113](data/ar_ag_transfer_8l4_20261113/README.md) | Eight-L4 BF16 transfer of RS scheduling optimizations to AllReduce and AllGather: CPU callback batching, size-limited GPU progress, per-source AG readiness, parallel submission, paired NCCL confirmation and eager/graph checks. |
| [rs_opt_8l4_20261113](data/rs_opt_8l4_20261113/provenance.json) | Eight-L4 BF16 ReduceScatter CPU scheduling: fixed worker-cap, batch-publication, direct-result and CPU destination-stripe ablations, paired NCCL controls over 1 KiB--4 MiB, reversed-order confirmation, raw per-call samples, calibration replay and eager/graph/numerical checks. |
| [optimization_20261110/README.md](data/optimization_20261110/README.md) | Follow-up four-L4 and two-node AR/AG/RS optimization: peer-read small-message path, compact source-completion events, topology-specific local schedules, complete paired samples, bottleneck evidence, and retained failed probes. |
| [multinode_serial_20261111](data/multinode_serial_20261111/README.md) | Two-node AR/AG/RS direct local CUDA-IPC variants against paired NCCL 2.28.7, 1 KiB–1 GiB, with 20 warmups, three alternating runs, complete per-call checks, raw samples, launch-geometry losses, and NCCL protocol diagnostics. |
| [nccl_spike_2x4l4_20261111](data/nccl_spike_2x4l4_20261111/README.md) | Targeted two-node NCCL 2.28.7 protocol diagnosis: tuning logs show default RING+LL at the spike sizes; three alternating checked rounds compare LL and forced SIMPLE with the same ring and channels. |
| [nccl_tuned_2x4l4_20261111](data/nccl_tuned_2x4l4_20261111/README.md) | Eight freshly paired RING+SIMPLE rechecks at NCCL spike sizes, merged with the complete default-control sweep into one explicit per-size NCCL policy for the current two-node figures. All 63 points retain source and configuration labels. |
| [ag_host_router_2x4l4_20261115](data/ag_host_router_2x4l4_20261115/README.md) | Current README multi-node AllGather: fixed 256 KiB shared-host-packet route, 1 MiB–1 GiB FP32 output/GPU, checked host timing and same four-process NCCL control; full samples, negative variants and validation. |
| [host_rdma_multicast_2x4l4_20261111](data/host_rdma_multicast_2x4l4_20261111/README.md) | Experimental two-copy host-RDMA AllGather with three-slot pipeline and NUMA-local GPU multicast. Retains the 1 KiB–2 MiB three-way comparison, exact RDMA-write and UCX-auto variants, tail checks, and large-size regressions. |
| [ucx_eager_nccl_2x4l4_20261111](data/ucx_eager_nccl_2x4l4_20261111/README.md) | Matched-binary two-node host AllGather: forced UCX eager versus paired NCCL 2.28.7 at 1–256 KiB, with tuned NCCL at 128/256 KiB, full samples, protocol trace, and automatic-UCX diagnostics. |
| [host_multicast_selected_2x4l4_20261111](data/host_multicast_selected_2x4l4_20261111/README.md) | Historical selected two-node AllGather figure: host multicast with paired NCCL at 1–256 KiB, previous direct-GPU paired points thereafter, and explicit path/process-layout provenance for every size. |
| [cpu_router_4l4_apply_20261111](data/cpu_router_4l4_apply_20261111/README.md) | Fresh BF16 CPU-router transfer test on this single four-L4 host: paired NCCL over 1 KiB–1 GiB in both placements, 8,940 checked timed samples, automatic replay, compact/GPU-progress negative variants, and topology identity. |
| [cpu_router_4l4_workers_20261111](data/cpu_router_4l4_workers_20261111/README.md) | Four-L4 BF16 CPU worker sweep and adapted eight-worker-per-NUMA AllGather-only profile: 29,100 paired timed calls, both placements, negative AR/RS results, and 126 fresh-process selection checks. |
| [cpu_router_ag_4l4_geometry_20261111](data/cpu_router_ag_4l4_geometry_20261111/README.md) | Four-L4 BF16 AllGather transfer-grid and packet-cap optimization: 248-block/4-MiB-cap profile, 3,900 checked paired calls, 126 automatic decisions, negative small-size results, and Nsight diagnosis. |
| [cpu_router_4l4_figures_20261111](data/cpu_router_4l4_figures_20261111/README.md) | Archived BF16 four-L4 AR/AG/RS figures: every UCCL point runs the CPU router, including regressions; NCCL 2.28.7 is a paired control only. |
| [fp32_packet_ag_4l4_20261111](data/fp32_packet_ag_4l4_20261111/README.md) | Earlier four-L4 FP32 AllGather: source-side push/peer paths through 16 MiB, bitwise host-packet router from 32 MiB, 2,070 checked paired calls and all negative candidates. |
| [fp32_packet_ag_per_source_4l4_20261111](data/fp32_packet_ag_per_source_4l4_20261111/README.md) | Updated four-L4 FP32 AllGather: GPU per-source readiness moves the host-packet crossover to 16 MiB; 2,070 checked paired calls, 4–16 MiB ablation, delayed-producer and graph correctness logs. |
| [host_memory_bw_4l4_20261111](data/host_memory_bw_4l4_20261111/README.md) | Large-working-set STREAM-style host DRAM triad on this four-L4 machine: per-NUMA isolated and simultaneous two-socket bandwidth with 12 physical or 24 logical CPU threads, 24 raw runs and 240 timed passes. |
| [refresh_20261110/README.md](data/refresh_20261110/README.md) | Fresh four-L4 and two-node AR/AG/RS plus dense/MoE fusion. Raw logs, per-call or per-round samples, selected paths, hardware/software identities, matched timing scopes, and negative results. The old mixed-timer two-node archive is excluded from new comparisons. |
| [hardware_profile_4l4_20261110_summary.csv](data/hardware_profile_4l4_20261110_summary.csv) and [raw runs](data/hardware_profile_4l4_20261110_run1.txt) | November 10: standalone GPU topology, directed and four-GPU concurrent peer copies, concurrent H2D/D2H, and NUMA CPU copy/BF16 reduction. Three complete runs × three repetitions, useful-byte bandwidth; no NCCL baseline or collective timing. [Method and limits](../docs/hardware-profiler.md). |
| [peer_ag_rs_4l4_20261110.txt](data/peer_ag_rs_4l4_20261110.txt) | Earlier four-L4 float32 direct-peer AG/RS versus NCCL 2.28.7, 1 KiB–64 MiB. Three alternating repetitions; slowest-rank CUDA events; correctness checks before and after timing. The new sweep checks every call. [Design and negative results](../docs/peer-ag-rs.md). |
| [ag_midrange_4l4_20261111](data/ag_midrange_4l4_20261111/README.md) | Four-L4 sequential peer-write AllGather on both local and l40 hosts, 1 KiB–1 GiB, paired with NCCL 2.28.7; 20 warmups, three rotated runs of ten checked calls. Includes full samples, negative candidates and Nsight profiles. |
| [rs_ar_serial_4l4_20261111](data/rs_ar_serial_4l4_20261111/README.md) | Four-L4 ReduceScatter serial-write and fused-reduction sweep on both hosts, 1 KiB–1 GiB, paired with NCCL 2.28.7. Includes checked samples, large-size losses, and archived AllReduce serial-write candidates that did not show a repeatable gain. |
| [multinode_peer_20261110/summary.csv](data/multinode_peer_20261110/summary.csv), [summary_large.csv](data/multinode_peer_20261110/summary_large.csv), [summary_ar.csv](data/multinode_peer_20261110/summary_ar.csv) | Earlier two-node float32 AG/RS/AR archive, up to 1 GiB. UCCL used CUDA-event time while `nccl-tests` reported host time, so the ratios are diagnostic only. [Raw logs and correction](data/multinode_peer_20261110/README.md), [design](../docs/multinode.md). |
| [gpu-progress/ablation_provenance.json](data/gpu-progress/ablation_provenance.json) and linked CSVs | November 5: CPU versus GPU packet progress, same GPU reduction and packet layout; 16,800 checked timings, all 21 AG/RS sizes. [Direct NCCL results](../docs/baseline-comparison.md). |
| [reduction-placement/ablation_provenance.json](data/reduction-placement/ablation_provenance.json) and linked CSVs | November 5: default CPU, 8-MiB CPU, flat-GPU and interleaved-GPU paths, same executable; 33,600 checked full-sweep timings plus three 1-GiB packet-size probes. AllGather and ReduceScatter, including every regression. [Architecture, blue bar figures and limitations](../docs/reduction-placement.md). |
| [allreduce_single_node.csv](data/allreduce_single_node.csv) | October 12: 1 KiB–1 GiB, 5 warmups, 10 timed iterations, three runs. NCCL 2.18.3/2.28.7 use nccl-tests 2.13.8; UCCL Lite uses its native harness. Fastest correct UCCL Lite variant per size. |
| [allreduce_two_node.csv](data/allreduce_two_node.csv) | October 12: three runs, slowest of eight ranks. Hierarchical UCCL Lite selects among 12 piece/distance configurations. CPU router uses 1 MiB pieces. |
| [cpu_router_4l4_calibration.csv](data/cpu_router_4l4_calibration.csv), [cpu_router_4l4_selection.csv](data/cpu_router_4l4_selection.csv) | New four-L4 BF16 adaptive-module validation against NCCL 2.28.7: three paired repetitions, 5 warmups / 10 iterations, 1–8 MiB AR/AG/RS. All 12 automatic decisions retain the fallback with no router initialization. These are integration calibration results, not the historical peak-throughput sweeps. |
| [collectives_8l4_20261111.csv](data/collectives_8l4_20261111.csv) | Current eight-L4 single-node AR/AG/RS figure data. AllGather uses the [tuned packet run](data/ag_packet_8l4_20261111/README.md), 128/256 KiB AllReduce uses the [compact router run](data/ar_compact_8l4_20261111/README.md), and other points retain their earlier sweep. |
| [ddio_8l4_20261111/README.md](data/ddio_8l4_20261111/README.md) | NUMA-local CPU BF16 reduction after CPU cache warming, LLC eviction, and four GPU SM/copy-engine host writes. Three runs per NUMA node distinguish cache sensitivity from an unverified DDIO mechanism. |
| [collectives_8l4_20261110.csv](data/collectives_8l4_20261110.csv) | Earlier eight-L4 single-node comparison before the AllGather packet update. |
| [shm_8l4_20261111/README.md](data/shm_8l4_20261111/README.md) | 8xL4 SHM logical no-P2P testbed: host-packet SM-kernel AllGather, AllReduce, and ReduceScatter versus verified NCCL SHM, 1 KiB–1 GiB. Two reversed paired sweeps retain 7,560 checked calls, full summary, and transport log. |
| [duplex_allgather_8l4_20261111/README.md](data/duplex_allgather_8l4_20261111/README.md) | Eight-L4 BF16 host-staged duplex AllGather versus the existing packet router and NCCL 2.28.9. Two paired sweeps retain 1,980 checked calls, a 1 GiB CUDA trace, overlap analysis, and small-size regressions. |
| [fusion_8l4_20261110/README.md](data/fusion_8l4_20261110/README.md) | Eight-L4 BF16 AG+GEMM, GEMM+RS, and GEMM+AR CPU-router experiments versus cuBLAS+NCCL 2.28.9. Original and tuned sweeps archive 6,300 checked calls; 3,900 feed the three-path README latency bars. |
| [collectives_small_8l4_20261110/README.md](data/collectives_small_8l4_20261110/README.md) | New 1 KiB–512 KiB BF16 measurements on the same host: local and router path selection, paired router/NCCL controls, and both All-to-All paths. Five sweeps, 5,700 output-checked calls. |
| [reducescatter_8l4_20261110_v2/README.md](data/reducescatter_8l4_20261110_v2/README.md) | Current eight-L4 BF16 ReduceScatter rerun: five paired repetitions against both the local fallback and NCCL, 1 KiB–1 GiB, plus packet/reduction-mode ablations and preserved regressions. |
| [router_vs_nccl_8l4_20261110/README.md](data/router_vs_nccl_8l4_20261110/README.md) | Direct paired CPU-router/NCCL 2.28.9 BF16 AR/AG/RS sweeps, both placements, 1 MiB–1 GiB. Three alternating repetitions × ten checked calls per backend and key. |
| [alltoall_8l4_20261110/README.md](data/alltoall_8l4_20261110/README.md) | Archived eight-L4 BF16 All-to-All experiment: NUMA host staging and peer+host variants against grouped NCCL send/recv, 1 MiB–1 GiB. All 990 timed calls plus warmups checked; NCCL wins all 11 sizes. |
| [local_fallback_8l4_20261110_v3/README.md](data/local_fallback_8l4_20261110_v3/README.md) | November 10: updated eight-L4 BF16 local fallback and size-specific router packet choice. Both placements, 1 MiB–1 GiB AR/AG/RS, 1,980 checked samples each, exact profiles, fresh automatic replays, and packet-size diagnostics. At 1 MiB AR the router beats the separate NCCL control. |
| [local_fallback_8l4_20261110/README.md](data/local_fallback_8l4_20261110/README.md) | Earlier same-day local v2/router calibration, retained with its small-message regression and complete negative results. Its profiles cannot select the revised code. |
| [cpu_router_8l4_20261110/README.md](data/cpu_router_8l4_20261110/README.md) | Earlier same-day eight-L4 BF16 router versus NCCL 2.28.9 control, both placements. Hardware probe, paired timings and old NCCL-backed profiles remain for external comparison; they cannot select against the new UCCL Lite fallback. |
| [primitives_8l4.csv](data/primitives_8l4.csv) | October 12 matched CPU-router primitive rerun after the ReduceScatter fix. Single node, eight L4s, BF16, no GPU P2P; both placements, two retained backends, 24 sizes, three runs. Used for standalone AG/RS charts, not automatic backend selection. [Provenance and port status](../docs/primitive-collectives.md). |
| [collectives.csv](data/collectives.csv) | Earlier compact single-node tables. The AllReduce rows are an older sweep; the current README uses the eight-L4 summary instead. |
| [dense_fusion.txt](data/dense_fusion.txt) | September dense shape sweeps, including NCCL, no-overlap, fusion, and Flux controls. Later blocks supersede earlier exploratory blocks. |
| [moe_layer.txt](data/moe_layer.txt) | Dispatch through combine. Twelve rounds × four iterations, min per variant after settling; not the three-run-median protocol used for the collective plots. |
| [tp_moe_prefill.csv](data/tp_moe_prefill.csv) | Complete TP=4 / EP=1 BF16 MoE layer, Qwen and DeepSeek shapes. Medians of three seed-run medians (1801/1901/2001); five rotating rounds × ten graph replays, slowest rank. Baseline searches 15 vLLM configurations. `previous_fused_ms` is whole-stripe readiness; `fused_ms` is shard readiness. |
| [tp_moe_decode.csv](data/tp_moe_decode.csv) | Complete TP-MoE layer at 1/2/4/8/16/32 tokens. Seeds 3201/3301; ten warmups, five rotating rounds × twenty graph replays; baseline tuned in graph mode. `fused-c512` is grouped fusion; `fused-decode` is direct decode. Ranges are paired seed ranges, not confidence intervals. |

Historical CSVs often preserve aggregate medians and ranges; the current
eight-L4 records also include every timed sample.

The dense and MoE benchmarks include their own baseline implementations. Large
fusion speedups are specific to those workloads; small cases can regress. The
L4 power limit changes clocks under sustained compute, so variant interleaving
and thermal settling matter. Cross-binary Flux comparisons have weaker controls.

TP-MoE timings charge router projection, top-k, GPU routing plan, shared
experts, both routed projections, activation, combination, and AllReduce.
Weights and hidden states are synthetic. These CSVs preserve existing layer
measurements, not application throughput results. Importing the data does not
import the application-side benchmark or imply a fresh validation of the headers.

## Figure conventions

Figures now use absolute metrics. Small collective messages use latency in
microseconds; large messages use bus or aggregate bandwidth in GB/s. Lower
latency and higher bandwidth are better. The current 8xL4 GPU figures use
blue for UCCL-Lite and orange for NCCL; historical figures retain their
recorded palettes. The fusion figures use paired latency bars for each shape.
The recorded min–max timing ranges remain in the CSVs rather than being
presented as confidence intervals.
The current four-L4 FP32 figures use the same two-panel style. AllGather
switches from direct GPU paths to the per-source host-packet router at 16 MiB; the
archived BF16 router-only figures remain in the detailed data.

The November 10 refresh uses the same two-panel latency/throughput style
as the current eight-L4 figures, with only the selected correct UCCL
Lite path and NCCL. Its four-L4 host-clock AR, four-L4 CUDA-event
AG/RS, and two-node host-clock figures are separate so timing
scopes are never combined. Fusion refresh figures use paired bars
of complete-operation latency at every shape, including losses.

The current eight-L4 and historical eight-GPU AG/RS figures split their
recorded sweeps between small-message latency and large-message bandwidth.
The [labeled dense shape plot](../docs/fusion.md) keeps all eight
shapes, including regressions. MoE shows full-layer latency at every recorded
token count, split into three panels with linear millisecond scales. The
two-node large-message latency bars select 64 MiB and 1 GiB, and
the small-latency figure selects readable sizes through 1 MiB. Full AG/RS
sweeps and the 512-KiB NCCL anomaly remain in the CSVs.

## Regenerate figures

```bash
python3 scripts/plot_4l4_collectives.py
python3 scripts/plot_results.py
python3 scripts/plot_multinode_peer.py
python3 scripts/summarize_ar_compact_8l4.py
python3 scripts/summarize_8l4_collectives.py
python3 scripts/plot_8l4_collectives.py
python3 scripts/summarize_8l4_shm_figure.py
python3 scripts/plot_8l4_shm.py
python3 scripts/plot_cpu_router_8l4.py
python3 scripts/plot_refresh_collectives.py
python3 scripts/plot_refresh_fusion.py
python3 scripts/plot_optimized_collectives.py
python3 scripts/plot_multinode_host_router.py
```

Requires Matplotlib and NumPy. `plot_results.py` reads committed CSV/text data
and regenerates four SVGs under `docs/assets/`: historical eight-GPU AG/RS,
the labeled dense-fusion shapes, and EP-MoE. The historical AG/RS
figures now appear in [primitive results](../docs/primitive-collectives.md).
`plot_multinode_peer.py` draws two archived diagnostic SVGs from the
mixed-timer two-node AG, RS, and AR summaries. The eight-L4 summarizer selects the fastest
measured UCCL Lite candidate before `plot_8l4_collectives.py` draws three
two-panel figures, one per collective. `plot_cpu_router_8l4.py` draws one diagnostic SVG
from the current eight-L4 paired BF16 local/router calibration and automatic
selection replay, with the earlier NCCL run as a separate control, for ten
older figures. The two refresh plotting scripts draw six collective and two
fusion SVGs. `plot_optimized_collectives.py` draws the archived follow-up
collective figures plus the current two-node host-router AllGather SVG. The latter can
also be regenerated alone with `plot_multinode_host_router.py`; it reads the
November 15 `shared_ready` and its own `nccl_shared_ready` control, with no
per-size variant selection. These scripts do not run GPU work or benchmarks.
Use `python3 scripts/plot_results.py --preview` to also export PNG and PDF copies
under ignored `build/figures/` for inspection or reuse in a technical report.
`python3 scripts/plot_multinode_peer.py --preview` exports PNG/PDF copies there.
`python3 scripts/plot_8l4_collectives.py --preview` exports the three current
eight-L4 figures there.
`python3 scripts/plot_cpu_router_8l4.py --preview` does the same for the
eight-L4 CPU-router figure.
For fresh measurements, use the commands in [build.md](../docs/build.md), keep
correctness enabled, and save the complete run configuration with each result.
