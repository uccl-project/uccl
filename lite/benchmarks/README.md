# Benchmarks and retained figure inputs

This directory contains buildable benchmark/probe sources and the compact
measurement inputs used by the current README or regression checks. Raw-run
archives, duplicate source trees, obsolete benchmark copies and superseded
figures are in the [immutable experiment archive](../docs/archived-experiments.md).

Historical results retain their recorded hardware, dtype, baseline, timing
scope, repetitions and variant-selection rules. They do not qualify the
current runtime. Negative results remain in the retained method notes and
linked archive. Future raw outputs should be written under `build/`.

## Active harnesses

| Directory | Purpose | Example build target |
| --- | --- | --- |
| `collectives/` | Standalone collective and native NCCL controls | `make nccl_4l4_mpi_baseline nccl_2x4l4_baseline` |
| `cpu_router/` | Router comparisons, transport probes and worker calibration | `make router-worker-calibrate compact-collectives` |
| `fusion/` | Dense and MoE complete-operation comparisons | `make fusion` |
| `joint/` | Current peer/router harness and independent native reference | `make joint-bench joint-nccl-reference` |
| `probes/` | Topology, memory, compute and network resource measurements | `make topology calibrate-hardware calibrate-host` |

See [build instructions](../docs/build.md) for dependencies and correctness
commands. The host-memory triad source is now `probes/host_memory_triad.cc`;
`make host-memory-bw-triad` retains its existing executable target.

## Retained datasets

Only inputs consumed by current figure renderers, required test fixtures, and
compact associated provenance/method notes are retained. Checksums in original
metadata can name archived files; the complete records are at the pinned
revision. The manifest below records the retained files themselves.

| Dataset | Retained files |
| --- | ---: |
| [benchmarks/data](../benchmarks/data) | 1 |
| [benchmarks/data/ag_combined_latest_2x4l4_20261117](../benchmarks/data/ag_combined_latest_2x4l4_20261117/README.md) | 4 |
| [benchmarks/data/ag_latency_2x4l4_20261116](../benchmarks/data/ag_latency_2x4l4_20261116) | 1 |
| [benchmarks/data/ar_small_fix_2x4l4_20261115](../benchmarks/data/ar_small_fix_2x4l4_20261115/README.md) | 2 |
| [benchmarks/data/fusion_8l4_models_20261111](../benchmarks/data/fusion_8l4_models_20261111/README.md) | 3 |
| [benchmarks/data/fusion_8l4_models_ag_rs_20261111](../benchmarks/data/fusion_8l4_models_ag_rs_20261111/README.md) | 3 |
| [benchmarks/data/fusion_rs_8l4_optimized_20261111](../benchmarks/data/fusion_rs_8l4_optimized_20261111/README.md) | 3 |
| [benchmarks/data/nccl_ascending_pro6000_20261115](../benchmarks/data/nccl_ascending_pro6000_20261115/README.md) | 3 |
| [benchmarks/data/nccl_local8_strict_20261119](../benchmarks/data/nccl_local8_strict_20261119/README.md) | 6 |
| [benchmarks/data/nccl_local_strict_4l4_20261119](../benchmarks/data/nccl_local_strict_4l4_20261119) | 2 |
| [benchmarks/data/nccl_tuned_2x4l4_20261116](../benchmarks/data/nccl_tuned_2x4l4_20261116/README.md) | 3 |
| [benchmarks/data/nccl_tuned_4l4_20261115](../benchmarks/data/nccl_tuned_4l4_20261115/README.md) | 2 |
| [benchmarks/data/nccl_tuned_pro6000_20261115](../benchmarks/data/nccl_tuned_pro6000_20261115/README.md) | 3 |
| [benchmarks/data/nccl_tuned_testbeds12_20261115](../benchmarks/data/nccl_tuned_testbeds12_20261115/README.md) | 3 |
| [benchmarks/data/overview_8l4_20261114](../benchmarks/data/overview_8l4_20261114/README.md) | 5 |
| [benchmarks/data/pro6000_ag_rs_cpp_20261116](../benchmarks/data/pro6000_ag_rs_cpp_20261116/README.md) | 3 |
| [benchmarks/data/pro6000_allreduce_ce_ring_cpp_20261116](../benchmarks/data/pro6000_allreduce_ce_ring_cpp_20261116/README.md) | 3 |
| [benchmarks/data/pro6000_b0ac7b4a_validation](../benchmarks/data/pro6000_b0ac7b4a_validation/README.md) | 4 |
| [benchmarks/data/pro6000_nccl_baselines_cpp_20261116](../benchmarks/data/pro6000_nccl_baselines_cpp_20261116/README.md) | 8 |
| [benchmarks/data/refresh_20261110](../benchmarks/data/refresh_20261110/README.md) | 3 |
| [benchmarks/data/rs_direct_medium_2x4l4_20261115](../benchmarks/data/rs_direct_medium_2x4l4_20261115/README.md) | 2 |
| [benchmarks/data/small_medium_4l4_20261113](../benchmarks/data/small_medium_4l4_20261113/README.md) | 5 |
| [benchmarks/data/uccl_reference_gap_20261118](../benchmarks/data/uccl_reference_gap_20261118/README.md) | 7 |
| [integration/l4_e2e/data/l4_20261116](../integration/l4_e2e/data/l4_20261116/README.md) | 5 |
| [integration/nccl_lite/data/e2e_large_messages_8l4_20261115](../integration/nccl_lite/data/e2e_large_messages_8l4_20261115) | 3 |
| [integration/peer_e2e/data/pro6000_20261115](../integration/peer_e2e/data/pro6000_20261115) | 3 |

The six model-fusion sample CSVs remain because their checked per-run medians
are the direct inputs to the published bars. Large raw JSON/log/trace archives
and packaged binaries are excluded.

## Regenerate the current README figures

Run from `lite/`, with Matplotlib and NumPy installed. These are CPU-only
renderers using historical inputs; they do not run new GPU benchmarks.

```bash
python3 -B scripts/plot_4l4_collectives.py
python3 -B scripts/plot_nccl_dropin.py
python3 -B scripts/plot_nccl_local8.py
python3 -B benchmarks/data/pro6000_b0ac7b4a_validation/plot.py
python3 -B scripts/plot_pro6000_nccl_baselines.py --summary benchmarks/data/pro6000_nccl_baselines_cpp_20261116
python3 -B scripts/plot_refresh_fusion.py
python3 -B scripts/plot_router_fusion_8l4_models.py
python3 -B scripts/plot_router_fusion_8l4_models_ag_rs.py --op ag-gemm
python3 -B scripts/plot_router_fusion_8l4_models_ag_rs.py --op gemm-rs --data-dir benchmarks/data/fusion_rs_8l4_optimized_20261111 --tag optimized
python3 -B integration/l4_e2e/plot.py --summary integration/l4_e2e/data/l4_20261116/summary_4l4.json --out docs/assets
python3 -B integration/l4_e2e/plot.py --summary integration/l4_e2e/data/l4_20261116/summary_2x4l4.json --out docs/assets
python3 -B integration/peer_e2e/plot.py --summary integration/peer_e2e/data/pro6000_20261115/summary.json
python3 -B integration/nccl_lite/plot_e2e_comparison.py --summary integration/nccl_lite/data/e2e_large_messages_8l4_20261115/summary.json
```

The logo is a static asset. Other historical renderers require their inputs
from the archive and are not part of this figure set.
