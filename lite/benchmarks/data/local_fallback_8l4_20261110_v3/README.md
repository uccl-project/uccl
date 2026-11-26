# Eight-L4 BF16 adaptive calibration, local fallback v3

This rerun measures the project-owned local collective against the CPU router
after two small-message changes: the local AllReduce/ReduceScatter path reduces
up to 2 MiB in pinned host memory, and the adaptive dispatcher prepares a
128 KiB-packet router for AllReduce through 4 MiB. Other router calls retain
512 KiB packets. NCCL 2.28.9 is a [separate control](../cpu_router_8l4_20261110/README.md),
never the automatic fallback. The prior [local v2 record](../local_fallback_8l4_20261110/README.md)
remains available; its profile identity cannot select the changed code.
The current ReduceScatter README figure uses a [later calibration](../reducescatter_8l4_20261110_v2/README.md)
with different packet and progress choices; this v3 record remains the
AllReduce/AllGather selection evidence.

## Machine and measurement

- `wxz-l4`: eight NVIDIA L4s, GPUs 0–3 on NUMA node 0 and 4–7 on node 1;
  peer access within each four-GPU group, no cross-group CUDA peer access.
- Intel Xeon CPU reported as `@ 2.20GHz`, CUDA 12.9, driver 580.178.04,
  CUTLASS 4.5.0. BF16, both placements, and 1 MiB–1 GiB doubling complete
  tensor bytes per rank.
- Each placement has 33 workload keys, two project-owned backends, three
  alternating paired repetitions, and ten timed calls per repetition:
  **1,980 checked samples per placement**. Each pair gets five warmups. The
  reported latency is the slowest-rank CUDA-event time; fill and verification
  run outside timing. Min/max bound three run medians, not confidence limits.
- `prepare()` allocates the local staging and router workspaces outside timing.
  The local path and the adaptive dispatcher are eager; neither silently runs
  NCCL collectives. Automatic selection needs an exact profile key and at
  least 3% median gain with non-overlapping run-median ranges.

At 1 MiB AllReduce, the new router wins the paired comparison and is faster
than the earlier, separately timed NCCL control:

| Placement | Local v3 | Router, 128 KiB packets | Router, 512 KiB diagnostic | NCCL control |
| --- | ---: | ---: | ---: | ---: |
| In place | 742.624 µs | **487.936 µs** | 799.232 µs | 587.776 µs |
| Out of place | 717.552 µs | **529.408 µs** | 775.168 µs | 578.560 µs |

The 512 KiB diagnostic is a separate forced-router run with
`--small-ar-packet-bytes 0`, not another paired calibration. The NCCL control
also comes from a separate run. At 2/4 MiB, the paired router AllReduce medians
are 832.512/1484.800 µs in place and 848.384/1471.488 µs out of place.
The 8 MiB and larger AllReduce calls use the original 512 KiB router packets.

At 1 GiB, the paired local/router medians are:

| Placement | AllReduce | AllGather | ReduceScatter |
| --- | --- | --- | --- |
| In place | 1479.323 / **277.644** ms | 183.350 / 181.601 ms | 933.820 / **169.574** ms |
| Out of place | 1443.508 / **276.350** ms | 187.138 / 181.695 ms | 881.901 / **170.279** ms |

Values are local/router. The local AllGather results remain close enough to
the router that the 3% selection margin retains local at 1 GiB in both
placements. Fresh automatic replays selected the router for **32/33** keys in
each placement and local AllGather for the remaining key. A separate fresh
1 GiB in-place AllGather replay selected local with
`dispatcher_router_initialized=0`. All calibration, diagnostic, and replay
invocations passed output checks. The local large-message AR/RS paths remain
negative results and are retained in the plot.

## Files and reproduction

| Files | Contents |
| --- | --- |
| `*_calibration.csv`, `*_samples.csv`, `*.profile` | Paired summaries, every timed call, and exact auto-selection evidence. |
| `*_auto.csv`, `*_auto.log` | Fresh-process automatic choices and correctness. |
| `*_packet_ablation.csv`, `*_packet_ablation.log` | Forced router with the small packet choice disabled. |
| `in_place_1g_ag_auto.*` | Fresh fallback-only 1 GiB AllGather check. |
| `*_calibration.log` | Selection reasons for each paired key. |

From `experimental/lite`, with new output paths:

```bash
make -j2 adaptive-collectives local-collectives-check \
  CUDA_HOME=/usr/local/cuda-12.9 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 \
  NCCL_HOME=build/nccl-2.28.9

env -i HOME="$HOME" PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=build/nccl-2.28.9/lib:/usr/local/cuda-12.9/lib64 \
  ./build/uccl_lite_collectives --gpus 8 --fallback local --op all \
  --in-place --begin 1048576 --end 1073741824 \
  --warmup 5 --iterations 10 --repeats 3 \
  --samples build/new-in-place-samples.csv --calibrate build/new-in-place.profile

# Omit --in-place and choose distinct files for the other placement.
# For selection replay, use --backend auto --profile instead of --calibrate.
python3 scripts/plot_cpu_router_8l4.py  # Requires matplotlib.
```

The [six-panel figure](../../../docs/assets/cpu-router-8l4.svg) plots both
project-owned backends at every size and the separate NCCL control. Rebuild
the exact local profile after changing the implementation, packet settings,
topology, software, or affinity.
