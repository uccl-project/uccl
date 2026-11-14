# Eight-L4 ReduceScatter regression recovery

The fixed 1–8 MiB BF16 ReduceScatter schedule now uses an opt-in mapped-host ring. Across both reversed sweeps in GPU-default and explicit SHM modes, it beats the previous implementation and paired NCCL at all four affected sizes. The 34 other size/mode points retain the previous algorithms; none has an aggregate slowdown above 5%.

## Paired results

Complete input bytes per GPU, BF16, out of place. Values are microseconds. The previous column is the pre-recovery `5b9a7f86` schedule remeasured in the same executable. These are fixed implementations, not a per-size best-of curve.

| Mode | MiB/GPU | Previous | Ring | NCCL | Latency reduction vs previous |
| --- | ---: | ---: | ---: | ---: | ---: |
| gpu | 1 | 310.27 | 281.09 | 338.43 | 9.4% |
| gpu | 2 | 543.74 | 464.64 | 571.12 | 14.5% |
| gpu | 4 | 937.22 | 835.58 | 1033.98 | 10.8% |
| gpu | 8 | 1682.94 | 1576.43 | 1943.54 | 6.3% |
| shm | 1 | 307.71 | 281.09 | 340.99 | 8.7% |
| shm | 2 | 546.82 | 464.64 | 580.10 | 15.0% |
| shm | 4 | 927.49 | 833.79 | 1036.26 | 10.1% |
| shm | 8 | 1682.43 | 1574.91 | 1942.02 | 6.4% |

[All 42 points](results.md), [per-sweep candidates](candidates.csv), and [validation](validation.json) retain every result. NCCL timings changed from the earlier morning experiments; only freshly paired controls support the speedups above. Historical measurements remain untouched.

A separate [direct archived-binary comparison](original_binary_control.csv)
also improves all eight target size/mode keys by 6.0–15.2%. It reruns the exact
pre-change executable from the full refresh, verified by SHA-256, against the
new executable. GPU mode runs original then updated; SHM reverses that order.
Each run includes paired NCCL, three repetitions, five warmups and ten timed
calls: 960 additional checked calls and 480 warmups total. This confirms the
improvement independently of the same-executable `previous` selector.
[Validation and original executable identity](original_binary_validation.json).

## Hardware and timing

`wxz-l4`: eight NVIDIA L4 GPUs, two Intel Xeon sockets reported as CPU @ 2.20 GHz, ranks 0–3 on NUMA 0 and 4–7 on NUMA 1. CUDA 12.9, driver 580.178.04, NCCL 2.28.9. Submission CPU 47; four launch workers per NUMA node; ring uses 16 CTAs, 256 threads, and two host banks. See [topology](topology.txt) and the full-refresh hardware record.

Two opposite-order sweeps per mode, 21 powers of two from 1 KiB to 1 GiB, three rotated backend repetitions, five warmups and ten timed calls per repetition. Total: 7,560 checked timed calls and 3,780 checked warmups. Every call changes the signed input and checks the complete output plus guards. Input producers finish before timing. The slowest-GPU CUDA-event interval includes host submission gaps and ends after the output is ready; preparation and validation are excluded. The table takes the median of the two sweep medians, each the median of three repetition medians.

GPU mode leaves NCCL P2P at default; SHM disables P2P explicitly. UCCL uses mapped host memory in both. Fresh transport probes recorded 16 `SHM/direct/direct` channel edges in each mode. These are separate testbed runs, not a P2P-versus-SHM bandwidth claim.

## Implementation and scope

[The ring design](../../../docs/host-ring-reduce-scatter.md) exchanges adjacent-rank BF16 partials with per-CTA system-scope credits. GPU arithmetic and progress eliminate CPU reduction waits. Every hop rounds to BF16; an independent scalar oracle verifies that order. Generic eager/graph paths keep the existing command queue and completion protocol.

`PacketPipelineOptions::hostRingReduceScatter` is disabled by default. Applications must opt in and calibrate exact dtype, size, placement, topology, software and timing scope. Unsupported or unmeasured Auto workloads retain the caller fallback without router allocation. The benchmark uses Force for the fixed measured schedule. No automatic production policy or serving hook is enabled.

CPU packet/worker/interleave/direct-result tuning, CPU loop unrolling, flat GPU reduction, recursive halving, and hierarchical alternatives did not yield a reliable improvement across all four sizes. [WORKLOG.md](WORKLOG.md), raw candidate logs, source snapshots and the reverted CPU trial preserve these negative results. Interrupted runs and the accidentally stale parallel-launch binaries are explicitly excluded.

## Correctness and limitations

[Correctness records](correctness.json) cover the rebuilt library's policy (35),
dispatcher (74), CPU BF16 oracle (860,132), NCCL identity (8), and router-core
checks. Dispatcher CUDA checks and generic eager/graph tests pass in GPU and
SHM modes. The ring test checks every BF16 encoding against an independent
scalar oracle, 270 asynchronous changing-extent calls, three replays of a
140-call graph, output guards, queue wrap, and recovery after rejected calls.
The final local eight-rank run checks 2,097,152 scalar outputs.

The separate full-range AR/AG controls add 7,560 checked timed calls and
3,780 warmups across GPU and SHM modes. They use one sweep per mode/operation,
three rotated repetitions and the same five-warmup/ten-call protocol.
The `current` and `previous` selectors resolve to identical implementations at
all 84 control points, with no aggregate slowdown above 5%.
[All control timings](controls.csv) and [validation](control_validation.json)
are retained. These selector controls share the revised executable; they are
not an independent comparison of old and new generic kernel binaries.

The separate documented four-L4 host, `l41` / `mibura-sky-test-02`, also passes
the ring test (1,048,576 scalar outputs), generic eager test and 1,000 captured
mixed-collective graph test. [Remote logs and executable hashes](four_l4/checks.json)
are retained; these were run on the separate four-GPU host, in addition to the
local four-rank subset checks.

Full eight-GPU Compute Sanitizer instrumentation timed out after 180 seconds
without reporting an error; it is not counted as a pass. A separately bounded
four-GPU memcheck instruments the first four ring dispatches, and an eight-GPU
memcheck instruments the first eight; both pass with zero errors. These smoke
checks do not establish full sanitizer coverage of
queue wrap or graph replay. All those cases pass the native correctness tests.

## Reproduction

Build from the Lite root using external CUDA/NCCL/CUTLASS installations:

```bash
make BUILD=build/rs_recovery_8l4_20261114 CUDA_HOME=/usr/local/cuda-12.9 \
  NCCL_HOME=build/nccl-2.28.9 CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 \
  compact-collectives router-checks adaptive-collectives
python3 scripts/run_rs_recovery.py full
python3 scripts/run_rs_recovery.py controls
python3 scripts/summarize_rs_recovery.py --controls
python3 benchmarks/data/overview_8l4_20261114/build.py
python3 benchmarks/data/overview_8l4_20261114/plot.py --op reduce_scatter --preview
```

The runner refuses existing raw run names; use a new data directory for another run. It also rejects occupied GPUs and aborts its own job if foreign GPU users appear. Source hashes, executable hashes, complete commands, environments and exit statuses are retained. Generated binaries and traces live under `build/`. AR/AG figures retain their earlier full-refresh measurements; the separate fresh controls validate behavior without replacing those historical curves.
