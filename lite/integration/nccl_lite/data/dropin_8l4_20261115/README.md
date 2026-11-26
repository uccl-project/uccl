# Eight-L4 NCCL drop-in validation, 2026-11-15

The final NCCL ABI interposer runs stock DDP, stock FSDP and SGLang without
an application communication hook or an extra FSDP reduce-scatter process
group. It selects 26 calibrated BF16 shapes and delegates other calls to the
same real NCCL library. **Every measured latency metric passed the 5%
regression gate.** This applies to the workloads and configuration below;
it is not a performance guarantee for unmeasured applications.

## Full training steps

| Mode | NCCL (ms) | Drop-in (ms) | Latency change |
| --- | ---: | ---: | ---: |
| DDP | 470.298 | 400.760 | −14.79% |
| Stock FSDP | 699.435 | 624.826 | −10.67% |

The randomly initialized causal transformer has 637,847,552 parameters,
12 blocks, width 2,048, vocabulary 8,192, BF16 parameters/gradients and FP32
cross-entropy. Each GPU processes one 128-token sequence per step; four
generated token batches rotate with the same seeds in every job. SGD uses
learning rate 0.001. DDP uses 128-MiB buckets and gradient bucket views.
FSDP uses its default hook, normal bound process-group initialization and
normal `summon_full_params`; there is no separate RS group/verification stream.

Timing includes gradient clearing, forward, loss, backward, communication,
SGD, host submission and final CUDA completion. Each sample is the maximum
rank wall time. Model construction, batch generation, pre-timing barriers,
initial transport preparation and numerical verification are excluded.
Each fresh job has three warmups and 20 timed steps; the ABBA order supplies
two jobs and **40 timed steps per backend/mode**. Reported values are medians
of job medians. DDP's independent job medians are 471.447/469.149 ms (NCCL)
and 395.578/405.942 ms (drop-in). FSDP's are 700.091/698.779 ms and
624.561/625.091 ms. No best-run selection is used.

All recorded parameter samples agree across ranks and backends. The maximum
complete loss-history differences are 0.00103760 (DDP) and 0.00103664 (FSDP).
Parameter verification samples eight elements per parameter; it does not
compare every trained parameter or establish training convergence.
[Raw samples](training/samples.jsonl), [summary](training/summary.json), and
[commands, environment and hashes](training/manifest.json) retain the evidence.
Each drop-in job's log records nonzero actual packet AR or AG/RS counts.

## SGLang serving

| Requests × input × output tokens | NCCL whole batch (s) | Drop-in (s) | Latency change |
| --- | ---: | ---: | ---: |
| 8 × 1024 × 1 | 3.10662 | 3.10852 | +0.06% |
| 32 × 2048 × 1 | 23.60588 | 19.07899 | −19.18% |
| 8 × 64 × 32 | 4.99765 | 5.01441 | +0.34% |

The pretrained GLM-4.5-Air-FP8 checkpoint is
`zai-org/GLM-4.5-Air-FP8@f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc`.
SGLang uses TP8/DP8/EP8, BF16 collectives, Triton MoE/attention, eager execution,
disabled radix cache/custom all-reduce, context length 4,096, at most 32 running
requests, static memory fraction 0.87, and seed 12345. The requested global
prefill chunk of 8,192 becomes 1,024 per DP rank in SGLang's parsed configuration.
Torch/OpenMP/MKL use one host thread and SGLang CPU affinity is enabled for
both backends. No fusion hook is loaded.

The client submits each batch concurrently to the loopback streaming HTTP
endpoint. Whole-batch time ends when the final response finishes and includes
queueing, inference and streaming. Prompts and output-length limits are identical
between backends. Model loading and request warmups are excluded. Four fresh
servers run NCCL → drop-in → drop-in → NCCL; case order reverses each phase.
Each case has one full-shape warmup and three timed repeats per server:
**six timed batches per backend/case**, 36 total. Values are medians of two
independent job medians, with no per-case best-run selection.

The two drop-in job medians for long prefill are 19.179/18.979 s; NCCL's are
23.540/23.672 s. All measured whole-batch, last-TTFT, request-median TTFT and
request-median TPOT values pass the 5% gate, including when comparing the
slower drop-in job median to the faster NCCL job median. Decode request-median
TPOT is 146.644 ms versus NCCL's 145.680 ms (+0.66%).

**Output comparison has a limit:** both one-token prefill cases have identical
output-text hashes across all repeats/backends. Decode has repeated-run
variability in both backends; they share the dominant text hash but have
different additional hashes. These runs therefore do not establish identical
decode text or model-quality equivalence. The collective tests below check
the transport/reduction outputs independently. The hashes cover returned text,
not token IDs; all raw hash observations are retained.

[Serving samples](serving/samples.jsonl), [server commands/environment](serving/manifest.json)
and the [combined summary](summary.json) include every repetition and metric.

## Hardware, software and calibrated selection

- `wxz-l4`: eight NVIDIA L4 GPUs, four per NUMA node, partial GPU P2P,
  PCIe Gen4, no NVLink; 23,034 MiB reported GPU memory. The exposed CPU model
  is `Intel(R) Xeon(R) CPU @ 2.20GHz`, 96 logical CPUs on two NUMA nodes.
- Driver 580.178.04; CUDA compiler 13.0.88, runtime 13.0; target `sm_89`;
  PyTorch 2.9.1+cu128; runtime NCCL **2.28.9**. Torch's compiled NCCL version is
  2.27.5, but the requested 2.28.9 library is explicitly preloaded and the
  training jobs verify exactly one real NCCL mapping.
- Both controls and drop-in set `NCCL_NET=Socket`, `NCCL_NET_PLUGIN=none`,
  `NCCL_PROFILER_PLUGIN=none`. The inherited gIB transport did not initialize
  on this host; the auto-loaded GPUViz profiler aborted a native-NCCL
  communicator-recreation diagnostic. Algorithm, protocol, channel, cuMem
  and NVLS settings are not tuned by the drop-in launcher.
- Normal dynamic clocks. The existing integration lock serializes all GPU
  jobs. Hardware UUIDs, topology, CPU layout/affinity, environment and binary
  hashes are in the [calibration manifest](calibration/manifest.json).

Calibration measures 36 exact operation/count pairs, each out-of-place and
in-place at 2/4/8/16-byte alignment. Every fresh ABBA job uses three repeats
of three warmups plus eight timed calls per combination. CUDA event time
includes the caller-stream enqueue interval, staging copies and completion;
each repeat uses the maximum rank median. A shape is accepted only if the
slower packet job beats the faster NCCL control by at least 5% in **every**
combination. The fixed packet configuration is not swept per size.
[All accepted and rejected points](calibration/summary.json) and raw samples
remain available; 26 pass and ten fall back. Complete AG output / RS input is
`count × 8 × 2` bytes; AR size is `count × 2` bytes.

[profile.tsv](profile.tsv) is a local profile, bound to these identities and
the exact compiled library. It does not enable routing on another eight-GPU
machine or after a binary/environment change. The candidate profile retained
under `calibration/` is explicitly marked and rejected by the deployment
launcher. [Source hashes](source_manifest.json) identify the final source;
compiled binaries remain under `build/`.

## Correctness and retained negative result

- [36 full-buffer signed/index-dependent cases](checks/affinity_numerics.json),
  mixed rank layouts, four alignment classes and alternating CUDA streams;
  every rank's AR helper/packet-worker CPU affinity is checked.
- [Mixed asynchronous operations and direct eager-to-graph transitions](checks/final_check_bridge.log),
  in-place buffers, non-BF16/non-SUM fallback, nested groups and repeated
  native graph replay.
- [Eight-rank](checks/fallback8.log) and [four-rank](checks/fallback4.log)
  ProcessGroupNCCL fallback, split communicators and grouped P2P; no packet
  buffers/workers initialized on these fallback paths.
- [Existing eight-process packet check](checks/legacy_packet_check.log),
  [policy](checks/router_policy_final.log), [dispatcher](checks/dispatcher_final.log),
  [dispatcher CUDA](checks/dispatcher_cuda_final.log), and the documented
  four-GPU `0,1,4,5` topology: [732 eager cases](checks/router_eager4_final.log)
  and [1,000 captured mixed collectives](checks/router_graph4_final.log).

The [rejected private-stream prototype](rejected/stock_smoke.jsonl) measured
1,475.47 ms DDP and 1,296.41 ms FSDP in unpaired two-warmup/three-step smoke
jobs. Microbenchmark wins were insufficient: extra stream handoffs regressed
the application. The final implementation enqueues on the caller's stream
and adds an event dependency only on stream changes. A separate helper-affinity
bug moved the AR helper onto the scheduler CPUs; its [observed affinity](rejected/helper_affinity_before.json)
is retained, and final checks verify the dedicated helper CPU. None of these
prototype timings are substituted into the release result tables.

The historical Testbed 1/4 figures remain measurements of their original
configurations. In particular, historical adapted-FSDP numbers are not stock
FSDP validation, and the PRO 6000 application adapter is not this NCCL drop-in.

See [DROPIN.md](../../DROPIN.md) for build, launch and reproduction commands.
