# L4 parameter-sharded application measurements, 2026-11-16

The final cohort uses pretrained **GPT-2-XL**, BF16, fixed **GPU-progress packet
AllGather + existing peer ReduceScatter**. Training rematerializes each block
with activation checkpointing (96 AG / 48 RS calls per full SGD step). Serving
uses prefill plus seven KV decode passes (384 AG calls per HTTP batch).

`final_4l4/` and `final_2x4l4/` retain all per-node JSON and logs from fresh
NCCL → UCCL → UCCL → NCCL processes, five warmups and ten measurements per
case/job. Every plotted cell is the median of two job medians: twenty measured
calls/backend/case. The same backend/configuration is used for all cases;
there is no per-size best implementation or best-run selection. Two-node
layout is **DP2 × shard4**, with twice the one-host global batch. It is not
global eight-way sharding or a strong-scaling comparison.

Each JSON retains complete wall samples, model/collective numerical checks,
loss histories, complete trained-parameter hashes and Torch peak memory.
Two-node files include both hosts. Their intervals already contain a maximum
across nodes; summaries also verify the per-sample maximum across retained
records. `summary_*.json` links the full-parameter comparison reports. Temporary
parameter tensors remain outside the source tree and are removed after checking.

`rematerialization_xl.json` compares a complete SGD update with and without
checkpointing: all **1,475,558,400** trainable parameters and all losses match
exactly. `validation/` retains policy, dispatcher, eager and graph checks on
both hosts. Adapter graph capture is rejected; graph checks validate the
underlying packet executor, not a graph-capable application adapter.

`provenance.json` records checkpoint, source/library hashes, packet settings,
CPU binding and protocol. `sources_packet/` snapshots the measured integration.
The CPU-router and peer kernel sources and automatic dispatch policies are
unchanged. The NCCL baseline constructs no packet executor or workers.

Read [the experiment](../../README.md) for precise timing and adapter scope.

## Controls and rejected work

- `controls/prepared_peer/` retains a complete earlier unbound BF16
  GPT-2-large ABBA cohort using peer AG/RS, without block checkpointing.
  Serving improved only 4.8–5.2% on one host and 8.3–8.9% on two hosts;
  training had no consistent large win. Those samples are not reused as
  controls for the new model/workload.
- `pilots/affinity_restricted_abba` and `four_l4`: MPI singleton initialization
  restricted the entire single-process/four-GPU job to one CPU core. Large
  apparent gains are CPU-budget sensitivity results, not accepted unbound
  NCCL comparisons. The accepted launcher disables binding. All eight early
  JSON files survive a post-job launcher editing error that truncated the last
  stdout log.
- `pilots/global_shard8`: genuine eight-rank global parameter sharding gave
  only 2.3% lower serving latency and 4.6% slower training in a tiny pilot.
- `pilots/replica2_shard4`, `unbound_large_screen`, `unbound_xl_screen`:
  smaller batch/model controls with limited gains.
- `pilots/fp32_screen`, `fp32_two_screen`: FP32 communication / BF16 compute
  trials, rejected for limited gains; `fp32_trial.patch` applies against the
  archived `sources_bf16/` snapshot, not the current packet integration.
- `pilots/packet_screen`, `packet_both_screen`, `packet_both16_screen`:
  packet AG helped, but replacing peer RS with packet RS at 4/16 MiB packet
  sizes regressed training. `packet_rs16_trial.cc` retains that rejected source.
- `pilots/packet_xl_screen` and `packet_xl_two_screen` are short backend/workload
  selection trials (one warmup/five measured calls) for the final fixed model
  and backend. They never enter either figure.

Optional `profile_*.json` are separate caller-stream event diagnostics;
instrumented timings do not enter the figures. Their communication intervals
include readiness waits and host submission gaps, not pure kernel occupancy,
PCIe utilization, or an overlap decomposition.
