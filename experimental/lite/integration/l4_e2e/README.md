# UCCL Lite parameter-sharded serving and training on L4

This experiment uses pretrained **GPT-2-XL**, BF16, and one fixed backend:
**GPU-progress packet AllGather + existing prepared peer ReduceScatter**.
Both workloads are explicit application adapters, not stock SGLang, FSDP,
or an NCCL drop-in. The packet executor is explicitly selected; no automatic
routing policy, collective kernel or CPU-router core is changed.

<!-- BEGIN_RESULTS -->
Both independent UCCL Lite jobs beat both fresh default NCCL controls in
**every serving and training case**. No pilot sample enters these results.

### Four L4 GPUs

| Operation / global shape | NCCL (ms) | UCCL Lite (ms) | Latency reduction | Throughput ratio |
| --- | ---: | ---: | ---: | ---: |
| serving 8 × 32 × 8 | 1209.15 | 980.53 | 18.9% | 1.233× |
| serving 8 × 128 × 8 | 1223.17 | 993.14 | 18.8% | 1.232× |
| serving 16 × 256 × 8 | 1298.92 | 1063.83 | 18.1% | 1.221× |
| training 8 × 32 | 495.27 | 463.14 | 6.5% | 1.069× |
| training 8 × 64 | 504.36 | 472.23 | 6.4% | 1.068× |
| training 16 × 128 | 577.90 | 544.08 | 5.9% | 1.062× |

### Two four-L4 hosts (DP2 × shard4)

| Operation / global shape | NCCL (ms) | UCCL Lite (ms) | Latency reduction | Throughput ratio |
| --- | ---: | ---: | ---: | ---: |
| serving 16 × 32 × 8 | 1223.85 | 983.95 | 19.6% | 1.244× |
| serving 16 × 128 × 8 | 1237.71 | 996.82 | 19.5% | 1.242× |
| serving 32 × 256 × 8 | 1313.85 | 1067.37 | 18.8% | 1.231× |
| training 16 × 32 | 561.45 | 508.12 | 9.5% | 1.105× |
| training 16 × 64 | 570.62 | 517.07 | 9.4% | 1.104× |
| training 32 × 128 | 645.47 | 589.80 | 8.6% | 1.094× |

Serving shapes are global batch × input × output tokens; training shapes
are global batch × sequence length. Values are medians of two independent
job medians. Throughput ratios are inverse latency at fixed batch, not a
saturated-service QPS test. Full repetitions and job ranges are retained.
<!-- END_RESULTS -->

## Workload and messages

GPT-2-XL has 48 blocks, width 1600, 25 attention heads, FFN width 6400 and
vocabulary 50,257. The checkpoint is
`openai-community/gpt2-xl@15ea56dee5df4983c59b2538573817e1667135e2`.
Every block contains 30,740,800 trainable parameters: **61,481,600 bytes =
58.6334 MiB** complete AG output / complete RS input per GPU in BF16. Each
rank contributes/receives 14.6584 MiB. All block parameters are used; causal
mask buffers are excluded. There are **1,475,558,400 trainable parameters**;
embeddings and the final norm are frozen replicas.

Training checkpoints each block, discarding materialized weights/activations
and recomputing the block during backward. This produces **96 AG + 48 RS**
calls per complete SGD step. It is actual rematerialized computation, not
extra dummy collectives. Both backends use the same checkpointing, precision,
inputs, placement, buffers, SDPA, GELU, eager execution and SGD. A separate
complete SGD comparison with/without checkpointing has identical losses and
**zero difference across every trainable parameter after the update**.

Serving returns eight tokens using one prefill plus seven KV decode forwards:
**384 AG calls per HTTP batch**. Block checkpointing is inactive under inference
`no_grad`. Prompt lengths are 32/128/256; training lengths are 32/64/128.
One-host global batches are 8/8/16. Two-host batches are 16/16/32, preserving
per-GPU work (**weak scaling**, not a strong-scaling speedup).

The model fits one L4. Deliberate parameter sharding provides a controlled,
communication-heavy comparison, not evidence that this is its best deployment.
These are fixed generated-token training batches, not convergence/model-quality
measurements. The earlier smaller-model/no-checkpoint cohort is a separate
control, not a reused denominator for the new workload.

## Backend and placement

One Python/MPI process owns GPUs 0–3 per host. A four-rank group shards each
block equally. The two-node layout is **DP2 × shard4**, with matching owned
gradient shards summed by two-rank NCCL AllReduce after each local RS. Both
backends use the same DP synchronization. Serving has two coordinated local
sharded replicas; it does not exchange model weights between hosts.

- UCCL Lite AG uses the existing BF16 `PacketPipeline`: NUMA-local mapped host
  packets, GPU progress, per-source readiness, fixed **4-MiB packets**, two
  slots, 248 transport blocks (runtime occupancy capped), 256 threads and an
  eight-worker/node cap. The log reports **zero CPU routing workers** in this
  GPU-progress mode. RS uses the existing `PeerCollectives` BF16 executor.
- NCCL uses default four-rank AG/RS, plus the same two-rank DP AllReduce for
  training on two hosts. No algorithm, protocol, channel-count, P2P-level,
  transport or copy-engine override is supplied. This is not a claim against
  every tuned NCCL configuration.
- The bridge validates full four-GPU P2P, BF16, per-device placement,
  out-of-place buffers, exact prepared extent and eager execution. The packet
  capacity is explicitly 64 MiB complete. Idle-stream preparation allocates
  packet state outside timing. The NCCL baseline constructs no packet executor,
  buffers or workers. Unsupported packet/global-eight-rank combinations reject.
- No backend or variant is selected per batch size. The current dispatcher
  profiles are unchanged; these measurements do not auto-enable a route based
  on GPU count. CPU-router core remains free of NCCL/MSCCL++/application hooks.

This adapter does not implement FSDP prefetch/overlap, production scheduling,
streaming responses or application graph capture. The underlying packet
executor's eager/graph tests are separate from the eager-only adapter.
A retained **globally eight-way-sharded GPT-2-large pilot** showed only 2.3%
lower serving latency and 4.6% slower training; its result must not be assigned
to the new replicated layout or to the separate FP32 multi-node kernels.

## Timing and repetitions

Serving measures complete loopback HTTP JSON submission, server parsing,
input upload, prefill, KV decoding, output download and complete JSON response
reception. Requests contain token IDs; tokenization/detokenization are outside
timing. EOS is ignored to return exactly eight tokens. This is one batched
request per replica, not independent concurrent client connections or a
saturated/streaming load test. Both clients submit the same global batch;
each node executes its half. The two-node result is the maximum complete
client interval per repetition.

Training includes gradient clearing, forward, FP32 cross-entropy, checkpoint
recomputation, backward, AG/RS, DP synchronization, SGD (`lr=.001`,
`foreach=True`), host submission gaps and completion on every GPU. Each rank's
mean loss is weighted by 1/(total GPU count). Four fixed token batches rotate
identically across trials. Batch preparation, optimizer construction, restoring
parameters between cases, loss reporting and parameter verification are outside
timing. Initial model/library loading and communication preparation are excluded
for both backends. There is no overlap attribution hidden in the timing.

Formal order is **NCCL → UCCL → UCCL → NCCL**, fresh processes for every
serving/training job. Every case has five full-shape warmups plus **ten timed
calls/job**. A plotted cell is the median of two job medians: **twenty measured
calls/backend/case**. Whiskers span independent job medians, not confidence
intervals. Every selected case and both jobs are retained; pilots and
instrumented diagnostics supply no formal sample. MPI barriers precede timing;
per-iteration max reductions run after the timer and are excluded for both.

CPU binding is disabled: `OMPI_MCA_hwloc_base_binding_policy=none` for singleton
initialization, `--bind-to none` for the two-node launcher. All 48 CPUs remain
available per process; Torch/OpenMP/MKL use one host thread. An early singleton
run inherited Open MPI's one-core binding and produced inflated apparent gains;
those raw records remain separate under rejected CPU-budget controls.

## Hardware and software

- `mibura-sky-test-02` alone, or with `mibura-sky-test-01` (`10.10.55.1`):
  four NVIDIA L4/host, **23,034 MiB/GPU** (92,136 MiB/host), full local P2P,
  no NVLink. GPUs 0/1 and 2/3 are on their respective NUMA/NIC groups; two
  physical IB rails per host. Xeon Silver 4410Y, two NUMA nodes, 48 logical CPUs.
- Driver 595.58.03 on node 0, 580.178.04 on node 1; normal dynamic clocks.
  CUDA compiler 12.9, sm_89; PyTorch 2.7.1+cu128, Transformers 4.53.3,
  preloaded **NCCL 2.28.7** on both backends/hosts, Open MPI 4.1.7rc1.
- Checkpoint, source, bridge/archive and NCCL hashes are in the manifest and
  environment records. Torch allocated-memory peaks exclude NCCL/pinned-host
  allocations. Local calibration is separate from historical eight-L4 figures.

## Validation and controls

Every process checks complete signed, destination-dependent AG/RS tensors at
4, 8, 16, 32, 64 MiB and the exact 58.6334-MiB model extent, outside timing.
AG preserves all bits; RS uses a complete FP32-sum/BF16-round oracle. Every
process compares all reference logits and four-token KV decode against the
unsharded Transformers checkpoint. The summarizer checks generated token hashes,
complete loss histories, configurations and the actual **2 AG / 1 RS** training
communication count. Full parameter reports compare every trainable element
across alternate phases and both nodes with fixed `atol=.002, rtol=.02`; actual
observed errors are retained rather than inferred from hashes/probes.

Both hosts pass router policy (35 checks), host dispatcher (74 checks), CUDA
dispatcher capture/recovery checks, 732 eager mixed producer/consumer cases,
and 1000 captured mixed packet collectives with five replays/ring wrap. These
validate the existing packet executor, not adapter graph support. The separate
rematerialization check compares a complete SGD update with/without checkpointing
and matches all 1,475,558,400 parameters exactly.

[The data index](data/l4_20261116/README.md) retains the complete prior peer-only
ABBA cohort and all rejected pilots: CPU-affinity restrictions, global eight-way
sharding, smaller/larger models, FP32 communication, and slower packet RS at
4/16-MiB packet sizes. Historical results are not validation of a subsequent
source revision or an automatic routing policy. This is an experimental eager
application integration, not production serving/training support.

## Separate communication diagnostic

Four-GPU caller-stream event diagnostics use three warmups, three uninstrumented
calls and three instrumented calls on the first-case shapes (direct generation,
not the HTTP client). Ranges below span the four GPUs' median intervals; values
sum AG/RS intervals **within one caller stream**, not across concurrent GPUs.

| Workload/backend | AG interval (ms) | RS interval (ms) | Communication / total stream |
| --- | ---: | ---: | ---: |
| Generation, NCCL | 1088–1098 | — | 89.8–90.6% |
| Generation, packet AG | 846–847 | — | 86.6–86.8% |
| Training, NCCL | 272–273 | 135–136 | 82.7–83.1% |
| Training, packet AG + peer RS | 221–225 | 142–148 | 80.5–81.0% |

The observed gain comes from AG; the peer RS intervals are slightly slower,
which limits training benefit. These intervals include producer readiness and
host submission gaps. The remainder includes compute and its submission gaps;
this is not pure kernel occupancy, PCIe utilization, or an overlap decomposition.
The diagnostic uses fixed generated token IDs, rather than the HTTP prompts.
Instrumentation changes timing and supplies no sample to the formal figures.

## Reproduction

Use external Python/checkpoint dependencies and the Makefile; binaries remain
under `build/`:

```bash
make -C integration/l4_e2e -j4
export E2E_PYTHON=/home/zhongjie/flux/.venv/bin/python
export E2E_CHECKPOINT=/home/zhongjie/.cache/uccl-lite-gpt2-xl
export E2E_NCCL=/home/zhongjie/third_party/nccl/build/lib/libnccl.so.2
export E2E_ORDER='nccl packet packet nccl'

E2E_OUT=build/l4_e2e/packet_final_4l4 integration/l4_e2e/run.sh \
  --checkpoint-blocks --warmup 5 --repeats 10 --save-parameters \
  --cases 8:32:8 8:128:8 16:256:8 --train-cases 8:32 8:64 16:128
E2E_NODES=2 E2E_OUT=build/l4_e2e/packet_final_2x4l4 integration/l4_e2e/run.sh \
  --checkpoint-blocks --replica-nodes --warmup 5 --repeats 10 --save-parameters \
  --cases 16:32:8 16:128:8 32:256:8 --train-cases 16:32 16:64 32:128
```

Both hosts need the same paths/binary. The launcher explicitly identifies these
hosts and IB interfaces. Checkpoints/environments are external; the second
host's dependencies live on its SSD, not its nearly full root filesystem.
Copy both nodes' JSON into the matching final dataset directory. Output files
must be fresh. Full snapshots default to an isolated `/dev/shm` directory;
check only after timing and retain reports before deleting tensors:

```bash
python3 integration/l4_e2e/validate_parameters.py --device cuda \
  --prefix packet_final_4l4 --out build/l4_e2e/parameters_4l4_node0.json
# Use packet_final_2x4l4 on both hosts for the two-node reports.
python3 integration/l4_e2e/check_rematerialization.py \
  --checkpoint "$E2E_CHECKPOINT" --out build/l4_e2e/rematerialization_xl.json
python3 integration/l4_e2e/summarize.py \
  --data integration/l4_e2e/data/l4_20261116/final_4l4 \
  --out integration/l4_e2e/data/l4_20261116/summary_4l4.json
python3 integration/l4_e2e/plot.py \
  --summary integration/l4_e2e/data/l4_20261116/summary_4l4.json --out docs/assets
# Repeat summary/plot for final_2x4l4 / summary_2x4l4.json.
```

Separate `profile_communication.py --backend nccl|packet --checkpoint-blocks`
runs caller-stream CUDA-event diagnostics. Their fractions include producer
readiness and host submission gaps and are not pure kernel/PCIe utilization.
Instrumentation changes timing; those samples never enter the figures.

The final 24 per-node records have zero reference-logit difference and matching
greedy/KV decode tokens. Complete loss-history maxima differ by 0.00773 /
0.00881 (one/two hosts). All **27 full-parameter comparisons** pass; maximum
observed parameter difference is **0.0009766**, relative L2 **1.59e-6**. Both
DP replicas have identical complete trained-parameter hashes in every phase
and case. These check numerical execution, not convergence/model quality.
