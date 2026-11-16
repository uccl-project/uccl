# Superseded prepared-peer control cohort

These are the earlier GPT-2-large, no-checkpoint control results. They are
not the final packet/GPT-2-XL figures.

# L4 application measurements, 2026-11-16

The accepted figures use `final_4l4/` and `final_2x4l4/`, with independent
NCCL → UCCL → UCCL → NCCL processes, five warmups and twenty measurements
per case/job. Each JSON retains complete wall-clock samples, numerical oracles,
reference checks, loss histories, full-parameter hashes and Torch peak memory.
Two-node JSON includes both nodes; reported intervals already contain a maximum
across nodes, outside the measured interval. Summaries additionally take the
per-sample maximum across retained node records.

Read [the experiment](../../README.md) for scope and limitations. These are
experimental weight-sharded GPT-2-large endpoints and SGD steps, not stock
SGLang/FSDP. The two-node layout uses DP2 × shard4 and doubles global batch
relative to four GPUs. It is not globally eight-way-sharded or a strong-scaling
comparison. Full parameter reports compare every one of 708,387,840 trainable
parameters at each of three shapes, for each alternate phase against phase 0.
Snapshots themselves are temporary external files, excluded from this dataset.

`provenance.json` identifies sources, libraries, rejected controls and the CPU
binding correction. `environment-node*.txt` records physical hardware, driver,
topology, compiler and hashes. The accepted single-node binary was rebuilt from
the identical BF16 sources after rejecting the FP32 trial; its build hash and
the earlier two-node build hash are recorded separately. `sources_bf16/` keeps
the exact measured adapter source. No collective kernel or routing policy was
changed by the accepted experiment.

`pilots/` never contributes a formal sample:

- `four_l4` and `affinity_restricted_abba`: MPI singleton initialization bound
  the one-process/four-GPU job to one CPU core. These are CPU-budget sensitivity
  results, not an accepted unbound NCCL comparison. The first ABBA launcher was
  edited during execution, causing an error after its final Python process
  finished and truncating that final stdout log; all eight JSON records remain.
- `global_shard8`: genuinely global eight-rank parameter sharding. Serving
  improved only 2.3%, training regressed 4.6% in the tiny pilot.
- `replica2_shard4`: global batch 8/four output tokens, small serving gain and
  no training benefit. The final weak-scaling workload uses twice the one-host
  global batch; it retains all three selected cases.
- `unbound_large_screen`, `unbound_xl_screen`: corrected CPU affinity and
  larger BF16 model probes, one warmup/three samples per backend.
- `fp32_screen`, `fp32_two_screen`: FP32 parameter/gradient communication with
  BF16 compute, one warmup/three samples. The staged FP32 RS was rejected for
  lack of useful gain; `fp32_trial.patch` reproduces this temporary extension
  against `sources_bf16/`. Its compiled libraries are not retained in source.

`profile_*.json`, when present, are separate four-GPU caller-stream CUDA-event
diagnostics. Their instrumented timings do not enter either plotted figure.
The communication fraction includes readiness waits and host submission gaps;
it is not pure kernel occupancy, PCIe utilization, or an overlap decomposition.
