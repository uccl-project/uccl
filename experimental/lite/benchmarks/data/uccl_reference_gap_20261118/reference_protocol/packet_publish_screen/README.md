# Packet FIFO publication and coalesced flush experiments

Two four-L4 hosts, FP32 AllGather and FP32 SUM AllReduce/ReduceScatter, fixed
1-GiB allocations, 1–128 KiB, both layouts, three repetitions with counterbalanced
variant order. 20 warmups, AG 30 / AR 10 / RS 10 timed calls. Each point is the median of
three maximum-rank per-call means including GPU and internal send retirement.
The baseline is one fixed output-overlap/ordered candidate; no per-size winner
curve is constructed and no native NCCL implementation is executed.

Both candidates and the FIFO-publication near-wrap build pass 198 mixed ABI
calls/rank over two communicator lifetimes, including delayed 128-KiB AG, pointer
alignment/layout variation, stream alternation and allocation recycling. The
near-wrap build confirms one packet epoch reset per rank per lifetime. All 54
screen jobs pass full-output, exact route and native-absence checks.

The coalesced candidate uses MSCCL++ TriggerData|TriggerSync for the final AG send
and waits for that exact FIFO ticket at the existing completion boundary. The
proxy handles data before sync, preserving the outgoing write and retirement.
It retains the existing producer fences. Its improvement is modest/layout-dependent;
it is not yet selected in the next base.

The FIFO-publication candidate retains the all-producer CTA barrier and MSCCL++
FIFO system-release publication, whose CPU consumer acquires the commit before
posting RDMA. It omits the additional per-producer system fence. The ordering
rationale is an inference from this exact source path and the transitive
synchronization relations in the [NVIDIA PTX memory model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model).
Dependency source hashes are retained. The final outgoing completion ticket, bank
reuse rules, reader synchronization and wrap reset are unchanged. This measured
variant consistently improves AG 32/64/128 KiB and AR 64 KiB, but does not close
AG 128 KiB against the original reference near 33 us. It requires combination and
full-range qualification, not an admission based on this focused screen.

| Variant | Operation / bytes | out (us) | in (us) |
| --- | --- | ---: | ---: |
| uccl_output_ordered | AG 32 KiB | 26.885 | 26.310 |
| uccl_coalesced_packet | AG 32 KiB | 26.077 | 26.289 |
| uccl_fifo_packet | AG 32 KiB | 25.241 | 25.098 |
| uccl_output_ordered | AG 64 KiB | 29.904 | 29.704 |
| uccl_coalesced_packet | AG 64 KiB | 29.460 | 28.806 |
| uccl_fifo_packet | AG 64 KiB | 27.656 | 27.848 |
| uccl_output_ordered | AG 128 KiB | 36.359 | 36.705 |
| uccl_coalesced_packet | AG 128 KiB | 36.398 | 35.422 |
| uccl_fifo_packet | AG 128 KiB | 34.209 | 35.016 |
| uccl_output_ordered | AR 64 KiB | 45.108 | 45.393 |
| uccl_coalesced_packet | AR 64 KiB | 46.509 | 44.667 |
| uccl_fifo_packet | AR 64 KiB | 43.736 | 43.446 |

All other rows and repetitions are retained in summary.csv and manifest.json.
The prior full paired result remains 25/126 slower rows until a complete new
fixed library is validated. No default release has been promoted. The independent
512-KiB streamed RS screen is separately promising and has not yet been combined
with this publication change.
