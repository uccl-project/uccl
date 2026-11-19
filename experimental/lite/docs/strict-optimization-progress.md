> Scope correction (2026-11-18): the completed result below is relative to
> native NCCL with a 5% tolerance. It does **not** satisfy the active requirement
> of zero regression against README UCCL Lite. See
> [the active reference audit](../benchmarks/data/uccl_reference_gap_20261118/README.md).

# Strict UCCL optimization result

The scoped goal is verified: **126/126 FP32 AG/SUM RS/SUM AR size/layout rows
pass the established 5% per-row gate**, using unchanged MPI nccl-tests on two
four-NVIDIA-L4 hosts, 21 powers of two from 1 KiB through 1 GiB, both layouts.
No native NCCL library loads or fallback calls occur in candidate processes.
Native NCCL is an independent baseline only. This does not claim zero absolute
slowdown or full NCCL API coverage.

See the [final report and completion audit](../benchmarks/data/strict_release_2x4l4_20261117/README.md).
The admitted result contains two independent ABBA rounds per operation, 10
warmups/50 timed iterations, four job means/backend. Worst candidate/native
ratios are AG 1.04125, RS 1.03403 and AR 1.04712. The second AR round was fully
repeated under a pre-recorded decision because a controller interruption lost
one process exit status. Original logs remain; no row was selected by timing.

## Retained implementation

- Strict NCCL ABI ownership with no native NCCL dependency, forwarding or
  communicator. MSCCL++ provides bootstrap, registration, algorithms runtime,
  MemoryChannel, PortChannel, semaphores, IB proxy and LL16Packet.
- Small LL protocols use separate local/network packet regions, alternating
  banks, generation reset and bounded completion retirement. Serial peer ready
  notifications run in independent lanes.
- Medium AG uses vectorized staging/four CTAs. AR streams sums directly to peer
  scratch and writes output while gathering. Actual 16-byte-aligned pointers
  select vector IO; other legal float-aligned pointers retain safe accesses.
  RS streams both standard-layout halves without a complete packed input.
- Medium banks are 2-MiB aligned and bounded to their reserved 8-MiB capacity.
  AR uses them at 8 MiB. Short final ownership chunks use common piece boundaries.
- AG/AR at 16 MiB and above, and RS at 8 MiB and above, use explicit CE graphs
  with direct caller input/RS output. No local CPU CUDA-copy worker is required.
- LL and 8-MiB AR retire completions every 32 calls, within the configured CQ
  bound. Input/output reuse is safe after stream completion. Synchronization,
  finalize and destruction drain outstanding internal transport work.

## Final evidence

The current source hashes and shared-library SHA match the tested artifacts.
The SONAME is libnccl.so.2 and dependencies contain no native NCCL library.
The final report preserves all raw results, hardware/software identities and
commands. Reproduction instructions are in [MSCCLPP.md](../integration/nccl_lite/MSCCLPP.md).

Validation includes the 126-call/rank mixed NCCL ABI over two lifetimes,
168-call/rank non-power-of-two/alignment/recycling core check, 604-call/rank
asynchronous queue/lifetime stress, forced generation-wrap ABI at 1-GiB capacity,
1101-iteration small-message checks, and 30 standalone AR/AG/RS samples each.
Both affected standalone CUDA targets compile. No sanitizer or network-fault
recovery certification is claimed.

## Historical evidence

[RS/LL and medium-path experiments](../benchmarks/data/strict_ll_rs_graph_2x4l4_20261117/README.md)
preserve failed variants, correctness/build failures, diagnostic captures and
outliers. A 90-row medium-only pass did not establish the 1-GiB result: nccl-tests
rotates addresses across its maximum allocation. All admitted performance data
uses the full workload. Earlier 91-, 53-, 15-, and 4-row failures remain recorded;
historical curves do not validate changed source. No best-per-size envelope
was assembled.

Remote cleanup removed only completed binary replicas whose SHA-256 matched
preserved local copies. Source, logs, measurements and negative results remain.
