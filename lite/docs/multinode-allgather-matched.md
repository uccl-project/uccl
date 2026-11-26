# AllGather: dual RDMA and matched peer exchange

These explicit experiments test two different hypotheses on two four-L4 hosts.
They do not change the existing automatic policy. GPU numbering is 0–3 within
each node, with NUMA groups `{0,1}` and `{2,3}`. Let each GPU contribute S bytes;
complete AllGather output is N=8S bytes per GPU.

## Two RDMA destinations

`--ag-pipeline sm|ce --ag-dual-rdma 1` sends each input chunk to both its
corresponding remote GPU and that GPU's NUMA mate. There are two independent IB
connections/QPs on the rank's configured rail. Both writes are submitted without
waiting for the first one to finish; this is two RDMA writes, not hardware multicast.

| Source | Remote RDMA destinations | Designated remote forwarder | Remaining targets |
| --- | --- | --- | --- |
| A0 | B0, B1 | B0 | B2, B3 |
| A1 | B1, B0 | B1 | B2, B3 |
| A2 | B2, B3 | B2 | B0, B1 |
| A3 | B3, B2 | B3 | B0, B1 |

B→A is symmetric. Every source also broadcasts its input to all three local
peers. On receipt, the designated forwarder skips its same-NUMA mate, which gets
the duplicate through RDMA. The receiver copies that duplicate from separate
staging to the correct output slot. Each output shard still has exactly one writer.
Both receive channels are waited on, and both send queues are drained before return.

Per node per direction, network payload rises from 4S to 8S. Local peer payload
drops from 24S to 20S, but cross-NUMA payload remains 16S: the removed local copy
is within NUMA. Thus this variant does not itself eliminate the cross-NUMA fan-out.
Primary and duplicate receive channels each have two alternating slots; neither
partner can advance two invocations without the other's progress. Slot offsets
use maximum configured shard size. The existing local scratch allocation is
large enough for the duplicate slots; the input and output layouts do not change.

## Matched recursive doubling

`--ag-pipeline matched-sm` uses SM stores for payload; `--ag-pipeline matched`
uses CUDA copy-engine transfers. Both use GPU-driven ready handshakes.

For each chunk, receive the corresponding remote shard first and write the two
owned shards into standard global-rank output positions. Treat them as a logical
pair; no packing or full-output unpacking is needed.

1. With default `--ag-first-mask 2`, rank r exchanges its two owned shards with
   local peer `r^2`. The pairs are 0↔2 and 1↔3. Each GPU now has four shards.
2. Exchange those four shards with `r^1`, forming pairs 0↔1 and 2↔3. Each GPU
   now has all eight shards.

`--ag-first-mask 1` reverses the matching order. Both orders have 24S total local
payload per node, but cross-first moves only 8S across NUMA, while within-first
moves 16S. The existing ring's corresponding total is 12S. Merely repeating
`r^1` cannot connect all four local GPUs.

Before every phase, **both endpoints publish ready and wait for peer ready**.
A rank becomes ready only after its prior phase's reads and sends complete.
The sender then transmits only to that one peer; it signals data completion and
waits for its incoming data. CE sends are flushed before advancing. SM completion
uses system-ordered GPU semaphore signals. Grid synchronization includes every
CTA in the phase boundaries.

This prevents payloads from different local phases arriving concurrently at a
GPU whose previous phase is still active. A simple loop over peers, even if its
individual copies are sequential, does not provide this guarantee across ranks.
The guarantee covers **P2P payloads**. Small control notifications and the
independent IB lookahead still exist; this is not exclusive ownership of every
transaction on the GPU's PCIe link.

Within a phase, sender-owned and incoming shard intervals are disjoint. At the
next phase they become sources only after data completion. At the next chunk or
invocation, the ready handshake prevents an early peer from overwriting a GPU
still reading its output. Network receives continue to use alternating staging
slots. This harness advances those slots per eager launch; graph replay is not
implemented or validated.

## Why matching may not improve end-to-end latency

The original ring already has one payload ingress and one egress per step,
although they can be different peers. Matching's potential gain over ring is
therefore fewer local stages, same-peer bidirectional transfers, and reduced
cross-NUMA payload, rather than removal of ring fan-in.

The initial matched implementation packed/unpacked the full output. Pilots
rejected that overhead and the final implementation uses output directly.
Ready handshakes and waiting for the remote chunk before local exchange remain
on the critical path. Small messages may be dominated by these dependencies;
better isolated P2P bandwidth does not automatically lower collective latency.

The same idea can inform a future AllReduce experiment: paired local
ReduceScatter (`^1 → ^2`), inter-node exchange, then paired AllGather
(`^2 → ^1`). That would replace six local ring steps with four paired steps while
retaining the same local payload volume. For ReduceScatter, the paired reduction
half alone applies. Those AR/RS algorithms are designs only; this change measures
AllGather and leaves AR/RS selection unchanged.

See [the paired results, negative pilots and reproduction commands](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/ag_dual_matched_2x4l4_20261115/README.md).
