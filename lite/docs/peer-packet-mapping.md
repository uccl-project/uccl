# Topology-derived peer-packet rank mapping

The retained FP32 packet algorithm still has four execution ranks in two pairs.
The runtime adapter now derives those pairs from physical NUMA placement rather
than requiring the caller to assign logical ranks 0/1 and 2/3 to them.
`pairedNumaRankOrder(topology)` returns execution-rank → caller-rank order,
grouping by discovered NUMA node and preserving caller order within each pair.
For physical node IDs `[1,0,1,0]`, it returns `[1,3,0,2]`. Unknown NUMA placement,
a 3+1 split, a single node or another participant count remains unsupported.
Peer/mapped-host, dtype, exact extent and alignment checks still apply.

Every rank agrees on ordered device UUIDs, CUDA ordinals, permutation, physical
NUMA nodes and capacity before initializing transport. The underlying executor
uses the remapped rank and the caller's actual device. Shared host storage
accepts an explicit device and physical node IDs; node masks handle IDs spanning
more than one machine word. The CPU-copy helper likewise receives a physical
node separately from its logical protocol group. Existing network callers keep
their original default placement contract.

An identity permutation uses the original direct buffers. Other permutations
allocate private input/output staging during prepare. AG writes execution-order
staging and copies each shard to its logical output position. RS first rearranges
logical input shards, so execution rank `w` reduces the caller shard `order[w]`.
These copies preserve all bits. AR uses the existing FP32 reduction tree;
reassociation is permitted under the numerical checker, not a promise of bitwise
agreement with every possible source order.

An adapter completion event covers the final permutation as well as transport.
It orders subsequent calls on different streams, including AG→RS→AR producer/
consumer chains. The caller's legal AG/RS aliases are interpreted in logical
rank space. There are no per-enqueue allocations or routing decisions. External
capture remains unsupported for this candidate.

The planner admits this as a candidate, then the normal exact-request trials
measure its complete execution, including permutations. Historical timings and
the presence of four GPUs cannot qualify it. `LocalSelection` exposes the rank
order alongside correctness flags and samples.

## Validation

`make peer-packet-mapping-check` builds a direct-registry client accepting
`quick|full`, with one process per GPU. Its independent CPU oracle checks signed
FP32 reductions, bitwise AG edge values, mixed aliases/alignment and complete
allocations/guards over two lifetimes. `full` reaches 64 MiB complete AG/RS
extents; `quick` covers 64/4096/65536 elements. Additional queued AG→RS→AR chains
test post-permutation stream ordering. Disagreeing mappings reject collectively
before transport setup; external capture rejects and subsequent execution
recovers.

Current hardware is four L4s on the documented dual-NUMA Xeon Silver 4410Y host.
Device orders `2,0,3,1`, `3,2,1,0`, and `0,1,2,3` exercise interleaved,
group-reversed and identity mappings. Noncontiguous physical NUMA IDs are covered
by CPU planner checks; this host has actual node IDs 0/1. This does not establish
correctness on another GPU model, node count, or NUMA grouping.

Logs are `build/packet-mapping-{full,reversed,identity,prepared}.log`.
`packet-mapping-memcheck.log` passes the interleaved quick case with zero device
memory errors on all four ranks; its timing is not performance evidence.
The independent four-rank native comparison in `packet-mapping-native.log`
selects packet for five of 18 FP32 requests and retains NCCL 2.28.7 for 13.
It uses counts 64/257/65536, two layout patterns, three paired ABBA rounds,
two warmups and three timed calls/block, and maximum-rank host completion
including permutation work. All samples and negative outcomes are retained;
this is exact-request admission on one machine, not a best-variant curve or
a claim for all sizes.

Because shared host storage is also used by the retained network algorithm,
`build/packet-mapping-network/` reruns two real four-L4 hosts: 126 full-buffer
ABI calls/rank, two lifetimes, matching path/cohort evidence and all-rank
rejection of inconsistent profiles. `packet-mapping-graph-regression.log`
preserves the BF16 owned-graph path. The default packet implementation uses GPU
replica progress; the separately compiled CPU-copy helper's updated node
argument is not a new CPU-copy performance result.
