# Hierarchical collectives on two four-L4 hosts

This is an experimental design and measurement record, not an automatic routing
policy. The existing MSCCL++ paths remain the default. The November 15 experiment
adds an explicit remote-first ReduceScatter; AllReduce and AllGather retain their
existing implementations.

## Placement and schedule

The measured hosts are `mibura-sky-test-02` and `mibura-sky-test-01` (`l40`).
On each host GPUs 0/1 share NUMA 0 with `mlx5_0`; GPUs 2/3 share NUMA 1 with
`mlx5_1`. There is local CUDA peer access but no NVLink. Cross-NUMA traffic
traverses the CPU interconnect. Four network participants stripe destination
shards over the two HCAs; reducing locally avoids sending four copies of each
contribution over IB.

For full FP32 input extent N per rank, ReduceScatter assigns each local rank r
two N/8-byte destinations, global ranks r and r+4. Each rank:

1. Pushes each destination's inputs into disjoint peer scratch slots, or, for
   the explicitly selected pull experiment, reads immutable peer inputs.
2. Reduces the destination on the **other** node first and posts its N/8-byte IB
   transfer to the corresponding local rank on that node.
3. Reduces its local destination while that IB transfer progresses.
4. Completes a local reader handshake, waits for IB, and adds the received
   partial to its local result. Flush protects the outbound source's lifetime.

The push path moves the same payload as the existing direct local RS, but exposes
one half of the reduction to the network earlier. The pull path removes scratch
writes but substitutes PCIe reads. The pilot measurements reject the assumption
that those reads are cheaper on this topology. Increasing CTA count also needs
measurement: grid barriers and cross-NUMA contention can outweigh parallelism.

Every scratch interval has one source writer. Every output interval has one
owner. Outbound and local-result halves are separate, so local accumulation never
modifies a NIC source. The local completion handshake protects input/scratch
reuse across consecutive calls. Two alternating receive slots protect the
previous call from a faster remote writer; completing call i+1 requires receiving
its partner's i+1 send, so neither side can lap the other by two calls. Slot
offsets use the configured maximum shard size. This harness advances the slot
at each eager launch; it does not implement CUDA graph replay. Grid size is checked against the actual kernel's
resident occupancy. The existing harness limits inputs to FP32 and complete
extents divisible by 512 bytes (at least 1024), so every vectorized shard is
16-byte aligned without a truncated tail. Unsupported RS mode combinations are
rejected. Reduction order is fixed, but differs from the serial reference.

Use `--op reducescatter --rs-overlap-local 1 -B 1 -T 256` for push or
`--rs-pull-local 1` for the negative pull comparison. Neither changes `--rs-auto`.
The explicit `-B` matters: the general executable defaults to 24 blocks.

## Optional CPU-router design

The next architectural experiment is a two-level NUMA packet hierarchy, not a
replacement of the independently linked CPU-router core with MPI or MSCCL++.
Keep the network adapter and collective schedule in `src/collectives/`; the
caller supplies transport progress, topology validation, and fallback.

| Collective | Within each NUMA pair | Across nodes | Final local phase |
| --- | --- | --- | --- |
| AllReduce | Reduce two GPU contributions into destination-striped packets; combine the two NUMA partials | Exchange reduced stripes over the stripe owner's HCA | Add remote partial, publish immutable result to both NUMA groups and multicast to GPUs |
| ReduceScatter | Reduce and partition by final global destination | Send only remote-destination partials; compute local destinations concurrently | Add received partial and deliver only to the owning GPU |
| AllGather | Pack two distinct source shards once | Forward each group's packet to remote groups; exchange with the other local group concurrently | Multicast shared immutable packets to each pair's GPUs |

For tiny packets, one packet per NUMA group amortizes transport submissions.
For intermediate messages, use a bounded packet pipeline with independent
upload, network, and download progress. Choose packet sizes from measurements,
not a fixed presumed crossover. The existing host multicast AllGather provides
a working transport prototype; it uses four MPI processes for eight GPUs, so it
needs its own same-layout NCCL control.

`RouterCore::process` can accept a caller-provided FP32 reduction operator;
its default BF16 operator must not be reused for FP32 bytes.
`RouterCore::forward` and callback-based `multicast` can express host forwarding
and shared-packet publication. A network adapter must retain source spans through
send completion and establish receive visibility before publishing GPU readiness.
A sender completion alone cannot release a packet still being read by GPUs.
Use generation counters and credits for all network consumers and GPU readers;
never overwrite a slot until all acknowledge that generation. CPU workers belong
to the NUMA group of their registered host allocation and HCA.

Preparation must explicitly validate dtype, exact size, rank/GPU placement,
P2P matrix, GPU-to-NUMA and HCA mapping, software identity, and calibrated paired
timings before allocating pinned buffers or starting workers. A rejected request
invokes the caller's fallback without router initialization. The new benchmark
flags are manual experiments, not evidence for enabling a production route.

Before implementing that adapter, establish a matched AR/AG/RS baseline and
profile readiness-to-post, network completion, and final GPU completion. A CPU
path must remove enough GPU/proxy synchronization to pay for its staging and
worker overhead. Require router-policy and dispatcher checks, eager/graph tests,
repeated-generation and delayed-consumer cases, and two-host end-to-end checks
before promotion. Current November 15 validation covers the GPU RS experiment,
not this proposed CPU adapter or graph replay.

See [the current measurement record](https://github.com/zhongjiechen/uccl-zhongjie-dev/blob/cea1faf6473fc76b5bd192db025f5291169d3b58/lite/benchmarks/data/multinode_overlap_20261115/README.md).
