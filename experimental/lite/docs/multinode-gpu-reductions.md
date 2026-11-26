# Paired GPU reductions borrowed from AllGather

`multinode_allreduce --reduce-matched ce|sm` implements explicit two-node FP32
ReduceScatter and AllReduce experiments. **All reductions execute on GPUs.**
The CPU/proxies only progress MSCCL++ transfers. These are experimental options;
they do not replace the existing automatic policies.

The transferable AllGather techniques are NUMA-aware pairing, bounded pieces,
keeping outbound data immutable, and ready/data handshakes that protect reuse.
Sharing a host packet is not required by this GPU-memory implementation.

## Local reduction and gathering

The documented NUMA pairs are GPUs `{0,1}` and `{2,3}`. N denotes input bytes
per GPU, and the local node has four GPUs.

1. Exchange with `rank ^ 1`, keeping the half of destination chunks assigned
   to this GPU's low rank bit. GPUs add incoming values to their own input.
2. Exchange the remaining partner's partial with `rank ^ 2`, keeping one
   quarter of the original destinations. GPUs add the two pair partials.
3. Exchange node partials over the corresponding GPU's IB channel. ReduceScatter
   sends only the remote node's destination shard. A GPU adds its own node
   partial and the received partial to produce the final sum.
4. For AllReduce, gather the results with `rank ^ 2` and then `rank ^ 1`.
   Results are written directly into their final output positions.

ReduceScatter groups the destination GPU's two node shards logically; its
original rank-ordered input stays immutable and needs no pack kernel. Only
partial sums use a packed scratch layout. AllReduce operates in place in the
existing harness. No CPU arithmetic or CPU-router initialization is introduced.

The local reduction moves 3N bytes per node, as does the ring. Cross-NUMA
payload is N rather than the ring's 1.5N. The reverse gather has the same
comparison. This cuts cross-NUMA payload by one third without reducing total
local payload. It does not imply an end-to-end speedup: the paired schedule
still has handshake and reduction dependencies and waits for each IB piece.

`ce` submits payload copies through CUDA-IPC PortChannels. `sm` issues GPU
peer stores. Both use GPU ready handshakes; they never infer readiness merely
from a locally completed send. Every phase drains its outgoing local payload
before advancing. SM stores are system-fenced before completion publication.

## Earlier ReduceScatter network submission

`--rs-matched-early 1` splits the cross-NUMA exchange into remote-destination
and local-destination halves. Each GPU first receives and reduces the remote
half, then submits its IB send. The local half's transfer and GPU reduction
continue while that send is in flight. The halves occupy disjoint positions
in `ibSend`; the outbound half remains immutable through the final IB flush.
The existing per-piece lookahead and receive-slot protocol remain in effect.

This adds a local data-completion round, but no additional payload. It can
hide more of the final IB piece; finer pieces may lose that benefit to extra
notifications. The option requires matched ReduceScatter and defaults off.
The more aggressive experiment that splits *both* local stages was rejected
and retained only as a patch in the
[early-send measurement record](../benchmarks/data/gpu_early_2x4l4_20261115/README.md).

## Pieces and lifetime

`--reduce-piece-bytes K` bounds each destination shard's piece. K must be a
multiple of 512 bytes, between 512 bytes and 1 GiB. Omitting it transfers a
whole shard per phase. ReduceScatter's local owner chunk contains two such
shards, while AllReduce's owner chunk contains one. A shorter final piece is
supported; all accepted extents preserve 16-byte copy alignment.

Pieces are submitted in order. `--reduce-depth D` (0–64, default 0) posts a
piece's IB exchange and performs D further local reductions before waiting
for that piece. Effective depth is capped at the number of pieces minus one.
The prologue fills the pipeline and the epilogue drains it; notifications are
consumed in order. Outbound partials and receive ranges are disjoint between
pieces. AllReduce may gather completed pieces while later IB writes remain in
flight. Depth zero preserves the original serial-piece schedule.

No extra payload allocation is needed for lookahead: all piece ranges already
have separate positions in the full-shard staging buffers. This is an eager,
bounded software pipeline, not graph replay. Progress and notifications remain
part of measured latency, including the cost of finer pieces. See the
[lookahead measurements](../benchmarks/data/gpu_overlap_2x4l4_20261115/README.md).

Network receives use two alternating slots sized from the maximum configured
message. A partner cannot reach invocation i+2 without receiving invocation
i+1, which requires the other side to finish consuming invocation i. Outgoing
node partials remain immutable until IB flush completes. Local input/scratch
reuse is protected by both endpoints' ready handshakes. Slots advance at eager
host launch; captured graph replay is neither implemented nor validated.

Occupancy is checked for the selected reduction kernel before a multi-CTA
launch, because the algorithm uses grid barriers. Other reduction modes cannot
be combined with `--reduce-matched`. Existing automatic policies are unchanged.

## Build and run

```bash
make multinode_allreduce multinode_cpu_router \
  MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx \
  MSCCLPP_IB_BUILD=/home/zhongjie/third_party/mscclpp/build-ib-pcieccl
HOSTS=10.10.55.2:4,10.10.55.1:4 SOCKET_IFNAME=ibp55s0f0 \
MPIRUN=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun \
  bash scripts/run_multinode.sh multinode_allreduce --op reducescatter \
  --reduce-matched ce --reduce-piece-bytes 4194304 --reduce-depth 2 -B 48 -T 256 \
  -b 1024 -e 67108864 -w 5 -n 10 --check-each 1 -c 1 --gpu-timing 0
```

Stage the same binary and libraries at the same paths on both hosts first.
Use `--op allreduce` to include the reverse paired gather. Numeric aliases 1
and 2 select CE and SM for the archived pilot scripts.

See the [measurements and validation record](../benchmarks/data/gpu_reduce_2x4l4_20261115/README.md).
The earlier GPU peer-read/host-staging trial was rejected: GPU arithmetic alone
did not make its staging and peer-access schedule competitive. Its patch and
negative measurements are retained with the record, not selected by this path.
