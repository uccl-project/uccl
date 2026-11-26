# Qualified default UCCL Lite policy

The default `graph512` policy is qualified on the recorded two four-L4 hosts.
Its [full paired comparison](../../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/graph512_full/README.md)
passes 126/126 points, and the [default-build qualification](../../benchmarks/data/uccl_reference_gap_20261118/reference_protocol/qualified_release/README.md)
records all checks and artifact identities. The earlier failed configurations
remain archived; these results do not validate other placements or API modes.

The table applies to FP32, eight ranks, a prepared 1-GiB capacity, and the profiled
powers of two. Size means `max(inputBytes, outputBytes)`: the complete output for
AG, complete input for RS, and buffer size for AR. It does not mean each rank's
AG input or RS output. Both legal in-place and out-of-place layouts use this policy.

| Operation | Complete extent | UCCL Lite path |
| --- | --- | --- |
| AG | 1–64 KiB | Single-CTA hierarchy with LL16Packet transport |
| AG | 128 KiB | Single-CTA raw transfer with explicit send retirement |
| AG | 256 KiB–4 MiB | Cooperative host-packet gather |
| AG | 8 MiB–1 GiB | Persistent host-packet gather |
| AR | 1–64 KiB | Single-CTA hierarchy with LL16Packet transport |
| AR | 128 KiB | Single-CTA raw hierarchy |
| AR | 256–512 KiB | Direct GPU ring using alternating medium banks |
| AR | 1 MiB–1 GiB | UCCL Lite RS followed by host-packet AG |
| RS | 1–32 KiB | Single-CTA hierarchy with LL16Packet transport |
| RS | 64–256 KiB | Single-CTA raw hierarchy |
| RS | 512 KiB–8 MiB | Streamed GPU ring; 8 MiB sends the remote half first |
| RS | 16–64 MiB | One-piece copy/compute graph, alternating medium banks |
| RS | 128 MiB | Two-piece graph, alternating banks within existing large storage |
| RS | 256 MiB–1 GiB | Multi-piece graph with the large-buffer reuse handshake |

At 16 MiB, scatter-graph reduction kernels use 512 CTAs with 256 threads; other
graph reduction grids remain 256 CTAs. This applies to public RS and hybrid AR
without changing copy-engine nodes, graph ordering or per-element arithmetic.

The private RS stage of hybrid AR uses graphs from 8 MiB; standalone RS uses
graphs from 16 MiB. Persistent gather retains four upload CTAs, twenty-four consumer CTAs at
8–32 MiB and sixteen for bulk, plus its control CTA, subject to cooperative
residency. Its two
host progress workers per host poll for work. These resource costs are part of
the measured configuration.

MSCCL++ supplies algorithm registration/context management, bootstrap,
registration, peer channels, semaphores, packet types and IB runtime components.
The algorithms and NCCL ABI wrapper remain UCCL Lite. No native NCCL collective
is loaded, forwarded to or used as fallback. The full API and placement limits
are listed in [MSCCLPP.md](MSCCLPP.md#supported-scope).

Graph binding reuse checks allocation identity as well as pointer equality;
recycled addresses invalidate bindings. Pointer-descriptor reuse starts only
after its writer has completed. Bank transitions retain the synchronization
required to protect local and network readers. These caches never reuse input
data or collective results.

The diagnostic synchronization entry point waits for the same completion event
using `cudaEventQuery`, then retires remaining transport work. The default
also skips redundant device switching only when the exact current CUDA context
matches the previously observed primary context. Ordinary stream completion and
caller-buffer ownership requirements are unchanged.

The default Makefile target selects this policy. Exact original build commands
and source snapshots are archived with the measurements. No qualification curve
combines different libraries' best points.
