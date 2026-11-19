# GPU readiness gates

The full zero-regression target remains active. This experiment changes only
how the persistent actors observe readiness: one publisher warp acquires the
original system-memory counters and publishes per-slot GPU tokens. Producers
and consumers acquire those device-scope tokens before their block barriers.
The host reads are still performed with system acquire semantics; a warp
synchronization transfers them to the lane issuing the device release. Slot
reuse still requires all local/remote reader ACKs and outgoing retirement.
Receive credits and the MSCCL++ CPU-only RDMA worker are unchanged.

The first fixed build retains cooperative AG through 8 MiB. On two four-L4
hosts, FP32, it passes 60 component calls/rank over two lifetimes and the 144-call
mixed ABI. Three unchanged full nccl-tests AG sweeps (1 KiB–1 GiB, both layouts,
20 warmups/30 iterations, maximum-rank host enqueue/completion batch means)
give 16-MiB out/in 638.30/637.61 us versus fresh UCCL 672.00 us; 32-MiB
1250.58/1250.43 us versus 1308.87 us; 1-GiB 39.223/39.228 ms versus 41.922 ms.
No per-size minima are selected. These are cross-harness diagnostics, not matched
admission. Native NCCL is absent and every size executes UCCL without fallback.

The subsequent experiment lowers only the cooperative cutoff from 8 to 4 MiB,
keeping the prepared packet capacity unchanged. This tests the new persistent
path at 8 MiB instead of the earlier cooperative path. It must pass its own
correctness and full-curve checks before any combined performance claim.
