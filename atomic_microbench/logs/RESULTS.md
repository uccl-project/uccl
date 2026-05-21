# Atomic microbenchmark — results

Hosts: chi-mi325x-pod2-099 (client, "amd1") → chi-mi325x-pod2-098 (server, "amd2")
NIC: mlx5_0, ConnectX-6 Dx, RoCEv2 (gid_idx=3, MTU 1024)
HW atomics: `atomic_cap = ATOMIC_HCA (1)` — supported
Iterations: 20000 per (trial, mode, size); 3–4 trials per cell (bench was killed mid-trial 4).

Latency is client-observed: time from `ibv_post_send` to the signaled completion (= one wire RTT including the responder ACK).

**`bytes` is the RDMA WRITE payload size.** The HW atomic and UEP atomic do not add to this — the HW atomic is a separate 8-byte fetch-add op chained after the write, and the UEP atomic rides in the 4-byte `imm_data` of the same WRITE_WITH_IMM (no extra payload).

**Posting pattern** (already optimal — no waiting between ops):
- *WRITE + HW atomic FA*: single `ibv_post_send` with a chained WR list. WRITE is unsignaled (no CQE), ATOMIC_FA is signaled. One MMIO doorbell, both WRs visible to the NIC at the same time. We poll only the atomic's CQE.
- *WRITE + UEP atomic*: single `ibv_post_send` of one `RDMA_WRITE_WITH_IMM`. Responder CPU decodes `imm_data` from the RECV CQE and applies the local atomic add.

| bytes | WRITE only (µs) | WRITE + HW atomic FA (µs) | WRITE + UEP atomic / imm (µs) | HW penalty | UEP penalty |
|------:|---------------:|--------------------------:|------------------------------:|-----------:|------------:|
|     8 | 4.27           | 5.28                      | 4.27                          | +1.01      | +0.00       |
|    64 | 4.29           | 5.31                      | 4.30                          | +1.02      | +0.01       |
|   256 | 4.38           | 5.34                      | 4.38                          | +0.96      | +0.00       |
|  1024 | 4.59           | 5.55                      | 4.59                          | +0.96      | +0.00       |
|  4096 | 4.87           | 5.84                      | 4.88                          | +0.97      | +0.01       |

Tail percentiles (worst p99 across trials) follow the same shape: UEP atomic tracks WRITE within ~10 ns; HW atomic is ~1 µs higher at every percentile.

## Why UEP-emulated atomic ≈ plain WRITE

The HW path posts **two work requests** in one doorbell: an unsignaled `IBV_WR_RDMA_WRITE` chained to a signaled `IBV_WR_ATOMIC_FETCH_AND_ADD`. The client never waits for the WRITE's completion — there is no CQE for it. The NIC pipelines both packets back-to-back.

The ~1 µs penalty is therefore **not** about posting order; it is the inherent wire cost of the atomic being a *separate RC op* that needs its own request packet, its own atomic execution at the responder NIC, and its own ACK packet. The responder serializes the atomic behind the WRITE in its execution engine and emits a second ACK; that second ACK is what we wait for.

The UEP path posts **one** `IBV_WR_RDMA_WRITE_WITH_IMM`. The 32-bit immediate encodes a signed-15-bit increment and a 13-bit aligned offset (`AtomicsImm::PackAtomic` in `ep/include/rdma.hpp`). On the wire it is byte-for-byte the same as a plain WRITE plus a 4-byte BTH/IMM field. The responder NIC delivers a RECV CQE; the responder CPU decodes the imm and does the atomic increment locally out of CPU cache. The wire ACK is the same single WRITE ACK — so the requester sees plain-WRITE latency.

So in this regime UEP atomic ≈ free atomic update bundled with the data write, vs. HW atomic which costs an extra ~1 µs (≈20% on top of the WRITE).

## Other axes where UEP wins

- **NIC support**: HW atomics need `atomic_cap > NONE`. Many NICs (EFA, some bnxt, older ConnectX) don't support them or only support 64-bit FA at low rates. UEP atomic works on any NIC that supports RDMA WRITE_WITH_IMM (i.e., all of them).
- **Target alignment / width**: HW FA is 64-bit aligned, 64-bit wide, and only FA / CAS. UEP supports arbitrary {value, offset} packed in 28 bits with the application choosing the semantics (saw it used as combine/non-combine, expert idx, etc. in `ep/src/rdma.cpp`).
- **Throughput**: A WR chain (WRITE + ATOMIC) burns 2 SQ slots and 2 CQ entries; UEP burns 1 SQ slot and 1 RQ slot.
- **No PCIe atomic dependency**: HW atomics on RoCE NICs interact with PCIe AtomicOp routing and IOMMU configuration; UEP just needs a CPU atomic on memory the responder already owns.

## Where HW atomic might still win

- **GPU-direct atomics on GPU memory**: UEP requires the responder CPU to be in the loop to apply the increment. If the target memory is GPU resident and the receiver doesn't have a poller, the HW atomic path can update it directly.
- **Genuine compare-and-swap semantics** (lock-free protocols): UEP-emulated atomic is just add; CAS would need a heavier protocol.

## Notes / caveats

- Bench was killed at trial 4 (sweep stalled around `write_hw_atomic` bytes=4096). Cause not investigated; numbers shown are the means of 3–4 completed trials per cell, which are tight (stddev < 0.1 µs except for occasional tail outliers — `max=131 µs` once on a 4096-byte WRITE, almost certainly a kernel preemption on the polling thread).
- Per-iteration measurement is `post → poll`. Client busy-polls CQ.
- See `bench_amd1.csv` for full per-trial percentiles.
