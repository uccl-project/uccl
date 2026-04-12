# DeepEP-Lite L4 Performance Benchmarks

Low-latency mode on L4 GPUs (sm_89, PCIe Gen4 x16, no NVLink, no GDR).

**Testbed**: 2 nodes (l40/l41), 4× L4 each, 1× mlx5 400Gbps RDMA NIC per node.
All data transfers use host-staged RDMA (`cudaMallocHost` buffers, no GPUDirect).

**Benchmark**: `bench/test_low_latency.py`, `num_topk=4`, `num_experts=8`, `--disable-nvlink`.
All experiments pass correctness verification (16 configs × varying fp8/scale/hook).

---

## Summary Table

| Setup | Tokens | Hidden | D+C BW (GB/s) | D BW (GB/s) | C BW (GB/s) | D+C Latency (μs) |
|-------|-------:|-------:|---------------:|------------:|------------:|------------------:|
| 1n×2g | 128 | 2048 | 5.95 | 6.42 | 5.09 | 526 |
| 1n×2g | 256 | 4096 | 7.27 | 5.33 | 6.57 | 1,735 |
| 1n×2g | 512 | 7168 | 6.86 | 5.03 | 6.14 | 6,461 |
| 1n×4g | 128 | 2048 | 3.67 | 5.20 | 3.29 | 851 |
| 1n×4g | 256 | 4096 | 3.60 | 3.75 | 3.49 | 3,497 |
| 1n×4g | 512 | 7168 | 3.52 | 3.93 | 3.44 | 12,584 |
| 2n×1g | 128 | 2048 | 13.61 | 6.07 | 13.43 | 230 |
| 2n×1g | 256 | 4096 | 21.95 | 12.08 | 19.12 | 574 |
| 2n×1g | 512 | 7168 | 22.86 | 21.37 | 20.33 | 1,939 |
| 2n×4g | 128 | 2048 | 1.02 | 0.96 | 0.96 | 3,058 |
| 2n×4g | 256 | 4096 | 1.02 | 1.10 | 0.98 | 12,352 |
| 2n×4g | 512 | 7168 | 1.01 | 1.10 | 0.98 | 44,040 |

## Detailed Timing (Rank 0)

| Setup | Tokens | Hidden | D Send (μs) | D Recv (μs) | C Send (μs) | C Recv (μs) |
|-------|-------:|-------:|------------:|------------:|------------:|------------:|
| 1n×2g | 128 | 2048 | — | — | — | — |
| 1n×2g | 256 | 4096 | — | — | — | — |
| 1n×2g | 512 | 7168 | — | — | — | — |
| 1n×4g | 128 | 2048 | — | — | — | — |
| 1n×4g | 256 | 4096 | — | — | — | — |
| 1n×4g | 512 | 7168 | — | — | — | — |
| 2n×1g | 128 | 2048 | 147 | 17 | 53 | 60 |
| 2n×1g | 256 | 4096 | 280 | 44 | 104 | 120 |
| 2n×1g | 512 | 7168 | 549 | 130 | 261 | 253 |
| 2n×4g | 128 | 2048 | 148 | 48 | 104 | 119 |
| 2n×4g | 256 | 4096 | 279 | 44 | 104 | 120 |
| 2n×4g | 512 | 7168 | 547 | 135 | 259 | 253 |

## Analysis

### Single-Node (intranode RDMA loopback)

- **2 GPUs**: 5.9–7.3 GB/s D+C. Good utilization of PCIe bandwidth with only
  one peer to communicate with.
- **4 GPUs**: 3.5–3.7 GB/s D+C. Bandwidth drops ~50% due to shared PCIe root
  complex and single NIC contention across 4 proxies.

### Multi-Node (internode RDMA)

- **2n×1g** (1 GPU/node): Best per-rank bandwidth — **up to 22.9 GB/s** D+C at
  512 tokens. Each GPU has exclusive NIC access. Combine is faster than dispatch
  because receive-side processing is lighter (no routing scatter).
- **2n×4g** (4 GPUs/node): ~1.0 GB/s D+C across all sizes. Bandwidth is
  bottlenecked by 4 GPUs sharing a single NIC + PCIe root complex. Latency
  scales linearly with data size, confirming bandwidth saturation.

### Bottleneck: PCIe + Single NIC

The dominant bottleneck for 4-GPU configurations is the shared PCIe Gen4 x16
link (~32 GB/s theoretical, ~25 GB/s practical) and single mlx5 NIC. Each GPU
competes for the same interconnect.

For 2n×4g, the effective per-GPU bandwidth is ~0.25 GB/s — far below what each
L4's PCIe link could deliver individually. Multi-NIC or NVSwitch would be
needed to scale.

### Comparison with H100 (NVLink + GDR)

| Metric | H100 (estimate) | L4 (measured, 2n×1g) | Ratio |
|--------|------------------:|---------------------:|------:|
| D+C BW (512t×7168h) | ~50 GB/s | 22.86 GB/s | 2.2× |
| LL dispatch latency | ~5 μs | ~550 μs | 110× |

The L4 achieves surprisingly close aggregate bandwidth to H100 for large
messages (only 2.2× slower), but latency is 100×+ higher due to PCIe round
trips replacing NVLink direct access.
