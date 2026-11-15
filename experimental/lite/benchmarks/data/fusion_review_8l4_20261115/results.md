# Fresh fusion results against optimized standalone collectives

On the 24 tested operation/shape pairs, the unchanged resident fusion schedule has lower aggregate medians than paired NCCL at 24/24 points and than the optimized sequential UCCL control at 17/24 points. These are descriptive median counts, not significance tests or an automatic dispatch policy. Several differences are near ties. The optimized sequential control changes the conclusion at some shapes, so historical fusion gains must not be used as evidence against the new standalone schedule.

The largest regressions against the current sequential control are 34.9% for 32B AG+GEMM at M=64, and 11.8% / 8.4% for 9B GEMM+AR at M=2048 / 8192. Prefill fusion remains useful at other shapes, including 32B GEMM+RS and AG+GEMM.

All values below are milliseconds, medians of two reversed-run medians, BF16 on eight L4s. Each method/configuration has 30 checked calls per run. See [method and validation](README.md) and [all configuration medians](summary.csv). Historical-config sequential is a fresh measurement of the old configuration, not a copied November 11 measurement. The fused column uses the current-profile process with the original resident settings (no CPU batching). No best variant per size is selected.

| Operation | Model | M | NCCL + cuBLAS | Historical-config sequential | Current sequential | Fused | Current / fused |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gemm-ar | 9B | 1 | 0.2048 | 0.1518 | 0.1782 | 0.1336 | 1.333× |
| gemm-ar | 9B | 64 | 0.5007 | 0.3891 | 0.3942 | 0.3640 | 1.083× |
| gemm-ar | 9B | 2048 | 7.2796 | 5.7554 | 5.0191 | 5.6090 | 0.895× |
| gemm-ar | 9B | 8192 | 28.6410 | 22.4696 | 19.0262 | 20.6208 | 0.923× |
| gemm-ar | 32B | 1 | 0.2130 | 0.1679 | 0.1718 | 0.1508 | 1.139× |
| gemm-ar | 32B | 64 | 0.5716 | 0.4326 | 0.4393 | 0.4234 | 1.037× |
| gemm-ar | 32B | 2048 | 9.5736 | 7.6562 | 6.6353 | 6.5034 | 1.020× |
| gemm-ar | 32B | 8192 | 37.0775 | 29.2475 | 24.9974 | 24.8696 | 1.005× † |
| gemm-rs | 9B | 1 | 0.1753 | 0.1608 | 0.1551 | 0.1280 | 1.212× |
| gemm-rs | 9B | 64 | 0.3320 | 0.3159 | 0.3082 | 0.3105 | 0.993× |
| gemm-rs | 9B | 2048 | 4.0346 | 3.1450 | 3.1711 | 3.1081 | 1.020× |
| gemm-rs | 9B | 8192 | 15.7834 | 11.9224 | 11.8902 | 11.9155 | 0.998× |
| gemm-rs | 32B | 1 | 0.1884 | 0.1528 | 0.1620 | 0.1498 | 1.082× |
| gemm-rs | 32B | 64 | 0.3809 | 0.3425 | 0.3561 | 0.3625 | 0.982× † |
| gemm-rs | 32B | 2048 | 5.5078 | 4.4227 | 4.4124 | 4.0056 | 1.102× |
| gemm-rs | 32B | 8192 | 21.0417 | 16.3397 | 16.3062 | 15.1457 | 1.077× |
| ag-gemm | 9B | 8 | 0.1754 | 0.1464 | 0.1654 | 0.1096 | 1.509× |
| ag-gemm | 9B | 64 | 0.1904 | 0.1436 | 0.1521 | 0.1580 | 0.963× |
| ag-gemm | 9B | 2048 | 1.6855 | 1.4595 | 1.3430 | 1.3386 | 1.003× † |
| ag-gemm | 9B | 8192 | 6.2917 | 5.4546 | 5.0358 | 4.6244 | 1.089× |
| ag-gemm | 32B | 8 | 0.1935 | 0.1562 | 0.1725 | 0.1339 | 1.289× |
| ag-gemm | 32B | 64 | 0.2747 | 0.1933 | 0.1510 | 0.2038 | 0.741× |
| ag-gemm | 32B | 2048 | 3.5881 | 3.1416 | 2.9473 | 2.6445 | 1.115× |
| ag-gemm | 32B | 8192 | 13.4203 | 11.7286 | 10.8355 | 9.4564 | 1.146× |

A ratio above 1 favors fusion. † means the winner versus current sequential changes between runs; these points should be treated as unresolved near ties, rather than stable gains/losses. The full run medians remain in the CSV.

The CPU job batching diagnostic does not establish a uniform benefit. It runs fused alone, while the unbatched control interleaves three methods. It remains an explicit option and is not promoted to a new default.

Keep the existing resident synchronization and cooperative occupancy contract. Update sequential controls and per-shape claims first. GPU progress, interleaved RS, host-ring RS and parallel standalone graph submission require separate resident integration/validation; bypassing the current guards is not a valid optimization. This study does not enable any application routing backend or production dispatch rule.

## Fixed 512-KiB AR prefill candidate

An explicit packet-cap change improves all four tested AR prefill shapes in both reversed runs. Fixed 512-KiB packets reduce fused latency by **7.5%–19.4%** versus fixed 128-KiB packets and by **9.0%–12.2%** versus the paired current sequential control. Each cap uses its own fresh interleaved NCCL/sequential/fused controls. These measurements are separate from the unchanged-default 24-shape table above; no best-cap-per-size curve is constructed.

| Model | M | Fused 128 KiB (ms) | Fused 512 KiB (ms) | Current sequential (ms) | Reduction vs 128 KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9B | 2048 | 5.6717 | 4.5722 | 5.0417 | 19.4% |
| 9B | 8192 | 20.5703 | 17.3258 | 19.0403 | 15.8% |
| 32B | 2048 | 6.4269 | 5.9433 | 6.6668 | 7.5% |
| 32B | 8192 | 24.5233 | 21.9891 | 25.0494 | 10.3% |

[All packet medians](packet_summary.csv) retain both runs and the NCCL controls. The packet cap also changes active CPU lanes and resident transport crew sizes under the existing schedule, so this is not an isolated measurement of packet count alone. The tradeoff can differ for smaller shapes.

Use the existing explicit option `--packet-kib 512` (or `PacketPipelineOptions::packetBytes = 512UL << 10`) for these validated AR prefill shapes on this hardware/software configuration. No new automatic policy is installed, and other sizes, model pairs and timing scopes require their own calibration. Reproduce one tested shape with:

```bash
NCCL_NET=Socket ./build/fusion_review_8l4_20261115/uccl_lite_router_fusion_8l4 \
  --op gemm-ar --m 2048 --n 4096 --k 1536 \
  --router-profile current --packet-kib 512 \
  --warmup 2 --repeats 3 --iterations 10
```
