# Lite AllReduce vs NCCL Performance

Date: 2026-06-10

Primary tables and plots use out-of-place nccl-tests results. Ratios greater than `1.00x` favor Lite: latency speedup is `NCCL us / Lite us`, and bus bandwidth ratio is `Lite GB/s / NCCL GB/s`. In-place values are retained in `plot_data.csv` because AllReduce can diverge materially for some paths.

## Benchmark settings and sources

| Setting | Value |
| --- | --- |
| Iterations | `50` |
| Warmup | `20` |
| Size convention | latency: `128B-1MiB`; bus bandwidth: `1MiB-1GiB` |
| Multi-node hosts | `10.10.55.1,10.10.55.2` |
| NCCL multi-node baseline | `NCCL_NET_GDR_LEVEL=0` no-GDR where explicitly re-run |
| Generated plots | `plots/allreduce/` |

All plotted source rows reported `#wrong=0`.

## Summary

| Setup | Latency geomean Lite speedup | BusBW geomean Lite/NCCL | 1GiB Lite GB/s | 1GiB NCCL GB/s | 1GiB ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1nx4g` | 0.44x | 0.82x | 17.59 | 17.55 | 1.00x |
| `2nx1g` | 0.12x | 0.63x | 15.32 | 15.37 | 1.00x |
| `2nx2g` | 0.57x | 0.89x | 14.75 | 15.84 | 0.93x |
| `2nx4g` | 1.84x | 1.01x | 16.24 | 15.08 | 1.08x |

## 1nx4g

### Latency, 128B-1MiB (us)

| Size | Lite us | NCCL us | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 6.03 | 13.02 | 2.16x |
| 256B | 5.97 | 12.50 | 2.09x |
| 512B | 7.04 | 12.82 | 1.82x |
| 1KiB | 9.19 | 12.87 | 1.40x |
| 2KiB | 14.62 | 13.15 | 0.90x |
| 4KiB | 24.14 | 13.60 | 0.56x |
| 8KiB | 41.23 | 14.33 | 0.35x |
| 16KiB | 79.42 | 15.48 | 0.19x |
| 32KiB | 84.37 | 16.74 | 0.20x |
| 64KiB | 156.87 | 22.83 | 0.15x |
| 128KiB | 156.00 | 35.26 | 0.23x |
| 256KiB | 270.03 | 60.51 | 0.22x |
| 512KiB | 611.41 | 93.42 | 0.15x |
| 1MiB | 1288.60 | 138.80 | 0.11x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite GB/s | NCCL GB/s | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 1.22 | 11.33 | 0.11x |
| 2MiB | 14.07 | 14.17 | 0.99x |
| 4MiB | 15.92 | 15.93 | 1.00x |
| 8MiB | 16.83 | 16.78 | 1.00x |
| 16MiB | 17.23 | 17.04 | 1.01x |
| 32MiB | 17.27 | 17.13 | 1.01x |
| 64MiB | 17.43 | 17.23 | 1.01x |
| 128MiB | 17.47 | 17.34 | 1.01x |
| 256MiB | 17.53 | 17.48 | 1.00x |
| 512MiB | 17.57 | 17.55 | 1.00x |
| 1GiB | 17.59 | 17.55 | 1.00x |

Plots: [`1nx4g_latency_128B_1MiB.png`](../plots/allreduce/1nx4g_latency_128B_1MiB.png), [`1nx4g_busbw_1MiB_1GiB.png`](../plots/allreduce/1nx4g_busbw_1MiB_1GiB.png)

## 2nx1g

### Latency, 128B-1MiB (us)

| Size | Lite us | NCCL us | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 166.36 | 12.21 | 0.07x |
| 256B | 161.07 | 12.58 | 0.08x |
| 512B | 331.79 | 12.73 | 0.04x |
| 1KiB | 263.89 | 12.86 | 0.05x |
| 2KiB | 449.82 | 13.38 | 0.03x |
| 4KiB | 418.30 | 14.42 | 0.03x |
| 8KiB | 329.33 | 15.74 | 0.05x |
| 16KiB | 167.24 | 18.40 | 0.11x |
| 32KiB | 169.26 | 23.04 | 0.14x |
| 64KiB | 176.99 | 33.63 | 0.19x |
| 128KiB | 186.01 | 55.52 | 0.30x |
| 256KiB | 195.97 | 104.51 | 0.53x |
| 512KiB | 221.10 | 213.05 | 0.96x |
| 1MiB | 293.19 | 106.41 | 0.36x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite GB/s | NCCL GB/s | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 3.58 | 9.85 | 0.36x |
| 2MiB | 5.59 | 12.01 | 0.47x |
| 4MiB | 8.73 | 13.42 | 0.65x |
| 8MiB | 9.12 | 14.32 | 0.64x |
| 16MiB | 8.83 | 14.55 | 0.61x |
| 32MiB | 1.65 | 14.66 | 0.11x |
| 64MiB | 14.75 | 14.70 | 1.00x |
| 128MiB | 15.56 | 14.70 | 1.06x |
| 256MiB | 15.94 | 14.72 | 1.08x |
| 512MiB | 16.22 | 14.72 | 1.10x |
| 1GiB | 15.32 | 15.37 | 1.00x |

Plots: [`2nx1g_latency_128B_1MiB.png`](../plots/allreduce/2nx1g_latency_128B_1MiB.png), [`2nx1g_busbw_1MiB_1GiB.png`](../plots/allreduce/2nx1g_busbw_1MiB_1GiB.png)

## 2nx2g

### Latency, 128B-1MiB (us)

| Size | Lite us | NCCL us | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 27.16 | 18.54 | 0.68x |
| 256B | 41.21 | 17.97 | 0.44x |
| 512B | 41.07 | 18.28 | 0.45x |
| 1KiB | 33.71 | 18.58 | 0.55x |
| 2KiB | 41.34 | 18.65 | 0.45x |
| 4KiB | 34.90 | 19.30 | 0.55x |
| 8KiB | 37.74 | 20.69 | 0.55x |
| 16KiB | 50.09 | 22.44 | 0.45x |
| 32KiB | 57.44 | 24.92 | 0.43x |
| 64KiB | 105.33 | 34.71 | 0.33x |
| 128KiB | 82.73 | 53.00 | 0.64x |
| 256KiB | 100.74 | 89.59 | 0.89x |
| 512KiB | 131.41 | 175.03 | 1.33x |
| 1MiB | 196.66 | 153.93 | 0.78x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite GB/s | NCCL GB/s | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 8.00 | 10.22 | 0.78x |
| 2MiB | 10.37 | 13.36 | 0.78x |
| 4MiB | 12.30 | 14.88 | 0.83x |
| 8MiB | 13.81 | 15.38 | 0.90x |
| 16MiB | 14.23 | 15.31 | 0.93x |
| 32MiB | 14.44 | 15.39 | 0.94x |
| 64MiB | 14.53 | 15.55 | 0.93x |
| 128MiB | 14.64 | 15.64 | 0.94x |
| 256MiB | 14.70 | 15.67 | 0.94x |
| 512MiB | 14.73 | 15.74 | 0.94x |
| 1GiB | 14.75 | 15.84 | 0.93x |

Plots: [`2nx2g_latency_128B_1MiB.png`](../plots/allreduce/2nx2g_latency_128B_1MiB.png), [`2nx2g_busbw_1MiB_1GiB.png`](../plots/allreduce/2nx2g_busbw_1MiB_1GiB.png)

## 2nx4g

### Latency, 128B-1MiB (us)

| Size | Lite us | NCCL us | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 11.96 | 21.83 | 1.83x |
| 256B | 11.93 | 22.13 | 1.85x |
| 512B | 11.99 | 22.60 | 1.88x |
| 1KiB | 12.49 | 23.47 | 1.88x |
| 2KiB | 12.62 | 26.25 | 2.08x |
| 4KiB | 13.62 | 31.11 | 2.28x |
| 8KiB | 16.54 | 32.41 | 1.96x |
| 16KiB | 21.49 | 34.74 | 1.62x |
| 32KiB | 30.59 | 39.12 | 1.28x |
| 64KiB | 49.45 | 44.05 | 0.89x |
| 128KiB | 71.80 | 67.79 | 0.94x |
| 256KiB | 107.52 | 212.80 | 1.98x |
| 512KiB | 140.32 | 740.66 | 5.28x |
| 1MiB | 199.90 | 493.68 | 2.47x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite GB/s | NCCL GB/s | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 9.18 | 3.72 | 2.47x |
| 2MiB | 11.14 | 11.37 | 0.98x |
| 4MiB | 13.38 | 14.49 | 0.92x |
| 8MiB | 14.44 | 15.92 | 0.91x |
| 16MiB | 14.99 | 15.92 | 0.94x |
| 32MiB | 15.01 | 14.69 | 1.02x |
| 64MiB | 5.94 | 14.79 | 0.40x |
| 128MiB | 16.23 | 14.89 | 1.09x |
| 256MiB | 16.23 | 14.95 | 1.09x |
| 512MiB | 16.23 | 15.00 | 1.08x |
| 1GiB | 16.24 | 15.08 | 1.08x |

Plots: [`2nx4g_latency_128B_1MiB.png`](../plots/allreduce/2nx4g_latency_128B_1MiB.png), [`2nx4g_busbw_1MiB_1GiB.png`](../plots/allreduce/2nx4g_busbw_1MiB_1GiB.png)

## Source logs

### 1nx4g
- lite: `.tmp/collective-benchmarks/ar-topologies-20260610-064813/allreduce_1nx4g_mscclpp_128B_1G.log`
- nccl: `.tmp/collective-benchmarks/ar-topologies-20260610-064813/allreduce_1nx4g_nccl_128B_1G.log`

### 2nx1g
- lite: `.tmp/collective-benchmarks/ar-topologies-20260610-064813/allreduce_2nx1g_mscclpp_128B_1G.log`
- lite: `.tmp/collective-benchmarks/ar-2nx1g-lite-current-20260610-143936/lite_current_2nx1g_1M_64M.log`
- lite: `.tmp/collective-benchmarks/ar-2nx1g-small-final-20260610-160846/lite_2nx1g_128B_1M.log`
- lite: `.tmp/collective-benchmarks/ar-2nx1g-large-seq-20260610-145943/lite_2nx1g_64M_512M.log`
- lite: `.tmp/collective-benchmarks/ar-2nx1g-large-repeat-20260610-150034/lite_2nx1g_128M_512M_repeat.log`
- lite: `.tmp/collective-benchmarks/ar-2nx1g-1g-final-20260610-155839/lite_2nx1g_1G.log`
- nccl: `.tmp/collective-benchmarks/ar-topologies-20260610-064813/allreduce_2nx1g_nccl_128B_1G.log`
- nccl: `.tmp/collective-benchmarks/ar-2nx1g-nccl-fresh-20260610-140134/nccl_2nx1g_1M_64M.log`
- nccl: `.tmp/collective-benchmarks/ar-2nx1g-small-final-20260610-160846/nccl_2nx1g_128B_1M.log`
- nccl: `.tmp/collective-benchmarks/ar-2nx1g-large-seq-20260610-145943/nccl_2nx1g_64M_512M.log`
- nccl: `.tmp/collective-benchmarks/ar-2nx1g-1g-final-20260610-155839/nccl_2nx1g_1G.log`

### 2nx2g
- lite: `.tmp/collective-benchmarks/ar-topologies-20260610-064813/allreduce_2nx2g_mscclpp_128B_1G.log`
- lite: `.tmp/collective-benchmarks/ar-topologies-20260610-064813/allreduce_2nx2g_mscclpp_small_final_guardfix.log`
- nccl: `.tmp/collective-benchmarks/ar-topologies-20260610-064813/allreduce_2nx2g_nccl_128B_1G.log`

### 2nx4g
- lite: `.tmp/collective-benchmarks/ar-small-opt-20260610-020910/allreduce_lite_small_final_reviewfix_128B_1M.log`
- lite: `.tmp/collective-benchmarks/ar-final-1M-1G-20260610-055550/allreduce_mscclpp_1M_1G.log`
- nccl: `.tmp/collective-benchmarks/ar-small-opt-20260610-020910/allreduce_nccl_fresh_128B_1M.log`
- nccl: `.tmp/collective-benchmarks/ar-final-1M-1G-20260610-055550/allreduce_nccl_1M_1G.log`
