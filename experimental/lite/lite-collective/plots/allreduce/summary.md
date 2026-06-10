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
| Clean-run idle checks | `.tmp/collective-benchmarks/ar-clean-plots-20260610-161719/idle_checks.log` |
| Generated plots | `plots/allreduce/` |

All plotted source rows reported `#wrong=0`.

## Summary

| Setup | Latency geomean Lite speedup | BusBW geomean Lite/NCCL | 1GiB Lite GB/s | 1GiB NCCL GB/s | 1GiB ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1nx4g` | 0.44x | 0.81x | 17.54 | 17.54 | 1.00x |
| `2nx1g` | 0.09x | 0.63x | 16.23 | 14.55 | 1.12x |
| `2nx2g` | 1.15x | 0.91x | 14.71 | 15.34 | 0.96x |
| `2nx4g` | 1.58x | 1.02x | 16.28 | 14.82 | 1.10x |

## 1nx4g

### Latency, 128B-1MiB (us)

| Size | Lite us | NCCL us | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 5.84 | 12.62 | 2.16x |
| 256B | 5.98 | 12.51 | 2.09x |
| 512B | 6.97 | 12.91 | 1.85x |
| 1KiB | 9.24 | 13.04 | 1.41x |
| 2KiB | 14.50 | 13.07 | 0.90x |
| 4KiB | 24.00 | 13.41 | 0.56x |
| 8KiB | 41.62 | 14.25 | 0.34x |
| 16KiB | 79.24 | 15.57 | 0.20x |
| 32KiB | 84.99 | 16.93 | 0.20x |
| 64KiB | 158.25 | 22.87 | 0.14x |
| 128KiB | 160.35 | 35.30 | 0.22x |
| 256KiB | 277.11 | 63.92 | 0.23x |
| 512KiB | 616.34 | 93.27 | 0.15x |
| 1MiB | 1309.10 | 136.12 | 0.10x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite GB/s | NCCL GB/s | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 1.20 | 11.55 | 0.10x |
| 2MiB | 14.12 | 14.07 | 1.00x |
| 4MiB | 15.88 | 15.84 | 1.00x |
| 8MiB | 16.84 | 16.88 | 1.00x |
| 16MiB | 17.13 | 17.20 | 1.00x |
| 32MiB | 17.21 | 17.31 | 0.99x |
| 64MiB | 17.39 | 17.40 | 1.00x |
| 128MiB | 17.46 | 17.47 | 1.00x |
| 256MiB | 17.54 | 17.53 | 1.00x |
| 512MiB | 17.59 | 17.57 | 1.00x |
| 1GiB | 17.54 | 17.54 | 1.00x |

Plots: [`1nx4g_latency_128B_1MiB.png`](1nx4g_latency_128B_1MiB.png), [`1nx4g_busbw_1MiB_1GiB.png`](1nx4g_busbw_1MiB_1GiB.png)

## 2nx1g

### Latency, 128B-1MiB (us)

| Size | Lite us | NCCL us | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 8420.67 | 12.13 | 0.00x |
| 256B | 157.40 | 12.15 | 0.08x |
| 512B | 159.17 | 12.23 | 0.08x |
| 1KiB | 160.80 | 12.40 | 0.08x |
| 2KiB | 159.80 | 12.94 | 0.08x |
| 4KiB | 166.07 | 13.61 | 0.08x |
| 8KiB | 166.22 | 14.82 | 0.09x |
| 16KiB | 164.39 | 16.93 | 0.10x |
| 32KiB | 257.37 | 21.06 | 0.08x |
| 64KiB | 176.07 | 28.68 | 0.16x |
| 128KiB | 286.35 | 46.58 | 0.16x |
| 256KiB | 194.65 | 83.81 | 0.43x |
| 512KiB | 397.07 | 159.54 | 0.40x |
| 1MiB | 387.44 | 105.65 | 0.27x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite GB/s | NCCL GB/s | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 2.71 | 9.92 | 0.27x |
| 2MiB | 5.58 | 12.02 | 0.46x |
| 4MiB | 8.58 | 13.33 | 0.64x |
| 8MiB | 9.07 | 14.14 | 0.64x |
| 16MiB | 7.69 | 14.31 | 0.54x |
| 32MiB | 1.98 | 14.38 | 0.14x |
| 64MiB | 15.02 | 14.44 | 1.04x |
| 128MiB | 15.62 | 14.50 | 1.08x |
| 256MiB | 15.97 | 14.53 | 1.10x |
| 512MiB | 16.15 | 14.51 | 1.11x |
| 1GiB | 16.23 | 14.55 | 1.12x |

Plots: [`2nx1g_latency_128B_1MiB.png`](2nx1g_latency_128B_1MiB.png), [`2nx1g_busbw_1MiB_1GiB.png`](2nx1g_busbw_1MiB_1GiB.png)

## 2nx2g

### Latency, 128B-1MiB (us)

| Size | Lite us | NCCL us | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 11.97 | 18.15 | 1.52x |
| 256B | 11.71 | 17.89 | 1.53x |
| 512B | 11.88 | 18.29 | 1.54x |
| 1KiB | 11.95 | 18.61 | 1.56x |
| 2KiB | 12.66 | 18.83 | 1.49x |
| 4KiB | 13.12 | 19.37 | 1.48x |
| 8KiB | 15.75 | 20.69 | 1.31x |
| 16KiB | 20.52 | 22.50 | 1.10x |
| 32KiB | 28.31 | 24.99 | 0.88x |
| 64KiB | 44.00 | 34.74 | 0.79x |
| 128KiB | 77.98 | 52.58 | 0.67x |
| 256KiB | 102.11 | 89.47 | 0.88x |
| 512KiB | 132.96 | 173.75 | 1.31x |
| 1MiB | 196.51 | 152.52 | 0.78x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite GB/s | NCCL GB/s | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 8.00 | 10.31 | 0.78x |
| 2MiB | 10.35 | 13.30 | 0.78x |
| 4MiB | 12.25 | 14.92 | 0.82x |
| 8MiB | 13.80 | 15.18 | 0.91x |
| 16MiB | 14.22 | 14.92 | 0.95x |
| 32MiB | 14.43 | 15.02 | 0.96x |
| 64MiB | 14.52 | 15.10 | 0.96x |
| 128MiB | 14.62 | 15.18 | 0.96x |
| 256MiB | 14.67 | 15.23 | 0.96x |
| 512MiB | 14.70 | 15.28 | 0.96x |
| 1GiB | 14.71 | 15.34 | 0.96x |

Plots: [`2nx2g_latency_128B_1MiB.png`](2nx2g_latency_128B_1MiB.png), [`2nx2g_busbw_1MiB_1GiB.png`](2nx2g_busbw_1MiB_1GiB.png)

## 2nx4g

### Latency, 128B-1MiB (us)

| Size | Lite us | NCCL us | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 12.04 | 21.42 | 1.78x |
| 256B | 12.11 | 21.13 | 1.74x |
| 512B | 12.27 | 21.58 | 1.76x |
| 1KiB | 12.30 | 22.36 | 1.82x |
| 2KiB | 12.77 | 24.72 | 1.94x |
| 4KiB | 13.93 | 31.30 | 2.25x |
| 8KiB | 17.02 | 32.35 | 1.90x |
| 16KiB | 22.27 | 34.32 | 1.54x |
| 32KiB | 31.58 | 37.80 | 1.20x |
| 64KiB | 57.24 | 41.77 | 0.73x |
| 128KiB | 80.55 | 60.94 | 0.76x |
| 256KiB | 120.70 | 100.38 | 0.83x |
| 512KiB | 138.88 | 417.54 | 3.01x |
| 1MiB | 196.91 | 582.82 | 2.96x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite GB/s | NCCL GB/s | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 9.32 | 3.15 | 2.96x |
| 2MiB | 11.13 | 11.36 | 0.98x |
| 4MiB | 13.42 | 14.69 | 0.91x |
| 8MiB | 14.47 | 15.90 | 0.91x |
| 16MiB | 15.07 | 15.68 | 0.96x |
| 32MiB | 7.57 | 14.51 | 0.52x |
| 64MiB | 10.09 | 14.65 | 0.69x |
| 128MiB | 16.28 | 14.74 | 1.10x |
| 256MiB | 16.28 | 14.81 | 1.10x |
| 512MiB | 16.27 | 14.80 | 1.10x |
| 1GiB | 16.28 | 14.82 | 1.10x |

Plots: [`2nx4g_latency_128B_1MiB.png`](2nx4g_latency_128B_1MiB.png), [`2nx4g_busbw_1MiB_1GiB.png`](2nx4g_busbw_1MiB_1GiB.png)

## Source logs

### 1nx4g
- lite: `.tmp/collective-benchmarks/ar-clean-plots-20260610-161719/lite_1nx4g_128B_1G.log`
- nccl: `.tmp/collective-benchmarks/ar-clean-plots-20260610-161719/nccl_1nx4g_128B_1G.log`

### 2nx1g
- lite: `.tmp/collective-benchmarks/ar-clean-plots-20260610-161719/lite_2nx1g_128B_1G.log`
- nccl: `.tmp/collective-benchmarks/ar-clean-plots-20260610-161719/nccl_2nx1g_128B_1G.log`

### 2nx2g
- lite: `.tmp/collective-benchmarks/ar-clean-plots-20260610-161719/lite_2nx2g_128B_1G.log`
- nccl: `.tmp/collective-benchmarks/ar-clean-plots-20260610-161719/nccl_2nx2g_128B_1G.log`

### 2nx4g
- lite: `.tmp/collective-benchmarks/ar-clean-plots-20260610-161719/lite_2nx4g_128B_1G.log`
- nccl: `.tmp/collective-benchmarks/ar-clean-plots-20260610-161719/nccl_2nx4g_128B_1G.log`
