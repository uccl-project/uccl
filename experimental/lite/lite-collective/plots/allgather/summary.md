# Lite AllGather vs NCCL Performance

Date: 2026-06-09

All rows use out-of-place nccl-tests results. Ratios greater than `1.00x` favor Lite: latency speedup is `NCCL us / Lite us`, and bus bandwidth ratio is `Lite GB/s / NCCL GB/s`.

## Benchmark settings and sources

| Setting | Value |
| --- | --- |
| Iterations | `50` |
| Warmup | `20` |
| Size convention | latency: `128B-1MiB`; bus bandwidth: `1MiB-1GiB` |
| Multi-node hosts | `10.10.55.1,10.10.55.2` |
| 1nx4g host baseline | NCCL SHM-only: `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=0` |
| 2nx NCCL baseline | Per-size tuned NCCL envelope from `.tmp/collective-benchmarks/ag-nccl-tune-20260609-022425/` |
| Refactor validation | `.tmp/collective-benchmarks/ag-refactor-validate-20260609-103556/` |
| Generated plots | `plots/allgather/` |

All plotted source rows reported `#wrong=0`.

## Summary

| Setup | Latency geomean Lite speedup | BusBW geomean Lite/NCCL | 1GiB Lite GB/s | 1GiB NCCL GB/s | 1GiB ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1nx4g_host` | 1.14x | 1.08x | 17.17 | 16.43 | 1.05x |
| `1nx4g_cudaipc` | 1.00x | 1.14x | 19.95 | 16.46 | 1.21x |
| `2nx1g` | 1.04x | 1.32x | 17.55 | 12.89 | 1.36x |
| `2nx2g` | 1.04x | 1.23x | 20.62 | 15.54 | 1.33x |
| `2nx4g` | 1.18x | 1.17x | 18.64 | 15.73 | 1.18x |

## 1nx4g host-memory (no CudaIpc payload) vs NCCL SHM-only

### Latency, 128B-1MiB (us)

| Size | Lite us | NCCL us | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 8.36 | 9.02 | 1.08x |
| 256B | 8.34 | 8.89 | 1.07x |
| 512B | 8.31 | 9.03 | 1.09x |
| 1KiB | 8.39 | 9.14 | 1.09x |
| 2KiB | 9.03 | 9.21 | 1.02x |
| 4KiB | 10.46 | 9.69 | 0.93x |
| 8KiB | 10.86 | 9.84 | 0.91x |
| 16KiB | 11.03 | 10.75 | 0.97x |
| 32KiB | 11.82 | 11.71 | 0.99x |
| 64KiB | 12.99 | 14.57 | 1.12x |
| 128KiB | 15.92 | 20.82 | 1.31x |
| 256KiB | 22.12 | 34.35 | 1.55x |
| 512KiB | 34.83 | 59.72 | 1.71x |
| 1MiB | 60.34 | 82.92 | 1.37x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite GB/s | NCCL GB/s | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 13.03 | 9.48 | 1.37x |
| 2MiB | 14.11 | 12.50 | 1.13x |
| 4MiB | 14.69 | 14.52 | 1.01x |
| 8MiB | 15.91 | 15.77 | 1.01x |
| 16MiB | 17.18 | 16.04 | 1.07x |
| 32MiB | 16.94 | 16.26 | 1.04x |
| 64MiB | 17.32 | 16.37 | 1.06x |
| 128MiB | 17.58 | 16.41 | 1.07x |
| 256MiB | 17.05 | 16.43 | 1.04x |
| 512MiB | 17.13 | 16.43 | 1.04x |
| 1GiB | 17.17 | 16.43 | 1.05x |

Plots: [`1nx4g_host_latency_128B_1MiB.png`](../plots/allgather/1nx4g_host_latency_128B_1MiB.png), [`1nx4g_host_busbw_1MiB_1GiB.png`](../plots/allgather/1nx4g_host_busbw_1MiB_1GiB.png)

## 1nx4g CudaIpc vs NCCL

### Latency, 128B-1MiB (us)

| Size | Lite us | NCCL us | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 8.68 | 9.92 | 1.14x |
| 256B | 8.85 | 8.89 | 1.00x |
| 512B | 9.11 | 8.99 | 0.99x |
| 1KiB | 9.15 | 9.13 | 1.00x |
| 2KiB | 9.21 | 9.19 | 1.00x |
| 4KiB | 9.38 | 9.34 | 1.00x |
| 8KiB | 9.79 | 9.92 | 1.01x |
| 16KiB | 10.76 | 10.67 | 0.99x |
| 32KiB | 13.88 | 11.54 | 0.83x |
| 64KiB | 14.58 | 14.49 | 0.99x |
| 128KiB | 20.80 | 20.70 | 1.00x |
| 256KiB | 33.63 | 34.39 | 1.02x |
| 512KiB | 60.08 | 59.50 | 0.99x |
| 1MiB | 82.88 | 82.34 | 0.99x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite GB/s | NCCL GB/s | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 9.49 | 9.55 | 0.99x |
| 2MiB | 12.52 | 12.52 | 1.00x |
| 4MiB | 14.55 | 14.57 | 1.00x |
| 8MiB | 17.78 | 15.71 | 1.13x |
| 16MiB | 19.06 | 16.01 | 1.19x |
| 32MiB | 19.72 | 16.29 | 1.21x |
| 64MiB | 19.99 | 16.35 | 1.22x |
| 128MiB | 19.92 | 16.43 | 1.21x |
| 256MiB | 19.89 | 16.43 | 1.21x |
| 512MiB | 19.92 | 16.45 | 1.21x |
| 1GiB | 19.95 | 16.46 | 1.21x |

Plots: [`1nx4g_cudaipc_latency_128B_1MiB.png`](../plots/allgather/1nx4g_cudaipc_latency_128B_1MiB.png), [`1nx4g_cudaipc_busbw_1MiB_1GiB.png`](../plots/allgather/1nx4g_cudaipc_busbw_1MiB_1GiB.png)

## 2nx1g

### Latency, 128B-1MiB (us)

| Size | Lite us | NCCL us | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 9.42 | 8.80 | 0.93x |
| 256B | 8.73 | 8.90 | 1.02x |
| 512B | 8.81 | 9.07 | 1.03x |
| 1KiB | 8.50 | 9.14 | 1.08x |
| 2KiB | 8.53 | 9.34 | 1.09x |
| 4KiB | 10.05 | 9.75 | 0.97x |
| 8KiB | 9.82 | 10.65 | 1.08x |
| 16KiB | 9.99 | 12.35 | 1.24x |
| 32KiB | 13.98 | 14.24 | 1.02x |
| 64KiB | 17.13 | 18.41 | 1.07x |
| 128KiB | 22.90 | 26.68 | 1.17x |
| 256KiB | 36.05 | 38.02 | 1.05x |
| 512KiB | 46.38 | 45.51 | 0.98x |
| 1MiB | 72.96 | 67.69 | 0.93x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite GB/s | NCCL GB/s | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 7.19 | 7.75 | 0.93x |
| 2MiB | 11.05 | 9.70 | 1.14x |
| 4MiB | 14.04 | 11.12 | 1.26x |
| 8MiB | 17.27 | 12.02 | 1.44x |
| 16MiB | 18.71 | 12.50 | 1.50x |
| 32MiB | 18.77 | 12.74 | 1.47x |
| 64MiB | 18.19 | 12.82 | 1.42x |
| 128MiB | 17.82 | 12.86 | 1.39x |
| 256MiB | 17.69 | 12.88 | 1.37x |
| 512MiB | 17.60 | 12.89 | 1.37x |
| 1GiB | 17.55 | 12.89 | 1.36x |

Plots: [`2nx1g_latency_128B_1MiB.png`](../plots/allgather/2nx1g_latency_128B_1MiB.png), [`2nx1g_busbw_1MiB_1GiB.png`](../plots/allgather/2nx1g_busbw_1MiB_1GiB.png)

## 2nx2g

### Latency, 128B-1MiB (us)

| Size | Lite us | NCCL us | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 11.35 | 12.07 | 1.06x |
| 256B | 11.39 | 12.19 | 1.07x |
| 512B | 11.31 | 12.36 | 1.09x |
| 1KiB | 11.36 | 12.54 | 1.10x |
| 2KiB | 11.21 | 12.59 | 1.12x |
| 4KiB | 11.26 | 12.96 | 1.15x |
| 8KiB | 15.58 | 13.55 | 0.87x |
| 16KiB | 24.51 | 14.86 | 0.61x |
| 32KiB | 16.60 | 16.28 | 0.98x |
| 64KiB | 17.35 | 20.66 | 1.19x |
| 128KiB | 29.53 | 29.77 | 1.01x |
| 256KiB | 33.26 | 45.48 | 1.37x |
| 512KiB | 54.35 | 64.60 | 1.19x |
| 1MiB | 84.00 | 80.82 | 0.96x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite GB/s | NCCL GB/s | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 9.36 | 9.73 | 0.96x |
| 2MiB | 12.61 | 11.99 | 1.05x |
| 4MiB | 15.67 | 14.05 | 1.12x |
| 8MiB | 18.17 | 15.01 | 1.21x |
| 16MiB | 19.68 | 15.26 | 1.29x |
| 32MiB | 20.59 | 15.37 | 1.34x |
| 64MiB | 20.52 | 15.44 | 1.33x |
| 128MiB | 20.80 | 15.51 | 1.34x |
| 256MiB | 20.72 | 15.49 | 1.34x |
| 512MiB | 20.66 | 15.52 | 1.33x |
| 1GiB | 20.62 | 15.54 | 1.33x |

Plots: [`2nx2g_latency_128B_1MiB.png`](../plots/allgather/2nx2g_latency_128B_1MiB.png), [`2nx2g_busbw_1MiB_1GiB.png`](../plots/allgather/2nx2g_busbw_1MiB_1GiB.png)

## 2nx4g

### Latency, 128B-1MiB (us)

| Size | Lite us | NCCL us | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 15.24 | 16.99 | 1.11x |
| 256B | 14.62 | 16.44 | 1.12x |
| 512B | 15.28 | 17.20 | 1.13x |
| 1KiB | 15.65 | 17.39 | 1.11x |
| 2KiB | 16.20 | 17.55 | 1.08x |
| 4KiB | 16.54 | 17.86 | 1.08x |
| 8KiB | 17.63 | 18.46 | 1.05x |
| 16KiB | 20.19 | 19.73 | 0.98x |
| 32KiB | 17.10 | 21.57 | 1.26x |
| 64KiB | 18.40 | 23.55 | 1.28x |
| 128KiB | 28.42 | 31.34 | 1.10x |
| 256KiB | 38.60 | 51.06 | 1.32x |
| 512KiB | 53.35 | 86.82 | 1.63x |
| 1MiB | 79.22 | 109.42 | 1.38x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite GB/s | NCCL GB/s | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 11.58 | 8.39 | 1.38x |
| 2MiB | 13.57 | 12.23 | 1.11x |
| 4MiB | 16.43 | 14.36 | 1.14x |
| 8MiB | 17.53 | 16.04 | 1.09x |
| 16MiB | 18.06 | 16.28 | 1.11x |
| 32MiB | 18.08 | 15.70 | 1.15x |
| 64MiB | 18.00 | 15.71 | 1.15x |
| 128MiB | 18.69 | 15.77 | 1.19x |
| 256MiB | 18.62 | 15.80 | 1.18x |
| 512MiB | 18.59 | 15.79 | 1.18x |
| 1GiB | 18.64 | 15.73 | 1.18x |

Plots: [`2nx4g_latency_128B_1MiB.png`](../plots/allgather/2nx4g_latency_128B_1MiB.png), [`2nx4g_busbw_1MiB_1GiB.png`](../plots/allgather/2nx4g_busbw_1MiB_1GiB.png)

## Plot files

- `1nx4g_host_latency_128B_1MiB.png`
- `1nx4g_host_busbw_1MiB_1GiB.png`
- `1nx4g_cudaipc_latency_128B_1MiB.png`
- `1nx4g_cudaipc_busbw_1MiB_1GiB.png`
- `2nx1g_latency_128B_1MiB.png`
- `2nx1g_busbw_1MiB_1GiB.png`
- `2nx2g_latency_128B_1MiB.png`
- `2nx2g_busbw_1MiB_1GiB.png`
- `2nx4g_latency_128B_1MiB.png`
- `2nx4g_busbw_1MiB_1GiB.png`
