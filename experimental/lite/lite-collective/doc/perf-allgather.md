# Lite AllGather vs NCCL Performance

Date: 2026-06-08

Code state: current worktree with the AllGather `rdmaReady` signal changed to
RDMA atomic update. All rows below use out-of-place results; in-place results
were similar and are kept in the raw logs.

Benchmark settings:

| Setting | Value |
| --- | --- |
| Hosts | `10.10.55.1,10.10.55.2` |
| Iterations | `50` |
| Warmup | `20` |
| Size range | `128B` to `1GiB` |
| Interface | `NCCL_SOCKET_IFNAME=ibp55s0f0`, `MSCCLPP_SOCKET_IFNAME=ibp55s0f0` |
| HCAs | `NCCL_IB_HCA=mlx5_0,mlx5_1`, `MSCCLPP_HCA_DEVICES=mlx5_0,mlx5_1` |
| NCCL no-GDR | `NCCL_NET_GDR_LEVEL=0` |
| NCCL buffer | `NCCL_BUFFSIZE=4194304` |
| Raw logs | `.tmp/collective-benchmarks/ag-perf-allgather-20260608-164008/` |

Ratios greater than `1.00x` favor Lite. Latency speedup is `NCCL us / Lite us`;
bus bandwidth ratio is `Lite GB/s / NCCL GB/s`. All benchmark rows reported
`#wrong=0`.

## 2nx1g

### Latency, 128B-1MiB (us)

| Size | Lite | NCCL | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 9.54 | 9.09 | 0.95x |
| 256B | 8.66 | 8.99 | 1.04x |
| 512B | 8.82 | 9.00 | 1.02x |
| 1KiB | 8.78 | 9.13 | 1.04x |
| 2KiB | 8.93 | 9.31 | 1.04x |
| 4KiB | 10.29 | 9.74 | 0.95x |
| 8KiB | 9.72 | 10.71 | 1.10x |
| 16KiB | 10.13 | 12.34 | 1.22x |
| 32KiB | 14.07 | 14.37 | 1.02x |
| 64KiB | 18.07 | 18.56 | 1.03x |
| 128KiB | 24.02 | 28.35 | 1.18x |
| 256KiB | 37.01 | 48.48 | 1.31x |
| 512KiB | 47.62 | 54.49 | 1.14x |
| 1MiB | 71.93 | 96.35 | 1.34x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite | NCCL | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 7.29 | 5.44 | 1.34x |
| 2MiB | 11.28 | 9.67 | 1.17x |
| 4MiB | 14.05 | 11.10 | 1.27x |
| 8MiB | 16.99 | 12.00 | 1.42x |
| 16MiB | 18.66 | 12.22 | 1.53x |
| 32MiB | 18.69 | 12.30 | 1.52x |
| 64MiB | 18.18 | 12.37 | 1.47x |
| 128MiB | 17.84 | 12.40 | 1.44x |
| 256MiB | 17.68 | 12.42 | 1.42x |
| 512MiB | 17.60 | 12.43 | 1.42x |
| 1GiB | 17.55 | 12.43 | 1.41x |

## 2nx2g

### Latency, 128B-1MiB (us)

| Size | Lite | NCCL | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 11.35 | 12.43 | 1.10x |
| 256B | 11.38 | 12.30 | 1.08x |
| 512B | 11.47 | 12.39 | 1.08x |
| 1KiB | 11.23 | 12.56 | 1.12x |
| 2KiB | 11.12 | 12.64 | 1.14x |
| 4KiB | 11.40 | 12.95 | 1.14x |
| 8KiB | 15.62 | 13.64 | 0.87x |
| 16KiB | 22.15 | 14.88 | 0.67x |
| 32KiB | 16.76 | 16.45 | 0.98x |
| 64KiB | 17.27 | 21.65 | 1.25x |
| 128KiB | 28.28 | 29.97 | 1.06x |
| 256KiB | 32.93 | 48.16 | 1.46x |
| 512KiB | 51.53 | 109.35 | 2.12x |
| 1MiB | 85.38 | 88.79 | 1.04x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite | NCCL | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 9.21 | 8.86 | 1.04x |
| 2MiB | 12.50 | 11.95 | 1.05x |
| 4MiB | 15.47 | 13.99 | 1.11x |
| 8MiB | 17.95 | 14.95 | 1.20x |
| 16MiB | 19.37 | 15.00 | 1.29x |
| 32MiB | 20.25 | 15.12 | 1.34x |
| 64MiB | 20.23 | 15.18 | 1.33x |
| 128MiB | 20.50 | 15.23 | 1.35x |
| 256MiB | 20.60 | 15.25 | 1.35x |
| 512MiB | 20.58 | 15.26 | 1.35x |
| 1GiB | 20.55 | 15.26 | 1.35x |

## 2nx4g

### Latency, 128B-1MiB (us)

| Size | Lite | NCCL | Lite speedup |
| ---: | ---: | ---: | ---: |
| 128B | 15.31 | 17.43 | 1.14x |
| 256B | 14.96 | 17.42 | 1.16x |
| 512B | 15.65 | 17.55 | 1.12x |
| 1KiB | 15.64 | 17.91 | 1.15x |
| 2KiB | 16.00 | 18.23 | 1.14x |
| 4KiB | 16.81 | 18.38 | 1.09x |
| 8KiB | 17.70 | 18.99 | 1.07x |
| 16KiB | 20.43 | 20.04 | 0.98x |
| 32KiB | 17.00 | 22.47 | 1.32x |
| 64KiB | 17.99 | 24.51 | 1.36x |
| 128KiB | 28.40 | 33.87 | 1.19x |
| 256KiB | 38.72 | 70.26 | 1.81x |
| 512KiB | 53.43 | 294.83 | 5.52x |
| 1MiB | 79.62 | 118.55 | 1.49x |

### Bus bandwidth, 1MiB-1GiB (GB/s)

| Size | Lite | NCCL | Lite / NCCL |
| ---: | ---: | ---: | ---: |
| 1MiB | 11.52 | 7.74 | 1.49x |
| 2MiB | 13.29 | 10.72 | 1.24x |
| 4MiB | 16.10 | 14.26 | 1.13x |
| 8MiB | 17.41 | 15.60 | 1.12x |
| 16MiB | 18.01 | 15.63 | 1.15x |
| 32MiB | 18.16 | 14.66 | 1.24x |
| 64MiB | 18.17 | 14.74 | 1.23x |
| 128MiB | 18.92 | 14.84 | 1.27x |
| 256MiB | 18.83 | 14.86 | 1.27x |
| 512MiB | 18.82 | 14.88 | 1.26x |
| 1GiB | 18.83 | 14.88 | 1.27x |
