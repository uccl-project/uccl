# RDMA All-to-All Throughput Results - Shared QP Comparison

Architecture: **One SRD QP per NIC** (connectionless mode with different address handlers per destination)

Unit: **GB/s**

## Nonrail-Optimized (Cross-Rail Traffic)

| Message Size (KB) | 2 Nodes | 3 Nodes | 4 Nodes |
|-------------------|---------|---------|---------|
| 8                 | 6.5     | 9.2     | 10.9    |
| 16                | 11.2    | 14.2    | 15.5    |
| 32                | 18.1    | 19.6    | 20.5    |
| 64                | 25.1    | 23.1    | 23.2    |
| 128               | 31.1    | 25.6    | 24.2    |
| 256               | 36.6    | 27.3    | 25.3    |
| 512               | 38.4    | 28.0    | 24.9    |
| 1024              | 40.1    | 28.2    | 25.0    |
| 2048              | 41.1    | 28.4    | 25.0    |
| 4096              | 41.5    | 28.5    | 25.0    |

---
## Rail-Optimized (No Cross-Rail Traffic)

| Message Size (KB) | 2 Nodes | 3 Nodes | 4 Nodes |
|-------------------|---------|---------|---------|
| 8                 | 4.1     | 7.6     | 10.3    |
| 16                | 4.2     | 7.7     | 10.1    |
| 32                | 7.8     | 13.2    | 16.9    |
| 64                | 12.7    | 19.9    | 24.1    |
| 128               | 18.8    | 26.8    | 30.2    |
| 256               | 24.2    | 32.3    | 33.1    |
| 512               | 31.0    | 34.6    | 35.3    |
| 1024              | 37.1    | 37.6    | 37.4    |
| 2048              | 38.5    | 39.6    | 39.5    |
| 4096              | 42.3    | 42.2    | 41.8    |

---


