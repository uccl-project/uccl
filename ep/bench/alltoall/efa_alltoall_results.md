# RDMA All-to-All Throughput Results

Both tests keep the same number of total messages sent by each GPU, and the same max inflight messaages sent by each GPU. The difference is that in cross-rail case, one GPU's peers are any GPUs in other nodes; in rail-aligned case, one GPU's peers are only rail-aligned GPUs in other nodes. 

Message size of 8KB would roughly represent DeepEP/UCCL-EP low-latency performance. 

### RDMA cross-rail all-to-all results (Unit: GB/s)

| Message Size (KB) | 2 Nodes (p5) |
|-------------------|--------------|
| 8                 | 6.1          |
| 16                | 8.6          |
| 32                | 14.1         |
| 64                | 15.6         |
| 128               | 15.7         |
| 256               | 15.3         |
| 512               | 17.9         |
| 1024              | 19.0         |
| 2048              | 19.9         |
| 4096              | 20.1         |

---

### RDMA rail-aligned all-to-all results (Unit: GB/s)

| Message Size (KB) | 2 Nodes (p5) |
| ----------------- | ------------ |
| 8                 | 13.8         |
| 16                | 21.1         |
| 32                | 35.6         |
| 64                | 36.9         |
| 128               | 41.6         |
| 256               | 41.9         |
| 512               | 43.2         |
| 1024              | 43.5         |
| 2048              | 43.4         |
| 4096              | 43.7         |

### To reproduce

Run [test_alltoall_rail.cpp](./test_alltoall_rail.cpp) and [test_alltoall.cpp](test_alltoall.cpp).

