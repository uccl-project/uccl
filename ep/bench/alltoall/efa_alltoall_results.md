# RDMA All-to-All Throughput Results

Both tests keep the same number of total messages sent by each GPU, and the same max inflight messaages sent by each GPU. The difference is that in cross-rail case, one GPU's peers are any GPUs in other nodes; in rail-aligned case, one GPU's peers are only rail-aligned GPUs in other nodes. 

### RDMA cross-rail all-to-all results (Unit: GB/s)

| Message Size (KB) | 2 Nodes |
|-------------------|---------|
| 8                 | 6.1     |
| 16                |     |
| 32                |     |
| 64                |     |
| 128               |     |
| 256               |     |
| 512               |     |
| 1024              |     |
| 2048              |     |
| 4096              |     |

---

### RDMA rail-aligned all-to-all results (Unit: GB/s)

| Message Size (KB) | 2 Nodes |
| ----------------- | ------- |
| 8                 | 13.8    |
| 16                |     |
| 32                |     |
| 64                |     |
| 128               |     |
| 256               |     |
| 512               |     |
| 1024              |     |
| 2048              |     |
| 4096              |     |

### To reproduce

Run [test_alltoall_rail.cpp](./test_alltoall_rail.cpp) and [test_alltoall.cpp](test_alltoall.cpp).

