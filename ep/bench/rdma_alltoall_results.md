# RDMA All-to-All Throughput Results

Unit: **Gb/s**

| Message Size (KB) | 2 Nodes | 3 Nodes | 4 Nodes |
|-------------------|---------|---------|---------|
| 8                 | 4.8     | 8.6     | 10.9    |
| 16                | 11.2    | 13.1    | 15.3    |
| 32                | 17.9    | 19.9    | 20.3    |
| 64                | 24.8    | 23.2    | 22.7    |
| 128               | 30.3    | 27.2    | 24.0    |
| 256               | 34.5    | 27.9    | 24.6    |
| 512               | 37.8    | 28.3    | 24.8    |
| 1024              | 39.5    | 28.5    | 24.9    |
| 2048              | 40.7    | 28.7    | 24.9    |
| 4096              | 41.0    | 28.9    | 24.9    |

---

## Observations
- **2 Nodes**: Throughput increases steadily with message size and reaches ~41 Gb/s at around 2 MB, which is close to the peak of the platform.  
- **3 Nodes**: Throughput grows until ~32 KB, then flattens and stabilizes at ~29 Gb/s.  
- **4 Nodes**: Throughput saturates early (around 128 KB) and stays nearly constant at ~25 Gb/s.  
