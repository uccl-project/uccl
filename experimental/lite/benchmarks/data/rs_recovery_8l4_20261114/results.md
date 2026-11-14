# Full-range ReduceScatter confirmation

Hardware: wxz-l4, eight NVIDIA L4 GPUs, two NUMA nodes (ranks 0–3 and 4–7). CUDA 12.9, driver 580.178.04, NCCL 2.28.9. Previous means the fixed pre-recovery selection from 5b9a7f86, measured again in the same executable. GPU mode uses default NCCL P2P policy; SHM explicitly disables P2P. Both use mapped-host UCCL transport.

BF16, out of place. Complete input bytes per GPU. Two opposite-order sweeps, each with three repetitions of five warmups and ten timed calls per size/backend. Median of the two sweep medians; each sweep median is the median of its three repetition medians. Slowest-GPU input-ready CUDA events include enqueue gaps. Every output is checked. No per-size winner selection: current fixes the ring at 1–8 MiB and retains previous algorithms elsewhere.

| Mode | Complete bytes/GPU | Previous µs | Current µs | Paired NCCL µs | NCCL/current |
| --- | ---: | ---: | ---: | ---: | ---: |
| gpu | 1024 | 52.99 | 53.25 | 93.18 | 1.750× |
| gpu | 2048 | 52.74 | 53.50 | 93.68 | 1.751× |
| gpu | 4096 | 54.02 | 53.74 | 94.98 | 1.767× |
| gpu | 8192 | 53.50 | 54.25 | 96.77 | 1.784× |
| gpu | 16384 | 55.30 | 54.77 | 101.35 | 1.851× |
| gpu | 32768 | 58.86 | 59.14 | 107.01 | 1.810× |
| gpu | 65536 | 74.22 | 74.24 | 118.53 | 1.597× |
| gpu | 131072 | 107.52 | 106.99 | 140.54 | 1.314× |
| gpu | 262144 | 147.46 | 146.18 | 189.93 | 1.299× |
| gpu | 524288 | 199.17 | 197.12 | 254.98 | 1.294× |
| gpu | 1048576 | 310.27 | 281.09 | 338.43 | 1.204× |
| gpu | 2097152 | 543.74 | 464.64 | 571.12 | 1.229× |
| gpu | 4194304 | 937.22 | 835.58 | 1033.98 | 1.237× |
| gpu | 8388608 | 1682.94 | 1576.43 | 1943.54 | 1.233× |
| gpu | 16777216 | 3036.42 | 3036.42 | 3759.36 | 1.238× |
| gpu | 33554432 | 5672.96 | 5729.28 | 7407.36 | 1.293× |
| gpu | 67108864 | 11019.01 | 10963.19 | 14649.86 | 1.336× |
| gpu | 134217728 | 21641.22 | 21676.03 | 29058.30 | 1.341× |
| gpu | 268435456 | 43284.22 | 43281.92 | 57636.35 | 1.332× |
| gpu | 536870912 | 86572.28 | 86530.05 | 114639.62 | 1.325× |
| gpu | 1073741824 | 173152.26 | 173050.88 | 227950.85 | 1.317× |
| shm | 1024 | 51.46 | 51.43 | 94.72 | 1.842× |
| shm | 2048 | 51.95 | 51.71 | 95.74 | 1.851× |
| shm | 4096 | 51.71 | 51.97 | 92.42 | 1.778× |
| shm | 8192 | 52.46 | 52.98 | 94.98 | 1.793× |
| shm | 16384 | 55.02 | 54.27 | 99.07 | 1.825× |
| shm | 32768 | 57.34 | 57.86 | 108.29 | 1.872× |
| shm | 65536 | 72.93 | 73.72 | 116.99 | 1.587× |
| shm | 131072 | 107.22 | 106.24 | 140.28 | 1.320× |
| shm | 262144 | 144.90 | 145.92 | 189.68 | 1.300× |
| shm | 524288 | 193.02 | 193.77 | 250.87 | 1.295× |
| shm | 1048576 | 307.71 | 281.09 | 340.99 | 1.213× |
| shm | 2097152 | 546.82 | 464.64 | 580.10 | 1.248× |
| shm | 4194304 | 927.49 | 833.79 | 1036.26 | 1.243× |
| shm | 8388608 | 1682.43 | 1574.91 | 1942.02 | 1.233× |
| shm | 16777216 | 3021.56 | 3027.97 | 3766.27 | 1.244× |
| shm | 33554432 | 5725.70 | 5721.60 | 7441.66 | 1.301× |
| shm | 67108864 | 10952.19 | 10983.42 | 14678.02 | 1.336× |
| shm | 134217728 | 21651.46 | 21643.52 | 29121.54 | 1.346× |
| shm | 268435456 | 43240.45 | 43288.58 | 57708.80 | 1.333× |
| shm | 536870912 | 86498.30 | 86418.17 | 114904.05 | 1.330× |
| shm | 1073741824 | 173203.45 | 173032.96 | 229218.30 | 1.325× |
