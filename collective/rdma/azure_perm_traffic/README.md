# Azure Permutation Traffic Benchmark

This benchmark is done on Azure [HBv2/HBrsv2](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/high-performance-compute/hbv2-series) VM instances with microsoft-dsvm::ubuntu-hpc::2204 image. Each VM has a 200Gbps CX-6 IB vNIC and connects to others via a fully-provisioned fattree. Our results show that when running permutation traffic, there are always some VM pairs that cannot only reach ~100Gbps. 

## Configurating azure_perm_traffic/transport_config.h

`kServiceLevel`: Control the service level. Only `0-3` is valid in Azure.

`kRCMode`: Let UCCL use RC(true) or UC(false).

`kRCSize`, `NCCL_MIN_POST_RECV`: Set them to a very big value (e.g., `5000000`) to let UCCL fallback to single RC per flow like NCCL.

`ENGINE_CPU_START_LIST[0], NUM_ENGINES`: Control the engine CPU starts from. Currently, `1` and `3` is the best. UCCL will use `CPU 1,2,3`. The NIC is attached to `NUMA0 (CPU0-3)` and the `run.sh` script binds the application to core 0. Otherewise, the application can't reach 200Gbps. 

`kBypassPacing`: Timely pacing. Currently, we bypass timely pacing but a constant window limit: `kMaxUnAckedBytesPerFlow`, `kMaxUnAckedBytesPerEngineLow` and `kMaxUnAckedBytesPerEngineHigh` as it gives good performance than pacing.

## Run benchmark

0. Clone this repo on a master node.
1. Add all **public IPs** in `ips`. These IPs are `74/128.xxx.xxx.xxx`.
2. Run `setup_nodes.sh` to install environment dependencies on all nodes: cuda, mpi, etc.
3. Add all **eth0 IPs** in `hostname`. These IPs are `10.0.0.x`.
4. For Permutation Traffic, generate `matrix.txt` using the `gen_permutation_full_bisection.py` script (not needed for sequential alltoall and alltoall).
5. Run `sync_repo.sh` to copy this repo to all nodes. It will also compile `uccl/rdma` and `uccl/rdma/azure_perm_traffic`.
6. ​Run the test by executing:

```bash
./run.sh <number_of_nodes>
```