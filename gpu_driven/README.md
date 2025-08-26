# UCCL GPU-Driven Peer-to-Peer Engine

An efficient prototype that demonstrates **end-to-end GPU-direct peer-to-peer (P2P) data communication** across machines using **GPUDirect RDMA** and lightweight **CPU proxies**.  

For UCCL's host/CPU-driven P2P engine, see [p2p](../p2p/) folder.

## Overview
1.	Each rank pins its GPU buffer with GPUDirect RDMA and exchanges RDMAConnectionInfo.
2.	Rank 0 writes batched commands into a host-mapped ring buffer managed by local CPU proxy.
3.	The CPU proxy of that rank polls that ring, posts `IBV_WR_RDMA_WRITE_WITH_IMM`, and recycles WQEs on completion.
4.	Rank 1’s proxy (on the remote node) posts matching receives and funnels completed work into a peer-copy kernel (optional) that pushes data to additional GPUs through NVLink. This step mimicks the requirements in MoE models where a token can be routed to multiple experts on the remote node via NVLink for de-duplication.

## Install

Installing `gpu_driven` as a Python package:
```bash
./build_and_install.sh cuda gpudriven 3.11
```
Alternatively, 
```bash
make
```

## Benchmark
In `bench` folder:

```bash
# On **sender** node (rank 0)
python benchmark_remote.py --rank 1 --peer-ip <rank_1_ip> --size-mb 256

# On **receiver** node (rank 1)
python benchmark_remote.py --rank 0 --peer-ip <rank_0_ip> --size-mb 256 --wait-sec 2
```