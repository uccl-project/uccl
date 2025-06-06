# Minimal P2P Prototype

This prototype demonstrates a **GPU→CPU proxy→RDMA→NVLink** message path using:

* **Cuda‑allocated host‑pinned ring buffer** (single‑producer GPU, single‑consumer CPU).
* **GPU kernel** enqueues *transfer commands* asynchronously and returns immediately.
* **CPU proxy thread** polls the ring buffer and issues an RDMA WRITE (GPUDirect) to the remote node. When compiled with `NO_RDMA=1` it instead simulates the transfer locally.
* **Receiver proxy** (run as a separate process on the peer node) would copy the remotely‑written GPU buffer to other local GPUs via `cudaMemcpyPeerAsync`, showcasing the NVLink hop.

> The RDMA section contains only a stub — integrate your own verb setup for real hardware.  
> The aim is to give you a *compilable* skeleton that you can extend.

## Build
```bash
make            # with RDMA (needs libibverbs + GPUDirect)
make NO_RDMA=1  # without RDMA
```

## Run (two nodes)
```bash
# Node A (rank 0)
./benchmark 0 <nodeB_ip> 1048576

# Node B (rank 1)
./benchmark 1 <nodeA_ip> 1048576
```

The program prints the latency of pushing commands from the GPU. Extend the `rdma_write_stub()` and add receiver‑side logic to measure end‑to‑end transfer latency.
