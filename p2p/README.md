# Minimal P2P Prototype

This prototype demonstrates a **GPU→CPU proxy→RDMA→NVLink** message path using:

* **Cuda‑allocated host‑pinned ring buffer** (single‑producer GPU, single‑consumer CPU).
* **GPU kernel** enqueues *transfer commands* asynchronously and returns immediately.
* **CPU proxy thread** polls the ring buffer and issues an RDMA WRITE (GPUDirect) to the remote node. 
* **Receiver proxy** (run as a separate process on the peer node) would copy the remotely‑written GPU buffer to other local GPUs.

> The RDMA section contains only a stub — integrate your own verb setup for real hardware.  
> The aim is to give you a *compilable* skeleton that you can extend.

## Build
```bash
make            
```

## Run (two nodes)
```bash
./benchmark_remote 0 192.168.0.100 # sender
./benchmark_remote 1 192.168.0.58 # receiver
```

The program prints the latency of pushing commands from the GPU. Extend the `rdma_write_stub()` and add receiver‑side logic to measure end‑to‑end transfer latency.
