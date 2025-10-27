# UCCL GPU-Driven Expert Parallelism Engine

GPU-driven communication (e.g., DeepEP) is the key to efficient and large-scale EP, but it cannot run on heterogeneous platforms in the public cloud due to tight coupling between GPU and NIC. UCCL-EP has exactly the same interface and functionality as [DeepEP](https://github.com/deepseek-ai/DeepEP), but allows you to run GPU-driven communication for MoE models on public clouds, such as AWS, with superior performance to the state-of-the-art. Our ultimate goal with UCCL-EP is to democratize EP for heterogeneous GPUs and NIC vendors, including AMD GPUs, Broadcom NICs, AMD Pensando NICs, and more. 

For UCCL's host/CPU-driven P2P engine, see [p2p](../p2p/) folder.

## Build on CUDA for testing

We provide a script to install dependencies (tested on p5en). Then under a Python environment: 
```bash
# Under uccl/ep
./install_deps.sh
```

We first configure the environmental variable before `make install`
```bash
# Normal mode
export MAKE_NORMAL_MODE=1
# Low latency mode
export MAKE_NORMAL_MODE=0
```
Next, in a conda environment: 
```bash
make -j install
```

Alternatively, you can build `uccl.ep` wheel using docker:
```bash
# Under uccl
bash build_and_install.sh cuda ep
```

## Build on ROCm for testing

Build `uccl.ep` wheel for ROCm using docker:
```bash
# Under uccl
bash build_and_install.sh rocm ep

# Install rocm7 into local Python env
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0
```

Test import `uccl.ep`
```bash
python -c "import torch; import uccl.ep"
```

## Example APIs

Dispatch and combine: 
```python
packed_recv_x, packed_recv_count, handle, event, hook = buffer.low_latency_dispatch(
    current_x,
    topk_idx,
    num_tokens,
    num_experts,
    use_fp8=dispatch_use_fp8,
    round_scale=round_scale,
    use_ue8m0=use_ue8m0,
    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
    async_finish=not return_recv_hook,
    return_recv_hook=return_recv_hook,
)

combined_x, event, hook = buffer.low_latency_combine(
    simulated_gemm_x,
    topk_idx,
    topk_weights,
    handle,
    use_logfmt=use_logfmt,
    async_finish=not return_recv_hook,
    zero_copy=zero_copy,
    return_recv_hook=return_recv_hook,
    out=out,
)
```

Initialization and tear down:
```python
proxies, workers = initialize_uccl(scratch, num_rdma_bytes, rank, num_ranks, group, args.num_experts)
destroy_uccl(proxies, workers)
```

## Benchmark
In `ep` folder, the benchmark can be run with `torchrun`. 

### Intranode Test

```bash
torchrun --standalone --nproc_per_node=8 \
  bench/test_intranode.py --num-tokens 4096 \
  --hidden 7168 --num-topk 8 --num-experts 256
```

### Internode Low Latency Test

```bash
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=<rank> \
  --master_addr=<ip> --master_port=12355 \
  bench/test_low_latency.py --num-tokens=128 \
  --hidden=7168 --num-topk=8 --num-experts=288
```

### Internode Normal Mode (Throughput) Test

```bash
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=<rank> \
  --master_addr=<ip> --master_port=12355 \
  bench/test_internode.py  --num-tokens=4096 \
  --hidden=7168 --num-topk=8 --num-experts=288 --test-ll-compatibility
```

Please refer to [ep/bench/baseline/](ep/bench/baseline/) for more baselines including `torch.distributed`, NVSHMEM, and pplx-kernels. 

## Results

### Normal kernels with NVLink and RDMA forwarding

We test normal kernels on **H200 (8× GPUs per node)** with each node connected to an **EFA 400 Gb/s RDMA** network card.
We follow the **DeepSeek-V3 pretraining** configuration (4096 tokens per batch, 7168 hidden, top-4 groups, top-8 experts, FP8 dispatch and BF16 combine).

|   Type    | Dispatch #EP | Bottleneck bandwidth | Combine #EP | Bottleneck bandwidth |
|:---------:|:-------------:|:--------------------:|:------------:|:--------------------:|
| Intranode | 8  | 320 GB/s (NVLink) | 8  | 319 GB/s (NVLink) |
| Internode | 16 | 50 GB/s (RDMA)    | 16 | 18 GB/s (RDMA)    |
| Internode | 24 | 53 GB/s (RDMA)    | 24 | 26 GB/s (RDMA)    |
| Internode | 32 | 54 GB/s (RDMA)    | 32 | 43 GB/s (RDMA)    |

### Low-latency kernels with pure RDMA

We test low-latency kernels on **H200 (8× GPUs + EFA 400 Gb/s)** following a **DeepSeek-V3 inference** setting (128 tokens per batch, 7168 hidden, top-8 experts, FP8 dispatch / BF16 combine).

| Dispatch #EP | Latency | RDMA bandwidth | Combine #EP | Latency | RDMA bandwidth |
|:-------------:|:--------:|:---------------:|:------------:|:--------:|:---------------:|
| 16 | 226 µs | 36 GB/s | 16 | 293 µs | 48 GB/s |
| 24 | 386 µs | 20 GB/s | 24 | 580 µs | 26 GB/s |
| 32 | 465 µs | 16 GB/s | 32 | 694 µs | 25 GB/s |
