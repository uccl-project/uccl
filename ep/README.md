# UCCL GPU-Driven Expert Parallelism Engine

GPU-driven communication (e.g., DeepEP) is the key to efficient and large-scale EP, but it cannot run on heterogeneous platforms in the public cloud due to tight coupling between GPU and NIC. UCCL-EP has exactly the same interface and functionality as [DeepEP](https://github.com/deepseek-ai/DeepEP), but allows you to run GPU-driven communication for MoE models on public clouds, such as AWS, with superior performance to the state-of-the-art. Our ultimate goal with UCCL-EP is to democratize EP for heterogeneous GPUs and NIC vendors, including AMD GPUs, Broadcom NICs, AMD Pensando NICs, and more. 

For UCCL's host/CPU-driven P2P engine, see [p2p](../p2p/) folder.

## Build on CUDA for testing

We provide a script to install dependencies (tested on p5en and p6-b200). 
```bash
# Under uccl/ep
./install_deps.sh
```

Then under a Python environment, you can directly build and install:
```bash
# Under uccl/ep
make -j install
```

Alternatively, you can build `uccl.ep` wheel using docker:
```bash
# Under uccl
bash build_and_install.sh cuda ep
```

## Build on ROCm for testing

You can build `uccl.ep` wheel for ROCm7 using docker:
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

Please refer to [bench/baseline](bench/baseline) for running more baselines including Torch, NVSHMEM, and pplx-kernels on EFA. 

## Results

### Normal kernels with NVLink and RDMA forwarding

#### On p5en

We test normal kernels on **8x H200 + 16x 200Gb/s EFA** with each GPU connected to two **200 Gb/s EFA RDMA** network cards.
We follow the **DeepSeek-V3 pretraining** configuration (4096 tokens per batch, 7168 hidden, top-4 groups, top-8 experts, FP8 dispatch and BF16 combine).

|   Type    | Dispatch #EP | Bottleneck bandwidth & latency | Combine #EP | Bottleneck bandwidth & latency |
|:---------:|:-------------:|:--------------------:|:------------:|:--------------------:|
| Intranode | 8  | 320 GB/s (NVLink), 500 µs | 8  | 319 GB/s (NVLink), 973 µs |
| Internode | 16 | 50 GB/s (RDMA), 1196 µs | 16 | 18 GB/s (RDMA), 6379 µs    |
| Internode | 24 | 53 GB/s (RDMA), 1633 µs | 24 | 26 GB/s (RDMA), 6365 µs    |
| Internode | 32 | 54 GB/s (RDMA), 2022 µs | 32 | 43 GB/s (RDMA), 4899 µs    |

#### On p6-b200

We test normal kernels on **8x B200 + 8x 400Gb/s EFA** with each GPU connected to a **400Gb/s EFA RDMA** network card.

|   Type    | Dispatch #EP | Bottleneck bandwidth & latency | Combine #EP | Bottleneck bandwidth & latency |
|:---------:|:-------------:|:--------------------:|:------------:|:--------------------:|
| Intranode | 8  | 280 GB/s (NVLink), 571 µs | 8  | 426 GB/s (NVLink), 727 µs |
| Internode | 16 | 53 GB/s (RDMA), 1141 µs | 16 | 60 GB/s (RDMA), 1965 µs    |
| Internode | 24 | 53 GB/s (RDMA), 1637 µs | 24 | 59 GB/s (RDMA), 2887 µs    |
| Internode | 32 | 53 GB/s (RDMA), 2072 µs | 32 | 57 GB/s (RDMA), 3724 µs    |

### Low-latency kernels with pure RDMA

#### On p5en

We test low-latency kernels on **8x H200 + 16x 200Gb/s EFA**, following a **DeepSeek-V3 inference** setting (128 tokens per batch, 7168 hidden, top-8 experts, FP8 dispatch / BF16 combine).

| Dispatch #EP | Latency | RDMA bandwidth | Combine #EP | Latency | RDMA bandwidth |
|:-------------:|:--------:|:---------------:|:------------:|:--------:|:---------------:|
| 16 | 226 µs | 36 GB/s | 16 | 293 µs | 48 GB/s |
| 24 | 386 µs | 20 GB/s | 24 | 580 µs | 26 GB/s |
| 32 | 465 µs | 16 GB/s | 32 | 694 µs | 25 GB/s |

#### On p6-b200

We test low-latency kernels on **8x B200 + 8x 400Gb/s EFA**.

| Dispatch #EP | Latency | RDMA bandwidth | Combine #EP | Latency | RDMA bandwidth |
|:-------------:|:--------:|:---------------:|:------------:|:--------:|:---------------:|
| 16 | 228 µs | 33 GB/s | 16 | 318 µs | 46 GB/s |
| 24 | 448 µs | 17 GB/s | 24 | 566 µs | 26 GB/s |
| 32 | 406 µs | 19 GB/s | 32 | 617 µs | 24 GB/s |
