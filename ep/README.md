# UCCL GPU-Driven Expert-parallelism Engine

UCCL EP engine provides a similar interface from [DeepEP](https://github.com/deepseek-ai/DeepEP). 

For UCCL's host/CPU-driven P2P engine, see [p2p](../p2p/) folder.

## Build on CUDA for testing

Installing `ep` as a Python package:
```bash
# under uccl
bash build_and_install.sh cuda ep
```

Alternatively, in a Python environment 
```bash
# under uccl/ep
make -j install
```

## Build on ROCm for testing

build rocm image
```bash
# requiring rocm7
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0
# under uccl
bash build_and_install.sh rocm ep
```

test import uccl.ep
```bash
python -c "import torch;import uccl.ep"
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
OMP_NUM_THREADS=8 torchrun \
  --standalone --nproc_per_node=8 bench/test_intranode.py \
  --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 256
```

### Internode Low Latency Test

```bash
OMP_NUM_THREADS=8 torchrun \
  --nnodes=3 --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=10.1.227.34 --master_port=12357 \
  bench/test_low_latency.py \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=288
```

### Internode Normal Mode (Throughput) Test

```bash
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=0 \
  --master_addr=ip --master_port=12355 \
 bench/test_internode.py  --num-tokens=4096 \
  --hidden=7168 --num-topk=8 --num-experts=288 --test-ll-compatibility
```

## Expected Results

### Normal kernels with NVLink and RDMA forwarding

We test normal kernels on **H200 (8× GPUs per node)** with each node connected to an **EFA 400 Gb/s RDMA** network card.
We follow the **DeepSeek-V3/R1 pretraining** configuration (4096 tokens per batch, 7168 hidden, top-4 groups, top-8 experts, FP8 dispatching and BF16 combining).

|   Type    | Dispatch #EP | Bottleneck bandwidth | Combine #EP | Bottleneck bandwidth |
|:---------:|:-------------:|:--------------------:|:------------:|:--------------------:|
| Intranode | 8  | 320 GB/s (NVLink) | 8  | 319 GB/s (NVLink) |
| Internode | 16 | 50 GB/s (RDMA)    | 16 | 18 GB/s (RDMA)    |
| Internode | 24 | 53 GB/s (RDMA)    | 24 | 26 GB/s (RDMA)    |
| Internode | 32 | 54 GB/s (RDMA)    | 32 | 43 GB/s (RDMA)    |

### Low-latency kernels with pure RDMA

We test low-latency kernels on **H200 (8× GPUs + EFA 400 Gb/s)** following a **DeepSeek-V3/R1 production-style** setting (128 tokens per batch, 7168 hidden, top-8 experts, FP8 dispatch / BF16 combine).

| Dispatch #EP | Latency | RDMA bandwidth | Combine #EP | Latency | RDMA bandwidth |
|:-------------:|:--------:|:---------------:|:------------:|:--------:|:---------------:|
| 16 | 226 µs | 36 GB/s | 16 | 293 µs | 48 GB/s |
| 24 | 386 µs | 20 GB/s | 24 | 580 µs | 26 GB/s |
| 32 | 465 µs | 16 GB/s | 32 | 694 µs | 25 GB/s |
