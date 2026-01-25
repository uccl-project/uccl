# UCCL-EP: GPU-initiated Expert-parallel Communication

GPU-initiated expert-parallel communication (e.g., DeepEP) is the key to efficient and large-scale EP. However, it cannot run on heterogeneous platforms due to tight coupling between GPU and NIC (e.g., with IBGDA). UCCL-EP has the same interface and functionality as [DeepEP](https://github.com/deepseek-ai/DeepEP), and enables GPU-initiated communication for MoE models across heterogeneous GPUs (e.g., Nvidia, AMD) and NICs (e.g., EFA, Broadcom, CX7), with superior performance to the state-of-the-art. 

## Prerequisite

We provide a script to install dependencies (tested on p5en, p6-b200, AMD MI300x), assuming under a Python environment: 
```bash
./install_deps.sh
```

## Build on CUDA

You can directly build and install into your Python env:
```bash
python setup.py install
```

You can also use `make` to build and install (might deprecate in the future): 
```bash
make -j install
```

Alternatively, you can build `uccl.ep` wheel using docker then install:
```bash
# Under uccl
bash build_and_install.sh cuda ep
```
> Note: docker-built `uccl.ep` wheel currently does not work on p6-b200, see https://github.com/uccl-project/uccl/issues/554. 

## Build on ROCm

You can directly build and install into your Python env:
```bash
python setup.py install
```

Alternatively, you can build `uccl.ep` wheel for ROCm7 using docker then install:
```bash
# Under uccl
bash build_and_install.sh rocm ep
```

## Test import `uccl.ep`
```bash
python -c "import torch; import uccl.ep"
```

Note: 
* If you hit some `CUDA error: invalid device function`, it is likely that the GPU arch auto-detection fails. You can forcely specify the arch by setting `TORCH_CUDA_ARCH_LIST=gfx950` (eg, default gfx942 for MI300X/MI325X, gfx950 for MI355X) during compilation. 
* If you hit any weird compilation errors, try `python setup.py clean`.

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

Notes:
* To avoid possible hangs, we suggest setting env variables explicitly including `NCCL_IB_GID_INDEX`, `UCCL_IB_GID_INDEX`, `NCCL_SOCKET_IFNAME`, and `UCCL_SOCKET_IFNAME`:
  * `UCCL_IB_GID_INDEX` should be the same as `NCCL_IB_GID_INDEX` like if you were using NCCL. 
  * `UCCL_SOCKET_IFNAME` should be the interface that you would use for the `--master_addr` in `torchrun`. 
* For Broadcom Thor-2 and AMD Pollara AI NIC, we suggest setting `UCCL_IB_MAX_INFLIGHT_NORMAL=1` to enforce stricter flow control, avoiding CQE error 12.  
* Please refer to [bench/baseline](bench/baseline) for running more baselines including Torch, NVSHMEM, and pplx-kernels on EFA. 

| Environment Variable | Description | Default Value |
|---------------------|-------------|---------------|
| UCCL_SOCKET_IFNAME | Boostrapping interface | null |
| UCCL_IB_GID_INDEX | GID index in RDMA network | -1 |
| UCCL_IB_MAX_INFLIGHT_BYTES | Max inflight bytes per GPU/NIC | 2097152/8388608 (IB/EFA) |
| UCCL_IB_MAX_INFLIGHT_LOW_LATENCY | Max inflight writes per GPU/NIC in LL | 32 |
| UCCL_IB_MAX_INFLIGHT_NORMAL | Max inflight writes per GPU/NIC in HT | 8 |
| UCCL_IB_SL | Service level in RDMA network | 3/8 (IB/EFA) |
| UCCL_IB_TC | Traffic class in RDMA network | 104/0 (IB/EFA) |


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

#### On AMD MI300X with CX7 InfiniBand

|   Type    | FP8 Dispatch #EP | Bottleneck bandwidth| BF16 Dispatch #EP |Bottleneck bandwidth | Combine #EP | Bottleneck bandwidth |
|:---------:|:------------:|:--------------------:|:-----------:|:--------------------:|:--------------------:|:--------------------:|
| Intranode |       8      |    260 GB/s (xGMI)   |     8       |   295 GB/s (xGMI)    |  8       |  304GB/s (xGMI)     |
| Internode |      16      |    74 GB/s (RDMA)    |     16      |    82 GB/s (RDMA)    |  16      |   78 GB/s (RDMA)    |
| Internode |      32      |    60 GB/s (RDMA)    |     32      |    61 GB/s (RDMA)    |  32      |   60 GB/s (RDMA)    |
| Internode |      64      |    52 GB/s (RDMA)    |     32      |    53 GB/s (RDMA)    |  64      |   51 GB/s (RDMA)    |

#### On AMD MI300X with Broadcom Thor2

|   Type    | FP8 Dispatch #EP | Bottleneck bandwidth| BF16 Dispatch #EP |Bottleneck bandwidth | Combine #EP | Bottleneck bandwidth |
|:---------:|:------------:|:--------------------:|:-----------:|:--------------------:|:--------------------:|:--------------------:|
| Internode |      16      |    71 GB/s (RDMA)    |     16      |    81 GB/s (RDMA)    | 16      |    45 GB/s (RDMA)    |
| Internode |      32      |    49 GB/s (RDMA)    |     32      |    55 GB/s (RDMA)    |  32      |    50 GB/s (RDMA)    |


### Low-latency kernels with pure RDMA

#### AMD MI300X with CX7 InfiniBand

| Dispatch #EP | Latency | RDMA bandwidth | Combine #EP | Latency | RDMA bandwidth |
|:------------:|:-------:|:--------------:|:-----------:|:-------:|:--------------:|
|      8       |   65 us  |    114 GB/s     |      8      |  92 us  |    157 GB/s    |
|      16      | 136 us  |    55 GB/s     |     16      | 207 us  |    70 GB/s     |
|      32      | 224 us  |    30 GB/s     |     32      | 341 us  |    42 GB/s     |

#### On p6-b200

We test low-latency kernels on **8x B200 + 8x 400Gb/s EFA**.

| Dispatch #EP | Latency | RDMA bandwidth | Combine #EP | Latency | RDMA bandwidth |
|:-------------:|:--------:|:---------------:|:------------:|:--------:|:---------------:|
| 16 | 228 µs | 33 GB/s | 16 | 318 µs | 46 GB/s |
| 24 | 448 µs | 17 GB/s | 24 | 566 µs | 26 GB/s |
| 32 | 406 µs | 19 GB/s | 32 | 617 µs | 24 GB/s |

#### On p5en

We test low-latency kernels on **8x H200 + 16x 200Gb/s EFA**, following a **DeepSeek-V3 inference** setting (128 tokens per batch, 7168 hidden, top-8 experts, FP8 dispatch / BF16 combine).

| Dispatch #EP | Latency | RDMA bandwidth | Combine #EP | Latency | RDMA bandwidth |
|:-------------:|:--------:|:---------------:|:------------:|:--------:|:---------------:|
| 16 | 226 µs | 36 GB/s | 16 | 293 µs | 48 GB/s |
| 24 | 386 µs | 20 GB/s | 24 | 580 µs | 26 GB/s |
| 32 | 465 µs | 16 GB/s | 32 | 694 µs | 25 GB/s |
