# Megatron / Primus benchmarks

## Benchmark Results: UCCL-EP vs NCCL (AWS p5en.48xlarge)

**Cluster**: 4× p5en.48xlarge bare metal (32× H200 GPUs, 16× EFAv3 NICs per node, 192 CPU cores per node)  
**Model**: DeepSeek-V3 style MoE (256 experts, top-k=8, hidden=7168, ffn-hidden=2048)  
**Routing**: Uniform (`--moe-router-force-load-balancing`)  
**Precision**: BF16 + FP8 hybrid  
**NCCL**: 2.28.9 with aws-ofi-nccl 1.18.0, Ring/Simple protocol  
**Megatron**: core_r0.17.0

### Key Result

| Config | NCCL | UCCL-EP | Speedup |
|---|---|---|---|
| EP=32, 12 layers | 152.2 TFLOPS | **188.0 TFLOPS** | **1.24x** |

### NCCL Channel Sweep (NCCL_MAX_NCHANNELS)

| NCCL_MAX_NCHANNELS | NCCL (TFLOPS) | UCCL-EP (TFLOPS) |
|---|---|---|
| None (unset) | 151.7 | 79.8 |
| 8 | 148.5 | 180.0 |
| 16 | 150.3 | 183.2 |
| 32 | 152.2 | 188.0 |

---

## Reproducing the Benchmark

### Hardware Requirements

- 4× p5en.48xlarge (or equivalent: 8× H200 per node, EFAv3 networking)
- Shared filesystem (FSx Lustre or similar) across all nodes
- Bare metal access recommended (virtualization will add overhead)

### Software Stack

| Component | Version |
|---|---|
| Python | 3.13 |
| PyTorch | 2.10.0+cu130 |
| CUDA toolkit | 13.0 |
| NCCL | 2.28.9 |
| Transformer Engine | 2.11.0 |
| EFA libfabric | 2.4.0 |
| aws-ofi-nccl | 1.18.0 |
| Megatron-LM | core_r0.17.0 |

### Environment Setup

```bash
# Common environment variables (both UCCL-EP and NCCL)
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NCCL_DEBUG=WARN
export NCCL_PROTO=simple
export NCCL_ALGO=Ring
export NCCL_SOCKET_IFNAME=^docker,lo,veth_def_agent
export NCCL_IB_GID_INDEX=0
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# UCCL-EP specific (add these for UCCL-EP runs only)
export NCCL_MAX_NCHANNELS=8          
export UCCL_SOCKET_IFNAME=^docker,lo,veth_def_agent
export UCCL_IB_GID_INDEX=0
export NETS_PER_GPU=2
export PYTHONPATH=/path/to/uccl_ep_hook_site:/path/to/megatron-lm
```

### Model Parameters

| Parameter | Value |
|---|---|
| `--num-layers` | 12 (EP=32) or 6 (EP=16) |
| `--hidden-size` | 7168 |
| `--ffn-hidden-size` | 2048 |
| `--num-attention-heads` | 128 |
| `--seq-length` | 4096 |
| `--micro-batch-size` | 1 |
| `--global-batch-size` | 32 |
| `--num-experts` | 256 |
| `--moe-router-topk` | 8 |
| `--expert-model-parallel-size` | 32 (or 16) |
| `--train-iters` | 60 |

### UCCL-EP Launch with Megatron (EP=32, 4 nodes)

Run on each node (change `--node_rank` to 0, 1, 2, 3):

```bash
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=NODE_RANK \
  --rdzv_backend=c10d --rdzv_endpoint=MASTER_ADDR:PORT \
  pretrain_gpt.py \
  --num-layers 12 --hidden-size 7168 --ffn-hidden-size 2048 \
  --num-attention-heads 128 --seq-length 4096 --max-position-embeddings 4096 \
  --micro-batch-size 1 --global-batch-size 32 --train-iters 60 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 32 --num-experts 256 --moe-router-topk 8 \
  --moe-router-load-balancing-type aux_loss --moe-grouped-gemm \
  --moe-router-force-load-balancing \
  --moe-token-dispatcher-type flex --transformer-impl transformer_engine \
  --bf16 --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max \
  --moe-flex-dispatcher-backend deepep --moe-router-dtype fp32 \
  --enable-experimental --moe-permute-fusion --moe-router-fusion \
  --no-gradient-accumulation-fusion --no-bias-gelu-fusion \
  --use-distributed-optimizer --lr 1e-4 --min-lr 1e-5 --weight-decay 0.0 \
  --clip-grad 1.0 --attention-dropout 0.0 --hidden-dropout 0.0 \
  --tokenizer-type NullTokenizer --vocab-size 32000 --mock-data \
  --split 99,1,0 --init-method-std 0.02 --log-interval 1 --log-throughput \
  --swiglu --disable-bias-linear \
  --recompute-granularity full --recompute-method uniform \
  --recompute-num-layers 1 --eval-interval 1000 --eval-iters 0
```

### NCCL Launch (EP=32, 4 nodes)

```bash
export PYTHONPATH=/path/to/megatron-lm
# Do NOT set UCCL_* env vars or NCCL_MAX_NCHANNELS

torchrun --nnodes=4 --nproc_per_node=8 --node_rank=NODE_RANK \
  --rdzv_backend=c10d --rdzv_endpoint=MASTER_ADDR:PORT \
  pretrain_gpt.py \
  --num-layers 12 --hidden-size 7168 --ffn-hidden-size 2048 \
  --num-attention-heads 128 --seq-length 4096 --max-position-embeddings 4096 \
  --micro-batch-size 1 --global-batch-size 32 --train-iters 60 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 32 --num-experts 256 --moe-router-topk 8 \
  --moe-router-load-balancing-type aux_loss --moe-grouped-gemm \
  --moe-router-force-load-balancing \
  --moe-token-dispatcher-type alltoall --transformer-impl transformer_engine \
  --bf16 --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max \
  --moe-pad-expert-input-to-capacity --moe-expert-capacity-factor 1.0 \
  --use-distributed-optimizer --lr 1e-4 --min-lr 1e-5 --weight-decay 0.0 \
  --clip-grad 1.0 --attention-dropout 0.0 --hidden-dropout 0.0 \
  --tokenizer-type NullTokenizer --vocab-size 32000 --mock-data \
  --split 99,1,0 --init-method-std 0.02 --log-interval 1 --log-throughput \
  --no-gradient-accumulation-fusion --no-bias-gelu-fusion --swiglu \
  --disable-bias-linear --recompute-granularity full --recompute-method uniform \
  --recompute-num-layers 1 --eval-interval 1000 --eval-iters 0
```

### Dispatch/Combine Config Tuning

UCCL-EP dispatch and combine kernels are controlled by `Config(num_sms, nvl_send, nvl_recv, rdma_send, rdma_recv)`:

| Parameter | Description |
|---|---|
| `num_sms` | SMs used by the kernel (default: `Buffer.num_sms = 20`) |
| `nvl_send` | Max NVLink chunked send tokens (intra-node pipelining) |
| `nvl_recv` | Max NVLink chunked recv tokens |
| `rdma_send` | Max RDMA chunked send tokens (inter-node pipelining) |
| `rdma_recv` | Max RDMA chunked recv tokens |

The tuned values below produced the best results on EFAv3.

#### Defaults vs Tuned (32 ranks, EP=32)

| | `nvl_send` | `nvl_recv` | `rdma_send` | `rdma_recv` |
|---|---|---|---|---|
| **Dispatch default** | 8 | 512 | 16 | 512 |
| **Dispatch tuned** | 8 | 512 | **24** | 512 |
| **Combine default** | 1 | 512 | 8 | 512 |
| **Combine tuned** | **3** | 512 | **24** | 512 |

To apply the tuned config, edit `get_dispatch_config` and `get_combine_config` in `buffer.py`:

```python
# get_dispatch_config, 32 ranks:
32: Config(Buffer.num_sms, 8, 512, 24, 512)

# get_combine_config, 32 ranks:
32: Config(Buffer.num_sms, 3, 512, 24, 512)
```
---

## Known or potential Issues

- **Container CPU jitter**: The use of container can affect UCCL-proxy thread, inflating FIFO round-trip time. Use bare metal for best performance. 
- **Tuning number of NCCL channels**: When UCCL-EP is active, NCCL auto-selects 32 channels for optimizer collectives (vs 16 in NCCL-only runs). This causes EFA SRD congestion and collapses performance. Setting `NCCL_MAX_NCHANNELS` resolves this.

## Reproducing AMD Primus Training frameworks results

From the repo root:

```bash
git submodule update --init thirdparty/Primus
```

Scripts default `PRIMUS_PATH` to `thirdparty/Primus` under this repo (see `primus_slurm_pretrain_cli.sh`).
