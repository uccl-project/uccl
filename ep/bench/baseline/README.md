# MoE All-to-All Baseline Benchmarks

Benchmark comparing different MoE all-to-all communication methods:
1. CPU + PyTorch Distributed
2. CPU + NVSHMEM
3. CUDA + PyTorch Distributed (with CUDA kernels)
4. CUDA + NVSHMEM (with CUDA kernels)
5. pplx kernel EP

## Assumption 
### 1. Build and install [pplx-kernels](https://github.com/perplexityai/pplx-kernels)

> **Note:** Our baseline implementations are adapted from pplx-kernels. To maintain consistency and fair comparison, we continue to use PyTorchStreamWrapper, nvshmem_init APIs provided by pplx-kernels.

## Quick Start


### 1. Build CUDA Extension (Recommended)

```bash
cd /path/to/uccl/ep/bench/baseline
./build_pack_unpack.sh
```

Verify:
```bash
python -c "import moe_pack_unpack; print('CUDA extension loaded!')"
```

### 3. Run Benchmark

**Single node (all GPUs):**
```bash
python bench_nvshmem_sparse_uccl.py --dp-size 1
```

**With different data types:**
```bash
python bench_nvshmem_sparse_uccl.py --dp-size 1 --in-dtype float16
```

## Multi-Node Setup

**Node 0 (Master):**
```bash
export MASTER_ADDR=<master_ip>
export MASTER_PORT=29500
export WORLD_SIZE=<total_gpus>
export WORLD_LOCAL_SIZE=<gpus_per_node>
export NODE_RANK=0
python bench_nvshmem_sparse_uccl.py --dp-size 1
```

**Node 1+ (Workers):**
```bash
export MASTER_ADDR=<master_ip>
export MASTER_PORT=29500
export WORLD_SIZE=<total_gpus>
export WORLD_LOCAL_SIZE=<gpus_per_node>
export NODE_RANK=<node_id>
python bench_nvshmem_sparse_uccl.py --dp-size 1
```

## Output

Results saved to: `uccl/ep/bench/data/<timestamp>_unified_moe_separated.tsv`

## Command Line Options

- `--dp-size`: Data parallel size (default: 1)
- `--in-dtype`: Input dtype: `bfloat16`, `float16`, `float8_e4m3fn`, `float8_e5m2` (default: `float8_e4m3fn`)
- `--out-dtype`: Output dtype: `bfloat16`, `float16`, `float8_e4m3fn`, `float8_e5m2`  (default: `bfloat16`)

## Configuration

### Default Data Types
- `--in-dtype`: Default is `float8_e4m3fn` (options: `bfloat16`, `float16`, `float8_e4m3fn`, `float8_e5m2`)
- `--out-dtype`: Default is `bfloat16` (options: `bfloat16`, `float16`, `float8_e4m3fn`, `float8_e5m2`)

### Testing Larger Workloads
To test with larger workloads, you can manually configure the `configs` list in `bench_nvshmem_spare_uccl.py`:
```python
configs = [
    # Custom configurations: (num_experts, experts_per_token, hidden_dim, max_num_tokens)
    MoEConfig(128, 8, 4096, 8192, in_dtype, out_dtype),
    MoEConfig(256, 16, 8192, 16384, in_dtype, out_dtype),
    # Add your custom configs here...
]
```



