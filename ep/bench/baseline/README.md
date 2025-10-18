# MoE All-to-All Baseline Benchmarks

Benchmark comparing different MoE all-to-all communication methods:
1. CPU + PyTorch Distributed
2. CPU + NVSHMEM
3. CUDA + PyTorch Distributed (with CUDA kernels)
4. CUDA + NVSHMEM (with CUDA kernels)

## Quick Start

### 1. Setup Environment

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 2. Build CUDA Extension (Recommended)

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
- `--in-dtype`: Input dtype: `bfloat16` or `float16` (default: `bfloat16`)
- `--out-dtype`: Output dtype: `bfloat16` or `float16` (default: `bfloat16`)

## Benchmark Configurations

**V2-Lite**: 64 experts, 6 experts/token, 2048 hidden dim
**R1**: 256 experts, 8 experts/token, 7168 hidden dim
**Tokens**: 1, 4, 8, 16, 32, 64, 128

## Troubleshooting

**CUDA extension build fails**: Benchmark will fall back to CPU pack/unpack (slower but works)

**Import error**: Rebuild extension with `./build_pack_unpack.sh`

**OOM error**: Reduce token count or use fewer GPUs
