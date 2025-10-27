# MoE All-to-All Baseline Benchmarks

## Methods

### Standard Implementations
1. **PyTorch Distributed**  
2. **NVSHMEM**  
   

### Optimized Implementations
3. **CUDA + PyTorch Distributed**  
4. **CUDA + NVSHMEM**  


5. **PPLX Kernel EP**  
   Specialized expert parallelism kernel implementation

---
**Standard** Python-based packing/unpacking operations
**Key Optimization:** Efficient CUDA kernels for packing/unpacking operations significantly reduce overhead compared to Python-based implementations.

## Requirements
### 1. Build and install [pplx-kernels](https://github.com/perplexityai/pplx-kernels)

> **Note:** Our baseline implementations are adapted from pplx-kernels. To maintain consistency and fair comparison, we continue to use PyTorchStreamWrapper, nvshmem_init APIs provided by pplx-kernels.

## Quick Start


### 1. Build CUDA Extension (Recommended)

```bash
cd /path/to/uccl/ep/bench/baseline
./build_pack_unpack.sh
```

### 2. Run Benchmark for torch and nvshmem 

**Single node (all GPUs):**
```bash
python bench_nvshmem_sparse_uccl.py --dp-size 1
```

**With different data types:**
```bash
python bench_nvshmem_sparse_uccl.py --dp-size 1 --in-dtype float16
```

### 3. Run Benchmark for pplx (if you already installed)

```bash
cd pplx-kernels
python3 -m tests.bench_all_to_all
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
pytest -svx tests/test_all_to_all.py ##pplx
```

**Node 1+ (Workers):**
```bash
export MASTER_ADDR=<master_ip>
export MASTER_PORT=29500
export WORLD_SIZE=<total_gpus>
export WORLD_LOCAL_SIZE=<gpus_per_node>
export NODE_RANK=<node_id>
python bench_nvshmem_sparse_uccl.py --dp-size 1
pytest -svx tests/test_all_to_all.py ##pplx
```

## Output

Results saved to: `uccl/ep/bench/data/<timestamp>_unified_moe_separated.tsv`





