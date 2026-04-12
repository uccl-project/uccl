# DeepEP-Lite Handoff: Multi-Node Benchmark Context

## Goal

Run DeepEP official benchmark (`bench/test_low_latency.py`) on 2 nodes x 4 GPUs = 8 GPUs total.
Single-node (4 GPU) already works and benchmarks correctly. Multi-node needs debugging.

## Testbed

| Node | Hostname | Public IP | RDMA IP | Python | GPUs |
|------|----------|-----------|---------|--------|------|
| l40 | mibura-sky-test-01 | 38.123.21.7 | 4.14.153.89 | conda env `uccl` (Python 3.12) | 4x L4 (sm_89) |
| l41 | mibura-sky-test-02 | 38.123.21.8 | 4.14.153.90 | virtualenv `~/zhongjie/zj_py` (Python 3.13) | 4x L4 (sm_89) |

- NFS: l40 exports `/home/yangz/nfs/` to l41 (same path on both nodes)
- RDMA NIC: mlx5_0 on both nodes (InfiniBand, PORT_ACTIVE)
- nvidia_peermem loaded on both nodes
- SSH: `ssh l41` works from l40

## Build

```bash
cd /home/yangz/nfs/zhongjie/uccl/experimental/lite/ep
conda run -n uccl make -j SM=89
# Binary: ep.abi3.so
```

## Install on Both Nodes

```bash
# l40 (conda)
cp ep.abi3.so /home/yangz/nfs/miniconda3/envs/uccl/lib/python3.12/site-packages/uccl/ep.abi3.so

# l41 (virtualenv) - run from l40 via ssh or NFS
ssh l41 "cp /home/yangz/nfs/zhongjie/uccl/experimental/lite/ep/ep.abi3.so ~/zhongjie/zj_py/lib/python3.13/site-packages/uccl/ep.abi3.so"
```

## How to Run Multi-Node (CRITICAL: different Python envs!)

**l41 uses virtualenv, NOT conda.** This is the most important thing to remember.

### Manual 2-node launch (1 GPU per node, simplest):

**Terminal/shell 1 — start l41 first:**
```bash
ssh l41 "nohup bash -c '
source ~/zhongjie/zj_py/bin/activate
cd /home/yangz/nfs/zhongjie/uccl/experimental/lite/ep
NCCL_IB_HCA=mlx5_0 CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=2 --nproc_per_node=1 --node_rank=1 \
    --master_addr=4.14.153.89 --master_port=29600 \
    bench/test_low_latency.py --num-tokens=128 --hidden=2048 --num-topk=1 --num-experts=2 --disable-nvlink
' > /home/yangz/nfs/zhongjie/l41_bench.log 2>&1 &"
```

Wait 5 seconds, then **start l40:**
```bash
cd /home/yangz/nfs/zhongjie/uccl/experimental/lite/ep
conda run -n uccl env NCCL_IB_HCA=mlx5_0 CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=2 --nproc_per_node=1 --node_rank=0 \
    --master_addr=4.14.153.89 --master_port=29600 \
    bench/test_low_latency.py --num-tokens=128 --hidden=2048 --num-topk=1 --num-experts=2 --disable-nvlink
```

### Full 8-GPU launch (4 GPUs per node):

**l41 (background):**
```bash
ssh l41 "nohup bash -c '
source ~/zhongjie/zj_py/bin/activate
cd /home/yangz/nfs/zhongjie/uccl/experimental/lite/ep
NCCL_IB_HCA=mlx5_0 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nnodes=2 --nproc_per_node=4 --node_rank=1 \
    --master_addr=4.14.153.89 --master_port=29600 \
    bench/test_low_latency.py --num-tokens=128 --hidden=2048 --num-topk=4 --num-experts=8 --disable-nvlink
' > /home/yangz/nfs/zhongjie/l41_bench.log 2>&1 &"
```

Wait 5s, then **l40:**
```bash
cd /home/yangz/nfs/zhongjie/uccl/experimental/lite/ep
conda run -n uccl env NCCL_IB_HCA=mlx5_0 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr=4.14.153.89 --master_port=29600 \
    bench/test_low_latency.py --num-tokens=128 --hidden=2048 --num-topk=4 --num-experts=8 --disable-nvlink
```

### Benchmark args constraints

- `num_experts` must be divisible by `num_ranks` (total GPUs across all nodes)
- `num_qps_per_rank = num_experts // num_ranks`
- For 2 GPUs total: `--num-experts=2 --num-topk=1`
- For 8 GPUs total: `--num-experts=8 --num-topk=4` (or `--num-experts=16 --num-topk=4`)

### Check l41 log after run:
```bash
cat /home/yangz/nfs/zhongjie/l41_bench.log
```

### Kill leftover processes:
```bash
# Find PIDs first
ssh l41 "ps aux | grep torchrun | grep -v grep"
# Then kill specific PIDs
ssh l41 "kill <PID>"
```

## Single-Node Test (verified working)

```bash
cd /home/yangz/nfs/zhongjie/uccl/experimental/lite/ep

# Simple test (4 GPU)
conda run -n uccl env CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 bench/test_internode_simple.py

# Benchmark (4 GPU)
conda run -n uccl env CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    bench/test_low_latency.py --num-tokens=128 --hidden=2048 --num-topk=4 --num-experts=16 --disable-nvlink
```

Single-node benchmark results (all correctness tests passed):

| Config (4-GPU) | D+C BW | Dispatch BW | Combine BW |
|----------------|--------|-------------|------------|
| 128t x 2048h x 16e | 3.32 GB/s | 3.09 GB/s | 3.46 GB/s |
| 256t x 4096h x 16e | 3.65 GB/s | 3.70 GB/s | 3.71 GB/s |
| 512t x 7168h x 16e | 3.65 GB/s | 3.97 GB/s | 3.54 GB/s |

## Multi-Node Status & Known Issues

### What works
- SSH from l40 to l41: confirmed working
- ep.abi3.so installed on both nodes
- Single dispatch+combine across nodes: DATA TRANSFER CORRECT (tested previously)
- l41 test output shows `All tests passed!` for the EP operations themselves

### What's broken: multi-node hangs/timeout
- l40 side often times out (never produces output from the test script)
- Possible causes:
  1. **Rendezvous timeout**: torchrun's TCP store might have issues with the 2-node setup
  2. **Port conflicts**: leftover processes from previous runs occupying ports
  3. **NCCL init**: dist.init_process_group might hang if NCCL can't find the IB device

### CQE errors during cleanup (non-blocking, cosmetic)
- After the test completes, proxy threads get `status=5 (Work Request Flushed Error)` 
- These are expected during shutdown — QPs transition to ERROR state
- The actual EP dispatch/combine operations succeed before these errors appear

### Shared PD fix (already applied)
- Root cause of previous multi-node CQE errors (status=10, remote access error) was 
  multiple proxy threads each opening independent ibv_context/PD/MR
- Fix: all threads share one ibv_context/PD/MR via `g_shared_rdma_cache`
- This is already in the current code and working for single-node

## Key Files

| File | Purpose |
|------|---------|
| `src/rdma.cpp` | RDMA transport, shared PD cache (lines ~255-312), CQE handling |
| `src/proxy.cpp` | Proxy init, QP setup, cleanup with `release_shared_rdma_resources()` |
| `src/uccl_ep.cc` | Python bindings, buffer allocation |
| `include/rdma.hpp` | RDMA declarations (shared PD decl is unconditional) |
| `bench/test_low_latency.py` | Official DeepEP benchmark (THE target script) |
| `bench/test_internode_simple.py` | Simple dispatch+combine test |
| `bench/utils.py` | `init_dist_under_torchrun()`, `initialize_uccl()`, proxy setup |
| `bench/buffer.py` | Buffer wrapper with PCIE_INTRANODE awareness |
| `Makefile` | Build with `PCIE_INTRANODE=1`, `DISABLE_SM90_FEATURES` for sm<90 |

## Compile Flags

- `PCIE_INTRANODE=1` (default): sets `NUM_MAX_NVL_PEERS=1`, all peers are RDMA
- `DISABLE_SM90_FEATURES`: auto-set for SM < 90 (L4 = sm_89)
- `SM=89`: target architecture

## Debugging Tips

1. Always check l41 log on NFS: `cat /home/yangz/nfs/zhongjie/l41_bench.log`
2. Use different `--master_port` values to avoid conflicts with stale processes
3. Kill l41 processes between runs (check with `ssh l41 "ps aux | grep torchrun"`)
4. For verbose RDMA debugging, the code prints QPN/rkey info during proxy setup
5. CQE errors with `status=5` during shutdown are cosmetic (WR Flushed after QP→ERR)
6. CQE errors with `status=10` (remote access) during data transfer are the real bug
