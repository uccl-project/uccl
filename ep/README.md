# UCCL GPU-Driven Expert-parallelism Engine

UCCL EP engine provides a similar interface from [DeepEP](https://github.com/deepseek-ai/DeepEP). 

For UCCL's host/CPU-driven P2P engine, see [p2p](../p2p/) folder.

## Install

Installing `ep` as a Python package:
```bash
./build_and_install.sh cuda ep 3.11
```
Alternatively, in a Python environment 
```bash
make -j install
```

## Build on ROCm for testing

build rocm image
```bash
bash build.sh rocm all 3.10
```

start docker container
```bash
docker run -d \
    --name dev_uccl-builder-rocm \
    --ipc=host \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    --cap-add=SYS_PTRACE \
    --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --group-add video \
    --privileged \
    -v "$PWD":/workspace/uccl \
    -w "/workspace/uccl" \
    uccl-builder-rocm sleep infinity
```

build ep
```bash
docker exec -it dev_uccl-builder-rocm /bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv
source .venv/bin/activate

cd ep
bash install_deps.sh base

python setup.py build
mkdir -p uccl/lib
cp build/**/*.so uccl/
```

build uccl for develop

```bash
cd /workspace/uccl/ep
python setup.py develop
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

Node 0: 
```bash
OMP_NUM_THREADS=8 torchrun \
  --nnodes=3 --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=10.1.227.34 --master_port=12357 \
  bench/test_low_latency.py \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=384
```

Node 1: 
```
OMP_NUM_THREADS=8 torchrun \
  --nnodes=3 --nproc_per_node=8 \
  --node_rank=1 \
  --master_addr=10.1.227.34 --master_port=12357 \
  bench/test_low_latency.py \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=384
```

Node 2:
```
OMP_NUM_THREADS=8 torchrun \
  --nnodes=3 --nproc_per_node=8 \
  --node_rank=2 \
  --master_addr=10.1.227.34 --master_port=12357 \
  bench/test_low_latency.py \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=384
```
