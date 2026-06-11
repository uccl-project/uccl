# UCCL-GIN

A standalone GPU-initiated networking backend that mirrors the
[NCCL GIN](https://github.com/NVIDIA/nccl) device API on
[AWS EFA](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html). It provides
the same `put` / `put_value` / `red_add_rel` / `quiet` surface as NCCL's
`ncclGin` so that communication libraries (DeepEP V2, NCCL-EP, etc.) can run on
EFA without per-kernel source changes.

Under the hood, cross-node traffic is routed through UCCL's D2H ring → CPU proxy
→ EFA verbs transport substrate. Intra-node (NVLink) traffic is delegated to
NCCL unchanged.

```
DeepEP V2 / NCCL-EP kernel
  │  gin.put<Rail>(recv, send, bytes, dst)
  │  gin.red_add_rel<Rail>(ptr, val, dst)
  │  gin.put_tail_add<Rail>(recv, send, bytes, dst, count, tail_off)
  ▼
uccl_gin::UCCLGin            ←  this project
  │  Rail → TransferCmd → D2H ring → CPU proxy → EFA RDMA write / write-with-imm
  │  Lsa  → delegate to NCCL (NVLink, not yet in standalone)
  ▼
UCCL CPU proxy + raw EFA ibverbs
```

## Why

EFA has no RDMA atomics, no ordering guarantees between WRs, and poor small-message
throughput (4 KiB → ~2.8 GB/s per NIC). CX7 IB can do per-token RDMA writes at
line rate; EFA requires coalescing.

UCCL-GIN addresses this with:
- **Piggyback tail** — payload and counter delta in one `WRITE_WITH_IMM` (no ordering gap)
- **Sender-side completion dependency** — finish atomics wait for payload WR CQEs
- **Per-slot metadata check** — receiver validates payload visibility before consuming
- **Compact channel staging** — batched multi-token writes (when integrated with DeepEP)

## Architecture

```
experimental/uccl_gin/
├── uccl_gin/                  # Public device API (what kernels include)
│   ├── uccl_gin.cuh           # handle::UCCLGin
│   ├── uccl_gin_rail.cuh      # rail_put / rail_put_tail_add / rail_red_add
│   └── resources.cuh          # UCCLGinResources POD + profiling counters
├── transport/                 # Internal D2H + proxy + EFA layer (copied from ep/)
├── context.hpp / context.cpp  # Host setup: EFA QP/CQ/MR, peer exchange, resources
├── bindings.cpp               # CPython extension (_uccl_gin)
├── tests/
│   ├── microbench.cu          # C++/MPI standalone microbench
│   ├── test_context.py        # Context lifecycle smoke test
│   ├── test_microbench.py     # Microbench wrapper
│   └── test_primitives.py     # Per-primitive correctness tests
├── python/uccl_gin/           # Python helpers + context smoke/stress scripts
├── ARCHITECTURE.md            # Full design doc
├── PLAN.md                    # Phased plan
├── Makefile
└── README.md                  # This file
```

## Primitives

| Primitive | Status | Description |
|-----------|--------|-------------|
| `put<Rail>` | ✅ | RDMA write from symmetric window to remote window |
| `put_tail_add<Rail>` | ✅ | RDMA write + piggyback tail counter advance (UCCL-specific) |
| `red_add_rel<Rail>` | ✅ | Ordered remote atomic via empty write-with-imm + CPU proxy |
| `put_value<Rail>` | ✅ | Inline single-word write to remote window slot |
| `quiet` | ✅ | Drain one D2H lane/proxy thread through prior WRITE CQEs |
| `flush` | ✅ | Drain all D2H queues in this UCCLGin context |
| `put<Lsa>` | — | Intra-node NVLink (compose NCCLGin, not in standalone) |
| `signal` | — | Future |

## Prerequisites

- **Hardware**: AWS p5 / p5en / p6 instances with EFA
- **GPU**: NVIDIA H100 / H200 (SM 90+)
- **CUDA**: 13.0+
- **NCCL**: 2.30+ (`nvidia-nccl-cu13` wheel)
- **MPI**: OpenMPI 5
- **Python**: 3.12 with venv

## Build

```bash
source /path/to/venv/bin/activate
make -C experimental/uccl_gin \
    CUDA_HOME=/usr/local/cuda-13.0 \
    PYTHON=$VIRTUAL_ENV/bin/python \
    SM=90 \
    -j
```

Options:
- `CUDA_HOME` — CUDA toolkit path (default `/usr/local/cuda-13.0`)
- `SM` — GPU architecture (90 for H100/H200)
- `NUM_PROXY_THS` — proxy threads per GPU (default 4)
- `UCCL_GIN_WITH_NCCL_GIN` — build NCCL-GIN reference path (default 1)

## Test

### C++ microbench (all primitives at once)

```bash
mpirun --oversubscribe \
  --host $NODE0:8,$NODE1:8 -np 16 -npernode 8 \
  -x LD_LIBRARY_PATH -x NCCL_NET_PLUGIN=ofi -x FI_PROVIDER=efa \
  -x FI_EFA_USE_DEVICE_RDMA=1 -x LOCAL_WORLD_SIZE=8 \
  build/uccl_gin_microbench \
    --sizes 1024,4096,65536,262144,1048576 \
    --iters 10 --warmup 2
```

Selected primitive, correctness-only:
```bash
  --only put-add --correctness-only    # put + red_add_rel
  --only tail-add --correctness-only   # put_tail_add + quiet
  --only quiet --correctness-only      # put + quiet
  --only red-add --correctness-only    # red_add_rel counter
```

Expected output:
```
UCCL-red_add counter: PASS
UCCL-put_value: PASS
bytes        NCCL     UCCL-put/add   UCCL-tail/q    UCCL-put+q
65536        -        PASS           PASS           PASS
all correctness PASS
```

### Python

```bash
UCCL_GIN_ROOT=$PWD UCCL_GIN_MPI_HOSTS=$NODE0:8,$NODE1:8 \
UCCL_GIN_RUN_PRIMITIVES=1 \
python -m pytest tests/test_primitives.py -v
```

### Context stress

```bash
UCCL_GIN_CONTEXT_STRESS_ITERS=100 \
mpirun ... python -m uccl_gin.context_stress
```

## Python API

```python
from uccl_gin import Context, mpi_rank

ctx = Context(max_message_bytes=1 << 20, local_world_size=8, ifname="enp71s0")
resources = ctx.resources()  # dict: window_base, atomic_tail_base, ...
ctx.close()
```

## Key design decisions

1. **Not a subclass of NCCLGin** — independent struct with `if constexpr` dispatch.
   Lsa/World delegation is composition, not inheritance.
2. **Tail storage separate from GPU window** — `PackAtomicWithSeq` offset is 13-bit
   (≤8191 bytes). Tails live in a compact `atomic_tail_base` indexed by
   `(channel, src_rank)`.
3. **`put_tail_add` is UCCL-GIN specific** — fuses payload and counter into one
   `WRITE_WITH_IMM` to close the EFA ordering gap that NCCL solves with FORCE_SO.
4. **16-byte TransferCmd ABI unchanged** from UCCL EP V1.
5. **Rail topology is paired remote only** in this standalone backend:
   `dst_rank` must be the same local rank on a different node.

## See also

- [ARCHITECTURE.md](ARCHITECTURE.md) — full design doc
- [UCCL EFA Programming Guide](https://uccl-project.github.io/posts/efa-programming/)
- [NCCL Device API](https://github.com/NVIDIA/nccl)
- [DeepEP V2](https://github.com/deepseek-ai/DeepEP)
