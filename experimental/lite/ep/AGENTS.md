# DeepEP-Lite

`experimental/lite/ep` is the DeepEPv2 Lite `ElasticBuffer` codebase with
UCCL-EP integrated as the EP data transport. The active path is the UCCL CPU
proxy (`EP_USE_UCCL_PROXY=1`) for both GDR and no-GDR runs. NCCL is still
loaded for communicator/bootstrap and DeepEP interface compatibility, but NCCL
GIN is not used for EP payload movement.

Design notes, rationale, and benchmark numbers live in
`doc/uccl-ep-deepepv2-migration.md`. This file is the operational checklist.

## Goals

- Run UCCL-EP + DeepEPv2 efficiently on consumer / low-end systems without
  NVLink or GDR. no-NVLink/no-GDR is the target path, not a fallback.
- Stage inter-node payloads through host memory; route same-node no-NVLink
  multi-GPU traffic through the POSIX shared host window.
- Validation hardware: NVIDIA L4 / `sm_89`, ConnectX over ibverbs.

## Runtime modes

Common environment template (keep absolute paths in your shell, not in
committed docs):

```bash
EP_DIR=$(pwd)
PYTHONPATH=$EP_DIR
PYTHON_BIN=${PYTHON_BIN:-python}
EP_USE_UCCL_PROXY=1
EP_FORCE_NO_NVLINK=1
NCCL_GIN_TYPE=2
DISABLE_SM90_FEATURES=1
EP_SUPPRESS_NCCL_CHECK=1
EP_FORCE_PROCESS_EXIT=1
UCCL_IB_HCA=${UCCL_IB_HCA:-<ib-device>}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<gpu-list>}
EP_NCCL_ROOT_DIR=${EP_NCCL_ROOT_DIR:-<local-nccl-root>}
EP_TORCH_NVSHMEM_STUB=${EP_TORCH_NVSHMEM_STUB:-}
LD_PRELOAD=$EP_NCCL_ROOT_DIR/lib/libnccl.so.2${EP_TORCH_NVSHMEM_STUB:+:$EP_TORCH_NVSHMEM_STUB}
LD_LIBRARY_PATH=$EP_NCCL_ROOT_DIR/lib:${CONDA_PREFIX:-<python-env>}/lib:${LD_LIBRARY_PATH:-}
EP_JIT_CACHE_DIR=${EP_JIT_CACHE_DIR:-$EP_DIR/.jit-cache}
```

| Mode | Settings |
| --- | --- |
| GDR | `UCCL_FORCE_NO_GDR=0`, `EP_FORCE_HOST_WINDOW=0`, `NCCL_NET_GDR_LEVEL=5` |
| no-GDR | `UCCL_FORCE_NO_GDR=1`, `EP_FORCE_HOST_WINDOW=1`, `NCCL_NET_GDR_LEVEL=0` |

With `EP_FORCE_NO_NVLINK=1` and `local_world_size > 1`, UCCL proxy uses the
shared host window even in GDR mode. `EP_UCCL_FORCE_GPU_WINDOW=1` is for
diagnostics only — it is not a supported fast path on L4/PCIe.

Use explicit test CLI flags (`--do-handle-copy-modes`, `--expert-alignment-modes`,
`--fp8-dispatch-modes`, `--num-bench-tests`, `--trace-steps`); do not add
hidden test environment switches.

## Build and run

```bash
conda activate uccl
make -j SM=89 PYTHON=$PYTHON_BIN
```

Built-in benchmark for the validation shape:

```bash
timeout 15s $PYTHON_BIN tests/elastic/test_ep.py \
  --num-processes 2 \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 \
  --test-first-only --skip-check --num-gpu-timeout-secs=3 \
  --do-handle-copy-modes=0 --expert-alignment-modes=128 \
  --fp8-dispatch-modes=0 --num-bench-tests=5
```

For 2-node × 1-GPU runs, set `WORLD_SIZE=2`, `LOCAL_WORLD_SIZE=1`, matching
`MASTER_ADDR`/`MASTER_PORT`, and `RANK=0` / `RANK=1` on the two nodes. The
first run after a JIT-visible change can spend the timeout compiling; rerun
with the same `EP_JIT_CACHE_DIR` after warmup. `run_multinode.sh` wraps this
launch pattern.

Full-path validation and per-stage profiling:

```bash
timeout 15s $PYTHON_BIN tests/elastic/run_full_path_bench.py \
  --transport nogdr --num-processes=4 --validate-only --trace-steps

timeout 15s $PYTHON_BIN tests/elastic/run_full_path_bench.py \
  --transport nogdr --num-processes=4 \
  --measure-stages=reduced_combine --num-bench-tests=1
```

`--measure-stages` accepts `all` or a comma-separated subset of `dispatch`,
`expanded_dispatch`, `cached_dispatch`, `combine`, `reduced_combine`.
Unselected stages still run once to preserve the data path; their bandwidth
fields print as `nan`.

## Correctness notes

- UCCL Gin-compatible barrier signal/shadow storage must be indexed by
  `(barrier_tag, rank)`. A single per-rank counter causes different DeepEP
  phases to interfere and can hang cached dispatch or combine.
- `ElasticBuffer::sync_uccl_peers()` must wait until every UCCL proxy reports
  ready after `start_dual()`. Without this, GDR kernels can enqueue commands
  before all QPs and ACK paths are ready; debug logging can mask the race.
- For UCCL `kNumScaleupRanks == 1` combine, write directly from the scaleup
  warp into the scaleout send/recv buffer and issue UCCL puts. The generic
  hybrid two-stage path causes an extra GPU PCIe read from host-window memory
  on no-GDR and makes the GDR/no-GDR comparison unfair.
- One elected lane must post direct-combine completion signals after payload
  puts so the UCCL D2H ring preserves data-before-signal ordering.
- In UCCL/no-NVLink mode with `local_world_size > 1`, prefer the shared host
  window even when `UCCL_FORCE_NO_GDR=0`. Same-node GPU-window traffic over
  PCIe is slower and less robust than host memcpy on L4.
- Bump `EP_UCCL_PROXY_TRANSPORT_VERSION` whenever JIT-visible device headers,
  barrier layout, or transport semantics change.

## Modification guidelines

- Keep UCCL transport changes scoped to `experimental/lite/ep`.
- Do not modify top-level `ep/` for DeepEPv2 work; that path is legacy
  DeepEP v1 UCCL-EP reference code.
- Keep the UCCL path gated by `EP_USE_UCCL_PROXY=1`; use `UCCL_FORCE_NO_GDR`
  and `EP_FORCE_HOST_WINDOW` to select GDR vs no-GDR for single-local-rank
  runs. Multi-local-rank no-NVLink runs default to shared host windows
  unless `EP_UCCL_FORCE_GPU_WINDOW=1` is set explicitly.
- Do not re-enable NVLink/LSA peer bypass, multiple QPs, or multiple-reduction
  combine without rerunning correctness and performance on the two-node
  validation setup.
- Guard SM90-only instructions/features behind `DISABLE_SM90_FEATURES` or
  equivalent architecture checks.
