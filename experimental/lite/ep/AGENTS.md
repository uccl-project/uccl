# DeepEP-Lite

`experimental/lite/ep` is the DeepEPv2 Lite `ElasticBuffer` codebase. The
validated no-GPUDirect-RDMA path in this branch uses a UCCL-EP style CPU proxy
transport, not NCCL GIN for the EP data path.

Target environment:

- NVIDIA L4 / `sm_89`
- no NVLink
- no GPUDirect RDMA
- DeepEPv2 dispatch/combine kernels
- UCCL CPU proxy with host-pinned shared windows and ibverbs

## Required UCCL no-GDR runtime mode

```bash
EP_USE_UCCL_PROXY=1
UCCL_FORCE_NO_GDR=1
EP_FORCE_NO_NVLINK=1
NCCL_NET_GDR_LEVEL=0
DISABLE_SM90_FEATURES=1
EP_TEST_DISABLE_FP8=1
EP_NCCL_ROOT_DIR=/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/deps_nccl_2304/nvidia/nccl
LD_PRELOAD=$EP_NCCL_ROOT_DIR/lib/libnccl.so.2:/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/torch-stubs/libtorch_nvshmem.so
LD_LIBRARY_PATH=$EP_NCCL_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}
```

Notes:

- NCCL is still loaded for communicator/bootstrap compatibility, but
  `EP_USE_UCCL_PROXY=1` switches the EP transport to the UCCL CPU proxy path.
- GPU-visible RDMA/workspace windows are allocated in POSIX shared host-pinned
  memory and mapped into CUDA. GPU kernels post D2H ring commands; CPU proxy
  threads perform host memcpy for same-node peers or ibverbs RDMA writes for
  remote-node peers.
- `EP_FORCE_NO_NVLINK=1` forces the conservative DeepEPv2 policy used here:
  one QP and no multiple-reduction combine.

## Build and test

Activate the node-specific Python environment before rebuilding:

```bash
# l40
conda activate uccl

# l41
source ~/zhongjie/zj_py/bin/activate
```

```bash
make -j SM=89
make -j install
```

Single-node example:

```bash
CUDA_VISIBLE_DEVICES=2,3 /home/yangz/nfs/miniconda3/bin/python3 \
  tests/elastic/test_ep.py --num-processes=2 \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 \
  --test-first-only --num-cpu-timeout-secs=120 --num-gpu-timeout-secs=120
```

Multi-node example:

```bash
bash run_multinode.sh --gpus-per-node 1 --gpu-list 2 \
  --python-bin /home/yangz/nfs/miniconda3/bin/python3 \
  --test-args "--allow-hybrid-mode 0 --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 --test-first-only --num-cpu-timeout-secs=120 --num-gpu-timeout-secs=120"
```

Constraint: `num_experts` must be divisible by total GPU count.

## UCCL no-GPUDirect-RDMA benchmark (L4, 128 tok x 7168 hid x top-8 x 64 exp)

All runs used DeepEPv2 `tests/elastic/test_ep.py --test-first-only` with
correctness checks enabled, BF16 (`EP_TEST_DISABLE_FP8=1`), no NVLink, one QP,
and UCCL CPU proxy no-GDR mode. Cells are `DeepEPv2 SU BW / legacy BW @
latency`, averaged across ranks. Legacy BW uses the old DeepEP low-latency
numerator (`valid_topk * hidden * 2` for BF16).

| Setup | Physical GPUs | Dispatch | Combine | Reduced combine |
| --- | --- | ---: | ---: | ---: |
| 1n x 2g | l40: GPU2,3 | 0.80 / 3.17 GB/s @ 4614.500 us | 0.49 / 1.95 GB/s @ 7491.000 us | 0.72 / 0.72 GB/s @ 20348.000 us |
| 1n x 4g | l40: GPU0,1,2,3 | 1.12 / 2.47 GB/s @ 5875.250 us | 0.39 / 0.86 GB/s @ 16896.250 us | 0.30 / 0.30 GB/s @ 47774.750 us |
| 2n x 1g | l40/l41: GPU2 | 0.64 / 2.54 GB/s @ 5756.500 us | 0.39 / 1.58 GB/s @ 9268.500 us | 0.56 / 0.56 GB/s @ 25953.500 us |
| 2n x 4g | l40/l41: GPU0,1,2,3 | 1.18 / 1.73 GB/s @ 8276.250 us | 0.17 / 0.24 GB/s @ 58363.250 us | 0.14 / 0.14 GB/s @ 99801.750 us |

Correctness-only validation was also rerun for the same four configurations
with `--skip-perf-test`.

## Modification guidelines

- Keep UCCL transport changes scoped to `experimental/lite/ep`.
- Do not modify top-level `ep/` for DeepEPv2 work; it is legacy DeepEP v1
  UCCL-EP reference code.
- Keep the no-GDR path gated by `EP_USE_UCCL_PROXY=1` and `UCCL_FORCE_NO_GDR=1`.
- Do not re-enable NVLink/LSA peer bypass, multiple QPs, or multiple-reduction
  combine in this mode without rerunning correctness on l40+l41.
- Guard SM90-only instructions/features behind `DISABLE_SM90_FEATURES` or
  equivalent architecture checks.
