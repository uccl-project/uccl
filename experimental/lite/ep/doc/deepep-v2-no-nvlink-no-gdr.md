# DeepEPv2 UCCL proxy no-GPUDirect-RDMA notes

This document records the validated no-GPUDirect-RDMA path for
`experimental/lite/ep`, the DeepEPv2 Lite `ElasticBuffer` codebase. The current
implementation ports the UCCL-EP CPU proxy architecture into DeepEPv2. It does
not use NCCL GIN for EP data movement.

## Runtime mode

Validated L4 mode:

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

NCCL remains present for communicator/bootstrap compatibility. With
`EP_USE_UCCL_PROXY=1`, DeepEPv2 dispatch/combine uses UCCL D2H rings plus CPU
proxy threads for the EP transport.

## Architecture

- DeepEPv2 communication/workspace windows are allocated in POSIX shared
  host-pinned memory.
- CUDA maps those host windows so GPU kernels can access them over PCIe.
- GPU kernels post WRITE, PUT_VALUE, and QUIET commands into host-mapped D2H
  rings.
- CPU proxy threads poll the rings and execute host memcpy for same-node peers
  or ibverbs RDMA writes for remote-node peers.
- CPU proxy completion advances ring acknowledgement state; GPU kernels wait on
  that state before reusing send buffers.
- Same-rank operations are bypassed locally; same-node peer operations still use
  the CPU proxy/shared-memory path required by the no-GDR design.

## Main integration points

| Area | Files | Purpose |
| --- | --- | --- |
| Python bootstrap | `deep_ep/buffers/elastic.py` | Enables UCCL mode, exchanges peer metadata with `all_gather_object`, and creates per-run shared-memory IDs. |
| C++ runtime | `csrc/elastic/buffer.hpp` | Allocates host-pinned mapped windows, starts/stops proxy threads, and passes UCCL context into kernels. |
| UCCL runtime | `csrc/uccl/` | Provides shared-buffer allocation, D2H rings, CPU proxy threads, RDMA setup, and ibverbs operations. |
| Device transport | `deep_ep/include/deep_ep/common/handle.cuh` | Implements the UCCL transport shim for `put`, `put_value`, and `flush/quiet`. |
| Barrier | `deep_ep/include/deep_ep/common/comm.cuh` | Uses UCCL `put_value` plus host-mapped polling for DeepEPv2 barriers. |
| Kernels | `deep_ep/include/deep_ep/impls/{dispatch,combine,barrier}.cuh` and `csrc/kernels/elastic/*.hpp` | Threads UCCL D2H addresses and signal state through DeepEPv2 dispatch/combine/barrier kernels. |
| Build/JIT | `setup.py`, `csrc/jit/compiler.hpp` | Builds UCCL proxy sources and adds UCCL JIT cache keys/includes. |
| Launcher | `run_multinode.sh` | Runs l40+l41 tests with UCCL no-GDR defaults. |

## Important fixes made during validation

- Removed legacy DeepEPv1 rank-pair filtering based on `MAX_NUM_GPUS`; DeepEPv2
  world ranks must connect according to actual node/IP topology.
- Routed self-target WRITE/PUT_VALUE/ATOMIC commands to local/shared-memory
  handling instead of posting RDMA to self.
- Avoided full-warp shuffle helpers in UCCL `put`, because DeepEPv2 can call
  transport puts inside divergent top-k branches.
- Allowed each active deduplicated lane to post its own payload command, which
  is required for `topk > 1`.
- Added unique per-run POSIX shared-memory IDs and non-creator size waits to
  avoid stale or partially initialized shared-memory mappings.
- UCCL `quiet` posts and waits across all D2H rings so GPU kernels do not reuse
  send buffers before CPU/verbs completion.

## Benchmark matrix

Configuration:

- Shape: `128 tok x 7168 hid x top-8 x 64 exp`
- Test: `tests/elastic/test_ep.py --test-first-only`
- Correctness checks: enabled (`--skip-check` was not used)
- Precision path: BF16 (`EP_TEST_DISABLE_FP8=1`)
- Runtime: `EP_USE_UCCL_PROXY=1`, `UCCL_FORCE_NO_GDR=1`,
  `EP_FORCE_NO_NVLINK=1`, `NCCL_NET_GDR_LEVEL=0`
- QPs: forced to `1/1`
- Cells are `DeepEPv2 SU BW / legacy BW @ latency`, averaged across ranks. The
  legacy numerator is the old DeepEP low-latency benchmark numerator:
  `valid_topk * hidden * 2` for BF16.

| Setup | Physical GPUs | Dispatch | Combine | Reduced combine | Log |
| --- | --- | ---: | ---: | ---: | --- |
| 1n x 2g | l40: GPU2,3 | 0.80 / 3.17 GB/s @ 4614.500 us | 0.49 / 1.95 GB/s @ 7491.000 us | 0.72 / 0.72 GB/s @ 20348.000 us | `/tmp/uccl_1n2g_final_perf.log` |
| 1n x 4g | l40: GPU0,1,2,3 | 1.12 / 2.47 GB/s @ 5875.250 us | 0.39 / 0.86 GB/s @ 16896.250 us | 0.30 / 0.30 GB/s @ 47774.750 us | `/tmp/uccl_1n4g_final_perf.log` |
| 2n x 1g | l40/l41: GPU2 | 0.64 / 2.54 GB/s @ 5756.500 us | 0.39 / 1.58 GB/s @ 9268.500 us | 0.56 / 0.56 GB/s @ 25953.500 us | `/tmp/uccl_2n1g_final_perf.log` |
| 2n x 4g | l40/l41: GPU0,1,2,3 | 1.18 / 1.73 GB/s @ 8276.250 us | 0.17 / 0.24 GB/s @ 58363.250 us | 0.14 / 0.14 GB/s @ 99801.750 us | `/tmp/uccl_2n4g_final_perf.log` |

Correctness-only validation was also rerun for the same four configurations
with `--skip-perf-test`.

## Reproduction commands

Single-node:

```bash
cd experimental/lite/ep
export PYTHONPATH=$PWD
export EP_NCCL_ROOT_DIR=/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/deps_nccl_2304/nvidia/nccl
export LD_LIBRARY_PATH=$EP_NCCL_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=$EP_NCCL_ROOT_DIR/lib/libnccl.so.2:/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/torch-stubs/libtorch_nvshmem.so
export EP_SUPPRESS_NCCL_CHECK=1
export EP_USE_UCCL_PROXY=1
export UCCL_FORCE_NO_GDR=1
export EP_FORCE_NO_NVLINK=1
export NCCL_NET_GDR_LEVEL=0
export DISABLE_SM90_FEATURES=1
export EP_TEST_DISABLE_FP8=1
export EP_FORCE_PROCESS_EXIT=1
export EP_JIT_CACHE_DIR=/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/jit-cache

CUDA_VISIBLE_DEVICES=2,3 /home/yangz/nfs/miniconda3/bin/python3 \
  tests/elastic/test_ep.py --num-processes=2 \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 \
  --test-first-only --num-cpu-timeout-secs=120 --num-gpu-timeout-secs=120

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/yangz/nfs/miniconda3/bin/python3 \
  tests/elastic/test_ep.py --num-processes=4 \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 \
  --test-first-only --num-cpu-timeout-secs=120 --num-gpu-timeout-secs=120
```

Multi-node:

```bash
cd experimental/lite/ep

bash run_multinode.sh --gpus-per-node 1 --gpu-list 2 \
  --python-bin /home/yangz/nfs/miniconda3/bin/python3 \
  --test-args "--allow-hybrid-mode 0 --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 --test-first-only --num-cpu-timeout-secs=120 --num-gpu-timeout-secs=120"

bash run_multinode.sh --gpus-per-node 4 --gpu-list 0,1,2,3 \
  --python-bin /home/yangz/nfs/miniconda3/bin/python3 \
  --test-args "--allow-hybrid-mode 0 --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 --test-first-only --num-cpu-timeout-secs=120 --num-gpu-timeout-secs=120"
```
