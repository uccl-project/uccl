# DeepEPv2 no-NVLink/no-GPUDirect RDMA migration notes

This document records the changes made to migrate `experimental/lite/ep` from the
old UCCL-EP/DeepEPv1-style implementation to DeepEPv2 PR605's NCCL GIN
`ElasticBuffer` path, and the extra adaptations needed to make it run on L4
without NVLink, IBGDA, UCCL-EP, or GPUDirect RDMA.

The important conclusion is that NCCL GIN made the migration possible, but the
upstream DeepEPv2 code still needed runtime and kernel changes for this
environment. It was not a macro-only change.

## Runtime mode used on L4

The validated mode uses:

```bash
EP_FORCE_NO_NVLINK=1
NCCL_NET_GDR_LEVEL=0
NCCL_GIN_TYPE=2
DISABLE_SM90_FEATURES=1
EP_TEST_DISABLE_FP8=1
EP_NCCL_ROOT_DIR=/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/deps_nccl_2304/nvidia/nccl
LD_PRELOAD=$EP_NCCL_ROOT_DIR/lib/libnccl.so.2:/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/torch-stubs/libtorch_nvshmem.so
```

- `NCCL_GIN_TYPE=2` is NCCL GIN proxy mode. The default GDAKI mode timed out or
  wedged in this no-GPUDirect setup.
- `NCCL_NET_GDR_LEVEL=0` disables GPUDirect RDMA.
- `EP_FORCE_NO_NVLINK=1` makes the runtime treat all peer GPU traffic as
  network/proxy GIN traffic rather than NVLink/LSA traffic.
- `DISABLE_SM90_FEATURES=1` enables SM89 fallbacks for NVIDIA L4.
- `EP_TEST_DISABLE_FP8=1` keeps the validation path BF16. FP8 can be revisited
  separately; the migration target here was DeepEPv2 correctness in no-GDR mode.

## File-level modification summary

| Area | Files | What changed | Why it is needed |
| --- | --- | --- | --- |
| DeepEPv2 import | `deep_ep/`, `csrc/`, `tests/elastic/`, `tests/utils/`, `LICENSE.deepep_v2`, `README.md` | Added DeepEPv2 PR605 package/runtime/tests and removed the old UCCL-EP source, wrappers, benches, and preserved-doc copy. | Replaces the old `uccl.ep` path with DeepEPv2 `ElasticBuffer`. |
| Build/install | `setup.py`, `Makefile` | Builds `deep_ep._C`, removes the legacy NVSHMEM build branch, supports NCCL 2.30.4 root/rpath, defaults SM89 fallback for L4, and adds a no-GDR test target. | Upstream needs NCCL host/device GIN APIs from 2.30.4; bundled PyTorch NCCL was insufficient. |
| NCCL runtime backend | `csrc/kernels/backend/nccl.cu`, `api.cuh`, `lazy_driver.hpp` | Reworked communicator/window setup, no-NVLink logical domains, proxy GIN requirements, optional host-window experiment, and lazy CUDA driver symbols. | Upstream assumes LSA/NVLink domains and GDR-like behavior; L4 no-GDR needs a pure GIN proxy path. |
| GIN address helper | `deep_ep/include/deep_ep/common/handle.cuh` | Added explicit local base pointer support, no-NVLink accessibility rules, and `put_value`. | GIN offsets must be relative to the NCCL-registered window, not an unavailable remote LSA pointer. |
| Workspace layout | `deep_ep/include/deep_ep/common/layout.cuh` | Reserved per-rank GIN barrier slots in workspace. | Needed for a barrier that works with proxy GIN without relying on NCCL indexed signals. |
| Barrier | `deep_ep/include/deep_ep/common/comm.cuh` | Replaced failing GIN signal/readSignal barrier with `put_value` into peer workspace + local polling. | NCCL GIN proxy did not reliably deliver/read the upstream signal barrier cross-node in no-GDR mode. |
| Dispatch/combine kernels | `deep_ep/include/deep_ep/impls/dispatch.cuh`, `combine.cuh`, `barrier.cuh`, `hybrid_dispatch.cuh`, `hybrid_combine.cuh` | Construct `NCCLGin` with explicit workspace base, add GIN flush/fence ordering where send buffers are consumed by proxy GIN, and keep final barriers. | Prevents proxy GIN from reading stale GPU send-buffer contents and ensures offsets are correct. |
| SM89 fallback | `deep_ep/include/deep_ep/common/ptx.cuh`, `csrc/jit/launch_runtime.hpp`, generated kernels | Added fallbacks for SM90-only `elect.sync`, TMA/mbarrier, PDL, cluster launch, and register allocation paths. | L4 is SM89; upstream PR605 targets Hopper/SM90-style device instructions. |
| Runtime policy | `deep_ep/buffers/elastic.py`, `tests/elastic/test_ep.py` | `EP_FORCE_NO_NVLINK=1` forces non-hybrid direct mode, disables multiple-reduction combine, and clamps to one GIN QP; tests mirror those settings. | Multiple QPs plus proxy GIN `flush` were not sufficient to order remote completion; single QP preserves correctness. |
| Legacy cleanup | `src/`, `include/`, `bench/`, `deep_ep_wrapper/`, `csrc/legacy/`, `csrc/kernels/legacy/`, `deep_ep/buffers/legacy.py`, `csrc/kernels/backend/nvshmem.cu` | Removed the old UCCL-EP implementation and the unused DeepEP V1/NVSHMEM build path from this lite tree. | Keeps the package focused on the validated DeepEPv2/NCCL GIN path and avoids stale code paths that are no longer tested. |
| Launcher | `run_multinode.sh` | Uses l40+l41, physical GPU list, NCCL 2.30.4, proxy GIN by default, no GDR, and process-exit workaround. | Provides reproducible multi-node runs on the target machines. |
| JIT cache | `csrc/jit/compiler.hpp` | Adds cache invalidation flags and waits for cache visibility after concurrent atomic rename on NFS. | Multi-rank JIT compilation on NFS can otherwise lose the winner's generated cubin briefly. |

## Code changes that are not just macros

### 1. No-NVLink topology and registered-window mapping

`csrc/kernels/backend/nccl.cu` changes the physical/logical domain model when
`EP_FORCE_NO_NVLINK=1`:

- `get_physical_domain_size()` returns `(num_ranks, 1)`, i.e. every rank is a
  non-NVLink peer.
- The runtime sets `num_nvl_ranks=1`, `nvl_rank_idx=0`,
  `num_rdma_ranks=num_ranks`, and `rdma_rank_idx=rank_idx`.
- In non-hybrid mode it maps all ranks into one scale-up domain:
  `num_scaleout_ranks=1`, `num_scaleup_ranks=num_ranks`.
- `nvl_window_ptrs` contains only the local mapped window pointer in no-NVLink
  mode, so peer access goes through GIN instead of LSA pointer translation.

This is a real runtime behavior change. Without it, upstream DeepEPv2 tries to
interpret local LSA/NVLink domains that do not exist on L4 PCIe-only machines.

### 2. NCCL GIN proxy communicator setup

The same backend queries NCCL communicator properties and configures
`ncclDevCommRequirements_t`. For proxy GIN it intentionally does not set a
custom `ginQueueDepth`, because proxy mode rejects that field. The launcher
defaults `NCCL_GIN_TYPE=2`, selecting NCCL's proxy backend.

This was necessary because the default NCCL selection on these nodes chose a
GDAKI-like path that assumes GPU-direct signaling/progress behavior and timed
out in no-GDR runs.

### 3. Explicit GIN base pointer and offset calculation

`common/handle.cuh` extends `NCCLGin` with an optional `local_base_ptr`.
All GIN `put`, `get`, `putValue`, and VA-signal offsets are computed as:

```cpp
reinterpret_cast<uint64_t>(ptr) - lsa_base_ptr
```

where `lsa_base_ptr` is now explicitly set to the local registered workspace
base when constructed by dispatch/combine/barrier kernels.

This matters because in no-NVLink mode `ncclGetLsaPointer()` is not a valid way
to derive a remote-accessible base for all peers. The registered window offset
must match the pointer passed to `ncclCommWindowRegister`.

### 4. No-NVLink peer accessibility

`NCCLGin::is_nvlink_accessible()` is changed under `EP_FORCE_NO_NVLINK` so only
the local rank is considered symmetric-pointer accessible. Every other peer
returns `nullptr` and falls back to GIN put/get.

That prevents accidental local-store/TMA bypass to peer memory when no NVLink
or peer LSA mapping exists.

### 5. Proxy-safe GIN barrier

Upstream DeepEPv2 used GIN signal/readSignal-style barriers. On l40/l41 with
proxy GIN and `NCCL_NET_GDR_LEVEL=0`, those barriers either timed out or failed
to order subsequent kernels.

The new barrier in `common/comm.cuh`:

1. Flushes the data QP(s) for source-buffer reuse.
2. On SM0, uses QP0 to write a monotonically increasing per-rank value into each
   peer's registered workspace via `gin.put_value`.
3. Polls local `workspace.get_gin_barrier_signal_ptr(peer_rank)` with
   `ld_acquire_sys` until every peer's value reaches the expected generation.

`common/layout.cuh` reserves `kNumMaxRanks * sizeof(uint64_t)` for these
barrier slots and shifts the following workspace sections.

This is one of the key code changes: it avoids relying on NCCL proxy signal
delivery and instead uses normal GIN put-value traffic into the same registered
window as the data path.

### 6. Dispatch/combine send-buffer ordering

In `dispatch.cuh` and `combine.cuh`, before issuing GIN puts from GPU send
buffers, the kernels now wait for the local TMA/global stores and add a
system-scope fence:

```cpp
ptx::tma_store_wait();
ptx::fence_acq_rel_sys();
```

This is needed because NCCL proxy GIN has a CPU/proxy progress path. Without the
fence, the proxy can observe the descriptor and consume the source buffer before
all GPU writes to that buffer are globally visible.

### 7. Single-QP conservative policy

NCCL documents GIN `flush` as making source buffers safe to reuse; it does not
guarantee that remote memory for all preceding puts on all QPs is settled before
a later barrier on a different QP. In practice, multi-QP proxy GIN produced
combine correctness mismatches.

Therefore `deep_ep/buffers/elastic.py` forces:

- `num_allocated_qps = 1`
- `get_theoretical_num_qps(...) = 1`
- explicit dispatch/combine `num_qps = 1`

whenever `EP_FORCE_NO_NVLINK=1`.

This is a throughput tradeoff, not a functional requirement of DeepEPv2 in
general. It is the conservative correctness setting for this no-GDR proxy path.

### 8. Disabling multiple-reduction combine in no-GDR proxy mode

The upstream multiple-reduction path was correct in normal DeepEPv2 environments
but produced reduced-combine mismatches in this no-GDR proxy setup. The fallback
sets `allow_multiple_reduction=False`, which uses the single-reduction/expanded
send path. This increases communication volume but keeps BF16 combine results
bitwise-correct against the test reference.

Both `ElasticBuffer` construction and `tests/elastic/test_ep.py` override the
setting when `EP_FORCE_NO_NVLINK=1`, so benchmark commands cannot accidentally
run the unsafe path.

### 9. SM89 fallback implementation

The upstream kernels use SM90 features such as `elect.sync`, TMA/mbarrier, PDL,
cluster launch attributes, and warpgroup register allocation controls. For L4:

- `ptx::elect_one_sync()` falls back to lane 0 election.
- TMA load/store helpers fall back to ordinary byte/global copies.
- mbarrier helpers become no-ops or phase flips.
- PDL calls are compiled out under `DISABLE_SM90_FEATURES`.
- JIT launch disables cluster dimensions and PDL when `DISABLE_SM90_FEATURES`
  is set.

This is code-level behavior, even though it is selected by a compile-time macro.

### 10. Host-window experiment kept off

`EP_FORCE_HOST_WINDOW` adds an experimental CPU-backed CUDA VMM allocation path
for the registered NCCL window. It is default off because full EP data buffers
backed by host NUMA memory caused illegal-address failures on L4. The code is
left as an explicit experiment, not part of the validated path.

## Macro/config-only changes

These are mostly selectors or cache invalidators; they do not by themselves make
the no-GDR mode correct:

- `DISABLE_SM90_FEATURES`: selects the SM89-compatible code paths.
- `EP_FORCE_NO_NVLINK`: selects no-NVLink runtime policy and JIT code paths.
- `DEEP_EP_*` JIT signature defines: invalidate stale cached cubins after the
  barrier/base/fence changes.
- `EP_SUPPRESS_NCCL_CHECK`: allows running with PyTorch already having loaded a
  different NCCL while `LD_PRELOAD` supplies NCCL 2.30.4. The real requirement is
  still to preload/link NCCL 2.30.4 for GIN host APIs.

## Benchmark matrix

Configuration:

- Shape: `128 tok x 7168 hid x top-8 x 64 exp`
- Test: `tests/elastic/test_ep.py --test-first-only`
- Correctness checks: enabled (`--skip-check` was not used)
- Precision path: BF16 (`EP_TEST_DISABLE_FP8=1`)
- Runtime: `EP_FORCE_NO_NVLINK=1`, `NCCL_NET_GDR_LEVEL=0`, `NCCL_GIN_TYPE=2`
- PCIe direct P2P is not explicitly disabled in NCCL for this matrix.
- QPs: forced to `1/1`
- Bandwidth columns below are the `SU` bottleneck bandwidth printed by DeepEP.
  `SO` is `0 GB/s` in this direct non-hybrid logical layout.
- Legacy BW columns use the old DeepEP low-latency benchmark numerator:
  valid top-k selections times the per-selection payload bytes. For the BF16
  path here, that is `valid_topk * hidden * 2` bytes per rank.
- Values are averages across ranks in that run.

| Setup | Physical GPUs | Ranks | Dispatch BW | Dispatch latency | Combine BW | Combine latency | Reduced combine BW | Reduced combine latency | Legacy BW (D / C / RC) | Log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1n x 2g | l40: GPU2,3 | 2 | 6.00 GB/s | 616.961 us | 6.00 GB/s | 580.603 us | 6.00 GB/s | 2507.500 us | 23.70 / 25.19 / 5.83 GB/s | `/tmp/deepep_matrix_1n2g.log` |
| 1n x 4g | l40: GPU0,1,2,3 | 4 | 7.75 GB/s | 863.959 us | 6.25 GB/s | 1027.750 us | 4.00 GB/s | 3510.250 us | 16.79 / 14.12 / 4.13 GB/s | `/tmp/deepep_matrix_1n4g.log` |
| 2n x 1g | l40/l41: GPU2 | 2 | 6.00 GB/s | 586.572 us | 7.00 GB/s | 547.059 us | 6.00 GB/s | 2609.000 us | 24.93 / 26.73 / 5.60 GB/s | `/tmp/deepep_matrix_2n1g.log` |
| 2n x 2g | l40/l41: GPU2,3 | 4 | 7.00 GB/s | 926.900 us | 6.00 GB/s | 1048.000 us | 4.00 GB/s | 3469.750 us | 15.65 / 13.84 / 4.18 GB/s | `/tmp/deepep_matrix_2n2g.log` |
| 2n x 4g | l40/l41: GPU0,1,2,3 | 8 | 5.00 GB/s | 1904.750 us | 5.00 GB/s | 2074.875 us | 5.25 GB/s | 2680.375 us | 7.50 / 6.88 / 5.33 GB/s | `/tmp/deepep_matrix_2n4g.log` |

Notes:

- The 4g cases necessarily use GPU0-3. This differs from the original GPU2/3
  constraint because a 4-GPU-per-node benchmark cannot be run on only two
  physical GPUs.
- The warning `Failed to get NVLink connection speed` is expected in this mode;
  the runtime forces no-NVLink behavior and does not depend on the NVLink speed
  query succeeding.
- The performance is intentionally conservative because no-GDR proxy GIN uses
  one QP and avoids multiple-reduction combine for correctness.

### GPUDirect RDMA enabled inter-node results

This mode keeps the same DeepEPv2 no-NVLink policy and GIN proxy backend, but
enables GPUDirect RDMA by setting:

```bash
NCCL_NET_GDR_LEVEL=SYS
NCCL_GIN_TYPE=2
```

The 2n x 1g debug log confirms NCCL network channels such as
`via NET/IB/0/GDRDMA`. These are the previously recorded GDR runs; they did
not explicitly add `NCCL_P2P_DISABLE=1`, although the DeepEP data path is still
forced through the no-NVLink policy. Values are averages across ranks.

| Setup | Physical GPUs | Ranks | Dispatch BW | Dispatch latency | Expanded dispatch BW | Expanded dispatch latency | Cached dispatch BW | Cached dispatch latency | Combine BW | Combine latency | Reduced combine BW | Reduced combine latency | Log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2n x 1g | l40/l41: GPU2 | 2 | 6.00 GB/s | 582.707 us | 6.00 GB/s | 583.630 us | 6.00 GB/s | 581.384 us | 7.00 GB/s | 547.618 us | 6.00 GB/s | 2603.500 us | `/tmp/deepep_gdr_2n1g.log` |
| 2n x 2g | l40/l41: GPU2,3 | 4 | 7.00 GB/s | 927.808 us | 7.00 GB/s | 927.287 us | 7.00 GB/s | 926.298 us | 6.00 GB/s | 1041.000 us | 5.00 GB/s | 2879.500 us | `/tmp/deepep_gdr_2n2g.log` |
| 2n x 4g | l40/l41: GPU0,1,2,3 | 8 | 5.00 GB/s | 1896.875 us | 5.00 GB/s | 1911.750 us | 5.00 GB/s | 1906.375 us | 5.00 GB/s | 2083.625 us | 5.00 GB/s | 2961.625 us | `/tmp/deepep_gdr_2n4g.log` |

Legacy BW (D / ED / CD / C / RC):

| Setup | Legacy BW |
| --- | ---: |
| 2n x 1g | 25.09 / 25.05 / 25.15 / 26.70 / 5.62 GB/s |
| 2n x 2g | 15.64 / 15.65 / 15.66 / 13.94 / 5.04 GB/s |
| 2n x 4g | 7.53 / 7.47 / 7.49 / 6.85 / 4.82 GB/s |

### No direct P2P + no GPUDirect RDMA intra-node results

This mode adds:

```bash
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=0
NCCL_NET_DISABLE_INTRA=1
NCCL_NET_GDR_LEVEL=0
```

With these settings, NCCL's ordinary intra-node channels can be forced to SHM
(`via SHM/direct/direct` in NCCL logs), but the DeepEPv2 data path still requires
NCCL GIN and remains `GIN_IB_PROXY`. If IB is fully disabled, GIN initialization
fails with `GIN is unavailable`; there is no validated GIN-over-SHM path in this
NCCL build.

| Setup | Physical GPUs | Ranks | Dispatch BW | Dispatch latency | Expanded dispatch BW | Expanded dispatch latency | Cached dispatch BW | Cached dispatch latency | Combine BW | Combine latency | Reduced combine BW | Reduced combine latency | Log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1n x 2g | l40: GPU2,3 | 2 | 6.00 GB/s | 615.182 us | 6.00 GB/s | 616.032 us | 6.00 GB/s | 613.049 us | 6.00 GB/s | 577.398 us | 6.00 GB/s | 2549.000 us | `/tmp/deepep_gin_shm_1n2g.log` |
| 1n x 4g | l40: GPU0,1,2,3 | 4 | 7.75 GB/s | 862.379 us | 8.00 GB/s | 861.427 us | 8.00 GB/s | 860.244 us | 6.50 GB/s | 1015.250 us | 4.00 GB/s | 3507.750 us | `/tmp/deepep_gin_shm_1n4g.log` |

Legacy BW (D / ED / CD / C / RC):

| Setup | Legacy BW |
| --- | ---: |
| 1n x 2g | 23.77 / 23.74 / 23.85 / 25.33 / 5.74 GB/s |
| 1n x 4g | 16.82 / 16.84 / 16.86 / 14.29 / 4.14 GB/s |

## Reproduction commands

Single-node:

```bash
cd experimental/lite/ep
export PYTHONPATH=$PWD
export EP_NCCL_ROOT_DIR=/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/deps_nccl_2304/nvidia/nccl
export LD_LIBRARY_PATH=$EP_NCCL_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=$EP_NCCL_ROOT_DIR/lib/libnccl.so.2:/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/torch-stubs/libtorch_nvshmem.so
export EP_SUPPRESS_NCCL_CHECK=1
export EP_FORCE_NO_NVLINK=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_GIN_TYPE=2
export DISABLE_SM90_FEATURES=1
export EP_TEST_DISABLE_FP8=1
export EP_FORCE_PROCESS_EXIT=1
export EP_JIT_CACHE_DIR=/home/yangz/nfs/zhongjie/copilot-deps/deepep-v2/jit-cache

# Optional: disable direct P2P and force NCCL ordinary intra-node channels to SHM.
# GIN itself remains GIN_IB_PROXY.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_NET_DISABLE_INTRA=1

CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=34453 \
  /home/yangz/nfs/miniconda3/envs/uccl/bin/python tests/elastic/test_ep.py \
  --num-processes 2 --allow-hybrid-mode 0 \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 \
  --test-first-only --num-cpu-timeout-secs=120 --num-gpu-timeout-secs=120

CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=34531 \
  /home/yangz/nfs/miniconda3/envs/uccl/bin/python tests/elastic/test_ep.py \
  --num-processes 4 --allow-hybrid-mode 0 \
  --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 \
  --test-first-only --num-cpu-timeout-secs=120 --num-gpu-timeout-secs=120
```

Multi-node:

```bash
cd experimental/lite/ep

bash run_multinode.sh --gpus-per-node 2 --gpu-list 2,3 \
  --master-port 34620 \
  --test-args "--allow-hybrid-mode 0 --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 --test-first-only --num-cpu-timeout-secs=120 --num-gpu-timeout-secs=120"

bash run_multinode.sh --gpus-per-node 1 --gpu-list 2 \
  --master-port 34811 \
  --test-args "--allow-hybrid-mode 0 --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 --test-first-only --num-cpu-timeout-secs=120 --num-gpu-timeout-secs=120"

bash run_multinode.sh --gpus-per-node 4 --gpu-list 0,1,2,3 \
  --master-port 34706 \
  --test-args "--allow-hybrid-mode 0 --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 --test-first-only --num-cpu-timeout-secs=120 --num-gpu-timeout-secs=120"

# GPUDirect RDMA enabled inter-node mode:
NCCL_NET_GDR_LEVEL=SYS bash run_multinode.sh --gpus-per-node 1 --gpu-list 2 \
  --master-port 35101 \
  --test-args "--allow-hybrid-mode 0 --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=64 --test-first-only --num-cpu-timeout-secs=120 --num-gpu-timeout-secs=120"
```
