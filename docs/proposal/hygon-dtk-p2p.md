# Proposal: Hygon DCU P2P Support via DTK

## Background

Hygon (海光) produces DCUs (Deep Computing Units) that are widely deployed in
Chinese data centers. Hygon ships the **DTK** (Deep Technology Kit), a
HIP-compatible SDK that mirrors AMD ROCm in API surface but is built for
Hygon's GCN-derived hardware.

UCCL already supports AMD GPU P2P via the ROCm/HIP code path. This proposal
extends P2P transport support to Hygon DCUs by adapting the existing HIP
abstractions for DTK.

## DTK vs. AMD ROCm: Key Differences

| Feature | AMD ROCm | Hygon DTK |
|---|---|---|
| SDK root | `/opt/rocm` | `/opt/dtk` |
| HIP runtime lib | `libamdhip64.so` | `libamdhip64.so` (same name) |
| HIP IPC API | Yes | Yes |
| `hipDeviceCanAccessPeer` | Yes | Yes |
| RCCL | `/opt/rocm/lib/librccl.so` | `/opt/dtk/rccl/lib/librccl.so` |
| `hipMemGetHandleForAddressRange` (DMA-BUF) | Yes | **No** |
| Build macro | `-D__HIP_PLATFORM_AMD__` | `-D__HIP_PLATFORM_AMD__` |

The only missing piece is the DMA-BUF GPU direct RDMA path
(`hipMemGetHandleForAddressRange` / `hipMemRangeHandleTypeDmaBufFd`), which is
a performance optimization. Its absence does not block RDMA transport; UCCL
falls back to standard `ibv_reg_mr` memory registration automatically.

## Design

### 1. `include/util/gpu_rt.h`

Guard the DMA-BUF type definitions with a new compile-time flag
`__HAS_HIP_DMABUF__`. Provide a stub (`typedef int gpuMemRangeHandleType`)
when the flag is absent so the existing `reg_mem_gpu_dmabuf()` code compiles
on DTK (the function is a no-op at runtime since `dlsym` returns null).

```cpp
#ifdef __HAS_HIP_DMABUF__
#define gpuMemRangeHandleType hipMemRangeHandleType
#define GPU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD hipMemRangeHandleTypeDmaBufFd
#define GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME \
  "hipMemGetHandleForAddressRange"
#else
typedef int gpuMemRangeHandleType;  // stub: DMA-BUF unavailable
#define GPU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD 0
#define GPU_DRIVER_GET_HANDLE_FOR_ADDRESS_RANGE_NAME ""
#endif
```

### 2. `p2p/Makefile.dtk` (new file)

A copy of `Makefile.rocm` with two key differences:

- `HIP_HOME ?= /opt/dtk` — points to Hygon's DTK installation
- `-D__HAS_HIP_DMABUF__` is **not** added (DMA-BUF disabled on DTK)

All other build flags, source files, and link options are identical to
`Makefile.rocm`.

### 3. `p2p/Makefile.rocm` and `p2p/Makefile.therock`

Add `-D__HAS_HIP_DMABUF__` to their `CXXFLAGS` to preserve existing
AMD ROCm behavior.

## Runtime Behaviour on Hygon DCU

| Transport | Works on DTK | Notes |
|---|---|---|
| RDMA (default) | Yes | `ibv_reg_mr` path used instead of DMA-BUF |
| NCCL/RCCL | Yes | Set `UCCL_NCCL_SO=/opt/dtk/rccl/lib/librccl.so` |
| GPU IPC (intra-node P2P) | Yes | `hipIpcGetMemHandle` / `hipIpcOpenMemHandle` |
| DMA-BUF GPUDirect RDMA | No | DTK lacks `hipMemGetHandleForAddressRange` |

## Key Data Structures

No new data structures are required. The proposal reuses:

- `gpuIpcMemHandle_t` → `hipIpcMemHandle_t` (from `gpu_rt.h`)
- `IpcTransferInfo` (in `p2p/engine.h`) — unchanged
- `RdmaContext` (in `p2p/rdma/rdma_context.{h,cc}`) — DMA-BUF path silently
  disabled at runtime when `gpuGetHandleForRange_func` is null

## Build Instructions

```bash
cd p2p
make clean -f Makefile.dtk
make -j$(nproc) -f Makefile.dtk
make install -f Makefile.dtk
```

To override the DTK path:

```bash
make -j$(nproc) -f Makefile.dtk HIP_HOME=/opt/dtk-23.10.1
```

For NCCL/RCCL transport:

```bash
export UCCL_P2P_TRANSPORT=nccl
export UCCL_NCCL_SO=/opt/dtk/rccl/lib/librccl.so
```

## Future Work

- DMA-BUF support once Hygon DTK adds `hipMemGetHandleForAddressRange`
- Add `dtk` target to `build.sh` / `build_inner.sh` for container-based builds
- Collective communication support (`collective/rdma/Makefile.dtk`)
