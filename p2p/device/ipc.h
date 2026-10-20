#pragma once

#include "util/gpu_rt.h"

namespace uccl::device {
// Synchronous imports normally use the local device. Some runtimes require
// the allocation owner's context, including copies to/from host buffers.
inline int ipc_open_device(int local, int owner) {
  return Backend::ipc_owner_context && owner >= 0 ? owner : local;
}

// Keep the engine's existing async import policy independent of the sync
// policy: async mappings and completion events stay owned by this context.
inline int ipc_async_device(int local, int owner) {
  return owner >= 0 ? owner : local;
}

template <typename CopyKind>
inline gpuError_t ipc_copy_async(void* dst, int dst_device, void const* src,
                                 int src_device, size_t size, bool host,
                                 CopyKind kind, gpuStream_t stream) {
  if constexpr (Backend::ipc_owner_context) {
    if (!host && dst_device != src_device)
      return gpuMemcpyPeerAsync(dst, dst_device, const_cast<void*>(src),
                                src_device, size, stream);
  }
  return gpuMemcpyAsync(dst, src, size, kind, stream);
}
}  // namespace uccl::device
