#pragma once

#include "util/gpu_rt.h"
#include <map>
#include <mutex>
#include <utility>

namespace uccl::device {
// Whether `local` can map memory that lives on `owner`. Both are ordinals in
// this process. An importer without peer access cannot map another GPU's IPC
// handle into its own context (for example PCIe GPUs with P2P disabled, or
// most virtualized hosts), so the handle has to be opened in the owner's
// context instead. Cached per pair; an ordinal that is not visible here is
// treated as accessible so the caller keeps its previous behavior.
inline bool peer_accessible(int local, int owner) {
  if (owner < 0 || local < 0 || owner == local) return true;
  static std::mutex mu;
  static std::map<std::pair<int, int>, bool> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find({local, owner});
  if (it != cache.end()) return it->second;
  int count = 0;
  bool accessible = true;
  if (gpuGetDeviceCount(&count) == gpuSuccess && local < count &&
      owner < count) {
    int can = 0;
    if (gpuDeviceCanAccessPeer(&can, local, owner) == gpuSuccess)
      accessible = can != 0;
  }
  (void)gpuGetLastError();
  cache[{local, owner}] = accessible;
  return accessible;
}

// Synchronous imports normally use the local device. Some runtimes require
// the allocation owner's context, including copies to/from host buffers, and
// so does any import without peer access to the owner.
inline bool ipc_use_owner_context(int local, int owner) {
  return owner >= 0 &&
         (Backend::ipc_owner_context || !peer_accessible(local, owner));
}

inline int ipc_open_device(int local, int owner) {
  return ipc_use_owner_context(local, owner) ? owner : local;
}

// CUDA/HIP handles map into the importer's local context; exporter ordinals
// belong to its own visible devices. Direct addresses are in the same process.
inline int ipc_async_device(int local, int owner, bool direct_address) {
  return (Backend::ipc_async_owner_context || direct_address ||
          ipc_use_owner_context(local, owner)) &&
                 owner >= 0
             ? owner
             : local;
}

template <typename CopyKind>
inline gpuError_t ipc_copy_async(void* dst, int dst_device, void const* src,
                                 int src_device, size_t size, bool host,
                                 CopyKind kind, gpuStream_t stream) {
  // Device-to-device across GPUs without a mapping into one context: let the
  // runtime route the copy between the two contexts.
  if (!host && dst_device != src_device &&
      (Backend::ipc_owner_context || !peer_accessible(src_device, dst_device)))
    return gpuMemcpyPeerAsync(dst, dst_device, const_cast<void*>(src),
                              src_device, size, stream);
  return gpuMemcpyAsync(dst, src, size, kind, stream);
}
}  // namespace uccl::device
