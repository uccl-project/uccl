#pragma once
// Runtime dlsym wrapper for libnccl (or librccl on ROCm).
// Uses nccl_types.h for type definitions - no nccl.h needed at compile time.
// Custom library path: set UCCL_NCCL_SO environment variable.

#include "nccl_types.h"
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <dlfcn.h>

namespace uccl {
namespace nccl_dl {

inline void* get_handle() {
  static void* h = [] {
    char const* custom = std::getenv("UCCL_NCCL_SO");
    void* handle = custom ? dlopen(custom, RTLD_NOW) : nullptr;
    if (!handle) handle = dlopen("libnccl.so", RTLD_NOW);
    if (!handle) handle = dlopen("libnccl.so.2", RTLD_NOW);
    if (!handle) handle = dlopen("librccl.so", RTLD_NOW);  // ROCm
    if (!handle) {
      std::fprintf(stderr, "UCCL P2P: failed to load libnccl.so: %s\n",
                   dlerror());
    }
    return handle;
  }();
  return h;
}

inline bool is_available() { return get_handle() != nullptr; }

inline void* resolve(char const* name) {
  void* sym = dlsym(get_handle(), name);
  if (!sym) {
    std::fprintf(stderr, "UCCL P2P: dlsym(%s) failed: %s\n", name, dlerror());
    std::abort();
  }
  return sym;
}

#define UCCL_NCCL_WRAP(func_name)                                     \
  template <typename... Args>                                         \
  inline auto func_name(Args&&... args)                               \
      ->decltype(::func_name(std::forward<Args>(args)...)) {          \
    using FnType = decltype(&::func_name);                            \
    static FnType fn = reinterpret_cast<FnType>(resolve(#func_name)); \
    return fn(std::forward<Args>(args)...);                           \
  }

UCCL_NCCL_WRAP(ncclGetUniqueId)
UCCL_NCCL_WRAP(ncclCommInitRank)
UCCL_NCCL_WRAP(ncclCommDestroy)
UCCL_NCCL_WRAP(ncclSend)
UCCL_NCCL_WRAP(ncclRecv)
UCCL_NCCL_WRAP(ncclGetErrorString)

#undef UCCL_NCCL_WRAP

}  // namespace nccl_dl
}  // namespace uccl
