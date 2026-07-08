#pragma once
// Runtime dlsym wrapper for libfabric.
// Custom library path: set UCCL_LIBFABRIC_SO.

#include <rdma/fabric.h>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

namespace fabric_dl {

inline void* get_handle() {
  static void* h = [] {
    char const* custom = std::getenv("UCCL_LIBFABRIC_SO");
    void* handle = custom ? dlopen(custom, RTLD_NOW | RTLD_GLOBAL) : nullptr;
    if (!handle) handle = dlopen("libfabric.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) handle = dlopen("libfabric.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      std::fprintf(stderr, "UCCL P2P: failed to load libfabric.so: %s\n",
                   dlerror());
    }
    return handle;
  }();
  return h;
}

inline bool is_available() { return get_handle() != nullptr; }

inline void* resolve(char const* name) {
  void* handle = get_handle();
  if (!handle) {
    std::fprintf(stderr, "UCCL P2P: cannot resolve %s without libfabric\n",
                 name);
    std::abort();
  }
  void* sym = dlsym(handle, name);
  if (!sym) {
    std::fprintf(stderr, "UCCL P2P: dlsym(%s) failed: %s\n", name, dlerror());
    std::abort();
  }
  return sym;
}

}  // namespace fabric_dl
