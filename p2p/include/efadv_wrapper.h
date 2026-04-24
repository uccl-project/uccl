#pragma once
// Runtime dlsym wrapper for libefa (EFA vendor extensions).
// Keeps #include <infiniband/efadv.h> for types and structs.
// Custom library path: set UCCL_EFA_SO environment variable.

#include <infiniband/efadv.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace uccl {
namespace efa_dl {

inline void* get_handle() {
  static void* h = [] {
    char const* custom = std::getenv("UCCL_EFA_SO");
    void* handle = custom ? dlopen(custom, RTLD_NOW) : nullptr;
    if (!handle) handle = dlopen("libefa.so", RTLD_NOW);
    if (!handle) handle = dlopen("libefa.so.1", RTLD_NOW);
    if (!handle) {
      std::fprintf(stderr,
                   "UCCL P2P: failed to load libefa.so: %s\n", dlerror());
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

#define UCCL_EFA_WRAP(func_name)                                            \
  template <typename... Args>                                               \
  inline auto func_name(Args&&... args)                                     \
      -> decltype(::func_name(std::forward<Args>(args)...)) {              \
    using FnType = decltype(&::func_name);                                  \
    static FnType fn = reinterpret_cast<FnType>(resolve(#func_name));       \
    return fn(std::forward<Args>(args)...);                                 \
  }

UCCL_EFA_WRAP(efadv_create_qp_ex)

#undef UCCL_EFA_WRAP

}  // namespace efa_dl
}  // namespace uccl
