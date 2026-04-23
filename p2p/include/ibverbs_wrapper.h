#pragma once
// Runtime dlsym wrapper for libibverbs.
// Keeps #include <infiniband/verbs.h> for types and inline macros.
// Wraps actual library functions via dlopen/dlsym so that libuccl_p2p.so
// does not need to link against -libverbs at compile time.
//
// Usage: replace `ibv_xxx(...)` calls with `uccl::ibv_dl::ibv_xxx(...)`,
//        or add `using namespace uccl::ibv_dl;` at the top of the file.
//
// Custom library path: set UCCL_IBV_SO environment variable.

#include <infiniband/verbs.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

namespace uccl {
namespace ibv_dl {

// Lazy-load libibverbs.so handle.
inline void* get_handle() {
  static void* h = [] {
    char const* custom = std::getenv("UCCL_IBV_SO");
    void* handle = custom ? dlopen(custom, RTLD_NOW) : nullptr;
    if (!handle) handle = dlopen("libibverbs.so", RTLD_NOW);
    if (!handle) handle = dlopen("libibverbs.so.1", RTLD_NOW);
    if (!handle) {
      std::fprintf(stderr,
                   "UCCL P2P: failed to load libibverbs.so: %s\n", dlerror());
    }
    return handle;
  }();
  return h;
}

inline bool is_available() { return get_handle() != nullptr; }

// Helper to resolve a symbol, abort on failure.
inline void* resolve(char const* name) {
  void* sym = dlsym(get_handle(), name);
  if (!sym) {
    std::fprintf(stderr, "UCCL P2P: dlsym(%s) failed: %s\n", name, dlerror());
    std::abort();
  }
  return sym;
}

// Macro to declare a dlsym wrapper for an ibv_* library function.
// The wrapper has the same signature as the original and resolves the
// symbol on first call (static function pointer, thread-safe via C++11).
#define UCCL_IBV_WRAP(func_name)                                            \
  template <typename... Args>                                               \
  inline auto func_name(Args&&... args)                                     \
      -> decltype(::func_name(std::forward<Args>(args)...)) {              \
    using FnType = decltype(&::func_name);                                  \
    static FnType fn = reinterpret_cast<FnType>(resolve(#func_name));       \
    return fn(std::forward<Args>(args)...);                                 \
  }

// --- Device management ---
UCCL_IBV_WRAP(ibv_get_device_list)
UCCL_IBV_WRAP(ibv_free_device_list)
UCCL_IBV_WRAP(ibv_get_device_name)
UCCL_IBV_WRAP(ibv_open_device)
UCCL_IBV_WRAP(ibv_close_device)
UCCL_IBV_WRAP(ibv_query_device)

// --- Port and GID ---
UCCL_IBV_WRAP(ibv_query_port)
UCCL_IBV_WRAP(ibv_query_gid)

// --- Protection domain ---
UCCL_IBV_WRAP(ibv_alloc_pd)
UCCL_IBV_WRAP(ibv_dealloc_pd)

// --- Memory registration ---
UCCL_IBV_WRAP(ibv_reg_mr)
UCCL_IBV_WRAP(ibv_dereg_mr)

// --- Completion queue ---
UCCL_IBV_WRAP(ibv_create_cq_ex)
UCCL_IBV_WRAP(ibv_destroy_cq)

// --- Queue pair ---
UCCL_IBV_WRAP(ibv_create_qp_ex)
UCCL_IBV_WRAP(ibv_destroy_qp)
UCCL_IBV_WRAP(ibv_modify_qp)

// --- Address handle ---
UCCL_IBV_WRAP(ibv_create_ah)

// --- Data path ---
UCCL_IBV_WRAP(ibv_post_send)
UCCL_IBV_WRAP(ibv_post_recv)
UCCL_IBV_WRAP(ibv_poll_cq)

// --- Error strings ---
UCCL_IBV_WRAP(ibv_wc_status_str)

#undef UCCL_IBV_WRAP

// NOTE: The following are inline macros/functions in verbs.h that work
// through struct function pointers. They do NOT call into libibverbs.so
// and do NOT need dlsym wrapping:
//   ibv_wr_start, ibv_wr_complete, ibv_wr_rdma_write, ibv_wr_rdma_write_imm,
//   ibv_wr_rdma_read, ibv_wr_set_inline_data, ibv_wr_set_sge_list,
//   ibv_wr_set_ud_addr, ibv_qp_to_qp_ex, ibv_cq_ex_to_cq,
//   ibv_start_poll, ibv_next_poll, ibv_end_poll,
//   ibv_wc_read_opcode, ibv_wc_read_byte_len, ibv_wc_read_imm_data

}  // namespace ibv_dl
}  // namespace uccl
