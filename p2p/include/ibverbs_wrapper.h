#pragma once
// Runtime dlsym wrapper for libibverbs.
// Keeps #include <infiniband/verbs.h> for types and inline macros.
// Only wraps true library functions - NOT the verbs.h macros that dispatch
// through struct function pointers (ibv_query_port, ibv_reg_mr, ibv_post_send,
// ibv_post_recv, ibv_poll_cq, ibv_wr_*, ibv_wc_read_*, etc.).
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

inline void* resolve(char const* name) {
  void* sym = dlsym(get_handle(), name);
  if (!sym) {
    std::fprintf(stderr, "UCCL P2P: dlsym(%s) failed: %s\n", name, dlerror());
    std::abort();
  }
  return sym;
}

// Wrapper macro - only safe for names that are NOT #define macros in verbs.h.
#define UCCL_IBV_WRAP(func_name)                                            \
  template <typename... Args>                                               \
  inline auto func_name(Args&&... args)                                     \
      -> decltype(::func_name(std::forward<Args>(args)...)) {              \
    using FnType = decltype(&::func_name);                                  \
    static FnType fn = reinterpret_cast<FnType>(resolve(#func_name));       \
    return fn(std::forward<Args>(args)...);                                 \
  }

// --- True library functions (safe to wrap) ---
UCCL_IBV_WRAP(ibv_get_device_list)
UCCL_IBV_WRAP(ibv_free_device_list)
UCCL_IBV_WRAP(ibv_get_device_name)
UCCL_IBV_WRAP(ibv_open_device)
UCCL_IBV_WRAP(ibv_close_device)
UCCL_IBV_WRAP(ibv_query_device)
UCCL_IBV_WRAP(ibv_query_gid)
UCCL_IBV_WRAP(ibv_alloc_pd)
UCCL_IBV_WRAP(ibv_dealloc_pd)
UCCL_IBV_WRAP(ibv_dereg_mr)
UCCL_IBV_WRAP(ibv_create_cq_ex)
UCCL_IBV_WRAP(ibv_destroy_cq)
UCCL_IBV_WRAP(ibv_create_qp_ex)
UCCL_IBV_WRAP(ibv_destroy_qp)
UCCL_IBV_WRAP(ibv_modify_qp)
UCCL_IBV_WRAP(ibv_create_ah)
UCCL_IBV_WRAP(ibv_wc_status_str)

#undef UCCL_IBV_WRAP

// The following are macros in verbs.h that dispatch through struct function
// pointers or the compat layer. They do NOT call into libibverbs.so directly
// and must NOT be wrapped (the preprocessor would conflict):
//   ibv_query_port, ibv_reg_mr, ibv_reg_dmabuf_mr,
//   ibv_post_send, ibv_post_recv, ibv_poll_cq,
//   ibv_wr_start, ibv_wr_complete, ibv_wr_rdma_write, ibv_wr_rdma_write_imm,
//   ibv_wr_rdma_read, ibv_wr_set_inline_data, ibv_wr_set_sge_list,
//   ibv_wr_set_ud_addr, ibv_qp_to_qp_ex, ibv_cq_ex_to_cq,
//   ibv_start_poll, ibv_next_poll, ibv_end_poll,
//   ibv_wc_read_opcode, ibv_wc_read_byte_len, ibv_wc_read_imm_data
//
// These macros ultimately call through context->ops or qp->context->ops
// which are populated when ibv_open_device() creates the context. Since
// ibv_open_device IS wrapped above, the function pointers get set up
// correctly via dlsym.

}  // namespace ibv_dl
}  // namespace uccl
