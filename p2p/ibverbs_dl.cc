// Linker-level dlsym wrappers for libibverbs functions.
// This file provides the ibv_* symbols that the RDMA code calls, resolving
// them at runtime via dlopen/dlsym instead of linking against -libverbs.
//
// These are C functions (extern "C") to match the libibverbs ABI.

#include "include/ibverbs_wrapper.h"
#include <infiniband/verbs.h>

// Undefine macros from verbs.h so we can define our own wrapper functions.
#undef ibv_get_device_list
#undef ibv_query_port
#undef ibv_reg_mr
#undef ibv_reg_mr_iova

// Eagerly load libibverbs.so at library load time with RTLD_GLOBAL so that
// compat-layer functions in verbs.h (ibv_query_port, ibv_reg_mr, etc.) can
// resolve their symbols from the dlopen'd library.
__attribute__((constructor))
static void ibv_dl_init() { uccl::ibv_dl::get_handle(); }

static void* ibv_handle() { return uccl::ibv_dl::get_handle(); }
static void* ibv_resolve(char const* name) { return uccl::ibv_dl::resolve(name); }

extern "C" {

// We need to provide implementations for every ibv_* symbol that the RDMA code
// references. Each function dlsym's the real implementation on first call.

#define IBV_DL_FUNC(ret, name, args, call_args) \
  ret name args { \
    using FnType = ret(*)args; \
    static FnType fn = reinterpret_cast<FnType>(ibv_resolve(#name)); \
    return fn call_args; \
  }

// Device management
// verbs.h defines: #define ibv_get_device_list(n) __ibv_get_device_list(n)
// The macro redirects to __ibv_get_device_list, so we provide that symbol.
// The actual library exports "ibv_get_device_list".
struct ibv_device** __ibv_get_device_list(int* num_devices) {
  using FnType = struct ibv_device**(*)(int*);
  static FnType fn = reinterpret_cast<FnType>(ibv_resolve("ibv_get_device_list"));
  return fn(num_devices);
}

IBV_DL_FUNC(void, ibv_free_device_list, (struct ibv_device** list), (list))
IBV_DL_FUNC(const char*, ibv_get_device_name, (struct ibv_device* device), (device))
IBV_DL_FUNC(struct ibv_context*, ibv_open_device, (struct ibv_device* device), (device))
IBV_DL_FUNC(int, ibv_close_device, (struct ibv_context* context), (context))
IBV_DL_FUNC(int, ibv_query_device, (struct ibv_context* context, struct ibv_device_attr* device_attr), (context, device_attr))

// Port and GID
IBV_DL_FUNC(int, ibv_query_gid, (struct ibv_context* context, uint8_t port_num, int index, union ibv_gid* gid), (context, port_num, index, gid))

// Protection domain
IBV_DL_FUNC(struct ibv_pd*, ibv_alloc_pd, (struct ibv_context* context), (context))
IBV_DL_FUNC(int, ibv_dealloc_pd, (struct ibv_pd* pd), (pd))

// Memory registration
IBV_DL_FUNC(int, ibv_dereg_mr, (struct ibv_mr* mr), (mr))

// Completion queue
// ibv_create_cq is declared in verbs.h, resolved via RTLD_GLOBAL.
IBV_DL_FUNC(int, ibv_destroy_cq, (struct ibv_cq* cq), (cq))

// Queue pair
IBV_DL_FUNC(int, ibv_destroy_qp, (struct ibv_qp* qp), (qp))
IBV_DL_FUNC(int, ibv_modify_qp, (struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask), (qp, attr, attr_mask))
// ibv_create_qp, ibv_qp_to_qp_ex are declared in verbs.h and resolved
// via RTLD_GLOBAL dlopen of libibverbs.so in the constructor above.

// NOTE: ibv_query_port, ibv_reg_mr, ibv_reg_dmabuf_mr are compat-layer
// functions declared in verbs.h with inline wrappers. They call through
// to the real library via verbs_get_ctx()->ops, so they don't need
// our dlsym wrappers. The symbols are resolved by the compat functions
// in verbs.h that call ibv_cmd_* internally.

// Address handle
IBV_DL_FUNC(struct ibv_ah*, ibv_create_ah, (struct ibv_pd* pd, struct ibv_ah_attr* attr), (pd, attr))

// Error strings
IBV_DL_FUNC(const char*, ibv_wc_status_str, (enum ibv_wc_status status), (status))

#undef IBV_DL_FUNC

}  // extern "C"
