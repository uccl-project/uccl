// Linker-level dlsym wrappers for libibverbs functions.
// This file provides the ibv_* symbols that the RDMA code calls, resolving
// them at runtime via dlopen/dlsym instead of linking against -libverbs.
//
// These are C functions (extern "C") to match the libibverbs ABI.

#include "include/ibverbs_wrapper.h"
#include <infiniband/verbs.h>

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
struct ibv_device** __ibv_get_device_list(int* num_devices) {
  using FnType = struct ibv_device**(*)(int*);
  static FnType fn = reinterpret_cast<FnType>(ibv_resolve("__ibv_get_device_list"));
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
IBV_DL_FUNC(int, ibv_destroy_cq, (struct ibv_cq* cq), (cq))

// Queue pair
IBV_DL_FUNC(int, ibv_destroy_qp, (struct ibv_qp* qp), (qp))
IBV_DL_FUNC(int, ibv_modify_qp, (struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask), (qp, attr, attr_mask))

// Address handle
IBV_DL_FUNC(struct ibv_ah*, ibv_create_ah, (struct ibv_pd* pd, struct ibv_ah_attr* attr), (pd, attr))

// Error strings
IBV_DL_FUNC(const char*, ibv_wc_status_str, (enum ibv_wc_status status), (status))

#undef IBV_DL_FUNC

}  // extern "C"
