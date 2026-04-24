// Linker-level dlsym wrapper for libefa functions.
// Provides efadv_* symbols via dlopen/dlsym instead of linking -lefa.

#include "include/efadv_wrapper.h"
#include <infiniband/efadv.h>

static void* efa_resolve(char const* name) { return uccl::efa_dl::resolve(name); }

extern "C" {

struct ibv_qp* efadv_create_qp_ex(
    struct ibv_context* context,
    struct ibv_qp_init_attr_ex* qp_init_attr_ex,
    struct efadv_qp_init_attr* efa_qp_init_attr,
    uint32_t inlen) {
  using FnType = struct ibv_qp*(*)(struct ibv_context*,
                                    struct ibv_qp_init_attr_ex*,
                                    struct efadv_qp_init_attr*, uint32_t);
  static FnType fn = reinterpret_cast<FnType>(efa_resolve("efadv_create_qp_ex"));
  return fn(context, qp_init_attr_ex, efa_qp_init_attr, inlen);
}

}  // extern "C"
