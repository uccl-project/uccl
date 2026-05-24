// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_IBVERBS_WRAPPER_HPP_
#define MSCCLPP_IBVERBS_WRAPPER_HPP_

#include <infiniband/verbs.h>
#include <string>

namespace mscclpp {

struct IBVerbs {
 private:
  static void* dlsym(std::string const& symbol, bool allowReturnNull = false);

 public:
#define REGISTER_IBV_FUNC_WITH_NAME(name__, func__)                     \
  template <typename... Args>                                           \
  static inline auto(name__)(Args && ... args) {                        \
    static_assert(sizeof(&::func__) > 0,                                \
                  #func__ " is expected be a function, not a macro");   \
    static decltype(&::func__) impl = nullptr;                          \
    if (!impl)                                                          \
      impl = reinterpret_cast<decltype(impl)>(IBVerbs::dlsym(#func__)); \
    return impl(std::forward<Args>(args)...);                           \
  }

#define REGISTER_IBV_FUNC(func__) REGISTER_IBV_FUNC_WITH_NAME(func__, func__)

  ///
  /// Usual cases where we can link the function directly
  ///

  REGISTER_IBV_FUNC(ibv_free_device_list)
  REGISTER_IBV_FUNC(ibv_alloc_pd)
  REGISTER_IBV_FUNC(ibv_dealloc_pd)
  REGISTER_IBV_FUNC(ibv_open_device)
  REGISTER_IBV_FUNC(ibv_close_device)
  REGISTER_IBV_FUNC(ibv_query_device)
  REGISTER_IBV_FUNC(ibv_create_cq)
  REGISTER_IBV_FUNC(ibv_destroy_cq)
  REGISTER_IBV_FUNC(ibv_create_qp)
  REGISTER_IBV_FUNC(ibv_destroy_qp)
  REGISTER_IBV_FUNC(ibv_modify_qp)
  REGISTER_IBV_FUNC(ibv_dereg_mr)
  REGISTER_IBV_FUNC(ibv_query_gid)
  REGISTER_IBV_FUNC(ibv_wc_status_str)

  static bool isDmabufSupported();
  static struct ibv_mr* ibv_reg_dmabuf_mr(struct ibv_pd*, uint64_t, size_t,
                                          uint64_t, int, int);

  ///
  /// Below is for cases where the API (may be / is) a macro. Refer to
  /// `infiniband/verbs.h`.
  ///

#if !defined(ibv_get_device_list)
  REGISTER_IBV_FUNC(ibv_get_device_list)
#else  // defined(ibv_get_device_list)
#undef ibv_get_device_list
  REGISTER_IBV_FUNC(ibv_static_providers)
  static inline struct ibv_device** ibv_get_device_list(int* num_devices) {
    using FuncType = struct ibv_device** (*)(int*);
    static FuncType impl = nullptr;
    if (!impl)
      impl = reinterpret_cast<FuncType>(IBVerbs::dlsym("ibv_get_device_list"));
    IBVerbs::ibv_static_providers(
        NULL, _RDMA_STATIC_PREFIX(RDMA_STATIC_PROVIDERS), NULL);
    return impl(num_devices);
  }
#endif  // defined(ibv_get_device_list)

#undef ibv_query_port
  static inline int ibv_query_port(struct ibv_context* context,
                                   uint8_t port_num,
                                   struct ibv_port_attr* port_attr) {
    static decltype(&::ibv_query_port) impl = nullptr;
    if (!impl)
      impl = reinterpret_cast<decltype(impl)>(IBVerbs::dlsym("ibv_query_port"));
    struct verbs_context* vctx = verbs_get_ctx_op(context, query_port);
    if (!vctx) {
      int rc;
      ::memset(port_attr, 0, sizeof(*port_attr));
      rc = impl(context, port_num, (struct _compat_ibv_port_attr*)port_attr);
      return rc;
    }
    return vctx->query_port(context, port_num, port_attr, sizeof(*port_attr));
  }

#undef ibv_reg_mr
  static inline struct ibv_mr* ibv_reg_mr(struct ibv_pd* pd, void* addr,
                                          size_t length, int access) {
    static decltype(&::ibv_reg_mr) impl = nullptr;
    static decltype(&::ibv_reg_mr_iova2) impl_iova2 = nullptr;
    int is_access_const =
        __builtin_constant_p(((int)(access)&IBV_ACCESS_OPTIONAL_RANGE) == 0);
    if (is_access_const && (access & IBV_ACCESS_OPTIONAL_RANGE) == 0) {
      if (!impl)
        impl = reinterpret_cast<decltype(impl)>(IBVerbs::dlsym("ibv_reg_mr"));
      return impl(pd, addr, length, (int)access);
    } else {
      if (!impl_iova2)
        impl_iova2 = reinterpret_cast<decltype(impl_iova2)>(
            IBVerbs::dlsym("ibv_reg_mr_iova2"));
      return impl_iova2(pd, addr, length, (uintptr_t)addr, access);
    }
  }

  ///
  /// Below is for cases where the API (may be / is) a static function. Refer to
  /// `infiniband/verbs.h`.
  ///

  static inline int ibv_post_send(struct ibv_qp* qp, struct ibv_send_wr* wr,
                                  struct ibv_send_wr** bad_wr) {
    return qp->context->ops.post_send(qp, wr, bad_wr);
  }

  static inline int ibv_post_recv(struct ibv_qp* qp, struct ibv_recv_wr* wr,
                                  struct ibv_recv_wr** bad_wr) {
    return qp->context->ops.post_recv(qp, wr, bad_wr);
  }

  static inline int ibv_poll_cq(struct ibv_cq* cq, int num_entries,
                                struct ibv_wc* wc) {
    return cq->context->ops.poll_cq(cq, num_entries, wc);
  }
};

}  // namespace mscclpp

#endif  // MSCCLPP_IBVERBS_WRAPPER_HPP_
