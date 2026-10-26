#pragma once
#include "engine.h"
#include "transport/rdma_ops.h"
#include "transport/rdma_request.h"
#if UCCL_P2P_ENABLE_NCCL
#include "transport/nccl_ops.h"
#endif
#if UCCL_P2P_ENABLE_CXI
#include "transport/cxi_ops.h"
#endif

// One variant dispatch per call. Transport adapters own signature and request
// conversions; identical endpoint APIs are called directly. The RDMA group
// fast paths below still bypass this dispatch.
inline ConnID uccl_connect(GenericEndpoint const& ep, int remote_gpuidx,
                           std::string remote_ip, uint16_t remote_port) {
  return std::visit(
      [&](auto const& s) -> ConnID {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::uccl_connect(s, remote_gpuidx, remote_ip,
                                            remote_port);
      },
      ep);
}

inline uint16_t get_p2p_listen_port(GenericEndpoint const& ep) {
  return std::visit(
      [&](auto const& s) -> uint16_t {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::get_p2p_listen_port(s);
      },
      ep);
}

inline int get_p2p_listen_fd(GenericEndpoint const& ep) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::get_p2p_listen_fd(s);
      },
      ep);
}

inline ConnID uccl_accept(GenericEndpoint const& ep, std::string& remote_ip,
                          int* remote_gpuidx) {
  return std::visit(
      [&](auto const& s) -> ConnID {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::uccl_accept(s, remote_ip, remote_gpuidx);
      },
      ep);
}

inline void uccl_stop_accept(GenericEndpoint const& ep) {
  std::visit([](auto const& s) { s->stop_accept(); }, ep);
}

inline bool uccl_accept_stopped(GenericEndpoint const& ep) {
  return std::visit([](auto const& s) { return s->accept_stopped(); }, ep);
}

inline bool uccl_regmr(GenericEndpoint const& ep, void* data, size_t len,
                       struct P2PMhandle* mhandle) {
  return std::visit(
      [&](auto const& s) -> bool {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::uccl_regmr(s, data, len, mhandle);
      },
      ep);
}

inline bool uccl_prepare_compress_ctx(GenericEndpoint const& ep, void* data,
                                      size_t len, CompressCtx compress_ctx) {
  return std::visit(
      [&](auto const& s) -> bool {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::uccl_prepare_compress_ctx(s, data, len,
                                                         compress_ctx);
      },
      ep);
}

inline bool uccl_poll_ureq_once(GenericEndpoint const& ep, UcclRequest* ureq) {
  return std::visit(
      [&](auto const& s) -> bool {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::uccl_poll_ureq_once(s, ureq);
      },
      ep);
}

inline void uccl_drive_send(GenericEndpoint const& ep) {
  return std::visit(
      [&](auto const& s) {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::uccl_drive_send(s);
      },
      ep);
}

inline void uccl_flush_send(GenericEndpoint const& ep) {
  return std::visit(
      [&](auto const& s) {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::uccl_flush_send(s);
      },
      ep);
}

inline SendConnection* uccl_resolve_send_group(GenericEndpoint const& ep,
                                               uint64_t peer_id) {
  return std::visit(
      [&](auto const& s) -> SendConnection* {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::uccl_resolve_send_group(s, peer_id);
      },
      ep);
}

inline bool uccl_check_wr_fast(SendConnection* send_group, int64_t wr_id) {
  return send_group->check_completion(wr_id);
}

inline void uccl_drive_recv(GenericEndpoint const& ep) {
  return std::visit(
      [&](auto const& s) {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::uccl_drive_recv(s);
      },
      ep);
}

inline bool uccl_check_ureq_once(GenericEndpoint const& ep, UcclRequest* ureq) {
  return std::visit(
      [&](auto const& s) -> bool {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::uccl_check_ureq_once(s, ureq);
      },
      ep);
}

inline int uccl_read_async(GenericEndpoint const& ep, Conn* conn,
                           P2PMhandle* local_mh, void* dst, size_t size,
                           FifoItem const& slot_item, UcclRequest* ureq) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::uccl_read_async(s, conn, local_mh, dst, size,
                                               slot_item, ureq);
      },
      ep);
}

inline int uccl_write_async(GenericEndpoint const& ep, Conn* conn,
                            P2PMhandle* local_mh, void* src, size_t size,
                            FifoItem const& slot_item, UcclRequest* ureq) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::uccl_write_async(s, conn, local_mh, src, size,
                                                slot_item, ureq);
      },
      ep);
}

// RDMA hot paths use a pre-resolved group and bypass variant dispatch.
inline int uccl_write_async_on_group(SendConnection* send_group, Conn* conn,
                                     P2PMhandle* local_mh, void* src,
                                     size_t size, FifoItem const& slot_item,
                                     UcclRequest* ureq) {
  ureq->type = ReqType::ReqWrite;
  return set_request_on_group(send_group, conn, local_mh, src, size, slot_item,
                              ureq);
}

inline int uccl_read_async_on_group(SendConnection* send_group, Conn* conn,
                                    P2PMhandle* local_mh, void* dst,
                                    size_t size, FifoItem const& slot_item,
                                    UcclRequest* ureq) {
  ureq->type = ReqType::ReqRead;
  return set_request_on_group(send_group, conn, local_mh, dst, size, slot_item,
                              ureq);
}

inline int prepare_fifo_metadata(GenericEndpoint const& ep, P2PMhandle* mhandle,
                                 void const* data, size_t size, char* out_buf) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::prepare_fifo_metadata(s, mhandle, data, size,
                                                     out_buf);
      },
      ep);
}

inline void uccl_deregmr(GenericEndpoint const& ep, P2PMhandle* mhandle) {
  return std::visit(
      [&](auto const& s) {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::uccl_deregmr(s, mhandle);
      },
      ep);
}

inline int uccl_get_numa_node(GenericEndpoint const& ep, int gpu_idx) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::uccl_get_numa_node(s, gpu_idx);
      },
      ep);
}

inline int uccl_send_notification(GenericEndpoint const& ep, uint64_t peer_id,
                                  NotifyMsg const& notification) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::uccl_send_notification(s, peer_id, notification);
      },
      ep);
}

inline bool initialize_rdma_ctx_for_gpu(GenericEndpoint const& ep, int dev) {
  return std::visit(
      [&](auto const& s) -> bool {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::initialize_rdma_ctx_for_gpu(s, dev);
      },
      ep);
}

inline void create_unified_p2p_socket(GenericEndpoint const& ep) {
  return std::visit(
      [&](auto const& s) {
        using T = std::decay_t<decltype(*s)>;
        return EndpointOps<T>::create_unified_p2p_socket(s);
      },
      ep);
}

inline std::shared_ptr<EpollClient> get_oob_client(GenericEndpoint const& ep) {
  return std::visit([](auto const& s) { return s->get_oob_client(); }, ep);
}

inline std::string get_oob_conn_key(GenericEndpoint const& ep,
                                    uint64_t peer_id) {
  return std::visit([&](auto const& s) { return s->get_oob_conn_key(peer_id); },
                    ep);
}
