#pragma once
#include "transport/endpoint_ops.h"
// Convert between the global UcclRequest and NcclRequest.
// Preserve the completion handle across submission and subsequent polls.
static inline NcclRequest to_nccl_req(UcclRequest const& r) {
  NcclRequest out{};
  out.type = static_cast<NcclReqType>(static_cast<int>(r.type));
  out.peer_id = r.peer_id;
  out.context = r.context;
  out.engine_idx = r.engine_idx;
  return out;
}

static inline void from_nccl_req(NcclRequest const& in, UcclRequest* out) {
  out->type = static_cast<ReqType>(static_cast<int>(in.type));
  out->peer_id = in.peer_id;
  out->engine_idx = in.engine_idx;
  out->context = in.context;
}

// Convert NCCL ConnID to the common ConnID used by Endpoint.
static inline ConnID to_conn_id(NcclConnID const& in) {
  ConnID out{};
  out.context = in.context;
  out.sock_fd = in.sock_fd;
  out.dev = in.dev;
  out.peer_id = in.peer_id;
  return out;
}

template <>
struct EndpointOps<NCCLEndpoint> {
  static ConnID uccl_connect(std::shared_ptr<NCCLEndpoint> const& s,
                             int remote_gpuidx, std::string const& remote_ip,
                             uint16_t remote_port) {
    int local_gpu = s->gpu_index();
    auto c =
        s->uccl_connect(0, local_gpu, 0, remote_gpuidx, remote_ip, remote_port);
    return to_conn_id(c);
  }

  static uint16_t get_p2p_listen_port(std::shared_ptr<NCCLEndpoint> const& s) {
    return s->get_p2p_listen_port(0);
  }

  static int get_p2p_listen_fd(std::shared_ptr<NCCLEndpoint> const& s) {
    return s->get_p2p_listen_fd(0);
  }

  static ConnID uccl_accept(std::shared_ptr<NCCLEndpoint> const& s,
                            std::string& remote_ip, int* remote_gpuidx) {
    int remote_dev = 0;
    int local_gpu = s->gpu_index();
    auto c =
        s->uccl_accept(0, -1, local_gpu, remote_ip, &remote_dev, remote_gpuidx);
    return to_conn_id(c);
  }

  static bool uccl_regmr(std::shared_ptr<NCCLEndpoint> const& s, void* data,
                         size_t len, struct P2PMhandle* mhandle) {
    (void)data;
    (void)len;
    (void)mhandle;
    return true;
  }

  static bool uccl_prepare_compress_ctx(std::shared_ptr<NCCLEndpoint> const& s,
                                        void* data, size_t len,
                                        CompressCtx const& compress_ctx) {
    (void)s;
    (void)data;
    (void)len;
    (void)compress_ctx;
    return false;
  }

  static bool uccl_poll_ureq_once(std::shared_ptr<NCCLEndpoint> const& s,
                                  UcclRequest* ureq) {
    NcclRequest nreq = to_nccl_req(*ureq);
    bool ret = s->uccl_poll_ureq_once(&nreq);
    from_nccl_req(nreq, ureq);
    return ret;
  }

  static void uccl_drive_send(std::shared_ptr<NCCLEndpoint> const& s) {}

  static void uccl_flush_send(std::shared_ptr<NCCLEndpoint> const& s) {}

  static SendConnection* uccl_resolve_send_group(
      std::shared_ptr<NCCLEndpoint> const& s, uint64_t peer_id) {
    return nullptr;
  }

  static void uccl_drive_recv(std::shared_ptr<NCCLEndpoint> const& s) {}

  static bool uccl_check_ureq_once(std::shared_ptr<NCCLEndpoint> const& s,
                                   UcclRequest* ureq) {
    return uccl_poll_ureq_once(s, ureq);
  }

  static int uccl_read_async(std::shared_ptr<NCCLEndpoint> const& s, Conn* conn,
                             P2PMhandle* local_mh, void* dst, size_t size,
                             FifoItem const& slot_item, UcclRequest* ureq) {
    (void)local_mh;
    ureq->type = ReqType::ReqRead;
    ureq->peer_id = conn->uccl_conn_id_.peer_id;
    NcclFifoItem nccl_item{};
    nccl_item.addr = slot_item.addr;
    nccl_item.size = static_cast<uint32_t>(size);
    std::memset(nccl_item.padding, 0, sizeof(nccl_item.padding));
    NcclRequest nreq = to_nccl_req(*ureq);
    int ret = s->uccl_read_async(
        reinterpret_cast<NcclFlow*>(conn->uccl_conn_id_.context), nullptr, dst,
        size, nccl_item, &nreq);
    from_nccl_req(nreq, ureq);
    return ret;
  }

  static int uccl_write_async(std::shared_ptr<NCCLEndpoint> const& s,
                              Conn* conn, P2PMhandle* local_mh, void* src,
                              size_t size, FifoItem const& slot_item,
                              UcclRequest* ureq) {
    (void)local_mh;
    ureq->type = ReqType::ReqWrite;
    ureq->peer_id = conn->uccl_conn_id_.peer_id;
    NcclFifoItem nccl_item{};
    nccl_item.addr = slot_item.addr;
    nccl_item.size = static_cast<uint32_t>(size);
    std::memset(nccl_item.padding, 0, sizeof(nccl_item.padding));
    NcclRequest nreq = to_nccl_req(*ureq);
    int ret = s->uccl_write_async(
        reinterpret_cast<NcclFlow*>(conn->uccl_conn_id_.context), nullptr, src,
        size, nccl_item, &nreq);
    from_nccl_req(nreq, ureq);
    return ret;
  }

  static int prepare_fifo_metadata(std::shared_ptr<NCCLEndpoint> const& s,
                                   P2PMhandle* mhandle, void const* data,
                                   size_t size, char* out_buf) {
    (void)mhandle;
    return s->prepare_fifo_metadata(nullptr, nullptr, data, size, out_buf);
  }

  static void uccl_deregmr(std::shared_ptr<NCCLEndpoint> const& s,
                           P2PMhandle* mhandle) {
    (void)mhandle;
  }

  static int uccl_get_numa_node(std::shared_ptr<NCCLEndpoint> const& s,
                                int gpu_idx) {
    return get_numa_node_from_iface();
  }

  static int uccl_send_notification(std::shared_ptr<NCCLEndpoint> const& s,
                                    uint64_t peer_id,
                                    NotifyMsg const& notification) {
    if (!s) return -1;

    return s->send_notification(peer_id, notification);
  }

  static bool initialize_rdma_ctx_for_gpu(
      std::shared_ptr<NCCLEndpoint> const& s, int dev) {
    return s->initialize_engine_by_dev(dev, true);
  }

  static void create_unified_p2p_socket(
      std::shared_ptr<NCCLEndpoint> const& s) {
    (void)s;
  }
};
