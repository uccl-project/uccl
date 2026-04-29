#pragma once
#include "engine.h"

// ── Helpers ─────────────────────────────────────────────────────────────────

template <class T>
struct always_false : std::false_type {};

// Convert between the global ucclRequest and uccl::ucclRequest.
// They have different layouts (uccl:: version has an extra context pointer).
static inline uccl::ucclRequest to_nccl_req(ucclRequest const& r) {
  uccl::ucclRequest out{};
  out.type = static_cast<uccl::ReqType>(r.type);
  out.n = r.n;
  out.context = nullptr;
  out.engine_idx = r.engine_idx;
  return out;
}
static inline void from_nccl_req(uccl::ucclRequest const& in,
                                 ucclRequest* out) {
  out->type = static_cast<ReqType>(in.type);
  out->n = in.n;
  out->engine_idx = in.engine_idx;
}

// Convert NCCL ConnID to the common ConnID used by Endpoint.
static inline ConnID to_conn_id(uccl::ConnID const& in) {
  ConnID out{};
  out.context = in.context;
  out.sock_fd = in.sock_fd;
  out.dev = in.dev;
  out.flow_id = in.flow_id;
  return out;
}

// RDMA-only: build a one-sided request from FifoItem metadata.
static inline int set_request(std::shared_ptr<NICEndpoint> const& obj,
                              Conn* conn, P2PMhandle* local_mh, void* src,
                              size_t size, FifoItem const& slot_item,
                              ucclRequest* ureq) {
  auto remote_mem = std::make_shared<RemoteMemInfo>();
  remote_mem->addr = slot_item.addr;
  remote_mem->length = slot_item.size;
  remote_mem->type = MemoryType::GPU;
  remote_mem->rkey_array.copyFrom(slot_item.padding);
  auto local_mem = std::make_shared<RegMemBlock>(src, size, MemoryType::GPU);
  local_mem->mr_array = local_mh->mr_array;

  auto req = std::make_shared<RDMASendRequest>(local_mem, remote_mem);
  req->compress_ctx = local_mh->compress_ctx;
  req->to_rank_id = conn->uccl_conn_id_.flow_id;
  req->send_type =
      (ureq->type == ReqType::ReqRead) ? SendType::Read : SendType::Write;

  ureq->engine_idx = obj->writeOrRead(req);
  ureq->n = conn->uccl_conn_id_.flow_id;
  return ureq->engine_idx;
}

// ── Dispatch functions ──────────────────────────────────────────────────────
// Each function uses std::visit to dispatch to the correct endpoint type
// at runtime. The if constexpr branches are resolved at compile time so
// there is no overhead beyond the variant type check.

inline ConnID uccl_connect(RDMAEndPoint const& ep, int remote_gpuidx,
                           std::string remote_ip, uint16_t remote_port) {
  return std::visit(
      [&](auto const& s) -> ConnID {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, tcp::TCPEndpoint>) {
          int local_gpu = s->gpuIndex();
          auto c = s->uccl_connect(0, local_gpu, 0, remote_gpuidx, remote_ip,
                                   remote_port);
          return to_conn_id(c);
        } else {
          return s->uccl_connect(remote_gpuidx, remote_ip, remote_port);
        }
      },
      ep);
}

inline uint16_t get_p2p_listen_port(RDMAEndPoint const& ep) {
  return std::visit(
      [](auto const& s) -> uint16_t {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, tcp::TCPEndpoint>)
          return s->get_p2p_listen_port(0);
        else
          return s->get_p2p_listen_port();
      },
      ep);
}

inline int get_p2p_listen_fd(RDMAEndPoint const& ep) {
  return std::visit(
      [](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, tcp::TCPEndpoint>)
          return s->get_p2p_listen_fd(0);
        else
          return s->get_p2p_listen_fd();
      },
      ep);
}

inline ConnID uccl_accept(RDMAEndPoint const& ep, std::string& remote_ip,
                          int* remote_gpuidx) {
  return std::visit(
      [&](auto const& s) -> ConnID {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, tcp::TCPEndpoint>) {
          int remote_dev = 0;
          int local_gpu = s->gpuIndex();
          auto c = s->uccl_accept(0, -1, local_gpu, remote_ip, &remote_dev,
                                  remote_gpuidx);
          return to_conn_id(c);
        } else {
          return s->uccl_accept(remote_ip, remote_gpuidx);
        }
      },
      ep);
}

inline void stop_accept(RDMAEndPoint const& ep) {
  std::visit([](auto const& s) { s->stop_accept(); }, ep);
}

inline bool uccl_regmr(RDMAEndPoint const& ep, void* data, size_t len,
                       struct P2PMhandle* mhandle) {
  return std::visit(
      [&](auto const& s) -> bool {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, tcp::TCPEndpoint>) {
          (void)data;
          (void)len;
          (void)mhandle;
          return true;
        } else {
          return s->uccl_regmr(data, len, mhandle->mr_array,
                               mhandle->cache_refs, mhandle->compress_ctx) >= 0;
        }
      },
      ep);
}

inline int uccl_send_async(RDMAEndPoint const& ep, Conn* conn,
                           P2PMhandle* mhandle, void const* data,
                           size_t const size, ucclRequest* ureq) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, tcp::TCPEndpoint>) {
          (void)mhandle;
          ureq->type = ReqType::ReqTx;
          ureq->n = conn->uccl_conn_id_.flow_id;
          uccl::ucclRequest nreq = to_nccl_req(*ureq);
          int ret = s->uccl_send_async(
              reinterpret_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context),
              nullptr, data, size, &nreq);
          from_nccl_req(nreq, ureq);
          return ret;
        } else {
          auto send_mem = std::make_shared<RegMemBlock>(const_cast<void*>(data),
                                                        size, MemoryType::GPU);
          send_mem->mr_array = mhandle->mr_array;
          auto remote_mem = std::make_shared<RemoteMemInfo>();
          auto send_req =
              std::make_shared<RDMASendRequest>(send_mem, remote_mem);
          send_req->compress_ctx = mhandle->compress_ctx;
          ureq->type = ReqType::ReqTx;
          send_req->to_rank_id = conn->uccl_conn_id_.flow_id;
          ureq->engine_idx = s->sendWithoutInnerQueue(send_req);
          while (ureq->engine_idx < 0) {
            ureq->engine_idx = s->sendWithoutInnerQueue(send_req);
          }
          ureq->n = conn->uccl_conn_id_.flow_id;
          return ureq->engine_idx;
        }
      },
      ep);
}

inline int uccl_recv_async(RDMAEndPoint const& ep, Conn* conn,
                           P2PMhandle* mhandles, void** data, int* size, int n,
                           ucclRequest* ureq) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, tcp::TCPEndpoint>) {
          (void)mhandles;
          ureq->type = ReqType::ReqRx;
          ureq->n = conn->uccl_conn_id_.flow_id;
          uccl::ucclRequest nreq = to_nccl_req(*ureq);
          int ret = s->uccl_recv_async(
              reinterpret_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context),
              nullptr, data, size, n, &nreq);
          from_nccl_req(nreq, ureq);
          return ret;
        } else {
          auto recv_mem =
              std::make_shared<RegMemBlock>(data[0], size[0], MemoryType::GPU);
          recv_mem->mr_array = mhandles->mr_array;
          auto recv_req = std::make_shared<RDMARecvRequest>(recv_mem);
          recv_req->compress_ctx = mhandles->compress_ctx;
          ureq->type = ReqType::ReqRx;
          ureq->engine_idx = s->recv(conn->uccl_conn_id_.flow_id, recv_req);
          ureq->n = conn->uccl_conn_id_.flow_id;
          return ureq->engine_idx;
        }
      },
      ep);
}

inline bool uccl_poll_ureq_once(RDMAEndPoint const& ep, ucclRequest* ureq) {
  return std::visit(
      [&](auto const& s) -> bool {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, tcp::TCPEndpoint>) {
          uccl::ucclRequest nreq = to_nccl_req(*ureq);
          bool ret = s->uccl_poll_ureq_once(&nreq);
          from_nccl_req(nreq, ureq);
          return ret;
        } else {
          if (ureq->type == ReqType::ReqTx || ureq->type == ReqType::ReqWrite ||
              ureq->type == ReqType::ReqRead) {
            s->sendRoutine();
            return s->checkSendComplete_once(ureq->n, ureq->engine_idx);
          } else if (ureq->type == ReqType::ReqRx) {
            s->recvRoutine();
            return s->checkRecvComplete_once(ureq->n, ureq->engine_idx);
          }
          UCCL_LOG(ERROR) << "Invalid request type: " << ureq->type;
          return false;
        }
      },
      ep);
}

inline int uccl_read_async(RDMAEndPoint const& ep, Conn* conn,
                           P2PMhandle* local_mh, void* dst, size_t size,
                           FifoItem const& slot_item, ucclRequest* ureq) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, tcp::TCPEndpoint>) {
          (void)local_mh;
          ureq->type = ReqType::ReqRead;
          ureq->n = conn->uccl_conn_id_.flow_id;
          uccl::FifoItem tcp_item{};
          tcp_item.addr = slot_item.addr;
          tcp_item.size = static_cast<uint32_t>(size);
          std::memset(tcp_item.padding, 0, sizeof(tcp_item.padding));
          uccl::ucclRequest nreq = to_nccl_req(*ureq);
          int ret = s->uccl_read_async(
              reinterpret_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context),
              nullptr, dst, size, tcp_item, &nreq);
          from_nccl_req(nreq, ureq);
          return ret;
        } else {
          ureq->type = ReqType::ReqRead;
          return set_request(s, conn, local_mh, dst, size, slot_item, ureq);
        }
      },
      ep);
}

inline int uccl_write_async(RDMAEndPoint const& ep, Conn* conn,
                            P2PMhandle* local_mh, void* src, size_t size,
                            FifoItem const& slot_item, ucclRequest* ureq) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, tcp::TCPEndpoint>) {
          (void)local_mh;
          ureq->type = ReqType::ReqWrite;
          ureq->n = conn->uccl_conn_id_.flow_id;
          uccl::FifoItem tcp_item{};
          tcp_item.addr = slot_item.addr;
          tcp_item.size = static_cast<uint32_t>(size);
          std::memset(tcp_item.padding, 0, sizeof(tcp_item.padding));
          uccl::ucclRequest nreq = to_nccl_req(*ureq);
          int ret = s->uccl_write_async(
              reinterpret_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context),
              nullptr, src, size, tcp_item, &nreq);
          from_nccl_req(nreq, ureq);
          return ret;
        } else {
          ureq->type = ReqType::ReqWrite;
          return set_request(s, conn, local_mh, src, size, slot_item, ureq);
        }
      },
      ep);
}

inline int prepare_fifo_metadata(RDMAEndPoint const& ep, Conn* conn,
                                 P2PMhandle* mhandle, void const* data,
                                 size_t size, char* out_buf) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, tcp::TCPEndpoint>) {
          (void)conn;
          (void)mhandle;
          return s->prepare_fifo_metadata(nullptr, nullptr, data, size,
                                          out_buf);
        } else {
          (void)conn;
          FifoItem remote_mem_info;
          remote_mem_info.addr = reinterpret_cast<uint64_t>(data);
          remote_mem_info.size = size;
          copyRKeysFromMRArrayToBytes(
              mhandle->mr_array, static_cast<char*>(remote_mem_info.padding),
              sizeof(remote_mem_info.padding));
          serialize_fifo_item(remote_mem_info, out_buf);
          return 0;
        }
      },
      ep);
}

inline void uccl_deregmr(RDMAEndPoint const& ep, P2PMhandle* mhandle) {
  std::visit(
      [&](auto const& s) {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, tcp::TCPEndpoint>) {
          (void)mhandle;
        } else {
          s->uccl_deregmr(mhandle->cache_refs);
        }
      },
      ep);
}

inline bool initialize_rdma_ctx_for_gpu(RDMAEndPoint const& ep, int dev) {
  return std::visit(
      [&](auto const& s) -> bool {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, tcp::TCPEndpoint>)
          return s->initialize_engine_by_dev(dev, true);
        else
          return s->initialize_rdma_ctx_for_gpu(dev);
      },
      ep);
}

inline void create_unified_p2p_socket(RDMAEndPoint const& ep) {
  std::visit(
      [](auto const& s) {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, tcp::TCPEndpoint>)
          (void)s;
        else
          s->create_unified_p2p_socket();
      },
      ep);
}

inline std::shared_ptr<EpollClient> get_oob_client(RDMAEndPoint const& ep) {
  return std::visit(
      [](auto const& s) -> std::shared_ptr<EpollClient> {
        return s->get_oob_client();
      },
      ep);
}

inline std::string get_oob_conn_key(RDMAEndPoint const& ep, uint64_t rank_id) {
  return std::visit(
      [&](auto const& s) -> std::string {
        return s->get_oob_conn_key(rank_id);
      },
      ep);
}
