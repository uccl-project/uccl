#pragma once
#include "engine.h"
#include "util/debug.h"
#include "util/gpu_rt.h"

// ── Helpers ─────────────────────────────────────────────────────────────────
// Convert between the global UcclRequest and NcclRequest.
// They have different layouts (NCCL version has an extra context pointer).
static inline NcclRequest to_nccl_req(UcclRequest const& r) {
  NcclRequest out{};
  out.type = static_cast<NcclReqType>(static_cast<int>(r.type));
  out.peer_id = r.peer_id;
  out.context = nullptr;
  out.engine_idx = r.engine_idx;
  return out;
}

static inline void from_nccl_req(NcclRequest const& in, UcclRequest* out) {
  out->type = static_cast<ReqType>(static_cast<int>(in.type));
  out->peer_id = in.peer_id;
  out->engine_idx = in.engine_idx;
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

// RDMA-only: build a one-sided request from FifoItem metadata.
//
// To minimize per-iov allocations on hot paths (writev/readv), the three
// objects (RemoteMemInfo, RegMemBlock, RDMASendRequest) are co-allocated in
// a single control block via std::make_shared, and aliased shared_ptrs are
// handed out via the aliasing constructor — sharing one refcount.
struct PooledSendBundle {
  RegMemBlock local_mem_obj;
  RemoteMemInfo remote_mem_obj;
  RDMASendRequest req;
  PooledSendBundle()
      : req(std::shared_ptr<RegMemBlock>(), std::shared_ptr<RemoteMemInfo>()) {}
};

static inline int set_request(std::shared_ptr<RDMAEndpoint> const& obj,
                              Conn* conn, P2PMhandle* local_mh, void* src,
                              size_t size, FifoItem const& slot_item,
                              UcclRequest* ureq) {
  auto bundle = std::make_shared<PooledSendBundle>();

  bundle->remote_mem_obj.addr = slot_item.addr;
  bundle->remote_mem_obj.length = slot_item.size;
  bundle->remote_mem_obj.type = MemoryType::GPU;
  bundle->remote_mem_obj.rkey_array.copy_from(slot_item.padding);

  bundle->local_mem_obj.addr = src;
  bundle->local_mem_obj.size = size;
  bundle->local_mem_obj.type = MemoryType::GPU;
  bundle->local_mem_obj.mr_array = local_mh->mr_array;

  // Aliasing constructor: shares ownership/control-block with `bundle`.
  bundle->req.local_mem =
      std::shared_ptr<RegMemBlock>(bundle, &bundle->local_mem_obj);
  bundle->req.remote_mem =
      std::shared_ptr<RemoteMemInfo>(bundle, &bundle->remote_mem_obj);
  bundle->req.compress_ctx = local_mh->compress_ctx;
  bundle->req.to_peer_id = conn->uccl_conn_id_.peer_id;
  bundle->req.send_type =
      (ureq->type == ReqType::ReqRead) ? SendType::Read : SendType::Write;
  bundle->req.need_signaled = true;

  auto req = std::shared_ptr<RDMASendRequest>(bundle, &bundle->req);

  ureq->engine_idx = obj->write_or_read(req);
  ureq->peer_id = conn->uccl_conn_id_.peer_id;
  return ureq->engine_idx;
}

// Same as set_request() but skips RDMAEndpoint::write_or_read() (which does a
// per-call map.find on send_channel_groups_) by taking a pre-resolved
// SendConnection pointer directly. Used by writev/readv hot paths.
static inline int set_request_on_group(SendConnection* send_group, Conn* conn,
                                       P2PMhandle* local_mh, void* src,
                                       size_t size, FifoItem const& slot_item,
                                       UcclRequest* ureq) {
  auto bundle = std::make_shared<PooledSendBundle>();

  bundle->remote_mem_obj.addr = slot_item.addr;
  bundle->remote_mem_obj.length = slot_item.size;
  bundle->remote_mem_obj.type = MemoryType::GPU;
  bundle->remote_mem_obj.rkey_array.copy_from(slot_item.padding);

  bundle->local_mem_obj.addr = src;
  bundle->local_mem_obj.size = size;
  bundle->local_mem_obj.type = MemoryType::GPU;
  bundle->local_mem_obj.mr_array = local_mh->mr_array;

  bundle->req.local_mem =
      std::shared_ptr<RegMemBlock>(bundle, &bundle->local_mem_obj);
  bundle->req.remote_mem =
      std::shared_ptr<RemoteMemInfo>(bundle, &bundle->remote_mem_obj);
  bundle->req.compress_ctx = local_mh->compress_ctx;
  bundle->req.to_peer_id = conn->uccl_conn_id_.peer_id;
  bundle->req.send_type =
      (ureq->type == ReqType::ReqRead) ? SendType::Read : SendType::Write;
  bundle->req.need_signaled = true;

  auto req = std::shared_ptr<RDMASendRequest>(bundle, &bundle->req);

  int64_t wr_id = -1;
  while (wr_id < 0) {
    wr_id = send_group->post_write_or_read(req);
    if (wr_id < 0) std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
  ureq->engine_idx = static_cast<int>(wr_id);
  ureq->peer_id = conn->uccl_conn_id_.peer_id;
  return ureq->engine_idx;
}

// ── Dispatch functions ──────────────────────────────────────────────────────
// Each function uses std::visit to dispatch to the correct endpoint type
// at runtime. The if constexpr branches are resolved at compile time so
// there is no overhead beyond the variant type check.

inline ConnID uccl_connect(GenericEndpoint const& ep, int remote_gpuidx,
                           std::string remote_ip, uint16_t remote_port) {
  return std::visit(
      [&](auto const& s) -> ConnID {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, NCCLEndpoint>) {
          int local_gpu = s->gpu_index();
          auto c = s->uccl_connect(0, local_gpu, 0, remote_gpuidx, remote_ip,
                                   remote_port);
          return to_conn_id(c);
        } else {
          return s->uccl_connect(remote_gpuidx, remote_ip, remote_port);
        }
      },
      ep);
}

inline uint16_t get_p2p_listen_port(GenericEndpoint const& ep) {
  return std::visit(
      [](auto const& s) -> uint16_t {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, NCCLEndpoint>)
          return s->get_p2p_listen_port(0);
        else
          return s->get_p2p_listen_port();
      },
      ep);
}

inline int get_p2p_listen_fd(GenericEndpoint const& ep) {
  return std::visit(
      [](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, NCCLEndpoint>)
          return s->get_p2p_listen_fd(0);
        else
          return s->get_p2p_listen_fd();
      },
      ep);
}

inline ConnID uccl_accept(GenericEndpoint const& ep, std::string& remote_ip,
                          int* remote_gpuidx) {
  return std::visit(
      [&](auto const& s) -> ConnID {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, NCCLEndpoint>) {
          int remote_dev = 0;
          int local_gpu = s->gpu_index();
          auto c = s->uccl_accept(0, -1, local_gpu, remote_ip, &remote_dev,
                                  remote_gpuidx);
          return to_conn_id(c);
        } else {
          return s->uccl_accept(remote_ip, remote_gpuidx);
        }
      },
      ep);
}

inline void stop_accept(GenericEndpoint const& ep) {
  std::visit([](auto const& s) { s->stop_accept(); }, ep);
}

inline bool uccl_regmr(GenericEndpoint const& ep, void* data, size_t len,
                       struct P2PMhandle* mhandle) {
  return std::visit(
      [&](auto const& s) -> bool {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, NCCLEndpoint>) {
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

inline int uccl_send_async(GenericEndpoint const& ep, Conn* conn,
                           P2PMhandle* mhandle, void const* data,
                           size_t const size, UcclRequest* ureq) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, NCCLEndpoint>) {
          (void)mhandle;
          ureq->type = ReqType::ReqTx;
          ureq->peer_id = conn->uccl_conn_id_.peer_id;
          NcclRequest nreq = to_nccl_req(*ureq);
          int ret = s->uccl_send_async(
              reinterpret_cast<NcclFlow*>(conn->uccl_conn_id_.context), nullptr,
              data, size, &nreq);
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
          send_req->to_peer_id = conn->uccl_conn_id_.peer_id;
          ureq->type = ReqType::ReqTx;
          do {
            ureq->engine_idx = s->send_without_inner_queue(send_req);
          } while (ureq->engine_idx < 0);
          ureq->peer_id = conn->uccl_conn_id_.peer_id;
          return ureq->engine_idx;
        }
      },
      ep);
}

inline int uccl_recv_async(GenericEndpoint const& ep, Conn* conn,
                           P2PMhandle* mhandles, void** data, int* size, int n,
                           UcclRequest* ureq) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, NCCLEndpoint>) {
          (void)mhandles;
          ureq->type = ReqType::ReqRx;
          ureq->peer_id = conn->uccl_conn_id_.peer_id;
          NcclRequest nreq = to_nccl_req(*ureq);
          int ret = s->uccl_recv_async(
              reinterpret_cast<NcclFlow*>(conn->uccl_conn_id_.context), nullptr,
              data, size, n, &nreq);
          from_nccl_req(nreq, ureq);
          return ret;
        } else {
          auto recv_mem =
              std::make_shared<RegMemBlock>(data[0], size[0], MemoryType::GPU);
          recv_mem->mr_array = mhandles->mr_array;
          auto recv_req = std::make_shared<RDMARecvRequest>(recv_mem);
          recv_req->compress_ctx = mhandles->compress_ctx;
          ureq->type = ReqType::ReqRx;
          ureq->engine_idx = s->recv(conn->uccl_conn_id_.peer_id, recv_req);
          ureq->peer_id = conn->uccl_conn_id_.peer_id;
          return ureq->engine_idx;
        }
      },
      ep);
}

inline bool uccl_poll_ureq_once(GenericEndpoint const& ep, UcclRequest* ureq) {
  return std::visit(
      [&](auto const& s) -> bool {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, NCCLEndpoint>) {
          NcclRequest nreq = to_nccl_req(*ureq);
          bool ret = s->uccl_poll_ureq_once(&nreq);
          from_nccl_req(nreq, ureq);
          return ret;
        } else {
          if (ureq->type == ReqType::ReqTx || ureq->type == ReqType::ReqWrite ||
              ureq->type == ReqType::ReqRead) {
            s->send_routine();
            return s->checkSendComplete_once(ureq->peer_id, ureq->engine_idx);
          } else if (ureq->type == ReqType::ReqRx) {
            s->recv_routine();
            return s->checkRecvComplete_once(ureq->peer_id, ureq->engine_idx);
          }
          UCCL_LOG(ERROR) << "Invalid request type: " << ureq->type;
          return false;
        }
      },
      ep);
}

// Drive the send/recv pollers once for the whole endpoint. Cheap to call
// repeatedly but should be called only once per outer poll pass when
// completing many in-flight requests (see writev/readv).
inline void uccl_drive_send(GenericEndpoint const& ep) {
  std::visit(
      [](auto const& s) {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (!std::is_same_v<T, NCCLEndpoint>) {
          s->send_routine();
        }
      },
      ep);
}

// Flush any batched send WRs (doorbell-batched posting). Use together with
// the thread-local g_uccl_batch_post flag.
inline void uccl_flush_send(GenericEndpoint const& ep) {
  std::visit(
      [](auto const& s) {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (!std::is_same_v<T, NCCLEndpoint>) {
          s->flush_all_sends();
        }
      },
      ep);
}

// Resolve the SendConnection for a peer_id once. Returns nullptr on the
// NCCL path or if not found. Callers can then use uccl_check_wr_fast() to
// avoid the per-call mutex + map lookup in checkSendComplete_once().
inline SendConnection* uccl_resolve_send_group(GenericEndpoint const& ep,
                                               uint64_t peer_id) {
  SendConnection* result = nullptr;
  std::visit(
      [&](auto const& s) {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (!std::is_same_v<T, NCCLEndpoint>) {
          result = s->get_send_group_raw(peer_id);
        }
      },
      ep);
  return result;
}

// Fast completion check using a pre-resolved SendConnection*.
inline bool uccl_check_wr_fast(SendConnection* send_group, int64_t wr_id) {
  return send_group->check(wr_id);
}

inline void uccl_drive_recv(GenericEndpoint const& ep) {
  std::visit(
      [](auto const& s) {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (!std::is_same_v<T, NCCLEndpoint>) {
          s->recv_routine();
        }
      },
      ep);
}

// Check completion of a single request without driving the send/recv pollers.
// Use together with uccl_drive_send/uccl_drive_recv to amortize the polling
// work across many in-flight requests.
inline bool uccl_check_ureq_once(GenericEndpoint const& ep, UcclRequest* ureq) {
  return std::visit(
      [&](auto const& s) -> bool {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, NCCLEndpoint>) {
          NcclRequest nreq = to_nccl_req(*ureq);
          bool ret = s->uccl_poll_ureq_once(&nreq);
          from_nccl_req(nreq, ureq);
          return ret;
        } else {
          if (ureq->type == ReqType::ReqTx || ureq->type == ReqType::ReqWrite ||
              ureq->type == ReqType::ReqRead) {
            return s->checkSendComplete_once(ureq->peer_id, ureq->engine_idx);
          } else if (ureq->type == ReqType::ReqRx) {
            return s->checkRecvComplete_once(ureq->peer_id, ureq->engine_idx);
          }
          UCCL_LOG(ERROR) << "Invalid request type: " << ureq->type;
          return false;
        }
      },
      ep);
}

inline int uccl_read_async(GenericEndpoint const& ep, Conn* conn,
                           P2PMhandle* local_mh, void* dst, size_t size,
                           FifoItem const& slot_item, UcclRequest* ureq) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, NCCLEndpoint>) {
          (void)local_mh;
          ureq->type = ReqType::ReqRead;
          ureq->peer_id = conn->uccl_conn_id_.peer_id;
          NcclFifoItem nccl_item{};
          nccl_item.addr = slot_item.addr;
          nccl_item.size = static_cast<uint32_t>(size);
          std::memset(nccl_item.padding, 0, sizeof(nccl_item.padding));
          NcclRequest nreq = to_nccl_req(*ureq);
          int ret = s->uccl_read_async(
              reinterpret_cast<NcclFlow*>(conn->uccl_conn_id_.context), nullptr,
              dst, size, nccl_item, &nreq);
          from_nccl_req(nreq, ureq);
          return ret;
        } else {
          ureq->type = ReqType::ReqRead;
          return set_request(s, conn, local_mh, dst, size, slot_item, ureq);
        }
      },
      ep);
}

inline int uccl_write_async(GenericEndpoint const& ep, Conn* conn,
                            P2PMhandle* local_mh, void* src, size_t size,
                            FifoItem const& slot_item, UcclRequest* ureq) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, NCCLEndpoint>) {
          (void)local_mh;
          ureq->type = ReqType::ReqWrite;
          ureq->peer_id = conn->uccl_conn_id_.peer_id;
          NcclFifoItem nccl_item{};
          nccl_item.addr = slot_item.addr;
          nccl_item.size = static_cast<uint32_t>(size);
          std::memset(nccl_item.padding, 0, sizeof(nccl_item.padding));
          NcclRequest nreq = to_nccl_req(*ureq);
          int ret = s->uccl_write_async(
              reinterpret_cast<NcclFlow*>(conn->uccl_conn_id_.context), nullptr,
              src, size, nccl_item, &nreq);
          from_nccl_req(nreq, ureq);
          return ret;
        } else {
          ureq->type = ReqType::ReqWrite;
          return set_request(s, conn, local_mh, src, size, slot_item, ureq);
        }
      },
      ep);
}

// RDMA-only fast variant: pre-resolved SendConnection*, no variant visit,
// no send_channel_groups_ lookup. Caller guarantees the path is RDMA.
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

inline int prepare_fifo_metadata(GenericEndpoint const& ep, Conn* conn,
                                 P2PMhandle* mhandle, void const* data,
                                 size_t size, char* out_buf) {
  return std::visit(
      [&](auto const& s) -> int {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, NCCLEndpoint>) {
          (void)conn;
          (void)mhandle;
          return s->prepare_fifo_metadata(nullptr, nullptr, data, size,
                                          out_buf);
        } else {
          (void)conn;
          FifoItem remote_mem_info;
          remote_mem_info.addr = reinterpret_cast<uint64_t>(data);
          remote_mem_info.size = size;
          copy_r_keys_from_mr_array_to_bytes(
              mhandle->mr_array, static_cast<char*>(remote_mem_info.padding),
              sizeof(remote_mem_info.padding));
          serialize_fifo_item(remote_mem_info, out_buf);
          return 0;
        }
      },
      ep);
}

inline void uccl_deregmr(GenericEndpoint const& ep, P2PMhandle* mhandle) {
  std::visit(
      [&](auto const& s) {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, NCCLEndpoint>) {
          (void)mhandle;
        } else {
          s->uccl_deregmr(mhandle->cache_refs);
        }
      },
      ep);
}

inline bool initialize_rdma_ctx_for_gpu(GenericEndpoint const& ep, int dev) {
  return std::visit(
      [&](auto const& s) -> bool {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, NCCLEndpoint>)
          return s->initialize_engine_by_dev(dev, true);
        else
          return s->initialize_rdma_ctx_for_gpu(dev);
      },
      ep);
}

inline void create_unified_p2p_socket(GenericEndpoint const& ep) {
  std::visit(
      [](auto const& s) {
        using T = std::decay_t<decltype(*s)>;
        if constexpr (std::is_same_v<T, NCCLEndpoint>)
          (void)s;
        else
          s->create_unified_p2p_socket();
      },
      ep);
}

inline std::shared_ptr<EpollClient> get_oob_client(GenericEndpoint const& ep) {
  return std::visit(
      [](auto const& s) -> std::shared_ptr<EpollClient> {
        return s->get_oob_client();
      },
      ep);
}

inline std::string get_oob_conn_key(GenericEndpoint const& ep,
                                    uint64_t peer_id) {
  return std::visit(
      [&](auto const& s) -> std::string {
        return s->get_oob_conn_key(peer_id);
      },
      ep);
}
