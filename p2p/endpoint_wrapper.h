#pragma once
#include "engine.h"

template <class T>
struct always_false : std::false_type {};

static inline int set_request(std::shared_ptr<NICEndpoint> const& obj,
                              Conn* conn, P2PMhandle* local_mh, void* src,
                              size_t size, FifoItem const& slot_item,
                              ucclRequest* ureq) {
  // Create RemoteMemInfo from FifoItem
  auto remote_mem = std::make_shared<RemoteMemInfo>();
  remote_mem->addr = slot_item.addr;
  remote_mem->length = slot_item.size;
  remote_mem->type = MemoryType::GPU;
  remote_mem->rkey_array.copyFrom(slot_item.padding);
  // Create RegMemBlock for local memory
  auto local_mem = std::make_shared<RegMemBlock>(src, size, MemoryType::GPU);
  local_mem->mr_array = local_mh->mr_array;

  auto req = std::make_shared<RDMASendRequest>(local_mem, remote_mem);
  req->to_rank_id = conn->uccl_conn_id_.flow_id;

  req->send_type = SendType::Write;
  ureq->engine_idx = obj->writeOrRead(req);
  ureq->n = conn->uccl_conn_id_.flow_id;

  return ureq->engine_idx;
}

#ifdef UCCL_P2P_USE_NCCL
static inline ConnID to_conn_id(uccl::ConnID const& in) {
  ConnID out{};
  out.context = in.context;
  out.sock_fd = in.sock_fd;
  out.flow_id = in.flow_id;
  out.dev = in.dev;
  return out;
}

static inline int tcp_set_request_write(
    std::shared_ptr<tcp::TCPEndpoint> const& obj, Conn* conn,
    P2PMhandle* local_mh, void* src, size_t size,
    FifoItem const& slot_item, ucclRequest* ureq) {
  (void)local_mh;
  ureq->type = uccl::ReqType::ReqWrite;
  ureq->n = conn->uccl_conn_id_.flow_id;
  auto const& tcp_item =
      reinterpret_cast<uccl::FifoItem const&>(slot_item);
  return obj->uccl_write_async(
      reinterpret_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), nullptr,
      src, size, tcp_item, ureq);
}

static inline int tcp_set_request_read(
    std::shared_ptr<tcp::TCPEndpoint> const& obj, Conn* conn,
    P2PMhandle* local_mh, void* dst, size_t size,
    FifoItem const& slot_item, ucclRequest* ureq) {
  (void)local_mh;
  ureq->type = uccl::ReqType::ReqRead;
  ureq->n = conn->uccl_conn_id_.flow_id;
  auto const& tcp_item =
      reinterpret_cast<uccl::FifoItem const&>(slot_item);
  return obj->uccl_read_async(
      reinterpret_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), nullptr,
      dst, size, tcp_item, ureq);
}
#endif

inline ConnID uccl_connect(RDMAEndPoint const& s, int local_gpuidx,
                           int remote_gpuidx, std::string remote_ip,
                           uint16_t remote_port) {
  return std::visit(
      [local_gpuidx, remote_gpuidx, remote_ip,
       remote_port](auto&& obj) -> ConnID {
        using T = std::decay_t<decltype(obj)>;
#ifdef UCCL_P2P_USE_NCCL
        if constexpr (std::is_same_v<T, std::shared_ptr<tcp::TCPEndpoint>>) {
          auto uccl_conn = obj->uccl_connect(0, local_gpuidx, 0, remote_gpuidx,
                                             remote_ip, remote_port);
          return to_conn_id(uccl_conn);
        } else
#endif
        {
          return obj->uccl_connect(remote_gpuidx, remote_ip, remote_port);
        }
      },
      s);
}

inline uint16_t get_p2p_listen_port(RDMAEndPoint const& s) {
  return std::visit(
      [](auto&& obj) -> uint16_t {
        using T = std::decay_t<decltype(obj)>;
#ifdef UCCL_P2P_USE_NCCL
        if constexpr (std::is_same_v<T, std::shared_ptr<tcp::TCPEndpoint>>) {
          return obj->get_p2p_listen_port(0);
        } else
#endif
        {
          return obj->get_p2p_listen_port();
        }
      },
      s);
}

inline int get_p2p_listen_fd(RDMAEndPoint const& s) {
  return std::visit(
      [](auto&& obj) -> int {
        using T = std::decay_t<decltype(obj)>;
#ifdef UCCL_P2P_USE_NCCL
        if constexpr (std::is_same_v<T, std::shared_ptr<tcp::TCPEndpoint>>) {
          return obj->get_p2p_listen_fd(0);
        } else
#endif
        {
          return obj->get_p2p_listen_fd();
        }
      },
      s);
}

inline ConnID uccl_accept(RDMAEndPoint const& s, int local_gpuidx,
                          std::string& remote_ip, int* remote_gpuidx) {
  return std::visit(
      [local_gpuidx, &remote_ip, remote_gpuidx](auto&& obj) -> ConnID {
        using T = std::decay_t<decltype(obj)>;
#ifdef UCCL_P2P_USE_NCCL
        if constexpr (std::is_same_v<T, std::shared_ptr<tcp::TCPEndpoint>>) {
          int remote_dev = 0;
          auto uccl_conn =
              obj->uccl_accept(0, -1, local_gpuidx, remote_ip, &remote_dev,
                               remote_gpuidx);
          return to_conn_id(uccl_conn);
        } else
#endif
        {
          return obj->uccl_accept(remote_ip, remote_gpuidx);
        }
      },
      s);
}

inline bool uccl_regmr(RDMAEndPoint const& s, void* data, size_t len,
                       P2PMhandle* mhandle) {
  return std::visit(
      [data, len, mhandle](auto&& obj) -> bool {
        using T = std::decay_t<decltype(obj)>;
#ifdef UCCL_P2P_USE_NCCL
        if constexpr (std::is_same_v<T, std::shared_ptr<tcp::TCPEndpoint>>) {
          (void)data;
          (void)len;
          (void)mhandle;
          return true;
        } else
#endif
        {
          return obj->uccl_regmr(data, len, mhandle->mr_array) >= 0;
        }
      },
      s);
}

inline int uccl_send_async(RDMAEndPoint const& s, Conn* conn,
                           P2PMhandle* mhandle, void const* data,
                           size_t const size, ucclRequest* ureq) {
  return std::visit(
      [conn, mhandle, data, size, ureq](auto&& obj) mutable -> int {
        using T = std::decay_t<decltype(obj)>;
#ifdef UCCL_P2P_USE_NCCL
        if constexpr (std::is_same_v<T, std::shared_ptr<tcp::TCPEndpoint>>) {
          ureq->type = uccl::ReqType::ReqTx;
          ureq->n = conn->uccl_conn_id_.flow_id;
          return obj->uccl_send_async(
              reinterpret_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context),
              nullptr, data, size, ureq);
        } else
#endif
        {
          auto send_mem = std::make_shared<RegMemBlock>(
              const_cast<void*>(data), size, MemoryType::GPU);
          send_mem->mr_array = mhandle->mr_array;
          auto remote_mem_placeholder = std::make_shared<RemoteMemInfo>();
          auto send_req = std::make_shared<RDMASendRequest>(
              send_mem, remote_mem_placeholder);
          ureq->type = uccl::ReqType::ReqTx;
          send_req->to_rank_id = conn->uccl_conn_id_.flow_id;
          ureq->engine_idx = obj->sendWithoutInnerQueue(send_req);
          while (ureq->engine_idx < 0) {
            ureq->engine_idx = obj->sendWithoutInnerQueue(send_req);
          }
          ureq->n = conn->uccl_conn_id_.flow_id;
          return ureq->engine_idx;
        }
      },
      s);
}

inline int uccl_recv_async(RDMAEndPoint const& s, Conn* conn,
                           P2PMhandle* mhandles, void** data, int* size, int n,
                           ucclRequest* ureq) {
  return std::visit(
      [conn, mhandles, data, size, n, ureq](auto&& obj) mutable -> int {
        using T = std::decay_t<decltype(obj)>;
#ifdef UCCL_P2P_USE_NCCL
        if constexpr (std::is_same_v<T, std::shared_ptr<tcp::TCPEndpoint>>) {
          ureq->type = uccl::ReqType::ReqRx;
          ureq->n = conn->uccl_conn_id_.flow_id;
          return obj->uccl_recv_async(
              reinterpret_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context),
              nullptr, data, size, n, ureq);
        } else
#endif
        {
          auto recv_mem =
              std::make_shared<RegMemBlock>(data[0], size[0], MemoryType::GPU);
          recv_mem->mr_array = mhandles->mr_array;
          auto recv_req = std::make_shared<RDMARecvRequest>(recv_mem);
          ureq->type = uccl::ReqType::ReqRx;
          ureq->engine_idx = obj->recv(conn->uccl_conn_id_.flow_id, recv_req);
          ureq->n = conn->uccl_conn_id_.flow_id;
          return ureq->engine_idx;
        }
      },
      s);
}

inline bool uccl_poll_ureq_once(RDMAEndPoint const& s, ucclRequest* ureq) {
  return std::visit(
      [ureq](auto&& obj) -> bool {
        using T = std::decay_t<decltype(obj)>;
#ifdef UCCL_P2P_USE_NCCL
        if constexpr (std::is_same_v<T, std::shared_ptr<tcp::TCPEndpoint>>) {
          return obj->uccl_poll_ureq_once(ureq);
        } else
#endif
        {
          if (ureq->type == uccl::ReqType::ReqTx ||
              ureq->type == uccl::ReqType::ReqWrite ||
              ureq->type == uccl::ReqType::ReqRead) {
            obj->sendRoutine();
            return obj->checkSendComplete_once(ureq->n, ureq->engine_idx);
          } else if (ureq->type == uccl::ReqType::ReqRx) {
            obj->recvRoutine();
            return obj->checkRecvComplete_once(ureq->n, ureq->engine_idx);
          }
          LOG(ERROR) << "Invalid request type: " << ureq->type;
          return false;
        }
      },
      s);
}

inline int uccl_read_async(RDMAEndPoint const& s, Conn* conn,
                           P2PMhandle* local_mh, void* dst, size_t size,
                           FifoItem const& slot_item, ucclRequest* ureq) {
  return std::visit(
      [conn, local_mh, dst, size, &slot_item, ureq](auto&& obj) -> int {
        using T = std::decay_t<decltype(obj)>;
#ifdef UCCL_P2P_USE_NCCL
        if constexpr (std::is_same_v<T, std::shared_ptr<tcp::TCPEndpoint>>) {
          return tcp_set_request_read(obj, conn, local_mh, dst, size,
                                      slot_item, ureq);
        } else
#endif
        {
          ureq->type = uccl::ReqType::ReqRead;
          return set_request(obj, conn, local_mh, dst, size, slot_item, ureq);
        }
      },
      s);
}

inline int uccl_write_async(RDMAEndPoint const& s, Conn* conn,
                            P2PMhandle* local_mh, void* src, size_t size,
                            FifoItem const& slot_item, ucclRequest* ureq) {
  return std::visit(
      [conn, local_mh, src, size, &slot_item, ureq](auto&& obj) -> int {
        using T = std::decay_t<decltype(obj)>;
#ifdef UCCL_P2P_USE_NCCL
        if constexpr (std::is_same_v<T, std::shared_ptr<tcp::TCPEndpoint>>) {
          return tcp_set_request_write(obj, conn, local_mh, src, size,
                                       slot_item, ureq);
        } else
#endif
        {
          ureq->type = uccl::ReqType::ReqWrite;
          return set_request(obj, conn, local_mh, src, size, slot_item, ureq);
        }
      },
      s);
}

inline int prepare_fifo_metadata(RDMAEndPoint const& s, Conn* conn,
                                 P2PMhandle* mhandle, void const* data,
                                 size_t size, char* out_buf) {
  (void)conn;
  return std::visit(
      [data, size, mhandle, out_buf](auto&& obj) -> int {
        using T = std::decay_t<decltype(obj)>;
#ifdef UCCL_P2P_USE_NCCL
        if constexpr (std::is_same_v<T, std::shared_ptr<tcp::TCPEndpoint>>) {
          FifoItem remote_mem_info{};
          remote_mem_info.addr = reinterpret_cast<uint64_t>(data);
          remote_mem_info.size = static_cast<uint32_t>(size);
          std::memset(remote_mem_info.padding, 0,
                      sizeof(remote_mem_info.padding));
          serialize_fifo_item(remote_mem_info, out_buf);
          return 0;
        } else
#endif
        {
          FifoItem remote_mem_info{};
          remote_mem_info.addr = reinterpret_cast<uint64_t>(data);
          remote_mem_info.size = static_cast<uint32_t>(size);
          copyRKeysFromMRArrayToBytes(
              mhandle->mr_array, static_cast<char*>(remote_mem_info.padding),
              sizeof(remote_mem_info.padding));
          serialize_fifo_item(remote_mem_info, out_buf);
          return 0;
        }
      },
      s);
}

inline void uccl_deregmr(RDMAEndPoint const& s, P2PMhandle* mhandle) {
  std::visit(
      [mhandle](auto&& obj) {
        using T = std::decay_t<decltype(obj)>;
#ifdef UCCL_P2P_USE_NCCL
        if constexpr (std::is_same_v<T, std::shared_ptr<tcp::TCPEndpoint>>) {
          (void)mhandle;
          return;
        } else
#endif
        {
          obj->uccl_deregmr(mhandle->mr_array);
        }
      },
      s);
}

inline bool initialize_rdma_ctx_for_gpu(RDMAEndPoint const& s, int dev) {
  return std::visit(
      [dev](auto&& obj) -> bool {
        using T = std::decay_t<decltype(obj)>;
#ifdef UCCL_P2P_USE_NCCL
        if constexpr (std::is_same_v<T, std::shared_ptr<tcp::TCPEndpoint>>) {
          (void)dev;
          return true;
        } else
#endif
        {
          return obj->initialize_rdma_ctx_for_gpu(dev);
        }
      },
      s);
}

inline void create_unified_p2p_socket(RDMAEndPoint const& s) {
  std::visit(
      [](auto&& obj) {
        using T = std::decay_t<decltype(obj)>;
#ifdef UCCL_P2P_USE_NCCL
        if constexpr (std::is_same_v<T, std::shared_ptr<tcp::TCPEndpoint>>) {
          return;
        } else
#endif
        {
          obj->create_unified_p2p_socket();
        }
      },
      s);
}
