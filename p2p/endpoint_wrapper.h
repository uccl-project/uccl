#pragma once
#include "engine.h"

#ifdef UCCL_P2P_USE_NCCL
#include "nccl/nccl_endpoint.h"
#endif

namespace unified {

template <class T>
struct always_false : std::false_type {};

inline void delete_ep(RDMAEndPoint const& s) {
  std::visit(
      [](auto&& ep) {
        using T = std::decay_t<decltype(ep)>;

        if constexpr (std::is_pointer_v<T>) {
          // raw pointer: we own it → delete
          delete ep;
        }
#ifdef UCCL_P2P_USE_NATIVE_RDMA
        else if constexpr (std::is_same_v<T, std::shared_ptr<NICEndpoint>>) {
          // shared_ptr: do nothing (shared_ptr handles lifetime)
        }
#endif
#ifdef UCCL_P2P_USE_NCCL
        else if constexpr (std::is_same_v<T,
                                          std::shared_ptr<tcp::TCPEndpoint>>) {
          // shared_ptr: do nothing (shared_ptr handles lifetime)
        }
#endif
        else {
          static_assert(always_false<T>::value,
                        "Unhandled type in RDMAEndPoint variant");
        }
      },
      s);
}

#ifdef UCCL_P2P_USE_NATIVE_RDMA
inline int set_request(std::shared_ptr<NICEndpoint> const& obj, Conn* conn,
                       unified::P2PMhandle* local_mh, void* src, size_t size,
                       FifoItem const& slot_item, uccl::ucclRequest* ureq) {
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
#endif

#ifdef UCCL_P2P_USE_NCCL
inline int tcp_set_request_write(std::shared_ptr<tcp::TCPEndpoint> const& obj,
                                 Conn* conn, unified::P2PMhandle* local_mh,
                                 void* src, size_t size,
                                 FifoItem const& slot_item,
                                 uccl::ucclRequest* ureq) {
  ureq->type = uccl::ReqType::ReqWrite;
  ureq->n = conn->uccl_conn_id_.flow_id;
  return obj->uccl_write_async(
      reinterpret_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), nullptr,
      src, size, slot_item, ureq);
}

inline int tcp_set_request_read(std::shared_ptr<tcp::TCPEndpoint> const& obj,
                                Conn* conn, unified::P2PMhandle* local_mh,
                                void* dst, size_t size,
                                FifoItem const& slot_item,
                                uccl::ucclRequest* ureq) {
  ureq->type = uccl::ReqType::ReqRead;
  ureq->n = conn->uccl_conn_id_.flow_id;
  return obj->uccl_read_async(
      reinterpret_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), nullptr,
      dst, size, slot_item, ureq);
}
#endif

inline uccl::ConnID uccl_connect(RDMAEndPoint const& s, int dev,
                                 int local_gpuidx, int remote_dev,
                                 int remote_gpuidx, std::string remote_ip,
                                 uint16_t remote_port) {
  return std::visit(
      [dev, local_gpuidx, remote_dev, remote_gpuidx, remote_ip,
       remote_port](auto&& obj) -> uccl::ConnID {
        using T = std::decay_t<decltype(obj)>;
#ifdef UCCL_P2P_USE_NCCL
        if constexpr (std::is_same_v<T, std::shared_ptr<tcp::TCPEndpoint>>) {
          return obj->uccl_connect(dev, local_gpuidx, remote_dev, remote_gpuidx,
                                   remote_ip, remote_port);
        } else
#endif
        {
          return obj->uccl_connect(dev, local_gpuidx, remote_dev, remote_gpuidx,
                                   remote_ip, remote_port);
        }
      },
      s);
}
inline uint16_t get_p2p_listen_port(RDMAEndPoint const& s, int dev) {
  return std::visit(
      [dev](auto&& obj) -> uint16_t { return obj->get_p2p_listen_port(dev); },
      s);
}

inline int get_p2p_listen_fd(RDMAEndPoint const& s, int dev) {
  return std::visit(
      [dev](auto&& obj) -> int { return obj->get_p2p_listen_fd(dev); }, s);
}

inline uccl::ConnID uccl_accept(RDMAEndPoint const& s, int dev, int listen_fd,
                                int local_gpuidx, std::string& remote_ip,
                                int* remote_dev, int* remote_gpuidx) {
  return std::visit(
      [dev, listen_fd, local_gpuidx, &remote_ip, remote_dev,
       remote_gpuidx](auto&& obj) -> uccl::ConnID {
        return obj->uccl_accept(dev, listen_fd, local_gpuidx, remote_ip,
                                remote_dev, remote_gpuidx);
      },
      s);
}
inline int uccl_regmr(RDMAEndPoint const& s, uccl::UcclFlow* flow, void* data,
                      size_t len, int type, struct uccl::Mhandle** mhandle) {
  return std::visit(
      [flow, data, len, type, mhandle](auto&& obj) -> int {
        return obj->uccl_regmr(flow, data, len, type, mhandle);
      },
      s);
}

inline bool uccl_regmr(RDMAEndPoint const& s, int dev, void* data, size_t len,
                       int type, struct P2PMhandle* mhandle) {
  return std::visit(
      [dev, data, len, type, mhandle](auto&& obj) mutable -> int {
        using T = std::decay_t<decltype(obj)>;

        if constexpr (std::is_pointer_v<T>) {
          obj->uccl_regmr(dev, data, len, type, &(mhandle->mhandle_));
          if (mhandle->mhandle_ == nullptr ||
              mhandle->mhandle_->mr == nullptr) {
            return false;
          }
        }
#ifdef UCCL_P2P_USE_NATIVE_RDMA
        else if constexpr (std::is_same_v<T, std::shared_ptr<NICEndpoint>>) {
          if (obj->uccl_regmr(data, len, mhandle->mr_array) < 0) {
            return false;
          }
        }
#endif
#ifdef UCCL_P2P_USE_NCCL
        else if constexpr (std::is_same_v<T,
                                          std::shared_ptr<tcp::TCPEndpoint>>) {
          // TCP doesn't need memory registration
          mhandle->mhandle_ = nullptr;
        }
#endif
        else {
          static_assert(always_false<T>::value,
                        "Unhandled type in RDMAEndPoint variant");
        }
        return true;
      },
      s);
}

inline int uccl_send_async(RDMAEndPoint const& s, Conn* conn,
                           P2PMhandle* mhandle, void const* data,
                           size_t const size, struct uccl::ucclRequest* ureq) {
  return std::visit(
      [conn, mhandle, data, size, ureq](auto&& obj) mutable -> int {
        using T = std::decay_t<decltype(obj)>;

        if constexpr (std::is_pointer_v<T>) {
          // raw pointer: we own it → delete
          struct uccl::Mhandle* mh = mhandle->mhandle_;
          return obj->uccl_send_async(
              static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), mh,
              data, size, ureq);
        }
#ifdef UCCL_P2P_USE_NATIVE_RDMA
        else if constexpr (std::is_same_v<T, std::shared_ptr<NICEndpoint>>) {
          auto send_mem = std::make_shared<RegMemBlock>(const_cast<void*>(data),
                                                        size, MemoryType::GPU);
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
          // std::cout<<"send_req::::::"<<send_req->wr_id<<std::endl;
          ureq->n = conn->uccl_conn_id_.flow_id;
          return ureq->engine_idx;
        }
#endif
#ifdef UCCL_P2P_USE_NCCL
        else if constexpr (std::is_same_v<T,
                                          std::shared_ptr<tcp::TCPEndpoint>>) {
          ureq->type = uccl::ReqType::ReqTx;
          ureq->n = conn->uccl_conn_id_.flow_id;
          return obj->uccl_send_async(
              reinterpret_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context),
              nullptr, data, size, ureq);
        }
#endif
        else {
          static_assert(always_false<T>::value,
                        "Unhandled type in RDMAEndPoint variant");
        }
        return 0;
      },
      s);
}

inline int uccl_recv_async(RDMAEndPoint const& s, Conn* conn,
                           unified::P2PMhandle* mhandles, void** data,
                           int* size, int n, struct uccl::ucclRequest* ureq) {
  return std::visit(
      [conn, mhandles, data, size, n, ureq](auto&& obj) mutable -> int {
        using T = std::decay_t<decltype(obj)>;

        if constexpr (std::is_pointer_v<T>) {
          return obj->uccl_recv_async(
              static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context),
              &(mhandles->mhandle_), data, size, n, ureq);
        }
#ifdef UCCL_P2P_USE_NATIVE_RDMA
        else if constexpr (std::is_same_v<T, std::shared_ptr<NICEndpoint>>) {
          auto recv_mem =
              std::make_shared<RegMemBlock>(data[0], size[0], MemoryType::GPU);
          recv_mem->mr_array = mhandles->mr_array;
          auto recv_req = std::make_shared<RDMARecvRequest>(recv_mem);
          ureq->type = uccl::ReqType::ReqRx;
          ureq->engine_idx = obj->recv(conn->uccl_conn_id_.flow_id, recv_req);
          ureq->n = conn->uccl_conn_id_.flow_id;
          return ureq->engine_idx;
        }
#endif
#ifdef UCCL_P2P_USE_NCCL
        else if constexpr (std::is_same_v<T,
                                          std::shared_ptr<tcp::TCPEndpoint>>) {
          ureq->type = uccl::ReqType::ReqRx;
          ureq->n = conn->uccl_conn_id_.flow_id;
          return obj->uccl_recv_async(
              reinterpret_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context),
              nullptr, data, size, n, ureq);
        }
#endif
        else {
          static_assert(always_false<T>::value,
                        "Unhandled type in RDMAEndPoint variant");
        }
        return true;
      },
      s);
}

inline bool uccl_poll_ureq_once(RDMAEndPoint const& s,
                                struct uccl::ucclRequest* ureq) {
  return std::visit(
      [ureq](auto&& obj) -> bool {
        using T = std::decay_t<decltype(obj)>;
        if constexpr (std::is_pointer_v<T>) {
          return obj->uccl_poll_ureq_once(ureq);
        }
#ifdef UCCL_P2P_USE_NATIVE_RDMA
        else if constexpr (std::is_same_v<T, std::shared_ptr<NICEndpoint>>) {
          if (ureq->type == uccl::ReqType::ReqTx ||
              ureq->type == uccl::ReqType::ReqWrite ||
              ureq->type == uccl::ReqType::ReqRead) {
            obj->sendRoutine();
            return obj->checkSendComplete_once(ureq->n, ureq->engine_idx);
          } else if (ureq->type == uccl::ReqType::ReqRx) {
            obj->recvRoutine();
            return obj->checkRecvComplete_once(ureq->n, ureq->engine_idx);
          }
        }
#endif
#ifdef UCCL_P2P_USE_NCCL
        else if constexpr (std::is_same_v<T,
                                          std::shared_ptr<tcp::TCPEndpoint>>) {
          // TCP operations are blocking, so always complete immediately
          return obj->uccl_poll_ureq_once(ureq);
        }
#endif
        else {
          static_assert(always_false<T>::value,
                        "Unhandled type in RDMAEndPoint variant");
        }
        return false;
      },
      s);
}

inline int uccl_read_async(RDMAEndPoint const& s, Conn* conn,
                           unified::P2PMhandle* local_mh, void* dst,
                           size_t size, FifoItem const& slot_item,
                           uccl::ucclRequest* ureq) {
  return std::visit(
      [conn, local_mh, dst, size, &slot_item, ureq](auto&& obj) -> int {
        using T = std::decay_t<decltype(obj)>;
        if constexpr (std::is_pointer_v<T>) {
          return obj->uccl_read_async(
              static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context),
              local_mh->mhandle_, dst, size,
              static_cast<uccl::FifoItem const&>(slot_item), ureq);
        }
#ifdef UCCL_P2P_USE_NATIVE_RDMA
        else if constexpr (std::is_same_v<T, std::shared_ptr<NICEndpoint>>) {
          ureq->type = uccl::ReqType::ReqRead;
          return set_request(obj, conn, local_mh, dst, size, slot_item, ureq);
        }
#endif
#ifdef UCCL_P2P_USE_NCCL
        else if constexpr (std::is_same_v<T,
                                          std::shared_ptr<tcp::TCPEndpoint>>) {
          return tcp_set_request_read(obj, conn, local_mh, dst, size, slot_item,
                                      ureq);
        }
#endif
        else {
          static_assert(always_false<T>::value,
                        "Unhandled type in RDMAEndPoint variant");
        }
        return 0;
      },
      s);
}

inline int uccl_write_async(RDMAEndPoint const& s, Conn* conn,
                            unified::P2PMhandle* local_mh, void* src,
                            size_t size, FifoItem const& slot_item,
                            uccl::ucclRequest* ureq) {
  return std::visit(
      [conn, local_mh, src, size, &slot_item, ureq](auto&& obj) -> int {
        using T = std::decay_t<decltype(obj)>;
        if constexpr (std::is_pointer_v<T>) {
          return obj->uccl_write_async(
              static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context),
              local_mh->mhandle_, src, size,
              static_cast<uccl::FifoItem const&>(slot_item), ureq);
        }
#ifdef UCCL_P2P_USE_NATIVE_RDMA
        else if constexpr (std::is_same_v<T, std::shared_ptr<NICEndpoint>>) {
          ureq->type = uccl::ReqType::ReqWrite;
          return set_request(obj, conn, local_mh, src, size, slot_item, ureq);
        }
#endif
#ifdef UCCL_P2P_USE_NCCL
        else if constexpr (std::is_same_v<T,
                                          std::shared_ptr<tcp::TCPEndpoint>>) {
          return tcp_set_request_write(obj, conn, local_mh, src, size,
                                       slot_item, ureq);
        }
#endif
        else {
          static_assert(always_false<T>::value,
                        "Unhandled type in RDMAEndPoint variant");
        }
        return 0;
      },
      s);
}

inline int prepare_fifo_metadata(RDMAEndPoint const& s, Conn* conn,
                                 P2PMhandle* mhandle, void const* data,
                                 size_t size, char* out_buf) {
  return std::visit(
      [conn, mhandle, data, size, out_buf](auto&& obj) -> int {
        using T = std::decay_t<decltype(obj)>;
        if constexpr (std::is_pointer_v<T>) {
          // raw pointer: call with mhandle_
          return obj->prepare_fifo_metadata(
              static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context),
              &(mhandle->mhandle_), data, size, out_buf);
        }
#ifdef UCCL_P2P_USE_NATIVE_RDMA
        else if constexpr (std::is_same_v<T, std::shared_ptr<NICEndpoint>>) {
          FifoItem remote_mem_info;
          remote_mem_info.addr = reinterpret_cast<uint64_t>(data);
          remote_mem_info.size = size;

          copyRKeysFromMRArrayToBytes(
              mhandle->mr_array, static_cast<char*>(remote_mem_info.padding),
              sizeof(remote_mem_info.padding));
          auto* rkeys1 = const_cast<RKeyArray*>(
              reinterpret_cast<const RKeyArray*>(remote_mem_info.padding));
          uccl::serialize_fifo_item(remote_mem_info, out_buf);
          return 0;
        }
#endif
#ifdef UCCL_P2P_USE_NCCL
        else if constexpr (std::is_same_v<T,
                                          std::shared_ptr<tcp::TCPEndpoint>>) {
          // For TCP, just store address and size (no rkeys needed)
          FifoItem remote_mem_info;
          remote_mem_info.addr = reinterpret_cast<uint64_t>(data);
          remote_mem_info.size = size;
          std::memset(remote_mem_info.padding, 0,
                      sizeof(remote_mem_info.padding));
          uccl::serialize_fifo_item(remote_mem_info, out_buf);
          return 0;
        }
#endif
        else {
          static_assert(always_false<T>::value,
                        "Unhandled type in RDMAEndPoint variant");
        }
      },
      s);
}

inline void uccl_deregmr(RDMAEndPoint const& s, P2PMhandle* mhandle) {
  std::visit(
      [mhandle](auto&& obj) {
        using T = std::decay_t<decltype(obj)>;

        if constexpr (std::is_pointer_v<T>) {
          // raw pointer: call with mhandle_
          obj->uccl_deregmr(mhandle->mhandle_);
        }
#ifdef UCCL_P2P_USE_NATIVE_RDMA
        else if constexpr (std::is_same_v<T, std::shared_ptr<NICEndpoint>>) {
          obj->uccl_deregmr(mhandle->mr_array);
        }
#endif
#ifdef UCCL_P2P_USE_NCCL
        else if constexpr (std::is_same_v<T,
                                          std::shared_ptr<tcp::TCPEndpoint>>) {
          // TCP doesn't need memory deregistration - no-op
        }
#endif
        else {
          static_assert(always_false<T>::value,
                        "Unhandled type in RDMAEndPoint variant");
        }
      },
      s);
}

inline int get_best_dev_idx(RDMAEndPoint const& s, int gpu_idx) {
  return std::visit(
      [gpu_idx](auto&& obj) -> int { return obj->get_best_dev_idx(gpu_idx); },
      s);
}

inline bool initialize_engine_by_dev(RDMAEndPoint const& s, int dev,
                                     bool enable_p2p_listen) {
  return std::visit(
      [dev, enable_p2p_listen](auto&& obj) -> bool {
        return obj->initialize_engine_by_dev(dev, enable_p2p_listen);
      },
      s);
}

inline void create_unified_p2p_socket(RDMAEndPoint const& s) {
  std::visit([](auto&& obj) { obj->create_unified_p2p_socket(); }, s);
}

}  // namespace unified
