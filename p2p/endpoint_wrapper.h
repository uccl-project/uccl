#pragma once
#include "engine.h"

#define EFA = 1
namespace my_namespace {

template <class T>
struct always_false : std::false_type {};

inline void delete_ep(RDMAEndPoint const& s) {
  std::visit(
      [](auto&& ep) {
        using T = std::decay_t<decltype(ep)>;

        if constexpr (std::is_pointer_v<T>) {
          // raw pointer: we own it → delete
          delete ep;
        } else if constexpr (std::is_same_v<T, std::shared_ptr<EFAEndpoint>>) {
          // shared_ptr: do nothing (shared_ptr handles lifetime)
        } else {
          static_assert(always_false<T>::value,
                        "Unhandled type in RDMAEndPoint variant");
        }
      },
      s);
}

inline uccl::ConnID uccl_connect(RDMAEndPoint const& s, int dev,
                                 int local_gpuidx, int remote_dev,
                                 int remote_gpuidx, std::string remote_ip,
                                 uint16_t remote_port) {
  return std::visit(
      [dev, local_gpuidx, remote_dev, remote_gpuidx, remote_ip,
       remote_port](auto&& obj) -> uccl::ConnID {
        return obj->uccl_connect(dev, local_gpuidx, remote_dev, remote_gpuidx,
                                 remote_ip, remote_port);
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
          // raw pointer: we own it → delete
          struct uccl::Mhandle* mh = mhandle->mhandle_;  // 必须是 Mhandle*
          return obj->uccl_regmr(dev, data, len, type, &mh);

          mhandle->mhandle_ = mh;
          if (mh == nullptr || mh->mr == nullptr) {
            return false;
          }
        } else if constexpr (std::is_same_v<T, std::shared_ptr<EFAEndpoint>>) {
          // shared_ptr: call the EFAEndpoint's regmr method
          if (obj->uccl_regmr(data, len, mhandle->mr_map) < 0) {
            return false;
          }
        } else {
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
        } else if constexpr (std::is_same_v<T, std::shared_ptr<EFAEndpoint>>) {
          auto send_mem = std::make_shared<RegMemBlock>(const_cast<void*>(data),
                                                        size, MemoryType::GPU);
          send_mem->mr_map = mhandle->mr_map;
          auto remote_mem_placeholder = std::make_shared<RemoteMemInfo>();
          auto send_req = std::make_shared<EFASendRequest>(
              send_mem, remote_mem_placeholder);
          ureq->type = uccl::ReqType::ReqTx;
          ureq->engine_idx = obj->send(conn->uccl_conn_id_.flow_id, send_req);
          std::cout<<"send_req::::::"<<send_req->wr_id<<std::endl;
          ureq->n = conn->uccl_conn_id_.flow_id;
          return ureq->engine_idx;
        } else {
          static_assert(always_false<T>::value,
                        "Unhandled type in RDMAEndPoint variant");
        }
        return 0;
      },
      s);
}

inline int uccl_recv_async(RDMAEndPoint const& s, Conn* conn,
                           my_namespace::P2PMhandle* mhandles, void** data,
                           int* size, int n, struct uccl::ucclRequest* ureq) {
  return std::visit(
      [conn, mhandles, data, size, n, ureq](auto&& obj) mutable -> int {
        using T = std::decay_t<decltype(obj)>;

        if constexpr (std::is_pointer_v<T>) {
          return obj->uccl_recv_async(
              static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context),
              &(mhandles->mhandle_), data, size, n, ureq);
        } else if constexpr (std::is_same_v<T, std::shared_ptr<EFAEndpoint>>) {
          auto recv_mem = std::make_shared<RegMemBlock>(data[0], size[0],
                                                        MemoryType::GPU);
          recv_mem->mr_map = mhandles->mr_map;
          auto recv_req = std::make_shared<EFARecvRequest>(recv_mem);
          ureq->type = uccl::ReqType::ReqRx;
          ureq->engine_idx = obj->recv(conn->uccl_conn_id_.flow_id, recv_req);
          ureq->n = conn->uccl_conn_id_.flow_id;
          return ureq->engine_idx;
        } else {
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
        } else if constexpr (std::is_same_v<T, std::shared_ptr<EFAEndpoint>>) {
          if (ureq->type == uccl::ReqType::ReqTx) {
            // LOG(INFO) << "Checking send complete for engine_idx: " << ureq->engine_idx << ", n: " << ureq->n;
            return obj->checkSendComplete_once(ureq->n, ureq->engine_idx);
          } else if (ureq->type == uccl::ReqType::ReqRx) {
            // LOG(INFO) << "Checking recv complete for engine_idx: " << ureq->engine_idx << ", n: " << ureq->n;
            return obj->checkRecvComplete_once(ureq->n, ureq->engine_idx);
          }
        } else {
          static_assert(always_false<T>::value,
                        "Unhandled type in RDMAEndPoint variant");
        }
        return false;
      },
      s);
}

inline int uccl_read_async(RDMAEndPoint const& s, uccl::UcclFlow* flow,
                           my_namespace::P2PMhandle* local_mh, void* dst,
                           size_t size, uccl::FifoItem const& slot_item,
                           uccl::ucclRequest* ureq) {
  return std::visit(
      [flow, local_mh, dst, size, &slot_item, ureq](auto&& obj) -> int {
        return obj->uccl_read_async(flow, local_mh->mhandle_, dst, size,
                                    slot_item, ureq);
      },
      s);
}

inline int uccl_write_async(RDMAEndPoint const& s, uccl::UcclFlow* flow,
                            my_namespace::P2PMhandle* local_mh, void* src,
                            size_t size, uccl::FifoItem const& slot_item,
                            uccl::ucclRequest* ureq) {
  return std::visit(
      [flow, local_mh, src, size, &slot_item, ureq](auto&& obj) -> int {
        return obj->uccl_write_async(flow, local_mh->mhandle_, src, size,
                                     slot_item, ureq);
      },
      s);
}

inline int prepare_fifo_metadata(RDMAEndPoint const& s, uccl::UcclFlow* flow,
                                 struct uccl::Mhandle** mhandles,
                                 void const* data, size_t size, char* out_buf) {
  return std::visit(
      [flow, mhandles, data, size, out_buf](auto&& obj) -> int {
        return obj->prepare_fifo_metadata(flow, mhandles, data, size, out_buf);
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
        } else if constexpr (std::is_same_v<T, std::shared_ptr<EFAEndpoint>>) {
          // shared_ptr: call with mr_map
          obj->uccl_deregmr(mhandle->mr_map);
        } else {
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
}  // namespace my_namespace
