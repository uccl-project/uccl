#pragma once
#include "engine.h"

namespace unified {

template <class T>
struct always_false : std::false_type {};

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

inline uccl::ConnID uccl_connect(RDMAEndPoint const& s, int dev,
                                 int local_gpuidx, int remote_dev,
                                 int remote_gpuidx, std::string remote_ip,
                                 uint16_t remote_port) {
  return s->uccl_connect(dev, local_gpuidx, remote_dev, remote_gpuidx,
                         remote_ip, remote_port);
}
inline uint16_t get_p2p_listen_port(RDMAEndPoint const& s, int dev) {
  return s->get_p2p_listen_port(dev);
}

inline int get_p2p_listen_fd(RDMAEndPoint const& s, int dev) {
  return s->get_p2p_listen_fd(dev);
}

inline uccl::ConnID uccl_accept(RDMAEndPoint const& s, int dev, int listen_fd,
                                int local_gpuidx, std::string& remote_ip,
                                int* remote_dev, int* remote_gpuidx) {
  return s->uccl_accept(dev, listen_fd, local_gpuidx, remote_ip, remote_dev,
                        remote_gpuidx);
}

inline bool uccl_regmr(RDMAEndPoint const& s, int dev, void* data, size_t len,
                       int type, struct P2PMhandle* mhandle) {
  return s->uccl_regmr(data, len, mhandle->mr_array) >= 0;
}

inline int uccl_send_async(RDMAEndPoint const& s, Conn* conn,
                           P2PMhandle* mhandle, void const* data,
                           size_t const size, struct uccl::ucclRequest* ureq) {
  auto send_mem = std::make_shared<RegMemBlock>(const_cast<void*>(data), size,
                                                MemoryType::GPU);
  send_mem->mr_array = mhandle->mr_array;
  auto remote_mem_placeholder = std::make_shared<RemoteMemInfo>();
  auto send_req =
      std::make_shared<RDMASendRequest>(send_mem, remote_mem_placeholder);
  ureq->type = uccl::ReqType::ReqTx;
  send_req->to_rank_id = conn->uccl_conn_id_.flow_id;
  ureq->engine_idx = s->sendWithoutInnerQueue(send_req);
  while (ureq->engine_idx < 0) {
    ureq->engine_idx = s->sendWithoutInnerQueue(send_req);
  }
  ureq->n = conn->uccl_conn_id_.flow_id;
  return ureq->engine_idx;
}

inline int uccl_recv_async(RDMAEndPoint const& s, Conn* conn,
                           unified::P2PMhandle* mhandles, void** data,
                           int* size, int n, struct uccl::ucclRequest* ureq) {
  auto recv_mem =
      std::make_shared<RegMemBlock>(data[0], size[0], MemoryType::GPU);
  recv_mem->mr_array = mhandles->mr_array;
  auto recv_req = std::make_shared<RDMARecvRequest>(recv_mem);
  ureq->type = uccl::ReqType::ReqRx;
  ureq->engine_idx = s->recv(conn->uccl_conn_id_.flow_id, recv_req);
  ureq->n = conn->uccl_conn_id_.flow_id;
  return ureq->engine_idx;
}

inline bool uccl_poll_ureq_once(RDMAEndPoint const& s,
                                struct uccl::ucclRequest* ureq) {
  if (ureq->type == uccl::ReqType::ReqTx ||
      ureq->type == uccl::ReqType::ReqWrite ||
      ureq->type == uccl::ReqType::ReqRead) {
    s->sendRoutine();
    return s->checkSendComplete_once(ureq->n, ureq->engine_idx);
  } else if (ureq->type == uccl::ReqType::ReqRx) {
    s->recvRoutine();
    return s->checkRecvComplete_once(ureq->n, ureq->engine_idx);
  }
}

inline int uccl_read_async(RDMAEndPoint const& s, Conn* conn,
                           unified::P2PMhandle* local_mh, void* dst,
                           size_t size, FifoItem const& slot_item,
                           uccl::ucclRequest* ureq) {
  ureq->type = uccl::ReqType::ReqRead;
  return set_request(s, conn, local_mh, dst, size, slot_item, ureq);
}

inline int uccl_write_async(RDMAEndPoint const& s, Conn* conn,
                            unified::P2PMhandle* local_mh, void* src,
                            size_t size, FifoItem const& slot_item,
                            uccl::ucclRequest* ureq) {
  ureq->type = uccl::ReqType::ReqWrite;
  return set_request(s, conn, local_mh, src, size, slot_item, ureq);
}

inline int prepare_fifo_metadata(RDMAEndPoint const& s, Conn* conn,
                                 P2PMhandle* mhandle, void const* data,
                                 size_t size, char* out_buf) {
  FifoItem remote_mem_info;
  remote_mem_info.addr = reinterpret_cast<uint64_t>(data);
  remote_mem_info.size = size;

  copyRKeysFromMRArrayToBytes(mhandle->mr_array,
                              static_cast<char*>(remote_mem_info.padding),
                              sizeof(remote_mem_info.padding));
  auto* rkeys1 = const_cast<RKeyArray*>(
      reinterpret_cast<RKeyArray const*>(remote_mem_info.padding));
  uccl::serialize_fifo_item(remote_mem_info, out_buf);
  return 0;
}

inline void uccl_deregmr(RDMAEndPoint const& s, P2PMhandle* mhandle) {
  s->uccl_deregmr(mhandle->mr_array);
}

inline int get_best_dev_idx(RDMAEndPoint const& s, int gpu_idx) {
  return s->get_best_dev_idx(gpu_idx);
}

inline bool initialize_engine_by_dev(RDMAEndPoint const& s, int dev,
                                     bool enable_p2p_listen) {
  return s->initialize_engine_by_dev(dev, enable_p2p_listen);
}

inline void create_unified_p2p_socket(RDMAEndPoint const& s) {
  s->create_unified_p2p_socket();
}

}  // namespace unified
