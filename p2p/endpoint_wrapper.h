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

inline ConnID uccl_connect(RDMAEndPoint const& s, int remote_gpuidx,
                           std::string remote_ip, uint16_t remote_port) {
  return s->uccl_connect(remote_gpuidx, remote_ip, remote_port);
}
inline uint16_t get_p2p_listen_port(RDMAEndPoint const& s) {
  return s->get_p2p_listen_port();
}

inline int get_p2p_listen_fd(RDMAEndPoint const& s) {
  return s->get_p2p_listen_fd();
}

inline ConnID uccl_accept(RDMAEndPoint const& s, std::string& remote_ip,
                          int* remote_gpuidx) {
  return s->uccl_accept(remote_ip, remote_gpuidx);
}

inline bool uccl_regmr(RDMAEndPoint const& s, void* data, size_t len,
                       struct P2PMhandle* mhandle) {
  return s->uccl_regmr(data, len, mhandle->mr_array) >= 0;
}

inline int uccl_send_async(RDMAEndPoint const& s, Conn* conn,
                           P2PMhandle* mhandle, void const* data,
                           size_t const size, struct ucclRequest* ureq) {
  auto send_mem = std::make_shared<RegMemBlock>(const_cast<void*>(data), size,
                                                MemoryType::GPU);
  send_mem->mr_array = mhandle->mr_array;
  auto remote_mem_placeholder = std::make_shared<RemoteMemInfo>();
  auto send_req =
      std::make_shared<RDMASendRequest>(send_mem, remote_mem_placeholder);
  ureq->type = ReqType::ReqTx;
  send_req->to_rank_id = conn->uccl_conn_id_.flow_id;
  ureq->engine_idx = s->sendWithoutInnerQueue(send_req);
  while (ureq->engine_idx < 0) {
    ureq->engine_idx = s->sendWithoutInnerQueue(send_req);
  }
  ureq->n = conn->uccl_conn_id_.flow_id;
  return ureq->engine_idx;
}

inline int uccl_recv_async(RDMAEndPoint const& s, Conn* conn,
                           P2PMhandle* mhandles, void** data, int* size, int n,
                           struct ucclRequest* ureq) {
  auto recv_mem =
      std::make_shared<RegMemBlock>(data[0], size[0], MemoryType::GPU);
  recv_mem->mr_array = mhandles->mr_array;
  auto recv_req = std::make_shared<RDMARecvRequest>(recv_mem);
  ureq->type = ReqType::ReqRx;
  ureq->engine_idx = s->recv(conn->uccl_conn_id_.flow_id, recv_req);
  ureq->n = conn->uccl_conn_id_.flow_id;
  return ureq->engine_idx;
}

inline bool uccl_poll_ureq_once(RDMAEndPoint const& s,
                                struct ucclRequest* ureq) {
  if (ureq->type == ReqType::ReqTx || ureq->type == ReqType::ReqWrite ||
      ureq->type == ReqType::ReqRead) {
    s->sendRoutine();
    return s->checkSendComplete_once(ureq->n, ureq->engine_idx);
  } else if (ureq->type == ReqType::ReqRx) {
    s->recvRoutine();
    return s->checkRecvComplete_once(ureq->n, ureq->engine_idx);
  }
  LOG(ERROR) << "Invalid request type: " << ureq->type;
  return false;
}

inline int uccl_read_async(RDMAEndPoint const& s, Conn* conn,
                           P2PMhandle* local_mh, void* dst, size_t size,
                           FifoItem const& slot_item, ucclRequest* ureq) {
  ureq->type = ReqType::ReqRead;
  return set_request(s, conn, local_mh, dst, size, slot_item, ureq);
}

inline int uccl_write_async(RDMAEndPoint const& s, Conn* conn,
                            P2PMhandle* local_mh, void* src, size_t size,
                            FifoItem const& slot_item, ucclRequest* ureq) {
  ureq->type = ReqType::ReqWrite;
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
  serialize_fifo_item(remote_mem_info, out_buf);
  return 0;
}

inline void uccl_deregmr(RDMAEndPoint const& s, P2PMhandle* mhandle) {
  s->uccl_deregmr(mhandle->mr_array);
}

inline bool initialize_rdma_ctx_for_gpu(RDMAEndPoint const& s, int dev) {
  return s->initialize_rdma_ctx_for_gpu(dev);
}

inline void create_unified_p2p_socket(RDMAEndPoint const& s) {
  s->create_unified_p2p_socket();
}
