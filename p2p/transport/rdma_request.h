#pragma once
#include "engine.h"
// RDMA-only: build a one-sided request from FifoItem metadata.
//
// The request owns a shared pair of memory descriptors. Keeping the request
// outside their allocation avoids a shared_ptr cycle during raw transfers.
struct SendRequestMemory {
  RegMemBlock local_mem_obj;
  RemoteMemInfo remote_mem_obj;
};

// A compress_only handle has no user MR: it may only be used as the exact
// registered source of a write.
static inline bool compress_only_send_rejected(P2PMhandle const* local_mh,
                                               void const* src, size_t size,
                                               UcclRequest const* ureq) {
  if (likely(!local_mh->compress_only)) return false;
  if (ureq->type == ReqType::ReqRead) {
    UCCL_LOG(ERROR) << "compress_only handle cannot be a read destination";
    return true;
  }
  if (src != local_mh->registered_addr || size != local_mh->registered_len) {
    UCCL_LOG(ERROR) << "compress_only handle must send exactly the buffer "
                       "passed to reg_compress_only";
    return true;
  }
  return false;
}

static inline int set_request(std::shared_ptr<RDMAEndpoint> const& obj,
                              Conn* conn, P2PMhandle* local_mh, void* src,
                              size_t size, FifoItem const& slot_item,
                              UcclRequest* ureq) {
  if (compress_only_send_rejected(local_mh, src, size, ureq))
    return SendConnection::kPostError;
  auto memory = std::make_shared<SendRequestMemory>();

  memory->remote_mem_obj.addr = slot_item.addr;
  memory->remote_mem_obj.length = slot_item.size;
  memory->remote_mem_obj.type = MemoryType::GPU;
  memory->remote_mem_obj.rkey_array.copy_from(slot_item.padding);

  memory->local_mem_obj.addr = src;
  memory->local_mem_obj.size = size;
  memory->local_mem_obj.type = MemoryType::GPU;
  memory->local_mem_obj.mr_array = local_mh->mr_array;

  auto req = std::make_shared<RDMASendRequest>(
      std::shared_ptr<RegMemBlock>(memory, &memory->local_mem_obj),
      std::shared_ptr<RemoteMemInfo>(memory, &memory->remote_mem_obj));
  if (src == local_mh->registered_addr && size == local_mh->registered_len)
    req->compress_ctx = local_mh->compress_ctx;
  req->compress_only = local_mh->compress_only;
  req->to_peer_id = conn->uccl_conn_id_.peer_id;
  req->send_type =
      (ureq->type == ReqType::ReqRead) ? SendType::Read : SendType::Write;
  req->need_signaled = true;

  int64_t wr_id = obj->write_or_read(req);
  if (wr_id == SendConnection::kPostError) return SendConnection::kPostError;
  ureq->engine_idx = wr_id;
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
  if (compress_only_send_rejected(local_mh, src, size, ureq))
    return SendConnection::kPostError;
  auto memory = std::make_shared<SendRequestMemory>();

  memory->remote_mem_obj.addr = slot_item.addr;
  memory->remote_mem_obj.length = slot_item.size;
  memory->remote_mem_obj.type = MemoryType::GPU;
  memory->remote_mem_obj.rkey_array.copy_from(slot_item.padding);

  memory->local_mem_obj.addr = src;
  memory->local_mem_obj.size = size;
  memory->local_mem_obj.type = MemoryType::GPU;
  memory->local_mem_obj.mr_array = local_mh->mr_array;

  auto req = std::make_shared<RDMASendRequest>(
      std::shared_ptr<RegMemBlock>(memory, &memory->local_mem_obj),
      std::shared_ptr<RemoteMemInfo>(memory, &memory->remote_mem_obj));
  if (src == local_mh->registered_addr && size == local_mh->registered_len)
    req->compress_ctx = local_mh->compress_ctx;
  req->compress_only = local_mh->compress_only;
  req->to_peer_id = conn->uccl_conn_id_.peer_id;
  req->send_type =
      (ureq->type == ReqType::ReqRead) ? SendType::Read : SendType::Write;
  req->need_signaled = true;

  int64_t wr_id = -1;
  while (wr_id < 0) {
    wr_id = send_group->post_write_or_read(req);
    if (wr_id == SendConnection::kPostError) return SendConnection::kPostError;
    if (wr_id < 0) std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
  ureq->engine_idx = static_cast<int>(wr_id);
  ureq->peer_id = conn->uccl_conn_id_.peer_id;
  return ureq->engine_idx;
}
