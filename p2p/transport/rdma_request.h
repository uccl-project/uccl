#pragma once
#include "engine.h"
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
  bundle->req.compress_only = local_mh->compress_only;
  bundle->req.to_peer_id = conn->uccl_conn_id_.peer_id;
  bundle->req.send_type =
      (ureq->type == ReqType::ReqRead) ? SendType::Read : SendType::Write;
  bundle->req.need_signaled = true;

  auto req = std::shared_ptr<RDMASendRequest>(bundle, &bundle->req);

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
  bundle->req.compress_only = local_mh->compress_only;
  bundle->req.to_peer_id = conn->uccl_conn_id_.peer_id;
  bundle->req.send_type =
      (ureq->type == ReqType::ReqRead) ? SendType::Read : SendType::Write;
  bundle->req.need_signaled = true;

  auto req = std::shared_ptr<RDMASendRequest>(bundle, &bundle->req);

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
