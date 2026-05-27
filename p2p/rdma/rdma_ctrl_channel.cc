#include "rdma_ctrl_channel.h"
#include "util/debug.h"
#include "util/util.h"

SendControlChannel::SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                                       uint32_t channel_id)
    : RDMADataChannel(ctx, channel_id) {}

SendControlChannel::SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                                       ChannelMetaData const& remote_meta,
                                       uint32_t channel_id)
    : RDMADataChannel(ctx, remote_meta, channel_id) {}

SendControlChannel::SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                                       std::shared_ptr<RegMemBlock> mem_block,
                                       uint32_t channel_id)
    : RDMADataChannel(ctx, channel_id) {
  rb_ =
      std::make_unique<RingBuffer<SendReqMetaOnRing, kRingCapacity>>(mem_block);
}

SendControlChannel::SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                                       ChannelMetaData const& remote_meta,
                                       std::shared_ptr<RegMemBlock> mem_block,
                                       uint32_t channel_id)
    : RDMADataChannel(ctx, remote_meta, channel_id) {
  rb_ =
      std::make_unique<RingBuffer<SendReqMetaOnRing, kRingCapacity>>(mem_block);
}

void SendControlChannel::bindWriteMetaRing(
    std::shared_ptr<RegMemBlock> local_mirror,
    RemoteMemInfo const& remote_ring) {
  write_meta_local_ = local_mirror;
  write_meta_remote_ =
      std::make_shared<RemoteMemInfo>(remote_ring.addr, remote_ring.length,
                                      remote_ring.rkey_array, remote_ring.type);
}

bool SendControlChannel::pushWriteMeta(WriteReqMeta const& meta,
                                       uint32_t slot) {
  if (unlikely(!write_meta_local_ || !write_meta_remote_)) return false;
  auto* local = static_cast<WriteReqMeta*>(write_meta_local_->addr) + slot;
  *local = meta;

  auto src = std::make_shared<RegMemBlock>(local, sizeof(WriteReqMeta),
                                           write_meta_local_->mr_array,
                                           write_meta_local_->type);
  auto dst = std::make_shared<RemoteMemInfo>(
      write_meta_remote_->addr + slot * sizeof(WriteReqMeta),
      sizeof(WriteReqMeta), write_meta_remote_->rkey_array,
      write_meta_remote_->type);

  ImmData imm;
  imm.set_chunk_count(1);
  imm.set_index(static_cast<uint16_t>(slot));
  imm.set_write_meta();

  auto req = std::make_shared<RDMASendRequest>(src, dst, imm.raw());
  req->channel_id = kControlChannelID;
  req->wr_id = static_cast<int64_t>(slot);
  req->send_type = SendType::Send;  // -> RDMA_WRITE_WITH_IMM
  return RDMADataChannel::send(req) >= 0;
}

int SendControlChannel::getOneSendRequestMeta(SendReqMeta& meta) {
  // Pop from rb_ and generate req, return false if empty
  return rb_->pop_with_convert(meta, from_ring_meta);
}

bool SendControlChannel::getOneSendRequest(
    std::shared_ptr<RDMASendRequest>& req) {
  // Pop from rb_ and generate req, return false if empty
  SendReqMeta meta;
  int index = getOneSendRequestMeta(meta);
  if (index < 0) {
    UCCL_LOG(INFO, UCCL_RDMA)
        << "getOneSendRequest - Ring buffer is empty, cannot pop";
    return false;
  }

  // Create RemoteMemInfo from the popped meta
  auto remote_mem = std::make_shared<RemoteMemInfo>(meta.remote_mem);
  // req should already have local_mem set, just update remote_mem
  req->remote_mem = remote_mem;
  req->channel_id = meta.channel_id;
  req->imm_data.set_index(index);

  // Log the received request with all information
  UCCL_LOG(INFO, UCCL_RDMA) << "getOneSendRequest - Received request: " << *req;
  UCCL_LOG(INFO, UCCL_RDMA) << "  SendReqMeta from ring buffer: " << meta;

  return true;
}

bool SendControlChannel::noblockingPoll() {
  std::vector<CQMeta> cq_datas;
  if (RDMADataChannel::pollOnce(cq_datas)) {
    for (auto const& cq_data : cq_datas) {
      UCCL_LOG(INFO, UCCL_RDMA)
          << "SendControlChannel::noblockingPoll - Polled completion: "
          << cq_data;
      if (cq_data.hasIMM()) {
        rb_->modify_and_advance_write(cq_data.imm.index(), check_in_progress,
                                      set_in_progress);
      }
    }
    return true;
  }
  return false;
}

RecvControlChannel::RecvControlChannel(std::shared_ptr<RdmaContext> ctx,
                                       std::shared_ptr<RegMemBlock> mem_block,
                                       uint32_t channel_id)
    : RDMADataChannel(ctx, channel_id) {
  local_info_ = mem_block;
  rb_ = std::make_unique<RingBuffer<SendReqMetaOnRing, kRingCapacity>>(
      local_info_);
}

RecvControlChannel::RecvControlChannel(std::shared_ptr<RdmaContext> ctx,
                                       MetaInfoToExchange const& remote_meta,
                                       std::shared_ptr<RegMemBlock> mem_block,
                                       uint32_t channel_id)
    : RDMADataChannel(ctx, remote_meta.channel_meta, channel_id) {
  local_info_ = mem_block;
  rb_ = std::make_unique<RingBuffer<SendReqMetaOnRing, kRingCapacity>>(
      local_info_);
  remote_info_ = std::make_unique<RemoteMemInfo>(remote_meta.mem_meta);
  empty_rb_ =
      std::make_unique<EmptyRingBuffer<SendReqMetaOnRing, kRingCapacity>>(
          reinterpret_cast<void*>(remote_meta.mem_meta.addr));
}

int RecvControlChannel::postSendReq(std::shared_ptr<RDMARecvRequest> rev_req) {
  SendReqMeta req_meta(rev_req);
  UCCL_LOG(INFO, UCCL_RDMA)
      << "postSendReq - Created SendReqMeta: " << req_meta;

  int index = rb_->push_with_convert(req_meta, to_ring_meta);
  if (index < 0) {
    UCCL_LOG(INFO, UCCL_RDMA)
        << "postSendReq - Failed to push to ring buffer, index: " << index;
    return index;
  }

  UCCL_LOG(INFO, UCCL_RDMA)
      << "postSendReq - Successfully pushed to ring buffer at index: " << index;
  if (!remote_mem_ptr_) {
    remote_mem_ptr_ = std::make_shared<RemoteMemInfo>(
        empty_rb_->getElementAddress(index), empty_rb_->sizeInBytes(),
        remote_info_->rkey_array, MemoryType::HOST);
  } else {
    remote_mem_ptr_->addr = empty_rb_->getElementAddress(index);
    remote_mem_ptr_->length = empty_rb_->sizeInBytes();
  }
  if (!local_mem_ptr_) {
    local_mem_ptr_ = std::make_shared<RegMemBlock>(
        reinterpret_cast<void*>(rb_->getElementAddress(index)),
        rb_->elementSize(), local_info_->mr_array, MemoryType::HOST);
  } else {
    local_mem_ptr_->addr =
        reinterpret_cast<void*>(rb_->getElementAddress(index));
    local_mem_ptr_->size = rb_->elementSize();
  }

  std::shared_ptr<RDMASendRequest> send_ptr =
      std::make_shared<RDMASendRequest>(local_mem_ptr_, remote_mem_ptr_, index);
  send_ptr->channel_id = kControlChannelID;
  send_ptr->wr_id = index;
  while (RDMADataChannel::send(send_ptr) < 0) {
    if (!has_concurrent_poller_.load(std::memory_order_acquire)) {
      noblockingPoll();
    }
    std::this_thread::yield();
  }
  return index;
}

std::shared_ptr<SendReqMeta> RecvControlChannel::recv_done(uint64_t index) {
  // Increment the received chunk count
  rb_->modify_at(index, increment_received_chunk);

  // Check if all chunks have been received
  if (rb_->check_at(index, check_all_chunks_received)) {
    // All chunks received, mark as done and remove completed items
    auto req = std::make_shared<SendReqMeta>(rb_->at(index).meta);
    UCCL_LOG(INFO, UCCL_RDMA)
        << "recv_done - All chunks received for index: " << index
        << ", marking as done.";

    rb_->modify_at(index, set_is_done);
    rb_->remove_while(check_is_done);
    return req;
  }
  return nullptr;
}

bool RecvControlChannel::noblockingPoll() {
  std::vector<CQMeta> cq_datas;
  if (!RDMADataChannel::pollOnce(cq_datas)) return false;
  for (auto const& cq_data : cq_datas) {
    UCCL_LOG(INFO, UCCL_RDMA)
        << "RecvControlChannel::noblockingPoll - CQE: " << cq_data
        << ", is_write_meta=" << cq_data.imm.is_write_meta()
        << ", local_ring=" << (write_meta_local_ ? "set" : "NULL");
    if (cq_data.hasIMM() && cq_data.imm.is_write_meta() && write_meta_local_) {
      uint16_t slot = cq_data.imm.plain_index();
      auto* entry = static_cast<WriteReqMeta*>(write_meta_local_->addr) + slot;
      UCCL_LOG(INFO, UCCL_RDMA)
          << "RecvControlChannel: queued WriteReqMeta slot=" << slot
          << " wr_id=" << entry->wr_id
          << " decompress_offset=" << entry->decompress_offset
          << " compressed_size=" << entry->compressed_size;
      pending_write_metas_.push_back(*entry);
    }
  }
  return true;
}

void RecvControlChannel::bindWriteMetaRing(
    std::shared_ptr<RegMemBlock> local_ring) {
  write_meta_local_ = local_ring;
}

std::vector<WriteReqMeta> RecvControlChannel::drainPendingWriteMetas() {
  std::vector<WriteReqMeta> out;
  out.swap(pending_write_metas_);
  return out;
}

bool RecvControlChannel::check_done(uint64_t index) {
  return rb_->check_at(index, check_is_done);
}
