#pragma once
#include "define.h"
#include "efa_channel.h"
#include "ring_spsc.h"

class SendControlChannel : public EFAChannel {
 public:
  explicit SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                              uint32_t channel_id = 0)
      : EFAChannel(ctx, channel_id) {}

  explicit SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                              ChannelMetaData const& remote_meta,
                              uint32_t channel_id = 0)
      : EFAChannel(ctx, remote_meta, channel_id) {}

  explicit SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                              std::shared_ptr<RegMemBlock> mem_block,
                              uint32_t channel_id = 0)
      : EFAChannel(ctx, channel_id) {
    rb_ = std::make_unique<RingBuffer<SendReqMetaOnRing, kRingCapacity>>(
        mem_block);
  }

  explicit SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                              ChannelMetaData const& remote_meta,
                              std::shared_ptr<RegMemBlock> mem_block,
                              uint32_t channel_id = 0)
      : EFAChannel(ctx, remote_meta, channel_id) {
    rb_ = std::make_unique<RingBuffer<SendReqMetaOnRing, kRingCapacity>>(
        mem_block);
  }

  void connect(ChannelMetaData const& remote_meta,
               std::shared_ptr<RegMemBlock> mem_block) {
    // Initialize rb_ with the shared_ptr<RegMemBlock>
    rb_ = std::make_unique<RingBuffer<SendReqMetaOnRing, kRingCapacity>>(
        mem_block);
    EFAChannel::connect(remote_meta);
  }

  int getOneSendRequestMeta(SendReqMeta& meta) {
    // Pop from rb_ and generate req, return false if empty
    return rb_->pop_with_convert(meta, from_ring_meta);
  }

  inline bool hasSendRequest() { return !rb_->empty(); }
  // not thread safe
  bool getOneSendRequest(std::shared_ptr<EFASendRequest>& req) {
    // Pop from rb_ and generate req, return false if empty
    SendReqMeta meta;
    int index = getOneSendRequestMeta(meta);
    if (index < 0) {
      LOG(INFO) << "getOneSendRequest - Ring buffer is empty, cannot pop";
      return false;
    }

    // Create RemoteMemInfo from the popped meta
    auto remote_mem = std::make_shared<RemoteMemInfo>(meta.remote_mem);
    // req should already have local_mem set, just update remote_mem
    req->remote_mem = remote_mem;
    req->channel_id = meta.channel_id;
    req->imm_data = index;

    // Log the received request with all information
    LOG(INFO) << "getOneSendRequest - Received request: " << *req;
    LOG(INFO) << "  SendReqMeta from ring buffer: " << meta;

    return true;
  }

  bool noblockingPoll() {
    std::vector<CQMeta> cq_datas;
    if (EFAChannel::poll_once(cq_datas)) {
      for (auto const& cq_data : cq_datas) {
        LOG(INFO) << "SendControlChannel::noblockingPoll - Polled completion: "
                  << cq_data;
        if (cq_data.hasIMM()) {
          rb_->modify_and_advance_write(cq_data.imm, check_in_progress,
                                        set_in_progress);
        }
      }
      return true;
    }
    return false;
  }

 private:
  std::unique_ptr<RingBuffer<SendReqMetaOnRing, kRingCapacity>> rb_;
};

class RecvControlChannel : public EFAChannel {
 public:
  explicit RecvControlChannel(std::shared_ptr<RdmaContext> ctx,
                              uint32_t channel_id = 0)
      : EFAChannel(ctx, channel_id) {}

  explicit RecvControlChannel(std::shared_ptr<RdmaContext> ctx,
                              ChannelMetaData const& remote_meta,
                              uint32_t channel_id = 0)
      : EFAChannel(ctx, remote_meta, channel_id) {}

  explicit RecvControlChannel(std::shared_ptr<RdmaContext> ctx,
                              std::shared_ptr<RegMemBlock> mem_block,
                              uint32_t channel_id = 0)
      : EFAChannel(ctx, channel_id) {
    local_info_ = mem_block;
    rb_ = std::make_unique<RingBuffer<SendReqMetaOnRing, kRingCapacity>>(
        local_info_);
  }

  explicit RecvControlChannel(std::shared_ptr<RdmaContext> ctx,
                              MetaInfoToExchange const& remote_meta,
                              std::shared_ptr<RegMemBlock> mem_block,
                              uint32_t channel_id = 0)
      : EFAChannel(ctx, remote_meta.channel_meta, channel_id) {
    local_info_ = mem_block;
    rb_ = std::make_unique<RingBuffer<SendReqMetaOnRing, kRingCapacity>>(
        local_info_);
    remote_info_ = std::make_unique<RemoteMemInfo>(remote_meta.mem_meta);
    empty_rb_ =
        std::make_unique<EmptyRingBuffer<SendReqMetaOnRing, kRingCapacity>>(
            reinterpret_cast<void*>(remote_meta.mem_meta.addr));
  }

  void connect(MetaInfoToExchange const& remote_meta) {
    // Initialize remote info and empty ring buffer (assumes local_info_ and rb_
    // already initialized)
    remote_info_ = std::make_unique<RemoteMemInfo>(remote_meta.mem_meta);
    empty_rb_ =
        std::make_unique<EmptyRingBuffer<SendReqMetaOnRing, kRingCapacity>>(
            reinterpret_cast<void*>(remote_meta.mem_meta.addr));
    EFAChannel::connect(remote_meta.channel_meta);
  }

  int postSendReq(std::shared_ptr<EFARecvRequest> rev_req) {
    LOG(INFO) << "postSendReq - Received EFARecvRequest: " << *rev_req;

    SendReqMeta req_meta;
    req_meta.rank_id = rev_req->from_rank_id;
    req_meta.channel_id = rev_req->channel_id;
    req_meta.remote_mem = rev_req->local_mem;
    req_meta.expected_chunk_count =
        getMessageChunkCount(rev_req->local_mem->size);
    req_meta.received_chunk_count = 0;
    LOG(INFO) << "postSendReq - Created SendReqMeta: " << req_meta;

    int index = rb_->push_with_convert(req_meta, to_ring_meta);
    if (index < 0) {
      LOG(INFO) << "postSendReq - Failed to push to ring buffer, index: "
                << index;
      return index;
    }

    LOG(INFO) << "postSendReq - Successfully pushed to ring buffer at index: "
              << index;
    std::shared_ptr<RemoteMemInfo> remote_mem_ptr =
        std::make_shared<RemoteMemInfo>(
            empty_rb_->getElementAddress(index), remote_info_->rkey,
            empty_rb_->sizeInBytes(), MemoryType::HOST);
    std::shared_ptr<RegMemBlock> local_mem_ptr = std::make_shared<RegMemBlock>(
        reinterpret_cast<void*>(rb_->getElementAddress(index)),
        rb_->elementSize(), MemoryType::HOST, local_info_->mr, false);
    std::shared_ptr<EFASendRequest> send_ptr =
        std::make_shared<EFASendRequest>(local_mem_ptr, remote_mem_ptr, index);
    EFAChannel::send(send_ptr);
    return index;
  }

  void recv_done(uint64_t index) {
    // Increment the received chunk count
    rb_->modify_at(index, increment_received_chunk);

    // Check if all chunks have been received
    if (rb_->check_at(index, check_all_chunks_received)) {
      // All chunks received, mark as done and remove completed items
      rb_->modify_at(index, set_is_done);
      rb_->remove_while(check_is_done);
    }
  }
  bool noblockingPoll() {
    std::vector<CQMeta> cq_datas;
    if (EFAChannel::poll_once(cq_datas)) {
      for (auto const& cq_data : cq_datas) {
        LOG(INFO) << "RecvControlChannel::noblockingPoll - Polled completion: "
                  << cq_data;
      }
      return true;
    }
    return false;
  }
  bool check_done(uint64_t index) {
    return rb_->check_at(index, check_is_done);
  }

 private:
  std::unique_ptr<EmptyRingBuffer<SendReqMetaOnRing, kRingCapacity>> empty_rb_;
  std::unique_ptr<RemoteMemInfo> remote_info_;
  std::shared_ptr<RegMemBlock> local_info_;
  std::unique_ptr<RingBuffer<SendReqMetaOnRing, kRingCapacity>> rb_;
};
