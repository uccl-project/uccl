#pragma once
#include "define.h"
#include "ring_spsc.h"
#include "efa_channel.h"

class SendControlChannel : public EFAChannel {
 public:
  explicit SendControlChannel(std::shared_ptr<RdmaContext> ctx)
      : EFAChannel(ctx) {}

  explicit SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                               ChannelMetaData const& remote_meta)
      : EFAChannel(ctx, remote_meta) {}

  explicit SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                               std::shared_ptr<RegMemBlock> mem_block)
      : EFAChannel(ctx) {
    rb_ = std::make_unique<RingBuffer<SendReqMetaOnRing, RING_CAPACITY>>(mem_block);
  }

  explicit SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                               ChannelMetaData const& remote_meta,
                               std::shared_ptr<RegMemBlock> mem_block)
      : EFAChannel(ctx, remote_meta) {
    rb_ = std::make_unique<RingBuffer<SendReqMetaOnRing, RING_CAPACITY>>(mem_block);
  }

  void connect(ChannelMetaData const& remote_meta, std::shared_ptr<RegMemBlock> mem_block) {
    // Initialize rb_ with the shared_ptr<RegMemBlock>
    rb_ = std::make_unique<RingBuffer<SendReqMetaOnRing, RING_CAPACITY>>(mem_block);
    EFAChannel::connect(remote_meta);
  }

  // not thread safe
  bool getOneSendRequest(std::shared_ptr<EFASendRequest>& req) {
    // Pop from rb_ and generate req, return false if empty
    SendReqMeta meta;
    if (!rb_->pop_with_convert(meta,from_ring_meta)) {
      return false;
    }

    // Create RemoteMemInfo from the popped meta
    auto remote_mem = std::make_shared<RemoteMemInfo>(meta.remote_mem);
    // req should already have local_mem set, just update remote_mem
    req->remote_mem = remote_mem;
    return true;
  }

  bool noblockingPoll() {
    CQMeta cq_data;
    if (EFAChannel::poll_once(cq_data)) {
      if(cq_data.hasIMM()){
        rb_->modify_and_advance_write(cq_data.imm, check_in_progress, set_in_progress);
      }
      return true;
    }
    return false;
  }

 private:
  std::unique_ptr<RingBuffer<SendReqMetaOnRing, RING_CAPACITY>> rb_;
};


class RecvControlChannel : public EFAChannel {
 public:
  explicit RecvControlChannel(std::shared_ptr<RdmaContext> ctx)
      : EFAChannel(ctx) {}

  explicit RecvControlChannel(std::shared_ptr<RdmaContext> ctx,
                               ChannelMetaData const& remote_meta)
      : EFAChannel(ctx, remote_meta) {}

  explicit RecvControlChannel(std::shared_ptr<RdmaContext> ctx,
                               std::shared_ptr<RegMemBlock> mem_block)
      : EFAChannel(ctx) {
    local_info_ = mem_block;
    rb_ = std::make_unique<RingBuffer<SendReqMetaOnRing, RING_CAPACITY>>(local_info_);
  }

  explicit RecvControlChannel(std::shared_ptr<RdmaContext> ctx,
                               MetaInfoToExchange const& remote_meta,
                               std::shared_ptr<RegMemBlock> mem_block)
      : EFAChannel(ctx, remote_meta.channel_meta) {
    local_info_ = mem_block;
    rb_ = std::make_unique<RingBuffer<SendReqMetaOnRing, RING_CAPACITY>>(local_info_);
    remote_info_ = std::make_unique<RemoteMemInfo>(remote_meta.mem_meta);
    empty_rb_ = std::make_unique<EmptyRingBuffer<SendReqMetaOnRing, RING_CAPACITY>>(reinterpret_cast<void*>(remote_meta.mem_meta.addr));
  }

  void connect(MetaInfoToExchange const& remote_meta) {
    // Initialize remote info and empty ring buffer (assumes local_info_ and rb_ already initialized)
    remote_info_ = std::make_unique<RemoteMemInfo>(remote_meta.mem_meta);
    empty_rb_ = std::make_unique<EmptyRingBuffer<SendReqMetaOnRing, RING_CAPACITY>>(reinterpret_cast<void*>(remote_meta.mem_meta.addr));
    EFAChannel::connect(remote_meta.channel_meta);
  }

  int postSendReq(std::shared_ptr<EFARecvRequest> rev_req){
    SendReqMeta req_meta;
    req_meta.rank_id = rev_req->from_rank_id;
    req_meta.channel_id = rev_req->channel_id;
    req_meta.remote_mem = rev_req->local_mem;
    int index = rb_->push_with_convert(req_meta, to_ring_meta);
    if(index < 0){
      return index;
    }
    std::shared_ptr<RemoteMemInfo> remote_mem_ptr = std::make_shared<RemoteMemInfo>(empty_rb_->getElementAddress(index),
    remote_info_->rkey, empty_rb_->sizeInBytes(),MemoryType::HOST);
    std::shared_ptr<RegMemBlock> local_mem_ptr = std::make_shared<RegMemBlock>(reinterpret_cast<void*>(rb_->getElementAddress(index)),
    rb_->elementSize(),MemoryType::HOST,local_info_->mr,false);
    std::shared_ptr<EFASendRequest> send_ptr = std::make_shared<EFASendRequest>(local_mem_ptr,remote_mem_ptr,index);
    if(EFAChannel::send(send_ptr)){
      return true;
    }

    std::cout<<"error!!!!!!!!!!!!"<<std::endl;
    return index;
  }

  void recv_done(uint64_t index){
    rb_->modify_at(index, set_is_done);
    rb_->remove_while(check_is_done);
  }

  bool check_done(uint64_t index){
    return rb_->check_at(index,check_is_done);
  }

 private:
  std::unique_ptr<EmptyRingBuffer<SendReqMetaOnRing, RING_CAPACITY>> empty_rb_;
  std::unique_ptr<RemoteMemInfo> remote_info_;
  std::shared_ptr<RegMemBlock> local_info_;
  std::unique_ptr<RingBuffer<SendReqMetaOnRing, RING_CAPACITY>> rb_;

};
