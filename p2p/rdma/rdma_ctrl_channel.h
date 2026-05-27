#pragma once
#include "common.h"
#include "rdma_data_channel.h"
#include "ring_spsc.h"
#include "util/debug.h"

class SendControlChannel : public RDMADataChannel {
 public:
  explicit SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                              uint32_t channel_id = 0);

  explicit SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                              ChannelMetaData const& remote_meta,
                              uint32_t channel_id = 0);

  explicit SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                              std::shared_ptr<RegMemBlock> mem_block,
                              uint32_t channel_id = 0);

  explicit SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                              ChannelMetaData const& remote_meta,
                              std::shared_ptr<RegMemBlock> mem_block,
                              uint32_t channel_id = 0);

  // Bind the WriteReqMeta ring: local mirror (sender-owned, registered HOST
  // memory) + remote slot table on the receiver. Called once during the
  // Control-channel handshake from the peer's exchanged metadata.
  void bindWriteMetaRing(std::shared_ptr<RegMemBlock> local_mirror,
                         RemoteMemInfo const& remote_ring);

  // Push one WriteReqMeta to the receiver. Caller is responsible for choosing
  // a slot index that is not currently in flight (monotonic counter mod
  // kWriteMetaRingCapacity is sufficient because depth ≥ max in-flight).
  // Slot is filled locally + RDMA WRITE WITH IMM (with kWriteMetaBit) lands
  // the same 64 bytes on the receiver, triggering its IMM CQE handler.
  bool pushWriteMeta(WriteReqMeta const& meta, uint32_t slot);

  int getOneSendRequestMeta(SendReqMeta& meta);

  inline bool hasSendRequest() { return !rb_->empty(); }

  // not thread safe
  bool getOneSendRequest(std::shared_ptr<RDMASendRequest>& req);

  bool noblockingPoll();

 private:
  std::unique_ptr<RingBuffer<SendReqMetaOnRing, kRingCapacity>> rb_;
  std::shared_ptr<RegMemBlock> write_meta_local_;
  std::shared_ptr<RemoteMemInfo> write_meta_remote_;
};

class RecvControlChannel : public RDMADataChannel {
 public:
  explicit RecvControlChannel(std::shared_ptr<RdmaContext> ctx,
                              std::shared_ptr<RegMemBlock> mem_block,
                              uint32_t channel_id = 0);

  explicit RecvControlChannel(std::shared_ptr<RdmaContext> ctx,
                              MetaInfoToExchange const& remote_meta,
                              std::shared_ptr<RegMemBlock> mem_block,
                              uint32_t channel_id = 0);

  int postSendReq(std::shared_ptr<RDMARecvRequest> rev_req);

  std::shared_ptr<SendReqMeta> recv_done(uint64_t index);

  bool noblockingPoll();

  // Adopt the local WriteReqMeta ring (HOST memory, registered for RDMA
  // WRITE access from the peer sender).
  void bindWriteMetaRing(std::shared_ptr<RegMemBlock> local_ring);

  // Hand off accumulated WriteReqMeta entries since the last call. Owned by
  // RecvConnection, which drives decompress + ack.
  std::vector<WriteReqMeta> drainPendingWriteMetas();

  bool check_done(uint64_t index);

  // Called by RecvConnection::startPolling/stopPolling. While true,
  // postSendReq()'s retry skips noblockingPoll() to avoid concurrent
  // lazyPostRecvWrsN() on the non-atomic pending_post_recv_.
  void setHasConcurrentPoller(bool v) {
    has_concurrent_poller_.store(v, std::memory_order_release);
  }

 private:
  std::atomic<bool> has_concurrent_poller_{false};
  std::unique_ptr<EmptyRingBuffer<SendReqMetaOnRing, kRingCapacity>> empty_rb_;
  std::unique_ptr<RemoteMemInfo> remote_info_;
  std::shared_ptr<RegMemBlock> local_info_;
  std::unique_ptr<RingBuffer<SendReqMetaOnRing, kRingCapacity>> rb_;
  std::shared_ptr<RemoteMemInfo> remote_mem_ptr_;
  std::shared_ptr<RegMemBlock> local_mem_ptr_;
  std::shared_ptr<RegMemBlock> write_meta_local_;
  std::vector<WriteReqMeta> pending_write_metas_;
};
