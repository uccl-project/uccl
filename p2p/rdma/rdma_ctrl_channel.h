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
  void bind_write_meta_ring(std::shared_ptr<RegMemBlock> local_mirror,
                            RemoteMemInfo const& remote_ring);

  // Push one WriteReqMeta to the receiver. Caller is responsible for choosing
  // a slot index that is not currently in flight (monotonic counter mod
  // kWriteMetaRingCapacity is sufficient because depth ≥ max in-flight).
  // Slot is filled locally + RDMA WRITE WITH IMM (with kWriteMetaBit) lands
  // the same 64 bytes on the receiver, triggering its IMM CQE handler.
  bool push_write_meta(WriteReqMeta const& meta, uint32_t slot);

  int get_one_send_request_meta(SendReqMeta& meta);

  inline bool has_send_request() { return !rb_->empty(); }

  // not thread safe
  bool get_one_send_request(std::shared_ptr<RDMASendRequest>& req);

  bool noblocking_poll();

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

  int post_send_req(std::shared_ptr<RDMARecvRequest> rev_req);

  std::shared_ptr<SendReqMeta> recv_done(uint64_t index);

  bool noblocking_poll();

  // Adopt the local WriteReqMeta ring (HOST memory, registered for RDMA
  // WRITE access from the peer sender).
  void bind_write_meta_ring(std::shared_ptr<RegMemBlock> local_ring);

  // Hand off accumulated WriteReqMeta entries since the last call. Owned by
  // RecvConnection, which drives decompress + ack.
  std::vector<WriteReqMeta> drain_pending_write_metas();

  bool check_done(uint64_t index);

  // Called by RecvConnection::start_polling/stop_polling. While true,
  // post_send_req()'s retry skips noblocking_poll() to avoid concurrent
  // lazy_post_recv_wrs_n() on the non-atomic pending_post_recv_.
  void set_has_concurrent_poller(bool v) {
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
