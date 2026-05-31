#pragma once
#include "common.h"
#include "rdma_context.h"
#include "rdma_data_channel_impl.h"
#include "seq_num.h"
#include "transport_type.h"
#include "util/debug.h"
#include "util/util.h"
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

// When set on a thread, RDMADataChannel::submit_request() accumulates send
// requests in a per-channel batch instead of immediately posting them. The
// thread must subsequently call flush_batch() (typically via
// RDMAEndpoint::flush_all_sends()) to flush all pending work requests in a
// single doorbell per channel.
inline thread_local bool g_uccl_batch_post = false;

class RDMADataChannel {
 public:
  struct RawSendRequest {
    SendType send_type = SendType::Write;
    uint64_t wr_id = 0;
    uint64_t local_addr = 0;
    uint32_t local_len = 0;
    uint32_t local_key = 0;
    uint64_t remote_addr = 0;
    uint32_t remote_key = 0;
    ImmData imm_data = 0;
  };

  explicit RDMADataChannel(std::shared_ptr<RdmaContext> ctx,
                           uint32_t channel_id = 0);
  explicit RDMADataChannel(std::shared_ptr<RdmaContext> ctx,
                           ChannelMetaData const& remote_meta,
                           uint32_t channel_id = 0);

  RDMADataChannel(RDMADataChannel const&) = delete;
  RDMADataChannel& operator=(RDMADataChannel const&) = delete;

  ~RDMADataChannel();

  void establish_channel(ChannelMetaData const& remote_meta);
  int64_t submit_request(std::shared_ptr<RDMASendRequest> req);

  // Flush all accumulated batched send requests in one doorbell. Safe to call
  // when nothing is batched (returns 0).
  int flush_batch();
  int post_raw_batch(std::vector<RawSendRequest> const& batch);

  // Given a CQE wr_id, return all wr_ids that should be acknowledged in the
  // tracker. With selective signaling, a single signaled CQE represents
  // completion of the WR with that wr_id plus all earlier unsignaled WRs
  // posted as part of the same batched flush. For CQEs that do not match
  // any pending batch (i.e. immediate-path WRs that signaled themselves),
  // only the original wr_id is returned.
  void expand_completion(uint64_t cqe_wr_id, std::vector<uint64_t>& acks);

  int64_t read(std::shared_ptr<RDMASendRequest> req);
  int64_t send(std::shared_ptr<RDMASendRequest> req);
  bool poll_once(std::vector<CQMeta>& cq_datas);

  // Post an 8-byte RDMA WRITE ack to the sender's ack_ring.
  // Uses a pre-registered staging slot (bnxt_re rejects IBV_SEND_INLINE).
  int post_ack_write(uint64_t remote_addr, uint32_t remote_rkey,
                     uint32_t ack_slot, uint64_t value);

  // Get local metadata
  std::shared_ptr<ChannelMetaData> get_local_meta() const;

  // Get remote metadata
  std::shared_ptr<ChannelMetaData> get_remote_meta() const;

  // Get RdmaContext
  std::shared_ptr<RdmaContext> const get_context() const;
  uint64_t const get_context_id() const;
  uint32_t get_channel_id() const;

 private:
  static constexpr uint64_t kAckSignalMarker = 0xACCAFEFEull;
  static constexpr uint32_t kAckSigEvery = 64;

  std::shared_ptr<RdmaContext> ctx_;
  uint32_t channel_id_;

  struct ibv_cq_ex* cq_ex_;
  struct ibv_qp* qp_;
  struct ibv_ah* ah_;

  std::shared_ptr<ChannelMetaData> local_meta_;
  std::shared_ptr<ChannelMetaData> remote_meta_;

  std::shared_ptr<AtomicBitmapPacketTracker> tracker_;
  std::unique_ptr<RDMADataChannelImpl> impl_;

  // Per-channel buffer for batched doorbell posting. Protected by batch_mu_.
  std::vector<std::shared_ptr<RDMASendRequest>> batch_;
  size_t batch_bytes_ = 0;
  std::mutex batch_mu_;

  // Verbs posting on a QP must be single-producer. Serialize direct posts,
  // deferred flushes, raw vector batches, and ack writes on this channel.
  std::mutex post_mu_;

  // Selective-signaling state: each batched flush emits one signaled CQE.
  // When that CQE arrives, all preceding unsignaled WRs in the same batch
  // are guaranteed (by the RC ordering rules) to have completed and can be
  // acknowledged together. Each entry corresponds to one batched flush, in
  // post order on this channel's QP.
  struct PendingSignalGroup {
    uint64_t signal_wr_id;
    std::vector<uint64_t> unsignaled_wr_ids;
  };
  std::deque<PendingSignalGroup> pending_groups_;
  std::mutex pending_mu_;

  uint32_t ack_unsig_count_ = 0;  // selective-signal counter for post_ack_write
  std::mutex ack_mu_;             // serializes ibv_post_send for ack WRs

  std::vector<uint64_t> ack_staging_;  // staging buffer for post_ack_write
  struct ibv_mr* ack_staging_mr_ = nullptr;

  struct ibv_cq_ex* get_cq() const;
  struct ibv_qp* get_qp() const;

  // Post send request based on send_type
  // Returns 0 on success, error code on failure
  int __post_request_ex(std::shared_ptr<RDMASendRequest> req);
  int __post_request(std::shared_ptr<RDMASendRequest> req);
  int post_request(std::shared_ptr<RDMASendRequest> req);

  // Post all requests in `batch` using a single ibv_wr_start/ibv_wr_complete
  // pair (one doorbell). IB uses selective signaling; EFA SRD requires each
  // WR to be signaled even when doorbell-batched.
  int __flush_batch_ex(
      std::vector<std::shared_ptr<RDMASendRequest>> const& batch);

  // Legacy verbs path: build a linked ibv_send_wr list and post once. Only
  // the last WR is IBV_SEND_SIGNALED.
  int __flush_batch_legacy(
      std::vector<std::shared_ptr<RDMASendRequest>> const& batch);

  int __post_raw_batch_ex(std::vector<RawSendRequest> const& batch);
  int __post_raw_batch_legacy(std::vector<RawSendRequest> const& batch);

  void init_qp();

  // Prepare SGE list for send request
  // Returns the number of SGE entries filled
  int prepare_sge_list(struct ibv_sge* sge,
                       std::shared_ptr<RDMASendRequest> req);
};
