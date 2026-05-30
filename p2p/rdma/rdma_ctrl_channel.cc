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
  (void)mem_block;
}

SendControlChannel::SendControlChannel(std::shared_ptr<RdmaContext> ctx,
                                       ChannelMetaData const& remote_meta,
                                       std::shared_ptr<RegMemBlock> mem_block,
                                       uint32_t channel_id)
    : RDMADataChannel(ctx, remote_meta, channel_id) {
  (void)mem_block;
}

void SendControlChannel::bind_write_meta_ring(
    std::shared_ptr<RegMemBlock> local_mirror,
    RemoteMemInfo const& remote_ring) {
  write_meta_local_ = local_mirror;
  write_meta_remote_ =
      std::make_shared<RemoteMemInfo>(remote_ring.addr, remote_ring.length,
                                      remote_ring.rkey_array, remote_ring.type);
}

bool SendControlChannel::push_write_meta(WriteReqMeta const& meta,
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

bool SendControlChannel::noblocking_poll() {
  std::vector<CQMeta> cq_datas;
  if (RDMADataChannel::poll_once(cq_datas)) {
    for (auto const& cq_data : cq_datas) {
      UCCL_LOG(INFO, UCCL_RDMA)
          << "SendControlChannel::noblocking_poll - Polled completion: "
          << cq_data;
    }
    return true;
  }
  return false;
}

RecvControlChannel::RecvControlChannel(std::shared_ptr<RdmaContext> ctx,
                                       std::shared_ptr<RegMemBlock> mem_block,
                                       uint32_t channel_id)
    : RDMADataChannel(ctx, channel_id) {
  (void)mem_block;
}

RecvControlChannel::RecvControlChannel(std::shared_ptr<RdmaContext> ctx,
                                       MetaInfoToExchange const& remote_meta,
                                       std::shared_ptr<RegMemBlock> mem_block,
                                       uint32_t channel_id)
    : RDMADataChannel(ctx, remote_meta.channel_meta, channel_id) {
  (void)mem_block;
}

bool RecvControlChannel::noblocking_poll() {
  std::vector<CQMeta> cq_datas;
  if (!RDMADataChannel::poll_once(cq_datas)) return false;
  for (auto const& cq_data : cq_datas) {
    UCCL_LOG(INFO, UCCL_RDMA)
        << "RecvControlChannel::noblocking_poll - CQE: " << cq_data
        << ", is_write_meta=" << cq_data.imm.is_write_meta()
        << ", local_ring=" << (write_meta_local_ ? "set" : "NULL");
    if (cq_data.has_imm() && cq_data.imm.is_write_meta() && write_meta_local_) {
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

void RecvControlChannel::bind_write_meta_ring(
    std::shared_ptr<RegMemBlock> local_ring) {
  write_meta_local_ = local_ring;
}

std::vector<WriteReqMeta> RecvControlChannel::drain_pending_write_metas() {
  std::vector<WriteReqMeta> out;
  out.swap(pending_write_metas_);
  return out;
}
