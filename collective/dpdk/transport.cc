#include "transport.h"
#include "ether.h"
#include "ipv4.h"
#include "packet.h"
#include "udp.h"
#include "util/util.h"
#include <glog/logging.h>
#include <cstdint>
#include <set>

namespace uccl {

void TXTracking::receive_acks(uint32_t num_acked_pkts) {
  VLOG(3) << "Received " << num_acked_pkts
          << " acks :" << " num_unsent_msgbufs_ " << num_unsent_msgbufs_
          << " last_msgbuf_ " << last_msgbuf_ << " oldest_unsent_msgbuf "
          << oldest_unsent_msgbuf_ << " oldest_unacked_msgbuf_ "
          << oldest_unacked_msgbuf_;
  DCHECK_LE(num_acked_pkts, num_tracked_msgbufs_);
  while (num_acked_pkts) {
    // LOG(INFO) << "receive_acks: " << "\n\tnum_acked_pkts " << num_acked_pkts
    //           << "\n\tnum_unacked_msgbufs_ " << num_unacked_msgbufs_
    //           << "\n\tnum_unsent_msgbufs_ " << num_unsent_msgbufs_
    //           << "\n\toldest_unacked_msgbuf_ " << oldest_unacked_msgbuf_
    //           << "\n\tnewest_unacked_msgbuf_ " << newest_unacked_msgbuf_
    //           << "\n\toldest_unsent_msgbuf_ " << oldest_unsent_msgbuf_
    //           << "\n\tlast_msgbuf_ " << last_msgbuf_;

    auto msgbuf = oldest_unacked_msgbuf_;
    DCHECK(msgbuf != nullptr);

    // if (num_acked_pkts > 1) {
    //   // DCHECK_NE(msgbuf, last_msgbuf_) << "Releasing the last msgbuf!";
    //   DCHECK_NE(oldest_unacked_msgbuf_, oldest_unsent_msgbuf_)
    //       << "Releasing an unsent msgbuf!";
    //   oldest_unacked_msgbuf_ = msgbuf->next();
    //   DCHECK(oldest_unacked_msgbuf_ != nullptr) << num_acked_pkts;
    // } else {
    //   CHECK_EQ(num_tracked_msgbufs_, 1);
    //   oldest_unacked_msgbuf_ = nullptr;
    //   newest_unacked_msgbuf_ = nullptr;
    //   oldest_unsent_msgbuf_ = nullptr;
    //   last_msgbuf_ = nullptr;
    // }

    if (msgbuf->is_last()) {
      VLOG(3) << "Transmitted a complete message";
      // Tx a full message; wakeup app thread waiting on endpoint.
      DCHECK(!poll_ctxs_.empty());
      auto poll_ctx = poll_ctxs_.front();
      poll_ctxs_.pop_front();
      {
        std::lock_guard<std::mutex> lock(poll_ctx->mu);
        poll_ctx->done = true;
        poll_ctx->cv.notify_one();
      }
    }
    // Free transmitted frames that are acked
    oldest_unacked_msgbuf_ = msgbuf->next();
    PacketBuf::Release(msgbuf);

    num_unacked_msgbufs_--;
    num_tracked_msgbufs_--;
    num_acked_pkts--;

    if (oldest_unacked_msgbuf_ == nullptr) {
      DCHECK(num_acked_pkts == 0) << "num_acked_pkts " << num_acked_pkts;
      newest_unacked_msgbuf_ = nullptr;
    }

    if(num_tracked_msgbufs_ == 0) {

      DCHECK(num_acked_pkts == 0) << "num_acked_pkts " << num_acked_pkts;
      DCHECK(oldest_unacked_msgbuf_ == nullptr);
      DCHECK(newest_unacked_msgbuf_ == nullptr);

      oldest_unacked_msgbuf_ = nullptr;
      newest_unacked_msgbuf_ = nullptr;
    }

  }
}

void TXTracking::append(PacketBuf* msgbuf_head, PacketBuf* msgbuf_tail,
                        uint32_t num_frames, PollCtx* poll_ctx) {
  VLOG(3) << "Appending " << num_frames
          << " frames :" << " num_unsent_msgbufs_ " << num_unsent_msgbufs_
          << " last_msgbuf_ " << last_msgbuf_ << " oldest_unsent_msgbuf "
          << oldest_unsent_msgbuf_ << " oldest_unacked_msgbuf_ "
          << oldest_unacked_msgbuf_;

  if (poll_ctx) poll_ctxs_.push_back(poll_ctx);

  if (num_frames == 0) {
    DCHECK(msgbuf_head == nullptr);
    DCHECK(msgbuf_tail == nullptr);
    return;
  }

  // Append the message at the end of the chain of buffers, if any.
  if (last_msgbuf_ == nullptr) {
    // This is the first pending message buffer in the flow.
    DCHECK(oldest_unsent_msgbuf_ == nullptr);
    last_msgbuf_ = msgbuf_tail;
    oldest_unsent_msgbuf_ = msgbuf_head;
  } else {
    // This is not the first message buffer in the flow; let's enqueue the
    // new message buffer at the end of the chain.
    last_msgbuf_->set_next(msgbuf_head);
    // Update the last buffer pointer to point to the current buffer.
    last_msgbuf_ = msgbuf_tail;
    if (oldest_unsent_msgbuf_ == nullptr) oldest_unsent_msgbuf_ = msgbuf_head;
  }

  num_unsent_msgbufs_ += num_frames;
  num_tracked_msgbufs_ += num_frames;
}

std::optional<PacketBuf*> TXTracking::get_and_update_oldest_unsent() {
  if (num_unsent_msgbufs_)
    VLOG(3) << "Getting: num_unsent_msgbufs_ " << num_unsent_msgbufs_
            << " last_msgbuf_ " << last_msgbuf_ << " oldest_unsent_msgbuf "
            << oldest_unsent_msgbuf_ << " oldest_unacked_msgbuf_ "
            << oldest_unacked_msgbuf_;

  if (oldest_unsent_msgbuf_ == nullptr) {
    DCHECK_EQ(num_unsent_msgbufs_, 0);
    return std::nullopt;
  }

  auto msgbuf = oldest_unsent_msgbuf_;
  if (oldest_unsent_msgbuf_ != last_msgbuf_) {
    oldest_unsent_msgbuf_ = oldest_unsent_msgbuf_->next();
  } else {
    oldest_unsent_msgbuf_ = nullptr;
  }

  PacketBuf* unacked_msgbuf = PacketBuf::Allocate();

  DCHECK(unacked_msgbuf != nullptr);
  unacked_msgbuf->set_msg_flags(msgbuf->msg_flags());

  if (newest_unacked_msgbuf_ == nullptr) {
    DCHECK(oldest_unacked_msgbuf_ == nullptr);
    newest_unacked_msgbuf_ = unacked_msgbuf;
    oldest_unacked_msgbuf_ = newest_unacked_msgbuf_;
  } else {
    DCHECK(oldest_unacked_msgbuf_ != nullptr);
    DCHECK(unacked_msgbuf != nullptr);
    newest_unacked_msgbuf_->set_next(unacked_msgbuf);
    newest_unacked_msgbuf_ = unacked_msgbuf;
  }

  // if (oldest_unacked_msgbuf_ == nullptr) oldest_unacked_msgbuf_ = msgbuf;

  num_unacked_msgbufs_++;
  num_unsent_msgbufs_--;

  // LOG(INFO) << "get_and_update_oldest_unsent: num_unacked_msgbufs_ "
  //           << num_unacked_msgbufs_ << " num_unsent_msgbufs_ "
  //           << num_unsent_msgbufs_ << " oldest_unacked_msgbuf_ "
  //           << oldest_unacked_msgbuf_ << " newest_unacked_msgbuf_ "
  //           << newest_unacked_msgbuf_ << " oldest_unsent_msgbuf_ "
  //           << oldest_unsent_msgbuf_ << " last_msgbuf_ " << last_msgbuf_;

  return msgbuf;
}

RXTracking::ConsumeRet RXTracking::consume(swift::Pcb* pcb, PacketBuf* msgbuf) {
  uint8_t* pkt_addr = msgbuf->get_pkt_addr();
  auto frame_len = msgbuf->get_packet_len();
  auto const* ucclh =
      reinterpret_cast<UcclPktHdr const*>(pkt_addr + kNetHdrLen);
  // auto const* payload =
  //     reinterpret_cast<UcclPktHdr const*>(pkt_addr + kNetHdrLen +
  //     kUcclHdrLen);
  auto const seqno = ucclh->seqno.value();
  auto const expected_seqno = pcb->rcv_nxt;

  if (swift::seqno_lt(seqno, expected_seqno)) {
    VLOG(3) << "Received old packet: " << seqno << " < " << expected_seqno;
    socket_->push_packet(msgbuf->get_pkt());
    return kOldPkt;
  }

  size_t const distance = seqno - expected_seqno;
  if (distance >= kReassemblyMaxSeqnoDistance) {
    VLOG(3) << "Packet too far ahead. Dropping as we can't handle SACK. "
            << "seqno: " << seqno << ", expected: " << expected_seqno;
    socket_->push_packet(msgbuf->get_pkt());
    return kOOOUntrackable;
  }

  // Only iterate through the deque if we must, i.e., for ooo packts only
  auto it = reass_q_.begin();
  if (seqno != expected_seqno) {
    it = reass_q_.lower_bound(seqno);
    if (it != reass_q_.end() && it->first == seqno) {
      VLOG(3) << "Received duplicate packet: " << seqno;
      // Duplicate packet. Drop it.
      socket_->push_packet(msgbuf->get_pkt());
      return kOOOTrackableDup;
    }
    VLOG(3) << "Received OOO trackable packet: " << seqno
            << " payload_len: " << frame_len - kNetHdrLen - kUcclHdrLen
            << " reass_q size " << reass_q_.size();
  } else {
    VLOG(3) << "Received expected packet: " << seqno
            << " payload_len: " << frame_len - kNetHdrLen - kUcclHdrLen;
  }

  // Buffer the packet in the frame pool. It may be out-of-order.
  reass_q_.insert(it, {seqno, msgbuf});

  // Update the SACK bitmap for the newly received packet.
  pcb->sack_bitmap_bit_set(distance);

  // These frames will be freed when the message is delivered to the app.
  push_inorder_msgbuf_to_app(pcb);

  return kOOOTrackableExpectedOrInOrder;
}

void RXTracking::push_inorder_msgbuf_to_app(swift::Pcb* pcb) {
  while (!reass_q_.empty() && reass_q_.begin()->first == pcb->rcv_nxt) {
    auto* msgbuf = reass_q_.begin()->second;
    reass_q_.erase(reass_q_.begin());

    // Stash this ready message in case application threads have not
    // supplied the app buffer while the engine keeps receiving messages.
    ready_msg_queue_.push_back(msgbuf);
    try_copy_msgbuf_to_appbuf(nullptr);

    pcb->advance_rcv_nxt();
    pcb->sack_bitmap_shift_left_one();
  }
}

void RXTracking::try_copy_msgbuf_to_appbuf(Channel::Msg* rx_work) {
  if (rx_work) {
    VLOG(3) << "ready_msg_queue_ size: " << ready_msg_queue_.size()
            << " app_buf_queue_ size: " << app_buf_queue_.size();
    app_buf_queue_.push_back({*rx_work});
  }

  while (!ready_msg_queue_.empty() && !app_buf_queue_.empty()) {
    PacketBuf* ready_msg = ready_msg_queue_.front();
    ready_msg_queue_.pop_front();
    DCHECK(ready_msg) << ready_msg->print_chain();

    if (deser_msgs_head_ == nullptr) {
      deser_msgs_head_ = ready_msg;
      deser_msgs_tail_ = ready_msg;
    } else {
      deser_msgs_tail_->set_next(ready_msg);
      deser_msgs_tail_ = ready_msg;
    }

    if (ready_msg->is_last()) {
      ready_msg->set_next(nullptr);

      auto& [rx_deser_work] = app_buf_queue_.front();
      rx_deser_work.deser_msgs = deser_msgs_head_;

      // Make sure the deser thread sees the deserialized messages.
      rx_deser_work.poll_ctx->write_barrier();
      Channel::enqueue_sp(channel_->rx_deser_q_, &rx_deser_work);

      app_buf_queue_.pop_front();
      deser_msgs_head_ = nullptr;
      deser_msgs_tail_ = nullptr;
      VLOG(2) << "Received a complete message";
    }
  }
}

std::string UcclFlow::to_string() const {
  std::string s;
  s += "\n\t\t\t[CC] pcb:         " + pcb_.to_string() +
       (kCCType == CCType::kCubicPP
            ? "\n\t\t\t     cubic_pp[0]: " + cubic_pp_[0].to_string()
            : "\n\t\t\t     cubic:       " + cubic_g_.to_string()) +
       "\n\t\t\t     timely:      " + timely_g_.to_string() +
       "\n\t\t\t[TX] pending msgbufs unsent: " +
       std::to_string(tx_tracking_.num_unsent_msgbufs()) +
       "\n\t\t\t[RX] ready msgs unconsumed: " +
       std::to_string(rx_tracking_.ready_msg_queue_.size());
  return s;
}

void UcclFlow::rx_messages() {
  VLOG(3) << "Received " << pending_rx_msgbufs_.size() << " packets";

  uint32_t num_data_frames_recvd = 0;
  uint32_t path_id = 0;
  uint16_t dst_port = 0;
  uint16_t dst_port_rtt_probe = 0;
  uint64_t timestamp1 = 0, timestamp2 = 0;
  bool received_rtt_probe = false;

  for (auto msgbuf : pending_rx_msgbufs_) {
    // ebpf_transport has filtered out invalid pkts.
    auto* pkt_addr = msgbuf->get_pkt_addr();
    auto* udph =
        reinterpret_cast<udphdr*>(pkt_addr + sizeof(ethhdr) + sizeof(iphdr));
    auto* ucclh = reinterpret_cast<UcclPktHdr*>(pkt_addr + kNetHdrLen);
    auto* ucclsackh = reinterpret_cast<UcclSackHdr*>(
        reinterpret_cast<uint8_t*>(ucclh) + kUcclHdrLen);

    switch (ucclh->net_flags) {
      case UcclPktHdr::UcclFlags::kAckRttProbe:
        // Sender gets the RTT probe response, update the flow.
        process_rttprobe_rsp(ucclh->timestamp1, ucclh->timestamp2,
                             ucclsackh->timestamp3, ucclsackh->timestamp4,
                             ucclh->path_id);
        [[fallthrough]];
      case UcclPktHdr::UcclFlags::kAck:
        // ACK packet, update the flow.
        process_ack(ucclh);
        // Free the received frame.
        socket_->push_packet(msgbuf->get_pkt());
        break;
      case UcclPktHdr::UcclFlags::kDataRttProbe:
        // Receiver gets the RTT probe, relay it back in the ACK.
        // If multiple RTT probe, we take the last one's timestamp.
        received_rtt_probe = true;
        path_id = ucclh->path_id;
        dst_port_rtt_probe = htons(udph->dest);
        timestamp1 = ucclh->timestamp1;
        timestamp2 = ucclh->timestamp2;
        [[fallthrough]];
      case UcclPktHdr::UcclFlags::kData:
        // Data packet, process the payload. The frame will be freed
        // once the engine copies the payload into app buffer
        rx_tracking_.consume(&pcb_, msgbuf);
        num_data_frames_recvd++;
        // Sender's dst_port selection are symmetric.
        dst_port = htons(udph->dest);
        break;
      case UcclPktHdr::UcclFlags::kRssProbe:
        if (ucclh->engine_id == local_engine_idx_) {
          // Probe packets arrive the remote engine!
          ucclh->net_flags = UcclPktHdr::UcclFlags::kRssProbeRsp;
          ucclh->engine_id = remote_engine_idx_;
          msgbuf->mark_txpulltime_free();
          // Reverse so to send back
          reverse_packet_l2l3(msgbuf);
          socket_->send_packet(msgbuf->get_pkt());
        } else {
          socket_->push_packet(msgbuf->get_pkt());
        }
        break;
      case UcclPktHdr::UcclFlags::kRssProbeRsp:
        // RSS probing rsp packet, ignore.
        LOG_EVERY_N(INFO, 10000)
            << "[Flow] RSS probing rsp packet received, ignoring...";
        socket_->push_packet(msgbuf->get_pkt());
        break;
      default:
        CHECK(false) << "Unsupported UcclFlags: "
                     << std::bitset<8>((uint8_t)ucclh->net_flags);
    }
  }
  pending_rx_msgbufs_.clear();

  // Send one ack for a bunch of received packets.
  if (num_data_frames_recvd) {
    // Avoiding client sending too much packet which would empty msgbuf.
    if (rx_tracking_.ready_msg_queue_.size() <= kMaxReadyRxMsgbufs) {
      auto net_flags = received_rtt_probe ? UcclPktHdr::UcclFlags::kAckRttProbe
                                          : UcclPktHdr::UcclFlags::kAck;
      auto dst_port_reverse =
          received_rtt_probe ? dst_port_rtt_probe : dst_port;

      Packet* ack_pkt =
          craft_ackpacket(path_id, dst_port_reverse, pcb_.seqno(), pcb_.ackno(),
                          net_flags, timestamp1, timestamp2);
      socket_->send_packet(ack_pkt);
    }
  }

  deserialize_and_append_to_txtracking();

  // Sending data frames that can be send per cwnd.
  transmit_pending_packets();
}

void UcclFlow::tx_messages(Channel::Msg& tx_deser_work) {
  // This happens to NCCL plugin!!!
  if (tx_deser_work.len == 0) {
    std::lock_guard<std::mutex> lock(tx_deser_work.poll_ctx->mu);
    tx_deser_work.poll_ctx->done = true;
    tx_deser_work.poll_ctx->cv.notify_one();
    return;
  }

  pending_tx_msgs_.push_back({tx_deser_work, 0});

  VLOG(3) << "tx_messages size: " << tx_deser_work.len << " bytes";

  deserialize_and_append_to_txtracking();

  // Append these tx frames to the flow's tx queue, and trigger
  // intial tx. Future received ACKs will trigger more tx.
  transmit_pending_packets();
}

void UcclFlow::process_rttprobe_rsp(uint64_t ts1, uint64_t ts2, uint64_t ts3,
                                    uint64_t ts4, uint32_t path_id) {
  auto rtt_ns = (ts4 - ts1) - (ts3 - ts2);
  auto sample_rtt_tsc = ns_to_cycles(rtt_ns, freq_ghz);
  port_path_rtt_[path_id] = sample_rtt_tsc;

  if constexpr (kCCType == CCType::kTimely) {
    timely_g_.timely_update_rate(rdtsc(), sample_rtt_tsc);
  }
  if constexpr (kCCType == CCType::kTimelyPP) {
    timely_pp_[path_id].timely_update_rate(rdtsc(), sample_rtt_tsc);
  }

  VLOG(3) << "sample_rtt_us " << to_usec(sample_rtt_tsc, freq_ghz)
          << " us, avg_rtt_diff " << timely_g_.timely_.get_avg_rtt_diff()
          << " us, timely rate " << timely_g_.timely_.get_rate_gbps()
          << " Gbps";

#ifdef RTT_STATS
  rtt_stats_.update(rtt_ns / 1000);
  if (++rtt_probe_count_ % 100000 == 0) {
    FILE* fp = fopen("rtt_stats.txt", "w");
    rtt_stats_.print(fp);
    fclose(fp);
  }
#endif
}

bool UcclFlow::periodic_check() {
  // TODO(yang): send RST packet, indicating removal of the flow.
  if (pcb_.max_rto_rexmits_consectutive_reached()) return false;

  pcb_.advance_rto_tick();

  auto& ready_wheel = pcb_.get_ready_rto_wheel();
  while (!ready_wheel.empty()) {
    auto [msgbuf, seqno] = ready_wheel.front();
    ready_wheel.pop_front();
    if (swift::seqno_ge(seqno, pcb_.snd_una)) {
      rto_retransmit((PacketBuf*)msgbuf, seqno);
    }
  }

  return true;
}

void UcclFlow::process_ack(UcclPktHdr const* ucclh) {
  auto const* ucclsackh = reinterpret_cast<UcclSackHdr const*>(
      reinterpret_cast<uint8_t const*>(ucclh) + kUcclHdrLen);
  auto ackno = ucclh->ackno.value();

  if (swift::seqno_lt(ackno, pcb_.snd_una)) {
    VLOG(3) << "Received old ACK " << ackno;
    return;
  } else if (swift::seqno_eq(ackno, pcb_.snd_una)) {
    VLOG(3) << "Received duplicate ACK " << ackno;
    // Duplicate ACK.
    pcb_.duplicate_acks++;
    // Update the number of out-of-order acknowledgements.
    pcb_.snd_ooo_acks = ucclsackh->sack_bitmap_count.value();

    if (pcb_.duplicate_acks < kFastRexmitDupAckThres) {
      // We have not reached the threshold yet, so we do not do
      // anything.
    } else if (pcb_.duplicate_acks == kFastRexmitDupAckThres) {
      // Fast retransmit.
      fast_retransmit();
    } else {
      // We have already done the fast retransmit, so we are now
      // in the fast recovery phase.
      auto sack_bitmap_count = ucclsackh->sack_bitmap_count.value();
      // We check the SACK bitmap to see if there are more undelivered
      // packets. In fast recovery mode we get after a fast
      // retransmit, we will retransmit all missing packets that we
      // find from the SACK bitmap, when enumerating the SACK bitmap
      // for up to sack_bitmap_count ACKs.
      auto* msgbuf = tx_tracking_.get_oldest_unacked_msgbuf();
      VLOG(2) << "Fast recovery " << ackno << " sack_bitmap_count "
              << sack_bitmap_count;

      // Avoid sending too many packets.

      // NOTICE(Nelson): DPDK does not support unpulled_tx_pkts() anymore.
      // if (socket_->unpulled_tx_pkts() > kMaxUnackedPktsPerEngine) return;
      auto num_unacked_pkts = tx_tracking_.num_unacked_msgbufs();
      if (num_unacked_pkts >= kMaxUnackedPktsPerEngine) return;
      auto unacked_pkt_budget = kMaxUnackedPktsPerEngine - num_unacked_pkts;
      // auto txq_free_entries =
      //     socket_->send_queue_free_entries(unacked_pkt_budget);
      // auto hard_budget =
      //     std::min(std::min(txq_free_entries, unacked_pkt_budget),
      //              (uint32_t)kSackBitmapSize);

      auto hard_budget =
          std::min(unacked_pkt_budget, (uint32_t)kSackBitmapSize);

      uint32_t index = 0;
      while (sack_bitmap_count && msgbuf && index < hard_budget) {
        size_t const sack_bitmap_bucket_idx =
            index / swift::Pcb::kSackBitmapBucketSize;
        size_t const sack_bitmap_idx_in_bucket =
            index % swift::Pcb::kSackBitmapBucketSize;
        auto sack_bitmap =
            ucclsackh->sack_bitmap[sack_bitmap_bucket_idx].value();
        if ((sack_bitmap & (1ULL << sack_bitmap_idx_in_bucket)) == 0) {
          // We found a missing packet.
          auto seqno = pcb_.snd_una + index;

          VLOG(2) << "Fast recovery retransmitting " << seqno;
          auto const* missing_ucclh = reinterpret_cast<UcclPktHdr const*>(
              msgbuf->get_pkt_addr() + kNetHdrLen);
          // TODO(yang): tmp fix---they should be equal, need to
          // refine the way we maintain tx_but_unacked msgbufs chains.
          if (seqno == missing_ucclh->seqno.value()) {
            auto path_id = get_path_id_with_lowest_rtt();
#ifdef REXMIT_SET_PATH
            tx_tracking_.dec_unacked_pkts_pp(get_path_id(seqno));
            tx_tracking_.inc_unacked_pkts_pp(path_id);
            set_path_id(seqno, path_id);
#endif
            prepare_datapacket(msgbuf, path_id, seqno,
                               UcclPktHdr::UcclFlags::kData);
            msgbuf->mark_not_txpulltime_free();
            missing_frames_.push_back(msgbuf->get_pkt());
            pcb_.add_to_rto_wheel(msgbuf, seqno);
            pcb_.fast_recovers++;
          }
        } else {
          sack_bitmap_count--;
        }
        index++;
        msgbuf = msgbuf->next();
      }
      if (!missing_frames_.empty()) {
        VLOG(2) << "Fast recovery retransmitting " << missing_frames_.size()
                << " missing packets";
        // TODO(yang): handling the cases where the number of
        // missing frames is larger than the free send_queue size.
        socket_->send_packets(missing_frames_.data(), missing_frames_.size());
        missing_frames_.clear();
      }
    }
  } else if (swift::seqno_gt(ackno, pcb_.snd_nxt)) {
    VLOG(3) << "Received ACK for untransmitted data.";
  } else {
    VLOG(3) << "Received valid ACK " << ackno;
    // This is a valid ACK, acknowledging new data.
    size_t num_acked_packets = ackno - pcb_.snd_una;
    tx_tracking_.receive_acks(num_acked_packets);

    if constexpr (kCCType == CCType::kCubic) {
      cubic_g_.cubic_on_recv_ack(num_acked_packets);
    }
    if constexpr (kCCType == CCType::kCubicPP) {
      uint32_t accumu_acks = 0;
      auto last_path_id = kMaxPath;
      uint32_t seqno = pcb_.snd_una;
      for (size_t i = 0; i < num_acked_packets; i++, seqno++) {
        auto path_id = get_path_id(seqno);
        if (path_id != last_path_id && last_path_id != kMaxPath) {
          cubic_pp_[last_path_id].cubic_on_recv_ack(accumu_acks);
          accumu_acks = 0;
        }
        last_path_id = path_id;
        accumu_acks++;
        tx_tracking_.dec_unacked_pkts_pp(path_id);
        VLOG(3) << "Hybrid acked seqno " << seqno << " path_id " << path_id;
      }
      if (accumu_acks) {
        cubic_pp_[last_path_id].cubic_on_recv_ack(accumu_acks);
      }
    } else {
      uint32_t seqno = pcb_.snd_una;
      for (size_t i = 0; i < num_acked_packets; i++, seqno++) {
        auto path_id = get_path_id(seqno);
        tx_tracking_.dec_unacked_pkts_pp(path_id);
        VLOG(3) << "Hybrid acked seqno " << seqno << " path_id " << path_id;
      }
    }

    pcb_.snd_una = ackno;
    pcb_.duplicate_acks = 0;
    pcb_.snd_ooo_acks = 0;
    pcb_.rto_rexmits_consectutive = 0;
  }
}

void UcclFlow::fast_retransmit() {
  // Retransmit the oldest unacknowledged message buffer.
  auto* msgbuf = tx_tracking_.get_oldest_unacked_msgbuf();
  auto seqno = pcb_.snd_una;
  VLOG(3) << "Fast retransmitting oldest unacked packet " << pcb_.snd_una;

  if (msgbuf && seqno != pcb_.snd_nxt) {
    auto path_id = get_path_id_with_lowest_rtt();
#ifdef REXMIT_SET_PATH
    tx_tracking_.dec_unacked_pkts_pp(get_path_id(seqno));
    tx_tracking_.inc_unacked_pkts_pp(path_id);
    set_path_id(seqno, path_id);
#endif
    prepare_datapacket(msgbuf, path_id, seqno, UcclPktHdr::UcclFlags::kData);
    auto const* ucclh = reinterpret_cast<UcclPktHdr const*>(
        msgbuf->get_pkt_addr() + kNetHdrLen);
    DCHECK_EQ(seqno, ucclh->seqno.value());
    msgbuf->mark_not_txpulltime_free();
    socket_->send_packet(msgbuf->get_pkt());
    pcb_.add_to_rto_wheel(msgbuf, seqno);
    pcb_.fast_rexmits++;
  }
}

void UcclFlow::rto_retransmit(PacketBuf* msgbuf, uint32_t seqno) {
  VLOG(3) << "RTO retransmitting oldest unacked packet " << seqno;
  auto path_id = get_path_id_with_lowest_rtt();
#ifdef REXMIT_SET_PATH
  tx_tracking_.dec_unacked_pkts_pp(get_path_id(seqno));
  tx_tracking_.inc_unacked_pkts_pp(path_id);
  set_path_id(seqno, path_id);
#endif
  prepare_datapacket(msgbuf, path_id, seqno, UcclPktHdr::UcclFlags::kData);
  msgbuf->mark_not_txpulltime_free();
  socket_->send_packet(msgbuf->get_pkt());
  pcb_.add_to_rto_wheel(msgbuf, seqno);
  pcb_.rto_rexmits++;
  pcb_.rto_rexmits_consectutive++;

  if constexpr (kCCType == CCType::kCubic) {
    cubic_g_.cubic_on_packet_loss();
    VLOG(2) << "rto " << cubic_g_.to_string() << " inflight "
            << pcb_.snd_nxt - pcb_.snd_una << " "
            << tx_tracking_.num_unacked_msgbufs();
  }
  if constexpr (kCCType == CCType::kCubicPP) {
    auto path_id = get_path_id(seqno);
    cubic_pp_[path_id].cubic_on_packet_loss();
  }
}

/**
 * @brief Helper function to transmit a number of packets from the queue
 * of pending TX data.
 */
void UcclFlow::transmit_pending_packets() {
  // Avoid sending too many packets.
  // auto num_unacked_pkts = tx_tracking_.num_unacked_msgbufs();
  // if (num_unacked_pkts >= kMaxUnackedPktsPerEngine) return;

  // auto unacked_pkt_budget = kMaxUnackedPktsPerEngine - num_unacked_pkts;
  // auto txq_free_entries =
  //     socket_->send_queue_free_entries(unacked_pkt_budget);
  // auto hard_budget = std::min(txq_free_entries, unacked_pkt_budget);

  // auto hard_budget = socket_->send_queue_free_entries();
  // NOTICE(Nelson): DPDK does not support send_queue_free_entries() anymore.
  auto hard_budget = SEND_BATCH_SIZE;

  uint32_t permitted_packets = 0;

  if constexpr (kCCType == CCType::kTimely || kCCType == CCType::kTimelyPP) {
    permitted_packets = timely_g_.timely_ready_packets(hard_budget);
  }
  if constexpr (kCCType == CCType::kCubic) {
    permitted_packets = std::min(hard_budget, cubic_g_.cubic_effective_wnd());
  }
  if constexpr (kCCType == CCType::kCubicPP) {
    permitted_packets = std::min(hard_budget, SEND_BATCH_SIZE);
  }

  // static uint64_t transmit_tries = 0;
  // static uint64_t transmit_success = 0;
  // transmit_tries++;
  // if (permitted_packets != 0) transmit_success++;
  // if (transmit_tries % 10000 == 0) {
  //     LOG(INFO) << "transmitting success rate: "
  //               << (double)transmit_success / transmit_tries;
  // }

  // LOG_EVERY_N(INFO, 10000)
  //     << "permitted_packets " << permitted_packets << " num_unacked_pkts "
  //     << num_unacked_pkts << " txq_free_entries " << txq_free_entries
  //     << " num_unsent_pkts " << tx_tracking_.num_unsent_msgbufs()
  //     << " pending_tx_msgs_ " << pending_tx_msgs_.size();

  // Prepare the packets.
  auto now_tsc = rdtsc();
  for (uint32_t i = 0; i < permitted_packets; i++) {
    uint32_t path_id = 0;
    uint32_t path_cwnd = 0;
    uint32_t path_unacked = 0;
    bool found_path = false;

    if constexpr (kCCType == CCType::kCubicPP) {
      // Avoiding sending too many packets on the same path.
      if (i % kSwitchPathThres == 0) {
        int tries = 0;
        while (tries++ < 16) {
          path_id = get_path_id_with_lowest_rtt();
          path_unacked = tx_tracking_.get_unacked_pkts_pp(path_id);
          path_cwnd = cubic_pp_[path_id].cubic_cwnd();
          if (path_unacked + kSwitchPathThres <= path_cwnd &&
              tx_tracking_.is_available_for_tx(path_id, now_tsc)) {
            found_path = true;
            break;
          }
        }
        if (!found_path) {
          // We cannot find a path with enough space to send packets.
          VLOG(2) << "[CubicPP] Cannot find path with available cwnd: "
                  << tx_tracking_.unacked_pkts_pp_to_string();
          break;
        }
      }
    } else {
      path_id = get_path_id_with_lowest_rtt();
    }

    auto msgbuf_opt = tx_tracking_.get_and_update_oldest_unsent();
    if (!msgbuf_opt.has_value()) break;
    auto* msgbuf = msgbuf_opt.value();
    auto seqno = pcb_.get_snd_nxt();
    set_path_id(seqno, path_id);
    tx_tracking_.inc_unacked_pkts_pp(path_id);
    tx_tracking_.set_last_tx_tsc_pp(path_id, now_tsc);
    VLOG(3) << "Transmitting seqno: " << seqno << " path_id: " << path_id;

    if (msgbuf->is_last()) {
      VLOG(2) << "Transmitting seqno: " << seqno << " payload_len: "
              << msgbuf->get_packet_len() - kNetHdrLen - kUcclHdrLen;
    }
    auto net_flags = (i == 0) ? UcclPktHdr::UcclFlags::kDataRttProbe
                              : UcclPktHdr::UcclFlags::kData;
    prepare_datapacket(msgbuf, path_id, seqno, net_flags);
    msgbuf->mark_not_txpulltime_free();
    pending_tx_frames_.push_back(msgbuf->get_pkt());

    pcb_.add_to_rto_wheel(msgbuf, seqno);
  }

  // TX both data and ack frames.
  if (pending_tx_frames_.empty()) {
    // NOTICE(Nelson): DPDK does not support kick_tx_and_pull() anymore.
    // socket_->kick_tx_and_pull();
    return;
  }
  VLOG(3) << "tx packets " << pending_tx_frames_.size();

  socket_->send_packets(pending_tx_frames_.data(), pending_tx_frames_.size());

  pending_tx_frames_.clear();
}

void UcclFlow::deserialize_and_append_to_txtracking() {
  while (tx_tracking_.num_unsent_msgbufs() < kMaxTwPkts &&
         !pending_tx_msgs_.empty()) {
    auto deser_budget = kMaxTwPkts - tx_tracking_.num_unsent_msgbufs();

    auto& [tx_work, cur_offset] = pending_tx_msgs_.front();
    PacketBuf* cur_msgbuf = tx_work.deser_msgs;
    PacketBuf* tx_msgbuf_head = cur_msgbuf;
    PacketBuf* tx_msgbuf_tail = nullptr;
    uint32_t num_tx_frames = 0;
    size_t remaining_bytes = tx_work.len - cur_offset;

    uint32_t path_id = kMaxPath;
    if constexpr (kCCType == CCType::kTimelyPP) {
      path_id = get_path_id_with_lowest_rtt();
    }

    auto now_tsc = rdtsc();
    while (cur_msgbuf != nullptr && num_tx_frames < deser_budget) {
      // The flow will free these Tx frames when receiving ACKs.
      cur_msgbuf->mark_not_txpulltime_free();
      if (remaining_bytes == tx_work.len) cur_msgbuf->mark_first();

      auto payload_len =
          cur_msgbuf->get_packet_len() - kNetHdrLen - kUcclHdrLen;

      // Both queue on one timing wheel.
      if constexpr (kCCType == CCType::kTimely) {
        timely_g_.timely_pace_packet(
            now_tsc, payload_len + kNetHdrLen + kUcclHdrLen, cur_msgbuf);
      }
      if constexpr (kCCType == CCType::kTimelyPP) {
        // TODO(yang): consider per-path rate limiting? If so, we need to
        // maintain prev_desired_tx_tsc_ for each path, calculate two
        // timestamps (one from timely_g_, one from
        // timely_pp_[path_id]), and insert the larger one into the
        // timely_g_.
        double rate = timely_pp_[path_id].timely_rate();
        timely_g_.timely_pace_packet_with_rate(
            now_tsc, payload_len + kNetHdrLen + kUcclHdrLen, cur_msgbuf, rate);
      }

      remaining_bytes -= payload_len;
      if (remaining_bytes == 0) {
        DCHECK_EQ(cur_msgbuf->next(), nullptr);
        cur_msgbuf->mark_last();
      }

      tx_msgbuf_tail = cur_msgbuf;
      cur_msgbuf = cur_msgbuf->next();
      num_tx_frames++;
    }

    DCHECK(tx_msgbuf_tail != nullptr)
        << deser_budget << " " << num_tx_frames << " ";

    tx_msgbuf_tail->set_next(nullptr);

    // LOG_EVERY_N(INFO, 10000)
    //     << "deser unsent_msgbufs " << tx_tracking_.num_unsent_msgbufs()
    //     << " deser_budget " << deser_budget << " pending_tx_msgs "
    //     << pending_tx_msgs_.size() << " successfully added to timingwheel "
    //     << num_tx_frames << " tx_tracking poll_ctxs "
    //     << tx_tracking_.poll_ctxs_.size();

    if (remaining_bytes == 0) {
      // This message has been fully deserialized and added to tx tracking.
      pending_tx_msgs_.pop_front();
    } else {
      // Resuming the deserialization of this message in the next iteration.
      tx_work.deser_msgs = cur_msgbuf;
      cur_offset = tx_work.len - remaining_bytes;
    }

    tx_tracking_.append(
        tx_msgbuf_head, tx_msgbuf_tail, num_tx_frames,
        (tx_msgbuf_head && tx_msgbuf_head->is_first() && num_tx_frames)
            ? tx_work.poll_ctx
            : nullptr);
  }
}

void UcclFlow::prepare_l2header(uint8_t* pkt_addr) const {
  Ethernet* eth = reinterpret_cast<Ethernet*>(pkt_addr);
  eth->src_addr = local_l2_addr_;
  eth->dst_addr = remote_l2_addr_;
  eth->eth_type = be16_t(Ethernet::kIpv4);

  // memcpy(eh->h_source, local_l2_addr_, ETH_ALEN);
  // memcpy(eh->h_dest, remote_l2_addr_, ETH_ALEN);
  // eh->h_proto = htons(ETH_P_IP);
}

void UcclFlow::prepare_l3header(uint8_t* pkt_addr,
                                uint32_t payload_bytes) const {
  Ipv4* ipv4 = reinterpret_cast<Ipv4*>(pkt_addr + sizeof(Ethernet));
  ipv4->version_ihl = 0x45;
  ipv4->type_of_service = 0;
  ipv4->packet_id = be16_t(0x1513);
  ipv4->fragment_offset = be16_t(0);
  ipv4->time_to_live = Ipv4::kDefaultTTL;
  ipv4->total_length = be16_t(sizeof(iphdr) + sizeof(udphdr) + payload_bytes);
  ipv4->next_proto_id = Ipv4::Proto::kUdp;
  ipv4->src_addr = local_addr_;
  ipv4->dst_addr = remote_addr_;
  ipv4->hdr_checksum = 0;
}

void UcclFlow::prepare_l4header(uint8_t* pkt_addr, uint32_t payload_bytes,
                                uint16_t dst_port) const {
  Udp* udp = reinterpret_cast<Udp*>(pkt_addr + sizeof(Ethernet) + sizeof(Ipv4));
#ifdef USE_MULTIPATH
  udp->src_port = BASE_PORT;
  udp->dst_port = dst_port;
#else
  udp->src_port = BASE_PORT;
  udp->dst_port = BASE_PORT;
#endif

  udp->len = be16_t(sizeof(Udp) + payload_bytes);
  udp->cksum = be16_t(0);
}

void UcclFlow::prepare_l3l4checksum(Packet* pkt) const {
  // Offload IPv4 and UDP checksums to hardware.
  pkt->set_l2_len(sizeof(Ethernet));
  pkt->set_l3_len(sizeof(Ipv4));
  pkt->offload_udpv4_csum();
}

void UcclFlow::prepare_datapacket(PacketBuf* msgbuf, uint32_t path_id,
                                  uint32_t seqno,
                                  UcclPktHdr::UcclFlags const net_flags) {
  // Header length after before the payload.
  uint32_t frame_len = msgbuf->get_packet_len();
  DCHECK_LE(frame_len, DPDK_MTU);
  uint8_t* pkt_addr = msgbuf->get_pkt_addr();

  // Prepare network headers.
  prepare_l2header(pkt_addr);
  prepare_l3header(pkt_addr, frame_len - kNetHdrLen);
  prepare_l4header(pkt_addr, frame_len - kNetHdrLen, dst_ports_[path_id]);

  // Prepare the Uccl-specific header.
  auto* ucclh = reinterpret_cast<UcclPktHdr*>(pkt_addr + kNetHdrLen);
  ucclh->magic = be16_t(UcclPktHdr::kMagic);
  ucclh->engine_id = remote_engine_idx_;
  ucclh->path_id = (uint16_t)path_id;
  ucclh->net_flags = net_flags;
  ucclh->ackno = be32_t(UINT32_MAX);
  // This fills the FrameBuf flags into the outgoing packet msg_flags.
  ucclh->msg_flags = msgbuf->msg_flags();
  ucclh->frame_len = be16_t(frame_len);

  ucclh->seqno = be32_t(seqno);
  ucclh->flow_id = be64_t(flow_id_);

  ucclh->timestamp1 =
      (net_flags == UcclPktHdr::UcclFlags::kDataRttProbe)
          ? get_monotonic_time_ns() + socket_->send_queue_estimated_latency_ns()
          : 0;

  ucclh->timestamp2 = 0;  // let the receiver ebpf fill this in.

  prepare_l3l4checksum(msgbuf->get_pkt());
}

Packet* UcclFlow::craft_ackpacket(uint32_t path_id, uint16_t dst_port,
                                  uint32_t seqno, uint32_t ackno,
                                  UcclPktHdr::UcclFlags const net_flags,
                                  uint64_t ts1, uint64_t ts2) {
  size_t const kControlPayloadBytes = kUcclHdrLen + kUcclSackHdrLen;
  auto pkt = socket_->pop_packet(kNetHdrLen + kControlPayloadBytes);
  auto msgbuf = PacketBuf::Create(pkt);
  // Let AFXDPSocket::pull_complete_queue() free control frames.
  msgbuf->mark_txpulltime_free();

  uint8_t* pkt_addr = msgbuf->get_pkt_addr();
  prepare_l2header(pkt_addr);
  prepare_l3header(pkt_addr, kControlPayloadBytes);
  prepare_l4header(pkt_addr, kControlPayloadBytes, dst_port);

  auto* ucclh = (UcclPktHdr*)(pkt_addr + kNetHdrLen);
  ucclh->magic = be16_t(UcclPktHdr::kMagic);
  ucclh->engine_id = remote_engine_idx_;
  ucclh->path_id = (uint16_t)path_id;
  ucclh->net_flags = net_flags;
  ucclh->msg_flags = 0;
  ucclh->frame_len = be16_t(kNetHdrLen + kControlPayloadBytes);
  ucclh->seqno = be32_t(seqno);
  ucclh->ackno = be32_t(ackno);
  ucclh->flow_id = be64_t(flow_id_);
  ucclh->timestamp1 = ts1;
  ucclh->timestamp2 = ts2;

  auto* ucclsackh = (UcclSackHdr*)(pkt_addr + kNetHdrLen + kUcclHdrLen);

  for (size_t i = 0; i < sizeof(UcclSackHdr::sack_bitmap) /
                             sizeof(UcclSackHdr::sack_bitmap[0]);
       ++i) {
    ucclsackh->sack_bitmap[i] = be64_t(pcb_.sack_bitmap[i]);
  }
  ucclsackh->sack_bitmap_count = be16_t(pcb_.sack_bitmap_count);

  ucclsackh->timestamp3 =
      (net_flags == UcclPktHdr::UcclFlags::kAckRttProbe)
          ? get_monotonic_time_ns() + socket_->send_queue_estimated_latency_ns()
          : 0;
  ucclsackh->timestamp4 = 0;  // let the sender ebpf fill this in.

  prepare_l3l4checksum(msgbuf->get_pkt());

  return msgbuf->get_pkt();
}

Packet* UcclFlow::craft_rssprobe_packet(uint16_t dst_port) {
  size_t const kRssProbePayloadBytes = kUcclHdrLen;
  auto pkt = socket_->pop_packet(kNetHdrLen + kRssProbePayloadBytes);
  auto msgbuf = PacketBuf::Create(pkt);
  // Let AFXDPSocket::pull_complete_queue() free control frames.
  msgbuf->mark_txpulltime_free();

  uint8_t* pkt_addr = msgbuf->get_pkt_addr();
  prepare_l2header(pkt_addr);
  prepare_l3header(pkt_addr, kRssProbePayloadBytes);
  prepare_l4header(pkt_addr, kRssProbePayloadBytes, dst_port);

  auto* udph = (udphdr*)(pkt_addr + sizeof(ethhdr) + sizeof(iphdr));
  udph->dest = htons(dst_port);

  auto* ucclh = (UcclPktHdr*)(pkt_addr + kNetHdrLen);
  ucclh->magic = be16_t(UcclPktHdr::kMagic);
  ucclh->engine_id = remote_engine_idx_;
  ucclh->path_id = 0;
  ucclh->net_flags = UcclPktHdr::UcclFlags::kRssProbe;
  ucclh->msg_flags = 0;
  ucclh->frame_len = be16_t(kNetHdrLen + kRssProbePayloadBytes);
  ucclh->seqno = be32_t(UINT32_MAX);
  ucclh->ackno = be32_t(UINT32_MAX);
  ucclh->flow_id = be64_t(flow_id_);
  ucclh->timestamp1 = 0;
  ucclh->timestamp2 = 0;

  prepare_l3l4checksum(msgbuf->get_pkt());

  return msgbuf->get_pkt();
}

void UcclFlow::reverse_packet_l2l3(PacketBuf* msgbuf) {
  auto* pkt_addr = msgbuf->get_pkt_addr();
  auto* eth = (ethhdr*)pkt_addr;
  auto* ipv4h = (iphdr*)(pkt_addr + sizeof(ethhdr));
  auto* udp = (udphdr*)(pkt_addr + sizeof(ethhdr) + sizeof(iphdr));

  unsigned char tmp_mac[ETH_ALEN];
  uint32_t tmp_ip;

  memcpy(tmp_mac, eth->h_source, ETH_ALEN);
  memcpy(eth->h_source, eth->h_dest, ETH_ALEN);
  memcpy(eth->h_dest, tmp_mac, ETH_ALEN);

  tmp_ip = ipv4h->saddr;
  ipv4h->saddr = ipv4h->daddr;
  ipv4h->daddr = tmp_ip;

  udp->check = 0;
  ipv4h->check = 0;
  ipv4h->check = ipv4_checksum(ipv4h, sizeof(iphdr));
}

void UcclEngine::run() {
  Channel::Msg rx_work;
  Channel::Msg tx_deser_work;

  Packet** pkts = new Packet*[RECV_BATCH_SIZE];

  while (!shutdown_) {
    // Calculate the cycles elapsed since last periodic processing.
    auto now_tsc = rdtsc();
    auto const elapsed_tsc = now_tsc - last_periodic_tsc_;

    if (elapsed_tsc >= kSlowTimerIntervalTsc_) {
      // Perform periodic processing.
      periodic_process();
      last_periodic_tsc_ = now_tsc;
    }

    if (Channel::dequeue_sc(channel_->rx_task_q_, &rx_work)) {
      VLOG(3) << "Rx jring dequeue";
      active_flows_map_[rx_work.flow_id]->rx_supply_app_buf(rx_work);
    }

    uint32_t rcvd = socket_->recv_packets(pkts, RECV_BATCH_SIZE);

    if (rcvd) {
      // LOG(INFO) << "rcvd: " << rcvd;
      process_rx_msg(pkts, rcvd);
    }

    if (Channel::dequeue_sc(channel_->tx_deser_q_, &tx_deser_work)) {
      // Make data written by the app thread visible to the engine.
      tx_deser_work.poll_ctx->read_barrier();

      VLOG(3) << "Tx deser jring dequeue";
      active_flows_map_[tx_deser_work.flow_id]->tx_messages(tx_deser_work);
    }

    for (auto& [flow_id, flow] : active_flows_map_) {
      flow->transmit_pending_packets();
    }

#ifndef THREADED_MEMCPY
    deser_th_func(std::vector<UcclEngine*>{this});
#endif
  }

  delete[] pkts;

  // This will reset flow pcb state.
  for (auto [flow_id, flow] : active_flows_map_) {
    flow->shutdown();
    delete flow;
  }
  // This will flush all unpolled tx frames.
  // socket_->shutdown();
}

void UcclEngine::deser_th_func(std::vector<UcclEngine*> engines) {
  Channel::Msg tx_deser_work;
  Channel::Msg rx_deser_work;

#ifdef THREADED_MEMCPY
  while (!engines[0]->shutdown_) {
#endif
    for (auto engine : engines) {
      if (Channel::dequeue_sc(engine->channel_->tx_task_q_, &tx_deser_work)) {
        // Make data written by the app thread visible to the deser
        // thread.
        tx_deser_work.poll_ctx->read_barrier();
        // LOG(INFO) << "Tx jring dequeue";
        VLOG(3) << "Tx jring dequeue";

        // deser tx_work into a framebuf chain, then pass to deser_th.
        PacketBuf* deser_msgs_head = nullptr;
        PacketBuf* deser_msgs_tail = nullptr;
        auto* app_buf_cursor = tx_deser_work.data;
        auto remaining_bytes = tx_deser_work.len;
        while (remaining_bytes > 0) {
          auto payload_len = std::min(
              remaining_bytes, (size_t)DPDK_MTU - kNetHdrLen - kUcclHdrLen);
          auto pkt = engine->socket_->pop_packet(payload_len + kNetHdrLen +
                                                 kUcclHdrLen);
          auto* msgbuf = PacketBuf::Create(pkt);
#ifndef EMULATE_ZC
          auto pkt_payload_addr =
              msgbuf->get_pkt_addr() + kNetHdrLen + kUcclHdrLen;
          memcpy(pkt_payload_addr, app_buf_cursor, payload_len);
#endif

          remaining_bytes -= payload_len;
          app_buf_cursor = (uint8_t*)app_buf_cursor + payload_len;

          if (deser_msgs_head == nullptr) {
            deser_msgs_head = msgbuf;
            deser_msgs_tail = msgbuf;
          } else {
            deser_msgs_tail->set_next(msgbuf);
            deser_msgs_tail = msgbuf;
          }
        }
        deser_msgs_tail->set_next(nullptr);
        tx_deser_work.deser_msgs = deser_msgs_head;

        // Make sure the app thread sees the deserialized messages.
        tx_deser_work.poll_ctx->write_barrier();
        Channel::enqueue_sp(engine->channel_->tx_deser_q_, &tx_deser_work);
      }
      if (Channel::dequeue_sc(engine->channel_->rx_deser_q_, &rx_deser_work)) {
        // Make data written by engine thread visible to the deser
        // thread.
        rx_deser_work.poll_ctx->read_barrier();
        VLOG(3) << "Rx ser jring dequeue";

        PacketBuf* ready_msg = rx_deser_work.deser_msgs;
        auto* app_buf = rx_deser_work.data;
        auto* app_buf_len_p = rx_deser_work.len_p;
        auto* poll_ctx = rx_deser_work.poll_ctx;
        size_t cur_offset = 0;

        while (ready_msg != nullptr) {
          auto* pkt_addr = ready_msg->get_pkt_addr();
          DCHECK(pkt_addr) << "pkt_addr is nullptr when copy to app buf "
                           << std::hex << "0x" << ready_msg << std::dec
                           << ready_msg->to_string();
          auto* payload_addr = pkt_addr + kNetHdrLen + kUcclHdrLen;
          auto payload_len =
              ready_msg->get_packet_len() - kNetHdrLen - kUcclHdrLen;

          auto const* ucclh =
              reinterpret_cast<UcclPktHdr const*>(pkt_addr + kNetHdrLen);
          VLOG(2) << "payload_len: " << payload_len << " seqno: " << std::dec
                  << ucclh->seqno.value();
#ifndef EMULATE_ZC
          memcpy((uint8_t*)app_buf + cur_offset, payload_addr, payload_len);
#endif
          cur_offset += payload_len;

          // We have a complete message. Let's deliver it to the app.
          if (ready_msg->is_last()) {
            *app_buf_len_p = cur_offset;

            // Wakeup app thread waiting on endpoint.
            poll_ctx->write_barrier();
            {
              std::lock_guard<std::mutex> lock(poll_ctx->mu);
              poll_ctx->done = true;
              poll_ctx->cv.notify_one();
            }
            VLOG(2) << "Received a complete message " << cur_offset << " bytes";
          }

          Packet* ready_msg_tmp = ready_msg->get_pkt();
          ready_msg = ready_msg->next();
          // Free received frames that have been copied to app buf.
          engine->socket_->push_packet(ready_msg_tmp);
        }
      }
    }
#ifdef THREADED_MEMCPY
  }
#endif
}

void UcclEngine::process_rx_msg(Packet** pkts, uint32_t rcvd) {
  for (uint32_t i = 0; i < rcvd; i++) {
    auto* pkt = pkts[i];

    auto* msgbuf = PacketBuf::Create(pkt);
    auto* pkt_addr = msgbuf->get_pkt_addr();

    auto* ethh = reinterpret_cast<ethhdr*>(pkt_addr);
    if (ntohs(ethh->h_proto) != ETH_P_IP) {
      LOG(INFO) << "Non-IP packet, EtherType: 0x" << std::hex
                << ntohs(ethh->h_proto);
      continue;
    }

    auto* iph = reinterpret_cast<iphdr*>(pkt_addr + sizeof(ethhdr));
    if (iph->protocol != IPPROTO_UDP) {
      LOG(INFO) << "Non-UDP packet, IP protocol: " << (int)iph->protocol;
      continue;
    }

    auto* ucclh = reinterpret_cast<UcclPktHdr*>(pkt_addr + kNetHdrLen);

    // Record the incoming packet UcclPktHdr.msg_flags in
    // FrameBuf.
    msgbuf->set_msg_flags(ucclh->msg_flags);

    if (msgbuf->is_last()) {
      VLOG(2) << "Received seqno: " << ucclh->seqno.value() << " payload_len: "
              << msgbuf->get_packet_len() - kNetHdrLen - kUcclHdrLen;
    }

    auto flow_id = ucclh->flow_id.value();

    auto it = active_flows_map_.find(flow_id);
    if (it == active_flows_map_.end()) {
      LOG_EVERY_N(ERROR, 1000000)
          << "process_rx_msg unknown flow " << std::hex << "0x" << flow_id
          << " engine_id " << local_engine_idx_ << " pkt->engine_id "
          << (int)ucclh->engine_id;
      for (auto [flow_id, flow] : active_flows_map_) {
        LOG_EVERY_N(ERROR, 1000000)
            << "                active flow " << std::hex << "0x" << flow_id;
      }
      socket_->push_packet(msgbuf->get_pkt());
      continue;
    }
    it->second->pending_rx_msgbufs_.push_back(msgbuf);
  }
  for (auto& [flow_id, flow] : active_flows_map_) {
    flow->rx_messages();
  }
}

/**
 * @brief Method to perform periodic processing. This is called by the
 * main engine cycle (see method `Run`).
 */
void UcclEngine::periodic_process() {
  // Advance the periodic ticks counter.
  periodic_ticks_++;
  handle_rto();
  process_ctl_reqs();
}

void UcclEngine::handle_rto() {
  for (auto [flow_id, flow] : active_flows_map_) {
    auto is_active_flow = flow->periodic_check();
    DCHECK(is_active_flow);
  }
}

void UcclEngine::process_ctl_reqs() {
  Channel::CtrlMsg ctrl_work;
  if (Channel::dequeue_sc(channel_->ctrl_task_q_, &ctrl_work)) {
    switch (ctrl_work.opcode) {
      case Channel::CtrlMsg::Op::kInstallFlow:
        handle_install_flow_on_engine(ctrl_work);
        break;
      default:
        break;
    }
  }
}

void UcclEngine::handle_install_flow_on_engine(Channel::CtrlMsg& ctrl_work) {
  LOG(INFO) << "[Engine] handle_install_flow_on_engine " << local_engine_idx_;
  int ret;
  std::string local_ip_str = ip_to_str(htonl(local_addr_));
  auto flow_id = ctrl_work.flow_id;
  auto remote_addr = ctrl_work.remote_ip;
  std::string remote_ip_str = ip_to_str(htonl(remote_addr));
  auto remote_mac_char = ctrl_work.remote_mac;
  auto remote_engine_idx = ctrl_work.remote_engine_idx;
  auto* poll_ctx = ctrl_work.poll_ctx;

  auto* flow = new UcclFlow(local_addr_, remote_addr, local_l2_addr_,
                            remote_mac_char, local_engine_idx_,
                            remote_engine_idx, socket_, channel_, flow_id);
  std::tie(std::ignore, ret) = active_flows_map_.insert({flow_id, flow});
  DCHECK(ret);

  LOG(INFO) << "[Engine] start RSS probing";

  // RSS probing to get a list of dst_port matching remote engine queue and,
  // reversely, matching local engine queue. Basically, symmetric dst_ports.
  std::set<uint16_t> dst_ports_set;
  Packet** pkts = new Packet*[RECV_BATCH_SIZE];

  for (uint32_t i = BASE_PORT; i < 65536;
       i = (i + 1) % (65536 - BASE_PORT) + BASE_PORT) {
    uint16_t dst_port = i;
    uint32_t sent = socket_->send_packet(flow->craft_rssprobe_packet(dst_port));

    if (!sent) {
      LOG(INFO) << "[Engine] failed to send RSS probe packet to port "
                << dst_port;
      continue;
    }

    uint32_t rcvd = socket_->recv_packets(pkts, RECV_BATCH_SIZE);
    if (rcvd) {
      VLOG(3) << "[Engine] received " << rcvd << " RSS probe packets";
    }

    for (uint32_t j = 0; j < rcvd; j++) {
      auto* pkt = pkts[j];
      auto* msgbuf = PacketBuf::Create(pkt);
      auto* pkt_addr = msgbuf->get_pkt_addr();
      auto* udp =
          reinterpret_cast<Udp*>(pkt_addr + sizeof(Ethernet) + sizeof(Ipv4));
      auto* ucclh = reinterpret_cast<UcclPktHdr*>(pkt_addr + kNetHdrLen);

      if (ucclh->net_flags == UcclPktHdr::UcclFlags::kRssProbe) {
        VLOG(3) << "[Engine] received RSS probe packet";
        if (ucclh->engine_id == local_engine_idx_) {
          // Probe packets arrive the remote engine!
          ucclh->net_flags = UcclPktHdr::UcclFlags::kRssProbeRsp;
          ucclh->engine_id = remote_engine_idx;
          msgbuf->mark_txpulltime_free();
          // Reverse so to send back
          flow->reverse_packet_l2l3(msgbuf);
          socket_->send_packet(msgbuf->get_pkt());
        } else {
          socket_->push_packet(pkt);
        }
      } else {
        VLOG(3) << "[Engine] received RSS probe rsp packet";
        DCHECK(ucclh->net_flags == UcclPktHdr::UcclFlags::kRssProbeRsp);
        if (ucclh->engine_id == local_engine_idx_) {
          // Probe rsp packets arrive this engine!
          dst_ports_set.insert(ntohs(udp->dst_port.port.value()));
        }
        socket_->push_packet(pkt);
      }
    }
    if (dst_ports_set.size() >= kMaxPath) break;
  }

  delete[] pkts;

  LOG(INFO) << "[Engine] handle_install_flow_on_engine dst_ports size: "
            << dst_ports_set.size();
  DCHECK_GE(dst_ports_set.size(), kMaxPath);

  flow->dst_ports_.reserve(kMaxPath);
  auto it = dst_ports_set.begin();
  std::advance(it, kMaxPath);
  std::copy(dst_ports_set.begin(), it, std::back_inserter(flow->dst_ports_));

  LOG(INFO) << "[Engine] install FlowID " << std::hex << "0x" << flow_id << ": "
            << local_ip_str << Format("(%d)", local_engine_idx_) << " <-> "
            << remote_ip_str << Format("(%d)", remote_engine_idx);

  // Wakeup app thread waiting on endpoint.
  {
    std::lock_guard<std::mutex> lock(poll_ctx->mu);
    poll_ctx->done = true;
    poll_ctx->cv.notify_one();
  }
}

std::string UcclEngine::status_to_string() {
  std::string s;
  for (auto [flow_id, flow] : active_flows_map_) {
    s += Format(
        "\n\t\tEngine %d Flow 0x%lx: %s (%u) <-> %s (%u)", local_engine_idx_,
        flow_id, ip_to_str(htonl(flow->local_addr_)).c_str(),
        flow->local_engine_idx_, ip_to_str(htonl(flow->remote_addr_)).c_str(),
        flow->remote_engine_idx_);
    s += flow->to_string();
  }
  s += "\n\t\t\t[DPDK] " + socket_->to_string();
  return s;
}

Endpoint::Endpoint(uint16_t port_id, char const* interface_name, int num_queues,
                   int engine_cpu_start)
    : num_queues_(num_queues),
      stats_thread_([this]() { stats_thread_fn(); }),
      dpdk_factory_(port_id, num_queues, num_queues) {
  LOG(INFO) << "Creating DPDKFactory";
  // Create UDS socket and get umem_fd and xsk_ids.
  dpdk_factory_.Init();

  local_ip_str_ = get_dev_ip(interface_name);
  local_mac_str_ = get_dev_mac(interface_name);

  CHECK_LE(num_queues, NUM_CPUS / 4)
      << "num_queues should be less than or equal to the number of CPUs / 4";

  LOG(INFO) << "Creating Channels";

  // Create multiple engines, each got its xsk and umem from the
  // daemon. Each engine has its own thread and channel to let the endpoint
  // communicate with.
  for (int i = 0; i < num_queues; i++) channel_vec_[i] = new Channel();

  LOG(INFO) << "Creating Engines";

  std::vector<std::future<std::unique_ptr<UcclEngine>>> engine_futures;
  for (int i = 0; i < num_queues; i++) {
    std::promise<std::unique_ptr<UcclEngine>> engine_promise;
    auto engine_future = engine_promise.get_future();
    engine_futures.emplace_back(std::move(engine_future));

    // Spawning a new thread to init engine and run the engine loop.
    engine_th_vec_.emplace_back(std::make_unique<std::thread>(
        [this, num_queues, i, engine_th_cpuid = engine_cpu_start + i,
         engine_promise = std::move(engine_promise)]() mutable {
          pin_thread_to_cpu(engine_th_cpuid);
          LOG(INFO) << "[Engine] thread " << i << " running on CPU "
                    << engine_th_cpuid;

          auto engine = std::make_unique<UcclEngine>(
              i, channel_vec_[i], dpdk_factory_.CreateSocket(i), local_ip_str_,
              local_mac_str_);
          auto* engine_ptr = engine.get();

          engine_promise.set_value(std::move(engine));
          engine_ptr->run();
        }));
  }
  std::vector<UcclEngine*> engines;
  for (auto& engine_future : engine_futures) {
    engine_vec_.emplace_back(std::move(engine_future.get()));
    engines.push_back(engine_vec_.back().get());
  }

#ifdef THREADED_MEMCPY
  for (int i = 0; i < num_queues; i++) {
    // Placing deser thread on engine_th_cpuid + num_queues.
    deser_th_vec_.emplace_back(std::make_unique<std::thread>(
        [i, deser_th_cpuid = engine_cpu_start + num_queues + i,
         engines = std::vector<UcclEngine*>{engines[i]}]() {
          pin_thread_to_cpu(deser_th_cpuid);
          LOG(INFO) << "[Engine] deser thread " << i << " running on CPU "
                    << deser_th_cpuid;
          UcclEngine::deser_th_func(engines);
        }));
  }
#endif

  ctx_pool_ = new SharedPool<PollCtx*, true>(
      kMaxInflightMsg, [](PollCtx* ctx) { ctx->clear(); });
  ctx_pool_buf_ = new uint8_t[kMaxInflightMsg * sizeof(PollCtx)];
  for (uint32_t i = 0; i < kMaxInflightMsg; i++) {
    ctx_pool_->push(new (ctx_pool_buf_ + i * sizeof(PollCtx)) PollCtx());
  }

  // Create listening socket
  listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  DCHECK(listen_fd_ >= 0) << "ERROR: opening socket";

  int flag = 1;
  DCHECK(setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(int)) >=
         0)
      << "ERROR: setsockopt SO_REUSEADDR fails";

  struct sockaddr_in serv_addr;
  bzero((char*)&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  serv_addr.sin_port = htons(kBootstrapPort);
  DCHECK(bind(listen_fd_, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) >= 0)
      << "ERROR: binding";

  DCHECK(!listen(listen_fd_, 128)) << "ERROR: listen";
  LOG(INFO) << "[Endpoint] server ready, listening on port " << kBootstrapPort;
}

Endpoint::~Endpoint() {
  for (auto& engine : engine_vec_) engine->shutdown();
  for (auto& engine_th : engine_th_vec_) engine_th->join();
  for (auto& deser_th : deser_th_vec_) deser_th->join();
  for (int i = 0; i < num_queues_; i++) delete channel_vec_[i];

  delete ctx_pool_;
  delete[] ctx_pool_buf_;

  close(listen_fd_);

  {
    std::lock_guard<std::mutex> lock(bootstrap_fd_map_mu_);
    for (auto& [flow_id, boostrap_id] : bootstrap_fd_map_) {
      close(boostrap_id);
    }
  }

  {
    std::lock_guard<std::mutex> lock(stats_mu_);
    shutdown_ = true;
    stats_cv_.notify_all();
  }
  stats_thread_.join();
}

/*
[Nelson Cheung]: ens1f1np1 is occupied by the dpdk, so we need to use eno49np0
to connect to the server.
*/

ConnID Endpoint::uccl_connect(std::string bootstrap_remote_ip,
                              std::string bootstrap_local_ip) {
  struct sockaddr_in serv_addr;
  struct hostent* server;
  int bootstrap_fd;

  bootstrap_fd = socket(AF_INET, SOCK_STREAM, 0);
  DCHECK(bootstrap_fd >= 0);

  server = gethostbyname(bootstrap_remote_ip.c_str());
  DCHECK(server);

  bzero((char*)&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  bcopy((char*)server->h_addr, (char*)&serv_addr.sin_addr.s_addr,
        server->h_length);
  serv_addr.sin_port = htons(kBootstrapPort);

  // Force the socket to bind to the local IP address.
  sockaddr_in localaddr = {};
  localaddr.sin_family = AF_INET;
  localaddr.sin_addr.s_addr = str_to_ip(bootstrap_local_ip.c_str());
  bind(bootstrap_fd, (sockaddr*)&localaddr, sizeof(localaddr));

  LOG(INFO) << "[Endpoint] connecting to " << bootstrap_remote_ip << ":"
            << kBootstrapPort;

  // Connect and set nonblocking and nodelay
  while (
      connect(bootstrap_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr))) {
    LOG(INFO) << "[Endpoint] connecting... Make sure the server is up.";
    sleep(1);
  }

  int flag = 1;
  setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void*)&flag, sizeof(int));

  auto local_engine_idx = find_least_loaded_engine_idx_and_update();
  CHECK_GE(local_engine_idx, 0);

  FlowID flow_id;
  while (true) {
    int ret = receive_message(bootstrap_fd, &flow_id, sizeof(FlowID));
    DCHECK(ret == sizeof(FlowID));
    LOG(INFO) << "[Endpoint] connect: receive proposed FlowID: " << std::hex
              << "0x" << flow_id;

    // Check if the flow ID is unique, and return it to the server.
    bool unique;
    {
      std::lock_guard<std::mutex> lock(bootstrap_fd_map_mu_);
      unique = (bootstrap_fd_map_.find(flow_id) == bootstrap_fd_map_.end());
      if (unique) bootstrap_fd_map_[flow_id] = bootstrap_fd;
    }

    ret = send_message(bootstrap_fd, &unique, sizeof(bool));
    DCHECK(ret == sizeof(bool));

    if (unique) break;
  }

  uint32_t local_ip_int = str_to_ip(local_ip_str_);
  int ret = send_message(bootstrap_fd, &local_ip_int, sizeof(local_ip_int));
  DCHECK(ret == sizeof(local_ip_int));

  uint32_t remote_ip_int;
  ret = receive_message(bootstrap_fd, &remote_ip_int, sizeof(remote_ip_int));
  DCHECK(ret == sizeof(remote_ip_int));
  std::string remote_ip = ip_to_str(remote_ip_int);

  LOG(INFO) << "[Endpoint] remote IP: " << remote_ip;

  // while (true);

  install_flow_on_engine(flow_id, remote_ip, local_engine_idx, bootstrap_fd);

  return ConnID{.flow_id = flow_id,
                .engine_idx = (uint32_t)local_engine_idx,
                .boostrap_id = bootstrap_fd};
}

ConnID Endpoint::uccl_accept() {
  struct sockaddr_in cli_addr;
  socklen_t clilen = sizeof(cli_addr);
  int bootstrap_fd;

  // Accept connection and set nonblocking and nodelay
  bootstrap_fd = accept(listen_fd_, (struct sockaddr*)&cli_addr, &clilen);
  DCHECK(bootstrap_fd >= 0);
  std::string client_ip = ip_to_str(cli_addr.sin_addr.s_addr);

  LOG(INFO) << "[Endpoint] accept from " << client_ip << ":"
            << cli_addr.sin_port;

  int flag = 1;
  setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void*)&flag, sizeof(int));

  auto local_engine_idx = find_least_loaded_engine_idx_and_update();
  CHECK_GE(local_engine_idx, 0);

  // Generate unique flow ID for both client and server.
  FlowID flow_id;
  while (true) {
    flow_id = U64Rand(0, std::numeric_limits<FlowID>::max());
    bool unique;
    {
      std::lock_guard<std::mutex> lock(bootstrap_fd_map_mu_);
      unique = (bootstrap_fd_map_.find(flow_id) == bootstrap_fd_map_.end());
      if (unique) {
        // Speculatively insert the flow ID.
        bootstrap_fd_map_[flow_id] = bootstrap_fd;
      } else {
        continue;
      }
    }

    LOG(INFO) << "[Endpoint] accept: propose FlowID: " << std::hex << "0x"
              << flow_id;

    // Ask client if this is unique
    int ret = send_message(bootstrap_fd, &flow_id, sizeof(FlowID));
    DCHECK(ret == sizeof(FlowID));
    bool unique_from_client;
    ret = receive_message(bootstrap_fd, &unique_from_client, sizeof(bool));
    DCHECK(ret == sizeof(bool));

    if (unique_from_client) {
      break;
    } else {
      // Remove the speculatively inserted flow ID.
      std::lock_guard<std::mutex> lock(bootstrap_fd_map_mu_);
      DCHECK(1 == bootstrap_fd_map_.erase(flow_id));
    }
  }

  uint32_t remote_ip_int;
  int ret =
      receive_message(bootstrap_fd, &remote_ip_int, sizeof(remote_ip_int));
  DCHECK(ret == sizeof(remote_ip_int));

  uint32_t local_ip_int = str_to_ip(local_ip_str_);
  ret = send_message(bootstrap_fd, &local_ip_int, sizeof(local_ip_int));
  DCHECK(ret == sizeof(local_ip_int));

  std::string remote_ip = ip_to_str(remote_ip_int);
  LOG(INFO) << "[Endpoint] remote IP: " << remote_ip;

  // while (true);

  install_flow_on_engine(flow_id, remote_ip, local_engine_idx, bootstrap_fd);

  return ConnID{.flow_id = flow_id,
                .engine_idx = (uint32_t)local_engine_idx,
                .boostrap_id = bootstrap_fd};
}

bool Endpoint::uccl_send(ConnID conn_id, void const* data, size_t const len,
                         bool busypoll) {
  auto* poll_ctx = uccl_send_async(conn_id, data, len);
  return busypoll ? uccl_poll(poll_ctx) : uccl_wait(poll_ctx);
}

bool Endpoint::uccl_recv(ConnID conn_id, void* data, size_t* len_p,
                         bool busypoll) {
  auto* poll_ctx = uccl_recv_async(conn_id, data, len_p);
  return busypoll ? uccl_poll(poll_ctx) : uccl_wait(poll_ctx);
}

PollCtx* Endpoint::uccl_send_async(ConnID conn_id, void const* data,
                                   size_t const len) {
  auto* poll_ctx = ctx_pool_->pop();
  Channel::Msg msg = {
      .opcode = Channel::Msg::Op::kTx,
      .flow_id = conn_id.flow_id,
      .data = const_cast<void*>(data),
      .len = len,
      .len_p = nullptr,
      .deser_msgs = nullptr,
      .poll_ctx = poll_ctx,
  };
  poll_ctx->write_barrier();
  Channel::enqueue_mp(channel_vec_[conn_id.engine_idx]->tx_task_q_, &msg);
  return poll_ctx;
}

PollCtx* Endpoint::uccl_recv_async(ConnID conn_id, void* data, size_t* len_p) {
  auto* poll_ctx = ctx_pool_->pop();
  Channel::Msg msg = {
      .opcode = Channel::Msg::Op::kRx,
      .flow_id = conn_id.flow_id,
      .data = data,
      .len = 0,
      .len_p = len_p,
      .deser_msgs = nullptr,
      .poll_ctx = poll_ctx,
  };
  Channel::enqueue_mp(channel_vec_[conn_id.engine_idx]->rx_task_q_, &msg);
  return poll_ctx;
}

bool Endpoint::uccl_wait(PollCtx* ctx) {
  {
    std::unique_lock<std::mutex> lock(ctx->mu);
    ctx->cv.wait(lock, [&ctx] { return ctx->done.load(); });
  }
  fence_and_clean_ctx(ctx);
  return true;
}

bool Endpoint::uccl_poll(PollCtx* ctx) {
  while (!uccl_poll_once(ctx));
  return true;
}

bool Endpoint::uccl_poll_once(PollCtx* ctx) {
  if (!ctx->done.load()) return false;
  fence_and_clean_ctx(ctx);
  return true;
}

void Endpoint::install_flow_on_engine(FlowID flow_id,
                                      std::string const& remote_ip,
                                      uint32_t local_engine_idx,
                                      int bootstrap_fd) {
  int ret;

  uint8_t local_mac_char[ETH_ALEN];
  std::string local_mac = local_mac_str_;
  VLOG(3) << "[Endpoint] local MAC: " << local_mac;
  str_to_mac(local_mac, local_mac_char);
  ret = send_message(bootstrap_fd, local_mac_char, ETH_ALEN);
  DCHECK(ret == ETH_ALEN);

  char remote_mac_char[ETH_ALEN];
  ret = receive_message(bootstrap_fd, remote_mac_char, ETH_ALEN);
  DCHECK(ret == ETH_ALEN);
  std::string remote_mac = mac_to_str(remote_mac_char);
  VLOG(3) << "[Endpoint] remote MAC: " << remote_mac;

  // Sync remote engine index.
  uint32_t remote_engine_idx;
  ret = send_message(bootstrap_fd, &local_engine_idx, sizeof(uint32_t));
  ret = receive_message(bootstrap_fd, &remote_engine_idx, sizeof(uint32_t));
  DCHECK(ret == sizeof(uint32_t));

  // Install flow and dst ports on engine.
  auto* poll_ctx = new PollCtx();
  Channel::CtrlMsg ctrl_msg = {};
  ctrl_msg.opcode = Channel::CtrlMsg::Op::kInstallFlow;
  ctrl_msg.flow_id = flow_id;
  ctrl_msg.remote_ip = htonl(str_to_ip(remote_ip));
  ctrl_msg.remote_engine_idx = remote_engine_idx;
  ctrl_msg.poll_ctx = poll_ctx;
  str_to_mac(remote_mac, ctrl_msg.remote_mac);
  Channel::enqueue_mp(channel_vec_[local_engine_idx]->ctrl_task_q_, &ctrl_msg);

  // Wait until the flow has been installed on the engine.
  {
    std::unique_lock<std::mutex> lock(poll_ctx->mu);
    poll_ctx->cv.wait(lock, [poll_ctx] { return poll_ctx->done.load(); });
  }
  delete poll_ctx;

  // sync so to receive flow_id packets.
  net_barrier(bootstrap_fd);
}

inline int Endpoint::find_least_loaded_engine_idx_and_update() {
  std::lock_guard<std::mutex> lock(engine_load_vec_mu_);
  if (engine_load_vec_.empty()) return -1;  // Handle empty vector case

  auto minElementIter =
      std::min_element(engine_load_vec_.begin(), engine_load_vec_.end());
  *minElementIter += 1;
  return std::distance(engine_load_vec_.begin(), minElementIter);
}

inline void Endpoint::fence_and_clean_ctx(PollCtx* ctx) {
  // Make the data written by the engine thread visible to the app thread.
  ctx->read_barrier();
  ctx_pool_->push(ctx);
}

void Endpoint::stats_thread_fn() {
  if (GetEnvVar("UCCL_ENGINE_QUIET") == "1") return;

  while (!shutdown_) {
    {
      std::unique_lock<std::mutex> lock(stats_mu_);
      bool shutdown =
          stats_cv_.wait_for(lock, std::chrono::seconds(kStatsTimerIntervalSec),
                             [this] { return shutdown_.load(); });
      if (shutdown) break;
    }
    if (engine_vec_.empty()) continue;

    uint32_t cnt = 0;
    std::string s;
    s += "\n\t[Uccl Engine] ";
    for (auto& engine : engine_vec_) {
      s += engine->status_to_string();
      if (++cnt == 2) break;
    }
    if (cnt < engine_vec_.size())
      s += Format("\n\t\t... %d more engines", engine_vec_.size() - cnt);
    LOG(INFO) << s;
  }
}

}  // namespace uccl