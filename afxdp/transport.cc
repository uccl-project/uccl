#include "transport.h"

namespace uccl {

void TXTracking::receive_acks(uint32_t num_acked_pkts) {
    VLOG(3) << "Received " << num_acked_pkts << " acks :"
            << " num_unsent_msgbufs_ " << num_unsent_msgbufs_
            << " last_msgbuf_ " << last_msgbuf_ << " oldest_unsent_msgbuf "
            << oldest_unsent_msgbuf_ << " oldest_unacked_msgbuf_ "
            << oldest_unacked_msgbuf_;
    DCHECK_LE(num_acked_pkts, num_tracked_msgbufs_);
    while (num_acked_pkts) {
        auto msgbuf = oldest_unacked_msgbuf_;
        DCHECK(msgbuf != nullptr);
        if (num_tracked_msgbufs_ > 1) {
            DCHECK_NE(msgbuf, last_msgbuf_) << "Releasing the last msgbuf!";
            DCHECK_NE(oldest_unacked_msgbuf_, oldest_unsent_msgbuf_)
                << "Releasing an unsent msgbuf!";
            oldest_unacked_msgbuf_ = msgbuf->next();
            DCHECK(oldest_unacked_msgbuf_ != nullptr) << num_acked_pkts;
        } else {
            oldest_unacked_msgbuf_ = nullptr;
            oldest_unsent_msgbuf_ = nullptr;
            last_msgbuf_ = nullptr;
            CHECK_EQ(num_tracked_msgbufs_, 1);
        }

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
        socket_->push_frame(msgbuf->get_frame_offset());

        num_unacked_msgbufs_--;
        num_tracked_msgbufs_--;
        num_acked_pkts--;
    }
}

void TXTracking::append(FrameBuf *msgbuf_head, FrameBuf *msgbuf_tail,
                        uint32_t num_frames, PollCtx *poll_ctx) {
    VLOG(3) << "Appending " << num_frames << " frames :"
            << " num_unsent_msgbufs_ " << num_unsent_msgbufs_
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
        oldest_unacked_msgbuf_ = msgbuf_head;
    } else {
        // This is not the first message buffer in the flow.
        DCHECK(oldest_unacked_msgbuf_ != nullptr)
            << oldest_unacked_msgbuf_->print_chain();
        // Let's enqueue the new message buffer at the end of the chain.
        last_msgbuf_->set_next(msgbuf_head);
        // Update the last buffer pointer to point to the current buffer.
        last_msgbuf_ = msgbuf_tail;
        if (oldest_unsent_msgbuf_ == nullptr)
            oldest_unsent_msgbuf_ = msgbuf_head;
    }

    num_unsent_msgbufs_ += num_frames;
    num_tracked_msgbufs_ += num_frames;
}

std::optional<FrameBuf *> TXTracking::get_and_update_oldest_unsent() {
    if (num_unsent_msgbufs_)
        VLOG(3) << "Getting: num_unsent_msgbufs_ " << num_unsent_msgbufs_
                << " last_msgbuf_ " << last_msgbuf_ << " oldest_unsent_msgbuf "
                << oldest_unsent_msgbuf_ << " oldest_unacked_msgbuf_ "
                << oldest_unacked_msgbuf_;
    if (oldest_unsent_msgbuf_ == nullptr) {
        DCHECK_EQ(num_unsent_msgbufs(), 0);
        return std::nullopt;
    }

    auto msgbuf = oldest_unsent_msgbuf_;
    if (oldest_unsent_msgbuf_ != last_msgbuf_) {
        oldest_unsent_msgbuf_ = oldest_unsent_msgbuf_->next();
    } else {
        oldest_unsent_msgbuf_ = nullptr;
    }

    num_unacked_msgbufs_++;
    num_unsent_msgbufs_--;
    return msgbuf;
}

RXTracking::ConsumeRet RXTracking::consume(swift::Pcb *pcb, FrameBuf *msgbuf) {
    uint8_t *pkt_addr = msgbuf->get_pkt_addr();
    auto frame_len = msgbuf->get_frame_len();
    const auto *ucclh =
        reinterpret_cast<const UcclPktHdr *>(pkt_addr + kNetHdrLen);
    const auto *payload = reinterpret_cast<const UcclPktHdr *>(
        pkt_addr + kNetHdrLen + kUcclHdrLen);
    const auto seqno = ucclh->seqno.value();
    const auto expected_seqno = pcb->rcv_nxt;

    if (swift::seqno_lt(seqno, expected_seqno)) {
        VLOG(3) << "Received old packet: " << seqno << " < " << expected_seqno;
        socket_->push_frame(msgbuf->get_frame_offset());
        return kOldPkt;
    }

    const size_t distance = seqno - expected_seqno;
    if (distance >= kReassemblyMaxSeqnoDistance) {
        VLOG(3) << "Packet too far ahead. Dropping as we can't handle SACK. "
                << "seqno: " << seqno << ", expected: " << expected_seqno;
        socket_->push_frame(msgbuf->get_frame_offset());
        return kOOOUntrackable;
    }

    // Only iterate through the deque if we must, i.e., for ooo packts only
    auto it = reass_q_.begin();
    if (seqno != expected_seqno) {
        it = reass_q_.lower_bound(seqno);
        if (it != reass_q_.end() && it->first == seqno) {
            VLOG(3) << "Received duplicate packet: " << seqno;
            // Duplicate packet. Drop it.
            socket_->push_frame(msgbuf->get_frame_offset());
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
    reass_q_.insert(
        it, std::pair<int, FrameBuf *>(static_cast<int>(seqno), msgbuf));

    // Update the SACK bitmap for the newly received packet.
    pcb->sack_bitmap_bit_set(distance);

    // These frames will be freed when the message is delivered to the app.
    push_inorder_msgbuf_to_app(pcb);

    return kOOOTrackableExpectedOrInOrder;
}

void RXTracking::push_inorder_msgbuf_to_app(swift::Pcb *pcb) {
    while (!reass_q_.empty() &&
           static_cast<uint32_t>(reass_q_.begin()->first) == pcb->rcv_nxt) {
        auto *msgbuf = reass_q_.begin()->second;
        reass_q_.erase(reass_q_.begin());

        // Stash this ready message in case application threads have not
        // supplied the app buffer while the engine keeps receiving messages.
        ready_msg_queue_.push_back(msgbuf);
        try_copy_msgbuf_to_appbuf(nullptr, nullptr, nullptr);

        pcb->advance_rcv_nxt();
        pcb->sack_bitmap_shift_left_one();
    }
}

void RXTracking::try_copy_msgbuf_to_appbuf(void *app_buf, size_t *app_buf_len_p,
                                           PollCtx *poll_ctx) {
    if (app_buf && app_buf_len_p && poll_ctx) {
        VLOG(3) << "ready_msg_queue_ size: " << ready_msg_queue_.size()
                << " app_buf_queue_ size: " << app_buf_queue_.size();
        app_buf_queue_.push_back({app_buf, app_buf_len_p, poll_ctx, 0});
    }

    while (!ready_msg_queue_.empty() && !app_buf_queue_.empty()) {
        FrameBuf *ready_msg = ready_msg_queue_.front();
        ready_msg_queue_.pop_front();
        DCHECK(ready_msg) << ready_msg->print_chain();

        auto &[app_buf, app_buf_len_p, poll_ctx, cur_offset] =
            app_buf_queue_.front();

        auto *pkt_addr = ready_msg->get_pkt_addr();
        DCHECK(pkt_addr) << "pkt_addr is nullptr when copy to app buf "
                         << std::hex << "0x" << ready_msg << std::dec
                         << ready_msg->to_string();
        auto *payload_addr = pkt_addr + kNetHdrLen + kUcclHdrLen;
        auto payload_len =
            ready_msg->get_frame_len() - kNetHdrLen - kUcclHdrLen;

        const auto *ucclh =
            reinterpret_cast<const UcclPktHdr *>(pkt_addr + kNetHdrLen);
        VLOG(2) << "payload_len: " << payload_len << " seqno: " << std::dec
                << ucclh->seqno.value();

#ifndef EMULATE_ZC
        memcpy((uint8_t *)app_buf + cur_offset, payload_addr, payload_len);
#endif
        cur_offset += payload_len;

        auto ready_frame_offset = ready_msg->get_frame_offset();

        // We have a complete message. Let's deliver it to the app.
        if (ready_msg->is_last()) {
            *app_buf_len_p = cur_offset;

            // Wakeup app thread waiting on endpoint.
            std::atomic_store_explicit(&poll_ctx->fence, true,
                                       std::memory_order_release);
            {
                std::lock_guard<std::mutex> lock(poll_ctx->mu);
                poll_ctx->done = true;
                poll_ctx->cv.notify_one();
            }

            app_buf_queue_.pop_front();

            VLOG(2) << "Received a complete message " << cur_offset << " bytes";
        }

        // Free received frames that have been copied to app buf.
        socket_->push_frame(ready_frame_offset);
    }
}

std::string UcclFlow::to_string() const {
    std::string s;
    s += "\n\t\t\t" + pcb_.to_string() +
         "\n\t\t\t[TX] pending msgbufs unsent: " +
         std::to_string(tx_tracking_.num_unsent_msgbufs()) +
         "\n\t\t\t[RX] ready msgs unconsumed: " +
         std::to_string(rx_tracking_.ready_msg_queue_.size());
    return s;
}

void UcclFlow::rx_messages() {
    VLOG(3) << "Received " << pending_rx_msgbufs_.size() << " packets";
    RXTracking::ConsumeRet consume_ret;
    uint32_t num_data_frames_recvd = 0;
    uint32_t path_id = 0;
    uint16_t dst_port = 0;
    uint16_t dst_port_rtt_probe = 0;
    uint64_t timestamp1 = 0, timestamp2 = 0;
    bool received_rtt_probe = false;

    for (auto msgbuf : pending_rx_msgbufs_) {
        // ebpf_transport has filtered out invalid pkts.
        auto *pkt_addr = msgbuf->get_pkt_addr();
        auto *udph = reinterpret_cast<udphdr *>(pkt_addr + sizeof(ethhdr) +
                                                sizeof(iphdr));
        auto *ucclh = reinterpret_cast<UcclPktHdr *>(pkt_addr + kNetHdrLen);
        auto *ucclsackh = reinterpret_cast<UcclSackHdr *>(
            reinterpret_cast<uint8_t *>(ucclh) + kUcclHdrLen);

        switch (ucclh->net_flags) {
            case UcclPktHdr::UcclFlags::kAckRttProbe:
                // Sender gets the RTT probe response, update the flow.
                process_rttprobe_rsp(ucclh->timestamp1, ucclh->timestamp2,
                                     ucclsackh->timestamp3,
                                     ucclsackh->timestamp4, ucclh->path_id);
            case UcclPktHdr::UcclFlags::kAck:
                // ACK packet, update the flow.
                process_ack(ucclh);
                // Free the received frame.
                socket_->push_frame(msgbuf->get_frame_offset());
                break;
            case UcclPktHdr::UcclFlags::kDataRttProbe:
                // Receiver gets the RTT probe, relay it back in the ACK.
                // If multiple RTT probe, we take the last one's timestamp.
                received_rtt_probe = true;
                path_id = ucclh->path_id;
                dst_port_rtt_probe = htons(udph->dest);
                timestamp1 = ucclh->timestamp1;
                timestamp2 = ucclh->timestamp2;
            case UcclPktHdr::UcclFlags::kData:
                // Data packet, process the payload. The frame will be freed
                // once the engine copies the payload into app buffer
                consume_ret = rx_tracking_.consume(&pcb_, msgbuf);
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
                    socket_->send_packet(
                        {msgbuf->get_frame_offset(), msgbuf->get_frame_len()});
                } else {
                    socket_->push_frame(msgbuf->get_frame_offset());
                }
                break;
            case UcclPktHdr::UcclFlags::kRssProbeRsp:
                // RSS probing rsp packet, ignore.
                LOG_EVERY_N(INFO, 10000)
                    << "[Flow] RSS probing rsp packet received, ignoring...";
                socket_->push_frame(msgbuf->get_frame_offset());
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
        if (rx_tracking_.ready_msg_queue_.size() <= kMaxReadyMsgbufs) {
            auto net_flags = received_rtt_probe
                                 ? UcclPktHdr::UcclFlags::kAckRttProbe
                                 : UcclPktHdr::UcclFlags::kAck;
            auto dst_port_reverse =
                received_rtt_probe ? dst_port_rtt_probe : dst_port;

            AFXDPSocket::frame_desc ack_frame = craft_ackpacket(
                path_id, dst_port_reverse, pcb_.seqno(), pcb_.ackno(),
                net_flags, timestamp1, timestamp2);
            socket_->send_packet(ack_frame);
        }
    }

    deserialize_and_append_to_txtracking();

    // Sending data frames that can be send per cwnd.
    transmit_pending_packets();
}

void UcclFlow::rx_supply_app_buf(Channel::Msg &rx_work) {
    rx_tracking_.try_copy_msgbuf_to_appbuf(rx_work.data, rx_work.len_p,
                                           rx_work.poll_ctx);
}

void UcclFlow::tx_messages(Channel::Msg &tx_work) {
    // This happens to NCCL plugin!!!
    if (tx_work.len == 0) {
        std::lock_guard<std::mutex> lock(tx_work.poll_ctx->mu);
        tx_work.poll_ctx->done = true;
        tx_work.poll_ctx->cv.notify_one();
        return;
    }

    pending_tx_msgs_.push_back({tx_work, 0});

    VLOG(3) << "tx_messages size: " << tx_work.len << " bytes";

    deserialize_and_append_to_txtracking();

    // Append these tx frames to the flow's tx queue, and trigger
    // intial tx. Future received ACKs will trigger more tx.
    transmit_pending_packets();
}

void UcclFlow::process_rttprobe_rsp(uint64_t ts1, uint64_t ts2, uint64_t ts3,
                                    uint64_t ts4, uint32_t path_id) {
    auto rtt_ns = (ts4 - ts1) - (ts3 - ts2);
    auto sample_rtt_tsc = ns_to_cycles(rtt_ns, freq_ghz);
#ifdef LATENCY_CC
    pcb_.timely_update_rate(rdtsc(), sample_rtt_tsc);
#endif
    port_path_rtt_[path_id] = sample_rtt_tsc;

    VLOG(3) << "sample_rtt_us " << to_usec(sample_rtt_tsc, freq_ghz)
            << " us, avg_rtt_diff " << pcb_.timely.get_avg_rtt_diff()
            << " us, timely rate " << pcb_.timely.get_rate_gbps() << " Gbps";

#ifdef RTT_STATS
    rtt_stats_.update(rtt_ns / 1000);
    if (++rtt_probe_count_ % 100000 == 0) {
        FILE *fp = fopen("rtt_stats.txt", "w");
        rtt_stats_.print(fp);
        fclose(fp);
    }
#endif
}

bool UcclFlow::periodic_check() {
    if (pcb_.rto_disabled()) return true;

    pcb_.rto_advance();

    // TODO(ilias): send RST packet, indicating removal of the flow.
    if (pcb_.max_rto_rexmits_consectutive_reached()) {
        return false;
    }

    if (pcb_.rto_expired()) {
        // Retransmit the oldest unacknowledged message buffer.
        rto_retransmit();
    }

    return true;
}

void UcclFlow::process_ack(const UcclPktHdr *ucclh) {
    const auto *ucclsackh = reinterpret_cast<const UcclSackHdr *>(
        reinterpret_cast<const uint8_t *>(ucclh) + kUcclHdrLen);
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
            auto *msgbuf = tx_tracking_.get_oldest_unacked_msgbuf();
            VLOG(2) << "Fast recovery " << ackno << " sack_bitmap_count "
                    << sack_bitmap_count;
            uint32_t index = 0;
            while (sack_bitmap_count && msgbuf && index < kSackBitmapSize) {
                const size_t sack_bitmap_bucket_idx =
                    index / swift::Pcb::kSackBitmapBucketSize;
                const size_t sack_bitmap_idx_in_bucket =
                    index % swift::Pcb::kSackBitmapBucketSize;
                auto sack_bitmap =
                    ucclsackh->sack_bitmap[sack_bitmap_bucket_idx].value();
                if ((sack_bitmap & (1ULL << sack_bitmap_idx_in_bucket)) == 0) {
                    // We found a missing packet.
                    auto seqno = pcb_.snd_una + index;

                    VLOG(2) << "Fast recovery retransmitting " << seqno;
                    const auto *missing_ucclh =
                        reinterpret_cast<const UcclPktHdr *>(
                            msgbuf->get_pkt_addr() + kNetHdrLen);
                    // TODO(yang): tmp fix---they should be equal, need to
                    // refine the way we maintain tx_but_unacked msgbufs chains.
                    if (seqno == missing_ucclh->seqno.value()) {
                        // DCHECK_EQ(seqno, missing_ucclh->seqno.value())
                        //     << " seqno mismatch at index " << index;
                        auto path_id = get_path_id_with_lowest_rtt();
                        set_path_id(seqno, path_id);
                        prepare_datapacket(msgbuf, path_id, seqno,
                                           UcclPktHdr::UcclFlags::kData);
                        msgbuf->mark_not_txpulltime_free();
                        missing_frames_.push_back({msgbuf->get_frame_offset(),
                                                   msgbuf->get_frame_len()});
                    }
                    pcb_.rto_reset();
                } else {
                    sack_bitmap_count--;
                }
                index++;
                msgbuf = msgbuf->next();
            }
            if (!missing_frames_.empty()) {
                // TODO(yang): handling the cases where the number of
                // missing frames is larger than the free send_queue size.
                socket_->send_packets(missing_frames_);
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

#ifndef LATENCY_CC
        uint32_t accumu_acks = 0;
        auto last_path_id = kPortEntropy;
        for (uint32_t seqno = pcb_.snd_una; seqno < ackno; seqno++) {
            auto path_id = get_path_id(seqno);
            if (path_id != last_path_id && last_path_id != kPortEntropy) {
                pcb_cc_[last_path_id].cubic_on_recv_ack(accumu_acks);
                accumu_acks = 0;
            }
            last_path_id = path_id;
            accumu_acks++;
        }
        if (accumu_acks) {
            pcb_cc_[last_path_id].cubic_on_recv_ack(accumu_acks);
        }
#endif

        pcb_.snd_una = ackno;
        pcb_.duplicate_acks = 0;
        pcb_.snd_ooo_acks = 0;
        pcb_.rto_rexmits_consectutive = 0;
        pcb_.rto_maybe_reset();
    }
}

void UcclFlow::fast_retransmit() {
    VLOG(3) << "Fast retransmitting oldest unacked packet " << pcb_.snd_una;
    // Retransmit the oldest unacknowledged message buffer.
    auto *msg_buf = tx_tracking_.get_oldest_unacked_msgbuf();
    if (msg_buf) {
        auto path_id = get_path_id_with_lowest_rtt();
        set_path_id(pcb_.snd_una, path_id);
        prepare_datapacket(msg_buf, path_id, pcb_.snd_una,
                           UcclPktHdr::UcclFlags::kData);
        const auto *ucclh = reinterpret_cast<const UcclPktHdr *>(
            msg_buf->get_pkt_addr() + kNetHdrLen);
        DCHECK_EQ(pcb_.snd_una, ucclh->seqno.value());
        msg_buf->mark_not_txpulltime_free();
        socket_->send_packet(
            {msg_buf->get_frame_offset(), msg_buf->get_frame_len()});
    }
    pcb_.rto_reset();
    pcb_.fast_rexmits++;
}

void UcclFlow::rto_retransmit() {
    VLOG(3) << "RTO retransmitting oldest unacked packet " << pcb_.snd_una;
    auto *msg_buf = tx_tracking_.get_oldest_unacked_msgbuf();
    if (msg_buf) {
        auto path_id = get_path_id_with_lowest_rtt();
        set_path_id(pcb_.snd_una, path_id);
        prepare_datapacket(msg_buf, path_id, pcb_.snd_una,
                           UcclPktHdr::UcclFlags::kData);
        msg_buf->mark_not_txpulltime_free();
        socket_->send_packet(
            {msg_buf->get_frame_offset(), msg_buf->get_frame_len()});
    }
#ifndef LATENCY_CC
    auto path_id = get_path_id(pcb_.snd_una);
    pcb_cc_[path_id].cubic_on_packet_loss();
#endif
    pcb_.rto_reset();
    pcb_.rto_rexmits++;
    pcb_.rto_rexmits_consectutive++;
}

/**
 * @brief Helper function to transmit a number of packets from the queue
 * of pending TX data.
 */
void UcclFlow::transmit_pending_packets() {
    // Avoid sending too many packets.
    auto num_unacked_pkts = tx_tracking_.num_unacked_msgbufs();
    if (num_unacked_pkts >= MAX_UNACKED_PKTS) return;

    auto unacked_pkt_budget = MAX_UNACKED_PKTS - num_unacked_pkts;
    auto txq_free_entries =
        socket_->send_queue_free_entries(unacked_pkt_budget);
    auto hard_budget = std::min(txq_free_entries, unacked_pkt_budget);

    // Choosing a path to send a batch of packets.
    auto path_id = get_path_id_with_lowest_rtt();

#ifdef LATENCY_CC
    auto permitted_packets = pcb_.timely_ready_packets(hard_budget);
#else
    // TODO(yang): control the size of batch.
    auto &pcb_cc = pcb_cc_[path_id];
    auto permitted_packets =
        std::min(hard_budget, pcb_cc.cubic_effective_wnd());
#endif

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
    for (uint32_t i = 0; i < permitted_packets; i++) {
        auto msg_buf_opt = tx_tracking_.get_and_update_oldest_unsent();
        if (!msg_buf_opt.has_value()) break;

        auto *msg_buf = msg_buf_opt.value();
        auto seqno = pcb_.get_snd_nxt();
        set_path_id(seqno, path_id);

        if (msg_buf->is_last()) {
            VLOG(2) << "Transmitting seqno: " << seqno << " payload_len: "
                    << msg_buf->get_frame_len() - kNetHdrLen - kUcclHdrLen;
        }
        auto net_flags = (i == 0) ? UcclPktHdr::UcclFlags::kDataRttProbe
                                  : UcclPktHdr::UcclFlags::kData;
        prepare_datapacket(msg_buf, path_id, seqno, net_flags);
        msg_buf->mark_not_txpulltime_free();
        pending_tx_frames_.push_back(
            {msg_buf->get_frame_offset(), msg_buf->get_frame_len()});
    }

    // TX both data and ack frames.
    if (pending_tx_frames_.empty()) return;
    VLOG(3) << "tx packets " << pending_tx_frames_.size();

    socket_->send_packets(pending_tx_frames_);
    pending_tx_frames_.clear();

    if (pcb_.rto_disabled()) pcb_.rto_enable();
}

void UcclFlow::deserialize_and_append_to_txtracking() {
    if (pending_tx_msgs_.empty()) return;
    if (tx_tracking_.num_unsent_msgbufs() >= MAX_TIMING_WHEEL_PKTS) return;

    auto &[tx_work, cur_offset] = pending_tx_msgs_.front();
    auto deser_budget =
        MAX_TIMING_WHEEL_PKTS - tx_tracking_.num_unsent_msgbufs();

    FrameBuf *tx_msgbuf_head = nullptr;
    FrameBuf *tx_msgbuf_tail = nullptr;
    uint32_t num_tx_frames = 0;

    auto remaining_bytes = tx_work.len - cur_offset;
    auto *app_buf_cursor = (uint8_t *)tx_work.data + cur_offset;

    auto now_tsc = rdtsc();
    FrameBuf *last_msgbuf = nullptr;
    while (remaining_bytes > 0 && num_tx_frames < deser_budget) {
        //  Deserializing the message into MTU-sized frames.
        auto payload_len = std::min(
            remaining_bytes, (size_t)AFXDP_MTU - kNetHdrLen - kUcclHdrLen);

        // Get a frame from the socket.
        auto frame_offset = socket_->pop_frame();
        auto *msgbuf = FrameBuf::Create(frame_offset, socket_->umem_buffer_,
                                        kNetHdrLen + kUcclHdrLen + payload_len);

        if (remaining_bytes == tx_work.len) msgbuf->mark_first();

        // The flow will free these Tx frames when receiving ACKs.
        msgbuf->mark_not_txpulltime_free();

#ifdef LATENCY_CC
        // Queue on the timing wheel.
        pcb_.timely_pace_packet(now_tsc, payload_len + kNetHdrLen + kUcclHdrLen,
                                msgbuf);
#endif

#ifndef EMULATE_ZC
        // VLOG(3) << "Deser copy " << msgbuf << " " << num_tx_frames;
        auto pkt_payload_addr =
            msgbuf->get_pkt_addr() + kNetHdrLen + kUcclHdrLen;
        memcpy(pkt_payload_addr, app_buf_cursor, payload_len);
#endif
        remaining_bytes -= payload_len;
        app_buf_cursor += payload_len;

        if (tx_msgbuf_head == nullptr) {
            tx_msgbuf_head = msgbuf;
        } else {
            last_msgbuf->set_next(msgbuf);
        }

        if (remaining_bytes == 0) {
            msgbuf->mark_last();
            msgbuf->set_next(nullptr);
        }

        last_msgbuf = msgbuf;
        num_tx_frames++;
    }
    tx_msgbuf_tail = last_msgbuf;
    if (tx_msgbuf_tail) tx_msgbuf_tail->set_next(nullptr);

    // LOG_EVERY_N(INFO, 10000)
    //     << "deser unsent_msgbufs " << tx_tracking_.num_unsent_msgbufs()
    //     << " deser_budget " << deser_budget << " pending_tx_msgs "
    //     << pending_tx_msgs_.size() << " successfully added to timingwheel "
    //     << num_tx_frames << " tx_tracking poll_ctxs "
    //     << tx_tracking_.poll_ctxs_.size();

    // This message has been fully deserialized and added to tx tracking.
    if (remaining_bytes == 0)
        pending_tx_msgs_.pop_front();
    else
        cur_offset = tx_work.len - remaining_bytes;

    tx_tracking_.append(
        tx_msgbuf_head, tx_msgbuf_tail, num_tx_frames,
        (tx_msgbuf_head && tx_msgbuf_head->is_first() && num_tx_frames)
            ? tx_work.poll_ctx
            : nullptr);

    // Recursively call this function to append more messages to the tx.
    if (tx_tracking_.num_unsent_msgbufs() < MAX_TIMING_WHEEL_PKTS &&
        !pending_tx_msgs_.empty()) {
        deserialize_and_append_to_txtracking();
    }
}

void UcclFlow::prepare_l2header(uint8_t *pkt_addr) const {
    auto *eh = (ethhdr *)pkt_addr;
    memcpy(eh->h_source, local_l2_addr_, ETH_ALEN);
    memcpy(eh->h_dest, remote_l2_addr_, ETH_ALEN);
    eh->h_proto = htons(ETH_P_IP);
}

void UcclFlow::prepare_l3header(uint8_t *pkt_addr,
                                uint32_t payload_bytes) const {
    auto *ipv4h = (iphdr *)(pkt_addr + sizeof(ethhdr));
    ipv4h->ihl = 5;
    ipv4h->version = 4;
    ipv4h->tos = 0x0;
    ipv4h->id = htons(0x1513);
    ipv4h->frag_off = htons(0);
    ipv4h->ttl = 64;
#ifdef USE_TCP
    ipv4h->protocol = IPPROTO_TCP;
    ipv4h->tot_len = htons(sizeof(iphdr) + sizeof(tcphdr) + payload_bytes);
#else
    ipv4h->protocol = IPPROTO_UDP;
    ipv4h->tot_len = htons(sizeof(iphdr) + sizeof(udphdr) + payload_bytes);
#endif
    ipv4h->saddr = htonl(local_addr_);
    ipv4h->daddr = htonl(remote_addr_);
    ipv4h->check = 0;
    // AWS would block traffic if ipv4 checksum is not calculated.
    ipv4h->check = ipv4_checksum(ipv4h, sizeof(iphdr));
}

void UcclFlow::prepare_l4header(uint8_t *pkt_addr, uint32_t payload_bytes,
                                uint16_t dst_port) const {
#ifdef USE_TCP
    auto *tcph = (tcphdr *)(pkt_addr + sizeof(ethhdr) + sizeof(iphdr));
    memset(tcph, 0, sizeof(tcphdr));
#ifdef USE_MULTIPATH
    tcph->source = htons(BASE_PORT);
    tcph->dest = htons(dst_port);
#else
    tcph->source = htons(BASE_PORT);
    tcph->dest = htons(BASE_PORT);
#endif
    tcph->doff = 5;
    tcph->check = 0;
#ifdef ENABLE_CSUM
    tcph->check = ipv4_udptcp_cksum(IPPROTO_TCP, local_addr_, remote_addr_,
                                    sizeof(tcphdr) + payload_bytes, tcph);
#endif
#else
    auto *udph = (udphdr *)(pkt_addr + sizeof(ethhdr) + sizeof(iphdr));
#ifdef USE_MULTIPATH
    udph->source = htons(BASE_PORT);
    udph->dest = htons(dst_port);
#else
    udph->source = htons(BASE_PORT);
    udph->dest = htons(BASE_PORT);
#endif
    udph->len = htons(sizeof(udphdr) + payload_bytes);
    udph->check = htons(0);
#ifdef ENABLE_CSUM
    udph->check = ipv4_udptcp_cksum(IPPROTO_UDP, local_addr_, remote_addr_,
                                    sizeof(udphdr) + payload_bytes, udph);
#endif
#endif
}

void UcclFlow::prepare_datapacket(FrameBuf *msg_buf, uint32_t path_id,
                                  uint32_t seqno,
                                  const UcclPktHdr::UcclFlags net_flags) {
    // Header length after before the payload.
    uint32_t frame_len = msg_buf->get_frame_len();
    DCHECK_LE(frame_len, AFXDP_MTU);
    uint8_t *pkt_addr = msg_buf->get_pkt_addr();

    // Prepare network headers.
    prepare_l2header(pkt_addr);
    prepare_l3header(pkt_addr, frame_len - kNetHdrLen);
    prepare_l4header(pkt_addr, frame_len - kNetHdrLen, dst_ports_[path_id]);

    // Prepare the Uccl-specific header.
    auto *ucclh = reinterpret_cast<UcclPktHdr *>(pkt_addr + kNetHdrLen);
    ucclh->magic = be16_t(UcclPktHdr::kMagic);
    ucclh->engine_id = remote_engine_idx_;
    ucclh->path_id = (uint8_t)path_id;
    ucclh->net_flags = net_flags;
    ucclh->ackno = be32_t(UINT32_MAX);
    // This fills the FrameBuf flags into the outgoing packet msg_flags.
    ucclh->msg_flags = msg_buf->msg_flags();
    ucclh->frame_len = be16_t(frame_len);

    ucclh->seqno = be32_t(seqno);
    ucclh->flow_id = be64_t(flow_id_);

    ucclh->timestamp1 = (net_flags == UcclPktHdr::UcclFlags::kDataRttProbe)
                            ? get_monotonic_time_ns() +
                                  socket_->send_queue_estimated_latency_ns()
                            : 0;
    ucclh->timestamp2 = 0;  // let the receiver ebpf fill this in.
}

AFXDPSocket::frame_desc UcclFlow::craft_ackpacket(
    uint32_t path_id, uint16_t dst_port, uint32_t seqno, uint32_t ackno,
    const UcclPktHdr::UcclFlags net_flags, uint64_t ts1, uint64_t ts2) {
    const size_t kControlPayloadBytes = kUcclHdrLen + kUcclSackHdrLen;
    auto frame_offset = socket_->pop_frame();
    auto msgbuf = FrameBuf::Create(frame_offset, socket_->umem_buffer_,
                                   kNetHdrLen + kControlPayloadBytes);
    // Let AFXDPSocket::pull_complete_queue() free control frames.
    msgbuf->mark_txpulltime_free();

    uint8_t *pkt_addr = (uint8_t *)socket_->umem_buffer_ + frame_offset;
    prepare_l2header(pkt_addr);
    prepare_l3header(pkt_addr, kControlPayloadBytes);
    prepare_l4header(pkt_addr, kControlPayloadBytes, dst_port);

    auto *ucclh = (UcclPktHdr *)(pkt_addr + kNetHdrLen);
    ucclh->magic = be16_t(UcclPktHdr::kMagic);
    ucclh->engine_id = remote_engine_idx_;
    ucclh->path_id = (uint8_t)path_id;
    ucclh->net_flags = net_flags;
    ucclh->msg_flags = 0;
    ucclh->frame_len = be16_t(kNetHdrLen + kControlPayloadBytes);
    ucclh->seqno = be32_t(seqno);
    ucclh->ackno = be32_t(ackno);
    ucclh->flow_id = be64_t(flow_id_);
    ucclh->timestamp1 = ts1;
    ucclh->timestamp2 = ts2;

    auto *ucclsackh = (UcclSackHdr *)(pkt_addr + kNetHdrLen + kUcclHdrLen);

    for (size_t i = 0; i < sizeof(UcclSackHdr::sack_bitmap) /
                               sizeof(UcclSackHdr::sack_bitmap[0]);
         ++i) {
        ucclsackh->sack_bitmap[i] = be64_t(pcb_.sack_bitmap[i]);
    }
    ucclsackh->sack_bitmap_count = be16_t(pcb_.sack_bitmap_count);

    ucclsackh->timestamp3 = (net_flags == UcclPktHdr::UcclFlags::kAckRttProbe)
                                ? get_monotonic_time_ns() +
                                      socket_->send_queue_estimated_latency_ns()
                                : 0;
    ucclsackh->timestamp4 = 0;  // let the sender ebpf fill this in.

    return {frame_offset, kNetHdrLen + kControlPayloadBytes};
}

AFXDPSocket::frame_desc UcclFlow::craft_rssprobe_packet(uint16_t dst_port) {
    const size_t kRssProbePayloadBytes = kUcclHdrLen;
    auto frame_offset = socket_->pop_frame();
    auto msgbuf = FrameBuf::Create(frame_offset, socket_->umem_buffer_,
                                   kNetHdrLen + kRssProbePayloadBytes);
    // Let AFXDPSocket::pull_complete_queue() free control frames.
    msgbuf->mark_txpulltime_free();

    uint8_t *pkt_addr = (uint8_t *)socket_->umem_buffer_ + frame_offset;
    prepare_l2header(pkt_addr);
    prepare_l3header(pkt_addr, kRssProbePayloadBytes);
    prepare_l4header(pkt_addr, kRssProbePayloadBytes, dst_port);

    auto *udph = (udphdr *)(pkt_addr + sizeof(ethhdr) + sizeof(iphdr));
    udph->dest = htons(dst_port);

    auto *ucclh = (UcclPktHdr *)(pkt_addr + kNetHdrLen);
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

    return {frame_offset, kNetHdrLen + kRssProbePayloadBytes};
}

void UcclFlow::reverse_packet_l2l3(FrameBuf *msg_buf) {
    auto *pkt_addr = msg_buf->get_pkt_addr();
    auto *eth = (ethhdr *)pkt_addr;
    auto *ipv4h = (iphdr *)(pkt_addr + sizeof(ethhdr));
    auto *udp = (udphdr *)(pkt_addr + sizeof(ethhdr) + sizeof(iphdr));

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
    Channel::Msg tx_work;

    while (!shutdown_) {
        // Calculate the cycles elapsed since last periodic processing.
        auto now_tsc = rdtsc();
        const auto elapsed_tsc = now_tsc - last_periodic_tsc_;

        if (elapsed_tsc >= kSlowTimerIntervalTsc_) {
            // Perform periodic processing.
            periodic_process();
            last_periodic_tsc_ = now_tsc;
        }

        if (jring_sc_dequeue_bulk(channel_->rx_cmdq_, &rx_work, 1, nullptr) ==
            1) {
            VLOG(3) << "Rx jring dequeue";
            active_flows_map_[rx_work.flow_id]->rx_supply_app_buf(rx_work);
        }

        auto frames = socket_->recv_packets(RECV_BATCH_SIZE);
        if (frames.size()) process_rx_msg(frames);

        if (jring_sc_dequeue_bulk(channel_->tx_cmdq_, &tx_work, 1, nullptr) ==
            1) {
            // Make data written by the app thread visible to the engine.
            std::ignore = std::atomic_load_explicit(&tx_work.poll_ctx->fence,
                                                    std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);

            VLOG(3) << "Tx jring dequeue";
            active_flows_map_[tx_work.flow_id]->tx_messages(tx_work);
        }

        for (auto &[flow_id, flow] : active_flows_map_) {
            flow->transmit_pending_packets();
        }
    }

    // This will reset flow pcb state.
    for (auto [flow_id, flow] : active_flows_map_) {
        flow->shutdown();
        delete flow;
    }
    // This will flush all unpolled tx frames.
    socket_->shutdown();
}

void UcclEngine::process_rx_msg(
    std::vector<AFXDPSocket::frame_desc> &pkt_msgs) {
    for (auto &frame : pkt_msgs) {
        auto *msgbuf = FrameBuf::Create(frame.frame_offset,
                                        socket_->umem_buffer_, frame.frame_len);
        auto *pkt_addr = msgbuf->get_pkt_addr();
        auto *ucclh = reinterpret_cast<UcclPktHdr *>(pkt_addr + kNetHdrLen);

        // Record the incoming packet UcclPktHdr.msg_flags in
        // FrameBuf.
        msgbuf->set_msg_flags(ucclh->msg_flags);

        /**
         * Yang: Work around an AFXDP bug that would receive the same packets
         * with differet lengths multiple times under high traffic volume. This
         * is likely caused by race conditions in the the kernel:
         * https://blog.cloudflare.com/a-debugging-story-corrupt-packets-in-af_xdp-kernel-bug-or-user-error/
         */
        if (ucclh->frame_len.value() != frame.frame_len) {
            VLOG(3) << "Received invalid frame length: "
                    << "xdp_desc->len " << frame.frame_len
                    << " ucclh->frame_len " << ucclh->frame_len.value()
                    << " net_flags " << std::bitset<8>((int)ucclh->net_flags)
                    << " seqno " << ucclh->seqno.value() << " ackno "
                    << ucclh->ackno.value() << " flow_id " << std::hex << "0x"
                    << ucclh->flow_id.value() << " engine_id "
                    << (int)ucclh->engine_id;
            socket_->push_frame(frame.frame_offset);
            continue;
        }

        if (msgbuf->is_last()) {
            VLOG(2) << "Received seqno: " << ucclh->seqno.value()
                    << " payload_len: "
                    << msgbuf->get_frame_len() - kNetHdrLen - kUcclHdrLen;
        }

        auto flow_id = ucclh->flow_id.value();

        auto it = active_flows_map_.find(flow_id);
        if (it == active_flows_map_.end()) {
            LOG_EVERY_N(ERROR, 1000000) << "process_rx_msg unknown flow "
                                        << std::hex << "0x" << flow_id;
            for (auto [flow_id, flow] : active_flows_map_) {
                LOG_EVERY_N(ERROR, 1000000) << "                active flow "
                                            << std::hex << "0x" << flow_id;
            }
            socket_->push_frame(msgbuf->get_frame_offset());
            continue;
        }
        it->second->pending_rx_msgbufs_.push_back(msgbuf);
    }
    for (auto &[flow_id, flow] : active_flows_map_) {
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
    if (jring_sc_dequeue_bulk(channel_->ctrl_cmdq_, &ctrl_work, 1, nullptr) ==
        1) {
        switch (ctrl_work.opcode) {
            case Channel::CtrlMsg::kInstallFlow:
                handle_install_flow_on_engine(ctrl_work);
                break;
            default:
                break;
        }
    }
}

void UcclEngine::handle_install_flow_on_engine(Channel::CtrlMsg &ctrl_work) {
    LOG(INFO) << "[Engine] handle_install_flow_on_engine " << local_engine_idx_;
    int ret;
    std::string local_ip_str = ip_to_str(htonl(local_addr_));
    auto flow_id = ctrl_work.flow_id;
    auto remote_addr = ctrl_work.remote_ip;
    std::string remote_ip_str = ip_to_str(htonl(remote_addr));
    auto remote_mac_char = ctrl_work.remote_mac;
    auto remote_engine_idx = ctrl_work.remote_engine_idx;
    auto *poll_ctx = ctrl_work.poll_ctx;

    auto *flow = new UcclFlow(local_addr_, remote_addr, local_l2_addr_,
                              remote_mac_char, local_engine_idx_,
                              remote_engine_idx, socket_, channel_, flow_id);
    std::tie(std::ignore, ret) = active_flows_map_.insert({flow_id, flow});
    DCHECK(ret);

    // RSS probing to get a list of dst_port matching remote engine queue and,
    // reversely, matching local engine queue. Basically, symmetric dst_ports.
    std::set<uint16_t> dst_ports_set;
    for (int i = BASE_PORT; i < 65536;
         i = (i + 1) % (65536 - BASE_PORT) + BASE_PORT) {
        uint16_t dst_port = i;
        auto frame = flow->craft_rssprobe_packet(dst_port);
        socket_->send_packet(frame);

        auto frames = socket_->recv_packets(RECV_BATCH_SIZE);
        for (auto &frame : frames) {
            auto *msgbuf = FrameBuf::Create(
                frame.frame_offset, socket_->umem_buffer_, frame.frame_len);
            auto *pkt_addr = msgbuf->get_pkt_addr();
            auto *udph = reinterpret_cast<udphdr *>(pkt_addr + sizeof(ethhdr) +
                                                    sizeof(iphdr));
            auto *ucclh = reinterpret_cast<UcclPktHdr *>(pkt_addr + kNetHdrLen);

            if (ucclh->net_flags == UcclPktHdr::UcclFlags::kRssProbe) {
                VLOG(3) << "[Engine] received RSS probe packet";
                if (ucclh->engine_id == local_engine_idx_) {
                    // Probe packets arrive the remote engine!
                    ucclh->net_flags = UcclPktHdr::UcclFlags::kRssProbeRsp;
                    ucclh->engine_id = remote_engine_idx;
                    msgbuf->mark_txpulltime_free();
                    // Reverse so to send back
                    flow->reverse_packet_l2l3(msgbuf);
                    socket_->send_packet(
                        {msgbuf->get_frame_offset(), msgbuf->get_frame_len()});
                } else {
                    socket_->push_frame(frame.frame_offset);
                }
            } else {
                VLOG(3) << "[Engine] received RSS probe rsp packet";
                DCHECK(ucclh->net_flags == UcclPktHdr::UcclFlags::kRssProbeRsp);
                if (ucclh->engine_id == local_engine_idx_) {
                    // Probe rsp packets arrive this engine!
                    dst_ports_set.insert(ntohs(udph->dest));
                }
                socket_->push_frame(frame.frame_offset);
            }
        }
        if (dst_ports_set.size() >= kPortEntropy) break;
    }

    LOG(INFO) << "[Engine] handle_install_flow_on_engine dst_ports size: "
              << dst_ports_set.size();
    DCHECK_GE(dst_ports_set.size(), kPortEntropy);

    flow->dst_ports_.reserve(kPortEntropy);
    auto it = dst_ports_set.begin();
    std::advance(it, kPortEntropy);
    std::copy(dst_ports_set.begin(), it, std::back_inserter(flow->dst_ports_));

    LOG(INFO) << "[Engine] install FlowID " << std::hex << "0x" << flow_id
              << ": " << local_ip_str << Format("(%d)", local_engine_idx_)
              << " <-> " << remote_ip_str << Format("(%d)", remote_engine_idx);

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
        s += Format("\n\t\tEngine %d Flow 0x%lx: %s (%u) <-> %s (%u)",
                    local_engine_idx_, flow_id,
                    ip_to_str(htonl(flow->local_addr_)).c_str(),
                    flow->local_engine_idx_,
                    ip_to_str(htonl(flow->remote_addr_)).c_str(),
                    flow->remote_engine_idx_);
        s += flow->to_string();
    }
    s += socket_->to_string();
    return s;
}

Endpoint::Endpoint(const char *interface_name, int num_queues,
                   uint64_t num_frames, int engine_cpu_start)
    : num_queues_(num_queues), stats_thread_([this]() { stats_thread_fn(); }) {
    // Create UDS socket and get the umem_id.
    static std::once_flag flag_once;
    std::call_once(flag_once, [interface_name, num_frames]() {
        AFXDPFactory::init(interface_name, num_frames, "ebpf_transport.o",
                           "ebpf_transport");
    });

    local_ip_str_ = get_dev_ip(interface_name);
    local_mac_str_ = get_dev_mac(interface_name);

    CHECK_LE(num_queues, NUM_CPUS / 4)
        << "num_queues should be less than or equal to the number of CPUs / 4";

    // Create multiple engines, each got its xsk and umem from the
    // daemon. Each engine has its own thread and channel to let the endpoint
    // communicate with.
    for (int i = 0; i < num_queues; i++) channel_vec_[i] = new Channel();

    for (int queue_id = 0, engine_cpu_id = engine_cpu_start;
         queue_id < num_queues; queue_id++, engine_cpu_id++) {
        engine_vec_.emplace_back(std::make_unique<UcclEngine>(
            queue_id, channel_vec_[queue_id], local_ip_str_, local_mac_str_));
        engine_th_vec_.emplace_back(std::make_unique<std::thread>(
            [engine_ptr = engine_vec_.back().get(), queue_id, engine_cpu_id]() {
                LOG(INFO) << "[Engine] thread " << queue_id
                          << " running on CPU " << engine_cpu_id;
                pin_thread_to_cpu(engine_cpu_id);
                engine_ptr->run();
            }));
    }

    ctx_pool_ = new SharedPool<PollCtx *, true>(kMaxInflightMsg);
    ctx_pool_buf_ = new uint8_t[kMaxInflightMsg * sizeof(PollCtx)];
    for (int i = 0; i < kMaxInflightMsg; i++) {
        ctx_pool_->push(new (ctx_pool_buf_ + i * sizeof(PollCtx)) PollCtx());
    }

    // Create listening socket
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    DCHECK(listen_fd_ >= 0) << "ERROR: opening socket";

    int flag = 1;
    DCHECK(setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &flag,
                      sizeof(int)) >= 0)
        << "ERROR: setsockopt SO_REUSEADDR fails";

    struct sockaddr_in serv_addr;
    bzero((char *)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(kBootstrapPort);
    DCHECK(bind(listen_fd_, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) >=
           0)
        << "ERROR: binding";

    DCHECK(!listen(listen_fd_, 128)) << "ERROR: listen";
    LOG(INFO) << "[Endpoint] server ready, listening on port "
              << kBootstrapPort;
}

Endpoint::~Endpoint() {
    for (auto &engine : engine_vec_) engine->shutdown();
    for (auto &engine_th : engine_th_vec_) engine_th->join();
    for (int i = 0; i < num_queues_; i++) delete channel_vec_[i];

    delete ctx_pool_;
    delete[] ctx_pool_buf_;

    close(listen_fd_);

    {
        std::lock_guard<std::mutex> lock(bootstrap_fd_map_mu_);
        for (auto &[flow_id, boostrap_id] : bootstrap_fd_map_) {
            close(boostrap_id);
        }
    }

    static std::once_flag flag_once;
    std::call_once(flag_once, []() { AFXDPFactory::shutdown(); });

    {
        std::lock_guard<std::mutex> lock(stats_mu_);
        shutdown_ = true;
        stats_cv_.notify_all();
    }
    stats_thread_.join();
}

ConnID Endpoint::uccl_connect(std::string remote_ip) {
    struct sockaddr_in serv_addr;
    struct hostent *server;
    int bootstrap_fd;

    bootstrap_fd = socket(AF_INET, SOCK_STREAM, 0);
    DCHECK(bootstrap_fd >= 0);

    server = gethostbyname(remote_ip.c_str());
    DCHECK(server);

    bzero((char *)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr,
          server->h_length);
    serv_addr.sin_port = htons(kBootstrapPort);

    // Force the socket to bind to the local IP address.
    sockaddr_in localaddr = {0};
    localaddr.sin_family = AF_INET;
    localaddr.sin_addr.s_addr = str_to_ip(local_ip_str_.c_str());
    bind(bootstrap_fd, (sockaddr *)&localaddr, sizeof(localaddr));

    LOG(INFO) << "[Endpoint] connecting to " << remote_ip << ":"
              << kBootstrapPort;

    // Connect and set nonblocking and nodelay
    while (connect(bootstrap_fd, (struct sockaddr *)&serv_addr,
                   sizeof(serv_addr))) {
        LOG(INFO) << "[Endpoint] connecting... Make sure the server is up.";
        sleep(1);
    }

    fcntl(bootstrap_fd, F_SETFL, O_NONBLOCK);
    int flag = 1;
    setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag,
               sizeof(int));

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
            unique =
                (bootstrap_fd_map_.find(flow_id) == bootstrap_fd_map_.end());
            if (unique) bootstrap_fd_map_[flow_id] = bootstrap_fd;
        }

        ret = send_message(bootstrap_fd, &unique, sizeof(bool));
        DCHECK(ret == sizeof(bool));

        if (unique) break;
    }

    install_flow_on_engine(flow_id, remote_ip, local_engine_idx, bootstrap_fd);

    return ConnID{.flow_id = flow_id,
                  .engine_idx = (uint32_t)local_engine_idx,
                  .boostrap_id = bootstrap_fd};
}

ConnID Endpoint::uccl_accept(std::string &remote_ip) {
    struct sockaddr_in cli_addr;
    socklen_t clilen = sizeof(cli_addr);
    int bootstrap_fd;

    // Accept connection and set nonblocking and nodelay
    bootstrap_fd = accept(listen_fd_, (struct sockaddr *)&cli_addr, &clilen);
    DCHECK(bootstrap_fd >= 0);
    remote_ip = ip_to_str(cli_addr.sin_addr.s_addr);

    LOG(INFO) << "[Endpoint] accept from " << remote_ip << ":"
              << cli_addr.sin_port;

    fcntl(bootstrap_fd, F_SETFL, O_NONBLOCK);
    int flag = 1;
    setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag,
               sizeof(int));

    auto local_engine_idx = find_least_loaded_engine_idx_and_update();
    CHECK_GE(local_engine_idx, 0);

    // Generate unique flow ID for both client and server.
    FlowID flow_id;
    while (true) {
        flow_id = U64Rand(0, std::numeric_limits<FlowID>::max());
        bool unique;
        {
            std::lock_guard<std::mutex> lock(bootstrap_fd_map_mu_);
            unique =
                (bootstrap_fd_map_.find(flow_id) == bootstrap_fd_map_.end());
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

    install_flow_on_engine(flow_id, remote_ip, local_engine_idx, bootstrap_fd);

    return ConnID{.flow_id = flow_id,
                  .engine_idx = (uint32_t)local_engine_idx,
                  .boostrap_id = bootstrap_fd};
}

bool Endpoint::uccl_send(ConnID conn_id, const void *data, const size_t len,
                         bool busypoll) {
    auto *poll_ctx = uccl_send_async(conn_id, data, len);
    return busypoll ? uccl_poll(poll_ctx) : uccl_wait(poll_ctx);
}

bool Endpoint::uccl_recv(ConnID conn_id, void *data, size_t *len_p,
                         bool busypoll) {
    auto *poll_ctx = uccl_recv_async(conn_id, data, len_p);
    return busypoll ? uccl_poll(poll_ctx) : uccl_wait(poll_ctx);
}

PollCtx *Endpoint::uccl_send_async(ConnID conn_id, const void *data,
                                   const size_t len) {
    auto *poll_ctx = ctx_pool_->pop();
    Channel::Msg msg = {
        .opcode = Channel::Msg::Op::kTx,
        .flow_id = conn_id.flow_id,
        .data = const_cast<void *>(data),
        .len = len,
        .len_p = nullptr,
        .poll_ctx = poll_ctx,
    };
    std::atomic_store_explicit(&poll_ctx->fence, true,
                               std::memory_order_release);
    while (jring_mp_enqueue_bulk(channel_vec_[conn_id.engine_idx]->tx_cmdq_,
                                 &msg, 1, nullptr) != 1);
    return poll_ctx;
}

PollCtx *Endpoint::uccl_recv_async(ConnID conn_id, void *data, size_t *len_p) {
    auto *poll_ctx = ctx_pool_->pop();
    Channel::Msg msg = {
        .opcode = Channel::Msg::Op::kRx,
        .flow_id = conn_id.flow_id,
        .data = data,
        .len = 0,
        .len_p = len_p,
        .poll_ctx = poll_ctx,
    };
    while (jring_mp_enqueue_bulk(channel_vec_[conn_id.engine_idx]->rx_cmdq_,
                                 &msg, 1, nullptr) != 1);
    return poll_ctx;
}

bool Endpoint::uccl_wait(PollCtx *ctx) {
    {
        std::unique_lock<std::mutex> lock(ctx->mu);
        ctx->cv.wait(lock, [&ctx] { return ctx->done.load(); });
    }
    fence_and_clean_ctx(ctx);
    return true;
}

bool Endpoint::uccl_poll(PollCtx *ctx) {
    while (!uccl_poll_once(ctx));
    return true;
}

bool Endpoint::uccl_poll_once(PollCtx *ctx) {
    if (!ctx->done.load()) return false;
    fence_and_clean_ctx(ctx);
    return true;
}

void Endpoint::install_flow_on_engine(FlowID flow_id,
                                      const std::string &remote_ip,
                                      uint32_t local_engine_idx,
                                      int bootstrap_fd) {
    int ret;

    char local_mac_char[ETH_ALEN];
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
    auto *poll_ctx = new PollCtx();
    Channel::CtrlMsg ctrl_msg = {
        .opcode = Channel::CtrlMsg::Op::kInstallFlow,
        .flow_id = flow_id,
        .remote_ip = htonl(str_to_ip(remote_ip)),
        .remote_engine_idx = remote_engine_idx,
        .poll_ctx = poll_ctx,
    };
    str_to_mac(remote_mac, ctrl_msg.remote_mac);
    while (jring_mp_enqueue_bulk(channel_vec_[local_engine_idx]->ctrl_cmdq_,
                                 &ctrl_msg, 1, nullptr) != 1);

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

inline void Endpoint::fence_and_clean_ctx(PollCtx *ctx) {
    // Make the data written by the engine thread visible to the app thread.
    std::ignore =
        std::atomic_load_explicit(&ctx->fence, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);

    ctx->clear();
    ctx_pool_->push(ctx);
}

void Endpoint::stats_thread_fn() {
    if (GetEnvVar("UCCL_ENGINE_QUIET") == "1") return;

    while (!shutdown_) {
        {
            std::unique_lock<std::mutex> lock(stats_mu_);
            bool shutdown = stats_cv_.wait_for(
                lock, std::chrono::seconds(kStatsTimerIntervalSec),
                [this] { return shutdown_.load(); });
            if (shutdown) break;
        }

        if (engine_vec_.empty()) continue;
        std::string s;
        s += "\n\t[Uccl Engine] ";
        for (auto &engine : engine_vec_) {
            s += engine->status_to_string();
        }
        LOG(INFO) << s;
    }
}

}  // namespace uccl