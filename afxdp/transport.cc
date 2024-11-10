#include "transport.h"

namespace uccl {

Endpoint::Endpoint(Channel *channel)
    : channel_(channel), next_avail_conn_id_(0x1), ctx_pool_(kMaxInflightMsg) {
    for (int i = 0; i < kMaxInflightMsg; i++) {
        ctx_pool_.push(new PollCtx());
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
    LOG(INFO) << "Server ready, listening on port " << kBootstrapPort;
}

Endpoint::~Endpoint() {
    // TODO(yang): free all ctx in ctx_pool_.
    close(listen_fd_);
}

ConnectionID Endpoint::uccl_connect(std::string remote_ip) {
    struct sockaddr_in serv_addr;
    struct hostent *server;
    int bootstrap_fd;

    bootstrap_fd = socket(AF_INET, SOCK_STREAM, 0);
    DCHECK(bootstrap_fd >= 0) << "ERROR: opening socket";

    server = gethostbyname(remote_ip.c_str());
    DCHECK(server) << "ERROR: no such host";

    bzero((char *)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr,
          server->h_length);
    serv_addr.sin_port = htons(kBootstrapPort);

    LOG(INFO) << "Connecting to " << remote_ip << " (0x" << std::hex
              << serv_addr.sin_addr.s_addr << std::dec << ":" << kBootstrapPort
              << ")";

    // Connect and set nonblocking and nodelay
    while (connect(bootstrap_fd, (struct sockaddr *)&serv_addr,
                   sizeof(serv_addr))) {
        LOG(INFO) << "Connecting... Make sure the server is up.";
        sleep(1);
    }

    int flag = 1;
    setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag,
               sizeof(int));

    ConnectionID conn_id;
    int ret = read(bootstrap_fd, &conn_id, sizeof(ConnectionID));
    DCHECK(ret == sizeof(ConnectionID)) << "ERROR: reading connection_id";

    DCHECK(bootstrap_fd_map_.find(conn_id) == bootstrap_fd_map_.end())
        << "Dup ConnectionID";
    bootstrap_fd_map_[conn_id] = bootstrap_fd;

    return conn_id;
}

std::tuple<ConnectionID, std::string> Endpoint::uccl_accept() {
    struct sockaddr_in cli_addr;
    socklen_t clilen = sizeof(cli_addr);
    int bootstrap_fd;

    // Accept connection and set nonblocking and nodelay
    bootstrap_fd = accept(listen_fd_, (struct sockaddr *)&cli_addr, &clilen);
    DCHECK(bootstrap_fd >= 0) << "ERROR: accept";
    auto ip_str = ip_to_str(cli_addr.sin_addr.s_addr);

    LOG(INFO) << "Accepting from " << ip_str << " (0x" << std::hex
              << cli_addr.sin_addr.s_addr << std::dec << ":"
              << cli_addr.sin_port << ")";

    int flag = 1;
    setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag,
               sizeof(int));

    // TODO(yang): making it unique across all servers.
    ConnectionID conn_id = next_avail_conn_id_++;

    DCHECK(bootstrap_fd_map_.find(conn_id) == bootstrap_fd_map_.end())
        << "Dup ConnectionID";
    bootstrap_fd_map_[conn_id] = bootstrap_fd;

    int ret = write(bootstrap_fd, &conn_id, sizeof(ConnectionID));
    DCHECK(ret == sizeof(ConnectionID)) << "ERROR: writing connection_id";

    return std::make_tuple(conn_id, ip_str);
}

bool Endpoint::uccl_send(ConnectionID connection_id, const void *data,
                         const size_t len) {
    auto *poll_ctx = uccl_send_async(connection_id, data, len);
    return uccl_poll(poll_ctx);
}

PollCtx *Endpoint::uccl_send_async(ConnectionID connection_id, const void *data,
                                   const size_t len) {
    auto *poll_ctx = ctx_pool_.pop();
    Channel::Msg msg = {
        .opcode = Channel::Msg::Op::kTx,
        .data = const_cast<void *>(data),
        .len_tosend = len,
        .len_recvd = nullptr,
        .connection_id = connection_id,
        .poll_ctx = poll_ctx,
    };
    std::atomic_store_explicit(&poll_ctx->fence, true,
                               std::memory_order_release);
    while (jring_mp_enqueue_bulk(channel_->tx_ring_, &msg, 1, nullptr) != 1);
    return poll_ctx;
}

bool Endpoint::uccl_recv(ConnectionID connection_id, void *data, size_t *len) {
    auto *poll_ctx = uccl_recv_async(connection_id, data, len);
    return uccl_poll(poll_ctx);
}

PollCtx *Endpoint::uccl_recv_async(ConnectionID connection_id, void *data,
                                   size_t *len) {
    auto *poll_ctx = ctx_pool_.pop();
    Channel::Msg msg = {
        .opcode = Channel::Msg::Op::kRx,
        .data = data,
        .len_tosend = 0,
        .len_recvd = len,
        .connection_id = connection_id,
        .poll_ctx = poll_ctx,
    };
    while (jring_mp_enqueue_bulk(channel_->rx_ring_, &msg, 1, nullptr) != 1);
    return poll_ctx;
}

bool Endpoint::uccl_poll(PollCtx *ctx) {
    while (!uccl_poll_once(ctx));
    return true;
}

bool Endpoint::uccl_poll_once(PollCtx *ctx) {
    bool res;
    {
        std::unique_lock<std::mutex> lock(ctx->mu);
        res = ctx->cv.wait_for(lock, std::chrono::microseconds(1000),
                               [&ctx] { return ctx->done; });
    }
    if (!res) return false;

    // Make the data written by the engine thread visible to the app thread.
    std::ignore =
        std::atomic_load_explicit(&ctx->fence, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);

    ctx->clear();
    ctx_pool_.push(ctx);
    return true;
}

void TXTracking::receive_acks(uint32_t num_acked_pkts) {
    VLOG(3) << "Received " << num_acked_pkts << " acks "
            << "num_tracked_msgbufs " << num_tracked_msgbufs_;
    while (num_acked_pkts) {
        auto msgbuf = oldest_unacked_msgbuf_;
        DCHECK(msgbuf != nullptr);
        // if (msgbuf != last_msgbuf_) {
        if (num_tracked_msgbufs_ != 1) {
            DCHECK_NE(oldest_unacked_msgbuf_, oldest_unsent_msgbuf_)
                << "Releasing an unsent msgbuf!";
            oldest_unacked_msgbuf_ = msgbuf->next();
            DCHECK(oldest_unacked_msgbuf_ != nullptr);
        } else {
            oldest_unacked_msgbuf_ = nullptr;
            oldest_unsent_msgbuf_ = nullptr;
            last_msgbuf_ = nullptr;
            // CHECK_EQ(num_tracked_msgbufs_, 1);
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
        num_tracked_msgbufs_--;
        num_acked_pkts--;
    }
}

void TXTracking::append(FrameBuf *msgbuf_head, FrameBuf *msgbuf_tail,
                        uint32_t num_frames, PollCtx *poll_ctx) {
    poll_ctxs_.push_back(poll_ctx);
    VLOG(3) << "Appending " << num_frames << " frames "
            << " num_unsent_msgbufs_ " << num_unsent_msgbufs_
            << " last_msgbuf_ " << last_msgbuf_;
    DCHECK(msgbuf_head->is_first());
    DCHECK(msgbuf_tail->is_last());
    // Append the message at the end of the chain of buffers, if any.
    if (last_msgbuf_ == nullptr) {
        // This is the first pending message buffer in the flow.
        DCHECK(oldest_unsent_msgbuf_ == nullptr);
        last_msgbuf_ = msgbuf_tail;
        oldest_unsent_msgbuf_ = msgbuf_head;
        oldest_unacked_msgbuf_ = msgbuf_head;
    } else {
        // This is not the first message buffer in the flow.
        DCHECK(oldest_unacked_msgbuf_ != nullptr);
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
    VLOG(3) << "Get: unsent messages " << num_unsent_msgbufs_
            << " oldest_unsent_msgbuf " << oldest_unsent_msgbuf_;
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
        it = std::find_if(reass_q_.begin(), reass_q_.end(),
                          [&seqno](const reasm_queue_ent_t &entry) {
                              return entry.seqno >= seqno;
                          });
        VLOG(3) << "Received OOO packet: reass_q size " << reass_q_.size();
        if (it != reass_q_.end() && it->seqno == seqno) {
            VLOG(3) << "Received duplicate packet: " << seqno;
            // Duplicate packet. Drop it.
            socket_->push_frame(msgbuf->get_frame_offset());
            return kOOOTrackableDup;
        }
    }

    // Buffer the packet in the frame pool. It may be out-of-order.
    const size_t payload_len = frame_len - kNetHdrLen - kUcclHdrLen;
    // This records the incoming network packet UcclPktHdr.msg_flags in
    // FrameBuf.
    msgbuf->set_msg_flags(ucclh->msg_flags);

    if (seqno == expected_seqno) {
        VLOG(3) << "Received expected packet: " << seqno;
        reass_q_.emplace_front(msgbuf, seqno);
    } else {
        VLOG(3) << "Received OOO trackable packet: " << seqno;
        reass_q_.insert(it, reasm_queue_ent_t(msgbuf, seqno));
    }

    // Update the SACK bitmap for the newly received packet.
    pcb->sack_bitmap_bit_set(distance);

    // These frames will be freed when the message is delivered to the app.
    push_inorder_msgbuf_to_app(pcb);

    return kOOOTrackableExpectedOrInOrder;
}

void RXTracking::push_inorder_msgbuf_to_app(swift::Pcb *pcb) {
    while (!reass_q_.empty() && reass_q_.front().seqno == pcb->rcv_nxt) {
        auto &front = reass_q_.front();
        auto *msgbuf = front.msgbuf;
        reass_q_.pop_front();

        if (cur_msg_train_head_ == nullptr) {
            DCHECK(msgbuf->is_first());
            cur_msg_train_head_ = msgbuf;
            cur_msg_train_tail_ = msgbuf;
        } else {
            cur_msg_train_tail_->set_next(msgbuf);
            cur_msg_train_tail_ = msgbuf;
        }

        if (cur_msg_train_tail_->is_last()) {
            // Stash cur_msg_train_head/tail_ in case application threads
            // have not supplied the app buffer while the engine is keeping
            // receiving messages? Stash this ready message
            ready_msg_stash_.push_back(
                {cur_msg_train_head_, cur_msg_train_tail_});
            try_copy_msgbuf_to_appbuf(nullptr, nullptr, nullptr);

            // Reset the message train for the next message.
            cur_msg_train_head_ = nullptr;
            cur_msg_train_tail_ = nullptr;
        }

        pcb->advance_rcv_nxt();

        pcb->sack_bitmap_shift_right_one();
    }
}

void RXTracking::try_copy_msgbuf_to_appbuf(void *app_buf, size_t *app_buf_len,
                                           PollCtx *poll_ctx) {
    if (app_buf && app_buf_len && poll_ctx)
        app_buf_stash_.push_back({app_buf, app_buf_len, poll_ctx});

    VLOG(2) << "ready_msg_stash_ size: " << ready_msg_stash_.size()
            << " app_buf_stash_ size: " << app_buf_stash_.size();

    while (!ready_msg_stash_.empty() && !app_buf_stash_.empty()) {
        ready_msg_t ready_msg = ready_msg_stash_.front();
        ready_msg_stash_.pop_front();
        app_buf_t app_buf_desc = app_buf_stash_.front();
        app_buf_stash_.pop_front();

        // We have a complete message. Let's deliver it to the app.
        auto *msgbuf_iter = ready_msg.msg_head;
        size_t app_buf_pos = 0;
        while (true) {
            auto *pkt_addr = msgbuf_iter->get_pkt_addr();
            auto *payload_addr = pkt_addr + kNetHdrLen + kUcclHdrLen;
            auto payload_len =
                msgbuf_iter->get_frame_len() - kNetHdrLen - kUcclHdrLen;

            memcpy((uint8_t *)app_buf_desc.buf + app_buf_pos, payload_addr,
                   payload_len);
            app_buf_pos += payload_len;

            auto *msgbuf_iter_tmp = msgbuf_iter;

            if (msgbuf_iter->is_last()) {
                socket_->push_frame(msgbuf_iter_tmp->get_frame_offset());
                break;
            }
            msgbuf_iter = msgbuf_iter->next();
            DCHECK(msgbuf_iter);

            // Free received frames that have been copied to app buf.
            socket_->push_frame(msgbuf_iter_tmp->get_frame_offset());
        }

        *app_buf_desc.buf_len = app_buf_pos;

        // Wakeup app thread waiting on endpoint.
        std::atomic_store_explicit(&app_buf_desc.poll_ctx->fence, true,
                                   std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(app_buf_desc.poll_ctx->mu);
            app_buf_desc.poll_ctx->done = true;
            app_buf_desc.poll_ctx->cv.notify_one();
        }

        VLOG(3) << "Received a complete message " << app_buf_pos << " bytes";
    }
}

std::string UcclFlow::to_string() const {
    std::string s;
    s += "\t\t" + pcb_.to_string() + "\n\t\t[TX] pending msgbufs unsent: " +
         std::to_string(tx_tracking_.num_unsent_msgbufs()) +
         "\n\t\t[RX] ready msgs unconsumed: " +
         std::to_string(rx_tracking_.ready_msg_stash_.size());
    return s;
}

void UcclFlow::rx_messages(std::vector<FrameBuf *> msgbufs) {
    VLOG(3) << "Received " << msgbufs.size() << " packets";
    uint32_t num_data_frames_recvd = 0;
    bool ecn_recvd = false;
    RXTracking::ConsumeRet consume_ret;
    for (auto msgbuf : msgbufs) {
        // ebpf_transport has filtered out invalid pkts.
        auto *pkt_addr = msgbuf->get_pkt_addr();
        auto *ucclh = reinterpret_cast<UcclPktHdr *>(pkt_addr + kNetHdrLen);

        switch (ucclh->net_flags) {
            case UcclPktHdr::UcclFlags::kAck:
                // ACK packet, update the flow.
                process_ack(ucclh);
                // Free the received frame.
                socket_->push_frame(msgbuf->get_frame_offset());
                break;
            case UcclPktHdr::UcclFlags::kAckEcn:
                process_ack(ucclh);
                socket_->push_frame(msgbuf->get_frame_offset());
                // Need to slowdown the sender.
                ecn_recvd = true;
                break;
            case UcclPktHdr::UcclFlags::kData:
                // Data packet, process the payload. The frame will be freed
                // once the engine copies the payload into app buffer
                consume_ret = rx_tracking_.consume(&pcb_, msgbuf);
                num_data_frames_recvd++;
                break;
            default:
                VLOG(3) << "Unsupported UcclFlags: "
                        << std::bitset<8>((uint8_t)ucclh->net_flags);
        }
    }
    // Send one ack for a bunch of received packets.
    if (num_data_frames_recvd) {
        if (rx_tracking_.ready_msg_stash_.size() <= kReadyMsgThresholdForEcn) {
            socket_->send_packet(craft_ack(pcb_.seqno(), pcb_.ackno()));
        } else {
            socket_->send_packet(
                craft_ack_with_ecn(pcb_.seqno(), pcb_.ackno()));
        }
    }

    if (ecn_recvd) {
        // update the cwnd and rate.
        pcb_.mutliplicative_decrease();
    } else {
        pcb_.additive_increase();
    }

    // Sending data frames that can be send per cwnd.
    transmit_pending_packets();
}

void UcclFlow::tx_messages(FrameBuf *msg_head, FrameBuf *msg_tail,
                           uint32_t num_frames, PollCtx *poll_ctx) {
    if (num_frames)
        tx_tracking_.append(msg_head, msg_tail, num_frames, poll_ctx);

    // TODO(ilias): We first need to check whether the cwnd is < 1, so
    // that we fallback to rate-based CC.

    // Calculate the effective window (in # of packets) to check whether
    // we can send more packets.
    transmit_pending_packets();
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
    auto ackno = ucclh->ackno.value();
    if (swift::seqno_lt(ackno, pcb_.snd_una)) {
        VLOG(3) << "Received old ACK " << ackno;
        return;
    } else if (swift::seqno_eq(ackno, pcb_.snd_una)) {
        VLOG(3) << "Received duplicate ACK " << ackno;
        // Duplicate ACK.
        pcb_.duplicate_acks++;
        // Update the number of out-of-order acknowledgements.
        pcb_.snd_ooo_acks = ucclh->sack_bitmap_count.value();

        if (pcb_.duplicate_acks < swift::Pcb::kFastRexmitDupAckThres) {
            // We have not reached the threshold yet, so we do not do
            // anything.
        } else if (pcb_.duplicate_acks == swift::Pcb::kFastRexmitDupAckThres) {
            // Fast retransmit.
            VLOG(3) << "Fast retransmit " << ackno;
            fast_retransmit();
        } else {
            // We have already done the fast retransmit, so we are now
            // in the fast recovery phase.
            auto sack_bitmap_count = ucclh->sack_bitmap_count.value();
            // We check the SACK bitmap to see if there are more undelivered
            // packets. In fast recovery mode we get after a fast
            // retransmit, we will retransmit all missing packets that we
            // find from the SACK bitmap, when enumerating the SACK bitmap
            // for up to sack_bitmap_count ACKs.
            auto *msgbuf = tx_tracking_.get_oldest_unacked_msgbuf();
            VLOG(3) << "Fast recovery " << ackno << " sack_bitmap_count "
                    << sack_bitmap_count;
            size_t index = 0;
            while (sack_bitmap_count && msgbuf) {
                constexpr size_t sack_bitmap_bucket_size =
                    sizeof(ucclh->sack_bitmap[0]) * 8;
                const size_t sack_bitmap_bucket_idx =
                    index / sack_bitmap_bucket_size;
                const size_t sack_bitmap_idx_in_bucket =
                    index % sack_bitmap_bucket_size;
                auto sack_bitmap =
                    ucclh->sack_bitmap[sack_bitmap_bucket_idx].value();
                if ((sack_bitmap & (1ULL << sack_bitmap_idx_in_bucket)) == 0) {
                    // We found a missing packet.
                    VLOG(3) << "Fast recovery sack_bitmap_count "
                            << sack_bitmap_count;
                    auto seqno = pcb_.snd_una + index;
                    prepare_datapacket(msgbuf, seqno);
                    msgbuf->mark_not_txpulltime_free();
                    missing_frames_.push_back(
                        {msgbuf->get_frame_offset(), msgbuf->get_frame_len()});
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

        pcb_.snd_una = ackno;
        pcb_.duplicate_acks = 0;
        pcb_.snd_ooo_acks = 0;
        pcb_.rto_rexmits_consectutive = 0;
        pcb_.rto_maybe_reset();
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
#ifdef USING_TCP
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

void UcclFlow::prepare_l4header(uint8_t *pkt_addr,
                                uint32_t payload_bytes) const {
#ifdef USING_TCP
    auto *tcph = (tcphdr *)(pkt_addr + sizeof(ethhdr) + sizeof(iphdr));
    memset(tcph, 0, sizeof(tcphdr));
#ifdef USING_MULTIPATH
    static uint16_t rand_port = 0;
    tcph->source = htons(local_port_ + (rand_port++) % 8);
    tcph->dest = htons(remote_port_ + (rand_port++) % 8);
#else
    tcph->source = htons(local_port_);
    tcph->dest = htons(remote_port_);
#endif
    tcph->doff = 5;
    // TODO(yang): tcpdump shows wrong checksum. Need to fix it.
    // tcph->check = tcp_hdr_chksum(htonl(local_addr_), htonl(remote_addr_),
    //                              5 * sizeof(uint32_t) + payload_bytes);
#else
    auto *udph = (udphdr *)(pkt_addr + sizeof(ethhdr) + sizeof(iphdr));
#ifdef USING_MULTIPATH
    static uint16_t rand_port = 0;
    udph->source = htons(local_port_ + (rand_port++) % 8);
    udph->dest = htons(remote_port_ + (rand_port++) % 8);
#else
    udph->source = htons(local_port_);
    udph->dest = htons(remote_port_);
#endif
    udph->len = htons(sizeof(udphdr) + payload_bytes);
    udph->check = htons(0);
    // TODO(yang): Calculate the UDP checksum.
#endif
}

void UcclFlow::prepare_ucclhdr(uint8_t *pkt_addr, uint32_t seqno,
                               uint32_t ackno,
                               const UcclPktHdr::UcclFlags &net_flags,
                               uint8_t msg_flags) const {
    auto *ucclh = (UcclPktHdr *)(pkt_addr + kNetHdrLen);
    ucclh->magic = be16_t(UcclPktHdr::kMagic);
    ucclh->net_flags = net_flags;
    ucclh->msg_flags = msg_flags;
    ucclh->seqno = be32_t(seqno);
    ucclh->ackno = be32_t(ackno);

    for (size_t i = 0; i < sizeof(UcclPktHdr::sack_bitmap) /
                               sizeof(UcclPktHdr::sack_bitmap[0]);
         ++i) {
        ucclh->sack_bitmap[i] = be64_t(pcb_.sack_bitmap[i]);
    }
    ucclh->sack_bitmap_count = be16_t(pcb_.sack_bitmap_count);
}

AFXDPSocket::frame_desc UcclFlow::craft_ctlpacket(
    uint32_t seqno, uint32_t ackno, const UcclPktHdr::UcclFlags &flags) const {
    const size_t kControlPayloadBytes = kUcclHdrLen;
    auto frame_offset = socket_->pop_frame();
    auto msgbuf = FrameBuf::Create(frame_offset, socket_->umem_buffer_,
                                   kNetHdrLen + kControlPayloadBytes);
    // Let AFXDPSocket::pull_complete_queue() free control frames.
    msgbuf->mark_txpulltime_free();

    uint8_t *pkt_addr = (uint8_t *)socket_->umem_buffer_ + frame_offset;
    prepare_l2header(pkt_addr);
    prepare_l3header(pkt_addr, kControlPayloadBytes);
    prepare_l4header(pkt_addr, kControlPayloadBytes);
    prepare_ucclhdr(pkt_addr, seqno, ackno, flags);

    return {frame_offset, kNetHdrLen + kControlPayloadBytes};
}

void UcclFlow::prepare_datapacket(FrameBuf *msg_buf, uint32_t seqno) const {
    // Header length after before the payload.
    uint32_t frame_len = msg_buf->get_frame_len();
    CHECK_LE(frame_len, AFXDP_MTU);
    uint8_t *pkt_addr = msg_buf->get_pkt_addr();

    // Prepare network headers.
    prepare_l2header(pkt_addr);
    prepare_l3header(pkt_addr, frame_len - kNetHdrLen);
    prepare_l4header(pkt_addr, frame_len - kNetHdrLen);

    // Prepare the Uccl-specific header.
    auto *ucclh = reinterpret_cast<UcclPktHdr *>(pkt_addr + kNetHdrLen);
    ucclh->magic = be16_t(UcclPktHdr::kMagic);
    ucclh->net_flags = UcclPktHdr::UcclFlags::kData;
    ucclh->ackno = be32_t(UINT32_MAX);
    // This fills the FrameBuf.flags into the outgoing packet
    // UcclPktHdr.msg_flags.
    ucclh->msg_flags = msg_buf->msg_flags();

    ucclh->seqno = be32_t(seqno);
}

void UcclFlow::fast_retransmit() {
    VLOG(3) << "Fast retransmitting oldest unacked packet " << pcb_.snd_una;
    // Retransmit the oldest unacknowledged message buffer.
    auto *msg_buf = tx_tracking_.get_oldest_unacked_msgbuf();
    if (msg_buf) {
        prepare_datapacket(msg_buf, pcb_.snd_una);
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
        prepare_datapacket(msg_buf, pcb_.snd_una);
        msg_buf->mark_not_txpulltime_free();
        socket_->send_packet(
            {msg_buf->get_frame_offset(), msg_buf->get_frame_len()});
    }
    pcb_.rto_reset();
    pcb_.rto_rexmits++;
    pcb_.rto_rexmits_consectutive++;
}

/**
 * @brief Helper function to transmit a number of packets from the queue
 * of pending TX data.
 */
void UcclFlow::transmit_pending_packets() {
    auto remaining_packets =
        std::min(pcb_.effective_wnd(), tx_tracking_.num_unsent_msgbufs());
    remaining_packets = std::min(
        remaining_packets, socket_->send_queue_free_entries(remaining_packets));
    if (remaining_packets == 0) return;

    // Prepare the packets.
    for (uint16_t i = 0; i < remaining_packets; i++) {
        auto msg_buf_opt = tx_tracking_.get_and_update_oldest_unsent();
        if (!msg_buf_opt.has_value()) break;

        auto *msg_buf = msg_buf_opt.value();
        auto seqno = pcb_.get_snd_nxt();
        prepare_datapacket(msg_buf, seqno);
        msg_buf->mark_not_txpulltime_free();
        pending_tx_frames_.push_back(
            {msg_buf->get_frame_offset(), msg_buf->get_frame_len()});
    }

    // TX both data and ack frames.
    if (pending_tx_frames_.empty()) return;
    VLOG(3) << "transmit_pending_packets " << pending_tx_frames_.size();
    socket_->send_packets(pending_tx_frames_);
    pending_tx_frames_.clear();

    if (pcb_.rto_disabled()) pcb_.rto_enable();
}

void UcclEngine::run() {
    // TODO(yang): maintain a queue of rx_work and tx_work
    Channel::Msg rx_work;
    Channel::Msg tx_work;

    while (!shutdown_) {
        // Calculate the time elapsed since the last periodic
        // processing.
        auto now = rdtsc_to_us(rdtsc());
        const auto elapsed = now - last_periodic_timestamp_;

        if (elapsed >= kSlowTimerIntervalUs) {
            // Perform periodic processing.
            periodic_process();
            last_periodic_timestamp_ = now;
        }

        if (jring_sc_dequeue_bulk(channel_->rx_ring_, &rx_work, 1, nullptr) ==
            1) {
            VLOG(3) << "Rx jring dequeue";
            supply_rx_app_buf(rx_work.data, rx_work.len_recvd,
                              rx_work.poll_ctx);
        }

        auto frames = socket_->recv_packets(RECV_BATCH_SIZE);
        if (frames.size()) {
            VLOG(3) << "Rx recv_packets " << frames.size();
            std::vector<FrameBuf *> msgbufs;
            msgbufs.reserve(frames.size());
            for (auto &frame : frames) {
                auto *msgbuf = FrameBuf::Create(
                    frame.frame_offset, socket_->umem_buffer_, frame.frame_len);
                msgbufs.push_back(msgbuf);
            }
            process_rx_msg(msgbufs);
        }

        if (jring_sc_dequeue_bulk(channel_->tx_ring_, &tx_work, 1, nullptr) ==
            1) {
            // Make data written by the app thread visible to the engine.
            std::ignore = std::atomic_load_explicit(&tx_work.poll_ctx->fence,
                                                    std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);

            VLOG(3) << "Tx jring dequeue";
            auto [tx_msgbuf_head, tx_msgbuf_tail, num_tx_frames] =
                deserialize_msg(tx_work.data, tx_work.len_tosend);
            VLOG(3) << "Tx process_tx_msg";
            // Append these tx frames to the flow's tx queue, and trigger
            // intial tx. Future received ACKs will trigger more tx.
            process_tx_msg(tx_msgbuf_head, tx_msgbuf_tail, num_tx_frames,
                           tx_work.poll_ctx);
        }

        // process_tx_msg(nullptr, nullptr, 0);
    }

    // This will reset flow pcb state.
    flow_->shutdown();
    // This will flush all unpolled tx frames.
    socket_->shutdown();

    delete flow_;
    delete socket_;
}

/**
 * @brief Method to perform periodic processing. This is called by the
 * main engine cycle (see method `Run`).
 */
void UcclEngine::periodic_process() {
    // Advance the periodic ticks counter.
    periodic_ticks_++;
    if (periodic_ticks_ % kDumpStatusTicks == 0) dump_status();
    handle_rto();
    process_ctl_reqs();
}

void UcclEngine::handle_rto() {
    // TODO(yang): maintain active_flows_map_
    auto is_active_flow = flow_->periodic_check();
    DCHECK(is_active_flow);
}

void UcclEngine::dump_status() {
    std::string s;
    s += "\n\t\t[Uccl Engine] " + Format("%x [queue %d] <-> %x [queue 0]\n",
                                         flow_->local_addr_, socket_->queue_id_,
                                         flow_->remote_addr_);
    s += flow_->to_string();
    s += socket_->to_string();
    // TODO(yang): Add more status information.
    s += "\n";
    LOG(INFO) << s;
}

std::tuple<FrameBuf *, FrameBuf *, uint32_t> UcclEngine::deserialize_msg(
    void *app_buf, size_t app_buf_len) {
    FrameBuf *tx_msgbuf_head = nullptr;
    FrameBuf *tx_msgbuf_tail = nullptr;
    uint32_t num_tx_frames = 0;

    auto remaining_bytes = app_buf_len;

    //  Deserializing the message into MTU-sized frames.
    FrameBuf *last_msgbuf = nullptr;
    while (remaining_bytes > 0) {
        auto payload_len = std::min(
            remaining_bytes, (size_t)AFXDP_MTU - kNetHdrLen - kUcclHdrLen);
        auto frame_offset = socket_->pop_frame();
        auto *msgbuf = FrameBuf::Create(frame_offset, socket_->umem_buffer_,
                                        kNetHdrLen + kUcclHdrLen + payload_len);
        // The engine will free these Tx frames when receiving ACKs.
        msgbuf->mark_not_txpulltime_free();

        VLOG(3) << "Deser msgbuf " << msgbuf << " " << num_tx_frames;
        auto pkt_payload_addr =
            msgbuf->get_pkt_addr() + kNetHdrLen + kUcclHdrLen;
        memcpy(pkt_payload_addr, app_buf, payload_len);
        remaining_bytes -= payload_len;
        app_buf += payload_len;

        if (tx_msgbuf_head == nullptr) {
            msgbuf->mark_first();
            tx_msgbuf_head = msgbuf;
        } else {
            last_msgbuf->set_next(msgbuf);
        }

        if (remaining_bytes == 0) {
            msgbuf->mark_last();
            msgbuf->set_next(nullptr);
            tx_msgbuf_tail = msgbuf;
        }

        last_msgbuf = msgbuf;
        num_tx_frames++;
    }
    CHECK(tx_msgbuf_head->is_first());
    CHECK(tx_msgbuf_tail->is_last());
    return std::make_tuple(tx_msgbuf_head, tx_msgbuf_tail, num_tx_frames);
}

}  // namespace uccl