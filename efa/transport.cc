#include "transport.h"
#include "transport_config.h"
#include "transport_header.h"
#include <glog/logging.h>
#include <infiniband/verbs.h>

namespace uccl {

void TXTracking::receive_acks(uint32_t num_acked_pkts) {
  VLOG(3) << "Received " << num_acked_pkts << " acks :"
          << " num_unsent_msgbufs_ " << num_unsent_msgbufs_ << " last_msgbuf_ "
          << last_msgbuf_ << " oldest_unsent_msgbuf " << oldest_unsent_msgbuf_
          << " oldest_unacked_msgbuf_ " << oldest_unacked_msgbuf_;
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
      CHECK_EQ(num_tracked_msgbufs_, 1);
      oldest_unacked_msgbuf_ = nullptr;
      oldest_unsent_msgbuf_ = nullptr;
      last_msgbuf_ = nullptr;
    }

    if (msgbuf->is_last()) {
      // Tx a full message; wakeup app thread waiting on endpoint.
      auto poll_ctx = (PollCtx*)msgbuf->get_poll_ctx();
      DCHECK(poll_ctx);
      if (--(poll_ctx->num_unfinished) == 0) {
        std::lock_guard<std::mutex> lock(poll_ctx->mu);
        poll_ctx->done = true;
        poll_ctx->cv.notify_one();
      }
    }
    // Free transmitted frames that are acked
    socket_->push_pkt_hdr(msgbuf->get_pkt_hdr_addr());
    // The pkt_data directly from the app buffer, thus no need to free.
    socket_->push_frame_desc((uint64_t)msgbuf);

    num_unacked_msgbufs_--;
    num_tracked_msgbufs_--;
    num_acked_pkts--;
  }
}

void TXTracking::append(FrameDesc* msgbuf_head, FrameDesc* msgbuf_tail,
                        uint32_t num_frames) {
  VLOG(3) << "Appending " << num_frames << " frames :"
          << " num_unsent_msgbufs_ " << num_unsent_msgbufs_ << " last_msgbuf_ "
          << last_msgbuf_ << " oldest_unsent_msgbuf " << oldest_unsent_msgbuf_
          << " oldest_unacked_msgbuf_ " << oldest_unacked_msgbuf_;

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

uint32_t TXTracking::convert_permitted_packets_to_bytes(
    uint32_t permitted_packets) {
  uint32_t total_packets = 0;
  uint32_t permitted_bytes = 0;

  auto msgbuf = oldest_unsent_msgbuf_;
  while (msgbuf != nullptr) {
    total_packets++;
    permitted_bytes += msgbuf->get_pkt_data_len();
    if (total_packets >= permitted_packets) break;
    msgbuf = msgbuf->next();
  }

  return permitted_bytes;
}

// Traverse the chain of frames to figure out how many packets are permitted to
// be sent.
uint32_t TXTracking::convert_permitted_bytes_to_packets(
    uint32_t permitted_bytes) {
  uint32_t total_bytes = 0;
  uint32_t permitted_packets = 0;

  auto msgbuf = oldest_unsent_msgbuf_;
  while (msgbuf != nullptr) {
    total_bytes += msgbuf->get_pkt_data_len();
    permitted_packets++;
    if (total_bytes >= permitted_bytes) break;
    msgbuf = msgbuf->next();
  }

  return permitted_packets;
}

std::optional<FrameDesc*> TXTracking::get_and_update_oldest_unsent() {
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

  if (oldest_unacked_msgbuf_ == nullptr) oldest_unacked_msgbuf_ = msgbuf;

  num_unacked_msgbufs_++;
  num_unsent_msgbufs_--;
  return msgbuf;
}

RXTracking::ConsumeRet RXTracking::consume(UcclFlow* flow, FrameDesc* msgbuf) {
  auto* pcb = flow->get_pcb();
  auto* eqds_cc = flow->get_eqds_cc();

  uint8_t* pkt_addr = (uint8_t*)msgbuf->get_pkt_hdr_addr() + EFA_UD_ADDITION;
  auto* ucclh = reinterpret_cast<UcclPktHdr*>(pkt_addr);
  auto const frame_len = ucclh->frame_len.value();
  auto const seqno = ucclh->seqno.value();
  auto const expected_seqno = pcb->rcv_nxt;

  if (swift::seqno_lt(seqno, expected_seqno)) {
    VLOG(3) << "Received old packet: " << seqno << " < " << expected_seqno;
    socket_->push_pkt_hdr(msgbuf->get_pkt_hdr_addr());
    socket_->push_pkt_data(msgbuf->get_pkt_data_addr());
    socket_->push_frame_desc((uint64_t)msgbuf);
    return kOldPkt;
  }

  const size_t distance = seqno - expected_seqno;
  if (distance >= kReassemblyMaxSeqnoDistance) {
    VLOG(3) << "Packet too far ahead. Dropping as we can't handle SACK. "
            << "seqno: " << seqno << ", expected: " << expected_seqno;
    socket_->push_pkt_hdr(msgbuf->get_pkt_hdr_addr());
    socket_->push_pkt_data(msgbuf->get_pkt_data_addr());
    socket_->push_frame_desc((uint64_t)msgbuf);
    return kOOOUntrackable;
  }

  // Only iterate through the deque if we must, i.e., for ooo packts only
  auto it = reass_q_.begin();
  if (seqno != expected_seqno) {
    it = reass_q_.lower_bound(seqno);
    if (it != reass_q_.end() && it->first == seqno) {
      VLOG(3) << "Received duplicate packet: " << seqno;
      // Duplicate packet. Drop it.
      socket_->push_pkt_hdr(msgbuf->get_pkt_hdr_addr());
      socket_->push_pkt_data(msgbuf->get_pkt_data_addr());
      socket_->push_frame_desc((uint64_t)msgbuf);
      return kOOOTrackableDup;
    }
    VLOG(3) << "Received OOO trackable packet: " << seqno
            << " payload_len: " << frame_len - kUcclPktHdrLen
            << " reass_q size " << reass_q_.size();
  } else {
    VLOG(3) << "Received expected packet: " << seqno
            << " payload_len: " << frame_len - kUcclPktHdrLen;
  }

  num_unconsumed_msgbufs_++;
  // if (num_unconsumed_msgbufs_ >= kMaxUnconsumedRxMsgbufs)
  //     LOG(INFO) << "num_unconsumed_msgbufs_: " << num_unconsumed_msgbufs_;

  // Buffer the packet in the frame pool. It may be out-of-order.
  reass_q_.insert(it, {seqno, msgbuf});

  VLOG_IF(3, num_unconsumed_msgbufs_ >= kMaxUnconsumedRxMsgbufs)
      << "seqno: " << seqno << " expected_seqno: " << expected_seqno
      << " distance: " << distance << " pcb->rcv_nxt: " << pcb->rcv_nxt
      << " reass_q_.begin()->first: " << reass_q_.begin()->first;

  // Update the SACK bitmap for the newly received packet.
  pcb->sack_bitmap_bit_set(distance);

  // These frames will be freed when the message is delivered to the app.
  push_inorder_msgbuf_to_app(pcb);

  if constexpr (kReceiverCCType == ReceiverCCType::kEQDS) {
    if (eqds_cc->handle_pull_target(ucclh->pullno.value())) {
      flow->request_pull();
    }
  }

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

#ifdef SCATTERED_MEMCPY
void RXTracking::try_copy_msgbuf_to_appbuf(Channel::Msg* rx_work) {
  if (rx_work) {
    if (rx_work->opcode == Channel::Msg::kRxFreePtrs) {
      int iov_n = rx_work->len;
      void** iov_addrs = (void**)rx_work->data;
      for (int i = 0; i < iov_n; i++) {
        socket_->push_pkt_data((uint64_t)iov_addrs[i]);
      }
      return;
    }
    VLOG(3) << "num_unconsumed_msgbufs: " << num_unconsumed_msgbufs()
            << " app_buf_queue_ size: " << app_buf_queue_.size();
    app_buf_queue_.push_back({*rx_work});
  }

  while (!ready_msg_queue_.empty() && !app_buf_queue_.empty()) {
    FrameDesc* ready_msg = ready_msg_queue_.front();
    ready_msg_queue_.pop_front();
    DCHECK(ready_msg) << ready_msg->print_chain();
    auto& [rx_copy_work] = app_buf_queue_.front();

    auto const* ucclh = reinterpret_cast<UcclPktHdr const*>(
        ready_msg->get_pkt_hdr_addr() + EFA_UD_ADDITION);
    auto payload_len = ucclh->frame_len.value() - kUcclPktHdrLen;

    auto* req = rx_copy_work.req;
    DCHECK(iov_n_ < kMaxIovs);
    req->iov_addrs[iov_n_] = (void*)ready_msg->get_pkt_data_addr();
    VLOG(3) << "gpu_idx " << socket_->gpu_idx() << " iov_addrs[" << iov_n_
            << "]: " << std::hex << req->iov_addrs[iov_n_];
    req->iov_lens[iov_n_] = (int)payload_len;
    req->dst_offsets[iov_n_] = deser_msg_len_;
    deser_msg_len_ += payload_len;
    iov_n_++;
    num_unconsumed_msgbufs_--;

    if (ready_msg->is_last()) {
      ready_msg->set_next(nullptr);

      // Copy the complete message to the app buffer.
      req->iov_n = iov_n_;
      *rx_copy_work.len_p = deser_msg_len_;

      auto* poll_ctx = rx_copy_work.poll_ctx;
      // Wakeup app thread waiting on endpoint.
      if (--(poll_ctx->num_unfinished) == 0) {
        poll_ctx->write_barrier();
        std::lock_guard<std::mutex> lock(poll_ctx->mu);
        poll_ctx->done = true;
        poll_ctx->cv.notify_one();
      }

      app_buf_queue_.pop_front();
      deser_msg_len_ = 0;
      iov_n_ = 0;
    }

    // Header and frame desc are no longer needed.
    socket_->push_pkt_hdr(ready_msg->get_pkt_hdr_addr());
    socket_->push_frame_desc((uint64_t)ready_msg);
  }
}

#else

void RXTracking::try_copy_msgbuf_to_appbuf(Channel::Msg* rx_work) {
  if (rx_work) {
    VLOG(3) << "num_unconsumed_msgbufs: " << num_unconsumed_msgbufs()
            << " app_buf_queue_ size: " << app_buf_queue_.size();
    app_buf_queue_.push_back({*rx_work});
  } else {
    // Channel::Msg rx_work2;
    // while (Channel::dequeue_sc(channel_->rx_task_q_, &rx_work2)) {
    //     VLOG(3) << "Rx jring dequeue";
    //     active_flows_map_[rx_work2.flow_id]->rx_supply_app_buf(rx_work2);
    // }
    // LOG_EVERY_N(INFO, 1000000)
    //     << "num_unconsumed_msgbufs: " << num_unconsumed_msgbufs()
    //     << " ready_msg_queue_ size: " << ready_msg_queue_.size()
    //     << " app_buf_queue_ size: " << app_buf_queue_.size();
  }

  while (!ready_msg_queue_.empty() && !app_buf_queue_.empty()) {
    FrameDesc* ready_msg = ready_msg_queue_.front();
    ready_msg_queue_.pop_front();
    DCHECK(ready_msg) << ready_msg->print_chain();

#ifdef EMULATE_RC_ZC
    auto const* ucclh = reinterpret_cast<UcclPktHdr const*>(
        ready_msg->get_pkt_hdr_addr() + EFA_UD_ADDITION);
    auto payload_len = ucclh->frame_len.value() - kUcclPktHdrLen;
    deser_msg_len_ += payload_len;
#endif

    if (deser_msgs_head_ == nullptr) {
      deser_msgs_head_ = ready_msg;
      deser_msgs_tail_ = ready_msg;
    } else {
      deser_msgs_tail_->set_next(ready_msg);
      deser_msgs_tail_ = ready_msg;
    }

    if (ready_msg->is_last()) {
      ready_msg->set_next(nullptr);

      auto& [rx_copy_work] = app_buf_queue_.front();
      rx_copy_work.deser_msgs = deser_msgs_head_;

      // The len field is not used in RX, we reuse it to pass counter
      // address so that copy thread can update.
      rx_copy_work.reserved = (uint64_t)(&num_unconsumed_msgbufs_);

      // Copy the complete message to the app buffer.
#ifdef EMULATE_RC_ZC
      *rx_copy_work.len_p = deser_msg_len_;
      Channel::enqueue_sp(channel_->rx_copy_done_q_, &rx_copy_work);

      auto* poll_ctx = rx_copy_work.poll_ctx;
      // Wakeup app thread waiting on endpoint.
      if (--(poll_ctx->num_unfinished) == 0) {
        poll_ctx->write_barrier();
        std::lock_guard<std::mutex> lock(poll_ctx->mu);
        poll_ctx->done = true;
        poll_ctx->cv.notify_one();
      }
#else
      rx_copy_work.poll_ctx->write_barrier();
      Channel::enqueue_sp(channel_->rx_copy_q_, &rx_copy_work);
#endif

      app_buf_queue_.pop_front();
      deser_msgs_head_ = nullptr;
      deser_msgs_tail_ = nullptr;
      deser_msg_len_ = 0;
    }
  }

  Channel::Msg rx_copy_done_work;
  while (Channel::dequeue_sc(channel_->rx_copy_done_q_, &rx_copy_done_work)) {
    auto* num_unconsumed_msgbufs = (uint32_t*)rx_copy_done_work.reserved;
    auto ready_msg = rx_copy_done_work.deser_msgs;

    while (ready_msg != nullptr) {
      // Free received frames that have been copied to app buf.
      socket_->push_pkt_hdr(ready_msg->get_pkt_hdr_addr());
      socket_->push_pkt_data(ready_msg->get_pkt_data_addr());
      socket_->push_frame_desc((uint64_t)ready_msg);

      ready_msg = ready_msg->next();
      (*num_unconsumed_msgbufs)--;
    }
  }
}

#endif

void RXTracking::copy_thread_func(uint32_t engine_idx, UcclEngine* engine) {
  // see
  // https://forums.developer.nvidia.com/t/persistent-kernel-does-not-work-properly-on-some-gpus/264019/5
  CHECK(GetEnvVar("CUDA_MODULE_LOADING") == "EAGER");

  copy_param_t* copy_param = new copy_param_t();
  cudaStream_t copy_stream;

  auto ret = cudaSetDevice(get_gpu_idx_by_engine_idx(engine_idx));
  CHECK(ret == cudaSuccess) << "cudaSetDevice failed";
  ret = cudaStreamCreate(&copy_stream);
  CHECK(ret == cudaSuccess) << "Failed to create cuda stream";

  // Note: these two macro conflicts with each other.

  // Test if this would block NCCL kernel launching---sadly, it does.
  // #define TEST_CONCURRENT
  // Test if empty copy would incur high perf overhead---it does not.
  // #define TEST_EMPTY_COPY

#ifdef TEST_CONCURRENT
  launchPersistentScatteredMemcpy(4, copy_stream);
#endif

  // Temporarily buffering Msg that are being copied by the cuda kernel.
  std::deque<Channel::Msg> ongoing_copy_queue;
  auto* socket = engine->socket_;
  auto* channel = engine->channel_;

  while (!engine->shutdown_) {
    if (!ongoing_copy_queue.empty()) {
#if defined(TEST_CONCURRENT) || defined(TEST_EMPTY_COPY)
      if (true) {
#else
      if (pollScatteredMemcpy(copy_stream)) {
#endif
        while (!ongoing_copy_queue.empty()) {
          auto& rx_copy_done_work = ongoing_copy_queue.back();
          auto poll_ctx = rx_copy_done_work.poll_ctx;
          auto* app_buf_len_p = rx_copy_done_work.len_p;

          Channel::enqueue_sp(channel->rx_copy_done_q_, &rx_copy_done_work);

          // Wakeup app thread waiting on endpoint.
          if (--(poll_ctx->num_unfinished) == 0) {
            poll_ctx->write_barrier();
            std::lock_guard<std::mutex> lock(poll_ctx->mu);
            poll_ctx->done = true;
            poll_ctx->cv.notify_one();
          }
          VLOG(2) << "copy_thread_func: Received a complete message "
                  << *app_buf_len_p << " bytes";

          ongoing_copy_queue.pop_back();
        }
      }
    }

    Channel::Msg rx_copy_work;
    int copy_idx = 0;

    // Using adapative batching to launch a larger memcpy kernel; this is
    // good for small messages with high inflights.
    while (Channel::dequeue_sc(channel->rx_copy_q_, &rx_copy_work)) {
      FrameDesc* ready_msg = rx_copy_work.deser_msgs;
      auto* app_buf = rx_copy_work.data;
      auto* app_buf_len_p = rx_copy_work.len_p;
      auto* poll_ctx = rx_copy_work.poll_ctx;
      poll_ctx->read_barrier();

      VLOG(2) << "copy_idx: " << copy_idx;
      size_t cur_offset = 0;
      while (ready_msg != nullptr) {
        auto* pkt_addr =
            (uint8_t*)ready_msg->get_pkt_hdr_addr() + EFA_UD_ADDITION;
        DCHECK(pkt_addr) << "pkt_addr is nullptr when copy to app buf "
                         << std::hex << "0x" << ready_msg << std::dec
                         << ready_msg->to_string();
        auto const* ucclh = reinterpret_cast<UcclPktHdr const*>(pkt_addr);
        auto payload_len = ucclh->frame_len.value() - kUcclPktHdrLen;
        VLOG(2) << "copy_thread_func: payload_len: " << payload_len
                << " seqno: " << std::dec << ucclh->seqno.value();

        copy_param->dst[copy_idx] = (uint64_t)app_buf + cur_offset;
        copy_param->src[copy_idx] = ready_msg->get_pkt_data_addr();
        copy_param->len[copy_idx] = (uint32_t)payload_len;

        copy_idx++;
        cur_offset += payload_len;

        // We have a complete message. Let's deliver it to the app.
        if (ready_msg->is_last()) {
          *app_buf_len_p = cur_offset;
          // Delaying wake up the app thread until all messages are
          // copied.
        }

        ready_msg = ready_msg->next();
      }
      ongoing_copy_queue.push_back(rx_copy_work);

      if (copy_idx > MAX_COPIES / 2) {
        break;
      }
    }

    if (copy_idx) {
#if !defined(TEST_CONCURRENT) && !defined(TEST_EMPTY_COPY)
      launchScatteredMemcpyAsync(copy_idx, copy_param, copy_stream);
#endif
    }
  }
}

std::string UcclFlow::to_string() const {
  std::string s;
  s += "\n\t\t\t[CC] pcb:         " + pcb_.to_string() +
       (kSenderCCType == SenderCCType::kCubicPP
            ? "\n\t\t\t     cubic_pp[0]: " + cubic_pp_[0].to_string()
            : "\n\t\t\t     cubic:       " + cubic_g_.to_string()) +
       "\n\t\t\t     timely:      " + timely_g_.to_string() +
       "\n\t\t\t[TX] msgbufs unsent: " +
       std::to_string(tx_tracking_.num_unsent_msgbufs()) +
       "\n\t\t\t[RX] msgbufs unconsumed: " +
       std::to_string(rx_tracking_.num_unconsumed_msgbufs());
  return s;
}

void UcclFlow::rx_messages() {
  VLOG(3) << "Received " << pending_rx_msgbufs_.size() << " packets";
  RXTracking::ConsumeRet consume_ret;
  uint32_t num_data_frames_recvd = 0;
  uint32_t path_id = 0;
  uint32_t probe_path_id = 0;
  uint64_t timestamp1 = 0, timestamp2 = 0, timestamp4 = 0;
  bool received_rtt_probe = false;

  for (auto* msgbuf : pending_rx_msgbufs_) {
    // efa_transport has filtered out invalid pkts.
#ifdef USE_SRD_FOR_CTRL
    uint8_t* pkt_addr = (uint8_t*)msgbuf->get_pkt_hdr_addr() +
                        (msgbuf->get_pkt_data_len() == 0 ? 0 : EFA_UD_ADDITION);
#else
    uint8_t* pkt_addr = (uint8_t*)msgbuf->get_pkt_hdr_addr() + EFA_UD_ADDITION;
#endif
    auto* ucclh = reinterpret_cast<UcclPktHdr*>(pkt_addr);
    auto* ucclsackh = reinterpret_cast<UcclSackHdr*>(pkt_addr + kUcclPktHdrLen);

    switch (ucclh->net_flags) {
      case UcclPktHdr::UcclFlags::kCredit:
        process_credit(reinterpret_cast<UcclPullHdr*>(ucclh));
        credit_qp_ctx_->engine_push_pkt_hdr(msgbuf->get_pkt_hdr_addr());
        credit_qp_ctx_->engine_push_frame_desc((uint64_t)msgbuf);
        break;
      case UcclPktHdr::UcclFlags::kAckRttProbe:
        timestamp4 = msgbuf->get_cpe_time_tsc();
        // Sender gets the RTT probe response, update the flow.
        process_rttprobe_rsp(ucclh->timestamp1, ucclh->timestamp2,
                             ucclsackh->timestamp3, timestamp4, ucclh->path_id);
      case UcclPktHdr::UcclFlags::kAck:
        last_received_rwnd_ = ucclsackh->rwnd.value();
        // ACK packet, update the flow.
        process_ack(ucclh);
        // Free the received frame.
        socket_->push_pkt_hdr(msgbuf->get_pkt_hdr_addr());
        socket_->push_frame_desc((uint64_t)msgbuf);
        break;
      case UcclPktHdr::UcclFlags::kDataRttProbe:
        // Receiver gets the RTT probe, relay it back in the ACK.
        // If multiple RTT probe, we take the last one's timestamp.
        received_rtt_probe = true;
        probe_path_id = ucclh->path_id;
        timestamp1 = ucclh->timestamp1;
        timestamp2 = msgbuf->get_cpe_time_tsc();
      case UcclPktHdr::UcclFlags::kData:
        // Data packet, process the payload. The frame will be freed
        // once the engine copies the payload into app buffer
        consume_ret = rx_tracking_.consume(this, msgbuf);
        num_data_frames_recvd++;
        path_id = ucclh->path_id;
        break;
      default:
        CHECK(false) << "Unsupported UcclFlags: "
                     << std::bitset<8>((uint8_t)ucclh->net_flags);
    }
  }
  pending_rx_msgbufs_.clear();

  // Send one ack for a bunch of received packets.
  if (num_data_frames_recvd) {
    // To avoid receiver-side buffer empty.
    if (rx_tracking_.num_unconsumed_msgbufs() < kMaxUnconsumedRxMsgbufs) {
      auto net_flags = received_rtt_probe ? UcclPktHdr::UcclFlags::kAckRttProbe
                                          : UcclPktHdr::UcclFlags::kAck;
      // Ack following the probe path if received, or the last path.
      path_id = received_rtt_probe ? probe_path_id : path_id;
      path_id = data_path_id_to_ctrl_path_id(path_id);
      auto [src_qp_idx, dst_qp_idx] = path_id_to_src_dst_qp_for_ctrl(path_id);

      // Avoiding client sending too much packet which would empty msgbuf.
      auto rwnd =
          kMaxUnconsumedRxMsgbufs - rx_tracking_.num_unconsumed_msgbufs();
      FrameDesc* ack_frame =
          craft_ackpacket(path_id, pcb_.seqno(), pcb_.ackno(), net_flags,
                          timestamp1, timestamp2, rwnd);
      ack_frame->set_dest_ah(remote_ah_);
      ack_frame->set_dest_qpn(remote_meta_->qpn_list_ctrl[dst_qp_idx]);

      socket_->post_send_wr(ack_frame, src_qp_idx);
    }
  }

  deserialize_and_append_to_txtracking();
  transmit_pending_packets_drr(true);
}

void UcclFlow::tx_prepare_messages(Channel::Msg& tx_work) {
  // deser tx_work into a FrameDesc chain, then pass to deser_th.
  FrameDesc* deser_msgs_head = nullptr;
  FrameDesc* deser_msgs_tail = nullptr;
  auto* app_buf_cursor = tx_work.data;
  auto remaining_bytes = tx_work.len;
  auto* app_mr = tx_work.mhandle->mr;
  auto* poll_ctx = tx_work.poll_ctx;

  if constexpr (kReceiverCCType == ReceiverCCType::kEQDS) {
    eqds_cc_.inc_backlog(tx_work.len);
  }

  while (remaining_bytes > 0 || tx_work.len == 0) {
    auto payload_len = std::min(remaining_bytes, (int)kUcclPktDataMaxLen);
    auto frame_desc = socket_->pop_frame_desc();
    auto pkt_hdr = socket_->pop_pkt_hdr();

    // For tx, we do not allocate data buffer; therefore, we
    // should not free it later.
    auto* msgbuf = FrameDesc::Create(frame_desc, pkt_hdr, kUcclPktHdrLen,
                                     (uint64_t)app_buf_cursor, payload_len,
                                     app_mr->lkey, 0);
    msgbuf->set_poll_ctx(poll_ctx);

    // auto pkt_payload_addr = msgbuf->get_pkt_addr() + kUcclPktHdrLen;
    // memcpy(pkt_payload_addr, app_buf_cursor, payload_len);

    remaining_bytes -= payload_len;
    app_buf_cursor += payload_len;

    if (deser_msgs_head == nullptr) {
      deser_msgs_head = msgbuf;
      deser_msgs_tail = msgbuf;
    } else {
      deser_msgs_tail->set_next(msgbuf);
      deser_msgs_tail = msgbuf;
    }

    if (tx_work.len == 0) break;
  }
  deser_msgs_head->mark_first();
  deser_msgs_tail->mark_last();
  deser_msgs_tail->set_next(nullptr);

  tx_work.deser_msgs = deser_msgs_head;
  pending_tx_msgs_.push_back({tx_work, 0});

  VLOG(3) << "tx_prepare_messages size: " << tx_work.len << " bytes";

  deserialize_and_append_to_txtracking();
  transmit_pending_packets_drr(false);
}

void UcclFlow::process_rttprobe_rsp(uint64_t ts1, uint64_t ts2, uint64_t ts3,
                                    uint64_t ts4, uint32_t path_id) {
  if (unlikely(ts4 <= ts1 || ts3 <= ts2)) return;
  auto sender_latency = ts4 - ts1;
  auto receiver_latency = ts3 - ts2;
  if (unlikely(sender_latency <= receiver_latency)) return;

  auto sample_rtt_tsc = sender_latency - receiver_latency;
  port_path_rtt_[path_id] = sample_rtt_tsc;

  if constexpr (kSenderCCType == SenderCCType::kTimely) {
    timely_g_.timely_update_rate(rdtsc(), sample_rtt_tsc);
  }
  if constexpr (kSenderCCType == SenderCCType::kTimelyPP) {
    timely_pp_[path_id].timely_update_rate(rdtsc(), sample_rtt_tsc);
  }

  VLOG(3) << "sample_rtt_us " << to_usec(sample_rtt_tsc, freq_ghz)
          << " us, avg_rtt_diff " << timely_g_.timely_.get_avg_rtt_diff()
          << " us, timely rate " << timely_g_.timely_.get_rate_gbps()
          << " Gbps, "
          << "ts1 " << ts1 << " ts2 " << ts2 << " ts3 " << ts3 << " ts4 "
          << ts4;

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
  if (pcb_.max_rto_rexmits_consectutive_reached()) {
    DCHECK(false) << "Max RTO retransmits reached";
  }

  pcb_.advance_rto_tick();

  auto& ready_wheel = pcb_.get_ready_rto_wheel();
  while (!ready_wheel.empty()) {
    auto [msgbuf, seqno] = ready_wheel.front();
    ready_wheel.pop_front();
    if (swift::seqno_ge(seqno, pcb_.snd_una)) {
      if (!rto_retransmit((FrameDesc*)msgbuf, seqno))
        pcb_.add_to_rto_wheel(msgbuf, seqno);
    }
  }

  return true;
}

void UcclFlow::process_credit(UcclPullHdr const* ucclh) {
  auto pullno = ucclh->pullno.value();

  if (eqds_cc_.handle_pull(pullno)) {
    // Kick transmission.
    transmit_pending_packets_drr(true);
  }
}

void UcclFlow::process_ack(UcclPktHdr const* ucclh) {
  auto const* ucclsackh = reinterpret_cast<UcclSackHdr const*>(
      reinterpret_cast<uint8_t const*>(ucclh) + kUcclPktHdrLen);
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
      if (socket_->send_queue_wrs() >= kMaxUnackedPktsPerEngine) return;
      auto num_unacked_pkts = tx_tracking_.num_unacked_msgbufs();
      if (num_unacked_pkts >= kMaxUnackedPktsPerEngine) return;
      auto unacked_pkt_budget = kMaxUnackedPktsPerEngine - num_unacked_pkts;
      auto txq_free_entries = socket_->send_queue_free_space();
      auto hard_budget =
          std::min(std::min(txq_free_entries, unacked_pkt_budget),
                   (uint32_t)kSackBitmapSize);
      auto src_qp_idx = get_src_qp_rr();
      hard_budget = std::min(hard_budget,
                             socket_->send_queue_free_space_per_qp(src_qp_idx));

      uint32_t index = 0;
      while (sack_bitmap_count && msgbuf && index < hard_budget) {
        if (!can_rtx(msgbuf)) break;

        const size_t sack_bitmap_bucket_idx =
            index / swift::Pcb::kSackBitmapBucketSize;
        const size_t sack_bitmap_idx_in_bucket =
            index % swift::Pcb::kSackBitmapBucketSize;
        auto sack_bitmap =
            ucclsackh->sack_bitmap[sack_bitmap_bucket_idx].value();
        if ((sack_bitmap & (1ULL << sack_bitmap_idx_in_bucket)) == 0) {
          // We found a missing packet.
          auto seqno = pcb_.snd_una + index;

          // Unlike standard TCP or Machnet, we do not skip holes in
          // the SACK bitmap that have already been retransmitted, but
          // keep retransmitting them. This is because we have set a
          // relatively high kFastRexmitDupAckThres due to multi-path
          // and out-of-order delivery.

          VLOG(2) << "Fast recovery retransmitting " << seqno;
          auto const* missing_ucclh =
              reinterpret_cast<UcclPktHdr const*>(msgbuf->get_pkt_hdr_addr());
          // TODO(yang): tmp fix---they should be equal, need to
          // refine the way we maintain tx_but_unacked msgbufs chains.
          if (seqno == missing_ucclh->seqno.value()) {
            auto dst_qp_idx = get_dst_qp_pow2(src_qp_idx);
            auto path_id = src_dst_qp_to_path_id(src_qp_idx, dst_qp_idx);
#ifdef REXMIT_SET_PATH
            tx_tracking_.dec_unacked_pkts_pp(get_path_id(seqno));
            tx_tracking_.inc_unacked_pkts_pp(path_id);
            set_path_id(seqno, path_id);
#endif
            prepare_datapacket(msgbuf, path_id, seqno,
                               UcclPktHdr::UcclFlags::kData);
            // All frame will use src_qp_idx.
            msgbuf->set_src_qp_idx(UINT16_MAX);
            msgbuf->set_dest_ah(remote_ah_);
            msgbuf->set_dest_qpn(remote_meta_->qpn_list[dst_qp_idx]);

            missing_frames_.push_back(msgbuf);
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
                << " missing packets "
                << " ackno " << ackno << " duplicate_acks "
                << pcb_.duplicate_acks;
        // TODO(yang): handling the cases where the number of
        // missing frames is larger than the free send_queue size.
        socket_->post_send_wrs(missing_frames_, src_qp_idx);
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

    if constexpr (kSenderCCType == SenderCCType::kCubic) {
      cubic_g_.cubic_on_recv_ack(num_acked_packets);
    }
    if constexpr (kSenderCCType == SenderCCType::kCubicPP) {
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
  if (!msgbuf || !can_rtx(msgbuf)) return;
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
    auto const* ucclh =
        reinterpret_cast<UcclPktHdr const*>(msgbuf->get_pkt_hdr_addr());
    DCHECK_EQ(seqno, ucclh->seqno.value());

    auto [src_qp_idx, dst_qp_idx] = path_id_to_src_dst_qp(path_id);
    msgbuf->set_src_qp_idx(UINT16_MAX);
    msgbuf->set_dest_ah(remote_ah_);
    msgbuf->set_dest_qpn(remote_meta_->qpn_list[dst_qp_idx]);
    socket_->post_send_wr(msgbuf, src_qp_idx);
    pcb_.add_to_rto_wheel(msgbuf, seqno);
    pcb_.fast_rexmits++;
  }
}

bool UcclFlow::rto_retransmit(FrameDesc* msgbuf, uint32_t seqno) {
  // Avoid too many inflight WQEs.
  if (!can_rtx(msgbuf)) return false;

  VLOG(3) << "RTO retransmitting oldest unacked packet " << seqno;
  auto path_id = get_path_id_with_lowest_rtt();
#ifdef REXMIT_SET_PATH
  tx_tracking_.dec_unacked_pkts_pp(get_path_id(seqno));
  tx_tracking_.inc_unacked_pkts_pp(path_id);
  set_path_id(seqno, path_id);
#endif
  prepare_datapacket(msgbuf, path_id, seqno, UcclPktHdr::UcclFlags::kData);

  auto [src_qp_idx, dst_qp_idx] = path_id_to_src_dst_qp(path_id);
  msgbuf->set_src_qp_idx(UINT16_MAX);
  msgbuf->set_dest_ah(remote_ah_);
  msgbuf->set_dest_qpn(remote_meta_->qpn_list[dst_qp_idx]);
  socket_->post_send_wr(msgbuf, src_qp_idx);
  pcb_.add_to_rto_wheel(msgbuf, seqno);
  pcb_.rto_rexmits++;
  pcb_.rto_rexmits_consectutive++;

  if constexpr (kSenderCCType == SenderCCType::kCubic) {
    cubic_g_.cubic_on_packet_loss();
    VLOG(2) << "rto " << cubic_g_.to_string() << " inflight "
            << pcb_.snd_nxt - pcb_.snd_una << " "
            << tx_tracking_.num_unacked_msgbufs();
  }
  if constexpr (kSenderCCType == SenderCCType::kCubicPP) {
    auto path_id = get_path_id(seqno);
    cubic_pp_[path_id].cubic_on_packet_loss();
  }

  return true;
}

/**
 * @brief Helper function to transmit a number of packets from the queue
 * of pending TX data.
 */
uint32_t UcclFlow::transmit_pending_packets(uint32_t budget) {
  uint32_t permitted_packets = 0;

  if (!has_pending_packets()) return permitted_packets;

  // Avoid too many inflight WQEs.
  if (socket_->send_queue_wrs() >= kMaxUnackedPktsPerEngine)
    return permitted_packets;
  // Avoid too many unacked packets.
  auto num_unacked_pkts = tx_tracking_.num_unacked_msgbufs();
  if (num_unacked_pkts >= kMaxUnackedPktsPerEngine) return permitted_packets;

  auto unacked_pkt_budget = kMaxUnackedPktsPerEngine - num_unacked_pkts;
  auto txq_free_entries = socket_->send_queue_free_space();
  auto hard_budget = std::min(std::min(txq_free_entries, unacked_pkt_budget),
                              (uint32_t)kSackBitmapSize);
  auto src_qp_idx = get_src_qp_rr();
  hard_budget =
      std::min(hard_budget, socket_->send_queue_free_space_per_qp(src_qp_idx));
  // if (last_received_rwnd_ == 0) last_received_rwnd_ = 1;
  hard_budget = std::min(hard_budget, last_received_rwnd_);
  hard_budget = std::min(hard_budget, tx_tracking_.num_unsent_msgbufs());
  hard_budget = std::min(hard_budget, budget);

  if (hard_budget == 0) return permitted_packets;

  permitted_packets = hard_budget;

  if constexpr (kSenderCCType == SenderCCType::kTimely ||
                kSenderCCType == SenderCCType::kTimelyPP) {
    if constexpr (kReceiverCCType != ReceiverCCType::kNone) {
      // Just check, don't pop.
      permitted_packets = timely_g_.timely_check_ready_packets(hard_budget);
    } else
      permitted_packets = timely_g_.timely_pop_ready_packets(hard_budget);
  }
  if constexpr (kSenderCCType == SenderCCType::kCubic) {
    permitted_packets =
        std::min(permitted_packets, cubic_g_.cubic_effective_wnd());
  }
  if constexpr (kSenderCCType == SenderCCType::kCubicPP) {
    permitted_packets = hard_budget;
  }
  if constexpr (kReceiverCCType == ReceiverCCType::kEQDS) {
    permitted_packets = std::min(
        permitted_packets,
        tx_tracking_.convert_permitted_bytes_to_packets(eqds_cc_.credit()));

    if (permitted_packets &&
        !eqds_cc_.spend_credit(tx_tracking_.convert_permitted_packets_to_bytes(
            permitted_packets))) {
      permitted_packets = 0;
    }

    if constexpr (kSenderCCType == SenderCCType::kTimely ||
                  kSenderCCType == SenderCCType::kTimelyPP) {
      if (permitted_packets) {
        auto old_check_v = permitted_packets;
        permitted_packets =
            timely_g_.timely_pop_ready_packets(permitted_packets);
        DCHECK(old_check_v == permitted_packets);
      }
    }
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
    uint16_t dst_qp_idx = UINT16_MAX;
    uint32_t path_id = 0;
    uint32_t path_cwnd = 0;
    uint32_t path_unacked = 0;
    bool found_path = false;

    if constexpr (kSenderCCType == SenderCCType::kCubicPP) {
      // Avoiding sending too many packets on the same path.
      if (i % kSwitchPathThres == 0) {
        int tries = 0;
        while (tries++ < 16) {
          dst_qp_idx = get_dst_qp_pow2(src_qp_idx);
          path_id = src_dst_qp_to_path_id(src_qp_idx, dst_qp_idx);

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
      dst_qp_idx = get_dst_qp_pow2(src_qp_idx);
      path_id = src_dst_qp_to_path_id(src_qp_idx, dst_qp_idx);
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
      auto const* ucclh =
          reinterpret_cast<UcclPktHdr const*>(msgbuf->get_pkt_hdr_addr());
      VLOG(2) << "Transmitting seqno: " << seqno
              << " payload_len: " << ucclh->frame_len.value() - kUcclPktHdrLen;
    }
    auto net_flags = (i == 0) ? UcclPktHdr::UcclFlags::kDataRttProbe
                              : UcclPktHdr::UcclFlags::kData;
    prepare_datapacket(msgbuf, path_id, seqno, net_flags);
    msgbuf->set_src_qp_idx(UINT16_MAX);
    msgbuf->set_dest_ah(remote_ah_);
    msgbuf->set_dest_qpn(remote_meta_->qpn_list[dst_qp_idx]);
    pending_tx_frames_.push_back(msgbuf);

    if constexpr (kReceiverCCType == ReceiverCCType::kEQDS) {
      eqds_cc_.dec_backlog(msgbuf->get_pkt_data_len());
    }

    pcb_.add_to_rto_wheel(msgbuf, seqno);
  }

  // TX both data and ack frames.
  if (pending_tx_frames_.empty()) return permitted_packets;
  VLOG(3) << "tx packets " << pending_tx_frames_.size();
  // Considering ack coalescing.
  last_received_rwnd_ -= pending_tx_frames_.size();

  socket_->post_send_wrs(pending_tx_frames_, src_qp_idx);
  pending_tx_frames_.clear();

  return permitted_packets;
}

void UcclFlow::deserialize_and_append_to_txtracking() {
  if (pending_tx_msgs_.empty()) return;
  if (tx_tracking_.num_unsent_msgbufs() >= kMaxPktsInTimingWheel) return;
  auto deser_budget = kMaxPktsInTimingWheel - tx_tracking_.num_unsent_msgbufs();

  auto& [tx_work, cur_offset] = pending_tx_msgs_.front();
  FrameDesc* cur_msgbuf = tx_work.deser_msgs;
  FrameDesc* tx_msgbuf_head = cur_msgbuf;
  FrameDesc* tx_msgbuf_tail = nullptr;
  uint32_t num_tx_frames = 0;
  size_t remaining_bytes = tx_work.len - cur_offset;

  uint32_t path_id = kMaxPath;
  if constexpr (kSenderCCType == SenderCCType::kTimelyPP) {
    path_id = get_path_id_with_lowest_rtt();
  }

  auto now_tsc = rdtsc();
  while (cur_msgbuf != nullptr && num_tx_frames < deser_budget) {
    // The flow will free these Tx frames when receiving ACKs.
    if (remaining_bytes == tx_work.len) {
      DCHECK(cur_msgbuf->is_first());
    }

    auto payload_len = cur_msgbuf->get_pkt_data_len();

    // Both queue on one timing wheel.
    if constexpr (kSenderCCType == SenderCCType::kTimely) {
      timely_g_.timely_pace_packet(now_tsc, payload_len + kUcclPktHdrLen,
                                   cur_msgbuf);
    }
    if constexpr (kSenderCCType == SenderCCType::kTimelyPP) {
      // TODO(yang): consider per-path rate limiting? If so, we need to
      // maintain prev_desired_tx_tsc_ for each path, calculate two
      // timestamps (one from timely_g_, one from
      // timely_pp_[path_id]), and insert the larger one into the
      // timely_g_.
      double rate = timely_pp_[path_id].timely_rate();
      timely_g_.timely_pace_packet_with_rate(
          now_tsc, payload_len + kUcclPktHdrLen, cur_msgbuf, rate);
    }

    remaining_bytes -= payload_len;
    if (remaining_bytes == 0) {
      DCHECK(cur_msgbuf->next() == nullptr && cur_msgbuf->is_last());
    }

    tx_msgbuf_tail = cur_msgbuf;
    cur_msgbuf = cur_msgbuf->next();
    num_tx_frames++;
  }
  // This because deser_bugdget is > 0.
  DCHECK(num_tx_frames > 0);
  tx_msgbuf_tail->set_next(nullptr);

  if (remaining_bytes == 0) {
    DCHECK(cur_msgbuf == nullptr);
    // This message has been fully deserialized and added to tx tracking.
    pending_tx_msgs_.pop_front();
  } else {
    // Resuming the deserialization of this message in the next iteration.
    DCHECK(cur_msgbuf != nullptr);
    tx_work.deser_msgs = cur_msgbuf;
    cur_offset = tx_work.len - remaining_bytes;
  }

  tx_tracking_.append(tx_msgbuf_head, tx_msgbuf_tail, num_tx_frames);

  // Recursively call this function to append more messages to the tx.
  if (tx_tracking_.num_unsent_msgbufs() < kMaxPktsInTimingWheel &&
      !pending_tx_msgs_.empty()) {
    deserialize_and_append_to_txtracking();
  }
}

void UcclFlow::prepare_datapacket(FrameDesc* msgbuf, uint32_t path_id,
                                  uint32_t seqno,
                                  const UcclPktHdr::UcclFlags net_flags) {
  // Header length after before the payload.
  uint32_t frame_len = msgbuf->get_pkt_hdr_len() + msgbuf->get_pkt_data_len();
  DCHECK_LE(frame_len, EFA_MTU);
  uint8_t* pkt_addr = (uint8_t*)msgbuf->get_pkt_hdr_addr();
  auto* ucclh = reinterpret_cast<UcclPktHdr*>(pkt_addr);

  // Prepare the Uccl-specific header.
  ucclh->magic = be16_t(UcclPktHdr::kMagic);
  ucclh->engine_id = remote_engine_idx_;
  ucclh->path_id = (uint16_t)path_id;
  ucclh->net_flags = net_flags;
  ucclh->ackno = be32_t(UINT32_MAX);
  // This fills the FrameDesc flags into the outgoing packet msg_flags.
  ucclh->msg_flags = msgbuf->get_msg_flags();
  ucclh->frame_len = be16_t(frame_len);
  ucclh->seqno = be32_t(seqno);
  ucclh->flow_id = be64_t(peer_flow_id_);

  ucclh->timestamp1 =
      (net_flags == UcclPktHdr::UcclFlags::kDataRttProbe)
          ? rdtsc() + ns_to_cycles(socket_->send_queue_estimated_latency_ns(),
                                   freq_ghz)
          : 0;
  // let the receiver fill this in when receiving this data packet.
  ucclh->timestamp2 = 0;

  if constexpr (kReceiverCCType == ReceiverCCType::kEQDS) {
    ucclh->pullno = be16_t(eqds_cc_.compute_pull_target());
  }
}

bool UcclFlow::send_pullpacket(PullQuanta const& pullno) {
  const size_t kControlPayloadBytes = kUcclPullHdrLen;

  auto frame_desc_buff = credit_qp_ctx_->pacer_pop_frame_desc();
  auto pkt_hdr_buff = credit_qp_ctx_->pacer_pop_pkt_hdr();
  auto frame = FrameDesc::Create(frame_desc_buff, pkt_hdr_buff,
                                 kControlPayloadBytes, 0, 0, 0, 0);

  uint8_t* pkt_addr = (uint8_t*)pkt_hdr_buff;
  auto* ucclh = (UcclPullHdr*)(pkt_addr);
  ucclh->magic = be16_t(UcclPktHdr::kMagic);
  ucclh->net_flags = UcclPktHdr::UcclFlags::kCredit;
  ucclh->frame_len = be16_t(kControlPayloadBytes);
  ucclh->flow_id = be64_t(peer_flow_id_);

  ucclh->pullno = be16_t(pullno);

  frame->set_dest_ah(remote_ah_);
  frame->set_dest_qpn(
      remote_meta_->qpn_list_credit[credit_qpidx_rr_++ % kMaxDstQPCredit]);

  struct ibv_sge sge = {
      .addr = frame->get_pkt_hdr_addr(),
      .length = frame->get_pkt_hdr_len(),
      .lkey = credit_qp_ctx_->get_pacer_hdr_pool_lkey(),
  };

  struct ibv_send_wr wr, *bad_wr;
  wr.wr_id = (uint64_t)frame;
  wr.opcode = IBV_WR_SEND;
  wr.num_sge = 1;
  wr.sg_list = &sge;
  wr.next = nullptr;
  wr.wr.ud.ah = frame->get_dest_ah();
  wr.wr.ud.remote_qpn = frame->get_dest_qpn();
  wr.wr.ud.remote_qkey = QKEY;
  wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;

  DCHECK(ibv_post_send(
             credit_qp_ctx_->get_qp_by_idx(credit_qpidx_rr_ % kMaxSrcQPCredit),
             &wr, &bad_wr) == 0);

  return true;
}

FrameDesc* UcclFlow::craft_ackpacket(uint32_t path_id, uint32_t seqno,
                                     uint32_t ackno,
                                     const UcclPktHdr::UcclFlags net_flags,
                                     uint64_t ts1, uint64_t ts2,
                                     uint32_t rwnd) {
  const size_t kControlPayloadBytes = kUcclPktHdrLen + kUcclSackHdrLen;
  auto frame_desc = socket_->pop_frame_desc();
  auto pkt_hdr = socket_->pop_pkt_hdr();
  auto msgbuf =
      FrameDesc::Create(frame_desc, pkt_hdr, kControlPayloadBytes, 0, 0, 0, 0);

  uint8_t* pkt_addr = (uint8_t*)pkt_hdr;
  auto* ucclh = (UcclPktHdr*)(pkt_addr);
  ucclh->magic = be16_t(UcclPktHdr::kMagic);
  ucclh->engine_id = remote_engine_idx_;
  ucclh->path_id = (uint16_t)path_id;
  ucclh->net_flags = net_flags;
  ucclh->msg_flags = 0;
  ucclh->frame_len = be16_t(kControlPayloadBytes);
  ucclh->seqno = be32_t(seqno);
  ucclh->ackno = be32_t(ackno);
  ucclh->flow_id = be64_t(peer_flow_id_);
  ucclh->timestamp1 = ts1;
  ucclh->timestamp2 = ts2;

  auto* ucclsackh = (UcclSackHdr*)(pkt_addr + kUcclPktHdrLen);
  for (size_t i = 0; i < sizeof(UcclSackHdr::sack_bitmap) /
                             sizeof(UcclSackHdr::sack_bitmap[0]);
       ++i) {
    ucclsackh->sack_bitmap[i] = be64_t(pcb_.sack_bitmap[i]);
  }
  ucclsackh->sack_bitmap_count = be16_t(pcb_.sack_bitmap_count);

  ucclsackh->timestamp3 =
      (net_flags == UcclPktHdr::UcclFlags::kAckRttProbe)
          ? rdtsc() + ns_to_cycles(socket_->send_queue_estimated_latency_ns(),
                                   freq_ghz)
          : 0;
  ucclsackh->rwnd = be32_t(rwnd);

  return msgbuf;
}

void UcclEngine::sender_only_run() {
  Channel::Msg rx_work;
  Channel::Msg tx_work;

  while (!shutdown_) {
    // Calculate the cycles elapsed since last periodic processing.
    auto now_tsc = rdtsc();
    auto const elapsed_tsc = now_tsc - last_periodic_tsc_;

    if (elapsed_tsc >= kSlowTimerIntervalTsc_) {
      // Perform periodic processing.
      periodic_process();
      last_periodic_tsc_ = now_tsc;
    }

    auto [frames, polled_send_acks] = socket_->poll_ctrl_cq(RECV_BATCH_SIZE);

    if constexpr (kReceiverCCType == ReceiverCCType::kEQDS) {
      // Receive credits.
      auto _frames = credit_qp_ctx_->engine_poll_credit_cq();
      frames.insert(frames.end(), _frames.begin(), _frames.end());
    }

    if (frames.size()) {
      process_rx_msg(frames);
    }

    while (Channel::dequeue_sc(channel_->tx_task_q_, &tx_work)) {
      // Make data written by the app thread visible to the engine.
      tx_work.poll_ctx->read_barrier();
      active_flows_map_[tx_work.flow_id]->tx_prepare_messages(tx_work);
    }

    for (auto& [flow_id, flow] : active_flows_map_) {
      flow->deserialize_and_append_to_txtracking();

      flow->transmit_pending_packets_drr(false);
    }

    socket_->poll_send_cq();
  }

  // This will reset flow pcb state.
  for (auto [flow_id, flow] : active_flows_map_) {
    flow->shutdown();
    delete flow;
  }
  // This will flush all unpolled tx frames.
  socket_->shutdown();
}

void UcclEngine::receiver_only_run() {
  Channel::Msg rx_work;
  Channel::Msg tx_work;

  while (!shutdown_) {
    // Calculate the cycles elapsed since last periodic processing.
    auto now_tsc = rdtsc();
    auto const elapsed_tsc = now_tsc - last_periodic_tsc_;

    if (elapsed_tsc >= kSlowTimerIntervalTsc_) {
      // Perform periodic processing.
      periodic_process();
      last_periodic_tsc_ = now_tsc;
    }

    while (Channel::dequeue_sc(channel_->rx_task_q_, &rx_work)) {
      active_flows_map_[rx_work.flow_id]->rx_supply_app_buf(rx_work);
    }

    auto frames = socket_->poll_recv_cq(RECV_BATCH_SIZE);
    auto [_frames, polled_send_acks] = socket_->poll_ctrl_cq(RECV_BATCH_SIZE);
    frames.insert(frames.end(), _frames.begin(), _frames.end());

    if (frames.size()) {
      process_rx_msg(frames);
    }

    for (auto& [flow_id, flow] : active_flows_map_) {
      // Driving the rx buffer matching to incoming packets.
      flow->rx_tracking_.try_copy_msgbuf_to_appbuf(nullptr);
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

void UcclEngine::run() {
  Channel::Msg rx_work;
  Channel::Msg tx_work;

  while (!shutdown_) {
    // Calculate the cycles elapsed since last periodic processing.
    auto now_tsc = rdtsc();
    auto const elapsed_tsc = now_tsc - last_periodic_tsc_;

    if (elapsed_tsc >= kSlowTimerIntervalTsc_) {
      // Perform periodic processing.
      periodic_process();
      last_periodic_tsc_ = now_tsc;
    }

    while (Channel::dequeue_sc(channel_->rx_task_q_, &rx_work)) {
      active_flows_map_[rx_work.flow_id]->rx_supply_app_buf(rx_work);
    }

    auto frames = socket_->poll_recv_cq(RECV_BATCH_SIZE);
    auto [_frames, polled_send_acks] = socket_->poll_ctrl_cq(RECV_BATCH_SIZE);
    frames.insert(frames.end(), _frames.begin(), _frames.end());

    if constexpr (kReceiverCCType == ReceiverCCType::kEQDS) {
      // Receive credits.
      _frames = credit_qp_ctx_->engine_poll_credit_cq();
      frames.insert(frames.end(), _frames.begin(), _frames.end());
    }

    if (frames.size()) {
      process_rx_msg(frames);
    }

    while (Channel::dequeue_sc(channel_->tx_task_q_, &tx_work)) {
      // Make data written by the app thread visible to the engine.
      tx_work.poll_ctx->read_barrier();
      active_flows_map_[tx_work.flow_id]->tx_prepare_messages(tx_work);
    }

    for (auto& [flow_id, flow] : active_flows_map_) {
      // Driving the rx buffer matching to incoming packets.
      flow->rx_tracking_.try_copy_msgbuf_to_appbuf(nullptr);
      flow->deserialize_and_append_to_txtracking();

      flow->transmit_pending_packets_drr(false);
    }

    socket_->poll_send_cq();
  }

  // This will reset flow pcb state.
  for (auto [flow_id, flow] : active_flows_map_) {
    flow->shutdown();
    delete flow;
  }
  // This will flush all unpolled tx frames.
  socket_->shutdown();
}

void UcclEngine::process_rx_msg(std::vector<FrameDesc*>& pkt_msgs) {
  for (auto& msgbuf : pkt_msgs) {
#ifdef USE_SRD_FOR_CTRL
    uint8_t* pkt_addr = (uint8_t*)msgbuf->get_pkt_hdr_addr() +
                        (msgbuf->get_pkt_data_len() == 0 ? 0 : EFA_UD_ADDITION);
#else
    uint8_t* pkt_addr = (uint8_t*)msgbuf->get_pkt_hdr_addr() + EFA_UD_ADDITION;
#endif
    auto* ucclh = reinterpret_cast<UcclPktHdr*>(pkt_addr);
    auto frame_len = ucclh->frame_len.value();

    // Record the incoming packet UcclPktHdr.msg_flags in
    // FrameDesc.
    msgbuf->set_msg_flags(ucclh->msg_flags);

    if (msgbuf->is_last()) {
      VLOG(3) << "Received seqno: " << ucclh->seqno.value()
              << " payload_len: " << frame_len - kUcclPktHdrLen;
    }

    auto flow_id = ucclh->flow_id.value();

    auto it = active_flows_map_.find(flow_id);
    if (it == active_flows_map_.end()) {
      LOG_EVERY_N(ERROR, 1000000) << "process_rx_msg unknown flow " << flow_id;
      for (auto [flow_id, flow] : active_flows_map_) {
        LOG_EVERY_N(ERROR, 1000000)
            << "                active flow " << flow_id;
      }
      socket_->push_pkt_hdr(msgbuf->get_pkt_hdr_addr());
      // In case of ack packets which have no payload.
      if (msgbuf->get_pkt_data_len()) {
        auto pkt_data_addr = msgbuf->get_pkt_data_addr();
        CHECK(pkt_data_addr);
        socket_->push_pkt_data(pkt_data_addr);
      }
      socket_->push_frame_desc((uint64_t)msgbuf);
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

  process_ctl_reqs();

  if constexpr (kSplitSendRecvEngine) {
    if (local_engine_idx_ % 2) return;
  }

  handle_rto();
}

void UcclEngine::handle_rto() {
  for (auto [flow_id, flow] : active_flows_map_) {
    auto is_active_flow = flow->periodic_check();
    DCHECK(is_active_flow);
  }
}

void UcclEngine::process_ctl_reqs() {
  Channel::CtrlMsg ctrl_work;
  while (Channel::dequeue_sc(channel_->ctrl_task_q_, &ctrl_work)) {
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
  int ret;
  auto flow_id = ctrl_work.flow_id;
  auto remote_addr = ctrl_work.remote_ip;
  auto remote_ip_str = ip_to_str(htonl(remote_addr));
  auto remote_engine_idx = ctrl_work.remote_engine_idx;
  auto* local_meta = ctrl_work.local_meta;
  auto* remote_meta = ctrl_work.remote_meta;
  auto is_sender = ctrl_work.is_sender;
  auto* poll_ctx = ctrl_work.poll_ctx;
  poll_ctx->read_barrier();

  LOG(INFO) << "[Engine] handle_install_flow_on_engine " << local_engine_idx_
            << " for flow " << flow_id;

  auto* flow = new UcclFlow(local_ip_str_, remote_ip_str, local_meta,
                            remote_meta, local_engine_idx_, remote_engine_idx,
                            socket_, credit_qp_ctx_, eqds_channel_, channel_,
                            flow_id, active_flows_map_, is_sender);
  auto* dev = EFAFactory::GetEFADevice(socket_->dev_idx());
  flow->remote_ah_ = dev->create_ah(remote_meta->gid);

  std::tie(std::ignore, ret) = active_flows_map_.insert({flow_id, flow});
  DCHECK(ret);

  std::string arrow = is_sender ? "->" : "<-";
  LOG(INFO) << "[Engine] install FlowID " << flow_id << ": " << local_ip_str_
            << Format("(%d)", local_engine_idx_) << arrow << remote_ip_str
            << Format("(%d)", remote_engine_idx);

  if constexpr (kReceiverCCType == ReceiverCCType::kEQDS) {
    if (!is_sender) flow->request_pull();
  }

  // Wakeup app thread waiting on endpoint.
  if (--(poll_ctx->num_unfinished) == 0) {
    std::lock_guard<std::mutex> lock(poll_ctx->mu);
    poll_ctx->done = true;
    poll_ctx->cv.notify_one();
  }
}

std::string UcclEngine::status_to_string(bool abbrev) {
  std::string s;
  int cnt = 0;
  s += Format("\n\tEngine %d active flows %lu:", local_engine_idx_,
              active_flows_map_.size());
  for (auto [flow_id, flow] : active_flows_map_) s += Format(" %lu", flow_id);
  s += socket_->to_string();
  if (abbrev) return s;

  for (auto [flow_id, flow] : active_flows_map_) {
    std::string arrow = flow->is_sender_ ? "->" : "<-";
    s += Format("\n\t\tFlow %lu: %s (%u) %s %s (%u)", flow_id,
                flow->local_ip_str_.c_str(), flow->local_engine_idx_,
                arrow.c_str(), flow->remote_ip_str_.c_str(),
                flow->remote_engine_idx_);
    s += flow->to_string();
    cnt++;
    if (cnt == 4) break;
  }
  if (cnt < active_flows_map_.size())
    s += Format("\n\t\t... %d more flows", active_flows_map_.size() - cnt);
  return s;
}

bool Endpoint::create_engine_and_add_to_engine_future(
    int engine_idx, int gpu_idx,
    std::vector<std::future<std::unique_ptr<UcclEngine>>>& engine_futures) {
  LOG(INFO) << "Creating Engines by Engine Idx " << engine_idx << " GPU Idx "
            << gpu_idx;
  {
    std::lock_guard<std::mutex> lock(engine_map_mutex_);
    if (engine_id_to_engine_map_.find(engine_idx) !=
        engine_id_to_engine_map_.end()) {
      LOG(INFO) << "Engine " << engine_idx << " already exists";
      return false;
    }
  }

  auto dev_idx = get_dev_idx_by_engine_idx(engine_idx);
  auto socket_idx = engine_idx;

  CHECK_EQ(gpu_idx, get_gpu_idx_by_engine_idx(engine_idx))
      << "gpu_idx " << gpu_idx
      << " != " << get_gpu_idx_by_engine_idx(engine_idx);

  std::string local_ip_str;
  auto ret = util_efa_get_ip_from_dev_idx(0, &local_ip_str);
  CHECK_EQ(ret, 0) << "Failed to get IP address from dev idx 0";

  // Creating engines sequentially to have inorder QPNs.
  auto engine = std::make_unique<UcclEngine>(
      local_ip_str, gpu_idx, dev_idx, socket_idx, channel_vec_[engine_idx],
      eqds_[dev_idx]->credit_qp_ctx_[get_engine_off_by_engine_idx(engine_idx)],
      &eqds_[dev_idx]->channel_);

  std::promise<std::unique_ptr<UcclEngine>> engine_promise;
  auto engine_future = engine_promise.get_future();
  engine_futures.emplace_back(std::move(engine_future));

  // GPU 0-3 on numa 0, and GPU 4-7 on numa 1.
  // This can be found with nvidia-smi topo -m.
  auto engine_cpu_start = ENGINE_CPU_START[gpu_idx / 4];
  // Total possible GPUs: 8 * kNumEnginesPerVdev, separated into two numas.
  auto engine_th_cpuid =
      engine_cpu_start + engine_idx % (8 * kNumEnginesPerVdev / 2);

  // Spawning a new thread to init engine and run the engine loop.
  engine_th_vec_.emplace_back(
      std::make_unique<std::thread>([this, engine_idx, engine_th_cpuid,
                                     engine_promise = std::move(engine_promise),
                                     engine = std::move(engine)]() mutable {
        pin_thread_to_cpu(engine_th_cpuid);
        LOG(INFO) << "[Engine] thread " << engine_idx << " running on CPU "
                  << engine_th_cpuid;

        auto* engine_ptr = engine.get();
        engine_promise.set_value(std::move(engine));

        if constexpr (kSplitSendRecvEngine) {
          if (engine_idx % 2 == 0)
            engine_ptr->sender_only_run();
          else
            engine_ptr->receiver_only_run();
        } else
          engine_ptr->run();
      }));
  return true;
}

bool Endpoint::initialize_engine_by_gpu_idx(int gpu_idx) {
  // TODO(MaoZiming): check only called once.
  static std::once_flag flag_once;
  std::call_once(flag_once, [this, gpu_idx]() {
    listen_port_cur_.store(kBootstrapPort + gpu_idx * 1000);
  });

  std::vector<std::future<std::unique_ptr<UcclEngine>>> engine_futures;
  for (int i = 0; i < kNumEnginesPerVdev; i++) {
    auto engine_idx = gpu_idx * kNumEnginesPerVdev + i;
    create_engine_and_add_to_engine_future(engine_idx, gpu_idx, engine_futures);
  }

  int i = 0;
  for (auto& engine_future : engine_futures) {
    // engine_vec_.emplace_back(std::move(engine_future.get()));
    auto engine_idx = gpu_idx * kNumEnginesPerVdev + i;
    {
      std::lock_guard<std::mutex> lock(engine_map_mutex_);
      engine_id_to_engine_map_[engine_idx] = std::move(engine_future.get());
    }
    i++;
  }
  return true;
}

Endpoint::Endpoint() : stats_thread_([this]() { stats_thread_fn(); }) {
  LOG(INFO) << "Creating EFAFactory (Lazy Init)";

  static std::once_flag flag_once;
  std::call_once(flag_once, []() { EFAFactory::Init(); });

  CHECK_LE(kNumEngines, NUM_CPUS / 4)
      << "num_queues should be less than or equal to the number of CPUs "
         "/ 4";

  LOG(INFO) << "Creating Channels";
  for (int i = 0; i < kNumEngines; i++) channel_vec_[i] = new Channel();
  LOG(INFO) << "Creating Pacers";
  for (int i = 0; i < NUM_DEVICES; i++) eqds_[i] = new eqds::EQDS(i);

#ifdef LAZY_CREATE_ENGINE
  LOG(INFO) << "Endpoint() skips creating Engines";
#else
  LOG(INFO) << "Creating Engines";
  std::vector<std::future<std::unique_ptr<UcclEngine>>> engine_futures;

  for (int i = 0; i < kNumEngines; i++) {
    auto gpu_idx = get_gpu_idx_by_engine_idx(i);
    create_engine_and_add_to_engine_future(i, gpu_idx, engine_futures);
  }

  std::vector<UcclEngine*> engines;
  for (auto& engine_future : engine_futures) {
    engine_vec_.emplace_back(std::move(engine_future.get()));
    engines.push_back(engine_vec_.back().get());
  }
#endif

  ctx_pool_ = new SharedPool<PollCtx*, true>(NUM_FRAMES * 4);
  ctx_pool_buf_ = new uint8_t[NUM_FRAMES * 4 * sizeof(PollCtx)];

  for (int i = 0; i < NUM_FRAMES * 4; i++) {
    ctx_pool_->push(new (ctx_pool_buf_ + i * sizeof(PollCtx)) PollCtx());
  }

#if !defined(EMULATE_RC_ZC) && !defined(SCATTERED_MEMCPY)
  // Creating copy thread for each engine.
  for (int i = 0; i < kNumEngines; i++) {
    auto gpu_idx = get_gpu_idx_by_engine_idx(i);
    auto engine_cpu_start = ENGINE_CPU_START[gpu_idx / 4];
    auto num_engines_per_numa =
        gpu_idx / 4 == 0 ? kNumEngines : kNumEngines / 2;
    auto copy_th_cpuid = engine_cpu_start + num_engines_per_numa +
                         i % (8 * kNumEnginesPerVdev / 2);

    copy_th_vec_.emplace_back(std::make_unique<std::thread>(
        [this, i, engine = engines[i], copy_th_cpuid]() {
          pin_thread_to_cpu(copy_th_cpuid);
          LOG(INFO) << "[Copy] thread " << i << " running on CPU "
                    << copy_th_cpuid;
          RXTracking::copy_thread_func(i, engine);
        }));
  }
#endif
}

// TODO(MaoZiming): Deprecate this constructor. Use Endpoint()
Endpoint::Endpoint(int gpu)
    : gpu_(gpu), stats_thread_([this]() { stats_thread_fn(); }) {
  listen_port_cur_.store(kBootstrapPort + gpu_ * 1000);

  LOG(INFO) << "Creating EFAFactory";
  // Create UDS socket and get umem_fd and xsk_ids.
  static std::once_flag flag_once;
  std::call_once(flag_once, [gpu]() { EFAFactory::Init(gpu); });

  CHECK_LE(kNumEngines, NUM_CPUS / 4)
      << "num_queues should be less than or equal to the number of CPUs "
         "/ 4";

  LOG(INFO) << "Creating Channels";

  // Create multiple engines. Each engine has its own thread and channel to
  // let the endpoint communicate with.
  for (int i = 0; i < kNumEngines; i++) channel_vec_[i] = new Channel();

  // Receiver-driven CC pacer.
  LOG(INFO) << "Creating Pacers";
  for (int i = 0; i < NUM_DEVICES; i++) eqds_[i] = new eqds::EQDS(i);

  LOG(INFO) << "Creating Engines";

  std::vector<std::future<std::unique_ptr<UcclEngine>>> engine_futures;
  for (int i = 0; i < kNumEngines; i++) {
    create_engine_and_add_to_engine_future(i, gpu_, engine_futures);
  }
  std::vector<UcclEngine*> engines;
  for (auto& engine_future : engine_futures) {
    engine_vec_.emplace_back(std::move(engine_future.get()));
    engines.push_back(engine_vec_.back().get());
  }

  ctx_pool_ = new SharedPool<PollCtx*, true>(NUM_FRAMES * 4);
  ctx_pool_buf_ = new uint8_t[NUM_FRAMES * 4 * sizeof(PollCtx)];
  for (int i = 0; i < NUM_FRAMES * 4; i++) {
    ctx_pool_->push(new (ctx_pool_buf_ + i * sizeof(PollCtx)) PollCtx());
  }

#if !defined(EMULATE_RC_ZC) && !defined(SCATTERED_MEMCPY)
  // Creating copy thread for each engine.
  for (int i = 0; i < kNumEngines; i++) {
    auto gpu_idx = get_gpu_idx_by_engine_idx(i);
    auto engine_cpu_start = ENGINE_CPU_START[gpu_idx / 4];
    auto num_engines_per_numa =
        gpu_idx / 4 == 0 ? kNumEngines : kNumEngines / 2;
    auto copy_th_cpuid = engine_cpu_start + num_engines_per_numa +
                         i % (8 * kNumEnginesPerVdev / 2);

    copy_th_vec_.emplace_back(std::make_unique<std::thread>(
        [this, i, engine = engines[i], copy_th_cpuid]() {
          pin_thread_to_cpu(copy_th_cpuid);
          LOG(INFO) << "[Copy] thread " << i << " running on CPU "
                    << copy_th_cpuid;
          RXTracking::copy_thread_func(i, engine);
        }));
  }
#endif
}

Endpoint::~Endpoint() {
  for (auto& engine : engine_vec_) engine->shutdown();
  {
    std::lock_guard<std::mutex> lock(engine_map_mutex_);
    for (auto& [engine_id, engine] : engine_id_to_engine_map_) {
      engine->shutdown();
    }
  }
  for (auto& engine_th : engine_th_vec_) engine_th->join();
  for (auto& copy_th : copy_th_vec_) copy_th->join();
  for (int i = 0; i < kNumEngines; i++) delete channel_vec_[i];
  for (auto listen_fd : listen_fd_vec_) close(listen_fd);
  delete ctx_pool_;
  delete[] ctx_pool_buf_;

  // Receiver-driven CC pacer.
  for (int i = 0; i < NUM_DEVICES; i++) delete eqds_[i];

  {
    std::lock_guard<std::mutex> lock(fd_map_mu_);
    for (auto& [flow_id, boostrap_id] : fd_map_) {
      close(boostrap_id);
    }
  }

  static std::once_flag flag_once;
  std::call_once(flag_once, []() { EFAFactory::Shutdown(); });

  {
    std::lock_guard<std::mutex> lock(stats_mu_);
    shutdown_ = true;
    stats_cv_.notify_all();
  }
  stats_thread_.join();
}

std::tuple<uint16_t, int> Endpoint::uccl_listen() {
  // Create listening socket
  auto listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  DCHECK(listen_fd >= 0) << "ERROR: opening socket";

  int flag = 1;
  DCHECK(setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(int)) >=
         0)
      << "ERROR: setsockopt SO_REUSEADDR fails";

  auto listen_port = listen_port_cur_.fetch_add(1);

  struct sockaddr_in serv_addr;
  bzero((char*)&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  serv_addr.sin_port = htons(listen_port);

  // Get the device currently assigned to this process
  int gpu;
  cudaGetDevice(&gpu);

  int localRank = 0;
  // Try to determine local rank from environment variable
  char const* localRankEnv = std::getenv("LOCAL_RANK");
  if (localRankEnv) {
    localRank = std::atoi(localRankEnv);
  } else {
    localRank = gpu;  // fallback guess
  }

  LOG(INFO) << "[Endpoint] Rank " << localRank << " trying to bind on port "
            << listen_port << " (fd=" << listen_fd << ")";

  int bind_ret =
      bind(listen_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
  if (bind_ret < 0) {
    LOG(ERROR) << "[Endpoint] Rank " << localRank << " failed to bind port "
               << listen_port << " (errno: " << errno << ", " << strerror(errno)
               << ")";
  }
  DCHECK(bind_ret >= 0) << "ERROR: binding";

  DCHECK(!listen(listen_fd, 128)) << "ERROR: listen";
  LOG(INFO) << "[Endpoint] server ready, listening on port " << listen_port;

  std::lock_guard<std::mutex> lock(listen_mu_);
  listen_port_vec_.push_back(listen_port);
  listen_fd_vec_.push_back(listen_fd);

  return {listen_port, listen_fd};
}

ConnID Endpoint::uccl_connect(int local_vdev, int remote_vdev,
                              std::string remote_ip, uint16_t listen_port) {
  int local_pdev = get_pdev(local_vdev);
  int remote_pdev = get_pdev(remote_vdev);

  struct sockaddr_in serv_addr = {};
  struct hostent* server;
  int ret;
  int bootstrap_fd;
  bool local_lock_first = false;
  bool is_sender = true;

  bootstrap_fd = socket(AF_INET, SOCK_STREAM, 0);
  DCHECK(bootstrap_fd >= 0) << "uccl_connect: socket()";

  server = gethostbyname(remote_ip.c_str());
  DCHECK(server) << "uccl_connect: gethostbyname()";

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = str_to_ip(remote_ip.c_str());
  serv_addr.sin_port = htons(listen_port);

  // Connect and set nonblocking and nodelay
  while (
      connect(bootstrap_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr))) {
    LOG(INFO) << "[Endpoint] connecting... Make sure the server is up.";
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  LOG(INFO) << "[Endpoint] connected to <" << remote_ip << ", " << remote_vdev
            << ">:" << listen_port << " bootstrap_fd " << bootstrap_fd;

  fcntl(bootstrap_fd, F_SETFL, O_NONBLOCK);
  int flag = 1;
  setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void*)&flag, sizeof(int));

  FlowID flow_id;
  while (true) {
    int ret = receive_message(bootstrap_fd, &flow_id, sizeof(FlowID));
    DCHECK(ret == sizeof(FlowID));
    LOG(INFO) << "[Endpoint] connect: receive proposed FlowID: " << flow_id;

    // Check if the flow ID is unique, and return it to the server.
    bool unique;
    {
      std::lock_guard<std::mutex> lock(fd_map_mu_);
      unique = (fd_map_.find(flow_id) == fd_map_.end());
      if (unique) fd_map_[flow_id] = bootstrap_fd;
    }

    ret = send_message(bootstrap_fd, &unique, sizeof(bool));
    DCHECK(ret == sizeof(bool));

    if (unique) {
      // Send our device ID to the server.
      ret = send_message(bootstrap_fd, &local_vdev, sizeof(int));
      DCHECK(ret == sizeof(int)) << "uccl_connect: send_message()";
      break;
    }
  }
  uint32_t local_engine_idx;
  if constexpr (kSplitSendRecvEngine)
    local_engine_idx = find_dedicated_engine_idx(local_vdev, is_sender);
  else
    local_engine_idx =
        find_least_loaded_engine_idx_and_update(local_vdev, flow_id, is_sender);
  CHECK_GE(local_engine_idx, 0);

  install_flow_on_engine(flow_id, remote_ip, local_engine_idx, bootstrap_fd,
                         is_sender);

  return ConnID{.flow_id = flow_id,
                .engine_idx = (uint32_t)local_engine_idx,
                .boostrap_id = bootstrap_fd};
}

ConnID Endpoint::uccl_accept(int local_vdev, int* remote_vdev,
                             std::string& remote_ip, int listen_fd) {
  int local_pdev = get_pdev(local_vdev);

  struct sockaddr_in cli_addr;
  socklen_t clilen = sizeof(cli_addr);
  int bootstrap_fd;
  int ret;
  bool local_lock_first = false;
  bool is_sender = false;

  // Accept connection and set nonblocking and nodelay
  bootstrap_fd = accept(listen_fd, (struct sockaddr*)&cli_addr, &clilen);
  DCHECK(bootstrap_fd >= 0) << "uccl_accept: accept()";
  remote_ip = ip_to_str(cli_addr.sin_addr.s_addr);

  LOG(INFO) << "[Endpoint] accept from " << remote_ip << ":"
            << cli_addr.sin_port << " bootstrap_fd " << bootstrap_fd;

  fcntl(bootstrap_fd, F_SETFL, O_NONBLOCK);
  int flag = 1;
  setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void*)&flag, sizeof(int));

  // Generate unique flow ID for both client and server.
  FlowID flow_id;
  while (true) {
    // generate flow_id sequentially for better debugging
    static std::atomic<uint64_t> base_flow_id = 0;
    flow_id = base_flow_id++;
    bool unique;
    {
      std::lock_guard<std::mutex> lock(fd_map_mu_);
      unique = (fd_map_.find(flow_id) == fd_map_.end());
      if (unique) {
        // Speculatively insert the flow ID.
        fd_map_[flow_id] = bootstrap_fd;
      } else {
        continue;
      }
    }

    LOG(INFO) << "[Endpoint] accept: propose FlowID: " << flow_id;

    // Allowing flow src and dst to be the same process.
    auto peer_flow_id = get_peer_flow_id(flow_id);
    int ret = send_message(bootstrap_fd, &peer_flow_id, sizeof(FlowID));
    DCHECK(ret == sizeof(FlowID));

    bool unique_from_client;
    ret = receive_message(bootstrap_fd, &unique_from_client, sizeof(bool));
    DCHECK(ret == sizeof(bool));

    if (unique_from_client) {
      // Receive the remote_dev from client.
      ret = receive_message(bootstrap_fd, remote_vdev, sizeof(int));
      DCHECK(ret == sizeof(int));
      break;
    } else {
      // Remove the speculatively inserted flow ID.
      std::lock_guard<std::mutex> lock(fd_map_mu_);
      DCHECK(1 == fd_map_.erase(flow_id));
    }
  }

  uint32_t local_engine_idx;
  if constexpr (kSplitSendRecvEngine)
    local_engine_idx = find_dedicated_engine_idx(local_vdev, is_sender);
  else
    local_engine_idx =
        find_least_loaded_engine_idx_and_update(local_vdev, flow_id, is_sender);
  CHECK_GE(local_engine_idx, 0);

  install_flow_on_engine(flow_id, remote_ip, local_engine_idx, bootstrap_fd,
                         is_sender);

  return ConnID{.flow_id = flow_id,
                .engine_idx = (uint32_t)local_engine_idx,
                .boostrap_id = bootstrap_fd};
}

bool Endpoint::uccl_send(ConnID conn_id, void const* data, int const len,
                         Mhandle* mhandle, bool busypoll) {
  auto* poll_ctx = uccl_send_async(conn_id, data, len, mhandle);
  return busypoll ? uccl_poll(poll_ctx) : uccl_wait(poll_ctx);
}

bool Endpoint::uccl_recv(ConnID conn_id, void* data, int* len_p,
                         Mhandle* mhandle, bool busypoll) {
  auto* poll_ctx = uccl_recv_async(conn_id, data, len_p, mhandle);
  return busypoll ? uccl_poll(poll_ctx) : uccl_wait(poll_ctx);
}

bool Endpoint::uccl_recv_multi(ConnID conn_id, void** data, int* len_p,
                               Mhandle** mhandle, int n, bool busypoll) {
  auto* poll_ctx = uccl_recv_multi_async(conn_id, data, len_p, mhandle, n);
  return busypoll ? uccl_poll(poll_ctx) : uccl_wait(poll_ctx);
}

PollCtx* Endpoint::uccl_send_async(ConnID conn_id, void const* data,
                                   int const len, Mhandle* mhandle) {
  PollCtx* poll_ctx = ctx_pool_->pop();
  poll_ctx->num_unfinished = 1;
  poll_ctx->engine_idx = conn_id.engine_idx;

  Channel::Msg msg = {
      .opcode = Channel::Msg::Op::kTx,
      .len = len,
      .len_p = nullptr,
      .data = const_cast<void*>(data),
      .mhandle = mhandle,
      .flow_id = conn_id.flow_id,
      .deser_msgs = nullptr,
      .poll_ctx = poll_ctx,
  };
  poll_ctx->write_barrier();
  Channel::enqueue_mp(channel_vec_[conn_id.engine_idx]->tx_task_q_, &msg);
  return poll_ctx;
}

PollCtx* Endpoint::uccl_recv_async(ConnID conn_id, void* data, int* len_p,
                                   Mhandle* mhandle) {
  PollCtx* poll_ctx = ctx_pool_->pop();
  poll_ctx->num_unfinished = 1;
  poll_ctx->engine_idx = conn_id.engine_idx;

  Channel::Msg msg = {
      .opcode = Channel::Msg::Op::kRx,
      .len = 0,
      .len_p = len_p,
      .data = data,
      .mhandle = mhandle,
      .flow_id = conn_id.flow_id,
      .deser_msgs = nullptr,
      .poll_ctx = poll_ctx,
  };
  Channel::enqueue_mp(channel_vec_[conn_id.engine_idx]->rx_task_q_, &msg);
  return poll_ctx;
}
PollCtx* Endpoint::uccl_recv_scattered_async(ConnID conn_id, UcclRequest* req,
                                             Mhandle* mhandle) {
  PollCtx* poll_ctx = ctx_pool_->pop();
  poll_ctx->num_unfinished = 1;
  poll_ctx->engine_idx = conn_id.engine_idx;

  Channel::Msg msg = {
      .opcode = Channel::Msg::Op::kRxScattered,
      .len = 0,
      .len_p = &(req->recv_len[0]),
      .req = req,
      .mhandle = mhandle,
      .flow_id = conn_id.flow_id,
      .deser_msgs = nullptr,
      .poll_ctx = poll_ctx,
  };
  Channel::enqueue_mp(channel_vec_[conn_id.engine_idx]->rx_task_q_, &msg);
  return poll_ctx;
}

void Endpoint::uccl_recv_free_ptrs(ConnID conn_id, int iov_n,
                                   void** iov_addrs) {
  Channel::Msg msg = {
      .opcode = Channel::Msg::Op::kRxFreePtrs,
      .len = iov_n,
      .len_p = nullptr,
      .data = (void*)iov_addrs,
      .mhandle = nullptr,
      .flow_id = conn_id.flow_id,
      .deser_msgs = nullptr,
      .poll_ctx = nullptr,
  };
  Channel::enqueue_mp(channel_vec_[conn_id.engine_idx]->rx_task_q_, &msg);
}

PollCtx* Endpoint::uccl_recv_multi_async(ConnID conn_id, void** data,
                                         int* len_p, Mhandle** mhandle, int n) {
  PollCtx* poll_ctx = ctx_pool_->pop();
  poll_ctx->num_unfinished = n;
  poll_ctx->engine_idx = conn_id.engine_idx;

  Channel::Msg msg[kMaxMultiRecv];
  for (int i = 0; i < n; i++) {
    msg[i] = {
        .opcode = Channel::Msg::Op::kRx,
        .len = 0,
        .len_p = &(len_p[i]),
        .data = data[i],
        .mhandle = mhandle[i],
        .flow_id = conn_id.flow_id,
        .deser_msgs = nullptr,
        .poll_ctx = poll_ctx,
    };
  }
  Channel::enqueue_mp_multi(channel_vec_[conn_id.engine_idx]->rx_task_q_,
                            (void*)msg, n);
  return poll_ctx;
}

PollCtx* Endpoint::uccl_flush_async(ConnID conn_id, void** data, int* len_p,
                                    Mhandle** mhandle, int n) {
  PollCtx* poll_ctx = ctx_pool_->pop();
  poll_ctx->num_unfinished = n;
  poll_ctx->engine_idx = conn_id.engine_idx;
  // UD does not require flush.
  poll_ctx->done = true;
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
  while (!uccl_poll_once(ctx))
    ;
  return true;
}

bool Endpoint::uccl_poll_once(PollCtx* ctx) {
  if (!ctx->done.load()) return false;
  fence_and_clean_ctx(ctx);
  return true;
}

int Endpoint::uccl_regmr_dmabuf(int dev, void* addr, size_t len, int type,
                                int offset, int fd, struct Mhandle** mhandle) {
#ifdef LAZY_CREATE_ENGINE
  auto factory_dev = EFAFactory::GetEFADevice(dev);
#else
  auto factory_dev = EFAFactory::GetEFADevice(gpu_);
#endif
  *mhandle = new Mhandle();

  (*mhandle)->mr = ibv_reg_dmabuf_mr(
      factory_dev->pd, offset, len, (uint64_t)addr, fd, IBV_ACCESS_LOCAL_WRITE);

  return 0;
}

int Endpoint::uccl_regmr(int dev, void* addr, size_t len,
                         int type /*unsed for now*/, struct Mhandle** mhandle) {
#ifdef LAZY_CREATE_ENGINE
  auto factory_dev = EFAFactory::GetEFADevice(dev);
#else
  auto factory_dev = EFAFactory::GetEFADevice(gpu_);
#endif

  *mhandle = new Mhandle();
  (*mhandle)->mr =
      ibv_reg_mr(factory_dev->pd, addr, len, IBV_ACCESS_LOCAL_WRITE);

  return 0;
}

void Endpoint::uccl_deregmr(struct Mhandle* mhandle) {
  ibv_dereg_mr(mhandle->mr);
  delete mhandle;
}

void Endpoint::install_flow_on_engine(FlowID flow_id,
                                      std::string const& remote_ip,
                                      uint32_t local_engine_idx,
                                      int bootstrap_fd, bool is_sender) {
  int ret;

  // Sync remote engine index.
  uint32_t remote_engine_idx;
  ret = send_message(bootstrap_fd, &local_engine_idx, sizeof(uint32_t));
  ret = receive_message(bootstrap_fd, &remote_engine_idx, sizeof(uint32_t));
  DCHECK(ret == sizeof(uint32_t));

  // Exchange ConnMeta with the peer.
  auto* efa_socket = engine_id_to_engine_map_[local_engine_idx]->socket_;
  ConnMeta* remote_meta = new ConnMeta();
  ConnMeta* local_meta = new ConnMeta();
  efa_socket->get_conn_metadata(local_meta);

  // Only operating bootstrap_fd on a the creation thread, not on each
  // engine thread, as it will create read/write hang.
  std::stringstream str;
  str << "[Engine] local_meta->qpn_list: ";
  for (int i = 0; i < kMaxSrcDstQP; i++) {
    str << local_meta->qpn_list[i] << " ";
  }
  str << "\n [Engine] local_meta->qpn_list_ctrl: ";
  for (int i = 0; i < kMaxSrcDstQPCtrl; i++) {
    str << local_meta->qpn_list_ctrl[i] << " ";
  }

  if constexpr (kReceiverCCType == ReceiverCCType::kEQDS) {
    for (int i = 0; i < kMaxSrcDstQPCredit; i++) {
      local_meta->qpn_list_credit[i] =
          engine_id_to_engine_map_[local_engine_idx]->credit_qp_ctx_->get_qpn(
              i);
    }

    str << "\n [Engine] local_meta->qpn_list_credit: ";
    for (int i = 0; i < kMaxSrcDstQPCredit; i++) {
      str << local_meta->qpn_list_credit[i] << " ";
    }
  }
  LOG(INFO) << str.str();

  send_message(bootstrap_fd, local_meta, sizeof(ConnMeta));
  receive_message(bootstrap_fd, remote_meta, sizeof(ConnMeta));

  // Install flow and dst ports on engine.
  auto* poll_ctx = new PollCtx();
  poll_ctx->num_unfinished = 1;
  Channel::CtrlMsg ctrl_msg = {
      .opcode = Channel::CtrlMsg::Op::kInstallFlow,
      .flow_id = flow_id,
      .remote_ip = htonl(str_to_ip(remote_ip)),
      .remote_engine_idx = remote_engine_idx,
      .local_meta = local_meta,
      .remote_meta = remote_meta,
      .is_sender = is_sender,
      .poll_ctx = poll_ctx,
  };
  poll_ctx->write_barrier();
  Channel::enqueue_mp(channel_vec_[local_engine_idx]->ctrl_task_q_, &ctrl_msg);

  // Wait until the flow has been installed on the engine.
  {
    std::unique_lock<std::mutex> lock(poll_ctx->mu);
    poll_ctx->cv.wait(lock, [&poll_ctx] { return poll_ctx->done.load(); });
  }
  delete poll_ctx;

  // sync so to receive flow_id packets.
  net_barrier(bootstrap_fd);
}

inline int Endpoint::find_dedicated_engine_idx(int vdev_idx, bool is_sender) {
  auto si = vdev_idx * kNumEnginesPerVdev;
  auto ei = (vdev_idx + 1) * kNumEnginesPerVdev;

  auto minElementIter = engine_load_vec_.begin() + si;
  if (!is_sender) minElementIter = engine_load_vec_.begin() + ei - 1;

  *minElementIter += 1;
  return std::distance(engine_load_vec_.begin(), minElementIter);
}

inline int Endpoint::find_least_loaded_engine_idx_and_update(int vdev_idx,
                                                             FlowID flow_id,
                                                             bool is_sender) {
  std::lock_guard<std::mutex> lock(engine_load_vec_mu_);

  auto si = vdev_idx * kNumEnginesPerVdev;
  auto ei = (vdev_idx + 1) * kNumEnginesPerVdev;

  auto minElementIter = std::min_element(engine_load_vec_.begin() + si,
                                         engine_load_vec_.begin() + ei);

  *minElementIter += 1;
  return std::distance(engine_load_vec_.begin(), minElementIter);
}

inline void Endpoint::fence_and_clean_ctx(PollCtx* ctx) {
  // Make the data written by the engine thread visible to the app thread.
  ctx->read_barrier();
  ctx->clear();
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

    int cnt = 0;
    std::string s;
    s += "\n[Uccl Engine] ";
    for (auto& engine : engine_vec_) {
      s += engine->status_to_string(cnt >= 2);
      cnt++;
    }
    if (cnt < engine_vec_.size())
      s += Format("\n\t... %d more engines", engine_vec_.size() - cnt);
    LOG(INFO) << s;
  }
}

}  // namespace uccl