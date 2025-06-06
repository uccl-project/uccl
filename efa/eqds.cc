#include "eqds.h"
#include "transport_config.h"
#include "util_list.h"
#include <glog/logging.h>
#include <infiniband/verbs.h>

namespace uccl {

namespace eqds {

// Make progress on the pacer.
void EQDS::run_pacer(void) {
  auto now = rdtsc();
  handle_pull_request();

  // It is our responsibility to poll Tx completion events.
  handle_poll_cq();

  if (now - last_pacing_tsc_ >= pacing_interval_tsc_) {
    handle_grant_credit();
    last_pacing_tsc_ = now;
  }
}

// Handle registration requests.
void EQDS::handle_pull_request(void) {
  EQDSChannel::Msg msg;
  int budget = 0;
  EQDSCC* sink;

  while (jring_sc_dequeue_bulk(channel_.cmdq_, &msg, 1, nullptr) == 1) {
    switch (msg.opcode) {
      case EQDSChannel::Msg::kRequestPull:
        sink = msg.eqds_cc;
        if (!sink->in_active_list()) {
          if (sink->in_idle_list()) {
            // Remove it from the idle list.
            sink->remove_from_idle_list();
          }
          // Add it to the active list.
          sink->add_to_active_list(&active_senders_);
          VLOG(5) << "Registered in pacer pull queue.";
        } else {
          // Already in the active list. Do nothing.
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        break;
      default:
        LOG(ERROR) << "Unknown opcode: " << msg.opcode;
        break;
    }
    if (++budget >= 16) break;
  }
}

// Handle Credit CQ TX events.
void EQDS::handle_poll_cq(void) {
  for (int i = 0; i < kNumEnginesPerVdev * 2; i++)
    credit_qp_ctx_[i]->pacer_poll_credit_cq();
}

bool EQDS::send_pull_packet(EQDSCC* eqds_cc) {
  return eqds_cc->send_pullpacket(eqds_cc->get_latest_pull());
}

// Grant credit to the sender of this flow.
bool EQDS::grant_credit(EQDSCC* eqds_cc, bool idle, uint32_t* total_inc) {
  uint32_t increment;

  if (!idle)
    increment = std::min(kCreditPerPull, eqds_cc->backlog());
  else
    increment = kCreditPerPull;

  if (!increment) return true;

  *total_inc += increment;

  eqds_cc->inc_lastest_pull(increment);

  if (!send_pull_packet(eqds_cc)) {
    eqds_cc->dec_latest_pull(increment);
    VLOG(5) << "Failed to send pull packet.";
  }

  return eqds_cc->backlog() == 0;
}

// Grant credits to senders.
void EQDS::handle_grant_credit(void) {
  struct list_head *pos, *n;
  uint32_t total_inc = 0;
  uint32_t budget = 0;

  if (!list_empty(&active_senders_)) {
    while (!list_empty(&active_senders_) && budget < kSendersPerPull) {
      list_for_each_safe(pos, n, &active_senders_) {
        auto item = list_entry(pos, struct active_item, active_link);
        auto* sink = item->eqds_cc;
        list_del(pos);

        if (grant_credit(sink, false, &total_inc)) {
          // Grant done, add it to idle sender list.
          DCHECK(!sink->in_idle_list());
          sink->add_to_idle_list(&idle_senders_);
        } else {
          // We have not satisfied its demand, re-add it to the active
          // sender list.
          sink->add_to_active_list(&active_senders_);
        }

        if (++budget >= kSendersPerPull) {
          if (total_inc < kCreditPerPull * kSendersPerPull) {
            // We have unutilized credits, try to grant more.
            continue;
          }
          break;
        }
      }
    }
  } else {
    // No active sender.
    list_for_each_safe(pos, n, &idle_senders_) {
      auto item = list_entry(pos, struct idle_item, idle_link);
      auto* sink = item->eqds_cc;
      list_del(pos);

      if (grant_credit(sink, true, &total_inc) && !sink->idle_credit_enough()) {
        // Grant done but we can still grant more credit for this
        // sender.
        sink->add_to_idle_list(&idle_senders_);
      }

      break;
    }
  }
}

void CreditQPContext::__post_recv_wrs_for_credit(int nb, uint32_t qpidx) {
  post_rq_cnt_[qpidx] += nb;

  while (post_rq_cnt_[qpidx] >= kMaxRecvWrDeficit) {
    // Post kMaxChainedWr at once.
    int nr_post = std::min(post_rq_cnt_[qpidx], kMaxChainedWr);
    uint64_t pkt_hdr_buf, frame_desc_buf;

    for (int i = 0; i < nr_post; i++) {
      pkt_hdr_buf = engine_pop_pkt_hdr();
      frame_desc_buf = engine_pop_frame_desc();

      auto* frame_desc =
          FrameDesc::Create(frame_desc_buf, pkt_hdr_buf,
                            EFA_UD_ADDITION + kUcclPullHdrLen, 0, 0, 0, 0);

      frame_desc->set_src_qp_idx(qpidx);

      rq_sges_[i].addr = pkt_hdr_buf;
      rq_sges_[i].length = EFA_UD_ADDITION + kUcclPullHdrLen;
      rq_sges_[i].lkey = engine_hdr_pool_->get_lkey();

      rq_wrs_[i].wr_id = (uint64_t)frame_desc;
    }

    rq_wrs_[nr_post - 1].next = nullptr;

    struct ibv_recv_wr* bad_wr;
    DCHECK(ibv_post_recv(credit_qp_list_[qpidx], rq_wrs_, &bad_wr) == 0)
        << qpidx;

    rq_wrs_[nr_post - 1].next =
        (nr_post == kMaxChainedWr) ? nullptr : &rq_wrs_[nr_post];

    post_rq_cnt_[qpidx] -= nr_post;
  }
}

std::vector<FrameDesc*> CreditQPContext::engine_poll_credit_cq(void) {
  std::vector<FrameDesc*> frames;
  struct ibv_wc wcs[kMaxPollBatch];
  int nr_cqe = ibv_poll_cq(engine_credit_cq_, kMaxPollBatch, wcs);

  for (int i = 0; i < nr_cqe; i++) {
    struct ibv_wc* wc = &wcs[i];
    DCHECK(wc->status == IBV_WC_SUCCESS) << wc->status;
    DCHECK(wc->opcode == IBV_WC_RECV);

    auto* frame = (FrameDesc*)wc->wr_id;

    frames.push_back(frame);

    __post_recv_wrs_for_credit(1, frame->get_src_qp_idx());
  }

  return frames;
}

int CreditQPContext::pacer_poll_credit_cq(void) {
  struct ibv_wc wcs[kMaxPollBatch];
  int nr_cqe = ibv_poll_cq(pacer_credit_cq_, kMaxPollBatch, wcs);

  for (int i = 0; i < nr_cqe; i++) {
    struct ibv_wc* wc = &wcs[i];
    DCHECK(wc->status == IBV_WC_SUCCESS) << wc->status;
    DCHECK(wc->opcode == IBV_WC_SEND);

    auto* frame = (FrameDesc*)wc->wr_id;

    pacer_push_pkt_hdr(frame->get_pkt_hdr_addr());
    pacer_push_frame_desc((uint64_t)frame);
  }

  return nr_cqe;
}

PullQuanta EQDSCC::compute_pull_target() {
  uint32_t pull_target_bytes = backlog_bytes_;

  if (pull_target_bytes > kEQDSMaxCwnd) pull_target_bytes = kEQDSMaxCwnd;

  if (pull_target_bytes > credit_pull_ + credit_spec_)
    pull_target_bytes -= (credit_pull_ + credit_spec_);
  else
    pull_target_bytes = 0;

  pull_target_bytes += unquantize(pull_);

  uint32_t old_pull_target_bytes = unquantize(last_sent_pull_target_);

  if (!in_speculating_ && credit_spec_ > 0 &&
      pull_target_bytes - old_pull_target_bytes < PULL_QUANTUM / 2) {
    if (credit_spec_ > PULL_QUANTUM)
      credit_spec_ -= PULL_QUANTUM;
    else
      credit_spec_ = 0;
    pull_target_bytes += PULL_QUANTUM;
  }

  last_sent_pull_target_ = quantize_ceil(pull_target_bytes);

  // if (pull_ > 1040000) {
  //     LOG_EVERY_N(INFO, 1) << pull_ << ", last_sent_pull_target_: " <<
  //     last_sent_pull_target_;
  // }

  return last_sent_pull_target_;
}

};  // namespace eqds
};  // namespace uccl