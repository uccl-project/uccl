#include "eqds.h"
#include "rdma_io.h"
#include "transport_config.h"
#include "util/list.h"
#include "util_rdma.h"
#include <glog/logging.h>
#include <infiniband/verbs.h>

namespace uccl {

namespace eqds {

PullQuanta EQDSCC::compute_pull_target(void* context, uint32_t chunk_size) {
  SubUcclFlow* subflow = reinterpret_cast<SubUcclFlow*>(context);

  uint32_t pull_target_bytes = subflow->backlog_bytes_;

  if constexpr (kSenderCCA != SENDER_CCA_NONE) {
    if (pull_target_bytes > subflow->pcb.timely_cc.get_wnd() + chunk_size) {
      pull_target_bytes = subflow->pcb.timely_cc.get_wnd() + chunk_size;
    }
  }

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

  return last_sent_pull_target_;
}

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

void EQDS::handle_grant_credit() {
  struct list_head *pos, *n;
  uint32_t budget = 0;
  uint16_t total_consumed;
  PullQuanta consumed;

  if (!list_empty(&active_senders_)) {
    while (!list_empty(&active_senders_) &&
           (budget < kSendersPerPull ||
            total_consumed < kCreditPerPull * kSendersPerPull)) {
      list_for_each_safe(pos, n, &active_senders_) {
        auto item = list_entry(pos, struct active_item, active_link);
        auto* sink = item->eqds_cc;
        list_del(pos);
        if (grant_credit(sink, false, &consumed)) {
          // Grant done, add it to idle sender list.
          DCHECK(list_empty(&sink->idle_item.idle_link));
          list_add_tail(&sink->idle_item.idle_link, &idle_senders_);
        } else {
          // We have not satisfied its demand, re-add it to the active
          // sender list.
          list_add_tail(pos, &active_senders_);
        }

        total_consumed += consumed;

        if (total_consumed >= kCreditPerPull * kSendersPerPull)
          break;
        else
          continue;

        if (++budget >= kSendersPerPull) {
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

      if (grant_credit(sink, true, &consumed) && !sink->idle_credit_enough()) {
        // Grant done but we can still grant more credit for this
        // sender.
        list_add_tail(&sink->idle_item.idle_link, &idle_senders_);
      }
    }
  }
}

void EQDS::handle_poll_cq(void) {
  struct list_head *pos, *n;
  list_for_each_safe(pos, n, &poll_cq_list_) {
    auto item = list_entry(pos, struct pacer_credit_cq_item, poll_link);
    auto pc_qpw = item->pc_qpw;
    if (poll_cq(pc_qpw)) {
      // Remove it from the poll list since is has no pending completion
      // event.
      list_del(pos);
    }
  }
}

bool EQDS::poll_cq(struct PacerCreditQPWrapper* pc_qpw) {
  if (!pc_qpw->poll_cq_cnt_) return true;
  auto cq_ex = pc_qpw->pacer_credit_cq_;
  int cq_budget = 0;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return false;

  while (1) {
    if (cq_ex->status == IBV_WC_SUCCESS) {
      auto chunk_addr = cq_ex->wr_id;
      pc_qpw->pacer_credit_chunk_pool_->free_buff(chunk_addr);
    } else {
      LOG(ERROR) << "pacer credit CQ state error: " << cq_ex->status;
    }

    pc_qpw->poll_cq_cnt_--;

    if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
  }
  ibv_end_poll(cq_ex);

  return pc_qpw->poll_cq_cnt_ == 0;
}

void EQDS::handle_pull_request(void) {
  EQDSChannel::Msg msg;
  int budget = 0;
  EQDSCC* sink;

  while (jring_sc_dequeue_bulk(channel_.cmdq_, &msg, 1, nullptr) == 1) {
    switch (msg.opcode) {
      case EQDSChannel::Msg::kRequestPull:
        sink = msg.eqds_cc;
        if (list_empty(&sink->active_item.active_link)) {
          if (!list_empty(&sink->idle_item.idle_link)) {
            // Remove it from the idle list.
            list_del(&sink->idle_item.idle_link);
          }
          // Add it to the active list.
          list_add_tail(&sink->active_item.active_link, &active_senders_);
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

bool EQDS::send_pull_packet(EQDSCC* eqds_cc) {
  uint64_t chunk_addr;
  auto* pc_qpw = eqds_cc->pc_qpw_;

  if (pc_qpw->pacer_credit_chunk_pool_->alloc_buff(&chunk_addr)) return false;

  auto* pullhdr = reinterpret_cast<struct UcclPullHdr*>(chunk_addr);
  pullhdr->fid = be16_t(eqds_cc->fid_);
  pullhdr->pullno = be16_t(eqds_cc->latest_pull_);

  struct ibv_sge sge = {
      .addr = chunk_addr,
      .length = CreditChunkBuffPool::kPktSize,
      .lkey = pc_qpw->pacer_credit_chunk_pool_->get_lkey(),
  };

  struct ibv_send_wr wr, *bad_wr;
  wr.num_sge = 1;
  wr.sg_list = &sge;

  wr.wr_id = chunk_addr;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  wr.next = nullptr;

  DCHECK(ibv_post_send(pc_qpw->credit_qp_, &wr, &bad_wr) == 0);

  pc_qpw->poll_cq_cnt_++;

  if (list_empty(&pc_qpw->poll_item.poll_link)) {
    // Add to the poll list.
    list_add_tail(&pc_qpw->poll_item.poll_link, &poll_cq_list_);
  }

  return true;
}

// Grant credit to the sender of this flow.
bool EQDS::grant_credit(EQDSCC* eqds_cc, bool idle, PullQuanta* ret_increment) {
  PullQuanta increment;

  if (!idle)
    increment = std::min(kCreditPerPull, eqds_cc->backlog());
  else
    increment = kCreditPerPull;

  eqds_cc->latest_pull_ += increment;

  if (!send_pull_packet(eqds_cc)) {
    eqds_cc->latest_pull_ -= increment;
    VLOG(5) << "Failed to send pull packet.";
  }

  *ret_increment = increment;

  return eqds_cc->backlog() == 0;
}

// For original EQDS, it stalls the pacer when ECN ratio reaches a threshold
// (i.e., 10%). Here we use resort to RTT-based stall.
void EQDS::update_cc_state(void) {}

}  // namespace eqds

};  // namespace uccl