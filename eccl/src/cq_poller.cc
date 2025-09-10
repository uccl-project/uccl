#include "cq_poller.h"
#include "transport.h"
#include <arpa/inet.h>  // ntohl
#include <cstring>
#include <iostream>

static constexpr int kMaxPollEntriesDefault = 16;

CQPoller::CQPoller(Communicator* comm, ibv_cq* assign_cq, int poll_batch)
    : comm_(comm),
      cq_(assign_cq),
      poll_batch_(poll_batch > 0 ? poll_batch : kMaxPollEntriesDefault) {}

CQPoller::~CQPoller() { stop(); }

void CQPoller::start() {
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true))
    return;  // already started
  thr_ = std::thread(&CQPoller::run_loop, this);
  std::cout << "Communicator " << comm_->local_rank_ << " CQPoller with cq "
            << cq_ << " started!" << std::endl;
}

void CQPoller::stop() {
  bool expected = true;
  if (!running_.compare_exchange_strong(expected, false))
    return;  // already stopped or not started
  if (thr_.joinable()) thr_.join();
  std::cout << "Communicator " << comm_->local_rank_ << " CQPoller with cq "
            << cq_ << " Closed!" << std::endl;
}

void CQPoller::run_loop() {
  const std::chrono::microseconds kIdleSleepMin(1);    // 1us
  const std::chrono::microseconds kIdleSleepMax(200);  // 200us
  std::chrono::microseconds idle_sleep = kIdleSleepMin;

  std::vector<ibv_wc> wcs;
  wcs.resize(poll_batch_);

  while (running_.load(std::memory_order_acquire)) {
    bool any_work = false;

    if (!cq_) continue;

    int ne = ibv_poll_cq(cq_, poll_batch_, wcs.data());
    if (ne < 0) {
      // CQ poll error
      std::cerr << "[CQPoller] ibv_poll_cq error: " << ne << " (cq=" << cq_
                << ")\n";
      continue;
    }
    if (ne == 0) continue;  // no completions on this CQ

    any_work = true;
    idle_sleep = kIdleSleepMin;  // got work, reset backoff

    for (int i = 0; i < ne; ++i) {
      ibv_wc& wc = wcs[i];

      unsigned req_id = static_cast<unsigned>(wc.wr_id);
      std::shared_ptr<Request> req;
      {
        std::lock_guard<std::mutex> lk(comm_->req_mu_);
        auto it = comm_->requests_map_.find(req_id);
        if (it != comm_->requests_map_.end()) req = it->second;
      }

      if (wc.status != IBV_WC_SUCCESS) {
        if (req) {
          req->on_comm_done(false);
        }
        std::cerr << "[CQPoller] wc error and cannot map req: status="
                  << wc.status << " opcode=" << wc.opcode
                  << " wr_id=" << wc.wr_id << " imm=" << wc.imm_data
                  << " vendor_err=" << wc.vendor_err << "\n";
        continue;
      }

      if (!req) {
        std::cerr << "[CQPoller] success wc but unknown req_id=" << req_id
                  << " opcode=" << wc.opcode << "\n";
        continue;
      }

      switch (wc.opcode) {
        case IBV_WC_SEND:
          // 本端 send 的 signaled 完成
          req->on_comm_done(true);
          break;

        case IBV_WC_RECV_RDMA_WITH_IMM:
          // remote 发来的 IMM -> data 到达，通知 request
          req->on_comm_done(true);
          // 如果你维护 per-q/pool，需要在这里补贴 recv pool：
          // comm_->replenish_recv_for_qp(cq_idx, 1);
          break;

        case IBV_WC_RECV:
          // 带 payload 的 recv 完成（视你协议可当作事件）
          req->on_comm_done(true);
          // comm_->replenish_recv_for_qp(cq_idx, 1);
          break;

        default:
          // 其它完成（RDMA_WRITE 本地完成、RDMA_READ 完成等）
          req->on_comm_done(true);
          break;
      }
    }  // for each wc

    if (!any_work) {
      // Backoff when idle to avoid busy spin
      std::this_thread::sleep_for(idle_sleep);
      // exponential backoff (capped)
      idle_sleep = std::min(kIdleSleepMax, idle_sleep * 2);
    } else {
      // quick yield to let other threads run; ensure not starving
      std::this_thread::yield();
    }
  }  // for each cq

  // thread exiting: optional cleanup/log
  std::cerr << "[CQPoller] exiting poll loop\n";
}
