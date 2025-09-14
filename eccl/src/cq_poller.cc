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

      if (wc.status != IBV_WC_SUCCESS) {
        std::cerr << "[CQPoller] wc error and cannot map req: status="
                  << wc.status << " opcode=" << wc.opcode
                  << " wr_id=" << wc.wr_id << " imm=" << wc.imm_data
                  << " vendor_err=" << wc.vendor_err << "\n";
        continue;
      }

      switch (wc.opcode) {
        case IBV_WR_RDMA_WRITE_WITH_IMM: {
          //  std::cout << "wc id is " << wc.wr_id << std::endl;
          unsigned req_id = static_cast<uint32_t>(wc.wr_id);
          std::shared_ptr<Request> req;

          {
            std::lock_guard<std::mutex> lk(comm_->req_mu_);
            auto it = comm_->requests_map_.find(req_id);
            if (it != comm_->requests_map_.end()) req = it->second;
          }

          if (!req) {
            std::cerr << "[CQPoller] send success wc but unknown req_id="
                      << req_id << " opcode=" << wc.opcode << "\n";
            break;
          }
          req->on_comm_done(true);
          break;
        }

        case IBV_WC_RECV_RDMA_WITH_IMM: {
          unsigned req_id = ntohl(static_cast<uint32_t>(wc.imm_data));
          std::shared_ptr<Request> req;

          {
            std::lock_guard<std::mutex> lk(comm_->req_mu_);
            auto it = comm_->requests_map_.find(req_id);
            if (it != comm_->requests_map_.end()) req = it->second;
          }

          if (!req) {
            std::cerr << "[CQPoller] recv success wc but unknown req_id="
                      << req_id << " opcode=" << wc.opcode
                      << " add it to pending queue first" << std::endl;
            while (jring_mp_enqueue_bulk(comm_->pending_req_id_to_deal_,
                                         &req_id, 1, nullptr) != 1) {
            }
            continue;
          }
          req->on_comm_done(true);
          break;
        }

        default:
          std::cerr << "[CQPoller] unknown wc wr_id=" << wc.wr_id
                    << " opcode=" << wc.opcode << "\n";
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
