#include "transport.h"
#include "rdma_io.h"
#include "transport_config.h"
#include "util/list.h"
#include "util/net.h"
#include "util/util.h"
#include "util_rdma.h"
#include "util_timer.h"
#include <infiniband/verbs.h>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <endian.h>

namespace uccl {

void UcclFlow::poll_flow_cq(void) {
  if (!flow_cq_cnt_) return;

  auto comm_base = &recv_comm_.base;
  auto cq = comm_base->flow_cq;
  struct ibv_wc wcs[kMaxBatchCQ];

  int nb_cqe = ibv_poll_cq(cq, kMaxBatchCQ, wcs);
  for (auto i = 0; i < nb_cqe; i++) {
    auto opcode = wcs[i].opcode;
    if (opcode == IBV_WC_RDMA_WRITE) {
      // RC send completion.
      if constexpr (kRCSize > 0) {
        if (wcs[i].qp_num == comm_base->rc_qp->qp_num) {
          auto* rc_or_flush_done = (uint64_t*)wcs[i].wr_id;
          *rc_or_flush_done = true;
        }
      }
    } else if (opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      // RC recv completion.
      auto ureq = (struct ucclRequest*)wcs[i].wr_id;
      ureq->recv.data_len[0] = ntohl(wcs[i].imm_data);
      ureq->rc_or_flush_done = true;
      if (ureq->recv.data_len[0] <= NCCL_MIN_POST_RECV) {
        auto* flow = (UcclFlow*)ureq->context;
        flow->set_last_rc_size(ureq->recv.data_len[0]);
      }
    } else if (opcode == IBV_WC_RDMA_READ) {
      // GPU flush completion.
      auto* rc_or_flush_done = (uint64_t*)wcs[i].wr_id;
      *rc_or_flush_done = true;
    }
  }
  flow_cq_cnt_ -= nb_cqe;
}

void UcclFlow::post_flush(struct Mhandle** mhandles, void** data, int* size,
                          int n, uint64_t* flush_done, int last) {
  struct ibv_send_wr wr = {};
  wr.wr_id = (uint64_t)flush_done;
  wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(data[last]);
  wr.wr.rdma.rkey = mhandles[last]->mr->rkey;
  wr.sg_list = &recv_comm_.gpu_flush_sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;

  struct ibv_send_wr* bad_wr;
  DCHECK(ibv_post_send(recv_comm_.gpu_flush_qp, &wr, &bad_wr) == 0);

  flow_cq_cnt_++;

  UCCL_LOG_EP << "Post flush: addr: " << wr.wr.rdma.remote_addr
              << ", rkey: " << wr.wr.rdma.rkey;
}

void UcclFlow::rc_recv(void* data, int size, struct Mhandle* mhandle,
                       struct ibv_send_wr* wr, struct ibv_sge* sge,
                       struct ucclRequest* ureq) {
  auto* comm_base = &recv_comm_.base;
  struct RemFifo* rem_fifo = comm_base->fifo;
  int slot = rem_fifo->fifo_tail % kMaxReq;
  auto elems = rem_fifo->elems[slot];
  auto qp = comm_base->fifo_qp;

  elems[0].addr = reinterpret_cast<uint64_t>(data);
  elems[0].rkey = mhandle->mr->rkey;
  elems[0].nmsgs = 1;
  // For sender to check if the receiver is ready.
  elems[0].idx = rem_fifo->fifo_tail + 1;
  elems[0].size = size;
  // For sender to know we are using RC.
  elems[0].engine_offset = RDMAEndpoint::RC_MAGIC;

  UCCL_LOG_EP << "rc_recv: posted recv addr: " << elems[0].addr
              << ", rkey: " << elems[0].rkey << ", size: " << elems[0].size;

  memset(wr, 0, sizeof(*wr));
  // Figure out the remote address to write.
  wr->wr.rdma.remote_addr =
      comm_base->remote_fifo_addr + slot * kMaxRecv * sizeof(struct FifoItem);
  wr->wr.rdma.rkey = comm_base->remote_fifo_rkey;

  sge->lkey = comm_base->fifo_mr->lkey;
  sge->addr = (uint64_t)elems;
  sge->length = 1 * sizeof(struct FifoItem);

  wr->sg_list = sge;
  wr->num_sge = 1;

  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->send_flags = IBV_SEND_INLINE;

  // Occasionally post a request with the IBV_SEND_SIGNALED flag.
  if (slot == 0) {
    wr->send_flags |= IBV_SEND_SIGNALED;
    flow_cq_cnt_++;
  }

  struct ibv_send_wr* bad_wr;
  DCHECK(ibv_post_send(qp, wr, &bad_wr) == 0);

  // Post a recv buffer for consuming immedate data.
  struct ibv_recv_wr recv_wr = {};
  recv_wr.wr_id = (uint64_t)ureq;
  recv_wr.sg_list = nullptr;
  recv_wr.num_sge = 0;
  recv_wr.next = nullptr;
  struct ibv_recv_wr* bad_recv_wr;
  DCHECK(ibv_post_recv(comm_base->rc_qp, &recv_wr, &bad_recv_wr) == 0);
  flow_cq_cnt_++;

  UCCL_LOG_EP << "rc_recv: supplies buffer at recv slot: " << slot;

  rem_fifo->fifo_tail++;
}

struct FifoItem* UcclFlow::post_fifo(uint32_t engine_idx, void** data,
                                     int* size, int n, struct Mhandle** mhandle,
                                     struct ibv_send_wr* wr,
                                     struct ibv_sge* sge) {
  auto* comm_base = &recv_comm_.base;
  memset(wr, 0, sizeof(*wr));
  struct RemFifo* rem_fifo = comm_base->fifo;
  int slot = rem_fifo->fifo_tail % kMaxReq;
  auto elems = rem_fifo->elems[slot];

  for (int i = 0; i < n; i++) {
    elems[i].addr = reinterpret_cast<uint64_t>(data[i]);
    elems[i].rkey = mhandle[i]->mr->rkey;
    elems[i].nmsgs = n;
    // For sender to check if the receiver is ready.
    elems[i].idx = rem_fifo->fifo_tail + 1;
    elems[i].size = size[i];
    // For sender to decide the engine.
    elems[i].engine_offset = engine_idx % ep_->num_engines_per_dev_;

    // elems[i].rid is filled by engine. See supply_rx_buff.

    UCCL_LOG_EP << "recv_async: posted recv addr: " << elems[i].addr
                << ", rkey: " << elems[i].rkey << ", size: " << elems[i].size;
  }

  // Figure out the remote address to write.
  wr->wr.rdma.remote_addr =
      comm_base->remote_fifo_addr + slot * kMaxRecv * sizeof(struct FifoItem);
  wr->wr.rdma.rkey = comm_base->remote_fifo_rkey;

  sge->lkey = comm_base->fifo_mr->lkey;
  sge->addr = (uint64_t)elems;
  sge->length = n * sizeof(struct FifoItem);

  wr->sg_list = sge;
  wr->num_sge = 1;

  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->send_flags = IBV_SEND_INLINE;

  // Occasionally post a request with the IBV_SEND_SIGNALED flag.
  if (slot == 0) {
    wr->send_flags |= IBV_SEND_SIGNALED;
    flow_cq_cnt_++;
  }

  UCCL_LOG_EP << "recv_async: provided buffer at recv slot: " << slot;

  rem_fifo->fifo_tail++;

  return elems;
}

void UcclRDMAEngine::rc_handle_completion(void) {
  int work = 0;
  for (auto& it : rdma_ctx_map_) {
    // Update ratio and offset
    it.second->update_clock(nic_ts_ratio_, nic_ts_offset_);

    if constexpr (kReceiverCCA == RECEIVER_CCA_EQDS) {
      work += it.second->poll_credit_cq();
      it.second->check_credit_rq(!work);
    }
  }

  io_ctx_.rc_poll_send_cq();

  io_ctx_.rc_poll_recv_cq();

  io_ctx_.check_srq(false);
}

void UcclRDMAEngine::uc_handle_completion(void) {
  int work = 0;
  // First, poll the CQ for Ctrl QPs and Credit QPs.

  io_ctx_.poll_ctrl_cq();

  for (auto& it : rdma_ctx_map_) {
    // Update ratio and offset
    it.second->update_clock(nic_ts_ratio_, nic_ts_offset_);

    if constexpr (kReceiverCCA == RECEIVER_CCA_EQDS)
      work += it.second->poll_credit_cq();
  }

  io_ctx_.uc_poll_send_cq();
  io_ctx_.uc_poll_recv_cq();

  for (auto& it : rdma_ctx_map_) {
    if constexpr (kReceiverCCA == RECEIVER_CCA_EQDS)
      it.second->check_credit_rq(!work);
  }

  io_ctx_.check_ctrl_rq(false);

  io_ctx_.check_srq(false);
}

void UcclRDMAEngine::handle_rx_work(void) {
  Channel::Msg rx_work;
  int budget = kMaxRxWork;

  while (!pending_rx_works_.empty() && budget--) {
    // Process pending rx works.
    auto it = pending_rx_works_.front();
    auto rdma_ctx = it.first;
    auto ureq = it.second;

    UCCL_LOG_ENGINE << "Process rx work.";
    if (rdma_ctx->supply_rx_buff(ureq) == 0) {
      pending_rx_works_.pop_front();
    } else {
      UCCL_LOG_ENGINE << "Too many inflight recv requests.";
      return;
    }
  }

  if (budget < 0) return;

  while (budget--) {
    if (jring_sc_dequeue_bulk(channel_->rx_cmdq_, &rx_work, 1, nullptr) == 0)
      break;
    // Make data written by the app thread visible to the engine.
    std::ignore = std::atomic_load_explicit(&rx_work.poll_ctx->fence,
                                            std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);

    auto peer_id = rx_work.peer_id;
    auto it = rdma_ctx_map_.find(peer_id);
    DCHECK(it != rdma_ctx_map_.end());
    auto rdma_ctx = it->second;

    UCCL_LOG_ENGINE << "Process rx work.";
    if (rdma_ctx->supply_rx_buff(rx_work.ureq)) {
      pending_rx_works_.push_back(std::make_pair(rdma_ctx, rx_work.ureq));
    }
  }
}

inline void RDMAEndpoint::initialize_resources(int total_num_engines) {
  // Initialize all dynamic arrays
  if (num_devices_ <= 0) {
    printf("num_devices_ is 0.");
    exit(EXIT_FAILURE);
  }

  channel_vec_.resize(total_num_engines);
  engine_vec_.resize(total_num_engines);
  engine_load_vec_.resize(total_num_engines);
  for (int i = 0; i < total_num_engines; i++) {
    engine_load_vec_[i] = std::make_unique<std::atomic<uint32_t>>(0);
  }
  eqds_.resize(num_devices_);
  p2p_listen_ports_.resize(num_devices_);
  std::fill(p2p_listen_ports_.begin(), p2p_listen_ports_.end(), 0);
  p2p_listen_fds_.resize(num_devices_);
  std::fill(p2p_listen_fds_.begin(), p2p_listen_fds_.end(), -1);

  peer_map_.resize(num_devices_);
  peer_map_mu_.resize(num_devices_);

  for (int i = 0; i < num_devices_; i++) {
    peer_map_mu_[i] = std::make_unique<std::mutex>();
  }

  flow_id_spin_.resize(num_devices_);
  next_flow_id_.resize(num_devices_);
  for (int i = 0; i < num_devices_; i++) {
    flow_id_spin_[i].resize(MAX_PEER);
    next_flow_id_[i].resize(MAX_PEER);
  }

  active_flows_vec_.resize(num_devices_);
  active_flows_spin_.resize(num_devices_);

  UCCL_LOG_EP << "Initialized " << total_num_engines << " engines for "
              << num_devices_ << " devices totally, with "
              << num_engines_per_dev_ << " engines per device";
}

void RDMAEndpoint::cleanup_resources() {
  for (auto& flows : active_flows_vec_) {
    for (auto* flow : flows) {
      if (flow) {
        delete flow;
      }
    }
    flows.clear();
  }

  active_flows_vec_.clear();
  active_flows_spin_.clear();

  for (auto* channel : channel_vec_) {
    if (channel) {
      delete channel;
    }
  }
  channel_vec_.clear();
  engine_load_vec_.clear();
  engine_vec_.clear();
  engine_id_to_engine_map_.clear();

  for (auto* eqds_ptr : eqds_) {
    if (eqds_ptr) {
      delete eqds_ptr;
    }
  }
  eqds_.clear();

  for (int fd : p2p_listen_fds_) {
    if (fd >= 0) {
      close(fd);
    }
  }
  p2p_listen_fds_.clear();
  p2p_listen_ports_.clear();

  peer_map_.clear();
  peer_map_mu_.clear();

  for (auto& spins : flow_id_spin_) {
    spins.clear();
  }
  flow_id_spin_.clear();
  for (auto& flow_ids : next_flow_id_) {
    flow_ids.clear();
  }
  next_flow_id_.clear();

  for (auto& boostrap_fd : fd_vec_) {
    close(boostrap_fd);
  }
  fd_vec_.clear();

  if (ctx_pool_) {
    delete ctx_pool_;
    ctx_pool_ = nullptr;
  }
  if (ctx_pool_buf_) {
    delete[] ctx_pool_buf_;
    ctx_pool_buf_ = nullptr;
  }
}

void UcclRDMAEngine::handle_tx_work(void) {
  Channel::Msg tx_work;
  int budget;
  uint32_t bytes = 0;

  budget = pending_tx_works_.size();
  while (!pending_tx_works_.empty() && budget--) {
    auto it = pending_tx_works_.front();
    pending_tx_works_.pop_front();
    auto rdma_ctx = it.first;
    auto ureq = it.second;
    if (!ureq) {
      UCCL_LOG_ERROR << "Null ureq in tx work\n";
      continue;
    }
    UCCL_LOG_ENGINE << "Process tx work.";
    if (!rdma_ctx->tx_message(ureq)) {
      pending_tx_works_.push_back(std::make_pair(rdma_ctx, ureq));
    }
  }

  budget = kMaxTxWork;
  while (budget--) {
    if (jring_sc_dequeue_bulk(channel_->tx_cmdq_, &tx_work, 1, nullptr) == 0)
      break;
    // Make data written by the app thread visible to the engine.
    std::ignore = std::atomic_load_explicit(&tx_work.poll_ctx->fence,
                                            std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);

    auto peer_id = tx_work.peer_id;
    auto it = rdma_ctx_map_.find(peer_id);
    DCHECK(it != rdma_ctx_map_.end());
    auto rdma_ctx = it->second;

    UCCL_LOG_ENGINE << "Process tx work.";
    if (!rdma_ctx->tx_message(tx_work.ureq)) {
      // Push the message to the pending transmit queue.
      pending_tx_works_.push_back(std::make_pair(rdma_ctx, tx_work.ureq));
    }

    bytes += tx_work.ureq->send.data_len;

    if (bytes >= kMaxTxBytesThres) break;
  }
}

void UcclRDMAEngine::handle_timing_wheel(void) {
  if (io_ctx_.bypass_pacing()) return;
  for (auto& it : rdma_ctx_map_) {
    it.second->burst_timing_wheel();
  }
}

void UcclRDMAEngine::run() {
  while (!shutdown_) {
    // Calculate the cycles elapsed since last periodic processing.
    auto now_tsc = rdtsc();
    auto const elapsed_tsc = now_tsc - last_periodic_tsc_;

    if (elapsed_tsc >= kSlowTimerIntervalTsc_) {
      // Perform periodic processing.
      periodic_process();
      last_periodic_tsc_ = now_tsc;
    }

    handle_clock_synchronization();

    handle_rx_work();

    handle_tx_work();

    handle_timing_wheel();

    handle_completion();
  }
  UCCL_LOG_ENGINE << "Engine " << engine_idx_ << " shutdown";
}

/**
 * @brief Method to perform periodic processing. This is called by the
 * main engine cycle (see method `Run`).
 */
void UcclRDMAEngine::periodic_process() {
  // Handle RTOs for all UC QPs.
  if (!io_ctx_.is_rc_mode()) handle_rto();

  // Handle control plane requests.
  process_ctl_reqs();
}

void UcclRDMAEngine::handle_rto() {
  if (is_no_rto()) return;

  auto expired_qp_vec = rto_tm_.check_expired();

  for (auto data : expired_qp_vec) {
    auto* rdma_ctx = reinterpret_cast<struct RDMAContext*>(data.rdma_ctx);
    auto* subflow = reinterpret_cast<struct SubUcclFlow*>(data.flow);

    DCHECK(rdma_ctx && subflow);

    rdma_ctx->mark_flow_timeout(subflow);

    rdma_ctx->rto_retransmit_for_flow(subflow);
  }
}

/// TODO: handle error case
void UcclRDMAEngine::process_ctl_reqs() {
  Channel::CtrlMsg ctrl_work;

  // Process pending install flow works.
  auto nr_pending_works = pending_install_flow_works_.size();
  while (nr_pending_works--) {
    auto ctrl_work = pending_install_flow_works_.front();
    pending_install_flow_works_.pop_front();
    handle_install_flow_on_engine(ctrl_work);
  }

  while (jring_sc_dequeue_bulk(channel_->ctrl_cmdq_, &ctrl_work, 1, nullptr) ==
         1) {
    switch (ctrl_work.opcode) {
      case Channel::CtrlMsg::kInstallCtx:
        UCCL_LOG_ENGINE << "[Engine#" << engine_idx_ << "] "
                        << "kInstallCtx";
        handle_install_ctx_on_engine(ctrl_work);
        break;
      case Channel::CtrlMsg::kInstallFlow:
        UCCL_LOG_ENGINE << "[Engine#" << engine_idx_ << "] "
                        << "kInstallFlow";
        handle_install_flow_on_engine(ctrl_work);
        break;
      default:
        break;
    }
  }
}

void UcclRDMAEngine::handle_install_flow_on_engine(
    Channel::CtrlMsg& ctrl_work) {
  if (rdma_ctx_map_.find(ctrl_work.peer_id) == rdma_ctx_map_.end()) {
    pending_install_flow_works_.push_back(ctrl_work);
    return;
  }

  auto* rdma_ctx = rdma_ctx_map_[ctrl_work.peer_id];
  auto* poll_ctx = ctrl_work.poll_ctx;
  auto flow_id = ctrl_work.meta.install_flow.flow_id;
  auto* flow = reinterpret_cast<UcclFlow*>(ctrl_work.meta.install_flow.context);
  auto is_send = ctrl_work.meta.install_flow.is_send;

  DCHECK(flow_id < MAX_FLOW) << flow_id << ", " << MAX_FLOW;

  if (is_send) {
    rdma_ctx->add_sender_flow(flow, flow_id);
    io_ctx_.record_sender_ctx_mapping(ctrl_work.peer_id, flow_id, rdma_ctx);
  } else {
    rdma_ctx->add_receiver_flow(flow, flow_id);
    if constexpr (kReceiverCCA == RECEIVER_CCA_EQDS) {
      auto* subflow = flow->sub_flows_[engine_idx_ % ucclParamNUM_ENGINES()];

      subflow->pcb.eqds_cc.set_fid(flow_id);
      // All subflows belong to the same RDMAContext share the same
      // PacerCreditQPWrapper.
      subflow->pcb.eqds_cc.set_pacer_credit_qpw(&rdma_ctx->pc_qpw_);

      subflow->pcb.eqds_cc.init_active_item();
      subflow->pcb.eqds_cc.init_idle_item();

      subflow->pcb.eqds_cc.highest_pull_target_.store(
          eqds::EQDSCC::INIT_PULL_QUANTA);
      subflow->pcb.eqds_cc.latest_pull_ = eqds::EQDSCC::INIT_PULL_QUANTA;

      eqds_->request_pull(&subflow->pcb.eqds_cc);
    }
  }

  UCCL_LOG_ENGINE << "Installed flow: " << flow_id
                  << ", peerid: " << ctrl_work.peer_id
                  << " on engine: " << engine_idx_
                  << (is_send ? " (send)" : " (recv)")
                  << ", RDMAContext: " << rdma_ctx;

  uccl_wakeup(poll_ctx);
}

void UcclRDMAEngine::handle_install_ctx_on_engine(Channel::CtrlMsg& ctrl_work) {
  int ret;
  auto meta = ctrl_work.meta;
  auto info = &meta.install_ctx;

  int bootstrap_fd = info->bootstrap_fd;
  auto dev = dev_;

  RDMAContext* rdma_ctx;

  auto* next_install_engine = info->next_install_engine;

  {
    DCHECK(rdma_ctx_map_.find(ctrl_work.peer_id) == rdma_ctx_map_.end())
        << "PID:" << getpid() << ", peer_id:" << ctrl_work.peer_id
        << ", dev:" << dev << ", poll_ctx:" << ctrl_work.poll_ctx;

    rdma_ctx = RDMAFactory::CreateContext(
        &rto_tm_, &engine_outstanding_bytes_, eqds_, dev,
        engine_idx_ % ucclParamNUM_ENGINES(), meta, &io_ctx_);
    std::tie(std::ignore, ret) =
        rdma_ctx_map_.insert({ctrl_work.peer_id, rdma_ctx});
    DCHECK(ret);
  }

  // Create a thread to handle the QP setup to avoid blocking the engine.
  std::thread qp_setup_thread([this, ctrl_work, rdma_ctx, bootstrap_fd, dev,
                               next_install_engine]() {
    UCCL_LOG_ENGINE << "Engine#" << engine_idx_ << " launched QP setup thread";
    auto meta = ctrl_work.meta;
    auto info = &meta.install_ctx;
    auto* poll_ctx = ctrl_work.poll_ctx;
    // Send QPN to remote peer.
    int const size = sizeof(uint32_t);
    auto total_size = kTotalQP * size;

    if (!rdma_ctx->rc_mode()) {
      total_size += size; /* ctrl qpn */
      total_size += size; /* peer id */
    }

    char buf[total_size];
    for (auto i = 0; i < ucclParamPORT_ENTROPY(); i++) {
      memcpy(buf + i * size, &rdma_ctx->dp_qps_[i].qp->qp_num,
             sizeof(uint32_t));
    }

    if constexpr (kReceiverCCA == RECEIVER_CCA_EQDS) {
      memcpy(buf + ucclParamPORT_ENTROPY() * size,
             &rdma_ctx->credit_qp_->qp_num, sizeof(uint32_t));
    }

    // Send ctrl qpn and our peer id to remote peer.
    if (!rdma_ctx->rc_mode()) {
      memcpy(buf + kTotalQP * size, &io_ctx_.ctrl_qp_->qp_num,
             sizeof(uint32_t));
      memcpy(buf + kTotalQP * size + size, &ctrl_work.peer_id,
             sizeof(uint32_t));
    }

    // Wait until our turn to use bootstrap fd.
    auto engine_offset = engine_idx_ % ucclParamNUM_ENGINES();
    while (next_install_engine->load() != engine_offset) {
      // yield CPU
      std::this_thread::yield();
    }

    int ret = send_message(bootstrap_fd, buf, total_size);
    DCHECK(ret == total_size);

    // Receive QPN from remote peer.
    ret = receive_message(bootstrap_fd, buf, total_size);
    DCHECK(ret == total_size);

    // Let other engines to use bootstrap fd.
    next_install_engine->store(next_install_engine->load() + 1);

    // Modify QPs to RTR and RTS.
    for (auto i = 0; i < ucclParamPORT_ENTROPY(); i++) {
      auto remote_qpn = *reinterpret_cast<uint32_t*>(buf + i * size);
      auto qp = rdma_ctx->dp_qps_[i].qp;

      ret = modify_qp_rtr(qp, dev, &rdma_ctx->remote_ctx_, remote_qpn);
      DCHECK(ret == 0) << "Failed to modify data path QP to RTR";

      ret = modify_qp_rts(qp, rdma_ctx->rc_mode());
      DCHECK(ret == 0) << "Failed to modify data path QP to RTS";
    }

    if constexpr (kReceiverCCA == RECEIVER_CCA_EQDS) {
      auto credit_rqpn =
          *reinterpret_cast<uint32_t*>(buf + ucclParamPORT_ENTROPY() * size);
      auto credit_qp = rdma_ctx->credit_qp_;
      ret = modify_qp_rtr(credit_qp, dev, &rdma_ctx->remote_ctx_, credit_rqpn);
      DCHECK(ret == 0) << "Failed to modify Ctrl QP to RTR";
      ret = modify_qp_rts(credit_qp, false);
      DCHECK(ret == 0) << "Failed to modify Ctrl QP to RTS";
    }

    if (!rdma_ctx->rc_mode()) {
      auto ctrl_qpn = *reinterpret_cast<uint32_t*>(buf + kTotalQP * size);
      auto remote_peer_id =
          *reinterpret_cast<uint32_t*>(buf + kTotalQP * size + size);
      rdma_ctx->remote_ctx_.remote_ctrl_qpn = ctrl_qpn;
      rdma_ctx->remote_ctx_.remote_peer_id = remote_peer_id;
    }

    UCCL_LOG_ENGINE << "Installed ctx: " << ctrl_work.peer_id
                    << " on engine: " << engine_idx_
                    << ", RDMAContext: " << rdma_ctx;

    uccl_wakeup(poll_ctx);
  });

  // Detach the thread to allow it to run independently.
  qp_setup_thread.detach();
}

RDMAEndpoint::RDMAEndpoint(int num_engines_per_dev)
    : num_engines_per_dev_(num_engines_per_dev),
      stats_thread_([this]() { stats_thread_fn(); }) {
#ifndef DISABLE_CALL_ONCE_STATIC
  static std::once_flag flag_once;
  std::call_once(flag_once, [&]() {
#endif
    num_devices_ = RDMAFactory::init_devs();
    if (num_devices_ <= 0) {
      printf("Failed to initialize RDMA devices.\n");
      exit(EXIT_FAILURE);
    }
#ifndef DISABLE_CALL_ONCE_STATIC
  });
#endif

  rdma_ctl_ = rdma_ctl;

  ctx_pool_ = new SharedPool<PollCtx*, true>(
      kMaxInflightMsg, [](PollCtx* ctx) { ctx->clear(); });
  ctx_pool_buf_ = new uint8_t[kMaxInflightMsg * sizeof(PollCtx)];
  for (int i = 0; i < kMaxInflightMsg; i++) {
    ctx_pool_->push(new (ctx_pool_buf_ + i * sizeof(PollCtx)) PollCtx());
  }
  int total_num_engines = num_devices_ * num_engines_per_dev;
  initialize_resources(total_num_engines);

  for (int i = 0; i < total_num_engines; i++) channel_vec_[i] = new Channel();

  if constexpr (kReceiverCCA == RECEIVER_CCA_EQDS) {
    // Receiver-driven congestion control per device.
    for (int i = 0; i < get_num_devices(); i++) {
      auto factory_dev = RDMAFactory::get_factory_dev(i);
      eqds_[i] = new eqds::EQDS(i, factory_dev->link_bw);
    }
  }

  ib_relaxed_ordering_enabled_ = ncclIbRelaxedOrderingCapable();
  UCCL_LOG_EP << "IB relaxed ordering enabled: "
              << ib_relaxed_ordering_enabled_;
}

int try_bind_listen_socket(int* sock_fd, int base_port,
                           int max_attempts = 100) {
  struct sockaddr_in addr;
  int port = base_port;

  for (int attempt = 0; attempt < max_attempts; ++attempt, ++port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      perror("socket");
      return -1;
    }

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);

    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
      // The backlog will be silently truncated to the value in
      // /proc/sys/net/core/somaxconn
      if (listen(fd, 16384) != 0) {
        perror("listen");
        close(fd);
        return -1;
      }
      *sock_fd = fd;
      return port;
    }
    close(fd);
  }
  return -1;
}

bool RDMAEndpoint::initialize_engine_by_dev(int dev,
                                            bool enable_p2p_listen = false) {
  bool called = false;
#ifndef DISABLE_CALL_ONCE_STATIC
  static std::vector<std::once_flag> flags_per_dev_(MAX_IB_DEVS);
  std::call_once(flags_per_dev_[dev], [this, dev, enable_p2p_listen,
                                       &called]() {
#endif
    called = true;
    int start_engine_idx = dev * num_engines_per_dev_;
    int end_engine_idx = (dev + 1) * num_engines_per_dev_ - 1;
    int numa_node = RDMAFactory::get_factory_dev(dev)->numa_node;

    for (int engine_id = start_engine_idx; engine_id <= end_engine_idx;
         engine_id++) {
      int engine_cpu_id =
          ENGINE_CPU_START_LIST[dev] + engine_id % num_engines_per_dev_;
      UcclRDMAEngine* engine_ptr;
      {
        std::lock_guard<std::mutex> lock(engine_map_mu_);
        if (engine_id_to_engine_map_.find(engine_id) !=
            engine_id_to_engine_map_.end()) {
          UCCL_LOG_ENGINE << "Engine " << engine_id << " already exists.";
          exit(EXIT_FAILURE);
        }
        engine_id_to_engine_map_[engine_id] = std::make_unique<UcclRDMAEngine>(
            dev, engine_id, channel_vec_[engine_id], eqds_[dev]);

        engine_ptr = engine_id_to_engine_map_[engine_id].get();
      }
      {
        std::lock_guard<std::mutex> lock(engine_th_mu_);
        engine_th_vec_.emplace_back(std::make_unique<std::thread>(
            [engine_ptr, engine_id, engine_cpu_id, numa_node]() {
              if (ucclParamPIN_TO_NUMA()) {
                UCCL_LOG_ENGINE << "[Engine#" << engine_id << "] "
                                << "running on NUMA node " << numa_node;
                pin_thread_to_numa(numa_node);
              } else {
                DCHECK(engine_cpu_id < NUM_CPUS)
                    << "The target CPU id " << engine_cpu_id
                    << " is out of range, available CPUs:" << NUM_CPUS;
                UCCL_LOG_ENGINE << "[Engine#" << engine_id << "] "
                                << "running on CPU " << engine_cpu_id;
                pin_thread_to_cpu(engine_cpu_id);
              }
              engine_ptr->run();
            }));
      }
    }
    if (enable_p2p_listen) {
      p2p_listen_ports_[dev] = create_listen_socket(&p2p_listen_fds_[dev]);
      DCHECK(p2p_listen_ports_[dev] >= 0)
          << "Failed to bind after trying many ports!";
      printf("P2P listening on port %d\n", p2p_listen_ports_[dev]);
    }
#ifndef DISABLE_CALL_ONCE_STATIC
  });
#endif
  return called;
}

inline uint32_t RDMAEndpoint::find_pot_load_engine_idx(int dev) {
  auto c1 = find_oblivious_engine_idx(dev);
  auto c2 = find_least_loaded_engine_idx(dev);
  return engine_load_vec_[c1]->load() < engine_load_vec_[c2]->load() ? c1 : c2;
}

inline uint32_t RDMAEndpoint::find_least_loaded_engine_idx(int dev) {
  auto first_engine_idx = find_first_engine_idx_on_dev(dev);
  auto last_engine_idx = first_engine_idx + num_engines_per_dev_ - 1;

  uint32_t min_load = std::numeric_limits<uint32_t>::max();
  uint32_t candidate = 0;
  for (uint32_t i = first_engine_idx; i <= last_engine_idx; i++) {
    uint32_t load = engine_load_vec_[i]->load();
    if (load < min_load) {
      min_load = load;
      candidate = i;
    }
  }
  return candidate;
}

inline uint32_t RDMAEndpoint::find_oblivious_engine_idx(int dev) {
  return find_first_engine_idx_on_dev(dev) + std::rand() % num_engines_per_dev_;
}

inline uint32_t RDMAEndpoint::find_rr_engine_idx(int dev,
                                                 uint32_t* next_candidate) {
  uint32_t candidate = find_first_engine_idx_on_dev(dev) + *next_candidate;
  *next_candidate = (*next_candidate + 1) % num_engines_per_dev_;
  return candidate;
}

void UcclRDMAEngine::release() {
  for (auto& it : rdma_ctx_map_) {
    delete it.second;
  }
  rdma_ctx_map_.clear();
}

RDMAEndpoint::~RDMAEndpoint() {
  for (auto& [engine_id, engine] : engine_id_to_engine_map_) {
    engine->shutdown();
  }

  for (auto& engine_th : engine_th_vec_) {
    if (engine_th) {
      engine_th->join();
    }
  }
  for (auto& [engine_id, engine] : engine_id_to_engine_map_) {
    engine->release();
  }

  cleanup_resources();

  {
    std::lock_guard<std::mutex> lock(stats_mu_);
    shutdown_ = true;
    stats_cv_.notify_all();
  }

  if (stats_thread_.joinable()) {
    stats_thread_.join();
  }
}

PollCtx* RDMAEndpoint::install_flow_on_engine(uint32_t engine_idx,
                                              PeerID peer_id,
                                              union CtrlMeta meta) {
  auto* cmdq = channel_vec_[engine_idx]->ctrl_cmdq_;

  auto* poll_ctx = ctx_pool_->pop();
  Channel::CtrlMsg ctrl_msg = {
      .opcode = Channel::CtrlMsg::Op::kInstallFlow,
      .peer_id = peer_id,
      .meta = meta,
      .poll_ctx = poll_ctx,
  };

  while (jring_mp_enqueue_bulk(cmdq, &ctrl_msg, 1, nullptr) != 1) {
  }

  return poll_ctx;
}

PollCtx* RDMAEndpoint::install_ctx_on_engine(uint32_t engine_idx,
                                             PeerID peer_id,
                                             union CtrlMeta meta) {
  auto* cmdq = channel_vec_[engine_idx]->ctrl_cmdq_;

  auto* poll_ctx = ctx_pool_->pop();
  Channel::CtrlMsg ctrl_msg = {
      .opcode = Channel::CtrlMsg::Op::kInstallCtx,
      .peer_id = peer_id,
      .meta = meta,
      .poll_ctx = poll_ctx,
  };

  while (jring_mp_enqueue_bulk(cmdq, &ctrl_msg, 1, nullptr) != 1) {
  }

  UCCL_LOG_EP << "Request to install context on engine" << engine_idx
              << " for peer " << peer_id;

  return poll_ctx;
}

std::vector<PollCtx*> RDMAEndpoint::install_flow_on_engines(
    int dev, PeerID peer_id, FlowID flow_id, UcclFlow* flow, bool is_send) {
  union CtrlMeta meta = {};
  auto* info = &meta.install_flow;

  info->flow_id = flow_id;
  info->context = flow;
  info->is_send = is_send;

  std::vector<PollCtx*> poll_ctx_vec;
  for (int i = 0; i < num_engines_per_dev_; i++) {
    auto engine_idx = find_first_engine_idx_on_dev(dev) + i;
    auto* poll_ctx = install_flow_on_engine(engine_idx, peer_id, meta);
    poll_ctx_vec.push_back(poll_ctx);
  }

  UCCL_LOG_EP << "Installed flow " << flow_id << " on all engines";
  return poll_ctx_vec;
}

void RDMAEndpoint::install_ctx_on_engines(int fd, int dev, PeerID peer_id,
                                          std::string remote_ip, int remote_dev,
                                          int remote_gpuidx) {
  union CtrlMeta meta = {};
  auto* info = &meta.install_ctx;

  // synchronize GID and PortAttr with remote peer.
  int ret;
  auto factory_dev = RDMAFactory::get_factory_dev(dev);
  DCHECK(factory_dev) << "install_ctx_on_engines: get_factory_dev()";

  ret = send_message(fd, &factory_dev->gid.raw, 16);
  DCHECK(ret == 16) << "Failed to send GID";
  ret = receive_message(fd, &info->remote_gid.raw, 16);
  DCHECK(ret == 16) << "Failed to receive GID";

  ret = send_message(fd, &factory_dev->port_attr, sizeof(ibv_port_attr));
  DCHECK(ret == sizeof(ibv_port_attr)) << "Failed to send PortAttr";
  ret = receive_message(fd, &info->remote_port_attr, sizeof(ibv_port_attr));
  DCHECK(ret == sizeof(ibv_port_attr)) << "Failed to receive PortAttr";

  // Update peer map. Let other connect() go ahead.
  peer_map_mu_[dev]->lock();
  peer_map_[dev][{remote_ip, remote_dev, remote_gpuidx}].ready = 1;
  peer_map_[dev][{remote_ip, remote_dev, remote_gpuidx}].remote_gid =
      info->remote_gid;
  peer_map_[dev][{remote_ip, remote_dev, remote_gpuidx}].remote_port_attr =
      info->remote_port_attr;
  peer_map_mu_[dev]->unlock();

  std::atomic<int> next_install_engine = 0;
  info->next_install_engine = &next_install_engine;
  info->bootstrap_fd = fd;

  std::vector<PollCtx*> poll_ctx_vec;

  for (int i = 0; i < num_engines_per_dev_; i++) {
    auto engine_idx = find_first_engine_idx_on_dev(dev) + i;
    auto* poll_ctx = install_ctx_on_engine(engine_idx, peer_id, meta);
    poll_ctx_vec.push_back(poll_ctx);
  }
  for (auto* poll_ctx : poll_ctx_vec) {
    uccl_wait(poll_ctx);
  }
}

ConnID RDMAEndpoint::uccl_connect(int dev, int local_gpuidx, int remote_dev,
                                  int remote_gpuidx, std::string remote_ip,
                                  uint16_t remote_port) {
  struct sockaddr_in serv_addr = {};
  struct hostent* server;
  int ret;
  int bootstrap_fd;
  PeerID peer_id;
  struct RemoteRDMAContext remote_ctx;
  FlowID flow_id;

  bootstrap_fd = socket(AF_INET, SOCK_STREAM, 0);
  DCHECK(bootstrap_fd >= 0) << "uccl_connect: socket()";

  server = gethostbyname(remote_ip.c_str());
  DCHECK(server) << "uccl_connect: gethostbyname() " << remote_ip;

  // Force the socket to bind to the local IP address.
  sockaddr_in localaddr = {};
  localaddr.sin_family = AF_INET;
  auto* factory_dev = RDMAFactory::get_factory_dev(dev);
  DCHECK(factory_dev) << "uccl_connect: get_factory_dev()";
  localaddr.sin_addr.s_addr = str_to_ip(factory_dev->local_ip_str.c_str());
  ret = bind(bootstrap_fd, (sockaddr*)&localaddr, sizeof(localaddr));
  DCHECK(ret == 0) << "uccl_connect: bind()";

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = str_to_ip(remote_ip.c_str());
  serv_addr.sin_port = htons(remote_port);

  UCCL_LOG_EP << "connecting to "
              << "<" << remote_ip << ", " << remote_dev << ">:" << remote_port
              << " local/remote gpuidx: " << local_gpuidx << "/"
              << remote_gpuidx;

  // Connect and set nonblocking and nodelay
  while (
      connect(bootstrap_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr))) {
    UCCL_LOG_EP << "connecting... Make sure the server is up.";
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  int flag = 1;
  setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void*)&flag, sizeof(int));
  uint16_t local_port = 0;
  {
    struct sockaddr_in local_addr_check = {};
    socklen_t addr_len = sizeof(local_addr_check);
    ret = getsockname(bootstrap_fd, (struct sockaddr*)&local_addr_check,
                      &addr_len);
    DCHECK(ret == 0) << "getsockname() failed";
    local_port = ntohs(local_addr_check.sin_port);
  }

  bool first_call = false;
  bool should_install_ctx = false;

  // Send our dev, gpu to the other side.
  int buf[2] = {dev, local_gpuidx};
  ret = send_message(bootstrap_fd, buf, sizeof(int) * 2);
  DCHECK(ret == sizeof(int) * 2) << "uccl_connect: send_message()";

  bool is_leader =
      is_local_leader(dev, local_gpuidx, factory_dev->local_ip_str, local_port,
                      remote_dev, remote_gpuidx, remote_ip, remote_port);

  peer_map_mu_[dev]->lock();
  auto it = peer_map_[dev].find({remote_ip, remote_dev, remote_gpuidx});
  if (it == peer_map_[dev].end()) {
    peer_id = alloc_peer_id();
    peer_map_[dev].insert(
        {{remote_ip, remote_dev, remote_gpuidx}, {peer_id, {}, {}, 0}});
    first_call = true;
  } else {
    peer_id = it->second.peer_id;
    first_call = false;
  }
  peer_map_mu_[dev]->unlock();

  CHECK(peer_id < MAX_PEER)
      << "peer_id:" << peer_id << ", MAX_PEER:" << MAX_PEER;

  if (is_leader) {
    // We are the leader, we can install ctx if we are the first call.
    should_install_ctx = first_call;
    ret = send_message(bootstrap_fd, &first_call, sizeof(bool));
    DCHECK(ret == sizeof(bool)) << "uccl_connect: send_message()";
  } else {
    // We are not the leader, let the remote side to determine if we should
    // install ctx.
    ret = receive_message(bootstrap_fd, &should_install_ctx, sizeof(bool));
    DCHECK(ret == sizeof(bool)) << "uccl_connect: receive_message()";
  }

  if (should_install_ctx) {
    UCCL_LOG_EP << "connect: install_ctx_on_engines for dev/peer: " << dev
                << "/" << peer_id;
    install_ctx_on_engines(bootstrap_fd, dev, peer_id, remote_ip, remote_dev,
                           remote_gpuidx);
  }

  // Negotiate FlowID with server.
  ret = receive_message(bootstrap_fd, &flow_id, sizeof(FlowID));
  DCHECK(ret == sizeof(FlowID)) << "uccl_connect: receive_message()";

  UCCL_LOG_EP << "PID:" << getpid()
              << " connect: receive proposed FlowID: " << std::hex << "0x"
              << flow_id << " for dev/peer: " << dev << "/" << peer_id
              << ", local_gpuidx:" << local_gpuidx
              << ", remote_gpuidx:" << remote_gpuidx;

  // Create a new UcclFlow.
  auto* flow =
      new UcclFlow(this, dev, peer_id, flow_id, remote_ip, remote_dev, true);
  DCHECK(flow);

  auto poll_ctx_vec =
      install_flow_on_engines(dev, peer_id, flow_id, flow, true);

  auto remote_fifo_qpn = flow->create_fifo_and_gpuflush(bootstrap_fd, dev);

  while (1) {
    peer_map_mu_[dev]->lock();
    auto it = peer_map_[dev].find({remote_ip, remote_dev, remote_gpuidx});
    DCHECK(it != peer_map_[dev].end());

    if (it->second.ready == 1) {
      remote_ctx.remote_gid = it->second.remote_gid;
      remote_ctx.remote_port_attr = it->second.remote_port_attr;
      peer_map_mu_[dev]->unlock();
      break;
    }

    peer_map_mu_[dev]->unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  flow->modify_fifo(bootstrap_fd, dev, remote_ctx, remote_fifo_qpn);

  active_flows_spin_[dev].Lock();
  active_flows_vec_[dev].push_back(flow);
  active_flows_spin_[dev].Unlock();

  {
    std::lock_guard<std::mutex> lock(fd_vec_mu_);
    fd_vec_.push_back(bootstrap_fd);
  }

  for (auto* poll_ctx : poll_ctx_vec) {
    uccl_wait(poll_ctx);
  }

  return ConnID{.context = flow,
                .sock_fd = bootstrap_fd,
                .flow_id = flow_id,
                .peer_id = peer_id,
                .dev = dev};
}

ConnID RDMAEndpoint::uccl_accept(int dev, int listen_fd, int local_gpuidx,
                                 std::string& remote_ip, int* remote_dev,
                                 int* remote_gpuidx) {
  struct sockaddr_in cli_addr;
  socklen_t clien = sizeof(cli_addr);
  int bootstrap_fd;
  int ret;
  PeerID peer_id;
  struct RemoteRDMAContext remote_ctx;
  FlowID flow_id;

  auto* factory_dev = RDMAFactory::get_factory_dev(dev);
  DCHECK(factory_dev) << "uccl_accept: get_factory_dev()";

  bootstrap_fd = accept(listen_fd, (struct sockaddr*)&cli_addr, &clien);

  uint16_t local_port;
  {
    struct sockaddr_in local_addr = {};
    socklen_t len = sizeof(local_addr);
    int ret = getsockname(bootstrap_fd, (struct sockaddr*)&local_addr, &len);
    DCHECK(ret == 0) << "getsockname() failed ";
    local_port = ntohs(local_addr.sin_port);
  }

  DCHECK(bootstrap_fd >= 0) << "uccl_accept: accept()";
  remote_ip = ip_to_str(cli_addr.sin_addr.s_addr);

  UCCL_LOG_EP << "accept from " << remote_ip << ":" << cli_addr.sin_port;

  int flag = 1;
  setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void*)&flag, sizeof(int));

  bool first_call = false;
  bool should_install_ctx = false;

  int buf[2];
  ret = receive_message(bootstrap_fd, buf, sizeof(int) * 2);
  DCHECK(ret == sizeof(int) * 2) << "uccl_accept: receive_message()";
  *remote_dev = buf[0];
  *remote_gpuidx = buf[1];
  uint16_t remote_port = ntohs(cli_addr.sin_port);
  bool is_leader =
      is_local_leader(dev, local_gpuidx, factory_dev->local_ip_str, local_port,
                      *remote_dev, *remote_gpuidx, remote_ip, remote_port);
  peer_map_mu_[dev]->lock();
  auto it = peer_map_[dev].find({remote_ip, *remote_dev, *remote_gpuidx});
  if (it == peer_map_[dev].end()) {
    peer_id = alloc_peer_id();
    peer_map_[dev].insert(
        {{remote_ip, *remote_dev, *remote_gpuidx}, {peer_id, {}, {}, 0}});
    first_call = true;
  } else {
    peer_id = it->second.peer_id;
    first_call = false;
  }
  peer_map_mu_[dev]->unlock();

  if (is_leader) {
    // We are the leader, we can install ctx if we are the first call.
    should_install_ctx = first_call;
    ret = send_message(bootstrap_fd, &first_call, sizeof(bool));
    DCHECK(ret == sizeof(bool)) << "uccl_accept: send_message()";
  } else {
    // We are not the leader, let the remote side to determine if we should
    // install ctx.
    ret = receive_message(bootstrap_fd, &should_install_ctx, sizeof(bool));
    DCHECK(ret == sizeof(bool)) << "uccl_accept: receive_message()";
  }

  if (should_install_ctx) {
    UCCL_LOG_EP << "accept: install_ctx_on_engines for dev/peer: " << dev << "/"
                << peer_id;
    install_ctx_on_engines(bootstrap_fd, dev, peer_id, remote_ip, *remote_dev,
                           *remote_gpuidx);
  }

  // Negotiate FlowID with client.
  flow_id_spin_[dev][peer_id].Lock();
  flow_id = next_flow_id_[dev][peer_id]++;
  flow_id_spin_[dev][peer_id].Unlock();

  ret = send_message(bootstrap_fd, &flow_id, sizeof(FlowID));
  DCHECK(ret == sizeof(FlowID));

  UCCL_LOG_EP << "PID:" << getpid() << " accept: propose FlowID: " << std::hex
              << "0x" << flow_id << " for dev/peer: " << dev << "/" << peer_id;

  // Create a new UcclFlow.
  auto* flow =
      new UcclFlow(this, dev, peer_id, flow_id, remote_ip, *remote_dev, false);
  DCHECK(flow);

  auto poll_ctx_vec =
      install_flow_on_engines(dev, peer_id, flow_id, flow, false);

  auto remote_fifo_qpn = flow->create_fifo_and_gpuflush(bootstrap_fd, dev);

  while (1) {
    peer_map_mu_[dev]->lock();

    auto it = peer_map_[dev].find({remote_ip, *remote_dev, *remote_gpuidx});
    DCHECK(it != peer_map_[dev].end());

    if (it->second.ready == 1) {
      remote_ctx.remote_gid = it->second.remote_gid;
      remote_ctx.remote_port_attr = it->second.remote_port_attr;
      peer_map_mu_[dev]->unlock();
      break;
    }

    peer_map_mu_[dev]->unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  flow->modify_fifo(bootstrap_fd, dev, remote_ctx, remote_fifo_qpn);

  active_flows_spin_[dev].Lock();
  active_flows_vec_[dev].push_back(flow);
  active_flows_spin_[dev].Unlock();

  {
    std::lock_guard<std::mutex> lock(fd_vec_mu_);
    fd_vec_.push_back(bootstrap_fd);
  }

  for (auto* poll_ctx : poll_ctx_vec) {
    uccl_wait(poll_ctx);
  }

  return ConnID{.context = flow,
                .sock_fd = bootstrap_fd,
                .flow_id = flow_id,
                .peer_id = peer_id,
                .dev = dev};
}

bool UcclFlow::check_fifo_ready(int* ret_slot, int* ret_nmsgs) {
  int slot = send_comm_.fifo_head % kMaxReq;
  auto rem_fifo = send_comm_.base.fifo;
  volatile struct FifoItem* slots = rem_fifo->elems[slot];

  auto idx = send_comm_.fifo_head + 1;
  if (slots[0].idx != idx) return false;

  // Wait until all slots are ready
  auto nmsgs = slots[0].nmsgs;
  for (int i = 1; i < nmsgs; i++)
    while (slots[i].idx != idx) {
    }

  UCCL_LOG_EP << "send_async: found that receiver is ready to receive";

  __sync_synchronize();

  *ret_slot = slot;
  *ret_nmsgs = nmsgs;

  return true;
}

void UcclFlow::rc_send(struct ucclRequest* ureq) {
  auto* qp = send_comm_.base.rc_qp;
  auto size = ureq->send.data_len;
  auto laddr = ureq->send.laddr;
  auto raddr = ureq->send.raddr;
  auto lkey = ureq->send.lkey;
  auto rkey = ureq->send.rkey;

  struct ibv_sge sge;
  struct ibv_send_wr wr, *bad_wr = nullptr;

  sge.addr = laddr;
  sge.lkey = lkey;
  sge.length = size;

  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.next = nullptr;

  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.imm_data = htonl(size);
  wr.wr.rdma.remote_addr = raddr;
  wr.wr.rdma.rkey = rkey;

  wr.send_flags = IBV_SEND_SIGNALED;

  if (size <= kMaxInline) {
    wr.send_flags |= IBV_SEND_INLINE;
  }

  wr.wr_id = (uint64_t)&ureq->rc_or_flush_done;

  DCHECK(ibv_post_send(qp, &wr, &bad_wr) == 0) << "Failed to post send";
  flow_cq_cnt_++;
}

void UcclFlow::post_multi_send(struct ucclRequest** ureqs,
                               uint32_t engine_offset) {
  if (engine_offset == RDMAEndpoint::RC_MAGIC) {
    ureqs[0]->type = ReqTxRC;
    rc_send(ureqs[0]);
    return;
  }

  DCHECK(engine_offset < ucclParamNUM_ENGINES()) << engine_offset;

  uint32_t engine_idx = ep_->find_first_engine_idx_on_dev(dev_) + engine_offset;
  auto txq = ep_->channel_vec_[engine_idx]->tx_cmdq_;
  auto n = ureqs[0]->n;
  Channel::Msg msgs[kMaxRecv];
  for (int i = 0; i < n; i++) {
    msgs[i].opcode = Channel::Msg::Op::kTx;
    msgs[i].peer_id = peer_id_;
    ureqs[i]->mid = i;
    msgs[i].ureq = ureqs[i];
    msgs[i].poll_ctx = ureqs[i]->poll_ctx;
  }

  while (jring_mp_enqueue_bulk(txq, msgs, n, nullptr) != n) {
  }

  UCCL_LOG_EP << "Enqueue tx work to engine " << engine_idx;
}

void UcclFlow::rc_read(struct ucclRequest* ureq) {
  auto* qp = send_comm_.base.rc_qp;
  auto size = ureq->send.data_len;
  auto laddr = ureq->send.laddr;
  auto raddr = ureq->send.raddr;
  auto lkey = ureq->send.lkey;
  auto rkey = ureq->send.rkey;

  ibv_sge sge;
  ibv_send_wr wr, *bad = nullptr;

  sge.addr = laddr;
  sge.length = size;
  sge.lkey = lkey;

  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.next = nullptr;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.wr.rdma.remote_addr = raddr;
  wr.wr.rdma.rkey = rkey;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr_id = (uint64_t)&ureq->rc_or_flush_done;

  DCHECK(ibv_post_send(qp, &wr, &bad) == 0) << "RC read post failed";
  flow_cq_cnt_++;
  printf(
      "RC read posted: laddr: %lx, raddr: %lx, size: %u, lkey: %u, rkey: %u\n",
      laddr, raddr, size, lkey, rkey);
}

void UcclFlow::post_multi_write(struct ucclRequest** ureqs,
                                uint32_t engine_offset) {
  DCHECK(engine_offset != RDMAEndpoint::RC_MAGIC);
  uint32_t engine_idx = ep_->find_first_engine_idx_on_dev(dev_) + engine_offset;
  auto txq = ep_->channel_vec_[engine_idx]->tx_cmdq_;
  int n = ureqs[0]->n;

  Channel::Msg msgs[kMaxRecv];
  for (int i = 0; i < n; ++i) {
    msgs[i].opcode = Channel::Msg::Op::kWrite;
    msgs[i].peer_id = peer_id_;
    ureqs[i]->mid = i;
    msgs[i].ureq = ureqs[i];
    msgs[i].poll_ctx = ureqs[i]->poll_ctx;
  }
  while (jring_mp_enqueue_bulk(txq, msgs, n, nullptr) != n) {
  }
}

void UcclFlow::post_multi_read(ucclRequest** ureqs, uint32_t engine_offset) {
  DCHECK(engine_offset != RDMAEndpoint::RC_MAGIC);
  uint32_t engine_idx = ep_->find_first_engine_idx_on_dev(dev_) + engine_offset;
  auto txq = ep_->channel_vec_[engine_idx]->tx_cmdq_;
  int n = ureqs[0]->n;

  Channel::Msg msgs[kMaxRecv];
  for (int i = 0; i < n; ++i) {
    msgs[i].opcode = Channel::Msg::Op::kRead;
    msgs[i].peer_id = peer_id_;
    ureqs[i]->mid = i;
    msgs[i].ureq = ureqs[i];
    msgs[i].poll_ctx = ureqs[i]->poll_ctx;
  }
  while (jring_mp_enqueue_bulk(txq, msgs, n, nullptr) != n) {
  }
}

int RDMAEndpoint::prepare_fifo_metadata(UcclFlow* flow,
                                        struct Mhandle** mhandles,
                                        void const* data, size_t size,
                                        char* out_buf) {
  void* data_arr[1] = {const_cast<void*>(data)};
  int size_arr[1] = {static_cast<int>(size)};
  struct ibv_send_wr wr;
  struct ibv_sge sge;

  uint32_t engine_idx = 0;
  FifoItem* slots =
      flow->post_fifo(engine_idx, data_arr, size_arr, 1, mhandles, &wr, &sge);
  if (!slots) {
    fprintf(stderr,
            "prepare_fifo_metadata failed: post_fifo returned nullptr\n");
    return -1;
  }

  // Serialize metadata into out_buf
  FifoItem item;
  item.addr = slots[0].addr;
  item.size = size;
  item.rkey = slots[0].rkey;
  item.nmsgs = 1;
  item.rid = slots[0].rid;
  item.idx = slots[0].idx;
  item.engine_offset = slots[0].engine_offset;
  memset(item.padding, 0, sizeof(item.padding));

  serialize_fifo_item(item, out_buf);
  return 0;
}

int RDMAEndpoint::uccl_send_async(UcclFlow* flow, struct Mhandle* mhandle,
                                  void const* data, size_t const size,
                                  struct ucclRequest* ureq) {
  ureq->type = ReqTx;
  ureq->send.data_len = size;

  int slot, nmsg;

  if (!flow->check_fifo_ready(&slot, &nmsg)) return -1;
  DCHECK(slot < kMaxReq && nmsg <= kMaxRecv) << slot << ", nmsg" << nmsg;
  auto send_comm = &flow->send_comm_;
  auto ureqs = send_comm->fifo_ureqs[slot];
  auto rem_fifo = send_comm->base.fifo;
  volatile struct FifoItem* slots = rem_fifo->elems[slot];

  for (int i = 0; i < nmsg; i++) {
    if (ureqs[i] != nullptr) continue;
    DCHECK(!(slots[i].size < 0 || slots[i].addr == 0 || slots[i].rkey == 0))
        << slots[i].size << ", " << slots[i].addr << ", " << slots[i].rkey;

    if (size > slots[i].size) {
      // Can't send more than what the receiver can receive.
      // Adjust data_len to the actual size sent.
      ureq->send.data_len = slots[i].size;
    }

    ureq->send.laddr = (uint64_t)data;
    ureq->send.lkey = mhandle->mr->lkey;
    ureq->send.raddr = slots[i].addr;
    ureq->send.rkey = slots[i].rkey;
    ureq->n = nmsg;
    ureq->send.rid = slots[i].rid;
    ureq->send.sent_offset = 0;
    ureq->send.acked_bytes = 0;
    if (slots[i].engine_offset == RDMAEndpoint::RC_MAGIC)
      ureq->rc_or_flush_done = false;
    else {
      ureq->poll_ctx = ctx_pool_->pop();
      if constexpr (kEngineLBPolicy >= ENGINE_POLICY_LOAD) {
        ureq->engine_idx = slots[i].engine_offset;
        inc_load_on_engine(ureq->engine_idx);
      }
    }
    ureq->context = flow;
    ureq->send.inc_backlog = 0;
    // Track this request.
    ureqs[i] = ureq;

    // If this is a multi-recv, send only when all requests have matched.
    for (int i = 0; i < nmsg; i++) {
      if (ureqs[i] == nullptr) return 0;
    }

    // All requests have matched. Post works to the engine.
    flow->post_multi_send(ureqs, slots[i].engine_offset);

    // Move the head of the FIFO.
    send_comm->fifo_head++;

    memset((void*)slots, 0, sizeof(struct FifoItem));
    memset(ureqs, 0, kMaxRecv * sizeof(struct ucclRequest*));

    UCCL_LOG_EP << "send_async: posted " << nmsg << " requests"
                << " on engine " << slots[i].engine_offset << " size: " << size
                << " slot: " << slot << ", flow " << flow << ", flow->dev "
                << flow->dev_;

    return 0;
  }

  return 0;
}

int RDMAEndpoint::uccl_write_async(UcclFlow* flow, Mhandle* local_mh, void* src,
                                   size_t size, FifoItem const& slot_item,
                                   ucclRequest* ureq) {
  ureq->type = ReqWrite;
  ureq->n = 1;
  ureq->context = flow;
  ureq->send.laddr = reinterpret_cast<uint64_t>(src);
  ureq->send.lkey = local_mh->mr->lkey;
  ureq->send.raddr = slot_item.addr;
  ureq->send.rkey = slot_item.rkey;
  ureq->send.data_len =
      std::min<uint32_t>(static_cast<uint32_t>(size), slot_item.size);
  ureq->send.rid = slot_item.rid;

  DCHECK(slot_item.engine_offset != RDMAEndpoint::RC_MAGIC);

  if (ureq->poll_ctx == nullptr) {
    ureq->poll_ctx = ctx_pool_->pop();
    if (!ureq->poll_ctx) {
      fprintf(stderr, "uccl_write_async: ctx_pool empty, cannot post\n");
      return -1;
    }
  }
  ureq->engine_idx = slot_item.engine_offset;
  DCHECK(ureq->context) << "uccl_write_async: ureq->context is null";
  ucclRequest* one[1] = {ureq};
  flow->post_multi_write(one, slot_item.engine_offset);
  return 0;
}

int RDMAEndpoint::uccl_read_async(UcclFlow* flow, Mhandle* local_mh, void* dst,
                                  size_t size, FifoItem const& slot_item,
                                  ucclRequest* ureq) {
  ureq->type = ReqRead;
  ureq->n = 1;
  ureq->context = flow;
  ureq->send.laddr = reinterpret_cast<uint64_t>(dst);
  ureq->send.lkey = local_mh->mr->lkey;
  ureq->send.raddr = slot_item.addr;
  ureq->send.rkey = slot_item.rkey;
  ureq->send.data_len =
      std::min<uint32_t>(static_cast<uint32_t>(size), slot_item.size);
  ureq->send.rid = slot_item.rid;

  DCHECK(slot_item.engine_offset != RDMAEndpoint::RC_MAGIC);

  if (ureq->poll_ctx == nullptr) {
    ureq->poll_ctx = ctx_pool_->pop();
    if (!ureq->poll_ctx) {
      fprintf(stderr, "uccl_read_async_direct: ctx_pool empty, cannot post\n");
      return -1;
    }
  }
  ureq->engine_idx = slot_item.engine_offset;
  DCHECK(ureq->context) << "uccl_read_async_direct: ureq->context is null";
  ucclRequest* one[1] = {ureq};
  flow->post_multi_read(one, slot_item.engine_offset);
  return 0;
}

bool RDMAEndpoint::uccl_poll_ureq_once(struct ucclRequest* ureq) {
#ifdef __HIP_PLATFORM_AMD__
  if (ureq->type == ReqFlush) return true;
#endif

  bool ret;
  UcclFlow* flow = reinterpret_cast<UcclFlow*>(ureq->context);
  flow->poll_flow_cq();
  if (ureq->type == ReqTxRC || ureq->type == ReqRxRC ||
      ureq->type == ReqFlush ||
      (ureq->type == ReqRead && ureq->poll_ctx == nullptr)) {
    ret = ureq->rc_or_flush_done;
  } else {
    ret = uccl_poll_once(ureq->poll_ctx);
  }
  if ((ureq->type == ReqRx || ureq->type == ReqRxRC) && ret) {
    flow->dec_outstanding_reqs();
    if constexpr (kRCSize > 0) {
      if (ureq->recv.data_len[0] <= kRCSize && ureq->n == 1) {
        // This message should have used RC.
        // Give subsequent messages a chance to use RC.
        flow->set_last_rc_size(0);
      }
    }
  }

  if constexpr (kEngineLBPolicy >= ENGINE_POLICY_LOAD) {
    if (ureq->type == ReqTx || ureq->type == ReqRx)
      dec_load_on_engine(ureq->engine_idx);
  }

  return ret;
}

int RDMAEndpoint::uccl_flush(UcclFlow* flow, struct Mhandle** mhandles,
                             void** data, int* size, int n,
                             struct ucclRequest* ureq) {
  flow->poll_flow_cq();

  int last = flow->check_need_flush(size, n);
  if (last == -1) return 0;

#ifndef __HIP_PLATFORM_AMD__
  flow->post_flush(mhandles, data, size, n, &ureq->rc_or_flush_done, last);
#else
  ureq->rc_or_flush_done = true;
#endif

  ureq->type = ReqFlush;

  return 0;
}

int RDMAEndpoint::uccl_recv_async(UcclFlow* flow, struct Mhandle** mhandles,
                                  void** data, int* size, int n,
                                  struct ucclRequest* ureq) {
  uint32_t candidate;
  auto dev = flow->dev_;
  PollCtx* pacer_ctx;

  // Limit the maximum inflight requests for each flow.
  if (!flow->check_room()) return -1;

  flow->inc_outstanding_reqs();

  if constexpr (kRCSize > 0) {
    if (size[0] <= NCCL_MIN_POST_RECV && n == 1 &&
        flow->get_last_rc_size() <= kRCSize) {
      // set/get_last_rc_size is a workaround for NCCL using 65536 as the
      // minimum post recv size. Therefore, the receiver cannot determine in
      // advance whether the actual size of the message is <= kRCSize (and
      // thus use RC for the message). This workaround is based on the fact
      // that if a message <= kRCSize was sent previously, then the subsequent
      // message with post recv size == 65536 is also likely to be <= kRCSize.
      flow->rc_recv(data[0], size[0], mhandles[0], &ureq->recv.wr,
                    &ureq->recv.sge, ureq);
      ureq->type = ReqRxRC;
      ureq->context = flow;
      ureq->rc_or_flush_done = false;
      ureq->n = 1;

      flow->poll_flow_cq();
      return 0;
    }
  }

  // Select a engine to serve this request.
  if constexpr (kEngineLBPolicy == ENGINE_POLICY_BIND) {
    candidate = find_first_engine_idx_on_dev(dev) + flow->next_engine_offset_;
  } else if constexpr (kEngineLBPolicy == ENGINE_POLICY_RR) {
    candidate = find_rr_engine_idx(dev, &flow->next_engine_offset_);
  } else if constexpr (kEngineLBPolicy == ENGINE_POLICY_OBLIVIOUS) {
    candidate = find_oblivious_engine_idx(dev);
  } else if constexpr (kEngineLBPolicy == ENGINE_POLICY_LOAD_POT) {
    candidate = find_pot_load_engine_idx(dev);
    inc_load_on_engine(candidate);
    ureq->engine_idx = candidate;
  } else if constexpr (kEngineLBPolicy == ENGINE_POLICY_LOAD) {
    candidate = find_least_loaded_engine_idx(dev);
    inc_load_on_engine(candidate);
    ureq->engine_idx = candidate;
  }

  // Prepare to send recv buffer to sender.
  // Note that the real transmission is triggered by engine.
  auto elems = flow->post_fifo(candidate, data, size, n, mhandles,
                               &ureq->recv.wr, &ureq->recv.sge);
  ureq->type = ReqRx;
  ureq->context = flow;
  ureq->n = n;
  for (int i = 0; i < n; i++) ureq->recv.data_len[i] = size[i];
  ureq->poll_ctx = ctx_pool_->pop();
  ureq->recv.elems = elems;
  ureq->recv.qp = flow->recv_comm_.base.fifo_qp;

  Channel::Msg msg = {
      .opcode = Channel::Msg::Op::kRx,
      .peer_id = flow->peer_id_,
      .ureq = ureq,
      .poll_ctx = ureq->poll_ctx,
  };

  auto rxq = channel_vec_[candidate]->rx_cmdq_;
  while (jring_mp_enqueue_bulk(rxq, &msg, 1, nullptr) != 1) {
  }

  UCCL_LOG_EP << "recv_async: posted " << n << " requests"
              << " on engine " << candidate << " size: " << size[0];

  flow->poll_flow_cq();

  return 0;
}

void RDMAEndpoint::stats_thread_fn() {
  if (GetEnvVar("UCCL_ENGINE_QUIET") == "1") return;

  while (!shutdown_) {
    {
      std::unique_lock<std::mutex> lock(stats_mu_);
      bool shutdown =
          stats_cv_.wait_for(lock, std::chrono::seconds(kStatsTimerIntervalSec),
                             [this] { return shutdown_.load(); });
      if (shutdown) break;
    }

    if (engine_id_to_engine_map_.empty()) {
      // No engines created yet, skip stats.
      continue;
    }
    std::string s;
    uint32_t eidx = 0;
    for (auto& [engine_id, engine] : engine_id_to_engine_map_) {
#ifdef STATS
      s = engine->status_to_string();
      if (!s.empty()) {
        std::cout << "[Engine#" << std::to_string(eidx++) << "]\n";
        std::cout << s;
      }
#endif
    }
  }
}

int RDMAEndpoint::uccl_regmr_dmabuf(UcclFlow* flow, void* addr, size_t len,
                                    int type, int offset, int fd,
                                    struct Mhandle** mhandle) {
  return uccl_regmr_dmabuf(flow->dev_, addr, len, type, offset, fd, mhandle);
}

int RDMAEndpoint::uccl_regmr_dmabuf(int dev, void* addr, size_t len,
                                    int type /*unsed for now*/, int offset,
                                    int fd, struct Mhandle** mhandle) {
  auto factory_dev = RDMAFactory::get_factory_dev(dev);

  unsigned int flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  if (ib_relaxed_ordering_enabled_) flags |= IBV_ACCESS_RELAXED_ORDERING;

  *mhandle = new Mhandle();
  (*mhandle)->mr = ibv_reg_dmabuf_mr(factory_dev->pd, offset, len,
                                     (uint64_t)addr, fd, flags);

  return 0;
}

int RDMAEndpoint::uccl_regmr(UcclFlow* flow, void* addr, size_t len,
                             int type /*unsed for now*/,
                             struct Mhandle** mhandle) {
  return uccl_regmr(flow->dev_, addr, len, type, mhandle);
}

int RDMAEndpoint::uccl_regmr(int dev, void* addr, size_t len,
                             int type /*unsed for now*/,
                             struct Mhandle** mhandle) {
  auto factory_dev = RDMAFactory::get_factory_dev(dev);

  unsigned int flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  if (ib_relaxed_ordering_enabled_) flags |= IBV_ACCESS_RELAXED_ORDERING;

  *mhandle = new Mhandle();
  if (ib_relaxed_ordering_enabled_) {
    (*mhandle)->mr =
        ibv_reg_mr_iova2(factory_dev->pd, addr, len, (uint64_t)addr, flags);
  } else {
    (*mhandle)->mr = ibv_reg_mr(factory_dev->pd, addr, len, flags);
  }
  if (!(*mhandle)->mr) {
    std::cerr << "ibv_reg_mr failed (" << strerror(errno) << "), len=" << len
              << " addr=" << addr << "\n";
    delete *mhandle;
    *mhandle = nullptr;
    return -1;
  }
  return 0;
}

void RDMAEndpoint::uccl_deregmr(struct Mhandle* mhandle) {
  ibv_dereg_mr(mhandle->mr);
  delete mhandle;
}

std::string UcclRDMAEngine::status_to_string() {
  std::string s;

  for (auto rdma_ctx : rdma_ctx_map_) {
    s += "    [Context#" + std::to_string(rdma_ctx.first) + "]";
    s += rdma_ctx.second->to_string();
    s += "    # of active timers:" + std::to_string(rto_tm_.size()) + "\n";
  }

  return s;
}

RDMAContext::RDMAContext(TimerManager* rto, uint32_t* engine_unacked_bytes,
                         eqds::EQDS* eqds, int dev, uint32_t engine_offset,
                         union CtrlMeta meta, SharedIOContext* io_ctx)
    : rto_(rto),
      engine_unacked_bytes_(engine_unacked_bytes),
      eqds_(eqds),
      io_ctx_(io_ctx),
      engine_offset_(engine_offset),
      wheel_({freq_ghz, us_to_cycles(kWheelSlotWidthUs, freq_ghz),
              us_to_cycles(kWheelHorizonUs, freq_ghz), kBktPoolSize,
              RDMAFactory::get_factory_dev(dev)->link_bw}) {
  int ret;
  auto* factory_dev = RDMAFactory::get_factory_dev(dev);

  context_ = factory_dev->context;
  gid_idx_ = factory_dev->gid_idx;
  is_roce_ = factory_dev->is_roce;

  port_entropy_ = ucclParamPORT_ENTROPY();
  dp_qps_.resize(port_entropy_);
  chunk_size_ = (ucclParamCHUNK_SIZE_KB() << 10);

  link_speed_ = util_rdma_get_link_speed_from_ibv_speed(
      factory_dev->port_attr.active_speed, factory_dev->port_attr.active_width);
  remote_ctx_.remote_gid = meta.install_ctx.remote_gid;
  remote_ctx_.remote_port_attr = meta.install_ctx.remote_port_attr;

  remote_ctx_.dest_ah = util_rdma_create_ah(
      factory_dev->pd, factory_dev->ib_port_num, remote_ctx_.remote_gid,
      remote_ctx_.remote_port_attr, is_roce_, gid_idx_);
  UCCL_INIT_CHECK(remote_ctx_.dest_ah != nullptr, "util_rdma_create_ah failed");

  mtu_bytes_ =
      util_rdma_get_mtu_from_ibv_mtu(factory_dev->port_attr.active_mtu);

  pd_ = factory_dev->pd;

  // Create data path QPs. (UC/RC)
  struct ibv_qp_init_attr qp_init_attr;
  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  qp_init_attr.qp_context = this;
  qp_init_attr.send_cq = ibv_cq_ex_to_cq(io_ctx->send_cq_ex_);
  qp_init_attr.recv_cq = ibv_cq_ex_to_cq(io_ctx->recv_cq_ex_);
  if (!rc_mode())
    qp_init_attr.qp_type = IBV_QPT_UC;
  else
    qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.cap.max_send_wr = 2 * kMaxReq * kMaxRecv;
  qp_init_attr.cap.max_send_sge = kMaxSge;
  qp_init_attr.cap.max_inline_data = 0;
  qp_init_attr.srq = io_ctx->srq_;

  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = factory_dev->ib_port_num;
  qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  qpAttr.max_rd_atomic = 8;
  qpAttr.max_dest_rd_atomic = 8;

  for (int i = 0; i < ucclParamPORT_ENTROPY(); i++) {
    struct ibv_qp* qp = ibv_create_qp(pd_, &qp_init_attr);
    UCCL_INIT_CHECK(qp != nullptr, "ibv_create_qp failed for data path QP");

    // Modify QP state to INIT.
    UCCL_INIT_CHECK(ibv_modify_qp(qp, &qpAttr,
                                  IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                      IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
                    "ibv_modify_qp failed");

    dp_qps_[i].qp = qp;
    qpn2idx_.insert({qp->qp_num, i});

    io_ctx->record_qpn_ctx_mapping(qp->qp_num, this);
  }

  // Initialize work request extension buffer pool.
  wr_ex_pool_.emplace();

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));

  if constexpr (kReceiverCCA == RECEIVER_CCA_EQDS) {
    // Create Credit QP, SCQ/RCQ and MR for engine or pacer.
    util_rdma_create_qp_seperate_cq(
        context_, &credit_qp_, IBV_QPT_UC, true, false,
        (struct ibv_cq**)&pacer_credit_cq_ex_,
        (struct ibv_cq**)&engine_credit_cq_ex_, false, kCQSize, pd_,
        factory_dev->ib_port_num, eqds::CreditChunkBuffPool::kNumChunk,
        eqds::CreditChunkBuffPool::kNumChunk, 1, 1);

    auto addr =
        mmap(nullptr, eqds::CreditChunkBuffPool::kCreditMRSize,
             PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    UCCL_INIT_CHECK(addr != MAP_FAILED, "mmap failed");
    engine_credit_mr_ =
        ibv_reg_mr(pd_, addr, eqds::CreditChunkBuffPool::kCreditMRSize,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    UCCL_INIT_CHECK(engine_credit_mr_ != nullptr,
                    "ibv_reg_mr failed for engine credit MR");

    addr = mmap(nullptr, eqds::CreditChunkBuffPool::kCreditMRSize,
                PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    UCCL_INIT_CHECK(addr != MAP_FAILED, "mmap failed");
    pacer_credit_mr_ =
        ibv_reg_mr(pd_, addr, eqds::CreditChunkBuffPool::kCreditMRSize,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    UCCL_INIT_CHECK(pacer_credit_mr_ != nullptr,
                    "ibv_reg_mr failed for pacer credit MR");

    // Initialize Credit packet buffer pool for engine and pacer, respectively.
    engine_credit_chunk_pool_.emplace(engine_credit_mr_);
    pacer_credit_chunk_pool_.emplace(pacer_credit_mr_);

    pc_qpw_.credit_qp_ = credit_qp_;
    pc_qpw_.pacer_credit_cq_ = pacer_credit_cq_ex_;
    pc_qpw_.pacer_credit_chunk_pool_ = &(*pacer_credit_chunk_pool_);

    INIT_LIST_HEAD(&pc_qpw_.poll_item.poll_link);
    pc_qpw_.poll_item.pc_qpw = &pc_qpw_;

    // Populate recv work requests on Credit QP for consuming credit packets.
    {
      struct ibv_sge sge;
      for (int i = 0; i < (eqds::CreditChunkBuffPool::kNumChunk - 1) / 2; i++) {
        uint64_t chunk_addr;
        UCCL_INIT_CHECK(engine_credit_chunk_pool_->alloc_buff(&chunk_addr) == 0,
                        "Failed to allocate buffer for credit packet");
        sge.addr = chunk_addr;
        sge.length = eqds::CreditChunkBuffPool::kChunkSize;
        sge.lkey = engine_credit_chunk_pool_->get_lkey();
        wr.wr_id = chunk_addr;
        wr.next = nullptr;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        struct ibv_recv_wr* bad_wr;
        UCCL_INIT_CHECK(ibv_post_recv(credit_qp_, &wr, &bad_wr) == 0,
                        "ibv_post_recv failed");
      }
    }

    for (int i = 0; i < kPostRQThreshold; i++) {
      credit_recv_wrs_.recv_sges[i].lkey =
          engine_credit_chunk_pool_->get_lkey();
      credit_recv_wrs_.recv_sges[i].length =
          eqds::CreditChunkBuffPool::kChunkSize;
      credit_recv_wrs_.recv_wrs[i].sg_list = &credit_recv_wrs_.recv_sges[i];
      credit_recv_wrs_.recv_wrs[i].num_sge = 1;
      credit_recv_wrs_.recv_wrs[i].next =
          (i == kPostRQThreshold - 1) ? nullptr
                                      : &credit_recv_wrs_.recv_wrs[i + 1];
    }
  }

  // Timing wheel.
  wheel_.catchup();
}

RDMAContext::~RDMAContext() {
  if (!rc_mode()) {
    if (credit_qp_ != nullptr) {
      ibv_destroy_qp(credit_qp_);
    }
    if (engine_credit_mr_ != nullptr) {
      munmap(engine_credit_mr_->addr, engine_credit_mr_->length);
      ibv_dereg_mr(engine_credit_mr_);
    }
    if (pacer_credit_mr_ != nullptr) {
      munmap(pacer_credit_mr_->addr, pacer_credit_mr_->length);
      ibv_dereg_mr(pacer_credit_mr_);
    }
  }

  for (int i = 0; i < ucclParamPORT_ENTROPY(); i++) {
    ibv_destroy_qp(dp_qps_[i].qp);
  }

  if (pd_ != nullptr) {
    ibv_dealloc_pd(pd_);
  }

  UCCL_LOG_RE << "RDMAContext destroyed";
}

int RDMAContext::supply_rx_buff(struct ucclRequest* ureq) {
  DCHECK(ureq);
  auto* elems = ureq->recv.elems;
  DCHECK(elems);

  auto* flow = reinterpret_cast<UcclFlow*>(ureq->context);
  auto* subflow = flow->sub_flows_[engine_offset_];

  auto req = subflow->alloc_recvreq();
  if (req == nullptr) return -1;
  DCHECK(ureq->n == 1);
  for (int i = 0; i < ureq->n; i++) {
    // For sender to encode the request id in the immediate data.
    elems[i].rid = subflow->get_recvreq_id(req);
  }

  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(ureq->recv.qp, &ureq->recv.wr, &bad_wr);
  DCHECK(ret == 0) << ret;

  req->type = RecvRequest::RECV;
  req->ureq = ureq;
  memset(req->received_bytes, 0, sizeof(uint32_t) * kMaxRecv);
  req->fin_msg = 0;

  UCCL_LOG_IO << "Really supply rx buff by posting buffers to FIFO QP, rid#"
              << subflow->get_recvreq_id(req);

  return 0;
}

bool RDMAContext::receiverCC_tx_message(struct ucclRequest* ureq) {
  auto* flow = reinterpret_cast<UcclFlow*>(ureq->context);
  auto* subflow = flow->sub_flows_[engine_offset_];
  auto* eqds = &subflow->pcb.eqds_cc;

  auto size = ureq->send.data_len;
  auto laddr = ureq->send.laddr;
  auto raddr = ureq->send.raddr;
  auto lkey = ureq->send.lkey;
  auto rkey = ureq->send.rkey;
  auto* sent_offset = &ureq->send.sent_offset;
  uint64_t wr_addr;
  bool queued = false;
  uint32_t chunk_size;

  auto now = rdtsc();

  if (ureq->send.inc_backlog == 0) {
    ureq->send.inc_backlog = 1;
    subflow->backlog_bytes_ += size;
  }

  if (subflow->in_rtx) {
    // We have to wait for the retransmission to finish.
    // Drain the retransmission queue.
    drain_rtx_queue(subflow);
    return false;
  }

  while (*sent_offset < size || size == 0 /* zero-length message */) {
    chunk_size = EventOnChunkSize(subflow, size - *sent_offset);

    if (chunk_size == 0 && size) return false;

    subflow->backlog_bytes_ -= chunk_size;

    auto pull_target = eqds->compute_pull_target(subflow, chunk_size);

    DCHECK(wr_ex_pool_->alloc_buff(&wr_addr) == 0);
    struct wr_ex* wr_ex = reinterpret_cast<struct wr_ex*>(wr_addr);
    auto wr = &wr_ex->wr;

    wr_ex->sge.addr = laddr + *sent_offset;
    wr_ex->sge.lkey = lkey;
    wr_ex->sge.length = chunk_size;

    wr->wr.rdma.remote_addr = raddr + *sent_offset;
    wr->wr.rdma.rkey = rkey;

    IMMDataEQDS imm_data(0);

    imm_data.SetTarget(pull_target);

    imm_data.SetFID(flow->flowid());
    if ((*sent_offset + chunk_size == size)) {
      // Last chunk of the message.
      imm_data.SetHINT(1);
    }
    imm_data.SetRID(ureq->send.rid);

    imm_data.SetCSN(subflow->pcb.get_snd_nxt().to_uint32());

    wr->imm_data = htonl(imm_data.GetImmData());

    // We use high 8 bits of wr_id to store CSN.
    // Lower 56 bits to store subflow pointer.
    if (io_ctx_->is_rc_mode())
      wr->wr_id = (1ULL * imm_data.GetCSN()) << 56 | (uint64_t)subflow;
    else
      wr->wr_id = 0;

    // Select QP.
    auto qpidx = EventOnSelectPath(subflow, chunk_size);
    auto qpw = &dp_qps_[qpidx];

    wr->send_flags = 0;
    if (qpw->signal_cnt_++ % kSignalInterval == 0) {
      wr->send_flags = IBV_SEND_SIGNALED;
    }
    // if (size <= kMaxInline) {
    //   wr->send_flags |= IBV_SEND_INLINE;
    // }
    wr_ex->qpidx = qpidx;

    struct ibv_send_wr* bad_wr;
    DCHECK(ibv_post_send(qpw->qp, wr, &bad_wr) == 0);

    // Track this chunk.
    subflow->txtracking.track_chunk(ureq, wr_ex, now, imm_data.GetCSN(),
                                    imm_data.GetHINT());
    if (!io_ctx_->is_rc_mode()) {
      // Arm timer for TX
      arm_timer_for_flow(subflow);
    }

    *sent_offset += chunk_size;

    UCCL_LOG_IO << "Tx: flow#" << flow->flowid() << ", req id#"
                << ureq->send.rid << ", msg id#" << ureq->mid
                << ", csn:" << imm_data.GetCSN()
                << ", remaining bytes:" << size - *sent_offset
                << ", pull target:" << (uint32_t)pull_target << " with QP#"
                << qpidx;

    subflow->unacked_bytes_ += chunk_size;
    *engine_unacked_bytes_ += chunk_size;

    /* zero-length message */
    if (size == 0) break;
  }

  return true;
}

bool RDMAContext::senderCC_tx_message(struct ucclRequest* ureq) {
  auto* flow = reinterpret_cast<UcclFlow*>(ureq->context);
  DCHECK(flow);
  auto* subflow = flow->sub_flows_[engine_offset_];

  auto size = ureq->send.data_len;
  auto laddr = ureq->send.laddr;
  auto raddr = ureq->send.raddr;
  auto lkey = ureq->send.lkey;
  auto rkey = ureq->send.rkey;
  auto* sent_offset = &ureq->send.sent_offset;
  uint64_t wr_addr;
  bool queued = false;
  uint32_t chunk_size;
  uint32_t qpidx;

  auto now = rdtsc();

  while (*sent_offset < size || size == 0 /* zero-length message */) {
    chunk_size = EventOnChunkSize(subflow, size - *sent_offset);

    if (chunk_size == 0 && size) return false;

    if (io_ctx_->bypass_pacing()) {
      DCHECK(wr_ex_pool_->alloc_buff(&wr_addr) == 0);
      struct wr_ex* wr_ex = reinterpret_cast<struct wr_ex*>(wr_addr);
      auto wr = &wr_ex->wr;

      wr_ex->sge.addr = laddr + *sent_offset;
      wr_ex->sge.lkey = lkey;
      wr_ex->sge.length = chunk_size;

      wr->wr.rdma.remote_addr = raddr + *sent_offset;
      wr->wr.rdma.rkey = rkey;

      IMMData imm_data(0);

      imm_data.SetFID(flow->flowid());
      if ((*sent_offset + chunk_size == size)) {
        // Last chunk of the message.
        imm_data.SetHINT(1);
      }
      imm_data.SetRID(ureq->send.rid);

      imm_data.SetCSN(subflow->pcb.get_snd_nxt().to_uint32());

      wr->imm_data = htonl(imm_data.GetImmData());

      // We use high 8 bits of wr_id to store CSN.
      // Lower 56 bits to store subflow pointer.
      if (io_ctx_->is_rc_mode())
        wr->wr_id = (1ULL * imm_data.GetCSN()) << 56 | (uint64_t)subflow;
      else
        wr->wr_id = 0;

      // Select QP.
      qpidx = select_qpidx_pot(chunk_size, subflow);
      auto qpw = &dp_qps_[qpidx];

      wr->send_flags = 0;
      if (qpw->signal_cnt_++ % kSignalInterval == 0) {
        wr->send_flags = IBV_SEND_SIGNALED;
      }
      // if (size <= kMaxInline) {
      //   wr->send_flags |= IBV_SEND_INLINE;
      // }
      wr_ex->qpidx = qpidx;

      struct ibv_send_wr* bad_wr;
      DCHECK(ibv_post_send(qpw->qp, wr, &bad_wr) == 0);

      // Track this chunk.
      subflow->txtracking.track_chunk(ureq, wr_ex, now, imm_data.GetCSN(),
                                      imm_data.GetHINT());
      if (!io_ctx_->is_rc_mode()) {
        // Arm timer for TX
        arm_timer_for_flow(subflow);
      }

      *sent_offset += chunk_size;

      UCCL_LOG_IO << "Tx: flow#" << flow->flowid() << "/" << flow << ", req id#"
                  << ureq->send.rid << ", msg id#" << ureq->mid
                  << ", csn:" << imm_data.GetCSN()
                  << ", remaining bytes:" << size - *sent_offset << " with QP#"
                  << qpidx;

      subflow->unacked_bytes_ += chunk_size;
      *engine_unacked_bytes_ += chunk_size;
      /* zero-length message */
      if (size == 0) break;

      continue;
    }

    // Prepare SGE.
    DCHECK(wr_ex_pool_->alloc_buff(&wr_addr) == 0);
    struct wr_ex* wr_ex = reinterpret_cast<struct wr_ex*>(wr_addr);
    auto wr = &wr_ex->wr;
    wr_ex->sge.addr = laddr + *sent_offset;
    wr_ex->sge.lkey = lkey;
    wr_ex->sge.length = chunk_size;

    // wr->sg_list/num_sge/next/opcode are already set.

    wr->wr.rdma.remote_addr = raddr + *sent_offset;
    wr->wr.rdma.rkey = rkey;

    UCCL_LOG_IO << "remote_addr: " << wr->wr.rdma.remote_addr
                << ", rkey: " << wr->wr.rdma.rkey;

    IMMData imm_data(0);

    imm_data.SetFID(flow->flowid());
    if ((*sent_offset + chunk_size == size)) {
      // Last chunk of the message.
      imm_data.SetHINT(1);
    }
    imm_data.SetRID(ureq->send.rid);

    imm_data.SetCSN(subflow->pcb.get_snd_nxt().to_uint32());

    wr->imm_data = htonl(imm_data.GetImmData());

    // We use high 8 bits of wr_id to store CSN.
    // Lower 56 bits to store subflow pointer.
    if (io_ctx_->is_rc_mode())
      wr->wr_id = (1ULL * imm_data.GetCSN()) << 56 | (uint64_t)subflow;
    else
      wr->wr_id = 0;
    bool roce = is_roce();
    {
      auto wheel = &wheel_;
      uint32_t hdr_overhead;
      if (likely(chunk_size == chunk_size_ && mtu_bytes_ == 4096)) {
        hdr_overhead = roce ? MAX_CHUNK_IB_4096_HDR_OVERHEAD
                            : MAX_CHUNK_ROCE_IPV4_4096_HDR_OVERHEAD;
      } else {
        auto num_mtu = (chunk_size + mtu_bytes_) / mtu_bytes_;
        hdr_overhead =
            num_mtu * (roce ? ROCE_IPV4_HDR_OVERHEAD : IB_HDR_OVERHEAD);
      }

      // Enforce global cwnd.
      queued = EventOnQueueData(subflow, wr_ex, chunk_size + hdr_overhead, now);

      if (queued) {
        // Queue the SGE on the timing wheel.
        subflow->in_wheel_cnt_++;
        // For future tracking.
        wr_ex->ureq = ureq;
        UCCL_LOG_IO << "Queued " << chunk_size
                    << " bytes to timing wheel for flow#" << flow->flowid();
      } else {
        // Transmit this chunk directly.
        // Select QP.
        auto qpidx = select_qpidx_pot(chunk_size, subflow);
        auto qpw = &dp_qps_[qpidx];
        // There is no need to signal every WQE since we don't handle TX
        // completions. But we still need occasionally post a request
        // with the IBV_SEND_SIGNALED flag. See
        // https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/.
        wr_ex->wr.send_flags = 0;
        if (qpw->signal_cnt_++ % kSignalInterval == 0) {
          wr_ex->wr.send_flags = IBV_SEND_SIGNALED;
        }
        // if (size <= kMaxInline) {
        //   wr_ex->wr.send_flags |= IBV_SEND_INLINE;
        // }
        wr_ex->qpidx = qpidx;
        struct ibv_send_wr* bad_wr;
        auto ret = ibv_post_send(qpw->qp, wr, &bad_wr);
        DCHECK(ret == 0) << ret;

        // Track this chunk.
        subflow->txtracking.track_chunk(ureq, wr_ex, now, imm_data.GetCSN(),
                                        imm_data.GetHINT());
        if (!io_ctx_->is_rc_mode()) {
          // Arm timer for TX
          arm_timer_for_flow(subflow);
        }

        UCCL_LOG_IO << "Directly sent " << chunk_size << " bytes to QP#"
                    << qpidx;
      }
    }

    *sent_offset += chunk_size;

    UCCL_LOG_IO << "Tx: flow#" << flow->flowid() << ", req id#"
                << ureq->send.rid << ", msg id#" << ureq->mid
                << ", csn:" << imm_data.GetCSN()
                << ", remaining bytes:" << size - *sent_offset << " with QP#"
                << qpidx;

    subflow->unacked_bytes_ += chunk_size;
    *engine_unacked_bytes_ += chunk_size;
    /* zero-length message */
    if (size == 0) break;
  }

  return true;
}

bool RDMAContext::senderCC_tx_write(struct ucclRequest* ureq) {
  auto* flow = reinterpret_cast<UcclFlow*>(ureq->context);
  auto* subflow = flow->sub_flows_[engine_offset_];

  auto size = ureq->send.data_len;
  auto laddr = ureq->send.laddr;
  auto raddr = ureq->send.raddr;
  auto lkey = ureq->send.lkey;
  auto rkey = ureq->send.rkey;
  uint32_t* sent_offset = &ureq->send.sent_offset;
  uint64_t wr_addr;
  uint32_t chunk_size;
  if (size == 0) {
    DCHECK(false) << "RDMA WRITE len 0";
    return true;
  }

  DCHECK(size > 0) << "RDMA WRITE size: " << size;

  while (*sent_offset < size || size == 0) {
    chunk_size = EventOnChunkSize(subflow, size - *sent_offset);

    if (chunk_size == 0 && size) return false;

    CHECK_EQ(wr_ex_pool_->alloc_buff(&wr_addr), 0);
    struct wr_ex* wr_ex = reinterpret_cast<struct wr_ex*>(wr_addr);
    auto* wr = &wr_ex->wr;

    wr_ex->sge.addr = laddr + *sent_offset;
    wr_ex->sge.lkey = lkey;
    wr_ex->sge.length = chunk_size;

    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->wr.rdma.remote_addr = raddr + *sent_offset;
    wr->wr.rdma.rkey = rkey;
    wr->sg_list = &wr_ex->sge;
    wr->num_sge = 1;
    wr->send_flags = IBV_SEND_SIGNALED;
    // We use high 8 bits of wr_id to store CSN.
    // Lower 56 bits to store subflow pointer.

    int csn = subflow->pcb.get_snd_nxt().to_uint32();
    wr->wr_id = (1ULL * csn) << 56 | (uint64_t)subflow;

    uint32_t qpidx = select_qpidx_pot(chunk_size, subflow);
    auto& qpw = dp_qps_[qpidx];
    wr_ex->qpidx = qpidx;

    struct ibv_send_wr* bad;
    int ret = ibv_post_send(qpw.qp, wr, &bad);
    if (ret) {
      fprintf(stderr,
              "ibv_post_send failed: ret=%d (%s)   wr_id=%lu  lkey=0x%x "
              "rkey=0x%x  len=%u\n",
              ret, strerror(ret), wr->wr_id, wr_ex->sge.lkey, wr->wr.rdma.rkey,
              wr_ex->sge.length);
      return false;
    }
    int hint = 0;
    if (*sent_offset + chunk_size == size) {
      // Last chunk of the message.
      hint = 1;
    }
    subflow->txtracking.track_chunk(ureq, wr_ex, rdtsc(), csn, hint);
    *sent_offset += chunk_size;

    subflow->unacked_bytes_ += chunk_size;
    *engine_unacked_bytes_ += chunk_size;

    if (size == 0) break;

    continue;
  }

  return true;
}

bool RDMAContext::senderCC_tx_read(struct ucclRequest* ureq) {
  auto* flow = reinterpret_cast<UcclFlow*>(ureq->context);
  auto* subflow = flow->sub_flows_[engine_offset_];

  auto size = ureq->send.data_len;
  auto laddr = ureq->send.laddr;
  auto raddr = ureq->send.raddr;
  auto lkey = ureq->send.lkey;
  auto rkey = ureq->send.rkey;
  uint32_t* sent_offset = &ureq->send.sent_offset;
  uint64_t wr_addr;
  uint32_t chunk_size;
  if (size == 0) {
    DCHECK(false) << "RDMA READ len 0";
    return true;
  }

  DCHECK(size > 0) << "RDMA READ size: " << size;

  while (*sent_offset < size || size == 0) {
    chunk_size = EventOnChunkSize(subflow, size - *sent_offset);

    if (chunk_size == 0 && size) return false;

    CHECK_EQ(wr_ex_pool_->alloc_buff(&wr_addr), 0);
    struct wr_ex* wr_ex = reinterpret_cast<struct wr_ex*>(wr_addr);
    auto* wr = &wr_ex->wr;

    wr_ex->sge.addr = laddr + *sent_offset;
    wr_ex->sge.lkey = lkey;
    wr_ex->sge.length = chunk_size;

    wr->opcode = IBV_WR_RDMA_READ;
    wr->wr.rdma.remote_addr = raddr + *sent_offset;
    wr->wr.rdma.rkey = rkey;
    wr->sg_list = &wr_ex->sge;
    wr->num_sge = 1;
    wr->send_flags = IBV_SEND_SIGNALED;
    // We use high 8 bits of wr_id to store CSN.
    // Lower 56 bits to store subflow pointer.

    int csn = subflow->pcb.get_snd_nxt().to_uint32();
    wr->wr_id = (1ULL * csn) << 56 | (uint64_t)subflow;

    uint32_t qpidx = select_qpidx_pot(chunk_size, subflow);
    auto& qpw = dp_qps_[qpidx];
    wr_ex->qpidx = qpidx;

    struct ibv_send_wr* bad;
    int ret = ibv_post_send(qpw.qp, wr, &bad);
    if (ret) {
      fprintf(stderr,
              "ibv_post_send failed: ret=%d (%s)   wr_id=%lu  lkey=0x%x "
              "rkey=0x%x  len=%u\n",
              ret, strerror(ret), wr->wr_id, wr_ex->sge.lkey, wr->wr.rdma.rkey,
              wr_ex->sge.length);
      return false;
    }
    int hint = 0;
    if (*sent_offset + chunk_size == size) {
      // Last chunk of the message.
      hint = 1;
    }
    subflow->txtracking.track_chunk(ureq, wr_ex, rdtsc(), csn, hint);
    *sent_offset += chunk_size;

    subflow->unacked_bytes_ += chunk_size;
    *engine_unacked_bytes_ += chunk_size;

    if (size == 0) break;

    continue;
  }

  return true;
}

void RDMAContext::uc_post_acks() {
  int num_ack = 0;
  struct list_head *pos, *n;
  auto chunk_addr = io_ctx_->pop_ctrl_chunk();

  list_for_each_safe(pos, n, &ack_list_) {
    auto ack_item = list_entry(pos, struct ack_item, ack_link);
    auto subflow = ack_item->subflow;
    DCHECK(num_ack < kMaxBatchCQ);
    craft_ack(subflow, chunk_addr, num_ack++);
    list_del(pos);
  }
  try_post_acks(num_ack, chunk_addr, false);
  if (num_ack == 0) {
    io_ctx_->push_ctrl_chunk(chunk_addr);
  }

  INIT_LIST_HEAD(&ack_list_);
}

template <typename T>
void RDMAContext::rc_rx_ack(T* wc_or_cq_ex) {
  static constexpr bool is_wc = std::is_same_v<T, struct ibv_wc>;
  auto now = rdtsc();

  auto wr_id = wc_or_cq_ex->wr_id;
  auto csn = (wr_id >> 56) & 0xff;
  auto subflow = reinterpret_cast<SubUcclFlow*>((wr_id & 0xffffffffffffff));

  auto pair = subflow->txtracking.ack_rc_transmitted_chunks(
      subflow, this, csn, now, &subflow->unacked_bytes_, engine_unacked_bytes_);

#ifdef TEST_CC_REACTION
  if constexpr (is_wc) {
    auto hw_ts = convert_nic_to_host(ibv_wc_read_completion_ts(wc_or_cq_ex));

    static bool first = true;
    static double avg_react_delay = 0.0;
    static int count = 0;
    auto reaction_delay = to_usec(now - hw_ts, freq_ghz);

    if (reaction_delay <
            500 /* filter wrong values (probabaly due to clock sync) */
        && count++ > 5000 /* warmup */) {
      if (first) {
        avg_react_delay = reaction_delay;
        first = false;
      } else {
        avg_react_delay =
            (avg_react_delay * count + reaction_delay) / (count + 1);
      }
      LOG_EVERY_N(INFO, 1000)
          << "CC decision delay: " << reaction_delay
          << "us, Average CC decision delay: " << avg_react_delay << "us";
    }
  }
#endif

  subflow->update_scoreboard_rtt(pair.first, pair.second);

  UCCL_LOG_IO << "Received ACK for csn: " << csn;
}

void RDMAContext::check_credit_rq(bool force) {
  // Populate recv work requests for consuming credit packets.
  while (credit_recv_wrs_.post_rq_cnt >= kPostRQThreshold) {
    struct ibv_recv_wr* bad_wr;
    for (int i = 0; i < kPostRQThreshold; i++) {
      uint64_t chunk_addr;
      DCHECK(engine_credit_chunk_pool_->alloc_buff(&chunk_addr) == 0);
      credit_recv_wrs_.recv_sges[i].addr = chunk_addr;
      credit_recv_wrs_.recv_wrs[i].wr_id = chunk_addr;
    }
    DCHECK(ibv_post_recv(credit_qp_, &credit_recv_wrs_.recv_wrs[0], &bad_wr) ==
           0);
    UCCL_LOG_IO << "Posted " << credit_recv_wrs_.post_rq_cnt
                << " recv requests for Credit QP";
    credit_recv_wrs_.post_rq_cnt -= kPostRQThreshold;
  }

  if (force && credit_recv_wrs_.post_rq_cnt) {
    struct ibv_recv_wr* bad_wr;
    for (int i = 0; i < credit_recv_wrs_.post_rq_cnt; i++) {
      uint64_t chunk_addr;
      DCHECK(engine_credit_chunk_pool_->alloc_buff(&chunk_addr) == 0);
      credit_recv_wrs_.recv_sges[i].addr = chunk_addr;
      credit_recv_wrs_.recv_wrs[i].wr_id = chunk_addr;
    }
    credit_recv_wrs_.recv_wrs[credit_recv_wrs_.post_rq_cnt - 1].next = nullptr;
    DCHECK(ibv_post_recv(credit_qp_, &credit_recv_wrs_.recv_wrs[0], &bad_wr) ==
           0);
    UCCL_LOG_IO << "Posted " << credit_recv_wrs_.post_rq_cnt
                << " recv requests for Credit QP";
    credit_recv_wrs_.recv_wrs[credit_recv_wrs_.post_rq_cnt - 1].next =
        &credit_recv_wrs_.recv_wrs[credit_recv_wrs_.post_rq_cnt];
    credit_recv_wrs_.post_rq_cnt = 0;
  }
}

void RDMAContext::drain_rtx_queue(SubUcclFlow* subflow) {
  fast_retransmit_for_flow(subflow);
}

bool RDMAContext::try_retransmit_chunk(SubUcclFlow* subflow,
                                       struct wr_ex* wr_ex) {
  if (!EventOnTxRTXData(subflow, wr_ex)) return false;

  auto* lossy_qpw = &dp_qps_[wr_ex->qpidx];
  struct ibv_send_wr retr_wr, *bad_wr;

  // Use SEND/RECV for retransmission through the original lossy QP.
  memset(&retr_wr, 0, sizeof(retr_wr));
  struct ibv_sge retr_sge[2];

  uint64_t retr_hdr;

  retr_hdr = io_ctx_->pop_retr_hdr();

  struct retr_chunk_hdr* hdr =
      reinterpret_cast<struct retr_chunk_hdr*>(retr_hdr);
  hdr->remote_addr = wr_ex->wr.wr.rdma.remote_addr;
  // Network byte order.
  hdr->imm_data = wr_ex->wr.imm_data;

  retr_sge[0].addr = retr_hdr;
  retr_sge[0].length = sizeof(struct retr_chunk_hdr);
  retr_sge[0].lkey = io_ctx_->get_retr_hdr_lkey();

  retr_sge[1] = wr_ex->sge;

  auto* cqe_desc = io_ctx_->pop_cqe_desc();
  cqe_desc->data = retr_hdr;
  retr_wr.wr_id = (uint64_t)cqe_desc;
  retr_wr.sg_list = retr_sge;
  retr_wr.num_sge = 2;
  retr_wr.opcode = IBV_WR_SEND;
  retr_wr.send_flags = IBV_SEND_SIGNALED;
  retr_wr.next = nullptr;

  int ret = ibv_post_send(lossy_qpw->qp, &retr_wr, &bad_wr);
  DCHECK(ret == 0) << ret;

  UCCL_LOG_IO << "successfully retransmit chunk for QP#"
              << std::distance(dp_qps_.begin(), dp_qps_.begin() + wr_ex->qpidx)
              << ", remote_addr: " << wr_ex->wr.wr.rdma.remote_addr
              << ", chunk_size: " << wr_ex->sge.length
              << ", csn: " << IMMData(ntohl(wr_ex->wr.imm_data)).GetCSN()
              << " for flow: " << subflow->fid_;

  return true;
}

void RDMAContext::rx_credit(uint64_t pkt_addr) {
  auto* ucclpullh = reinterpret_cast<UcclPullHdr*>(pkt_addr);

  auto fid = ucclpullh->fid.value();
  auto pullno = ucclpullh->pullno.value();

  auto* flow = reinterpret_cast<UcclFlow*>(sender_flow_tbl_[fid]);

  if (unlikely(!flow)) {
    // No Bug. This only happens during connection setup when the sender
    // is not ready while the receiver has already started to send credits.
    return;
  }

  auto* subflow = flow->sub_flows_[engine_offset_];

  EventOnRxCredit(subflow, pullno);
}

template <typename T>
void RDMAContext::uc_rx_ack(T* wc_or_cq_ex, UcclSackHdr* ucclsackh) {
  static constexpr bool is_wc = std::is_same_v<T, struct ibv_wc>;
  uint64_t t5;
  auto t6 = rdtsc();

  auto fid = ucclsackh->fid.value();
  auto qpidx = ucclsackh->path.value();
  auto ackno = ucclsackh->ackno.value();

  DCHECK(fid < MAX_FLOW);
  auto* flow = reinterpret_cast<UcclFlow*>(sender_flow_tbl_[fid]);
  auto* subflow = flow->sub_flows_[engine_offset_];

  bool update_sackbitmap = false;

  if (UINT_CSN::uintcsn_seqno_lt(ackno, subflow->pcb.snd_una)) {
    UCCL_LOG_IO << "Received old ACK " << ackno << " for flow" << fid << "/"
                << flow << " by Ctrl QP";
  } else if (UINT_CSN::uintcsn_seqno_gt(ackno, subflow->pcb.snd_nxt)) {
    UCCL_LOG_IO << "Received ACK for untransmitted data "
                << "ackno: " << ackno
                << ", snd_nxt: " << subflow->pcb.snd_nxt.to_uint32()
                << " for flow" << fid << "/" << flow << " by Ctrl QP";
  } else if (UINT_CSN::uintcsn_seqno_eq(ackno, subflow->pcb.snd_una)) {
    UCCL_LOG_IO << "Received duplicate ACK " << ackno << " for flow" << fid
                << "/" << flow
                << ", snd_una: " << subflow->pcb.snd_una.to_uint32()
                << " by Ctrl QP";

    EventOnRxNACK(subflow, ucclsackh);

    update_sackbitmap = true;

    subflow->pcb.duplicate_acks++;
    subflow->pcb.snd_ooo_acks = ucclsackh->sack_bitmap_count.value();
    int fast_rexmit_thres = ((is_roce()) ? ROCE_DUP_ACK_THRES : 65536);

    if (subflow->pcb.duplicate_acks < fast_rexmit_thres) {
      // We have not reached the threshold yet, so we do not do
      // retransmission.
    } else if (subflow->pcb.duplicate_acks == fast_rexmit_thres) {
      // Fast retransmit.
      fast_retransmit_for_flow(subflow);
    } else {
      // We have already done the fast retransmit, so we are now
      // in the fast recovery phase.
      auto sack_bitmap_count = ucclsackh->sack_bitmap_count.value();
      // We check the SACK bitmap to see if there are more undelivered
      // chunks. In fast recovery mode we get after a fast
      // retransmit, we will retransmit all missing chunks that we
      // find from the SACK bitmap, when enumerating the SACK bitmap
      // for up to sack_bitmap_count ACKs.
      uint32_t index = 0;
      while (sack_bitmap_count && index < kSackBitmapSize &&
             !subflow->txtracking.empty()) {
        auto bucket_idx = index / PCB::kSackBitmapBucketSize;
        auto sack_bitmap = ucclsackh->sack_bitmap[bucket_idx].value();

        auto cursor = index % PCB::kSackBitmapBucketSize;

        if ((sack_bitmap & (1ULL << cursor)) == 0) {
          // We found a hole.
          auto seqno = subflow->pcb.snd_una + index;
          auto chunk = subflow->txtracking.get_unacked_chunk_from_idx(index);
          if (seqno == chunk.csn) {
            auto wr_ex = chunk.wr_ex;
            if (try_retransmit_chunk(subflow, wr_ex)) {
              subflow->pcb.stats_fast_rexmits++;
            } else {
              // We can't retransmit the chunk due to lack of
              // credits. Quit the loop.
              index = kSackBitmapSize;
            }
          }
          // Rearm timer for Retransmission.
          rearm_timer_for_flow(subflow);
        } else {
          sack_bitmap_count--;
        }
        index++;
      }
    }

  } else {
    UCCL_LOG_IO << "Received valid ACK " << ackno << " for flow" << fid << "/"
                << flow << " by Ctrl QP";

    EventOnRxACK(subflow, ucclsackh);

    update_sackbitmap = true;
    auto num_acked_chunks = UINT_CSN(ackno) - subflow->pcb.snd_una;
    auto remote_queueing_tsc =
        us_to_cycles((ucclsackh->remote_queueing.value()), freq_ghz);
    if constexpr (kTestNoHWTimestamp || is_wc)
      t5 = t6;
    else
      t5 = convert_nic_to_host(ibv_wc_read_completion_ts(wc_or_cq_ex));

    DCHECK(engine_offset_ < ucclParamNUM_ENGINES());
    auto reduced_bytes = subflow->unacked_bytes_;
    auto newrtt_tsc = subflow->txtracking.ack_transmitted_chunks(
        subflow, this, num_acked_chunks.to_uint32(), t5, t6,
        remote_queueing_tsc, &subflow->unacked_bytes_);
    reduced_bytes -= subflow->unacked_bytes_;
    *engine_unacked_bytes_ -= reduced_bytes;
    if (qpidx < port_entropy_)
      subflow->update_scoreboard_rtt(newrtt_tsc, qpidx);
    else {
      // This ack is for retransmitted chunk.
      // Don't update scoreboard for retransmitted chunks.
    }

    subflow->pcb.snd_una = ackno;
    subflow->pcb.duplicate_acks = 0;
    subflow->pcb.snd_ooo_acks = 0;
    subflow->pcb.rto_rexmits_consectutive = 0;
    if (!subflow->txtracking.empty()) {
      // Rearm timer if we still have unacked chunks.
      rearm_timer_for_flow(subflow);
    } else {
      disarm_timer_for_flow(subflow);
    }
  }

  // For duplicate ACKs and valid ACKs, we may need to update the SACK bitmap
  // at the sender side.
  if (update_sackbitmap) {
    for (int i = 0; i < kSackBitmapSize / PCB::kSackBitmapBucketSize; i++)
      subflow->pcb.tx_sack_bitmap[i] = ucclsackh->sack_bitmap[i].value();
    subflow->pcb.tx_sack_bitmap_count = ucclsackh->sack_bitmap_count.value();
    subflow->pcb.tx_sack_bitmap_base = ackno;
  }
}

int RDMAContext::poll_credit_cq(void) {
  uint64_t chunk_addr;
  int work = 0;
  while (1) {
    struct ibv_poll_cq_attr poll_cq_attr = {};
    auto cq_ex = engine_credit_cq_ex_;
    if (ibv_start_poll(cq_ex, &poll_cq_attr)) return work;

    int cq_budget = 0;

    while (1) {
      if (cq_ex->status == IBV_WC_SUCCESS) {
        // Completion for receiving ACKs.
        chunk_addr = cq_ex->wr_id;
        if (ibv_wc_read_opcode(cq_ex) == IBV_WC_RECV) {
          rx_credit(chunk_addr);
          credit_recv_wrs_.post_rq_cnt++;
        }
        engine_credit_chunk_pool_->free_buff(chunk_addr);
      } else {
        LOG(ERROR) << "Credit CQ state error: " << cq_ex->status;
      }

      if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
    }

    ibv_end_poll(cq_ex);

    work += cq_budget;

    if (cq_budget < kMaxBatchCQ) break;
  }

  return work;
}

void RDMAContext::burst_timing_wheel(void) {
  auto wheel = &wheel_;
  struct ibv_send_wr* bad_wr;

  wheel->reap(rdtsc());

  auto num_chunks = std::min(kMaxBurstTW, (uint32_t)wheel->ready_queue_.size());
  auto kMaxUnAckedBytesPerEngineHigh =
      (is_roce() ? kMaxUnAckedBytesPerEngineHighForRoCE
                 : kMaxUnAckedBytesPerEngineHighForIB);

  for (auto i = 0; i < num_chunks; i++) {
    struct wr_ex* wr_ex =
        reinterpret_cast<struct wr_ex*>(wheel->ready_queue_.front().sslot_);
    auto* wr = &wr_ex->wr;
    auto* flow = reinterpret_cast<UcclFlow*>(wr_ex->ureq->context);
    auto* subflow = flow->sub_flows_[engine_offset_];
    // Select QP.
    auto qpidx = select_qpidx_pot(wr_ex->sge.length, subflow);
    auto qpw = &dp_qps_[qpidx];

    wr->send_flags = 0;
    if (qpw->signal_cnt_++ % kSignalInterval == 0) {
      wr->send_flags = IBV_SEND_SIGNALED;
    }
    wr_ex->qpidx = qpidx;

    auto ret = ibv_post_send(qpw->qp, &wr_ex->wr, &bad_wr);
    DCHECK(ret == 0) << ret;

    IMMData imm_data(ntohl(wr_ex->wr.imm_data));
    // Track this chunk.
    subflow->txtracking.track_chunk(wr_ex->ureq, wr_ex, rdtsc(),
                                    imm_data.GetCSN(), imm_data.GetHINT());
    if (!io_ctx_->is_rc_mode()) {
      // Arm timer for TX
      arm_timer_for_flow(subflow);
    }
    UCCL_LOG_IO << "Burst send: csn: " << imm_data.GetCSN() << " with QP#"
                << wr_ex->qpidx;

    subflow->in_wheel_cnt_--;

    wheel->ready_queue_.pop_front();

    if (*engine_unacked_bytes_ >= kMaxUnAckedBytesPerEngineHigh) {
      // The code is here because we want to at least send one chunk.
      // Push the message to the pending transmit queue.
      return;
    }
  }
}

void RDMAContext::try_update_csn(SubUcclFlow* subflow) {
  while (!subflow->rxtracking.ready_csn_.empty() &&
         subflow->rxtracking.ready_csn_.begin()->first.to_uint32() ==
             subflow->pcb.rcv_nxt.to_uint32()) {
    struct RecvRequest* req = reinterpret_cast<struct RecvRequest*>(
        subflow->rxtracking.ready_csn_.begin()->second);
    if (req) {
      // This is the last chunk of a message whose size is mismatched with
      // the expected size. I.e., send size < recv size. Fix
      // req->ureq->data_len[0] and wakeup application.
      req->ureq->recv.data_len[0] = req->received_bytes[0];
      // Wakeup app thread.
      uccl_wakeup(req->ureq->poll_ctx);
      UCCL_LOG_IO << "Rx message complete.";
      // Free the request.
      subflow->free_recvreq(req);
    }

    subflow->rxtracking.ready_csn_.erase(
        subflow->rxtracking.ready_csn_.begin());

    // Data is already DMAed to the application buffer.
    // Nothing more to do.

    subflow->pcb.advance_rcv_nxt();
    UCCL_LOG_IO << "try_update_csn:"
                << " rcv_nxt: " << subflow->pcb.rcv_nxt.to_uint32();

    if (!io_ctx_->is_rc_mode()) {
      subflow->pcb.sack_bitmap_shift_left_one();
    }
  }
}

template <typename T>
void RDMAContext::uc_rx_rtx_chunk(T* wc_or_cq_ex, uint64_t chunk_addr) {
  static constexpr bool is_wc = std::is_same_v<T, struct ibv_wc>;
  UCCL_LOG_IO << "uc_rx_rtx_chunk";
  auto now = rdtsc();

  uint32_t byte_len;
  if constexpr (is_wc) {
    byte_len = wc_or_cq_ex->byte_len;
  } else {
    byte_len = ibv_wc_read_byte_len(wc_or_cq_ex);
  }
  auto chunk_len = byte_len - sizeof(struct retr_chunk_hdr);

  struct retr_chunk_hdr* hdr =
      reinterpret_cast<struct retr_chunk_hdr*>(chunk_addr);
  auto imm_data = IMMData(ntohl(hdr->imm_data));

  auto last_chunk = imm_data.GetHINT();
  auto csn = imm_data.GetCSN();
  auto rid = imm_data.GetRID();
  auto fid = imm_data.GetFID();

  DCHECK(fid < MAX_FLOW);
  auto* flow = reinterpret_cast<UcclFlow*>(receiver_flow_tbl_[fid]);
  auto* subflow = flow->sub_flows_[engine_offset_];

  UCCL_LOG_IO << "Received retransmission chunk: (csn, rid, fid): " << csn
              << ", " << rid << ", " << fid;

  // Locate request by rid
  DCHECK(rid < kMaxReq);
  auto req = subflow->get_recvreq_by_id(rid);
  if (req->type != RecvRequest::RECV || req->ureq->context != flow) {
    UCCL_LOG_IO << "Can't find corresponding request or this request is "
                   "invalid for this retransmission chunk. Dropping. "
                << req->type;
    subflow->pcb.stats_retr_chunk_drop++;
    return;
  }

  // Compare CSN with the expected CSN.
  auto ecsn = subflow->pcb.rcv_nxt;
  auto distance = UINT_CSN(csn) - ecsn;

  if (UINT_CSN::uintcsn_seqno_lt(UINT_CSN(csn), ecsn)) {
    // Original chunk is already received.
    UCCL_LOG_IO << "Original chunk is already received. Dropping "
                   "retransmission chunk for flow"
                << fid;
    subflow->pcb.stats_retr_chunk_drop++;
    return;
  }

  if (distance.to_uint32() > kReassemblyMaxSeqnoDistance) {
    UCCL_LOG_IO << "Chunk too far ahead. Dropping as we can't handle SACK. "
                << "csn: " << csn << ", ecsn: " << ecsn.to_uint32();
    subflow->pcb.stats_retr_chunk_drop++;
    return;
  }

  if (subflow->pcb.sack_bitmap_bit_is_set(distance.to_uint32())) {
    UCCL_LOG_IO << "Original chunk is already received. Dropping "
                   "retransmission chunk for flow"
                << fid;
    subflow->pcb.stats_retr_chunk_drop++;
    return;
  }

  UCCL_LOG_IO << "This retransmission chunk is accepted!!!";
// Accept this retransmission chunk.
#ifdef CPU_MEMORY
  memcpy(reinterpret_cast<void*>(hdr->remote_addr),
         reinterpret_cast<void*>(chunk_addr + sizeof(struct retr_chunk_hdr)),
         chunk_len);
#else
#ifndef __HIP_PLATFORM_AMD__
  cudaMemcpy(
      reinterpret_cast<void*>(hdr->remote_addr),
      reinterpret_cast<void*>(chunk_addr + sizeof(struct retr_chunk_hdr)),
      chunk_len, cudaMemcpyHostToDevice);
#else
  DCHECK(hipMemcpy(reinterpret_cast<void*>(hdr->remote_addr),
                   reinterpret_cast<void*>(chunk_addr +
                                           sizeof(struct retr_chunk_hdr)),
                   chunk_len, hipMemcpyHostToDevice) == hipSuccess);
#endif
#endif

  subflow->pcb.stats_accept_retr++;

  subflow->pcb.sack_bitmap_bit_set(distance.to_uint32());

  auto* msg_size = &req->ureq->recv.elems[0].size;
  uint32_t* received_bytes = req->received_bytes;
  received_bytes[0] += chunk_len;

  if (!last_chunk) {
    req = nullptr;
  }

  subflow->rxtracking.ready_csn_.insert({csn, req});

  try_update_csn(subflow);

  /// FIXME: Should we send ACK immediately here?
  if (list_empty(&subflow->ack.ack_link))
    list_add_tail(&subflow->ack.ack_link, &ack_list_);
  // Don't let sender update the path's rtt.
  subflow->next_ack_path_ = std::numeric_limits<uint16_t>::max();

  EventOnRxRTXData(subflow, &imm_data);

  return;
}

template <typename T>
void RDMAContext::rc_rx_chunk(T* wc_or_cq_ex) {
  static constexpr bool is_wc = std::is_same_v<T, struct ibv_wc>;
  uint32_t byte_len, imm_value;
  if constexpr (is_wc) {
    byte_len = wc_or_cq_ex->byte_len;
    imm_value = wc_or_cq_ex->imm_data;
  } else {
    byte_len = ibv_wc_read_byte_len(wc_or_cq_ex);
    imm_value = ibv_wc_read_imm_data(wc_or_cq_ex);
  }
  auto imm_data = IMMData(ntohl(imm_value));

  auto last_chunk = imm_data.GetHINT();
  auto csn = imm_data.GetCSN();
  auto rid = imm_data.GetRID();
  auto fid = imm_data.GetFID();

  DCHECK(fid < MAX_FLOW);
  auto* flow = reinterpret_cast<UcclFlow*>(receiver_flow_tbl_[fid]);
  DCHECK(flow) << fid << ", RDMAContext ptr: " << this;
  auto* subflow = flow->sub_flows_[engine_offset_];

  UCCL_LOG_IO << "Received chunk: (byte_len, csn, rid, fid): " << byte_len
              << ", " << csn << ", " << rid << ", " << fid;

  // Locate request by rid
  DCHECK(rid < kMaxReq);
  auto req = subflow->get_recvreq_by_id(rid);
  DCHECK(req->ureq);

  if (req->type != RecvRequest::RECV || req->ureq->context != flow) {
    LOG(ERROR) << "Can't find corresponding request or this request is "
                  "invalid for this chunk. Dropping. "
               << req->type;
    // This should never happen.
    CHECK(0);
    return;
  }

  // There is no need to check CSN as RC provides reliable delivery.
  auto* msg_size = &req->ureq->recv.elems[0].size;
  uint32_t* received_bytes = req->received_bytes;
  received_bytes[0] += byte_len;

  if (!last_chunk) {
    req = nullptr;
  }

  subflow->rxtracking.ready_csn_.insert({csn, req});

  try_update_csn(subflow);

  EventOnRxData(subflow, &imm_data);
}

template <typename T>
void RDMAContext::uc_rx_chunk(T* wc_or_cq_ex) {
  static constexpr bool is_wc = std::is_same_v<T, struct ibv_wc>;
  auto now = rdtsc();
  uint32_t byte_len, imm_value, qp_num;
  if constexpr (is_wc) {
    byte_len = wc_or_cq_ex->byte_len;
    imm_value = wc_or_cq_ex->imm_data;
    qp_num = wc_or_cq_ex->qp_num;
  } else {
    byte_len = ibv_wc_read_byte_len(wc_or_cq_ex);
    imm_value = ibv_wc_read_imm_data(wc_or_cq_ex);
    qp_num = ibv_wc_read_qp_num(wc_or_cq_ex);
  }
  auto imm_data = IMMData(ntohl(imm_value));
  auto qpidx = qpn2idx_[qp_num];

  auto last_chunk = imm_data.GetHINT();
  auto csn = imm_data.GetCSN();
  auto rid = imm_data.GetRID();
  auto fid = imm_data.GetFID();

  DCHECK(fid < MAX_FLOW);
  auto* flow = reinterpret_cast<UcclFlow*>(receiver_flow_tbl_[fid]);
  DCHECK(flow) << fid << ", RDMAContext ptr: " << this;
  auto* subflow = flow->sub_flows_[engine_offset_];

  UCCL_LOG_IO << "Received chunk: (byte_len, csn, rid, fid): " << byte_len
              << ", " << csn << ", " << rid << ", " << fid << " from QP#"
              << qpidx;

  // Locate request by rid
  DCHECK(rid < kMaxReq);
  auto req = subflow->get_recvreq_by_id(rid);
  if (req->type != RecvRequest::RECV || req->ureq->context != flow) {
    UCCL_LOG_IO << "Can't find corresponding request or this request is "
                   "invalid for this chunk. Dropping. ";
    subflow->pcb.stats_chunk_drop++;
    return;
  }

  // Compare CSN with the expected CSN.
  auto ecsn = subflow->pcb.rcv_nxt;
  auto distance = UINT_CSN(csn) - ecsn;

  if (UINT_CSN::uintcsn_seqno_lt(UINT_CSN(csn), ecsn)) {
    UCCL_LOG_IO << "Chunk lag behind. Dropping as we can't handle SACK. "
                << "csn: " << csn << ", ecsn: " << ecsn.to_uint32();
    subflow->pcb.stats_chunk_drop++;
    return;
  }

  if (distance.to_uint32() > kReassemblyMaxSeqnoDistance) {
    UCCL_LOG_IO << "Chunk too far ahead. Dropping as we can't handle SACK. "
                << "csn: " << csn << ", ecsn: " << ecsn.to_uint32();
    subflow->pcb.stats_chunk_drop++;
    return;
  }

  // It's impossible to receive duplicate OOO chunks in uc_rx_chunk().
  // This is because the rtx chunk is retransmitted after the original chunk
  // using the same QP. According to the nature of UC QP, if the original chunk
  // is not lost, the original chunk is always handled before its corresponding
  // rtx chunk. In addition, all rtx chunks are handled in uc_rx_rtx_chunk()
  // rather than uc_rx_chunk(). Therefore, there is no need for us to check
  // duplicate OOO chunks here. But uc_rx_rtx_chunk() needs to handle this case.
  DCHECK(!subflow->pcb.sack_bitmap_bit_is_set(distance.to_uint32()));

  // Always use the latest timestamp.
  if constexpr (kTestNoHWTimestamp || is_wc)
    subflow->pcb.t_remote_nic_rx = now;
  else
    subflow->pcb.t_remote_nic_rx = ibv_wc_read_completion_ts(wc_or_cq_ex);

  subflow->pcb.sack_bitmap_bit_set(distance.to_uint32());

  auto* msg_size = &req->ureq->recv.elems[0].size;
  uint32_t* received_bytes = req->received_bytes;
  received_bytes[0] += byte_len;

  if (!last_chunk) {
    req = nullptr;
  }

  subflow->rxtracking.ready_csn_.insert({csn, req});

  try_update_csn(subflow);

  if (distance.to_uint32()) {
    subflow->rxtracking.encounter_ooo();
#ifdef STATS
    subflow->pcb.stats_ooo++;
    subflow->pcb.stats_maxooo =
        std::max(subflow->pcb.stats_maxooo, distance.to_uint32());
    if (subflow->rxtracking.real_ooo()) subflow->pcb.stats_real_ooo++;
#endif
  }

  subflow->rxtracking.cumulate_wqe();
  subflow->rxtracking.cumulate_bytes(byte_len);

  if (list_empty(&subflow->ack.ack_link))
    list_add_tail(&subflow->ack.ack_link, &ack_list_);
  subflow->next_ack_path_ = qpidx;

  // Send ACK if needed.
  if (subflow->rxtracking.need_imm_ack()) {
    auto chunk_addr = io_ctx_->pop_ctrl_chunk();
    craft_ack(subflow, chunk_addr, 0);
    try_post_acks(1, chunk_addr, true);

    subflow->rxtracking.clear_imm_ack();
    list_del(&subflow->ack.ack_link);
  }

  EventOnRxData(subflow, &imm_data);
}

void RDMAContext::try_post_acks(int num_ack, uint64_t chunk_addr, bool force) {
  if (num_ack == 0) return;

  auto idx = io_ctx_->nr_tx_ack_wr_;

  io_ctx_->tx_ack_sge_[idx].addr = chunk_addr;
  io_ctx_->tx_ack_sge_[idx].length = CtrlChunkBuffPool::kPktSize * num_ack;
  io_ctx_->tx_ack_sge_[idx].lkey = io_ctx_->get_ctrl_chunk_lkey();

  CQEDesc* cqe_desc = io_ctx_->pop_cqe_desc();
  cqe_desc->data = chunk_addr;
  io_ctx_->tx_ack_wr_[idx].wr_id = (uint64_t)cqe_desc;

  io_ctx_->tx_ack_wr_[idx].imm_data = htonl(num_ack);

  io_ctx_->tx_ack_wr_[idx].wr.ud.ah = remote_ctx_.dest_ah;
  io_ctx_->tx_ack_wr_[idx].wr.ud.remote_qpn = remote_ctx_.remote_ctrl_qpn;
  io_ctx_->tx_ack_wr_[idx].wr.ud.remote_qkey = remote_ctx_.remote_ctrl_qpn;

  if (io_ctx_->tx_ack_sge_[idx].length <= kMaxInline) {
    io_ctx_->tx_ack_wr_[idx].send_flags |= IBV_SEND_INLINE;
  } else {
    io_ctx_->tx_ack_wr_[idx].send_flags &= ~IBV_SEND_INLINE;
  }

  io_ctx_->tx_ack_wr_[idx].next =
      (idx == kMaxAckWRs - 1) ? nullptr : &io_ctx_->tx_ack_wr_[idx + 1];

  io_ctx_->nr_tx_ack_wr_++;

  UCCL_LOG_IO << "Post " << num_ack << " ACKs";

  if (force || io_ctx_->nr_tx_ack_wr_ == kMaxAckWRs) io_ctx_->flush_acks();
}

void RDMAContext::craft_ack(SubUcclFlow* subflow, uint64_t chunk_addr,
                            int idx_in_chunk) {
  uint64_t pkt_addr = chunk_addr + CtrlChunkBuffPool::kPktSize * idx_in_chunk;
  auto* ucclsackh = reinterpret_cast<UcclSackHdr*>(pkt_addr);

  ucclsackh->peer_id = be16_t(remote_ctx_.remote_peer_id);
  ucclsackh->fid = be16_t(subflow->fid_);
  ucclsackh->ackno = be16_t(subflow->pcb.ackno().to_uint32());
  ucclsackh->path = be16_t(subflow->next_ack_path_);

  auto t4 = rdtsc();
  uint64_t t2;
  if constexpr (kTestNoHWTimestamp)
    t2 = subflow->pcb.t_remote_nic_rx;
  else
    t2 = convert_nic_to_host(subflow->pcb.t_remote_nic_rx);

  ucclsackh->remote_queueing = be64_t(to_usec(t4 - t2, freq_ghz));

  for (size_t i = 0; i < sizeof(UcclSackHdr::sack_bitmap) /
                             sizeof(UcclSackHdr::sack_bitmap[0]);
       ++i) {
    ucclsackh->sack_bitmap[i] = be64_t(subflow->pcb.sack_bitmap[i]);
  }
  ucclsackh->sack_bitmap_count = be16_t(subflow->pcb.sack_bitmap_count);

  UCCL_LOG_IO << "craft_ack ackno: " << subflow->pcb.ackno().to_uint32()
              << " for flow: " << subflow->fid_;
}

void RDMAContext::__retransmit_for_flow(void* context, bool rto) {
  SubUcclFlow* subflow = reinterpret_cast<SubUcclFlow*>(context);

  if (subflow->txtracking.empty()) {
    UCCL_LOG_IO << "No unacked chunk to retransmit for flow" << subflow->fid_;
    return;
  }

  if (subflow->pcb.rto_rexmits_consectutive >= kRTOAbortThreshold) {
    UCCL_LOG_IO << "RTO retransmission threshold reached." << subflow->fid_;
  }

  // Case#1: SACK bitmap at the sender side is empty. Retransmit the oldest
  // unacked chunk.
  auto sack_bitmap_count = subflow->pcb.tx_sack_bitmap_count;
  if (!sack_bitmap_count) {
    auto chunk = subflow->txtracking.get_oldest_unacked_chunk();
    auto wr_ex = chunk.wr_ex;
    try_retransmit_chunk(subflow, wr_ex);
    // Arm timer for Retransmission
    rearm_timer_for_flow(subflow);
    if (rto) {
      subflow->pcb.stats_rto_rexmits++;
      subflow->pcb.rto_rexmits_consectutive++;
    } else {
      subflow->pcb.stats_fast_rexmits++;
    }
    return;
  }

  // Case#2: Retransmit the unacked chunks according to the SACK bitmap.
  bool done = false;
  auto tx_sack_bitmap_base = UINT_CSN(subflow->pcb.tx_sack_bitmap_base);

  uint32_t index = 0;
  while (sack_bitmap_count && index < kSackBitmapSize &&
         !subflow->txtracking.empty()) {
    auto bucket_idx = index / PCB::kSackBitmapBucketSize;
    auto sack_bitmap = subflow->pcb.tx_sack_bitmap[bucket_idx];

    auto cursor = index % PCB::kSackBitmapBucketSize;

    if ((sack_bitmap & (1ULL << cursor)) == 0) {
      // We found a hole.
      auto seqno = tx_sack_bitmap_base + index;
      DCHECK(index < subflow->txtracking.track_size());
      auto chunk = subflow->txtracking.get_unacked_chunk_from_idx(index);
      if (seqno == chunk.csn) {
        auto wr_ex = chunk.wr_ex;
        if (try_retransmit_chunk(subflow, wr_ex)) {
          done = true;
        } else {
          // We can't retransmit the chunk due to lack of credits.
          // Quit the loop.
          index = kSackBitmapSize;
        }
      } else {
        // This bit is stale and its corresponding chunk is already
        // acked. Do nothing.
        UCCL_LOG_IO << "Stale SACK bit for seqno: " << seqno.to_uint32()
                    << ", chunk.csn: " << chunk.csn << ", tx_sack_bitmap_base: "
                    << tx_sack_bitmap_base.to_uint32();
      }
    } else {
      sack_bitmap_count--;
    }
    index++;
  }

  // Arm timer for Retransmission
  rearm_timer_for_flow(subflow);
  if (done) {
    if (rto) {
      subflow->pcb.stats_rto_rexmits++;
      subflow->pcb.rto_rexmits_consectutive++;
    } else {
      subflow->pcb.stats_fast_rexmits++;
    }
  }
}

uint32_t RDMAContext::select_qpidx_pot(uint32_t msize, void* subflow_context) {
  if (can_use_last_choice(msize)) return last_qp_choice_;

  auto* sublfow = reinterpret_cast<SubUcclFlow*>(subflow_context);
  auto q1 = select_qpidx_rand();
  auto q2 = select_qpidx_rand();

  // Return the QP with lower RTT.
  auto qpidx =
      sublfow->scoreboard_rtt_[q1] < sublfow->scoreboard_rtt_[q2] ? q1 : q2;
  last_qp_choice_ = qpidx;
  return qpidx;
}

// Try to arm a timer for the given flow. If the timer is already armed, do
// nothing.
void RDMAContext::arm_timer_for_flow(void* context) {
  auto* subflow = reinterpret_cast<SubUcclFlow*>(context);
  if (!subflow->rto_armed) {
    if constexpr (kConstRTO) {
      rto_->arm_timer({this, subflow});
    } else {
      rto_->arm_timer({this, subflow},
                      std::max(kRTORTT * subflow->pcb.timely_cc.get_avg_rtt(),
                               kMinRTOUsec));
    }
    subflow->rto_armed = true;
  }
}

// Try to rearm a timer for the given flow. If the timer is not armed, arm it.
// If the timer is already armed, rearm it.
void RDMAContext::rearm_timer_for_flow(void* context) {
  auto* subflow = reinterpret_cast<SubUcclFlow*>(context);
  if (subflow->rto_armed) {
    if constexpr (kConstRTO) {
      rto_->rearm_timer({this, subflow});
    } else {
      rto_->rearm_timer({this, subflow},
                        std::max(kRTORTT * subflow->pcb.timely_cc.get_avg_rtt(),
                                 kMinRTOUsec));
    }
  } else {
    arm_timer_for_flow(subflow);
  }
}

void RDMAContext::mark_flow_timeout(void* context) {
  auto* subflow = reinterpret_cast<SubUcclFlow*>(context);
  subflow->rto_armed = false;
}

void RDMAContext::disarm_timer_for_flow(void* context) {
  auto* subflow = reinterpret_cast<SubUcclFlow*>(context);
  if (subflow->rto_armed) {
    rto_->disarm_timer({this, subflow});
    subflow->rto_armed = false;
  }
}

std::string RDMAContext::to_string() {
  std::string s;
  s.clear();

  uint32_t stats_rto_rexmits = 0;
  uint32_t stats_fast_rexmits = 0;
  uint32_t stats_accept_retr = 0;

  uint32_t stats_chunk_drop = 0;
  uint32_t stats_retr_chunk_drop = 0;
  uint32_t stats_ooo = 0;
  uint32_t stats_real_ooo = 0;
  uint32_t stats_maxooo = 0;

  for (int fid = 0; fid < nr_flows_; fid++) {
    {
      auto* flow = reinterpret_cast<UcclFlow*>(receiver_flow_tbl_[fid]);
      if (flow) {
        auto* subflow = flow->sub_flows_[engine_offset_];
        stats_accept_retr += subflow->pcb.stats_accept_retr;

        stats_chunk_drop += subflow->pcb.stats_chunk_drop;
        stats_retr_chunk_drop += subflow->pcb.stats_retr_chunk_drop;
        stats_ooo += subflow->pcb.stats_ooo;
        stats_real_ooo += subflow->pcb.stats_real_ooo;
        stats_maxooo = std::max(stats_maxooo, subflow->pcb.stats_maxooo);
        subflow->pcb.stats_maxooo = 0;  // Inaccurate is fine.
      }
    }
    {
      auto* flow = reinterpret_cast<UcclFlow*>(sender_flow_tbl_[fid]);
      if (flow) {
        auto* subflow = flow->sub_flows_[engine_offset_];
        stats_rto_rexmits += subflow->pcb.stats_rto_rexmits;
        stats_fast_rexmits += subflow->pcb.stats_fast_rexmits;
      }
    }
  }

  s += "\tRTO retr:" + std::to_string(stats_rto_rexmits) +
       "/Fast retr:" + std::to_string(stats_fast_rexmits) +
       "/Eat retr:" + std::to_string(stats_accept_retr) +
       "/Chunk drop:" + std::to_string(stats_chunk_drop) +
       "/Retr drop:" + std::to_string(stats_retr_chunk_drop) +
       "/OOO: " + std::to_string(stats_ooo) +
       "/ROOO: " + std::to_string(stats_real_ooo) +
       "/MAXOOO: " + std::to_string(stats_maxooo);

  s += "\n";

  return s;
}
// Initialize the templates to avoid the linker error.
template void RDMAContext::uc_rx_chunk<struct ibv_wc>(struct ibv_wc* wc);
template void RDMAContext::uc_rx_chunk<struct ibv_cq_ex>(struct ibv_cq_ex* wc);
template void RDMAContext::uc_rx_ack<struct ibv_wc>(struct ibv_wc* wc,
                                                    UcclSackHdr* ucclsackh);
template void RDMAContext::uc_rx_ack<struct ibv_cq_ex>(struct ibv_cq_ex* wc,
                                                       UcclSackHdr* ucclsackh);
template void RDMAContext::uc_rx_rtx_chunk<struct ibv_wc>(struct ibv_wc* wc,
                                                          uint64_t chunk_addr);
template void RDMAContext::uc_rx_rtx_chunk<struct ibv_cq_ex>(
    struct ibv_cq_ex* wc, uint64_t chunk_addr);
template void RDMAContext::rc_rx_ack<struct ibv_wc>(struct ibv_wc* wc);
template void RDMAContext::rc_rx_ack<struct ibv_cq_ex>(struct ibv_cq_ex* wc);
template void RDMAContext::rc_rx_chunk<struct ibv_wc>(struct ibv_wc* wc);
template void RDMAContext::rc_rx_chunk<struct ibv_cq_ex>(struct ibv_cq_ex* wc);

}  // namespace uccl
