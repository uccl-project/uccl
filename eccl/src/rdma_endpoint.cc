#include "transport.h"
#include <arpa/inet.h>  // ntohl

RDMAEndpoint::RDMAEndpoint(std::shared_ptr<Config> config, Communicator* comm)
    : config_(config), comm_(comm) {}

RDMAEndpoint::~RDMAEndpoint() {
  int local_rank = comm_ ? comm_->local_rank_ : -1;
  {
    std::lock_guard<std::mutex> lk(qp_list_mu_);
    for (size_t i = 0; i < qp_list_.size(); i++) {
      ibv_qp* qp = qp_list_[i];
      if (qp) {
        if (ibv_destroy_qp(qp)) {
          std::cerr << "[WARN] Communicator " << local_rank
                    << " Failed to destroy QP[" << i << "]" << std::endl;
        } else {
          std::cout << "[INFO] Communicator " << local_rank << " QP[" << i
                    << "] destroyed" << std::endl;
        }
      }
    }
    qp_list_.clear();
  }
  {
    std::lock_guard<std::mutex> lk(qp_info_list_mu_);
    qp_info_list_.clear();
  }
  {
    std::lock_guard<std::mutex> lk(remote_qp_info_list_mu_);
    remote_qp_info_list_.clear();
  }

  std::cout << "[INFO] Communicator " << local_rank
            << " RDMAEndpoint resources released" << std::endl;
}

bool RDMAEndpoint::connect_to(int peer_rank) {
  int local_rank = comm_->local_rank_;

  // Create QPs
  struct ibv_qp_init_attr qp_init_attr = {};
  qp_init_attr.send_cq = comm_->cq_list_[peer_rank % comm_->cq_list_.size()];
  qp_init_attr.recv_cq = comm_->cq_list_[peer_rank % comm_->cq_list_.size()];
  qp_init_attr.qp_type = IBV_QPT_RC;  // Reliable Connection
  qp_init_attr.cap.max_send_wr = config_->qp_max_recv_wr;
  qp_init_attr.cap.max_recv_wr = config_->qp_max_recv_wr;
  qp_init_attr.cap.max_send_sge = config_->qp_max_sge;
  qp_init_attr.cap.max_recv_sge = config_->qp_max_sge;
  qp_init_attr.sq_sig_all = 0;

  for (int i = 0; i < config_->qp_count_per_ep; i++) {
    std::lock_guard<std::mutex> lock(qp_list_mu_);
    ibv_qp* new_qp = ibv_create_qp(comm_->pd_, &qp_init_attr);
    if (!new_qp) {
      std::cerr << "[ERROR] Communicator " << local_rank
                << " Failed to create QP" << std::endl;
      return false;
    }
    qp_list_.push_back(new_qp);
    QpInfo new_info{.qp_num = new_qp->qp_num,
                    .psn = (uint32_t)rand() & 0xffffff,
                    .lid = comm_->lid};
    if (comm_->support_rdma_roce) {
      memcpy(new_info.gid, comm_->gid, 16);
    }
    qp_info_list_.push_back(new_info);
  }

  // Modify QPs to INIT
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = 1;
  attr.pkey_index = 0;
  attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  int flags =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

  for (int i = 0; i < config_->qp_count_per_ep; i++) {
    if (ibv_modify_qp(qp_list_[i], &attr, flags)) {
      std::cerr << "[ERROR] Communicator " << local_rank
                << " Failed to modify QP to INIT: " << strerror(errno)
                << std::endl;
      return false;
    }
  }

  // Exchange QP info
  RDMAInfo local_info{};
  {
    std::lock_guard<std::mutex> lk(qp_info_list_mu_);
    local_info.qps = qp_info_list_;
  }
  std::string key =
      "qpinfo:" + std::to_string(local_rank) + "->" + std::to_string(peer_rank);
  if (!comm_->redis_client_->publish(key, local_info)) {
    std::cerr << "[ERROR] Communicator " << local_rank
              << " Failed to publish QP info for peer " << peer_rank
              << std::endl;
    return false;
  }

  RDMAInfo remote_info;
  std::string peer_key =
      "qpinfo:" + std::to_string(peer_rank) + "->" + std::to_string(local_rank);
  if (!comm_->redis_client_->wait_and_fetch(peer_key, remote_info, -1)) {
    std::cerr << "[ERROR] Communicator " << local_rank
              << " Timeout waiting QP info from peer " << peer_rank
              << std::endl;
    return false;
  }

  {
    std::lock_guard<std::mutex> lk(remote_qp_info_list_mu_);
    remote_qp_info_list_.clear();
    size_t n = std::min(qp_info_list_.size(), remote_info.qps.size());
    remote_qp_info_list_.resize(n);
    for (size_t i = 0; i < n; i++) {
      remote_qp_info_list_[i] = remote_info.qps[i];
    }
  }

  std::cout << "[INFO] Communicator " << local_rank
            << " QP info exchanged with rank " << peer_rank
            << ", local_qps=" << local_info.qps.size()
            << ", remote_qps=" << remote_info.qps.size() << std::endl;

  // Modify QPs to RTR
  for (int i = 0; i < config_->qp_count_per_ep; i++) {
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = comm_->active_mtu;
    attr.dest_qp_num = remote_qp_info_list_[i].qp_num;
    attr.rq_psn = remote_qp_info_list_[i].psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    if (comm_->support_rdma_roce) {
      attr.ah_attr.is_global = 1;
      attr.ah_attr.port_num = 1;
      attr.ah_attr.sl = 0;
      attr.ah_attr.src_path_bits = 0;
      attr.ah_attr.grh.hop_limit = 1;
      memcpy(&attr.ah_attr.grh.dgid, remote_qp_info_list_[i].gid, 16);
      attr.ah_attr.grh.sgid_index = 1;
    } else {
      attr.ah_attr.is_global = 0;
      attr.ah_attr.dlid = remote_qp_info_list_[i].lid;
      attr.ah_attr.port_num = 1;
      attr.ah_attr.sl = 0;
      attr.ah_attr.src_path_bits = 0;
      attr.ah_attr.static_rate = 0;
      memset(&attr.ah_attr.grh, 0, sizeof(attr.ah_attr.grh));
    }
    int rtr_flags = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV |
                    IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                    IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    if (ibv_modify_qp(qp_list_[i], &attr, rtr_flags)) {
      std::cerr << "[ERROR] Communicator " << local_rank
                << " Failed to modify QP to RTR: " << strerror(errno)
                << std::endl;
      return false;
    }
    std::cout << "[INFO] Communicator " << local_rank << " QP[" << i
              << "] modified to RTR state" << std::endl;
  }

  // Modify QPs to RTS
  for (int i = 0; i < config_->qp_count_per_ep; i++) {
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = qp_info_list_[i].psn;
    attr.max_rd_atomic = 1;

    int rts_flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                    IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
    if (ibv_modify_qp(qp_list_[i], &attr, rts_flags)) {
      std::cerr << "[ERROR] Communicator " << local_rank
                << " Failed to modify QP to RTS: " << strerror(errno)
                << std::endl;
      return false;
    }
    std::cout << "[INFO] Communicator " << local_rank << " QP[" << i
              << "] modified to RTS state" << std::endl;
  }

  // Pre post some recv wrs
  for (int i = 0; i < config_->qp_count_per_ep; i++) {
    post_recv_imm_(qp_list_[i], 16);  // keep same as cq poll batch
  }

  return true;
}

bool RDMAEndpoint::accept_from(int rank) { return connect_to(rank); }

bool RDMAEndpoint::send_async(int to_rank, std::shared_ptr<Request> creq) {
  {
    std::lock_guard<std::mutex> lk(qp_list_mu_);
    if (qp_list_.empty()) return false;
  }

  size_t total = creq->len;
  if (total == 0) return false;

  // Use qp0 send for single qp transfer test
  ibv_qp* qp0 = nullptr;
  {
    std::lock_guard<std::mutex> lk(qp_list_mu_);
    qp0 = qp_list_.empty() ? nullptr : qp_list_[0];
  }
  if (!qp0) return false;

  creq->pending_signaled.store(1, std::memory_order_relaxed);
  creq->running.store(true, std::memory_order_release);

  MR remote_mr = comm_->get_remote_mr(to_rank, creq->remote_mr_id);
  MR local_mr = comm_->get_local_mr(creq->local_mr_id);

  ibv_sge sge;
  sge.addr = reinterpret_cast<uint64_t>(creq->buf) + creq->offset;
  sge.length = static_cast<uint32_t>(creq->len);
  sge.lkey = local_mr.key;

  ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = static_cast<uint64_t>(
      static_cast<uint32_t>(creq->id));  // wr_id : which creq
  // std::cout << "wr id is " << creq->id << std::endl;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr =
      reinterpret_cast<uint64_t>(remote_mr.address) + creq->offset;
  wr.wr.rdma.rkey = remote_mr.key;
  wr.imm_data = htonl(static_cast<uint32_t>(creq->id));

  ibv_send_wr* bad_wr = nullptr;
  int ret = ibv_post_send(qp0, &wr, &bad_wr);
  if (ret) {
    perror("ibv_post_send failed");
    creq->pending_signaled.store(0, std::memory_order_relaxed);
    creq->running.store(false, std::memory_order_release);
    return false;
  }

  return true;
}

// TODO: multi qps transfer
// bool RDMAEndpoint::send_async(int to_rank, std::shared_ptr<Request> creq) {
//   {
//     std::lock_guard<std::mutex> lk(qp_list_mu_);
//     if (qp_list_.empty()) return false;
//   }
//   const size_t chunk_size = config_->rdma_chunk_size;
//   if (chunk_size == 0) return false;

//   size_t total = creq->len;
//   if (total == 0) return false;

//   // calcu all chunk
//   size_t chunk_count = (total + chunk_size - 1) / chunk_size;
//   size_t qp_count;
//   {
//     std::lock_guard<std::mutex> lk(qp_list_mu_);
//     qp_count = qp_list_.size();
//     if (qp_count == 0) return false;
//   }

//   // assign chunk to qp
//   struct Chunk {
//     size_t chunk_idx;
//     size_t off;
//     size_t len;
//   };
//   std::vector<std::vector<Chunk>> assign(qp_count);
//   size_t chunk_per_qp = (chunk_count + qp_count - 1) / qp_count;  // ceil
//   for (size_t ci = 0; ci < chunk_count; ++ci) {
//     size_t qp_idx = ci / chunk_per_qp;
//     if (qp_idx >= qp_count) qp_idx = qp_count - 1;
//     size_t off = creq->offset + ci * chunk_size;
//     size_t len = std::min(chunk_size, creq->len - ci * chunk_size);
//     assign[qp_idx].push_back({ci, off, len});
//   }

//   int used_qps = 0;
//   for (size_t i = 0; i < qp_count; ++i)
//     if (!assign[i].empty()) ++used_qps;
//   creq->pending_signaled.store(used_qps, std::memory_order_relaxed);
//   creq->pending_signaled.store(used_qps, std::memory_order_relaxed);
//   creq->running.store(true, std::memory_order_release);

//   // check mr
//   MR remote_mr = comm_->get_remote_mr(to_rank, creq->remote_mr_id);
//   MR local_mr = comm_->get_local_mr(creq->local_mr_id);

//   for (size_t qp_idx = 0; qp_idx < qp_count; ++qp_idx) {
//     if (assign[qp_idx].empty()) continue;

//     ibv_qp* qp = nullptr;
//     {
//       std::lock_guard<std::mutex> lk(qp_list_mu_);
//       qp = qp_list_[qp_idx];
//     }
//     if (!qp) {
//       creq->pending_signaled.store(0, std::memory_order_relaxed);
//       return false;
//     }

//     auto& chunks = assign[qp_idx];
//     std::vector<ibv_send_wr> wrs(chunks.size());
//     std::vector<ibv_sge> sges(chunks.size());

//     for (size_t j = 0; j < chunks.size(); ++j) {
//       Chunk const& c = chunks[j];

//       // sge
//       sges[j].addr = reinterpret_cast<uint64_t>(creq->buf) + c.off;
//       sges[j].length = static_cast<uint32_t>(c.len);
//       sges[j].lkey = local_mr.key;

//       // wr
//       memset(&wrs[j], 0, sizeof(ibv_send_wr));
//       wrs[j].wr_id = static_cast<uint64_t>(creq->id);
//       wrs[j].sg_list = &sges[j];
//       wrs[j].num_sge = 1;
//       wrs[j].wr.rdma.remote_addr =
//           reinterpret_cast<uint64_t>(remote_mr.address) + c.off;
//       wrs[j].wr.rdma.rkey = remote_mr.key;

//       if (j + 1 == chunks.size()) {
//         wrs[j].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
//         wrs[j].send_flags = IBV_SEND_SIGNALED;
//         wrs[j].imm_data = htonl(static_cast<uint32_t>(creq->id));
//       } else {
//         wrs[j].opcode = IBV_WR_RDMA_WRITE;
//         wrs[j].send_flags = 0;
//       }

//       wrs[j].next = (j + 1 < chunks.size()) ? &wrs[j + 1] : nullptr;
//     }

//     ibv_send_wr* bad_wr = nullptr;
//     int ret = ibv_post_send(qp, &wrs[0], &bad_wr);
//     if (ret) {
//       perror("ibv_post_send failed");
//       creq->pending_signaled.store(0, std::memory_order_relaxed);
//       return false;
//     }
//   }

//   return true;
// }

bool RDMAEndpoint::post_recv_imm_(ibv_qp* qp, uint64_t count) {
  for (uint64_t i = 0; i < count; ++i) {
    ibv_recv_wr rr;
    memset(&rr, 0, sizeof(rr));

    rr.wr_id = reinterpret_cast<uint64_t>(
        qp);  // wr_id : which qp; recv get creq->id from imm data
    rr.num_sge = 0;
    rr.sg_list = nullptr;

    ibv_recv_wr* bad_rr = nullptr;
    int ret = ibv_post_recv(qp, &rr, &bad_rr);
    if (ret) {
      std::cerr << "[RDMAEndpoint] ibv_post_recv failed at " << i << ": "
                << strerror(errno) << std::endl;
      return false;
    }
  }
  return true;
}

bool RDMAEndpoint::recv_async(int from_rank, std::shared_ptr<Request> creq) {
  {
    std::lock_guard<std::mutex> lk(qp_list_mu_);
    if (qp_list_.empty()) return false;
  }

  size_t total = creq->len;
  if (total == 0) return false;

  // Use qp0 recv for single qp tansfer test
  ibv_qp* qp0 = nullptr;
  {
    std::lock_guard<std::mutex> lk(qp_list_mu_);
    if (qp_list_.empty()) return false;
    qp0 = qp_list_[0];
  }
  if (!qp0) return false;

  creq->pending_signaled.store(1, std::memory_order_relaxed);
  creq->running.store(true, std::memory_order_release);

  auto ok = post_recv_imm_(qp0, 1);
  if (!ok) {
    creq->pending_signaled.store(0, std::memory_order_relaxed);
    creq->running.store(false, std::memory_order_release);
    return false;
  }

  return true;
}

// TODO: multi qps transfer
// bool RDMAEndpoint::recv_async(int from_rank, std::shared_ptr<Request> creq) {
//   const size_t chunk_size = config_->rdma_chunk_size;
//   if (chunk_size == 0) return false;

//   size_t total = creq->len;
//   if (total == 0) return false;

//   // calc all chunk
//   size_t chunk_count = (total + chunk_size - 1) / chunk_size;
//   size_t qp_count;
//   {
//     std::lock_guard<std::mutex> lk(qp_list_mu_);
//     qp_count = qp_list_.size();
//     if (qp_count == 0) return false;
//   }

//   // assign chunk to qp (block distribution, same as sender)
//   struct Chunk {
//     size_t chunk_idx;
//     size_t off;
//     size_t len;
//   };
//   std::vector<std::vector<Chunk>> assign(qp_count);
//   size_t chunk_per_qp = (chunk_count + qp_count - 1) / qp_count;  // ceil
//   for (size_t ci = 0; ci < chunk_count; ++ci) {
//     size_t qp_idx = ci / chunk_per_qp;
//     if (qp_idx >= qp_count) qp_idx = qp_count - 1;
//     size_t off = creq->offset + ci * chunk_size;
//     size_t len = std::min(chunk_size, creq->len - ci * chunk_size);
//     assign[qp_idx].push_back({ci, off, len});
//   }

//   // how many QPs will actually produce an IMM (used_qps)
//   int used_qps = 0;
//   for (size_t i = 0; i < qp_count; ++i)
//     if (!assign[i].empty()) ++used_qps;
//   if (used_qps == 0) {
//     // nothing to receive
//     creq->finished.store(true, std::memory_order_release);
//     return true;
//   }

//   // record that we expect `used_qps` IMM notifications for this request
//   // reuse pending_signaled as pending IMM count on receiver
//   creq->pending_signaled.store(used_qps, std::memory_order_relaxed);
//   creq->running.store(true, std::memory_order_release);

//   // For each qp that has assigned chunks, post a single zero-byte recv WR
//   for (size_t qp_idx = 0; qp_idx < qp_count; ++qp_idx) {
//     if (assign[qp_idx].empty()) continue;

//     ibv_qp* qp = nullptr;
//     {
//       std::lock_guard<std::mutex> lk(qp_list_mu_);
//       qp = qp_list_[qp_idx];
//     }
//     if (!qp) {
//       // failure: clear expectation and return false
//       creq->pending_signaled.store(0, std::memory_order_relaxed);
//       return false;
//     }

//     // Prepare a zero-byte receive WR to capture the incoming IMM.
//     // For RDMA_WRITE_WITH_IMM the recv WR can be 0-byte; the completion
//     // will be IBV_WC_RECV_RDMA_WITH_IMM containing wc.imm_data.
//     ibv_recv_wr rr;
//     memset(&rr, 0, sizeof(rr));
//     rr.wr_id = static_cast<uint64_t>(creq->id);  // optional: helps mapping
//     rr.next = nullptr;
//     rr.num_sge = 0;  // zero-byte receive (no payload buffer)
//     rr.sg_list = nullptr;

//     ibv_recv_wr* bad_rr = nullptr;
//     int ret = ibv_post_recv(qp, &rr, &bad_rr);
//     if (ret) {
//       perror("ibv_post_recv failed");
//       creq->pending_signaled.store(0, std::memory_order_relaxed);
//       return false;
//     }
//   }  // per-qp
//   return true;
// }
