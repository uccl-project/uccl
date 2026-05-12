#include "rdma_adapter.h"
#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>

namespace UKernel {
namespace Transport {

namespace {

constexpr auto kPollSleep = std::chrono::microseconds(10);
constexpr auto kQpMaxSendWr = 1024;
constexpr int kQpTimeout = 14;
constexpr int kQpRetryCnt = 7;
constexpr int kQpRnrRetry = 7;
constexpr double kEwmaAlpha = 0.125;

int detect_gid(ibv_context* ctx, ibv_port_attr const* port_attr,
               union ibv_gid* out_gid) {
  char const* dev = ibv_get_device_name(ctx->device);
  union ibv_gid gid;
  for (int i = 0; i < port_attr->gid_tbl_len; ++i) {
    if (ibv_query_gid(ctx, 1, i, &gid) != 0) continue;
    if (gid.global.subnet_prefix == 0 && gid.global.interface_id == 0) continue;

    char path[512];
    snprintf(path, sizeof(path),
             "/sys/class/infiniband/%s/ports/1/gid_attrs/types/%d", dev, i);
    FILE* fp = fopen(path, "r");
    if (!fp) continue;
    char buf[64] = {};
    bool roce = false;
    if (fgets(buf, sizeof(buf), fp)) {
      roce = (strstr(buf, "RoCE v2") != nullptr);
    }
    fclose(fp);
    if (!roce) continue;

    bool v4 = (gid.raw[0] == 0 && gid.raw[1] == 0 && gid.raw[2] == 0 &&
               gid.raw[3] == 0 && gid.raw[4] == 0 && gid.raw[5] == 0 &&
               gid.raw[6] == 0 && gid.raw[7] == 0 && gid.raw[8] == 0 &&
               gid.raw[9] == 0 && gid.raw[10] == 0xFF && gid.raw[11] == 0xFF);
    if (v4) {
      *out_gid = gid;
      return i;
    }
  }
  for (int i = 0; i < port_attr->gid_tbl_len; ++i) {
    if (ibv_query_gid(ctx, 1, i, &gid) != 0) continue;
    if (gid.global.subnet_prefix != 0 || gid.global.interface_id != 0) {
      *out_gid = gid;
      return i;
    }
  }
  memset(out_gid, 0, sizeof(*out_gid));
  return 0;
}

int pick_dev() {
  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  if (!list || n == 0) return -1;
  ibv_free_device_list(list);
  return 0;
}

}  // namespace

uint64_t RdmaTransportAdapter::now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// ── constructor / destructor ────────────────────────────────────────────────

RdmaTransportAdapter::RdmaTransportAdapter(int local_gpu_idx,
                                           RdmaTransportConfig config)
    : local_gpu_idx_(local_gpu_idx), config_(config) {
  if (config_.num_qps < 1 || config_.num_qps > kMaxQPs) config_.num_qps = 4;

  std::srand(static_cast<unsigned>(now_ns() & 0xFFFFFFFFU));

  local_dev_idx_ = pick_dev();
  if (local_dev_idx_ < 0) {
    std::cerr << "[ERROR] RdmaAdapter: no RDMA device for GPU "
              << local_gpu_idx_ << std::endl;
    return;
  }

  int ndev = 0;
  ibv_device** devs = ibv_get_device_list(&ndev);
  if (!devs || local_dev_idx_ >= ndev) {
    if (devs) ibv_free_device_list(devs);
    return;
  }
  ibv_device* dev = devs[local_dev_idx_];

  ibv_context* raw = ibv_open_device(dev);
  ibv_free_device_list(devs);
  if (!raw) return;

  ctx_handle_.reset(raw, [](ibv_context* c) {
    if (c) ibv_close_device(c);
  });
  ctx_ = raw;

  if (ibv_query_device(ctx_, &dev_attr_) != 0) {
    std::cerr << "[ERROR] RdmaAdapter: ibv_query_device failed" << std::endl;
    ctx_handle_.reset();
    ctx_ = nullptr;
    return;
  }

  pd_ = ibv_alloc_pd(ctx_);
  if (!pd_) {
    ctx_handle_.reset();
    ctx_ = nullptr;
    return;
  }

  ibv_port_attr pattr = {};
  if (ibv_query_port(ctx_, 1, &pattr) != 0) {
    ibv_dealloc_pd(pd_);
    pd_ = nullptr;
    ctx_handle_.reset();
    ctx_ = nullptr;
    return;
  }
  lid_ = pattr.lid;
  gid_index_ = detect_gid(ctx_, &pattr, &gid_);

  auto* buf = new uint8_t[kSignalBufBytes]();
  signal_mr_ = ibv_reg_mr(pd_, buf, kSignalBufBytes,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_READ);
  if (!signal_mr_) {
    delete[] buf;
    ibv_dealloc_pd(pd_);
    pd_ = nullptr;
    ctx_handle_.reset();
    ctx_ = nullptr;
    return;
  }
  signal_addr_ = reinterpret_cast<uint64_t>(buf);
  signal_rkey_ = signal_mr_->rkey;

  slots_ = std::make_unique<RequestSlot[]>(kSlotCount);

  stop_.store(false, std::memory_order_release);
  poll_thread_ = std::thread([this] { poll_loop(); });

  std::cout << "[INFO] RdmaAdapter: dev=" << ibv_get_device_name(ctx_->device)
            << " lid=" << lid_ << " gid_idx=" << (int)gid_index_ << std::endl;
}

RdmaTransportAdapter::~RdmaTransportAdapter() {
  stop_.store(true, std::memory_order_release);
  cv_.notify_all();
  if (poll_thread_.joinable()) poll_thread_.join();

  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [rank, p] : peers_) destroy_peer_qps(*p);
    peers_.clear();
    for (auto& [id, mr] : mr_map_) ibv_dereg_mr(mr);
    mr_map_.clear();
  }

  if (signal_mr_) {
    delete[] reinterpret_cast<uint8_t*>(signal_addr_);
    ibv_dereg_mr(signal_mr_);
    signal_mr_ = nullptr;
  }
  if (pd_) {
    ibv_dealloc_pd(pd_);
    pd_ = nullptr;
  }
}

// ── QP helpers ──────────────────────────────────────────────────────────────

bool RdmaTransportAdapter::create_qp_set(ibv_qp** qps, ibv_cq** cq, int count,
                                          int cq_size) {
  *cq = ibv_create_cq(ctx_, cq_size, nullptr, nullptr, 0);
  if (!*cq) return false;

  ibv_qp_init_attr attr = {};
  attr.send_cq = *cq;
  attr.recv_cq = *cq;
  attr.qp_type = IBV_QPT_RC;
  attr.cap.max_send_wr = kQpMaxSendWr;
  attr.cap.max_recv_wr = config_.recv_wr_pool_per_qp;
  attr.cap.max_send_sge = 1;
  attr.cap.max_recv_sge = 4;
  attr.cap.max_inline_data = 0;

  for (int i = 0; i < count; ++i) {
    qps[i] = ibv_create_qp(pd_, &attr);
    if (!qps[i]) {
      for (int j = 0; j < i; ++j) {
        ibv_destroy_qp(qps[j]);
        qps[j] = nullptr;
      }
      ibv_destroy_cq(*cq);
      *cq = nullptr;
      return false;
    }
  }
  return true;
}

bool RdmaTransportAdapter::qps_to_init(ibv_qp** qps, int count) {
  for (int i = 0; i < count; ++i) {
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(qps[i], &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS) != 0) {
      return false;
    }
  }
  return true;
}

bool RdmaTransportAdapter::qps_to_rtr(ibv_qp** qps, int count,
                                      RdmaPeerConnectSpec const& remote,
                                      bool is_put) {
  for (int i = 0; i < count; ++i) {
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num =
        is_put ? remote.remote_recv_qpns[i] : remote.remote_send_qpns[i];
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 16;
    attr.min_rnr_timer = 12;

    if (remote.remote_lid != 0) {
      attr.ah_attr.is_global = 0;
      attr.ah_attr.dlid = remote.remote_lid;
      attr.ah_attr.sl = 0;
      attr.ah_attr.src_path_bits = 0;
      attr.ah_attr.port_num = 1;
    } else {
      attr.ah_attr.is_global = 1;
      attr.ah_attr.port_num = 1;
      memcpy(&attr.ah_attr.grh.dgid, remote.remote_gid_raw.data(),
             sizeof(attr.ah_attr.grh.dgid));
      attr.ah_attr.grh.sgid_index = gid_index_;
      attr.ah_attr.grh.hop_limit = 64;
    }

    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

    if (ibv_modify_qp(qps[i], &attr, flags) != 0) return false;
  }
  return true;
}

bool RdmaTransportAdapter::qps_to_rts(ibv_qp** qps, int count) {
  for (int i = 0; i < count; ++i) {
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.timeout = kQpTimeout;
    attr.retry_cnt = kQpRetryCnt;
    attr.rnr_retry = kQpRnrRetry;
    attr.max_rd_atomic = 16;

    int flags = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;

    if (ibv_modify_qp(qps[i], &attr, flags) != 0) return false;
  }
  return true;
}

int RdmaTransportAdapter::find_qp_idx(ibv_qp* const* qps, int count,
                                       uint32_t qp_num) {
  for (int i = 0; i < count; ++i) {
    if (qps[i] && qps[i]->qp_num == qp_num) return i;
  }
  return -1;
}

bool RdmaTransportAdapter::create_signal_qp(RdmaPeer& p) {
  p.signal_cq = ibv_create_cq(ctx_, 256, nullptr, nullptr, 0);
  if (!p.signal_cq) return false;

  ibv_qp_init_attr attr = {};
  attr.send_cq = p.signal_cq;
  attr.recv_cq = p.signal_cq;
  attr.qp_type = IBV_QPT_RC;
  attr.cap.max_send_wr = 64;
  attr.cap.max_recv_wr = 16;
  attr.cap.max_send_sge = 1;
  attr.cap.max_recv_sge = 1;
  attr.cap.max_inline_data = 16;

  p.signal_qp = ibv_create_qp(pd_, &attr);
  if (!p.signal_qp) {
    ibv_destroy_cq(p.signal_cq);
    p.signal_cq = nullptr;
    return false;
  }

  ibv_qp_attr qp_attr = {};
  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = 1;
  qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  if (ibv_modify_qp(p.signal_qp, &qp_attr,
                    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                        IBV_QP_ACCESS_FLAGS) != 0) {
    ibv_destroy_qp(p.signal_qp);
    p.signal_qp = nullptr;
    ibv_destroy_cq(p.signal_cq);
    p.signal_cq = nullptr;
    return false;
  }
  return true;
}

bool RdmaTransportAdapter::connect_signal_qp(RdmaPeer& p,
                                             uint32_t remote_qpn) {
  if (!p.signal_qp) return false;

  ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_4096;
  attr.dest_qp_num = remote_qpn;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = 16;
  attr.min_rnr_timer = 12;

  if (p.remote_lid != 0) {
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = p.remote_lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = 1;
  } else {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = 1;
    memcpy(&attr.ah_attr.grh.dgid, &p.remote_gid,
           sizeof(attr.ah_attr.grh.dgid));
    attr.ah_attr.grh.sgid_index = gid_index_;
    attr.ah_attr.grh.hop_limit = 64;
  }

  int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
              IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
              IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

  if (ibv_modify_qp(p.signal_qp, &attr, flags) != 0) return false;

  ibv_qp_attr rts_attr = {};
  rts_attr.qp_state = IBV_QPS_RTS;
  rts_attr.sq_psn = 0;
  rts_attr.timeout = kQpTimeout;
  rts_attr.retry_cnt = kQpRetryCnt;
  rts_attr.rnr_retry = kQpRnrRetry;
  rts_attr.max_rd_atomic = 16;

  int rts_flags = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                  IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;

  if (ibv_modify_qp(p.signal_qp, &rts_attr, rts_flags) != 0) return false;

  return true;
}

bool RdmaTransportAdapter::init_signal_pool(RdmaPeer& p) {
  static constexpr int kSignalPoolSz = 16;
  auto pool = std::make_unique<RecvPool>();
  pool->buffer.resize(static_cast<size_t>(kSignalPoolSz) * 32);
  pool->mr = ibv_reg_mr(pd_, pool->buffer.data(), pool->buffer.size(),
                        IBV_ACCESS_LOCAL_WRITE);
  if (!pool->mr) return false;

  pool->sges.resize(kSignalPoolSz);
  pool->wrs.resize(kSignalPoolSz);
  for (int i = 0; i < kSignalPoolSz; ++i) {
    pool->sges[i].addr = reinterpret_cast<uint64_t>(pool->buffer.data() +
                                                    static_cast<size_t>(i) * 32);
    pool->sges[i].length = 32;
    pool->sges[i].lkey = pool->mr->lkey;

    pool->wrs[i].wr_id = 0;
    pool->wrs[i].sg_list = &pool->sges[i];
    pool->wrs[i].num_sge = 1;
    pool->wrs[i].next = (i + 1 < kSignalPoolSz) ? &pool->wrs[i + 1] : nullptr;
  }

  ibv_recv_wr* bad = nullptr;
  if (ibv_post_recv(p.signal_qp, &pool->wrs[0], &bad) != 0) {
    return false;
  }
  p.signal_pool = std::move(pool);
  p.signal_post_idx = 0;
  std::cerr << "[init_signal_pool] posted " << kSignalPoolSz
            << " recv WRs on signal_qp" << std::endl;
  return true;
}

void RdmaTransportAdapter::destroy_signal_qp(RdmaPeer& p) {
  if (p.signal_pool && p.signal_pool->mr) {
    ibv_dereg_mr(p.signal_pool->mr);
  }
  p.signal_pool.reset();
  if (p.signal_qp) {
    ibv_destroy_qp(p.signal_qp);
    p.signal_qp = nullptr;
  }
  if (p.signal_cq) {
    ibv_destroy_cq(p.signal_cq);
    p.signal_cq = nullptr;
  }
}

bool RdmaTransportAdapter::repost_signal_recv(RdmaPeer& p) {
  if (!p.signal_pool) return false;
  int idx = p.signal_post_idx;
  static constexpr int kSignalPoolSz = 16;
  p.signal_pool->wrs[idx].next = nullptr;
  ibv_recv_wr* bad = nullptr;
  if (ibv_post_recv(p.signal_qp, &p.signal_pool->wrs[idx], &bad) != 0)
    return false;
  p.signal_post_idx = (idx + 1) % kSignalPoolSz;
  return true;
}

// ── peer init / destroy ─────────────────────────────────────────────────────

bool RdmaTransportAdapter::init_peer_qps(RdmaPeer& p) {
  int cq_size = config_.recv_wr_pool_per_qp * p.num_qps * 2 + kQpMaxSendWr;
  return create_qp_set(p.send_qps, &p.send_cq, p.num_qps, cq_size) &&
         create_qp_set(p.recv_qps, &p.recv_cq, p.num_qps, cq_size) &&
         qps_to_init(p.send_qps, p.num_qps) &&
         qps_to_init(p.recv_qps, p.num_qps) &&
         create_signal_qp(p);
}

void RdmaTransportAdapter::destroy_peer_qps(RdmaPeer& p) {
  for (int q = 0; q < p.num_qps; ++q) {
    if (p.send_qps[q]) {
      ibv_destroy_qp(p.send_qps[q]);
      p.send_qps[q] = nullptr;
    }
    if (p.recv_qps[q]) {
      ibv_destroy_qp(p.recv_qps[q]);
      p.recv_qps[q] = nullptr;
    }
    if (p.recv_pools[q] && p.recv_pools[q]->mr) {
      ibv_dereg_mr(p.recv_pools[q]->mr);
    }
    p.recv_pools[q].reset();
  }
  if (p.send_cq) {
    ibv_destroy_cq(p.send_cq);
    p.send_cq = nullptr;
  }
  if (p.recv_cq) {
    ibv_destroy_cq(p.recv_cq);
    p.recv_cq = nullptr;
  }
  destroy_signal_qp(p);
}

// ── recv pools ───────────────────────────────────────────────────────────────

bool RdmaTransportAdapter::init_recv_pools(RdmaPeer& p) {
  int n = config_.recv_wr_pool_per_qp;
  for (int q = 0; q < p.num_qps; ++q) {
    auto pool = std::make_unique<RecvPool>();
    pool->buffer.resize(static_cast<size_t>(n) * 32);
    pool->mr = ibv_reg_mr(pd_, pool->buffer.data(), pool->buffer.size(),
                          IBV_ACCESS_LOCAL_WRITE);
    if (!pool->mr) return false;

    pool->sges.resize(n);
    pool->wrs.resize(n);
    for (int i = 0; i < n; ++i) {
      pool->sges[i].addr = reinterpret_cast<uint64_t>(pool->buffer.data() +
                                                      static_cast<size_t>(i) *
                                                          32);
      pool->sges[i].length = 32;
      pool->sges[i].lkey = pool->mr->lkey;

      pool->wrs[i].wr_id = 0;
      pool->wrs[i].sg_list = &pool->sges[i];
      pool->wrs[i].num_sge = 1;
      pool->wrs[i].next = (i + 1 < n) ? &pool->wrs[i + 1] : nullptr;
    }

    ibv_recv_wr* bad = nullptr;
    if (ibv_post_recv(p.recv_qps[q], &pool->wrs[0], &bad) != 0) {
      return false;
    }
    p.recv_pools[q] = std::move(pool);
    p.recv_post_idx[q] = 0;
  }
  return true;
}

bool RdmaTransportAdapter::repost_one_recv(RdmaPeer& p, int qp_idx) {
  auto& pool = p.recv_pools[qp_idx];
  if (!pool) return false;
  int idx = p.recv_post_idx[qp_idx];
  int n = config_.recv_wr_pool_per_qp;
  pool->wrs[idx].next = nullptr;
  ibv_recv_wr* bad = nullptr;
  if (ibv_post_recv(p.recv_qps[qp_idx], &pool->wrs[idx], &bad) != 0)
    return false;
  p.recv_post_idx[qp_idx] = (idx + 1) % n;
  return true;
}

// ── connection management ───────────────────────────────────────────────────

RdmaConnectInit RdmaTransportAdapter::get_connect_init(int peer_rank) {
  RdmaConnectInit init;
  init.num_qps = static_cast<uint8_t>(config_.num_qps);
  init.lid = lid_;
  memcpy(init.gid_raw, gid_.raw, sizeof(init.gid_raw));
  init.gid_index = gid_index_;
  init.signal_addr = signal_addr_;
  init.signal_rkey = signal_rkey_;
  init.dev_idx = local_dev_idx_;
  init.gpu_idx = local_gpu_idx_;

  if (peer_rank >= 0) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peers_.find(peer_rank);
    if (it == peers_.end()) {
      auto p = std::make_unique<RdmaPeer>();
      p->num_qps = static_cast<uint8_t>(config_.num_qps);
      if (init_peer_qps(*p)) {
        p->qps_in_init = true;
        for (int i = 0; i < p->num_qps; ++i) {
          init.send_qpns[i] = p->send_qps[i]->qp_num;
          init.recv_qpns[i] = p->recv_qps[i]->qp_num;
        }
        init.signal_qpn = p->signal_qp ? p->signal_qp->qp_num : 0;
        peers_[peer_rank] = std::move(p);
      }
    } else if (it->second->qps_in_init) {
      for (int i = 0; i < it->second->num_qps; ++i) {
        init.send_qpns[i] = it->second->send_qps[i]->qp_num;
        init.recv_qpns[i] = it->second->recv_qps[i]->qp_num;
      }
      init.signal_qpn = it->second->signal_qp ? it->second->signal_qp->qp_num : 0;
    }
  }
  return init;
}

bool RdmaTransportAdapter::ensure_put_path(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_put_path(spec.peer_rank)) return true;
  if (spec.type != PeerConnectType::Connect) return false;
  if (!std::holds_alternative<RdmaPeerConnectSpec>(spec.detail)) return false;
  return do_connect(spec.peer_rank,
                    std::get<RdmaPeerConnectSpec>(spec.detail));
}

bool RdmaTransportAdapter::ensure_wait_path(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_wait_path(spec.peer_rank)) return true;
  if (spec.type != PeerConnectType::Accept) return false;
  if (!std::holds_alternative<RdmaPeerConnectSpec>(spec.detail)) return false;
  return do_accept(spec.peer_rank,
                   std::get<RdmaPeerConnectSpec>(spec.detail));
}

bool RdmaTransportAdapter::has_put_path(int rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peers_.find(rank);
  return it != peers_.end() && it->second->put_ready;
}

bool RdmaTransportAdapter::has_wait_path(int rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peers_.find(rank);
  return it != peers_.end() && it->second->wait_ready;
}

bool RdmaTransportAdapter::do_connect(int rank,
                                      RdmaPeerConnectSpec const& remote) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peers_.find(rank);
    if (it != peers_.end() && it->second->put_ready) return true;
    if (it != peers_.end() && !it->second->qps_in_init) {
      it->second.reset();
    }
  }

  RdmaPeer* p = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peers_.find(rank);
    if (it != peers_.end() && it->second->qps_in_init) {
      p = it->second.get();
    }
  }

  if (!p) {
    auto np = std::make_unique<RdmaPeer>();
    np->num_qps = remote.num_qps > 0 ? remote.num_qps
                                     : static_cast<uint8_t>(config_.num_qps);
    if (!init_peer_qps(*np)) return false;
    np->qps_in_init = true;
    {
      std::lock_guard<std::mutex> lk(mu_);
      peers_[rank] = std::move(np);
      p = peers_[rank].get();
    }
  }

  memcpy(p->remote_send_qpns, remote.remote_send_qpns,
         sizeof(p->remote_send_qpns));
  memcpy(p->remote_recv_qpns, remote.remote_recv_qpns,
         sizeof(p->remote_recv_qpns));
  p->remote_lid = remote.remote_lid;
  memcpy(p->remote_gid.raw, remote.remote_gid_raw.data(),
         sizeof(p->remote_gid.raw));
  p->remote_gid_index = remote.remote_gid_index;
  p->remote_signal_addr = remote.remote_signal_addr;
  p->remote_signal_rkey = remote.remote_signal_rkey;
  p->remote_signal_qpn = remote.remote_signal_qpn;

  if (!qps_to_rtr(p->send_qps, p->num_qps, remote, true)) return false;
  if (!qps_to_rts(p->send_qps, p->num_qps)) return false;

  if (!qps_to_rtr(p->recv_qps, p->num_qps, remote, false)) return false;
  if (!qps_to_rts(p->recv_qps, p->num_qps)) return false;
  if (!p->wait_ready && !init_recv_pools(*p)) return false;

  if (!p->signal_qp) {
    if (!create_signal_qp(*p)) return false;
  }
  if (!connect_signal_qp(*p, p->remote_signal_qpn)) return false;
  if (!p->signal_pool && !init_signal_pool(*p)) return false;

  std::cerr << "[do_connect] SIGNAL_READY rank=" << rank
            << " signal_qpn=" << p->signal_qp->qp_num
            << " remote_qpn=" << p->remote_signal_qpn << std::endl;
  p->put_ready = true;
  p->wait_ready = true;

  return true;
}

bool RdmaTransportAdapter::do_accept(int rank,
                                     RdmaPeerConnectSpec const& remote) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peers_.find(rank);
    if (it != peers_.end() && it->second->wait_ready && it->second->put_ready)
      return true;
    if (it != peers_.end() && !it->second->qps_in_init) {
      it->second.reset();
    }
  }

  RdmaPeer* p = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peers_.find(rank);
    if (it != peers_.end() && it->second->qps_in_init) {
      p = it->second.get();
    }
  }

  if (!p) {
    auto np = std::make_unique<RdmaPeer>();
    np->num_qps = remote.num_qps > 0 ? remote.num_qps
                                     : static_cast<uint8_t>(config_.num_qps);
    if (!init_peer_qps(*np)) return false;
    np->qps_in_init = true;
    {
      std::lock_guard<std::mutex> lk(mu_);
      peers_[rank] = std::move(np);
      p = peers_[rank].get();
    }
  }

  memcpy(p->remote_send_qpns, remote.remote_send_qpns,
         sizeof(p->remote_send_qpns));
  memcpy(p->remote_recv_qpns, remote.remote_recv_qpns,
         sizeof(p->remote_recv_qpns));
  p->remote_lid = remote.remote_lid;
  memcpy(p->remote_gid.raw, remote.remote_gid_raw.data(),
         sizeof(p->remote_gid.raw));
  p->remote_gid_index = remote.remote_gid_index;
  p->remote_signal_addr = remote.remote_signal_addr;
  p->remote_signal_rkey = remote.remote_signal_rkey;
  p->remote_signal_qpn = remote.remote_signal_qpn;

  if (!qps_to_rtr(p->send_qps, p->num_qps, remote, true)) return false;
  if (!qps_to_rts(p->send_qps, p->num_qps)) return false;

  if (!qps_to_rtr(p->recv_qps, p->num_qps, remote, false)) return false;
  if (!qps_to_rts(p->recv_qps, p->num_qps)) return false;
  if (!p->wait_ready && !init_recv_pools(*p)) return false;

  if (!p->signal_qp) {
    if (!create_signal_qp(*p)) return false;
  }
  if (!connect_signal_qp(*p, p->remote_signal_qpn)) return false;
  if (!p->signal_pool && !init_signal_pool(*p)) return false;

  std::cerr << "[do_accept] SIGNAL_READY rank=" << rank
            << " signal_qpn=" << p->signal_qp->qp_num
            << " remote_qpn=" << p->remote_signal_qpn << std::endl;
  p->put_ready = true;
  p->wait_ready = true;

  return true;
}

// ── QP selection ─────────────────────────────────────────────────────────────

int RdmaTransportAdapter::select_qp(RdmaPeer& p, uint32_t msize) {
  if (msize <= kCacheSizeThresh &&
      p.cached_qp_valid_.load(std::memory_order_acquire)) {
    uint32_t prev =
        p.consecutive_cached_bytes_.fetch_add(msize, std::memory_order_relaxed);
    if (prev + msize <= kCacheConsecutiveThresh) {
      return p.last_qp_.load(std::memory_order_relaxed);
    }
  }
  p.cached_qp_valid_.store(false, std::memory_order_release);
  p.consecutive_cached_bytes_.store(0, std::memory_order_release);

  if (p.num_qps == 1) {
    p.last_qp_.store(0, std::memory_order_relaxed);
    p.cached_qp_valid_.store(true, std::memory_order_release);
    p.consecutive_cached_bytes_.store(msize, std::memory_order_release);
    return 0;
  }

  int a = std::rand() % p.num_qps;
  int b = std::rand() % p.num_qps;
  while (a == b) b = std::rand() % p.num_qps;
  int chosen = (p.rtt_tracker[a].ewma_rtt_ns < p.rtt_tracker[b].ewma_rtt_ns)
                   ? a
                   : b;
  p.last_qp_.store(chosen, std::memory_order_relaxed);
  p.cached_qp_valid_.store(true, std::memory_order_release);
  p.consecutive_cached_bytes_.store(msize, std::memory_order_release);
  return chosen;
}

// ── send path ────────────────────────────────────────────────────────────────

unsigned RdmaTransportAdapter::put_async(int rank, void* local_ptr,
                                         uint32_t local_buf_id,
                                         void* remote_ptr,
                                         uint32_t remote_buf_id, size_t len) {
  if (!has_put_path(rank) || len == 0) return 0;

  RdmaPeer* p = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peers_.find(rank);
    if (it == peers_.end() || !it->second->put_ready) return 0;
    p = it->second.get();
  }

  ibv_mr* lmr = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = mr_map_.find(local_buf_id);
    if (it != mr_map_.end()) lmr = it->second;
  }
  if (!lmr) {
    lmr = ibv_reg_mr(pd_, local_ptr, len,
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!lmr) return 0;
  }

  uint64_t raddr = 0;
  uint32_t rkey = 0;
  if (remote_buf_id != 0) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = p->remote_buffers.find(remote_buf_id);
    if (it != p->remote_buffers.end()) {
      raddr = it->second.addr;
      rkey = it->second.rkey;
    }
  }
  if (remote_ptr && raddr == 0) {
    raddr = reinterpret_cast<uint64_t>(remote_ptr);
  }
  if (raddr == 0 || rkey == 0) return 0;

  unsigned rid = 0;
  RequestSlot* slot = acquire_slot(&rid);
  if (!slot) return 0;
  slot->kind = RequestKind::DataPut;
  slot->peer_rank = rank;

  auto ck = chunk_split(len);
  slot->total_chunks = ck.count;
  slot->completed_chunks.store(0, std::memory_order_release);
  slot->regular_chunk_size = ck.regular_size;
  slot->last_chunk_size = ck.last_size;

  std::cerr << "[put_async] start rank=" << rank << " rid=" << rid
            << " len=" << len << " chunks=" << ck.count
            << " unacked=" << p->unacked_bytes_.load() << std::endl;

  size_t off = 0;
  for (uint32_t ci = 0; ci < ck.count; ++ci) {
    uint32_t sz = (ci + 1 == ck.count) ? ck.last_size : ck.regular_size;
    int q = select_qp(*p, sz);

    while (p->unacked_bytes_.load(std::memory_order_acquire) + sz >
           kMaxInflightBytes) {
      std::cerr << "[put_async] FLOW_BLOCK unacked="
                << p->unacked_bytes_.load(std::memory_order_acquire)
                << " + sz=" << sz << " > max=" << kMaxInflightBytes
                << " rank=" << rank << " ci=" << ci << std::endl;
      std::unique_lock<std::mutex> lk(cv_mu_);
      cv_.wait_for(lk, kPollSleep, [&] {
        return p->unacked_bytes_.load(std::memory_order_acquire) + sz <=
                   kMaxInflightBytes ||
               slot->failed.load(std::memory_order_acquire) ||
               stop_.load(std::memory_order_acquire);
      });
    }

    uint64_t wr_id = (static_cast<uint64_t>(rid) << 32) | ci;

    ibv_sge sge = {};
    sge.addr = reinterpret_cast<uint64_t>(static_cast<uint8_t*>(local_ptr) +
                                          off);
    sge.length = sz;
    sge.lkey = lmr->lkey;

    ibv_send_wr wr = {};
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = raddr + off;
    wr.wr.rdma.rkey = rkey;

    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(p->send_qps[q], &wr, &bad) != 0) {
      std::cerr << "[put_async] POST_FAILED rank=" << rank << " ci=" << ci
                << " qp=" << q << " sz=" << sz << " rid=" << rid << std::endl;
      slot->failed.store(true, std::memory_order_release);
      slot->completed.store(true, std::memory_order_release);
      return rid;
    }

    uint64_t prev = p->unacked_bytes_.fetch_add(sz, std::memory_order_relaxed);
    std::cerr << "[put_async] POST_OK ci=" << ci << " sz=" << sz
              << " unacked: " << prev << " -> " << (prev + sz)
              << " rid=" << rid << " qp=" << q << std::endl;
    p->rtt_tracker[q].last_send_ns.store(now_ns(), std::memory_order_release);
    off += sz;
  }

  return rid;
}

unsigned RdmaTransportAdapter::signal_async(int rank, uint64_t tag) {
  if (!has_put_path(rank)) return 0;

  RdmaPeer* p = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peers_.find(rank);
    if (it == peers_.end() || !it->second->put_ready) return 0;
    p = it->second.get();
  }

  unsigned rid = 0;
  RequestSlot* slot = acquire_slot(&rid);
  if (!slot) return 0;
  slot->kind = RequestKind::Signal;
  slot->peer_rank = rank;
  slot->expected_tag = tag;
  slot->total_chunks = 1;
  slot->completed_chunks.store(0, std::memory_order_release);

  uint8_t msg_id = p->next_msg_id++ & kMsgIdMask;

  uint64_t* tag_addr =
      reinterpret_cast<uint64_t*>(signal_addr_ +
                                  static_cast<uintptr_t>(msg_id) * 8);
  *tag_addr = tag;

  uint32_t imm = imm_encode(msg_id, 0, 0);

  ibv_sge sge = {};
  sge.addr = reinterpret_cast<uint64_t>(tag_addr);
  sge.length = 8;
  sge.lkey = signal_mr_->lkey;

  ibv_send_wr wr = {};
  wr.wr_id = static_cast<uint64_t>(rid) << 32;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  wr.imm_data = imm;
  wr.wr.rdma.remote_addr =
      p->remote_signal_addr + static_cast<uintptr_t>(msg_id) * 8;
  wr.wr.rdma.rkey = p->remote_signal_rkey;

  ibv_send_wr* bad = nullptr;
  if (ibv_post_send(p->signal_qp, &wr, &bad) != 0) {
    std::cerr << "[signal_async] FAILED ibv_post_send rank=" << rank
              << " msg_id=" << (int)msg_id << std::endl;
    slot->failed.store(true, std::memory_order_release);
    slot->completed.store(true, std::memory_order_release);
    return rid;
  }

  std::cerr << "[signal_async] posted rank=" << rank << " msg_id=" << (int)msg_id
            << " tag=" << tag << " rid=" << rid << std::endl;
  return rid;
}

// ── recv path ────────────────────────────────────────────────────────────────

unsigned RdmaTransportAdapter::wait_async(int rank, uint64_t expected_tag,
                                          std::optional<WaitTarget> target) {
  if (!has_wait_path(rank)) return 0;

  unsigned rid = 0;
  RequestSlot* slot = acquire_slot(&rid);
  if (!slot) return 0;
  slot->peer_rank = rank;
  slot->expected_tag = expected_tag;

  RdmaPeer* p = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peers_.find(rank);
    if (it == peers_.end()) return 0;
    p = it->second.get();
  }

  if (!target.has_value()) {
    slot->kind = RequestKind::SignalWait;
  } else {
    slot->kind = RequestKind::DataWait;
  }

  unsigned idx =
      p->next_expected_dispatch.fetch_add(1, std::memory_order_relaxed) &
      kMsgIdMask;
  p->trackers[idx].wait_slot.store(rid, std::memory_order_release);

  return rid;
}

// ── completion ──────────────────────────────────────────────────────────────

bool RdmaTransportAdapter::poll_completion(unsigned id) {
  auto* slot = resolve_const(id);
  if (!slot) return true;
  return slot->completed.load(std::memory_order_acquire);
}

bool RdmaTransportAdapter::wait_completion(unsigned id) {
  int spins = 0;
  while (!poll_completion(id)) {
    if (++spins % 100000 == 0)
      std::cerr << "[wait_completion] still waiting rid=" << id
                << " spins=" << spins << std::endl;
    std::this_thread::sleep_for(kPollSleep);
  }
  return true;
}

bool RdmaTransportAdapter::request_failed(unsigned id) {
  auto* slot = resolve_const(id);
  if (!slot) return false;
  return slot->failed.load(std::memory_order_acquire);
}

void RdmaTransportAdapter::release_request(unsigned id) { free_slot(id); }

// ── memory registration ─────────────────────────────────────────────────────

bool RdmaTransportAdapter::register_memory(uint32_t buf_id, void* ptr,
                                           size_t len) {
  if (!pd_ || !ptr || len == 0) return false;
  std::lock_guard<std::mutex> lk(mu_);
  if (mr_map_.find(buf_id) != mr_map_.end()) return true;
  ibv_mr* mr = ibv_reg_mr(pd_, ptr, len,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_READ);
  if (!mr) return false;
  mr_map_[buf_id] = mr;
  return true;
}

void RdmaTransportAdapter::deregister_memory(uint32_t buf_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = mr_map_.find(buf_id);
  if (it != mr_map_.end()) {
    ibv_dereg_mr(it->second);
    mr_map_.erase(it);
  }
}

bool RdmaTransportAdapter::is_memory_registered(uint32_t buf_id) const {
  std::lock_guard<std::mutex> lk(mu_);
  return mr_map_.find(buf_id) != mr_map_.end();
}

uint32_t RdmaTransportAdapter::get_memory_rkey(uint32_t buf_id) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = mr_map_.find(buf_id);
  return (it != mr_map_.end()) ? it->second->rkey : 0;
}

void RdmaTransportAdapter::register_remote_buffer(int rank, uint32_t buf_id,
                                                  uint64_t addr,
                                                  uint32_t rkey) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peers_.find(rank);
  if (it != peers_.end()) it->second->remote_buffers[buf_id] = {addr, rkey};
}

void RdmaTransportAdapter::set_remote_signal_info(int rank, uint64_t addr,
                                                  uint32_t rkey) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peers_.find(rank);
  if (it != peers_.end()) {
    it->second->remote_signal_addr = addr;
    it->second->remote_signal_rkey = rkey;
  }
}

// ── polling ──────────────────────────────────────────────────────────────────

void RdmaTransportAdapter::poll_loop() {
  unsigned poll_cycles = 0;
  unsigned signal_cq_polls = 0;
  while (!stop_.load(std::memory_order_acquire)) {
    std::vector<std::pair<int, RdmaPeer*>> active;
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (auto& [rank, p] : peers_) {
        if (p->send_cq || p->recv_cq || p->signal_cq)
          active.push_back({rank, p.get()});
      }
    }

    bool any = false;
    for (auto& [rank, p] : active) {
      if (p->send_cq &&
          poll_cq_set(*p, rank, p->send_cq, p->send_qps, p->num_qps, false))
        any = true;
      if (p->recv_cq &&
          poll_cq_set(*p, rank, p->recv_cq, p->recv_qps, p->num_qps, true))
        any = true;
      if (p->signal_cq) {
        signal_cq_polls++;
        if (poll_signal_cq(*p, rank)) any = true;
      }
      check_dispatch(*p, rank);
    }
    poll_cycles++;
    if (poll_cycles % 500000 == 0) {
      std::cerr << "[poll_loop] cycles=" << poll_cycles
                << " signal_cq_polls=" << signal_cq_polls
                << " any=" << any << std::endl;
    }
    if (!any) {
      std::this_thread::sleep_for(kPollSleep);
    }
  }
}

bool RdmaTransportAdapter::poll_cq_set(RdmaPeer& p, int rank, ibv_cq* cq,
                                        ibv_qp* const* qps, int qp_count,
                                        bool is_recv_side) {
  ibv_wc wc;
  bool any = false;
  while (ibv_poll_cq(cq, 1, &wc) > 0) {
    any = true;
    if (wc.status != IBV_WC_SUCCESS) {
      if (is_recv_side && (wc.opcode == IBV_WC_RECV ||
                           wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM)) {
        int qp = find_qp_idx(qps, qp_count, wc.qp_num);
        if (qp >= 0) repost_one_recv(p, qp);
      }
      unsigned rid = static_cast<unsigned>(wc.wr_id >> 32);
      std::cerr << "[poll_cq_set] ERROR status=" << (int)wc.status
                << " opcode=" << (int)wc.opcode << " rid=" << rid
                << " byte_len=" << wc.byte_len
                << " unacked_before=" << p.unacked_bytes_.load()
                << " is_recv=" << is_recv_side << std::endl;
      RequestSlot* s = resolve_slot(rid);
      if (wc.opcode == IBV_WC_RDMA_WRITE || wc.opcode == IBV_WC_SEND) {
        if (s && !s->completed.load(std::memory_order_acquire)) {
          s->failed.store(true, std::memory_order_release);
          s->completed.store(true, std::memory_order_release);
          uint64_t total_bytes = static_cast<uint64_t>(s->total_chunks - 1) * s->regular_chunk_size + s->last_chunk_size;
          uint64_t before = p.unacked_bytes_.fetch_sub(total_bytes, std::memory_order_relaxed);
          std::cerr << "[poll_cq_set] ERROR_DONE rid=" << rid
                    << " total_bytes=" << total_bytes
                    << " unacked: " << before << " -> " << (before - total_bytes)
                    << std::endl;
        } else if (s && s->completed.load(std::memory_order_acquire)) {
          // already completed (previous error or all success), don't decrement again
        } else {
          p.unacked_bytes_.fetch_sub(wc.byte_len, std::memory_order_relaxed);
        }
      } else {
        if (s && !s->completed.load(std::memory_order_acquire)) {
          s->failed.store(true, std::memory_order_release);
          s->completed.store(true, std::memory_order_release);
        }
      }
      cv_.notify_all();
      continue;
    }

    switch (wc.opcode) {
      case IBV_WC_RDMA_WRITE:
      case IBV_WC_SEND: {
        unsigned rid = static_cast<unsigned>(wc.wr_id >> 32);
        int qp = find_qp_idx(qps, qp_count, wc.qp_num);
        if (qp >= 0) {
          uint64_t send_ns =
              p.rtt_tracker[qp].last_send_ns.load(std::memory_order_acquire);
          if (send_ns != 0) {
            uint64_t rtt_ns = now_ns() - send_ns;
            p.rtt_tracker[qp].ewma_rtt_ns =
                kEwmaAlpha * static_cast<double>(rtt_ns) +
                (1.0 - kEwmaAlpha) * p.rtt_tracker[qp].ewma_rtt_ns;
          }
        }

        RequestSlot* s = resolve_slot(rid);
        if (!s) break;
        uint32_t done =
            s->completed_chunks.fetch_add(1, std::memory_order_acq_rel) + 1;
        std::cerr << "[poll_cq_set] CQE_DECR rid=" << rid
                  << " wc_byte_len=" << wc.byte_len
                  << " done_chunks=" << done << "/" << s->total_chunks
                  << " unacked_before=" << p.unacked_bytes_.load()
                  << " is_recv=" << is_recv_side << std::endl;
        if (done >= s->total_chunks) {
          uint64_t total_bytes = static_cast<uint64_t>(s->total_chunks - 1) * s->regular_chunk_size + s->last_chunk_size;
          uint64_t before = p.unacked_bytes_.fetch_sub(total_bytes, std::memory_order_relaxed);
          std::cerr << "[poll_cq_set] ALL_DONE rid=" << rid
                    << " total_bytes=" << total_bytes
                    << " unacked: " << before << " -> " << (before - total_bytes)
                    << std::endl;
          s->completed.store(true, std::memory_order_release);
        }
        cv_.notify_all();
        break;
      }

      case IBV_WC_RECV_RDMA_WITH_IMM: {
        int qp = find_qp_idx(qps, qp_count, wc.qp_num);
        if (qp >= 0) repost_one_recv(p, qp);

        uint32_t imm = wc.imm_data;
        uint8_t msg_id = imm_msg_id(imm);

        // Only signal IMMs arrive (total=0); data WRs use plain RDMA_WRITE
        auto& t = p.trackers[msg_id];
        t.completed.store(true, std::memory_order_release);

        check_dispatch(p, rank);
        break;
      }

      case IBV_WC_RECV: {
        int qp = find_qp_idx(qps, qp_count, wc.qp_num);
        if (qp >= 0) repost_one_recv(p, qp);
        break;
      }

      default:
        break;
    }
  }
  return any;
}

bool RdmaTransportAdapter::poll_signal_cq(RdmaPeer& p, int rank) {
  ibv_wc wc;
  bool any = false;
  int ret;
  while ((ret = ibv_poll_cq(p.signal_cq, 1, &wc)) > 0) {
    any = true;
    if (wc.status != IBV_WC_SUCCESS) {
      if (wc.opcode == IBV_WC_RECV ||
          wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        repost_signal_recv(p);
      }
      unsigned rid = static_cast<unsigned>(wc.wr_id >> 32);
      std::cerr << "[poll_signal_cq] ERROR status=" << wc.status
                << " opcode=" << wc.opcode << " rid=" << rid
                << " vendor_err=" << wc.vendor_err << std::endl;
      RequestSlot* s = resolve_slot(rid);
      if (s && !s->completed.load(std::memory_order_acquire)) {
        s->failed.store(true, std::memory_order_release);
        s->completed.store(true, std::memory_order_release);
      }
      cv_.notify_all();
      continue;
    }

    switch (wc.opcode) {
      case IBV_WC_RDMA_WRITE: {
        unsigned rid = static_cast<unsigned>(wc.wr_id >> 32);
        std::cerr << "[poll_signal_cq] SEND_CQE rid=" << rid
                  << " status=" << wc.status << std::endl;
        RequestSlot* s = resolve_slot(rid);
        if (s) {
          s->completed.store(true, std::memory_order_release);
        }
        cv_.notify_all();
        break;
      }

      case IBV_WC_RECV_RDMA_WITH_IMM: {
        repost_signal_recv(p);
        uint32_t imm = wc.imm_data;
        uint8_t msg_id = imm_msg_id(imm);
        std::cerr << "[poll_signal_cq] RECV_IMM rank=" << rank
                  << " msg_id=" << (int)msg_id << std::endl;
        auto& t = p.trackers[msg_id];
        t.completed.store(true, std::memory_order_release);
        check_dispatch(p, rank);
        break;
      }

      case IBV_WC_RECV: {
        repost_signal_recv(p);
        break;
      }

      default:
        break;
    }
  }
  if (ret < 0) {
    std::cerr << "[poll_signal_cq] POLL_ERROR ret=" << ret
              << " errno=" << errno << std::endl;
  }
  return any;
}

void RdmaTransportAdapter::check_dispatch(RdmaPeer& p, int rank) {
  unsigned expected = p.next_expected_dispatch.load(std::memory_order_acquire);
  if (p.dispatch_cursor < expected) {
    std::cerr << "[check_dispatch] enter rank=" << rank
              << " cursor=" << p.dispatch_cursor << " next_expected=" << expected
              << std::endl;
  }
  while (p.dispatch_cursor < expected) {
    unsigned idx = p.dispatch_cursor & kMsgIdMask;
    auto& t = p.trackers[idx];

    if (!t.completed.load(std::memory_order_acquire)) {
      std::cerr << "[check_dispatch] tracker[" << idx
                << "] not completed, breaking (cursor=" << p.dispatch_cursor
                << ")" << std::endl;
      {
        ibv_qp_attr a;
        ibv_qp_init_attr ia;
        if (ibv_query_qp(p.signal_qp, &a, IBV_QP_STATE, &ia) == 0) {
          std::cerr << "[check_dispatch] signal_qp_state=" << (int)a.qp_state
                    << " (2=RTR,3=RTS,6=ERR)" << std::endl;
        }
      }
      break;
    }
    unsigned wait_id = t.wait_slot.load(std::memory_order_acquire);
    if (wait_id == 0) {
      std::cerr << "[check_dispatch] tracker[" << idx
                << "] wait_id=0, breaking" << std::endl;
      break;
    }

    RequestSlot* s = resolve_slot(wait_id);
    if (s && s->peer_rank == rank) {
      if (s->kind == RequestKind::SignalWait) {
        uint64_t* tag_addr = reinterpret_cast<uint64_t*>(
            signal_addr_ + static_cast<uintptr_t>(idx) * 8);
        if (*tag_addr == s->expected_tag) {
          std::cerr << "[check_dispatch] SignalWait idx=" << idx
                    << " tag matched, completing slot" << std::endl;
          s->completed.store(true, std::memory_order_release);
        } else {
          std::cerr << "[check_dispatch] SignalWait idx=" << idx
                    << " TAG MISMATCH got=" << *tag_addr
                    << " expected=" << s->expected_tag << ", breaking"
                    << std::endl;
          break;
        }
      } else if (s->kind == RequestKind::DataWait) {
        std::cerr << "[check_dispatch] DataWait idx=" << idx
                  << " completing slot wait_id=" << wait_id << std::endl;
        s->completed.store(true, std::memory_order_release);
      }
    } else {
      std::cerr << "[check_dispatch] resolve_slot failed wait_id=" << wait_id
                << " peer_rank=" << (s ? s->peer_rank : -1)
                << " expected_rank=" << rank << ", breaking" << std::endl;
      break;
    }

    t.completed.store(false, std::memory_order_release);
    t.wait_slot.store(0, std::memory_order_release);
    t.total = 0;
    t.done = 0;
    p.dispatch_cursor++;
    cv_.notify_all();
    expected = p.next_expected_dispatch.load(std::memory_order_acquire);
  }
}

// ── slot management ─────────────────────────────────────────────────────────

RdmaTransportAdapter::RequestSlot* RdmaTransportAdapter::acquire_slot(
    unsigned* out) {
  if (!out || !slots_) return nullptr;
  for (uint32_t n = 0; n < kSlotCount; ++n) {
    uint32_t idx = cursor_.fetch_add(1, std::memory_order_relaxed) & kSlotMask;
    auto& s = slots_[idx];
    uint8_t expect = 0;
    if (!s.state.compare_exchange_strong(expect, 1,
                                         std::memory_order_acq_rel)) {
      continue;
    }
    uint32_t gen = s.generation.load(std::memory_order_acquire);
    if (gen == 0) {
      gen = 1;
      s.generation.store(gen, std::memory_order_release);
    }
    s.kind = RequestKind::DataPut;
    s.peer_rank = -1;
    s.expected_tag = 0;
    s.completed.store(false, std::memory_order_release);
    s.failed.store(false, std::memory_order_release);
    s.total_chunks = 0;
    s.completed_chunks.store(0, std::memory_order_release);
    s.regular_chunk_size = 0;
    s.last_chunk_size = 0;
    *out = make_request_id(idx, gen);
    std::cerr << "[acquire_slot] idx=" << idx << " gen=" << gen
              << " rid=" << *out << " cursor=" << cursor_.load() << std::endl;
    return &s;
  }
  return nullptr;
}

RdmaTransportAdapter::RequestSlot* RdmaTransportAdapter::resolve_slot(
    unsigned id) {
  if (id == 0 || !slots_) return nullptr;
  uint32_t gen = slot_generation(id);
  if (gen == 0) return nullptr;
  uint32_t idx = slot_index(id);
  auto& s = slots_[idx];
  if (s.generation.load(std::memory_order_acquire) != gen) return nullptr;
  if (s.state.load(std::memory_order_acquire) == 0) return nullptr;
  return &s;
}

const RdmaTransportAdapter::RequestSlot* RdmaTransportAdapter::resolve_const(
    unsigned id) const {
  if (id == 0 || !slots_) return nullptr;
  uint32_t gen = slot_generation(id);
  if (gen == 0) return nullptr;
  uint32_t idx = slot_index(id);
  auto& s = slots_[idx];
  if (s.generation.load(std::memory_order_acquire) != gen) return nullptr;
  if (s.state.load(std::memory_order_acquire) == 0) return nullptr;
  return &s;
}

void RdmaTransportAdapter::free_slot(unsigned id) {
  if (id == 0 || !slots_) return;
  uint32_t gen = slot_generation(id);
  if (gen == 0) return;
  uint32_t idx = slot_index(id);
  auto& s = slots_[idx];
  uint8_t st = s.state.load(std::memory_order_acquire);
  if (st <= 1 && !s.completed.load(std::memory_order_acquire)) return;
  s.state.store(0, std::memory_order_release);
  s.total_chunks = 0;
  s.completed_chunks.store(0, std::memory_order_release);
  s.generation.fetch_add(1, std::memory_order_release);
  uint32_t ng = s.generation.load(std::memory_order_acquire);
  std::cerr << "[free_slot] rid=" << id << " idx=" << idx << " old_gen=" << gen
            << " new_gen=" << ng << " st=" << (int)st << " completed="
            << (int)s.completed.load() << std::endl;
  cv_.notify_all();
}

RdmaTransportAdapter::ChunkResult RdmaTransportAdapter::chunk_split(
    size_t len) const {
  ChunkResult r = {};
  uint32_t cs = static_cast<uint32_t>(config_.chunk_size_kb) * 1024;
  r.regular_size = cs;

  uint32_t n = static_cast<uint32_t>((len + cs - 1) / cs);
  if (n > kMaxChunks) {
    n = kMaxChunks;
    r.regular_size = static_cast<uint32_t>(len / n);
  }
  r.count = n;
  if (n > 1) {
    r.last_size =
        static_cast<uint32_t>(len - static_cast<uint64_t>(r.regular_size) *
                                       (n - 1));
  } else {
    r.last_size = static_cast<uint32_t>(len);
  }
  return r;
}

}  // namespace Transport
}  // namespace UKernel
