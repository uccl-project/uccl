#include "rdma_adapter.h"
#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <unistd.h>

#include "util/util.h"

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

int pick_dev_for_gpu(int gpu_idx) {
  auto gpu_cards = uccl::get_gpu_cards();
  if (gpu_idx < 0 || static_cast<size_t>(gpu_idx) >= gpu_cards.size()) return 0;

  auto nics = uccl::get_rdma_nics();
  if (nics.empty()) return 0;

  int best_idx = 0;
  uint32_t best_dist = UINT32_MAX;
  int ndev = 0;
  ibv_device** devs = ibv_get_device_list(&ndev);
  if (!devs) return 0;

  for (int j = 0; j < ndev; ++j) {
    std::string name = ibv_get_device_name(devs[j]);
    auto it = std::find_if(nics.begin(), nics.end(),
        [&](auto const& p) { return p.first == name; });
    if (it == nics.end()) continue;
    uint32_t d = uccl::safe_pcie_distance(gpu_cards[gpu_idx], it->second);
    if (d < best_dist) { best_dist = d; best_idx = j; }
  }
  ibv_free_device_list(devs);
  return best_idx;
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

  local_dev_idx_ = pick_dev_for_gpu(local_gpu_idx_);

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
    fprintf(stderr, "[ERROR] RdmaAdapter: ibv_query_device failed\n");
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
  attr.cap.max_recv_wr = 1;
  attr.cap.max_send_sge = 1;
  attr.cap.max_recv_sge = 1;
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
                                      RdmaPeerConnectSpec const& remote) {
  for (int i = 0; i < count; ++i) {
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote.remote_signal_qpn;
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
  if (ibv_post_recv(p.signal_qp, &p.signal_pool->wrs[idx], &bad) != 0) {
    fprintf(stderr, "[repost_signal_recv] FAILED idx=%d errno=%d\n", idx, errno);
    return false;
  }
  p.signal_post_idx = (idx + 1) % kSignalPoolSz;
  return true;
}

// ── peer init / destroy ─────────────────────────────────────────────────────

bool RdmaTransportAdapter::init_peer_qps(RdmaPeer& p) {
  int cq_size = kQpMaxSendWr * 2;
  return create_qp_set(p.send_qps, &p.send_cq, p.num_qps, cq_size) &&
         qps_to_init(p.send_qps, p.num_qps) &&
         create_signal_qp(p);
}

void RdmaTransportAdapter::destroy_peer_qps(RdmaPeer& p) {
  for (int q = 0; q < p.num_qps; ++q) {
    if (p.send_qps[q]) {
      ibv_destroy_qp(p.send_qps[q]);
      p.send_qps[q] = nullptr;
    }
  }
  if (p.send_cq) {
    ibv_destroy_cq(p.send_cq);
    p.send_cq = nullptr;
  }
  destroy_signal_qp(p);
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
        init.signal_qpn = p->signal_qp ? p->signal_qp->qp_num : 0;
        peers_[peer_rank] = std::move(p);
      }
    } else if (it->second->qps_in_init) {
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
  return setup_peer_path(spec.peer_rank,
                         std::get<RdmaPeerConnectSpec>(spec.detail));
}

bool RdmaTransportAdapter::ensure_wait_path(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_wait_path(spec.peer_rank)) return true;
  if (spec.type != PeerConnectType::Accept) return false;
  if (!std::holds_alternative<RdmaPeerConnectSpec>(spec.detail)) return false;
  return setup_peer_path(spec.peer_rank,
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

bool RdmaTransportAdapter::setup_peer_path(int rank,
                                          RdmaPeerConnectSpec const& remote) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peers_.find(rank);
    if (it != peers_.end() && it->second->put_ready && it->second->wait_ready)
      return true;
    if (it != peers_.end() && !it->second->qps_in_init) {
      peers_.erase(it);
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

  p->remote_lid = remote.remote_lid;
  memcpy(p->remote_gid.raw, remote.remote_gid_raw.data(),
         sizeof(p->remote_gid.raw));
  p->remote_gid_index = remote.remote_gid_index;
  p->remote_signal_addr = remote.remote_signal_addr;
  p->remote_signal_rkey = remote.remote_signal_rkey;
  p->remote_signal_qpn = remote.remote_signal_qpn;

  if (!qps_to_rtr(p->send_qps, p->num_qps, remote)) return false;
  if (!qps_to_rts(p->send_qps, p->num_qps)) return false;

  if (!p->signal_qp) {
    if (!create_signal_qp(*p)) return false;
  }
  if (!connect_signal_qp(*p, p->remote_signal_qpn)) return false;
  if (!p->signal_pool && !init_signal_pool(*p)) return false;

  p->put_ready = true;
  p->wait_ready = true;

  {
    auto state_name = [](ibv_qp_state s) -> char const* {
      switch (s) {
        case IBV_QPS_RESET: return "RESET";
        case IBV_QPS_INIT:  return "INIT";
        case IBV_QPS_RTR:   return "RTR";
        case IBV_QPS_RTS:   return "RTS";
        case IBV_QPS_SQD:   return "SQD";
        case IBV_QPS_SQE:   return "SQE";
        case IBV_QPS_ERR:   return "ERR";
        default:            return "?";
      }
    };
    ibv_qp_attr qattr;
    ibv_qp_init_attr iattr;
    for (int i = 0; i < p->num_qps; ++i) {
      ibv_query_qp(p->send_qps[i], &qattr, IBV_QP_STATE, &iattr);
      fprintf(stderr, "[RDMA] rank=%d send_qp[%d] qpn=%u state=%s\n",
              rank, i, p->send_qps[i]->qp_num, state_name(qattr.qp_state));
    }
    ibv_query_qp(p->signal_qp, &qattr, IBV_QP_STATE, &iattr);
    fprintf(stderr, "[RDMA] rank=%d signal_qp qpn=%u state=%s\n",
            rank, p->signal_qp->qp_num, state_name(qattr.qp_state));
  }

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

  static thread_local std::mt19937 rng(std::random_device{}());
  int a = rng() % p.num_qps;
  int b = rng() % p.num_qps;
  while (a == b) b = rng() % p.num_qps;
  int chosen = (p.qp_state[a].ewma_rtt_ns < p.qp_state[b].ewma_rtt_ns)
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
  if (!lmr) return 0;

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
  slot->chunk_size = ck.chunk_size;
  slot->last_chunk_size = ck.last_size;

  fprintf(stderr, "[put_async] START rank=%d rid=%u len=%zu chunks=%u\n",
          rank, rid, len, ck.count);

  size_t off = 0;
  for (uint32_t ci = 0; ci < ck.count; ++ci) {
    uint32_t sz = (ci + 1 == ck.count) ? ck.last_size : ck.chunk_size;
    int q = select_qp(*p, sz);

    while (p->qp_state[q].unacked_bytes.load(std::memory_order_acquire) + sz >
           kMaxInflightBytesPerQp) {
      std::unique_lock<std::mutex> lk(cv_mu_);
      cv_.wait_for(lk, kPollSleep, [&] {
        return p->qp_state[q].unacked_bytes.load(std::memory_order_acquire) + sz <=
                   kMaxInflightBytesPerQp ||
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
      slot->failed.store(true, std::memory_order_release);
      slot->completed.store(true, std::memory_order_release);
      return rid;
    }

    p->qp_state[q].unacked_bytes.fetch_add(sz, std::memory_order_relaxed);
    p->qp_state[q].last_send_ns.store(now_ns(), std::memory_order_release);
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

  uint8_t msg_id = p->next_msg_id.fetch_add(1, std::memory_order_relaxed) & kMsgIdMask;

  uint64_t* tag_addr =
      reinterpret_cast<uint64_t*>(signal_addr_ +
                                  static_cast<uintptr_t>(msg_id) * 8);
  *tag_addr = tag;

  uint32_t imm = imm_encode(msg_id);

  fprintf(stderr, "[signal_async] rank=%d msg_id=%u tag=0x%lx remote_qpn=%u remote_addr=0x%lx+0x%lx rkey=0x%x\n",
          rank, msg_id, (unsigned long)tag, p->remote_signal_qpn,
          (unsigned long)p->remote_signal_addr,
          (unsigned long)(static_cast<uintptr_t>(msg_id) * 8),
          p->remote_signal_rkey);

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
    fprintf(stderr, "[signal_async] POST_FAILED rank=%d msg_id=%u\n", rank, msg_id);
    slot->failed.store(true, std::memory_order_release);
    slot->completed.store(true, std::memory_order_release);
    return rid;
  }

  fprintf(stderr, "[signal_async] OK rank=%d msg_id=%u rid=%u\n", rank, msg_id, rid);
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

  slot->kind = RequestKind::SignalWait;

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
  while (!poll_completion(id)) {
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
  while (!stop_.load(std::memory_order_acquire)) {
    std::vector<std::pair<int, RdmaPeer*>> active;
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (auto& [rank, p] : peers_) {
        if (p->send_cq || p->signal_cq)
          active.push_back({rank, p.get()});
      }
    }

    bool any = false;
    for (auto& [rank, p] : active) {
      if (p->send_cq &&
          poll_cq_set(*p, rank, p->send_cq, p->send_qps, p->num_qps))
        any = true;
      if (p->signal_cq && poll_signal_cq(*p, rank))
        any = true;
      check_dispatch(*p, rank);
    }
    if (!any) {
      std::this_thread::sleep_for(kPollSleep);
    }
  }
}

bool RdmaTransportAdapter::poll_cq_set(RdmaPeer& p, int rank, ibv_cq* cq,
                                         ibv_qp* const* qps, int qp_count) {
  ibv_wc wc;
  bool any = false;
  int polls = 0;
  int wc_count = 0;
  while ((polls = ibv_poll_cq(cq, 1, &wc)) > 0) {
    wc_count++;
    any = true;
    if (wc.status != IBV_WC_SUCCESS) {
      unsigned rid = static_cast<unsigned>(wc.wr_id >> 32);
      RequestSlot* s = resolve_slot(rid);
      int qp = find_qp_idx(qps, qp_count, wc.qp_num);
      if (qp >= 0 &&
          (wc.opcode == IBV_WC_RDMA_WRITE || wc.opcode == IBV_WC_SEND)) {
        p.qp_state[qp].unacked_bytes.fetch_sub(wc.byte_len, std::memory_order_relaxed);
      }
      if (s && !s->completed.load(std::memory_order_acquire)) {
        s->failed.store(true, std::memory_order_release);
        s->completed.store(true, std::memory_order_release);
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
              p.qp_state[qp].last_send_ns.load(std::memory_order_acquire);
          if (send_ns != 0) {
            uint64_t rtt_ns = now_ns() - send_ns;
            p.qp_state[qp].ewma_rtt_ns =
                kEwmaAlpha * static_cast<double>(rtt_ns) +
                (1.0 - kEwmaAlpha) * p.qp_state[qp].ewma_rtt_ns;
          }
        }

        RequestSlot* s = resolve_slot(rid);
        if (!s) break;
        if (qp >= 0)
          p.qp_state[qp].unacked_bytes.fetch_sub(wc.byte_len, std::memory_order_relaxed);
        uint32_t done =
            s->completed_chunks.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (done >= s->total_chunks) {
          s->completed.store(true, std::memory_order_release);
          fprintf(stderr, "[CQE] ALL_DONE rid=%u rank=%d\n",
                  rid, rank);
        }
        cv_.notify_all();
        break;
      }

      default:
        break;
    }
  }
  if (wc_count > 0) {
    fprintf(stderr, "[poll_cq_set] rank=%d wc_count=%d\n",
            rank, wc_count);
  }
  return any;
}

bool RdmaTransportAdapter::poll_signal_cq(RdmaPeer& p, int rank) {
  ibv_wc wc;
  bool any = false;
  int wc_count = 0;
  while (ibv_poll_cq(p.signal_cq, 1, &wc) > 0) {
    wc_count++;
    any = true;
    if (wc.status != IBV_WC_SUCCESS) {
      fprintf(stderr, "[poll_signal_cq] ERROR rank=%d status=%d opcode=%d wrid=0x%lx\n",
              rank, wc.status, wc.opcode, (unsigned long)wc.wr_id);
      if (wc.opcode == IBV_WC_RECV ||
          wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        (void)repost_signal_recv(p);
      }
      unsigned rid = static_cast<unsigned>(wc.wr_id >> 32);
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
        RequestSlot* s = resolve_slot(rid);
        if (s) {
          s->completed.store(true, std::memory_order_release);
        }
        cv_.notify_all();
        break;
      }

      case IBV_WC_RECV_RDMA_WITH_IMM: {
        (void)repost_signal_recv(p);
        uint32_t imm = wc.imm_data;
        uint8_t msg_id = imm_msg_id(imm);
        fprintf(stderr, "[poll_signal_cq] RECV_IMM rank=%d msg_id=%u imm=0x%x\n",
                rank, msg_id, imm);
        auto& t = p.trackers[msg_id];
        t.completed.store(true, std::memory_order_release);
        check_dispatch(p, rank);
        break;
      }

      case IBV_WC_RECV: {
        (void)repost_signal_recv(p);
        break;
      }

      default:
        break;
    }
  }
  if (wc_count > 0) {
    fprintf(stderr, "[poll_signal_cq] rank=%d wc_count=%d\n", rank, wc_count);
  }
  return any;
}

void RdmaTransportAdapter::check_dispatch(RdmaPeer& p, int rank) {
  unsigned expected = p.next_expected_dispatch.load(std::memory_order_acquire);
  while (p.dispatch_cursor < expected) {
    unsigned idx = p.dispatch_cursor & kMsgIdMask;
    auto& t = p.trackers[idx];

    if (!t.completed.load(std::memory_order_acquire)) {
      fprintf(stderr, "[check_dispatch] rank=%d cursor=%u expected=%u idx=%u NOT_COMPLETED\n",
              rank, p.dispatch_cursor, expected, idx);
      break;
    }
    unsigned wait_id = t.wait_slot.load(std::memory_order_acquire);
    if (wait_id == 0) {
      fprintf(stderr, "[check_dispatch] rank=%d cursor=%u expected=%u idx=%u completed BUT wait_id=0\n",
              rank, p.dispatch_cursor, expected, idx);
      break;
    }

    RequestSlot* s = resolve_slot(wait_id);
    if (s && s->peer_rank == rank) {
      uint64_t* tag_addr = reinterpret_cast<uint64_t*>(
          signal_addr_ + static_cast<uintptr_t>(idx) * 8);
      fprintf(stderr, "[check_dispatch] rank=%d cursor=%u idx=%u tag=0x%lx expected_tag=0x%lx wait_id=%u\n",
              rank, p.dispatch_cursor, idx, (unsigned long)*tag_addr,
              (unsigned long)s->expected_tag, wait_id);
      if (*tag_addr == s->expected_tag) {
        s->completed.store(true, std::memory_order_release);
        fprintf(stderr, "[check_dispatch] MATCH rank=%d idx=%u wait_id=%u\n",
                rank, idx, wait_id);
      } else {
        fprintf(stderr, "[check_dispatch] TAG_MISMATCH rank=%d idx=%u\n",
                rank, idx);
        break;
      }
    } else {
      fprintf(stderr, "[check_dispatch] rank=%d cursor=%u idx=%u BAD_SLOT s=%p peer=%d\n",
              rank, p.dispatch_cursor, idx, (void*)s, s ? s->peer_rank : -1);
      break;
    }

    t.completed.store(false, std::memory_order_release);
    t.wait_slot.store(0, std::memory_order_release);
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
    s.chunk_size = 0;
    s.last_chunk_size = 0;
    *out = make_request_id(idx, gen);
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
  cv_.notify_all();
}

RdmaTransportAdapter::ChunkResult RdmaTransportAdapter::chunk_split(
    size_t len) const {
  ChunkResult r = {};
  uint32_t cs = static_cast<uint32_t>(config_.chunk_size_kb) * 1024;
  r.chunk_size = cs;

  uint32_t n = static_cast<uint32_t>((len + cs - 1) / cs);
  if (n > kMaxChunks) {
    n = kMaxChunks;
    r.chunk_size = static_cast<uint32_t>(len / n);
  }
  r.count = n;
  if (n > 1) {
    r.last_size =
        static_cast<uint32_t>(len - static_cast<uint64_t>(r.chunk_size) *
                                       (n - 1));
  } else {
    r.last_size = static_cast<uint32_t>(len);
  }
  return r;
}

}  // namespace Transport
}  // namespace UKernel
