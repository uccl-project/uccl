#include "rdma_adapter.h"
#include "../util/utils.h"
#include "util/util.h"
#include <arpa/inet.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <unistd.h>

namespace UKernel {
namespace Transport {

namespace {

constexpr auto kCvTimeout = std::chrono::microseconds(1);
constexpr auto kQpMaxSendWr = 1024;
constexpr int kQpTimeout = 14;
constexpr int kQpRetryCnt = 7;
constexpr int kQpRnrRetry = 7;
constexpr double kEwmaAlpha = 0.125;
constexpr size_t kTaskRingSize = 1024;

template <typename T>
bool enqueue_elem(jring_t* ring, T const& elem, std::atomic<bool> const& stop) {
  while (!stop.load(std::memory_order_acquire) &&
         jring_mp_enqueue_bulk(ring, &elem, 1, nullptr) != 1)
    std::this_thread::yield();
  return !stop.load(std::memory_order_acquire);
}

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

  int best_idx = -1;
  uint32_t best_dist = UINT32_MAX;
  int ndev = 0;
  ibv_device** devs = ibv_get_device_list(&ndev);
  if (!devs) return 0;

  for (int j = 0; j < ndev; ++j) {
    std::string name = ibv_get_device_name(devs[j]);
    auto it = std::find_if(nics.begin(), nics.end(),
                           [&](auto const& p) { return p.first == name; });
    if (it == nics.end()) continue;

    char state_path[512];
    snprintf(state_path, sizeof(state_path),
             "/sys/class/infiniband/%s/ports/1/state", name.c_str());
    FILE* fp = fopen(state_path, "r");
    if (!fp) continue;
    int port_state = 0;
    char buf[32] = {};
    if (fgets(buf, sizeof(buf), fp)) port_state = atoi(buf);
    fclose(fp);
    if (port_state != 4) {  // 4 = IBV_PORT_ACTIVE
      fprintf(stderr, "[pick_dev] skip %s (port state=%d, not active)\n",
              name.c_str(), port_state);
      continue;
    }

    uint32_t d = uccl::safe_pcie_distance(gpu_cards[gpu_idx], it->second);
    if (d < best_dist) {
      best_dist = d;
      best_idx = j;
    }
  }
  ibv_free_device_list(devs);

  if (best_idx < 0) {
    fprintf(stderr, "[pick_dev] no active RDMA port found\n");
    return -1;
  }
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
  if (!devs || local_dev_idx_ < 0 || local_dev_idx_ >= ndev) {
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

  // Create jring-based task queues
  send_ring_ = create_ring(sizeof(RingElem), kTaskRingSize);
  recv_ring_ = create_ring(sizeof(RingElem), kTaskRingSize);
  if (!send_ring_ || !recv_ring_) {
    if (send_ring_) { free(send_ring_); send_ring_ = nullptr; }
    if (recv_ring_) { free(recv_ring_); recv_ring_ = nullptr; }
    ibv_dealloc_pd(pd_);
    pd_ = nullptr;
    ctx_handle_.reset();
    ctx_ = nullptr;
    return;
  }

  stop_.store(false, std::memory_order_release);
  send_worker_ = std::thread([this] { send_worker(); });
  recv_worker_ = std::thread([this] { recv_worker(); });
  poll_thread_ = std::thread([this] { poll_loop(); });
}

RdmaTransportAdapter::~RdmaTransportAdapter() {
  stop_.store(true, std::memory_order_release);
  cv_.notify_all();
  if (send_worker_.joinable()) send_worker_.join();
  if (recv_worker_.joinable()) recv_worker_.join();
  if (poll_thread_.joinable()) poll_thread_.join();

  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [rank, p] : peers_) destroy_peer_qps(*p);
    peers_.clear();
    for (auto& [id, mr] : mr_map_) ibv_dereg_mr(mr);
    mr_map_.clear();
  }

  if (send_ring_) { free(send_ring_); send_ring_ = nullptr; }
  if (recv_ring_) { free(recv_ring_); recv_ring_ = nullptr; }

  if (pd_) {
    ibv_dealloc_pd(pd_);
    pd_ = nullptr;
  }
}

// ── QP helpers ──────────────────────────────────────────────────────────────

bool RdmaTransportAdapter::create_qp_set(ibv_qp** qps, ibv_cq** cq, int count,
                                         int cq_size, int max_recv_wr) {
  *cq = ibv_create_cq(ctx_, cq_size, nullptr, nullptr, 0);
  if (!*cq) return false;

  ibv_qp_init_attr attr = {};
  attr.send_cq = *cq;
  attr.recv_cq = *cq;
  attr.qp_type = IBV_QPT_RC;
  attr.cap.max_send_wr = kQpMaxSendWr;
  attr.cap.max_recv_wr = static_cast<uint32_t>(max_recv_wr);
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
    attr.dest_qp_num = remote.remote_data_qpns[i];
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

    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                IBV_QP_MIN_RNR_TIMER;

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

bool RdmaTransportAdapter::init_signal_pool(RdmaPeer& p) {
  auto pool = std::make_unique<RecvPool>();
  pool->buffer.resize(4);
  pool->mr = ibv_reg_mr(pd_, pool->buffer.data(), pool->buffer.size(),
                        IBV_ACCESS_LOCAL_WRITE);
  if (!pool->mr) return false;

  pool->sges.resize(1);
  pool->wrs.resize(1);
  pool->sges[0].addr = reinterpret_cast<uint64_t>(pool->buffer.data());
  pool->sges[0].length = 4;
  pool->sges[0].lkey = pool->mr->lkey;
  pool->wrs[0].wr_id = 0;
  pool->wrs[0].sg_list = &pool->sges[0];
  pool->wrs[0].num_sge = 1;

  ibv_recv_wr* bad = nullptr;
  if (ibv_post_recv(p.signal_qp, &pool->wrs[0], &bad) != 0) {
    return false;
  }
  p.signal_pool = std::move(pool);
  return true;
}

bool RdmaTransportAdapter::repost_signal_recv(RdmaPeer& p) {
  if (!p.signal_pool) return false;
  ibv_recv_wr* bad = nullptr;
  if (ibv_post_recv(p.signal_qp, &p.signal_pool->wrs[0], &bad) != 0) {
    fprintf(stderr, "[repost_signal_recv] FAILED errno=%d\n", errno);
    return false;
  }
  return true;
}

// ── peer init / destroy ─────────────────────────────────────────────────────

bool RdmaTransportAdapter::init_peer_qps(RdmaPeer& p) {
  int cq_size = kQpMaxSendWr * 2;
  if (!create_qp_set(p.data_qps, &p.data_cq, p.num_qps, cq_size)) return false;
  if (!qps_to_init(p.data_qps, p.num_qps)) return false;

  p.signal_cq = ibv_create_cq(ctx_, 256, nullptr, nullptr, 0);
  if (!p.signal_cq) return false;

  ibv_qp_init_attr attr = {};
  attr.send_cq = p.signal_cq;
  attr.recv_cq = p.signal_cq;
  attr.qp_type = IBV_QPT_RC;
  attr.cap.max_send_wr = kQpMaxSendWr;
  attr.cap.max_recv_wr = 1;
  attr.cap.max_send_sge = 1;
  attr.cap.max_recv_sge = 1;
  attr.cap.max_inline_data = 16;

  p.signal_qp = ibv_create_qp(pd_, &attr);
  if (!p.signal_qp) return false;
  if (!qps_to_init(&p.signal_qp, 1)) return false;

  return true;
}

void RdmaTransportAdapter::destroy_peer_qps(RdmaPeer& p) {
  for (int q = 0; q < p.num_qps; ++q) {
    if (p.data_qps[q]) {
      ibv_destroy_qp(p.data_qps[q]);
      p.data_qps[q] = nullptr;
    }
  }
  if (p.data_cq) {
    ibv_destroy_cq(p.data_cq);
    p.data_cq = nullptr;
  }
  if (p.signal_qp) {
    ibv_destroy_qp(p.signal_qp);
    p.signal_qp = nullptr;
  }
  if (p.signal_cq) {
    ibv_destroy_cq(p.signal_cq);
    p.signal_cq = nullptr;
  }
  if (p.signal_pool && p.signal_pool->mr) {
    ibv_dereg_mr(p.signal_pool->mr);
  }
  p.signal_pool.reset();
}

// ── connection management ───────────────────────────────────────────────────

RdmaPeerConnectSpec RdmaTransportAdapter::get_connect_init(int peer_rank) {
  RdmaPeerConnectSpec init;
  init.num_qps = static_cast<uint8_t>(config_.num_qps);
  init.remote_lid = lid_;
  memcpy(init.remote_gid_raw.data(), gid_.raw, sizeof(init.remote_gid_raw));
  init.remote_signal_qpn = 0;
  init.local_dev_idx = local_dev_idx_;
  init.local_gpu_idx = local_gpu_idx_;

  if (peer_rank >= 0) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peers_.find(peer_rank);
    if (it == peers_.end()) {
      auto p = std::make_unique<RdmaPeer>();
      p->num_qps = static_cast<uint8_t>(config_.num_qps);
      if (init_peer_qps(*p)) {
        p->qps_created = true;
        for (int i = 0; i < p->num_qps; ++i)
          init.remote_data_qpns[i] = p->data_qps[i]->qp_num;
        init.remote_signal_qpn = p->signal_qp ? p->signal_qp->qp_num : 0;
        peers_[peer_rank] = std::move(p);
      }
    } else if (it->second->qps_created) {
      for (int i = 0; i < it->second->num_qps; ++i)
        init.remote_data_qpns[i] = it->second->data_qps[i]->qp_num;
      init.remote_signal_qpn =
          it->second->signal_qp ? it->second->signal_qp->qp_num : 0;
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
    if (it != peers_.end() && !it->second->qps_created) {
      peers_.erase(it);
    }
  }

  RdmaPeer* p = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peers_.find(rank);
    if (it != peers_.end() && it->second->qps_created) {
      p = it->second.get();
    }
  }

  if (!p) {
    auto np = std::make_unique<RdmaPeer>();
    np->num_qps = remote.num_qps > 0 ? remote.num_qps
                                     : static_cast<uint8_t>(config_.num_qps);
    if (!init_peer_qps(*np)) return false;
    np->qps_created = true;
    {
      std::lock_guard<std::mutex> lk(mu_);
      peers_[rank] = std::move(np);
      p = peers_[rank].get();
    }
  }

  p->remote_lid = remote.remote_lid;
  memcpy(p->remote_gid.raw, remote.remote_gid_raw.data(),
         sizeof(p->remote_gid.raw));

  p->remote_signal_qpn = remote.remote_signal_qpn;
  memcpy(p->remote_data_qpns, remote.remote_data_qpns,
         sizeof(p->remote_data_qpns));

  if (!qps_to_rtr(p->data_qps, p->num_qps, remote)) return false;
  if (!qps_to_rts(p->data_qps, p->num_qps)) return false;

  {
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
    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                IBV_QP_MIN_RNR_TIMER;
    if (ibv_modify_qp(p->signal_qp, &attr, flags) != 0) return false;
  }
  if (!qps_to_rts(&p->signal_qp, 1)) return false;

  if (!p->signal_pool && !init_signal_pool(*p)) return false;

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

  if (msize <= static_cast<uint32_t>(config_.chunk_size_kb) * 1024) {
    int chosen = (p.last_qp_.load(std::memory_order_relaxed) + 1) % p.num_qps;
    p.last_qp_.store(chosen, std::memory_order_relaxed);
    p.cached_qp_valid_.store(true, std::memory_order_release);
    p.consecutive_cached_bytes_.store(msize, std::memory_order_release);
    return chosen;
  }

  int a = p.last_qp_.load(std::memory_order_relaxed);
  int b = (a + 1) % p.num_qps;
  int chosen = (p.qp_state[a].ewma_rtt_ns < p.qp_state[b].ewma_rtt_ns) ? a : b;
  p.last_qp_.store(chosen, std::memory_order_relaxed);
  p.cached_qp_valid_.store(true, std::memory_order_release);
  p.consecutive_cached_bytes_.store(msize, std::memory_order_release);
  return chosen;
}

// ── send path (enqueue to jring) ─────────────────────────────────────────────

unsigned RdmaTransportAdapter::put_async(int rank, void* local_ptr,
                                         uint32_t local_buf_id,
                                         void* remote_ptr,
                                         uint32_t remote_buf_id, size_t len,
                                         unsigned comm_rid) {
  if (!has_put_path(rank) || len == 0) return 0;

  uint64_t raddr = 0;
  uint32_t rkey = 0;
  uint32_t lkey = 0;

  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = mr_map_.find(local_buf_id);
    if (it != mr_map_.end()) lkey = it->second->lkey;
  }
  if (lkey == 0) return 0;

  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peers_.find(rank);
    if (it == peers_.end() || !it->second->put_ready) return 0;
    if (remote_buf_id != 0) {
      auto rit = it->second->remote_buffers.find(remote_buf_id);
      if (rit != it->second->remote_buffers.end()) {
        raddr = rit->second.addr;
        rkey = rit->second.rkey;
      }
    }
  }
  if (remote_ptr && raddr == 0) {
    raddr = reinterpret_cast<uint64_t>(remote_ptr);
  }
  if (raddr == 0 || rkey == 0) return 0;

  RingElem e{comm_rid, rank, Kind::DataPut, local_ptr, remote_ptr,
             local_buf_id, remote_buf_id, len, 0, raddr, rkey, lkey};
  if (!enqueue_elem(send_ring_, e, stop_)) return 0;
  return 1;
}

unsigned RdmaTransportAdapter::signal_async(int rank, uint64_t tag,
                                            unsigned comm_rid) {
  if (!has_put_path(rank)) return 0;

  RingElem e{comm_rid, rank, Kind::Signal, nullptr, nullptr,
             0, 0, 0, tag, 0, 0, 0};
  if (!enqueue_elem(send_ring_, e, stop_)) return 0;
  return 1;
}

// ── recv path (enqueue to jring) ─────────────────────────────────────────────

unsigned RdmaTransportAdapter::wait_async(int rank, uint64_t expected_tag,
                                          std::optional<WaitTarget> target,
                                          unsigned comm_rid) {
  if (!has_wait_path(rank)) return 0;

  RingElem e{comm_rid, rank, Kind::SignalWait, nullptr, nullptr,
             0, 0, 0, expected_tag, 0, 0, 0};
  if (!enqueue_elem(recv_ring_, e, stop_)) return 0;
  return 1;
}

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

// ── send_worker ──────────────────────────────────────────────────────────────

void RdmaTransportAdapter::send_worker() {
  RingElem e;
  while (!stop_.load(std::memory_order_acquire)) {
    if (jring_sc_dequeue_bulk(send_ring_, &e, 1, nullptr) != 1) {
      std::this_thread::yield();
      continue;
    }

    if (e.kind == Kind::DataPut) {
      // Look up peer
      RdmaPeer* p = nullptr;
      {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = peers_.find(e.peer_rank);
        if (it != peers_.end() && it->second->put_ready) p = it->second.get();
      }
      if (!p) {
        publish_completion(e.comm_rid, true);
        continue;
      }

      uint32_t cs = static_cast<uint32_t>(config_.chunk_size_kb) * 1024;
      ChunkResult ck;
      if (e.len <= cs) {
        ck.count = 1;
        ck.chunk_size = static_cast<uint32_t>(e.len);
        ck.last_size = static_cast<uint32_t>(e.len);
      } else {
        ck = chunk_split(e.len);
      }

      uint32_t send_id = static_cast<uint32_t>(
          next_send_id_.fetch_add(1, std::memory_order_relaxed));
      {
        std::lock_guard<std::mutex> lk(pending_mu_);
        pending_sends_.try_emplace(send_id, e.comm_rid, ck.count);
      }

      bool failed = false;
      size_t off = 0;
      for (uint32_t ci = 0; ci < ck.count; ++ci) {
        uint32_t sz = (ci + 1 == ck.count) ? ck.last_size : ck.chunk_size;
        int q = select_qp(*p, sz);

        // Back-pressure: wait for inflight WRs to drop
        while (p->qp_state[q].unacked_wrs.load(std::memory_order_acquire) + 1 >
               kMaxInflightWrs) {
          if (stop_.load(std::memory_order_acquire)) {
            failed = true;
            break;
          }
          std::unique_lock<std::mutex> lk(cv_mu_);
          cv_.wait_for(lk, kCvTimeout, [&] {
            return p->qp_state[q].unacked_wrs.load(std::memory_order_acquire) +
                           1 <=
                       kMaxInflightWrs ||
                   stop_.load(std::memory_order_acquire);
          });
        }
        if (failed) break;

        uint64_t wr_id = (static_cast<uint64_t>(send_id) << 32) | ci;

        ibv_sge sge = {};
        sge.addr = reinterpret_cast<uint64_t>(static_cast<uint8_t*>(e.local_ptr) + off);
        sge.length = sz;
        sge.lkey = e.local_lkey;

        ibv_send_wr wr = {};
        wr.wr_id = wr_id;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = e.remote_addr + off;
        wr.wr.rdma.rkey = e.remote_rkey;

        ibv_send_wr* bad = nullptr;
        int rc = ibv_post_send(p->data_qps[q], &wr, &bad);
        if (rc != 0) {
          ibv_qp_attr qattr;
          ibv_qp_init_attr iattr;
          ibv_query_qp(p->data_qps[q], &qattr, IBV_QP_STATE, &iattr);
          fprintf(stderr,
                  "[send_worker] POST_FAILED ci=%u qp=%d qpn=%u state=%d errno=%d\n",
                  ci, q, p->data_qps[q]->qp_num, qattr.qp_state, errno);
          {
            std::lock_guard<std::mutex> lk(pending_mu_);
            auto it = pending_sends_.find(send_id);
            if (it != pending_sends_.end()) {
              it->second.completed_chunks.store(
                  ck.count, std::memory_order_release);
            }
          }
          publish_completion(e.comm_rid, true);
          failed = true;
          break;
        }

        p->qp_state[q].unacked_wrs.fetch_add(1, std::memory_order_relaxed);
        p->qp_state[q].last_send_ns.store(now_ns(), std::memory_order_release);
        off += sz;
      }

      if (failed) {
        // Completion already published above
        continue;
      }
    } else if (e.kind == Kind::Signal) {
      // Look up peer
      RdmaPeer* p = nullptr;
      {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = peers_.find(e.peer_rank);
        if (it != peers_.end() && it->second->put_ready) p = it->second.get();
      }
      if (!p) {
        publish_completion(e.comm_rid, true);
        continue;
      }

      uint32_t send_id = static_cast<uint32_t>(
          next_send_id_.fetch_add(1, std::memory_order_relaxed));
      {
        std::lock_guard<std::mutex> lk(pending_mu_);
        pending_sends_.try_emplace(send_id, e.comm_rid, 1);
      }

      uint8_t msg_id =
          p->next_msg_id.fetch_add(1, std::memory_order_relaxed) & kMsgIdMask;

      uint32_t payload = static_cast<uint32_t>(msg_id);
      ibv_sge sge = {};
      sge.addr = reinterpret_cast<uint64_t>(&payload);
      sge.length = sizeof(payload);

      ibv_send_wr wr = {};
      wr.wr_id = static_cast<uint64_t>(send_id) << 32;
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.opcode = IBV_WR_SEND;
      wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;

      ibv_send_wr* bad = nullptr;
      if (ibv_post_send(p->signal_qp, &wr, &bad) != 0) {
        fprintf(stderr, "[send_worker] SIGNAL_POST_FAILED rank=%d msg_id=%u\n",
                e.peer_rank, msg_id);
        {
          std::lock_guard<std::mutex> lk(pending_mu_);
          pending_sends_.erase(send_id);
        }
        publish_completion(e.comm_rid, true);
        continue;
      }
    } else {
      // Unknown kind → fail immediately
      publish_completion(e.comm_rid, true);
    }
  }

  // Drain remaining on shutdown
  RingElem drain;
  while (jring_mc_dequeue_bulk(send_ring_, &drain, 1, nullptr) == 1)
    publish_completion(drain.comm_rid, true);
}

// ── recv_worker ──────────────────────────────────────────────────────────────

void RdmaTransportAdapter::recv_worker() {
  RingElem e;
  while (!stop_.load(std::memory_order_acquire)) {
    if (jring_sc_dequeue_bulk(recv_ring_, &e, 1, nullptr) != 1) {
      std::this_thread::yield();
      continue;
    }

    if (e.kind != Kind::SignalWait) {
      publish_completion(e.comm_rid, true);
      continue;
    }

    // Look up peer
    RdmaPeer* p = nullptr;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = peers_.find(e.peer_rank);
      if (it != peers_.end()) p = it->second.get();
    }
    if (!p) {
      publish_completion(e.comm_rid, true);
      continue;
    }

    unsigned idx =
        p->next_expected_dispatch.fetch_add(1, std::memory_order_relaxed) &
        kMsgIdMask;
    p->trackers[idx].wait_comm_rid.store(e.comm_rid,
                                          std::memory_order_release);
  }

  // Drain remaining on shutdown
  RingElem drain;
  while (jring_mc_dequeue_bulk(recv_ring_, &drain, 1, nullptr) == 1)
    publish_completion(drain.comm_rid, true);
}

// ── polling ──────────────────────────────────────────────────────────────────

void RdmaTransportAdapter::poll_loop() {
  while (!stop_.load(std::memory_order_acquire)) {
    std::vector<std::pair<int, RdmaPeer*>> active;
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (auto& [rank, p] : peers_) {
        if (p->data_cq || p->signal_cq) active.push_back({rank, p.get()});
      }
    }

    bool any = false;
    for (auto& [rank, p] : active) {
      if (p->data_cq &&
          poll_cq_set(*p, rank, p->data_cq, p->data_qps, p->num_qps))
        any = true;
      if (p->signal_cq && poll_signal_cq(*p, rank)) any = true;
      check_dispatch(*p, rank);
    }
    if (!any) {
      std::this_thread::yield();
    }
  }
}

bool RdmaTransportAdapter::poll_cq_set(RdmaPeer& p, int rank, ibv_cq* cq,
                                       ibv_qp* const* qps, int qp_count) {
  ibv_wc wc[16];
  bool any = false;
  int n;
  while ((n = ibv_poll_cq(cq, 16, wc)) > 0) {
    for (int i = 0; i < n; ++i) {
      any = true;
      uint32_t send_id = static_cast<uint32_t>(wc[i].wr_id >> 32);

      if (wc[i].status != IBV_WC_SUCCESS) {
        int qp = find_qp_idx(qps, qp_count, wc[i].qp_num);
        if (qp >= 0)
          p.qp_state[qp].unacked_wrs.fetch_sub(1, std::memory_order_relaxed);

        PendingSend* ps = nullptr;
        {
          std::lock_guard<std::mutex> lk(pending_mu_);
          auto it = pending_sends_.find(send_id);
          if (it != pending_sends_.end()) ps = &it->second;
        }
        if (ps) {
          // Mark all chunks as completed to force publication
          ps->completed_chunks.store(ps->total_chunks,
                                      std::memory_order_release);
          publish_completion(ps->comm_rid, true);
          {
            std::lock_guard<std::mutex> lk(pending_mu_);
            pending_sends_.erase(send_id);
          }
        }
        cv_.notify_all();
        continue;
      }

      int qp = find_qp_idx(qps, qp_count, wc[i].qp_num);
      if (qp >= 0 &&
          (wc[i].opcode == IBV_WC_RDMA_WRITE || wc[i].opcode == IBV_WC_SEND)) {
        uint64_t send_ns =
            p.qp_state[qp].last_send_ns.load(std::memory_order_acquire);
        if (send_ns != 0) {
          uint64_t rtt_ns = now_ns() - send_ns;
          p.qp_state[qp].ewma_rtt_ns =
              kEwmaAlpha * static_cast<double>(rtt_ns) +
              (1.0 - kEwmaAlpha) * p.qp_state[qp].ewma_rtt_ns;
        }
      }
      if (qp >= 0)
        p.qp_state[qp].unacked_wrs.fetch_sub(1, std::memory_order_relaxed);

      // Look up pending send and track chunk completion
      {
        std::lock_guard<std::mutex> lk(pending_mu_);
        auto it = pending_sends_.find(send_id);
        if (it != pending_sends_.end()) {
          uint32_t done = it->second.completed_chunks.fetch_add(
              1, std::memory_order_acq_rel) + 1;
          if (done >= it->second.total_chunks) {
            publish_completion(it->second.comm_rid, false);
            pending_sends_.erase(it);
          }
        }
      }
      cv_.notify_all();
    }
  }
  return any;
}

bool RdmaTransportAdapter::poll_signal_cq(RdmaPeer& p, int rank) {
  ibv_wc wc;
  bool any = false;
  while (ibv_poll_cq(p.signal_cq, 1, &wc) > 0) {
    any = true;
    if (wc.status != IBV_WC_SUCCESS) {
      if (wc.opcode == IBV_WC_RECV) {
        (void)repost_signal_recv(p);
      }
      uint32_t send_id = static_cast<uint32_t>(wc.wr_id >> 32);
      {
        std::lock_guard<std::mutex> lk(pending_mu_);
        auto it = pending_sends_.find(send_id);
        if (it != pending_sends_.end()) {
          publish_completion(it->second.comm_rid, true);
          pending_sends_.erase(it);
        }
      }
      cv_.notify_all();
      continue;
    }

    switch (wc.opcode) {
      case IBV_WC_SEND: {
        // Signal send completed successfully
        uint32_t send_id = static_cast<uint32_t>(wc.wr_id >> 32);
        {
          std::lock_guard<std::mutex> lk(pending_mu_);
          auto it = pending_sends_.find(send_id);
          if (it != pending_sends_.end()) {
            publish_completion(it->second.comm_rid, false);
            pending_sends_.erase(it);
          }
        }
        cv_.notify_all();
        break;
      }

      case IBV_WC_RECV: {
        (void)repost_signal_recv(p);
        uint8_t msg_id =
            *reinterpret_cast<uint8_t*>(p.signal_pool->buffer.data());
        auto& t = p.trackers[msg_id];
        t.completed.store(true, std::memory_order_release);
        check_dispatch(p, rank);
        break;
      }

      default:
        break;
    }
  }
  return any;
}

void RdmaTransportAdapter::check_dispatch(RdmaPeer& p, int rank) {
  (void)rank;
  unsigned expected = p.next_expected_dispatch.load(std::memory_order_acquire);
  while (p.dispatch_cursor < expected) {
    unsigned idx = p.dispatch_cursor & kMsgIdMask;
    auto& t = p.trackers[idx];

    if (!t.completed.load(std::memory_order_acquire)) break;
    unsigned wait_rid = t.wait_comm_rid.load(std::memory_order_acquire);
    if (wait_rid == 0) break;

    publish_completion(wait_rid, false);

    t.completed.store(false, std::memory_order_release);
    t.wait_comm_rid.store(0, std::memory_order_release);
    p.dispatch_cursor++;
    cv_.notify_all();
    expected = p.next_expected_dispatch.load(std::memory_order_acquire);
  }
}

// ── chunk splitting ──────────────────────────────────────────────────────────

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
    r.last_size = static_cast<uint32_t>(
        len - static_cast<uint64_t>(r.chunk_size) * (n - 1));
  } else {
    r.last_size = static_cast<uint32_t>(len);
  }
  return r;
}

}  // namespace Transport
}  // namespace UKernel
