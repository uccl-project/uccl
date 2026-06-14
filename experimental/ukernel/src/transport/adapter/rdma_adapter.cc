#include "rdma_adapter.h"
#include "../communicator.h"
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
constexpr size_t kInitMrCapacity = 256;
constexpr size_t kInitPeerCapacity = 8;

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
  if (!send_ring_) {
    ibv_dealloc_pd(pd_);
    pd_ = nullptr;
    ctx_handle_.reset();
    ctx_ = nullptr;
    return;
  }

  // Allocate lock-free MR table
  mr_table_capacity_ = kInitMrCapacity;
  mr_table_.reset(new std::atomic<ibv_mr*>[kInitMrCapacity]());

  // Allocate peer table
  peer_capacity_ = kInitPeerCapacity;
  peer_table_.reset(new std::atomic<RdmaPeer*>[kInitPeerCapacity]());
  peer_owners_.reset(new std::unique_ptr<RdmaPeer>[kInitPeerCapacity]());

  // Allocate pending ring
  pending_ring_.reset(new PendingSlot[kRingSize]());

  stop_.store(false, std::memory_order_release);
  send_worker_ = std::thread([this] { send_worker(); });
  poll_thread_ = std::thread([this] { poll_loop(); });
}

RdmaTransportAdapter::~RdmaTransportAdapter() {
  stop_.store(true, std::memory_order_release);
  cv_.notify_all();
  if (send_worker_.joinable()) send_worker_.join();
  if (poll_thread_.joinable()) poll_thread_.join();

  // Destroy peers
  for (size_t r = 0; r < peer_capacity_; ++r) {
    if (peer_owners_[r]) {
      peer_table_[r].store(nullptr, std::memory_order_release);
      destroy_peer_qps(*peer_owners_[r]);
      peer_owners_[r].reset();
    }
  }

  // Deregister all MRs
  for (uint32_t id : registered_ids_) {
    ibv_mr* mr = mr_table_[id].load(std::memory_order_acquire);
    if (mr) {
      mr_table_[id].store(nullptr, std::memory_order_release);
      ibv_dereg_mr(mr);
    }
  }
  registered_ids_.clear();

  if (send_ring_) {
    free(send_ring_);
    send_ring_ = nullptr;
  }

  if (pd_) {
    ibv_dealloc_pd(pd_);
    pd_ = nullptr;
  }
}

// ── table sizing helpers ────────────────────────────────────────────────────

std::atomic<RdmaTransportAdapter::RdmaPeer*>&
RdmaTransportAdapter::ensure_peer_slot(int rank) {
  size_t idx = static_cast<size_t>(rank);
  if (idx < peer_capacity_) return peer_table_[idx];

  std::lock_guard<std::mutex> lk(peer_resize_mu_);
  if (idx < peer_capacity_) return peer_table_[idx];  // double-check

  size_t new_cap = std::max(idx + 1, peer_capacity_ * 2);
  auto new_table = std::make_unique<std::atomic<RdmaPeer*>[]>(new_cap);
  auto new_owners = std::make_unique<std::unique_ptr<RdmaPeer>[]>(new_cap);

  for (size_t i = 0; i < peer_capacity_; ++i) {
    new_table[i].store(peer_table_[i].load(std::memory_order_acquire),
                       std::memory_order_relaxed);
    new_owners[i] = std::move(peer_owners_[i]);
  }

  peer_table_ = std::move(new_table);
  peer_owners_ = std::move(new_owners);
  peer_capacity_ = new_cap;
  return peer_table_[idx];
}

void RdmaTransportAdapter::ensure_mr_slot(uint32_t id) {
  if (id < mr_table_capacity_) return;

  std::lock_guard<std::mutex> lk(mr_reg_mu_);
  if (id < mr_table_capacity_) return;  // double-check

  size_t new_cap = std::max<size_t>(id + 1, mr_table_capacity_ * 2);
  auto new_table = std::make_unique<std::atomic<ibv_mr*>[]>(new_cap);

  for (size_t i = 0; i < mr_table_capacity_; ++i) {
    new_table[i].store(mr_table_[i].load(std::memory_order_acquire),
                       std::memory_order_relaxed);
  }

  mr_table_ = std::move(new_table);
  mr_table_capacity_ = new_cap;
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

bool RdmaTransportAdapter::init_signal_pool(RdmaPeer& p) {
  auto pool = std::make_unique<RecvPool>();
  // tag_buf is already zero-initialized, no resize needed
  pool->mr =
      ibv_reg_mr(pd_, &pool->tag_buf, sizeof(uint64_t), IBV_ACCESS_LOCAL_WRITE);
  if (!pool->mr) return false;

  pool->sges.resize(1);
  pool->wrs.resize(1);
  pool->sges[0].addr = reinterpret_cast<uint64_t>(&pool->tag_buf);
  pool->sges[0].length = sizeof(uint64_t);
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
  // Clean up remote buffer table heap allocations
  for (size_t i = 0; i < p.remote_buf_capacity_; ++i) {
    p.remote_buffer_table_[i].store(nullptr, std::memory_order_release);
    p.remote_buf_owners_[i].reset();
  }

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
    RdmaPeer* p = peer_table_[static_cast<size_t>(peer_rank)].load(
        std::memory_order_acquire);
    if (!p || !p->qps_created) {
      // Create a new peer
      auto np = std::make_unique<RdmaPeer>();
      np->num_qps = static_cast<uint8_t>(config_.num_qps);
      if (init_peer_qps(*np)) {
        np->qps_created = true;
        for (int i = 0; i < np->num_qps; ++i)
          init.remote_data_qpns[i] = np->data_qps[i]->qp_num;
        init.remote_signal_qpn = np->signal_qp ? np->signal_qp->qp_num : 0;

        auto& slot = ensure_peer_slot(peer_rank);
        size_t idx = static_cast<size_t>(peer_rank);
        peer_owners_[idx] = std::move(np);
        slot.store(peer_owners_[idx].get(), std::memory_order_release);
      }
    } else {
      for (int i = 0; i < p->num_qps; ++i)
        init.remote_data_qpns[i] = p->data_qps[i]->qp_num;
      init.remote_signal_qpn = p->signal_qp ? p->signal_qp->qp_num : 0;
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
  if (rank < 0 || static_cast<size_t>(rank) >= peer_capacity_) return false;
  RdmaPeer* p =
      peer_table_[static_cast<size_t>(rank)].load(std::memory_order_acquire);
  return p != nullptr && p->put_ready;
}

bool RdmaTransportAdapter::has_wait_path(int rank) const {
  if (rank < 0 || static_cast<size_t>(rank) >= peer_capacity_) return false;
  RdmaPeer* p =
      peer_table_[static_cast<size_t>(rank)].load(std::memory_order_acquire);
  return p != nullptr && p->wait_ready;
}

bool RdmaTransportAdapter::setup_peer_path(int rank,
                                           RdmaPeerConnectSpec const& remote) {
  // Check if already fully ready
  {
    if (static_cast<size_t>(rank) < peer_capacity_) {
      RdmaPeer* p = peer_table_[static_cast<size_t>(rank)].load(
          std::memory_order_acquire);
      if (p && p->put_ready && p->wait_ready) return true;
    }
  }

  // Ensure slot exists
  auto& slot = ensure_peer_slot(rank);
  size_t idx = static_cast<size_t>(rank);

  RdmaPeer* p = slot.load(std::memory_order_acquire);
  if (p && p->qps_created && p->put_ready && p->wait_ready) return true;

  if (!p || !p->qps_created) {
    // Remove stale peer if exists
    if (peer_owners_[idx]) {
      slot.store(nullptr, std::memory_order_release);
      destroy_peer_qps(*peer_owners_[idx]);
      peer_owners_[idx].reset();
      p = nullptr;
    }

    auto np = std::make_unique<RdmaPeer>();
    np->num_qps = remote.num_qps > 0 ? remote.num_qps
                                     : static_cast<uint8_t>(config_.num_qps);
    if (!init_peer_qps(*np)) return false;
    np->qps_created = true;
    peer_owners_[idx] = std::move(np);
    p = peer_owners_[idx].get();
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

  // Publish peer to lock-free table (after QP setup complete)
  slot.store(p, std::memory_order_release);

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

unsigned RdmaTransportAdapter::send_put_async(int rank, void* local_ptr,
                                              uint32_t local_buf_id,
                                              void* remote_ptr,
                                              uint32_t remote_buf_id,
                                              size_t len, unsigned comm_rid) {
  if (!has_put_path(rank) || len == 0) return 0;

  // Lock-free local MR lookup (Task 1)
  uint32_t lkey = 0;
  if (local_buf_id < mr_table_capacity_) {
    ibv_mr* mr = mr_table_[local_buf_id].load(std::memory_order_acquire);
    if (mr) lkey = mr->lkey;
  }
  if (lkey == 0) return 0;

  // Lock-free peer + remote buffer lookup (Task 2)
  RdmaPeer* p =
      peer_table_[static_cast<size_t>(rank)].load(std::memory_order_acquire);
  if (!p || !p->put_ready) return 0;

  uint64_t raddr = 0;
  uint32_t rkey = 0;
  if (remote_buf_id != 0 && remote_buf_id < p->remote_buf_capacity_) {
    RemoteBufInfo* info =
        p->remote_buffer_table_[remote_buf_id].load(std::memory_order_acquire);
    if (info) {
      raddr = info->addr;
      rkey = info->rkey;
    }
  }
  if (remote_ptr && raddr == 0) {
    raddr = reinterpret_cast<uint64_t>(remote_ptr);
  }
  if (raddr == 0 || rkey == 0) return 0;

  RingElem e{comm_rid,   rank,         Kind::DataPut, local_ptr,
             remote_ptr, local_buf_id, remote_buf_id, len,
             0,          raddr,        rkey,          lkey};
  if (!enqueue_elem(send_ring_, e, stop_)) return 0;
  return 1;
}

unsigned RdmaTransportAdapter::send_signal_async(int rank, uint64_t tag,
                                                 unsigned comm_rid) {
  if (!has_put_path(rank)) return 0;

  RingElem e{comm_rid, rank, Kind::Signal, nullptr, nullptr, 0,
             0,        0,    tag,          0,       0,       0};
  if (!enqueue_elem(send_ring_, e, stop_)) return 0;
  return 1;
}

// ── recv path (enqueue to jring) ─────────────────────────────────────────────

unsigned RdmaTransportAdapter::wait_signal_async(
    int rank, uint64_t /*expected_tag*/, std::optional<WaitTarget> /*target*/,
    unsigned comm_rid) {
  if (!has_wait_path(rank)) return 0;
  // SignalWait path is handled by poll_loop pushing tags to Communicator.
  publish_completion(comm_rid, true);
  return 1;
}

// ── memory registration ─────────────────────────────────────────────────────

bool RdmaTransportAdapter::register_memory(uint32_t buf_id, void* ptr,
                                           size_t len) {
  if (!pd_ || !ptr || len == 0) return false;

  ensure_mr_slot(buf_id);

  // Check if already registered (fast path)
  {
    ibv_mr* existing = mr_table_[buf_id].load(std::memory_order_acquire);
    if (existing) return true;
  }

  ibv_mr* mr = ibv_reg_mr(pd_, ptr, len,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_READ);
  if (!mr) return false;

  mr_table_[buf_id].store(mr, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lk(mr_reg_mu_);
    registered_ids_.insert(buf_id);
  }
  return true;
}

void RdmaTransportAdapter::deregister_memory(uint32_t buf_id) {
  if (buf_id >= mr_table_capacity_) return;

  ibv_mr* mr = mr_table_[buf_id].exchange(nullptr, std::memory_order_acq_rel);
  if (mr) {
    ibv_dereg_mr(mr);
    std::lock_guard<std::mutex> lk(mr_reg_mu_);
    registered_ids_.erase(buf_id);
  }
}

bool RdmaTransportAdapter::is_memory_registered(uint32_t buf_id) const {
  if (buf_id >= mr_table_capacity_) return false;
  return mr_table_[buf_id].load(std::memory_order_acquire) != nullptr;
}

uint32_t RdmaTransportAdapter::get_memory_rkey(uint32_t buf_id) const {
  if (buf_id >= mr_table_capacity_) return 0;
  ibv_mr* mr = mr_table_[buf_id].load(std::memory_order_acquire);
  return mr ? mr->rkey : 0;
}

void RdmaTransportAdapter::register_remote_buffer(int rank, uint32_t buf_id,
                                                  uint64_t addr,
                                                  uint32_t rkey) {
  if (rank < 0 || static_cast<size_t>(rank) >= peer_capacity_) return;

  RdmaPeer* p =
      peer_table_[static_cast<size_t>(rank)].load(std::memory_order_acquire);
  if (!p) return;

  // Resize remote buffer table if needed (under peer's internal mutex)
  if (buf_id >= p->remote_buf_capacity_) {
    std::lock_guard<std::mutex> lk(p->remote_buf_mu_);
    if (buf_id >= p->remote_buf_capacity_) {
      size_t new_cap =
          std::max<size_t>(buf_id + 1, p->remote_buf_capacity_ * 2);
      if (new_cap < 64) new_cap = 64;

      auto new_table = std::make_unique<std::atomic<RemoteBufInfo*>[]>(new_cap);
      auto new_owners =
          std::make_unique<std::unique_ptr<RemoteBufInfo>[]>(new_cap);

      for (size_t i = 0; i < p->remote_buf_capacity_; ++i) {
        new_table[i].store(
            p->remote_buffer_table_[i].load(std::memory_order_acquire),
            std::memory_order_relaxed);
        new_owners[i] = std::move(p->remote_buf_owners_[i]);
      }

      p->remote_buffer_table_ = std::move(new_table);
      p->remote_buf_owners_ = std::move(new_owners);
      p->remote_buf_capacity_ = new_cap;
    }
  }

  // Allocate stable RemoteBufInfo (never moved after publication)
  auto info = std::make_unique<RemoteBufInfo>();
  info->addr = addr;
  info->rkey = rkey;
  p->remote_buffer_table_[buf_id].store(info.get(), std::memory_order_release);
  p->remote_buf_owners_[buf_id] = std::move(info);
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
      // Lock-free peer lookup (Task 2)
      RdmaPeer* p = nullptr;
      if (static_cast<size_t>(e.peer_rank) < peer_capacity_) {
        p = peer_table_[static_cast<size_t>(e.peer_rank)].load(
            std::memory_order_acquire);
      }
      if (!p || !p->put_ready) {
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

      // Allocate send_id and acquire ring slot (Task 3)
      uint32_t send_id = static_cast<uint32_t>(
          next_send_id_.fetch_add(1, std::memory_order_relaxed));
      uint32_t slot_idx = send_id % kRingSize;
      PendingSlot& slot = pending_ring_[slot_idx];

      // CAS-acquire the slot: 0 → send_id
      uint32_t expected = 0;
      while (!slot.send_id.compare_exchange_weak(expected, send_id,
                                                 std::memory_order_acquire)) {
        if (stop_.load(std::memory_order_acquire)) {
          publish_completion(e.comm_rid, true);
          goto next_elem;
        }
        expected = 0;
        std::this_thread::yield();
      }
      slot.comm_rid = e.comm_rid;
      slot.total_chunks = ck.count;
      slot.completed_chunks.store(0, std::memory_order_release);

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

        // Encode QP index into wr_id (Task 4)
        uint64_t wr_id = (static_cast<uint64_t>(send_id) << 32) |
                         (static_cast<uint64_t>(q) << 16) | ci;

        ibv_sge sge = {};
        sge.addr = reinterpret_cast<uint64_t>(
            static_cast<uint8_t*>(e.local_ptr) + off);
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
                  "[send_worker] POST_FAILED ci=%u qp=%d qpn=%u state=%d "
                  "errno=%d\n",
                  ci, q, p->data_qps[q]->qp_num, qattr.qp_state, errno);

          // Mark slot as complete and free (error path)
          slot.completed_chunks.store(ck.count, std::memory_order_release);
          publish_completion(e.comm_rid, true);
          // Only free if we still own the slot
          uint32_t exp = send_id;
          slot.send_id.compare_exchange_strong(exp, 0,
                                               std::memory_order_release);
          failed = true;
          break;
        }

        p->qp_state[q].unacked_wrs.fetch_add(1, std::memory_order_relaxed);
        p->qp_state[q].last_send_ns.store(now_ns(), std::memory_order_release);
        off += sz;
      }

      if (failed) {
        continue;
      }
    } else if (e.kind == Kind::Signal) {
      // Lock-free peer lookup (Task 2)
      RdmaPeer* p = nullptr;
      if (static_cast<size_t>(e.peer_rank) < peer_capacity_) {
        p = peer_table_[static_cast<size_t>(e.peer_rank)].load(
            std::memory_order_acquire);
      }
      if (!p || !p->put_ready) {
        publish_completion(e.comm_rid, true);
        continue;
      }

      uint32_t send_id = static_cast<uint32_t>(
          next_send_id_.fetch_add(1, std::memory_order_relaxed));
      uint32_t slot_idx = send_id % kRingSize;
      PendingSlot& slot = pending_ring_[slot_idx];

      // CAS-acquire slot for signal send
      uint32_t expected = 0;
      while (!slot.send_id.compare_exchange_weak(expected, send_id,
                                                 std::memory_order_acquire)) {
        if (stop_.load(std::memory_order_acquire)) {
          publish_completion(e.comm_rid, true);
          goto next_elem;
        }
        expected = 0;
        std::this_thread::yield();
      }
      slot.comm_rid = e.comm_rid;
      slot.total_chunks = 1;
      slot.completed_chunks.store(0, std::memory_order_release);

      uint64_t payload = e.tag;
      ibv_sge sge = {};
      sge.addr = reinterpret_cast<uint64_t>(&payload);
      sge.length = sizeof(payload);

      ibv_send_wr wr = {};
      // Signal QP: encode with reserved qp_idx 0xFF (Task 4)
      wr.wr_id = (static_cast<uint64_t>(send_id) << 32) | (0xFFULL << 16);
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.opcode = IBV_WR_SEND;
      wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;

      ibv_send_wr* bad = nullptr;
      if (ibv_post_send(p->signal_qp, &wr, &bad) != 0) {
        fprintf(stderr, "[send_worker] SIGNAL_POST_FAILED rank=%d tag=%lu\n",
                e.peer_rank, (unsigned long)payload);
        slot.completed_chunks.store(1, std::memory_order_release);
        publish_completion(e.comm_rid, true);
        uint32_t exp = send_id;
        slot.send_id.compare_exchange_strong(exp, 0, std::memory_order_release);
        continue;
      }
    } else {
      // Unknown kind → fail immediately
      publish_completion(e.comm_rid, true);
    }
  next_elem:;
  }

  // Drain remaining on shutdown
  RingElem drain;
  while (jring_mc_dequeue_bulk(send_ring_, &drain, 1, nullptr) == 1)
    publish_completion(drain.comm_rid, true);
}

// ── polling ──────────────────────────────────────────────────────────────────

void RdmaTransportAdapter::poll_loop() {
  while (!stop_.load(std::memory_order_acquire)) {
    bool any = false;

    // Iterate peer_table_ directly without lock (Task 5)
    for (size_t r = 0; r < peer_capacity_; ++r) {
      RdmaPeer* p = peer_table_[r].load(std::memory_order_acquire);
      if (!p) continue;
      int rank = static_cast<int>(r);
      if (p->data_cq && poll_cq_set(*p, rank)) any = true;
      if (p->signal_cq && poll_signal_cq(*p, rank)) any = true;
    }

    if (!any) {
      std::this_thread::yield();
    }
  }
}

bool RdmaTransportAdapter::poll_cq_set(RdmaPeer& p, int rank) {
  ibv_wc wc[16];
  bool any = false;
  int n;
  while ((n = ibv_poll_cq(p.data_cq, 16, wc)) > 0) {
    for (int i = 0; i < n; ++i) {
      any = true;
      uint32_t send_id = static_cast<uint32_t>(wc[i].wr_id >> 32);

      // Decode QP index directly from wr_id (Task 4)
      int qp = static_cast<int>((wc[i].wr_id >> 16) & 0xFFFF);

      if (wc[i].status != IBV_WC_SUCCESS) {
        if (qp >= 0 && qp < p.num_qps)
          p.qp_state[qp].unacked_wrs.fetch_sub(1, std::memory_order_relaxed);

        // Error path: mark slot as complete
        uint32_t slot_idx = send_id % kRingSize;
        PendingSlot& slot = pending_ring_[slot_idx];
        // Only process if we own this slot
        if (slot.send_id.load(std::memory_order_acquire) == send_id) {
          slot.completed_chunks.store(slot.total_chunks,
                                      std::memory_order_release);
          publish_completion(slot.comm_rid, true);
          uint32_t exp = send_id;
          slot.send_id.compare_exchange_strong(exp, 0,
                                               std::memory_order_release);
        }
        cv_.notify_all();
        continue;
      }

      if (qp >= 0 && qp < p.num_qps &&
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
      if (qp >= 0 && qp < p.num_qps)
        p.qp_state[qp].unacked_wrs.fetch_sub(1, std::memory_order_relaxed);

      // Track chunk completion via lock-free ring buffer (Task 3)
      uint32_t slot_idx = send_id % kRingSize;
      PendingSlot& slot = pending_ring_[slot_idx];

      // Verify slot ownership
      if (slot.send_id.load(std::memory_order_acquire) != send_id) {
        cv_.notify_all();
        continue;
      }

      uint32_t done =
          slot.completed_chunks.fetch_add(1, std::memory_order_acq_rel) + 1;
      if (done == slot.total_chunks) {
        // Try to claim completion: CAS send_id back to 0
        uint32_t expected = send_id;
        if (slot.send_id.compare_exchange_strong(expected, 0,
                                                 std::memory_order_release)) {
          publish_completion(slot.comm_rid, false);
        }
        // If CAS failed, error path or another thread already handled it
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
      uint32_t slot_idx = send_id % kRingSize;
      PendingSlot& slot = pending_ring_[slot_idx];
      if (slot.send_id.load(std::memory_order_acquire) == send_id) {
        slot.completed_chunks.store(slot.total_chunks,
                                    std::memory_order_release);
        publish_completion(slot.comm_rid, true);
        uint32_t exp = send_id;
        slot.send_id.compare_exchange_strong(exp, 0, std::memory_order_release);
      }
      cv_.notify_all();
      continue;
    }

    switch (wc.opcode) {
      case IBV_WC_SEND: {
        // Signal send completed successfully
        uint32_t send_id = static_cast<uint32_t>(wc.wr_id >> 32);
        uint32_t slot_idx = send_id % kRingSize;
        PendingSlot& slot = pending_ring_[slot_idx];
        if (slot.send_id.load(std::memory_order_acquire) == send_id) {
          uint32_t done =
              slot.completed_chunks.fetch_add(1, std::memory_order_acq_rel) + 1;
          if (done >= slot.total_chunks) {
            uint32_t expected = send_id;
            if (slot.send_id.compare_exchange_strong(
                    expected, 0, std::memory_order_release)) {
              publish_completion(slot.comm_rid, false);
            }
          }
        }
        cv_.notify_all();
        break;
      }

      case IBV_WC_RECV: {
        (void)repost_signal_recv(p);
        uint64_t tag = p.signal_pool->tag_buf;

        // Push directly to Communicator for tag matching.
        if (comm_) comm_->on_signal_received(rank, tag);
        break;
      }

      default:
        break;
    }
  }
  return any;
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
