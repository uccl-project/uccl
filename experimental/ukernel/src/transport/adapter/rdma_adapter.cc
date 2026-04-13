#include "rdma_adapter.h"
#include "collective/rdma/transport.h"
#include "util/gpu_rt.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

namespace UKernel {
namespace Transport {

namespace {
constexpr auto kRdmaAsyncRetryTimeout = std::chrono::seconds(30);
constexpr uint32_t kRdmaRetryYieldThreshold = 64;
constexpr uint32_t kRdmaRetrySleepThreshold = 256;
constexpr auto kRdmaRetryBackoff = std::chrono::microseconds(2);

inline void backoff_retry(uint32_t retries) {
  if (retries < kRdmaRetryYieldThreshold) return;
  if (retries < kRdmaRetrySleepThreshold) {
    std::this_thread::yield();
    return;
  }
  std::this_thread::sleep_for(kRdmaRetryBackoff);
}
}  // namespace

RdmaTransportAdapter::RdmaTransportAdapter(int local_gpu_idx, int world_size,
                                           RdmaTransportConfig config)
    : local_gpu_idx_(local_gpu_idx) {
  int num_engines = static_cast<int>(::ucclParamNUM_ENGINES());
  if (num_engines <= 0) {
    std::cerr << "[ERROR] Invalid RDMA num_engines=" << num_engines
              << std::endl;
    return;
  }
  if (config.num_engines > 0 && config.num_engines != num_engines) {
    std::cout << "[WARN] RDMA engine count mismatch: requested "
              << config.num_engines << ", runtime " << num_engines
              << ". Using runtime value." << std::endl;
  }

  endpoint_ = std::make_unique<::uccl::RDMAEndpoint>(num_engines);
  endpoint_->initialize_resources(num_engines * world_size);

  int dev_idx = endpoint_->get_best_dev_idx(local_gpu_idx_);
  if (dev_idx < 0) {
    std::cerr << "[ERROR] RDMA get_best_dev_idx failed for local gpu "
              << local_gpu_idx_ << std::endl;
    endpoint_.reset();
    return;
  }
  endpoint_->initialize_engine_by_dev(dev_idx, true);

  request_slots_ = std::make_unique<PendingRequestSlot[]>(kRequestSlotCount);
  for (uint32_t i = 0; i < kRequestSlotCount; ++i) {
    request_slots_[i].request = std::make_unique<::uccl::ucclRequest>();
    request_slots_[i].in_use = false;
    request_slots_[i].generation = 1;
    request_slots_[i].state = RequestState::Init;
    request_slots_[i].failed = false;
  }
}

RdmaTransportAdapter::~RdmaTransportAdapter() { endpoint_.reset(); }

uint16_t RdmaTransportAdapter::get_p2p_listen_port(int dev_idx) const {
  if (!endpoint_ || dev_idx < 0) return 0;
  return endpoint_->get_p2p_listen_port(dev_idx);
}

std::string RdmaTransportAdapter::get_p2p_listen_ip(int dev_idx) const {
  if (!endpoint_ || dev_idx < 0) return {};
  return endpoint_->get_p2p_listen_ip(dev_idx);
}

int RdmaTransportAdapter::get_best_dev_idx(int gpu_idx) const {
  if (!endpoint_) return -1;
  return endpoint_->get_best_dev_idx(gpu_idx);
}

bool RdmaTransportAdapter::has_send_peer(int peer_rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peer_contexts_.find(peer_rank);
  return it != peer_contexts_.end() && it->second.send_flow != nullptr;
}

bool RdmaTransportAdapter::has_recv_peer(int peer_rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peer_contexts_.find(peer_rank);
  return it != peer_contexts_.end() && it->second.recv_flow != nullptr;
}

bool RdmaTransportAdapter::ensure_peer(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_peer(spec.peer_rank)) return true;
  if (!std::holds_alternative<RdmaPeerConnectSpec>(spec.detail)) return false;

  auto const& rdma = std::get<RdmaPeerConnectSpec>(spec.detail);
  if (spec.type == PeerConnectType::Connect) {
    return connect_to_peer(spec.peer_rank, rdma.remote_ip, rdma.remote_port,
                           rdma.local_dev_idx, rdma.local_gpu_idx,
                           rdma.remote_dev_idx, rdma.remote_gpu_idx);
  }
  return accept_from_peer(spec.peer_rank, rdma.remote_ip, rdma.remote_dev_idx,
                          rdma.remote_gpu_idx, rdma.remote_port);
}

bool RdmaTransportAdapter::is_memory_registered(uint64_t mr_id) const {
  std::lock_guard<std::mutex> lk(mu_);
  return mr_id_to_mhandle_.find(mr_id) != mr_id_to_mhandle_.end();
}

bool RdmaTransportAdapter::register_memory(uint64_t mr_id, void* ptr,
                                           size_t len) {
  if (!endpoint_ || ptr == nullptr || len == 0) return false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (mr_id_to_mhandle_.find(mr_id) != mr_id_to_mhandle_.end()) return true;
  }

  ::uccl::Mhandle* mhandle = nullptr;
  int dev_idx = endpoint_->get_best_dev_idx(local_gpu_idx_);
  if (dev_idx < 0) return false;
  if (endpoint_->uccl_regmr(dev_idx, ptr, len, 0, &mhandle) != 0 || !mhandle) {
    return false;
  }

  std::lock_guard<std::mutex> lk(mu_);
  mr_id_to_mhandle_[mr_id] = mhandle;
  return true;
}

void RdmaTransportAdapter::deregister_memory(uint64_t mr_id) {
  if (!endpoint_) return;
  std::lock_guard<std::mutex> lk(mu_);
  auto it = mr_id_to_mhandle_.find(mr_id);
  if (it == mr_id_to_mhandle_.end()) return;
  endpoint_->uccl_deregmr(it->second);
  mr_id_to_mhandle_.erase(it);
}

bool RdmaTransportAdapter::connect_to_peer(int peer_rank, std::string remote_ip,
                                           uint16_t remote_port,
                                           int local_dev_idx, int local_gpu_idx,
                                           int remote_dev_idx,
                                           int remote_gpu_idx) {
  if (has_send_peer(peer_rank) && has_recv_peer(peer_rank)) return true;
  if (!endpoint_) return false;
  if (local_dev_idx < 0 || remote_dev_idx < 0 || remote_port == 0) return false;

  if (!has_send_peer(peer_rank)) {
    ::uccl::ConnID conn_id =
        endpoint_->uccl_connect(local_dev_idx, local_gpu_idx, remote_dev_idx,
                                remote_gpu_idx, remote_ip, remote_port);
    if (conn_id.context == nullptr) return false;

    std::lock_guard<std::mutex> lk(mu_);
    auto& ctx = peer_contexts_[peer_rank];
    ctx.peer_rank = peer_rank;
    ctx.send_flow = static_cast<::uccl::UcclFlow*>(conn_id.context);
    ctx.remote_ip = remote_ip;
    ctx.remote_dev_idx = remote_dev_idx;
    ctx.remote_gpu_idx = remote_gpu_idx;
  }

  if (!has_recv_peer(peer_rank)) {
    if (!accept_from_peer(peer_rank, remote_ip, remote_dev_idx, remote_gpu_idx,
                          remote_port)) {
      return false;
    }
  }
  return has_send_peer(peer_rank) && has_recv_peer(peer_rank);
}

bool RdmaTransportAdapter::accept_from_peer(
    int peer_rank, std::string const& expected_remote_ip,
    int expected_remote_dev_idx, int expected_remote_gpu_idx,
    uint16_t expected_remote_port) {
  if (has_send_peer(peer_rank) && has_recv_peer(peer_rank)) return true;
  if (!endpoint_) return false;

  if (!has_recv_peer(peer_rank)) {
    int dev_idx = endpoint_->get_best_dev_idx(local_gpu_idx_);
    if (dev_idx < 0) return false;

    int listen_fd = endpoint_->get_p2p_listen_fd(dev_idx);
    if (listen_fd < 0) return false;

    std::string remote_ip;
    int remote_dev = 0;
    int remote_gpuidx = 0;
    ::uccl::ConnID conn_id =
        endpoint_->uccl_accept(dev_idx, listen_fd, local_gpu_idx_, remote_ip,
                               &remote_dev, &remote_gpuidx);
    if (conn_id.context == nullptr) return false;

    if ((!expected_remote_ip.empty() && remote_ip != expected_remote_ip) ||
        (expected_remote_dev_idx >= 0 &&
         remote_dev != expected_remote_dev_idx) ||
        (expected_remote_gpu_idx >= 0 &&
         remote_gpuidx != expected_remote_gpu_idx)) {
      endpoint_->discard_conn(conn_id);
      return false;
    }

    std::lock_guard<std::mutex> lk(mu_);
    auto& ctx = peer_contexts_[peer_rank];
    ctx.peer_rank = peer_rank;
    ctx.recv_flow = static_cast<::uccl::UcclFlow*>(conn_id.context);
    ctx.remote_ip = remote_ip;
    ctx.remote_dev_idx = remote_dev;
    ctx.remote_gpu_idx = remote_gpuidx;
  }

  if (!has_send_peer(peer_rank)) {
    if (expected_remote_port == 0) return false;
    int local_dev_idx = endpoint_->get_best_dev_idx(local_gpu_idx_);
    if (local_dev_idx < 0) return false;
    if (!connect_to_peer(peer_rank, expected_remote_ip, expected_remote_port,
                         local_dev_idx, local_gpu_idx_, expected_remote_dev_idx,
                         expected_remote_gpu_idx)) {
      return false;
    }
  }

  return has_send_peer(peer_rank) && has_recv_peer(peer_rank);
}

RdmaTransportAdapter::PendingRequestSlot*
RdmaTransportAdapter::try_acquire_request_slot(unsigned* out_request_id) {
  if (out_request_id == nullptr || !request_slots_) return nullptr;
  std::lock_guard<std::mutex> lk(mu_);
  for (uint32_t n = 0; n < kRequestSlotCount; ++n) {
    uint32_t idx =
        request_alloc_cursor_.fetch_add(1, std::memory_order_relaxed) &
        kRequestSlotMask;
    auto& slot = request_slots_[idx];
    if (slot.in_use) continue;
    slot.in_use = true;
    slot.failed = false;
    slot.state = RequestState::Init;
    if (slot.generation == 0) slot.generation = 1;
    if (!slot.request) {
      slot.request = std::make_unique<::uccl::ucclRequest>();
    }
    *out_request_id = make_request_id(idx, slot.generation);
    return &slot;
  }
  return nullptr;
}

RdmaTransportAdapter::PendingRequestSlot*
RdmaTransportAdapter::resolve_request_slot_locked(unsigned request_id) {
  if (request_id == 0 || !request_slots_) return nullptr;
  uint32_t generation = request_generation(request_id);
  if (generation == 0) return nullptr;
  uint32_t idx = request_slot_index(request_id);
  auto& slot = request_slots_[idx];
  if (!slot.in_use || slot.generation != generation) return nullptr;
  return &slot;
}

void RdmaTransportAdapter::release_request_slot_locked(unsigned request_id) {
  PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
  if (!slot) return;
  if (slot->state == RequestState::Posted) return;
  slot->in_use = false;
  slot->failed = false;
  slot->state = RequestState::Init;
  uint32_t next_gen = slot->generation + 1;
  slot->generation = (next_gen == 0) ? 1 : next_gen;
}

unsigned RdmaTransportAdapter::send_async(
    int peer_rank, void* local_ptr, size_t len, uint64_t local_mr_id,
    std::optional<RemoteSlice> remote_hint, BounceBufferProvider bounce_provider) {
  void* send_ptr = local_ptr;
  uint64_t send_mr_id = local_mr_id;
  if (bounce_provider) {
    BounceBufferInfo info = bounce_provider(len);
    if (info.ptr != nullptr) {
      GPU_RT_CHECK(gpuMemcpy(info.ptr, local_ptr, len, gpuMemcpyDeviceToHost));
      send_ptr = info.ptr;
      send_mr_id = info.mr_id;
    }
  }

  uint64_t remote_mr_id = 0;
  RemoteSlice const* remote_slice_ptr = nullptr;
  if (remote_hint.has_value()) {
    remote_mr_id = remote_hint->mem_id;
    remote_slice_ptr = &(*remote_hint);
  }

  unsigned request_id = 0;
  if (try_acquire_request_slot(&request_id) == nullptr) return 0;
  int ret = send_async_rdma(peer_rank, send_ptr, len, send_mr_id, remote_mr_id,
                            request_id, remote_slice_ptr);
  if (ret != 0) {
    std::lock_guard<std::mutex> lk(mu_);
    release_request_slot_locked(request_id);
    return 0;
  }
  return request_id;
}

unsigned RdmaTransportAdapter::recv_async(int peer_rank, void* local_ptr,
                                          size_t len, uint64_t local_mr_id,
                                          BounceBufferProvider bounce_provider) {
  void* recv_ptr = local_ptr;
  uint64_t recv_mr_id = local_mr_id;
  if (bounce_provider) {
    BounceBufferInfo info = bounce_provider(len);
    if (info.ptr != nullptr) {
      recv_ptr = info.ptr;
      recv_mr_id = info.mr_id;
    }
  }

  unsigned request_id = 0;
  if (try_acquire_request_slot(&request_id) == nullptr) return 0;
  int ret = recv_async_rdma(peer_rank, recv_ptr, len, recv_mr_id, request_id);
  if (ret != 0) {
    std::lock_guard<std::mutex> lk(mu_);
    release_request_slot_locked(request_id);
    return 0;
  }
  return request_id;
}

bool RdmaTransportAdapter::request_failed(unsigned id) {
  std::lock_guard<std::mutex> lk(mu_);
  PendingRequestSlot* slot = resolve_request_slot_locked(id);
  return slot != nullptr && slot->failed;
}

int RdmaTransportAdapter::send_async_rdma(int peer_rank, void* local_ptr,
                                          size_t len, uint64_t local_mr_id,
                                          uint64_t remote_mr_id,
                                          uint64_t request_id,
                                          RemoteSlice const* remote_slice) {
  ::uccl::UcclFlow* flow = nullptr;
  ::uccl::Mhandle* local_mh = nullptr;
  ::uccl::ucclRequest* ureq = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto peer_it = peer_contexts_.find(peer_rank);
    if (peer_it == peer_contexts_.end()) return -1;
    flow = peer_it->second.send_flow;

    auto mh_it = mr_id_to_mhandle_.find(local_mr_id);
    if (mh_it != mr_id_to_mhandle_.end()) local_mh = mh_it->second;

    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot == nullptr || slot->state != RequestState::Init) return -1;
    if (!slot->request) slot->request = std::make_unique<::uccl::ucclRequest>();
    std::memset(slot->request.get(), 0, sizeof(::uccl::ucclRequest));
    slot->failed = false;
    ureq = slot->request.get();
  }

  if (!flow || !local_mh || !ureq) return -1;

  int ret = -1;
  if (remote_slice != nullptr && remote_mr_id != 0 &&
      remote_slice->has_write_hint()) {
    ::uccl::FifoItem slot_item{};
    slot_item.addr = remote_slice->write.addr;
    slot_item.rkey = remote_slice->write.key;
    slot_item.size = remote_slice->write.capacity == 0
                         ? static_cast<uint32_t>(len)
                         : remote_slice->write.capacity;
    slot_item.rid = remote_slice->write.rid;
    slot_item.engine_offset = remote_slice->write.engine_offset;
    slot_item.nmsgs = 1;
    slot_item.idx = 1;
    std::memset(slot_item.padding, 0, sizeof(slot_item.padding));
    ret = endpoint_->uccl_write_async(flow, local_mh, local_ptr, len, slot_item,
                                      ureq);
  }

  if (ret != 0) {
    auto deadline = std::chrono::steady_clock::now() + kRdmaAsyncRetryTimeout;
    uint32_t retries = 0;
    while (std::chrono::steady_clock::now() < deadline) {
      ret = endpoint_->uccl_send_async(flow, local_mh, local_ptr, len, ureq);
      if (ret == 0) break;
      backoff_retry(retries++);
    }
  }
  if (ret != 0) {
    std::lock_guard<std::mutex> lk(mu_);
    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot != nullptr) {
      slot->failed = true;
      slot->state = RequestState::Failed;
    }
    return -1;
  }

  std::lock_guard<std::mutex> lk(mu_);
  PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
  if (slot == nullptr) return -1;
  slot->failed = false;
  slot->state = RequestState::Posted;
  return 0;
}

int RdmaTransportAdapter::recv_async_rdma(int peer_rank, void* local_ptr,
                                          size_t len, uint64_t local_mr_id,
                                          uint64_t request_id) {
  ::uccl::UcclFlow* flow = nullptr;
  ::uccl::Mhandle* local_mh = nullptr;
  ::uccl::ucclRequest* ureq = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto peer_it = peer_contexts_.find(peer_rank);
    if (peer_it == peer_contexts_.end()) return -1;
    flow = peer_it->second.recv_flow;

    auto mh_it = mr_id_to_mhandle_.find(local_mr_id);
    if (mh_it != mr_id_to_mhandle_.end()) local_mh = mh_it->second;

    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot == nullptr || slot->state != RequestState::Init) return -1;
    if (!slot->request) slot->request = std::make_unique<::uccl::ucclRequest>();
    std::memset(slot->request.get(), 0, sizeof(::uccl::ucclRequest));
    slot->failed = false;
    ureq = slot->request.get();
  }

  if (!flow || !local_mh || !ureq) return -1;

  ::uccl::Mhandle* mh_array[1] = {local_mh};
  void* data_array[1] = {local_ptr};
  int size_array[1] = {static_cast<int>(len)};

  auto deadline = std::chrono::steady_clock::now() + kRdmaAsyncRetryTimeout;
  int ret = -1;
  uint32_t retries = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    ret = endpoint_->uccl_recv_async(flow, mh_array, data_array, size_array, 1,
                                     ureq);
    if (ret == 0) break;
    backoff_retry(retries++);
  }
  if (ret != 0) {
    std::lock_guard<std::mutex> lk(mu_);
    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot != nullptr) {
      slot->failed = true;
      slot->state = RequestState::Failed;
    }
    return -1;
  }

  std::lock_guard<std::mutex> lk(mu_);
  PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
  if (slot == nullptr) return -1;
  slot->failed = false;
  slot->state = RequestState::Posted;
  return 0;
}

bool RdmaTransportAdapter::poll_completion(unsigned request_id) {
  ::uccl::ucclRequest* req = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot == nullptr) return true;
    if (slot->state == RequestState::Completed ||
        slot->state == RequestState::Failed) {
      return true;
    }
    if (slot->state != RequestState::Posted) return false;
    if (!endpoint_ || !slot->request || !slot->request->context) {
      slot->failed = true;
      slot->state = RequestState::Failed;
      return true;
    }
    req = slot->request.get();
  }

  if (!endpoint_->uccl_poll_ureq_once(req)) return false;

  std::lock_guard<std::mutex> lk(mu_);
  PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
  if (slot == nullptr) return true;
  if (slot->state == RequestState::Posted) {
    slot->state = RequestState::Completed;
  }
  return true;
}

bool RdmaTransportAdapter::wait_completion(unsigned request_id) {
  uint32_t retries = 0;
  while (true) {
    if (poll_completion(request_id)) return true;
    backoff_retry(retries++);
  }
}

void RdmaTransportAdapter::release_request(unsigned request_id) {
  std::lock_guard<std::mutex> lk(mu_);
  release_request_slot_locked(request_id);
}

}  // namespace Transport
}  // namespace UKernel
