#include "uccl_adapter.h"
#include "collective/rdma/transport.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

namespace UKernel {
namespace Transport {

namespace {
constexpr auto kUcclAsyncRetryTimeout = std::chrono::seconds(30);
constexpr uint32_t kUcclRetryYieldThreshold = 64;
constexpr uint32_t kUcclRetrySleepThreshold = 256;
constexpr auto kUcclRetryBackoff = std::chrono::microseconds(2);

inline void backoff_retry(uint32_t retries) {
  if (retries < kUcclRetryYieldThreshold) return;
  if (retries < kUcclRetrySleepThreshold) {
    std::this_thread::yield();
    return;
  }
  std::this_thread::sleep_for(kUcclRetryBackoff);
}
}  // namespace

UcclTransportAdapter::UcclTransportAdapter(int local_gpu_idx, int world_size,
                                           UcclTransportConfig config)
    : local_gpu_idx_(local_gpu_idx) {
  int num_engines = static_cast<int>(::ucclParamNUM_ENGINES());
  if (num_engines <= 0) {
    std::cerr << "[ERROR] Invalid UCCL num_engines=" << num_engines
              << std::endl;
    return;
  }
  if (config.num_engines > 0 && config.num_engines != num_engines) {
    std::cout << "[WARN] UCCL engine count mismatch: requested "
              << config.num_engines << ", runtime " << num_engines
              << ". Using runtime value." << std::endl;
  }
  endpoint_ = std::make_unique<::uccl::RDMAEndpoint>(num_engines);
  endpoint_->initialize_resources(num_engines * world_size);

  int dev_idx = endpoint_->get_best_dev_idx(local_gpu_idx_);
  if (dev_idx < 0) {
    std::cerr << "[ERROR] UCCL get_best_dev_idx failed for local gpu "
              << local_gpu_idx_ << std::endl;
    endpoint_.reset();
    return;
  }
  endpoint_->initialize_engine_by_dev(dev_idx, true);

  request_slots_ = std::make_unique<PendingRequestSlot[]>(kRequestSlotCount);
  for (uint32_t i = 0; i < kRequestSlotCount; ++i) {
    request_slots_[i].request = std::make_unique<::uccl::ucclRequest>();
    request_slots_[i].kind = PendingRequestSlot::Kind::DataPut;
    request_slots_[i].state = RequestState::Free;
    request_slots_[i].generation = 1;
    request_slots_[i].control_tag = 0;
    request_slots_[i].expected_signal_tag = 0;
    request_slots_[i].control_mhandle = nullptr;
    request_slots_[i].failed = false;
  }
}

UcclTransportAdapter::~UcclTransportAdapter() {
  if (endpoint_ && request_slots_) {
    for (uint32_t i = 0; i < kRequestSlotCount; ++i) {
      if (request_slots_[i].control_mhandle != nullptr) {
        endpoint_->uccl_deregmr(request_slots_[i].control_mhandle);
        request_slots_[i].control_mhandle = nullptr;
      }
    }
  }
  endpoint_.reset();
}

UcclTransportAdapter::PendingRequestSlot*
UcclTransportAdapter::try_acquire_request_slot(unsigned* out_request_id) {
  if (out_request_id == nullptr || !request_slots_) return nullptr;
  std::lock_guard<std::mutex> lk(mu_);
  for (uint32_t n = 0; n < kRequestSlotCount; ++n) {
    uint32_t idx =
        request_alloc_cursor_.fetch_add(1, std::memory_order_relaxed) &
        kRequestSlotMask;
    auto& slot = request_slots_[idx];
    if (slot.state != RequestState::Free) continue;
    slot.state = RequestState::Reserved;
    slot.kind = PendingRequestSlot::Kind::DataPut;
    slot.control_tag = 0;
    slot.expected_signal_tag = 0;
    slot.failed = false;
    if (slot.generation == 0) slot.generation = 1;
    if (!slot.request) {
      slot.request = std::make_unique<::uccl::ucclRequest>();
    }
    *out_request_id = make_request_id(idx, slot.generation);
    return &slot;
  }
  return nullptr;
}

UcclTransportAdapter::PendingRequestSlot*
UcclTransportAdapter::resolve_request_slot_locked(unsigned request_id) {
  if (request_id == 0 || !request_slots_) return nullptr;
  uint32_t generation = request_generation(request_id);
  if (generation == 0) return nullptr;
  uint32_t idx = request_slot_index(request_id);
  auto& slot = request_slots_[idx];
  if (slot.generation != generation) return nullptr;
  if (slot.state == RequestState::Free) return nullptr;
  return &slot;
}

void UcclTransportAdapter::release_request_slot_locked(unsigned request_id) {
  PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
  if (slot == nullptr) return;
  if (slot->state == RequestState::InFlight) return;
  slot->control_tag = 0;
  slot->expected_signal_tag = 0;
  slot->kind = PendingRequestSlot::Kind::DataPut;
  slot->failed = false;
  slot->state = RequestState::Free;
  uint32_t next_gen = slot->generation + 1;
  slot->generation = (next_gen == 0) ? 1 : next_gen;
}

uint16_t UcclTransportAdapter::get_p2p_listen_port(int dev_idx) const {
  if (!endpoint_ || dev_idx < 0) return 0;
  return endpoint_->get_p2p_listen_port(dev_idx);
}

std::string UcclTransportAdapter::get_p2p_listen_ip(int dev_idx) const {
  if (!endpoint_ || dev_idx < 0) return {};
  return endpoint_->get_p2p_listen_ip(dev_idx);
}

int UcclTransportAdapter::get_best_dev_idx(int gpu_idx) const {
  if (!endpoint_) return -1;
  return endpoint_->get_best_dev_idx(gpu_idx);
}

bool UcclTransportAdapter::has_send_peer(int peer_rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peer_contexts_.find(peer_rank);
  return it != peer_contexts_.end() && it->second.send_flow != nullptr;
}

bool UcclTransportAdapter::has_recv_peer(int peer_rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peer_contexts_.find(peer_rank);
  return it != peer_contexts_.end() && it->second.recv_flow != nullptr;
}

bool UcclTransportAdapter::ensure_put_path(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_send_peer(spec.peer_rank)) return true;
  if (spec.type != PeerConnectType::Connect) return false;
  if (!std::holds_alternative<UcclPeerConnectSpec>(spec.detail)) return false;
  auto const& u = std::get<UcclPeerConnectSpec>(spec.detail);
  if (!connect_to_peer(spec.peer_rank, u.remote_ip, u.remote_port,
                       u.local_dev_idx, u.local_gpu_idx, u.remote_dev_idx,
                       u.remote_gpu_idx)) {
    return false;
  }
  return has_send_peer(spec.peer_rank);
}

bool UcclTransportAdapter::ensure_wait_path(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_recv_peer(spec.peer_rank)) return true;
  if (spec.type != PeerConnectType::Accept) return false;
  if (!std::holds_alternative<UcclPeerConnectSpec>(spec.detail)) return false;
  auto const& u = std::get<UcclPeerConnectSpec>(spec.detail);
  if (!accept_from_peer(spec.peer_rank, u.remote_ip, u.remote_dev_idx,
                        u.remote_gpu_idx, u.remote_port)) {
    return false;
  }
  return has_recv_peer(spec.peer_rank);
}

bool UcclTransportAdapter::is_memory_registered(uint32_t buffer_id) const {
  std::lock_guard<std::mutex> lk(mu_);
  return buffer_id_to_mhandle_.find(buffer_id) != buffer_id_to_mhandle_.end();
}

bool UcclTransportAdapter::connect_to_peer(int peer_rank, std::string remote_ip,
                                           uint16_t remote_port,
                                           int local_dev_idx, int local_gpu_idx,
                                           int remote_dev_idx,
                                           int remote_gpu_idx) {
  if (has_send_peer(peer_rank)) return true;
  if (!endpoint_) return false;
  if (local_dev_idx < 0 || remote_dev_idx < 0 || remote_port == 0) return false;
  ::uccl::ConnID conn_id =
      endpoint_->uccl_connect(local_dev_idx, local_gpu_idx, remote_dev_idx,
                              remote_gpu_idx, remote_ip, remote_port);
  if (conn_id.context == nullptr) {
    std::cerr << "[ERROR] UCCL connect returned null context for peer "
              << peer_rank << std::endl;
    return false;
  }

  std::lock_guard<std::mutex> lk(mu_);
  auto& ctx = peer_contexts_[peer_rank];
  ctx.send_flow = static_cast<::uccl::UcclFlow*>(conn_id.context);
  return true;
}

bool UcclTransportAdapter::accept_from_peer(
    int peer_rank, std::string const& expected_remote_ip,
    int expected_remote_dev_idx, int expected_remote_gpu_idx,
    uint16_t expected_remote_port) {
  (void)expected_remote_port;
  if (has_recv_peer(peer_rank)) return true;
  if (!endpoint_) return false;
  int dev_idx = endpoint_->get_best_dev_idx(local_gpu_idx_);
  if (dev_idx < 0) {
    std::cerr << "[ERROR] UCCL accept: invalid local dev for gpu "
              << local_gpu_idx_ << std::endl;
    return false;
  }
  int listen_fd = endpoint_->get_p2p_listen_fd(dev_idx);
  if (listen_fd < 0) {
    std::cerr << "[ERROR] UCCL accept: invalid listen fd for dev " << dev_idx
              << std::endl;
    return false;
  }
  std::string remote_ip;
  int remote_dev = 0;
  int remote_gpuidx = 0;
  ::uccl::ConnID conn_id =
      endpoint_->uccl_accept(dev_idx, listen_fd, local_gpu_idx_, remote_ip,
                             &remote_dev, &remote_gpuidx);
  if (conn_id.context == nullptr) {
    std::cerr << "[ERROR] UCCL accept returned null context for peer "
              << peer_rank << std::endl;
    return false;
  }

  if ((!expected_remote_ip.empty() && remote_ip != expected_remote_ip) ||
      (expected_remote_dev_idx >= 0 && remote_dev != expected_remote_dev_idx) ||
      (expected_remote_gpu_idx >= 0 && remote_gpuidx != expected_remote_gpu_idx)) {
    std::cerr << "[ERROR] UCCL accept peer mismatch for rank " << peer_rank
              << ": expected ip/dev/gpu=" << expected_remote_ip << "/"
              << expected_remote_dev_idx << "/" << expected_remote_gpu_idx
              << ", got " << remote_ip << "/" << remote_dev << "/"
              << remote_gpuidx << std::endl;
    endpoint_->discard_conn(conn_id);
    return false;
  }

  std::lock_guard<std::mutex> lk(mu_);
  auto& ctx = peer_contexts_[peer_rank];
  ctx.recv_flow = static_cast<::uccl::UcclFlow*>(conn_id.context);
  return true;
}

bool UcclTransportAdapter::register_memory(uint32_t buffer_id, void* ptr,
                                           size_t len) {
  if (!endpoint_ || ptr == nullptr || len == 0) return false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (buffer_id_to_mhandle_.find(buffer_id) != buffer_id_to_mhandle_.end()) {
      return true;
    }
  }

  ::uccl::Mhandle* mhandle = nullptr;
  int dev_idx = endpoint_->get_best_dev_idx(local_gpu_idx_);
  if (dev_idx < 0) return false;
  if (endpoint_->uccl_regmr(dev_idx, ptr, len, 0, &mhandle) != 0 || !mhandle) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lk(mu_);
    buffer_id_to_mhandle_[buffer_id] = mhandle;
  }

  return true;
}

void UcclTransportAdapter::deregister_memory(uint32_t buffer_id) {
  if (!endpoint_) return;
  std::lock_guard<std::mutex> lk(mu_);
  auto it = buffer_id_to_mhandle_.find(buffer_id);
  if (it != buffer_id_to_mhandle_.end()) {
    endpoint_->uccl_deregmr(it->second);
    buffer_id_to_mhandle_.erase(it);
  }
}

unsigned UcclTransportAdapter::put_async(
    int peer_rank, void* local_ptr, uint32_t local_buffer_id,
    void* remote_ptr, uint32_t remote_buffer_id, size_t len) {
  if (!has_put_path(peer_rank)) return 0;

  unsigned request_id = 0;
  PendingRequestSlot* slot = try_acquire_request_slot(&request_id);
  if (slot == nullptr) return 0;
  slot->kind = PendingRequestSlot::Kind::DataPut;
  int ret = send_async_uccl(peer_rank, local_ptr, len, local_buffer_id,
                            remote_buffer_id, request_id);
  if (ret != 0) {
    std::lock_guard<std::mutex> lk(mu_);
    release_request_slot_locked(request_id);
    return 0;
  }
  return request_id;
}

unsigned UcclTransportAdapter::signal_async(int peer_rank, uint64_t tag) {
  if (!has_put_path(peer_rank)) return 0;
  if (!endpoint_) return 0;

  unsigned request_id = 0;
  PendingRequestSlot* slot = try_acquire_request_slot(&request_id);
  if (slot == nullptr) return 0;
  slot->kind = PendingRequestSlot::Kind::SignalSend;
  slot->control_tag = tag;
  slot->expected_signal_tag = 0;

  if (slot->control_mhandle == nullptr) {
    int dev_idx = endpoint_->get_best_dev_idx(local_gpu_idx_);
    if (dev_idx < 0) {
      std::lock_guard<std::mutex> lk(mu_);
      release_request_slot_locked(request_id);
      return 0;
    }
    ::uccl::Mhandle* control_mh = nullptr;
    if (endpoint_->uccl_regmr(dev_idx, &slot->control_tag,
                              sizeof(slot->control_tag), 0,
                              &control_mh) != 0 ||
        control_mh == nullptr) {
      std::lock_guard<std::mutex> lk(mu_);
      release_request_slot_locked(request_id);
      return 0;
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      slot = resolve_request_slot_locked(request_id);
      if (slot == nullptr || slot->state != RequestState::Reserved) {
        endpoint_->uccl_deregmr(control_mh);
        if (slot != nullptr) release_request_slot_locked(request_id);
        return 0;
      }
      slot->control_mhandle = control_mh;
    }
  }

  int ret = send_async_uccl(peer_rank, &slot->control_tag,
                            sizeof(slot->control_tag),
                            /*local_buffer_id=*/0, /*remote_buffer_id=*/0,
                            request_id, slot->control_mhandle);
  if (ret != 0) {
    std::lock_guard<std::mutex> lk(mu_);
    release_request_slot_locked(request_id);
    return 0;
  }
  return request_id;
}

unsigned UcclTransportAdapter::wait_async(int peer_rank, uint64_t expected_tag,
                                          std::optional<WaitTarget> target) {
  if (!has_wait_path(peer_rank)) return 0;
  if (!endpoint_) return 0;

  if (!target.has_value()) {
    unsigned request_id = 0;
    PendingRequestSlot* slot = try_acquire_request_slot(&request_id);
    if (slot == nullptr) return 0;
    slot->kind = PendingRequestSlot::Kind::SignalWait;
    slot->control_tag = 0;
    slot->expected_signal_tag = expected_tag;

    if (slot->control_mhandle == nullptr) {
      int dev_idx = endpoint_->get_best_dev_idx(local_gpu_idx_);
      if (dev_idx < 0) {
        std::lock_guard<std::mutex> lk(mu_);
        release_request_slot_locked(request_id);
        return 0;
      }
      ::uccl::Mhandle* control_mh = nullptr;
      if (endpoint_->uccl_regmr(dev_idx, &slot->control_tag,
                                sizeof(slot->control_tag), 0,
                                &control_mh) != 0 ||
          control_mh == nullptr) {
        std::lock_guard<std::mutex> lk(mu_);
        release_request_slot_locked(request_id);
        return 0;
      }
      {
        std::lock_guard<std::mutex> lk(mu_);
        slot = resolve_request_slot_locked(request_id);
        if (slot == nullptr || slot->state != RequestState::Reserved) {
          endpoint_->uccl_deregmr(control_mh);
          if (slot != nullptr) release_request_slot_locked(request_id);
          return 0;
        }
        slot->control_mhandle = control_mh;
      }
    }

    int ret =
        recv_async_uccl(peer_rank, &slot->control_tag,
                        sizeof(slot->control_tag),
                        /*local_buffer_id=*/0, request_id,
                        slot->control_mhandle);
    if (ret != 0) {
      std::lock_guard<std::mutex> lk(mu_);
      release_request_slot_locked(request_id);
      return 0;
    }
    return request_id;
  }

  unsigned request_id = 0;
  PendingRequestSlot* slot = try_acquire_request_slot(&request_id);
  if (slot == nullptr) return 0;
  slot->kind = PendingRequestSlot::Kind::DataWait;
  int ret = recv_async_uccl(peer_rank, target->local_ptr, target->len,
                            target->local_buffer_id, request_id);
  if (ret != 0) {
    std::lock_guard<std::mutex> lk(mu_);
    release_request_slot_locked(request_id);
    return 0;
  }
  return request_id;
}

bool UcclTransportAdapter::request_failed(unsigned id) {
  std::lock_guard<std::mutex> lk(mu_);
  PendingRequestSlot* slot = resolve_request_slot_locked(id);
  return slot != nullptr && slot->failed;
}

int UcclTransportAdapter::send_async_uccl(int peer_rank, void* local_ptr,
                                           size_t len, uint32_t local_buffer_id,
                                           uint32_t remote_buffer_id,
                                           uint64_t request_id,
                                           ::uccl::Mhandle* local_mh_override) {
  (void)remote_buffer_id;
  ::uccl::UcclFlow* flow = nullptr;
  ::uccl::Mhandle* local_mh = nullptr;
  ::uccl::ucclRequest* ureq = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto peer_it = peer_contexts_.find(peer_rank);
    if (peer_it == peer_contexts_.end()) {
      std::cerr << "[ERROR] UCCL send_async missing peer context for rank "
                << peer_rank << " request " << request_id << std::endl;
      return -1;
    }
    flow = peer_it->second.send_flow;

    if (local_mh_override != nullptr) {
      local_mh = local_mh_override;
    } else {
      auto mh_it = buffer_id_to_mhandle_.find(local_buffer_id);
      if (mh_it != buffer_id_to_mhandle_.end()) {
        local_mh = mh_it->second;
      }
    }

    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot == nullptr || slot->state != RequestState::Reserved) return -1;
    if (!slot->request) {
      slot->request = std::make_unique<::uccl::ucclRequest>();
    }
    std::memset(slot->request.get(), 0, sizeof(::uccl::ucclRequest));
    slot->failed = false;
    ureq = slot->request.get();
  }

  if (!flow || !local_mh || !ureq) {
    std::cerr << "[ERROR] UCCL send_async missing "
              << (!flow ? "send flow" : "memory handle") << " for peer "
              << peer_rank << ", request " << request_id << ", buffer_id "
              << local_buffer_id << ", len " << len << ", ptr " << local_ptr
              << std::endl;
    return -1;
  }

  int ret = -1;
  auto deadline = std::chrono::steady_clock::now() + kUcclAsyncRetryTimeout;
  uint32_t retries = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    ret = endpoint_->uccl_send_async(flow, local_mh, local_ptr, len, ureq);
    if (ret == 0) break;
    backoff_retry(retries++);
  }
  if (ret != 0) {
    std::cerr << "[ERROR] UCCL send submit failed for peer " << peer_rank << ", request "
              << request_id << ", buffer_id " << local_buffer_id << ", len "
              << len << ", ptr " << local_ptr << std::endl;
    std::lock_guard<std::mutex> lk(mu_);
    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot != nullptr) {
      slot->failed = true;
      slot->state = RequestState::Failed;
    }
    return -1;
  }

  {
    std::lock_guard<std::mutex> lk(mu_);
    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot == nullptr) return -1;
    slot->failed = false;
    slot->state = RequestState::InFlight;
  }

  return 0;
}

int UcclTransportAdapter::recv_async_uccl(int peer_rank, void* local_ptr,
                                          size_t len, uint32_t local_buffer_id,
                                          uint64_t request_id,
                                          ::uccl::Mhandle* local_mh_override) {
  ::uccl::UcclFlow* flow = nullptr;
  ::uccl::Mhandle* local_mh = nullptr;
  ::uccl::ucclRequest* ureq = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto peer_it = peer_contexts_.find(peer_rank);
    if (peer_it == peer_contexts_.end()) {
      std::cerr << "[ERROR] UCCL recv_async missing peer context for rank "
                << peer_rank << " request " << request_id << std::endl;
      return -1;
    }
    flow = peer_it->second.recv_flow;

    if (local_mh_override != nullptr) {
      local_mh = local_mh_override;
    } else {
      auto mh_it = buffer_id_to_mhandle_.find(local_buffer_id);
      if (mh_it != buffer_id_to_mhandle_.end()) {
        local_mh = mh_it->second;
      }
    }

    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot == nullptr || slot->state != RequestState::Reserved) return -1;
    if (!slot->request) {
      slot->request = std::make_unique<::uccl::ucclRequest>();
    }
    std::memset(slot->request.get(), 0, sizeof(::uccl::ucclRequest));
    slot->failed = false;
    ureq = slot->request.get();
  }

  if (!flow || !local_mh || !ureq) {
    std::cerr << "[ERROR] UCCL recv_async missing "
              << (!flow ? "recv flow" : "memory handle") << " for peer "
              << peer_rank << ", request " << request_id << ", buffer_id "
              << local_buffer_id << ", len " << len << ", ptr " << local_ptr
              << std::endl;
    return -1;
  }

  ::uccl::Mhandle* mh_array[1] = {local_mh};
  void* data_array[1] = {local_ptr};
  int size_array[1] = {static_cast<int>(len)};

  auto deadline = std::chrono::steady_clock::now() + kUcclAsyncRetryTimeout;
  int ret = -1;
  uint32_t retries = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    ret = endpoint_->uccl_recv_async(flow, mh_array, data_array, size_array, 1,
                                     ureq);
    if (ret == 0) break;
    backoff_retry(retries++);
  }
  if (ret != 0) {
    std::cerr << "[ERROR] UCCL recv_async submit failed for peer " << peer_rank
              << ", request " << request_id << ", buffer_id " << local_buffer_id
              << ", len " << len << ", ptr " << local_ptr << std::endl;
    std::lock_guard<std::mutex> lk(mu_);
    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot != nullptr) {
      slot->failed = true;
      slot->state = RequestState::Failed;
    }
    return -1;
  }

  {
    std::lock_guard<std::mutex> lk(mu_);
    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot == nullptr) return -1;
    slot->failed = false;
    slot->state = RequestState::InFlight;
  }

  return 0;
}

bool UcclTransportAdapter::poll_completion(unsigned request_id) {
  ::uccl::ucclRequest* req = nullptr;
  PendingRequestSlot::Kind kind = PendingRequestSlot::Kind::DataPut;
  uint64_t expected_signal_tag = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot == nullptr) return true;
    if (slot->state == RequestState::Completed ||
        slot->state == RequestState::Failed) {
      return true;
    }
    if (slot->state != RequestState::InFlight) return false;
    if (!endpoint_ || !slot->request || !slot->request->context) {
      slot->failed = true;
      slot->state = RequestState::Failed;
      return true;
    }
    kind = slot->kind;
    expected_signal_tag = slot->expected_signal_tag;
    req = slot->request.get();
  }

  if (!endpoint_->uccl_poll_ureq_once(req)) return false;

  {
    std::lock_guard<std::mutex> lk(mu_);
    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot == nullptr) return true;
    if (slot->state == RequestState::InFlight) {
      bool ok = true;
      if (kind == PendingRequestSlot::Kind::SignalWait) {
        ok = (slot->control_tag == expected_signal_tag);
      }
      slot->failed = !ok;
      slot->state = ok ? RequestState::Completed : RequestState::Failed;
    }
  }
  return true;
}

bool UcclTransportAdapter::wait_completion(unsigned request_id) {
  uint32_t retries = 0;
  while (true) {
    if (poll_completion(request_id)) return true;
    backoff_retry(retries++);
  }
}

void UcclTransportAdapter::release_request(unsigned request_id) {
  std::lock_guard<std::mutex> lk(mu_);
  release_request_slot_locked(request_id);
}

}  // namespace Transport
}  // namespace UKernel
