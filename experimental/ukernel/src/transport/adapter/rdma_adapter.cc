#include "rdma_adapter.h"
#include "p2p/rdma/memory_allocator.h"
#include "p2p/rdma/rdma_endpoint.h"
#include "gpu_rt.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <thread>

namespace UKernel {
namespace Transport {

namespace {
constexpr uint32_t kBootstrapMagic = 0x554B5244;  // "UKRD"
constexpr uint32_t kBootstrapVersion = 2;
constexpr char kDefaultBootstrapIp[] = "127.0.0.1";
constexpr int kWaitSleepUs = 50;

inline char const* choose_local_ip(std::string const& configured_ip) {
  if (!configured_ip.empty()) return configured_ip.c_str();
  if (char const* env_ip = std::getenv("UHM_LOCAL_IP")) {
    if (env_ip[0] != '\0') return env_ip;
  }
  return kDefaultBootstrapIp;
}

}  // namespace

struct RdmaTransportAdapter::BootstrapPayload {
  uint32_t magic = kBootstrapMagic;
  uint32_t version = kBootstrapVersion;
  int32_t src_rank = -1;
  int32_t dst_rank = -1;
  uint32_t port = 0;
  char ip[64] = {};
};

RdmaTransportAdapter::RdmaTransportAdapter(int local_gpu_idx, int local_rank,
                                           int world_size,
                                           RdmaTransportConfig config)
    : local_gpu_idx_(local_gpu_idx),
      local_rank_(local_rank),
      world_size_(world_size),
      config_(std::move(config)) {
  request_slots_ = std::make_unique<PendingRequestSlot[]>(kRequestSlotCount);
  for (uint32_t i = 0; i < kRequestSlotCount; ++i) {
    request_slots_[i].in_use = false;
    request_slots_[i].generation = 1;
    request_slots_[i].state = RequestState::Init;
    request_slots_[i].failed = false;
  }
  initialized_ = initialize_endpoint();
}

RdmaTransportAdapter::~RdmaTransportAdapter() {
  std::lock_guard<std::mutex> lk(mu_);
  if (endpoint_) {
    for (auto& [buffer_id, local_mr] : buffer_id_to_local_mr_) {
      (void)buffer_id;
      if (local_mr.block) endpoint_->deregMem(local_mr.block);
    }
  }
  buffer_id_to_local_mr_.clear();
  pending_peer_oob_.clear();
  ready_peers_.clear();
  endpoint_.reset();
}

bool RdmaTransportAdapter::initialize_endpoint() {
  std::vector<size_t> device_ids =
      RdmaDeviceManager::instance().get_best_dev_idx(local_gpu_idx_);
  if (device_ids.empty()) {
    return false;
  }
  try {
    endpoint_ = std::make_unique<NICEndpoint>(local_gpu_idx_, local_rank_, 0,
                                              true, device_ids);
    return endpoint_ != nullptr;
  } catch (...) {
    endpoint_.reset();
    return false;
  }
}

MemoryType RdmaTransportAdapter::detect_memory_type(void* ptr) {
  if (ptr == nullptr) return MemoryType::HOST;
  gpuPointerAttribute_t attr{};
  gpuError_t err = gpuPointerGetAttributes(&attr, ptr);
  if (err == gpuSuccess && attr.type == gpuMemoryTypeDevice) {
    return MemoryType::GPU;
  }
  return MemoryType::HOST;
}

std::string RdmaTransportAdapter::serialize_bootstrap(
    BootstrapPayload const& payload) {
  auto const* p = reinterpret_cast<uint8_t const*>(&payload);
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (size_t i = 0; i < sizeof(BootstrapPayload); ++i) {
    oss << std::setw(2) << static_cast<unsigned>(p[i]);
  }
  return oss.str();
}

bool RdmaTransportAdapter::deserialize_bootstrap(std::string const& payload,
                                                 BootstrapPayload* out_payload) {
  if (out_payload == nullptr) return false;
  if (payload.size() != sizeof(BootstrapPayload) * 2) return false;
  BootstrapPayload decoded{};
  auto* p = reinterpret_cast<uint8_t*>(&decoded);
  for (size_t i = 0; i < sizeof(BootstrapPayload); ++i) {
    auto byte_str = payload.substr(i * 2, 2);
    char* end = nullptr;
    unsigned long v = std::strtoul(byte_str.c_str(), &end, 16);
    if (end == nullptr || *end != '\0' || v > 0xff) return false;
    p[i] = static_cast<uint8_t>(v);
  }
  if (decoded.magic != kBootstrapMagic ||
      decoded.version != kBootstrapVersion) {
    return false;
  }
  *out_payload = decoded;
  return true;
}

bool RdmaTransportAdapter::build_send_bootstrap(int peer_rank,
                                                std::string* out_payload) {
  if (!initialized_ || !endpoint_ || out_payload == nullptr || peer_rank < 0) {
    return false;
  }
  BootstrapPayload payload{};
  payload.src_rank = local_rank_;
  payload.dst_rank = peer_rank;
  payload.port = endpoint_->get_p2p_listen_port();
  std::snprintf(payload.ip, sizeof(payload.ip), "%s",
                choose_local_ip(config_.local_ip));
  *out_payload = serialize_bootstrap(payload);
  return true;
}

bool RdmaTransportAdapter::build_recv_bootstrap(
    int peer_rank, std::string const& remote_send_payload,
    std::string* out_payload) {
  if (!initialized_ || !endpoint_ || out_payload == nullptr || peer_rank < 0) {
    return false;
  }

  BootstrapPayload remote{};
  if (!deserialize_bootstrap(remote_send_payload, &remote)) return false;
  if (remote.src_rank != peer_rank || remote.dst_rank != local_rank_) {
    return false;
  }

  auto remote_oob =
      std::make_shared<OOBMetaData>(std::string(remote.ip), remote.port);
  {
    std::lock_guard<std::mutex> lk(mu_);
    pending_peer_oob_[peer_rank] = std::move(remote_oob);
  }

  BootstrapPayload ack{};
  ack.src_rank = local_rank_;
  ack.dst_rank = peer_rank;
  ack.port = endpoint_->get_p2p_listen_port();
  std::snprintf(ack.ip, sizeof(ack.ip), "%s", choose_local_ip(config_.local_ip));
  *out_payload = serialize_bootstrap(ack);
  return true;
}

bool RdmaTransportAdapter::finalize_send_bootstrap(
    int peer_rank, std::string const& remote_recv_payload) {
  if (!initialized_ || !endpoint_ || peer_rank < 0) return false;

  BootstrapPayload remote_recv{};
  if (!deserialize_bootstrap(remote_recv_payload, &remote_recv)) return false;
  if (remote_recv.src_rank != peer_rank || remote_recv.dst_rank != local_rank_) {
    return false;
  }

  std::shared_ptr<OOBMetaData> remote_oob;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_peer_oob_.find(peer_rank);
    if (it == pending_peer_oob_.end()) {
      remote_oob =
          std::make_shared<OOBMetaData>(std::string(remote_recv.ip), remote_recv.port);
      pending_peer_oob_[peer_rank] = remote_oob;
    } else {
      remote_oob = it->second;
    }
  }

  endpoint_->add_rank_oob_meta({{static_cast<uint64_t>(peer_rank), remote_oob}});
  int ret = endpoint_->build_connect(static_cast<uint64_t>(peer_rank), true,
                                     config_.connect_timeout_ms);
  if (ret < 0) {
    return false;
  }

  std::lock_guard<std::mutex> lk(mu_);
  ready_peers_.insert(peer_rank);
  return true;
}

void RdmaTransportAdapter::rollback_peer_bootstrap(int peer_rank) {
  if (peer_rank < 0) return;
  std::lock_guard<std::mutex> lk(mu_);
  pending_peer_oob_.erase(peer_rank);
  ready_peers_.erase(peer_rank);
}

bool RdmaTransportAdapter::is_memory_registered(uint32_t buffer_id) const {
  std::lock_guard<std::mutex> lk(mu_);
  return buffer_id_to_local_mr_.find(buffer_id) != buffer_id_to_local_mr_.end();
}

bool RdmaTransportAdapter::register_memory(uint32_t buffer_id, void* ptr,
                                           size_t len) {
  if (!initialized_ || !endpoint_ || ptr == nullptr || len == 0) return false;
  {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      auto it = buffer_registering_.find(buffer_id);
      if (it == buffer_registering_.end() || !it->second) break;
      mr_cv_.wait(lk);
    }
    if (buffer_id_to_local_mr_.find(buffer_id) != buffer_id_to_local_mr_.end()) {
      return true;
    }
    buffer_registering_[buffer_id] = true;
  }

  bool success = false;
  LocalMr local_mr;
  local_mr.block = std::make_shared<RegMemBlock>(ptr, len, detect_memory_type(ptr));
  try {
    success = endpoint_->regMem(local_mr.block);
  } catch (...) {
    success = false;
  }

  std::unique_lock<std::mutex> lk(mu_);
  if (success) {
    buffer_id_to_local_mr_.emplace(buffer_id, std::move(local_mr));
  }
  buffer_registering_.erase(buffer_id);
  lk.unlock();
  mr_cv_.notify_all();
  return success;
}

void RdmaTransportAdapter::deregister_memory(uint32_t buffer_id) {
  std::shared_ptr<RegMemBlock> block;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = buffer_id_to_local_mr_.find(buffer_id);
    if (it == buffer_id_to_local_mr_.end()) return;
    block = it->second.block;
    buffer_id_to_local_mr_.erase(it);
  }
  if (endpoint_ && block) {
    endpoint_->deregMem(block);
  }
}

bool RdmaTransportAdapter::query_memory_rkey(uint32_t buffer_id,
                                             uint32_t* out_rkey) const {
  std::array<uint32_t, kNICContextNumber> rkeys{};
  if (!query_memory_rkeys(buffer_id, &rkeys)) return false;
  for (uint32_t key : rkeys) {
    if (key != 0) {
      if (out_rkey) *out_rkey = key;
      return true;
    }
  }
  return false;
}

bool RdmaTransportAdapter::query_memory_rkeys(
    uint32_t buffer_id,
    std::array<uint32_t, kNICContextNumber>* out_rkeys) const {
  if (out_rkeys == nullptr) return false;
  std::lock_guard<std::mutex> lk(mu_);
  auto it = buffer_id_to_local_mr_.find(buffer_id);
  if (it == buffer_id_to_local_mr_.end() || !it->second.block) return false;
  for (uint32_t ctx = 0; ctx < kNICContextNumber; ++ctx) {
    (*out_rkeys)[ctx] = it->second.block->getKeyByContextID(ctx);
  }
  return true;
}

bool RdmaTransportAdapter::ensure_peer(PeerConnectSpec const& spec) {
  return has_peer(spec.peer_rank);
}

bool RdmaTransportAdapter::has_peer(int peer_rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  return ready_peers_.find(peer_rank) != ready_peers_.end();
}

RdmaTransportAdapter::PendingRequestSlot*
RdmaTransportAdapter::try_acquire_request_slot(unsigned* out_request_id) {
  if (out_request_id == nullptr) return nullptr;
  for (uint32_t attempt = 0; attempt < kRequestSlotCount; ++attempt) {
    uint32_t slot_idx =
        request_alloc_cursor_.fetch_add(1, std::memory_order_relaxed) &
        kRequestSlotMask;
    std::lock_guard<std::mutex> lk(mu_);
    PendingRequestSlot& slot = request_slots_[slot_idx];
    if (slot.in_use) continue;
    slot.in_use = true;
    slot.state = RequestState::Init;
    slot.failed = false;
    slot.request = AdapterRequest{};
    *out_request_id = make_request_id(slot_idx, slot.generation);
    return &slot;
  }
  return nullptr;
}

RdmaTransportAdapter::PendingRequestSlot*
RdmaTransportAdapter::resolve_request_slot_locked(unsigned request_id) {
  uint32_t slot_idx = request_slot_index(request_id);
  if (slot_idx >= kRequestSlotCount) return nullptr;
  PendingRequestSlot& slot = request_slots_[slot_idx];
  if (!slot.in_use || slot.generation != request_generation(request_id)) {
    return nullptr;
  }
  return &slot;
}

void RdmaTransportAdapter::release_request_slot_locked(unsigned request_id) {
  PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
  if (slot == nullptr) return;
  slot->in_use = false;
  slot->failed = false;
  slot->state = RequestState::Init;
  slot->request = AdapterRequest{};
  uint32_t next_gen = slot->generation + 1;
  slot->generation = (next_gen == 0) ? 1 : next_gen;
}

unsigned RdmaTransportAdapter::send_async(
    int peer_rank, void* local_ptr, size_t len, uint32_t local_buffer_id,
    std::optional<RemoteSlice> remote_hint,
    BounceBufferProvider bounce_provider) {
  if (!initialized_ || !endpoint_ || !has_peer(peer_rank)) return 0;

  void* send_ptr = local_ptr;
  uint32_t send_buffer_id = local_buffer_id;
  if (bounce_provider) {
    BounceBufferInfo info = bounce_provider(len);
    if (info.ptr != nullptr) {
      if (detect_memory_type(local_ptr) == MemoryType::GPU) {
        GPU_RT_CHECK(gpuMemcpy(info.ptr, local_ptr, len, gpuMemcpyDeviceToHost));
      } else {
        std::memcpy(info.ptr, local_ptr, len);
      }
      send_ptr = info.ptr;
      send_buffer_id = info.buffer_id;
    }
  }

  unsigned request_id = 0;
  if (try_acquire_request_slot(&request_id) == nullptr) return 0;

  std::shared_ptr<RegMemBlock> base_block;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = buffer_id_to_local_mr_.find(send_buffer_id);
    if (it == buffer_id_to_local_mr_.end() || !it->second.block) {
      release_request_slot_locked(request_id);
      return 0;
    }
    base_block = it->second.block;
  }

  int64_t token = -1;
  RequestKind kind = RequestKind::Send;
  try {
    auto send_mem = std::make_shared<RegMemBlock>(
        send_ptr, len, base_block->mr_array, base_block->type);
    if (remote_hint.has_value() && remote_hint->has_rdma_write_hint()) {
      std::cout << "[INFO] RDMA adapter ignoring one-sided write hint for peer "
                << peer_rank
                << " and using matched send/recv path for request stability"
                << std::endl;
    }
    auto placeholder = std::make_shared<RemoteMemInfo>();
    auto req = std::make_shared<RDMASendRequest>(send_mem, placeholder);
    req->from_rank_id = static_cast<uint32_t>(local_rank_);
    req->to_rank_id = static_cast<uint32_t>(peer_rank);
    req->send_type = SendType::Send;
    token = endpoint_->send(static_cast<uint64_t>(peer_rank), req);
    kind = RequestKind::Send;
  } catch (...) {
    token = -1;
  }

  std::lock_guard<std::mutex> lk(mu_);
  PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
  if (slot == nullptr) return 0;
  if (token < 0) {
    slot->failed = true;
    slot->state = RequestState::Failed;
    return request_id;
  }
  slot->request.kind = kind;
  slot->request.peer_rank = peer_rank;
  slot->request.token = token;
  slot->state = RequestState::Posted;
  slot->failed = false;
  return request_id;
}

unsigned RdmaTransportAdapter::recv_async(int peer_rank, void* local_ptr,
                                          size_t len, uint32_t local_buffer_id,
                                          BounceBufferProvider bounce_provider) {
  if (!initialized_ || !endpoint_ || !has_peer(peer_rank)) return 0;

  void* recv_ptr = local_ptr;
  uint32_t recv_buffer_id = local_buffer_id;
  if (bounce_provider) {
    BounceBufferInfo info = bounce_provider(len);
    if (info.ptr != nullptr) {
      recv_ptr = info.ptr;
      recv_buffer_id = info.buffer_id;
    }
  }

  unsigned request_id = 0;
  if (try_acquire_request_slot(&request_id) == nullptr) return 0;

  std::shared_ptr<RegMemBlock> base_block;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = buffer_id_to_local_mr_.find(recv_buffer_id);
    if (it == buffer_id_to_local_mr_.end() || !it->second.block) {
      release_request_slot_locked(request_id);
      return 0;
    }
    base_block = it->second.block;
  }

  int64_t token = -1;
  try {
    auto recv_mem = std::make_shared<RegMemBlock>(
        recv_ptr, len, base_block->mr_array, base_block->type);
    auto req = std::make_shared<RDMARecvRequest>(recv_mem);
    req->from_rank_id = static_cast<uint32_t>(peer_rank);
    req->to_rank_id = static_cast<uint32_t>(local_rank_);
    token = endpoint_->recv(static_cast<uint64_t>(peer_rank), req);
  } catch (...) {
    token = -1;
  }

  std::lock_guard<std::mutex> lk(mu_);
  PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
  if (slot == nullptr) return 0;
  if (token < 0) {
    slot->failed = true;
    slot->state = RequestState::Failed;
    return request_id;
  }
  slot->request.kind = RequestKind::Recv;
  slot->request.peer_rank = peer_rank;
  slot->request.token = token;
  slot->state = RequestState::Posted;
  slot->failed = false;
  return request_id;
}

bool RdmaTransportAdapter::is_backend_request_done(AdapterRequest const& request,
                                                   bool* ok) {
  if (!endpoint_) return false;
  bool done = false;
  try {
    if (request.kind == RequestKind::Recv) {
      done = endpoint_->checkRecvComplete_once(
          static_cast<uint64_t>(request.peer_rank),
          static_cast<uint64_t>(request.token));
    } else {
      done = endpoint_->checkSendComplete_once(
          static_cast<uint64_t>(request.peer_rank), request.token);
    }
  } catch (...) {
    if (ok) *ok = false;
    return true;
  }
  if (done && ok) *ok = true;
  return done;
}

bool RdmaTransportAdapter::poll_completion(unsigned id) {
  std::lock_guard<std::mutex> lk(mu_);
  PendingRequestSlot* slot = resolve_request_slot_locked(id);
  if (slot == nullptr) return false;
  if (slot->state == RequestState::Completed || slot->state == RequestState::Failed) {
    return true;
  }
  if (slot->state != RequestState::Posted) return false;

  bool ok = false;
  if (!is_backend_request_done(slot->request, &ok)) return false;
  slot->state = ok ? RequestState::Completed : RequestState::Failed;
  slot->failed = !ok;
  return true;
}

bool RdmaTransportAdapter::wait_completion(unsigned id) {
  while (true) {
    if (poll_completion(id)) return true;
    std::this_thread::sleep_for(std::chrono::microseconds(kWaitSleepUs));
  }
}

bool RdmaTransportAdapter::request_failed(unsigned id) {
  std::lock_guard<std::mutex> lk(mu_);
  PendingRequestSlot* slot = resolve_request_slot_locked(id);
  return slot != nullptr && slot->failed;
}

void RdmaTransportAdapter::release_request(unsigned id) {
  std::lock_guard<std::mutex> lk(mu_);
  release_request_slot_locked(id);
}

}  // namespace Transport
}  // namespace UKernel
