#include "rdma_adapter.h"
#include "p2p/rdma/memory_allocator.h"
#include "p2p/rdma/rdma_connection.h"
#include "p2p/rdma/rdma_context.h"
#include "p2p/rdma/rdma_ctrl_channel.h"
#include "p2p/rdma/rdma_data_channel.h"
#include "p2p/rdma/rdma_device.h"
#include "gpu_rt.h"
#include <array>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

namespace UKernel {
namespace Transport {

namespace {
constexpr uint32_t kHandshakeMagic = 0x554B5244;  // "UKRD"
constexpr uint32_t kHandshakeVersion = 1;
constexpr uint32_t kRdmaWaitSpinCount = 128;

inline size_t channel_to_context(uint32_t channel_id) {
  return (channel_id == 0) ? 0 : (channel_id - 1) % kNICContextNumber;
}
}  // namespace

RdmaTransportAdapter::RdmaTransportAdapter(int local_gpu_idx, int local_rank,
                                           int world_size,
                                           RdmaTransportConfig config)
    : local_gpu_idx_(local_gpu_idx), local_rank_(local_rank) {
  (void)world_size;
  (void)config;

  request_slots_ = std::make_unique<PendingRequestSlot[]>(kRequestSlotCount);
  for (uint32_t i = 0; i < kRequestSlotCount; ++i) {
    request_slots_[i].in_use = false;
    request_slots_[i].generation = 1;
    request_slots_[i].state = RequestState::Init;
    request_slots_[i].failed = false;
  }

  allocator_ = std::make_shared<MemoryAllocator>();
  initialized_ = initialize_contexts();
}

RdmaTransportAdapter::~RdmaTransportAdapter() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& [id, mr] : mr_id_to_local_mr_) {
    (void)id;
    for (auto const& ref : mr.cache_refs) {
      if (ref.context != nullptr) {
        ref.context->releaseCachedMr(ref.entry);
      }
    }
    mr.cache_refs.clear();
  }
  mr_id_to_local_mr_.clear();
  send_build_states_.clear();
  peer_contexts_.clear();
  contexts_.clear();
}

bool RdmaTransportAdapter::initialize_contexts() {
  std::vector<size_t> device_ids =
      RdmaDeviceManager::instance().get_best_dev_idx(local_gpu_idx_);
  if (device_ids.empty()) {
    std::cerr << "[ERROR] RDMA no usable device for gpu " << local_gpu_idx_
              << std::endl;
    return false;
  }

  std::unordered_map<size_t, std::shared_ptr<RdmaContext>> device_ctx_map;
  contexts_.clear();
  contexts_.reserve(kNICContextNumber);

  for (int i = 0; i < kNICContextNumber; ++i) {
    size_t device_id = device_ids[static_cast<size_t>(i) % device_ids.size()];
    auto it = device_ctx_map.find(device_id);
    if (it != device_ctx_map.end()) {
      contexts_.push_back(it->second);
      continue;
    }

    auto device = RdmaDeviceManager::instance().getDevice(device_id);
    if (!device) {
      std::cerr << "[ERROR] RDMA device id " << device_id << " not found"
                << std::endl;
      return false;
    }

    auto context =
        std::make_shared<RdmaContext>(device, static_cast<uint64_t>(contexts_.size()));
    contexts_.push_back(context);
    device_ctx_map.emplace(device_id, context);
  }

  if (contexts_.size() != kNICContextNumber) {
    std::cerr << "[ERROR] RDMA context slots init failed, expected "
              << kNICContextNumber << ", got " << contexts_.size() << std::endl;
    return false;
  }

  auto devs = RdmaDeviceManager::instance().get_best_dev_idx(local_gpu_idx_);
  if (!devs.empty()) {
    numa_node_ = RdmaDeviceManager::instance().get_numa_node(devs.front());
  }
  return true;
}

std::shared_ptr<RdmaContext> RdmaTransportAdapter::get_context_by_channel_id(
    uint32_t channel_id) const {
  if (contexts_.empty()) return nullptr;
  size_t idx = channel_to_context(channel_id);
  if (idx >= contexts_.size()) return nullptr;
  return contexts_[idx];
}

std::string RdmaTransportAdapter::serialize_handshake(
    HandshakePacket const& packet) {
  auto const* p = reinterpret_cast<uint8_t const*>(&packet);
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (size_t i = 0; i < sizeof(HandshakePacket); ++i) {
    oss << std::setw(2) << static_cast<unsigned>(p[i]);
  }
  return oss.str();
}

bool RdmaTransportAdapter::deserialize_handshake(std::string const& payload,
                                                 HandshakePacket* out_packet) {
  if (out_packet == nullptr) return false;
  if (payload.size() != sizeof(HandshakePacket) * 2) return false;
  HandshakePacket packet{};
  auto* p = reinterpret_cast<uint8_t*>(&packet);
  for (size_t i = 0; i < sizeof(HandshakePacket); ++i) {
    auto byte_str = payload.substr(i * 2, 2);
    char* end = nullptr;
    unsigned long v = std::strtoul(byte_str.c_str(), &end, 16);
    if (end == nullptr || *end != '\0' || v > 0xff) return false;
    p[i] = static_cast<uint8_t>(v);
  }
  *out_packet = packet;
  return true;
}

bool RdmaTransportAdapter::validate_remote_packet(HandshakePacket const& remote,
                                                  int expected_src_rank) const {
  if (remote.magic != kHandshakeMagic || remote.version != kHandshakeVersion) {
    return false;
  }
  if (remote.src_rank != expected_src_rank) return false;
  if (remote.dst_rank != local_rank_) return false;
  return true;
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

bool RdmaTransportAdapter::has_send_peer(int peer_rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peer_contexts_.find(peer_rank);
  return it != peer_contexts_.end() && it->second.send_conn != nullptr;
}

bool RdmaTransportAdapter::has_recv_peer(int peer_rank) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = peer_contexts_.find(peer_rank);
  return it != peer_contexts_.end() && it->second.recv_conn != nullptr;
}

bool RdmaTransportAdapter::build_send_bootstrap(int peer_rank,
                                                std::string* out_payload) {
  if (!initialized_ || out_payload == nullptr || peer_rank < 0) return false;

  auto send_group = std::make_shared<SendConnection>(numa_node_, true);
  auto ctrl_ctx = get_context_by_channel_id(kControlChannelID);
  if (!ctrl_ctx) return false;

  auto ctrl_mem =
      allocator_->allocate(kRingBufferSize, MemoryType::HOST, ctrl_ctx);
  auto send_ctrl = std::make_shared<SendControlChannel>(
      ctrl_ctx, ctrl_mem, kControlChannelID);

  SendBuildState state{};
  state.send_conn = send_group;
  state.send_ctrl = send_ctrl;

  HandshakePacket local{};
  local.magic = kHandshakeMagic;
  local.version = kHandshakeVersion;
  local.src_rank = local_rank_;
  local.dst_rank = peer_rank;
  local.gpu_idx = local_gpu_idx_;
  local.ctrl_meta = *send_ctrl->get_local_meta();
  local.ctrl_mem = RemoteMemInfo(ctrl_mem);

  for (int i = 0; i < kQpNumPerChannel; ++i) {
    uint32_t channel_id = static_cast<uint32_t>(i + 1);
    auto ctx = get_context_by_channel_id(channel_id);
    if (!ctx) return false;
    auto ch = std::make_shared<RDMADataChannel>(ctx, channel_id);
    send_group->addChannel(channel_id, ch);
    local.normal_meta[i] = *ch->get_local_meta();
    state.channels[static_cast<size_t>(i)] = ch;
  }

  {
    std::lock_guard<std::mutex> lk(mu_);
    send_build_states_[peer_rank] = std::move(state);
  }
  *out_payload = serialize_handshake(local);
  return true;
}

bool RdmaTransportAdapter::create_recv_from_remote_send(
    int peer_rank, HandshakePacket const& remote_send,
    HandshakePacket* out_local_recv) {
  if (out_local_recv == nullptr) return false;

  auto recv_group = std::make_shared<RecvConnection>(numa_node_, true);
  auto ctrl_ctx = get_context_by_channel_id(kControlChannelID);
  if (!ctrl_ctx) return false;

  auto ctrl_mem =
      allocator_->allocate(kRingBufferSize, MemoryType::HOST, ctrl_ctx);

  MetaInfoToExchange remote_ctrl_meta;
  remote_ctrl_meta.channel_meta = remote_send.ctrl_meta;
  remote_ctrl_meta.mem_meta = remote_send.ctrl_mem;

  auto recv_ctrl = std::make_shared<RecvControlChannel>(
      ctrl_ctx, remote_ctrl_meta, ctrl_mem, kControlChannelID);

  HandshakePacket local{};
  local.magic = kHandshakeMagic;
  local.version = kHandshakeVersion;
  local.src_rank = local_rank_;
  local.dst_rank = peer_rank;
  local.gpu_idx = local_gpu_idx_;
  local.ctrl_meta = *recv_ctrl->get_local_meta();
  local.ctrl_mem = RemoteMemInfo(ctrl_mem);

  for (int i = 0; i < kQpNumPerChannel; ++i) {
    uint32_t channel_id = static_cast<uint32_t>(i + 1);
    auto ctx = get_context_by_channel_id(channel_id);
    if (!ctx) return false;
    auto ch = std::make_shared<RDMADataChannel>(ctx, remote_send.normal_meta[i],
                                                channel_id);
    recv_group->addChannel(channel_id, ch);
    local.normal_meta[i] = *ch->get_local_meta();
  }

  recv_group->setControlChannel(std::move(recv_ctrl));
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto& ctx = peer_contexts_[peer_rank];
    ctx.recv_conn = recv_group;
  }

  *out_local_recv = local;
  return true;
}

bool RdmaTransportAdapter::build_recv_bootstrap(
    int peer_rank, std::string const& remote_send_payload,
    std::string* out_payload) {
  if (!initialized_ || out_payload == nullptr || peer_rank < 0) return false;

  HandshakePacket remote_send{};
  if (!deserialize_handshake(remote_send_payload, &remote_send)) return false;
  if (!validate_remote_packet(remote_send, peer_rank)) return false;

  HandshakePacket local_recv{};
  if (!create_recv_from_remote_send(peer_rank, remote_send, &local_recv)) {
    return false;
  }
  *out_payload = serialize_handshake(local_recv);
  return true;
}

bool RdmaTransportAdapter::finalize_send_from_remote_recv(
    int peer_rank, HandshakePacket const& remote_recv) {
  SendBuildState state{};
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = send_build_states_.find(peer_rank);
    if (it == send_build_states_.end()) return false;
    state = it->second;
    send_build_states_.erase(it);
  }

  if (!state.send_conn || !state.send_ctrl) return false;

  state.send_ctrl->establishChannel(remote_recv.ctrl_meta);
  for (int i = 0; i < kQpNumPerChannel; ++i) {
    auto ch = state.channels[static_cast<size_t>(i)];
    if (!ch) return false;
    ch->establishChannel(remote_recv.normal_meta[i]);
  }
  state.send_conn->setControlChannel(std::move(state.send_ctrl));

  std::lock_guard<std::mutex> lk(mu_);
  auto& ctx = peer_contexts_[peer_rank];
  ctx.send_conn = state.send_conn;
  return true;
}

bool RdmaTransportAdapter::finalize_send_bootstrap(
    int peer_rank, std::string const& remote_recv_payload) {
  if (!initialized_ || peer_rank < 0) return false;
  HandshakePacket remote_recv{};
  if (!deserialize_handshake(remote_recv_payload, &remote_recv)) return false;
  if (!validate_remote_packet(remote_recv, peer_rank)) return false;
  return finalize_send_from_remote_recv(peer_rank, remote_recv);
}

void RdmaTransportAdapter::rollback_peer_bootstrap(int peer_rank) {
  if (peer_rank < 0) return;
  std::lock_guard<std::mutex> lk(mu_);
  send_build_states_.erase(peer_rank);
  auto it = peer_contexts_.find(peer_rank);
  if (it == peer_contexts_.end()) return;
  it->second.send_conn.reset();
  it->second.recv_conn.reset();
  peer_contexts_.erase(it);
}

bool RdmaTransportAdapter::ensure_peer(PeerConnectSpec const& spec) {
  return has_peer(spec.peer_rank);
}

bool RdmaTransportAdapter::register_memory(uint64_t mr_id, void* ptr,
                                           size_t len) {
  if (!initialized_ || ptr == nullptr || len == 0) return false;
  {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      auto it = mr_registering_.find(mr_id);
      if (it == mr_registering_.end() || !it->second) break;
      mr_cv_.wait(lk);
    }
    if (mr_id_to_local_mr_.find(mr_id) != mr_id_to_local_mr_.end()) return true;
    mr_registering_[mr_id] = true;
  }

  LocalMr local_mr;
  local_mr.block = std::make_shared<RegMemBlock>(ptr, len, detect_memory_type(ptr));
  bool success = false;

  std::unordered_map<RdmaContext*, struct ibv_mr*> registered;
  for (size_t context_id = 0; context_id < contexts_.size(); ++context_id) {
    auto ctx = contexts_[context_id];
    if (!ctx) {
      for (auto const& ref : local_mr.cache_refs) {
        if (ref.context) ref.context->releaseCachedMr(ref.entry);
      }
      local_mr.cache_refs.clear();
      break;
    }

    auto found = registered.find(ctx.get());
    if (found != registered.end()) {
      local_mr.block->setMRByContextID(static_cast<uint32_t>(context_id),
                                       found->second);
      continue;
    }

    MrCacheEntry* entry = ctx->acquireCachedMr(ptr, len);
    if (!entry || !entry->mr) {
      for (auto const& ref : local_mr.cache_refs) {
        if (ref.context) ref.context->releaseCachedMr(ref.entry);
      }
      local_mr.cache_refs.clear();
      break;
    }

    local_mr.block->setMRByContextID(static_cast<uint32_t>(context_id),
                                     entry->mr);
    local_mr.cache_refs.push_back({ctx, entry});
    registered.emplace(ctx.get(), entry->mr);
  }

  if (local_mr.cache_refs.size() == contexts_.size()) {
    success = true;
  }

  std::unique_lock<std::mutex> lk(mu_);
  if (success) {
    mr_id_to_local_mr_.emplace(mr_id, std::move(local_mr));
  }
  mr_registering_.erase(mr_id);
  lk.unlock();
  mr_cv_.notify_all();
  return success;
}

void RdmaTransportAdapter::deregister_memory(uint64_t mr_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = mr_id_to_local_mr_.find(mr_id);
  if (it == mr_id_to_local_mr_.end()) return;

  for (auto const& ref : it->second.cache_refs) {
    if (ref.context != nullptr) {
      ref.context->releaseCachedMr(ref.entry);
    }
  }
  it->second.cache_refs.clear();
  mr_id_to_local_mr_.erase(it);
}

bool RdmaTransportAdapter::is_memory_registered(uint64_t mr_id) const {
  std::lock_guard<std::mutex> lk(mu_);
  return mr_id_to_local_mr_.find(mr_id) != mr_id_to_local_mr_.end();
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
    slot.request = AdapterRequest{};
    if (slot.generation == 0) slot.generation = 1;
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
  slot->request = AdapterRequest{};
  uint32_t next_gen = slot->generation + 1;
  slot->generation = (next_gen == 0) ? 1 : next_gen;
}

unsigned RdmaTransportAdapter::send_async(
    int peer_rank, void* local_ptr, size_t len, uint64_t local_mr_id,
    std::optional<RemoteSlice> remote_hint,
    BounceBufferProvider bounce_provider) {
  void* send_ptr = local_ptr;
  uint64_t send_mr_id = local_mr_id;
  if (bounce_provider) {
    BounceBufferInfo info = bounce_provider(len);
    if (info.ptr != nullptr) {
      if (detect_memory_type(local_ptr) == MemoryType::GPU) {
        GPU_RT_CHECK(gpuMemcpy(info.ptr, local_ptr, len, gpuMemcpyDeviceToHost));
      } else {
        std::memcpy(info.ptr, local_ptr, len);
      }
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
  (void)remote_mr_id;

  std::shared_ptr<SendConnection> send_conn;
  LocalMr local_mr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto peer_it = peer_contexts_.find(peer_rank);
    if (peer_it == peer_contexts_.end() || !peer_it->second.send_conn) return -1;
    send_conn = peer_it->second.send_conn;

    auto mh_it = mr_id_to_local_mr_.find(local_mr_id);
    if (mh_it == mr_id_to_local_mr_.end()) return -1;
    local_mr = mh_it->second;

    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot == nullptr || slot->state != RequestState::Init) return -1;
    slot->failed = false;
  }

  auto send_mem = std::make_shared<RegMemBlock>(
      local_ptr, len, local_mr.block->mr_array, local_mr.block->type);

  int64_t token = -1;
  RequestKind kind = RequestKind::Send;

  if (remote_slice != nullptr && remote_slice->has_write_hint()) {
    auto remote_mem = std::make_shared<RemoteMemInfo>();
    remote_mem->addr = remote_slice->write.addr;
    remote_mem->length =
        (remote_slice->write.capacity == 0) ? len : remote_slice->write.capacity;
    remote_mem->type = MemoryType::GPU;
    for (uint32_t ctx = 0; ctx < kNICContextNumber; ++ctx) {
      remote_mem->rkey_array.setKeyByContextID(ctx, remote_slice->write.key);
    }

    auto req = std::make_shared<RDMASendRequest>(send_mem, remote_mem);
    req->send_type = SendType::Write;
    token = send_conn->postWriteOrRead(req);
    kind = RequestKind::Write;
  } else {
    auto remote_mem_placeholder = std::make_shared<RemoteMemInfo>();
    auto req = std::make_shared<RDMASendRequest>(send_mem, remote_mem_placeholder);
    req->send_type = SendType::Send;
    token = send_conn->send(req);
    kind = RequestKind::Send;
  }

  if (token < 0) {
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
  slot->request.kind = kind;
  slot->request.peer_rank = peer_rank;
  slot->request.token = token;
  slot->failed = false;
  slot->state = RequestState::Posted;
  return 0;
}

int RdmaTransportAdapter::recv_async_rdma(int peer_rank, void* local_ptr,
                                          size_t len, uint64_t local_mr_id,
                                          uint64_t request_id) {
  std::shared_ptr<RecvConnection> recv_conn;
  LocalMr local_mr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto peer_it = peer_contexts_.find(peer_rank);
    if (peer_it == peer_contexts_.end() || !peer_it->second.recv_conn) return -1;
    recv_conn = peer_it->second.recv_conn;

    auto mh_it = mr_id_to_local_mr_.find(local_mr_id);
    if (mh_it == mr_id_to_local_mr_.end()) return -1;
    local_mr = mh_it->second;

    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot == nullptr || slot->state != RequestState::Init) return -1;
    slot->failed = false;
  }

  auto recv_mem = std::make_shared<RegMemBlock>(
      local_ptr, len, local_mr.block->mr_array, local_mr.block->type);
  auto recv_req = std::make_shared<RDMARecvRequest>(recv_mem);
  int64_t token = recv_conn->recv(recv_req);

  if (token < 0) {
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
  slot->request.kind = RequestKind::Recv;
  slot->request.peer_rank = peer_rank;
  slot->request.token = token;
  slot->failed = false;
  slot->state = RequestState::Posted;
  return 0;
}

bool RdmaTransportAdapter::is_backend_request_done(AdapterRequest const& request,
                                                   bool* ok) const {
  if (ok) *ok = false;

  std::shared_ptr<SendConnection> send_conn;
  std::shared_ptr<RecvConnection> recv_conn;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto peer_it = peer_contexts_.find(request.peer_rank);
    if (peer_it == peer_contexts_.end()) return true;
    send_conn = peer_it->second.send_conn;
    recv_conn = peer_it->second.recv_conn;
  }

  switch (request.kind) {
    case RequestKind::Send:
    case RequestKind::Write:
      if (!send_conn) return true;
      if (ok) *ok = true;
      return send_conn->check(request.token);
    case RequestKind::Recv:
      if (!recv_conn) return true;
      if (ok) *ok = true;
      return recv_conn->check(static_cast<uint64_t>(request.token));
    default:
      return true;
  }
}

bool RdmaTransportAdapter::poll_completion(unsigned request_id) {
  AdapterRequest req;
  {
    std::lock_guard<std::mutex> lk(mu_);
    PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
    if (slot == nullptr) return true;
    if (slot->state == RequestState::Completed ||
        slot->state == RequestState::Failed) {
      return true;
    }
    if (slot->state != RequestState::Posted) return false;
    req = slot->request;
  }

  bool backend_ok = false;
  bool done = is_backend_request_done(req, &backend_ok);
  if (!done) return false;

  std::lock_guard<std::mutex> lk(mu_);
  PendingRequestSlot* slot = resolve_request_slot_locked(request_id);
  if (slot == nullptr) return true;

  if (!backend_ok) {
    slot->failed = true;
    slot->state = RequestState::Failed;
    return true;
  }

  if (slot->state == RequestState::Posted) {
    slot->state = RequestState::Completed;
  }
  return true;
}

bool RdmaTransportAdapter::wait_completion(unsigned request_id) {
  uint32_t spins = 0;
  while (true) {
    if (poll_completion(request_id)) return true;
    if (spins < kRdmaWaitSpinCount) {
      ++spins;
      continue;
    }
    std::this_thread::yield();
  }
}

void RdmaTransportAdapter::release_request(unsigned request_id) {
  std::lock_guard<std::mutex> lk(mu_);
  release_request_slot_locked(request_id);
}

}  // namespace Transport
}  // namespace UKernel
