#include "ipc_adapter.h"
#include "../communicator.h"
#include "../util/utils.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>

namespace UKernel {
namespace Transport {

namespace {

constexpr int kIpcControlPollTimeoutMs = 50;
constexpr size_t kTaskRingSize = 1024;
constexpr size_t kIpcSizePerEngine = 1ul << 20;
constexpr int kIpcControlTimeoutMs = 50000;

bool ipc_force_relay_enabled() {
  char const* env = std::getenv("UHM_IPC_FORCE_RELAY");
  return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

int remaining_timeout_ms(std::chrono::steady_clock::time_point deadline) {
  auto now = std::chrono::steady_clock::now();
  if (now >= deadline) return 0;
  return static_cast<int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
          .count());
}

constexpr uint32_t kAckStatusOkDirect = 1;

bool enqueue_one_request_id(jring_t* ring, unsigned elem,
                            std::atomic<bool> const& stop) {
  unsigned elem_slot = elem;
  while (!stop.load(std::memory_order_acquire) &&
         jring_mp_enqueue_bulk(ring, &elem_slot, 1, nullptr) != 1) {
    std::this_thread::yield();
  }
  return !stop.load(std::memory_order_acquire);
}

}  // namespace

IpcAdapter::IpcAdapter(Communicator* comm, std::string ring_namespace,
                       int self_local_id)
    : next_match_seq_per_peer_(comm->world_size(),
                               std::array<uint64_t, 2>{1, 1}),
      shm_control_(std::make_shared<ShmRingExchanger>(
          comm->rank(), comm->world_size(), std::move(ring_namespace),
          self_local_id)),
      comm_(comm) {
  send_task_ring_ =
      UKernel::Transport::create_ring(sizeof(unsigned), kTaskRingSize);
  recv_task_ring_ =
      UKernel::Transport::create_ring(sizeof(unsigned), kTaskRingSize);
  request_slots_ = std::make_unique<IpcRequestSlot[]>(kRequestSlotCount);
  stop_.store(false);
  send_thread_ = std::thread([this] { send_thread_func(); });
  recv_thread_ = std::thread([this] { recv_thread_func(); });

  int n_streams = 2;
  GPU_RT_CHECK(gpuSetDevice(comm->local_gpu_idx()));
  ipc_streams_.resize(n_streams);
  for (int i = 0; i < n_streams; ++i) {
    GPU_RT_CHECK(
        gpuStreamCreateWithFlags(&ipc_streams_[i], gpuStreamNonBlocking));
  }
}

IpcAdapter::~IpcAdapter() { shutdown(); }

void IpcAdapter::shutdown() {
  bool expected = false;
  if (!shutdown_started_.compare_exchange_strong(expected, true)) return;

  stop_.store(true, std::memory_order_release);
  cv_.notify_all();
  if (send_thread_.joinable()) send_thread_.join();
  if (recv_thread_.joinable()) recv_thread_.join();

  if (shm_control_ != nullptr) {
    for (int peer_rank = 0; peer_rank < comm_->world_size(); ++peer_rank) {
      if (peer_rank == comm_->rank()) continue;
      if (shm_control_->is_peer_connected(peer_rank)) {
        shm_control_->close_peer(peer_rank);
      }
    }
  }

  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  GPU_RT_CHECK(gpuSetDevice(comm_->local_gpu_idx()));
  for (auto& stream : ipc_streams_) {
    if (stream != nullptr) GPU_RT_CHECK(gpuStreamDestroy(stream));
  }
  ipc_streams_.clear();
  GPU_RT_CHECK(gpuSetDevice(orig_device));

  if (send_task_ring_) {
    free(send_task_ring_);
    send_task_ring_ = nullptr;
  }
  if (recv_task_ring_) {
    free(recv_task_ring_);
    recv_task_ring_ = nullptr;
  }
}

bool IpcAdapter::connect_to(int rank) {
  return shm_control_ != nullptr && shm_control_->connect_to(rank, 30000);
}

bool IpcAdapter::accept_from(int rank) {
  return shm_control_ != nullptr && shm_control_->accept_from(rank, 30000);
}

bool IpcAdapter::ensure_peer(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_peer(spec.peer_rank)) return true;
  if (!(std::holds_alternative<IpcPeerConnectSpec>(spec.detail) ||
        std::holds_alternative<std::monostate>(spec.detail))) {
    return false;
  }
  if (spec.type == PeerConnectType::Connect) {
    return connect_to(spec.peer_rank);
  }
  return accept_from(spec.peer_rank);
}

void IpcAdapter::set_peer_local_id(int peer_rank, int local_id) {
  if (shm_control_ != nullptr) {
    shm_control_->set_peer_local_id(peer_rank, local_id);
  }
}

void IpcAdapter::close_peer(int peer_rank) {
  if (shm_control_ != nullptr) {
    shm_control_->close_peer(peer_rank);
  }
}

uint64_t IpcAdapter::next_send_match_seq(int rank) {
  std::lock_guard<std::mutex> lk(match_seq_mu_);
  int src = comm_->rank();
  int dst = rank;
  size_t dir = (src < dst) ? 0u : 1u;
  uint64_t counter = next_match_seq_per_peer_[rank][dir]++;
  // Encode edge direction bit in seq so opposite directions never collide.
  return (counter << 1) | static_cast<uint64_t>(dir);
}

uint64_t IpcAdapter::next_recv_match_seq(int rank) {
  std::lock_guard<std::mutex> lk(match_seq_mu_);
  int src = rank;
  int dst = comm_->rank();
  size_t dir = (src < dst) ? 0u : 1u;
  uint64_t counter = next_match_seq_per_peer_[rank][dir]++;
  return (counter << 1) | static_cast<uint64_t>(dir);
}

bool IpcAdapter::has_peer(int peer_rank) const {
  return shm_control_ != nullptr && shm_control_->is_peer_connected(peer_rank);
}

IpcAdapter::IpcRequestSlot* IpcAdapter::try_acquire_request_slot(
    unsigned* out_request_id) {
  if (out_request_id == nullptr || !request_slots_) return nullptr;
  for (uint32_t n = 0; n < kRequestSlotCount; ++n) {
    uint32_t idx =
        request_alloc_cursor_.fetch_add(1, std::memory_order_relaxed) &
        kRequestSlotMask;
    auto& slot = request_slots_[idx];
    RequestState expected = RequestState::Free;
    if (!slot.state.compare_exchange_strong(expected, RequestState::Running,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
      continue;
    }
    uint32_t gen = slot.generation.load(std::memory_order_acquire);
    if (gen == 0) {
      slot.generation.store(1, std::memory_order_release);
      gen = 1;
    }
    slot.id = make_request_id(idx, gen);
    slot.peer_rank = -1;
    slot.match_seq = 0;
    slot.buffer = nullptr;
    slot.size_bytes = 0;
    slot.remote_slice = {};
    slot.bounce_ptr = nullptr;
    slot.bounce_shm_name.clear();
    slot.bounce_provider = nullptr;
    *out_request_id = slot.id;
    return &slot;
  }
  return nullptr;
}

IpcAdapter::IpcRequestSlot* IpcAdapter::resolve_request_slot(
    unsigned request_id) {
  if (request_id == 0 || !request_slots_) return nullptr;
  uint32_t generation = request_generation(request_id);
  if (generation == 0) return nullptr;
  uint32_t idx = request_slot_index(request_id);
  auto& slot = request_slots_[idx];
  if (slot.generation.load(std::memory_order_acquire) != generation) {
    return nullptr;
  }
  if (slot.state.load(std::memory_order_acquire) == RequestState::Free) {
    return nullptr;
  }
  return &slot;
}

IpcAdapter::IpcRequestSlot* IpcAdapter::resolve_request_slot_const(
    unsigned request_id) const {
  if (request_id == 0 || !request_slots_) return nullptr;
  uint32_t generation = request_generation(request_id);
  if (generation == 0) return nullptr;
  uint32_t idx = request_slot_index(request_id);
  auto const& slot = request_slots_[idx];
  if (slot.generation.load(std::memory_order_acquire) != generation) {
    return nullptr;
  }
  if (slot.state.load(std::memory_order_acquire) == RequestState::Free) {
    return nullptr;
  }
  return const_cast<IpcRequestSlot*>(&slot);
}

void IpcAdapter::release_request_slot(unsigned request_id) {
  if (request_id == 0 || !request_slots_) return;
  uint32_t generation = request_generation(request_id);
  if (generation == 0) return;
  uint32_t idx = request_slot_index(request_id);
  auto& slot = request_slots_[idx];
  if (slot.generation.load(std::memory_order_acquire) != generation) return;
  auto st = slot.state.load(std::memory_order_acquire);
  if (st == RequestState::Running || st == RequestState::Queued) return;
  slot.state.store(RequestState::Free, std::memory_order_release);
  slot.failed.store(false, std::memory_order_release);
  slot.finished.store(false, std::memory_order_release);
  slot.remaining.store(0, std::memory_order_release);
  uint32_t old_gen = slot.generation.load(std::memory_order_acquire);
  while (true) {
    uint32_t next_gen = old_gen + 1;
    if (next_gen == 0) next_gen = 1;
    if (slot.generation.compare_exchange_weak(old_gen, next_gen,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
      break;
    }
  }
}

bool IpcAdapter::enqueue_request(unsigned request_id, IpcReqType type) {
  if (stop_.load(std::memory_order_acquire)) return false;
  if (type == IpcReqType::Send) {
    if (send_task_ring_ == nullptr ||
        !enqueue_one_request_id(send_task_ring_, request_id, stop_)) {
      return false;
    }
    pending_send_.fetch_add(1, std::memory_order_relaxed);
  } else {
    if (recv_task_ring_ == nullptr ||
        !enqueue_one_request_id(recv_task_ring_, request_id, stop_)) {
      return false;
    }
    pending_recv_.fetch_add(1, std::memory_order_relaxed);
  }
  cv_.notify_all();
  return true;
}

unsigned IpcAdapter::send_async(int peer_rank, void* local_ptr, size_t len,
                                uint64_t local_mr_id,
                                std::optional<RemoteSlice> remote_hint,
                                BounceBufferProvider bounce_provider) {
  (void)local_mr_id;

  uint32_t remote_mem_id = 0;
  size_t remote_offset = 0;
  if (remote_hint.has_value()) {
    remote_mem_id = remote_hint->mem_id;
    remote_offset = remote_hint->offset;
  }
  uint64_t match_seq = next_send_match_seq(peer_rank);
  RemoteSlice remote_slice{};
  remote_slice.mem_id = remote_mem_id;
  remote_slice.offset = remote_offset;
  if (remote_hint.has_value()) {
    remote_slice.write = remote_hint->write;
  }
  unsigned request_id = 0;
  IpcRequestSlot* req = try_acquire_request_slot(&request_id);
  if (!req) return 0;
  req->peer_rank = peer_rank;
  req->match_seq = match_seq;
  req->buffer = local_ptr;
  req->size_bytes = len;
  req->remote_slice = remote_slice;
  req->bounce_provider = std::move(bounce_provider);
  req->mark_queued(1);
  if (!enqueue_request(request_id, IpcReqType::Send)) {
    req->mark_failed();
    release_request_slot(request_id);
    return 0;
  }
  return request_id;
}

unsigned IpcAdapter::recv_async(int peer_rank, void* local_ptr, size_t len,
                                uint64_t local_mr_id,
                                BounceBufferProvider bounce_provider) {
  (void)local_mr_id;

  void* bounce_ptr = nullptr;
  std::string bounce_shm_name;
  if (bounce_provider) {
    auto info = bounce_provider(len);
    bounce_ptr = info.ptr;
    bounce_shm_name = info.shm_name;
  }

  uint64_t match_seq = next_recv_match_seq(peer_rank);
  unsigned request_id = 0;
  IpcRequestSlot* req = try_acquire_request_slot(&request_id);
  if (!req) return 0;
  req->peer_rank = peer_rank;
  req->match_seq = match_seq;
  req->buffer = local_ptr;
  req->size_bytes = len;
  req->remote_slice = {};
  req->bounce_ptr = bounce_ptr;
  req->bounce_shm_name = std::move(bounce_shm_name);
  req->mark_queued(1);
  if (!enqueue_request(request_id, IpcReqType::Recv)) {
    req->mark_failed();
    release_request_slot(request_id);
    return 0;
  }
  return request_id;
}

bool IpcAdapter::poll_completion(unsigned id) {
  IpcRequestSlot* req = resolve_request_slot_const(id);
  if (!req) return true;
  return req->is_finished();
}

bool IpcAdapter::wait_completion(unsigned id) {
  IpcRequestSlot* req = resolve_request_slot_const(id);
  if (!req) return true;
  while (!req->is_finished()) {
    std::this_thread::yield();
  }
  return true;
}

bool IpcAdapter::request_failed(unsigned id) {
  IpcRequestSlot* req = resolve_request_slot_const(id);
  if (!req) return false;
  return req->has_failed();
}

void IpcAdapter::release_request(unsigned id) { release_request_slot(id); }

bool IpcAdapter::send_one(IpcRequestSlot* creq) {
  if (!creq) return false;
  int to_rank = creq->peer_rank;
  if (!(creq && creq->buffer != nullptr)) {
    std::cerr << "[ERROR] send_ipc: data pointer is null!" << std::endl;
    return false;
  }
  creq->mark_running();

  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset = UKernel::Transport::finally(
      [&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(comm_->local_gpu_idx()));
  auto wait_sender_ack = [&](int timeout_ms, uint32_t* out_status) -> int {
    uint64_t out_seq = 0;
    uint32_t status = 0;
    if (!shm_control_->recv_ack(to_rank, &status, &out_seq, timeout_ms,
                                creq->match_seq)) {
      return 0;
    }
    if (out_seq != creq->match_seq || status != kAckStatusOkDirect) {
      std::cerr << "[ERROR] sender completion ack invalid: seq=" << out_seq
                << " status=" << status << " req=" << creq->id
                << " match_seq=" << creq->match_seq << std::endl;
      return -1;
    }
    if (out_status) *out_status = status;
    return 1;
  };

  auto copy_to_remote = [&](void* dst_ptr, int remote_gpu_idx) {
    void* src_ptr = creq->data();
    size_t n_streams =
        std::min(ipc_streams_.size(),
                 creq->size_bytes < kIpcSizePerEngine
                     ? size_t{1}
                     : std::max<size_t>(size_t{1},
                                        creq->size_bytes / kIpcSizePerEngine));
    size_t chunk_size = creq->size_bytes / n_streams;
    for (size_t i = 0; i < n_streams; ++i) {
      void* chunk_src = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(src_ptr) + i * chunk_size);
      void* chunk_dst = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(dst_ptr) + i * chunk_size);
      size_t copy_size =
          i == n_streams - 1 ? creq->size_bytes - i * chunk_size : chunk_size;
      if (remote_gpu_idx == comm_->local_gpu_idx()) {
        GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst, chunk_src, copy_size,
                                    gpuMemcpyDeviceToDevice, ipc_streams_[i]));
      } else {
        GPU_RT_CHECK(gpuMemcpyPeerAsync(chunk_dst, remote_gpu_idx, chunk_src,
                                        comm_->local_gpu_idx(), copy_size,
                                        ipc_streams_[i]));
      }
    }
    for (size_t i = 0; i < n_streams; ++i) {
      GPU_RT_CHECK(gpuStreamSynchronize(ipc_streams_[i]));
    }
  };

  auto relay_via_bounce = [&](uint64_t seq) -> bool {
    void* relay_bounce_ptr = creq->bounce_ptr;
    std::string relay_bounce_shm_name = creq->bounce_shm_name;
    if ((relay_bounce_ptr == nullptr || relay_bounce_shm_name.empty()) &&
        creq->bounce_provider) {
      auto info = creq->bounce_provider(creq->size_bytes);
      relay_bounce_ptr = info.ptr;
      relay_bounce_shm_name = info.shm_name;
    }
    if (relay_bounce_ptr == nullptr || relay_bounce_shm_name.empty()) {
      std::cerr << "[ERROR] IPC relay requires bounce buffer, req " << creq->id
                << " match_seq " << creq->match_seq << std::endl;
      return false;
    }
    void* src_ptr = creq->data();
    GPU_RT_CHECK(gpuMemcpy(relay_bounce_ptr, src_ptr, creq->size_bytes,
                           gpuMemcpyDeviceToHost));

    IpcCacheWire relay_cache{};
    relay_cache.size = creq->size_bytes;
    relay_cache.is_send = 1;
    relay_cache.remote_gpu_idx_ = comm_->local_gpu_idx();
    relay_cache.offset = 0;
    relay_cache.use_bounce_buffer = 1;
    std::strncpy(relay_cache.bounce_shm_name, relay_bounce_shm_name.c_str(),
                 sizeof(relay_cache.bounce_shm_name) - 1);

    if (!shm_control_->send_ipc_cache(to_rank, seq, relay_cache)) {
      std::cerr << "[ERROR] send_ipc_cache with bounce failed for req "
                << creq->id << " match_seq " << creq->match_seq << std::endl;
      return false;
    }

    uint32_t status = 0;
    if (wait_sender_ack(kIpcControlTimeoutMs, &status) <= 0) {
      std::cerr << "[ERROR] recv ack for bounce relay failed, req " << creq->id
                << " match_seq " << creq->match_seq << std::endl;
      return false;
    }
    return true;
  };

  // IPC direct-path safety rule:
  // only attempt by-mem_id fast path when binding_version is explicit.
  // Otherwise force handshake to avoid stale metadata reuse.
  if (creq->remote_slice.mem_id != 0 &&
      creq->remote_slice.binding_version != 0) {
    auto try_direct_by_remote_slice = [&](void** out_dst,
                                          int* out_remote_gpu) -> bool {
      if (!comm_->resolve_ipc_buffer_pointer(
              to_rank, creq->remote_slice.mem_id, creq->remote_slice.offset,
              creq->size_bytes, out_dst, out_remote_gpu)) {
        return false;
      }
      bool can_use_direct_peer = !ipc_force_relay_enabled() &&
                                 (*out_remote_gpu == comm_->local_gpu_idx());
      if (!can_use_direct_peer && !ipc_force_relay_enabled() &&
          *out_remote_gpu >= 0) {
        int can_access_peer = 0;
        GPU_RT_CHECK(gpuDeviceCanAccessPeer(
            &can_access_peer, comm_->local_gpu_idx(), *out_remote_gpu));
        can_use_direct_peer = (can_access_peer != 0);
      }
      return can_use_direct_peer;
    };

    bool have_fresh_meta = comm_->ipc_has_fresh_remote_ipc_buffer(
        to_rank, creq->remote_slice.mem_id, creq->remote_slice.binding_version);
    if (!have_fresh_meta) {
      have_fresh_meta = comm_->ipc_fetch_remote_ipc_buffer(
          to_rank, creq->remote_slice.mem_id,
          creq->remote_slice.binding_version);
      if (!have_fresh_meta) {
        comm_->ipc_invalidate_remote_ipc_buffer(to_rank,
                                                creq->remote_slice.mem_id);
      }
    }

    void* cached_dst = nullptr;
    int cached_remote_gpu = -1;
    bool can_use_direct =
        have_fresh_meta &&
        try_direct_by_remote_slice(&cached_dst, &cached_remote_gpu);
    if (can_use_direct && cached_dst != nullptr && cached_remote_gpu >= 0) {
      bool can_use_direct_peer = !ipc_force_relay_enabled() &&
                                 (cached_remote_gpu == comm_->local_gpu_idx());
      if (!can_use_direct_peer && !ipc_force_relay_enabled()) {
        int can_access_peer = 0;
        GPU_RT_CHECK(gpuDeviceCanAccessPeer(
            &can_access_peer, comm_->local_gpu_idx(), cached_remote_gpu));
        can_use_direct_peer = (can_access_peer != 0);
      }
      if (can_use_direct_peer) {
        copy_to_remote(cached_dst, cached_remote_gpu);
        if (!shm_control_->send_ack(to_rank, creq->match_seq,
                                    kAckStatusOkDirect)) {
          std::cerr << "[ERROR] send direct cached ack(" << to_rank
                    << ") failed for req " << creq->id << " match_seq "
                    << creq->match_seq << std::endl;
          return false;
        }
        return true;
      }
      // Remote destination is known but peer access is unavailable:
      // relay directly instead of paying an extra ipc_cache_req round trip.
      return relay_via_bounce(creq->match_seq);
    }
  }

  if (!shm_control_->send_ipc_cache_req(to_rank, creq->match_seq)) {
    std::cerr << "[ERROR] send_ipc_cache_req(" << to_rank << ") failed for req "
              << creq->id << " match_seq " << creq->match_seq << std::endl;
    return false;
  }

  IpcCacheWire got{};
  uint64_t seq = 0;
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(kIpcControlTimeoutMs);
  bool got_cache = false;
  while (!stop_.load(std::memory_order_acquire)) {
    int timeout_ms =
        std::min(kIpcControlPollTimeoutMs, remaining_timeout_ms(deadline));
    if (timeout_ms <= 0) break;
    if (shm_control_->recv_ipc_cache(to_rank, got, &seq, timeout_ms,
                                     creq->match_seq)) {
      got_cache = true;
      break;
    }
  }
  if (!got_cache) {
    std::cerr << "[ERROR] recv_ipc_cache(" << to_rank << ") failed for req "
              << creq->id << " match_seq " << creq->match_seq << std::endl;
    return false;
  }

  bool can_use_direct_peer =
      !ipc_force_relay_enabled() &&
      (got.remote_gpu_idx_ == static_cast<uint32_t>(comm_->local_gpu_idx()));
  if (!can_use_direct_peer) {
    int can_access_peer = 0;
    if (!ipc_force_relay_enabled()) {
      GPU_RT_CHECK(
          gpuDeviceCanAccessPeer(&can_access_peer, comm_->local_gpu_idx(),
                                 static_cast<int>(got.remote_gpu_idx_)));
      can_use_direct_peer = (can_access_peer != 0);
    }
  }

  if (!can_use_direct_peer) {
    return relay_via_bounce(seq);
  }

  IPCItem ipc = comm_->get_remote_ipc_cache(to_rank, got.handle);
  void* base = ipc.direct_ptr;
  if (base == nullptr) {
    GPU_RT_CHECK(
        gpuIpcOpenMemHandle(&base, got.handle, gpuIpcMemLazyEnablePeerAccess));

    IPCItem new_ipc{};
    new_ipc.handle = got.handle;
    new_ipc.direct_ptr = base;
    new_ipc.base_offset = got.offset;
    new_ipc.bytes = got.size;
    new_ipc.device_idx = static_cast<int>(got.remote_gpu_idx_);
    new_ipc.valid = true;
    comm_->register_remote_ipc_cache(to_rank, got.handle, new_ipc);
  }

  void* dst_ptr =
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + got.offset);
  copy_to_remote(dst_ptr, static_cast<int>(got.remote_gpu_idx_));

  if (!shm_control_->send_ack(to_rank, seq, kAckStatusOkDirect)) {
    std::cerr << "[ERROR] send_ack(" << to_rank << ") failed for req "
              << creq->id << " match_seq " << creq->match_seq << std::endl;
    return false;
  }
  return true;
}

bool IpcAdapter::recv_one(IpcRequestSlot* creq) {
  if (!creq) return false;
  int from_rank = creq->peer_rank;
  if (!(creq && creq->buffer != nullptr)) {
    std::cerr << "[ERROR] recv_ipc: data pointer is null!" << std::endl;
    return false;
  }
  creq->mark_running();
  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset = UKernel::Transport::finally(
      [&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  auto wait_sender_ack = [&](int timeout_ms, uint32_t* out_status) -> int {
    uint64_t out_seq = 0;
    uint32_t status = 0;
    if (!shm_control_->recv_ack(from_rank, &status, &out_seq, timeout_ms,
                                creq->match_seq)) {
      return 0;
    }
    if (out_seq != creq->match_seq || status != kAckStatusOkDirect) {
      std::cerr << "[ERROR] sender completion ack invalid: seq=" << out_seq
                << " status=" << status << " req=" << creq->id
                << " match_seq=" << creq->match_seq << std::endl;
      return -1;
    }
    if (out_status) *out_status = status;
    return 1;
  };

  // Phase 1: receiver waits for either:
  // 1) direct completion ACK from sender; or
  // 2) sender's bounce relay notification (ipc_cache/use_bounce_buffer=1); or
  // 3) sender's ipc_cache_req asking receiver to advertise destination handle.
  auto phase1_deadline = std::chrono::steady_clock::now() +
                         std::chrono::milliseconds(kIpcControlTimeoutMs);
  bool got_cache_req = false;
  auto handle_relay_cache = [&](IpcCacheWire const& relay_cache,
                                uint64_t relay_seq) -> bool {
    if (relay_seq != creq->match_seq) {
      std::cerr << "[ERROR] recv_ipc_cache seq mismatch: got " << relay_seq
                << " expected " << creq->match_seq << " req " << creq->id
                << std::endl;
      return false;
    }
    if (relay_cache.use_bounce_buffer == 0 ||
        relay_cache.bounce_shm_name[0] == '\0') {
      std::cerr << "[ERROR] unexpected ipc_cache without bounce for req "
                << creq->id << " match_seq " << creq->match_seq << std::endl;
      return false;
    }

    void* relay_bounce_ptr =
        comm_->get_or_open_bounce_shm(relay_cache.bounce_shm_name);
    if (relay_bounce_ptr == nullptr) {
      std::cerr << "[ERROR] get_or_open_bounce_shm failed for relay req "
                << creq->id << std::endl;
      return false;
    }

    GPU_RT_CHECK(gpuSetDevice(comm_->local_gpu_idx()));
    void* actual_dst = creq->data();
    GPU_RT_CHECK(gpuMemcpy(actual_dst, relay_bounce_ptr, creq->size_bytes,
                           gpuMemcpyHostToDevice));
    if (!shm_control_->send_ack(from_rank, creq->match_seq,
                                kAckStatusOkDirect)) {
      std::cerr << "[ERROR] send_ack for relay cache failed, req " << creq->id
                << " match_seq " << creq->match_seq << std::endl;
      return false;
    }
    return true;
  };
  while (!stop_.load(std::memory_order_acquire)) {
    uint32_t status = 0;
    int ack_result = wait_sender_ack(/*timeout_ms=*/0, &status);
    if (ack_result < 0) return false;
    if (ack_result > 0) {
      return true;
    }
    uint64_t req_seq = 0;
    if (shm_control_->recv_ipc_cache_req(from_rank, &req_seq,
                                         /*timeout_ms=*/0, creq->match_seq)) {
      got_cache_req = true;
      break;
    }
    IpcCacheWire relay_cache{};
    uint64_t relay_seq = 0;
    if (shm_control_->recv_ipc_cache(from_rank, relay_cache, &relay_seq,
                                     /*timeout_ms=*/0, creq->match_seq)) {
      return handle_relay_cache(relay_cache, relay_seq);
    }
    if (std::chrono::steady_clock::now() >= phase1_deadline) {
      std::cerr
          << "[ERROR] timed out waiting sender ack/relay/cache-req for req "
          << creq->id << " match_seq " << creq->match_seq << std::endl;
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (stop_.load(std::memory_order_acquire) || !got_cache_req) return false;

  GPU_RT_CHECK(gpuSetDevice(comm_->local_gpu_idx()));
  IpcCacheWire transfer_info{};
  transfer_info.size = creq->size_bytes;
  transfer_info.is_send = 0;
  transfer_info.remote_gpu_idx_ = comm_->local_gpu_idx();
  void* actual_dst = creq->data();

  void* base = nullptr;
  size_t base_size = 0;
  GPU_RT_CHECK(gpuMemGetAddressRange(&base, &base_size, actual_dst));
  GPU_RT_CHECK(gpuIpcGetMemHandle(&transfer_info.handle, base));
  transfer_info.offset = reinterpret_cast<uintptr_t>(actual_dst) -
                         reinterpret_cast<uintptr_t>(base);

  if (!shm_control_->send_ipc_cache(from_rank, creq->match_seq,
                                    transfer_info)) {
    std::cerr << "[ERROR] send_ipc_cache(" << from_rank << ") failed for req "
              << creq->id << " match_seq " << creq->match_seq << std::endl;
    return false;
  }

  // Phase 2: after advertising IPC cache, sender may either:
  // 1) copy directly and send ACK; or
  // 2) fall back to bounce relay and send ipc_cache(use_bounce_buffer=1).
  auto final_deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(kIpcControlTimeoutMs);
  while (!stop_.load(std::memory_order_acquire)) {
    uint32_t status = 0;
    int ack_result = wait_sender_ack(/*timeout_ms=*/0, &status);
    if (ack_result < 0) return false;
    if (ack_result > 0) return true;

    IpcCacheWire relay_cache{};
    uint64_t relay_seq = 0;
    if (shm_control_->recv_ipc_cache(from_rank, relay_cache, &relay_seq,
                                     /*timeout_ms=*/0, creq->match_seq)) {
      return handle_relay_cache(relay_cache, relay_seq);
    }

    if (std::chrono::steady_clock::now() >= final_deadline) {
      std::cerr << "[ERROR] timed out waiting sender completion (ack/relay) "
                << "for req " << creq->id << " match_seq " << creq->match_seq
                << std::endl;
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

void IpcAdapter::complete_task(IpcRequestSlot* req, bool ok) {
  if (!req) return;
  if (!ok) {
    req->mark_failed();
  } else {
    req->complete_one();
  }
}

void IpcAdapter::send_thread_func() {
  while (!stop_.load(std::memory_order_relaxed)) {
    {
      std::unique_lock<std::mutex> lk(cv_mu_);
      cv_.wait(lk, [&] {
        return stop_.load(std::memory_order_relaxed) ||
               pending_send_.load(std::memory_order_relaxed) > 0;
      });
    }
    if (stop_.load(std::memory_order_relaxed)) break;

    unsigned request_id = 0;
    if (jring_sc_dequeue_bulk(send_task_ring_, &request_id, 1, nullptr) != 1) {
      continue;
    }
    pending_send_.fetch_sub(1, std::memory_order_relaxed);
    IpcRequestSlot* req = resolve_request_slot(request_id);
    if (!req) continue;
    complete_task(req, send_one(req));
  }

  while (true) {
    unsigned request_id = 0;
    if (jring_mc_dequeue_bulk(send_task_ring_, &request_id, 1, nullptr) != 1) {
      break;
    }
    IpcRequestSlot* req = resolve_request_slot(request_id);
    if (!req) continue;
    complete_task(req, false);
  }
}

void IpcAdapter::recv_thread_func() {
  while (!stop_.load(std::memory_order_relaxed)) {
    {
      std::unique_lock<std::mutex> lk(cv_mu_);
      cv_.wait(lk, [&] {
        return stop_.load(std::memory_order_relaxed) ||
               pending_recv_.load(std::memory_order_relaxed) > 0;
      });
    }
    if (stop_.load(std::memory_order_relaxed)) break;

    unsigned request_id = 0;
    if (jring_sc_dequeue_bulk(recv_task_ring_, &request_id, 1, nullptr) != 1) {
      continue;
    }
    pending_recv_.fetch_sub(1, std::memory_order_relaxed);
    IpcRequestSlot* req = resolve_request_slot(request_id);
    if (!req) continue;
    complete_task(req, recv_one(req));
  }

  while (true) {
    unsigned request_id = 0;
    if (jring_mc_dequeue_bulk(recv_task_ring_, &request_id, 1, nullptr) != 1) {
      break;
    }
    IpcRequestSlot* req = resolve_request_slot(request_id);
    if (!req) continue;
    complete_task(req, false);
  }
}

}  // namespace Transport
}  // namespace UKernel
