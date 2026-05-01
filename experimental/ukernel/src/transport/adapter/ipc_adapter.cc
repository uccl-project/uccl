#include "ipc_adapter.h"
#include "../communicator.h"
#include "../util/utils.h"
#include <algorithm>
#include <chrono>

namespace UKernel {
namespace Transport {

namespace {

constexpr int kIpcControlTimeoutMs = 50000;
constexpr size_t kTaskRingSize = 1024;
constexpr size_t kIpcSizePerEngine = 1ul << 20;

int remaining_timeout_ms(std::chrono::steady_clock::time_point deadline) {
  auto now = std::chrono::steady_clock::now();
  if (now >= deadline) return 0;
  return static_cast<int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
          .count());
}

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
                       int self_local_id, int local_gpu_idx)
    : next_match_seq_per_peer_(comm->world_size(),
                               std::array<uint64_t, 2>{1, 1}),
      shm_control_(std::make_shared<ShmRingExchanger>(
          comm->rank(), comm->world_size(), std::move(ring_namespace),
          self_local_id)),
      peer_dir_state_(comm->world_size()),
      comm_(comm),
      local_gpu_idx_(local_gpu_idx) {
  send_task_ring_ = create_ring(sizeof(unsigned), kTaskRingSize);
  recv_task_ring_ = create_ring(sizeof(unsigned), kTaskRingSize);
  request_slots_ = std::make_unique<IpcRequestSlot[]>(kRequestSlotCount);
  stop_.store(false);
  send_thread_ = std::thread([this] { send_thread_func(); });
  recv_thread_ = std::thread([this] { recv_thread_func(); });

  int n_streams = 2;
  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
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
  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
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

// ── Connection / path state ────────────────────────────────────────────────

bool IpcAdapter::connect_to(int rank) {
  return shm_control_ != nullptr && shm_control_->connect_to(rank, 30000);
}

bool IpcAdapter::accept_from(int rank) {
  return shm_control_ != nullptr && shm_control_->accept_from(rank, 30000);
}

bool IpcAdapter::ensure_put_path(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_put_path(spec.peer_rank)) return true;
  if (spec.type != PeerConnectType::Connect) return false;
  if (!(std::holds_alternative<IpcPeerConnectSpec>(spec.detail) ||
        std::holds_alternative<std::monostate>(spec.detail))) return false;
  if (!connect_to(spec.peer_rank)) return false;
  std::lock_guard<std::mutex> lk(peer_dir_mu_);
  peer_dir_state_[static_cast<size_t>(spec.peer_rank)].put_ready = true;
  return true;
}

bool IpcAdapter::ensure_wait_path(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_wait_path(spec.peer_rank)) return true;
  if (spec.type != PeerConnectType::Accept) return false;
  if (!(std::holds_alternative<IpcPeerConnectSpec>(spec.detail) ||
        std::holds_alternative<std::monostate>(spec.detail))) return false;
  if (!accept_from(spec.peer_rank)) return false;
  std::lock_guard<std::mutex> lk(peer_dir_mu_);
  peer_dir_state_[static_cast<size_t>(spec.peer_rank)].wait_ready = true;
  return true;
}

void IpcAdapter::set_peer_local_id(int peer_rank, int local_id) {
  if (shm_control_) shm_control_->set_peer_local_id(peer_rank, local_id);
}

void IpcAdapter::close_peer(int peer_rank) {
  if (shm_control_) shm_control_->close_peer(peer_rank);
  if (peer_rank >= 0 && peer_rank < comm_->world_size()) {
    std::lock_guard<std::mutex> lk(peer_dir_mu_);
    peer_dir_state_[static_cast<size_t>(peer_rank)] = {};
  }
}

uint64_t IpcAdapter::next_send_match_seq(int rank) {
  std::lock_guard<std::mutex> lk(match_seq_mu_);
  int src = comm_->rank();
  int dst = rank;
  size_t dir = (src < dst) ? 0u : 1u;
  uint64_t counter = next_match_seq_per_peer_[rank][dir]++;
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

bool IpcAdapter::has_put_path(int peer_rank) const {
  if (peer_rank < 0 || peer_rank >= comm_->world_size()) return false;
  std::lock_guard<std::mutex> lk(peer_dir_mu_);
  return peer_dir_state_[static_cast<size_t>(peer_rank)].put_ready;
}

bool IpcAdapter::has_wait_path(int peer_rank) const {
  if (peer_rank < 0 || peer_rank >= comm_->world_size()) return false;
  std::lock_guard<std::mutex> lk(peer_dir_mu_);
  return peer_dir_state_[static_cast<size_t>(peer_rank)].wait_ready;
}

// ── Slot management ────────────────────────────────────────────────────────

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
                                            std::memory_order_acquire))
      continue;
    uint32_t gen = slot.generation.load(std::memory_order_acquire);
    if (gen == 0) {
      slot.generation.store(1, std::memory_order_release);
      gen = 1;
    }
    slot.id = make_request_id(idx, gen);
    slot.peer_rank = -1;
    slot.match_seq = 0;
    slot.req_type = IpcReqType::DataPut;
    slot.local_ptr = nullptr;
    slot.remote_ptr = nullptr;
    slot.size_bytes = 0;
    *out_request_id = slot.id;
    return &slot;
  }
  return nullptr;
}

IpcAdapter::IpcRequestSlot* IpcAdapter::resolve_request_slot(unsigned id) {
  if (id == 0 || !request_slots_) return nullptr;
  uint32_t generation = request_generation(id);
  if (generation == 0) return nullptr;
  uint32_t idx = request_slot_index(id);
  auto& slot = request_slots_[idx];
  if (slot.generation.load(std::memory_order_acquire) != generation) return nullptr;
  if (slot.state.load(std::memory_order_acquire) == RequestState::Free) return nullptr;
  return &slot;
}

IpcAdapter::IpcRequestSlot* IpcAdapter::resolve_request_slot_const(
    unsigned request_id) const {
  if (request_id == 0 || !request_slots_) return nullptr;
  uint32_t generation = request_generation(request_id);
  if (generation == 0) return nullptr;
  uint32_t idx = request_slot_index(request_id);
  auto const& slot = request_slots_[idx];
  if (slot.generation.load(std::memory_order_acquire) != generation) return nullptr;
  if (slot.state.load(std::memory_order_acquire) == RequestState::Free) return nullptr;
  return const_cast<IpcRequestSlot*>(&slot);
}

void IpcAdapter::release_request_slot(unsigned id) {
  if (id == 0 || !request_slots_) return;
  uint32_t generation = request_generation(id);
  if (generation == 0) return;
  uint32_t idx = request_slot_index(id);
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
                                              std::memory_order_acquire))
      break;
  }
}

bool IpcAdapter::enqueue_request(unsigned id, IpcReqType type) {
  if (stop_.load(std::memory_order_acquire)) return false;
  if (type == IpcReqType::DataPut || type == IpcReqType::Signal) {
    if (send_task_ring_ == nullptr ||
        !enqueue_one_request_id(send_task_ring_, id, stop_))
      return false;
    pending_send_.fetch_add(1, std::memory_order_relaxed);
  } else {
    if (recv_task_ring_ == nullptr ||
        !enqueue_one_request_id(recv_task_ring_, id, stop_))
      return false;
    pending_recv_.fetch_add(1, std::memory_order_relaxed);
  }
  cv_.notify_all();
  return true;
}

// ── Public API ─────────────────────────────────────────────────────────────

unsigned IpcAdapter::put_async(int peer_rank, void* local_ptr,
                               uint32_t local_buffer_id, void* remote_ptr,
                               uint32_t remote_buffer_id, size_t len) {
  (void)local_buffer_id;
  (void)remote_buffer_id;
  if (!has_put_path(peer_rank)) return 0;

  uint64_t match_seq = next_send_match_seq(peer_rank);
  unsigned rid = 0;
  IpcRequestSlot* req = try_acquire_request_slot(&rid);
  if (!req) return 0;
  req->peer_rank = peer_rank;
  req->match_seq = match_seq;
  req->req_type = IpcReqType::DataPut;
  req->local_ptr = local_ptr;
  req->remote_ptr = remote_ptr;
  req->size_bytes = len;
  req->mark_queued(1);
  if (!enqueue_request(rid, IpcReqType::DataPut)) {
    req->mark_failed();
    release_request_slot(rid);
    return 0;
  }
  return rid;
}

unsigned IpcAdapter::signal_async(int peer_rank, uint64_t tag) {
  if (!has_put_path(peer_rank)) return 0;
  unsigned rid = 0;
  IpcRequestSlot* req = try_acquire_request_slot(&rid);
  if (!req) return 0;
  req->peer_rank = peer_rank;
  req->match_seq = tag;
  req->req_type = IpcReqType::Signal;
  req->local_ptr = nullptr;
  req->remote_ptr = nullptr;
  req->size_bytes = 0;
  req->mark_queued(1);
  if (!enqueue_request(rid, IpcReqType::Signal)) {
    req->mark_failed();
    release_request_slot(rid);
    return 0;
  }
  return rid;
}

unsigned IpcAdapter::wait_async(int peer_rank, uint64_t expected_tag,
                                std::optional<WaitTarget> target) {
  if (!has_wait_path(peer_rank)) return 0;

  uint64_t match_seq =
      target.has_value() ? next_recv_match_seq(peer_rank) : expected_tag;
  unsigned rid = 0;
  IpcRequestSlot* req = try_acquire_request_slot(&rid);
  if (!req) return 0;
  req->peer_rank = peer_rank;
  req->match_seq = match_seq;
  req->req_type =
      target.has_value() ? IpcReqType::DataWait : IpcReqType::SignalWait;
  req->local_ptr = target.has_value() ? target->local_ptr : nullptr;
  req->size_bytes = target.has_value() ? target->len : 0;
  req->mark_queued(1);
  if (!enqueue_request(rid, req->req_type)) {
    req->mark_failed();
    release_request_slot(rid);
    return 0;
  }
  return rid;
}

bool IpcAdapter::poll_completion(unsigned id) {
  auto* req = resolve_request_slot_const(id);
  return !req || req->is_finished();
}

bool IpcAdapter::wait_completion(unsigned id) {
  auto* req = resolve_request_slot_const(id);
  if (!req) return true;
  while (!req->is_finished()) std::this_thread::yield();
  return true;
}

bool IpcAdapter::request_failed(unsigned id) {
  auto* req = resolve_request_slot_const(id);
  return req && req->has_failed();
}

void IpcAdapter::release_request(unsigned id) { release_request_slot(id); }

// ── send_one ───────────────────────────────────────────────────────────────
// DataPut: GPU peer copy → send Ack (payload=0).
// Signal: send ack with tag as status.

bool IpcAdapter::send_one(IpcRequestSlot* creq) {
  if (!creq) return false;
  int to_rank = creq->peer_rank;
  creq->mark_running();

  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset = finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });
  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

  if (creq->req_type == IpcReqType::Signal) {
    return shm_control_->send_ack(
        to_rank, creq->match_seq,
        static_cast<uint32_t>(creq->match_seq & 0xFFFFFFFFu));
  }

  // DataPut: GPU peer copy.
  void* src = creq->local_ptr;
  void* dst = creq->remote_ptr;
  if (dst == nullptr) {
    std::cerr << "[ERROR] IPC put_async required remote_ptr, req " << creq->id
              << std::endl;
    return false;
  }

  int remote_gpu = comm_->peer_gpu_idx(to_rank);
  if (remote_gpu < 0) remote_gpu = local_gpu_idx_;

  size_t n_streams =
      std::min(ipc_streams_.size(),
               creq->size_bytes < kIpcSizePerEngine
                   ? size_t{1}
                   : std::max<size_t>(size_t{1}, creq->size_bytes / kIpcSizePerEngine));
  size_t chunk = creq->size_bytes / n_streams;
  for (size_t i = 0; i < n_streams; ++i) {
    void* cs = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(src) + i * chunk);
    void* cd = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(dst) + i * chunk);
    size_t sz = (i == n_streams - 1) ? creq->size_bytes - i * chunk : chunk;
    if (remote_gpu == local_gpu_idx_) {
      GPU_RT_CHECK(gpuMemcpyAsync(cd, cs, sz, gpuMemcpyDeviceToDevice,
                                  ipc_streams_[i]));
    } else {
      GPU_RT_CHECK(gpuMemcpyPeerAsync(cd, remote_gpu, cs, local_gpu_idx_, sz,
                                      ipc_streams_[i]));
    }
  }
  for (auto& s : ipc_streams_) GPU_RT_CHECK(gpuStreamSynchronize(s));

  // payload=0 means data is already in receiver's GPU buffer.
  if (!shm_control_->send_ack(to_rank, creq->match_seq, 0)) {
    std::cerr << "[ERROR] send_ack for direct copy failed, req " << creq->id
              << std::endl;
    return false;
  }
  return true;
}

// ── recv_one ───────────────────────────────────────────────────────────────
// SignalWait: wait for Ack from sender.

bool IpcAdapter::recv_one(IpcRequestSlot* creq) {
  if (!creq) return false;
  int from_rank = creq->peer_rank;
  creq->mark_running();

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(kIpcControlTimeoutMs);

  while (!stop_.load(std::memory_order_acquire)) {
    uint32_t status = 0;
    uint64_t ack_seq = 0;
    if (shm_control_->recv_ack(from_rank, &status, &ack_seq,
                               /*timeout_ms=*/50, creq->match_seq)) {
      if (ack_seq == creq->match_seq) return true;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      std::cerr << "[ERROR] IPC recv timed out, req " << creq->id
                << " match_seq " << creq->match_seq << std::endl;
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

// ── Worker threads ─────────────────────────────────────────────────────────

void IpcAdapter::complete_task(IpcRequestSlot* req, bool ok) {
  if (!req) return;
  if (!ok)
    req->mark_failed();
  else
    req->complete_one();
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

    unsigned id = 0;
    if (jring_sc_dequeue_bulk(send_task_ring_, &id, 1, nullptr) != 1) continue;
    pending_send_.fetch_sub(1, std::memory_order_relaxed);
    auto* req = resolve_request_slot(id);
    if (!req) continue;
    complete_task(req, send_one(req));
  }

  while (true) {
    unsigned id = 0;
    if (jring_mc_dequeue_bulk(send_task_ring_, &id, 1, nullptr) != 1) break;
    auto* req = resolve_request_slot(id);
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

    unsigned id = 0;
    if (jring_sc_dequeue_bulk(recv_task_ring_, &id, 1, nullptr) != 1) continue;
    pending_recv_.fetch_sub(1, std::memory_order_relaxed);
    auto* req = resolve_request_slot(id);
    if (!req) continue;
    complete_task(req, recv_one(req));
  }

  while (true) {
    unsigned id = 0;
    if (jring_mc_dequeue_bulk(recv_task_ring_, &id, 1, nullptr) != 1) break;
    auto* req = resolve_request_slot(id);
    if (!req) continue;
    complete_task(req, false);
  }
}

}  // namespace Transport
}  // namespace UKernel
