#include "ipc_adapter.h"
#include "../communicator.h"
#include "../util/utils.h"
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

namespace {

constexpr int kIpcControlTimeoutMs = 50000;
constexpr size_t kTaskRingSize = 1024;
constexpr size_t kIpcSizePerEngine = 1ul << 20;

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
                       int local_gpu_idx)
    : seqs_(comm->world_size(),
                               std::array<uint64_t, 2>{1, 1}),
      ns_(std::move(ring_namespace)),
      dir_state_(comm->world_size()),
      comps_(comm->world_size()),
      comm_(comm),
      gpu_id_(local_gpu_idx) {
  send_ring_ = create_ring(sizeof(unsigned), kTaskRingSize);
  recv_ring_ = create_ring(sizeof(unsigned), kTaskRingSize);
  if (send_ring_ == nullptr || recv_ring_ == nullptr) {
    if (send_ring_ != nullptr) {
      free(send_ring_);
      send_ring_ = nullptr;
    }
    if (recv_ring_ != nullptr) {
      free(recv_ring_);
      recv_ring_ = nullptr;
    }
    throw std::runtime_error("IpcAdapter failed to allocate task rings");
  }
  slots_ = std::make_unique<Slot[]>(kSlotCnt);

  int n_streams = 2;
  GPU_RT_CHECK(gpuSetDevice(gpu_id_));
  ipc_ctx_.resize(n_streams);
  for (int i = 0; i < n_streams; ++i) {
    GPU_RT_CHECK(
        gpuStreamCreateWithFlags(&ipc_ctx_[i].first, gpuStreamNonBlocking));
    GPU_RT_CHECK(
        gpuEventCreateWithFlags(&ipc_ctx_[i].second, gpuEventDisableTiming));
  }

  stop_.store(false, std::memory_order_release);
  send_th_ = std::thread([this] { send_worker(); });
  recv_th_ = std::thread([this] { recv_worker(); });
}

IpcAdapter::~IpcAdapter() { shutdown(); }

void IpcAdapter::shutdown() {
  bool expected = false;
  if (!stop_.compare_exchange_strong(expected, true)) return;

  stop_.store(true, std::memory_order_release);
  if (send_th_.joinable()) send_th_.join();
  if (recv_th_.joinable()) recv_th_.join();

  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  GPU_RT_CHECK(gpuSetDevice(gpu_id_));
  for (auto& ctx : ipc_ctx_) {
    if (ctx.first != nullptr) GPU_RT_CHECK(gpuStreamDestroy(ctx.first));
    if (ctx.second != nullptr) GPU_RT_CHECK(gpuEventDestroy(ctx.second));
  }
  ipc_ctx_.clear();
  GPU_RT_CHECK(gpuSetDevice(orig_device));

  if (send_ring_) {
    free(send_ring_);
    send_ring_ = nullptr;
  }
  if (recv_ring_) {
    free(recv_ring_);
    recv_ring_ = nullptr;
  }
  for (size_t r = 0; r < comps_.size(); ++r)
    close_comp(static_cast<int>(r));
}

// ── Data-completion SHM (fast path for IPC GPU data transfers) ───────────

std::string IpcAdapter::comp_shm_name(int peer_rank) const {
  return Format("/uk_cmpl_%s_p%d_p%d", ns_.c_str(), peer_rank,
                comm_->rank());
}

bool IpcAdapter::ensure_local_comp(int peer_rank) {
  auto& pc = comps_[static_cast<size_t>(peer_rank)];
  if (pc.local != nullptr) return true;

  pc.shm_name = comp_shm_name(peer_rank);
  size_t sz = sizeof(IpcDataCompletion);
  int fd = shm_open(pc.shm_name.c_str(), O_CREAT | O_RDWR, 0666);
  if (fd < 0) return false;
  if (ftruncate(fd, static_cast<off_t>(sz)) != 0) {
    close(fd);
    return false;
  }
  void* ptr = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    close(fd);
    return false;
  }
  pc.shm_fd = fd;
  pc.shm_size = sz;
  pc.local = new (ptr) IpcDataCompletion();
  return true;
}

bool IpcAdapter::ensure_remote_comp(int peer_rank) {
  auto& pc = comps_[static_cast<size_t>(peer_rank)];
  if (pc.remote != nullptr) return true;

  // The remote completion SHM is the peer's *local* completion — the
  // peer created it with itself as receiver.
  std::string remote_name = Format(
      "/uk_cmpl_%s_p%d_p%d", ns_.c_str(), comm_->rank(), peer_rank);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  size_t sz = sizeof(IpcDataCompletion);
  while (true) {
    int fd = shm_open(remote_name.c_str(), O_RDWR, 0666);
    if (fd >= 0) {
      void* ptr = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if (ptr != MAP_FAILED) {
        pc.remote = reinterpret_cast<IpcDataCompletion*>(ptr);
        return true;
      }
      close(fd);
    }
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

void IpcAdapter::close_comp(int peer_rank) {
  auto& pc = comps_[static_cast<size_t>(peer_rank)];
  if (pc.local != nullptr) {
    munmap(pc.local, pc.shm_size);
    pc.local = nullptr;
  }
  if (pc.remote != nullptr) {
    auto* remote_copy = pc.remote;
    pc.remote = nullptr;
    munmap(remote_copy, sizeof(IpcDataCompletion));
  }
  if (pc.shm_fd >= 0) {
    close(pc.shm_fd);
    pc.shm_fd = -1;
  }
  if (!pc.shm_name.empty()) {
    shm_unlink(pc.shm_name.c_str());
    pc.shm_name.clear();
  }
}

// ── Connection / path state ────────────────────────────────────────────────

bool IpcAdapter::connect_to(int rank) { return ensure_remote_comp(rank); }

bool IpcAdapter::accept_from(int rank) { return ensure_local_comp(rank); }

bool IpcAdapter::ensure_put_path(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_put_path(spec.peer_rank)) return true;
  if (spec.type != PeerConnectType::Connect) return false;
  if (!(std::holds_alternative<IpcPeerConnectSpec>(spec.detail) ||
        std::holds_alternative<std::monostate>(spec.detail)))
    return false;
  if (!connect_to(spec.peer_rank)) return false;
  std::lock_guard<std::mutex> lk(dir_mu_);
  dir_state_[static_cast<size_t>(spec.peer_rank)].put_ready = true;
  return true;
}

bool IpcAdapter::ensure_wait_path(PeerConnectSpec const& spec) {
  if (spec.peer_rank < 0) return false;
  if (has_wait_path(spec.peer_rank)) return true;
  if (spec.type != PeerConnectType::Accept) return false;
  if (!(std::holds_alternative<IpcPeerConnectSpec>(spec.detail) ||
        std::holds_alternative<std::monostate>(spec.detail)))
    return false;
  if (!accept_from(spec.peer_rank)) return false;
  std::lock_guard<std::mutex> lk(dir_mu_);
  dir_state_[static_cast<size_t>(spec.peer_rank)].wait_ready = true;
  return true;
}

void IpcAdapter::close_comp(int peer_rank) {
  close_comp(peer_rank);
  if (peer_rank >= 0 && peer_rank < comm_->world_size()) {
    std::lock_guard<std::mutex> lk(dir_mu_);
    dir_state_[static_cast<size_t>(peer_rank)] = {};
  }
}

uint64_t IpcAdapter::next_send_match_seq(int rank) {
  std::lock_guard<std::mutex> lk(seq_mu_);
  int src = comm_->rank();
  int dst = rank;
  size_t dir = (src < dst) ? 0u : 1u;
  uint64_t counter = seqs_[rank][dir]++;
  return (counter << 1) | static_cast<uint64_t>(dir);
}

uint64_t IpcAdapter::next_recv_match_seq(int rank) {
  std::lock_guard<std::mutex> lk(seq_mu_);
  int src = rank;
  int dst = comm_->rank();
  size_t dir = (src < dst) ? 0u : 1u;
  uint64_t counter = seqs_[rank][dir]++;
  return (counter << 1) | static_cast<uint64_t>(dir);
}

bool IpcAdapter::has_put_path(int peer_rank) const {
  if (peer_rank < 0 || peer_rank >= comm_->world_size()) return false;
  std::lock_guard<std::mutex> lk(dir_mu_);
  return dir_state_[static_cast<size_t>(peer_rank)].put_ready;
}

bool IpcAdapter::has_wait_path(int peer_rank) const {
  if (peer_rank < 0 || peer_rank >= comm_->world_size()) return false;
  std::lock_guard<std::mutex> lk(dir_mu_);
  return dir_state_[static_cast<size_t>(peer_rank)].wait_ready;
}

// ── Slot management ────────────────────────────────────────────────────────

IpcAdapter::Slot* IpcAdapter::acquire_slot(
    unsigned* out_request_id) {
  if (out_request_id == nullptr || !slots_) return nullptr;
  for (uint32_t n = 0; n < kSlotCnt; ++n) {
    uint32_t idx =
        alloc_cursor_.fetch_add(1, std::memory_order_relaxed) &
        kSlotMask;
    auto& slot = slots_[idx];
    ReqState expected = ReqState::Free;
    if (!slot.state.compare_exchange_strong(expected, ReqState::Running,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire))
      continue;
    uint32_t gen = slot.generation.load(std::memory_order_acquire);
    if (gen == 0) {
      slot.generation.store(1, std::memory_order_release);
      gen = 1;
    }
    slot.id = make_rid(idx, gen);
    slot.peer_rank = -1;
    slot.match_seq = 0;
    slot.req_type = ReqType::DataPut;
    slot.local_ptr = nullptr;
    slot.remote_ptr = nullptr;
    slot.size_bytes = 0;
    *out_request_id = slot.id;
    return &slot;
  }
  return nullptr;
}

IpcAdapter::Slot* IpcAdapter::resolve_slot(unsigned id) {
  if (id == 0 || !slots_) return nullptr;
  uint32_t generation = slot_gen(id);
  if (generation == 0) return nullptr;
  uint32_t idx = slot_idx(id);
  auto& slot = slots_[idx];
  if (slot.generation.load(std::memory_order_acquire) != generation)
    return nullptr;
  if (slot.state.load(std::memory_order_acquire) == ReqState::Free)
    return nullptr;
  return &slot;
}

IpcAdapter::Slot* IpcAdapter::resolve_slot(
    unsigned request_id) const {
  if (request_id == 0 || !slots_) return nullptr;
  uint32_t generation = slot_gen(request_id);
  if (generation == 0) return nullptr;
  uint32_t idx = slot_idx(request_id);
  auto const& slot = slots_[idx];
  if (slot.generation.load(std::memory_order_acquire) != generation)
    return nullptr;
  if (slot.state.load(std::memory_order_acquire) == ReqState::Free)
    return nullptr;
  return const_cast<Slot*>(&slot);
}

void IpcAdapter::release_slot(unsigned id) {
  if (id == 0 || !slots_) return;
  uint32_t generation = slot_gen(id);
  if (generation == 0) return;
  uint32_t idx = slot_idx(id);
  auto& slot = slots_[idx];
  if (slot.generation.load(std::memory_order_acquire) != generation) return;
  auto st = slot.state.load(std::memory_order_acquire);
  if (st == ReqState::Running || st == ReqState::Queued) return;
  slot.state.store(ReqState::Free, std::memory_order_release);
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

bool IpcAdapter::enqueue(unsigned id, ReqType type) {
  if (stop_.load(std::memory_order_acquire)) return false;
  if (type == ReqType::DataPut || type == ReqType::Signal) {
    if (send_ring_ == nullptr ||
        !enqueue_one_request_id(send_ring_, id, stop_))
      return false;
  } else {
    if (recv_ring_ == nullptr ||
        !enqueue_one_request_id(recv_ring_, id, stop_))
      return false;
  }
  return true;
}

// ── Public API ─────────────────────────────────────────────────────────────

unsigned IpcAdapter::put_async(int peer_rank, void* local_ptr,
                               uint32_t local_buffer_id, void* remote_ptr,
                               uint32_t remote_buffer_id, size_t len,
                               unsigned comm_rid) {
  (void)local_buffer_id;
  (void)remote_buffer_id;
  if (!has_put_path(peer_rank)) return 0;

  uint64_t match_seq = next_send_match_seq(peer_rank);
  unsigned rid = 0;
  Slot* req = acquire_slot(&rid);
  if (!req) return 0;
  req->comm_rid = comm_rid;
  req->peer_rank = peer_rank;
  req->match_seq = match_seq;
  req->req_type = ReqType::DataPut;
  req->local_ptr = local_ptr;
  req->remote_ptr = remote_ptr;
  req->size_bytes = len;
  req->mark_queued(1);
  if (!enqueue(rid, ReqType::DataPut)) {
    req->mark_failed();
    release_slot(rid);
    return 0;
  }
  return rid;
}

unsigned IpcAdapter::signal_async(int peer_rank, uint64_t tag,
                                  unsigned comm_rid) {
  if (!has_put_path(peer_rank)) return 0;
  unsigned rid = 0;
  Slot* req = acquire_slot(&rid);
  if (!req) return 0;
  req->comm_rid = comm_rid;
  req->peer_rank = peer_rank;
  req->match_seq = tag;
  req->req_type = ReqType::Signal;
  req->local_ptr = nullptr;
  req->remote_ptr = nullptr;
  req->size_bytes = 0;
  req->mark_queued(1);
  if (!enqueue(rid, ReqType::Signal)) {
    req->mark_failed();
    release_slot(rid);
    return 0;
  }
  return rid;
}

unsigned IpcAdapter::wait_async(int peer_rank, uint64_t expected_tag,
                                std::optional<WaitTarget> target,
                                unsigned comm_rid) {
  if (!has_wait_path(peer_rank)) return 0;

  uint64_t match_seq =
      target.has_value() ? next_recv_match_seq(peer_rank) : expected_tag;
  unsigned rid = 0;
  Slot* req = acquire_slot(&rid);
  if (!req) return 0;
  req->comm_rid = comm_rid;
  req->peer_rank = peer_rank;
  req->match_seq = match_seq;
  req->req_type =
      target.has_value() ? ReqType::DataWait : ReqType::SignalWait;
  req->local_ptr = target.has_value() ? target->local_ptr : nullptr;
  req->size_bytes = target.has_value() ? target->len : 0;
  req->mark_queued(1);
  if (!enqueue(rid, req->req_type)) {
    req->mark_failed();
    release_slot(rid);
    return 0;
  }
  return rid;
}

void IpcAdapter::release(unsigned id) { release_slot(id); }

void IpcAdapter::done(Slot* req, bool ok) {
  if (!req) return;
  if (!ok)
    req->mark_failed();
  else
    req->complete_one();
  publish_completion(req->comm_rid, !ok);
}

void IpcAdapter::send_worker() {
  GPU_RT_CHECK(gpuSetDevice(gpu_id_));
  while (!stop_.load(std::memory_order_relaxed)) {
    unsigned id = 0;
    if (jring_sc_dequeue_bulk(send_ring_, &id, 1, nullptr) != 1) {
      std::this_thread::yield();
      continue;
    }
    auto* req = resolve_slot(id);
    if (!req) continue;
    done(req, send_one(req));
  }

  while (true) {
    unsigned id = 0;
    if (jring_mc_dequeue_bulk(send_ring_, &id, 1, nullptr) != 1) break;
    auto* req = resolve_slot(id);
    if (!req) continue;
    done(req, false);
  }
}

void IpcAdapter::recv_worker() {
  while (!stop_.load(std::memory_order_relaxed)) {
    unsigned id = 0;
    if (jring_sc_dequeue_bulk(recv_ring_, &id, 1, nullptr) != 1) {
      std::this_thread::yield();
      continue;
    }
    auto* req = resolve_slot(id);
    if (!req) continue;
    done(req, recv_one(req));
  }

  while (true) {
    unsigned id = 0;
    if (jring_mc_dequeue_bulk(recv_ring_, &id, 1, nullptr) != 1) break;
    auto* req = resolve_slot(id);
    if (!req) continue;
    done(req, false);
  }
}

}  // namespace Transport
}  // namespace UKernel
