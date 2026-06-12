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

template <typename T>
bool enqueue_elem(jring_t* ring, T const& elem, std::atomic<bool> const& stop) {
  while (!stop.load(std::memory_order_acquire) &&
         jring_mp_enqueue_bulk(ring, &elem, 1, nullptr) != 1)
    std::this_thread::yield();
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
  send_ring_ = create_ring(sizeof(RingElem), kTaskRingSize);
  recv_ring_ = create_ring(sizeof(RingElem), kTaskRingSize);
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

  // Clean up any stale IPC completion SHM from previous crashed runs
  for (int r = 0; r < comm_->world_size(); ++r) {
    if (r == comm_->rank()) continue;
    std::string name = comp_shm_name(r);
    shm_unlink(name.c_str());
  }

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
  shm_unlink(pc.shm_name.c_str());  // Clean up stale SHM from previous crashed run
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
  if (pc.local) { munmap(pc.local, pc.shm_size); pc.local = nullptr; }
  if (pc.remote) { munmap(pc.remote, sizeof(IpcDataCompletion)); pc.remote = nullptr; }
  if (pc.shm_fd >= 0) { close(pc.shm_fd); pc.shm_fd = -1; }
  if (!pc.shm_name.empty()) { shm_unlink(pc.shm_name.c_str()); pc.shm_name.clear(); }
  if (peer_rank < static_cast<int>(dir_state_.size())) {
    std::lock_guard<std::mutex> lk(dir_mu_);
    dir_state_[static_cast<size_t>(peer_rank)] = {};
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
  dir_state_[static_cast<size_t>(spec.peer_rank)].first = true;
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
  dir_state_[static_cast<size_t>(spec.peer_rank)].second = true;
  return true;
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
  return dir_state_[static_cast<size_t>(peer_rank)].first;
}

bool IpcAdapter::has_wait_path(int peer_rank) const {
  if (peer_rank < 0 || peer_rank >= comm_->world_size()) return false;
  std::lock_guard<std::mutex> lk(dir_mu_);
  return dir_state_[static_cast<size_t>(peer_rank)].second;
}

// ── Public API ─────────────────────────────────────────────────────────────

unsigned IpcAdapter::put_async(int peer, void* local_ptr, uint32_t,
                               void* remote_ptr, uint32_t, size_t bytes,
                               unsigned comm_rid) {
  if (!has_put_path(peer)) return 0;
  RingElem e{comm_rid, peer, ReqType::DataPut, next_send_match_seq(peer),
             local_ptr, remote_ptr, bytes};
  if (!enqueue_elem(send_ring_, e, stop_)) return 0;
  return 1;
}

unsigned IpcAdapter::signal_async(int peer, uint64_t tag, unsigned comm_rid) {
  if (!has_put_path(peer)) return 0;
  RingElem e{comm_rid, peer, ReqType::Signal, tag, nullptr, nullptr, 0};
  if (!enqueue_elem(send_ring_, e, stop_)) return 0;
  return 1;
}

unsigned IpcAdapter::wait_async(int peer, uint64_t tag,
                                std::optional<WaitTarget> target,
                                unsigned comm_rid) {
  if (!has_wait_path(peer)) return 0;
  uint64_t seq = target ? next_recv_match_seq(peer) : tag;
  void* ptr = target ? target->local_ptr : nullptr;
  size_t len = target ? target->len : 0;
  ReqType t = target ? ReqType::DataWait : ReqType::SignalWait;
  RingElem e{comm_rid, peer, t, seq, ptr, nullptr, len};
  if (!enqueue_elem(recv_ring_, e, stop_)) return 0;
  return 1;
}

void IpcAdapter::send_worker() {
  GPU_RT_CHECK(gpuSetDevice(gpu_id_));
  RingElem e;
  while (!stop_.load(std::memory_order_relaxed)) {
    if (jring_sc_dequeue_bulk(send_ring_, &e, 1, nullptr) != 1) {
      std::this_thread::yield();
      continue;
    }
    if (e.type == ReqType::DataPut) {
      bool ok = send_one(&e);
      if (ok) {
        size_t dir = (comm_->rank() < e.peer) ? 0u : 1u;
        comps_[e.peer].remote->last_completed[dir].store(
            e.seq, std::memory_order_release);
      }
      publish_completion(e.comm_rid, !ok);
    } else if (e.type == ReqType::Signal) {
      size_t dir = (comm_->rank() < e.peer) ? 0u : 1u;
      comps_[e.peer].remote->last_completed[dir].store(
          e.seq, std::memory_order_release);
      publish_completion(e.comm_rid, false);
    } else {
      publish_completion(e.comm_rid, true);
    }
  }
  RingElem drain;
  while (jring_mc_dequeue_bulk(send_ring_, &drain, 1, nullptr) == 1)
    publish_completion(drain.comm_rid, true);
}

void IpcAdapter::recv_worker() {
  RingElem e;
  while (!stop_.load(std::memory_order_relaxed)) {
    if (jring_sc_dequeue_bulk(recv_ring_, &e, 1, nullptr) != 1) {
      std::this_thread::yield();
      continue;
    }
    bool ok = (e.type == ReqType::DataWait || e.type == ReqType::SignalWait)
                  ? recv_one(&e) : false;
    publish_completion(e.comm_rid, !ok);
  }
  RingElem drain;
  while (jring_mc_dequeue_bulk(recv_ring_, &drain, 1, nullptr) == 1)
    publish_completion(drain.comm_rid, true);
}

bool IpcAdapter::send_one(RingElem* e) {
  if (!e || e->type != ReqType::DataPut) return false;
  void* src = e->local_ptr;
  void* dst = e->remote_ptr;
  if (!dst) {
    std::cerr << "[ERROR] IPC put_async no remote_ptr\n";
    return false;
  }

  int remote_gpu = comm_->peer_gpu_idx(e->peer);
  if (remote_gpu < 0) remote_gpu = gpu_id_;

  size_t bytes = e->bytes;
  size_t n_total = ipc_ctx_.size();
  size_t needed = (bytes + kIpcSizePerEngine - 1) / kIpcSizePerEngine;
  if (needed > n_total) needed = n_total;
  if (needed == 0) needed = 1;

  for (size_t i = 0; i < needed; ++i) {
    size_t offset = i * kIpcSizePerEngine;
    size_t chunk = std::min(kIpcSizePerEngine, bytes - offset);
    if (chunk == 0) break;
    char* src_chunk = static_cast<char*>(src) + offset;
    char* dst_chunk = static_cast<char*>(dst) + offset;
    gpuStream_t stream = ipc_ctx_[i].first;
    if (remote_gpu == gpu_id_)
      GPU_RT_CHECK(gpuMemcpyAsync(dst_chunk, src_chunk, chunk,
                                  gpuMemcpyDeviceToDevice, stream));
    else
      GPU_RT_CHECK(gpuMemcpyPeerAsync(dst_chunk, remote_gpu, src_chunk,
                                      gpu_id_, chunk, stream));
    GPU_RT_CHECK(gpuEventRecord(ipc_ctx_[i].second, stream));
  }

  for (size_t i = 0; i < needed; ++i)
    GPU_RT_CHECK(gpuEventSynchronize(ipc_ctx_[i].second));

  return true;
}

bool IpcAdapter::recv_one(RingElem* e) {
  if (!e) return false;
  if (e->type != ReqType::DataWait && e->type != ReqType::SignalWait)
    return false;

  size_t dir = (e->peer < comm_->rank()) ? 0u : 1u;
  uint64_t expected = e->seq;
  auto* counter = &comps_[e->peer].local->last_completed[dir];

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(kIpcControlTimeoutMs);
  while (!stop_.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    if (counter->load(std::memory_order_acquire) >= expected) return true;
    std::this_thread::yield();
  }
  if (!stop_.load(std::memory_order_acquire)) {
    std::cerr << "[ERROR] IPC recv timed out, peer " << e->peer
              << " match_seq " << e->seq << std::endl;
  }
  return false;
}

}  // namespace Transport
}  // namespace UKernel
