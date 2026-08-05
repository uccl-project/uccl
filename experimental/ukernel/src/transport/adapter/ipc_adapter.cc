#include "ipc_adapter.h"
#include "../communicator.h"
#include "../util/utils.h"
#include "util/uk_debug.h"
#include <algorithm>
#include <chrono>
#include <deque>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

// Host timestamp (us) for [tss] diagnostics (UK_CCL_DEBUG>=1): breaks
// down the put-completion -> signal chain on multi-rank allreduce.
static inline long long tss_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

namespace {

constexpr int kIpcControlTimeoutMs = 50000;
constexpr size_t kTaskRingSize = 1024;
// Default in-flight put window PER PEER (see IpcAdapter ctor;
// UK_CCL_IPC_BATCH overrides, 1 = old per-put sync behavior). Each peer
// owns an independent window + stream pool, so N peers run N x window
// copies concurrently instead of sharing one global window (which
// serialized 8-rank alltoall fan-out to 4 in-flight puts). The sliding
// window keeps up to this many copies per peer in flight and completes
// each peer FIFO (the receiver matches per-(peer,direction) sequences);
// different peers complete out of order. B300 allreduce sweep
// (256M, LT=8/BLK=32) measured window 4-64 all at 462-498 GB/s median —
// 4 per peer keeps the same host event pressure as before.
constexpr size_t kIpcSendBatchDefault = 4;

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
    : seqs_(comm->world_size(), std::array<uint64_t, 2>{1, 1}),
      ns_(std::move(ring_namespace)),
      dir_state_(comm->world_size()),
      comps_(comm->world_size()),
      comm_(comm),
      gpu_id_(local_gpu_idx) {
  send_rings_.resize(comm->world_size(), nullptr);
  for (int r = 0; r < comm->world_size(); ++r) {
    if (r == comm->rank()) continue;
    send_rings_[static_cast<size_t>(r)] =
        create_ring(sizeof(RingElem), kTaskRingSize);
  }
  recv_ring_ = create_ring(sizeof(RingElem), kTaskRingSize);
  for (size_t r = 0; r < send_rings_.size(); ++r)
    if (static_cast<int>(r) != comm->rank() && send_rings_[r] == nullptr)
      throw std::runtime_error("IpcAdapter failed to allocate send ring");
  if (recv_ring_ == nullptr) {
    for (auto* ring : send_rings_)
      if (ring != nullptr) free(ring);
    send_rings_.clear();
    throw std::runtime_error("IpcAdapter failed to allocate recv ring");
  }

  if (char const* v = std::getenv("UK_CCL_IPC_STREAMS_PER_PEER")) {
    size_t s = static_cast<size_t>(std::stoul(v));
    if (s > 0 && s <= 16) streams_per_peer_ = s;
  }
  size_t n_streams =
      static_cast<size_t>(std::max(1, comm->world_size())) * streams_per_peer_;
  GPU_RT_CHECK(gpuSetDevice(gpu_id_));
  ipc_ctx_.resize(n_streams);
  for (int i = 0; i < n_streams; ++i)
    GPU_RT_CHECK(gpuStreamCreateWithFlags(&ipc_ctx_[i], gpuStreamNonBlocking));

  stop_.store(false, std::memory_order_release);

  send_batch_ = kIpcSendBatchDefault;
  if (char const* v = std::getenv("UK_CCL_IPC_BATCH")) {
    size_t b = static_cast<size_t>(std::stoul(v));
    if (b > 0) send_batch_ = b;
  }

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

void IpcAdapter::stop() { stop_.store(true, std::memory_order_release); }

void IpcAdapter::shutdown() {
  stop_.store(true, std::memory_order_release);

  if (send_th_.joinable()) send_th_.join();
  if (recv_th_.joinable()) recv_th_.join();

  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  GPU_RT_CHECK(gpuSetDevice(gpu_id_));
  for (auto& s : ipc_ctx_)
    if (s != nullptr) GPU_RT_CHECK(gpuStreamDestroy(s));
  ipc_ctx_.clear();
  GPU_RT_CHECK(gpuSetDevice(orig_device));

  for (auto* ring : send_rings_) {
    if (ring != nullptr) free(ring);
  }
  send_rings_.clear();
  if (recv_ring_) {
    free(recv_ring_);
    recv_ring_ = nullptr;
  }
  for (size_t r = 0; r < comps_.size(); ++r) close_comp(static_cast<int>(r));
}

// Data-completion SHM (fast path for IPC GPU data transfers)

std::string IpcAdapter::comp_shm_name(int peer_rank) const {
  return Format("/uk_cmpl_%s_p%d_p%d", ns_.c_str(), peer_rank, comm_->rank());
}

bool IpcAdapter::ensure_local_comp(int peer_rank) {
  auto& pc = comps_[static_cast<size_t>(peer_rank)];
  if (pc.local != nullptr) return true;

  pc.shm_name = comp_shm_name(peer_rank);
  shm_unlink(
      pc.shm_name.c_str());  // Clean up stale SHM from previous crashed run
  size_t sz = sizeof(IpcDataCompletion) + sizeof(PeerSignalRing);
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
  pc.signal_ring = new (static_cast<char*>(ptr) + sizeof(IpcDataCompletion))
      PeerSignalRing();
  return true;
}

bool IpcAdapter::ensure_remote_comp(int peer_rank) {
  auto& pc = comps_[static_cast<size_t>(peer_rank)];
  if (pc.remote != nullptr) return true;

  // The remote completion SHM is the peer's *local* completion — the
  // peer created it with itself as receiver.
  std::string remote_name =
      Format("/uk_cmpl_%s_p%d_p%d", ns_.c_str(), comm_->rank(), peer_rank);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  size_t sz = sizeof(IpcDataCompletion) + sizeof(PeerSignalRing);
  while (true) {
    int fd = shm_open(remote_name.c_str(), O_RDWR, 0666);
    if (fd >= 0) {
      void* ptr = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if (ptr != MAP_FAILED) {
        pc.remote = reinterpret_cast<IpcDataCompletion*>(ptr);
        // Zero-copy GPU mapping: lets device kernels write fused
        // PutSignal tags straight into this ring. Optional — failure
        // only disables device-side fusion for this peer. Requires GPU
        // system atomics to host memory (the ring claim is an
        // atomicAdd from the kernel).
        int atomics_ok = 1;
#ifndef __HIP_PLATFORM_AMD__
        atomics_ok = 0;
        if (gpuDeviceGetAttribute(&atomics_ok,
                                  gpuDevAttrHostNativeAtomicSupported,
                                  gpu_id_) != gpuSuccess)
          atomics_ok = 0;
#endif
        int prev_dev = -1;
        if (atomics_ok && gpuGetDevice(&prev_dev) == gpuSuccess) {
          if (gpuSetDevice(gpu_id_) == gpuSuccess &&
              gpuHostRegister(ptr, sz, gpuHostRegisterMapped) == gpuSuccess) {
            void* dptr = nullptr;
            if (gpuHostGetDevicePointer(&dptr, ptr, 0) == gpuSuccess)
              pc.remote_device = dptr;
          }
          gpuSetDevice(prev_dev);
        }
        if (!pc.remote_device)
          UK_DBG(UK_DBG_LVL_TPT,
                 "[ipc r%d] peer %d signal ring not GPU-mapped "
                 "(atomics_ok=%d) — device-side fusion disabled",
                 comm_->rank(), peer_rank, atomics_ok);
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
  if (pc.local) {
    munmap(pc.local, pc.shm_size);
    pc.local = nullptr;
  }
  if (pc.remote) {
    munmap(pc.remote, sizeof(IpcDataCompletion));
    pc.remote = nullptr;
  }
  if (pc.shm_fd >= 0) {
    close(pc.shm_fd);
    pc.shm_fd = -1;
  }
  if (!pc.shm_name.empty()) {
    shm_unlink(pc.shm_name.c_str());
    pc.shm_name.clear();
  }
  if (peer_rank < static_cast<int>(dir_state_.size())) {
    std::lock_guard<std::mutex> lk(dir_mu_);
    dir_state_[static_cast<size_t>(peer_rank)] = {};
  }
}

// Connection / path state

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

// Public API

unsigned IpcAdapter::send_put_async(int peer, void* local_ptr, uint32_t,
                                    void* remote_ptr, uint32_t, size_t bytes,
                                    unsigned comm_rid) {
  if (!has_put_path(peer)) return 0;
  if (peer < 0 || static_cast<size_t>(peer) >= send_rings_.size() ||
      send_rings_[static_cast<size_t>(peer)] == nullptr)
    return 0;
  RingElem e{
      comm_rid,   peer, ReqType::DataPut, next_send_match_seq(peer), local_ptr,
      remote_ptr, bytes};
  if (!enqueue_elem(send_rings_[static_cast<size_t>(peer)], e, stop_)) return 0;
  return 1;
}

bool IpcAdapter::write_signal_ring(int peer, uint64_t tag) {
  // Inline fast path: write tag to remote peer's signal ring in SHM.
  auto* remote_ring = reinterpret_cast<PeerSignalRing*>(
      reinterpret_cast<char*>(comps_[peer].remote) + sizeof(IpcDataCompletion));

  // Multi-producer claim: plain signals come from the executor's enqueue
  // thread, fused PutSignal writes come from the send worker. Claim with
  // fetch_add; the per-slot ready flag tolerates out-of-order publishes
  // (the consumer stops at the first unready slot and catches up later).
  // In-flight claims are bounded by the producer count (≤ 2), far below
  // the ring size, so a new claim can never lap a stalled publisher.
  uint64_t w = remote_ring->write_idx.fetch_add(1, std::memory_order_acq_rel);
  size_t idx = w & (kSignalRingSize - 1);

  // Back-pressure: wait until this slot's previous lap was consumed.
  while (remote_ring->slots[idx].ready.load(std::memory_order_acquire)) {
    if (stop_.load(std::memory_order_relaxed)) return false;
    std::this_thread::yield();
  }

  remote_ring->slots[idx].tag = tag;
  remote_ring->slots[idx].ready.store(true, std::memory_order_release);
  return true;
}

unsigned IpcAdapter::send_signal_async(int peer, uint64_t tag,
                                       unsigned comm_rid) {
  if (!has_put_path(peer)) return 0;
  if (!write_signal_ring(peer, tag)) return 0;
  UK_DBG(UK_DBG_LVL_TPT, "[ipc-sig-send r%d] tag=%lu to p%d", comm_->rank(),
         (unsigned long)tag, peer);
  publish_sig_send_completion(comm_rid, false);
  return 1;
}

unsigned IpcAdapter::send_put_signal_async(int peer, void* local_ptr, uint32_t,
                                           void* remote_ptr, uint32_t,
                                           size_t len, uint64_t tag,
                                           unsigned comm_rid) {
  if (!has_put_path(peer)) return 0;
  if (peer < 0 || static_cast<size_t>(peer) >= send_rings_.size() ||
      send_rings_[static_cast<size_t>(peer)] == nullptr)
    return 0;
  RingElem e{comm_rid,
             peer,
             ReqType::PutSignal,
             next_send_match_seq(peer),
             local_ptr,
             remote_ptr,
             len,
             tag};
  if (!enqueue_elem(send_rings_[static_cast<size_t>(peer)], e, stop_)) return 0;
  return 1;
}

unsigned IpcAdapter::wait_signal_async(int peer, uint64_t /*tag*/,
                                       std::optional<WaitTarget> target,
                                       unsigned comm_rid) {
  if (!has_wait_path(peer)) return 0;

  if (target) {
    // DataWait: GPU-copy completion — through jring + recv_worker.
    uint64_t seq = next_recv_match_seq(peer);
    RingElem e{comm_rid,          peer,    ReqType::DataWait, seq,
               target->local_ptr, nullptr, target->len};
    if (!enqueue_elem(recv_ring_, e, stop_)) return 0;
    return 1;
  }

  // SignalWait: not handled by the IPC adapter inline; Communicator uses
  // drain_signal_tags() to drain incoming signal tags.
  return 1;
}

void* IpcAdapter::peer_signal_ring_device_ptr(int peer) const {
  if (peer < 0 || static_cast<size_t>(peer) >= comps_.size()) return nullptr;
  auto const& pc = comps_[static_cast<size_t>(peer)];
  if (!pc.remote_device) return nullptr;
  return static_cast<char*>(pc.remote_device) + sizeof(IpcDataCompletion);
}

size_t IpcAdapter::drain_signal_tags(int peer_rank, uint64_t* tags,
                                     size_t max) {
  if (!has_wait_path(peer_rank)) {
    static int once = 0;
    if (!once++)
      UK_DBG(UK_DBG_LVL_TPT, "[drain-sig-tags r%d] no wait path for p%d",
             comm_->rank(), peer_rank);
    return 0;
  }
  auto& pc = comps_[static_cast<size_t>(peer_rank)];
  PeerSignalRing* ring = pc.signal_ring;
  if (!ring) {
    static int once2 = 0;
    if (!once2++)
      UK_DBG(UK_DBG_LVL_TPT, "[drain-sig-tags r%d] no signal_ring for p%d",
             comm_->rank(), peer_rank);
    return 0;
  }

  uint64_t r = ring->read_idx.load(std::memory_order_relaxed);
  size_t count = 0;

  while (count < max) {
    uint64_t w = ring->write_idx.load(std::memory_order_acquire);
    if (r >= w) break;

    size_t idx = r & (kSignalRingSize - 1);
    if (!ring->slots[idx].ready.load(std::memory_order_acquire)) break;

    tags[count] = ring->slots[idx].tag;
    ring->slots[idx].ready.store(false, std::memory_order_release);
    r++;
    count++;
  }

  if (count > 0) {
    UK_DBG(UK_DBG_LVL_TPT,
           "[drain-sig-tags r%d] got %zu tags from p%d, first=%lu",
           comm_->rank(), count, peer_rank, (unsigned long)tags[0]);
    ring->read_idx.store(r, std::memory_order_release);
  }
  return count;
}

void IpcAdapter::send_worker() {
  GPU_RT_CHECK(gpuSetDevice(gpu_id_));
  UK_DBG(UK_DBG_LVL_TPT, "[ipc-send r%d] worker alive, waiting for ops",
         comm_->rank());
  struct PendingPut {
    RingElem e;
    size_t evi;
    bool ok;
  };
  // Per-peer event pools (one event per in-flight put per peer). A
  // peer's events are reused via its own free list only after that
  // put completed, so a busy peer never steals a slot from an idle one.
  size_t const window = send_batch_;
  int const n = comm_->world_size();
  int const rank = comm_->rank();
  std::vector<gpuEvent_t> evs(window * static_cast<size_t>(n));
  for (size_t i = 0; i < evs.size(); ++i)
    GPU_RT_CHECK(gpuEventCreateWithFlags(&evs[i], gpuEventDisableTiming));
  std::vector<std::deque<size_t>> free_evs(static_cast<size_t>(n));
  for (int p = 0; p < n; ++p)
    for (size_t i = 0; i < window; ++i)
      free_evs[static_cast<size_t>(p)].push_back(
          static_cast<size_t>(p) * window + i);
  std::vector<std::deque<PendingPut>> pending(static_cast<size_t>(n));
  std::vector<size_t> inflight(static_cast<size_t>(n), 0);
  std::vector<size_t> launch_serial(static_cast<size_t>(n), 0);

  while (!stop_.load(std::memory_order_relaxed)) {
    // Per-peer loop: complete ready fronts (each peer strictly FIFO —
    // the receiver matches per-(peer,direction) sequences), then
    // launch-ahead up to that peer's window. Peers never block each
    // other, so 7-peer fan-out runs 7 x window copies concurrently.
    bool any = false;
    for (int p = 0; p < n; ++p) {
      if (p == rank) continue;
      size_t const pidx = static_cast<size_t>(p);
      auto& pque = pending[pidx];
      auto& pevs = free_evs[pidx];
      while (inflight[pidx] > 0) {
        auto& front = pque.front();
        gpuError_t st = gpuEventQuery(evs[front.evi]);
        if (st == gpuSuccess) {
          complete_one(&front.e, front.ok);
          pevs.push_back(front.evi);
          pque.pop_front();
          --inflight[pidx];
          any = true;
        } else if (st == gpuErrorNotReady) {
          break;  // front still in flight; stream order keeps the rest
        } else {
          complete_one(&front.e, false);
          pevs.push_back(front.evi);
          pque.pop_front();
          --inflight[pidx];
          any = true;
          std::fprintf(stderr,
                       "[ipc-send r%d] event query failed p%d st=%d\n",
                       comm_->rank(), p, static_cast<int>(st));
        }
      }
      while (inflight[pidx] < window) {
        RingElem e;
        if (jring_sc_dequeue_bulk(send_rings_[pidx], &e, 1, nullptr) == 1) {
          size_t stream_idx =
              pidx * streams_per_peer_ +
              (launch_serial[pidx]++ % streams_per_peer_);
          bool ok = launch_one(&e, stream_idx);
          size_t evi = pevs.front();
          pevs.pop_front();
          if (ok) GPU_RT_CHECK(gpuEventRecord(evs[evi], ipc_ctx_[stream_idx]));
          pque.push_back({e, evi, ok});
          ++inflight[pidx];
          any = true;
        } else {
          break;
        }
      }
    }
    bool inflight_any = false;
    for (size_t v : inflight)
      if (v > 0) {
        inflight_any = true;
        break;
      }
    if (inflight_any) {
      // Copies in flight: poll events eagerly with a pause burst instead
      // of yielding to the scheduler — completion latency is on the
      // critical path (8-rank alltoall tail).
      for (int s = 0; s < 16 && !stop_.load(std::memory_order_relaxed); ++s)
        machnet_pause();
      if (any) continue;
      std::this_thread::yield();
    } else {
      std::this_thread::yield();
    }
  }
  // Shutdown drain: flush pending puts and the ring (best effort).
  for (int p = 0; p < n; ++p) {
    if (p == rank) continue;
    size_t const pidx = static_cast<size_t>(p);
    for (auto const& pp : pending[pidx]) complete_one(&pp.e, false);
    pending[pidx].clear();
    RingElem drain;
    while (jring_mc_dequeue_bulk(send_rings_[pidx], &drain, 1, nullptr) == 1)
      publish_put_completion(drain.comm_rid, true);
  }
  for (auto& ev : evs) GPU_RT_CHECK(gpuEventDestroy(ev));
}

void IpcAdapter::recv_worker() {
  RingElem e;
  while (!stop_.load(std::memory_order_relaxed)) {
    if (jring_sc_dequeue_bulk(recv_ring_, &e, 1, nullptr) != 1) {
      std::this_thread::yield();
      continue;
    }
    bool ok = (e.type == ReqType::DataWait) ? recv_one(&e) : false;
    publish_put_completion(e.comm_rid, !ok);
  }
  RingElem drain;
  while (jring_mc_dequeue_bulk(recv_ring_, &drain, 1, nullptr) == 1)
    publish_put_completion(drain.comm_rid, true);
}

bool IpcAdapter::launch_one(RingElem* e, size_t stream_idx) {
  if (!e || (e->type != ReqType::DataPut && e->type != ReqType::PutSignal))
    return false;
  void* src = e->local_ptr;
  void* dst = e->remote_ptr;
  UK_DBG(UK_DBG_LVL_TPT, "[ipc-send_one r%d] dst=%p src=%p peer=%d",
         comm_->rank(), dst, src, e->peer);
  if (!dst) {
    std::cerr << "[ERROR] IPC send_put_async no remote_ptr\n";
    return false;
  }

  int remote_gpu = comm_->peer_gpu_idx(e->peer);
  if (remote_gpu < 0) remote_gpu = gpu_id_;
  UK_DBG(UK_DBG_LVL_TPT, "[ipc-send_one r%d] remote_gpu=%d gpu_id=%d bytes=%lu",
         comm_->rank(), remote_gpu, gpu_id_, (unsigned long)e->bytes);

  // One stream per put (round-robin in send_worker): a single 1MB+
  // copy already saturates the P2P link (~52 GB/s measured), and
  // assigning consecutive puts to different streams lets them overlap
  // while each put keeps a single, unambiguous event point. Intra-put
  // chunking across streams would need per-chunk event tracking for
  // completion and buys nothing at link saturation.
  gpuStream_t stream = ipc_ctx_[stream_idx];
  if (remote_gpu == gpu_id_)
    GPU_RT_CHECK(gpuMemcpyAsync(dst, src, e->bytes, gpuMemcpyDeviceToDevice,
                                stream));
  else
    GPU_RT_CHECK(gpuMemcpyPeerAsync(dst, remote_gpu, src, gpu_id_, e->bytes,
                                    stream));
  return true;
}

void IpcAdapter::complete_one(RingElem const* e, bool ok) {
  if (ok) {
    size_t dir = (comm_->rank() < e->peer) ? 0u : 1u;
    comps_[e->peer].remote->last_completed[dir].store(
        e->seq, std::memory_order_release);
  }
  // Publish the put completion BEFORE the fused-signal ring write: the
  // signal write can back-pressure on the receiver's drain cadence
  // (observed ~150-250us stalls at 8 ranks), and the sender's collective
  // must not wait on it — the receiver completes when IT drains the
  // signal. Reordering keeps the sender's critical path to just the copy
  // event; the signal still lands after the data (same thread, same
  // order).
  publish_put_completion(e->comm_rid, !ok);
  if (ok) {
    // Fused PutSignal: the peer observes the tag only after the data
    // has landed, matching a separate Signal op's semantics.
    if (e->type == ReqType::PutSignal) {
      if (uk_dbg_lvl() >= 1)
        std::fprintf(stderr, "[tss] r%d sig_send peer=%d tag=%lu t=%lld\n",
                     comm_->rank(), e->peer, (unsigned long)e->tag, tss_us());
      ok = write_signal_ring(e->peer, e->tag);
    }
  }
}

bool IpcAdapter::recv_one(RingElem* e) {
  if (!e || e->type != ReqType::DataWait) return false;

  size_t dir = (e->peer < comm_->rank()) ? 0u : 1u;
  uint64_t expected = e->seq;
  auto& pc = comps_[e->peer];
  auto* counter = &pc.local->last_completed[dir];

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(kIpcControlTimeoutMs);
  while (!stop_.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    if (counter->load(std::memory_order_acquire) >= expected) return true;
    std::this_thread::yield();
  }
  if (!stop_.load(std::memory_order_acquire)) {
    std::cerr << "[ERROR] IPC recv timed out, peer " << e->peer << " match_seq "
              << e->seq << std::endl;
  }
  return false;
}

}  // namespace Transport
}  // namespace UKernel
