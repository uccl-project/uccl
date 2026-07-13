#pragma once

#include "backend/backend.h"
#include "coll_config.h"
#include "lower.h"
#include "util/jring.h"
#include "util/jrqueue.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {
enum class PeerTransportKind;
struct CommunicatorConfig;
struct SignalCompletion;
class Communicator;
}  // namespace Transport
namespace CCL {

enum class CollectiveOpStatus : uint32_t {
  Queued,
  Running,
  Completed,
  Failed,
};

using CollectiveOpHandle = uint64_t;
inline constexpr CollectiveOpHandle kInvalidHandle = 0;

struct SprayRun {
  static constexpr uint32_t kIndegreeDone =
      ~0u;  // sentinel: op already processed

  // Hot path: accessed every enqueue/drain cycle
  std::atomic<CollectiveOpStatus> status{CollectiveOpStatus::Queued};
  std::atomic<size_t> done_count{0};
  std::mutex mtx;
  std::vector<uint8_t> submitted;
  std::vector<uint32_t> ready;
  // Backpressure: ops rejected by backend are deferred and retried first
  // on the next enqueue cycle before pulling new ops from the ring.
  // This preserves priority (FIFO within each backend) and avoids
  // re-enqueue contention on the lock-free ring.
  std::vector<uint32_t> deferred_dev;
  std::vector<uint32_t> deferred_tpt;
  std::vector<uint32_t> deferred_sig;
  std::vector<Cmd> dev_cmds;
  std::vector<Cmd> tpt_cmds;

  // Lock-free drain path (drain threads, no mtx)
  std::vector<uint32_t> successor_data;  // contiguous successor list
  std::vector<uint32_t>
      successor_off;               // offset into data per op (size = nops+1)
  std::vector<uint32_t> indegree;  // __atomic_fetch_sub decrement, 0 = ready

  // Lock-free ready ring via jring (MP/SC, sized to nops at submit time)
  jring_t* ready_ring = nullptr;

  bool push_ready(uint32_t op) {
    if (!ready_ring) return false;
    jrpush(ready_ring, op);
    return true;
  }
  uint32_t pop_ready() {
    if (!ready_ring) return ~0u;
    uint32_t op = ~0u;
    if (jring_sc_dequeue_bulk(ready_ring, &op, 1, nullptr) == 1) return op;
    return ~0u;
  }

  void init_ready_ring(size_t nops) {
    uint32_t count = 1;
    while (count <= nops) count <<= 1;
    if (count < 1024) count = 1024;  // floor: avoid overflow under burst
    size_t sz = jring_get_buf_ring_size(sizeof(uint32_t), count);
    ready_ring = static_cast<jring_t*>(calloc(1, sz));
    jring_init(ready_ring, count, sizeof(uint32_t), 1, 0);  // MP/SC
  }

  ~SprayRun() {
    if (ready_ring) { free(ready_ring); ready_ring = nullptr; }
  }

  // Read-only after construction
  TiledResult tiled;

  // Buffer ID deduplication: same ptr+size = same ID
  uint32_t input_buf_id = 0;
  uint32_t output_buf_id = 0;
  uint32_t scratch_buf_id = 0;

  // GPU buffer pointers for GDR read-tail flush after RDMA writes
  void* output_buf_ptr = nullptr;
  void* input_buf_ptr = nullptr;
  void* scratch_buf_ptr = nullptr;

  // Per-op flush ranges indexed by op_idx (~0u = no flush).
  // After a WaitSignal for an RDMA Put, the output buffer at
  // [flush_off[i], flush_off[i]+flush_bytes[i]) must be flushed.
  std::vector<uint32_t> flush_off;
  std::vector<uint32_t> flush_bytes;

  // (backend_tag, be_idx) pairs for release cleanup
  // backend_tag: 0=dev, 1=tpt, 2=sig
  std::vector<std::pair<uint8_t, uint32_t>> be_slots;

  // Cold: rarely accessed
  std::string error;
};

struct SprayExecutorConfig {
  int gpu_id;
  int rank;
  int world_size;
  size_t device_task_capacity = 4096;
  size_t max_device_fifos = 2;
  int threads_per_block = 64;
  int blocks_per_worker = 1;
  size_t fifo_capacity = 256;
  size_t smem_size = 4096;  // dynamic shared memory for reduce kernel
  size_t max_concurrent_runs = 16;

  // Communicator settings
  std::string exchanger_ip = "0.0.0.0";
  int exchanger_port = 6979;
  int local_id = -1;
};

// Per-backend slot: be_idx → (run, op_idx) in one lookup.
// tag doubles as ready flag and generation counter.
struct alignas(64) BeSlot {
  std::atomic<uint32_t> tag{~0u};  // be_idx when ready, ~0u = empty
  SprayRun* run = nullptr;
  uint32_t op_idx = 0;
  PutPath put_path = PutPath::Device;
  uint64_t enqueue_ns = 0;  // steady_clock timestamp, for latency EWMA
};

static inline size_t round_up_pow2(size_t n) {
  if (n == 0) return 0;
  size_t c = 1;
  while (c < n) c <<= 1;
  return c;
}

// Lock-free slot table: be_idx → BeSlot, one-writer (enqueue) one-reader
// (drain) per slot.  Sized to backend capacity for index stability.
class BeSlotTable {
 public:
  explicit BeSlotTable(size_t capacity) {
    size_t cap = round_up_pow2(capacity);
    if (cap == 0) cap = 256;
    slots_.reset(new BeSlot[cap]);
    mask_ = cap - 1;
  }

  BeSlotTable() = default;

  void write(uint32_t be_idx, SprayRun* run, uint32_t op_idx,
             PutPath put_path) {
    auto& s = slots_[be_idx & mask_];
    s.run = run;
    s.op_idx = op_idx;
    s.put_path = put_path;
    s.enqueue_ns = std::chrono::steady_clock::now().time_since_epoch().count();
    s.tag.store(be_idx, std::memory_order_release);
  }

  BeSlot* wait(uint32_t be_idx, std::atomic<bool> const& stop) {
    auto& s = slots_[be_idx & mask_];
    uint32_t tag;
    while ((tag = s.tag.load(std::memory_order_acquire)) != be_idx) {
      if (stop.load(std::memory_order_relaxed)) return nullptr;
      std::this_thread::yield();
    }
    return &s;
  }

  void release(uint32_t be_idx) {
    auto& s = slots_[be_idx & mask_];
    uint32_t t = be_idx;
    s.tag.compare_exchange_strong(t, ~0u, std::memory_order_release,
                                  std::memory_order_relaxed);
  }

 private:
  std::unique_ptr<BeSlot[]> slots_;
  size_t mask_ = 0;
};

// Per-peer transport metrics for dynamic load balancing
struct PathMetrics {
  std::atomic<uint32_t> inflight{0};
  std::atomic<uint64_t> latency_ns{100000};  // 100 us default
};

struct PeerMetrics {
  PathMetrics device;
  PathMetrics ipc;
  PathMetrics rdma;
};

class SprayExecutor {
 public:
  static std::unique_ptr<SprayExecutor> create(
      SprayExecutorConfig const& config);
  SprayExecutor(BatchBackend* device_be, BatchBackend* tpt_be,
                BatchBackend* signal_be = nullptr, int world_size = 0);
  ~SprayExecutor();

  SprayExecutor(SprayExecutor const&) = delete;
  SprayExecutor& operator=(SprayExecutor const&) = delete;

  CollectiveOpHandle submit(CollectiveConfig const& cfg, void* input,
                            void* output, void* scratch);

  CollectiveOpStatus status(CollectiveOpHandle h) const;
  bool poll(CollectiveOpHandle h);
  bool wait(CollectiveOpHandle h,
            std::chrono::milliseconds to = std::chrono::milliseconds(0));
  void release(CollectiveOpHandle h);
  std::string error_message(CollectiveOpHandle h) const;

  size_t active_count() const;

 private:
  SprayRun* get(CollectiveOpHandle h);

  void enqueue_loop();
  void drain_dev_loop();
  void drain_tpt_loop();
  void drain_signal_loop();

  void collect_ready(SprayRun& run);
  void enqueue_to_ring(SprayRun& run);

  PutPath pick_put_path(int peer);
  void check_completions_();

  template <typename F>
  void drain_batch(BeSlot** slots, size_t n, F&& cb) {
    for (size_t i = 0; i < n; ++i) {
      auto& s = *slots[i];
      SprayRun* run = s.run;
      if (!run) continue;
      uint32_t op_idx = s.op_idx;
      uint32_t cur =
          __atomic_load_n(&run->indegree[op_idx], __ATOMIC_ACQUIRE);
      if (cur == SprayRun::kIndegreeDone) continue;
      run->done_count.fetch_add(1, std::memory_order_release);
      cb(s);

      uint32_t off = run->successor_off[op_idx];
      uint32_t end = run->successor_off[op_idx + 1];
      for (uint32_t j = off; j < end; ++j) {
        uint32_t succ = run->successor_data[j];
        if (__atomic_fetch_sub(&run->indegree[succ], 1, __ATOMIC_RELEASE) ==
            1)
          run->push_ready(succ);
      }
      __atomic_store_n(&run->indegree[op_idx], SprayRun::kIndegreeDone,
                       __ATOMIC_RELEASE);
    }
  }

  // Tensor to buffer ID dedup: same ptr = same ID
  std::unordered_map<uintptr_t, uint32_t> tensor_to_buf_id_;
  uint32_t next_buf_id_ = 1;
  uint32_t get_or_register_buf(void* ptr, size_t bytes);

  // Buffer registration indirection (set by factory, avoids link deps)
  void (*register_buf_fn_)(Transport::Communicator*, uint32_t, void*,
                           size_t) = nullptr;
  void (*peer_setup_fn_)(Transport::Communicator*, int, int) = nullptr;
  void (*resolve_buf_fn_)(Transport::Communicator*, int, int, uint32_t) =
      nullptr;
  bool (*same_host_fn_)(Transport::Communicator*, int) = nullptr;

  // Pin GPU buffer for BAR1 access (called once per buffer at registration).
  // The factory maps the buffer via GDRCopy; flush_rdma_fn_ reads from it.
  void (*pin_buf_fn_)(void* gpu_ptr, size_t bytes) = nullptr;

  // Flush GPU L2 cache for addresses written by RDMA.
  // Called from drain_signal_loop after WaitSignal completion.
  // gpu_buf_ptr must have been previously pinned via pin_buf_fn_.
  void (*flush_rdma_fn_)(void* gpu_buf_ptr, size_t offset,
                          size_t bytes) = nullptr;

  BatchBackend* device_be_;
  BatchBackend* tpt_be_;
  BatchBackend* signal_be_ = nullptr;
  std::unique_ptr<BatchBackend> owned_device_;
  std::unique_ptr<BatchBackend> owned_transport_;
  std::unique_ptr<BatchBackend> owned_signal_;
  std::shared_ptr<Transport::Communicator> owned_comm_;

  std::thread enqueue_th_;
  std::thread drain_th_dev_;
  std::thread drain_th_tpt_;
  std::thread drain_th_signal_;
  std::atomic<bool> stop_{false};

  BeSlotTable dev_slots_;
  BeSlotTable tpt_slots_;
  BeSlotTable sig_slots_;

  // Transport LB state
  int world_size_ = 0;
  std::unique_ptr<PeerMetrics[]> tpt_metrics_;

  // Backpressure
  size_t max_concurrent_runs_ = 16;
  std::atomic<size_t> active_runs_{0};

  std::unordered_map<CollectiveOpHandle, std::unique_ptr<SprayRun>> runs_;
  mutable std::mutex runs_mutex_;
  uint64_t next_handle_ = 1;
};

}  // namespace CCL
}  // namespace UKernel
