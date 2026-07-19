#pragma once

#include "backend/backend.h"
#include "coll_config.h"
#include "lower.h"
#include "util/jring.h"
#include "util/jrqueue.h"
#include "util/uk_debug.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <set>
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

// Immutable, pointer-independent execution plan for one collective
// shape (kind/bytes/tile/dtype/splits/inplace). Built once and shared
// by all runs with the same shape; buf IDs are resolved per run, so
// plans are reusable across different input/output pointers.
struct CollPlan {
  TiledResult tiled;
  size_t nops = 0;
  std::vector<uint32_t> successor_off;   // CSR offsets, size nops+1
  std::vector<uint32_t> successor_data;  // CSR successors
  std::vector<uint32_t> indegree_init;   // deps count per op
  std::vector<uint32_t> initial_ready;   // ops with no deps
};

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

  // Lock-free drain path (drain threads, no mtx).
  // Successor graph lives in the shared plan; only the mutable indegree
  // is per-run.
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
    if (jring_dequeue_bulk(ready_ring, &op, 1, nullptr) == 1) return op;
    return ~0u;
  }

  void init_ready_ring(size_t nops) {
    uint32_t count = 1;
    while (count <= nops) count <<= 1;
    if (count < 1024) count = 1024;  // floor: avoid overflow under burst
    size_t sz = jring_get_buf_ring_size(sizeof(uint32_t), count);
    ready_ring = static_cast<jring_t*>(calloc(1, sz));
    if (!ready_ring) {
      std::fprintf(stderr, "[SprayRun] calloc ready_ring failed sz=%zu\n", sz);
      std::abort();
    }
    jring_init(ready_ring, count, sizeof(uint32_t), 0, 1);
  }

  ~SprayRun() {
    if (ready_ring) { free(ready_ring); ready_ring = nullptr; }
  }

  // Shared immutable plan (ops + successor graph); read-only here.
  std::shared_ptr<CollPlan const> plan;

  // Buffer ID deduplication: same ptr+size = same ID
  uint32_t input_buf_id = 0;
  uint32_t output_buf_id = 0;
  uint32_t scratch_buf_id = 0;

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

// Backend path counters for load-balancing diagnostics.
struct PathCounters {
  size_t device = 0;
  size_t ipc = 0;
  size_t rdma = 0;
};

// Per-backend slot: be_idx → (run, op_idx) in one lookup.
// tag doubles as ready flag and generation counter.
// Consumed by try_claim(): snapshot + CAS-release in one atomic step.
struct alignas(64) BeSlot {
  std::atomic<uint32_t> tag{~0u};  // be_idx when ready, ~0u = empty
  SprayRun* run = nullptr;
  uint32_t op_idx = 0;
  PutPath put_path = PutPath::Device;
  uint64_t enqueue_ns = 0;  // steady_clock timestamp, for latency EWMA
};

// Snapshot of a claimed slot — carries the data so the slot can be
// immediately reused by a subsequent write() without corrupting the
// drain pipeline.
struct BeSlotSnap {
  SprayRun* run = nullptr;
  uint32_t op_idx = 0;
  PutPath put_path = PutPath::Device;
  uint64_t enqueue_ns = 0;
};

static inline size_t round_up_pow2(size_t n) {
  if (n == 0) return 0;
  size_t c = 1;
  while (c < n) c <<= 1;
  return c;
}

// Lock-free slot table: be_idx → BeSlot, one-writer (enqueue) one-reader
// (drain) per slot.  Sized to backend capacity for index stability.
// Slots are claimed by the drain thread via try_claim(), which
// atomically snapshots the data and releases the slot in one CAS.
// release() is a no-op safe-guard; try_claim() is the sole consumer.
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
             PutPath put_path, std::atomic<bool> const& stop) {
    auto& s = slots_[be_idx & mask_];
    // be_idx grows monotonically while slots are reused modulo table
    // size. Never overwrite a slot whose previous occupant has not been
    // claimed yet — its try_claim would otherwise spin forever on a tag
    // that has already moved on. Single writer (the enqueue thread), so
    // this is plain backpressure, not a race.
    while (s.tag.load(std::memory_order_acquire) != ~0u) {
      if (stop.load(std::memory_order_relaxed)) return;
      std::this_thread::yield();
    }
    s.run = run;
    s.op_idx = op_idx;
    s.put_path = put_path;
    s.enqueue_ns = std::chrono::steady_clock::now().time_since_epoch().count();
    s.tag.store(be_idx, std::memory_order_release);
  }

  BeSlotSnap try_claim(uint32_t be_idx, std::atomic<bool> const& stop) {
    auto& s = slots_[be_idx & mask_];
    uint32_t tag;
    uint64_t spins = 0;
    while ((tag = s.tag.load(std::memory_order_acquire)) != be_idx) {
      // A synchronous-completion backend may surface a completion
      // before the enqueue thread publishes the slot. Every completion
      // implies a publisher that is a few instructions behind, so wait
      // for it: dropping a legitimate completion stalls the whole run.
      if (stop.load(std::memory_order_relaxed)) return {};
      if ((++spins & 0xFFFFFu) == 0)
        std::fprintf(stderr,
                     "[BeSlotTable] try_claim be_idx=%u: waited %lu spins\n",
                     be_idx, (unsigned long)spins);
      std::this_thread::yield();
    }
    // Snapshot data before releasing the slot so a subsequent write()
    // cannot corrupt what the drain pipeline is about to process.
    BeSlotSnap snap{s.run, s.op_idx, s.put_path, s.enqueue_ns};
    uint32_t expected = be_idx;
    if (!s.tag.compare_exchange_strong(expected, ~0u,
                                       std::memory_order_release,
                                       std::memory_order_relaxed))
      return {};  // claimed by another thread or release(), drop it
    return snap;
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

  // Start background threads. Must be called after owned_* members and
  // backend comm pointers have been set by the factory.
  void start();

  CollectiveOpHandle submit(CollectiveConfig const& cfg, void* input,
                            void* output);

  // Prepare peer connections and buffer resources for the given
  // collective.  Must be called once before the first submit().
  // Derives needed peers from the algorithm DAG so only relevant
  // IPC / RDMA links are established.
  void prepare(CollectiveConfig const& cfg, void* input, void* output);

  CollectiveOpStatus status(CollectiveOpHandle h) const;
  bool poll(CollectiveOpHandle h);
  bool wait(CollectiveOpHandle h,
            std::chrono::milliseconds to = std::chrono::milliseconds(0));
  void release(CollectiveOpHandle h);
  std::string error_message(CollectiveOpHandle h) const;

  size_t active_count() const;

  // Pre-register a buffer with the Communicator.  Subsequent submits
  // using the same pointer reuse the same buffer ID and MR.
  uint32_t get_or_register_buf(void* ptr, size_t bytes);

  // Snapshot path counters (atomically).
  PathCounters get_path_counters() const {
    PathCounters c;
    c.device = put_path_device_.load(std::memory_order_relaxed);
    c.ipc = put_path_ipc_.load(std::memory_order_relaxed);
    c.rdma = put_path_rdma_.load(std::memory_order_relaxed);
    return c;
  }
  void reset_path_counters() {
    put_path_device_.store(0, std::memory_order_relaxed);
    put_path_ipc_.store(0, std::memory_order_relaxed);
    put_path_rdma_.store(0, std::memory_order_relaxed);
  }
  bool path_counters_enabled() const { return path_counters_enabled_; }

 private:
  SprayRun* get(CollectiveOpHandle h);

  void enqueue_loop();
  void drain_dev_loop();
  void drain_tpt_loop();
  void drain_signal_loop();

  void collect_ready(SprayRun& run);
  void enqueue_to_ring(SprayRun& run);

  PutPath pick_put_path(int peer);
  // One round of non-blocking progress across all backends: drain
  // completions, claim slots, and release dependencies. Safe to call
  // concurrently from the background drain threads and from user
  // threads in wait(). Returns the number of completions processed.
  size_t progress_once();
  // Mark run Completed exactly once (CAS); the winner releases the
  // active-run slot. Safe against concurrent callers.
  void finalize_run(SprayRun* run);
  int rank_or_neg1() const;

  template <typename F>
  void drain_batch(BeSlotSnap* snaps, size_t n, F&& cb) {
    static int dbg_count = 0;
    if (n > 0 && dbg_count < 5) {
      ++dbg_count;
      UK_DBG(UK_DBG_LVL_EXEC, "[drain-batch r%d] %zu ops completed",
             rank_or_neg1(), n);
    }
    for (size_t i = 0; i < n; ++i) {
      auto& s = snaps[i];
      SprayRun* run = s.run;
      if (!run) continue;
      uint32_t op_idx = s.op_idx;
      uint32_t cur =
          __atomic_load_n(&run->indegree[op_idx], __ATOMIC_ACQUIRE);
      if (cur == SprayRun::kIndegreeDone) continue;
      run->done_count.fetch_add(1, std::memory_order_release);
      cb(s);

      uint32_t off = run->plan->successor_off[op_idx];
      uint32_t end = run->plan->successor_off[op_idx + 1];
      for (uint32_t j = off; j < end; ++j) {
        uint32_t succ = run->plan->successor_data[j];
        if (__atomic_fetch_sub(&run->indegree[succ], 1, __ATOMIC_RELEASE) ==
            1)
          run->push_ready(succ);
      }
      __atomic_store_n(&run->indegree[op_idx], SprayRun::kIndegreeDone,
                       __ATOMIC_RELEASE);
      // Inline completion: flip the run's status as soon as its last op
      // drains instead of waiting for a periodic check_completions_
      // sweep (CAS-guarded, exactly-once).
      finalize_run(run);
    }
  }

  // Tensor to buffer ID dedup: same ptr = same ID
  std::unordered_map<uintptr_t, uint32_t> tensor_to_buf_id_;
  uint32_t next_buf_id_ = 1;

  // Buffer registration indirection (set by factory, avoids link deps)
  void (*register_buf_fn_)(Transport::Communicator*, uint32_t, void*,
                           size_t) = nullptr;
  void (*peer_setup_fn_)(Transport::Communicator*, int,
                         std::vector<int> const&) = nullptr;
  void (*resolve_buf_fn_)(Transport::Communicator*, int, int, uint32_t) =
      nullptr;
  bool (*same_host_fn_)(Transport::Communicator*, int) = nullptr;

  BatchBackend* device_be_;
  BatchBackend* tpt_be_;
  BatchBackend* signal_be_ = nullptr;
  std::unique_ptr<BatchBackend> owned_device_;
  std::unique_ptr<BatchBackend> owned_transport_;
  std::unique_ptr<BatchBackend> owned_signal_;
  std::shared_ptr<Transport::Communicator> owned_comm_;

  // Lazy-allocated scratch buffer for inplace staging (managed internally).
  void* internal_scratch_ = nullptr;
  size_t internal_scratch_cap_ = 0;

  bool prepared_ = false;
  std::set<int> prepared_peers_;

  std::thread enqueue_th_;
  std::thread drain_th_dev_;
  std::thread drain_th_tpt_;
  std::thread drain_th_signal_;
  std::atomic<bool> stop_{false};

  BeSlotTable dev_slots_;
  BeSlotTable tpt_slots_;
  BeSlotTable sig_slots_;
  // Scratch for batched submission (single-threaded: the enqueue loop).
  std::vector<uint32_t> be_idx_scratch_;

  // Transport LB state
  int world_size_ = 0;
  std::unique_ptr<PeerMetrics[]> tpt_metrics_;

  // Put path counters (enqueue-time, for diagnostics)
  std::atomic<size_t> put_path_device_{0};
  std::atomic<size_t> put_path_ipc_{0};
  std::atomic<size_t> put_path_rdma_{0};
  bool path_counters_enabled_ = false;

  // Backpressure
  size_t max_concurrent_runs_ = 16;
  std::atomic<size_t> active_runs_{0};

  // shared_ptr so the enqueue loop can snapshot running runs and process
  // them without holding runs_mutex_ (lifetime stays safe even if the
  // run completes and is released concurrently).
  std::unordered_map<CollectiveOpHandle, std::shared_ptr<SprayRun>> runs_;
  mutable std::mutex runs_mutex_;
  uint64_t next_handle_ = 1;

  // Plan cache: collective shape -> immutable plan. Plans are built once
  // per shape; submit() on a hit only copies the mutable scheduling
  // state (indegree) into the new run.
  std::mutex plan_cache_mu_;
  std::unordered_map<std::string, std::shared_ptr<CollPlan const>>
      plan_cache_;
  static constexpr size_t kMaxCachedPlans = 64;
};

}  // namespace CCL
}  // namespace UKernel
