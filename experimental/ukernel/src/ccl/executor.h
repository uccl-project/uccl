#pragma once

#include "backend/backend.h"
#include "backend/rdma_fused_proxy.h"
#include "coll_config.h"
#include "lower.h"
#include "util/jring.h"
#include "util/jrqueue.h"
#include "util/uk_debug.h"
#include "../../include/gpu_rt.h"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <map>
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
  // PutSignal fusion metadata (from the lowerer), indexed by op idx.
  // put_to_sig[put] = the signal op whose tag this put may carry (-1 =
  // none); several puts of one group map to the same signal. A fused
  // group is fully carried when sig_group_size[sig] of its puts were
  // accepted with the fusion flag; the Signal op then completes locally.
  // wait_group_size[ws] = group tiles (0 = not fusion-eligible): the
  // wait counts that many tag arrivals when the sender fuses the group.
  std::vector<int32_t> put_to_sig;
  std::vector<uint16_t> sig_group_size;
  std::vector<uint16_t> wait_group_size;
};

struct SprayRun {
  static constexpr uint32_t kIndegreeDone =
      ~0u;  // sentinel: op already processed

  // Hot path: accessed every enqueue/drain cycle
  std::atomic<CollectiveOpStatus> status{CollectiveOpStatus::Queued};
  std::atomic<size_t> done_count{0};
  // Ops accepted by a backend whose completion has not drained yet
  // (+1 on acceptance in enqueue_to_ring, -1 in drain_batch). A Failed
  // run is releasable only once this hits 0: late completions reach the
  // run through BeSlot raw pointers, so freeing it earlier is a
  // use-after-free (release() gates on this).
  std::atomic<size_t> inflight_ops{0};
  // In-flight WaitSignal ops (subset of inflight_ops): throttled per run
  // by UK_CCL_SIG_INFLIGHT_CAP in enqueue_to_ring, decremented in
  // drain_batch alongside inflight_ops.
  std::atomic<uint32_t> sig_inflight{0};
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

  // Per signal op: how many of its group's puts were accepted with the
  // fusion flag (see CollPlan::sig_group_size). The Signal op completes
  // locally once the count reaches the group size.
  std::vector<uint16_t> fused_sig_cnt;

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
    size_t count = 1;
    // Cap to avoid uint32_t overflow; jring APIs only accept uint32_t count.
    if (nops > RING_SZ_MASK) nops = RING_SZ_MASK;
    while (count <= nops) count <<= 1;
    if (count < 1024) count = 1024;  // floor: avoid overflow under burst
    // jring slots must be a power of two ≤ RING_SZ_MASK; clamp so the
    // ready ring stays valid even for degenerate (huge) op counts.
    if (count > ((RING_SZ_MASK >> 1) + 1)) count = (RING_SZ_MASK >> 1) + 1;
    size_t sz = jring_get_buf_ring_size(sizeof(uint32_t),
                                        static_cast<uint32_t>(count));
    // jring_get_buf_ring_size returns (size_t)-1 on invalid input; the
    // clamp above keeps count valid, so this is a real size (guard
    // doubles as a cheap invariant check).
    if (sz == static_cast<size_t>(-1)) {
      std::fprintf(stderr, "[SprayRun] invalid ready-ring size\n");
      std::abort();
    }
    ready_ring = static_cast<jring_t*>(calloc(1, sz));
    if (!ready_ring) {
      std::fprintf(stderr, "[SprayRun] calloc ready_ring failed sz=%zu\n", sz);
      std::abort();
    }
    jring_init(ready_ring, static_cast<uint32_t>(count), sizeof(uint32_t), 0, 1);
  }

  ~SprayRun() {
    if (ready_ring) {
      free(ready_ring);
      ready_ring = nullptr;
    }
  }

  // Shared immutable plan (ops + successor graph); read-only here.
  std::shared_ptr<CollPlan const> plan;

  // Buffer ID deduplication: same ptr+size = same ID
  uint32_t input_buf_id = 0;
  uint32_t output_buf_id = 0;
  uint32_t scratch_buf_id = 0;
  // Offset of each role's tensor within its registered allocation.
  uint64_t input_base_off = 0;
  uint64_t output_base_off = 0;
  uint64_t scratch_base_off = 0;

  // (backend_tag, be_idx) pairs for release cleanup
  // backend_tag: 0=dev, 1=tpt, 2=sig
  std::vector<std::pair<uint8_t, uint32_t>> be_slots;

  // Cold: rarely accessed
  std::string error;

  // Stream-ordered dependency management (NCCL-compatible).
  gpuStream_t user_stream = nullptr;  // submit() caller's stream
  gpuEvent_t input_ready = nullptr;   // record on user_stream, gate enqueue
  uint64_t done_seq = 0;              // monotonic completion sequence number

  // Watchdog state (enqueue_loop only): fail the run when done_count
  // stops advancing — turns silent deadlocks into loud errors.
  size_t watchdog_done = 0;
  std::chrono::steady_clock::time_point watchdog_ts{};

  // Signal-tag epoch for this run (assigned in submit from
  // next_run_epoch_); folded into every Signal/WaitSignal/fused tag.
  uint32_t tag_epoch = 0;

  // Debug counters: Signal ops completed locally (fused) vs dispatched
  // standalone to the signal backend — a run whose puts fused must have
  // sig_standalone == 0, or the peer sees duplicate arrivals.
  uint32_t sig_local = 0;
  uint32_t sig_standalone = 0;

  // Per-signal-group acceptance accounting (indexed by Signal op idx,
  // like fused_sig_cnt): fused_sig_cnt counts group puts accepted WITH
  // the fuse flag, accepted_sig_cnt counts ALL accepted group puts.
  // The Signal op must not be evaluated until accepted == grp — a
  // point-in-time check against fused only would dispatch a standalone
  // signal for a put that fuses one cycle later (duplicate arrival).
  std::vector<uint16_t> accepted_sig_cnt;
};

struct SprayExecutorConfig {
  int gpu_id;
  int rank;
  int world_size;
  size_t device_task_capacity = 4096;
  size_t max_device_fifos = 2;
  // 256 threads: the ILP-vectorized reduce needs this many to keep the
  // memory system fed (64 was latency-bound); 512 currently exceeds the
  // per-block register limit for the ILP reduce, so 256 is the sweet
  // spot until launch bounds land.
  int threads_per_block = 256;
  // <0 = auto: pick a per-GPU default at init from the device's compute
  // capability (A40-class 8, Hopper 16, Blackwell 32 — the few-SM-
  // friendly picks; see executor_factory). Set a positive value to force
  // (or override with UK_CCL_DEV_BLOCKS). The measured A40 baseline that
  // motivated the default 8: ReduceScatter 256MB 8.3ms -> 3.1ms (beats
  // native NCCL's 3.9ms), AllGather 256MB 3.8ms -> 2.8ms, small messages
  // improve too (256KB AllReduce 84us -> 77us); 16 blocks was marginal.
  int blocks_per_worker = -1;
  size_t fifo_capacity = 256;
  // Dynamic shared memory for the reduce kernel; follows the build's
  // REDUCE_SMEM_KB (default 4KB) so the launch config always matches the
  // kernel's TMA chunk sizing.
#ifndef UK_REDUCE_SMEM_KB
#define UK_REDUCE_SMEM_KB 4
#endif
  size_t smem_size = static_cast<size_t>(UK_REDUCE_SMEM_KB) * 1024;
  size_t max_concurrent_runs = 16;

  // Communicator settings
  std::string exchanger_ip = "0.0.0.0";
  int exchanger_port = 16998;  // matches all ccl test defaults
  int local_id = -1;
  // Grace period (µs) of continuous fifo emptiness after which device
  // worker kernels exit (relaunched on next enqueue). 0 = always
  // resident. Enable when the process also runs torch/CUDA work — an
  // always-spinning kernel deadlocks device-wide syncs.
  uint32_t device_idle_exit_us = 500;  // see WorkerPool::Config::idleExitAfterUs
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

  size_t capacity() const { return mask_ + 1; }

  void write(uint32_t be_idx, SprayRun* run, uint32_t op_idx, PutPath put_path,
             std::atomic<bool> const& stop) {
    auto& s = slots_[be_idx & mask_];
    // be_idx grows monotonically while slots are reused modulo table
    // size. Never overwrite a slot whose previous occupant has not been
    // claimed yet — its try_claim would otherwise spin forever on a tag
    // that has already moved on. Single writer (the enqueue thread), so
    // this is plain backpressure, not a race.
    uint64_t spins = 0;
    while (s.tag.load(std::memory_order_acquire) != ~0u) {
      if (stop.load(std::memory_order_relaxed)) return;
      if ((++spins & 0xFFFFFu) == 0) {
        // Zombie-slot forensics: identify the occupant whose completion
        // vanished (op kind via run->plan, backend tag, enqueue time).
        auto const& op = s.run->plan->tiled.ops[s.op_idx];
        std::fprintf(stderr,
                     "[BeSlotTable] write be_idx=%u blocked %lu spins: "
                     "occupant tag=%u op_idx=%u kind=%d bytes=%zu "
                     "src_peer=%d dst_peer=%d\n",
                     be_idx, (unsigned long)spins, s.tag.load(), s.op_idx,
                     (int)op.kind, op.bytes, (int)op.src_peer,
                     (int)op.dst_peer);
      }
      std::this_thread::yield();
    }
    s.run = run;
    s.op_idx = op_idx;
    s.put_path = put_path;
    s.enqueue_ns = std::chrono::steady_clock::now().time_since_epoch().count();
    s.tag.store(be_idx, std::memory_order_release);
  }

  // Non-blocking occupancy query for the slot write() would target.
  // Lets the enqueue thread defer an op instead of spinning in write()
  // when the table has wrapped — blocking there can deadlock a run
  // whose later ops (e.g. batched puts) produce the completions the
  // earlier ones wait for.
  bool occupied(uint32_t be_idx) const {
    return slots_[be_idx & mask_].tag.load(std::memory_order_acquire) != ~0u;
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
    if (!s.tag.compare_exchange_strong(expected, ~0u, std::memory_order_release,
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
                            void* output, gpuStream_t stream = nullptr);

  // Prepare peer connections and buffer resources for the given
  // collective. Idempotent and thread-safe: internally deduped on
  // (kind, peers, allocations, bytes), so callers should invoke it
  // before every submit() and let the executor skip repeats.
  // Derives needed peers from the algorithm DAG so only relevant
  // IPC / RDMA links are established. No-op for single-rank configs.
  void prepare(CollectiveConfig const& cfg, void* input, void* output);
  uintptr_t cached_alloc_base(void const* p);

  // Submit a fused RDMA command previously written by a device kernel into
  // the D2H ring. Returns true when the command was accepted by the
  // TransportBackend and a BeSlot was published for normal completion.
  // first_attempt == true only on the initial ring pop: the acceptance
  // accounting is mirrored exactly once; retries skip it.
  bool submit_fused_cmd(uint64_t cmd_index, bool first_attempt);

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
  uint32_t get_or_register_buf(void* ptr, size_t bytes, uint64_t* out_off,
                               char const* role);

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
  // Release one tentative inflight charge (op deferred before
  // acceptance; see pick_put_path). The drain side releases accepted
  // ops in update_path_metrics.
  void release_put_inflight(int peer, PutPath path);
  // One round of non-blocking progress across all backends: drain
  // completions, claim slots, and release dependencies. Safe to call
  // concurrently from the background drain threads and from user
  // threads in wait(). Returns the number of completions processed.
  size_t progress_once();
  // Mark run Completed exactly once (CAS); the winner releases the
  // active-run slot. Safe against concurrent callers.
  void finalize_run(SprayRun* run);
  // Mark run Failed exactly once (CAS) with an error message; the winner
  // releases the active-run slot. Used by the enqueue_loop watchdog.
  // The run stays allocated until its in-flight ops drain (late
  // completions still reference it through BeSlot raw pointers);
  // release() gates on SprayRun::inflight_ops.
  void fail_run(SprayRun* run, std::string msg);
  // Dump per-kind submission state of a run (enqueue_loop context).
  void dump_run_state(SprayRun* run, char const* why);
  int rank_or_neg1() const;
  // Invalidate prepare-cache entries referencing an allocation base
  // (stale-registration eviction path, get_or_register_buf).
  void invalidate_prepared_by_base(uintptr_t base);
  // Allocate or grow the internal scratch buffer (api_mu_ held).
  void ensure_internal_scratch(size_t bytes);

  // Mark an op completed without a backend completion — used when a
  // Signal op's tag already rode its partner fused PutSignal. Mirrors
  // the drain_batch per-op path: bump done_count, release successor
  // dependencies, and flip the run if this was the last op.
  void complete_op_local(SprayRun& run, uint32_t op_idx) {
    uint32_t cur = __atomic_load_n(&run.indegree[op_idx], __ATOMIC_ACQUIRE);
    if (cur == SprayRun::kIndegreeDone) return;
    run.done_count.fetch_add(1, std::memory_order_release);
    uint32_t off = run.plan->successor_off[op_idx];
    uint32_t end = run.plan->successor_off[op_idx + 1];
    for (uint32_t j = off; j < end; ++j) {
      uint32_t succ = run.plan->successor_data[j];
      if (__atomic_fetch_sub(&run.indegree[succ], 1, __ATOMIC_RELEASE) == 1)
        run.push_ready(succ);
    }
    __atomic_store_n(&run.indegree[op_idx], SprayRun::kIndegreeDone,
                     __ATOMIC_RELEASE);
    finalize_run(&run);
  }

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
      // One drained completion retires one in-flight op (see
      // SprayRun::inflight_ops), regardless of run state.
      run->inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
      uint32_t op_idx = s.op_idx;
      if (run->plan->tiled.ops[op_idx].kind == LogicalOpKind::Wait)
        run->sig_inflight.fetch_sub(1, std::memory_order_acq_rel);
      if (run->status.load(std::memory_order_acquire) !=
          CollectiveOpStatus::Running) {
        // Late completion for a Failed run (watchdog fired with ops
        // still out) or a Completed one (fused-signal local completion
        // raced the last drain): release the backend path charge, but
        // skip the scheduling bookkeeping — a Failed run's remaining
        // ops are never submitted, a Completed run has none left.
        cb(s);
        continue;
      }
      uint32_t cur = __atomic_load_n(&run->indegree[op_idx], __ATOMIC_ACQUIRE);
      if (cur == SprayRun::kIndegreeDone) continue;
      run->done_count.fetch_add(1, std::memory_order_release);
      cb(s);

      uint32_t off = run->plan->successor_off[op_idx];
      uint32_t end = run->plan->successor_off[op_idx + 1];
      for (uint32_t j = off; j < end; ++j) {
        uint32_t succ = run->plan->successor_data[j];
        if (__atomic_fetch_sub(&run->indegree[succ], 1, __ATOMIC_RELEASE) == 1)
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

  // Tensor registration table, keyed by ALLOCATION BASE (see
  // get_or_register_buf): one entry per CUDA allocation, no matter how
  // many tensors/offsets inside it are used. Registrations are
  // immutable; buf ids are minted in first-seen order, which is
  // rank-symmetric because all ranks drive the same collective sequence.
  struct BufReg {
    uint32_t id;
    void* alloc_base;
    size_t alloc_size;
  };
  std::unordered_map<uintptr_t, BufReg> tensor_to_buf_id_;
  // Canonical id assignment: 0 is the "no buffer" sentinel. Both user
  // allocations and scratch buffers mint from next_buf_id_ in first-seen
  // order (rank-symmetric because all ranks prepare the same shape
  // sequence). Scratch buffers use one id PER DISTINCT SCRATCH SIZE;
  // minting from the shared counter (instead of a fixed low pool) means
  // the id space is unbounded — a long-running app sweeping many
  // collective sizes (e.g. nccl-tests 8B→128M) no longer exhausts a
  // fixed pool. Rank symmetry of scratch ids still holds because every
  // rank drives the same (kind, size) prepare sequence.
  uint32_t next_buf_id_ = 1;

  // Buffer registration indirection (set by factory, avoids link deps)
  void (*register_buf_fn_)(Transport::Communicator*, uint32_t, void*,
                           size_t) = nullptr;
  void (*deregister_buf_fn_)(Transport::Communicator*, uint32_t) = nullptr;
  void (*peer_setup_fn_)(Transport::Communicator*, int,
                         std::vector<int> const&) = nullptr;
  void (*resolve_buf_fn_)(Transport::Communicator*, int, int,
                          uint32_t) = nullptr;
  bool (*same_host_fn_)(Transport::Communicator*, int) = nullptr;

  BatchBackend* device_be_;
  BatchBackend* tpt_be_;
  BatchBackend* signal_be_ = nullptr;
  std::unique_ptr<BatchBackend> owned_device_;
  std::unique_ptr<BatchBackend> owned_transport_;
  std::unique_ptr<BatchBackend> owned_signal_;
  std::shared_ptr<Transport::Communicator> owned_comm_;
  // Alternate producer for fused RDMA puts: GPU kernel writes cmd_index
  // into the D2H ring, progress() feeds the same TransportBackend.
  std::shared_ptr<RdmaFusedProxy> fused_proxy_;

  // Scratch buffers for Tmp regions / lowering staging, ONE PER
  // DISTINCT SIZE (key = staging bytes). Buffers stay allocated and
  // registered for the executor's lifetime: no mid-session dereg (its
  // RDMA QP flush can fail in-flight WRs) and no same-id
  // re-registration (its generation re-resolve races peer polls) — a
  // fresh size gets a fresh pool id, so peer re-resolution stays a
  // plain first-publish wait, and in-flight puts into older sizes keep
  // working. Trade-off: holds ~2x the largest size at peak.
  struct ScratchBuf {
    void* ptr;          // aligned base used for registration/addressing
    void* alloc_raw;    // original cudaMalloc result (must be freed with it)
    uint32_t id;
  };
  std::map<size_t, ScratchBuf> scratch_by_size_;

  // Serializes prepare()/submit() against each other: both mutate the
  // shared registration table (tensor_to_buf_id_), the internal scratch
  // allocation, and the prepared_* state. Lock order: api_mu_ ->
  // plan_cache_mu_ -> runs_mutex_; the enqueue/drain threads never take
  // api_mu_, so this cannot deadlock the progress engine.
  std::mutex api_mu_;

  bool prepared_ = false;
  std::set<int> prepared_peers_;
  // prepare() dedup cache (see prepare_key in executor.cc): one entry per
  // (kind, peers, input/output allocation, bytes) combination already
  // prepared. Cleared wholesale when full — entries are cheap to rebuild.
  std::set<std::string> prepared_keys_;
  // Reverse index: input/output allocation base → prepare keys
  // referencing it, so a stale-registration eviction
  // (get_or_register_buf) can invalidate exactly the affected entries.
  std::unordered_multimap<uintptr_t, std::string> prepared_key_bases_;
  // Fast-path caches for prepare(): the algorithm DAG is shape-
  // determined (no pointer dependence) and alloc-base lookups repeat
  // across nccl-tests iterations (buffers shift inside one allocation).
  // Both avoid per-collective DAG rebuilds and driver calls; measured
  // ~5-8us of the ~15us per-call prepare+submit overhead.
  std::unordered_map<std::string, CollAlgo> prepare_algo_cache_;
  std::unordered_map<uintptr_t, uintptr_t> alloc_base_cache_;
  static constexpr size_t kMaxPreparedKeys = 64;

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
  // Enqueue-loop wakeup: submit() notifies when a run is published, so
  // the loop does not sit through a scheduler yield between back-to-back
  // collectives (measured ~97us of the ~190us alltoall dispatch at 8
  // ranks came from the yield->wake latency).
  std::mutex wake_mu_;
  std::condition_variable wake_cv_;
  bool wake_pending_ = false;
  uint64_t next_handle_ = 1;
  // Monotonic run counter for signal-tag epoch salting. Assigned to
  // run->tag_epoch in submit() (under runs_mutex_). Ranks derive
  // identical epochs because collective issue order is rank-symmetric
  // (the same assumption tag matching already relies on).
  uint64_t next_run_epoch_ = 0;

  // Plan cache: collective shape -> immutable plan. Plans are built once
  // per shape; submit() on a hit only copies the mutable scheduling
  // state (indegree) into the new run.
  std::mutex plan_cache_mu_;
  std::unordered_map<std::string, std::shared_ptr<CollPlan const>> plan_cache_;
  static constexpr size_t kMaxCachedPlans = 64;

  // --- Stream-ordered dependency management (NCCL-compatible) ---

  // Completion flag: host-pinned (gpuHostAlloc with gpuHostAllocMapped),
  // CPU writes seq, GPU WaitValue polls. Monotonic uint64 avoids ABA.
  uint64_t* done_flag_host_ = nullptr;
  gpuDevicePtr_t done_flag_devptr_ = 0;
  std::atomic<uint64_t> next_done_seq_{1};

  // Event pool: pre-allocated gpuEventDisableTiming events for input deps.
  static constexpr int kEventPoolSize = 32;
  std::vector<gpuEvent_t> event_pool_;
  std::mutex event_pool_mu_;
  gpuEvent_t event_pool_acquire();
  void event_pool_release(gpuEvent_t ev);
};

}  // namespace CCL
}  // namespace UKernel
