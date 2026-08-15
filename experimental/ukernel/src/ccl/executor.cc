#include "executor.h"
#include "../../include/transport.h"
#include "coll_algo.h"
#include "backend/backend.h"
#include "coll_config.h"
#include "util/uk_debug.h"
#include "util/host_prof.h"
#include "utils.h"
#include <algorithm>
#include <csignal>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <pthread.h>

namespace UKernel {
namespace CCL {

namespace {
// SIGUSR2 requests a state dump of all running runs (printed by
// enqueue_loop, not the handler). Debug aid for distributed stalls:
// kill -USR2 <pid> on the surviving rank after the peer's watchdog fires.
std::atomic<bool> g_dump_all_requested{false};

// UK_CCL_OP_TRACE=1: print per-op completion latency (now - enqueue_ns)
// for the first ~40 ops of each kind — isolates the put/reduce/signal
// dependency-chain latency that aggregate HostProf buckets blur.
void trace_op(char const* kind, uint64_t enqueue_ns, int rank) {
  static bool const enabled = std::getenv("UK_CCL_OP_TRACE") != nullptr;
  if (!enabled) return;
  static std::atomic<int> tpt_n{0}, dev_n{0}, sig_n{0};
  std::atomic<int>* c =
      (kind[0] == 'p') ? &tpt_n : (kind[0] == 'd') ? &dev_n : &sig_n;
  int const n = c->fetch_add(1, std::memory_order_relaxed);
  if (n >= 40) return;
  auto const now = std::chrono::steady_clock::now().time_since_epoch().count();
  std::fprintf(stderr, "[trace r%d] %s#%d enq->done %.1fus\n", rank, kind, n,
               (now - static_cast<int64_t>(enqueue_ns)) / 1000.0);
}
}  // namespace

static void update_path_metrics(PathMetrics& m, uint64_t enqueue_ns) {
  m.inflight.fetch_sub(1, std::memory_order_relaxed);
  uint64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
  uint64_t sample = now - enqueue_ns;
  uint64_t old = m.latency_ns.load(std::memory_order_relaxed);
  uint64_t nv = (old * 7 + sample) / 8;
  while (!m.latency_ns.compare_exchange_weak(old, nv, std::memory_order_release,
                                             std::memory_order_relaxed))
    ;
}

static uint32_t role_to_buf(CollectiveBufferRole role, uint32_t in,
                            uint32_t out, uint32_t scr) {
  switch (role) {
    case CollectiveBufferRole::Input:
      return in;
    case CollectiveBufferRole::Output:
      return out;
    case CollectiveBufferRole::Scratch:
      return scr;
  }
  return 0;
}

// Role → base offset of the role's tensor within its registered
// allocation. With allocation-scoped registration, op offsets (relative
// to the tensor) must be shifted by the tensor's offset inside its
// allocation.
static uint64_t role_to_off(CollectiveBufferRole role, uint64_t in,
                            uint64_t out, uint64_t scr) {
  switch (role) {
    case CollectiveBufferRole::Input:
      return in;
    case CollectiveBufferRole::Output:
      return out;
    case CollectiveBufferRole::Scratch:
      return scr;
  }
  return 0;
}

// Effective in-place flag: the explicit CollectiveConfig::inplace wins
// (the NCCL shim sets it for AllGather/ReduceScatter, whose in-place
// layouts use distinct overlapping pointers that `input == output` would
// miss); otherwise fall back to pointer equality (AllReduce/AllToAll).
static inline bool cfg_inplace(CollectiveConfig const& cfg, void* input,
                               void* output) {
  return cfg.inplace || (input == output);
}

// Signal-tag epoch salting. Plan-level tags (make_tag) repeat across
// runs of the same shape, and rank skew (±1 run) lets a future run's
// arrivals satisfy — or be buffered for — the wrong run's waits:
// count-conserving but data-unsafe (a wait can fire before its run's
// data landed). Folding a per-executor monotonic run epoch into the
// high 32 bits makes cross-run matching impossible. Both ranks derive
// the same epoch for the k-th submit (rank-symmetric issue order).
// NOTE: salted tags no longer fit the 32-bit RDMA immediate, so fused
// RDMA puts send only the UNSALTED tag (== the salted tag's low 32
// bits) as the immediate, and the receiver matches immediates per-peer
// FIFO in arrival order — cross-run uniqueness comes from ordering (all
// fused puts to a peer are pinned to one QP), not from the tag.
// Standalone RDMA signals still carry the full 64-bit tag on the
// signal QP.
static inline uint64_t salt_tag(uint64_t base, uint32_t epoch) {
  return base | (static_cast<uint64_t>(epoch) << 32);
}

static inline bool tag_fits_rdma_imm(uint64_t tag) {
  return tag <= 0xFFFFFFFFull;
}

// RDMA put-signal fusion predicate — the sender's Put gate and the
// receiver's WaitSignal mirror both use it, so the two sides always
// agree on whether a group's tags travel as 32-bit immediates. Fusion
// requires the UNSALTED tag to fit the immediate and the RDMA path to
// support fused PutSignal. Same-host RDMA puts are excluded unless
// UK_CCL_PUT_PATH=rdma forces the path: the receiver cannot otherwise
// predict whether the load balancer routed a given put via IPC
// (shm-ring signal) or RDMA (immediate).
static bool rdma_imm_fusion_active(Transport::Communicator* comm, int peer,
                                   uint64_t unsalted_tag) {
  static const bool kForceRdma = []() {
    char const* v = std::getenv("UK_CCL_PUT_PATH");
    return v && std::string(v) == "rdma";
  }();
  if (!tag_fits_rdma_imm(unsalted_tag)) return false;
  if (!kForceRdma && comm->same_host(peer)) return false;
  return comm->can_fuse_put_signal(peer, Transport::PeerTransportKind::Rdma);
}

static Cmd make_cmd(TiledOp const& op, ReductionKind redop, ScalarType dtype,
                    uint32_t input_buf, uint32_t output_buf,
                    uint32_t scratch_buf, uint64_t input_off,
                    uint64_t output_off, uint64_t scr_off,
                    uint32_t tag_epoch) {
  Cmd c{};
  c.kind = op.kind;
  c.bytes = static_cast<uint32_t>(op.bytes);
  c.dtype = dtype;
  // Op offsets are relative to the role's tensor; shift by the tensor's
  // offset within its registered allocation (allocation-scoped
  // registration).
  c.src_off =
      op.src_off + role_to_off(op.src_buf_role, input_off, output_off, scr_off);
  c.dst_off =
      op.dst_off + role_to_off(op.dst_buf_role, input_off, output_off, scr_off);
  c.src_peer = op.src_peer;
  c.dst_peer = op.dst_peer;
  auto role_src = op.src_buf_role;
  auto role_dst = op.dst_buf_role;
  c.src_buf = role_to_buf(role_src, input_buf, output_buf, scratch_buf);
  c.dst_buf = role_to_buf(role_dst, input_buf, output_buf, scratch_buf);
  c.src2_buf = role_to_buf(op.src2_buf_role, input_buf, output_buf,
                           scratch_buf);
  c.copy_dst_buf = role_to_buf(op.copy_dst_buf_role, input_buf, output_buf,
                               scratch_buf);
  c.copy_dst_peer = op.copy_dst_peer;
  c.copy_dst_off = op.copy_dst_off;
  c.flag_slot = op.flag_slot;
  c.flag_count = op.flag_count;
  c.redop = (op.kind == ExecOpKind::Reduce) ? redop : ReductionKind::None;
  c.put_path = op.put_path_hint;  // None = auto (pick_put_path below)
  if (op.reduce_mode == 1) c.flags |= kCmdFlagReduce3Way;
  c.tag = salt_tag(op.tag, tag_epoch);
  if (op.fused_copy) c.flags |= kCmdFlagReduceCopy;
  if (op.kind == ExecOpKind::Put && op.flag_slot != ~0u)
    c.flags |= kCmdFlagCopySignal;
  return c;
}

// Opaque binary key covering everything that shapes a plan. Buffer
// pointers are deliberately excluded: plans reference buffers by role
// and are shared across different input/output pointers.
static std::string plan_key(CollectiveConfig const& cfg, bool inplace) {
  std::string k;
  k.reserve(128);
  auto add = [&k](uint64_t v) {
    k.append(reinterpret_cast<char const*>(&v), sizeof(v));
  };
  add(static_cast<uint64_t>(cfg.kind));
  add(static_cast<uint64_t>(cfg.nranks));
  add(static_cast<uint64_t>(cfg.rank));
  add(cfg.input_bytes);
  add(cfg.output_bytes);
  add(cfg.tile_bytes);
  add(static_cast<uint64_t>(cfg.dtype));
  add(static_cast<uint64_t>(cfg.reduction));
  add(static_cast<uint64_t>(cfg.signal_group_tiles));
  add(cfg.fuse_rs_reduce ? 1u : 0u);
  add(cfg.fuse_reduce_copy ? 1u : 0u);
  add(cfg.fuse_ag_copy ? 1u : 0u);
  add(cfg.device_flags ? 1u : 0u);
  add(inplace ? 1u : 0u);
  add(cfg.input_split_bytes.size());
  for (size_t v : cfg.input_split_bytes) add(v);
  add(cfg.output_split_bytes.size());
  for (size_t v : cfg.output_split_bytes) add(v);
  return k;
}

// Allocation base for a device pointer (falls back to the pointer
// itself when not GPU-allocated). Mirrors the registration keying in
// get_or_register_buf: the prepare cache is keyed on allocations, not
// raw pointers — nccl-tests shifts buffer offsets per iteration inside
// one big allocation, and torch hands out different pool addresses
// constantly, so keying on raw pointers would re-run the full
// peer-setup/resolve prepare on every call.
static void* alloc_base_of(void const* p) {
  void* base = nullptr;
  size_t size = 0;
  if (p && gpuMemGetAddressRange(&base, &size, const_cast<void*>(p)) ==
               gpuSuccess)
    return base;
  return const_cast<void*>(p);
}

// Opaque binary key for prepare() dedup: everything that determines
// peer setup and buffer resolution — the collective kind, the peer set,
// the allocations holding input/output, their byte counts, and the
// declared Tmp footprint (an in-place allreduce and its out-of-place
// twin differ only here: the former registers/resolves scratch).
static std::string prepare_key(CollectiveConfig const& cfg,
                               std::vector<int> const& peers,
                               uintptr_t input_base, uintptr_t output_base,
                               size_t tmp_total) {
  std::string k;
  k.reserve(96);
  auto add = [&k](uint64_t v) {
    k.append(reinterpret_cast<char const*>(&v), sizeof(v));
  };
  add(static_cast<uint64_t>(cfg.kind));
  add(cfg.input_bytes);
  add(cfg.output_bytes);
  add(tmp_total);
  add(peers.size());
  for (int p : peers) add(static_cast<uint64_t>(p));
  add(input_base);
  add(output_base);
  return k;
}

// Alloc-base lookup with a per-executor cache: nccl-tests shifts buffer
// offsets inside one allocation every iteration, so the raw pointer
// changes while the CUDA allocation base does not. The driver call per
// lookup was part of the per-call prepare overhead.
uintptr_t SprayExecutor::cached_alloc_base(void const* p) {
  uintptr_t const raw = reinterpret_cast<uintptr_t>(p);
  auto it = alloc_base_cache_.find(raw);
  if (it != alloc_base_cache_.end()) return it->second;
  uintptr_t const base = reinterpret_cast<uintptr_t>(alloc_base_of(p));
  if (alloc_base_cache_.size() >= 8192) alloc_base_cache_.clear();
  alloc_base_cache_.emplace(raw, base);
  return base;
}

// Build the immutable plan: tiling/lowering plus the successor CSR and
// the initial scheduling state that submit() would otherwise rebuild
// on every call.
static std::shared_ptr<CollPlan const> build_plan(CollectiveConfig const& cfg,
                                                  bool inplace) {
  auto plan = std::make_shared<CollPlan>();
  plan->tiled = build_tiled(cfg, inplace);
  size_t nops = plan->tiled.ops.size();
  plan->nops = nops;

  // Cmd.bytes is uint32_t (task ABI / RDMA WR width). Adaptive tiling
  // can in principle produce single ops beyond that for multi-TB
  // tensors; reject the plan loudly instead of silently truncating
  // every op's byte count in make_cmd.
  for (auto const& op : plan->tiled.ops) {
    if (op.bytes > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument(
          "executor: op bytes exceed the 32-bit Cmd.bytes limit");
  }

  plan->indegree_init.resize(nops);
  std::vector<uint32_t> succ_count(nops, 0);
  for (uint32_t i = 0; i < nops; ++i) {
    plan->indegree_init[i] =
        static_cast<uint32_t>(plan->tiled.ops[i].deps.size());
    for (uint32_t dep : plan->tiled.ops[i].deps) ++succ_count[dep];
  }

  plan->successor_off.resize(nops + 1);
  uint32_t off = 0;
  for (uint32_t i = 0; i < nops; ++i) {
    plan->successor_off[i] = off;
    off += succ_count[i];
  }
  plan->successor_off[nops] = off;
  plan->successor_data.resize(off);

  std::vector<uint32_t> pos = plan->successor_off;
  for (uint32_t i = 0; i < nops; ++i)
    for (uint32_t dep : plan->tiled.ops[i].deps)
      plan->successor_data[pos[dep]++] = i;

  for (uint32_t i = 0; i < nops; ++i)
    if (plan->tiled.ops[i].deps.empty()) plan->initial_ready.push_back(i);

  // PutSignal fusion metadata from the lowerer.
  plan->put_to_sig.assign(nops, -1);
  plan->sig_group_size.assign(nops, 0);
  plan->wait_group_size.assign(nops, 0);
  for (auto [sig_idx, put_idx] : plan->tiled.fused_put_signal)
    plan->put_to_sig[put_idx] = static_cast<int32_t>(sig_idx);
  for (auto [sig_idx, grp] : plan->tiled.sig_group_size)
    plan->sig_group_size[sig_idx] = static_cast<uint16_t>(grp);
  for (auto [ws_idx, grp] : plan->tiled.wait_group_size)
    plan->wait_group_size[ws_idx] = static_cast<uint16_t>(grp);
  return plan;
}

// Allocation-scoped buffer registration. The first time a pointer is
// seen, its whole CUDA allocation [base, base+size) is registered once
// under a new buf id; any later (ptr, bytes) inside the same allocation
// reuses that id. Op addressing shifts by (tensor_ptr - alloc_base) at
// make_cmd time, so id count stays O(allocations), not O(calls) — this
// is what keeps nccl-tests' per-iteration buffer shifting (and torch's
// caching-allocator pointers) from exploding registrations and the OOB
// KV store.
uint32_t SprayExecutor::get_or_register_buf(void* ptr, size_t bytes) {
  return get_or_register_buf(ptr, bytes, nullptr, "?");
}

uint32_t SprayExecutor::get_or_register_buf(void* ptr, size_t bytes,
                                            uint64_t* out_off,
                                            char const* role) {
  if (out_off) *out_off = 0;
  if (!ptr || !bytes) return 0;

  void* alloc_base = nullptr;
  size_t alloc_size = 0;
  bool have_range = gpuMemGetAddressRange(&alloc_base, &alloc_size, ptr) ==
                    gpuSuccess;

  if (have_range) {
    uintptr_t key = reinterpret_cast<uintptr_t>(alloc_base);
    auto it = tensor_to_buf_id_.find(key);
    if (it != tensor_to_buf_id_.end()) {
      BufReg const& r = it->second;
      if (r.alloc_size == alloc_size) {
        if (out_off) *out_off = reinterpret_cast<uintptr_t>(ptr) - key;
        return r.id;
      }
      // Size change at the same base means the VA was freed and
      // re-allocated: the old registration — and every peer's cached
      // resolve of it — points at a dead allocation. Evict it,
      // deregister the old MR, and mint a FRESH buf id below (peers key
      // resolves by id; reusing the id would silently alias the dead
      // VA). Prepare-cache entries that resolved the old id are
      // invalidated, so the next collective re-prepares and peers
      // re-resolve.
      //
      // Rank symmetry: buf ids are minted in first-seen order on every
      // rank, so this stays symmetric only if all ranks observe the
      // same alloc/free sequence — true when ranks drive the same
      // collective sequence over identical allocator behavior.
      // Remaining window: a peer may still PUT into the old MR until it
      // re-resolves (there is no cross-rank invalidation protocol), so
      // an allocation must not be recycled while collectives using it
      // can still be in flight.
      std::fprintf(stderr,
                   "[bufreg r%d] allocation %p size changed (%zu -> %zu); "
                   "evicting stale registration id=%u\n",
                   rank_or_neg1(), alloc_base, r.alloc_size, alloc_size, r.id);
      if (owned_comm_ && deregister_buf_fn_)
        deregister_buf_fn_(owned_comm_.get(), r.id);
      invalidate_prepared_by_base(key);
      tensor_to_buf_id_.erase(it);
      // Fall through to a fresh registration below.
    }
    uint32_t id = next_buf_id_++;
    tensor_to_buf_id_[key] = BufReg{id, alloc_base, alloc_size};
    if (owned_comm_ && register_buf_fn_)
      register_buf_fn_(owned_comm_.get(), id, alloc_base, alloc_size);
    if (out_off) *out_off = reinterpret_cast<uintptr_t>(ptr) - key;
    UK_DBG(UK_DBG_LVL_EXEC,
           "[bufreg r%d] id=%u base=%p alloc=%zu (ptr=%p bytes=%zu) role=%s",
           rank_or_neg1(), id, alloc_base, alloc_size, ptr, bytes, role);
    return id;
  }

  // Fallback for pointers not from the CUDA allocator: register the
  // requested extent directly, offset 0.
  uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
  auto it = tensor_to_buf_id_.find(key);
  if (it != tensor_to_buf_id_.end()) return it->second.id;
  uint32_t id = next_buf_id_++;
  tensor_to_buf_id_[key] = BufReg{id, ptr, bytes};
  if (owned_comm_ && register_buf_fn_)
    register_buf_fn_(owned_comm_.get(), id, ptr, bytes);
  UK_DBG(UK_DBG_LVL_EXEC, "[bufreg r%d] id=%u ptr=%p bytes=%zu role=%s (raw)",
         rank_or_neg1(), id, ptr, bytes, role);
  return id;
}

// Drop every prepare-cache entry that resolved buffers from allocation
// `base` (called under api_mu_ from get_or_register_buf on a stale
// registration). Affected shapes re-run prepare() — including peer
// buffer re-resolve — on their next collective.
void SprayExecutor::invalidate_prepared_by_base(uintptr_t base) {
  auto range = prepared_key_bases_.equal_range(base);
  for (auto it = range.first; it != range.second; ++it)
    prepared_keys_.erase(it->second);
  prepared_key_bases_.erase(range.first, range.second);
}

gpuEvent_t SprayExecutor::event_pool_acquire() {
  std::lock_guard<std::mutex> lk(event_pool_mu_);
  if (!event_pool_.empty()) {
    gpuEvent_t ev = event_pool_.back();
    event_pool_.pop_back();
    return ev;
  }
  gpuEvent_t ev = nullptr;
  GPU_RT_CHECK(gpuEventCreateWithFlags(&ev, gpuEventDisableTiming));
  return ev;
}

void SprayExecutor::event_pool_release(gpuEvent_t ev) {
  std::lock_guard<std::mutex> lk(event_pool_mu_);
  event_pool_.push_back(ev);
}

SprayExecutor::SprayExecutor(BatchBackend* device_be, BatchBackend* tpt_be,
                             BatchBackend* signal_be, int world_size)
    : device_be_(device_be),
      tpt_be_(tpt_be),
      signal_be_(signal_be),
      stop_(false),
      dev_slots_(device_be ? device_be->capacity() : 0),
      tpt_slots_(tpt_be ? tpt_be->capacity() : 0),
      sig_slots_(signal_be ? signal_be->capacity() : 0),
      world_size_(world_size) {
  if (world_size_ > 0)
    tpt_metrics_.reset(new PeerMetrics[static_cast<size_t>(world_size_)]{});

  char const* pc = std::getenv("UK_CCL_PATH_COUNTERS");
  path_counters_enabled_ =
      (pc && (strcmp(pc, "1") == 0 || strcmp(pc, "true") == 0));
}

void SprayExecutor::start() {
  // Stream-ordered deps: pinned+mapped completion flag for GPU WaitValue.
  GPU_RT_CHECK(gpuHostAlloc(reinterpret_cast<void**>(&done_flag_host_),
                            sizeof(uint64_t), gpuHostAllocMapped));
  GPU_RT_CHECK(gpuHostGetDevicePointer(
      reinterpret_cast<void**>(&done_flag_devptr_), done_flag_host_, 0));
  *done_flag_host_ = 0;

  // Pre-allocate event pool for input dependency gating.
  for (int i = 0; i < kEventPoolSize; ++i) {
    gpuEvent_t ev = nullptr;
    GPU_RT_CHECK(gpuEventCreateWithFlags(&ev, gpuEventDisableTiming));
    event_pool_.push_back(ev);
  }

  // SIGUSR2: dump all running runs from enqueue_loop (see namespace
  // comment above). One handler per process is enough.
  static std::once_flag sig_once;
  std::call_once(sig_once, [] {
    std::signal(SIGUSR2, [](int) {
      g_dump_all_requested.store(true, std::memory_order_relaxed);
    });
  });

  enqueue_th_ = std::thread(&SprayExecutor::enqueue_loop, this);
  pthread_setname_np(enqueue_th_.native_handle(), "ucl-enq");
  if (device_be_) {
    drain_th_dev_ = std::thread(&SprayExecutor::drain_dev_loop, this);
    pthread_setname_np(drain_th_dev_.native_handle(), "ucl-drain-dev");
  }
  if (tpt_be_) {
    drain_th_tpt_ = std::thread(&SprayExecutor::drain_tpt_loop, this);
    pthread_setname_np(drain_th_tpt_.native_handle(), "ucl-drain-tpt");
  }
  if (signal_be_) {
    drain_th_signal_ = std::thread(&SprayExecutor::drain_signal_loop, this);
    pthread_setname_np(drain_th_signal_.native_handle(), "ucl-drain-sig");
  }
}

SprayExecutor::~SprayExecutor() {
  stop_ = true;
  if (owned_comm_) owned_comm_->stop_transports();
  if (enqueue_th_.joinable()) enqueue_th_.join();
  if (drain_th_dev_.joinable()) drain_th_dev_.join();
  if (drain_th_tpt_.joinable()) drain_th_tpt_.join();
  if (drain_th_signal_.joinable()) drain_th_signal_.join();
  HostProf::print();
  // Explicitly release backends before communicator is destroyed.
  // Backends hold raw comm_ pointers that must remain valid during
  // their destructors (e.g. DeviceBackend tears down GPU task manager).
  device_be_ = nullptr;
  tpt_be_ = nullptr;
  signal_be_ = nullptr;
  owned_device_.reset();
  owned_transport_.reset();
  owned_signal_.reset();
  for (auto& [bytes, sb] : scratch_by_size_)
    GPU_RT_CHECK(gpuFree(sb.alloc_raw));
  // Stream-ordered dep cleanup.
  for (gpuEvent_t ev : event_pool_)
    if (ev) (void)gpuEventDestroy(ev);
  if (done_flag_host_) {
    (void)gpuFreeHost(done_flag_host_);
  }
}

SprayRun* SprayExecutor::get(CollectiveOpHandle h) {
  std::lock_guard lock(runs_mutex_);
  auto it = runs_.find(h);
  return it != runs_.end() ? it->second.get() : nullptr;
}

CollectiveOpStatus SprayExecutor::status(CollectiveOpHandle h) const {
  std::lock_guard lock(runs_mutex_);
  auto it = runs_.find(h);
  return it != runs_.end() ? it->second->status.load(std::memory_order_acquire)
                           : CollectiveOpStatus::Completed;
}

size_t SprayExecutor::active_count() const {
  std::lock_guard lock(runs_mutex_);
  size_t n = 0;
  for (auto& [h, r] : runs_)
    if (r->status.load(std::memory_order_acquire) ==
        CollectiveOpStatus::Running)
      ++n;
  return n;
}

std::string SprayExecutor::error_message(CollectiveOpHandle h) const {
  std::lock_guard lock(runs_mutex_);
  auto it = runs_.find(h);
  return it != runs_.end() ? it->second->error : std::string{};
}

// Allocate the scratch buffer for this size if absent (api_mu_ held).
// One buffer per distinct size, id minted from the shared counter so
// the id space is unbounded (see executor.h).
void SprayExecutor::ensure_internal_scratch(size_t bytes) {
  if (bytes == 0 || scratch_by_size_.count(bytes)) return;
  uint32_t id = next_buf_id_++;
  // RDMA DMA-BUF MR registration requires a 64KB-aligned buffer address;
  // cudaMalloc only guarantees 256B alignment, so an unlucky placement
  // (observed: 1M scratch at base%64K==2048) fails ibv_reg_dmabuf_mr and
  // breaks the whole run. Allocate the alignment margin and round up;
  // keep the raw allocation for gpuFree (it must free the original
  // pointer, not the aligned interior).
  constexpr size_t kScratchAlign = 65536;
  void* raw = nullptr;
  GPU_RT_CHECK(gpuMalloc(&raw, bytes + kScratchAlign - 1));
  uintptr_t aligned =
      (reinterpret_cast<uintptr_t>(raw) + kScratchAlign - 1) &
      ~(kScratchAlign - 1);
  void* ptr = reinterpret_cast<void*>(aligned);
  scratch_by_size_[bytes] = ScratchBuf{ptr, raw, id};
  if (owned_comm_ && register_buf_fn_)
    register_buf_fn_(owned_comm_.get(), id, ptr, bytes);
}

void SprayExecutor::prepare(CollectiveConfig const& cfg, void* input,
                            void* output) {
  std::lock_guard<std::mutex> lock(api_mu_);
  // Single-rank collectives have no peers: nothing to set up.
  if (cfg.nranks <= 1) return;
  if (!owned_comm_ || !peer_setup_fn_) return;

  // Fast path: the algorithm DAG is shape-determined (kind, ranks, byte
  // counts, in-place) and does not depend on buffer pointers, so cache
  // it keyed by shape. Rebuilding it per call plus two driver
  // alloc-base lookups was ~5-8us of the ~15us per-call prepare+submit
  // overhead (measured with a 2-rank 256KB ncclAllReduce probe).
  const bool inplace = cfg_inplace(cfg, input, output);
  std::string skey;
  skey.reserve(48);
  auto add = [&skey](uint64_t v) {
    skey.append(reinterpret_cast<char const*>(&v), sizeof(v));
  };
  add(static_cast<uint64_t>(cfg.kind));
  add(static_cast<uint64_t>(cfg.nranks));
  add(static_cast<uint64_t>(cfg.rank));
  add(cfg.input_bytes);
  add(cfg.output_bytes);
  add(inplace ? 1u : 0u);
  CollAlgo algo;
  auto ait = prepare_algo_cache_.find(skey);
  if (ait != prepare_algo_cache_.end()) {
    algo = ait->second;
  } else {
    algo = build_coll_algo(cfg, inplace);
    if (prepare_algo_cache_.size() >= 256) prepare_algo_cache_.clear();
    prepare_algo_cache_.emplace(skey, algo);
  }

  // Derive needed peers from the algorithm DAG.
  std::vector<int> peers;
  for (auto const& ch : algo.chunks) {
    if (ch.src_rank >= 0) peers.push_back(ch.src_rank);
    if (ch.dst_rank >= 0) peers.push_back(ch.dst_rank);
  }
  // Deduplicate and sort.
  std::sort(peers.begin(), peers.end());
  peers.erase(std::unique(peers.begin(), peers.end()), peers.end());

  // Dedup: peer setup, MR (re)registration and buffer resolution are
  // expensive and only needed once per (shape, allocations) combination.
  // Callers are expected to invoke prepare() before every submit.
  size_t tmp_total = 0;
  for (size_t b : algo.tmp_bytes) tmp_total += b;
  uintptr_t const ibase = cached_alloc_base(input);
  uintptr_t const obase = cached_alloc_base(output);
  std::string pkey = prepare_key(cfg, peers, ibase, obase, tmp_total);
  if (prepared_keys_.count(pkey)) return;
  if (prepared_keys_.size() >= kMaxPreparedKeys) {
    prepared_keys_.clear();
    prepared_key_bases_.clear();
  }

  {
    std::string plist;
    for (auto p : peers) {
      if (!plist.empty()) plist += ",";
      plist += std::to_string(p);
    }
    UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] peers=%s  -> peer_setup_fn start",
           cfg.rank, plist.c_str());
  }
  peer_setup_fn_(owned_comm_.get(), cfg.rank, peers);
  UK_DBG(UK_DBG_LVL_EXEC,
         "[prepare r%d] peer_setup_fn done  -> re_register_all_mrs", cfg.rank);
  owned_comm_->re_register_all_mrs();
  UK_DBG(UK_DBG_LVL_EXEC,
         "[prepare r%d] re_register_all_mrs done  -> register bufs", cfg.rank);
  prepared_peers_.insert(peers.begin(), peers.end());
  prepared_ = true;

  // Register and resolve user buffers.
  uint32_t in_id = get_or_register_buf(input, cfg.input_bytes, nullptr, "prep-in");
  UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] in_id=%u registered  -> resolve bufs",
         cfg.rank, in_id);
  uint32_t out_id =
      get_or_register_buf(output, cfg.output_bytes, nullptr, "prep-out");
  UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] out_id=%u registered", cfg.rank,
         out_id);
  for (int p : peers) {
    if (in_id && resolve_buf_fn_) {
      UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] resolve in_id=%u from peer %d ...",
             cfg.rank, in_id, p);
      resolve_buf_fn_(owned_comm_.get(), p, world_size_, in_id);
      UK_DBG(UK_DBG_LVL_EXEC,
             "[prepare r%d] resolve in_id=%u from peer %d done", cfg.rank,
             in_id, p);
    }
    if (out_id && out_id != in_id && resolve_buf_fn_) {
      UK_DBG(UK_DBG_LVL_EXEC,
             "[prepare r%d] resolve out_id=%u from peer %d ...", cfg.rank,
             out_id, p);
      resolve_buf_fn_(owned_comm_.get(), p, world_size_, out_id);
      UK_DBG(UK_DBG_LVL_EXEC,
             "[prepare r%d] resolve out_id=%u from peer %d done", cfg.rank,
             out_id, p);
    }
  }

  // Algorithms declaring Tmp regions (ReduceScatter's partial sums,
  // binary-tree's first-child partial, in-place allreduce) route them
  // through an executor scratch buffer under a fixed per-size pool id.
  // Allocate+register ours iff this plan needs one; resolve a peer's
  // iff this plan puts into its Tmp — doing it at submit time would be
  // too late (the peer needs the MR/handle published to resolve here).
  // The per-size pool id is derived identically on all ranks (same
  // shape sequence), and each size registers exactly once, so this
  // resolve is a plain first-publish wait.
  if (tmp_total > 0) ensure_internal_scratch(tmp_total);
  if (resolve_buf_fn_ && tmp_total > 0) {
    uint32_t const scr_id = scratch_by_size_[tmp_total].id;
    std::set<int> scratch_peers;
    for (auto const& ch : algo.chunks)
      if (ch.op == AlgoOpKind::Put && ch.dst.space == BufSpace::Tmp &&
          ch.dst_rank >= 0)
        scratch_peers.insert(ch.dst_rank);
    for (int p : scratch_peers) {
      UK_DBG(UK_DBG_LVL_EXEC,
             "[prepare r%d] resolve scr_id=%u from peer %d ...", cfg.rank,
             scr_id, p);
      resolve_buf_fn_(owned_comm_.get(), p, world_size_, scr_id);
      UK_DBG(UK_DBG_LVL_EXEC,
             "[prepare r%d] resolve scr_id=%u from peer %d done", cfg.rank,
             scr_id, p);
    }
  }

  prepared_keys_.insert(pkey);
  prepared_key_bases_.emplace(ibase, pkey);
  prepared_key_bases_.emplace(obase, std::move(pkey));
  UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] ALL DONE", cfg.rank);
  prepared_ = true;
}

CollectiveOpHandle SprayExecutor::submit(CollectiveConfig const& cfg,
                                         void* input, void* output,
                                         gpuStream_t stream) {
  std::lock_guard<std::mutex> api_lock(api_mu_);

  // Single-rank collective: no peers, no network ops. Out-of-place still
  // needs a local D2D copy, enqueued on the user stream itself so it
  // stays stream-ordered after the user's preceding kernels (the input
  // event gate is therefore trivially satisfied). The run completes
  // immediately on the CPU side and deliberately skips the WaitValue
  // output gate: the copy is already ordered on the user stream, and
  // nothing would ever write a done_seq for this run.
  if (cfg.nranks <= 1) {
    if (input != output && cfg.input_bytes > 0) {
      gpuError_t err = gpuMemcpyAsync(output, input, cfg.input_bytes,
                                      gpuMemcpyDeviceToDevice, stream);
      if (err != gpuSuccess)
        throw std::runtime_error(std::string("single-rank copy failed: ") +
                                 gpuGetErrorString(err));
    }
    std::lock_guard rlock(runs_mutex_);
    auto h = next_handle_++;
    auto run = std::make_shared<SprayRun>();
    run->status.store(CollectiveOpStatus::Completed,
                      std::memory_order_release);
    runs_[h] = std::move(run);
    return h;
  }

  if (owned_comm_ && !prepared_) {
    // Check all peers needed by this algorithm are prepared.
    CollAlgo algo = build_coll_algo(cfg, cfg_inplace(cfg, input, output));
    for (auto const& ch : algo.chunks) {
      if (ch.src_rank >= 0 && !prepared_peers_.count(ch.src_rank))
        throw std::runtime_error("prepare() not called for peer " +
                                 std::to_string(ch.src_rank));
      if (ch.dst_rank >= 0 && !prepared_peers_.count(ch.dst_rank))
        throw std::runtime_error("prepare() not called for peer " +
                                 std::to_string(ch.dst_rank));
    }
  }

  // Plan cache: identical collective shapes reuse the same immutable
  // plan; only mutable scheduling state is per-run.
  std::shared_ptr<CollPlan const> plan;
  {
    std::lock_guard lock(plan_cache_mu_);
    std::string key = plan_key(cfg, cfg_inplace(cfg, input, output));
    auto it = plan_cache_.find(key);
    if (it != plan_cache_.end()) {
      plan = it->second;
    } else {
      if (plan_cache_.size() >= kMaxCachedPlans) plan_cache_.clear();
      plan = build_plan(cfg, cfg_inplace(cfg, input, output));
      plan_cache_.emplace(std::move(key), plan);
    }
  }
  TiledResult const& tiled = plan->tiled;

  // Allocate or grow internal scratch buffer as needed.
  ensure_internal_scratch(tiled.staging_bytes_required);

  uint32_t in_id = 0, out_id = 0, scr_id = 0;
  uint64_t in_off = 0, out_off = 0, scr_off = 0;
  if (owned_comm_) {
    in_id = get_or_register_buf(input, tiled.input_bytes, &in_off, "sub-in");
    out_id = get_or_register_buf(output, tiled.output_bytes, &out_off, "sub-out");
    // Scratch uses a fixed per-size pool id (registered by
    // ensure_internal_scratch on first use of the size); each buffer is
    // its own allocation, so the base offset is 0.
    if (tiled.staging_bytes_required > 0) {
      scr_id = scratch_by_size_[tiled.staging_bytes_required].id;
      // A scratch id freshly minted by THIS submit (the plan's shape was
      // never prepared, e.g. an in-place AllReduce handshake whose Tmp
      // puts target the peer's scratch) must be resolved on the peer
      // before any put referencing it can be sent — prepare() only
      // resolved the scratch of the shape it was called with. Without
      // this, the peer's wait for this id never completes and the run
      // deadlocks (observed: AllToAll perf, whose prepare declares no
      // Tmp, deadlocks on the in-place AllReduce handshake's scratch).
      std::set<int> scratch_peers;
      for (auto const& op : tiled.ops)
        if (op.kind == ExecOpKind::Put &&
            op.dst_buf_role == CollectiveBufferRole::Scratch &&
            op.dst_peer != ~0u)
          scratch_peers.insert(static_cast<int>(op.dst_peer));
      if (resolve_buf_fn_ && !scratch_peers.empty()) {
        for (int p : scratch_peers) {
          UK_DBG(UK_DBG_LVL_EXEC,
                 "[submit r%d] resolve scr_id=%u from peer %d ...", cfg.rank,
                 scr_id, p);
          resolve_buf_fn_(owned_comm_.get(), p, world_size_, scr_id);
          UK_DBG(UK_DBG_LVL_EXEC,
                 "[submit r%d] resolve scr_id=%u from peer %d done", cfg.rank,
                 scr_id, p);
        }
      }
    }
  }

  std::lock_guard lock(runs_mutex_);

  while (active_runs_.load(std::memory_order_acquire) >= max_concurrent_runs_) {
    runs_mutex_.unlock();
    std::this_thread::yield();
    runs_mutex_.lock();
  }

  auto h = next_handle_++;
  if (plan->nops == 0) {
    auto run = std::make_shared<SprayRun>();
    run->plan = plan;
    run->status.store(CollectiveOpStatus::Completed, std::memory_order_release);
    runs_[h] = std::move(run);
    return h;
  }

  auto run = std::make_shared<SprayRun>();
  active_runs_.fetch_add(1, std::memory_order_release);
  run->status.store(CollectiveOpStatus::Running, std::memory_order_release);
  run->plan = plan;
  run->input_buf_id = in_id;
  run->output_buf_id = out_id;
  run->scratch_buf_id = scr_id;
  run->input_base_off = in_off;
  run->output_base_off = out_off;
  run->scratch_base_off = scr_off;
  // Signal-tag epoch: monotonic per executor, identical across ranks for
  // the k-th submit (rank-symmetric issue order). Skipped for nops==0
  // runs above, which is symmetric too (same plan => same nops).
  run->tag_epoch = static_cast<uint32_t>(++next_run_epoch_);
  size_t nops = plan->nops;
  run->submitted.resize(nops, 0);
  run->fused_sig_cnt.assign(nops, 0);
  run->accepted_sig_cnt.assign(nops, 0);

  UK_DBG(UK_DBG_LVL_EXEC, "[submit r%d] %zu ops", cfg.rank, nops);
  run->init_ready_ring(nops);
  run->indegree = plan->indegree_init;  // one memcpy from the template
  for (uint32_t op : plan->initial_ready) run->push_ready(op);

  // Input dependency: record an event on the user's stream; enqueue_loop
  // gates on it (non-blocking cudaEventQuery) before pushing any op to
  // backends, so the executor never reads the input buffer before the
  // user's preceding kernels on this stream complete.
  if (stream) {
    run->input_ready = event_pool_acquire();
    gpuEventRecord(run->input_ready, stream);
  }

  // Output dependency: enqueue a WaitValue on the user stream so
  // subsequent kernels on it observe the collective's completion.
  // Enqueued BEFORE the run is published to runs_, so the run cannot
  // complete before its wait is on the stream. done_flag is published
  // with a monotonic max write in finalize_run (see there) — a plain
  // store can regress under concurrent finalization and was the root
  // cause of the multi-iteration hang that once disabled this path.
  if (stream) {
    run->user_stream = stream;
    run->done_seq =
        next_done_seq_.fetch_add(1, std::memory_order_relaxed);

    gpuError_t werr = gpuStreamWaitValue32(
        stream, done_flag_devptr_, static_cast<unsigned int>(run->done_seq),
        CU_STREAM_WAIT_VALUE_GEQ);
    if (werr != gpuSuccess) {
      // Failed enqueue silently drops the output gate: subsequent user
      // kernels on the stream would race the collective. Fail loudly
      // instead (run_coll maps the throw to ncclInternalError).
      if (run->input_ready) {
        event_pool_release(run->input_ready);
        run->input_ready = nullptr;
      }
      throw std::runtime_error(std::string("stream wait-value enqueue "
                                           "failed: ") +
                               gpuGetErrorString(werr));
    }
  }

  runs_[h] = std::move(run);
  {
    std::lock_guard<std::mutex> lk(wake_mu_);
    wake_pending_ = true;
  }
  wake_cv_.notify_one();
  return h;
}

bool SprayExecutor::poll(CollectiveOpHandle h) {
  std::lock_guard lock(runs_mutex_);
  auto it = runs_.find(h);
  if (it == runs_.end()) return true;
  auto s = it->second->status.load(std::memory_order_acquire);
  return s == CollectiveOpStatus::Completed || s == CollectiveOpStatus::Failed;
}

bool SprayExecutor::wait(CollectiveOpHandle h, std::chrono::milliseconds to) {
  SprayRun* run = get(h);
  if (!run) return true;

  // Calling-thread-makes-progress (UCX/MPICH style): the waiting thread
  // drains backends itself instead of sleeping on a timer, so completion
  // is observed as soon as the last op drains rather than after a sleep
  // quantum (previously up to 10ms of tail latency per wait).
  auto const deadline = std::chrono::steady_clock::now() + to;
  bool const use_deadline = to.count() > 0;
  uint32_t empty = 0;
  uint32_t sleep_us = 100;
  while (run->status.load(std::memory_order_acquire) ==
         CollectiveOpStatus::Running) {
    if (use_deadline && std::chrono::steady_clock::now() >= deadline) break;
    size_t n = progress_once();
    finalize_run(run);
    if (n > 0) {
      empty = 0;
      sleep_us = 100;
      continue;
    }
    if (++empty < 2000) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
      sleep_us = std::min(sleep_us * 2, 1000u);
    }
  }
  return run->status.load(std::memory_order_acquire) !=
         CollectiveOpStatus::Failed;
}

void SprayExecutor::release(CollectiveOpHandle h) {
  std::lock_guard lock(runs_mutex_);
  auto it = runs_.find(h);
  if (it == runs_.end()) return;
  auto st = it->second->status.load(std::memory_order_acquire);
  if (st == CollectiveOpStatus::Queued || st == CollectiveOpStatus::Running)
    throw std::logic_error("cannot release running collective");
  // A Failed run is releasable only once quiesced: its in-flight ops
  // still complete through BeSlot raw pointers into the run, so erasing
  // it earlier is a use-after-free. Callers (e.g. the shim's reap) must
  // retry once the remaining completions have drained.
  size_t inflight = it->second->inflight_ops.load(std::memory_order_acquire);
  if (st == CollectiveOpStatus::Failed && inflight != 0)
    throw std::logic_error("cannot release failed collective: " +
                           std::to_string(inflight) +
                           " ops still in flight (run error: " +
                           it->second->error + ")");
  runs_.erase(it);
}

void SprayExecutor::collect_ready(SprayRun& run) {
  run.ready.clear();
  for (;;) {
    uint32_t op = run.pop_ready();
    if (op == ~0u) break;
    if (!run.submitted[op]) run.ready.push_back(op);
  }
  if (run.ready.empty()) {
    static int collect_zero_cnt = 0;
    if (uk_dbg_lvl() >= UK_DBG_LVL_ALL && ++collect_zero_cnt % 200000 == 0)
      UK_DBG(UK_DBG_LVL_ALL, "[enqueue r%d] collected 0 ready", rank_or_neg1());
  }
}

void SprayExecutor::enqueue_to_ring(SprayRun& run) {
  // Prepend deferred ops from prior cycle (preserves priority).
  // Signal deferred ops get highest priority so peer WaitSignal unblocks
  // promptly, avoiding head-of-line blocking from data-path backpressure.
  {
    run.ready.insert(run.ready.begin(), run.deferred_dev.begin(),
                     run.deferred_dev.end());
    run.ready.insert(run.ready.begin(), run.deferred_tpt.begin(),
                     run.deferred_tpt.end());
    run.ready.insert(run.ready.begin(), run.deferred_sig.begin(),
                     run.deferred_sig.end());

    run.deferred_dev.clear();
    run.deferred_tpt.clear();
    run.deferred_sig.clear();
  }
  if (run.ready.empty()) return;

  HostProf::Scope hps(HostProf::enq_us);
  run.dev_cmds.clear();
  run.tpt_cmds.clear();
  std::vector<uint32_t> dev_idx;
  std::vector<uint32_t> tpt_idx;
  size_t sig_dispatched = 0;

  static int op_kind_print_cnt = 0;
  for (uint32_t idx : run.ready) {
    Cmd c = make_cmd(run.plan->tiled.ops[idx], run.plan->tiled.reduction,
                     run.plan->tiled.dtype, run.input_buf_id,
                     run.output_buf_id, run.scratch_buf_id,
                     run.input_base_off, run.output_base_off,
                     run.scratch_base_off, run.tag_epoch);
    if (op_kind_print_cnt++ < 20) {
      UK_DBG(UK_DBG_LVL_EXEC,
             "[enqueue r%d] op[%u] kind=%d dst_peer=%u put_path=%d tag=%lu "
             "bytes=%u",
             rank_or_neg1(), idx, (int)c.kind, c.dst_peer, (int)c.put_path,
             (unsigned long)c.tag, c.bytes);
    }
    if (c.kind == ExecOpKind::Put && c.dst_peer != ~0u) {
      if (c.flags & kCmdFlagCopySignal) {
        // Fused AG copy: must run on the device backend (the task writes
        // the completion flag); never route to CE/RDMA.
        c.put_path = PutPath::Device;
      } else {
        // A builder-set hint (RS hybrid halves) wins; otherwise auto.
        if (c.put_path == PutPath::None)
          c.put_path = pick_put_path(static_cast<int>(c.dst_peer));
      }
      UK_DBG(UK_DBG_LVL_EXEC, "[pick r%d] op[%u] peer=%u -> path=%d",
             rank_or_neg1(), idx, c.dst_peer, (int)c.put_path);
      // On Nvidia consumer GPUs the PCIe BAR1 window is
      // typically 256 MiB.  Set UK_BAR1_WINDOW_MB to enable the
      // fallback to IPC for remote accesses exceeding that window.
      // Data-center GPUs (H100/A100) have full BAR1 mapping and
      // should leave this unset.
      static const size_t kBar1Bytes = []() -> size_t {
        char const* env = std::getenv("UK_BAR1_WINDOW_MB");
        return env ? std::stoull(env) * 1024 * 1024 : 0;
      }();
      if (!(c.flags & kCmdFlagCopySignal) && c.put_path == PutPath::Device &&
          kBar1Bytes > 0 &&
          c.dst_off + c.bytes > kBar1Bytes) {
        // Move the tentative charge from Device to IPC (reroute).
        tpt_metrics_[static_cast<size_t>(c.dst_peer)].device.inflight.fetch_sub(
            1, std::memory_order_relaxed);
        tpt_metrics_[static_cast<size_t>(c.dst_peer)].ipc.inflight.fetch_add(
            1, std::memory_order_relaxed);
        c.put_path = PutPath::Ipc;
      }
    }

    if (path_counters_enabled_ && c.kind == ExecOpKind::Put &&
        c.dst_peer != ~0u) {
      switch (c.put_path) {
        case PutPath::Device:
          put_path_device_.fetch_add(1, std::memory_order_relaxed);
          break;
        case PutPath::Ipc:
          put_path_ipc_.fetch_add(1, std::memory_order_relaxed);
          break;
        case PutPath::Rdma:
          put_path_rdma_.fetch_add(1, std::memory_order_relaxed);
          break;
        default:
          break;
      }
    }

    // PutSignal fusion: a partner Signal rides this Put when the path
    // supports it; the Signal op then completes locally once it becomes
    // ready. The decision is re-derived every cycle, and the per-run
    // fused_sig_cnt is bumped only when the fused put is actually
    // accepted by the backend (see the submission loops).
    //
    // Groups (grp > 1): every put of the group must fuse — the receiver
    // counts grp arrivals. Fused channels split into two matching layers:
    // device-kernel and IPC shm-ring writes share the 64-bit tag map,
    // RDMA immediates use per-peer FIFO. A group never straddles the two
    // layers because the fusion predicate is deterministic per peer
    // within a run: same-host puts reroute to IPC (always fusable) when
    // their picked path cannot fuse, and remote groups fall back to a
    // standalone Signal when RDMA cannot fuse — mirrored by the
    // receiver's wait shaping below.
    if (c.kind == ExecOpKind::Put && c.dst_peer != ~0u && owned_comm_ &&
        !run.plan->put_to_sig.empty()) {
      int32_t sig_idx = run.plan->put_to_sig[idx];
      if (sig_idx >= 0) {
        uint16_t grp = run.plan->sig_group_size[sig_idx];
        uint64_t salted_sig_tag =
            salt_tag(run.plan->tiled.ops[sig_idx].tag, run.tag_epoch);
        bool fuse = false;
        if (c.put_path == PutPath::Device) {
          fuse = device_be_->can_fuse_put_signal(static_cast<int>(c.dst_peer));
          if (!fuse && grp > 1 &&
              owned_comm_->same_host(static_cast<int>(c.dst_peer))) {
            // Device fusion unavailable: reroute to IPC so the group
            // still fully fuses (move the tentative charge).
            tpt_metrics_[static_cast<size_t>(c.dst_peer)]
                .device.inflight.fetch_sub(1, std::memory_order_relaxed);
            tpt_metrics_[static_cast<size_t>(c.dst_peer)]
                .ipc.inflight.fetch_add(1, std::memory_order_relaxed);
            c.put_path = PutPath::Ipc;
          }
        }
        if (!fuse && c.put_path != PutPath::Device) {
          auto tpt_kind = (c.put_path == PutPath::Rdma)
                              ? Transport::PeerTransportKind::Rdma
                              : Transport::PeerTransportKind::Ipc;
          // RDMA fusion sends the unsalted tag as the 32-bit
          // write-with-imm immediate; the receiver mirrors the decision
          // via kCmdFlagImmWait (see rdma_imm_fusion_active).
          fuse =
              (tpt_kind == Transport::PeerTransportKind::Rdma)
                  ? rdma_imm_fusion_active(owned_comm_.get(),
                                           static_cast<int>(c.dst_peer),
                                           run.plan->tiled.ops[sig_idx].tag)
                  : owned_comm_->can_fuse_put_signal(
                        static_cast<int>(c.dst_peer), tpt_kind);
          if (!fuse && grp > 1 && c.put_path == PutPath::Rdma &&
              owned_comm_->same_host(static_cast<int>(c.dst_peer))) {
            // Same-host group put the RDMA gate rejected: reroute to IPC
            // so the group still fully fuses (mirrors the receiver's
            // counted wait, which keys on IPC fusability).
            tpt_metrics_[static_cast<size_t>(c.dst_peer)]
                .rdma.inflight.fetch_sub(1, std::memory_order_relaxed);
            tpt_metrics_[static_cast<size_t>(c.dst_peer)]
                .ipc.inflight.fetch_add(1, std::memory_order_relaxed);
            c.put_path = PutPath::Ipc;
            fuse = owned_comm_->can_fuse_put_signal(
                static_cast<int>(c.dst_peer), Transport::PeerTransportKind::Ipc);
          }
        }
        if (fuse) {
          c.flags |= kCmdFlagPutSignal;
          c.tag = salted_sig_tag;
        }
      }
    }

    if (c.kind == ExecOpKind::Signal || c.kind == ExecOpKind::WaitSignal) {
      // Fused group accounting: a Signal whose group's puts were ALL
      // accepted WITH the fuse flag completes locally — no backend
      // dispatch. If the group cannot (fully) fuse, it must go
      // standalone. Crucially, do not judge until every group put has an
      // acceptance decision: the fused count is bumped at put ACCEPTANCE
      // (in the dev/tpt batches, which run AFTER this ready loop in the
      // same cycle), so a point-in-time check would dispatch a
      // standalone signal for a put that fuses moments later — the peer
      // then sees the tag twice (duplicate arrival, poisoned counts).
      if (c.kind == ExecOpKind::Signal && !run.plan->sig_group_size.empty() &&
          run.plan->sig_group_size[idx] > 0) {
        uint16_t grp = run.plan->sig_group_size[idx];
        if (run.fused_sig_cnt[idx] == grp) {
          run.submitted[idx] = 1;
          complete_op_local(run, idx);
          ++run.sig_local;
          ++sig_dispatched;
          continue;
        }
        if (run.accepted_sig_cnt[idx] < grp) {
          // Group puts not all accepted yet: re-evaluate next cycle.
          run.deferred_sig.push_back(idx);
          continue;
        }
        // All group puts accepted but not all fused: standalone below.
      }
      // Counted/imm wait shaping — the exact mirror of the sender's
      // fusion decision above (same predicates, same order):
      // - RDMA-fused groups (any size): each put's tag rides the 32-bit
      //   write-with-imm immediate, so the wait matches immediates
      //   per-peer FIFO (kCmdFlagImmWait, UNSALTED tag), counting one
      //   imm per group put.
      // - Same-host IPC-fusable groups: one shm-ring arrival per tile,
      //   counted tag-map wait on the salted tag.
      // - Otherwise: one standalone 64-bit signal (signal QP / shm
      //   ring), plain map wait.
      if (c.kind == ExecOpKind::WaitSignal && owned_comm_ &&
          !run.plan->wait_group_size.empty() &&
          run.plan->wait_group_size[idx] > 0) {
        uint16_t const grp = run.plan->wait_group_size[idx];
        uint64_t const unsalted_tag = run.plan->tiled.ops[idx].tag;
        if (rdma_imm_fusion_active(owned_comm_.get(),
                                   static_cast<int>(c.src_peer),
                                   unsalted_tag)) {
          c.wait_count = grp;
          c.flags |= kCmdFlagImmWait;
          c.tag = unsalted_tag;
        } else if (grp > 1 &&
                   owned_comm_->can_fuse_put_signal(
                       static_cast<int>(c.src_peer),
                       Transport::PeerTransportKind::Ipc)) {
          c.wait_count = grp;
        }
      }
      // Per-run in-flight cap for WaitSignals (UK_CCL_SIG_INFLIGHT_CAP,
      // default 4096): a WaitSignal holds its signal-backend slot until
      // the peer's data lands, so an unbounded first wave of waits can
      // occupy every slot and starve the Signals that would unblock
      // them (the 128M in-place stall noted in signal_backend.h). Only
      // WaitSignals are throttled — Signals NEVER defer on this cap,
      // and this cycle's data puts still go out below: they produce the
      // arrivals these waits are for (see the 256M deadlock note below).
      static const uint32_t kSigInflightCap = []() {
        char const* env = std::getenv("UK_CCL_SIG_INFLIGHT_CAP");
        return env ? static_cast<uint32_t>(std::stoul(env)) : 4096u;
      }();
      if (c.kind == ExecOpKind::WaitSignal &&
          run.sig_inflight.load(std::memory_order_relaxed) >=
              kSigInflightCap) {
        run.deferred_sig.push_back(idx);
        continue;
      }
      uint32_t be_idx = signal_be_->reserve_slot();
      if (be_idx != BatchBackend::kInvalidBeIdx) {
        // Ring full (table wrapped onto an unclaimed slot): defer the op
        // instead of blocking in write(). The batched dev/tpt puts below
        // must still go out this cycle — they produce the arrivals these
        // signals wait for, so spinning here deadlocks both ranks (seen
        // at 256M: 4096 initially-ready WaitSignals > 2048 slots).
        if (sig_slots_.occupied(be_idx)) {
          run.deferred_sig.push_back(idx);
          continue;
        }
        // Reserve-then-enqueue: publish the slot BEFORE the op can
        // complete (IPC signal sends complete synchronously).
        sig_slots_.write(be_idx, &run, idx, PutPath::None, stop_);
        if (signal_be_->do_enqueue_reserved(c, be_idx)) {
          run.be_slots.emplace_back(2, be_idx);
          run.inflight_ops.fetch_add(1, std::memory_order_release);
          if (c.kind == ExecOpKind::WaitSignal)
            run.sig_inflight.fetch_add(1, std::memory_order_release);
          run.submitted[idx] = 1;
          if (c.kind == ExecOpKind::Signal) ++run.sig_standalone;
          ++sig_dispatched;
        } else {
          sig_slots_.release(be_idx);
          run.deferred_sig.push_back(idx);
        }
      } else {
        // Backend without reserve support: enqueue first, then publish
        // the slot. Only safe when completions never arrive
        // synchronously during do_enqueue.
        uint32_t gen_idx = 0;
        if (signal_be_->do_enqueue(&c, 1, &gen_idx) == 1) {
          sig_slots_.write(gen_idx, &run, idx, PutPath::None, stop_);
          run.be_slots.emplace_back(2, gen_idx);
          run.inflight_ops.fetch_add(1, std::memory_order_release);
          if (c.kind == ExecOpKind::WaitSignal)
            run.sig_inflight.fetch_add(1, std::memory_order_release);
          run.submitted[idx] = 1;
          ++sig_dispatched;
        } else {
          run.deferred_sig.push_back(idx);
        }
      }
      continue;
    }

    if (c.kind == ExecOpKind::Put && c.dst_peer != ~0u &&
        c.put_path != PutPath::Device) {
      run.tpt_cmds.push_back(c);
      tpt_idx.push_back(idx);
    } else {
      run.dev_cmds.push_back(c);
      dev_idx.push_back(idx);
    }
  }

  // Submit device batch: reserve be_idx range → publish all slots →
  // submit (two-phase). A completion can therefore never arrive before
  // its slot is published, so the drain side's try_claim never spins on
  // this path.
  size_t dev_dispatched = 0;
  if (!run.dev_cmds.empty()) {
    size_t m = run.dev_cmds.size();
    // Never reserve more be_idx than the slot table holds: the two-phase
    // below writes ALL slots before submitting ANY op, so a wrap inside
    // one batch would block write() on a slot whose op has not been
    // submitted yet — a completion that can never arrive (deadlock seen
    // at 64M: batch > 512, write(be_idx=512) stuck on unsubmitted
    // be_idx=0). Ops beyond the cap stay deferred for the next cycle.
    size_t capped = std::min(m, dev_slots_.capacity());
    be_idx_scratch_.resize(capped);
    size_t reserved = device_be_->reserve_slots(be_idx_scratch_.data(), capped);
    if (reserved > 0) {
      for (size_t i = 0; i < reserved; ++i)
        dev_slots_.write(be_idx_scratch_[i], &run, dev_idx[i], PutPath::None,
                         stop_);
      size_t ok = device_be_->do_enqueue_reserved_batch(
          run.dev_cmds.data(), be_idx_scratch_.data(), reserved);
      run.inflight_ops.fetch_add(ok, std::memory_order_release);
      for (size_t i = 0; i < ok; ++i) {
        run.be_slots.emplace_back(0, be_idx_scratch_[i]);
        run.submitted[dev_idx[i]] = 1;
        int32_t sg = run.plan->put_to_sig[dev_idx[i]];
        if (sg >= 0) ++run.accepted_sig_cnt[sg];
        if (run.dev_cmds[i].flags & kCmdFlagPutSignal)
          ++run.fused_sig_cnt[sg];
      }
      // Roll back slots whose submission failed, defer the rest
      // (releasing their tentative inflight charges).
      for (size_t i = ok; i < reserved; ++i)
        dev_slots_.release(be_idx_scratch_[i]);
      for (size_t i = ok; i < m; ++i) {
        if (run.dev_cmds[i].dst_peer != ~0u)
          release_put_inflight(static_cast<int>(run.dev_cmds[i].dst_peer),
                               run.dev_cmds[i].put_path);
        run.deferred_dev.push_back(dev_idx[i]);
      }
      dev_dispatched = ok;
    } else {
      // Backend without reserve support: submit first, publish after.
      // try_claim spins for the rare completion that beats publication.
      size_t ok = device_be_->do_enqueue(run.dev_cmds.data(), m,
                                         be_idx_scratch_.data());
      run.inflight_ops.fetch_add(ok, std::memory_order_release);
      for (size_t i = 0; i < ok; ++i) {
        dev_slots_.write(be_idx_scratch_[i], &run, dev_idx[i], PutPath::None,
                         stop_);
        run.be_slots.emplace_back(0, be_idx_scratch_[i]);
        run.submitted[dev_idx[i]] = 1;
        int32_t sg = run.plan->put_to_sig[dev_idx[i]];
        if (sg >= 0) ++run.accepted_sig_cnt[sg];
        if (run.dev_cmds[i].flags & kCmdFlagPutSignal)
          ++run.fused_sig_cnt[sg];
      }
      for (size_t i = ok; i < m; ++i) {
        if (run.dev_cmds[i].dst_peer != ~0u)
          release_put_inflight(static_cast<int>(run.dev_cmds[i].dst_peer),
                               run.dev_cmds[i].put_path);
        run.deferred_dev.push_back(dev_idx[i]);
      }
      dev_dispatched = ok;
    }
  }

  // Submit transport batch (same two-phase scheme as device above).
  size_t tpt_dispatched = 0;
  if (!run.tpt_cmds.empty()) {
    size_t m = run.tpt_cmds.size();
    // Same wrap-deadlock guard as the device batch above.
    size_t capped = std::min(m, tpt_slots_.capacity());
    be_idx_scratch_.resize(capped);
    size_t reserved = tpt_be_->reserve_slots(be_idx_scratch_.data(), capped);
    if (reserved > 0) {
      for (size_t i = 0; i < reserved; ++i)
        tpt_slots_.write(be_idx_scratch_[i], &run, tpt_idx[i],
                         run.tpt_cmds[i].put_path, stop_);
      size_t ok = tpt_be_->do_enqueue_reserved_batch(
          run.tpt_cmds.data(), be_idx_scratch_.data(), reserved);
      run.inflight_ops.fetch_add(ok, std::memory_order_release);
      for (size_t i = 0; i < ok; ++i) {
        run.be_slots.emplace_back(1, be_idx_scratch_[i]);
        run.submitted[tpt_idx[i]] = 1;
        // The fused put was accepted: count it toward its signal group;
        // the Signal op completes locally once the whole group is out.
        int32_t sg = run.plan->put_to_sig[tpt_idx[i]];
        if (sg >= 0) ++run.accepted_sig_cnt[sg];
        if (run.tpt_cmds[i].flags & kCmdFlagPutSignal)
          ++run.fused_sig_cnt[sg];
      }
      for (size_t i = ok; i < reserved; ++i)
        tpt_slots_.release(be_idx_scratch_[i]);
      for (size_t i = ok; i < m; ++i) {
        release_put_inflight(static_cast<int>(run.tpt_cmds[i].dst_peer),
                             run.tpt_cmds[i].put_path);
        run.deferred_tpt.push_back(tpt_idx[i]);
      }
      tpt_dispatched = ok;
    } else {
      size_t ok =
          tpt_be_->do_enqueue(run.tpt_cmds.data(), m, be_idx_scratch_.data());
      run.inflight_ops.fetch_add(ok, std::memory_order_release);
      for (size_t i = 0; i < ok; ++i) {
        tpt_slots_.write(be_idx_scratch_[i], &run, tpt_idx[i],
                         run.tpt_cmds[i].put_path, stop_);
        run.be_slots.emplace_back(1, be_idx_scratch_[i]);
        run.submitted[tpt_idx[i]] = 1;
        int32_t sg = run.plan->put_to_sig[tpt_idx[i]];
        if (sg >= 0) ++run.accepted_sig_cnt[sg];
        if (run.tpt_cmds[i].flags & kCmdFlagPutSignal)
          ++run.fused_sig_cnt[sg];
      }
      for (size_t i = ok; i < m; ++i) {
        release_put_inflight(static_cast<int>(run.tpt_cmds[i].dst_peer),
                             run.tpt_cmds[i].put_path);
        run.deferred_tpt.push_back(tpt_idx[i]);
      }
      tpt_dispatched = ok;
    }
  }

  UK_DBG(UK_DBG_LVL_ALL, "[enqueue r%d] dispatched: dev=%zu tpt=%zu sig=%zu",
         rank_or_neg1(), dev_dispatched, tpt_dispatched, sig_dispatched);
  if (HostProf::enabled())
    HostProf::enq_ops.fetch_add(dev_dispatched + tpt_dispatched + sig_dispatched,
                                std::memory_order_relaxed);

  run.ready.clear();
}

void SprayExecutor::enqueue_loop() {
  // Run watchdog: fail a Running run whose done_count has not advanced
  // for this long (0 = disabled). Turns silent deadlocks (lost signal,
  // rejected submission, backend stall) into a loud wait() failure.
  static const std::chrono::milliseconds kRunWatchdogMs = []() {
    char const* env = std::getenv("UK_CCL_RUN_WATCHDOG_MS");
    return std::chrono::milliseconds(env ? std::stol(env) : 30000);
  }();
  while (!stop_) {
    // Snapshot running runs under the global lock, then process them
    // lock-free — collect_ready/enqueue_to_ring do backend submission
    // work that must not serialize submit/poll/wait behind runs_mutex_.
    std::vector<std::shared_ptr<SprayRun>> snapshot;
    {
      std::lock_guard lock(runs_mutex_);
      snapshot.reserve(runs_.size());
      for (auto& [h, run] : runs_) {
        if (run->status.load(std::memory_order_acquire) ==
            CollectiveOpStatus::Running)
          snapshot.push_back(run);
      }
    }
    auto const now = std::chrono::steady_clock::now();// SIGUSR2 dump: print every running run's state, then continue.
    if (g_dump_all_requested.exchange(false, std::memory_order_relaxed)) {
      if (snapshot.empty())
        std::fprintf(stderr, "[dump r%d] usr2: no running runs\n",
                     rank_or_neg1());
      for (auto& run : snapshot) {
        std::lock_guard rlock(run->mtx);
        dump_run_state(run.get(), "usr2:");
      }
      if (owned_comm_) owned_comm_->dump_signal_state();
    }
    for (auto& run : snapshot) {
      std::lock_guard rlock(run->mtx);

      // Input dependency gate: skip the entire run if the user's stream
      // hasn't reached the recorded event yet (cudaEventQuery is
      // non-blocking). Never block here: the enqueue thread must stay
      // available to dispatch ops that produce the very arrivals the
      // gated run (and its peers) wait for.
      if (run->input_ready &&
          gpuEventQuery(run->input_ready) == gpuErrorNotReady) {
        continue;
      }
      if (run->input_ready) {
        event_pool_release(run->input_ready);
        run->input_ready = nullptr;
      }

      // Watchdog: once the input gate has passed, the run must keep
      // making completion progress. Only enqueue_loop touches the
      // watchdog fields, so no atomics are needed beyond done_count.
      size_t dn = run->done_count.load(std::memory_order_acquire);
      if (dn != run->watchdog_done) {
        run->watchdog_done = dn;
        run->watchdog_ts = now;
      } else if (run->watchdog_ts.time_since_epoch().count() == 0) {
        run->watchdog_ts = now;  // start the clock
      } else if (kRunWatchdogMs.count() > 0 &&
                 now - run->watchdog_ts > kRunWatchdogMs) {
        fail_run(run.get(),
                 "no completion progress for " +
                     std::to_string(kRunWatchdogMs.count()) +
                     " ms (done=" + std::to_string(dn) + "/" +
                     std::to_string(run->plan->tiled.ops.size()) + ")");
        continue;
      }

      collect_ready(*run);
      enqueue_to_ring(*run);
    }
    if (snapshot.empty()) {
      // Nothing running: sleep until submit() publishes a run. The
      // wake flag closes the lost-notify race; the timeout is a safety
      // net for notifiers that do not set it (none today).
      std::unique_lock<std::mutex> lk(wake_mu_);
      wake_cv_.wait_for(lk, std::chrono::microseconds(200),
                        [this] {
                          return stop_.load(std::memory_order_relaxed) ||
                                 wake_pending_;
                        });
      wake_pending_ = false;
    } else {
    }
  }
}

void SprayExecutor::drain_dev_loop() {
  // NOTE: do NOT pin this thread with gpuSetDevice at loop start — a
  // concurrent context attach from a second thread while the enqueue
  // thread is mid-launch hangs the worker kernel launch on this driver
  // (observed locally, GPU idle + launch never returns). do_drain's own
  // save/restore handles the device, and a failed restore is a warning
  // (see DeviceBackend::do_drain), which fixes the B300 256M abort.
  uint32_t be_buf[256];
  BeSlotSnap snap_buf[256];
  int iter = 0;
  static int dbg_count = 0;
  while (!stop_) {
    if (uk_dbg_lvl() >= UK_DBG_LVL_ALL && ++iter % 10000 == 0)
      UK_DBG(UK_DBG_LVL_ALL, "[drain-dev r%d] alive iter=%d", rank_or_neg1(),
             iter);
    size_t n;
    {
      HostProf::Scope hps(HostProf::dev_us);
      n = device_be_->do_drain(be_buf, 256);
    }
    if (HostProf::enabled() && n > 0)
      HostProf::dev_ops.fetch_add(n, std::memory_order_relaxed);
    if (n > 0 && dbg_count < 5) {
      ++dbg_count;
      UK_DBG(UK_DBG_LVL_EXEC,
             "[drain-dev r%d] do_drain returned %zu completions (count=%d)",
             rank_or_neg1(), n, dbg_count);
    }
    if (n == 0) {
      std::this_thread::yield();
      continue;
    }
    // Drain all available batches; finalize_run runs inline in
    // drain_batch, so completion is observed without a runs_ sweep.
    while (n > 0) {
      size_t valid = 0;
      for (size_t i = 0; i < n; ++i) {
        auto snap = dev_slots_.try_claim(be_buf[i], stop_);
        if (!snap.run) {
          if (stop_.load(std::memory_order_relaxed)) return;
          continue;
        }
        trace_op("dev", snap.enqueue_ns, rank_or_neg1());
        snap_buf[valid++] = snap;
      }
      drain_batch(snap_buf, valid, [this](BeSlotSnap& s) {
        auto& op = s.run->plan->tiled.ops[s.op_idx];
        if (op.kind == ExecOpKind::Put && op.dst_peer != ~0u) {
          int peer = static_cast<int>(op.dst_peer);
          if (peer >= 0 && peer < world_size_)
            update_path_metrics(tpt_metrics_[peer].device, s.enqueue_ns);
        }
      });
      {
        HostProf::Scope hps(HostProf::dev_us);
        n = device_be_->do_drain(be_buf, 256);
      }
      if (HostProf::enabled() && n > 0)
        HostProf::dev_ops.fetch_add(n, std::memory_order_relaxed);
    }
  }
}

void SprayExecutor::drain_tpt_loop() {
  uint32_t be_buf[256];
  BeSlotSnap snap_buf[256];
  int iter = 0;
  static int dbg_count = 0;
  while (!stop_) {
    if (uk_dbg_lvl() >= UK_DBG_LVL_ALL && ++iter % 10000 == 0)
      UK_DBG(UK_DBG_LVL_ALL, "[drain-tpt r%d] alive iter=%d", rank_or_neg1(),
             iter);
    size_t nd;
    {
      HostProf::Scope hps(HostProf::tpt_us);
      nd = tpt_be_->do_drain(be_buf, 256);
    }
    if (HostProf::enabled() && nd > 0)
      HostProf::tpt_ops.fetch_add(nd, std::memory_order_relaxed);
    if (nd > 0 && dbg_count < 5) {
      ++dbg_count;
      UK_DBG(UK_DBG_LVL_EXEC,
             "[drain-tpt r%d] do_drain returned %zu completions (count=%d)",
             rank_or_neg1(), nd, dbg_count);
    }
    if (nd == 0) {
      for (int s = 0; s < 16 && !stop_; ++s) machnet_pause();
      std::this_thread::yield();
      continue;
    }
    while (nd > 0) {
      size_t valid = 0;
      for (size_t i = 0; i < nd; ++i) {
        auto snap = tpt_slots_.try_claim(be_buf[i], stop_);
        if (!snap.run) {
          if (stop_.load(std::memory_order_relaxed)) return;
          continue;
        }
        trace_op("put", snap.enqueue_ns, rank_or_neg1());
        snap_buf[valid++] = snap;
      }
      drain_batch(snap_buf, valid, [this](BeSlotSnap& s) {
        if (s.put_path == PutPath::None) return;
        int peer = static_cast<int>(s.run->plan->tiled.ops[s.op_idx].dst_peer);
        if (peer < 0 || peer >= world_size_) return;
        auto& m = (s.put_path == PutPath::Ipc) ? tpt_metrics_[peer].ipc
                                               : tpt_metrics_[peer].rdma;
        update_path_metrics(m, s.enqueue_ns);
      });
      {
        HostProf::Scope hps(HostProf::tpt_us);
        nd = tpt_be_->do_drain(be_buf, 256);
      }
      if (HostProf::enabled() && nd > 0)
        HostProf::tpt_ops.fetch_add(nd, std::memory_order_relaxed);
    }
  }
}

void SprayExecutor::drain_signal_loop() {
  uint32_t be_buf[256];
  BeSlotSnap snap_buf[256];
  int iter = 0;
  static int dbg_count = 0;
  while (!stop_) {
    if (uk_dbg_lvl() >= UK_DBG_LVL_ALL && ++iter % 10000 == 0)
      UK_DBG(UK_DBG_LVL_ALL, "[drain-sig r%d] alive iter=%d", rank_or_neg1(),
             iter);
    size_t ns;
    {
      HostProf::Scope hps(HostProf::sig_us);
      ns = signal_be_->do_drain(be_buf, 256);
    }
    if (HostProf::enabled() && ns > 0)
      HostProf::sig_ops.fetch_add(ns, std::memory_order_relaxed);
    if (ns > 0 && dbg_count < 5) {
      ++dbg_count;
      UK_DBG(UK_DBG_LVL_EXEC,
             "[drain-sig r%d] do_drain returned %zu completions (count=%d)",
             rank_or_neg1(), ns, dbg_count);
    }
    if (ns == 0) {
      // When waits are registered, arrivals are on the critical path:
      // spin on pauses instead of yielding to the scheduler (at 8 ranks
      // a yield can cost 50-200us of signal-drain latency). Yield only
      // when nothing is waiting.
      bool const waits_pending =
          owned_comm_ && owned_comm_->has_pending_signal_waits();
      int const burst = waits_pending ? 512 : 16;
      for (int s = 0; s < burst && !stop_; ++s) machnet_pause();
      if (!waits_pending || stop_.load(std::memory_order_relaxed))
        std::this_thread::yield();
      continue;
    }
    while (ns > 0) {
      size_t valid = 0;
      for (size_t i = 0; i < ns; ++i) {
        auto snap = sig_slots_.try_claim(be_buf[i], stop_);
        if (!snap.run) {
          if (stop_.load(std::memory_order_relaxed)) return;
          continue;
        }
        trace_op("sig", snap.enqueue_ns, rank_or_neg1());
        snap_buf[valid++] = snap;
      }
      drain_batch(snap_buf, valid, [](BeSlotSnap&) {});
      {
        HostProf::Scope hps(HostProf::sig_us);
        ns = signal_be_->do_drain(be_buf, 256);
      }
      if (HostProf::enabled() && ns > 0)
        HostProf::sig_ops.fetch_add(ns, std::memory_order_relaxed);
    }
  }
}

int SprayExecutor::rank_or_neg1() const {
  return owned_comm_ ? owned_comm_->rank() : -1;
}

void SprayExecutor::finalize_run(SprayRun* run) {
  if (run->status.load(std::memory_order_acquire) !=
      CollectiveOpStatus::Running)
    return;
  if (run->done_count.load(std::memory_order_acquire) <
      run->plan->tiled.ops.size())
    return;
  // Exactly-once: only the CAS winner flips status and releases the
  // active-run slot, no matter how many threads observe completion.
  CollectiveOpStatus expected = CollectiveOpStatus::Running;
  if (run->status.compare_exchange_strong(
          expected, CollectiveOpStatus::Completed, std::memory_order_release,
          std::memory_order_relaxed)) {
    active_runs_.fetch_sub(1, std::memory_order_release);
    // Output dependency: publish completion to the mapped done_flag the
    // user stream's WaitValue polls. Monotonic MAX write: several
    // drain/wait threads can finalize different runs concurrently and
    // out of seq order — a plain store could move the flag backwards
    // and strand a GPU WaitValue(GEQ) on an older, larger seq forever
    // (the multi-iteration hang that once disabled this path). The
    // SEQ_CST fence after the CAS loop flushes the winning store so the
    // GPU polling loop sees it (mfence on x86).
    if (run->user_stream) {
      uint64_t cur = __atomic_load_n(done_flag_host_, __ATOMIC_RELAXED);
      while (cur < run->done_seq &&
             !__atomic_compare_exchange_n(done_flag_host_, &cur,
                                          run->done_seq, /*weak=*/true,
                                          __ATOMIC_RELEASE,
                                          __ATOMIC_RELAXED))
        ;
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }
  }
}

void SprayExecutor::dump_run_state(SprayRun* run, char const* why) {
  // Called from enqueue_loop (run->mtx held). Per-kind submission state
  // plus the first few pending ops — the post-mortem for stalls.
  size_t nops = run->plan->tiled.ops.size();
  int sub[4] = {0, 0, 0, 0}, pend[4] = {0, 0, 0, 0};
  static char const* kNames[4] = {"Put", "Reduce", "Signal", "WaitSignal"};
  for (size_t i = 0; i < nops; ++i) {
    int k = static_cast<int>(run->plan->tiled.ops[i].kind);
    if (k < 0 || k > 3) continue;
    if (run->submitted[i])
      ++sub[k];
    else
      ++pend[k];
  }
  std::fprintf(stderr,
               "[dump r%d] %s done=%zu/%zu epoch=%u deferred dev=%zu tpt=%zu "
               "sig=%zu sig_local=%u sig_standalone=%u\n",
               rank_or_neg1(), why, run->done_count.load(), nops,
               run->tag_epoch, run->deferred_dev.size(),
               run->deferred_tpt.size(), run->deferred_sig.size(),
               run->sig_local, run->sig_standalone);
  for (int k = 0; k < 4; ++k)
    std::fprintf(stderr, "[dump r%d] %-10s submitted=%d pending=%d\n",
                 rank_or_neg1(), kNames[k], sub[k], pend[k]);
  int shown_per_kind[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < nops; ++i) {
    if (run->submitted[i]) continue;
    auto const& op = run->plan->tiled.ops[i];
    int k = static_cast<int>(op.kind);
    if (k < 0 || k > 3 || shown_per_kind[k] >= 6) continue;
    ++shown_per_kind[k];
    std::string depinfo;
    for (uint32_t d : op.deps) {
      depinfo += std::to_string(d);
      depinfo += (d < run->submitted.size() && run->submitted[d]) ? "(ok) " : "(X) ";
    }
    std::fprintf(stderr,
                 "[dump r%d] pending op[%zu] kind=%d indegree=%u "
                 "ndeps=%zu tag=%#lx src=%d dst=%d bytes=%zu deps=%s\n",
                 rank_or_neg1(), i, (int)op.kind,
                 i < run->indegree.size() ? run->indegree[i] : 9999u,
                 op.deps.size(), (unsigned long)op.tag, (int)op.src_peer,
                 (int)op.dst_peer, op.bytes, depinfo.c_str());
  }
}

void SprayExecutor::fail_run(SprayRun* run, std::string msg) {
  // Write the error before the CAS: a waiter that observes Failed with
  // acquire ordering is then guaranteed to see the message.
  run->error = std::move(msg);
  CollectiveOpStatus expected = CollectiveOpStatus::Running;
  if (run->status.compare_exchange_strong(
          expected, CollectiveOpStatus::Failed, std::memory_order_release,
          std::memory_order_relaxed)) {
    active_runs_.fetch_sub(1, std::memory_order_release);
    std::fprintf(stderr, "[executor r%d] run failed: %s\n", rank_or_neg1(),
                 run->error.c_str());
    dump_run_state(run, "failed:");
  }
}

size_t SprayExecutor::progress_once() {
  uint32_t be_buf[256];
  BeSlotSnap snap_buf[256];
  size_t total = 0;

  if (device_be_) {
    size_t n = device_be_->do_drain(be_buf, 256);
    size_t valid = 0;
    for (size_t i = 0; i < n; ++i) {
      auto snap = dev_slots_.try_claim(be_buf[i], stop_);
      if (snap.run) snap_buf[valid++] = snap;
    }
    if (valid) {
      drain_batch(snap_buf, valid, [this](BeSlotSnap& s) {
        auto& op = s.run->plan->tiled.ops[s.op_idx];
        if (op.kind == ExecOpKind::Put && op.dst_peer != ~0u) {
          int peer = static_cast<int>(op.dst_peer);
          if (peer >= 0 && peer < world_size_)
            update_path_metrics(tpt_metrics_[peer].device, s.enqueue_ns);
        }
      });
      total += valid;
    }
  }

  if (tpt_be_) {
    size_t n = tpt_be_->do_drain(be_buf, 256);
    size_t valid = 0;
    for (size_t i = 0; i < n; ++i) {
      auto snap = tpt_slots_.try_claim(be_buf[i], stop_);
      if (snap.run) snap_buf[valid++] = snap;
    }
    if (valid) {
      drain_batch(snap_buf, valid, [this](BeSlotSnap& s) {
        if (s.put_path == PutPath::None) return;
        int peer = static_cast<int>(s.run->plan->tiled.ops[s.op_idx].dst_peer);
        if (peer < 0 || peer >= world_size_) return;
        auto& m = (s.put_path == PutPath::Ipc) ? tpt_metrics_[peer].ipc
                                               : tpt_metrics_[peer].rdma;
        update_path_metrics(m, s.enqueue_ns);
      });
      total += valid;
    }
  }

  if (signal_be_) {
    size_t n = signal_be_->do_drain(be_buf, 256);
    size_t valid = 0;
    for (size_t i = 0; i < n; ++i) {
      auto snap = sig_slots_.try_claim(be_buf[i], stop_);
      if (snap.run) snap_buf[valid++] = snap;
    }
    if (valid) {
      drain_batch(snap_buf, valid, [](BeSlotSnap&) {});
      total += valid;
    }
  }

  return total;
}

PutPath SprayExecutor::pick_put_path(int peer) {
  if (!tpt_metrics_ || peer < 0 || peer >= world_size_) {
    return PutPath::Device;
  }
  // Optional forced path for A/B benchmarking, same-host peers only
  // (remote peers are always RDMA). UK_CCL_PUT_PATH=device|ipc|rdma;
  // unset or anything else = normal load balancing.
  static const PutPath forced = []() {
    char const* v = std::getenv("UK_CCL_PUT_PATH");
    if (!v) return PutPath::None;
    std::string s(v);
    if (s == "device") return PutPath::Device;
    if (s == "ipc") return PutPath::Ipc;
    if (s == "rdma") return PutPath::Rdma;
    return PutPath::None;
  }();
  if (forced != PutPath::None && same_host_fn_ &&
      same_host_fn_(owned_comm_.get(), peer)) {
    auto& pfm = tpt_metrics_[peer];
    PathMetrics* fm = (forced == PutPath::Device) ? &pfm.device
                      : (forced == PutPath::Ipc)  ? &pfm.ipc
                                                  : &pfm.rdma;
    fm->inflight.fetch_add(1, std::memory_order_relaxed);
    return forced;
  }
  // Same-host cross-GPU puts go over IPC: the CPU-side DMA path is the
  // fast transport (sliding-window puts; measured 67 GB/s aggregate for
  // 256MB AllGather vs 36 GB/s device and ~2 GB/s RDMA loopback). The
  // old latency-based balancer misrouted same-host puts onto the
  // device/RDMA paths — the sliding window inflates the IPC latency
  // metric (completions burst after a window sync), so IPC lost the
  // comparison and AllGather dropped from 3.8ms to 15.9ms. Device puts
  // remain reachable for local ops (reduce is not routed here) and via
  // UK_CCL_PUT_PATH; remote peers always use RDMA.
  if (!same_host_fn_ || !same_host_fn_(owned_comm_.get(), peer)) {
    tpt_metrics_[peer].rdma.inflight.fetch_add(1, std::memory_order_relaxed);
    return PutPath::Rdma;
  }
  tpt_metrics_[peer].ipc.inflight.fetch_add(1, std::memory_order_relaxed);
  return PutPath::Ipc;
}

void SprayExecutor::release_put_inflight(int peer, PutPath path) {
  if (!tpt_metrics_ || peer < 0 || peer >= world_size_) return;
  auto& pm = tpt_metrics_[peer];
  switch (path) {
    case PutPath::Device:
      pm.device.inflight.fetch_sub(1, std::memory_order_relaxed);
      break;
    case PutPath::Ipc:
      pm.ipc.inflight.fetch_sub(1, std::memory_order_relaxed);
      break;
    case PutPath::Rdma:
      pm.rdma.inflight.fetch_sub(1, std::memory_order_relaxed);
      break;
    default:
      break;
  }
}

}  // namespace CCL
}  // namespace UKernel
