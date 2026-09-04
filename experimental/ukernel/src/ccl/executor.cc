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
#include <cstdlib>
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

// At UK_CCL_DEBUG >= 3: print per-op completion latency (now -
// enqueue_ns) for the first ~40 ops of each kind — isolates the
// put/reduce/signal dependency-chain latency that aggregate HostProf
// buckets blur.
void trace_op(char const* kind, uint64_t enqueue_ns, int rank) {
  if (uk_dbg_lvl() < UK_DBG_LVL_ALL) return;
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

// Block-interleave the per-host rank groups so the ring's cross-node
// edges spread across distinct GPUs/NICs instead of the naive single
// edge. Rank-symmetric: groups are sorted by min rank, so every rank
// derives the same permutation. Falls back to identity when peer
// metadata is not established yet.
static std::vector<int> compute_ring_order(Transport::Communicator* comm,
                                           int nranks) {
  std::vector<int> order;
  std::map<std::string, std::vector<int>> by_host;
  std::vector<std::string> rank_host(static_cast<size_t>(nranks));
  try {
    for (int r = 0; r < nranks; ++r) {
      std::string id = comm->peer_host_id(r);
      if (id.empty()) return order;  // meta not established: identity
      by_host[id].push_back(r);
      rank_host[static_cast<size_t>(r)] = std::move(id);
    }
  } catch (...) {
    return order;  // identity
  }
  std::vector<std::vector<int>> groups;
  for (auto& kv : by_host) groups.push_back(std::move(kv.second));
  std::sort(groups.begin(), groups.end(),
            [](std::vector<int> const& a, std::vector<int> const& b) {
              return a.front() < b.front();
            });
  size_t min_sz = 0;
  if (!groups.empty()) {
    min_sz = groups[0].size();
    for (auto const& g : groups) min_sz = std::min(min_sz, g.size());
  }
  // Experiment knob: override the block-interleave width. The default
  // (min group / 2) spreads the ring's cross-node edges over distinct
  // GPUs/NICs; larger blocks approach identity, smaller blocks make the
  // ring cross more often.
  size_t const block = [&]() {
    char const* v = std::getenv("UK_CCL_RING_BLOCK");
    if (v && *v) {
      long b = std::strtol(v, nullptr, 10);
      if (b > 0) return static_cast<size_t>(b);
    }
    return std::max<size_t>(1, min_sz / 2);
  }();
  size_t nblocks = 0;
  for (auto const& g : groups)
    nblocks = std::max(nblocks, (g.size() + block - 1) / block);
  for (size_t blk = 0; blk < nblocks; ++blk)
    for (auto const& g : groups)
      for (size_t i = blk * block; i < (blk + 1) * block && i < g.size(); ++i)
        order.push_back(g[i]);

  // Every cross-host edge needs an on-fabric NIC on both endpoints;
  // otherwise the ring would fail path setup at submit time. Fall back
  // to the identity order (the known-good default) instead of emitting
  // an order that cannot run.
  for (size_t i = 0; i < order.size(); ++i) {
    int a = order[i];
    int b = order[(i + 1) % order.size()];
    if (rank_host[static_cast<size_t>(a)] ==
        rank_host[static_cast<size_t>(b)])
      continue;
    if (!comm->peer_rdma_capable(a) || !comm->peer_rdma_capable(b)) {
      UK_DBG(UK_DBG_LVL_EXEC,
             "[ring-order r%d] interleave order unusable (rank %d lacks "
             "on-fabric RDMA); falling back to identity",
             comm->rank(), a);
      return {};
    }
  }
  return order;
}

// Apply the opt-in NIC-aware ring order (UK_CCL_RING_INTERLEAVE=1) to a
// config copy. Used by BOTH prepare() and submit() so the peer setup and
// the plan see the same order.
static CollectiveConfig with_ring_order(CollectiveConfig cfg,
                                        Transport::Communicator* comm) {
  static bool const kInterleave = []() {
    char const* v = std::getenv("UK_CCL_RING_INTERLEAVE");
    return v && std::string(v) == "1";
  }();
  if (kInterleave && cfg.ring_order.empty() && cfg.nranks > 1 && comm) {
    cfg.ring_order = compute_ring_order(comm, cfg.nranks);
  }
  return cfg;
}

// Signal-tag epoch salting: plan tags repeat across runs, so fold a
// per-executor run epoch (rank-symmetric) into the high bits to keep
// cross-run matching impossible. RDMA immediates instead carry an
// epoch-encoded value (see encode_imm); standalone signals keep the
// full 64-bit salted tag.
static inline uint64_t salt_tag(uint64_t base, uint32_t epoch) {
  return base | (static_cast<uint64_t>(epoch) << 32);
}

// RDMA write-with-imm payload, unique per (run, tag): unsalted tag in
// the low 20 bits + run epoch in bits 20..31. Both sides derive it from
// (unsalted_tag, tag_epoch), so a skewed run can never match — or
// block — another run's waits.
static inline uint32_t encode_imm(uint64_t unsalted_tag, uint32_t epoch) {
  return (static_cast<uint32_t>(unsalted_tag) & 0xFFFFFu) |
         ((epoch & 0xFFFu) << 20);
}

// Fusion gate: the unsalted tag must fit the imm's 20-bit field;
// oversized tags fall back to a standalone signal on both sides.
static inline bool tag_fits_imm_field(uint64_t tag) {
  return tag <= 0xFFFFFu;
}

// RDMA put-signal predicate, mirrored by the sender and receiver:
// fusable iff the unsalted tag fits the imm field and the RDMA path
// supports it (same-host excluded unless UK_CCL_PUT_PATH=rdma forces
// the path).
static bool rdma_imm_fusion_active(Transport::Communicator* comm, int peer,
                                   uint64_t unsalted_tag) {
  // Decoupled-signal mode is the default: the adapter stripes message
  // chunks across the peer's QPs and sends the data-ready signal
  // separately (see rdma_adapter.cc); waits use the plain signal path,
  // mirrored by the sender. UK_CCL_RDMA_FUSED_PUT=1 restores the legacy
  // write-with-imm fusion.
  static bool const kFusedPut = []() {
    char const* v = std::getenv("UK_CCL_RDMA_FUSED_PUT");
    return v && std::string(v) == "1";
  }();
  if (!kFusedPut) return false;
  static const bool kForceRdma = []() {
    char const* v = std::getenv("UK_CCL_PUT_PATH");
    return v && std::string(v) == "rdma";
  }();
  if (!tag_fits_imm_field(unsalted_tag)) return false;
  if (!kForceRdma && comm->same_host(peer)) return false;
  return comm->can_fuse_put_signal(peer, Transport::PeerTransportKind::Rdma);
}

static Cmd make_cmd(TiledOp const& op, ReductionKind redop, ScalarType dtype,
                    uint32_t input_buf, uint32_t output_buf,
                    uint32_t scratch_buf, uint64_t input_off,
                    uint64_t output_off, uint64_t scr_off,
                    uint32_t tag_epoch) {
  Cmd c{};
  c.kind = op.kind;  // the command's identity is the logical op
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
  c.copy_dst_buf = role_to_buf(op.copy_dst_buf_role, input_buf, output_buf,
                               scratch_buf);
  c.copy_dst_peer = op.copy_dst_peer;
  c.copy_dst_off = op.copy_dst_off;
  c.flag_slot = op.flag_slot;
  c.flag_count = op.flag_count;
  c.redop = (op.kind == LogicalOpKind::Reduce ||
             op.kind == LogicalOpKind::ReducePut ||
             op.kind == LogicalOpKind::ReducePutSignal)
                ? redop
                : ReductionKind::None;
  c.put_path = op.put_path_hint;  // None = auto (pick_put_path below)
  c.tag = salt_tag(op.tag, tag_epoch);
  if (op.kind == LogicalOpKind::Reduce && op.fused_proxy_put_idx >= 0)
    c.flags |= kCmdFlagRdmaFusedProxy;
  if (op.kind == LogicalOpKind::PutSignal) {
    c.tag = op.tag;  // raw signal tag; channel encoding at enqueue time
  }
  if (op.kind == LogicalOpKind::Put && op.flag_slot != ~0u)
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
  add(static_cast<uint64_t>(cfg.channels));
  add(cfg.fuse_reduce_copy ? 1u : 0u);
  add(cfg.fuse_ag_copy ? 1u : 0u);
  add(cfg.device_flags ? 1u : 0u);
  add(inplace ? 1u : 0u);
  add(cfg.ring_order.size());
  for (int r : cfg.ring_order) add(static_cast<uint64_t>(r));
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

bool SprayExecutor::submit_fused_cmd(uint64_t cmd_index, bool first_attempt) {
  if (!fused_proxy_) return false;
  auto const& slot = fused_proxy_->pool().get(cmd_index);
  if (uk_dbg_lvl() >= UK_DBG_LVL_TPT)
    std::fprintf(stderr,
                 "[fused-submit r%d] idx=%llu peer=%d at=%.3fms first=%d\n",
                 rank_or_neg1(), (unsigned long long)cmd_index,
                 static_cast<int>(slot.cmd.dst_peer),
                 static_cast<double>(
                     std::chrono::system_clock::now()
                         .time_since_epoch()
                         .count()) /
                     1e6,
                 first_attempt ? 1 : 0);
  uint32_t be_idx = tpt_be_->reserve_slot();
  auto* run = static_cast<SprayRun*>(slot.run);
  // First-attempt accounting must precede BeSlot publication: wait()'s
  // progress_once() drains tpt completions concurrently, and a drain
  // could otherwise see the completion before these counters are set
  // (duplicate Signal / run UAF). Retries skip this block.
  if (first_attempt && run) {
    run->inflight_ops.fetch_add(1, std::memory_order_release);
    if (slot.op_idx < run->submitted.size())
      __atomic_store_n(&run->submitted[slot.op_idx], 1, __ATOMIC_RELAXED);
  }
  // Publish the BeSlot BEFORE enqueueing so a fast completion can never
  // beat the slot publication (same two-phase rule as normal submission).
  tpt_slots_.write(be_idx, run, slot.op_idx, slot.put_path, stop_);
  if (!tpt_be_->do_enqueue_reserved(slot.cmd, be_idx)) {
    tpt_slots_.release(be_idx);
    return false;
  }
  return true;
}

// Build the immutable plan: tiling/lowering plus the successor CSR and
// the initial scheduling state that submit() would otherwise rebuild
// on every call.
static std::shared_ptr<CollPlan const> build_plan(
    CollectiveConfig const& cfg, bool inplace,
    std::function<bool(int)> same_host = nullptr) {
  auto plan = std::make_shared<CollPlan>();
  plan->tiled = build_tiled(cfg, inplace, same_host);
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

  path_counters_enabled_ = uk_dbg_lvl() >= UK_DBG_LVL_EXEC;
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
  // comment above). Installed only when diagnostics are enabled; one
  // handler per process is enough.
  static std::once_flag sig_once;
  std::call_once(sig_once, [] {
    if (uk_dbg_lvl() >= UK_DBG_LVL_EXEC)
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

  // Defensive budget for the whole prepare phase (peer setup, MR
  // registration, buffer resolution). A wedged CUDA driver cannot be
  // interrupted from this thread, but every blocking step that does
  // return (OOB exchanges, buffer resolves, transport setup) is bounded
  // so a slow/hung path fails loudly with a named stage instead of
  // silently hanging or cascading into the run. The driver-ioctl hang
  // itself is addressed at the source (fused reduce alignment fix);
  // this is the net for everything else.
  static std::chrono::milliseconds const kPrepareTimeout = [] {
    char const* v = std::getenv("UK_CCL_PREPARE_TIMEOUT_MS");
    return std::chrono::milliseconds(
        v ? std::max<long>(1000, std::atol(v)) : 60000);
  }();
  auto const prepare_deadline =
      std::chrono::steady_clock::now() + kPrepareTimeout;
  auto check_prepare_deadline = [&](char const* stage) {
    if (std::chrono::steady_clock::now() > prepare_deadline) {
      throw std::runtime_error(std::string(
          "prepare timeout at stage '") + stage + "' after " +
          std::to_string(kPrepareTimeout.count()) +
          "ms (UK_CCL_PREPARE_TIMEOUT_MS; rank " +
          std::to_string(cfg.rank) + ")");
    }
  };

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
  add(cfg.ring_order.size());
  for (int r : cfg.ring_order) add(static_cast<uint64_t>(r));
  CollAlgo algo;
  auto ait = prepare_algo_cache_.find(skey);
  if (ait != prepare_algo_cache_.end()) {
    algo = ait->second;
  } else {
    algo = build_coll_algo(with_ring_order(cfg, owned_comm_.get()), inplace);
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
  check_prepare_deadline("peer_setup start");
  peer_setup_fn_(owned_comm_.get(), cfg.rank, peers);
  check_prepare_deadline("peer_setup done");
  UK_DBG(UK_DBG_LVL_EXEC,
         "[prepare r%d] peer_setup_fn done  -> re_register_all_mrs", cfg.rank);
  owned_comm_->re_register_all_mrs();
  check_prepare_deadline("re_register_all_mrs");
  UK_DBG(UK_DBG_LVL_EXEC,
         "[prepare r%d] re_register_all_mrs done  -> register bufs", cfg.rank);
  prepared_peers_.insert(peers.begin(), peers.end());
  prepared_ = true;

  // Register and resolve user buffers.
  uint32_t in_id = get_or_register_buf(input, cfg.input_bytes, nullptr, "prep-in");
  check_prepare_deadline("register input buf");
  UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] in_id=%u registered  -> resolve bufs",
         cfg.rank, in_id);
  uint32_t out_id =
      get_or_register_buf(output, cfg.output_bytes, nullptr, "prep-out");
  check_prepare_deadline("register output buf");
  UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] out_id=%u registered", cfg.rank,
         out_id);
  for (int p : peers) {
    if (in_id && resolve_buf_fn_) {
      check_prepare_deadline("resolve input buf");
      UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] resolve in_id=%u from peer %d ...",
             cfg.rank, in_id, p);
      resolve_buf_fn_(owned_comm_.get(), p, world_size_, in_id);
      check_prepare_deadline("resolve input buf done");
      UK_DBG(UK_DBG_LVL_EXEC,
             "[prepare r%d] resolve in_id=%u from peer %d done", cfg.rank,
             in_id, p);
    }
    if (out_id && out_id != in_id && resolve_buf_fn_) {
      check_prepare_deadline("resolve output buf");
      UK_DBG(UK_DBG_LVL_EXEC,
             "[prepare r%d] resolve out_id=%u from peer %d ...", cfg.rank,
             out_id, p);
      resolve_buf_fn_(owned_comm_.get(), p, world_size_, out_id);
      check_prepare_deadline("resolve output buf done");
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
      check_prepare_deadline("resolve scratch buf");
      UK_DBG(UK_DBG_LVL_EXEC,
             "[prepare r%d] resolve scr_id=%u from peer %d ...", cfg.rank,
             scr_id, p);
      resolve_buf_fn_(owned_comm_.get(), p, world_size_, scr_id);
      check_prepare_deadline("resolve scratch buf done");
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

  // Opt-in NIC-aware ring order, shared with prepare().
  CollectiveConfig cfg_eff = with_ring_order(cfg, owned_comm_.get());

  if (owned_comm_ && !prepared_) {
    // Check all peers needed by this algorithm are prepared.
    CollAlgo algo = build_coll_algo(cfg_eff, cfg_inplace(cfg, input, output));
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
    std::string key = plan_key(cfg_eff, cfg_inplace(cfg, input, output));
    auto it = plan_cache_.find(key);
    if (it != plan_cache_.end()) {
      plan = it->second;
    } else {
      if (plan_cache_.size() >= kMaxCachedPlans) plan_cache_.clear();
      plan = build_plan(
          cfg_eff, cfg_inplace(cfg, input, output),
          [this](int peer) {
            return same_host_fn_ && same_host_fn_(owned_comm_.get(), peer);
          });
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
        if (op.kind == LogicalOpKind::Put &&
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
    // submitted is also written by the fused-proxy thread (drain_tpt_loop)
    // for synthetic puts, so reads here must be atomic.
    if (!__atomic_load_n(&run.submitted[op], __ATOMIC_RELAXED))
      run.ready.push_back(op);
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
    for (uint32_t d : run.deferred_dev)
      if (!__atomic_load_n(&run.submitted[d], __ATOMIC_RELAXED))
        run.ready.push_back(d);
    for (uint32_t d : run.deferred_tpt)
      if (!__atomic_load_n(&run.submitted[d], __ATOMIC_RELAXED))
        run.ready.push_back(d);
    for (uint32_t d : run.deferred_sig)
      if (!__atomic_load_n(&run.submitted[d], __ATOMIC_RELAXED))
        run.ready.push_back(d);

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

  for (uint32_t idx : run.ready) {
    Cmd c = make_cmd(run.plan->tiled.ops[idx], run.plan->tiled.reduction,
                     run.plan->tiled.dtype, run.input_buf_id,
                     run.output_buf_id, run.scratch_buf_id,
                     run.input_base_off, run.output_base_off,
                     run.scratch_base_off, run.tag_epoch);
    if (c.kind == LogicalOpKind::Reduce &&
        (c.flags & kCmdFlagRdmaFusedProxy)) {
      // Host proxy: allocate the synthetic Put's Cmd in the fused pool
      // now. The device task writes this index into the D2H ring after
      // reducing to local dst; the proxy then posts the put — over IPC
      // (CE, host-acknowledged) for same-host peers and RDMA for remote
      // ones. The device-direct LD/ST peer copy is not arrival-ordered
      // on PCIe, so same-host fused copies must use the host path too.
      int32_t pop_idx = run.plan->tiled.ops[idx].fused_proxy_put_idx;
      if (pop_idx >= 0 && fused_proxy_) {
        auto const& pop = run.plan->tiled.ops[static_cast<size_t>(pop_idx)];
        Cmd put = make_cmd(pop, run.plan->tiled.reduction,
                           run.plan->tiled.dtype, run.input_buf_id,
                           run.output_buf_id, run.scratch_buf_id,
                           run.input_base_off, run.output_base_off,
                           run.scratch_base_off, run.tag_epoch);
        // The linked node is a PutSignal (lowered): it already carries
        // the data-ready tag. In fused mode the proxy put travels as an
        // RDMA write-with-imm carrying the epoch-encoded tag; in
        // decoupled mode it is sent as plain writes plus a standalone
        // signal carrying the salted tag. The receiver's wait registers
        // the matching value.
        put.tag =
            rdma_imm_fusion_active(owned_comm_.get(),
                                   static_cast<int>(put.dst_peer), pop.tag)
                ? encode_imm(pop.tag, run.tag_epoch)
                : salt_tag(pop.tag, run.tag_epoch);
        PutPath const put_path =
            same_host_fn_ &&
                    same_host_fn_(owned_comm_.get(),
                                  static_cast<int>(put.dst_peer))
                ? PutPath::Ipc
                : PutPath::Rdma;
        put.put_path = put_path;
        uint64_t pool_idx = fused_proxy_->pool().alloc(
            put, &run, static_cast<uint32_t>(pop_idx), put_path);
        if (pool_idx != UINT64_MAX) {
          c.rdma_fused_ring = fused_proxy_->ring().device_handle();
          c.rdma_fused_cmd_index = pool_idx;
        } else {
          // Pool exhausted: defer the Reduce instead of dispatching a
          // device task with a null ring handle / invalid index. The
          // deferred op is retried by the next enqueue cycle.
          run.deferred_dev.push_back(idx);
          continue;
        }
      }
    }
    if ((c.kind == LogicalOpKind::Put ||
         c.kind == LogicalOpKind::PutSignal) &&
        c.dst_peer != ~0u) {
      if (c.flags & kCmdFlagCopySignal) {
        // Fused AG copy: must run on the device backend (the task writes
        // the completion flag); never route to CE/RDMA.
        c.put_path = PutPath::Device;
      } else {
        // A builder-set hint (RS hybrid halves) wins; otherwise auto.
        if (c.put_path == PutPath::None)
          c.put_path = pick_put_path(static_cast<int>(c.dst_peer));
        // Remote peers are only reachable over RDMA: a Device/IPC hint
        // (e.g. an alltoall hybrid half) must not leak to a cross-node
        // peer — the CE/device path has no remote buffer to write.
        if (c.dst_peer != ~0u && same_host_fn_ &&
            !same_host_fn_(owned_comm_.get(), static_cast<int>(c.dst_peer)) &&
            c.put_path != PutPath::Rdma) {
          c.put_path = PutPath::Rdma;
        }
      }
      UK_DBG(UK_DBG_LVL_EXEC, "[pick r%d] op[%u] peer=%u -> path=%d",
             rank_or_neg1(), idx, c.dst_peer, (int)c.put_path);
      // PutSignal tag encoding: RDMA → epoch-encoded imm, else salted
      // 64-bit tag; the receiver mirrors this in its wait shaping.
      if (c.kind == LogicalOpKind::PutSignal && owned_comm_ &&
          !run.plan->tiled.ops[idx].proxy_posted) {
        uint64_t const unsalted = run.plan->tiled.ops[idx].tag;
        if (c.put_path == PutPath::Rdma &&
            rdma_imm_fusion_active(owned_comm_.get(),
                                   static_cast<int>(c.dst_peer), unsalted)) {
          c.tag = encode_imm(unsalted, run.tag_epoch);
        } else {
          c.tag = salt_tag(unsalted, run.tag_epoch);
        }
      }
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

    if (path_counters_enabled_ &&
        (c.kind == LogicalOpKind::Put ||
         c.kind == LogicalOpKind::PutSignal) &&
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

    if (c.kind == LogicalOpKind::Signal || c.kind == LogicalOpKind::Wait) {
      // Wait channel shaping, mirroring the sender's PutSignal encoding:
      // RDMA → imm value wait; same-host fusable groups → counted
      // tag-map wait; otherwise plain 64-bit signal wait.
      if (c.kind == LogicalOpKind::Wait && owned_comm_) {
        uint16_t const grp = run.plan->tiled.ops[idx].wait_count;
        uint64_t const unsalted_tag = run.plan->tiled.ops[idx].tag;
        if (rdma_imm_fusion_active(owned_comm_.get(),
                                   static_cast<int>(c.src_peer),
                                   unsalted_tag)) {
          c.wait_count = grp;
          c.flags |= kCmdFlagImmWait;
          c.tag = encode_imm(unsalted_tag, run.tag_epoch);
        } else if (grp > 1) {
          // Plain signal path: a G-tile group sends one standalone
          // signal per tile (RDMA decoupled mode or IPC), so the wait
          // counts G arrivals of the same tag.
          c.wait_count = grp;
        }
      }
      // Cap in-flight WaitSignals so an early wave of waits cannot
      // occupy every signal slot and starve the Signals that unblock
      // them; Signals never defer on this cap.
      static const uint32_t kSigInflightCap = []() {
        char const* env = std::getenv("UK_CCL_SIG_INFLIGHT_CAP");
        return env ? static_cast<uint32_t>(std::stoul(env)) : 4096u;
      }();
      if (c.kind == LogicalOpKind::Wait &&
          run.sig_inflight.load(std::memory_order_relaxed) >=
              kSigInflightCap) {
        run.deferred_sig.push_back(idx);
        continue;
      }
      uint32_t be_idx = signal_be_->reserve_slot();
      if (be_idx != BatchBackend::kInvalidBeIdx) {
        // Slot wrapped onto an unclaimed entry: defer; the batched
        // puts below must still go out this cycle or the waits never
        // arrive.
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
          if (c.kind == LogicalOpKind::Wait)
            run.sig_inflight.fetch_add(1, std::memory_order_release);
          run.submitted[idx] = 1;
          if (c.kind == LogicalOpKind::Signal) ++run.sig_standalone;
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
          if (c.kind == LogicalOpKind::Wait)
            run.sig_inflight.fetch_add(1, std::memory_order_release);
          run.submitted[idx] = 1;
          ++sig_dispatched;
        } else {
          run.deferred_sig.push_back(idx);
        }
      }
      continue;
    }

    if ((c.kind == LogicalOpKind::Put ||
         c.kind == LogicalOpKind::PutSignal) &&
        run.plan->tiled.ops[idx].proxy_posted) {
      // Proxy-posted PutSignal: submitted by the D2H proxy through
      // submit_fused_cmd(), not by the normal enqueue path.
      run.deferred_tpt.push_back(idx);
      continue;
    }

    if ((c.kind == LogicalOpKind::Put ||
         c.kind == LogicalOpKind::PutSignal) &&
        c.dst_peer != ~0u &&
        c.put_path != PutPath::Device) {
      run.tpt_cmds.push_back(c);
      tpt_idx.push_back(idx);
    } else {
      run.dev_cmds.push_back(c);
      dev_idx.push_back(idx);
    }
  }

  // Two-phase submission: reserve → publish ALL slots → submit, so a
  // completion can never beat its slot publication.
  size_t dev_dispatched = 0;
  if (!run.dev_cmds.empty()) {
    size_t m = run.dev_cmds.size();
    // Cap the batch at the slot-table size: a wrap inside one batch
    // would block write() on a slot whose op is not yet submitted — a
    // completion that never comes. Ops beyond the cap stay deferred.
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
        if (op.kind == LogicalOpKind::Put && op.dst_peer != ~0u) {
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
  while (!stop_) {
    if (uk_dbg_lvl() >= UK_DBG_LVL_ALL && ++iter % 10000 == 0)
      UK_DBG(UK_DBG_LVL_ALL, "[drain-tpt r%d] alive iter=%d", rank_or_neg1(),
             iter);
    if (fused_proxy_) fused_proxy_->progress();
    size_t nd;
    {
      HostProf::Scope hps(HostProf::tpt_us);
      nd = tpt_be_->do_drain(be_buf, 256);
    }
    if (HostProf::enabled() && nd > 0)
      HostProf::tpt_ops.fetch_add(nd, std::memory_order_relaxed);
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
  int sub[7] = {0, 0, 0, 0, 0, 0, 0}, pend[7] = {0, 0, 0, 0, 0, 0, 0};
  static char const* kNames[7] = {"Put",      "Reduce", "Signal", "Wait",
                                  "PutSignal", "ReducePut", "ReducePutSignal"};
  for (size_t i = 0; i < nops; ++i) {
    int k = static_cast<int>(run->plan->tiled.ops[i].kind);
    if (k < 0 || k > 6) continue;
    if (__atomic_load_n(&run->submitted[i], __ATOMIC_RELAXED))
      ++sub[k];
    else
      ++pend[k];
  }
  std::fprintf(stderr,
               "[dump r%d] %s done=%zu/%zu epoch=%u deferred dev=%zu tpt=%zu "
               "sig=%zu sig_standalone=%u\n",
               rank_or_neg1(), why, run->done_count.load(), nops,
               run->tag_epoch, run->deferred_dev.size(),
               run->deferred_tpt.size(), run->deferred_sig.size(),
               run->sig_standalone);
  for (int k = 0; k < 7; ++k)
    std::fprintf(stderr, "[dump r%d] %-10s submitted=%d pending=%d\n",
                 rank_or_neg1(), kNames[k], sub[k], pend[k]);
  int shown_per_kind[7] = {0, 0, 0, 0, 0, 0, 0};
  for (size_t i = 0; i < nops; ++i) {
    if (__atomic_load_n(&run->submitted[i], __ATOMIC_RELAXED)) continue;
    auto const& op = run->plan->tiled.ops[i];
    int k = static_cast<int>(op.kind);
    if (k < 0 || k > 6 || shown_per_kind[k] >= 6) continue;
    ++shown_per_kind[k];
    std::string depinfo;
    for (uint32_t d : op.deps) {
      depinfo += std::to_string(d);
      depinfo +=
          (d < run->submitted.size() &&
           __atomic_load_n(&run->submitted[d], __ATOMIC_RELAXED))
              ? "(ok) "
              : "(X) ";
    }
    std::fprintf(stderr,
                 "[dump r%d] pending op[%zu] kind=%d indegree=%u "
                 "ndeps=%zu tag=%#lx src=%d dst=%d bytes=%zu deps=%s\n",
                 rank_or_neg1(), i, (int)op.kind,
                 i < run->indegree.size() ? run->indegree[i] : 9999u,
                 op.deps.size(), (unsigned long)op.tag, (int)op.src_peer,
                 (int)op.dst_peer, op.bytes, depinfo.c_str());
  }
  // Submitted but not yet completed (indegree not set to done).
  int shown_incomplete = 0;
  for (size_t i = 0; i < nops && shown_incomplete < 10; ++i) {
    if (!__atomic_load_n(&run->submitted[i], __ATOMIC_RELAXED)) continue;
    if (run->indegree[i] == SprayRun::kIndegreeDone) continue;
    auto const& op = run->plan->tiled.ops[i];
    std::fprintf(stderr,
                 "[dump r%d] incomplete op[%zu] kind=%d indegree=%u "
                 "tag=%#lx src=%d dst=%d bytes=%zu\n",
                 rank_or_neg1(), i, (int)op.kind,
                 i < run->indegree.size() ? run->indegree[i] : 9999u,
                 (unsigned long)op.tag, (int)op.src_peer, (int)op.dst_peer,
                 op.bytes);
    ++shown_incomplete;
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
        if (op.kind == LogicalOpKind::Put && op.dst_peer != ~0u) {
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
  // Same-host puts go over IPC (CE): measured fastest for same-host
  // traffic; device/RDMA remain reachable via UK_CCL_PUT_PATH and remote
  // peers always use RDMA.
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
