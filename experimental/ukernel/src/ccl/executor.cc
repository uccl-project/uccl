#include "executor.h"
#include "../../include/transport.h"
#include "algo/chunk_graph.h"
#include "backend/backend.h"
#include "coll_config.h"
#include "utils.h"
#include "util/uk_debug.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <pthread.h>

namespace UKernel {
namespace CCL {

static void update_path_metrics(PathMetrics& m, uint64_t enqueue_ns) {
  m.inflight.fetch_sub(1, std::memory_order_relaxed);
  uint64_t now =
      std::chrono::steady_clock::now().time_since_epoch().count();
  uint64_t sample = now - enqueue_ns;
  uint64_t old = m.latency_ns.load(std::memory_order_relaxed);
  uint64_t nv = (old * 7 + sample) / 8;
  while (!m.latency_ns.compare_exchange_weak(
      old, nv, std::memory_order_release, std::memory_order_relaxed))
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

static Cmd make_cmd(TiledOp const& op, ReductionKind redop, uint32_t input_buf,
                    uint32_t output_buf, uint32_t scratch_buf) {
  Cmd c{};
  c.kind = op.kind;
  c.bytes = static_cast<uint32_t>(op.bytes);
  c.src_off = static_cast<uint32_t>(op.src_off);
  c.dst_off = static_cast<uint32_t>(op.dst_off);
  c.src_peer = op.src_peer;
  c.dst_peer = op.dst_peer;
  auto role_src = op.src_buf_role;
  auto role_dst = op.dst_buf_role;
  c.src_buf = role_to_buf(role_src, input_buf, output_buf, scratch_buf);
  c.dst_buf = role_to_buf(role_dst, input_buf, output_buf, scratch_buf);
  c.redop = (op.kind == ExecOpKind::Reduce) ? redop : ReductionKind::None;
  c.put_path = PutPath::None;
  c.tag = op.tag;
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
  add(inplace ? 1u : 0u);
  add(cfg.input_split_bytes.size());
  for (size_t v : cfg.input_split_bytes) add(v);
  add(cfg.output_split_bytes.size());
  for (size_t v : cfg.output_split_bytes) add(v);
  return k;
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

uint32_t SprayExecutor::get_or_register_buf(void* ptr, size_t bytes) {
  if (!ptr || !bytes) return 0;
  uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
  auto it = tensor_to_buf_id_.find(key);
  if (it != tensor_to_buf_id_.end()) return it->second;
  uint32_t id = next_buf_id_++;
  tensor_to_buf_id_[key] = id;
  if (owned_comm_ && register_buf_fn_)
    register_buf_fn_(owned_comm_.get(), id, ptr, bytes);
  return id;
}

SprayExecutor::SprayExecutor(BatchBackend* device_be, BatchBackend* tpt_be,
                              BatchBackend* signal_be, int world_size)
    : device_be_(device_be),
      tpt_be_(tpt_be),
      signal_be_(signal_be),
      dev_slots_(device_be ? device_be->capacity() : 0),
      tpt_slots_(tpt_be ? tpt_be->capacity() : 0),
      sig_slots_(signal_be ? signal_be->capacity() : 0),
      stop_(false),
      world_size_(world_size) {
  if (world_size_ > 0)
    tpt_metrics_.reset(new PeerMetrics[static_cast<size_t>(world_size_)]{});

  char const* pc = std::getenv("UK_CCL_PATH_COUNTERS");
  path_counters_enabled_ =
      (pc && (strcmp(pc, "1") == 0 || strcmp(pc, "true") == 0));
}

void SprayExecutor::start() {
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
  // Explicitly release backends before communicator is destroyed.
  // Backends hold raw comm_ pointers that must remain valid during
  // their destructors (e.g. DeviceBackend tears down GPU task manager).
  device_be_ = nullptr;
  tpt_be_ = nullptr;
  signal_be_ = nullptr;
  owned_device_.reset();
  owned_transport_.reset();
  owned_signal_.reset();
  if (internal_scratch_) { GPU_RT_CHECK(gpuFree(internal_scratch_)); }
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

void SprayExecutor::prepare(CollectiveConfig const& cfg, void* input,
                            void* output) {
  if (!owned_comm_ || !peer_setup_fn_) return;

  // Derive needed peers from the algorithm DAG.
  CollAlgo algo = build_coll_algo(cfg, input == output);
  std::vector<int> peers;
  for (auto const& ch : algo.chunks) {
    if (ch.src_rank >= 0)
      peers.push_back(ch.src_rank);
    if (ch.dst_rank >= 0)
      peers.push_back(ch.dst_rank);
  }
  // Deduplicate and sort.
  std::sort(peers.begin(), peers.end());
  peers.erase(std::unique(peers.begin(), peers.end()), peers.end());

  {
    std::string plist;
    for (auto p : peers) { if (!plist.empty()) plist += ","; plist += std::to_string(p); }
    UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] peers=%s  -> peer_setup_fn start", cfg.rank, plist.c_str());
  }
  peer_setup_fn_(owned_comm_.get(), cfg.rank, peers);
  UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] peer_setup_fn done  -> re_register_all_mrs", cfg.rank);
  owned_comm_->re_register_all_mrs();
  UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] re_register_all_mrs done  -> register bufs", cfg.rank);
  prepared_peers_.insert(peers.begin(), peers.end());
  prepared_ = true;

  // Register and resolve user buffers.
  uint32_t in_id = get_or_register_buf(input, cfg.input_bytes);
  UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] in_id=%u registered  -> resolve bufs", cfg.rank, in_id);
  uint32_t out_id = get_or_register_buf(output, cfg.output_bytes);
  UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] out_id=%u registered", cfg.rank, out_id);
  for (int p : peers) {
    if (in_id && resolve_buf_fn_) {
      UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] resolve in_id=%u from peer %d ...", cfg.rank, in_id, p);
      resolve_buf_fn_(owned_comm_.get(), p, world_size_, in_id);
      UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] resolve in_id=%u from peer %d done", cfg.rank, in_id, p);
    }
    if (out_id && out_id != in_id && resolve_buf_fn_) {
      UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] resolve out_id=%u from peer %d ...", cfg.rank, out_id, p);
      resolve_buf_fn_(owned_comm_.get(), p, world_size_, out_id);
      UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] resolve out_id=%u from peer %d done", cfg.rank, out_id, p);
    }
  }

  UK_DBG(UK_DBG_LVL_EXEC, "[prepare r%d] ALL DONE", cfg.rank);
  prepared_ = true;
}

CollectiveOpHandle SprayExecutor::submit(CollectiveConfig const& cfg,
                                         void* input, void* output) {
  if (owned_comm_ && !prepared_) {
    // Check all peers needed by this algorithm are prepared.
    CollAlgo algo = build_coll_algo(cfg, input == output);
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
    std::string key = plan_key(cfg, input == output);
    auto it = plan_cache_.find(key);
    if (it != plan_cache_.end()) {
      plan = it->second;
    } else {
      if (plan_cache_.size() >= kMaxCachedPlans) plan_cache_.clear();
      plan = build_plan(cfg, input == output);
      plan_cache_.emplace(std::move(key), plan);
    }
  }
  TiledResult const& tiled = plan->tiled;

  // Allocate or grow internal scratch buffer as needed.
  if (tiled.staging_bytes_required > 0) {
    if (tiled.staging_bytes_required > internal_scratch_cap_) {
      if (internal_scratch_) GPU_RT_CHECK(gpuFree(internal_scratch_));
      internal_scratch_cap_ = tiled.staging_bytes_required;
      GPU_RT_CHECK(gpuMalloc(&internal_scratch_, internal_scratch_cap_));
    }
  }

  uint32_t in_id = 0, out_id = 0, scr_id = 0;
  if (owned_comm_) {
    in_id = get_or_register_buf(input, tiled.input_bytes);
    out_id = get_or_register_buf(output, tiled.output_bytes);
    if (internal_scratch_ && tiled.staging_bytes_required > 0)
      scr_id = get_or_register_buf(internal_scratch_, tiled.staging_bytes_required);
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
  size_t nops = plan->nops;
  run->submitted.resize(nops, 0);
  run->fused_sig_cnt.assign(nops, 0);

  UK_DBG(UK_DBG_LVL_EXEC, "[submit r%d] %zu ops", cfg.rank, nops);
  run->init_ready_ring(nops);
  run->indegree = plan->indegree_init;  // one memcpy from the template
  for (uint32_t op : plan->initial_ready) run->push_ready(op);
  runs_[h] = std::move(run);
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
  if (it->second->status.load(std::memory_order_acquire) ==
          CollectiveOpStatus::Queued ||
      it->second->status.load(std::memory_order_acquire) ==
          CollectiveOpStatus::Running)
    throw std::logic_error("cannot release running collective");
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

  run.dev_cmds.clear();
  run.tpt_cmds.clear();
  std::vector<uint32_t> dev_idx;
  std::vector<uint32_t> tpt_idx;
  size_t sig_dispatched = 0;

  static int op_kind_print_cnt = 0;
  for (uint32_t idx : run.ready) {
    Cmd c = make_cmd(run.plan->tiled.ops[idx], run.plan->tiled.reduction,
                     run.input_buf_id,
                     run.output_buf_id, run.scratch_buf_id);
    if (op_kind_print_cnt++ < 20) {
      UK_DBG(UK_DBG_LVL_EXEC,
             "[enqueue r%d] op[%u] kind=%d dst_peer=%u put_path=%d tag=%lu bytes=%u",
             rank_or_neg1(), idx, (int)c.kind, c.dst_peer, (int)c.put_path,
             (unsigned long)c.tag, c.bytes);
    }

    if (c.kind == ExecOpKind::Put && c.dst_peer != ~0u) {
      c.put_path = pick_put_path(static_cast<int>(c.dst_peer));
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
      if (c.put_path == PutPath::Device && kBar1Bytes > 0 &&
          c.dst_off + c.bytes > kBar1Bytes) {
        // Move the tentative charge from Device to IPC (reroute).
        tpt_metrics_[static_cast<size_t>(c.dst_peer)]
            .device.inflight.fetch_sub(1, std::memory_order_relaxed);
        tpt_metrics_[static_cast<size_t>(c.dst_peer)]
            .ipc.inflight.fetch_add(1, std::memory_order_relaxed);
        c.put_path = PutPath::Ipc;
      }
    }

    if (path_counters_enabled_ &&
        c.kind == ExecOpKind::Put && c.dst_peer != ~0u) {
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
    // counts grp arrivals, and all fused channels (device-kernel and
    // IPC ring writes, RDMA immediates) share one matching layer, so
    // per-op paths may freely mix. A put whose picked path cannot fuse
    // (e.g. device kernel without a GPU-mapped ring) is rerouted to
    // IPC, which is always fusable for same-host peers. Remote groups
    // fall back to a standalone Signal when RDMA cannot fuse, mirrored
    // by the receiver's wait count.
    if (c.kind == ExecOpKind::Put && c.dst_peer != ~0u && owned_comm_ &&
        !run.plan->put_to_sig.empty()) {
      int32_t sig_idx = run.plan->put_to_sig[idx];
      if (sig_idx >= 0) {
        uint16_t grp = run.plan->sig_group_size[sig_idx];
        bool fuse = false;
        if (c.put_path == PutPath::Device) {
          fuse = device_be_->can_fuse_put_signal(
              static_cast<int>(c.dst_peer));
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
          fuse = owned_comm_->can_fuse_put_signal(
              static_cast<int>(c.dst_peer), tpt_kind);
        }
        if (fuse) {
          c.flags |= kCmdFlagPutSignal;
          c.tag = run.plan->tiled.ops[sig_idx].tag;
        }
      }
    }

    if (c.kind == ExecOpKind::Signal || c.kind == ExecOpKind::WaitSignal) {
      // Fused: every Put of this Signal's group already carried the tag
      // — no backend dispatch needed, complete it locally.
      if (c.kind == ExecOpKind::Signal &&
          !run.plan->sig_group_size.empty() &&
          run.plan->sig_group_size[idx] > 0 &&
          run.fused_sig_cnt[idx] == run.plan->sig_group_size[idx]) {
        run.submitted[idx] = 1;
        complete_op_local(run, idx);
        ++sig_dispatched;
        continue;
      }
      // Counted wait: when the sender fuses this group, each tile
      // arrives as its own signal, so the wait counts group_size
      // arrivals instead of one standalone signal. The sender's rule is
      // deterministic and mirrored here: same-host groups always fully
      // fuse (IPC is the guaranteed fallback), remote groups fuse iff
      // RDMA can fuse.
      if (c.kind == ExecOpKind::WaitSignal && owned_comm_ &&
          !run.plan->wait_group_size.empty() &&
          run.plan->wait_group_size[idx] > 1 &&
          (owned_comm_->can_fuse_put_signal(
               static_cast<int>(c.src_peer), Transport::PeerTransportKind::Ipc) ||
           owned_comm_->can_fuse_put_signal(
               static_cast<int>(c.src_peer),
               Transport::PeerTransportKind::Rdma))) {
        c.wait_count = run.plan->wait_group_size[idx];
      }
      uint32_t be_idx = signal_be_->reserve_slot();
      if (be_idx != BatchBackend::kInvalidBeIdx) {
        // Reserve-then-enqueue: publish the slot BEFORE the op can
        // complete (IPC signal sends complete synchronously).
        sig_slots_.write(be_idx, &run, idx, PutPath::None, stop_);
        if (signal_be_->do_enqueue_reserved(c, be_idx)) {
          run.be_slots.emplace_back(2, be_idx);
          run.submitted[idx] = 1;
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
    be_idx_scratch_.resize(m);
    size_t reserved =
        device_be_->reserve_slots(be_idx_scratch_.data(), m);
    if (reserved > 0) {
      for (size_t i = 0; i < reserved; ++i)
        dev_slots_.write(be_idx_scratch_[i], &run, dev_idx[i],
                         PutPath::None, stop_);
      size_t ok = device_be_->do_enqueue_reserved_batch(
          run.dev_cmds.data(), be_idx_scratch_.data(), reserved);
      for (size_t i = 0; i < ok; ++i) {
        run.be_slots.emplace_back(0, be_idx_scratch_[i]);
        run.submitted[dev_idx[i]] = 1;
        if (run.dev_cmds[i].flags & kCmdFlagPutSignal)
          ++run.fused_sig_cnt[run.plan->put_to_sig[dev_idx[i]]];
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
      for (size_t i = 0; i < ok; ++i) {
        dev_slots_.write(be_idx_scratch_[i], &run, dev_idx[i],
                         PutPath::None, stop_);
        run.be_slots.emplace_back(0, be_idx_scratch_[i]);
        run.submitted[dev_idx[i]] = 1;
        if (run.dev_cmds[i].flags & kCmdFlagPutSignal)
          ++run.fused_sig_cnt[run.plan->put_to_sig[dev_idx[i]]];
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
    be_idx_scratch_.resize(m);
    size_t reserved = tpt_be_->reserve_slots(be_idx_scratch_.data(), m);
    if (reserved > 0) {
      for (size_t i = 0; i < reserved; ++i)
        tpt_slots_.write(be_idx_scratch_[i], &run, tpt_idx[i],
                         run.tpt_cmds[i].put_path, stop_);
      size_t ok = tpt_be_->do_enqueue_reserved_batch(
          run.tpt_cmds.data(), be_idx_scratch_.data(), reserved);
      for (size_t i = 0; i < ok; ++i) {
        run.be_slots.emplace_back(1, be_idx_scratch_[i]);
        run.submitted[tpt_idx[i]] = 1;
        // The fused put was accepted: count it toward its signal group;
        // the Signal op completes locally once the whole group is out.
        if (run.tpt_cmds[i].flags & kCmdFlagPutSignal)
          ++run.fused_sig_cnt[run.plan->put_to_sig[tpt_idx[i]]];
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
      size_t ok = tpt_be_->do_enqueue(run.tpt_cmds.data(), m,
                                      be_idx_scratch_.data());
      for (size_t i = 0; i < ok; ++i) {
        tpt_slots_.write(be_idx_scratch_[i], &run, tpt_idx[i],
                         run.tpt_cmds[i].put_path, stop_);
        run.be_slots.emplace_back(1, be_idx_scratch_[i]);
        run.submitted[tpt_idx[i]] = 1;
        if (run.tpt_cmds[i].flags & kCmdFlagPutSignal)
          ++run.fused_sig_cnt[run.plan->put_to_sig[tpt_idx[i]]];
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

  run.ready.clear();
}

void SprayExecutor::enqueue_loop() {
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
    for (auto& run : snapshot) {
      std::lock_guard rlock(run->mtx);
      collect_ready(*run);
      enqueue_to_ring(*run);
    }
    if (snapshot.empty()) std::this_thread::yield();
  }
}

void SprayExecutor::drain_dev_loop() {
  uint32_t be_buf[256];
  BeSlotSnap snap_buf[256];
  int iter = 0;
  static int dbg_count = 0;
  while (!stop_) {
    if (uk_dbg_lvl() >= UK_DBG_LVL_ALL && ++iter % 10000 == 0)
      UK_DBG(UK_DBG_LVL_ALL, "[drain-dev r%d] alive iter=%d", rank_or_neg1(), iter);
    size_t n = device_be_->do_drain(be_buf, 256);
    if (n > 0 && dbg_count < 5) {
      ++dbg_count;
      UK_DBG(UK_DBG_LVL_EXEC, "[drain-dev r%d] do_drain returned %zu completions (count=%d)",
             rank_or_neg1(), n, dbg_count);
    }
    if (n == 0) { std::this_thread::yield(); continue; }
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
      n = device_be_->do_drain(be_buf, 256);
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
      UK_DBG(UK_DBG_LVL_ALL, "[drain-tpt r%d] alive iter=%d", rank_or_neg1(), iter);
    size_t nd = tpt_be_->do_drain(be_buf, 256);
    if (nd > 0 && dbg_count < 5) {
      ++dbg_count;
      UK_DBG(UK_DBG_LVL_EXEC, "[drain-tpt r%d] do_drain returned %zu completions (count=%d)",
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
      nd = tpt_be_->do_drain(be_buf, 256);
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
      UK_DBG(UK_DBG_LVL_ALL, "[drain-sig r%d] alive iter=%d", rank_or_neg1(), iter);
    size_t ns = signal_be_->do_drain(be_buf, 256);
    if (ns > 0 && dbg_count < 5) {
      ++dbg_count;
      UK_DBG(UK_DBG_LVL_EXEC, "[drain-sig r%d] do_drain returned %zu completions (count=%d)",
             rank_or_neg1(), ns, dbg_count);
    }
    if (ns == 0) {
      for (int s = 0; s < 16 && !stop_; ++s) machnet_pause();
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
        snap_buf[valid++] = snap;
      }
      drain_batch(snap_buf, valid, [](BeSlotSnap&) {});
      ns = signal_be_->do_drain(be_buf, 256);
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
  if (run->status.compare_exchange_strong(expected,
                                          CollectiveOpStatus::Completed,
                                          std::memory_order_release,
                                          std::memory_order_relaxed))
    active_runs_.fetch_sub(1, std::memory_order_release);
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
  if (forced != PutPath::None &&
      same_host_fn_ && same_host_fn_(owned_comm_.get(), peer)) {
    auto& pfm = tpt_metrics_[peer];
    PathMetrics* fm = (forced == PutPath::Device)   ? &pfm.device
                      : (forced == PutPath::Ipc)    ? &pfm.ipc
                                                    : &pfm.rdma;
    fm->inflight.fetch_add(1, std::memory_order_relaxed);
    return forced;
  }
  // Tentative charge: bump the chosen path now so a batch of picks
  // rotates and every path gets measured. The charge is reconciled
  // exactly once: released on deferral (release_put_inflight), moved on
  // reroute (BAR1 / fusion fallback), or balanced by the drain-side
  // decrement after acceptance. Deferred ops re-pick next cycle with a
  // fresh tentative charge, so nothing leaks.
  if (!same_host_fn_ || !same_host_fn_(owned_comm_.get(), peer)) {
    tpt_metrics_[peer].rdma.inflight.fetch_add(1, std::memory_order_relaxed);
    return PutPath::Rdma;
  }
  auto& pm = tpt_metrics_[peer];
  uint64_t dc = static_cast<uint64_t>(
                    pm.device.inflight.load(std::memory_order_relaxed)) *
                pm.device.latency_ns.load(std::memory_order_relaxed);
  uint64_t ic = static_cast<uint64_t>(
                    pm.ipc.inflight.load(std::memory_order_relaxed)) *
                pm.ipc.latency_ns.load(std::memory_order_relaxed);
  uint64_t rc = static_cast<uint64_t>(
                    pm.rdma.inflight.load(std::memory_order_relaxed)) *
                pm.rdma.latency_ns.load(std::memory_order_relaxed);

  PutPath choice;
  PathMetrics* chosen;
  if (ic <= dc && ic <= rc) {
    choice = PutPath::Ipc;
    chosen = &pm.ipc;
  } else if (dc <= rc) {
    choice = PutPath::Device;
    chosen = &pm.device;
  } else {
    choice = PutPath::Rdma;
    chosen = &pm.rdma;
  }
  chosen->inflight.fetch_add(1, std::memory_order_relaxed);
  return choice;
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
