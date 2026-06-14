#include "executor.h"
#include "../../include/transport.h"
#include "algo/chunk_graph.h"
#include "backend/async_backend.h"
#include "backend/backend.h"
#include "backend/transport_backend.h"
#include "coll_config.h"
#include "utils.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>

namespace UKernel {
namespace CCL {

// ── Helpers ─────────────────────────────────────────────────────────────

static CollectiveBufferRole buf_role(OpKind kind, bool is_src,
                                     bool copy_from_staging) {
  switch (kind) {
    case OpKind::Copy:
      return is_src ? (copy_from_staging ? CollectiveBufferRole::Scratch
                                         : CollectiveBufferRole::Input)
                    : CollectiveBufferRole::Output;
    case OpKind::Reduce:
      return is_src ? CollectiveBufferRole::Input : CollectiveBufferRole::Output;
    case OpKind::Send:
      return is_src ? CollectiveBufferRole::Input : CollectiveBufferRole::Output;
    case OpKind::Recv:
    case OpKind::RecvReduce:
      return CollectiveBufferRole::Output;
    case OpKind::Signal:
    case OpKind::SignalWait:
      return CollectiveBufferRole::Input;
    default:
      return CollectiveBufferRole::Input;
  }
}

static uint32_t buf_of(Op const& op, bool is_src) {
  auto r = buf_role(op.kind, is_src, op.copy_from_staging);
  return static_cast<uint32_t>(r) + 1;
}

static Cmd make_cmd(Op const& op, ReductionKind redop) {
  Cmd c{};
  c.kind = op.kind;
  c.bytes = static_cast<uint32_t>(op.bytes);
  c.src_off = static_cast<uint32_t>(op.src_off);
  c.dst_off = static_cast<uint32_t>(op.dst_off);
  c.src_peer = op.src_peer;
  c.dst_peer = op.dst_peer;
  c.src_buf = buf_of(op, true);
  c.dst_buf = buf_of(op, false);
  c.redop = (op.kind == OpKind::Reduce || op.kind == OpKind::RecvReduce)
                ? redop : ReductionKind::None;
  c.transport = static_cast<uint8_t>(Transport::PeerTransportKind::Unknown);
  c.tag = op.tag;
  return c;
}

// ── Constructor ───────────────────────────────────────────────────────

SprayExecutor::SprayExecutor(BatchBackend* device_be, BatchBackend* tpt_be)
    : device_be_(device_be), tpt_be_(tpt_be),
      owned_device_(), owned_transport_(), owned_comm_(),
      stop_(false) {
  std::memset(cmd_to_run_, 0, sizeof(cmd_to_run_));

  if (device_be_) {
    async_dev_ = std::make_unique<AsyncBackend>(device_be_, 2048, 2048);
    async_dev_->start();
  }
  if (tpt_be_) {
    async_tpt_ = std::make_unique<AsyncBackend>(tpt_be_, 2048, 2048);
    async_tpt_->start();
  }

  enqueue_th_ = std::thread(&SprayExecutor::enqueue_loop, this);
  if (async_dev_)
    drain_th_dev_ = std::thread(&SprayExecutor::drain_loop, this, async_dev_.get());
  if (async_tpt_)
    drain_th_tpt_ = std::thread(&SprayExecutor::drain_tpt_loop, this);
}

SprayExecutor::~SprayExecutor() {
  stop_ = true;
  if (enqueue_th_.joinable()) enqueue_th_.join();
  if (drain_th_dev_.joinable()) drain_th_dev_.join();
  if (drain_th_tpt_.joinable()) drain_th_tpt_.join();
  if (async_dev_) async_dev_->stop();
  if (async_tpt_) async_tpt_->stop();
}

// ── Lookup ───────────────────────────────────────────────────────────────

SprayRun* SprayExecutor::get(CollectiveOpHandle h) {
  std::lock_guard lock(runs_mutex_);
  auto it = runs_.find(h);
  return it != runs_.end() ? it->second.get() : nullptr;
}

CollectiveOpStatus SprayExecutor::status(CollectiveOpHandle h) const {
  auto* ex = const_cast<SprayExecutor*>(this);
  std::lock_guard lock(ex->runs_mutex_);
  auto it = ex->runs_.find(h);
  return it != ex->runs_.end() ? it->second->status : CollectiveOpStatus::Completed;
}

size_t SprayExecutor::active_count() const {
  auto* ex = const_cast<SprayExecutor*>(this);
  std::lock_guard lock(ex->runs_mutex_);
  size_t n = 0;
  for (auto& [h, r] : ex->runs_)
    if (r->status == CollectiveOpStatus::Running) ++n;
  return n;
}

std::string SprayExecutor::error_message(CollectiveOpHandle h) const {
  auto* ex = const_cast<SprayExecutor*>(this);
  std::lock_guard lock(ex->runs_mutex_);
  auto it = ex->runs_.find(h);
  return it != ex->runs_.end() ? it->second->error : std::string{};
}

// ── Submit ───────────────────────────────────────────────────────────────

CollectiveOpHandle SprayExecutor::submit_allreduce(
    CollectiveConfig const& cfg, void* input, void* output, void* scratch) {
  CollectiveConfig c = cfg;
  c.kind = CollKind::AllReduceRing;
  TiledResult tiled = build_tiled(c, input == output);

  BufSpec bufs[3] = {
      {input, tiled.input_bytes},
      {output, tiled.output_bytes},
      {scratch, tiled.staging_bytes_required},
  };

  if (device_be_) device_be_->init(bufs);
  if (tpt_be_) tpt_be_->init(bufs);

  std::lock_guard lock(runs_mutex_);
  auto h = next_handle_++;
  if (tiled.ops.empty()) {
    auto run = std::make_unique<SprayRun>();
    run->status = CollectiveOpStatus::Completed;
    runs_[h] = std::move(run);
    return h;
  }

  auto run = std::make_unique<SprayRun>();
  run->status = CollectiveOpStatus::Running;
  run->tiled = std::move(tiled);
  run->input = input; run->output = output; run->scratch = scratch;
  run->done.resize(run->tiled.ops.size(), false);
  run->submitted.resize(run->tiled.ops.size(), false);
  runs_[h] = std::move(run);
  return h;
}

CollectiveOpHandle SprayExecutor::submit_alltoall(
    CollectiveConfig const& cfg, void* input, void* output, void* scratch) {
  CollectiveConfig c = cfg;
  c.kind = CollKind::AllToAllPairwise;
  return submit_allreduce(c, input, output, scratch);
}

// ── Poll / Wait / Release ────────────────────────────────────────────────

bool SprayExecutor::poll(CollectiveOpHandle h) {
  std::lock_guard lock(runs_mutex_);
  auto it = runs_.find(h);
  if (it == runs_.end()) return true;
  auto s = it->second->status;
  return s == CollectiveOpStatus::Completed ||
         s == CollectiveOpStatus::Failed;
}

bool SprayExecutor::wait(CollectiveOpHandle h, std::chrono::milliseconds to) {
  constexpr int kSpin = 1000;
  int spin = 0;
  if (to.count() == 0) {
    while (!poll(h)) {
      if (spin < kSpin) { ++spin; std::this_thread::yield(); }
      else std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return true;
  }
  auto dl = std::chrono::steady_clock::now() + to;
  do {
    if (poll(h)) return true;
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < dl);
  return poll(h);
}

void SprayExecutor::release(CollectiveOpHandle h) {
  std::lock_guard lock(runs_mutex_);
  auto it = runs_.find(h);
  if (it == runs_.end()) return;
  if (it->second->status == CollectiveOpStatus::Queued ||
      it->second->status == CollectiveOpStatus::Running)
    throw std::logic_error("cannot release running collective");
  runs_.erase(it);
}

// ── Phase helpers ────────────────────────────────────────────────────────

void SprayExecutor::collect_ready(SprayRun& run) {
  run.ready.clear();
  auto& ops = run.tiled.ops;
  auto& layer = run.tiled.layers;

  for (uint32_t l = run.next_layer; l < layer.size(); ++l) {
    bool ld = true;
    for (uint32_t op : layer[l]) {
      if (run.done[op] || run.submitted[op]) continue;
      ld = false;
      bool ok = true;
      for (uint32_t d : ops[op].deps)
        if (!run.done[d]) { ok = false; break; }
      if (ok) run.ready.push_back(op);
    }
    if (ld) run.next_layer = l + 1;
  }
}

void SprayExecutor::enqueue_to_ring(SprayRun& run, AsyncBackend* async_be) {
  (void)async_be;
  if (run.ready.empty()) return;

  run.dev_cmds.clear();
  run.tpt_cmds.clear();

  for (uint32_t idx : run.ready) {
    Cmd c = make_cmd(run.tiled.ops[idx], run.tiled.reduction);

    // LB decision: for Send ops, pick IPC vs RDMA dynamically
    if (c.kind == OpKind::Send && c.dst_peer != ~0u) {
      auto tpt = pick_transport(static_cast<int>(c.dst_peer));
      c.transport = static_cast<uint8_t>(tpt);
      // Bump inflight counter for the chosen path
      auto& pm = tpt_metrics_[static_cast<int>(c.dst_peer)];
      if (tpt == Transport::PeerTransportKind::Ipc)
        pm.ipc.inflight.fetch_add(1, std::memory_order_relaxed);
      else if (tpt == Transport::PeerTransportKind::Rdma)
        pm.rdma.inflight.fetch_add(1, std::memory_order_relaxed);
    }

    CmdWithId cwi{c, 0};
    cwi.caller_id = next_cmd_idx_++;
    cmd_to_run_[cwi.caller_id & (kMaxCmdIdx - 1)] = {&run, idx};
    cmd_transport_[cwi.caller_id] = c.transport;

    // Route: Signal/SignalWait → TransportBackend; Send with explicit transport → TptBE
    if (c.kind == OpKind::Signal || c.kind == OpKind::SignalWait ||
        (async_tpt_ && (c.kind == OpKind::Send || c.kind == OpKind::Recv) &&
         c.transport != 0)) {
      run.tpt_cmds.push_back(cwi);
    } else if (async_tpt_ && tpt_be_->supports(c.kind)) {
      run.tpt_cmds.push_back(cwi);
    } else {
      run.dev_cmds.push_back(cwi);
    }
    run.submitted[idx] = true;
  }

  if (!run.dev_cmds.empty())
    async_dev_->try_enqueue(run.dev_cmds.data(), run.dev_cmds.size());

  if (!run.tpt_cmds.empty())
    async_tpt_->try_enqueue(run.tpt_cmds.data(), run.tpt_cmds.size());
}

// ── Thread loops ─────────────────────────────────────────────────────────

void SprayExecutor::enqueue_loop() {
  while (!stop_) {
    bool any = false;
    {
      std::lock_guard lock(runs_mutex_);
      for (auto& [h, run] : runs_) {
        if (run->status != CollectiveOpStatus::Running) continue;
        {
          std::lock_guard rlock(run->mtx);
          collect_ready(*run);
          enqueue_to_ring(*run, nullptr);
        }
        any = true;
      }
    }
    if (!any) std::this_thread::yield();
  }
}

void SprayExecutor::drain_loop(AsyncBackend* async_be) {
  uint32_t done_buf[256];
  while (!stop_) {
    size_t n = async_be->try_drain(done_buf, 256);
    if (n == 0) {
      std::this_thread::yield();
      continue;
    }

    for (size_t i = 0; i < n; ++i) {
      auto& m = cmd_to_run_[done_buf[i] & (kMaxCmdIdx - 1)];
      if (!m.run) continue;

      std::lock_guard rlock(m.run->mtx);
      if (!m.run->done[m.op_idx]) {
        m.run->done[m.op_idx] = true;
        m.run->done_count.fetch_add(1, std::memory_order_release);
      }
    }
    // Release m.run->mtx before locking runs_mutex_ to avoid deadlock:
    // enqueue_thread: runs_mutex_ → run->mtx
    // drain_thread:   run->mtx → ... (done above) → runs_mutex_

    std::lock_guard lock(runs_mutex_);
    for (auto& [h, run] : runs_) {
      if (run->status != CollectiveOpStatus::Running) continue;
      size_t dc = run->done_count.load(std::memory_order_acquire);
      if (dc >= run->tiled.ops.size())
        run->status = CollectiveOpStatus::Completed;
    }
  }
}

Transport::PeerTransportKind SprayExecutor::pick_transport(int peer) {
  auto& m = tpt_metrics_[peer];
  double cost_ipc = static_cast<double>(
      m.ipc.inflight.load(std::memory_order_relaxed)) *
      m.ipc.latency_us.load(std::memory_order_relaxed);
  double cost_rdma = static_cast<double>(
      m.rdma.inflight.load(std::memory_order_relaxed)) *
      m.rdma.latency_us.load(std::memory_order_relaxed);
  return cost_ipc <= cost_rdma ? Transport::PeerTransportKind::Ipc
                               : Transport::PeerTransportKind::Rdma;
}

void SprayExecutor::drain_tpt_loop() {
  uint32_t done_buf[256];
  while (!stop_) {
    // Channel 1: data completions via AsyncBackend (Send, Signal ops)
    size_t nd = async_tpt_->try_drain(done_buf, 256);
    for (size_t i = 0; i < nd; ++i) {
      uint32_t caller_id = done_buf[i];
      auto& m = cmd_to_run_[caller_id & (kMaxCmdIdx - 1)];
      if (!m.run) continue;

      std::lock_guard rlock(m.run->mtx);
      if (!m.run->done[m.op_idx]) {
        m.run->done[m.op_idx] = true;
        m.run->done_count.fetch_add(1, std::memory_order_release);

        auto it = cmd_transport_.find(caller_id);
        if (it != cmd_transport_.end()) {
          auto tpt = static_cast<Transport::PeerTransportKind>(it->second);
          if (tpt != Transport::PeerTransportKind::Unknown) {
            int peer = static_cast<int>(m.run->tiled.ops[m.op_idx].dst_peer);
            auto& pm = tpt_metrics_[peer];
            if (tpt == Transport::PeerTransportKind::Ipc)
              pm.ipc.inflight.fetch_sub(1, std::memory_order_relaxed);
            else if (tpt == Transport::PeerTransportKind::Rdma)
              pm.rdma.inflight.fetch_sub(1, std::memory_order_relaxed);
          }
          cmd_transport_.erase(it);
        }
      }
    }

    // Channel 2: direct signal completions (1-hop, bypasses AsyncBackend)
    auto* tpt_be = static_cast<TransportBackend*>(tpt_be_);
    size_t ns = tpt_be->drain_signals(done_buf, 256);
    for (size_t i = 0; i < ns; ++i) {
      uint32_t caller_id = done_buf[i];
      auto& m = cmd_to_run_[caller_id & (kMaxCmdIdx - 1)];
      if (!m.run) continue;

      std::lock_guard rlock(m.run->mtx);
      if (!m.run->done[m.op_idx]) {
        m.run->done[m.op_idx] = true;
        m.run->done_count.fetch_add(1, std::memory_order_release);
      }
    }

    // Mark completed runs
    {
      std::lock_guard lock(runs_mutex_);
      for (auto& [h, run] : runs_) {
        if (run->status != CollectiveOpStatus::Running) continue;
        size_t dc = run->done_count.load(std::memory_order_acquire);
        if (dc >= run->tiled.ops.size())
          run->status = CollectiveOpStatus::Completed;
      }
    }

    if (nd == 0 && ns == 0) {
      for (int s = 0; s < 16 && !stop_; ++s) _mm_pause();
      std::this_thread::yield();
    }
  }
}

}  // namespace CCL
}  // namespace UKernel
