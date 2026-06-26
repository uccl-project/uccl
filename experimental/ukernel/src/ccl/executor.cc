#include "executor.h"
#include "../../include/transport.h"
#include "algo/chunk_graph.h"
#include "backend/backend.h"
#include "coll_config.h"
#include "utils.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <pthread.h>
#include <stdexcept>
#include <thread>

namespace UKernel {
namespace CCL {

// ── Helpers ─────────────────────────────────────────────────────────────

static CollectiveBufferRole buf_role(OpKind kind, bool is_src,
                                      bool copy_from_staging) {
  switch (kind) {
    case OpKind::Put:
      return is_src ? (copy_from_staging ? CollectiveBufferRole::Scratch
                                          : CollectiveBufferRole::Input)
                     : CollectiveBufferRole::Output;
    case OpKind::Reduce:
      return is_src ? CollectiveBufferRole::Input
                    : CollectiveBufferRole::Output;
    case OpKind::Signal:
    case OpKind::WaitSignal:
      return CollectiveBufferRole::Output;
    default:
      return CollectiveBufferRole::Input;
  }
}

static uint32_t role_to_buf(CollectiveBufferRole role, uint32_t in, uint32_t out,
                            uint32_t scr) {
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

static Cmd make_cmd(Op const& op, ReductionKind redop, uint32_t input_buf,
                    uint32_t output_buf, uint32_t scratch_buf) {
  Cmd c{};
  c.kind = op.kind;
  c.bytes = static_cast<uint32_t>(op.bytes);
  c.src_off = static_cast<uint32_t>(op.src_off);
  c.dst_off = static_cast<uint32_t>(op.dst_off);
  c.src_peer = op.src_peer;
  c.dst_peer = op.dst_peer;
  auto role_src = buf_role(op.kind, true, op.copy_from_staging);
  auto role_dst = buf_role(op.kind, false, op.copy_from_staging);
  c.src_buf = role_to_buf(role_src, input_buf, output_buf, scratch_buf);
  c.dst_buf = role_to_buf(role_dst, input_buf, output_buf, scratch_buf);
  c.redop = (op.kind == OpKind::Reduce) ? redop : ReductionKind::None;
  c.transport = static_cast<uint8_t>(Transport::PeerTransportKind::Unknown);
  c.tag = op.tag;
  return c;
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

// ── Constructor ───────────────────────────────────────────────────────

SprayExecutor::SprayExecutor(BatchBackend* device_be, BatchBackend* tpt_be,
                             BatchBackend* signal_be, int world_size)
    : device_be_(device_be),
      tpt_be_(tpt_be),
      signal_be_(signal_be),
      owned_device_(),
      owned_transport_(),
      owned_comm_(),
      stop_(false),
      world_size_(world_size) {
  for (auto& m : cmd_to_run_) m = CmdRunMapping{};
  if (world_size_ > 0)
    tpt_metrics_.reset(new PeerMetrics[static_cast<size_t>(world_size_)]{});

  enqueue_th_ = std::thread(&SprayExecutor::enqueue_loop, this);
  pthread_setname_np(enqueue_th_.native_handle(), "ucl-enq");
  if (device_be_) {
    drain_th_dev_ =
        std::thread(&SprayExecutor::drain_loop, this, device_be_);
    pthread_setname_np(drain_th_dev_.native_handle(), "ucl-drain-dev");
  }
  if (tpt_be_) {
    drain_th_tpt_ = std::thread(&SprayExecutor::drain_tpt_loop, this);
    pthread_setname_np(drain_th_tpt_.native_handle(), "ucl-drain-tpt");
  }
  if (signal_be_) {
    drain_th_signal_ =
        std::thread(&SprayExecutor::drain_signal_loop, this);
    pthread_setname_np(drain_th_signal_.native_handle(), "ucl-drain-sig");
  }
}

SprayExecutor::~SprayExecutor() {
  stop_ = true;
  if (enqueue_th_.joinable()) enqueue_th_.join();
  if (drain_th_dev_.joinable()) drain_th_dev_.join();
  if (drain_th_tpt_.joinable()) drain_th_tpt_.join();
  if (drain_th_signal_.joinable()) drain_th_signal_.join();
}

// ── Lookup ───────────────────────────────────────────────────────────────

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
    if (r->status.load(std::memory_order_acquire) == CollectiveOpStatus::Running) ++n;
  return n;
}

std::string SprayExecutor::error_message(CollectiveOpHandle h) const {
  std::lock_guard lock(runs_mutex_);
  auto it = runs_.find(h);
  return it != runs_.end() ? it->second->error : std::string{};
}

// ── Submit ───────────────────────────────────────────────────────────────

CollectiveOpHandle SprayExecutor::submit_allreduce(CollectiveConfig const& cfg,
                                                   void* input, void* output,
                                                   void* scratch) {
  CollectiveConfig c = cfg;
  c.kind = CollKind::AllReduceRing;
  TiledResult tiled = build_tiled(c, input == output);

  std::lock_guard lock(runs_mutex_);
  auto h = next_handle_++;
  if (tiled.ops.empty()) {
    auto run = std::make_unique<SprayRun>();
    run->status.store(CollectiveOpStatus::Completed, std::memory_order_release);
    runs_[h] = std::move(run);
    return h;
  }

  auto run = std::make_unique<SprayRun>();
  run->status.store(CollectiveOpStatus::Running, std::memory_order_release);
  run->tiled = std::move(tiled);
  run->input_buf_id = get_or_register_buf(input, run->tiled.input_bytes);
  run->output_buf_id = get_or_register_buf(output, run->tiled.output_bytes);
  run->scratch_buf_id =
      get_or_register_buf(scratch, run->tiled.staging_bytes_required);
  run->done.resize(run->tiled.ops.size(), 0);
  run->submitted.resize(run->tiled.ops.size(), 0);

  // Build reverse dependency map and indegree for countdown-latch
  size_t nops = run->tiled.ops.size();
  run->successors.resize(nops);
  run->indegree.resize(nops, 0);
  for (uint32_t i = 0; i < nops; ++i) {
    run->indegree[i] = static_cast<uint32_t>(run->tiled.ops[i].deps.size());
    for (uint32_t dep : run->tiled.ops[i].deps)
      run->successors[dep].push_back(i);
  }

  runs_[h] = std::move(run);
  return h;
}

CollectiveOpHandle SprayExecutor::submit_alltoall(CollectiveConfig const& cfg,
                                                  void* input, void* output,
                                                  void* scratch) {
  CollectiveConfig c = cfg;
  c.kind = CollKind::AllToAllPairwise;
  return submit_allreduce(c, input, output, scratch);
}

// ── Poll / Wait / Release ────────────────────────────────────────────────

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

  auto check_complete = [&]() {
    if (run->done_count.load(std::memory_order_acquire) >= run->tiled.ops.size())
      run->status.store(CollectiveOpStatus::Completed,
                        std::memory_order_release);
  };

  if (to.count() == 0) {
    int spin = 0;
    int sleep_us = 100;
    while (run->status.load(std::memory_order_acquire) ==
           CollectiveOpStatus::Running) {
      if (spin < 1000) { ++spin; std::this_thread::yield(); }
      else { std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
             sleep_us = std::min(sleep_us * 2, 10000); }
      check_complete();
    }
    return run->status.load(std::memory_order_acquire) !=
           CollectiveOpStatus::Failed;
  }

  auto dl = std::chrono::steady_clock::now() + to;
  int spin = 0;
  int sleep_us = 100;
  while (run->status.load(std::memory_order_acquire) ==
         CollectiveOpStatus::Running) {
    if (std::chrono::steady_clock::now() >= dl) break;
    if (spin < 1000) { ++spin; std::this_thread::yield(); }
    else { std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
           sleep_us = std::min(sleep_us * 2, 10000); }
    check_complete();
  }
  return run->status.load(std::memory_order_acquire) !=
         CollectiveOpStatus::Failed;
}

void SprayExecutor::release(CollectiveOpHandle h) {
  std::lock_guard lock(runs_mutex_);
  auto it = runs_.find(h);
  if (it == runs_.end()) return;
  if (it->second->status.load(std::memory_order_acquire) == CollectiveOpStatus::Queued ||
      it->second->status.load(std::memory_order_acquire) == CollectiveOpStatus::Running)
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

void SprayExecutor::enqueue_to_ring(SprayRun& run) {
  if (run.ready.empty()) return;

  run.dev_cmds.clear();
  run.tpt_cmds.clear();
  std::vector<uint32_t> dev_idx;
  std::vector<uint32_t> tpt_idx;

  for (uint32_t idx : run.ready) {
    Cmd c = make_cmd(run.tiled.ops[idx], run.tiled.reduction,
                     run.input_buf_id, run.output_buf_id, run.scratch_buf_id);

    // LB decision: for Put ops, pick IPC vs RDMA dynamically
    if (c.kind == OpKind::Put && c.dst_peer != ~0u) {
      auto peer = static_cast<int>(c.dst_peer);
      auto tpt = pick_transport(peer);
      c.transport = static_cast<uint8_t>(tpt);
      if (peer >= 0 && peer < world_size_) {
        auto& pm = tpt_metrics_[peer];
        if (tpt == Transport::PeerTransportKind::Ipc)
          pm.ipc.inflight.fetch_add(1, std::memory_order_relaxed);
        else if (tpt == Transport::PeerTransportKind::Rdma)
          pm.rdma.inflight.fetch_add(1, std::memory_order_relaxed);
      }
    }

    CmdWithId cwi{c, 0};
    cwi.caller_id = next_cmd_idx_++;
    cmd_to_run_[cwi.caller_id & (kMaxCmdIdx - 1)] = {&run, idx, c.transport,
                                                      cwi.caller_id};

    if (c.kind == OpKind::Signal || c.kind == OpKind::WaitSignal) {
      signal_be_->try_enqueue(&cwi, 1);
      run.submitted[idx] = 1;
      continue;
    }

    if (c.kind == OpKind::Put) {
      if (c.transport == 0) {
        run.dev_cmds.push_back(cwi);
        dev_idx.push_back(idx);
      } else {
        run.tpt_cmds.push_back(cwi);
        tpt_idx.push_back(idx);
      }
    } else {
      run.dev_cmds.push_back(cwi);
      dev_idx.push_back(idx);
    }
  }

  // Submit device batch, stop on backpressure
  {
    size_t off = 0;
    while (off < run.dev_cmds.size()) {
      size_t n = device_be_->try_enqueue(run.dev_cmds.data() + off,
                                         run.dev_cmds.size() - off);
      for (size_t j = 0; j < n; ++j)
        run.submitted[dev_idx[off + j]] = 1;
      off += n;
      if (n == 0) break;
    }
  }

  // Submit transport batch, stop on backpressure
  {
    size_t off = 0;
    while (off < run.tpt_cmds.size()) {
      size_t n = tpt_be_->try_enqueue(run.tpt_cmds.data() + off,
                                      run.tpt_cmds.size() - off);
      for (size_t j = 0; j < n; ++j)
        run.submitted[tpt_idx[off + j]] = 1;
      off += n;
      if (n == 0) break;
    }
  }
}

// ── Thread loops ─────────────────────────────────────────────────────────

void SprayExecutor::enqueue_loop() {
  while (!stop_) {
    bool any = false;
    {
      std::lock_guard lock(runs_mutex_);
      for (auto& [h, run] : runs_) {
        if (run->status.load(std::memory_order_acquire) !=
            CollectiveOpStatus::Running)
          continue;
        {
          std::lock_guard rlock(run->mtx);
          collect_ready(*run);
          enqueue_to_ring(*run);
        }
        if (run->done_count.load(std::memory_order_acquire) >=
            run->tiled.ops.size())
          run->status.store(CollectiveOpStatus::Completed,
                            std::memory_order_release);
        any = true;
      }
    }
    if (!any) std::this_thread::yield();
  }
}

void SprayExecutor::drain_loop(BatchBackend* be) {
  uint32_t done_buf[256];
  while (!stop_) {
    size_t n = be->try_drain(done_buf, 256);
    if (n == 0) {
      std::this_thread::yield();
      continue;
    }
    drain_batch(done_buf, n, [](auto&, uint32_t) {});
  }
}

Transport::PeerTransportKind SprayExecutor::pick_transport(int peer) {
  if (peer < 0 || peer >= world_size_) return Transport::PeerTransportKind::Ipc;
  auto& m = tpt_metrics_[peer];
  uint64_t cost_ipc =
      static_cast<uint64_t>(m.ipc.inflight.load(std::memory_order_relaxed)) *
      m.ipc.latency_ns.load(std::memory_order_relaxed);
  uint64_t cost_rdma =
      static_cast<uint64_t>(m.rdma.inflight.load(std::memory_order_relaxed)) *
      m.rdma.latency_ns.load(std::memory_order_relaxed);
  return cost_ipc <= cost_rdma ? Transport::PeerTransportKind::Ipc
                                : Transport::PeerTransportKind::Rdma;
}

void SprayExecutor::drain_tpt_loop() {
  uint32_t done_buf[256];
  while (!stop_) {
    size_t nd = tpt_be_->try_drain(done_buf, 256);
    if (nd == 0) {
      for (int s = 0; s < 16 && !stop_; ++s) _mm_pause();
      std::this_thread::yield();
      continue;
    }

    drain_batch(done_buf, nd, [this](auto& m, uint32_t) {
      auto transport = m.transport;
      if (transport != 0) {
        auto tpt = static_cast<Transport::PeerTransportKind>(transport);
        if (tpt != Transport::PeerTransportKind::Unknown) {
          int peer = static_cast<int>(m.run->tiled.ops[m.op_idx].dst_peer);
          if (peer >= 0 && peer < world_size_) {
            if (tpt == Transport::PeerTransportKind::Ipc)
              tpt_metrics_[peer].ipc.inflight.fetch_sub(
                  1, std::memory_order_relaxed);
            else if (tpt == Transport::PeerTransportKind::Rdma)
              tpt_metrics_[peer].rdma.inflight.fetch_sub(
                  1, std::memory_order_relaxed);
          }
        }
      }
    });
  }
}

void SprayExecutor::drain_signal_loop() {
  uint32_t done_buf[256];
  while (!stop_) {
    size_t ns = signal_be_->try_drain(done_buf, 256);
    if (ns == 0) {
      for (int s = 0; s < 16 && !stop_; ++s) _mm_pause();
      std::this_thread::yield();
      continue;
    }
    drain_batch(done_buf, ns, [](auto&, uint32_t) {});
  }
}

}  // namespace CCL
}  // namespace UKernel
