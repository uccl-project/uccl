#include "executor.h"
#include "../../include/transport.h"
#include "algo/chunk_graph.h"
#include "backend/backend.h"
#include "coll_config.h"
#include "utils.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <pthread.h>

namespace UKernel {
namespace CCL {

static CollectiveBufferRole buf_role(ExecOpKind kind, bool is_src) {
  switch (kind) {
    case ExecOpKind::Put:
      return is_src ? CollectiveBufferRole::Input
                    : CollectiveBufferRole::Output;
    case ExecOpKind::Reduce:
      return is_src ? CollectiveBufferRole::Input
                    : CollectiveBufferRole::Output;
    case ExecOpKind::Signal:
    case ExecOpKind::WaitSignal:
      return CollectiveBufferRole::Output;
    default:
      return CollectiveBufferRole::Input;
  }
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
  auto role_src = buf_role(op.kind, true);
  auto role_dst = buf_role(op.kind, false);
  c.src_buf = role_to_buf(role_src, input_buf, output_buf, scratch_buf);
  c.dst_buf = role_to_buf(role_dst, input_buf, output_buf, scratch_buf);
  c.redop = (op.kind == ExecOpKind::Reduce) ? redop : ReductionKind::None;
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

// ReadyRing lock-free MPSC

bool SprayRun::ReadyRing::push(uint32_t op) {
  uint32_t t = tail.load(std::memory_order_relaxed);
  for (;;) {
    uint32_t next = (t + 1) & kMask;
    if (next == head.load(std::memory_order_acquire)) return false;  // full
    if (tail.compare_exchange_weak(t, next, std::memory_order_release,
                                   std::memory_order_relaxed)) {
      buf[t] = op;
      return true;
    }
  }
}

uint32_t SprayRun::ReadyRing::pop() {
  uint32_t h = head.load(std::memory_order_relaxed);
  if (h == tail.load(std::memory_order_acquire)) return ~0u;
  uint32_t op = buf[h];
  head.store((h + 1) & kMask, std::memory_order_release);
  return op;
}


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
  if (device_be_) {
    dev_caller_map_.reset(new std::atomic<uint32_t>[kCallerMapSize]);
    for (size_t i = 0; i < kCallerMapSize; ++i)
      dev_caller_map_[i].store(kMapSlotEmpty, std::memory_order_relaxed);
  }
  if (tpt_be_) {
    tpt_caller_map_.reset(new std::atomic<uint32_t>[kCallerMapSize]);
    for (size_t i = 0; i < kCallerMapSize; ++i)
      tpt_caller_map_[i].store(kMapSlotEmpty, std::memory_order_relaxed);
  }
  if (signal_be_) {
    sig_caller_map_.reset(new std::atomic<uint32_t>[kCallerMapSize]);
    for (size_t i = 0; i < kCallerMapSize; ++i)
      sig_caller_map_[i].store(kMapSlotEmpty, std::memory_order_relaxed);
  }
  if (world_size_ > 0)
    tpt_metrics_.reset(new PeerMetrics[static_cast<size_t>(world_size_)]{});

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
  if (enqueue_th_.joinable()) enqueue_th_.join();
  if (drain_th_dev_.joinable()) drain_th_dev_.join();
  if (drain_th_tpt_.joinable()) drain_th_tpt_.join();
  if (drain_th_signal_.joinable()) drain_th_signal_.join();
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


CollectiveOpHandle SprayExecutor::submit(CollectiveConfig const& cfg,
                                         void* input, void* output,
                                         void* scratch) {
  TiledResult tiled = build_tiled(cfg, input == output);

  std::lock_guard lock(runs_mutex_);

  // Backpressure: spin until a run slot opens up
  while (active_runs_.load(std::memory_order_acquire) >= max_concurrent_runs_) {
    runs_mutex_.unlock();
    std::this_thread::yield();
    runs_mutex_.lock();
  }

  auto h = next_handle_++;
  if (tiled.ops.empty()) {
    auto run = std::make_unique<SprayRun>();
    run->status.store(CollectiveOpStatus::Completed, std::memory_order_release);
    runs_[h] = std::move(run);
    return h;
  }

  auto run = std::make_unique<SprayRun>();
  active_runs_.fetch_add(1, std::memory_order_release);
  run->status.store(CollectiveOpStatus::Running, std::memory_order_release);
  run->tiled = std::move(tiled);
  run->input_buf_id = get_or_register_buf(input, run->tiled.input_bytes);
  run->output_buf_id = get_or_register_buf(output, run->tiled.output_bytes);
  run->scratch_buf_id =
      get_or_register_buf(scratch, run->tiled.staging_bytes_required);
  run->submitted.resize(run->tiled.ops.size(), 0);

  // Build flat successor table and atomic indegree for lock-free drain
  size_t nops = run->tiled.ops.size();
  run->indegree.resize(nops);

  // Pass 1: count successors per op
  std::vector<uint32_t> succ_count(nops, 0);
  for (uint32_t i = 0; i < nops; ++i) {
    __atomic_store_n(&run->indegree[i],
                     static_cast<uint32_t>(run->tiled.ops[i].deps.size()),
                     __ATOMIC_RELAXED);
    for (uint32_t dep : run->tiled.ops[i].deps) ++succ_count[dep];
  }

  // Pass 2: build flat successor table
  run->successor_off.resize(nops + 1);
  uint32_t off = 0;
  for (uint32_t i = 0; i < nops; ++i) {
    run->successor_off[i] = off;
    off += succ_count[i];
  }
  run->successor_off[nops] = off;
  run->successor_data.resize(off);

  // Fill
  std::vector<uint32_t> pos = run->successor_off;
  for (uint32_t i = 0; i < nops; ++i) {
    for (uint32_t dep : run->tiled.ops[i].deps)
      run->successor_data[pos[dep]++] = i;
  }

  // Seed ready ring with initial indegree==0 ops
  for (uint32_t i = 0; i < nops; ++i) {
    if (run->tiled.ops[i].deps.empty()) run->ready_ring.push(i);
  }

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

  auto check_complete = [&]() {
    if (run->done_count.load(std::memory_order_acquire) >=
        run->tiled.ops.size())
      run->status.store(CollectiveOpStatus::Completed,
                        std::memory_order_release);
  };

  if (to.count() == 0) {
    int spin = 0;
    int sleep_us = 100;
    while (run->status.load(std::memory_order_acquire) ==
           CollectiveOpStatus::Running) {
      if (spin < 1000) {
        ++spin;
        std::this_thread::yield();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        sleep_us = std::min(sleep_us * 2, 10000);
      }
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
    if (spin < 1000) {
      ++spin;
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
      sleep_us = std::min(sleep_us * 2, 10000);
    }
    check_complete();
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
    uint32_t op = run.ready_ring.pop();
    if (op == ~0u) break;
    if (!run.submitted[op]) run.ready.push_back(op);
  }
}

void SprayExecutor::enqueue_to_ring(SprayRun& run) {
  if (run.ready.empty()) return;

  run.dev_cmds.clear();
  run.tpt_cmds.clear();
  std::vector<uint32_t> dev_idx;
  std::vector<uint32_t> tpt_idx;

  for (uint32_t idx : run.ready) {
    Cmd c = make_cmd(run.tiled.ops[idx], run.tiled.reduction, run.input_buf_id,
                     run.output_buf_id, run.scratch_buf_id);

    // LB decision: for Put ops, pick IPC vs RDMA dynamically
    if (c.kind == ExecOpKind::Put && c.dst_peer != ~0u) {
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

    if (c.kind == ExecOpKind::Signal || c.kind == ExecOpKind::WaitSignal) {
      uint32_t be_idx;
      if (signal_be_->do_enqueue(&cwi.cmd, 1, &be_idx) > 0) {
        sig_caller_map_[be_idx & (kCallerMapSize - 1)].store(
            cwi.caller_id, std::memory_order_release);
        run.submitted[idx] = 1;
      }
      continue;
    }

    if (c.kind == ExecOpKind::Put) {
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
      uint32_t be_idx;
      size_t n = device_be_->do_enqueue(&run.dev_cmds[off].cmd, 1, &be_idx);
      if (n == 0) break;
      dev_caller_map_[be_idx & (kCallerMapSize - 1)].store(
          run.dev_cmds[off].caller_id, std::memory_order_release);
      run.submitted[dev_idx[off]] = 1;
      ++off;
    }
  }

  // Submit transport batch, stop on backpressure
  {
    size_t off = 0;
    while (off < run.tpt_cmds.size()) {
      uint32_t be_idx;
      size_t n = tpt_be_->do_enqueue(&run.tpt_cmds[off].cmd, 1, &be_idx);
      if (n == 0) break;
      tpt_caller_map_[be_idx & (kCallerMapSize - 1)].store(
          run.tpt_cmds[off].caller_id, std::memory_order_release);
      run.submitted[tpt_idx[off]] = 1;
      ++off;
    }
  }
}


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
        any = true;
      }
    }
    if (!any) std::this_thread::yield();
  }
}

void SprayExecutor::drain_dev_loop() {
  uint32_t be_buf[256];
  uint32_t caller_buf[256];
  while (!stop_) {
    size_t n = device_be_->do_drain(be_buf, 256);
    if (n == 0) {
      std::this_thread::yield();
      continue;
    }
    size_t valid = 0;
    for (size_t i = 0; i < n; ++i) {
      uint32_t caller_id;
      while (
          (caller_id = dev_caller_map_[be_buf[i] & (kCallerMapSize - 1)].load(
               std::memory_order_acquire)) == kMapSlotEmpty) {
        if (stop_) break;
        std::this_thread::yield();
      }
      if (stop_) return;
      caller_buf[valid++] = caller_id;
      dev_caller_map_[be_buf[i] & (kCallerMapSize - 1)].store(
          kMapSlotEmpty, std::memory_order_relaxed);
    }
    drain_batch(caller_buf, valid, [](auto&, uint32_t) {});
    check_completions_();
  }
}

void SprayExecutor::check_completions_() {
  std::lock_guard lock(runs_mutex_);
  for (auto& [h, run] : runs_) {
    if (run->status.load(std::memory_order_acquire) !=
        CollectiveOpStatus::Running)
      continue;
    if (run->done_count.load(std::memory_order_acquire) >=
        run->tiled.ops.size()) {
      run->status.store(CollectiveOpStatus::Completed,
                        std::memory_order_release);
      active_runs_.fetch_sub(1, std::memory_order_release);
    }
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
  uint32_t be_buf[256];
  uint32_t caller_buf[256];
  while (!stop_) {
    size_t nd = tpt_be_->do_drain(be_buf, 256);
    if (nd == 0) {
      for (int s = 0; s < 16 && !stop_; ++s) _mm_pause();
      std::this_thread::yield();
      continue;
    }

    size_t valid = 0;
    for (size_t i = 0; i < nd; ++i) {
      uint32_t caller_id;
      while (
          (caller_id = tpt_caller_map_[be_buf[i] & (kCallerMapSize - 1)].load(
               std::memory_order_acquire)) == kMapSlotEmpty) {
        if (stop_) break;
        std::this_thread::yield();
      }
      if (stop_) return;
      caller_buf[valid++] = caller_id;
      tpt_caller_map_[be_buf[i] & (kCallerMapSize - 1)].store(
          kMapSlotEmpty, std::memory_order_relaxed);
    }
    drain_batch(caller_buf, valid, [this](auto& m, uint32_t) {
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
    check_completions_();
  }
}

void SprayExecutor::drain_signal_loop() {
  uint32_t be_buf[256];
  uint32_t caller_buf[256];
  while (!stop_) {
    size_t ns = signal_be_->do_drain(be_buf, 256);
    if (ns == 0) {
      for (int s = 0; s < 16 && !stop_; ++s) _mm_pause();
      std::this_thread::yield();
      continue;
    }
    size_t valid = 0;
    for (size_t i = 0; i < ns; ++i) {
      uint32_t caller_id;
      while (
          (caller_id = sig_caller_map_[be_buf[i] & (kCallerMapSize - 1)].load(
               std::memory_order_acquire)) == kMapSlotEmpty) {
        if (stop_) break;
        std::this_thread::yield();
      }
      if (stop_) return;
      caller_buf[valid++] = caller_id;
      sig_caller_map_[be_buf[i] & (kCallerMapSize - 1)].store(
          kMapSlotEmpty, std::memory_order_relaxed);
    }
    drain_batch(caller_buf, valid, [](auto&, uint32_t) {});
    check_completions_();
  }
}

}  // namespace CCL
}  // namespace UKernel
