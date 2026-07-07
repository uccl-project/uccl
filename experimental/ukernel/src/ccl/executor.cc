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
  auto role_src = op.src_buf_role;
  auto role_dst = op.dst_buf_role;
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

  uint32_t in_id = 0, out_id = 0, scr_id = 0;
  if (owned_comm_ && peer_setup_fn_) {
    peer_setup_fn_(owned_comm_.get(), cfg.rank, world_size_);
  }

  if (owned_comm_) {
    in_id = get_or_register_buf(input, tiled.input_bytes);
    out_id = get_or_register_buf(output, tiled.output_bytes);
    if (scratch && tiled.staging_bytes_required > 0)
      scr_id = get_or_register_buf(scratch, tiled.staging_bytes_required);

    for (int p = 0; p < world_size_; ++p) {
      if (p == cfg.rank) continue;
      if (in_id && resolve_buf_fn_)
        resolve_buf_fn_(owned_comm_.get(), p, world_size_, in_id);
      if (out_id && out_id != in_id && resolve_buf_fn_)
        resolve_buf_fn_(owned_comm_.get(), p, world_size_, out_id);
    }
  }

  std::lock_guard lock(runs_mutex_);

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
  run->input_buf_id = in_id;
  run->output_buf_id = out_id;
  run->scratch_buf_id = scr_id;
  run->submitted.resize(run->tiled.ops.size(), 0);

  size_t nops = run->tiled.ops.size();
  run->init_ready_ring(nops);
  run->indegree.resize(nops);

  std::vector<uint32_t> succ_count(nops, 0);
  for (uint32_t i = 0; i < nops; ++i) {
    __atomic_store_n(&run->indegree[i],
                     static_cast<uint32_t>(run->tiled.ops[i].deps.size()),
                     __ATOMIC_RELAXED);
    for (uint32_t dep : run->tiled.ops[i].deps) ++succ_count[dep];
  }

  run->successor_off.resize(nops + 1);
  uint32_t off = 0;
  for (uint32_t i = 0; i < nops; ++i) {
    run->successor_off[i] = off;
    off += succ_count[i];
  }
  run->successor_off[nops] = off;
  run->successor_data.resize(off);

  std::vector<uint32_t> pos = run->successor_off;
  for (uint32_t i = 0; i < nops; ++i) {
    for (uint32_t dep : run->tiled.ops[i].deps)
      run->successor_data[pos[dep]++] = i;
  }

  size_t initial = 0;
  for (uint32_t i = 0; i < nops; ++i) {
    if (run->tiled.ops[i].deps.empty()) { run->push_ready(i); ++initial; }
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
  for (auto& [tag, be_idx] : it->second->be_slots) {
    if (tag == 0)
      dev_slots_.release(be_idx);
    else if (tag == 1)
      tpt_slots_.release(be_idx);
    else
      sig_slots_.release(be_idx);
  }
  runs_.erase(it);
}

void SprayExecutor::collect_ready(SprayRun& run) {
  run.ready.clear();
  for (;;) {
    uint32_t op = run.pop_ready();
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

    if (c.kind == ExecOpKind::Put && c.dst_peer != ~0u) {
      auto peer = static_cast<int>(c.dst_peer);
      c.transport = static_cast<uint8_t>(pick_put_path(peer));
    }

    if (c.kind == ExecOpKind::Signal || c.kind == ExecOpKind::WaitSignal) {
      uint32_t be_idx;
      if (signal_be_->do_enqueue(&c, 1, &be_idx) > 0) {
        sig_slots_.write(be_idx, &run, idx, 0);
        run.be_slots.emplace_back(2, be_idx);
        run.submitted[idx] = 1;
      } else {
        run.push_ready(idx);
      }
      continue;
    }

    if (c.kind == ExecOpKind::Put && c.transport != 0) {
      run.tpt_cmds.push_back(c);
      tpt_idx.push_back(idx);
    } else {
      run.dev_cmds.push_back(c);
      dev_idx.push_back(idx);
    }
  }

  // Submit device batch
  {
    size_t off = 0;
    while (off < run.dev_cmds.size()) {
      uint32_t be_idx;
      size_t n = device_be_->do_enqueue(&run.dev_cmds[off], 1, &be_idx);
      if (n == 0) break;
      dev_slots_.write(be_idx, &run, dev_idx[off], 0);
      run.be_slots.emplace_back(0, be_idx);
      run.submitted[dev_idx[off]] = 1;
      ++off;
    }
    for (size_t i = off; i < run.dev_cmds.size(); ++i)
      run.push_ready(dev_idx[i]);
  }

  // Submit transport batch
  {
    size_t off = 0;
    while (off < run.tpt_cmds.size()) {
      uint32_t be_idx;
      size_t n = tpt_be_->do_enqueue(&run.tpt_cmds[off], 1, &be_idx);
      if (n == 0) break;
      tpt_slots_.write(be_idx, &run, tpt_idx[off],
                       run.tpt_cmds[off].transport);
      run.be_slots.emplace_back(1, be_idx);
      run.submitted[tpt_idx[off]] = 1;
      ++off;
    }
    for (size_t i = off; i < run.tpt_cmds.size(); ++i)
      run.push_ready(tpt_idx[i]);
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
  BeSlot* slot_buf[256];
  while (!stop_) {
    size_t n = device_be_->do_drain(be_buf, 256);
    if (n == 0) {
      std::this_thread::yield();
      continue;
    }
    size_t valid = 0;
    for (size_t i = 0; i < n; ++i) {
      auto* s = dev_slots_.wait(be_buf[i], stop_);
      if (!s) return;
      slot_buf[valid++] = s;
    }
    drain_batch(slot_buf, valid, [this](BeSlot& s) {
      auto& op = s.run->tiled.ops[s.op_idx];
      if (op.kind == ExecOpKind::Put && op.dst_peer != ~0u) {
        int peer = static_cast<int>(op.dst_peer);
        if (peer >= 0 && peer < world_size_)
          update_path_metrics(tpt_metrics_[peer].device, s.enqueue_ns);
      }
    });
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

Transport::PeerTransportKind SprayExecutor::pick_put_path(int peer) {
  // FIXME: temporary — always IPC while debugging 3-way path selection
  if (tpt_metrics_ && peer >= 0 && peer < world_size_)
    tpt_metrics_[peer].ipc.inflight.fetch_add(1, std::memory_order_relaxed);
  return Transport::PeerTransportKind::Ipc;

  if (!tpt_metrics_ || peer < 0 || peer >= world_size_) {
    return Transport::PeerTransportKind::Unknown;
  }
  if (!same_host_fn_ || !same_host_fn_(owned_comm_.get(), peer)) {
    tpt_metrics_[peer].rdma.inflight.fetch_add(1, std::memory_order_relaxed);
    return Transport::PeerTransportKind::Rdma;
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

  Transport::PeerTransportKind choice;
  PathMetrics* chosen;
  if (ic <= dc && ic <= rc) {
    choice = Transport::PeerTransportKind::Ipc;
    chosen = &pm.ipc;
  } else if (dc <= rc) {
    choice = Transport::PeerTransportKind::Unknown;  // device
    chosen = &pm.device;
  } else {
    choice = Transport::PeerTransportKind::Rdma;
    chosen = &pm.rdma;
  }
  chosen->inflight.fetch_add(1, std::memory_order_relaxed);
  return choice;
}

void SprayExecutor::drain_tpt_loop() {
  uint32_t be_buf[256];
  BeSlot* slot_buf[256];
  while (!stop_) {
    size_t nd = tpt_be_->do_drain(be_buf, 256);
    if (nd == 0) {
      for (int s = 0; s < 16 && !stop_; ++s) _mm_pause();
      std::this_thread::yield();
      continue;
    }

    size_t valid = 0;
    for (size_t i = 0; i < nd; ++i) {
      auto* s = tpt_slots_.wait(be_buf[i], stop_);
      if (!s) return;
      slot_buf[valid++] = s;
    }
    drain_batch(slot_buf, valid, [this](BeSlot& s) {
      auto transport = s.transport;
      if (transport == 0) return;
      auto tpt = static_cast<Transport::PeerTransportKind>(transport);
      if (tpt == Transport::PeerTransportKind::Unknown) return;
      int peer = static_cast<int>(s.run->tiled.ops[s.op_idx].dst_peer);
      if (peer < 0 || peer >= world_size_) return;
      auto& m = (tpt == Transport::PeerTransportKind::Ipc)
                    ? tpt_metrics_[peer].ipc
                    : tpt_metrics_[peer].rdma;
      update_path_metrics(m, s.enqueue_ns);
    });
    check_completions_();
  }
}

void SprayExecutor::drain_signal_loop() {
  uint32_t be_buf[256];
  BeSlot* slot_buf[256];
  while (!stop_) {
    size_t ns = signal_be_->do_drain(be_buf, 256);
    if (ns == 0) {
      for (int s = 0; s < 16 && !stop_; ++s) _mm_pause();
      std::this_thread::yield();
      continue;
    }
    size_t valid = 0;
    for (size_t i = 0; i < ns; ++i) {
      auto* s = sig_slots_.wait(be_buf[i], stop_);
      if (!s) return;
      slot_buf[valid++] = s;
    }
    drain_batch(slot_buf, valid, [](BeSlot&) {});
    check_completions_();
  }
}

}  // namespace CCL
}  // namespace UKernel
