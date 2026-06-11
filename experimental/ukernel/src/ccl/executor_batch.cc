#include "executor_batch.h"
#include "algo/chunk_graph.h"
#include "backend/backend.h"
#include "coll_config.h"
#include "utils.h"
#include <algorithm>
#include <stdexcept>
#include <thread>

namespace UKernel {
namespace CCL {

// ── Helpers ─────────────────────────────────────────────────────────────

static uint32_t buf_of(Op const& op, bool is_src) {
  auto r = buf_role(op.kind, is_src, op.copy_from_staging);
  return static_cast<uint32_t>(r) + 1;  // Input=1, Output=2, Scratch=3
}

static Cmd make_cmd(Op const& op, ReductionKind redop) {
  Cmd c;
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
  return c;
}

// ── SprayExecutor ───────────────────────────────────────────────────────

SprayExecutor::SprayExecutor(BatchExecutorBackends backends) : be_(backends) {}

SprayRun* SprayExecutor::get(CollectiveOpHandle h) {
  auto it = runs_.find(h);
  return it != runs_.end() ? &it->second : nullptr;
}

CollectiveOpStatus SprayExecutor::status(CollectiveOpHandle h) const {
  auto it = runs_.find(h);
  return it != runs_.end() ? it->second.status : CollectiveOpStatus::Completed;
}

size_t SprayExecutor::active_count() const {
  size_t n = 0;
  for (auto& [h, r] : runs_)
    if (r.status == CollectiveOpStatus::Running) ++n;
  return n;
}

std::string SprayExecutor::error_message(CollectiveOpHandle h) const {
  auto it = runs_.find(h);
  return it != runs_.end() ? it->second.error : std::string{};
}

// ── submit ──────────────────────────────────────────────────────────────

CollectiveOpHandle SprayExecutor::submit_allreduce(
    CollectiveConfig const& cfg, void* input, void* output, void* scratch) {
  CollectiveConfig c = cfg;
  c.kind = CollKind::AllReduceRing;
  TiledResult tiled = build_tiled(c, input == output);
  if (tiled.ops.empty()) {
    auto h = next_handle_++;
    runs_[h] = {.status = CollectiveOpStatus::Completed};
    return h;
  }

  // Init backends with buffer specs
  BufSpec bufs[3] = {
      {input, tiled.input_bytes},
      {output, tiled.output_bytes},
      {scratch, tiled.staging_bytes_required},
  };
  if (be_.transport) be_.transport->init(bufs);
  if (be_.device) be_.device->init(bufs);

  auto h = next_handle_++;
  SprayRun run;
  run.status = CollectiveOpStatus::Running;
  run.tiled = std::move(tiled);
  run.input = input; run.output = output; run.scratch = scratch;
  run.done.resize(run.tiled.ops.size(), false);
  runs_[h] = std::move(run);
  return h;
}

CollectiveOpHandle SprayExecutor::submit_alltoall(
    CollectiveConfig const& cfg, void* input, void* output, void* scratch) {
  CollectiveConfig c = cfg;
  c.kind = CollKind::AllToAllPairwise;
  return submit_allreduce(c, input, output, scratch);
}

// ── poll / progress / wait ──────────────────────────────────────────────

void SprayExecutor::advance(SprayRun& run) {
  if (run.status != CollectiveOpStatus::Running) return;
  if (run.done_count >= run.tiled.ops.size()) {
    run.status = CollectiveOpStatus::Completed;
    return;
  }
  collect_ready(run);
  enqueue_ready(run);
  drain_done(run);
  if (run.done_count >= run.tiled.ops.size())
    run.status = CollectiveOpStatus::Completed;
}

bool SprayExecutor::poll(CollectiveOpHandle h) {
  auto* r = get(h);
  if (!r) return true;
  if (r->status != CollectiveOpStatus::Running) return true;
  advance(*r);
  return r->status == CollectiveOpStatus::Completed ||
         r->status == CollectiveOpStatus::Failed;
}

void SprayExecutor::progress() {
  for (auto& [h, run] : runs_) advance(run);
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
  auto it = runs_.find(h);
  if (it == runs_.end()) return;
  if (it->second.status == CollectiveOpStatus::Queued ||
      it->second.status == CollectiveOpStatus::Running)
    throw std::logic_error("cannot release running collective");
  runs_.erase(it);
}

// ── Three-phase advance ─────────────────────────────────────────────────

void SprayExecutor::collect_ready(SprayRun& run) {
  run.ready.clear();
  auto& ops = run.tiled.ops;
  auto& layer = run.tiled.layers;

  for (uint32_t l = run.next_layer; l < layer.size(); ++l) {
    bool ld = true;
    for (uint32_t op : layer[l]) {
      if (run.done[op]) continue;
      ld = false;
      bool ok = true;
      for (uint32_t d : ops[op].deps)
        if (!run.done[d]) { ok = false; break; }
      if (ok) run.ready.push_back(op);
    }
    if (ld) run.next_layer = l + 1;
  }
}

void SprayExecutor::enqueue_ready(SprayRun& run) {
  // Build command lists grouped by backend
  std::vector<Cmd> d_cmds, t_cmds;
  std::vector<uint32_t> d_map, t_map;

  for (uint32_t idx : run.ready) {
    Cmd c = make_cmd(run.tiled.ops[idx], run.tiled.reduction);
    if (be_.transport && be_.transport->supports(c.kind)) {
      t_cmds.push_back(c); t_map.push_back(idx);
    } else if (be_.device) {
      d_cmds.push_back(c); d_map.push_back(idx);
    }
  }

  // Spray to device backend — respect capacity
  if (be_.device && !d_cmds.empty()) {
    size_t cap = be_.device->capacity();
    size_t n = be_.device->enqueue(d_cmds.data(), std::min(d_cmds.size(), cap));
    run.dev_map.assign(d_map.begin(), d_map.begin() + n);
  }

  // Spray to transport backend
  if (be_.transport && !t_cmds.empty()) {
    size_t cap = be_.transport->capacity();
    size_t n = be_.transport->enqueue(t_cmds.data(), std::min(t_cmds.size(), cap));
    run.tpt_map.assign(t_map.begin(), t_map.begin() + n);
  }
}

void SprayExecutor::drain_done(SprayRun& run) {
  uint32_t comp[256];
  if (be_.device) {
    size_t n = be_.device->drain(comp, 256);
    for (size_t i = 0; i < n; ++i) {
      uint32_t op = run.dev_map[comp[i]];
      if (!run.done[op]) { run.done[op] = true; ++run.done_count; }
    }
  }
  if (be_.transport) {
    size_t n = be_.transport->drain(comp, 256);
    for (size_t i = 0; i < n; ++i) {
      uint32_t op = run.tpt_map[comp[i]];
      if (!run.done[op]) { run.done[op] = true; ++run.done_count; }
    }
  }
  // Yield if stalled
  if (run.done_count < run.tiled.ops.size())
    std::this_thread::yield();
}

// ── run_tiled (sync, for tests) ─────────────────────────────────────────

void SprayExecutor::run_tiled(TiledResult const& tiled,
                               void* input, void* output, void* scratch) {
  BufSpec bufs[3] = {
      {input, tiled.input_bytes},
      {output, tiled.output_bytes},
      {scratch, tiled.staging_bytes_required},
  };
  if (be_.transport) be_.transport->init(bufs);
  if (be_.device) be_.device->init(bufs);

  SprayRun run;
  run.status = CollectiveOpStatus::Running;
  run.tiled = tiled;
  run.input = input; run.output = output; run.scratch = scratch;
  run.done.resize(tiled.ops.size(), false);

  while (run.status == CollectiveOpStatus::Running) {
    advance(run);
    if (run.status == CollectiveOpStatus::Running)
      std::this_thread::yield();
  }
  if (run.status == CollectiveOpStatus::Failed)
    throw std::runtime_error(run.error);
}

}  // namespace CCL
}  // namespace UKernel
