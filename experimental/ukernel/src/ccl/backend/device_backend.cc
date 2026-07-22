#include "device_backend.h"
#include "../../../include/transport.h"
#include "../../device/task.h"
#include "../../device/worker.h"
#include "gpu_rt.h"
#include "util/uk_debug.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace UKernel {
namespace CCL {

DeviceBackend::DeviceBackend(DeviceBackendConfig const& cfg) : cfg_(cfg) {
  GPU_RT_CHECK(gpuGetDevice(&device_idx_));
  GPU_RT_CHECK(gpuDeviceGetAttribute(&sm_count_, gpuDevAttrMultiProcessorCount,
                                     device_idx_));
  pending_by_fifo_.resize(cfg_.max_fifos);
  ensure_runtime();
}

DeviceBackend::~DeviceBackend() {
  worker_pool_.reset();
  if (owns_task_manager_) {
    Device::TaskManager::instance().release();
    owns_task_manager_ = false;
  }
}

bool DeviceBackend::supports(ExecOpKind kind) const {
  return kind == ExecOpKind::Put || kind == ExecOpKind::Reduce;
}

void DeviceBackend::ensure_runtime() {
  static thread_local int tls_last_device = -1;
  if (tls_last_device != device_idx_) {
    GPU_RT_CHECK(gpuSetDevice(device_idx_));
    tls_last_device = device_idx_;
  }
  if (!Device::TaskManager::instance().inited()) {
    Device::TaskManager::instance().init(cfg_.task_capacity);
    owns_task_manager_ = true;
  }
  if (worker_pool_) return;
  Device::WorkerPool::Config wc;
  wc.numMaxWorkers = cfg_.max_fifos;
  wc.threadsPerBlock = cfg_.threads_per_block;
  wc.fifoCapacity = cfg_.fifo_capacity;
  wc.smemSize = cfg_.smem_size;
  wc.idleExitAfterUs = cfg_.idle_exit_after_us;
  worker_pool_ = std::make_unique<Device::WorkerPool>(wc);
  // Pre-create all workers
  for (uint32_t i = 0; i < cfg_.max_fifos; ++i) {
    worker_pool_->createWorker(i, cfg_.blocks_per_worker);
    worker_pool_->waitWorker(i);
  }
}
bool DeviceBackend::can_fuse_put_signal(int peer) const {
  return comm_ && comm_->ipc_signal_ring_device_ptr(peer) != nullptr;
}

bool DeviceBackend::build_task(Cmd const& c, Device::TaskArgs& args,
                               Device::TaskType& tt) {
  args.bytes = c.bytes;
  args.src_rank = (c.src_peer != ~0u) ? (int)c.src_peer : -1;
  args.dst_rank = (c.dst_peer != ~0u) ? (int)c.dst_peer : -1;
  args.src_device = device_idx_;
  args.dst_device = device_idx_;
  bool src_ok = (c.src_buf == 0), dst_ok = (c.dst_buf == 0);

  if (c.src_buf > 0) {
    if (c.src_peer != ~0u && comm_) {
      // Check cache first — resolved ptr never changes once set.
      // Cache stores the base pointer (offset=0); add c.src_off at
      // each use because try_resolve_remote_ipc_pointer already bakes
      // the passed offset into the returned pointer.
      void* cached = nullptr;
      for (auto& e : resolved_remote_cache_) {
        if (e.remote_rank == (int)c.src_peer && e.buffer_id == c.src_buf) {
          cached = e.ptr;
          args.src_device = e.device_idx;
          break;
        }
      }
      if (cached) {
        args.src = (char*)cached + c.src_off;
        src_ok = true;
      } else {
        size_t const need = c.src_off + c.bytes;
        if (comm_->try_resolve_remote_ipc_pointer(
                (int)c.src_peer, c.src_buf, 0, need, &cached,
                &args.src_device)) {
          resolved_remote_cache_.push_back(
              {(int)c.src_peer, c.src_buf, cached, args.src_device});
          args.src = (char*)cached + c.src_off;
          src_ok = true;
        }
      }
    } else if (comm_) {
      if (c.src_buf < kMaxLocalBufs && local_ptr_cache_[c.src_buf]) {
        args.src = (char*)local_ptr_cache_[c.src_buf] + c.src_off;
        src_ok = true;
      } else {
        auto ipc = comm_->get_ipc(c.src_buf);
        // A local IPC item describes the whole allocation (base_addr)
        // plus the buffer's offset within it (base_offset) — e.g. a
        // tensor inside a torch caching-allocator segment. The kernel
        // address must include base_offset, same as
        // try_resolve_remote_ipc_pointer does for remote items.
        char* base = (char*)(ipc.is_local ? (void*)ipc.base_addr
                                          : ipc.direct_ptr);
        if (base) {
          char* ptr = base + ipc.base_offset;
          if (c.src_buf < kMaxLocalBufs) local_ptr_cache_[c.src_buf] = ptr;
          args.src = ptr + c.src_off;
          src_ok = true;
        }
      }
    }
  }
  if (c.dst_buf > 0) {
    if (c.dst_peer != ~0u && comm_) {
      void* cached = nullptr;
      int cached_dev = device_idx_;
      for (auto& e : resolved_remote_cache_) {
        if (e.remote_rank == (int)c.dst_peer && e.buffer_id == c.dst_buf) {
          cached = e.ptr;
          cached_dev = e.device_idx;
          break;
        }
      }
      if (cached) {
        args.dst = (char*)cached + c.dst_off;
        args.dst_device = cached_dev;
        dst_ok = true;
      } else {
        size_t const need = c.dst_off + c.bytes;
        if (comm_->try_resolve_remote_ipc_pointer(
                (int)c.dst_peer, c.dst_buf, 0, need, &cached,
                &cached_dev)) {
          resolved_remote_cache_.push_back(
              {(int)c.dst_peer, c.dst_buf, cached, cached_dev});
          args.dst = (char*)cached + c.dst_off;
          args.dst_device = cached_dev;
          dst_ok = true;
        }
      }
    } else if (comm_) {
      if (c.dst_buf < kMaxLocalBufs && local_ptr_cache_[c.dst_buf]) {
        args.dst = (char*)local_ptr_cache_[c.dst_buf] + c.dst_off;
        dst_ok = true;
      } else {
        auto ipc = comm_->get_ipc(c.dst_buf);
        // Same base_offset requirement as the src path above.
        char* base = (char*)(ipc.is_local ? (void*)ipc.base_addr
                                          : ipc.direct_ptr);
        if (base) {
          char* ptr = base + ipc.base_offset;
          if (c.dst_buf < kMaxLocalBufs) local_ptr_cache_[c.dst_buf] = ptr;
          args.dst = ptr + c.dst_off;
          dst_ok = true;
        }
      }
    }
  }
  args.set_red_type(c.redop == ReductionKind::None ? Device::ReduceType::None
                    : c.redop == ReductionKind::Sum
                        ? Device::ReduceType::Sum
                        : Device::ReduceType::Sum);

  switch (c.kind) {
    case ExecOpKind::Put:
      tt = Device::TaskType::CollCopy;
      break;
    case ExecOpKind::Reduce:
      tt = Device::TaskType::CollReduce;
      break;
    default:
      return false;
  }

  if (c.flags & kCmdFlagPutSignal) {
    // Fused PutSignal: the kernel writes the tag into the peer's shm
    // signal ring after the copy (same channel as host-sent signals,
    // so the receiver stays a CPU-side poll).
    void* ring = comm_ ? comm_->ipc_signal_ring_device_ptr(
                             static_cast<int>(c.dst_peer))
                       : nullptr;
    if (!ring) {
      throw std::runtime_error(
          "[DeviceBackend] PutSignal flagged but peer signal ring is not "
          "GPU-mapped (dst_peer=" +
          std::to_string(c.dst_peer) + ")");
    }
    tt = Device::TaskType::CollPut;
    args.src2 = ring;
    args.redTypeRaw = c.tag;
  }

  if (!src_ok || !dst_ok) {
    throw std::runtime_error(
        std::string("[DeviceBackend] unresolved buffer ptr src_ok=") +
        std::to_string((int)src_ok) +
        " dst_ok=" + std::to_string((int)dst_ok) +
        " src_buf=" + std::to_string(c.src_buf) +
        " dst_buf=" + std::to_string(c.dst_buf));
  }
  return true;
}

uint32_t DeviceBackend::reserve_slot() {
  return cmd_next_.fetch_add(1, std::memory_order_relaxed);
}

bool DeviceBackend::do_enqueue_reserved(Cmd const& c, uint32_t be_idx) {
  return do_enqueue_reserved_batch(&c, &be_idx, 1) == 1;
}

size_t DeviceBackend::do_enqueue_reserved_batch(Cmd const* cmds,
                                                uint32_t const* be_idx,
                                                size_t n) {
  ensure_runtime();
  size_t accepted = 0;
  // Records of ops submitted in this batch, appended to pending_ under a
  // single lock at the end. Single-writer assumption: only the executor's
  // enqueue thread calls this (same as BeSlotTable).
  std::vector<CmdRec> recs;
  recs.reserve(n);
  while (accepted < n) {
    Cmd const& c = cmds[accepted];

    Device::TaskArgs args{};
    Device::TaskType tt;
    if (!build_task(c, args, tt)) {
      ++accepted;
      continue;
    }

    // Capacity check + FIFO pick under lock (cheap), heavy ops outside.
    uint32_t fid;
    {
      std::lock_guard<std::mutex> lk(pending_mu_);
      if (pending_total_ + recs.size() >= capacity()) {
        UK_DBG(UK_DBG_LVL_EXEC, "[dev-enq] capacity full %zu/%zu",
               pending_total_ + recs.size(), capacity());
        break;
      }
      fid = next_fifo_ % cfg_.max_fifos;
      next_fifo_ = (next_fifo_ + 1) % cfg_.max_fifos;
    }

    auto task = Device::TaskManager::instance().create_task(
        args, tt, Device::DataType::Fp32, 0);
    if (task.type_u8() == 0) {
      UK_DBG(UK_DBG_LVL_EXEC, "[dev-enq] create_task failed (pool empty?)");
      break;
    }

    uint64_t tid = worker_pool_->enqueue(task, fid);
    if (tid == Device::WorkerPool::kInvalidTaskId) {
      UK_DBG(UK_DBG_LVL_EXEC, "[dev-enq] worker enqueue failed fifo=%u", fid);
      break;
    }
    recs.push_back({fid, tid, task.args_index(), be_idx[accepted]});
    ++accepted;
  }
  if (!recs.empty()) {
    std::lock_guard<std::mutex> lk(pending_mu_);
    for (auto& r : recs) pending_by_fifo_[r.fifo_id].push_back(r);
    pending_total_ += recs.size();
  }
  return accepted;
}

size_t DeviceBackend::do_enqueue(Cmd const* cmds, size_t n,
                                 uint32_t* out_indices) {
  size_t accepted = 0;
  for (size_t i = 0; i < n; ++i) {
    uint32_t idx = reserve_slot();
    if (!do_enqueue_reserved(cmds[i], idx)) break;
    if (out_indices) out_indices[accepted] = idx;
    ++accepted;
  }
  return accepted;
}

size_t DeviceBackend::do_drain(uint32_t* completed, size_t max) {
  // do_drain may run on user threads (SprayExecutor::wait drives
  // progress); save/restore the caller's CUDA device around it.
  int prev_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&prev_device));
  if (prev_device != device_idx_) GPU_RT_CHECK(gpuSetDevice(device_idx_));
  size_t count = 0;
  // Args slots to recycle, freed in one batch after pending_mu_ is
  // released. Draining is capped by the buffer; callers loop anyway.
  uint32_t args_buf[256];
  {
    std::lock_guard<std::mutex> lk(pending_mu_);
    // A task that raced the kernel's idle exit gets its worker
    // relaunched here (enqueue-time checks are not atomic with the
    // kernel's exit decision).
    if (cfg_.idle_exit_after_us > 0) {
      for (uint32_t fid = 0; fid < cfg_.max_fifos; ++fid)
        if (!pending_by_fifo_[fid].empty())
          worker_pool_->relaunch_if_exited(fid);
    }
    // Per FIFO, completions are in-order, so pop only the done prefix of
    // each queue — cost is O(completed + fifos), not O(pending).
    for (uint32_t fid = 0; fid < cfg_.max_fifos && count < max &&
                           count < sizeof(args_buf) / sizeof(args_buf[0]);
         ++fid) {
      auto& q = pending_by_fifo_[fid];
      while (!q.empty() && count < max &&
             count < sizeof(args_buf) / sizeof(args_buf[0]) &&
             worker_pool_->is_done(q.front().task_id, fid)) {
        args_buf[count] = q.front().args_id;
        completed[count++] = q.front().cmd_idx;
        q.pop_front();
      }
    }
    pending_total_ -= count;

    // Stall forensics: pending work but nothing drains — dump fifo
    // head/tail vs the pending queue front to tell "kernel stuck on a
    // task" from "host accounting bug".
    static int stall_iters = 0;
    if (count == 0 && pending_total_ > 0 && (++stall_iters % 5000) == 0) {
      for (uint32_t fid = 0; fid < cfg_.max_fifos; ++fid) {
        if (pending_by_fifo_[fid].empty()) continue;
        auto ht = worker_pool_->fifo_head_tail(fid);
        std::fprintf(stderr,
                     "[dev-stall] fifo%u pending=%zu front_tid=%llu "
                     "head=%llu tail=%llu\n",
                     fid, pending_by_fifo_[fid].size(),
                     (unsigned long long)pending_by_fifo_[fid].front().task_id,
                     (unsigned long long)ht.first,
                     (unsigned long long)ht.second);
      }
    } else if (count > 0) {
      stall_iters = 0;
    }
  }
  Device::TaskManager::instance().free_task_args_batch(args_buf, count);
  if (prev_device != device_idx_) GPU_RT_CHECK(gpuSetDevice(prev_device));
  return count;
}

size_t DeviceBackend::capacity() const {
  return (size_t)cfg_.max_fifos * cfg_.fifo_capacity;
}

}  // namespace CCL
}  // namespace UKernel
