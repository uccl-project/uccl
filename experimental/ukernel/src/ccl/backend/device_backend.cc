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

namespace {
// CCL ScalarType → device-kernel DataType. The device layer supports
// Int8/Int32/Int64/Fp16/Fp32/Fp64/Bf16. UInt32/UInt64 map to the signed
// types: Sum/Prod are bit-exact (two's complement), but Max/Min on
// unsigned data compares signed — the shim rejects that combination
// (see nccl.cc). Int16 has no device kernel; the shim and Python
// bindings never produce it, so reject loudly if it ever appears.
Device::DataType to_device_dtype(ScalarType t) {
  switch (t) {
    case ScalarType::UInt8:
    case ScalarType::Int8:
      return Device::DataType::Int8;
    case ScalarType::Int32:
      return Device::DataType::Int32;
    case ScalarType::Int64:
      return Device::DataType::Int64;
    case ScalarType::Float16:
      return Device::DataType::Fp16;
    case ScalarType::Float32:
      return Device::DataType::Fp32;
    case ScalarType::Float64:
      return Device::DataType::Fp64;
    case ScalarType::BFloat16:
      return Device::DataType::Bf16;
    case ScalarType::Int16:
      throw std::invalid_argument(
          "DeviceBackend: Int16 has no device-kernel reduce");
  }
  return Device::DataType::Fp32;
}
}  // namespace

DeviceBackend::DeviceBackend(DeviceBackendConfig const& cfg) : cfg_(cfg) {
  GPU_RT_CHECK(gpuGetDevice(&device_idx_));
  GPU_RT_CHECK(gpuDeviceGetAttribute(&sm_count_, gpuDevAttrMultiProcessorCount,
                                     device_idx_));
  GPU_RT_CHECK(gpuDeviceGetAttribute(&host_atomic_supported_,
                                     gpuDevAttrHostNativeAtomicSupported,
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
    {
      std::lock_guard<std::mutex> lk(Device::deviceApiMutex());
      GPU_RT_CHECK(gpuSetDevice(device_idx_));
    }
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
  // Warm-up launch: create (then immediately destroy) worker 0 at init
  // while the CUDA context is quiescent (no executor threads, no IPC
  // copies yet). This loads the persistent-kernel module and exercises
  // the launch path; without it, the process's FIRST kernel launch
  // happens on the drain thread while the context is busy and hangs the
  // driver (CUDA 13.3, observed locally — cuLaunchKernel spins with the
  // GPU idle). Destroying it keeps all-CE collectives at zero
  // device-worker SM occupancy; the first real device op lazily creates
  // a fresh worker (a post-warm-up launch is safe).
  if (cfg_.max_fifos > 0) {
    if (!worker_pool_->createWorker(0, cfg_.blocks_per_worker)) {
      throw std::runtime_error(
          "DeviceBackend: failed to warm up worker 0 with "
          "blocks_per_worker=" +
          std::to_string(cfg_.blocks_per_worker));
    }
    worker_pool_->waitWorker(0);
    worker_pool_->destroyWorker(0);
  }
}

void DeviceBackend::ensure_worker(uint32_t fid) {
  if (worker_pool_->isWorkerBound(fid)) return;
  // Runs on the pinned device-drain thread (or a user thread in wait()
  // that save/restores around do_drain). createWorker is CAS-guarded on
  // the fifo's bound flag, so concurrent callers are safe: only one
  // binds, the others observe isWorkerBound and return.
  if (!worker_pool_->createWorker(fid, cfg_.blocks_per_worker)) {
    if (!worker_pool_->isWorkerBound(fid)) {
      throw std::runtime_error(
          "DeviceBackend: failed to create worker " + std::to_string(fid) +
          " with blocks_per_worker=" +
          std::to_string(cfg_.blocks_per_worker) +
          " (exceeds this GPU's SM count, or the FIFO is already bound)");
    }
  }
  worker_pool_->waitWorker(fid);
}
bool DeviceBackend::can_fuse_put_signal(int peer) const {
  return host_atomic_supported_ && comm_ &&
         comm_->ipc_signal_ring_device_ptr(peer) != nullptr;
}

bool DeviceBackend::build_task(Cmd const& c, Device::TaskArgs& args,
                               Device::TaskType& tt) {
  args.bytes = c.bytes;
  args.src_rank = (c.src_peer != ~0u) ? (int)c.src_peer : -1;
  args.dst_rank = (c.dst_peer != ~0u) ? (int)c.dst_peer : -1;
  args.src_device = device_idx_;
  args.dst_device = device_idx_;
  bool src_ok = (c.src_buf == 0), dst_ok = (c.dst_buf == 0);
  // A 3-way reduce must resolve its local Input contribution.
  bool src2_ok = !(c.flags & kCmdFlagReduce3Way);
  // A fused reduce+copy must resolve its peer accum target.
  bool copy_dst_ok = !(c.flags & kCmdFlagReduceCopy);

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
        if (comm_->try_resolve_remote_ipc_pointer((int)c.src_peer, c.src_buf, 0,
                                                  need, &cached,
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
        char* base =
            (char*)(ipc.is_local ? (void*)ipc.base_addr : ipc.direct_ptr);
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
        if (comm_->try_resolve_remote_ipc_pointer((int)c.dst_peer, c.dst_buf, 0,
                                                  need, &cached, &cached_dev)) {
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
        char* base =
            (char*)(ipc.is_local ? (void*)ipc.base_addr : ipc.direct_ptr);
        if (base) {
          char* ptr = base + ipc.base_offset;
          if (c.dst_buf < kMaxLocalBufs) local_ptr_cache_[c.dst_buf] = ptr;
          args.dst = ptr + c.dst_off;
          dst_ok = true;
        }
      }
    }
  }
  if (c.flags & kCmdFlagReduce3Way) {
    // Fused out-of-place reduce: src2 is this rank's local Input
    // contribution at the same offset as the shard.
    void* p2 = nullptr;
    if (c.src2_buf < kMaxLocalBufs && local_ptr_cache_[c.src2_buf]) {
      p2 = local_ptr_cache_[c.src2_buf];
    } else if (comm_) {
      auto ipc = comm_->get_ipc(c.src2_buf);
      char* base =
          (char*)(ipc.is_local ? (void*)ipc.base_addr : ipc.direct_ptr);
      if (base) {
        p2 = base + ipc.base_offset;
        if (c.src2_buf < kMaxLocalBufs)
          local_ptr_cache_[c.src2_buf] = p2;
      }
    }
    if (p2) {
      args.src2 = (char*)p2 + c.src_off;
      args.taskFlags |= Device::TaskArgs::kFlagReduce3Way;
      src2_ok = true;
    }
  }
  args.set_red_type(c.redop == ReductionKind::None  ? Device::ReduceType::None
                    : c.redop == ReductionKind::Sum ? Device::ReduceType::Sum
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
    void* ring =
        comm_ ? comm_->ipc_signal_ring_device_ptr(static_cast<int>(c.dst_peer))
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

  if (c.flags & kCmdFlagReduceCopy) {
    // Fused reduce+copy: after the reduce, copy dst to the peer's
    // accumulation buffer. The data-ready signal is a separate
    // host-written Signal op (the peer signal ring is not GPU-mapped on
    // B300 — gpuDevAttrHostNativeAtomicSupported is 0).
    void* pcd = nullptr;
    int cd_dev = device_idx_;
    for (auto& e : resolved_remote_cache_) {
      if (e.remote_rank == (int)c.copy_dst_peer && e.buffer_id == c.copy_dst_buf) {
        pcd = e.ptr;
        cd_dev = e.device_idx;
        break;
      }
    }
    if (!pcd && comm_) {
      size_t const need = c.copy_dst_off + c.bytes;
      if (comm_->try_resolve_remote_ipc_pointer(
              (int)c.copy_dst_peer, c.copy_dst_buf, 0, need, &pcd, &cd_dev)) {
        resolved_remote_cache_.push_back(
            {(int)c.copy_dst_peer, c.copy_dst_buf, pcd, cd_dev});
      }
    }
    if (pcd) {
      args.dst2 = (char*)pcd + c.copy_dst_off;
      args.taskFlags |= Device::TaskArgs::kFlagReduceCopy;
      if (c.flag_slot != ~0u) {
        // Device-completion flag: write the signal tag into the peer's
        // flag slot (plain store + fence, no atomics) when the task
        // completes.
        void* flag_area =
            comm_ ? comm_->ipc_device_flag_ptr((int)c.copy_dst_peer) : nullptr;
        if (flag_area) {
          args.src2 = static_cast<char*>(flag_area) +
                      static_cast<size_t>(c.flag_slot) * sizeof(uint64_t);
          args.signal_tag = c.tag;
          args.taskFlags |= Device::TaskArgs::kFlagSignalAfter;
          copy_dst_ok = true;
        }
      } else {
        copy_dst_ok = true;
      }
    }
  }
  if (c.flags & kCmdFlagCopySignal) {
    // Fused AG copy: the device copy task also writes the completion
    // flag slot (peer's flag area) when it finishes.
    if (c.flag_slot != ~0u) {
      void* flag_area =
          comm_ ? comm_->ipc_device_flag_ptr((int)c.dst_peer) : nullptr;
      if (flag_area) {
        args.src2 = static_cast<char*>(flag_area) +
                    static_cast<size_t>(c.flag_slot) * sizeof(uint64_t);
        args.signal_tag = c.tag;
        args.taskFlags |= Device::TaskArgs::kFlagSignalAfter;
      }
    }
  }

  if (!src_ok || !dst_ok || !src2_ok || !copy_dst_ok) {
    throw std::runtime_error(
        std::string("[DeviceBackend] unresolved buffer ptr src_ok=") +
        std::to_string((int)src_ok) + " dst_ok=" + std::to_string((int)dst_ok) +
        " src2_ok=" + std::to_string((int)src2_ok) +
        " copy_dst_ok=" + std::to_string((int)copy_dst_ok) +
        " src_buf=" + std::to_string(c.src_buf) +
        " src2_buf=" + std::to_string(c.src2_buf) +
        " copy_dst_buf=" + std::to_string(c.copy_dst_buf) +
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
        args, tt, to_device_dtype(cmds[accepted].dtype), 0);
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
  {
    std::lock_guard<std::mutex> lk(Device::deviceApiMutex());
    GPU_RT_CHECK(gpuGetDevice(&prev_device));
  }
  if (prev_device != device_idx_) {
    std::lock_guard<std::mutex> lk(Device::deviceApiMutex());
    GPU_RT_CHECK(gpuSetDevice(device_idx_));
  }
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
      // Lazy worker binding: the first device op of a collective (or
      // process) binds the persistent worker here, on the pinned drain
      // thread — never on the enqueue thread, whose first-ever kernel
      // launch hangs this driver (CUDA 13.3, observed locally).
      if (!q.empty() && !worker_pool_->isWorkerBound(fid)) ensure_worker(fid);
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
  if (prev_device != device_idx_) {
    // Restoring the caller's device can fail when its CUDA context cannot
    // be (re)created under memory pressure (observed on a VLLM-co-resident
    // B300: cudaSetDevice -> out of memory at 256M). The caller's next
    // ensure_runtime() re-pins to device_idx_, so degrade to a warning
    // instead of aborting the drain path.
    std::lock_guard<std::mutex> lk(Device::deviceApiMutex());
    gpuError_t err = gpuSetDevice(prev_device);
    if (err != gpuSuccess) {
      std::fprintf(stderr,
                   "[dev-drain] warning: restore device %d failed (%s); "
                   "thread left on device %d\n",
                   prev_device, gpuGetErrorString(err), device_idx_);
    }
  }
  return count;
}

size_t DeviceBackend::capacity() const {
  return (size_t)cfg_.max_fifos * cfg_.fifo_capacity;
}

}  // namespace CCL
}  // namespace UKernel
