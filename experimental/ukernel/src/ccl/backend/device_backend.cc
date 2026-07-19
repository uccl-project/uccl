#include "device_backend.h"
#include "../../../include/transport.h"
#include "../../device/task.h"
#include "../../device/worker.h"
#include "gpu_rt.h"
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
  worker_pool_ = std::make_unique<Device::WorkerPool>(wc);
  // Pre-create all workers
  for (uint32_t i = 0; i < cfg_.max_fifos; ++i) {
    worker_pool_->createWorker(i, cfg_.blocks_per_worker);
    worker_pool_->waitWorker(i);
  }
}
size_t DeviceBackend::do_enqueue(Cmd const* cmds, size_t n,
                                 uint32_t* out_indices) {
  ensure_runtime();
  size_t accepted = 0;
  while (accepted < n) {
    Cmd const& c = cmds[accepted];

    Device::TaskArgs args{};
    args.bytes = c.bytes;
    args.src_rank = (c.src_peer != ~0u) ? (int)c.src_peer : -1;
    args.dst_rank = (c.dst_peer != ~0u) ? (int)c.dst_peer : -1;
    args.src_device = device_idx_;
    args.dst_device = device_idx_;
    bool src_ok = (c.src_buf == 0), dst_ok = (c.dst_buf == 0);

    if (c.src_buf > 0) {
      if (c.src_peer != ~0u && comm_) {
        // Check cache first — resolved ptr never changes once set
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
        } else if (comm_->try_resolve_remote_ipc_pointer(
                       (int)c.src_peer, c.src_buf, c.src_off, c.bytes, &cached,
                       &args.src_device)) {
          resolved_remote_cache_.push_back(
              {(int)c.src_peer, c.src_buf, cached, args.src_device});
          args.src = (char*)cached + c.src_off;
          src_ok = true;
        }
      } else if (comm_) {
        if (c.src_buf < kMaxLocalBufs && local_ptr_cache_[c.src_buf]) {
          args.src = (char*)local_ptr_cache_[c.src_buf] + c.src_off;
          src_ok = true;
        } else {
          auto ipc = comm_->get_ipc(c.src_buf);
          void* ptr = ipc.is_local ? (void*)ipc.base_addr : ipc.direct_ptr;
          if (ptr) {
            if (c.src_buf < kMaxLocalBufs) local_ptr_cache_[c.src_buf] = ptr;
            args.src = (char*)ptr + c.src_off;
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
        } else if (comm_->try_resolve_remote_ipc_pointer(
                       (int)c.dst_peer, c.dst_buf, c.dst_off, c.bytes, &cached,
                       &cached_dev)) {
          resolved_remote_cache_.push_back(
              {(int)c.dst_peer, c.dst_buf, cached, cached_dev});
          args.dst = (char*)cached + c.dst_off;
          args.dst_device = cached_dev;
          dst_ok = true;
        }
      } else if (comm_) {
        if (c.dst_buf < kMaxLocalBufs && local_ptr_cache_[c.dst_buf]) {
          args.dst = (char*)local_ptr_cache_[c.dst_buf] + c.dst_off;
          dst_ok = true;
        } else {
          auto ipc = comm_->get_ipc(c.dst_buf);
          void* ptr = ipc.is_local ? (void*)ipc.base_addr : ipc.direct_ptr;
          if (ptr) {
            if (c.dst_buf < kMaxLocalBufs) local_ptr_cache_[c.dst_buf] = ptr;
            args.dst = (char*)ptr + c.dst_off;
            dst_ok = true;
          }
        }
      }
    }
    args.set_red_type(c.redop == ReductionKind::None ? Device::ReduceType::None
                      : c.redop == ReductionKind::Sum
                          ? Device::ReduceType::Sum
                          : Device::ReduceType::Sum);

    Device::TaskType tt;
    switch (c.kind) {
      case ExecOpKind::Put:
        tt = Device::TaskType::CollCopy;
        break;
      case ExecOpKind::Reduce:
        tt = Device::TaskType::CollReduce;
        break;
      default:
        ++accepted;
        continue;
    }

    if (!src_ok || !dst_ok) {
      throw std::runtime_error(
          std::string("[DeviceBackend] unresolved buffer ptr src_ok=") +
          std::to_string((int)src_ok) +
          " dst_ok=" + std::to_string((int)dst_ok) +
          " src_buf=" + std::to_string(c.src_buf) +
          " dst_buf=" + std::to_string(c.dst_buf));
    }

    // Reserve slot + fid under lock (cheap), heavy ops outside
    uint32_t cmd_idx;
    uint32_t fid;
    {
      std::lock_guard<std::mutex> lk(pending_mu_);
      if (pending_.size() >= capacity()) break;
      cmd_idx = cmd_next_++;
      fid = next_fifo_ % cfg_.max_fifos;
      next_fifo_ = (next_fifo_ + 1) % cfg_.max_fifos;
    }

    auto task = Device::TaskManager::instance().create_task(
        args, tt, Device::DataType::Fp32, 0);

    if (task.type_u8() == 0) { --cmd_next_; break; }

    uint64_t tid = worker_pool_->enqueue(task, fid);
    {
      std::lock_guard<std::mutex> lk(pending_mu_);
      if (tid == Device::WorkerPool::kInvalidTaskId) {
        --cmd_next_;
        break;
      }
      if (out_indices) out_indices[accepted] = cmd_idx;
      pending_.push_back({fid, tid, task.args_index(), cmd_idx});
    }
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
  std::lock_guard<std::mutex> lk(pending_mu_);
  size_t count = 0;
  for (size_t i = 0; i < pending_.size() && count < max;) {
    auto& rec = pending_[i];
    if (worker_pool_->is_done(rec.task_id, rec.fifo_id)) {
      Device::TaskManager::instance().free_task_args(rec.args_id);
      completed[count++] = rec.cmd_idx;
      if (i != pending_.size() - 1) rec = pending_.back();
      pending_.pop_back();
    } else {
      ++i;
    }
  }
  if (prev_device != device_idx_) GPU_RT_CHECK(gpuSetDevice(prev_device));
  return count;
}

size_t DeviceBackend::capacity() const {
  return (size_t)cfg_.max_fifos * cfg_.fifo_capacity;
}

}  // namespace CCL
}  // namespace UKernel
