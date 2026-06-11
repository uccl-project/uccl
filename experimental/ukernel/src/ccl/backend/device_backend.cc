#include "device_backend.h"
#include "../../device/task.h"
#include "../../device/worker.h"
#include "../../../include/gpu_rt.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace UKernel {
namespace CCL {

DeviceBackend::DeviceBackend(DeviceBackendConfig const& cfg) : cfg_(cfg) {
  GPU_RT_CHECK(gpuGetDevice(&device_idx_));
  GPU_RT_CHECK(gpuDeviceGetAttribute(&sm_count_,
              gpuDevAttrMultiProcessorCount, device_idx_));
}

DeviceBackend::~DeviceBackend() {
  worker_pool_.reset();
  if (owns_task_manager_) {
    Device::TaskManager::instance().release();
    owns_task_manager_ = false;
  }
}

bool DeviceBackend::supports(OpKind kind) const {
  return kind == OpKind::Copy || kind == OpKind::Reduce ||
         kind == OpKind::Send || kind == OpKind::RecvReduce ||
         kind == OpKind::Recv;
}

void DeviceBackend::ensure_runtime() {
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
    worker_pool_->createWorker(i, 1);
    worker_pool_->waitWorker(i);
  }
}

void DeviceBackend::init(BufSpec bufs[3]) {
  for (int i = 0; i < 3; ++i) bufs_[i] = bufs[i];
  ensure_runtime();
  inited_ = true;
}

size_t DeviceBackend::enqueue(Cmd const* cmds, size_t n) {
  if (!inited_) return 0;

  size_t accepted = 0;
  while (accepted < n && pending_.size() < capacity()) {
    Cmd const& c = cmds[accepted];

    // Build TaskArgs
    Device::TaskArgs args{};
    args.bytes = c.bytes;
    args.src_rank = (c.src_peer != ~0u) ? (int)c.src_peer : -1;
    args.dst_rank = (c.dst_peer != ~0u) ? (int)c.dst_peer : -1;
    args.src_device = device_idx_;
    args.dst_device = device_idx_;

    // Resolve src/dst pointers from registered buffers
    if (c.src_buf > 0 && c.src_buf <= 3 && bufs_[c.src_buf - 1].ptr) {
      args.src = (char*)bufs_[c.src_buf - 1].ptr + c.src_off;
    }
    if (c.dst_buf > 0 && c.dst_buf <= 3 && bufs_[c.dst_buf - 1].ptr) {
      args.dst = (char*)bufs_[c.dst_buf - 1].ptr + c.dst_off;
    }
    args.set_red_type(c.redop == ReductionKind::None
                          ? Device::ReduceType::None
                          : c.redop == ReductionKind::Sum
                                ? Device::ReduceType::Sum
                                : Device::ReduceType::Sum);

    // Map OpKind → TaskType
    Device::TaskType tt;
    switch (c.kind) {
      case OpKind::Copy:       tt = Device::TaskType::CollCopy; break;
      case OpKind::Reduce:     tt = Device::TaskType::CollReduce; break;
      case OpKind::Send:       tt = Device::TaskType::CollSend; break;
      case OpKind::Recv:       tt = Device::TaskType::CollRecv; break;
      case OpKind::RecvReduce: tt = Device::TaskType::CollRecvReduce; break;
      default: ++accepted; continue;
    }

    // Create task and enqueue
    uint32_t cmd_idx = cmd_next_++;
    auto task = Device::TaskManager::instance().create_task(
        args, tt, Device::DataType::Fp32, 0);

    // Round-robin FIFO assignment
    uint32_t fid = next_fifo_ % cfg_.max_fifos;
    next_fifo_ = (next_fifo_ + 1) % cfg_.max_fifos;

    uint64_t tid = worker_pool_->enqueue(task, fid);
    if (tid == Device::WorkerPool::kInvalidTaskId) break; // backpressure

    pending_.push_back({fid, tid, task.args_index(), cmd_idx});
    ++accepted;
  }
  return accepted;
}

size_t DeviceBackend::drain(uint32_t* completed, size_t max) {
  size_t count = 0;
  for (size_t i = 0; i < pending_.size() && count < max; ) {
    auto& rec = pending_[i];
    if (worker_pool_->is_done(rec.task_id, rec.fifo_id)) {
      Device::TaskManager::instance().free_task_args(rec.args_id);
      completed[count++] = rec.cmd_idx;
      if (i != pending_.size() - 1)
        rec = pending_.back();
      pending_.pop_back();
    } else {
      ++i;
    }
  }
  return count;
}

size_t DeviceBackend::capacity() const {
  return (size_t)cfg_.max_fifos * cfg_.fifo_capacity;
}

void DeviceBackend::set_signal_buffers(std::vector<GpuSignalPeer> const& peers) {
  gpu_signal_bufs_ = peers;
}

}  // namespace CCL
}  // namespace UKernel
