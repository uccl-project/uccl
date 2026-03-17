#include "communicator.h"
#include "ipc_channel.h"
#include "util/util.h"
#include <algorithm>

namespace UKernel {
namespace Transport {

IpcChannel::IpcChannel(Communicator* comm) : comm_(comm) {
  send_task_ring_ = uccl::create_ring(sizeof(IpcTask), kTaskRingSize);
  recv_task_ring_ = uccl::create_ring(sizeof(IpcTask), kTaskRingSize);
  stop_.store(false);
  send_thread_ = std::thread([this] { send_thread_func(); });
  recv_thread_ = std::thread([this] { recv_thread_func(); });

  int n_streams = 2;
  GPU_RT_CHECK(gpuSetDevice(comm->local_gpu_idx_));
  ipc_streams_.resize(n_streams);
  for (int i = 0; i < n_streams; ++i) {
    GPU_RT_CHECK(
        gpuStreamCreateWithFlags(&ipc_streams_[i], gpuStreamNonBlocking));
  }
}

IpcChannel::~IpcChannel() {
  stop_.store(true);
  cv_.notify_all();
  if (send_thread_.joinable()) send_thread_.join();
  if (recv_thread_.joinable()) recv_thread_.join();

  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  GPU_RT_CHECK(gpuSetDevice(comm_->local_gpu_idx_));
  for (auto& stream : ipc_streams_) {
    if (stream != nullptr) GPU_RT_CHECK(gpuStreamDestroy(stream));
  }
  GPU_RT_CHECK(gpuSetDevice(orig_device));

  if (send_task_ring_) free(send_task_ring_);
  if (recv_task_ring_) free(recv_task_ring_);
}

bool IpcChannel::connect_to(int rank) {
  return comm_->shm_control_->connect_to(rank, 30000);
}

bool IpcChannel::accept_from(int rank) {
  return comm_->shm_control_->accept_from(rank, 30000);
}

bool IpcChannel::send_async(int to_rank, std::shared_ptr<Request> creq) {
  if (!creq || creq->len == 0) return false;
  creq->pending_signaled.store(1, std::memory_order_relaxed);
  creq->running.store(true, std::memory_order_release);

  IpcTask task{IpcTaskType::SEND, to_rank, creq.get()};
  while (jring_mp_enqueue_bulk(send_task_ring_, &task, 1, nullptr) != 1) {
    std::this_thread::yield();
  }
  pending_send_.fetch_add(1, std::memory_order_relaxed);
  cv_.notify_all();
  return true;
}

bool IpcChannel::recv_async(int from_rank, std::shared_ptr<Request> creq) {
  if (!creq || creq->len == 0) return false;
  creq->pending_signaled.store(1, std::memory_order_relaxed);
  creq->running.store(true, std::memory_order_release);

  IpcTask task{IpcTaskType::RECV, from_rank, creq.get()};
  while (jring_mp_enqueue_bulk(recv_task_ring_, &task, 1, nullptr) != 1) {
    std::this_thread::yield();
  }
  pending_recv_.fetch_add(1, std::memory_order_relaxed);
  cv_.notify_all();
  return true;
}

bool IpcChannel::send_one(int to_rank, Request* creq) {
  UCCL_CHECK(creq && creq->buf != nullptr) << "send_ipc: data pointer is null!";

  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  IpcCacheWire got{};
  uint64_t seq = 0;
  if (!comm_->shm_control_->recv_ipc_cache(to_rank, got, &seq,
                                           kIpcControlTimeoutMs, creq->id)) {
    std::cerr << "[ERROR] recv_ipc_cache(" << to_rank
              << ") failed for req " << creq->id << std::endl;
    return false;
  }

  GPU_RT_CHECK(gpuSetDevice(comm_->local_gpu_idx_));

  IpcCache cache = comm_->get_remote_ipc_cache(to_rank, got.handle);
  void* base = cache.direct_ptr;
  if (base == nullptr) {
    GPU_RT_CHECK(
        gpuIpcOpenMemHandle(&base, got.handle, gpuIpcMemLazyEnablePeerAccess));

    IpcCache new_cache{};
    new_cache.handle = got.handle;
    new_cache.is_send = got.is_send;
    new_cache.direct_ptr = base;
    new_cache.offset = got.offset;
    new_cache.size = got.size;
    new_cache.device_idx = static_cast<int>(got.remote_gpu_idx_);
    comm_->register_remote_ipc_cache(to_rank, got.handle, new_cache);
  }

  void* dst_ptr =
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + got.offset);
  void* src_ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(creq->buf) + creq->offset);

  size_t n_streams = std::min(
      ipc_streams_.size(),
      creq->len < kIpcSizePerEngine ? size_t{1}
                                    : std::max<size_t>(size_t{1},
                                                       creq->len / kIpcSizePerEngine));
  size_t chunk_size = creq->len / n_streams;
  for (size_t i = 0; i < n_streams; ++i) {
    void* chunk_src = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(src_ptr) + i * chunk_size);
    void* chunk_dst = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(dst_ptr) + i * chunk_size);
    size_t copy_size =
        i == n_streams - 1 ? creq->len - i * chunk_size : chunk_size;
    if (got.remote_gpu_idx_ == static_cast<uint32_t>(comm_->local_gpu_idx_)) {
      GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst, chunk_src, copy_size,
                                  gpuMemcpyDeviceToDevice, ipc_streams_[i]));
    } else {
      GPU_RT_CHECK(gpuMemcpyPeerAsync(chunk_dst, got.remote_gpu_idx_, chunk_src,
                                      comm_->local_gpu_idx_, copy_size,
                                      ipc_streams_[i]));
    }
  }
  for (size_t i = 0; i < n_streams; ++i) {
    GPU_RT_CHECK(gpuStreamSynchronize(ipc_streams_[i]));
  }

  if (!comm_->shm_control_->send_ack(to_rank, seq, 1)) {
    std::cerr << "[ERROR] send_ack(" << to_rank << ") failed for req "
              << creq->id << std::endl;
    return false;
  }
  return true;
}

bool IpcChannel::recv_one(int from_rank, Request* creq) {
  UCCL_CHECK(creq && creq->buf != nullptr) << "recv_ipc: data pointer is null!";

  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(comm_->local_gpu_idx_));
  IpcCacheWire transfer_info{};
  transfer_info.size = creq->len;
  transfer_info.is_send = 0;
  transfer_info.remote_gpu_idx_ = comm_->local_gpu_idx_;
  void* actual_dst = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(creq->buf) + creq->offset);

  void* base = nullptr;
  size_t base_size = 0;
  GPU_RT_CHECK(gpuMemGetAddressRange(&base, &base_size, actual_dst));
  GPU_RT_CHECK(gpuIpcGetMemHandle(&transfer_info.handle, base));
  transfer_info.offset = reinterpret_cast<uintptr_t>(actual_dst) -
                         reinterpret_cast<uintptr_t>(base);

  if (!comm_->shm_control_->send_ipc_cache(from_rank, creq->id, transfer_info)) {
    std::cerr << "[ERROR] send_ipc_cache(" << from_rank
              << ") failed for req " << creq->id << std::endl;
    return false;
  }

  uint32_t status = 0;
  uint64_t out_seq = 0;
  if (!comm_->shm_control_->recv_ack(from_rank, &status, &out_seq,
                                     kIpcControlTimeoutMs, creq->id)) {
    std::cerr << "[ERROR] recv_ack(" << from_rank << ") failed for req "
              << creq->id << std::endl;
    return false;
  }
  if (out_seq != creq->id || status != 1) {
    std::cerr << "[ERROR] sender completion ack invalid: seq=" << out_seq
              << " status=" << status << " req=" << creq->id << std::endl;
    return false;
  }
  return true;
}

void IpcChannel::complete_task(Request* req, bool ok) {
  if (!req) return;
  if (!ok) req->failed.store(true, std::memory_order_release);
  req->running.store(false, std::memory_order_release);
  req->on_comm_done();
}

void IpcChannel::send_thread_func() {
  while (!stop_.load(std::memory_order_relaxed)) {
    {
      std::unique_lock<std::mutex> lk(cv_mu_);
      cv_.wait(lk, [&] {
        return stop_.load(std::memory_order_relaxed) ||
               pending_send_.load(std::memory_order_relaxed) > 0;
      });
    }
    if (stop_.load(std::memory_order_relaxed)) break;

    IpcTask task{};
    if (jring_sc_dequeue_bulk(send_task_ring_, &task, 1, nullptr) != 1) {
      continue;
    }
    pending_send_.fetch_sub(1, std::memory_order_relaxed);
    complete_task(task.req, send_one(task.peer_rank, task.req));
  }

  while (true) {
    IpcTask task{};
    if (jring_mc_dequeue_bulk(send_task_ring_, &task, 1, nullptr) != 1) break;
    complete_task(task.req, false);
  }
}

void IpcChannel::recv_thread_func() {
  while (!stop_.load(std::memory_order_relaxed)) {
    {
      std::unique_lock<std::mutex> lk(cv_mu_);
      cv_.wait(lk, [&] {
        return stop_.load(std::memory_order_relaxed) ||
               pending_recv_.load(std::memory_order_relaxed) > 0;
      });
    }
    if (stop_.load(std::memory_order_relaxed)) break;

    IpcTask task{};
    if (jring_sc_dequeue_bulk(recv_task_ring_, &task, 1, nullptr) != 1) {
      continue;
    }
    pending_recv_.fetch_sub(1, std::memory_order_relaxed);
    complete_task(task.req, recv_one(task.peer_rank, task.req));
  }

  while (true) {
    IpcTask task{};
    if (jring_mc_dequeue_bulk(recv_task_ring_, &task, 1, nullptr) != 1) break;
    complete_task(task.req, false);
  }
}

}  // namespace Transport
}  // namespace UKernel
