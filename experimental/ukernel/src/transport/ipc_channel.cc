#include "communicator.h"
#include "ipc_channel.h"
#include "util/util.h"

namespace UKernel {
namespace Transport {

IpcChannel::IpcChannel(Communicator* comm) : comm_(comm) {
  task_ring_ = uccl::create_ring(sizeof(IpcTask), kTaskRingSize);
  stop_.store(false);
  proxy_thread_ = std::thread([this] { proxy_thread_func(); });

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
  if (proxy_thread_.joinable()) proxy_thread_.join();
  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  GPU_RT_CHECK(gpuSetDevice(comm_->local_gpu_idx_));
  for (auto& stream : ipc_streams_) {
    if (stream != nullptr) GPU_RT_CHECK(gpuStreamDestroy(stream));
  }
  GPU_RT_CHECK(gpuSetDevice(orig_device));
  if (task_ring_) {
    free(task_ring_);
  }
}

bool IpcChannel::connect_to(int rank) { return comm_->uds_->connect_to(rank, 30000); }

bool IpcChannel::accept_from(int rank) {
  return comm_->uds_->accept_from(rank, 30000);
}

bool IpcChannel::send_async(int to_rank, std::shared_ptr<Request> creq) {
  if (!creq || creq->len == 0) return false;
  creq->pending_signaled.store(1, std::memory_order_relaxed);
  creq->running.store(true, std::memory_order_release);

  IpcTask t{IpcTaskType::SEND, to_rank, creq};

  while (jring_mp_enqueue_bulk(task_ring_, &t, 1, nullptr) != 1) {
    std::this_thread::yield();
  }
  pending_.fetch_add(1, std::memory_order_relaxed);
  cv_.notify_one();
  return true;
}

bool IpcChannel::recv_async(int from_rank, std::shared_ptr<Request> creq) {
  if (!creq || creq->len == 0) return false;
  creq->pending_signaled.store(1, std::memory_order_relaxed);
  creq->running.store(true, std::memory_order_release);

  IpcTask t{IpcTaskType::RECV, from_rank, creq};

  while (jring_mp_enqueue_bulk(task_ring_, &t, 1, nullptr) != 1) {
    std::this_thread::yield();
  }
  pending_.fetch_add(1, std::memory_order_relaxed);
  cv_.notify_one();
  return true;
}

bool IpcChannel::send_(int to_rank, std::shared_ptr<Request> creq) {
  UCCL_CHECK(creq && creq->buf != nullptr) << "send_ipc: data pointer is null!";

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  IpcCacheWire got{};
  uint64_t seq = 0;
  if (!comm_->uds_->recv_ipc_cache(to_rank, got, &seq, 50000)) {
    std::cerr << "[ERROR] recv_ipc_cache(" << to_rank << ") failed\n";
    return false;
  }

  GPU_RT_CHECK(gpuSetDevice(got.remote_gpu_idx_));

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

  int num_streams = std::min(ipc_streams_.size(),
                             creq->len < kIpcSizePerEngine
                                 ? 1
                                 : (size_t)creq->len / kIpcSizePerEngine);
  size_t chunk_size = creq->len / num_streams;

  GPU_RT_CHECK(gpuSetDevice(comm_->local_gpu_idx_));
  for (int i = 0; i < num_streams; ++i) {
    void* chunk_data = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(src_ptr) + i * chunk_size);
    void* chunk_dst_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(dst_ptr) + i * chunk_size);
    auto copy_size =
        i == num_streams - 1 ? creq->len - i * chunk_size : chunk_size;
    GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst_ptr, chunk_data, copy_size,
                                gpuMemcpyDeviceToDevice, ipc_streams_[i]));
  }

  for (auto& stream : ipc_streams_) {
    GPU_RT_CHECK(gpuStreamSynchronize(stream));
  }

  if (!comm_->uds_->send_ack(to_rank, seq, 1)) {
    std::cerr << "[ERROR] send_ack(" << to_rank << ") failed\n";
    return false;
  }

  return true;
}

bool IpcChannel::recv_(int from_rank, std::shared_ptr<Request> creq) {
  UCCL_CHECK(creq && creq->buf != nullptr) << "recv_ipc: data pointer is null!";

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(comm_->local_gpu_idx_));
  IpcCacheWire transfer_info = {};
  transfer_info.size = creq->len;
  transfer_info.is_send = 0;
  transfer_info.remote_gpu_idx_ = comm_->local_gpu_idx_;
  void* actual_dst = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(creq->buf) + creq->offset);
  GPU_RT_CHECK(gpuIpcGetMemHandle(&transfer_info.handle, actual_dst));

  void* base = nullptr;
  size_t base_size = 0;
  GPU_RT_CHECK(gpuMemGetAddressRange(&base, &base_size, actual_dst));
  transfer_info.offset = reinterpret_cast<uintptr_t>(actual_dst) -
                         reinterpret_cast<uintptr_t>(base);

  if (!comm_->uds_->send_ipc_cache(from_rank, creq->id, transfer_info)) {
    std::cerr << "[ERROR] send_ipc_cache(" << from_rank << ") failed\n";
    return false;
  }

  uint32_t status = 0;
  uint64_t out_seq = 0;
  uint64_t expect_seq = creq->id;
  if (!comm_->uds_->recv_ack(from_rank, &status, &out_seq, 5000, expect_seq)) {
    std::cerr << "[ERROR] recv_ack(" << from_rank << ") failed\n";
    return false;
  }
  if (out_seq != expect_seq || status != 1) {
    std::cerr << "[ERROR] sender completion ack invalid: seq=" << out_seq
              << " status=" << status << std::endl;
    return false;
  }

  return true;
}

void IpcChannel::proxy_thread_func() {
  while (!stop_.load(std::memory_order_relaxed)) {
    {
      std::unique_lock<std::mutex> lk(cv_mu_);
      cv_.wait(lk, [&] {
        return stop_.load(std::memory_order_relaxed) ||
               pending_.load(std::memory_order_relaxed) > 0;
      });
    }
    if (stop_.load(std::memory_order_relaxed)) break;

    IpcTask t;
    if (jring_sc_dequeue_bulk(task_ring_, &t, 1, nullptr) != 1) {
      continue;
    }
    pending_.fetch_sub(1, std::memory_order_relaxed);

    bool ok = false;
    if (t.type == IpcTaskType::SEND) {
      ok = send_(t.peer_rank, t.req);
    } else {
      ok = recv_(t.peer_rank, t.req);
    }

    if (t.req) {
      if (!ok) t.req->failed.store(true, std::memory_order_release);
      t.req->running.store(false, std::memory_order_release);
      t.req->on_comm_done();
    }
  }

  while (true) {
    IpcTask t;
    if (jring_mc_dequeue_bulk(task_ring_, &t, 1, nullptr) != 1) break;
    if (t.req) {
      t.req->failed.store(true, std::memory_order_release);
      t.req->on_comm_done();
    }
  }
}

}  // namespace Transport
}  // namespace UKernel
