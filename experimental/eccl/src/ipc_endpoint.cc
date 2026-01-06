#include "transport.h"
#include "util/util.h"

IPCEndpoint::IPCEndpoint(std::shared_ptr<Config> config, Communicator* comm)
    : config_(config), comm_(comm) {
  task_ring_ = uccl::create_ring(sizeof(IpcTask), kTaskRingSize);
  stop_.store(false);
  proxy_thread_ = std::thread([this] { proxy_thread_func(); });

  // int n_streams = std::max(1, (int)ucclParamNumGpuRtStreams()); // ?ucclParamNumGpuRtStreams
  int n_streams = 2;
  GPU_RT_CHECK(gpuSetDevice(comm->local_rank_));
  ipc_streams_.resize(n_streams);
  for (int i = 0; i < n_streams; ++i) {
    GPU_RT_CHECK(
        gpuStreamCreateWithFlags(&ipc_streams_[i], gpuStreamNonBlocking));
  }
}

IPCEndpoint::~IPCEndpoint() {
  stop_.store(true);
  cv_.notify_all();
  if (proxy_thread_.joinable()) proxy_thread_.join();
  if (task_ring_) { free(task_ring_); }
}

bool IPCEndpoint::connect_to(int rank) { return comm_->uds_->connect_to(rank, 30000); }

bool IPCEndpoint::accept_from(int rank) { return comm_->uds_->accept_from(rank, 30000); }

bool IPCEndpoint::send_async(int to_rank, std::shared_ptr<Request> creq) {
  if (!creq || creq->len == 0) return false;
  creq->pending_signaled.store(1, std::memory_order_relaxed);
  creq->running.store(true, std::memory_order_release);

  IpcTask t{IpcTaskType::SEND, to_rank, creq, 0, 0};

  while (jring_mp_enqueue_bulk(task_ring_, &t, 1, nullptr) != 1) {
    std::this_thread::yield();
  }
  pending_.fetch_add(1, std::memory_order_relaxed);
  cv_.notify_one();

  // std::cout << "produce IPC send creq to task_ring_" << std::endl;
  return true;
}

bool IPCEndpoint::recv_async(int from_rank, std::shared_ptr<Request> creq) {
  if (!creq || creq->len == 0) return false;
  creq->pending_signaled.store(1, std::memory_order_relaxed);
  creq->running.store(true, std::memory_order_release);

  IpcTask t{IpcTaskType::RECV, from_rank, creq, 0, 0};

  while (jring_mp_enqueue_bulk(task_ring_, &t, 1, nullptr) != 1) {
    std::this_thread::yield();
  }
  pending_.fetch_add(1, std::memory_order_relaxed);
  cv_.notify_one();

  // std::cout << "produce IPC recv creq to task_ring_" << std::endl;
  return true;
}

// TODO send_, recv_
bool IPCEndpoint::send_(int to_rank, std::shared_ptr<Request> creq) {
  CHECK(creq && creq->buf != nullptr) << "send_ipc: data pointer is null!";

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

  void* base = nullptr;
  GPU_RT_CHECK(gpuSetDevice(got.remote_gpu_idx_));
  GPU_RT_CHECK(gpuIpcOpenMemHandle(&base, got.handle,
                                   gpuIpcMemLazyEnablePeerAccess));
  void* dst_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) +
                                          got.offset);

  int num_streams =
      std::min(ipc_streams_.size(),
               creq->len < kIpcSizePerEngine ? 1 : (size_t)creq->len / kIpcSizePerEngine);
  size_t chunk_size = creq->len / num_streams;

  for (int i = 0; i < num_streams; ++i) {
    // Split data and dst_ptr into n_streams chunks
    void* chunk_data = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(creq->buf) + i * chunk_size);
    void* chunk_dst_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(dst_ptr) + i * chunk_size);
    auto copy_size = i == num_streams - 1 ? creq->len - i * chunk_size : chunk_size;
    // Works for both intra-GPU and inter-GPU copy
    GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst_ptr, chunk_data, copy_size,
                                gpuMemcpyDeviceToDevice, ipc_streams_[i]));
  }

  for (auto& stream : ipc_streams_) {
    GPU_RT_CHECK(gpuStreamSynchronize(stream));
  }

  // Notify receiver of completion
  comm_->uds_->send_ack(to_rank, 0, 1);

  // We close all IPC memory handles later when releasing this endpoint.
  return true;
}

bool IPCEndpoint::recv_(int from_rank, std::shared_ptr<Request> creq) {
  CHECK(creq && creq->buf != nullptr) << "recv_ipc: data pointer is null!";

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(comm_->local_rank_));
  // Generate IPC memory handle for our receive buffer
  IpcCacheWire transfer_info = {};  // Initialize to zero
  transfer_info.size = creq->len;
  transfer_info.is_send = 0;
  transfer_info.remote_gpu_idx_ = comm_->local_rank_;
  GPU_RT_CHECK(
      gpuIpcGetMemHandle(&transfer_info.handle, reinterpret_cast<void*>(creq->buf)));

  // Getting the base address.
  void* base = nullptr;
  size_t base_size;
  GPU_RT_CHECK(gpuMemGetAddressRange(&base, &base_size, creq->buf));
  auto data_offset =
      reinterpret_cast<uintptr_t>(creq->buf) - reinterpret_cast<uintptr_t>(base);
  transfer_info.offset = data_offset;

  comm_->uds_->send_ipc_cache(from_rank, 0, transfer_info);

  // Wait Notify of sender's completion
  uint32_t status = 0;
  uint64_t out_seq = 0;
  uint64_t expect_seq = 0;
  comm_->uds_->recv_ack(from_rank, &status, &out_seq, 5000, expect_seq);
  CHECK_EQ(out_seq, expect_seq) << "Sender reported failure";

  return true;
}

void IPCEndpoint::proxy_thread_func() {

  while (!stop_.load(std::memory_order_relaxed)) {
    // Wait until there is work
    {
      std::unique_lock<std::mutex> lk(cv_mu_);
      cv_.wait(lk, [&] {
        return stop_.load(std::memory_order_relaxed) ||
               pending_.load(std::memory_order_relaxed) > 0;
      });
    }
    if (stop_.load(std::memory_order_relaxed)) break;

    // pop one and run it to completion.
    IpcTask t;
    if (jring_sc_dequeue_bulk(task_ring_, &t, 1, nullptr) != 1) {
      // just continue to next wait.
      continue;
    }
    pending_.fetch_sub(1, std::memory_order_relaxed);

    bool ok = false;
    if (t.type == IpcTaskType::SEND) {
      std::cout << "consume a IPC send creq from task_ring_" << std::endl;
      ok = send_(t.peer_rank, t.req);
    } else {
      std::cout << "consume a IPC recv creq from task_ring_" << std::endl;
      ok = recv_(t.peer_rank, t.req);
    }

    // Tasks on an IPCEndpoint are strictly serialized and order-preserving,
    // so it is safe to process the current task directly.
    if (t.req) {
      if (!ok) t.req->failed.store(true, std::memory_order_release);
      t.req->running.store(false, std::memory_order_release);
      t.req->on_comm_done(ok);
    }
  }

  // Drain leftover tasks to avoid leaks
  while (true) {
    IpcTask t;
    if (jring_mc_dequeue_bulk(task_ring_, &t, 1, nullptr) != 1) break;
    if (t.req) t.req->on_comm_done(false);
  }
}
