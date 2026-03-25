#include "ipc_channel.h"
#include "communicator.h"
#include "ipc_cache.h"
#include "utils.h"
#include "util/util.h"
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

namespace {

constexpr int kIpcControlPollTimeoutMs = 50;

int remaining_timeout_ms(std::chrono::steady_clock::time_point deadline) {
  auto now = std::chrono::steady_clock::now();
  if (now >= deadline) return 0;
  return static_cast<int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
          .count());
}

constexpr uint32_t kAckStatusOkDirect = 1;
constexpr uint32_t kAckStatusOkCpuRelay = 2;

std::string relay_shm_name(int sender_rank, int receiver_rank,
                           uint64_t match_seq) {
  return uccl::Format("/uk_t_ipc_relay_%s_%d_%d_%llu",
                      generate_host_id(false).c_str(), sender_rank,
                      receiver_rank,
                      static_cast<unsigned long long>(match_seq));
}

struct RelayBufferMapping {
  std::string name;
  int fd = -1;
  void* ptr = nullptr;
  size_t bytes = 0;
  bool owner = false;
};

bool create_relay_mapping(std::string const& name, size_t bytes,
                          RelayBufferMapping& out) {
  shm_unlink(name.c_str());
  int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
  if (fd < 0) return false;
  if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    close(fd);
    shm_unlink(name.c_str());
    return false;
  }
  void* ptr =
      mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    close(fd);
    shm_unlink(name.c_str());
    return false;
  }
  out.name = name;
  out.fd = fd;
  out.ptr = ptr;
  out.bytes = bytes;
  out.owner = true;
  return true;
}

bool open_relay_mapping(std::string const& name, size_t bytes,
                        std::chrono::steady_clock::time_point deadline,
                        RelayBufferMapping& out) {
  while (true) {
    int fd = shm_open(name.c_str(), O_RDWR, 0666);
    if (fd >= 0) {
      void* ptr =
          mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if (ptr != MAP_FAILED) {
        out.name = name;
        out.fd = fd;
        out.ptr = ptr;
        out.bytes = bytes;
        out.owner = false;
        return true;
      }
      close(fd);
    }
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void close_relay_mapping(RelayBufferMapping& mapping) {
  if (mapping.ptr != nullptr) {
    munmap(mapping.ptr, mapping.bytes);
    mapping.ptr = nullptr;
  }
  if (mapping.fd >= 0) {
    close(mapping.fd);
    mapping.fd = -1;
  }
  if (mapping.owner && !mapping.name.empty()) {
    shm_unlink(mapping.name.c_str());
  }
  mapping.name.clear();
  mapping.bytes = 0;
  mapping.owner = false;
}

}  // namespace

IpcChannel::IpcChannel(Communicator* comm) : comm_(comm) {
  send_task_ring_ = uccl::create_ring(sizeof(IpcTask*), kTaskRingSize);
  recv_task_ring_ = uccl::create_ring(sizeof(IpcTask*), kTaskRingSize);
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

IpcChannel::~IpcChannel() { shutdown(); }

void IpcChannel::shutdown() {
  bool expected = false;
  if (!shutdown_started_.compare_exchange_strong(expected, true)) return;

  stop_.store(true, std::memory_order_release);
  cv_.notify_all();
  if (send_thread_.joinable()) send_thread_.join();
  if (recv_thread_.joinable()) recv_thread_.join();

  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  GPU_RT_CHECK(gpuSetDevice(comm_->local_gpu_idx_));
  for (auto& stream : ipc_streams_) {
    if (stream != nullptr) GPU_RT_CHECK(gpuStreamDestroy(stream));
  }
  ipc_streams_.clear();
  GPU_RT_CHECK(gpuSetDevice(orig_device));

  if (send_task_ring_) {
    free(send_task_ring_);
    send_task_ring_ = nullptr;
  }
  if (recv_task_ring_) {
    free(recv_task_ring_);
    recv_task_ring_ = nullptr;
  }
}

bool IpcChannel::connect_to(int rank) {
  return comm_->shm_control_->connect_to(rank, 30000);
}

bool IpcChannel::accept_from(int rank) {
  return comm_->shm_control_->accept_from(rank, 30000);
}

bool IpcChannel::send_async(int to_rank, std::shared_ptr<Request> creq) {
  if (!creq || creq->size_bytes == 0 || stop_.load(std::memory_order_acquire) ||
      send_task_ring_ == nullptr) {
    return false;
  }
  creq->mark_queued(1);

  auto* task = new IpcTask{IpcTaskType::SEND, to_rank, std::move(creq)};
  while (!stop_.load(std::memory_order_acquire) &&
         jring_mp_enqueue_bulk(send_task_ring_, &task, 1, nullptr) != 1) {
    std::this_thread::yield();
  }
  if (stop_.load(std::memory_order_acquire)) {
    delete task;
    return false;
  }
  pending_send_.fetch_add(1, std::memory_order_relaxed);
  cv_.notify_all();
  return true;
}

bool IpcChannel::recv_async(int from_rank, std::shared_ptr<Request> creq) {
  if (!creq || creq->size_bytes == 0 || stop_.load(std::memory_order_acquire) ||
      recv_task_ring_ == nullptr) {
    return false;
  }
  creq->mark_queued(1);

  auto* task = new IpcTask{IpcTaskType::RECV, from_rank, std::move(creq)};
  while (!stop_.load(std::memory_order_acquire) &&
         jring_mp_enqueue_bulk(recv_task_ring_, &task, 1, nullptr) != 1) {
    std::this_thread::yield();
  }
  if (stop_.load(std::memory_order_acquire)) {
    delete task;
    return false;
  }
  pending_recv_.fetch_add(1, std::memory_order_relaxed);
  cv_.notify_all();
  return true;
}

bool IpcChannel::send_one(int to_rank, Request* creq) {
  UCCL_CHECK(creq && creq->buffer != nullptr)
      << "send_ipc: data pointer is null!";
  creq->mark_running();

  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  IpcCacheWire got{};
  uint64_t seq = 0;
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(kIpcControlTimeoutMs);
  bool got_cache = false;
  while (!stop_.load(std::memory_order_acquire)) {
    int timeout_ms =
        std::min(kIpcControlPollTimeoutMs, remaining_timeout_ms(deadline));
    if (timeout_ms <= 0) break;
    if (comm_->shm_control_->recv_ipc_cache(to_rank, got, &seq, timeout_ms,
                                            creq->match_seq)) {
      got_cache = true;
      break;
    }
  }
  if (!got_cache) {
    std::cerr << "[ERROR] recv_ipc_cache(" << to_rank << ") failed for req "
              << creq->id << " match_seq " << creq->match_seq << std::endl;
    return false;
  }

  GPU_RT_CHECK(gpuSetDevice(comm_->local_gpu_idx_));

  bool can_use_direct_peer = (got.remote_gpu_idx_ ==
                              static_cast<uint32_t>(comm_->local_gpu_idx_));
  if (!can_use_direct_peer) {
    int can_access_peer = 0;
    GPU_RT_CHECK(gpuDeviceCanAccessPeer(
        &can_access_peer, comm_->local_gpu_idx_,
        static_cast<int>(got.remote_gpu_idx_)));
    can_use_direct_peer = (can_access_peer != 0);
  }

  if (!can_use_direct_peer) {
    RelayBufferMapping relay{};
    std::string relay_name =
        relay_shm_name(comm_->rank(), to_rank, creq->match_seq);
    if (!open_relay_mapping(relay_name, creq->size_bytes, deadline, relay)) {
      std::cerr << "[ERROR] open relay shm failed for req " << creq->id
                << " match_seq " << creq->match_seq << std::endl;
      return false;
    }
    auto relay_cleanup = uccl::finally([&]() { close_relay_mapping(relay); });

    void* src_ptr = creq->data();
    GPU_RT_CHECK(gpuMemcpy(relay.ptr, src_ptr, creq->size_bytes,
                           gpuMemcpyDeviceToHost));

    if (!comm_->shm_control_->send_ack(to_rank, seq, kAckStatusOkCpuRelay)) {
      std::cerr << "[ERROR] send relay ack(" << to_rank << ") failed for req "
                << creq->id << " match_seq " << creq->match_seq << std::endl;
      return false;
    }
    return true;
  }

  IpcCacheManager::IpcCache cache =
      comm_->get_remote_ipc_cache(to_rank, got.handle);
  void* base = cache.direct_ptr;
  if (base == nullptr) {
    GPU_RT_CHECK(
        gpuIpcOpenMemHandle(&base, got.handle, gpuIpcMemLazyEnablePeerAccess));

    IpcCacheManager::IpcCache new_cache{};
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
  void* src_ptr = creq->data();

  size_t n_streams = std::min(
      ipc_streams_.size(),
      creq->size_bytes < kIpcSizePerEngine
          ? size_t{1}
          : std::max<size_t>(size_t{1}, creq->size_bytes / kIpcSizePerEngine));
  size_t chunk_size = creq->size_bytes / n_streams;
  for (size_t i = 0; i < n_streams; ++i) {
    void* chunk_src = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(src_ptr) + i * chunk_size);
    void* chunk_dst = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(dst_ptr) + i * chunk_size);
    size_t copy_size =
        i == n_streams - 1 ? creq->size_bytes - i * chunk_size : chunk_size;
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

  if (!comm_->shm_control_->send_ack(to_rank, seq, kAckStatusOkDirect)) {
    std::cerr << "[ERROR] send_ack(" << to_rank << ") failed for req "
              << creq->id << " match_seq " << creq->match_seq << std::endl;
    return false;
  }
  return true;
}

bool IpcChannel::recv_one(int from_rank, Request* creq) {
  UCCL_CHECK(creq && creq->buffer != nullptr)
      << "recv_ipc: data pointer is null!";
  creq->mark_running();

  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(comm_->local_gpu_idx_));
  IpcCacheWire transfer_info{};
  transfer_info.size = creq->size_bytes;
  transfer_info.is_send = 0;
  transfer_info.remote_gpu_idx_ = comm_->local_gpu_idx_;
  void* actual_dst = creq->data();

  RelayBufferMapping relay{};
  std::string relay_name =
      relay_shm_name(from_rank, comm_->rank(), creq->match_seq);
  if (!create_relay_mapping(relay_name, creq->size_bytes, relay)) {
    std::cerr << "[ERROR] create relay shm failed for req " << creq->id
              << " match_seq " << creq->match_seq << std::endl;
    return false;
  }
  auto relay_cleanup = uccl::finally([&]() { close_relay_mapping(relay); });

  void* base = nullptr;
  size_t base_size = 0;
  GPU_RT_CHECK(gpuMemGetAddressRange(&base, &base_size, actual_dst));
  GPU_RT_CHECK(gpuIpcGetMemHandle(&transfer_info.handle, base));
  transfer_info.offset = reinterpret_cast<uintptr_t>(actual_dst) -
                         reinterpret_cast<uintptr_t>(base);

  if (!comm_->shm_control_->send_ipc_cache(from_rank, creq->match_seq,
                                           transfer_info)) {
    std::cerr << "[ERROR] send_ipc_cache(" << from_rank << ") failed for req "
              << creq->id << " match_seq " << creq->match_seq << std::endl;
    return false;
  }

  uint32_t status = 0;
  uint64_t out_seq = 0;
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(kIpcControlTimeoutMs);
  bool got_ack = false;
  while (!stop_.load(std::memory_order_acquire)) {
    int timeout_ms =
        std::min(kIpcControlPollTimeoutMs, remaining_timeout_ms(deadline));
    if (timeout_ms <= 0) break;
    if (comm_->shm_control_->recv_ack(from_rank, &status, &out_seq, timeout_ms,
                                      creq->match_seq)) {
      got_ack = true;
      break;
    }
  }
  if (!got_ack) {
    std::cerr << "[ERROR] recv_ack(" << from_rank << ") failed for req "
              << creq->id << " match_seq " << creq->match_seq << std::endl;
    return false;
  }
  if (out_seq != creq->match_seq ||
      (status != kAckStatusOkDirect && status != kAckStatusOkCpuRelay)) {
    std::cerr << "[ERROR] sender completion ack invalid: seq=" << out_seq
              << " status=" << status << " req=" << creq->id
              << " match_seq=" << creq->match_seq << std::endl;
    return false;
  }

  if (status == kAckStatusOkCpuRelay) {
    GPU_RT_CHECK(gpuMemcpy(actual_dst, relay.ptr, creq->size_bytes,
                           gpuMemcpyHostToDevice));
  }
  return true;
}

void IpcChannel::complete_task(std::shared_ptr<Request> const& req, bool ok) {
  if (!req) return;
  if (!ok) {
    req->mark_failed();
  } else {
    req->complete_one();
  }
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

    IpcTask* task = nullptr;
    if (jring_sc_dequeue_bulk(send_task_ring_, &task, 1, nullptr) != 1) {
      continue;
    }
    pending_send_.fetch_sub(1, std::memory_order_relaxed);
    std::unique_ptr<IpcTask> task_guard(task);
    complete_task(task_guard->req,
                  send_one(task_guard->peer_rank, task_guard->req.get()));
  }

  while (true) {
    IpcTask* task = nullptr;
    if (jring_mc_dequeue_bulk(send_task_ring_, &task, 1, nullptr) != 1) break;
    std::unique_ptr<IpcTask> task_guard(task);
    complete_task(task_guard->req, false);
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

    IpcTask* task = nullptr;
    if (jring_sc_dequeue_bulk(recv_task_ring_, &task, 1, nullptr) != 1) {
      continue;
    }
    pending_recv_.fetch_sub(1, std::memory_order_relaxed);
    std::unique_ptr<IpcTask> task_guard(task);
    complete_task(task_guard->req,
                  recv_one(task_guard->peer_rank, task_guard->req.get()));
  }

  while (true) {
    IpcTask* task = nullptr;
    if (jring_mc_dequeue_bulk(recv_task_ring_, &task, 1, nullptr) != 1) break;
    std::unique_ptr<IpcTask> task_guard(task);
    complete_task(task_guard->req, false);
  }
}

}  // namespace Transport
}  // namespace UKernel
