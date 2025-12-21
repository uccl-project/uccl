#pragma once

#include "util/gpu_rt.h"
#include "util/jring.h"
#include "util/util.h"
#include <glog/logging.h>
#include <netinet/tcp.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace tcp {

// Constants
static constexpr size_t kCopyBufferSize =
    16 * 1024 * 1024;  // 16MB staging buffer
static constexpr size_t kDefaultTCPThreads = 4;
static constexpr size_t kRequestRingSize = 1024;
static constexpr int kEpollMaxEvents = 64;
static constexpr int kEpollTimeoutMs = 1;

// Get number of TCP threads from environment
inline size_t get_tcp_thread_count() {
  char const* env = std::getenv("UCCL_TCP_THREADS");
  if (env && strlen(env) > 0) {
    int val = std::atoi(env);
    if (val > 0) {
      LOG(INFO) << "TCP: Using " << val << " threads from UCCL_TCP_THREADS";
      return static_cast<size_t>(val);
    }
  }
  LOG(INFO) << "TCP: Using default " << kDefaultTCPThreads << " threads";
  return kDefaultTCPThreads;
}

// Request types for the thread pool
enum class TCPRequestType : uint32_t {
  SEND =
      0,  // Send data (waits for RecvReady on ctrl, then sends on data conns)
  RECV = 1,  // Recv data (sends RecvReady on ctrl, receiver worker handles data
             // via epoll)
  WRITE = 2,  // RDMA-style write (dest_addr already known, no ctrl message)
  READ = 3,   // RDMA-style read (dest_addr already known, no ctrl message)
  SHUTDOWN = 255
};

// Chunk header sent before each data chunk on data connections
struct ChunkHeader {
  uint64_t dest_addr;   // Destination GPU address on receiver side
  uint64_t offset;      // Offset within the transfer
  uint64_t chunk_size;  // Size of this chunk
  uint64_t total_size;  // Total transfer size
  uint32_t request_id;  // Request ID for matching
  uint32_t flags;       // Flags

  static constexpr uint32_t kFlagLastChunk = 1;
};
static_assert(sizeof(ChunkHeader) == 40, "ChunkHeader size mismatch");

// Message sent on control connection by receiver to tell sender where to put
// data
struct RecvReadyMsg {
  uint64_t dest_addr;   // GPU buffer address on receiver
  uint64_t size;        // Expected size
  uint32_t request_id;  // For matching (future use)
  uint32_t reserved;
};
static_assert(sizeof(RecvReadyMsg) == 24, "RecvReadyMsg size mismatch");

// TCP request structure (submitted via jring)
struct alignas(64) TCPRequest {
  TCPRequestType type;
  int ctrl_fd;         // Control connection fd
  void* data;          // GPU memory pointer (local buffer)
  size_t size;         // Total size to transfer
  uint64_t dest_addr;  // Destination GPU addr (for WRITE/READ, known upfront)
  std::atomic<bool>* completed;  // Completion flag
  std::atomic<bool>* success;    // Success flag
  uint32_t request_id;           // Unique request ID

  // For sender: pointer to connection group for load-balanced sending
  void* conn_group;  // TCPConnectionGroup*

  TCPRequest()
      : type(TCPRequestType::SEND),
        ctrl_fd(-1),
        data(nullptr),
        size(0),
        dest_addr(0),
        completed(nullptr),
        success(nullptr),
        request_id(0),
        conn_group(nullptr) {}
};

// TCP connection wrapper
struct TCPConnection {
  int fd = -1;
  std::string local_ip;
  std::string remote_ip;
  int remote_port = 0;
  std::atomic<uint32_t> inflight_chunks{0};
  uint32_t worker_id = 0;

  TCPConnection() = default;

  TCPConnection(TCPConnection&& other) noexcept
      : fd(other.fd),
        local_ip(std::move(other.local_ip)),
        remote_ip(std::move(other.remote_ip)),
        remote_port(other.remote_port),
        inflight_chunks(other.inflight_chunks.load()),
        worker_id(other.worker_id) {
    other.fd = -1;
  }

  ~TCPConnection() {
    if (fd >= 0) {
      close(fd);
      fd = -1;
    }
  }

  bool is_valid() const { return fd >= 0; }
};

// Forward declaration
struct TCPConnectionGroup;

// Pending receive state (tracked by receiver worker)
struct PendingRecv {
  uint64_t dest_addr;
  size_t total_size;
  size_t received_size;
  uint32_t request_id;
  std::atomic<bool>* completed;
  std::atomic<bool>* success;
};

// TCP Receiver Worker - uses epoll to wait on data connections
class TCPReceiverWorker {
 public:
  explicit TCPReceiverWorker(uint32_t id)
      : worker_id_(id), running_(false), epoll_fd_(-1) {
    epoll_fd_ = epoll_create1(0);
    if (epoll_fd_ < 0) {
      LOG(ERROR) << "TCPReceiverWorker " << id << ": Failed to create epoll";
    }
    LOG(INFO) << "TCPReceiverWorker " << id << " initialized";
  }

  ~TCPReceiverWorker() {
    stop();
    if (epoll_fd_ >= 0) {
      close(epoll_fd_);
    }
  }

  void start() {
    if (running_) return;
    running_ = true;
    worker_thread_ = std::thread(&TCPReceiverWorker::worker_loop, this);
    LOG(INFO) << "TCPReceiverWorker " << worker_id_ << " started";
  }

  void stop() {
    if (!running_) return;
    running_ = false;
    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
    LOG(INFO) << "TCPReceiverWorker " << worker_id_ << " stopped";
  }

  // Add a data connection to epoll
  bool add_data_connection(int fd) {
    struct epoll_event ev;
    ev.events = EPOLLIN | EPOLLET;  // Edge-triggered, read only
    ev.data.fd = fd;

    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) < 0) {
      LOG(ERROR) << "TCPReceiverWorker " << worker_id_
                 << ": epoll_ctl ADD failed";
      return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    data_fds_.insert(fd);
    return true;
  }

  void remove_data_connection(int fd) {
    epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
    std::lock_guard<std::mutex> lock(mutex_);
    data_fds_.erase(fd);
  }

  // Register a pending receive (called when uccl_recv_async is issued)
  void register_pending_recv(uint64_t dest_addr, size_t size,
                             uint32_t request_id, std::atomic<bool>* completed,
                             std::atomic<bool>* success) {
    std::lock_guard<std::mutex> lock(mutex_);
    PendingRecv pr;
    pr.dest_addr = dest_addr;
    pr.total_size = size;
    pr.received_size = 0;
    pr.request_id = request_id;
    pr.completed = completed;
    pr.success = success;
    pending_recvs_[dest_addr] = pr;
  }

  uint32_t id() const { return worker_id_; }

 private:
  void worker_loop() {
    std::vector<epoll_event> events(kEpollMaxEvents);
    thread_local std::vector<char> staging_buffer(kCopyBufferSize);

    while (running_) {
      int n = epoll_wait(epoll_fd_, events.data(), kEpollMaxEvents,
                         kEpollTimeoutMs);
      if (n < 0) {
        if (errno == EINTR) continue;
        LOG(ERROR) << "TCPReceiverWorker " << worker_id_
                   << ": epoll_wait error";
        break;
      }

      for (int i = 0; i < n; ++i) {
        if (events[i].events & EPOLLIN) {
          process_incoming_data(events[i].data.fd, staging_buffer);
        }
      }
    }
  }

  void process_incoming_data(int fd, std::vector<char>& staging_buffer) {
    // Read chunk header
    ChunkHeader header;
    ssize_t ret = recv(fd, &header, sizeof(header), MSG_PEEK);
    if (ret < static_cast<ssize_t>(sizeof(header))) {
      // Not enough data yet (edge-triggered, will be notified again)
      return;
    }

    // Actually read the header
    if (!recv_exact(fd, &header, sizeof(header))) {
      LOG(ERROR) << "TCPReceiverWorker " << worker_id_
                 << ": failed to read chunk header";
      return;
    }

    if (header.chunk_size > kCopyBufferSize) {
      LOG(ERROR) << "TCPReceiverWorker " << worker_id_ << ": chunk too large";
      return;
    }

    // Read chunk data
    if (!recv_exact(fd, staging_buffer.data(), header.chunk_size)) {
      LOG(ERROR) << "TCPReceiverWorker " << worker_id_
                 << ": failed to read chunk data";
      return;
    }

    // Copy to GPU at dest_addr + offset
    void* gpu_dest = reinterpret_cast<void*>(header.dest_addr + header.offset);
    gpuError_t err = gpuMemcpy(gpu_dest, staging_buffer.data(),
                               header.chunk_size, gpuMemcpyHostToDevice);
    if (err != gpuSuccess) {
      LOG(ERROR) << "TCPReceiverWorker " << worker_id_ << ": GPU memcpy failed";
    }

    // Update pending recv tracking
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pending_recvs_.find(header.dest_addr);
    if (it != pending_recvs_.end()) {
      it->second.received_size += header.chunk_size;

      bool is_complete = (header.flags & ChunkHeader::kFlagLastChunk) ||
                         (it->second.received_size >= it->second.total_size);

      if (is_complete) {
        if (it->second.success) {
          it->second.success->store(true, std::memory_order_release);
        }
        if (it->second.completed) {
          it->second.completed->store(true, std::memory_order_release);
        }
        pending_recvs_.erase(it);
      }
    }
  }

  bool recv_exact(int fd, void* buf, size_t n) {
    char* ptr = static_cast<char*>(buf);
    size_t received = 0;
    while (received < n) {
      ssize_t ret = ::recv(fd, ptr + received, n - received, 0);
      if (ret < 0) {
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK)
          return false;  // Would block
        return false;
      }
      if (ret == 0) return false;  // Connection closed
      received += ret;
    }
    return true;
  }

  uint32_t worker_id_;
  std::atomic<bool> running_;
  int epoll_fd_;
  std::thread worker_thread_;

  mutable std::mutex mutex_;
  std::unordered_set<int> data_fds_;
  std::unordered_map<uint64_t, PendingRecv> pending_recvs_;  // Key: dest_addr
};

// TCP Sender Worker - processes send requests from jring
class TCPSenderWorker {
 public:
  explicit TCPSenderWorker(uint32_t id)
      : worker_id_(id), running_(false), request_ring_(nullptr) {
    request_ring_ = uccl::create_ring(sizeof(TCPRequest), kRequestRingSize);
    LOG(INFO) << "TCPSenderWorker " << id << " initialized";
  }

  ~TCPSenderWorker() {
    stop();
    if (request_ring_) {
      free(request_ring_);
    }
  }

  void start() {
    if (running_) return;
    running_ = true;
    worker_thread_ = std::thread(&TCPSenderWorker::worker_loop, this);
    LOG(INFO) << "TCPSenderWorker " << worker_id_ << " started";
  }

  void stop() {
    if (!running_) return;
    running_ = false;

    TCPRequest shutdown_req;
    shutdown_req.type = TCPRequestType::SHUTDOWN;
    submit_request(shutdown_req);

    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
    LOG(INFO) << "TCPSenderWorker " << worker_id_ << " stopped";
  }

  bool submit_request(TCPRequest const& req) {
    return jring_sp_enqueue_bulk(request_ring_, &req, 1, nullptr) == 1;
  }

  uint32_t id() const { return worker_id_; }

 private:
  void worker_loop() {
    while (running_) {
      if (!process_requests()) {
        std::this_thread::yield();
      }
    }
  }

  bool process_requests() {
    TCPRequest req;
    bool processed_any = false;

    while (jring_sc_dequeue_bulk(request_ring_, &req, 1, nullptr) == 1) {
      processed_any = true;

      if (req.type == TCPRequestType::SHUTDOWN) {
        return true;
      }

      bool success = false;
      switch (req.type) {
        case TCPRequestType::SEND:
          success = do_send(req);
          break;
        case TCPRequestType::RECV:
          success = do_recv_ctrl(req);  // Just send RecvReady on ctrl
          break;
        case TCPRequestType::WRITE:
          success = do_write(req);
          break;
        case TCPRequestType::READ:
          success = do_read(req);
          break;
        default:
          break;
      }

      // For RECV, completion is handled by receiver worker
      if (req.type != TCPRequestType::RECV) {
        if (req.success) {
          req.success->store(success, std::memory_order_release);
        }
        if (req.completed) {
          req.completed->store(true, std::memory_order_release);
        }
      }
    }
    return processed_any;
  }

  bool send_exact(int fd, void const* buf, size_t n) {
    char const* ptr = static_cast<char const*>(buf);
    size_t sent = 0;
    while (sent < n) {
      ssize_t ret = ::send(fd, ptr + sent, n - sent, MSG_NOSIGNAL);
      if (ret < 0) {
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          std::this_thread::yield();
          continue;
        }
        return false;
      }
      sent += ret;
    }
    return true;
  }

  bool recv_exact(int fd, void* buf, size_t n) {
    char* ptr = static_cast<char*>(buf);
    size_t received = 0;
    while (received < n) {
      ssize_t ret = ::recv(fd, ptr + received, n - received, 0);
      if (ret < 0) {
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          std::this_thread::yield();
          continue;
        }
        return false;
      }
      if (ret == 0) return false;
      received += ret;
    }
    return true;
  }

  // Send RecvReady message on control connection (for uccl_recv_async)
  bool do_recv_ctrl(TCPRequest& req) {
    RecvReadyMsg msg;
    msg.dest_addr = reinterpret_cast<uint64_t>(req.data);
    msg.size = req.size;
    msg.request_id = req.request_id;
    msg.reserved = 0;

    return send_exact(req.ctrl_fd, &msg, sizeof(msg));
  }

  // Wait for RecvReady on ctrl, then send data on data connections
  bool do_send(
      TCPRequest& req);  // Implemented after TCPConnectionGroup definition

  // Send data directly (dest_addr already known from FifoItem)
  bool do_write(
      TCPRequest& req);  // Implemented after TCPConnectionGroup definition

  // Request read from remote (trigger remote to send data to us)
  bool do_read(TCPRequest& req) {
    // For read, we need to trigger remote side to send data
    // The protocol message was already sent in uccl_read_async
    // This is handled by the receiver worker via epoll
    return true;
  }

  uint32_t worker_id_;
  std::atomic<bool> running_;
  jring_t* request_ring_;
  std::thread worker_thread_;
};

// Connection group for a peer (ctrl + data connections)
struct TCPConnectionGroup {
  int ctrl_fd = -1;  // Control connection
  std::vector<std::unique_ptr<TCPConnection>> data_connections;
  std::atomic<uint64_t> round_robin_idx{0};
  mutable std::shared_mutex mutex;

  // For tracking remaining data to send (decremented as RecvReady messages
  // arrive)
  std::atomic<size_t> pending_send_size{0};

  // Power-of-two choice selection for load balancing
  TCPConnection* select_data_connection() {
    std::shared_lock<std::shared_mutex> lock(mutex);
    if (data_connections.empty()) return nullptr;
    if (data_connections.size() == 1) return data_connections[0].get();

    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, data_connections.size() - 1);

    size_t idx1 = dist(gen);
    size_t idx2 = dist(gen);
    while (idx2 == idx1 && data_connections.size() > 1) {
      idx2 = dist(gen);
    }

    uint32_t load1 =
        data_connections[idx1]->inflight_chunks.load(std::memory_order_relaxed);
    uint32_t load2 =
        data_connections[idx2]->inflight_chunks.load(std::memory_order_relaxed);

    return (load1 <= load2) ? data_connections[idx1].get()
                            : data_connections[idx2].get();
  }

  void add_data_connection(std::unique_ptr<TCPConnection> conn) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data_connections.push_back(std::move(conn));
  }

  size_t data_connection_count() const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return data_connections.size();
  }
};

// Implementation of do_send after TCPConnectionGroup is defined
inline bool TCPSenderWorker::do_send(TCPRequest& req) {
  if (req.ctrl_fd < 0 || !req.data || req.size == 0) return false;

  auto* group = static_cast<TCPConnectionGroup*>(req.conn_group);
  if (!group) return false;

  // Wait for RecvReady message on control connection
  RecvReadyMsg ready_msg;
  if (!recv_exact(req.ctrl_fd, &ready_msg, sizeof(ready_msg))) {
    LOG(ERROR) << "TCPSenderWorker " << worker_id_
               << ": failed to recv RecvReady";
    return false;
  }

  uint64_t dest_addr = ready_msg.dest_addr;

  thread_local std::vector<char> staging_buffer(kCopyBufferSize);

  size_t offset = 0;
  while (offset < req.size) {
    // Select a data connection with power-of-two choice
    TCPConnection* conn = group->select_data_connection();
    if (!conn || !conn->is_valid()) {
      LOG(ERROR) << "TCPSenderWorker " << worker_id_
                 << ": no valid data connection";
      return false;
    }

    conn->inflight_chunks.fetch_add(1, std::memory_order_relaxed);

    size_t chunk_size = std::min(req.size - offset, kCopyBufferSize);
    bool is_last = (offset + chunk_size >= req.size);

    // Prepare chunk header
    ChunkHeader header;
    header.dest_addr = dest_addr;
    header.offset = offset;
    header.chunk_size = chunk_size;
    header.total_size = req.size;
    header.request_id = req.request_id;
    header.flags = is_last ? ChunkHeader::kFlagLastChunk : 0;

    // Send header on data connection
    if (!send_exact(conn->fd, &header, sizeof(header))) {
      conn->inflight_chunks.fetch_sub(1, std::memory_order_relaxed);
      return false;
    }

    // Copy from GPU to staging buffer
    gpuError_t err = gpuMemcpy(staging_buffer.data(),
                               static_cast<char const*>(req.data) + offset,
                               chunk_size, gpuMemcpyDeviceToHost);
    if (err != gpuSuccess) {
      conn->inflight_chunks.fetch_sub(1, std::memory_order_relaxed);
      return false;
    }

    // Send data
    if (!send_exact(conn->fd, staging_buffer.data(), chunk_size)) {
      conn->inflight_chunks.fetch_sub(1, std::memory_order_relaxed);
      return false;
    }

    conn->inflight_chunks.fetch_sub(1, std::memory_order_relaxed);
    offset += chunk_size;
  }

  return true;
}

// Implementation of do_write (dest_addr already known)
inline bool TCPSenderWorker::do_write(TCPRequest& req) {
  if (!req.data || req.size == 0) return false;

  auto* group = static_cast<TCPConnectionGroup*>(req.conn_group);
  if (!group) return false;

  uint64_t dest_addr = req.dest_addr;  // Already known from FifoItem

  thread_local std::vector<char> staging_buffer(kCopyBufferSize);

  size_t offset = 0;
  while (offset < req.size) {
    TCPConnection* conn = group->select_data_connection();
    if (!conn || !conn->is_valid()) return false;

    conn->inflight_chunks.fetch_add(1, std::memory_order_relaxed);

    size_t chunk_size = std::min(req.size - offset, kCopyBufferSize);
    bool is_last = (offset + chunk_size >= req.size);

    ChunkHeader header;
    header.dest_addr = dest_addr;
    header.offset = offset;
    header.chunk_size = chunk_size;
    header.total_size = req.size;
    header.request_id = req.request_id;
    header.flags = is_last ? ChunkHeader::kFlagLastChunk : 0;

    if (!send_exact(conn->fd, &header, sizeof(header))) {
      conn->inflight_chunks.fetch_sub(1, std::memory_order_relaxed);
      return false;
    }

    gpuError_t err = gpuMemcpy(staging_buffer.data(),
                               static_cast<char const*>(req.data) + offset,
                               chunk_size, gpuMemcpyDeviceToHost);
    if (err != gpuSuccess) {
      conn->inflight_chunks.fetch_sub(1, std::memory_order_relaxed);
      return false;
    }

    if (!send_exact(conn->fd, staging_buffer.data(), chunk_size)) {
      conn->inflight_chunks.fetch_sub(1, std::memory_order_relaxed);
      return false;
    }

    conn->inflight_chunks.fetch_sub(1, std::memory_order_relaxed);
    offset += chunk_size;
  }

  return true;
}

// TCP Thread Pool (sender workers + receiver workers)
class TCPThreadPool {
 public:
  explicit TCPThreadPool(size_t num_threads = 0) {
    if (num_threads == 0) {
      num_threads = get_tcp_thread_count();
    }

    // Create sender workers
    sender_workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      sender_workers_.push_back(std::make_unique<TCPSenderWorker>(i));
    }

    // Create receiver workers
    receiver_workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      receiver_workers_.push_back(std::make_unique<TCPReceiverWorker>(i));
    }

    LOG(INFO) << "TCPThreadPool created with " << num_threads
              << " sender + receiver workers";
  }

  ~TCPThreadPool() { stop(); }

  void start() {
    for (auto& w : sender_workers_) w->start();
    for (auto& w : receiver_workers_) w->start();
  }

  void stop() {
    for (auto& w : sender_workers_) w->stop();
    for (auto& w : receiver_workers_) w->stop();
  }

  // Assign a data connection to a receiver worker (round-robin)
  uint32_t assign_data_connection(int fd) {
    uint32_t id = next_receiver_.fetch_add(1, std::memory_order_relaxed) %
                  receiver_workers_.size();
    receiver_workers_[id]->add_data_connection(fd);
    return id;
  }

  // Submit request to a sender worker
  bool submit_send_request(TCPRequest const& req) {
    uint32_t id = next_sender_.fetch_add(1, std::memory_order_relaxed) %
                  sender_workers_.size();
    return sender_workers_[id]->submit_request(req);
  }

  // Register a pending receive with a receiver worker
  void register_pending_recv(uint64_t dest_addr, size_t size,
                             uint32_t request_id, std::atomic<bool>* completed,
                             std::atomic<bool>* success) {
    // Register with all receiver workers (data can come on any connection)
    for (auto& w : receiver_workers_) {
      w->register_pending_recv(dest_addr, size, request_id, completed, success);
    }
  }

  size_t size() const { return sender_workers_.size(); }

  TCPReceiverWorker* get_receiver_worker(size_t idx) {
    return idx < receiver_workers_.size() ? receiver_workers_[idx].get()
                                          : nullptr;
  }

 private:
  std::vector<std::unique_ptr<TCPSenderWorker>> sender_workers_;
  std::vector<std::unique_ptr<TCPReceiverWorker>> receiver_workers_;
  std::atomic<uint32_t> next_sender_{0};
  std::atomic<uint32_t> next_receiver_{0};
};

// Async request tracking
struct TCPAsyncHandle {
  std::atomic<bool> completed{false};
  std::atomic<bool> success{false};
  uint32_t request_id{0};
};

}  // namespace tcp
