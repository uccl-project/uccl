#include "tcp/tcp_worker_pool.h"

namespace tcp {

void PendingRecvMap::add(uint64_t dest_addr, size_t size, uint32_t request_id,
                         std::atomic<bool>* completed,
                         std::atomic<bool>* success) {
  std::lock_guard<std::mutex> lock(mutex_);
  pending_recvs_[dest_addr] = std::make_unique<PendingRecv>(
      dest_addr, size, request_id, completed, success);
}

bool PendingRecvMap::update_and_check_complete(uint64_t dest_addr,
                                               size_t chunk_size,
                                               bool is_last_chunk) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_recvs_.find(dest_addr);
  if (it == pending_recvs_.end()) {
    return false;  // Not found, ignore
  }

  PendingRecv* pr = it->second.get();
  size_t new_received =
      pr->received_size.fetch_add(chunk_size, std::memory_order_relaxed) +
      chunk_size;

  bool is_complete = is_last_chunk || (new_received >= pr->total_size);

  if (is_complete) {
    if (pr->success) {
      pr->success->store(true, std::memory_order_release);
    }
    if (pr->completed) {
      pr->completed->store(true, std::memory_order_release);
    }
    pending_recvs_.erase(it);
    return true;
  }

  return false;
}

TCPReceiverWorker::TCPReceiverWorker(uint32_t id, PendingRecvMap* pending_recvs)
    : worker_id_(id),
      running_(false),
      epoll_fd_(-1),
      pending_recvs_(pending_recvs) {
  epoll_fd_ = epoll_create1(0);
  if (epoll_fd_ < 0) {
    LOG(ERROR) << "TCPReceiverWorker " << id << ": Failed to create epoll";
  }
  LOG(INFO) << "TCPReceiverWorker " << id << " initialized";
}

TCPReceiverWorker::~TCPReceiverWorker() {
  stop();
  if (epoll_fd_ >= 0) {
    close(epoll_fd_);
  }
}

void TCPReceiverWorker::start() {
  if (running_) return;
  running_ = true;
  worker_thread_ = std::thread(&TCPReceiverWorker::worker_loop, this);
  LOG(INFO) << "TCPReceiverWorker " << worker_id_ << " started";
}

void TCPReceiverWorker::stop() {
  if (!running_) return;
  running_ = false;
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  LOG(INFO) << "TCPReceiverWorker " << worker_id_ << " stopped";
}

bool TCPReceiverWorker::add_data_connection(int fd) {
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

void TCPReceiverWorker::remove_data_connection(int fd) {
  epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
  std::lock_guard<std::mutex> lock(mutex_);
  data_fds_.erase(fd);
}

void TCPReceiverWorker::worker_loop() {
  std::vector<epoll_event> events(kEpollMaxEvents);
  thread_local std::vector<char> staging_buffer(kCopyBufferSize);

  while (running_) {
    int n =
        epoll_wait(epoll_fd_, events.data(), kEpollMaxEvents, kEpollTimeoutMs);
    if (n < 0) {
      if (errno == EINTR) continue;
      LOG(ERROR) << "TCPReceiverWorker " << worker_id_ << ": epoll_wait error";
      break;
    }

    for (int i = 0; i < n; ++i) {
      if (events[i].events & EPOLLIN) {
        process_incoming_data(events[i].data.fd, staging_buffer);
      }
    }
  }
}

void TCPReceiverWorker::process_incoming_data(
    int fd, std::vector<char>& staging_buffer) {
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
  gpuError_t err = gpuMemcpy(gpu_dest, staging_buffer.data(), header.chunk_size,
                             gpuMemcpyHostToDevice);
  if (err != gpuSuccess) {
    LOG(ERROR) << "TCPReceiverWorker " << worker_id_ << ": GPU memcpy failed";
  }

  // Update pending recv tracking (shared map)
  bool is_last = (header.flags & ChunkHeader::kFlagLastChunk) != 0;
  pending_recvs_->update_and_check_complete(header.dest_addr, header.chunk_size,
                                            is_last);
}

bool TCPReceiverWorker::recv_exact(int fd, void* buf, size_t n) {
  char* ptr = static_cast<char*>(buf);
  size_t received = 0;
  while (received < n) {
    ssize_t ret = ::recv(fd, ptr + received, n - received, 0);
    if (ret < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) return false;  // Would block
      return false;
    }
    if (ret == 0) return false;  // Connection closed
    received += ret;
  }
  return true;
}

TCPSenderWorker::TCPSenderWorker(uint32_t id)
    : worker_id_(id), running_(false), request_ring_(nullptr) {
  request_ring_ = uccl::create_ring(sizeof(TCPRequest), kRequestRingSize);
  LOG(INFO) << "TCPSenderWorker " << id << " initialized";
}

TCPSenderWorker::~TCPSenderWorker() {
  stop();
  if (request_ring_) {
    free(request_ring_);
  }
}

void TCPSenderWorker::start() {
  if (running_) return;
  running_ = true;
  worker_thread_ = std::thread(&TCPSenderWorker::worker_loop, this);
  LOG(INFO) << "TCPSenderWorker " << worker_id_ << " started";
}

void TCPSenderWorker::stop() {
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

bool TCPSenderWorker::submit_request(TCPRequest const& req) {
  return jring_sp_enqueue_bulk(request_ring_, &req, 1, nullptr) == 1;
}

void TCPSenderWorker::worker_loop() {
  while (running_) {
    if (!process_requests()) {
      std::this_thread::yield();
    }
  }
}

bool TCPSenderWorker::process_requests() {
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

bool TCPSenderWorker::send_exact(int fd, void const* buf, size_t n) {
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

bool TCPSenderWorker::recv_exact(int fd, void* buf, size_t n) {
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

bool TCPSenderWorker::do_recv_ctrl(TCPRequest& req) {
  RecvReadyMsg msg;
  msg.dest_addr = reinterpret_cast<uint64_t>(req.data);
  msg.size = req.size;
  msg.request_id = req.request_id;
  msg.reserved = 0;

  return send_exact(req.ctrl_fd, &msg, sizeof(msg));
}

bool TCPSenderWorker::do_send(TCPRequest& req) {
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

bool TCPSenderWorker::do_write(TCPRequest& req) {
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

bool TCPSenderWorker::do_read(TCPRequest& req) {
  // For read, we need to trigger remote side to send data
  // The protocol message was already sent in uccl_read_async
  // This is handled by the receiver worker via epoll
  return true;
}

TCPConnection* TCPConnectionGroup::select_data_connection() {
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

void TCPConnectionGroup::add_data_connection(
    std::unique_ptr<TCPConnection> conn) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  data_connections.push_back(std::move(conn));
}

size_t TCPConnectionGroup::data_connection_count() const {
  std::shared_lock<std::shared_mutex> lock(mutex);
  return data_connections.size();
}

TCPThreadPool::TCPThreadPool(size_t num_threads) {
  if (num_threads == 0) {
    num_threads = get_tcp_thread_count();
  }

  // Create sender workers
  sender_workers_.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    sender_workers_.push_back(std::make_unique<TCPSenderWorker>(i));
  }

  // Create receiver workers (pass shared pending_recvs_)
  receiver_workers_.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    receiver_workers_.push_back(
        std::make_unique<TCPReceiverWorker>(i, &pending_recvs_));
  }

  LOG(INFO) << "TCPThreadPool created with " << num_threads
            << " sender + receiver workers";
}

TCPThreadPool::~TCPThreadPool() { stop(); }

void TCPThreadPool::start() {
  for (auto& w : sender_workers_) w->start();
  for (auto& w : receiver_workers_) w->start();
}

void TCPThreadPool::stop() {
  for (auto& w : sender_workers_) w->stop();
  for (auto& w : receiver_workers_) w->stop();
}

uint32_t TCPThreadPool::assign_data_connection(int fd) {
  uint32_t id = next_receiver_.fetch_add(1, std::memory_order_relaxed) %
                receiver_workers_.size();
  receiver_workers_[id]->add_data_connection(fd);
  return id;
}

bool TCPThreadPool::submit_send_request(TCPRequest const& req) {
  uint32_t id = next_sender_.fetch_add(1, std::memory_order_relaxed) %
                sender_workers_.size();
  return sender_workers_[id]->submit_request(req);
}

void TCPThreadPool::register_pending_recv(uint64_t dest_addr, size_t size,
                                          uint32_t request_id,
                                          std::atomic<bool>* completed,
                                          std::atomic<bool>* success) {
  // Register in shared pending_recvs_ map
  pending_recvs_.add(dest_addr, size, request_id, completed, success);
}

TCPReceiverWorker* TCPThreadPool::get_receiver_worker(size_t idx) {
  return idx < receiver_workers_.size() ? receiver_workers_[idx].get()
                                        : nullptr;
}

}  // namespace tcp
