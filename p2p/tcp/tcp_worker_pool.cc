#include "tcp/tcp_worker_pool.h"

namespace tcp {

// RecvMatchQueue implementation
void RecvMatchQueue::push_recv(uint64_t dest_addr, size_t size,
                               uint32_t recv_request_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto info = std::make_unique<RecvMatchInfo>();
  info->dest_addr = dest_addr;
  info->size = size;
  info->recv_request_id = recv_request_id;
  info->received.store(0, std::memory_order_relaxed);
  pending_recvs_.push_back(std::move(info));
}

uint32_t RecvMatchQueue::get_next_send_seq_id() {
  return next_send_seq_id_.fetch_add(1, std::memory_order_relaxed);
}

bool RecvMatchQueue::get_recv_info(uint32_t send_seq_id,
                                   uint64_t* base_dest_addr,
                                   uint32_t* recv_request_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if already in progress
  auto it = in_progress_.find(send_seq_id);
  if (it != in_progress_.end() && it->second) {
    *base_dest_addr = it->second->dest_addr;
    *recv_request_id = it->second->recv_request_id;
    return true;
  }

  // Pop recvs from pending_recvs_ in order (0, 1, 2...) up to send_seq_id
  // We need to ensure recv N is assigned to send_seq_id N
  while (next_seq_to_assign_ <= send_seq_id) {
    if (pending_recvs_.empty()) {
      // Not enough recvs registered yet
      return false;
    }
    // Pop recv and assign to sequence number next_seq_to_assign_
    in_progress_[next_seq_to_assign_] = std::move(pending_recvs_.front());
    pending_recvs_.pop_front();
    next_seq_to_assign_++;
  }

  // Now we have the recv for send_seq_id in in_progress_
  it = in_progress_.find(send_seq_id);
  if (it == in_progress_.end() || !it->second) {
    // This shouldn't happen, but be defensive
    LOG(ERROR) << "RecvMatchQueue::get_recv_info: send_seq_id=" << send_seq_id
               << " not found after assignment";
    exit(1);
  }
  *base_dest_addr = it->second->dest_addr;
  *recv_request_id = it->second->recv_request_id;
  return true;
}

void RecvMatchQueue::add_received_bytes(uint32_t send_seq_id, size_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = in_progress_.find(send_seq_id);
  if (it == in_progress_.end()) return;

  size_t new_total =
      it->second->received.fetch_add(bytes, std::memory_order_relaxed) + bytes;
  if (new_total >= it->second->size) {
    // All bytes received, remove from in_progress
    in_progress_.erase(it);
  }
}

void PendingRecvMap::add(uint64_t dest_addr, size_t size, uint32_t request_id,
                         std::atomic<bool>* completed,
                         std::atomic<bool>* success) {
  std::lock_guard<std::mutex> lock(mutex_);
  // Key by request_id to support chunked transfers where dest_addr varies
  pending_recvs_[request_id] =
      std::make_unique<PendingTransfer>(size, request_id, completed, success);
}

bool PendingRecvMap::update_and_check_complete(uint32_t request_id,
                                               size_t chunk_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_recvs_.find(request_id);
  if (it == pending_recvs_.end()) {
    return false;  // Not found, ignore
  }

  PendingTransfer* pr = it->second.get();
  size_t new_received =
      pr->transferred_size.fetch_add(chunk_size, std::memory_order_relaxed) +
      chunk_size;

  // Complete only when all bytes received (chunks may arrive out of order)
  bool is_complete = (new_received >= pr->total_size);

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

void PendingSendMap::add(size_t size, uint32_t request_id,
                         std::atomic<bool>* completed,
                         std::atomic<bool>* success) {
  std::lock_guard<std::mutex> lock(mutex_);
  pending_sends_[request_id] =
      std::make_unique<PendingTransfer>(size, request_id, completed, success);
}

bool PendingSendMap::update_and_check_complete(uint32_t request_id,
                                               size_t chunk_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_sends_.find(request_id);
  if (it == pending_sends_.end()) {
    return false;  // Not found, ignore
  }

  PendingTransfer* ps = it->second.get();
  size_t new_sent =
      ps->transferred_size.fetch_add(chunk_size, std::memory_order_relaxed) +
      chunk_size;

  // Complete only when all bytes sent
  bool is_complete = (new_sent >= ps->total_size);

  if (is_complete) {
    if (ps->success) {
      ps->success->store(true, std::memory_order_release);
    }
    if (ps->completed) {
      ps->completed->store(true, std::memory_order_release);
    }
    pending_sends_.erase(it);
    return true;
  }

  return false;
}

// Common TCP I/O helper functions
bool send_exact(int fd, void const* buf, size_t n) {
  char const* ptr = static_cast<char const*>(buf);
  size_t sent = 0;
  while (sent < n) {
    ssize_t ret = ::send(fd, ptr + sent, n - sent, MSG_NOSIGNAL);
    if (ret < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // For simplicity, spin; could use poll/select
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
      if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
      return false;
    }
    if (ret == 0) return false;  // Connection closed
    received += ret;
  }
  return true;
}

// Use sendmsg for scatter-gather I/O (header + data in one syscall)
// This reduces syscall overhead and allows TCP to coalesce better
bool send_header_and_data(int fd, void const* header, size_t header_size,
                          void const* data, size_t data_size) {
  struct iovec iov[2];
  iov[0].iov_base = const_cast<void*>(header);
  iov[0].iov_len = header_size;
  iov[1].iov_base = const_cast<void*>(data);
  iov[1].iov_len = data_size;

  struct msghdr msg;
  std::memset(&msg, 0, sizeof(msg));
  msg.msg_iov = iov;
  msg.msg_iovlen = 2;

  size_t total = header_size + data_size;
  size_t sent = 0;

  while (sent < total) {
    // Adjust iov for partial sends
    size_t offset = sent;
    int iov_idx = 0;
    if (offset >= header_size) {
      // Header fully sent, only data remains
      iov[0].iov_base = nullptr;
      iov[0].iov_len = 0;
      iov[1].iov_base =
          static_cast<char*>(const_cast<void*>(data)) + (offset - header_size);
      iov[1].iov_len = data_size - (offset - header_size);
      iov_idx = 1;
    } else {
      // Still sending header
      iov[0].iov_base = static_cast<char*>(const_cast<void*>(header)) + offset;
      iov[0].iov_len = header_size - offset;
    }

    ssize_t ret = ::sendmsg(fd, &msg, MSG_NOSIGNAL);
    if (ret < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
      return false;
    }
    sent += ret;
  }
  return true;
}

TCPReceiverWorker::TCPReceiverWorker(uint32_t id, TCPThreadPool* thread_pool)
    : worker_id_(id),
      running_(false),
      epoll_fd_(-1),
      thread_pool_(thread_pool),
      staging_buffer_(nullptr) {
  // Allocate pinned memory for efficient GPU-host transfers
  gpuError_t err = gpuMallocHost(reinterpret_cast<void**>(&staging_buffer_),
                                 kStagingBufferSize);
  if (err != gpuSuccess) {
    LOG(ERROR) << "TCPReceiverWorker " << id
               << ": Failed to allocate pinned memory" << err;
  }
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
  if (staging_buffer_) {
    (void)gpuFreeHost(staging_buffer_);
    staging_buffer_ = nullptr;
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
               << ": epoll_ctl ADD failed for fd=" << fd << " errno=" << errno;
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
        process_event(events[i].data.fd);
      }
    }
  }
}

void TCPReceiverWorker::process_event(int fd) {
  // With edge-triggered epoll, we must drain all available data
  // Otherwise subsequent messages won't trigger EPOLLIN
  while (true) {
    // Check if a full header is available (non-blocking peek)
    TCPDataHeader header;
    ssize_t peeked = recv(fd, &header, sizeof(header), MSG_PEEK | MSG_DONTWAIT);
    if (peeked < static_cast<ssize_t>(sizeof(header))) {
      return;
    }

    // For SEND operations with kFlagNeedsMatch, check if we have a matching
    // recv before reading the header and data
    if ((header.flags & TCPDataHeader::kFlagNeedsMatch) &&
        static_cast<TCPDataMsgType>(header.msg_type) ==
            TCPDataMsgType::DATA_CHUNK) {
      RecvMatchQueue* match_queue = thread_pool_->get_match_queue(fd);
      if (!match_queue) {
        LOG(ERROR) << "TCPReceiverWorker " << worker_id_
                   << ": no match queue for fd=" << fd;
        exit(1);
      }
      uint32_t send_seq_id = header.request_id;  // send_seq_id is in request_id
      uint64_t base_dest_addr;
      uint32_t recv_request_id;
      // Check if we can match this send_seq_id - don't read data until we can
      while (!match_queue->get_recv_info(send_seq_id, &base_dest_addr,
                                         &recv_request_id)) {
        // No match available yet - yield and retry later
        std::this_thread::yield();
      }
    }

    // Now read the header (blocking is OK since we know data is there)
    if (!recv_exact(fd, &header, sizeof(header))) {
      // Error or connection closed
      exit(1);
    }

    if (static_cast<TCPDataMsgType>(header.msg_type) ==
        TCPDataMsgType::READ_REQUEST) {
      // Handle READ request - remote side wants us to send data
      process_read_request(fd, header);
    } else {
      // Handle data chunk (SEND/WRITE)
      process_data_chunk(fd, header);
    }
  }
}

void TCPReceiverWorker::process_read_request(int fd,
                                             TCPDataHeader const& header) {
  // Read from local memory at remote_addr and send to requester
  // The requester wants data from our remote_addr sent to their dest_addr
  if (header.size > kStagingBufferSize) {
    LOG(ERROR) << "TCPReceiverWorker " << worker_id_ << ": READ size too large";
    return;
  }

  // Copy from GPU to staging buffer
#if UCCL_TCP_GPU_MEMCPY
  void* src = reinterpret_cast<void*>(header.remote_addr);
  gpuError_t err =
      gpuMemcpy(staging_buffer_, src, header.size, gpuMemcpyDeviceToHost);
  if (err != gpuSuccess) {
    LOG(ERROR) << "TCPReceiverWorker " << worker_id_
               << ": GPU memcpy failed for READ";
    return;
  }
#endif

  // Send data back with a response header
  TCPDataHeader response;
  response.msg_type = static_cast<uint32_t>(TCPDataMsgType::DATA_CHUNK);
  response.flags = 0;
  response.request_id = header.request_id;
  response.reserved = 0;
  response.dest_addr = header.dest_addr;
  response.remote_addr = 0;
  response.size = header.size;
  response.total_size = header.total_size;

  if (!send_exact(fd, &response, sizeof(response))) {
    LOG(ERROR) << "TCPReceiverWorker " << worker_id_
               << ": failed to send READ response header";
    return;
  }

  if (!send_exact(fd, staging_buffer_, header.size)) {
    LOG(ERROR) << "TCPReceiverWorker " << worker_id_
               << ": failed to send READ response data";
    return;
  }
}

void TCPReceiverWorker::process_data_chunk(int fd,
                                           TCPDataHeader const& header) {
  // Header already read by process_event

  if (header.size > kStagingBufferSize) {
    LOG(ERROR) << "TCPReceiverWorker " << worker_id_ << ": chunk too large";
    return;
  }

  // Read chunk data
  if (!recv_exact(fd, staging_buffer_, header.size)) {
    LOG(ERROR) << "TCPReceiverWorker " << worker_id_
               << ": failed to read chunk data";
    return;
  }

  uint64_t actual_dest_addr = header.dest_addr;
  uint32_t recv_request_id = header.request_id;
  uint32_t send_seq_id = 0;

  // For SEND operations (kFlagNeedsMatch), look up dest_addr from match queue
  // Note: We already checked for match in process_event() before reading data,
  // so this should always succeed
  RecvMatchQueue* match_queue = nullptr;
  if (header.flags & TCPDataHeader::kFlagNeedsMatch) {
    match_queue = thread_pool_->get_match_queue(fd);
    if (!match_queue) {
      LOG(ERROR) << "TCPReceiverWorker " << worker_id_
                 << ": no match queue for fd=" << fd;
      return;
    }

    // header.request_id is the send_seq_id for SEND operations
    send_seq_id = header.request_id;

    // Get recv info - should always succeed since we checked in process_event()
    uint64_t base_dest_addr;
    if (!match_queue->get_recv_info(send_seq_id, &base_dest_addr,
                                    &recv_request_id)) {
      LOG(ERROR) << "TCPReceiverWorker " << worker_id_
                 << ": failed to get recv_info for send_seq_id=" << send_seq_id
                 << " (should not happen - checked in process_event)";
      return;
    }

    // dest_addr in header is the offset for this chunk
    actual_dest_addr = base_dest_addr + header.dest_addr;
  }

  // Copy to GPU at dest_addr
#if UCCL_TCP_GPU_MEMCPY
  void* gpu_dest = reinterpret_cast<void*>(actual_dest_addr);
  gpuError_t err =
      gpuMemcpy(gpu_dest, staging_buffer_, header.size, gpuMemcpyHostToDevice);
  if (err != gpuSuccess) {
    LOG(ERROR) << "TCPReceiverWorker " << worker_id_ << ": GPU memcpy failed";
  }
#endif

  // Update pending recv tracking (shared map) - keyed by recv_request_id
  // Completion is based on received_size >= total_size
  thread_pool_->get_pending_recvs()->update_and_check_complete(recv_request_id,
                                                               header.size);

  // Track bytes received in match queue and remove from in_progress when
  // complete
  if (match_queue) {
    match_queue->add_received_bytes(send_seq_id, header.size);
  }
}

TCPSenderWorker::TCPSenderWorker(uint32_t id, TCPThreadPool* thread_pool)
    : worker_id_(id),
      running_(false),
      request_ring_(nullptr),
      staging_buffer_(nullptr),
      thread_pool_(thread_pool) {
  // Allocate pinned memory for efficient GPU-host transfers
  gpuError_t err = gpuMallocHost(reinterpret_cast<void**>(&staging_buffer_),
                                 kStagingBufferSize);
  if (err != gpuSuccess) {
    LOG(ERROR) << "TCPSenderWorker " << id
               << ": Failed to allocate pinned memory" << err;
  }
  request_ring_ = uccl::create_ring(sizeof(TCPRequest), kRequestRingSize);
  LOG(INFO) << "TCPSenderWorker " << id << " initialized";
}

TCPSenderWorker::~TCPSenderWorker() {
  stop();
  if (request_ring_) {
    free(request_ring_);
  }
  if (staging_buffer_) {
    (void)gpuFreeHost(staging_buffer_);
    staging_buffer_ = nullptr;
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
  while (jring_sp_enqueue_bulk(request_ring_, &req, 1, nullptr) != 1) {
    if (!running_) return false;
    std::this_thread::yield();
  }
  return true;
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
      case TCPRequestType::WRITE:
        success = do_write(req);
        break;
      case TCPRequestType::READ:
        success = do_read(req);
        break;
      default:
        break;
    }

    // For SEND and WRITE, track bytes sent via shared map
    if (req.type == TCPRequestType::SEND || req.type == TCPRequestType::WRITE) {
      if (success) {
        thread_pool_->get_pending_sends()->update_and_check_complete(
            req.request_id, req.size);
      }
    }
  }
  return processed_any;
}

bool TCPSenderWorker::do_send(TCPRequest& req) {
  // For eager SEND, dest_addr is the offset (receiver looks up base from match
  // queue)
  if (!req.data || req.size == 0) return false;

  // Use pre-assigned connection (ensures disjoint connections per worker)
  TCPConnection* conn = req.assigned_conn;
  if (!conn || !conn->is_valid()) {
    LOG(ERROR) << "TCPSenderWorker " << worker_id_
               << ": no valid assigned connection";
    return false;
  }

  // Prepare header
  // - dest_addr is the offset (receiver adds base from match queue)
  // - flags include kFlagNeedsMatch so receiver knows to look up from match
  // queue
  // - request_id carries send_seq_id for matching on receiver
  TCPDataHeader header;
  header.msg_type = static_cast<uint32_t>(TCPDataMsgType::DATA_CHUNK);
  header.flags = req.flags;  // Already includes kFlagNeedsMatch from endpoint
  header.request_id = req.send_seq_id;  // send_seq_id for matching
  header.reserved = 0;
  header.dest_addr = req.dest_addr;  // Offset within transfer
  header.remote_addr = 0;            // Not used for DATA_CHUNK
  header.size = req.size;
  header.total_size = req.total_size;

  // Copy from GPU to staging buffer (always needed as req.data is GPU memory)
#if UCCL_TCP_GPU_MEMCPY
  gpuError_t err =
      gpuMemcpy(staging_buffer_, static_cast<char const*>(req.data), req.size,
                gpuMemcpyDeviceToHost);
  if (err != gpuSuccess) {
    return false;
  }
#endif

  // Send header + data in one syscall using scatter-gather I/O
  if (!send_header_and_data(conn->fd, &header, sizeof(header), staging_buffer_,
                            req.size)) {
    conn->inflight_chunks.fetch_sub(1, std::memory_order_relaxed);
    return false;
  }
  conn->inflight_chunks.fetch_sub(1, std::memory_order_relaxed);

  return true;
}

bool TCPSenderWorker::do_write(TCPRequest& req) {
  if (!req.data || req.size == 0) return false;

  // Use pre-assigned connection (ensures disjoint connections per worker)
  TCPConnection* conn = req.assigned_conn;
  if (!conn || !conn->is_valid()) return false;

  // dest_addr already includes offset from chunking
  TCPDataHeader header;
  header.msg_type = static_cast<uint32_t>(TCPDataMsgType::DATA_CHUNK);
  header.flags = req.flags;
  header.request_id = req.request_id;
  header.reserved = 0;
  header.dest_addr = req.dest_addr;
  header.remote_addr = 0;  // Not used for DATA_CHUNK
  header.size = req.size;
  header.total_size = req.total_size;

  // Copy from GPU to staging buffer (always needed as req.data is GPU memory)
#if UCCL_TCP_GPU_MEMCPY
  gpuError_t err =
      gpuMemcpy(staging_buffer_, static_cast<char const*>(req.data), req.size,
                gpuMemcpyDeviceToHost);
  if (err != gpuSuccess) {
    return false;
  }
#endif

  // Send header + data in one syscall using scatter-gather I/O
  if (!send_header_and_data(conn->fd, &header, sizeof(header), staging_buffer_,
                            req.size)) {
    conn->inflight_chunks.fetch_sub(1, std::memory_order_relaxed);
    return false;
  }
  conn->inflight_chunks.fetch_sub(1, std::memory_order_relaxed);

  return true;
}

bool TCPSenderWorker::do_read(TCPRequest& req) {
  // For READ, send request header on a data connection so remote
  // receiver worker can see it via epoll and trigger sending data back

  // Use pre-assigned connection (ensures disjoint connections per worker)
  TCPConnection* conn = req.assigned_conn;
  if (!conn || !conn->is_valid()) return false;

  // Send READ request header on data connection
  TCPDataHeader header;
  header.msg_type = static_cast<uint32_t>(TCPDataMsgType::READ_REQUEST);
  header.flags = 0;
  header.request_id = req.request_id;  // For tracking completion on return
  header.reserved = 0;
  header.dest_addr = req.dest_addr;      // Where to put data on our side
  header.remote_addr = req.remote_addr;  // Address to read from on remote side
  header.size = req.size;
  header.total_size = req.total_size;

  if (!send_exact(conn->fd, &header, sizeof(header))) {
    return false;
  }

  conn->inflight_chunks.fetch_sub(1, std::memory_order_relaxed);

  // Completion is handled by receiver worker when data arrives
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

  // Split threads: half for sender, half for receiver (minimum 1 each)
  size_t num_senders = std::max(size_t{1}, num_threads / 2);
  size_t num_receivers = std::max(size_t{1}, num_threads - num_senders);

  // Create sender workers (pass this for access to shared state)
  sender_workers_.reserve(num_senders);
  for (size_t i = 0; i < num_senders; ++i) {
    sender_workers_.push_back(std::make_unique<TCPSenderWorker>(i, this));
  }

  // Create receiver workers (pass this for access to shared state)
  receiver_workers_.reserve(num_receivers);
  for (size_t i = 0; i < num_receivers; ++i) {
    receiver_workers_.push_back(std::make_unique<TCPReceiverWorker>(i, this));
  }

  LOG(INFO) << "TCPThreadPool created with " << num_senders
            << " sender workers + " << num_receivers << " receiver workers";
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

uint32_t TCPThreadPool::assign_data_connection(int fd, TCPConnection* conn,
                                               RecvMatchQueue* match_queue) {
  uint32_t id = next_receiver_.fetch_add(1, std::memory_order_relaxed) %
                receiver_workers_.size();
  receiver_workers_[id]->add_data_connection(fd);
  conn->receiver_worker_id = id;

  id = next_sender_.fetch_add(1, std::memory_order_relaxed) %
       sender_workers_.size();
  conn->sender_worker_id = id;

  // Register fd -> match_queue mapping
  if (match_queue) {
    std::unique_lock<std::shared_mutex> lock(fd_match_queue_mutex_);
    fd_to_match_queue_[fd] = match_queue;
  }

  return id;
}

RecvMatchQueue* TCPThreadPool::get_match_queue(int fd) {
  std::shared_lock<std::shared_mutex> lock(fd_match_queue_mutex_);
  auto it = fd_to_match_queue_.find(fd);
  if (it == fd_to_match_queue_.end()) {
    return nullptr;
  }
  return it->second;
}

bool TCPThreadPool::submit_request(TCPRequest const& req) {
  uint32_t id = req.assigned_conn->sender_worker_id;
  return sender_workers_[id]->submit_request(req);
}

void TCPThreadPool::register_pending_recv(uint64_t dest_addr, size_t size,
                                          uint32_t request_id,
                                          std::atomic<bool>* completed,
                                          std::atomic<bool>* success) {
  // Register in shared pending_recvs_ map
  pending_recvs_.add(dest_addr, size, request_id, completed, success);
}

void TCPThreadPool::register_pending_send(size_t size, uint32_t request_id,
                                          std::atomic<bool>* completed,
                                          std::atomic<bool>* success) {
  // Register in shared pending_sends_ map
  pending_sends_.add(size, request_id, completed, success);
}

}  // namespace tcp
