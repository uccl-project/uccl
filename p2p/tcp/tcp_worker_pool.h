#pragma once

#include "util/gpu_rt.h"
#include "util/jring.h"
#include "util/util.h"
#include <glog/logging.h>
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

// Toggle GPU memcpy (set to 0 for testing without GPU)
#define UCCL_TCP_GPU_MEMCPY 0
// #ifndef UCCL_TCP_GPU_MEMCPY
// #define UCCL_TCP_GPU_MEMCPY 1
// #endif

static constexpr size_t kStagingBufferSize =
    16 * 1024 * 1024;  // 16MB staging buffer
static constexpr size_t kDefaultTCPThreads =
    8;  // More threads for high bandwidth
static constexpr size_t kRequestRingSize = 1024;
static constexpr int kEpollMaxEvents = 64;

// Common TCP I/O helper functions
bool send_exact(int fd, void const* buf, size_t n);
bool recv_exact(int fd, void* buf, size_t n);
// Use sendmsg for scatter-gather I/O (header + data in one syscall)
bool send_header_and_data(int fd, void const* header, size_t header_size,
                          void const* data, size_t data_size);
static constexpr int kEpollTimeoutMs = 100;  // Reduce spinning

// Get number of TCP threads from environment
inline size_t get_tcp_thread_count() {
  char const* env = std::getenv("UCCL_P2P_TCP_THREADS");
  if (env && strlen(env) > 0) {
    int val = std::atoi(env);
    if (val > 0) {
      LOG(INFO) << "TCP: Using " << val << " threads from UCCL_P2P_TCP_THREADS";
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
  WRITE = 1,  // RDMA-style write (dest_addr already known, no ctrl message)
  READ = 3,   // RDMA-style read (sends request on data conn, receiver worker
              // handles incoming data)
  SHUTDOWN = 255
};

// Message types on data connections (first 4 bytes of any message)
enum class TCPDataMsgType : uint32_t {
  DATA_CHUNK = 0,    // TCPDataHeader for SEND/WRITE data
  READ_REQUEST = 1,  // TCPDataHeader for READ request
};

// Unified message header for all data connection messages (DATA_CHUNK and
// READ_REQUEST) This eliminates the need for MSG_PEEK to determine message type
struct TCPDataHeader {
  uint32_t msg_type;    // TCPDataMsgType (DATA_CHUNK or READ_REQUEST)
  uint32_t flags;       // Flags (kFlagLastChunk, etc.)
  uint32_t request_id;  // Request ID for completion tracking
  uint32_t reserved;
  uint64_t dest_addr;    // Destination GPU address
  uint64_t remote_addr;  // Remote address (for READ), unused for DATA_CHUNK
  uint64_t size;         // Size of this chunk/request
  uint64_t total_size;   // Total transfer size

  static constexpr uint32_t kFlagLastChunk = 1;
};
static_assert(sizeof(TCPDataHeader) == 48, "TCPDataHeader size mismatch");

// Message sent on control connection by receiver to tell sender where to put
// data
struct RecvReadyMsg {
  uint64_t dest_addr;   // GPU buffer address on receiver
  uint64_t size;        // Expected size
  uint32_t request_id;  // For matching (future use)
  uint32_t reserved;
};
static_assert(sizeof(RecvReadyMsg) == 24, "RecvReadyMsg size mismatch");

// Forward declarations
struct TCPConnection;
struct TCPConnectionGroup;
class TCPReceiverWorker;
class TCPSenderWorker;

// TCP request structure (submitted via jring)
struct alignas(64) TCPRequest {
  TCPRequestType type;
  int ctrl_fd;           // Control connection fd
  void* data;            // GPU memory pointer (local buffer)
  size_t size;           // Chunk size to transfer
  size_t total_size;     // Total transfer size (for header)
  uint64_t dest_addr;    // Destination GPU addr (includes offset for chunks)
  uint64_t remote_addr;  // Remote addr to read from (for READ)
  std::atomic<bool>* completed;  // Completion flag
  std::atomic<bool>* success;    // Success flag
  uint32_t send_request_id;      // Sender's request ID (for pending_sends_)
  uint32_t recv_request_id;  // Receiver's request ID (for DATA_CHUNK header)
  uint32_t flags;            // Chunk flags (kFlagLastChunk, etc.)

  // For sender: pointer to connection group for load-balanced sending
  void* conn_group;  // TCPConnectionGroup*

  // Pre-assigned connection (selected before submission, routes to its owner)
  TCPConnection* assigned_conn;

  TCPRequest()
      : type(TCPRequestType::SEND),
        ctrl_fd(-1),
        data(nullptr),
        size(0),
        total_size(0),
        dest_addr(0),
        remote_addr(0),
        completed(nullptr),
        success(nullptr),
        send_request_id(0),
        recv_request_id(0),
        flags(0),
        conn_group(nullptr),
        assigned_conn(nullptr) {}
};

// TCP connection wrapper
struct TCPConnection {
  int fd = -1;
  std::string local_ip;
  std::string remote_ip;
  int remote_port = 0;
  std::atomic<uint32_t> inflight_chunks{0};
  uint32_t sender_worker_id = 0;  // Which sender worker owns this connection
  uint32_t receiver_worker_id =
      0;  // Which receiver worker monitors this connection

  TCPConnection() = default;

  TCPConnection(TCPConnection&& other) noexcept
      : fd(other.fd),
        local_ip(std::move(other.local_ip)),
        remote_ip(std::move(other.remote_ip)),
        remote_port(other.remote_port),
        inflight_chunks(other.inflight_chunks.load()),
        sender_worker_id(other.sender_worker_id),
        receiver_worker_id(other.receiver_worker_id) {
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

// Pending transfer tracking (shared across workers)
struct alignas(64) PendingTransfer {
  size_t total_size;
  std::atomic<size_t> transferred_size{0};
  uint32_t request_id;
  std::atomic<bool>* completed;
  std::atomic<bool>* success;

  PendingTransfer() = default;
  PendingTransfer(size_t size, uint32_t req_id, std::atomic<bool>* comp,
                  std::atomic<bool>* succ)
      : total_size(size),
        transferred_size(0),
        request_id(req_id),
        completed(comp),
        success(succ) {}
};

// Global pending receives map (shared across all receiver workers)
class PendingRecvMap {
 public:
  void add(uint64_t dest_addr, size_t size, uint32_t request_id,
           std::atomic<bool>* completed, std::atomic<bool>* success);

  // Update received size and check if complete. Returns true if this call
  // completed the receive. Keyed by request_id to support chunked transfers.
  // Completion is based purely on received_size >= total_size (not last chunk
  // flag)
  bool update_and_check_complete(uint32_t request_id, size_t chunk_size);

 private:
  mutable std::mutex mutex_;
  // Keyed by request_id (not dest_addr) to support chunked transfers
  std::unordered_map<uint32_t, std::unique_ptr<PendingTransfer>> pending_recvs_;
};

// Global pending sends map (shared across all sender workers)
class PendingSendMap {
 public:
  void add(size_t size, uint32_t request_id, std::atomic<bool>* completed,
           std::atomic<bool>* success);

  // Update sent size and check if complete. Returns true if this call
  // completed the send.
  bool update_and_check_complete(uint32_t request_id, size_t chunk_size);

 private:
  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, std::unique_ptr<PendingTransfer>> pending_sends_;
};

// TCP Receiver Worker - uses epoll to wait on data connections
class TCPReceiverWorker {
 public:
  TCPReceiverWorker(uint32_t id, PendingRecvMap* pending_recvs);
  ~TCPReceiverWorker();

  void start();
  void stop();

  // Add a data connection to epoll
  bool add_data_connection(int fd);
  void remove_data_connection(int fd);

  uint32_t id() const { return worker_id_; }

 private:
  void worker_loop();
  void process_event(int fd);
  void process_read_request(int fd, TCPDataHeader const& header);
  void process_data_chunk(int fd, TCPDataHeader const& header);

  uint32_t worker_id_;
  std::atomic<bool> running_;
  int epoll_fd_;
  std::thread worker_thread_;
  PendingRecvMap* pending_recvs_;  // Shared across all workers
  char* staging_buffer_;  // Pinned memory for efficient GPU-host transfers

  mutable std::mutex mutex_;
  std::unordered_set<int> data_fds_;
};

// TCP Sender Worker - processes send requests from jring
class TCPSenderWorker {
 public:
  TCPSenderWorker(uint32_t id, PendingSendMap* pending_sends);
  ~TCPSenderWorker();

  void start();
  void stop();

  bool submit_request(TCPRequest const& req);

  uint32_t id() const { return worker_id_; }

 private:
  void worker_loop();
  bool process_requests();

  bool do_send(TCPRequest& req);
  bool do_write(TCPRequest& req);
  bool do_read(TCPRequest& req);

  uint32_t worker_id_;
  std::atomic<bool> running_;
  jring_t* request_ring_;
  char* staging_buffer_;  // Pinned memory for efficient GPU-host transfers
  std::thread worker_thread_;
  PendingSendMap* pending_sends_;  // Shared across all sender workers
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

  // Select a connection and return it (caller routes request to its owner)
  TCPConnection* select_data_connection();

  // Add connection and assign to sender worker round-robin
  void add_data_connection(std::unique_ptr<TCPConnection> conn);

  size_t data_connection_count() const;
};

// TCP Thread Pool (sender workers + receiver workers)
class TCPThreadPool {
 public:
  explicit TCPThreadPool(size_t num_threads = 0);
  ~TCPThreadPool();

  void start();
  void stop();

  // Assign a data connection to a receiver worker (round-robin)
  uint32_t assign_data_connection(int fd, TCPConnection* conn);

  // Submit request to sender worker (routes based on assigned_conn's owner)
  bool submit_request(TCPRequest const& req);

  // Register a pending receive (shared across all receiver workers)
  void register_pending_recv(uint64_t dest_addr, size_t size,
                             uint32_t request_id, std::atomic<bool>* completed,
                             std::atomic<bool>* success);

  // Register a pending send (shared across all sender workers)
  void register_pending_send(size_t size, uint32_t request_id,
                             std::atomic<bool>* completed,
                             std::atomic<bool>* success);

 private:
  PendingRecvMap pending_recvs_;  // Shared pending receives
  PendingSendMap pending_sends_;  // Shared pending sends
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
