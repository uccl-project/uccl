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
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
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
  uint32_t flags;       // Flags (kFlagLastChunk, kFlagNeedsMatch, etc.)
  uint32_t request_id;  // Request ID for completion tracking (unused for SEND
                        // with kFlagNeedsMatch, used for WRITE/READ)
  uint32_t reserved;
  uint64_t dest_addr;    // Destination GPU address (offset from base for SEND)
  uint64_t remote_addr;  // Remote address (for READ), unused for DATA_CHUNK
  uint64_t size;         // Size of this chunk/request
  uint64_t total_size;   // Total transfer size

  static constexpr uint32_t kFlagLastChunk = 1;
  // Flag to indicate this chunk needs matching (SEND operation)
  // When set, receiver looks up base_dest_addr from match queue using
  // request_id as send_seq_id
  static constexpr uint32_t kFlagNeedsMatch = 2;
};
static_assert(sizeof(TCPDataHeader) == 48, "TCPDataHeader size mismatch");

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
  uint64_t dest_addr;    // Destination GPU addr (offset for SEND, absolute for
                         // WRITE)
  uint64_t remote_addr;  // Remote addr to read from (for READ)
  std::atomic<bool>* completed;  // Completion flag
  std::atomic<bool>* success;    // Success flag
  uint32_t request_id;           // Request ID (for sender completion tracking)
  uint32_t send_seq_id;          // Sequence ID for matching (for SEND)
  uint32_t flags;                // Chunk flags (kFlagLastChunk, etc.)

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
        request_id(0),
        send_seq_id(0),
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

// Matching info for a pending recv (used for eager send/recv without
// negotiation)
struct RecvMatchInfo {
  uint64_t dest_addr;
  size_t size;
  uint32_t recv_request_id;
  std::atomic<size_t> received{0};  // Bytes received so far (for clearing
                                    // current_)
};

// Per-connection-group matching queue for eager send/recv.
// uccl_recv_async registers recv info here; receiver workers match incoming
// data with registered recvs in FIFO order using send_seq_id.
class RecvMatchQueue {
 public:
  // Called by uccl_recv_async to register a recv
  void push_recv(uint64_t dest_addr, size_t size, uint32_t recv_request_id);

  // Called by receiver worker to get recv info for a specific send_seq_id.
  // If send_seq_id is new, pops from queue and creates in_progress entry.
  // Returns false if no recv available (queue empty and not in_progress).
  bool get_recv_info(uint32_t send_seq_id, uint64_t* base_dest_addr,
                     uint32_t* recv_request_id);

  // Called after processing a chunk - adds bytes and removes from in_progress
  // when complete
  void add_received_bytes(uint32_t send_seq_id, size_t bytes);

  // Get next send_seq_id for sender (increments atomically)
  uint32_t get_next_send_seq_id();

 private:
  mutable std::mutex mutex_;
  std::deque<std::unique_ptr<RecvMatchInfo>> pending_recvs_;
  // Map from send_seq_id to in-progress transfer info
  std::unordered_map<uint32_t, std::unique_ptr<RecvMatchInfo>> in_progress_;
  // Next sequence number to assign when popping from pending_recvs_
  uint32_t next_seq_to_assign_{0};
  // Next send_seq_id to assign to senders (atomic for thread safety)
  std::atomic<uint32_t> next_send_seq_id_{0};
};

// Forward declaration
class TCPThreadPool;

// TCP Receiver Worker - uses epoll to wait on data connections
class TCPReceiverWorker {
 public:
  TCPReceiverWorker(uint32_t id, TCPThreadPool* thread_pool);
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
  TCPThreadPool* thread_pool_;  // Access to shared state (pending_recvs,
                                // match_queues)
  char* staging_buffer_;  // Pinned memory for efficient GPU-host transfers

  mutable std::mutex mutex_;
  std::unordered_set<int> data_fds_;
};

// TCP Sender Worker - processes send requests from jring
class TCPSenderWorker {
 public:
  TCPSenderWorker(uint32_t id, TCPThreadPool* thread_pool);
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
  TCPThreadPool* thread_pool_;  // Access to shared state (pending_sends, etc.)
};

// Connection group for a peer (ctrl + data connections)
struct TCPConnectionGroup {
  int ctrl_fd = -1;  // Control connection
  std::vector<std::unique_ptr<TCPConnection>> data_connections;
  std::atomic<uint64_t> round_robin_idx{0};
  mutable std::shared_mutex mutex;

  // Match queue for eager send/recv (recv registers here, receiver workers
  // match, and send_seq_id generation)
  RecvMatchQueue match_queue;

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
  // Also registers the fd -> match_queue mapping
  uint32_t assign_data_connection(int fd, TCPConnection* conn,
                                  RecvMatchQueue* match_queue);

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

  // Get the match queue for a given fd (used by receiver workers)
  RecvMatchQueue* get_match_queue(int fd);

  // Get pending recvs map (for receiver workers)
  PendingRecvMap* get_pending_recvs() { return &pending_recvs_; }

  // Get pending sends map (for sender workers)
  PendingSendMap* get_pending_sends() { return &pending_sends_; }

 private:
  PendingRecvMap pending_recvs_;  // Shared pending receives
  PendingSendMap pending_sends_;  // Shared pending sends
  std::vector<std::unique_ptr<TCPSenderWorker>> sender_workers_;
  std::vector<std::unique_ptr<TCPReceiverWorker>> receiver_workers_;
  std::atomic<uint32_t> next_sender_{0};
  std::atomic<uint32_t> next_receiver_{0};

  // Mapping from data fd to its connection group's match queue
  mutable std::shared_mutex fd_match_queue_mutex_;
  std::unordered_map<int, RecvMatchQueue*> fd_to_match_queue_;
};

// Async request tracking
struct TCPAsyncHandle {
  std::atomic<bool> completed{false};
  std::atomic<bool> success{false};
  uint32_t request_id{0};
};

}  // namespace tcp
