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
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace tcp {

#define UCCL_TCP_GPU_MEMCPY 0

static constexpr size_t kStagingBufferSize = 16 * 1024 * 1024;
static constexpr size_t kDefaultTCPThreads = 10;
static constexpr size_t kRequestRingSize = 1024;
static constexpr int kEpollMaxEvents = 64;

bool send_exact(int fd, void const* buf, size_t n);
bool recv_exact(int fd, void* buf, size_t n);
bool send_header_and_data(int fd, void const* header, size_t header_size,
                          void const* data, size_t data_size);
static constexpr int kEpollTimeoutMs = 100;

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

enum class TCPRequestType : uint32_t {
  SEND = 0,
  WRITE = 1,
  READ = 3,
  SHUTDOWN = 255
};

enum class TCPDataMsgType : uint32_t {
  DATA_CHUNK = 0,
  READ_REQUEST = 1,
};

struct TCPDataHeader {
  uint32_t msg_type;
  uint32_t flags;
  uint32_t request_id;
  uint32_t reserved;
  uint64_t dest_addr;
  uint64_t remote_addr;
  uint64_t size;
  uint64_t total_size;

  static constexpr uint32_t kFlagLastChunk = 1;
  static constexpr uint32_t kFlagNeedsMatch = 2;
};
static_assert(sizeof(TCPDataHeader) == 48, "TCPDataHeader size mismatch");

struct TCPConnection;
struct TCPConnectionGroup;
class TCPReceiverWorker;
class TCPSenderWorker;

struct alignas(64) TCPRequest {
  TCPRequestType type;
  int ctrl_fd;
  void* data;
  size_t size;
  size_t total_size;
  uint64_t dest_addr;
  uint64_t remote_addr;
  std::atomic<bool>* completed;
  std::atomic<bool>* success;
  uint32_t request_id;
  uint32_t send_seq_id;
  uint32_t flags;
  void* conn_group;
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

struct TCPConnection {
  int fd = -1;
  std::string local_ip;
  std::string remote_ip;
  int remote_port = 0;
  uint32_t sender_worker_id = 0;
  uint32_t receiver_worker_id = 0;

  TCPConnection() = default;

  TCPConnection(TCPConnection&& other) noexcept
      : fd(other.fd),
        local_ip(std::move(other.local_ip)),
        remote_ip(std::move(other.remote_ip)),
        remote_port(other.remote_port),
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

class PendingRecvMap {
 public:
  void add(uint64_t dest_addr, size_t size, uint32_t request_id,
           std::atomic<bool>* completed, std::atomic<bool>* success);
  bool update_and_check_complete(uint32_t request_id, size_t chunk_size);

 private:
  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, std::unique_ptr<PendingTransfer>> pending_recvs_;
};

class PendingSendMap {
 public:
  void add(size_t size, uint32_t request_id, std::atomic<bool>* completed,
           std::atomic<bool>* success);
  bool update_and_check_complete(uint32_t request_id, size_t chunk_size);

 private:
  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, std::unique_ptr<PendingTransfer>> pending_sends_;
};

struct RecvMatchInfo {
  uint64_t dest_addr;
  size_t size;
  uint32_t recv_request_id;
  std::atomic<size_t> received{0};
};

class RecvMatchQueue {
 public:
  void push_recv(uint64_t dest_addr, size_t size, uint32_t recv_request_id);
  bool get_recv_info(uint32_t send_seq_id, uint64_t* base_dest_addr,
                     uint32_t* recv_request_id);
  void add_received_bytes(uint32_t send_seq_id, size_t bytes);
  uint32_t get_next_send_seq_id();

 private:
  mutable std::mutex mutex_;
  std::deque<std::unique_ptr<RecvMatchInfo>> pending_recvs_;
  std::unordered_map<uint32_t, std::unique_ptr<RecvMatchInfo>> in_progress_;
  uint32_t next_seq_to_assign_{0};
  std::atomic<uint32_t> next_send_seq_id_{0};
};

class TCPThreadPool;

class TCPReceiverWorker {
 public:
  TCPReceiverWorker(uint32_t id, TCPThreadPool* thread_pool);
  ~TCPReceiverWorker();

  void start();
  void stop();
  bool add_data_connection(int fd);
  void remove_data_connection(int fd);
  uint32_t id() const { return worker_id_; }

 private:
  void worker_loop();
  bool process_event(int fd);
  void process_read_request(int fd, TCPDataHeader const& header);
  void process_data_chunk(int fd, TCPDataHeader const& header);

  uint32_t worker_id_;
  std::atomic<bool> running_;
  int epoll_fd_;
  std::thread worker_thread_;
  TCPThreadPool* thread_pool_;
  char* staging_buffer_;
  mutable std::mutex mutex_;
  std::unordered_set<int> data_fds_;
};

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
  char* staging_buffer_;
  std::thread worker_thread_;
  TCPThreadPool* thread_pool_;
};

struct TCPConnectionGroup {
  int ctrl_fd = -1;
  std::vector<std::unique_ptr<TCPConnection>> data_connections;
  std::atomic<uint64_t> round_robin_idx{0};
  mutable std::shared_mutex mutex;
  RecvMatchQueue match_queue;

  TCPConnection* select_data_connection();
  void add_data_connection(std::unique_ptr<TCPConnection> conn);
  size_t data_connection_count() const;
};

class TCPThreadPool {
 public:
  explicit TCPThreadPool(size_t num_threads = 0);
  ~TCPThreadPool();

  void start();
  void stop();
  uint32_t assign_data_connection(int fd, TCPConnection* conn,
                                  RecvMatchQueue* match_queue);
  bool submit_request(TCPRequest const& req);
  void register_pending_recv(uint64_t dest_addr, size_t size,
                             uint32_t request_id, std::atomic<bool>* completed,
                             std::atomic<bool>* success);
  void register_pending_send(size_t size, uint32_t request_id,
                             std::atomic<bool>* completed,
                             std::atomic<bool>* success);
  RecvMatchQueue* get_match_queue(int fd);
  PendingRecvMap* get_pending_recvs() { return &pending_recvs_; }
  PendingSendMap* get_pending_sends() { return &pending_sends_; }

 private:
  PendingRecvMap pending_recvs_;
  PendingSendMap pending_sends_;
  std::vector<std::unique_ptr<TCPSenderWorker>> sender_workers_;
  std::vector<std::unique_ptr<TCPReceiverWorker>> receiver_workers_;
  std::atomic<uint32_t> next_sender_{0};
  std::atomic<uint32_t> next_receiver_{0};
  mutable std::shared_mutex fd_match_queue_mutex_;
  std::unordered_map<int, RecvMatchQueue*> fd_to_match_queue_;
};

struct TCPAsyncHandle {
  std::atomic<bool> completed{false};
  std::atomic<bool> success{false};
  uint32_t request_id{0};
};

}  // namespace tcp
