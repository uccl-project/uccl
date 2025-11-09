#pragma once

#include "tcpx/device/unpack_launch.h"
#include "tcpx/include/bootstrap.h"
#include "tcpx/include/tcpx_interface.h"
#include "tcpx/include/unpack_descriptor.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

extern thread_local bool inside_python;

namespace tcpx {

// Connection state shared across the endpoint and its dedicated progress worker.
// Each connection owns its TCPX handles, CUDA event pool, and the
// ConnProgressWorker that keeps its transfers moving (producer-consumer model).
class Endpoint;

struct Conn : std::enable_shared_from_this<Conn> {
  Conn() {
    recv_dev_handle = recv_dev_handle_storage.data();
    send_dev_handle = send_dev_handle_storage.data();
    std::memset(recv_dev_handle_storage.data(), 0,
                recv_dev_handle_storage.size());
    std::memset(send_dev_handle_storage.data(), 0,
                send_dev_handle_storage.size());
  }

  uint64_t conn_id = 0;
  std::string ip_addr;
  int remote_gpu_idx = -1;
  int remote_port = -1;
  int ctrl_sock_fd = -1;

  // CUDA Event Pool for recv unpack operations (循环复用)
  // 对齐原来的 ChannelWindow::events 设计。池大小与接收窗口上限保持一致，
  // 在 connect/accept 时根据 UCCL_TCPX_MAX_RECV_INFLIGHT 计算并初始化。
  std::vector<cudaEvent_t> recv_events;
  size_t recv_event_pool_size = 0;  // = recv_events.size()
  uint64_t event_counter = 0;       // 事件循环复用计数器

  // TCPX plugin handles
  void* send_comm = nullptr;
  void* recv_comm = nullptr;
  void* send_dev_handle = nullptr;
  void* recv_dev_handle = nullptr;

  alignas(16) std::array<uint8_t, 512> recv_dev_handle_storage;
  alignas(16) std::array<uint8_t, 512> send_dev_handle_storage;

  // Per-connection progress worker state. The nested struct encapsulates the
  // background thread plus its synchronization primitives so teardown can stop
  // the worker before TCPX handles are destroyed.
  struct ConnProgressWorker {
    ConnProgressWorker(Endpoint& endpoint, std::shared_ptr<Conn> owner);
    ~ConnProgressWorker();
    void enqueue(uint64_t transfer_id);
    void stop();

   private:
    void run();
    Endpoint& endpoint_;
    std::weak_ptr<Conn> conn_;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<uint64_t> queue_;
    std::unordered_set<uint64_t> inflight_;
    bool running_ = true;
  };
  std::unique_ptr<ConnProgressWorker> progress_worker;
};

struct FifoItem {
  uint64_t mr_id;    // Registered memory identifier advertised to the peer
  uint32_t size;     // Payload size that should be transferred
  uint32_t tag;      // TCPX-side tag used to match isend/irecv operations
  uint64_t offset;   // Byte offset within the registered MR base pointer
  uint64_t token;    // Reserved for future metadata (kept for alignment)
  char padding[32];  // Preserve 64-byte layout expected by uccl listener
};
static_assert(sizeof(struct FifoItem) == 64, "FifoItem size is not 64 bytes");

struct MrEntry {
  void* base = nullptr;
  size_t size = 0;
  int ptr_type = NCCL_PTR_CUDA;
  bool is_recv = true;
  // Cache the TCPX registration handles for each connection so we only call
  // tcpx_reg_mr once per direction.
  std::unordered_map<uint64_t, void*> send_handles;  // conn_id -> mhandle
  std::unordered_map<uint64_t, void*> recv_handles;  // conn_id -> mhandle
};

struct PendingTransfer {
  struct ChunkState {
    size_t offset = 0;
    size_t bytes = 0;
    uint32_t tag = 0;
    void* request = nullptr;
    void* dst_ptr = nullptr;
    bool needs_unpack = false;
    bool stage1_done = false;
    bool stage2_done = false;
    bool posted = false;
    // Bounce-buffer metadata used to launch the unpack kernel once the network
    // transfer finishes.
    rx::UnpackDescriptorBlock desc_block{};
    // CUDA event 从 Conn::recv_events pool 中获取（不拥有，只是引用）
    cudaEvent_t event = nullptr;
    // Event 在 pool 中的索引（用于调试）
    size_t event_idx = 0;
  };

  enum class Kind { kSend, kRecv, kRead };

  Kind kind = Kind::kRecv;
  uint64_t transfer_id = 0;
  uint64_t conn_id = 0;
  uint64_t mr_id = 0;
  size_t total_bytes = 0;
  uint32_t base_tag = 0;
  size_t next_chunk_to_post = 0;
  void* mhandle = nullptr;
  // Transfer is expressed as a vector of chunks that flow through
  // (TCP completion -> optional GPU unpack -> completion).
  std::vector<ChunkState> chunks;
  size_t chunks_completed = 0;
  std::deque<size_t> send_queue;
  std::deque<size_t> recv_stage1_queue;
  std::deque<size_t> recv_stage2_queue;
};

class Endpoint {
 public:
  /*
   * Create engine threads running in background for a single interface.
   *
   * input:
   *   local_gpu_idx: the GPU index to use for the engine
   *   num_cpus: the number of CPUs to use for the engine
   */
  Endpoint(uint32_t const local_gpu_idx, uint32_t const num_cpus);

  ~Endpoint();

  /*
   * Connect to a remote server via TCP.
   *
   * input:
   *   ip_addr: the IP address of the remote server
   *   remote_gpu_idx: the GPU index of the remote server
   *   remote_port: the port of the remote server (optional)
   * output:
   *   conn_id: the ID of the connection
   */
  bool connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
               uint64_t& conn_id);

  /*
   * Accept an incoming connection via TCP.
   *
   * output:
   *   ip_addr: the IP address of the remote server
   *   remote_gpu_idx: the GPU index of the remote server
   *   conn_id: the ID of the connection
   */
  bool accept(std::string& ip_addr, int& remote_gpu_idx, uint64_t& conn_id);

  std::vector<uint8_t> get_metadata();
  /*
   * Parse endpoint metadata to extract IP address, port, and GPU index.
   * Returns a tuple of (ip_address, port, gpu_index).
   */
  static std::tuple<std::string, uint16_t, int> parse_metadata(
      std::vector<uint8_t> const& metadata);

  bool advertise(uint64_t conn_id, uint64_t mr_id, void* addr, size_t len,
                 char* out_buf);
  bool queue_read_response(uint64_t conn_id, FifoItem const& fifo_item);
  uint32_t allocate_tag() { return next_tag_.fetch_add(1); }

  int get_sock_fd(uint64_t conn_id) const {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_map_.find(conn_id);
    if (it == conn_map_.end()) return -1;
    return it->second->ctrl_sock_fd;
  }

  /*Register the data with a specific interface. */
  bool reg(void const* data, size_t size, uint64_t& mr_id);
  bool dereg(uint64_t mr_id);
  bool find_mr_by_addr(uintptr_t addr, size_t size, uint64_t* mr_id) const;

  /* Read data from the remote server asynchronously. */
  bool read_async(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
                  FifoItem const& slot_item, uint64_t* transfer_id);
  /* Send data to the remote server asynchronously. */
  bool send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                  size_t size, uint64_t* transfer_id);
  bool send_async_with_tag(uint64_t conn_id, uint64_t mr_id, void const* data,
                           size_t size, uint32_t tag, uint64_t* transfer_id);
  /* Receive data from the remote server asynchronously. */
  bool recv_async(uint64_t conn_id, uint64_t mr_id, void* data, size_t size,
                  uint64_t* transfer_id);
  bool recv_async_with_tag(uint64_t conn_id, uint64_t mr_id, void* data,
                           size_t size, uint32_t tag, uint64_t* transfer_id);

  /* Poll the status of the asynchronous receive. */
  bool poll_async(uint64_t transfer_id, bool* is_done);
  bool progress_conn(uint64_t conn_id);

  size_t chunk_bytes() const { return chunk_bytes_; }

 private:
  int dev_id_ = -1;
  int ctrl_listen_fd_ = -1;
  void* listen_comms_ = nullptr;
  uint32_t local_gpu_idx_ = 0;
  int ctrl_port_ = 0;
  ncclNetHandle_v7 listen_handle_{};

  std::atomic<uint64_t> next_conn_id_ = 0;
  std::atomic<uint64_t> next_mr_id_{1};
  std::atomic<uint64_t> next_transfer_id_{1};
  std::atomic<uint32_t> next_tag_{1};

  mutable std::shared_mutex conn_mu_;
  std::unordered_map<uint64_t, std::shared_ptr<Conn>> conn_map_;

  mutable std::mutex mr_mu_;
  std::unordered_map<uint64_t, MrEntry> mr_map_;

  mutable std::mutex transfer_mu_;
  std::unordered_map<uint64_t, PendingTransfer> transfer_map_;

  cudaStream_t unpack_stream_ = nullptr;
  std::unique_ptr<device::UnpackLauncher> unpack_launcher_;

  size_t chunk_bytes_ = 0;
  bool debug_enabled_ = false;
  CUdevice cu_device_ = 0;
  CUcontext cu_context_ = nullptr;

  mutable std::mutex window_mu_;
  // Sliding-window counters (how many chunks are currently in-flight per
  // connection for send/recv directions).
  std::unordered_map<uint64_t, size_t> send_inflight_chunks_;
  std::unordered_map<uint64_t, size_t> recv_inflight_chunks_;
  std::condition_variable window_cv_;

  enum class ScheduleOutcome { kNoProgress, kProgress, kError };

  void free_conn_(std::shared_ptr<Conn> const& conn);
  ScheduleOutcome schedule_send_chunks_locked(Conn& conn,
                                              PendingTransfer& transfer);
  ScheduleOutcome schedule_recv_chunks_locked(Conn& conn,
                                              PendingTransfer& transfer);
  bool reserve_send_slot(uint64_t conn_id, size_t limit);
  bool reserve_recv_slot(uint64_t conn_id, size_t limit);
  void release_send_slot(uint64_t conn_id);
  void release_recv_slot(uint64_t conn_id);
  bool populate_conn_handles_(Conn& conn, uint64_t mr_id, bool is_recv,
                              void** mhandle_out);
  bool progress_transfer_locked(Conn& conn, PendingTransfer& transfer,
                                bool* schedule_send, bool* schedule_recv);
  // Unified advancement helper that assumes transfer_mu_ is already held. It
  // schedules additional chunks (Stage 0) and runs Stage 1/2. The bool output
  // tells callers whether the transfer finished so they can cleanup outside the
  // critical section.
  bool advance_transfer_locked(Conn& conn, PendingTransfer& transfer,
                               bool* transfer_complete);
  void finalize_transfer_locked(
      std::unordered_map<uint64_t, PendingTransfer>::iterator it, Conn& conn);
  // Utility to reset per-connection inflight counters and wake any producers
  // waiting for window space.
  void reset_conn_window_counters_(uint64_t conn_id);
  bool poll_chunk_request_(PendingTransfer& transfer,
                           PendingTransfer::ChunkState& chunk, bool* done,
                           int* received_size);
  bool enqueue_chunk_unpack_(PendingTransfer& transfer,
                             PendingTransfer::ChunkState& chunk,
                             tcpx::plugin::tcpxRequest* request, Conn& conn);
  bool finalize_recv_chunk_(Conn& conn, PendingTransfer::ChunkState& chunk);
  bool enqueue_unpack_(PendingTransfer& transfer,
                       tcpx::plugin::tcpxRequest* request, Conn& conn);
  bool complete_pending_transfer_(PendingTransfer& transfer, bool success);
  bool poll_request_(PendingTransfer& transfer, bool* done, int* received_size);
  bool post_send_(Conn& conn, uint64_t mr_id, MrEntry const& mr,
                  void const* data, size_t size, int tag,
                  uint64_t& transfer_id);
  bool post_recv_(Conn& conn, uint64_t mr_id, MrEntry const& mr, void* data,
                  size_t size, int tag, uint64_t& transfer_id,
                  bool needs_unpack);
  void start_conn_progress_worker_(std::shared_ptr<Conn> const& conn);
  void stop_conn_progress_worker_(std::shared_ptr<Conn> const& conn);
  void enqueue_transfer_for_progress_(std::shared_ptr<Conn> const& conn,
                                      uint64_t transfer_id);
  bool drive_transfer_(std::shared_ptr<Conn> const& conn, uint64_t transfer_id,
                       bool* transfer_done);

  friend struct Conn::ConnProgressWorker;
};

}  // namespace tcpx
