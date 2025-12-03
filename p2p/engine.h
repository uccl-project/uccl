#pragma once

#include "transport.h"
#include "util/gpu_rt.h"
#include "util/jring.h"
#include "util/net.h"
#include "util/shared_pool.h"
#include "util/util.h"
#include <infiniband/verbs.h>
#include <pybind11/pybind11.h>
#include <atomic>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <variant>
#include "efa_endpoint/efa_endpoint.h"
#include "transport.h"

namespace py = pybind11;

extern thread_local bool inside_python;
namespace my_namespace {

struct P2PMhandle {
  struct uccl::Mhandle* mhandle_;
  std::unordered_map<int64_t, struct ibv_mr*> mr_map;
};

using RDMAEndPoint =
    std::variant<uccl::RDMAEndpoint*, std::shared_ptr<EFAEndpoint>>;
}
struct MR {
  uint64_t mr_id_;
  my_namespace::P2PMhandle* mhandle_;
};

struct Conn {
  uint64_t conn_id_;
  uccl::ConnID uccl_conn_id_;
  std::string ip_addr_;
  int remote_gpu_idx_;
  int uds_sockfd_ = -1;  // Unix Domain Socket file descriptor for local IPC
};

struct PeerInfo {
  std::string ip_addr;  // IP address of the peer
  int gpu_idx;          // GPU index of the peer
};



class Endpoint {
  uint64_t const kRTTBytes = 1024 * 1024;
  uint64_t const kChunkSize = 1024 * 1024;
  uint32_t const kMaxInflightChunks = 8;
  static constexpr size_t kIpcAlignment = 1ul << 20;
  static constexpr size_t kIpcSizePerEngine = 1ul << 20;

 public:
  // Prepare transfer info structure for receiving IPC handle
  struct IpcTransferInfo {
    gpuIpcMemHandle_t handle;
    uintptr_t offset;
    size_t size;
    uint32_t operation;  // 0 = send_ipc request, 1 = recv_ipc response
  };

  /*
   * Create engine threads running in background for a single interface. It also
   * opens a TCP listening thread waiting for incoming connections.
   *
   * input:
   *   local_gpu_idx: the GPU index to use for the engine
   *   num_cpus: the number of CPUs to use for the engine
   */
  Endpoint(uint32_t const local_gpu_idx, uint32_t const num_cpus,
           uint32_t const rank_id);

  /*
   * Create endpoint without intializing the engine. Lazy creation of engine is
   * done during  memory registration. Additionally, open a unified P2P socket
   * for metadata exchanges.
   *
   * input:
   *   num_cpus: the number of CPUs to use for the engine
   */
  Endpoint(uint32_t const num_cpus, uint32_t const rank_id = 0);
  ~Endpoint();

  /*
   * Connect to a remote server via TCP, then build RDMA QP connections.
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

  std::vector<uint8_t> get_metadata();

  /*
   * Get the unified metadata for all devices.
   */
  std::vector<uint8_t> get_unified_metadata();

  /*
   * Parse endpoint metadata to extract IP address, port, and GPU index.
   * Returns a tuple of (ip_address, port, gpu_index).
   */
  static std::tuple<std::string, uint16_t, int> parse_metadata(
      std::vector<uint8_t> const& metadata);

  /*
   * Accept an incoming connection via TCP, then build RDMA QP connections.
   *
   * output:
   *   ip_addr: the IP address of the remote server
   *   remote_gpu_idx: the GPU index of the remote server
   *   conn_id: the ID of the connection
   */
  bool accept(std::string& ip_addr, int& remote_gpu_idx, uint64_t& conn_id);

  /*Register the data with a specific interface. */
  bool reg(void const* data, size_t size, uint64_t& mr_id);

  bool regv(std::vector<void const*> const& data_v,
            std::vector<size_t> const& size_v, std::vector<uint64_t>& mr_id_v);
  bool dereg(uint64_t mr_id);

  /*Send data to the remote server. Blocking. */
  bool send(uint64_t conn_id, uint64_t mr_id, void const* data, size_t size);

  /*Receive data from the remote server. Blocking.*/
  bool recv(uint64_t conn_id, uint64_t mr_id, void* data, size_t size);

  /* Send data to the remote server asynchronously. */
  bool send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                  size_t size, uint64_t* transfer_id);

  /* Receive data from the remote server asynchronously. */
  bool recv_async(uint64_t conn_id, uint64_t mr_id, void* data, size_t size,
                  uint64_t* transfer_id);

  /* Send a vector of data chunks. Blocking. */
  bool sendv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<void const*> data_v, std::vector<size_t> size_v,
             size_t num_iovs);

  /* Send a vector of data chunks asynchronously. */
  bool sendv_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                   std::vector<void const*> data_v, std::vector<size_t> size_v,
                   size_t num_iovs, uint64_t* transfer_id);

  /* Receive a vector of data chunks. Blocking. */
  bool recvv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<void*> data_v, std::vector<size_t> size_v,
             size_t num_iovs);

  /* Receive a vector of data chunks asynchronously. */
  bool recvv_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                   std::vector<void*> data_v, std::vector<size_t> size_v,
                   size_t num_iovs, uint64_t* transfer_id);

  /* Read data from the remote server. Blocking. */
  bool read(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
            uccl::FifoItem const& slot_item);

  /* Read data from the remote server asynchronously. */
  bool read_async(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
                  uccl::FifoItem const& slot_item, uint64_t* transfer_id);

  /* Read a vector of data chunks. */
  bool readv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<void*> dst_v, std::vector<size_t> size_v,
             std::vector<uccl::FifoItem> slot_item_v, size_t num_iovs);

  /* Read a vector of data chunks asynchronously. */
  bool readv_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                   std::vector<void*> dst_v, std::vector<size_t> size_v,
                   std::vector<uccl::FifoItem> slot_item_v, size_t num_iovs,
                   uint64_t* transfer_id);

  /* Write data to the remote server. Blocking. */
  bool write(uint64_t conn_id, uint64_t mr_id, void* src, size_t size,
             uccl::FifoItem const& slot_item);

  /* Write data to the remote server asynchronously. */
  bool write_async(uint64_t conn_id, uint64_t mr_id, void* src, size_t size,
                   uccl::FifoItem const& slot_item, uint64_t* transfer_id);

  /* Write a vector of data chunks. */
  bool writev(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
              std::vector<void*> src_v, std::vector<size_t> size_v,
              std::vector<uccl::FifoItem> slot_item_v, size_t num_iovs);

  /* Write a vector of data chunks asynchronously. */
  bool writev_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                    std::vector<void*> src_v, std::vector<size_t> size_v,
                    std::vector<uccl::FifoItem> slot_item_v, size_t num_iovs,
                    uint64_t* transfer_id);

  /* Write data to the remote server via CUDA/HIP IPC. Blocking. */
  bool write_ipc(uint64_t conn_id, uint64_t mr_id, void const* data,
                 size_t size, void const* meta, size_t meta_len);

  bool advertise(uint64_t conn_id, uint64_t mr_id, void* addr, size_t len,
                 char* out_buf);

  /* Advertise a vector of data chunks. */
  bool advertisev(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                  std::vector<void*> addr_v, std::vector<size_t> len_v,
                  std::vector<char*> out_buf_v, size_t num_iovs);

  /*Connect to a local process via Unix Domain Socket.*/
  bool connect_local(int remote_gpu_idx, uint64_t& conn_id);

  /*Accept an incoming local connection via Unix Domain Socket. */
  bool accept_local(int& remote_gpu_idx, uint64_t& conn_id);

  /* Send data to the remote server via CUDA/HIP IPC. Blocking. The
   * gpuIpcMemHandle_t will be passed via UDS from recv_ipc to send_ipc
   * function. */
  bool send_ipc(uint64_t conn_id, void* data, size_t size);

  bool recv_ipc(uint64_t conn_id, void* data, size_t size);

  bool send_ipc_async(uint64_t conn_id, void const* data, size_t size,
                      uint64_t* transfer_id);

  bool recv_ipc_async(uint64_t conn_id, void* data, size_t size,
                      uint64_t* transfer_id);

  /* One-sided write and read via IPC. */
  bool write_ipc(uint64_t conn_id, void const* data, size_t size,
                 IpcTransferInfo const& info);
  bool read_ipc(uint64_t conn_id, void* data, size_t size,
                IpcTransferInfo const& info);
  bool write_ipc_async(uint64_t conn_id, void const* data, size_t size,
                       IpcTransferInfo const& info, uint64_t* transfer_id);
  bool read_ipc_async(uint64_t conn_id, void* data, size_t size,
                      IpcTransferInfo const& info, uint64_t* transfer_id);
  bool advertise_ipc(uint64_t conn_id, void* addr, size_t len, char* out_buf);
  bool advertisev_ipc(uint64_t conn_id, std::vector<void*> addr_v,
                      std::vector<size_t> len_v, std::vector<char*> out_buf_v,
                      size_t num_iovs);

  /* Poll the status of the asynchronous receive. */
  bool poll_async(uint64_t transfer_id, bool* is_done);

  int get_sock_fd(uint64_t conn_id) const {
    auto it = conn_id_to_conn_.find(conn_id);
    if (it == conn_id_to_conn_.end()) {
      return -1;
    }
    return it->second->uccl_conn_id_.sock_fd;
  }

  /** Returns conn_id for @rank, or UINT64_MAX if unknown. */
  uint64_t conn_id_of_rank(int rank) const {
    auto it = rank2conn_.find(rank);
    return it != rank2conn_.end() ? it->second : UINT64_MAX;
  }

 private:
  gpuStream_t pick_stream() {
    if (streams_.empty()) return nullptr;
    uint32_t i =
        rr_stream_.fetch_add(1, std::memory_order_relaxed) % streams_.size();
    return streams_[i];
  }

  /** Rank‑indexed view of established connections (read‑only). */
  std::unordered_map<int, uint64_t> const& rank2conn() const {
    return rank2conn_;
  }

  /*
   * Create UDS socket path based on GPU index.
   */
  std::string get_uds_socket_path(int gpu_idx) const {
    return "/tmp/uccl_gpu_" + std::to_string(gpu_idx) + ".sock";
  }

  /*
   * Initialize UDS socket for listening.
   */
  void init_uds_socket();

  /*
   * Cleanup UDS socket resources.
   */
  void cleanup_uds_socket();

  /*
   * Initialize the engine
   * Internal helper function for lazy initialization.
   */
  void initialize_engine();

  int local_gpu_idx_;
  uint32_t num_cpus_;
  int numa_node_;

  my_namespace::RDMAEndPoint ep_;
  bool engine_initialized_ = false;

  std::atomic<uint64_t> next_conn_id_ = 0;
  std::atomic<uint64_t> next_mr_id_ = 0;
  std::atomic<uint64_t> next_transfer_id_ = 0;

  // Accessed by both app thread and proxy thread.
  mutable std::shared_mutex conn_mu_;
  std::unordered_map<uint64_t, Conn*> conn_id_to_conn_;
  std::unordered_map<uint64_t, uint64_t> conn_id_to_conn_efa_;
  mutable std::shared_mutex mr_mu_;
  std::unordered_map<uint64_t, MR*> mr_id_to_mr_;

  // Single-threaded.
  std::unordered_map<int, uint64_t> rank2conn_;

  // UDS socket for local connections
  int uds_listen_fd_ = -1;
  std::string uds_socket_path_;

  // Assuming 1TB GPU memory, 128KB KV block size.
  static constexpr size_t kMaxNumChunksPerTransfer = 1024ul * 1024 * 1024 / 128;
  std::atomic<uint32_t> rr_stream_{0};
  std::vector<gpuStream_t> streams_;
  std::vector<std::vector<gpuStream_t>> ipc_streams_;

  static constexpr size_t kTaskRingSize = 1024;

  enum class TaskType {
    SEND_NET,
    RECV_NET,
    SEND_IPC,
    RECV_IPC,
    WRITE_NET,
    READ_NET,
    WRITE_IPC,
    READ_IPC,
    SENDV,
    RECVV,
    WRITEV,
    READV,
  };
  struct TaskBatch {
    size_t num_iovs;  // Number of IO vectors
    std::shared_ptr<std::vector<void const*>> const_data_ptr;  // for SENDV
    std::shared_ptr<std::vector<void*>> data_ptr;  // for RECVV/READV/WRITEV
    std::shared_ptr<std::vector<size_t>> size_ptr;
    std::shared_ptr<std::vector<uint64_t>> mr_id_ptr;
    std::shared_ptr<std::vector<uccl::FifoItem>>
        slot_item_ptr;  // for READV/WRITEV

    TaskBatch() : num_iovs(0) {}

    TaskBatch(TaskBatch&& other) noexcept
        : num_iovs(other.num_iovs),
          const_data_ptr(std::move(other.const_data_ptr)),
          data_ptr(std::move(other.data_ptr)),
          size_ptr(std::move(other.size_ptr)),
          mr_id_ptr(std::move(other.mr_id_ptr)),
          slot_item_ptr(std::move(other.slot_item_ptr)) {}

    TaskBatch& operator=(TaskBatch&& other) noexcept {
      if (this != &other) {
        num_iovs = other.num_iovs;
        const_data_ptr = std::move(other.const_data_ptr);
        data_ptr = std::move(other.data_ptr);
        size_ptr = std::move(other.size_ptr);
        mr_id_ptr = std::move(other.mr_id_ptr);
        slot_item_ptr = std::move(other.slot_item_ptr);
      }
      return *this;
    }

    TaskBatch(TaskBatch const&) = delete;
    TaskBatch& operator=(TaskBatch const&) = delete;

    void const** const_data_v() const {
      if (!const_data_ptr) return nullptr;
      return const_data_ptr->data();
    }
    void** data_v() const {
      if (!data_ptr) return nullptr;
      return data_ptr->data();
    }
    size_t* size_v() const {
      if (!size_ptr) return nullptr;
      return size_ptr->data();
    }
    uint64_t* mr_id_v() const {
      if (!mr_id_ptr) return nullptr;
      return mr_id_ptr->data();
    }
    uccl::FifoItem* slot_item_v() const {
      if (!slot_item_ptr) return nullptr;
      return slot_item_ptr->data();
    }
  };

  struct alignas(64) Task {
    TaskType type;
    void* data;
    size_t size;
    uint64_t conn_id;
    uint64_t mr_id;
    std::atomic<bool> done;
    // For proxy to access the task.done
    Task* self_ptr;
  };

  struct alignas(64) NetRwTask {
    TaskType type;
    void* data;
    size_t size;
    uint64_t conn_id;
    uint64_t mr_id;
    std::atomic<bool> done;
    // For proxy to access the task.done
    NetRwTask* self_ptr;
    uccl::FifoItem slot_item;
  };

  struct alignas(64) IpcRwTask {
    TaskType type;
    void* data;
    size_t size;
    uint64_t conn_id;
    uint64_t mr_id;
    std::atomic<bool> done;
    // For proxy to access the task.done
    IpcRwTask* self_ptr;
    IpcTransferInfo ipc_info;
  };

  static constexpr size_t MAX_RESERVE_SIZE =
      uccl::max_sizeof<uccl::FifoItem, IpcTransferInfo, TaskBatch>();

  struct alignas(64) UnifiedTask {
    TaskType type;
    void* data;
    size_t size;
    uint64_t conn_id;
    uint64_t mr_id;
    std::atomic<bool> done;
    UnifiedTask* self_ptr;

    union SpecificData {
      struct {
        uint8_t reserved[MAX_RESERVE_SIZE];
      } base;

      struct {
        uccl::FifoItem slot_item;
        uint8_t reserved[MAX_RESERVE_SIZE - sizeof(uccl::FifoItem)];
      } net;

      struct {
        IpcTransferInfo ipc_info;
        uint8_t reserved[MAX_RESERVE_SIZE - sizeof(IpcTransferInfo)];
      } ipc;

      struct {
        TaskBatch task_batch;
        uint8_t reserved[MAX_RESERVE_SIZE - sizeof(TaskBatch)];
      } batch;

      SpecificData() : base{} {}
      // // Explicit trivial destructor so the union is not implicitly deleted
      ~SpecificData() {}
    } specific;

    UnifiedTask()
        : type(TaskType::SEND_NET),
          data(nullptr),
          size(0),
          conn_id(0),
          mr_id(0),
          done(false),
          self_ptr(this),
          specific() {}

    ~UnifiedTask() {
      if (is_batch_task()) {
        specific.batch.task_batch.~TaskBatch();
      }
    }

    inline uccl::FifoItem& slot_item() { return specific.net.slot_item; }

    inline uccl::FifoItem const& slot_item() const {
      return specific.net.slot_item;
    }

    inline IpcTransferInfo& ipc_info() { return specific.ipc.ipc_info; }

    inline IpcTransferInfo const& ipc_info() const {
      return specific.ipc.ipc_info;
    }

    inline TaskBatch& task_batch() { return specific.batch.task_batch; }

    inline TaskBatch const& task_batch() const {
      return specific.batch.task_batch;
    }

    inline bool is_batch_task() const {
      return type == TaskType::SENDV || type == TaskType::RECVV ||
             type == TaskType::WRITEV || type == TaskType::READV;
    }
  };

  inline UnifiedTask* create_task(uint64_t conn_id, uint64_t mr_id,
                                  TaskType type, void* data, size_t size) {
    UnifiedTask* task = new UnifiedTask();
    task->type = type;
    task->data = data;
    task->size = size;
    task->conn_id = conn_id;
    task->mr_id = mr_id;
    task->done = false;
    task->self_ptr = task;
    return task;
  }

  inline UnifiedTask* create_batch_task(uint64_t conn_id, TaskType type,
                                        TaskBatch&& batch) {
    UnifiedTask* task = new UnifiedTask();
    task->type = type;
    task->conn_id = conn_id;
    task->done = false;
    task->self_ptr = task;
    // Not used for batch operations
    task->mr_id = 0;
    task->data = nullptr;
    task->size = 0;
    // placement new
    new (&task->specific.batch.task_batch) TaskBatch(std::move(batch));
    return task;
  }

  inline UnifiedTask* create_sendv_task(
      uint64_t conn_id,
      std::shared_ptr<std::vector<void const*>> const_data_ptr,
      std::shared_ptr<std::vector<size_t>> size_ptr,
      std::shared_ptr<std::vector<uint64_t>> mr_id_ptr) {
    if (!const_data_ptr || !size_ptr || !mr_id_ptr ||
        const_data_ptr->size() != size_ptr->size() ||
        size_ptr->size() != mr_id_ptr->size()) {
      return nullptr;
    }
    size_t num_iovs = const_data_ptr->size();

    TaskBatch batch;
    batch.num_iovs = num_iovs;
    batch.const_data_ptr = std::move(const_data_ptr);  // Transfer ownership
    batch.size_ptr = std::move(size_ptr);
    batch.mr_id_ptr = std::move(mr_id_ptr);

    return create_batch_task(conn_id, TaskType::SENDV, std::move(batch));
  }

  inline UnifiedTask* create_recvv_task(uint64_t conn_id,
                                        std::vector<void*>&& data_v,
                                        std::vector<size_t>&& size_v,
                                        std::vector<uint64_t>&& mr_id_v) {
    if (data_v.size() != size_v.size() || size_v.size() != mr_id_v.size()) {
      return nullptr;
    }
    size_t num_iovs = data_v.size();

    auto data_ptr = std::make_shared<std::vector<void*>>(std::move(data_v));
    auto size_ptr = std::make_shared<std::vector<size_t>>(std::move(size_v));
    auto mr_id_ptr =
        std::make_shared<std::vector<uint64_t>>(std::move(mr_id_v));

    TaskBatch batch;
    batch.num_iovs = num_iovs;
    batch.data_ptr = std::move(data_ptr);
    batch.size_ptr = std::move(size_ptr);
    batch.mr_id_ptr = std::move(mr_id_ptr);

    return create_batch_task(conn_id, TaskType::RECVV, std::move(batch));
  }

  inline UnifiedTask* create_writev_task(
      uint64_t conn_id, std::vector<void*>&& data_v,
      std::vector<size_t>&& size_v, std::vector<uint64_t>&& mr_id_v,
      std::vector<uccl::FifoItem>&& slot_item_v) {
    if (data_v.size() != size_v.size() || size_v.size() != mr_id_v.size() ||
        mr_id_v.size() != slot_item_v.size()) {
      return nullptr;
    }
    size_t num_iovs = data_v.size();

    auto data_ptr = std::make_shared<std::vector<void*>>(std::move(data_v));
    auto size_ptr = std::make_shared<std::vector<size_t>>(std::move(size_v));
    auto mr_id_ptr =
        std::make_shared<std::vector<uint64_t>>(std::move(mr_id_v));
    auto slot_item_ptr =
        std::make_shared<std::vector<uccl::FifoItem>>(std::move(slot_item_v));

    TaskBatch batch;
    batch.num_iovs = num_iovs;
    batch.data_ptr = std::move(data_ptr);
    batch.size_ptr = std::move(size_ptr);
    batch.mr_id_ptr = std::move(mr_id_ptr);
    batch.slot_item_ptr = std::move(slot_item_ptr);

    return create_batch_task(conn_id, TaskType::WRITEV, std::move(batch));
  }

  inline UnifiedTask* create_readv_task(
      uint64_t conn_id, std::vector<void*>&& data_v,
      std::vector<size_t>&& size_v, std::vector<uint64_t>&& mr_id_v,
      std::vector<uccl::FifoItem>&& slot_item_v) {
    if (data_v.size() != size_v.size() || size_v.size() != mr_id_v.size() ||
        mr_id_v.size() != slot_item_v.size()) {
      return nullptr;
    }
    size_t num_iovs = data_v.size();

    auto data_ptr = std::make_shared<std::vector<void*>>(std::move(data_v));
    auto size_ptr = std::make_shared<std::vector<size_t>>(std::move(size_v));
    auto mr_id_ptr =
        std::make_shared<std::vector<uint64_t>>(std::move(mr_id_v));
    auto slot_item_ptr =
        std::make_shared<std::vector<uccl::FifoItem>>(std::move(slot_item_v));

    TaskBatch batch;
    batch.num_iovs = num_iovs;
    batch.data_ptr = std::move(data_ptr);
    batch.size_ptr = std::move(size_ptr);
    batch.mr_id_ptr = std::move(mr_id_ptr);
    batch.slot_item_ptr = std::move(slot_item_ptr);

    return create_batch_task(conn_id, TaskType::READV, std::move(batch));
  }

  inline UnifiedTask* create_net_task(uint64_t conn_id, uint64_t mr_id,
                                      TaskType type, void* data, size_t size,
                                      uccl::FifoItem const& slot_item) {
    UnifiedTask* task = create_task(conn_id, mr_id, type, data, size);
    task->slot_item() = slot_item;
    return task;
  }

  inline UnifiedTask* create_ipc_task(uint64_t conn_id, uint64_t mr_id,
                                      TaskType type, void* data, size_t size,
                                      IpcTransferInfo const& ipc_info) {
    UnifiedTask* task = create_task(conn_id, mr_id, type, data, size);
    task->ipc_info() = ipc_info;
    return task;
  }

  // For both net and ipc send/recv tasks.
  jring_t* send_unified_task_ring_;
  jring_t* recv_unified_task_ring_;

  std::atomic<bool> stop_{false};
  std::thread send_proxy_thread_;
  std::thread recv_proxy_thread_;
  void send_proxy_thread_func();
  void recv_proxy_thread_func();
};