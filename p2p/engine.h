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

namespace py = pybind11;

struct MR {
  uint64_t mr_id_;
  uccl::Mhandle* mhandle_;
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

static inline std::string get_oob_ip() {
  char uccl_ifname[MAX_IF_NAME_SIZE + 1];
  uccl::socketAddress uccl_ifaddr;
  int num_ifs =
      uccl::find_interfaces(uccl_ifname, &uccl_ifaddr, MAX_IF_NAME_SIZE, 1);
  CHECK(num_ifs == 1) << "No IP interface found";
  return uccl::get_dev_ip(uccl_ifname);
}

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
  Endpoint(uint32_t const local_gpu_idx, uint32_t const num_cpus);

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
  bool send(uint64_t conn_id, uint64_t mr_id, void const* data, size_t size,
            bool inside_python = true);

  /*Receive data from the remote server. Blocking.*/
  bool recv(uint64_t conn_id, uint64_t mr_id, void* data, size_t size,
            bool inside_python = true);

  /* Send data to the remote server asynchronously. */
  bool send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                  size_t size, uint64_t* transfer_id);

  /* Receive data from the remote server asynchronously. */
  bool recv_async(uint64_t conn_id, uint64_t mr_id, void* data, size_t size,
                  uint64_t* transfer_id);

  /* Send a vector of data chunks. Blocking. */
  bool sendv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<void const*> data_v, std::vector<size_t> size_v,
             size_t num_iovs, bool inside_python = true);

  /* Send a vector of data chunks asynchronously. */
  bool sendv_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                   std::vector<void const*> data_v, std::vector<size_t> size_v,
                   size_t num_iovs, uint64_t* transfer_id);

  /* Receive a vector of data chunks. Blocking. */
  bool recvv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<void*> data_v, std::vector<size_t> size_v,
             size_t num_iovs, bool inside_python = true);

  /* Receive a vector of data chunks asynchronously. */
  bool recvv_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                   std::vector<void*> data_v, std::vector<size_t> size_v,
                   size_t num_iovs, uint64_t* transfer_id);

  /* Read data from the remote server. Blocking. */
  bool read(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
            uccl::FifoItem const& slot_item, bool inside_python = true);

  /* Read data from the remote server asynchronously. */
  bool read_async(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
                  uccl::FifoItem const& slot_item, uint64_t* transfer_id);

  /* Read a vector of data chunks. */
  bool readv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<void*> dst_v, std::vector<size_t> size_v,
             std::vector<uccl::FifoItem> slot_item_v, size_t num_iovs);

  /* Write data to the remote server. Blocking. */
  bool write(uint64_t conn_id, uint64_t mr_id, void* src, size_t size,
             uccl::FifoItem const& slot_item, bool inside_python = true);

  /* Write data to the remote server asynchronously. */
  bool write_async(uint64_t conn_id, uint64_t mr_id, void* src, size_t size,
                   uccl::FifoItem const& slot_item, uint64_t* transfer_id);

  /* Write a vector of data chunks. */
  bool writev(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
              std::vector<void*> src_v, std::vector<size_t> size_v,
              std::vector<uccl::FifoItem> slot_item_v, size_t num_iovs);

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
  bool send_ipc(uint64_t conn_id, void* data, size_t size,
                bool inside_python = true);

  bool recv_ipc(uint64_t conn_id, void* data, size_t size,
                bool inside_python = true);

  bool send_ipc_async(uint64_t conn_id, void const* data, size_t size,
                      uint64_t* transfer_id);

  bool recv_ipc_async(uint64_t conn_id, void* data, size_t size,
                      uint64_t* transfer_id);

  /* One-sided write and read via IPC. */
  bool write_ipc(uint64_t conn_id, void const* data, size_t size,
                 IpcTransferInfo const& info, bool inside_python = true);
  bool read_ipc(uint64_t conn_id, void* data, size_t size,
                IpcTransferInfo const& info, bool inside_python = true);
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

  int local_gpu_idx_;
  uint32_t num_cpus_;
  int numa_node_;

  uccl::RDMAEndpoint* ep_;

  std::atomic<uint64_t> next_conn_id_ = 0;
  std::atomic<uint64_t> next_mr_id_ = 0;
  std::atomic<uint64_t> next_transfer_id_ = 0;

  // Accessed by both app thread and proxy thread.
  mutable std::shared_mutex conn_mu_;
  std::unordered_map<uint64_t, Conn*> conn_id_to_conn_;
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
  };
  struct TaskBatch {
    TaskType type;         // SENDV or RECVV
    size_t num_iovs;       // Number of IO vectors
    void* iov_data_block;  // Memory block containing data arrays

    // Get pointer array for const data (SENDV operations)
    void const** const_data_v() const {
      if (type != TaskType::SENDV) return nullptr;
      return static_cast<void const**>(iov_data_block);
    }

    // Get pointer array for mutable data (RECVV operations)
    void** data_v() const {
      if (type != TaskType::RECVV) return nullptr;
      return static_cast<void**>(iov_data_block);
    }

    // Get size array
    size_t* size_v() const {
      uintptr_t base = reinterpret_cast<uintptr_t>(iov_data_block);
      return reinterpret_cast<size_t*>(base + sizeof(void*) * num_iovs);
    }

    // Get memory region ID array
    uint64_t* mr_id_v() const {
      uintptr_t base = reinterpret_cast<uintptr_t>(iov_data_block);
      return reinterpret_cast<uint64_t*>(base + sizeof(void*) * num_iovs +
                                         sizeof(size_t) * num_iovs);
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
    } specific;

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

    // Check if this is a batch task
    inline bool is_batch_task() const {
      return type == TaskType::SENDV || type == TaskType::RECVV;
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
                                        TaskBatch const& batch) {
    UnifiedTask* task = new UnifiedTask();
    task->type = type;
    task->conn_id = conn_id;
    task->done = false;
    task->task_batch() = batch;
    task->self_ptr = task;
    // Not used for batch operations
    task->mr_id = 0;
    task->data = nullptr;
    task->size = 0;
    return task;
  }

  inline UnifiedTask* create_sendv_task(
      uint64_t conn_id, std::vector<void const*> const& const_data_v,
      std::vector<size_t> const& size_v, std::vector<uint64_t> const& mr_id_v) {
    size_t const num_iovs = const_data_v.size();
    // Calculate memory layout
    size_t const ptr_array_size = sizeof(void const*) * num_iovs;
    size_t const size_array_size = sizeof(size_t) * num_iovs;
    size_t const mr_id_array_size = sizeof(uint64_t) * num_iovs;
    size_t const total_data_size =
        ptr_array_size + size_array_size + mr_id_array_size;

    // Allocate and initialize data block
    char* block = new char[total_data_size];
    char* size_array_ptr = block + ptr_array_size;
    char* mr_id_array_ptr = block + ptr_array_size + size_array_size;

    std::memcpy(block, const_data_v.data(), ptr_array_size);
    std::memcpy(size_array_ptr, size_v.data(), size_array_size);
    std::memcpy(mr_id_array_ptr, mr_id_v.data(), mr_id_array_size);

    // Create TaskBatch
    TaskBatch batch;
    batch.type = TaskType::SENDV;
    batch.num_iovs = num_iovs;
    batch.iov_data_block = block;

    return create_batch_task(conn_id, TaskType::SENDV, batch);
  }

  // Create batch task for RECVV operations
  inline UnifiedTask* create_recvv_task(uint64_t conn_id,
                                        std::vector<void*> const& data_v,
                                        std::vector<size_t> const& size_v,
                                        std::vector<uint64_t> const& mr_id_v) {
    size_t const num_iovs = data_v.size();
    // Calculate memory layout
    size_t const ptr_array_size = sizeof(void*) * num_iovs;
    size_t const size_array_size = sizeof(size_t) * num_iovs;
    size_t const mr_id_array_size = sizeof(uint64_t) * num_iovs;
    size_t const total_data_size =
        ptr_array_size + size_array_size + mr_id_array_size;

    // Allocate and initialize data block
    char* block = new char[total_data_size];
    char* size_array_ptr = block + ptr_array_size;
    char* mr_id_array_ptr = block + ptr_array_size + size_array_size;

    std::memcpy(block, data_v.data(), ptr_array_size);
    std::memcpy(size_array_ptr, size_v.data(), size_array_size);
    std::memcpy(mr_id_array_ptr, mr_id_v.data(), mr_id_array_size);

    // Create TaskBatch
    TaskBatch batch;
    batch.type = TaskType::RECVV;
    batch.num_iovs = num_iovs;
    batch.iov_data_block = block;

    return create_batch_task(conn_id, TaskType::RECVV, batch);
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

  // Destroy UnifiedTask and cleanup resources
  inline void destroy_unified_task(UnifiedTask* task) {
    if (task == nullptr) {
      return;
    }

    // Cleanup batch task specific resources
    if (task->is_batch_task() && task->task_batch().iov_data_block != nullptr) {
      delete[] static_cast<char*>(task->task_batch().iov_data_block);
    }

    delete task;
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
