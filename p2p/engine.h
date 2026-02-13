#pragma once

#include "common.h"
#include "rdma/rdma_endpoint.h"
#include "util/gpu_rt.h"
#include "util/jring.h"
#include "util/net.h"
#include "util/shared_pool.h"
#include "util/util.h"
#ifdef UCCL_P2P_USE_NCCL
#include "nccl/nccl_endpoint.h"
#endif
#include <glog/logging.h>
#include <infiniband/verbs.h>
#include <pybind11/pybind11.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace py = pybind11;

extern thread_local bool inside_python;

int const kMaxNumGPUs = 8;

inline int parseLogLevelFromEnv() {
  char const* env = std::getenv("UCCL_P2P_LOG_LEVEL");
  if (!env) {
    return google::WARNING;
  }

  if (!strcasecmp(env, "INFO")) return google::INFO;
  if (!strcasecmp(env, "WARNING")) return google::WARNING;
  if (!strcasecmp(env, "ERROR")) return google::ERROR;
  if (!strcasecmp(env, "FATAL")) return google::FATAL;

  char* end = nullptr;
  long val = std::strtol(env, &end, 10);
  if (end != env && val >= 0 && val <= 3) {
    return static_cast<int>(val);
  }

  return google::WARNING;
}

#ifdef UCCL_P2P_USE_NCCL
using RDMAEndPoint = std::shared_ptr<tcp::TCPEndpoint>;
using ReqType = uccl::ReqType;
using ucclRequest = uccl::ucclRequest;
#else
using RDMAEndPoint = std::shared_ptr<NICEndpoint>;
enum ReqType { ReqTx, ReqRx, ReqRead, ReqWrite };
struct ucclRequest {
  enum ReqType type;
  uint32_t n;
  uint32_t engine_idx;
};
#endif
struct Mhandle {
  struct ibv_mr* mr;
};

struct P2PMhandle {
  MRArray mr_array;
};

struct MR {
  uint64_t mr_id_;
  P2PMhandle* mhandle_;
};

const size_t ShmRingDefaultElemCnt = 16;

struct ShmRingHandle {
  jring_t* ring = nullptr;
  int shm_fd = -1;
  size_t shm_size = 0;
  std::string shm_name;
};

struct Conn {
  uint64_t conn_id_;
  ConnID uccl_conn_id_;
  std::string ip_addr_;
  int remote_gpu_idx_;

  ShmRingHandle remote_inbox_;
  bool shm_attached_ = false;
};

#ifdef UCCL_P2P_USE_TCPX
using FifoItem = nccl_tcpx::FifoItem;
#else
using FifoItem = FifoItem;
#endif

// Custom hash function for std::vector<uint8_t>
struct VectorUint8Hash {
  std::size_t operator()(std::vector<uint8_t> const& vec) const {
    std::size_t hash = vec.size();
    for (uint8_t byte : vec) {
      hash = hash * 31 + static_cast<std::size_t>(byte);
    }
    return hash;
  }
};

class Endpoint {
  static constexpr size_t kIpcAlignment = 1ul << 20;
  static constexpr size_t kIpcSizePerEngine = 1ul << 20;
  static constexpr int kMaxInflightOps = 8;  // Max 8 concurrent Ops

 public:
  // Prepare transfer info structure for receiving IPC handle
  struct IpcTransferInfo {
    gpuIpcMemHandle_t handle;
    uintptr_t offset;
    size_t size;
    uint32_t operation;  // 0 = send_ipc request, 1 = recv_ipc response
  };
  // For ShmChannel
  enum class ShmMsgType : uint32_t {
    CONNECT = 0,
    IPC_HANDLE = 1,
    COMPLETION = 2,
  };
  struct ShmMsg {
    uint32_t src_gpu;
    ShmMsgType type;
    union {
      Endpoint::IpcTransferInfo info;
      uint32_t completion;
    };
    ShmMsg() : src_gpu(0), type(ShmMsgType::COMPLETION), completion(0) {}
  };

  /* Create engine threads running in background for a single interface. It also
   * opens a TCP listening thread waiting for incoming connections. */
  Endpoint(uint32_t const local_gpu_idx, uint32_t const num_cpus);

  /* Create endpoint without intializing the engine. Lazy creation of engine is
   * done during  memory registration. Additionally, open a unified P2P socket
   * for metadata exchanges. If passive_accept is true, the endpoint will not
   * call accept() but delegate it to a background thread.
   */
  Endpoint(uint32_t const num_cpus);
  ~Endpoint();

  /* Connect to a remote server via TCP, then build RDMA QP connections. */
  bool connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
               uint64_t& conn_id);

  std::vector<uint8_t> get_metadata();

  /* Get the unified metadata for all devices. */
  std::vector<uint8_t> get_unified_metadata();

  /* Parse endpoint metadata to extract IP address, port, and GPU index.
   * Returns a tuple of (ip_address, port, gpu_index). */
  static std::tuple<std::string, uint16_t, int> parse_metadata(
      std::vector<uint8_t> const& metadata);

  /* Accept an incoming connection via TCP, then build RDMA QP connections. */
  bool accept(std::string& ip_addr, int& remote_gpu_idx, uint64_t& conn_id);

  /* Register the data with a specific interface. */
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
            FifoItem const& slot_item);

  /* Read data from the remote server asynchronously. */
  bool read_async(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
                  FifoItem const& slot_item, uint64_t* transfer_id);

  /* Read a vector of data chunks. */
  bool readv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<void*> dst_v, std::vector<size_t> size_v,
             std::vector<FifoItem> slot_item_v, size_t num_iovs);

  /* Read a vector of data chunks asynchronously. */
  bool readv_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                   std::vector<void*> dst_v, std::vector<size_t> size_v,
                   std::vector<FifoItem> slot_item_v, size_t num_iovs,
                   uint64_t* transfer_id);

  /* Write data to the remote server. Blocking. */
  bool write(uint64_t conn_id, uint64_t mr_id, void* src, size_t size,
             FifoItem const& slot_item);

  /* Write data to the remote server asynchronously. */
  bool write_async(uint64_t conn_id, uint64_t mr_id, void* src, size_t size,
                   FifoItem const& slot_item, uint64_t* transfer_id);

  /* Write a vector of data chunks. */
  bool writev(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
              std::vector<void*> src_v, std::vector<size_t> size_v,
              std::vector<FifoItem> slot_item_v, size_t num_iovs);

  /* Write a vector of data chunks asynchronously. */
  bool writev_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                    std::vector<void*> src_v, std::vector<size_t> size_v,
                    std::vector<FifoItem> slot_item_v, size_t num_iovs,
                    uint64_t* transfer_id);

  /* Write data to the remote server via CUDA/HIP IPC. Blocking. */
  bool write_ipc(uint64_t conn_id, uint64_t mr_id, void const* data,
                 size_t size, void const* meta, size_t meta_len);

  bool advertise(uint64_t conn_id, uint64_t mr_id, void* addr, size_t len,
                 char* out_buf);

  /* Prepare Fifo without requiring a connection (for pre-computing fifo_item).
   */
  bool prepare_fifo(uint64_t mr_id, void* addr, size_t len, char* out_buf);

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

  /* Add a remote endpoint with metadata - connect only once per remote
   * endpoint. */
  bool add_remote_endpoint(std::vector<uint8_t> const& metadata,
                           uint64_t& conn_id);

  /* Start a background thread for accepting. */
  bool start_passive_accept();

  /***************************************************/
  /* API for Ray */
  /***************************************************/

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

  std::shared_ptr<EpollClient> get_oob_client() const {
    return ep_->get_oob_client();
  }

  std::string get_oob_conn_key(uint64_t conn_id) const {
    auto it = conn_id_to_conn_.find(conn_id);
    if (it == conn_id_to_conn_.end()) {
      return "";
    }
    uint64_t rank_id = it->second->uccl_conn_id_.flow_id;

    return ep_->get_oob_conn_key(rank_id);
  }

  inline MR* get_mr(uint64_t mr_id) const {
    std::shared_lock<std::shared_mutex> lock(mr_mu_);
    auto it = mr_id_to_mr_.find(mr_id);
    if (it == mr_id_to_mr_.end()) {
      return nullptr;
    }
    return it->second;
  }

  inline P2PMhandle* get_mhandle(uint64_t mr_id) const {
    auto mr = get_mr(mr_id);
    if (unlikely(mr == nullptr)) {
      return nullptr;
    }
    return mr->mhandle_;
  }

  inline Conn* get_conn(uint64_t conn_id) const {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_id_to_conn_.find(conn_id);
    if (it == conn_id_to_conn_.end()) {
      return nullptr;
    }
    return it->second;
  }

  inline RDMAEndPoint get_endpoint() const {
    return ep_;
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

  /* Initialize the engine Internal helper function for lazy initialization. */
  void initialize_engine();

  int local_gpu_idx_;
  uint32_t num_cpus_;
  int numa_node_;

  RDMAEndPoint ep_;
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

  std::unordered_map<std::vector<uint8_t>, uint64_t, VectorUint8Hash>
      remote_endpoint_to_conn_id_;

  // Single-threaded.
  std::unordered_map<int, uint64_t> rank2conn_;

  // JRing for local
  std::array<ShmRingHandle, kMaxNumGPUs> inbox_rings_;
  std::array<bool, kMaxNumGPUs> inbox_creators_;

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
    std::shared_ptr<std::vector<FifoItem>> slot_item_ptr;  // for READV/WRITEV

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
    FifoItem* slot_item_v() const {
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
  };

  struct alignas(64) NetRwTask {
    TaskType type;
    void* data;
    size_t size;
    uint64_t conn_id;
    uint64_t mr_id;
    FifoItem slot_item;
  };

  struct alignas(64) IpcRwTask {
    TaskType type;
    void* data;
    size_t size;
    uint64_t conn_id;
    uint64_t mr_id;
    IpcTransferInfo ipc_info;
  };

  static constexpr size_t MAX_RESERVE_SIZE =
      uccl::max_sizeof<FifoItem, IpcTransferInfo, TaskBatch>();

  struct UnifiedTask;

  struct TransferStatus {
    std::atomic<bool> done{false};
    std::shared_ptr<UnifiedTask> task_ptr;
  };

  struct alignas(64) UnifiedTask {
    TaskType type;
    void* data;
    size_t size;
    uint64_t conn_id;
    uint64_t mr_id;
    TransferStatus* status_ptr{nullptr};

    union SpecificData {
      struct {
        uint8_t reserved[MAX_RESERVE_SIZE];
      } base;

      struct {
        FifoItem slot_item;
        uint8_t reserved[MAX_RESERVE_SIZE - sizeof(FifoItem)];
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
      // Explicit trivial destructor so the union is not implicitly deleted
      ~SpecificData() {}
    } specific;

    UnifiedTask()
        : type(TaskType::SEND_NET),
          data(nullptr),
          size(0),
          conn_id(0),
          mr_id(0),
          status_ptr(nullptr),
          specific() {}

    ~UnifiedTask() {
      if (is_batch_task()) {
        specific.batch.task_batch.~TaskBatch();
      }
    }

    inline FifoItem& slot_item() { return specific.net.slot_item; }

    inline FifoItem const& slot_item() const { return specific.net.slot_item; }

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

  inline std::shared_ptr<UnifiedTask> create_task(uint64_t conn_id,
                                                  uint64_t mr_id, TaskType type,
                                                  void* data, size_t size) {
    auto task = std::make_shared<UnifiedTask>();
    task->type = type;
    task->data = data;
    task->size = size;
    task->conn_id = conn_id;
    task->mr_id = mr_id;
    return task;
  }

  inline std::shared_ptr<UnifiedTask> create_batch_task(uint64_t conn_id,
                                                        TaskType type,
                                                        TaskBatch&& batch) {
    auto task = std::make_shared<UnifiedTask>();
    task->type = type;
    task->conn_id = conn_id;
    // Not used for batch operations
    task->mr_id = 0;
    task->data = nullptr;
    task->size = 0;
    // placement new
    new (&task->specific.batch.task_batch) TaskBatch(std::move(batch));
    return task;
  }

  inline std::shared_ptr<UnifiedTask> create_sendv_task(
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

  inline std::shared_ptr<UnifiedTask> create_recvv_task(
      uint64_t conn_id, std::vector<void*>&& data_v,
      std::vector<size_t>&& size_v, std::vector<uint64_t>&& mr_id_v) {
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

  inline std::shared_ptr<UnifiedTask> create_writev_task(
      uint64_t conn_id, std::vector<void*>&& data_v,
      std::vector<size_t>&& size_v, std::vector<uint64_t>&& mr_id_v,
      std::vector<FifoItem>&& slot_item_v) {
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
        std::make_shared<std::vector<FifoItem>>(std::move(slot_item_v));

    TaskBatch batch;
    batch.num_iovs = num_iovs;
    batch.data_ptr = std::move(data_ptr);
    batch.size_ptr = std::move(size_ptr);
    batch.mr_id_ptr = std::move(mr_id_ptr);
    batch.slot_item_ptr = std::move(slot_item_ptr);

    return create_batch_task(conn_id, TaskType::WRITEV, std::move(batch));
  }

  inline std::shared_ptr<UnifiedTask> create_readv_task(
      uint64_t conn_id, std::vector<void*>&& data_v,
      std::vector<size_t>&& size_v, std::vector<uint64_t>&& mr_id_v,
      std::vector<FifoItem>&& slot_item_v) {
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
        std::make_shared<std::vector<FifoItem>>(std::move(slot_item_v));

    TaskBatch batch;
    batch.num_iovs = num_iovs;
    batch.data_ptr = std::move(data_ptr);
    batch.size_ptr = std::move(size_ptr);
    batch.mr_id_ptr = std::move(mr_id_ptr);
    batch.slot_item_ptr = std::move(slot_item_ptr);

    return create_batch_task(conn_id, TaskType::READV, std::move(batch));
  }

  inline std::shared_ptr<UnifiedTask> create_net_task(
      uint64_t conn_id, uint64_t mr_id, TaskType type, void* data, size_t size,
      FifoItem const& slot_item) {
    auto task = create_task(conn_id, mr_id, type, data, size);
    task->slot_item() = slot_item;
    return task;
  }

  inline std::shared_ptr<UnifiedTask> create_ipc_task(
      uint64_t conn_id, uint64_t mr_id, TaskType type, void* data, size_t size,
      IpcTransferInfo const& ipc_info) {
    auto task = create_task(conn_id, mr_id, type, data, size);
    task->ipc_info() = ipc_info;
    return task;
  }

  // For both net and ipc send/recv tasks.
  jring_t* send_unified_task_ring_ = nullptr;
  jring_t* recv_unified_task_ring_ = nullptr;

  std::atomic<bool> stop_{false};
  std::thread send_proxy_thread_;
  std::thread recv_proxy_thread_;
  void send_proxy_thread_func();
  void recv_proxy_thread_func();

  std::atomic<bool> passive_accept_stop_{false};
  bool passive_accept_;
  std::thread passive_accept_thread_;
  void passive_accept_thread_func();
};
