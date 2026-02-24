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

#ifdef UCCL_P2P_USE_TCPX
using FifoItem = nccl_tcpx::FifoItem;
#else
using FifoItem = FifoItem;
#endif

extern thread_local bool inside_python;

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

// Custom hash function for std::vector<uint8_t>
struct VectorUint8Hash {
  std::size_t operator()(std::vector<uint8_t> const& vec) const;
};

// Prepare transfer info structure for receiving IPC handle
struct IpcTransferInfo {
  gpuIpcMemHandle_t handle;
  uintptr_t offset;
  size_t size;
  uint32_t operation;  // 0 = send_ipc request, 1 = recv_ipc response
  bool is_host;        // true if this side's buffer is CPU memory
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
    IpcTransferInfo info;
    uint32_t completion;
  };
  ShmMsg();
};

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

  TaskBatch();
  TaskBatch(TaskBatch&& other) noexcept;
  TaskBatch& operator=(TaskBatch&& other) noexcept;

  TaskBatch(TaskBatch const&) = delete;
  TaskBatch& operator=(TaskBatch const&) = delete;

  void const** const_data_v() const;
  void** data_v() const;
  size_t* size_v() const;
  uint64_t* mr_id_v() const;
  FifoItem* slot_item_v() const;
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

static constexpr size_t kEndpointMaxReserveSize =
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
      uint8_t reserved[kEndpointMaxReserveSize];
    } base;

    struct {
      FifoItem slot_item;
      uint8_t reserved[kEndpointMaxReserveSize - sizeof(FifoItem)];
    } net;

    struct {
      IpcTransferInfo ipc_info;
      uint8_t reserved[kEndpointMaxReserveSize - sizeof(IpcTransferInfo)];
    } ipc;

    struct {
      TaskBatch task_batch;
      uint8_t reserved[kEndpointMaxReserveSize - sizeof(TaskBatch)];
    } batch;

    SpecificData();
    ~SpecificData();
  } specific;

  UnifiedTask();
  ~UnifiedTask();

  FifoItem& slot_item();
  FifoItem const& slot_item() const;

  IpcTransferInfo& ipc_info();
  IpcTransferInfo const& ipc_info() const;

  TaskBatch& task_batch();
  TaskBatch const& task_batch() const;

  bool is_batch_task() const;
};

// Tracks an in-flight IPC async copy (used by Endpoint).
struct IpcInflightOp {
  std::vector<gpuEvent_t> events;  // flattened events (all iovs × all streams)
  void* raw_ptr;                   // non-null for scalar ops
  TransferStatus* status;
  int gpu_idx;  // used for scalar ops
  // Vectorized only (populated when raw_ptr == nullptr):
  std::vector<void*> raw_ptrs_v;
  std::vector<int> gpu_idxs_v;
};

// -----------------------------------------------------------------------------
// The main P2P Endpoint class declaration
// -----------------------------------------------------------------------------

class Endpoint {
  static constexpr int kMaxNumGPUs = 8;
  static constexpr size_t kIpcAlignment = 1ul << 20;
  static constexpr size_t kIpcSizePerEngine = 1ul << 20;
  static constexpr int kMaxInflightOps = 8;  // Max 8 concurrent Ops
  static constexpr size_t ShmRingDefaultElemCnt = 16;
  static constexpr size_t kTaskRingSize = 1024;

  static std::once_flag glog_init_once;
  static int parse_log_level_from_env();

 public:
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
  bool writev_ipc(uint64_t conn_id, std::vector<void const*> data_v,
                  std::vector<size_t> size_v,
                  std::vector<IpcTransferInfo> info_v, size_t num_iovs);
  bool readv_ipc(uint64_t conn_id, std::vector<void*> data_v,
                 std::vector<size_t> size_v,
                 std::vector<IpcTransferInfo> info_v, size_t num_iovs);
  bool writev_ipc_async(uint64_t conn_id, std::vector<void const*> data_v,
                        std::vector<size_t> size_v,
                        std::vector<IpcTransferInfo> info_v, size_t num_iovs,
                        uint64_t* transfer_id);
  bool readv_ipc_async(uint64_t conn_id, std::vector<void*> data_v,
                       std::vector<size_t> size_v,
                       std::vector<IpcTransferInfo> info_v, size_t num_iovs,
                       uint64_t* transfer_id);
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

  int get_sock_fd(uint64_t conn_id) const;

  /** Returns conn_id for @rank, or UINT64_MAX if unknown. */
  uint64_t conn_id_of_rank(int rank) const;

  std::shared_ptr<EpollClient> get_oob_client() const;

  std::string get_oob_conn_key(uint64_t conn_id) const;

  MR* get_mr(uint64_t mr_id) const;

  P2PMhandle* get_mhandle(uint64_t mr_id) const;

  Conn* get_conn(uint64_t conn_id) const;

  RDMAEndPoint get_endpoint() const;

 private:
  int local_gpu_idx_;
  uint32_t num_cpus_;
  int numa_node_;
  RDMAEndPoint ep_;
  bool engine_initialized_ = false;
  std::atomic<uint64_t> next_conn_id_ = 0;
  std::atomic<uint64_t> next_mr_id_ = 0;
  std::atomic<uint64_t> next_transfer_id_ = 0;
  /* Accessed by both app thread and proxy thread. */
  mutable std::shared_mutex conn_mu_;
  std::unordered_map<uint64_t, Conn*> conn_id_to_conn_;
  std::unordered_map<uint64_t, uint64_t> conn_id_to_conn_efa_;
  mutable std::shared_mutex mr_mu_;
  std::unordered_map<uint64_t, MR*> mr_id_to_mr_;
  std::unordered_map<std::vector<uint8_t>, uint64_t, VectorUint8Hash>
      remote_endpoint_to_conn_id_;
  /* Single-threaded access only. */
  std::unordered_map<int, uint64_t> rank2conn_;
  /* JRing for local */
  std::array<ShmRingHandle, kMaxNumGPUs> inbox_rings_;
  std::array<bool, kMaxNumGPUs> inbox_creators_;
  std::vector<std::vector<gpuStream_t>> ipc_streams_;
  /* For both net and ipc send/recv tasks. */
  jring_t* send_unified_task_ring_ = nullptr;
  jring_t* recv_unified_task_ring_ = nullptr;
  jring_t* ipc_inflight_ring_ = nullptr;
  std::atomic<bool> stop_{false};
  std::thread send_proxy_thread_;
  std::thread recv_proxy_thread_;
  /* MPSC ring: caller threads push IpcInflightOp*, poller thread drains it. */
  std::thread ipc_poller_thread_;
  /* For background passive accept thread. */
  bool passive_accept_;
  std::atomic<bool> passive_accept_stop_{false};
  std::thread passive_accept_thread_;

  /* Initialize the engine Internal helper function for lazy initialization. */
  void initialize_engine();

  /* Background threads for send/recv/ipc/passive accept. */
  void send_proxy_thread_func();
  void recv_proxy_thread_func();
  void passive_accept_thread_func();
  void ipc_poller_thread_func();

  std::shared_ptr<UnifiedTask> create_task(uint64_t conn_id, uint64_t mr_id,
                                           TaskType type, void* data,
                                           size_t size);
  std::shared_ptr<UnifiedTask> create_batch_task(uint64_t conn_id,
                                                 TaskType type,
                                                 TaskBatch&& batch);
  std::shared_ptr<UnifiedTask> create_sendv_task(
      uint64_t conn_id,
      std::shared_ptr<std::vector<void const*>> const_data_ptr,
      std::shared_ptr<std::vector<size_t>> size_ptr,
      std::shared_ptr<std::vector<uint64_t>> mr_id_ptr);
  std::shared_ptr<UnifiedTask> create_recvv_task(
      uint64_t conn_id, std::vector<void*>&& data_v,
      std::vector<size_t>&& size_v, std::vector<uint64_t>&& mr_id_v);
  std::shared_ptr<UnifiedTask> create_writev_task(
      uint64_t conn_id, std::vector<void*>&& data_v,
      std::vector<size_t>&& size_v, std::vector<uint64_t>&& mr_id_v,
      std::vector<FifoItem>&& slot_item_v);
  std::shared_ptr<UnifiedTask> create_readv_task(
      uint64_t conn_id, std::vector<void*>&& data_v,
      std::vector<size_t>&& size_v, std::vector<uint64_t>&& mr_id_v,
      std::vector<FifoItem>&& slot_item_v);
  std::shared_ptr<UnifiedTask> create_net_task(uint64_t conn_id, uint64_t mr_id,
                                               TaskType type, void* data,
                                               size_t size,
                                               FifoItem const& slot_item);
  std::shared_ptr<UnifiedTask> create_ipc_task(uint64_t conn_id, uint64_t mr_id,
                                               TaskType type, void* data,
                                               size_t size,
                                               IpcTransferInfo const& ipc_info);
};
