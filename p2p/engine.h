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
// #define USE_REDIS
#ifdef USE_REDIS
#include <sw/redis++/redis++.h>
#endif

namespace py = pybind11;
constexpr uint64_t kNvlinkConn = UINT64_MAX;

struct MR {
  uint64_t mr_id_;
  uccl::Mhandle* mhandle_;
};

struct IpcCache {
  gpuIpcMemHandle_t handle;
  bool is_send;
  void* direct_ptr;
  uintptr_t offset;
  size_t size;
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
  const uint64_t kRTTBytes = 1024 * 1024;
  const uint64_t kChunkSize = 1024 * 1024;
  const uint32_t kMaxInflightChunks = 8;

 public:
  gpuStream_t pick_stream() {
    if (streams_.empty()) return nullptr;
    uint32_t i =
        rr_stream_.fetch_add(1, std::memory_order_relaxed) % streams_.size();
    return streams_[i];
  }

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

  bool connect(py::bytes const& metadata, uint64_t& conn_id);

  /*
   * Accept an incoming connection via TCP, then build RDMA QP connections.
   *
   * output:
   *   ip_addr: the IP address of the remote server
   *   remote_gpu_idx: the GPU index of the remote server
   *   conn_id: the ID of the connection
   */
  bool accept(std::string& ip_addr, int& remote_gpu_idx, uint64_t& conn_id);

  /*
   * Register the data with a specific interface. Typically, one data residing
   * on one GPU only needs to register to one NIC. Even if the data is
   * registered to multiple NICs, the GPU wouldn't have enough PCIe bandwidth
   * for multiple NICs.
   *
   * input:
   *   data: the data to register
   *   size: the size of the data
   * output:
   *   mr_id: the ID of the MR
   */
  bool reg(void const* data, size_t size, uint64_t& mr_id);

  bool regv(std::vector<void const*> const& data_v,
            std::vector<size_t> const& size_v, std::vector<uint64_t>& mr_id_v);

  /*
   * Send data to the remote server. Blocking.
   *
   * input:
   *   conn_id: the ID of the connection
   *   mr_id: the ID of the data
   *   data: the data to send
   *   size: the size of the data
   */
  bool send(uint64_t conn_id, uint64_t mr_id, void const* data, size_t size,
            bool inside_python = true);

  /*
   * Receive data from the remote server. Blocking.
   *
   * input:
   *   conn_id: the ID of the connection
   *   mr_id: the ID of the data
   * output:
   *   data: the data to receive
   *   size: the size of the data
   */
  bool recv(uint64_t conn_id, uint64_t mr_id, void* data, size_t size,
            bool inside_python = true);

  bool send_ipc(uint64_t conn_id, uint64_t mr_id, void const* data, size_t size,
                void const* meta, size_t meta_len);

  /* Send data to the remote server asynchronously. */
  bool send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                  size_t size, uint64_t* transfer_id);

  /* Receive data from the remote server asynchronously. */
  bool recv_async(uint64_t conn_id, uint64_t mr_id, void* data, size_t size,
                  uint64_t* transfer_id);

  /* Send a vector of data chunks. Blocking. */
  bool sendv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<void const*> data_v, std::vector<size_t> size_v,
             size_t num_iovs,bool inside_python = true);

  /* Send a vector of data chunks asynchronously. */
  bool sendv_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                   std::vector<void const*> data_v, std::vector<size_t> size_v,
                   size_t num_iovs, uint64_t* transfer_id);

  /* Receive a vector of data chunks. Blocking. */
  bool recvv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<void*> data_v, std::vector<size_t> size_v,
             size_t num_iovs,bool inside_python = true);

  /* Receive a vector of data chunks asynchronously. */
  bool recvv_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                   std::vector<void*> data_v, std::vector<size_t> size_v,
                   size_t num_iovs, uint64_t* transfer_id);

  /* Read data from the remote server. Blocking.
   *
   * input:
   *   conn_id: the ID of the connection
   *   mr_id: the ID of the data
   *   dst: the destination buffer
   *   size: the size of the data
   *   slot_item: the slot item to use for the transfer
   */
  bool read(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
            uccl::FifoItem const& slot_item, bool inside_python = true);

  /* Read data from the remote server asynchronously. */
  bool read_async(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
                  uccl::FifoItem const& slot_item, uint64_t* transfer_id);

  /* Read a vector of data chunks. */
  bool readv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
             std::vector<void*> dst_v, std::vector<size_t> size_v,
             std::vector<uccl::FifoItem> slot_item_v, size_t num_iovs);

  bool advertise(uint64_t conn_id, uint64_t mr_id, void* addr, size_t len,
                 char* out_buf);

  /* Advertise a vector of data chunks. */
  bool advertisev(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                  std::vector<void*> addr_v, std::vector<size_t> len_v,
                  std::vector<char*> out_buf_v, size_t num_iovs);

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

  /* Poll the status of the asynchronous receive. */
  bool poll_async(uint64_t transfer_id, bool* is_done);

  /**
   * Join a logical rendezvous group and connect to every other member.
   *
   * This helper publishes (ip, gpu_idx) to an external discovery service (e.g.,
   * Redis, a Ray named actor, etc.) under the given @group_name.  All callers
   * block until @world_size peers have registered.  Connections are then
   * established in rank‑ascending order (lower rank initiates), guaranteeing a
   * fully‑connected clique without duplicate dials.
   *
   * @param discovery_uri  URI for discovery backend. Examples:
   *                       "redis://127.0.0.1:6379" or "ray://actor:Store".
   * @param group_name     Logical namespace so multiple groups can coexist.
   * @param world_size     Total number of expected ranks in the group.
   * @param my_rank        Caller's rank (0‑based). Must be unique.
   *
   * @returns true on success, false otherwise.
   */
  bool join_group(std::string const& discovery_uri,
                  std::string const& group_name, int world_size, int my_rank,
                  int remote_gpu_idx, uint16_t remote_port);

  /**
   * Convenience constructor: create Endpoint and immediately join a group.
   * You may prefer this factory in Ray where each actor knows its rank and the
   * rendezvous, but not its peers' IP addresses.
   */
  static std::unique_ptr<Endpoint> create_and_join(
      std::string const& discovery_uri, std::string const& group_name,
      int world_size, int my_rank, uint32_t local_gpu_idx, uint32_t num_cpus,
      int remote_gpu_idx);

  int get_sock_fd(uint64_t conn_id) const;

  /** Returns conn_id for @rank, or UINT64_MAX if unknown. */
  uint64_t conn_id_of_rank(int rank) const;

  std::vector<uint8_t> get_endpoint_metadata();

  /*
   * Parse endpoint metadata to extract IP address, port, and GPU index.
   * Returns a tuple of (ip_address, port, gpu_index).
   */
  static std::tuple<std::string, uint16_t, int> parse_metadata(
      std::vector<uint8_t> const& metadata);

  /*
   * Connect to a local process via Unix Domain Socket.
   *
   * input:
   *   remote_gpu_idx: the GPU index of the remote process
   * output:
   *   conn_id: the ID of the connection
   */
  bool connect_local(int remote_gpu_idx, uint64_t& conn_id);

  /*
   * Accept an incoming local connection via Unix Domain Socket.
   *
   * output:
   *   remote_gpu_idx: the GPU index of the remote process
   *   conn_id: the ID of the connection
   */
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

 private:
  /** Rank‑indexed view of established connections (read‑only). */
  std::unordered_map<int, uint64_t> const& rank2conn() const {
    return rank2conn_;
  }

  /*
   * Create UDS socket path based on GPU index.
   */
  std::string get_uds_socket_path(int gpu_idx) const;

  /*
   * Initialize UDS socket for listening.
   */
  void init_uds_socket();

  /*
   * Cleanup UDS socket resources.
   */
  void cleanup_uds_socket();

#ifdef USE_REDIS
  bool publish_redis(std::string const& redis_uri, std::string const& key,
                     PeerInfo const& info);
  bool fetch_all_redis(std::string const& redis_uri,
                       std::string const& key_prefix, int world_size,
                       std::vector<PeerInfo>& out);
#endif

  bool publish_peer(std::string const& discovery_uri,
                    std::string const& group_name, int rank,
                    PeerInfo const& info);
  bool collect_peers(std::string const& discovery_uri,
                     std::string const& group_name, int world_size,
                     std::vector<PeerInfo>& out);

  int local_gpu_idx_;
  int remote_gpu_idx_;
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

  static constexpr size_t kIpcAlignment = 1ul << 20;
  static constexpr size_t kIpcSizePerEngine = 1ul << 20;
  // Prepare transfer info structure for receiving IPC handle
  struct IpcTransferInfo {
    gpuIpcMemHandle_t handle;
    uintptr_t offset;
    size_t size;
    uint32_t operation;  // 0 = send_ipc request, 1 = recv_ipc response
  };

  struct IpcEventInfo {
    gpuIpcEventHandle_t event_handle;
  };

  // IPC Buffer cache
  mutable std::shared_mutex ipc_cache_mu_;
  std::unordered_map<uint64_t, std::unordered_map<void*, struct IpcCache>>
      conn_id_and_ptr_to_ipc_cache_;

  static constexpr size_t kTaskRingSize = 1024;

  enum class TaskType {
    SEND,
    RECV,
    READ,
    WRITE,
    SEND_IPC,
    RECV_IPC,
    SENDV,
    RECVV,
    // TODO: SENDV_IPC, RECVV_IPC
  };

  struct alignas(64) TaskV {
    TaskType type;              // 任务类型，将是 SENDV 或 RECVV

    uint64_t conn_id;           // 连接 ID，与 Task 相同

    // --- 矢量化操作的核心数据 ---
    // 使用 vector 来存储多个数据块的信息
    std::vector<uint64_t> mr_id_v;
    // 为 sendv (const) 和 recvv (non-const) 分别准备
    std::vector<void const*> const_data_v;
    std::vector<void*> data_v;
    std::vector<size_t> size_v;
    size_t num_iovs;            // 数据块的数量

    std::atomic<bool> done;     // 完成状态标志，与 Task 相同
    TaskV* self_ptr;            // 指向自己的指针，与 Task 相同
};

struct TaskVPtrWrapper {
    TaskV* ptr;
    uint64_t padding; // 8 bytes of ptr + 8 bytes of padding = 16 bytes
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

  struct alignas(64) RWTask {
    TaskType type;
    void* data;
    size_t size;
    uint64_t conn_id;
    uint64_t mr_id;
    std::atomic<bool> done;
    // For proxy to access the task.done
    RWTask* self_ptr;
    uccl::FifoItem slot_item;
  };

  jring_t* send_task_ring_;
  jring_t* recv_task_ring_;
  jring_t* read_task_ring_;
  jring_t* write_task_ring_;
  jring_t* sendv_task_ring_;
  jring_t* recvv_task_ring_;

  std::atomic<bool> stop_{false};
  std::thread send_proxy_thread_;
  std::thread recv_proxy_thread_;
  std::thread sendv_proxy_thread_;
  std::thread recvv_proxy_thread_;
  void send_proxy_thread_func();
  void recv_proxy_thread_func();
  void sendv_proxy_thread_func();
  void recvv_proxy_thread_func();
};
