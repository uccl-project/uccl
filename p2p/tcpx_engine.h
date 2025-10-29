#pragma once

#include "tcpx/device/unpack_launch.h"
#include "tcpx/include/bootstrap.h"
#include "tcpx/include/tcpx_interface.h"
#include "tcpx/include/unpack_descriptor.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

extern thread_local bool inside_python;

namespace tcpx {

struct Conn {
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

  // TCPX plugin handles
  void* send_comm = nullptr;
  void* recv_comm = nullptr;
  void* send_dev_handle = nullptr;
  void* recv_dev_handle = nullptr;

  // Cached memory registrations per MR id
  std::unordered_map<uint64_t, void*> send_mhandles;
  std::unordered_map<uint64_t, void*> recv_mhandles;

  alignas(16) std::array<uint8_t, 512> recv_dev_handle_storage;
  alignas(16) std::array<uint8_t, 512> send_dev_handle_storage;
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
};

struct PendingTransfer {
  enum class Kind { kSend, kRecv, kRead };

  Kind kind = Kind::kRecv;
  uint64_t transfer_id = 0;
  uint64_t conn_id = 0;
  uint64_t mr_id = 0;
  size_t size = 0;
  int tag = 0;

  // TCPX request handle returned by isend/irecv/test
  void* request = nullptr;

  // Destination buffer for recv/read
  void* dst_ptr = nullptr;

  // Bounce buffer unpack
  bool needs_unpack = false;
  rx::UnpackDescriptorBlock desc_block{};
  cudaEvent_t completion_event = nullptr;
  bool event_recorded = false;
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
  std::unordered_map<uint64_t, std::unique_ptr<Conn>> conn_map_;

  mutable std::mutex mr_mu_;
  std::unordered_map<uint64_t, MrEntry> mr_map_;

  mutable std::mutex transfer_mu_;
  std::unordered_map<uint64_t, PendingTransfer> transfer_map_;

  cudaStream_t unpack_stream_ = nullptr;
  std::unique_ptr<device::UnpackLauncher> unpack_launcher_;

  static void free_conn_(std::unique_ptr<Conn>& conn);
  bool populate_conn_handles_(Conn& conn, uint64_t mr_id, MrEntry const& mr,
                              bool is_recv, void** mhandle_out);
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
};

}  // namespace tcpx
