#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <cuda_runtime_api.h>
#include <nccl.h>

namespace nccl_tcpx {

// 64-byte FIFO descriptor matching tcpx::FifoItem layout so uccl_engine can
// swap implementations.
struct FifoItem {
  uint64_t mr_id;
  uint32_t size;
  uint32_t tag;
  uint64_t offset;
  uint64_t token;
  char padding[32];
};
static_assert(sizeof(FifoItem) == 64, "FifoItem must be 64 bytes");

class Endpoint {
 public:
  explicit Endpoint(int num_cpus);
  ~Endpoint();

  bool connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
               uint64_t& conn_id);
  bool accept(std::string& ip_addr, int& remote_gpu_idx, uint64_t& conn_id);
  int get_sock_fd(uint64_t conn_id) const;
  std::vector<uint8_t> get_unified_metadata();
  static std::tuple<std::string, uint16_t, int> parse_metadata(
      std::vector<uint8_t> const& metadata);

  bool reg(void const* data, size_t size, uint64_t& mr_id);
  bool dereg(uint64_t mr_id);

  bool advertise(uint64_t conn_id, uint64_t mr_id, void const* addr, size_t len,
                 void* out_buf);

  bool deal_out_buf(uint64_t conn_id, char* out_buf);
  bool queue_read_response(uint64_t conn_id, FifoItem const& fifo_item);
  bool read_async(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
                  FifoItem const& slot_item, uint64_t* transfer_id);
  bool send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                  size_t size, uint64_t* transfer_id);
  bool recv_async(uint64_t conn_id, uint64_t mr_id, void* data, size_t size,
                  uint64_t* transfer_id);
  bool recv(uint64_t conn_id, uint64_t mr_id, void* data, size_t size);
  bool poll_async(uint64_t transfer_id, bool* is_done);

 private:
  struct MrEntry {
    void* base = nullptr;
    size_t size = 0;
  };

  struct Conn {
    int sock_fd = -1;
    int rank = -1;
    int remote_rank = -1;
    int local_gpu_idx = 0;
    int remote_gpu_idx = 0;
    ncclComm_t comm = nullptr;
    cudaStream_t stream = nullptr;
  };

  struct Transfer {
    cudaEvent_t event = nullptr;
  };

  bool setup_listener_();
  bool send_all_(int fd, void const* buf, size_t len);
  bool recv_all_(int fd, void* buf, size_t len);
  bool init_comm_(Conn& conn, ncclUniqueId const& uid, int rank);
  bool send_internal_(Conn& conn, void const* data, size_t size,
                      uint64_t& transfer_id);
  bool recv_internal_(Conn& conn, void* data, size_t size,
                      uint64_t& transfer_id);

  int local_gpu_idx_ = 0;
  int ctrl_listen_fd_ = -1;
  int ctrl_port_ = 0;
  bool initialized_{false};

  std::atomic<uint64_t> next_conn_id_{1};
  std::atomic<uint64_t> next_mr_id_{1};
  std::atomic<uint64_t> next_transfer_id_{1};

  mutable std::mutex conn_mu_;
  std::unordered_map<uint64_t, Conn> conn_map_;

  mutable std::mutex mr_mu_;
  std::unordered_map<uint64_t, MrEntry> mr_map_;

  mutable std::mutex transfer_mu_;
  std::unordered_map<uint64_t, Transfer> transfer_map_;
};

}  // namespace nccl_tcpx
