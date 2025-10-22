#pragma once

#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

extern thread_local bool inside_python;

namespace tcpx {

struct Conn {
  uint64_t conn_id_;
  std::string ip_addr_;
  int remote_gpu_idx_;
  int sock_fd;
};

struct FifoItem {
  uint64_t addr;
  uint32_t size;
  char padding[52];
};
static_assert(sizeof(struct FifoItem) == 64, "FifoItem size is not 64 bytes");
// TODO: Nix plugin need #define FIFO_ITEM_SIZE 64 -> #define FIFO_ITEM_SIZE
// sizeof(FifoItem)

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

  bool advertise(uint64_t conn_id, uint64_t mr_id, void* addr, size_t len,
                 char* out_buf);

  int get_sock_fd(uint64_t conn_id) const {
    auto it = conn_id_to_conn_.find(conn_id);
    if (it == conn_id_to_conn_.end()) return -1;
    return it->second->sock_fd;
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
  /* Receive data from the remote server asynchronously. */
  bool recv_async(uint64_t conn_id, uint64_t mr_id, void* data, size_t size,
                  uint64_t* transfer_id);

  /* Poll the status of the asynchronous receive. */
  bool poll_async(uint64_t transfer_id, bool* is_done);

 private:
  mutable std::shared_mutex conn_mu_;
  std::unordered_map<uint64_t, Conn*> conn_id_to_conn_;
};

}  // namespace tcpx