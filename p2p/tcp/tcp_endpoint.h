#pragma once

#include "tcp/tcp_worker_pool.h"
#include "util/gpu_rt.h"
#include "util/net.h"
#include <arpa/inet.h>
#include <glog/logging.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <transport.h>  // For common types like uccl::ConnID, FifoItem, etc.

namespace tcp {

static constexpr size_t kChunkSize = 128 * 1024;  // 128KB chunk size
static_assert(kChunkSize <= kStagingBufferSize,
              "kChunkSize must be <= kStagingBufferSize");
static constexpr size_t kMaxInflightChunks = 256;
static constexpr size_t kTCPBufferSize = 128 * 1024 * 1024;  // 128MB TCP buffer
static constexpr uint64_t kBandwidthPerConnection =
    20ULL * 1000 * 1000 * 1000;  // 20Gbps

// TCP doesn't need memory registration, so MRArray is a dummy type
struct MRArray {
  void* dummy = nullptr;
};

// Interface information
struct InterfaceInfo {
  std::string name;
  std::string ip_addr;
  uint64_t bandwidth_bps;
  int num_connections;
};

// Get interface bandwidth from sysfs or ethtool
uint64_t get_interface_bandwidth(std::string const& ifname);

// Parse interface list from environment variable
std::vector<InterfaceInfo> parse_tcp_interfaces();

// TCP Endpoint implementation
class TCPEndpoint {
 public:
  explicit TCPEndpoint(int gpu_index, uint16_t port = 0);
  ~TCPEndpoint();

  int gpuIndex() const { return gpu_index_; }

  // Connect to remote peer
  uccl::ConnID uccl_connect(int dev, int local_gpuidx, int remote_dev,
                            int remote_gpuidx, std::string remote_ip,
                            uint16_t remote_port);

  uint16_t get_p2p_listen_port(int dev) { return listen_port_; }

  int get_p2p_listen_fd(int dev) { return listen_fd_; }

  uccl::ConnID uccl_accept(int dev, int listen_fd, int local_gpuidx,
                           std::string& remote_ip, int* remote_dev,
                           int* remote_gpuidx);

  // Memory registration (no-op for TCP)
  int uccl_regmr(uccl::UcclFlow* flow, void* data, size_t len, int type,
                 struct uccl::Mhandle** mhandle);

  int uccl_regmr(void* data, size_t len, MRArray& mr_array);

  int uccl_regmr(int dev, void* data, size_t len, int type,
                 struct uccl::Mhandle** mhandle);

  void uccl_deregmr(struct uccl::Mhandle* mhandle);

  void uccl_deregmr(MRArray const& mr_array);

  // Send data asynchronously (data is in GPU memory)
  int uccl_send_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh,
                      void const* data, size_t size,
                      struct uccl::ucclRequest* ureq);

  // Receive data asynchronously (data is in GPU memory)
  int uccl_recv_async(uccl::UcclFlow* flow, struct uccl::Mhandle** mhandles,
                      void** data, int* sizes, int n,
                      struct uccl::ucclRequest* ureq);

  // Read from remote memory (dest_addr already known from FifoItem)
  int uccl_read_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh, void* dst,
                      size_t size, uccl::FifoItem const& slot_item,
                      uccl::ucclRequest* ureq);

  // Write to remote memory (dest_addr already known from FifoItem)
  int uccl_write_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh,
                       void* src, size_t size, uccl::FifoItem const& slot_item,
                       uccl::ucclRequest* ureq);

  // Poll for completion
  bool uccl_poll_ureq_once(struct uccl::ucclRequest* ureq);

  // Prepare FIFO metadata for advertisement
  int prepare_fifo_metadata(uccl::UcclFlow* flow,
                            struct uccl::Mhandle** mhandle, void const* data,
                            size_t size, char* out_buf);

  int get_best_dev_idx(int gpu_idx) { return 0; }

  bool initialize_engine_by_dev(int dev, bool enable_p2p_listen) {
    return true;
  }

  void create_unified_p2p_socket() {}

 private:
  // Interface info exchanged during negotiation
  struct InterfaceNegotiationInfo {
    char ip_addr[16];  // IPv4 address string
    int32_t num_connections;
    uint16_t data_port;  // Port for data connections on this interface
    uint16_t reserved;
  };

  struct NegotiationInfo {
    int32_t gpu_index;
    int32_t num_interfaces;
    int32_t total_connections;
    int32_t reserved;
    // Followed by InterfaceNegotiationInfo[num_interfaces]
  };

  void start_listening();

  int create_tcp_connection(std::string const& remote_ip, int remote_port);

  int create_tcp_connection_from_interface(std::string const& remote_ip,
                                           int remote_port,
                                           std::string const& local_ip);

  void setup_tcp_socket_options(int fd);

  std::shared_ptr<TCPConnectionGroup> get_connection_group(uint64_t conn_id);

  int gpu_index_;
  uint16_t listen_port_;
  int listen_fd_ = -1;  // Control connection listen fd (INADDR_ANY)
  std::vector<int> data_listen_fds_;         // Per-interface data listen fds
  std::vector<uint16_t> data_listen_ports_;  // Per-interface data listen ports
  std::atomic<uint64_t> next_conn_id_;
  std::atomic<uint32_t> next_request_id_;
  std::atomic<bool> running_;

  std::vector<InterfaceInfo> interfaces_;
  int total_connections_;

  std::unique_ptr<TCPThreadPool> thread_pool_;

  mutable std::shared_mutex conn_mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<TCPConnectionGroup>>
      connection_groups_;
};

}  // namespace tcp
