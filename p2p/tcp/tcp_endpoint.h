#pragma once

#include "efa/define.h"  // For common types like uccl::ConnID, FifoItem, etc.
#include "tcp/tcp_worker_pool.h"
#include "util/gpu_rt.h"
#include "util/net.h"
#include "util/util.h"
#include <arpa/inet.h>
#include <glog/logging.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <netinet/tcp.h>
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

namespace tcp {

// Constants
static constexpr size_t kChunkSize = 1024 * 1024 * 1024;  // 1GB chunk size
static constexpr size_t kMaxInflightChunks = 256;
static constexpr size_t kTCPBufferSize = 4 * 1024 * 1024;  // 4MB TCP buffer
static constexpr uint64_t kBandwidthPerConnection =
    10ULL * 1000 * 1000 * 1000;  // 10Gbps

// Interface information
struct InterfaceInfo {
  std::string name;
  std::string ip_addr;
  uint64_t bandwidth_bps;
  int num_connections;
};

// Get interface bandwidth from sysfs or ethtool
inline uint64_t get_interface_bandwidth(std::string const& ifname) {
  std::string speed_path = "/sys/class/net/" + ifname + "/speed";
  std::ifstream speed_file(speed_path);
  if (speed_file.is_open()) {
    int speed_mbps = 0;
    speed_file >> speed_mbps;
    if (speed_mbps > 0) {
      return static_cast<uint64_t>(speed_mbps) * 1000000ULL;
    }
  }

  int sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock < 0) return 0;

  struct ifreq ifr;
  struct ethtool_cmd ecmd;
  std::memset(&ifr, 0, sizeof(ifr));
  std::strncpy(ifr.ifr_name, ifname.c_str(), IFNAMSIZ - 1);
  ecmd.cmd = ETHTOOL_GSET;
  ifr.ifr_data = reinterpret_cast<char*>(&ecmd);

  uint64_t bandwidth = 0;
  if (ioctl(sock, SIOCETHTOOL, &ifr) >= 0) {
    uint32_t speed = ethtool_cmd_speed(&ecmd);
    if (speed != UINT32_MAX && speed > 0) {
      bandwidth = static_cast<uint64_t>(speed) * 1000000ULL;
    }
  }
  close(sock);
  return bandwidth;
}

// Parse interface list from environment variable
inline std::vector<InterfaceInfo> parse_tcp_interfaces() {
  std::vector<InterfaceInfo> interfaces;

  char const* env = std::getenv("UCCL_TCP_IFNAME");
  if (!env || strlen(env) == 0) {
    char ifNames[uccl::MAX_IFS * uccl::MAX_IF_NAME_SIZE];
    uccl::socketAddress ifAddrs[uccl::MAX_IFS];
    int nIfs = uccl::find_interfaces(ifNames, ifAddrs, uccl::MAX_IF_NAME_SIZE,
                                     uccl::MAX_IFS);

    for (int i = 0; i < nIfs; i++) {
      std::string name = &ifNames[i * uccl::MAX_IF_NAME_SIZE];
      if (name == "lo") continue;

      std::string ip = uccl::get_dev_ip(name.c_str());
      if (ip.empty()) continue;

      uint64_t bw = get_interface_bandwidth(name);
      if (bw == 0) bw = 10ULL * 1000 * 1000 * 1000;

      int num_conns =
          std::max(1, static_cast<int>(bw / kBandwidthPerConnection));
      interfaces.push_back({name, ip, bw, num_conns});
      break;
    }
  } else {
    std::string env_str(env);
    std::stringstream ss(env_str);
    std::string ifname;

    while (std::getline(ss, ifname, ',')) {
      ifname.erase(0, ifname.find_first_not_of(" \t"));
      ifname.erase(ifname.find_last_not_of(" \t") + 1);

      if (ifname.empty()) continue;

      std::string ip = uccl::get_dev_ip(ifname.c_str());
      if (ip.empty()) {
        LOG(WARNING) << "TCP: Interface " << ifname
                     << " has no IP address, skipping";
        continue;
      }

      uint64_t bw = get_interface_bandwidth(ifname);
      if (bw == 0) bw = 10ULL * 1000 * 1000 * 1000;

      int num_conns =
          std::max(1, static_cast<int>(bw / kBandwidthPerConnection));
      interfaces.push_back({ifname, ip, bw, num_conns});
      LOG(INFO) << "TCP: Using interface " << ifname << " (" << ip << "), "
                << "bandwidth=" << (bw / 1e9) << " Gbps, "
                << "connections=" << num_conns;
    }
  }

  if (interfaces.empty()) {
    LOG(WARNING) << "TCP: No valid interfaces found, using localhost";
    interfaces.push_back({"lo", "127.0.0.1", 10ULL * 1000 * 1000 * 1000, 1});
  }

  return interfaces;
}

// TCP Endpoint implementation
class TCPEndpoint {
 public:
  explicit TCPEndpoint(int gpu_index, uint16_t port = 0)
      : gpu_index_(gpu_index),
        listen_port_(port),
        next_conn_id_(0),
        next_request_id_(0),
        running_(true) {
    interfaces_ = parse_tcp_interfaces();

    total_connections_ = 0;
    for (auto const& iface : interfaces_) {
      total_connections_ += iface.num_connections;
    }

    LOG(INFO) << "TCPEndpoint initialized for GPU " << gpu_index_ << " with "
              << interfaces_.size() << " interfaces, " << total_connections_
              << " total connections per peer";

    thread_pool_ = std::make_unique<TCPThreadPool>();
    thread_pool_->start();

    start_listening();
  }

  ~TCPEndpoint() {
    running_ = false;

    if (thread_pool_) {
      thread_pool_->stop();
    }

    if (listen_fd_ >= 0) {
      close(listen_fd_);
      listen_fd_ = -1;
    }

    std::unique_lock<std::shared_mutex> lock(conn_mutex_);
    connection_groups_.clear();
  }

  int gpuIndex() const { return gpu_index_; }

  // Connect to remote peer
  uccl::ConnID uccl_connect(int dev, int local_gpuidx, int remote_dev,
                            int remote_gpuidx, std::string remote_ip,
                            uint16_t remote_port) {
    uint64_t conn_id = next_conn_id_.fetch_add(1, std::memory_order_relaxed);

    LOG(INFO) << "TCPEndpoint::uccl_connect to " << remote_ip << ":"
              << remote_port;

    auto group = std::make_shared<TCPConnectionGroup>();

    NegotiationInfo local_info;
    local_info.gpu_index = local_gpuidx;
    local_info.num_interfaces = interfaces_.size();
    local_info.total_connections = total_connections_;

    // Create control connection first
    int ctrl_fd = create_tcp_connection(remote_ip, remote_port);
    if (ctrl_fd < 0) {
      LOG(ERROR) << "Failed to create control connection";
      uccl::ConnID invalid_id;
      invalid_id.flow_id = UINT64_MAX;
      return invalid_id;
    }

    group->ctrl_fd = ctrl_fd;
    setup_tcp_socket_options(ctrl_fd);

    // Exchange negotiation info
    uccl::send_message(ctrl_fd, &local_info, sizeof(local_info));
    NegotiationInfo remote_info;
    uccl::receive_message(ctrl_fd, &remote_info, sizeof(remote_info));

    int actual_connections = std::min(static_cast<int>(total_connections_),
                                      remote_info.total_connections);
    actual_connections = std::max(1, actual_connections);

    LOG(INFO) << "Negotiated " << actual_connections
              << " data connections with peer";

    // Create data connections
    for (int i = 0; i < actual_connections; i++) {
      int iface_idx = i % interfaces_.size();
      int fd = create_tcp_connection_from_interface(
          remote_ip, remote_port, interfaces_[iface_idx].ip_addr);
      if (fd >= 0) {
        auto conn = std::make_unique<TCPConnection>();
        conn->fd = fd;
        conn->local_ip = interfaces_[iface_idx].ip_addr;
        conn->remote_ip = remote_ip;
        conn->remote_port = remote_port;
        setup_tcp_socket_options(fd);

        // Register data connection with receiver worker
        thread_pool_->assign_data_connection(fd);

        group->add_data_connection(std::move(conn));
      }
    }

    LOG(INFO) << "Created " << group->data_connection_count()
              << " data connections to peer";

    {
      std::unique_lock<std::shared_mutex> lock(conn_mutex_);
      connection_groups_[conn_id] = group;
    }

    uccl::ConnID result;
    result.flow_id = conn_id;
    result.context = reinterpret_cast<void*>(conn_id);
    return result;
  }

  uint16_t get_p2p_listen_port(int dev) { return listen_port_; }

  int get_p2p_listen_fd(int dev) { return listen_fd_; }

  uccl::ConnID uccl_accept(int dev, int listen_fd, int local_gpuidx,
                           std::string& remote_ip, int* remote_dev,
                           int* remote_gpuidx) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);

    // Accept control connection
    int ctrl_fd =
        accept(listen_fd, (struct sockaddr*)&client_addr, &client_len);
    if (ctrl_fd < 0) {
      LOG(ERROR) << "accept failed: " << strerror(errno);
      uccl::ConnID invalid_id;
      invalid_id.flow_id = UINT64_MAX;
      return invalid_id;
    }

    char ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, ip_str, INET_ADDRSTRLEN);
    remote_ip = ip_str;

    LOG(INFO) << "TCPEndpoint::uccl_accept from " << remote_ip;

    NegotiationInfo local_info;
    local_info.gpu_index = local_gpuidx;
    local_info.num_interfaces = interfaces_.size();
    local_info.total_connections = total_connections_;

    NegotiationInfo remote_info;
    uccl::receive_message(ctrl_fd, &remote_info, sizeof(remote_info));
    uccl::send_message(ctrl_fd, &local_info, sizeof(local_info));

    if (remote_gpuidx) *remote_gpuidx = remote_info.gpu_index;
    if (remote_dev) *remote_dev = 0;

    int actual_connections = std::min(static_cast<int>(total_connections_),
                                      remote_info.total_connections);
    actual_connections = std::max(1, actual_connections);

    uint64_t conn_id = next_conn_id_.fetch_add(1, std::memory_order_relaxed);

    auto group = std::make_shared<TCPConnectionGroup>();
    group->ctrl_fd = ctrl_fd;
    setup_tcp_socket_options(ctrl_fd);

    // Accept data connections
    for (int i = 0; i < actual_connections; i++) {
      int data_fd =
          accept(listen_fd, (struct sockaddr*)&client_addr, &client_len);
      if (data_fd >= 0) {
        auto conn = std::make_unique<TCPConnection>();
        conn->fd = data_fd;
        inet_ntop(AF_INET, &client_addr.sin_addr, ip_str, INET_ADDRSTRLEN);
        conn->remote_ip = ip_str;
        conn->remote_port = ntohs(client_addr.sin_port);
        setup_tcp_socket_options(data_fd);

        // Register data connection with receiver worker
        thread_pool_->assign_data_connection(data_fd);

        group->add_data_connection(std::move(conn));
      }
    }

    LOG(INFO) << "Accepted " << group->data_connection_count()
              << " data connections from peer";

    {
      std::unique_lock<std::shared_mutex> lock(conn_mutex_);
      connection_groups_[conn_id] = group;
    }

    uccl::ConnID result;
    result.flow_id = conn_id;
    result.context = reinterpret_cast<void*>(conn_id);
    return result;
  }

  // Memory registration (no-op for TCP)
  int uccl_regmr(uccl::UcclFlow* flow, void* data, size_t len, int type,
                 struct uccl::Mhandle** mhandle) {
    *mhandle = nullptr;
    return 0;
  }

  int uccl_regmr(void* data, size_t len, MRArray& mr_array) { return 0; }

  int uccl_regmr(int dev, void* data, size_t len, int type,
                 struct uccl::Mhandle** mhandle) {
    if (mhandle) *mhandle = nullptr;
    return 0;
  }

  void uccl_deregmr(struct uccl::Mhandle* mhandle) {}

  void uccl_deregmr(MRArray const& mr_array) {}

  // Send data asynchronously (data is in GPU memory)
  // Sender waits for RecvReady on ctrl connection before sending
  int uccl_send_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh,
                      void const* data, size_t size,
                      struct uccl::ucclRequest* ureq) {
    uint64_t conn_id = reinterpret_cast<uint64_t>(flow);
    auto group = get_connection_group(conn_id);
    if (!group) return -1;

    auto handle = new TCPAsyncHandle();
    uint32_t request_id =
        next_request_id_.fetch_add(1, std::memory_order_relaxed);
    handle->request_id = request_id;

    TCPRequest req;
    req.type = TCPRequestType::SEND;
    req.ctrl_fd = group->ctrl_fd;
    req.data = const_cast<void*>(data);
    req.size = size;
    req.dest_addr = 0;  // Will be received from ctrl connection
    req.completed = &handle->completed;
    req.success = &handle->success;
    req.request_id = request_id;
    req.conn_group = group.get();

    if (!thread_pool_->submit_send_request(req)) {
      delete handle;
      return -1;
    }

    if (ureq) {
      ureq->engine_idx = reinterpret_cast<int64_t>(handle);
      ureq->n = conn_id;
    }

    return 0;
  }

  // Receive data asynchronously (data is in GPU memory)
  // Receiver sends RecvReady on ctrl, then receiver worker handles data via
  // epoll
  int uccl_recv_async(uccl::UcclFlow* flow, struct uccl::Mhandle** mhandles,
                      void** data, int* sizes, int n,
                      struct uccl::ucclRequest* ureq) {
    if (n <= 0) return -1;
    uint64_t conn_id = reinterpret_cast<uint64_t>(flow);
    auto group = get_connection_group(conn_id);
    if (!group) return -1;

    auto handle = new TCPAsyncHandle();
    uint32_t request_id =
        next_request_id_.fetch_add(1, std::memory_order_relaxed);
    handle->request_id = request_id;

    // Register pending receive with receiver workers
    thread_pool_->register_pending_recv(reinterpret_cast<uint64_t>(data[0]),
                                        sizes[0], request_id,
                                        &handle->completed, &handle->success);

    // Send RecvReady message on control connection
    TCPRequest req;
    req.type = TCPRequestType::RECV;
    req.ctrl_fd = group->ctrl_fd;
    req.data = data[0];
    req.size = sizes[0];
    req.dest_addr = 0;
    req.completed = nullptr;  // Completion handled by receiver worker
    req.success = nullptr;
    req.request_id = request_id;
    req.conn_group = group.get();

    if (!thread_pool_->submit_send_request(req)) {
      delete handle;
      return -1;
    }

    if (ureq) {
      ureq->engine_idx = reinterpret_cast<int64_t>(handle);
      ureq->n = conn_id;
    }

    return 0;
  }

  // Poll for completion
  bool uccl_poll_ureq_once(struct uccl::ucclRequest* ureq) {
    if (!ureq) return false;

    TCPAsyncHandle* handle =
        reinterpret_cast<TCPAsyncHandle*>(ureq->engine_idx);
    if (!handle) return true;

    bool completed = handle->completed.load(std::memory_order_acquire);
    if (completed) {
      delete handle;
      ureq->engine_idx = 0;
    }
    return completed;
  }

  // Read from remote memory (dest_addr already known from FifoItem)
  int uccl_read_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh, void* dst,
                      size_t size, uccl::FifoItem const& slot_item,
                      uccl::ucclRequest* ureq) {
    uint64_t conn_id = reinterpret_cast<uint64_t>(flow);
    auto group = get_connection_group(conn_id);
    if (!group) return -1;

    auto handle = new TCPAsyncHandle();
    uint32_t request_id =
        next_request_id_.fetch_add(1, std::memory_order_relaxed);
    handle->request_id = request_id;

    // For READ, we're receiving data - register with receiver workers
    thread_pool_->register_pending_recv(reinterpret_cast<uint64_t>(dst), size,
                                        request_id, &handle->completed,
                                        &handle->success);

    // Send read request to remote side via control connection
    ReadWriteHeader header;
    header.type = static_cast<uint32_t>(TCPRequestType::READ);
    header.reserved = 0;
    header.remote_addr = slot_item.addr;  // Address on remote side
    header.dest_addr = reinterpret_cast<uint64_t>(dst);
    header.size = size;

    uccl::send_message(group->ctrl_fd, &header, sizeof(header));

    if (ureq) {
      ureq->engine_idx = reinterpret_cast<int64_t>(handle);
      ureq->n = conn_id;
    }

    return 0;
  }

  // Write to remote memory (dest_addr already known from FifoItem)
  int uccl_write_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh,
                       void* src, size_t size, uccl::FifoItem const& slot_item,
                       uccl::ucclRequest* ureq) {
    uint64_t conn_id = reinterpret_cast<uint64_t>(flow);
    auto group = get_connection_group(conn_id);
    if (!group) return -1;

    auto handle = new TCPAsyncHandle();
    uint32_t request_id =
        next_request_id_.fetch_add(1, std::memory_order_relaxed);
    handle->request_id = request_id;

    TCPRequest req;
    req.type = TCPRequestType::WRITE;
    req.ctrl_fd = group->ctrl_fd;
    req.data = src;
    req.size = size;
    req.dest_addr = slot_item.addr;  // Remote address already known
    req.completed = &handle->completed;
    req.success = &handle->success;
    req.request_id = request_id;
    req.conn_group = group.get();

    if (!thread_pool_->submit_send_request(req)) {
      delete handle;
      return -1;
    }

    if (ureq) {
      ureq->engine_idx = reinterpret_cast<int64_t>(handle);
      ureq->n = conn_id;
    }

    return 0;
  }

  // Prepare FIFO metadata for advertisement
  int prepare_fifo_metadata(uccl::UcclFlow* flow,
                            struct uccl::Mhandle** mhandle, void const* data,
                            size_t size, char* out_buf) {
    uccl::FifoItem item;
    item.addr = reinterpret_cast<uint64_t>(data);
    item.size = size;
    std::memset(item.padding, 0, sizeof(item.padding));
    uccl::serialize_fifo_item(item, out_buf);
    return 0;
  }

  int get_best_dev_idx(int gpu_idx) { return 0; }

  bool initialize_engine_by_dev(int dev, bool enable_p2p_listen) {
    return true;
  }

  void create_unified_p2p_socket() {}

 private:
  struct NegotiationInfo {
    int32_t gpu_index;
    int32_t num_interfaces;
    int32_t total_connections;
    int32_t reserved;
  };

  struct ReadWriteHeader {
    uint32_t type;
    uint32_t reserved;
    uint64_t remote_addr;
    uint64_t dest_addr;
    size_t size;
  };

  void start_listening() {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
      LOG(ERROR) << "Failed to create listen socket: " << strerror(errno);
      return;
    }

    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(listen_port_);

    if (bind(listen_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
      LOG(ERROR) << "Failed to bind listen socket: " << strerror(errno);
      close(listen_fd_);
      listen_fd_ = -1;
      return;
    }

    if (listen_port_ == 0) {
      socklen_t len = sizeof(addr);
      getsockname(listen_fd_, (struct sockaddr*)&addr, &len);
      listen_port_ = ntohs(addr.sin_port);
    }

    if (listen(listen_fd_, 128) < 0) {
      LOG(ERROR) << "Failed to listen: " << strerror(errno);
      close(listen_fd_);
      listen_fd_ = -1;
      return;
    }

    LOG(INFO) << "TCP listening on port " << listen_port_;
  }

  int create_tcp_connection(std::string const& remote_ip, int remote_port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_in remote_addr;
    std::memset(&remote_addr, 0, sizeof(remote_addr));
    remote_addr.sin_family = AF_INET;
    remote_addr.sin_port = htons(remote_port);
    inet_pton(AF_INET, remote_ip.c_str(), &remote_addr.sin_addr);

    int retries = 100;
    while (connect(fd, (struct sockaddr*)&remote_addr, sizeof(remote_addr)) <
           0) {
      if (--retries <= 0) {
        LOG(ERROR) << "Failed to connect to " << remote_ip << ":"
                   << remote_port;
        close(fd);
        return -1;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return fd;
  }

  int create_tcp_connection_from_interface(std::string const& remote_ip,
                                           int remote_port,
                                           std::string const& local_ip) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_in local_addr;
    std::memset(&local_addr, 0, sizeof(local_addr));
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = 0;
    inet_pton(AF_INET, local_ip.c_str(), &local_addr.sin_addr);

    if (bind(fd, (struct sockaddr*)&local_addr, sizeof(local_addr)) < 0) {
      LOG(WARNING) << "Failed to bind to local IP " << local_ip;
    }

    struct sockaddr_in remote_addr;
    std::memset(&remote_addr, 0, sizeof(remote_addr));
    remote_addr.sin_family = AF_INET;
    remote_addr.sin_port = htons(remote_port);
    inet_pton(AF_INET, remote_ip.c_str(), &remote_addr.sin_addr);

    int retries = 100;
    while (connect(fd, (struct sockaddr*)&remote_addr, sizeof(remote_addr)) <
           0) {
      if (--retries <= 0) {
        close(fd);
        return -1;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return fd;
  }

  void setup_tcp_socket_options(int fd) {
    int opt = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    int bufsize = kTCPBufferSize;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));

    setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &opt, sizeof(opt));
  }

  std::shared_ptr<TCPConnectionGroup> get_connection_group(uint64_t conn_id) {
    std::shared_lock<std::shared_mutex> lock(conn_mutex_);
    auto it = connection_groups_.find(conn_id);
    if (it == connection_groups_.end()) return nullptr;
    return it->second;
  }

  int gpu_index_;
  uint16_t listen_port_;
  int listen_fd_ = -1;
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
