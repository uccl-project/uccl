#include "tcp/tcp_endpoint.h"

namespace tcp {

// Helper functions

uint64_t get_interface_bandwidth(std::string const& ifname) {
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

std::vector<InterfaceInfo> parse_tcp_interfaces() {
  std::vector<InterfaceInfo> interfaces;

  char const* env = std::getenv("UCCL_P2P_TCP_IFNAME");
  if (!env || strlen(env) == 0) {
    char ifNames[MAX_IFS * MAX_IF_NAME_SIZE];
    uccl::socketAddress ifAddrs[MAX_IFS];
    int nIfs =
        uccl::find_interfaces(ifNames, ifAddrs, MAX_IF_NAME_SIZE, MAX_IFS);

    for (int i = 0; i < nIfs; i++) {
      std::string name = &ifNames[i * MAX_IF_NAME_SIZE];
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

TCPEndpoint::TCPEndpoint(int gpu_index, uint16_t port)
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

TCPEndpoint::~TCPEndpoint() {
  running_ = false;

  if (thread_pool_) {
    thread_pool_->stop();
  }

  if (listen_fd_ >= 0) {
    close(listen_fd_);
    listen_fd_ = -1;
  }

  // Close per-interface data listen sockets
  for (int fd : data_listen_fds_) {
    if (fd >= 0) {
      close(fd);
    }
  }
  data_listen_fds_.clear();

  std::unique_lock<std::shared_mutex> lock(conn_mutex_);
  connection_groups_.clear();
}

uccl::ConnID TCPEndpoint::uccl_connect(int dev, int local_gpuidx,
                                       int remote_dev, int remote_gpuidx,
                                       std::string remote_ip,
                                       uint16_t remote_port) {
  uint64_t conn_id = next_conn_id_.fetch_add(1, std::memory_order_relaxed);

  LOG(INFO) << "TCPEndpoint::uccl_connect to " << remote_ip << ":"
            << remote_port;

  auto group = std::make_shared<TCPConnectionGroup>();

  NegotiationInfo local_info;
  local_info.gpu_index = local_gpuidx;
  local_info.num_interfaces = interfaces_.size();
  local_info.total_connections = total_connections_;
  local_info.reserved = 0;

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

  // Exchange negotiation info (base info first)
  uccl::send_message(ctrl_fd, &local_info, sizeof(local_info));
  NegotiationInfo remote_info;
  uccl::receive_message(ctrl_fd, &remote_info, sizeof(remote_info));

  // Send our interface info (including data ports)
  for (size_t i = 0; i < interfaces_.size(); i++) {
    InterfaceNegotiationInfo iface_info;
    std::memset(&iface_info, 0, sizeof(iface_info));
    std::strncpy(iface_info.ip_addr, interfaces_[i].ip_addr.c_str(),
                 sizeof(iface_info.ip_addr) - 1);
    iface_info.num_connections = interfaces_[i].num_connections;
    iface_info.data_port =
        (i < data_listen_ports_.size()) ? data_listen_ports_[i] : listen_port_;
    iface_info.reserved = 0;
    uccl::send_message(ctrl_fd, &iface_info, sizeof(iface_info));
  }

  // Receive remote interface info (including data ports)
  std::vector<InterfaceNegotiationInfo> remote_interfaces(
      remote_info.num_interfaces);
  for (int i = 0; i < remote_info.num_interfaces; i++) {
    uccl::receive_message(ctrl_fd, &remote_interfaces[i],
                          sizeof(InterfaceNegotiationInfo));
  }

  int actual_connections = std::min(static_cast<int>(total_connections_),
                                    remote_info.total_connections);
  actual_connections = std::max(1, actual_connections);

  LOG(INFO) << "Negotiated " << actual_connections << " data connections with "
            << remote_info.num_interfaces << " remote interfaces";

  // Create data connections: match local interfaces to remote interfaces
  // Connect to remote interface IPs on their specific data ports
  int created_connections = 0;
  size_t local_iface_idx = 0;
  size_t remote_iface_idx = 0;

  while (created_connections < actual_connections) {
    auto const& local_iface = interfaces_[local_iface_idx % interfaces_.size()];
    auto const& remote_iface =
        remote_interfaces[remote_iface_idx % remote_interfaces.size()];

    std::string remote_iface_ip(remote_iface.ip_addr);
    uint16_t remote_data_port = remote_iface.data_port;

    int fd = create_tcp_connection_from_interface(
        remote_iface_ip, remote_data_port, local_iface.ip_addr);
    if (fd >= 0) {
      auto conn = std::make_unique<TCPConnection>();
      conn->fd = fd;
      conn->local_ip = local_iface.ip_addr;
      conn->remote_ip = remote_iface_ip;
      conn->remote_port = remote_data_port;
      setup_tcp_socket_options(fd);

      // Register data connection with sender and receiver workers
      thread_pool_->assign_data_connection(fd, conn.get());

      group->add_data_connection(std::move(conn));
      created_connections++;
    }

    // Round-robin through interfaces
    local_iface_idx++;
    remote_iface_idx++;
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

uccl::ConnID TCPEndpoint::uccl_accept(int dev, int listen_fd, int local_gpuidx,
                                      std::string& remote_ip, int* remote_dev,
                                      int* remote_gpuidx) {
  struct sockaddr_in client_addr;
  socklen_t client_len = sizeof(client_addr);

  // Accept control connection
  int ctrl_fd = accept(listen_fd, (struct sockaddr*)&client_addr, &client_len);
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
  local_info.reserved = 0;

  NegotiationInfo remote_info;
  uccl::receive_message(ctrl_fd, &remote_info, sizeof(remote_info));
  uccl::send_message(ctrl_fd, &local_info, sizeof(local_info));

  // Receive remote interface info (including data ports)
  std::vector<InterfaceNegotiationInfo> remote_interfaces(
      remote_info.num_interfaces);
  for (int i = 0; i < remote_info.num_interfaces; i++) {
    uccl::receive_message(ctrl_fd, &remote_interfaces[i],
                          sizeof(InterfaceNegotiationInfo));
  }

  // Send our interface info (including data ports)
  for (size_t i = 0; i < interfaces_.size(); i++) {
    InterfaceNegotiationInfo iface_info;
    std::memset(&iface_info, 0, sizeof(iface_info));
    std::strncpy(iface_info.ip_addr, interfaces_[i].ip_addr.c_str(),
                 sizeof(iface_info.ip_addr) - 1);
    iface_info.num_connections = interfaces_[i].num_connections;
    iface_info.data_port =
        (i < data_listen_ports_.size()) ? data_listen_ports_[i] : listen_port_;
    iface_info.reserved = 0;
    uccl::send_message(ctrl_fd, &iface_info, sizeof(iface_info));
  }

  if (remote_gpuidx) *remote_gpuidx = remote_info.gpu_index;
  if (remote_dev) *remote_dev = 0;

  int actual_connections = std::min(static_cast<int>(total_connections_),
                                    remote_info.total_connections);
  actual_connections = std::max(1, actual_connections);

  uint64_t conn_id = next_conn_id_.fetch_add(1, std::memory_order_relaxed);

  auto group = std::make_shared<TCPConnectionGroup>();
  group->ctrl_fd = ctrl_fd;
  setup_tcp_socket_options(ctrl_fd);

  // Accept data connections from per-interface listen sockets
  // Build fd_set once outside the loop
  fd_set listen_fds;
  FD_ZERO(&listen_fds);
  int max_fd = -1;

  for (int data_listen_fd : data_listen_fds_) {
    FD_SET(data_listen_fd, &listen_fds);
    if (data_listen_fd > max_fd) max_fd = data_listen_fd;
  }

  // Also include control listen_fd as fallback
  FD_SET(listen_fd, &listen_fds);
  if (listen_fd > max_fd) max_fd = listen_fd;

  int accepted_connections = 0;
  while (accepted_connections < actual_connections) {
    // Copy fd_set since select() modifies it
    fd_set read_fds = listen_fds;

    struct timeval timeout;
    timeout.tv_sec = 30;
    timeout.tv_usec = 0;

    int ready = select(max_fd + 1, &read_fds, nullptr, nullptr, &timeout);
    if (ready <= 0) {
      LOG(ERROR) << "Timeout waiting for data connection";
      break;
    }

    // Accept from whichever fd is ready
    int data_fd = -1;
    std::string local_iface_ip;

    for (size_t i = 0; i < data_listen_fds_.size(); i++) {
      if (FD_ISSET(data_listen_fds_[i], &read_fds)) {
        data_fd = accept(data_listen_fds_[i], (struct sockaddr*)&client_addr,
                         &client_len);
        if (data_fd >= 0 && i < interfaces_.size()) {
          local_iface_ip = interfaces_[i].ip_addr;
        }
        break;
      }
    }

    // Fallback to control listen_fd
    if (data_fd < 0 && FD_ISSET(listen_fd, &read_fds)) {
      data_fd = accept(listen_fd, (struct sockaddr*)&client_addr, &client_len);
    }

    if (data_fd < 0) {
      LOG(ERROR) << "Failed to accept data connection: " << strerror(errno);
      break;
    }

    auto conn = std::make_unique<TCPConnection>();
    conn->fd = data_fd;
    conn->local_ip = local_iface_ip;
    inet_ntop(AF_INET, &client_addr.sin_addr, ip_str, INET_ADDRSTRLEN);
    conn->remote_ip = ip_str;
    conn->remote_port = ntohs(client_addr.sin_port);
    setup_tcp_socket_options(data_fd);

    // Register data connection with sender and receiver workers
    thread_pool_->assign_data_connection(data_fd, conn.get());

    group->add_data_connection(std::move(conn));
    accepted_connections++;
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

int TCPEndpoint::uccl_regmr(uccl::UcclFlow* flow, void* data, size_t len,
                            int type, struct uccl::Mhandle** mhandle) {
  *mhandle = nullptr;
  return 0;
}

int TCPEndpoint::uccl_regmr(void* data, size_t len, MRArray& mr_array) {
  return 0;
}

int TCPEndpoint::uccl_regmr(int dev, void* data, size_t len, int type,
                            struct uccl::Mhandle** mhandle) {
  if (mhandle) *mhandle = nullptr;
  return 0;
}

void TCPEndpoint::uccl_deregmr(struct uccl::Mhandle* mhandle) {}

void TCPEndpoint::uccl_deregmr(MRArray const& mr_array) {}

int TCPEndpoint::uccl_send_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh,
                                 void const* data, size_t size,
                                 struct uccl::ucclRequest* ureq) {
  uint64_t conn_id = reinterpret_cast<uint64_t>(flow);
  auto group = get_connection_group(conn_id);
  if (!group) return -1;

  auto handle = new TCPAsyncHandle();

  // Wait for RecvReady message ONCE on control connection
  RecvReadyMsg ready_msg;
  if (!recv_exact(group->ctrl_fd, &ready_msg, sizeof(ready_msg))) {
    LOG(ERROR) << "uccl_send_async: failed to recv RecvReady";
    delete handle;
    return -1;
  }

  uint64_t base_dest_addr = ready_msg.dest_addr;
  // Use receiver's request_id for chunks so receiver can track completion
  uint32_t recv_request_id = ready_msg.request_id;
  // Generate our own request_id for sender-side completion tracking
  uint32_t send_request_id =
      next_request_id_.fetch_add(1, std::memory_order_relaxed);
  handle->request_id = send_request_id;

  // Register pending send with total size for sender completion tracking
  thread_pool_->register_pending_send(size, send_request_id, &handle->completed,
                                      &handle->success);

  // Chunk the message and distribute across workers
  size_t num_chunks = (size + kChunkSize - 1) / kChunkSize;
  size_t offset = 0;

  for (size_t i = 0; i < num_chunks; ++i) {
    size_t chunk_size = std::min(kChunkSize, size - offset);
    bool is_last = (i == num_chunks - 1);

    // Select different connection for each chunk (load balance across workers)
    TCPConnection* conn = group->select_data_connection();
    if (!conn) {
      delete handle;
      return -1;
    }

    TCPRequest req;
    req.type = TCPRequestType::SEND;
    req.ctrl_fd = group->ctrl_fd;
    req.data = const_cast<char*>(static_cast<char const*>(data) + offset);
    req.size = chunk_size;
    req.total_size = size;
    req.dest_addr = base_dest_addr + offset;  // Destination with offset
    req.flags = is_last ? TCPDataHeader::kFlagLastChunk : 0;
    req.request_id = send_request_id;       // For sender completion tracking
    req.recv_request_id = recv_request_id;  // For receiver completion tracking
    req.conn_group = group.get();
    req.assigned_conn = conn;

    thread_pool_->submit_request(req);

    // Track inflight chunks for load balancing
    conn->inflight_chunks.fetch_add(1, std::memory_order_relaxed);

    offset += chunk_size;
  }

  if (ureq) {
    ureq->engine_idx = reinterpret_cast<int64_t>(handle);
    ureq->n = conn_id;
  }

  return 0;
}

int TCPEndpoint::uccl_recv_async(uccl::UcclFlow* flow,
                                 struct uccl::Mhandle** mhandles, void** data,
                                 int* sizes, int n,
                                 struct uccl::ucclRequest* ureq) {
  if (n <= 0) return -1;
  uint64_t conn_id = reinterpret_cast<uint64_t>(flow);
  auto group = get_connection_group(conn_id);
  if (!group) return -1;

  auto handle = new TCPAsyncHandle();
  uint32_t request_id =
      next_request_id_.fetch_add(1, std::memory_order_relaxed);
  handle->request_id = request_id;

  // Register pending receive with receiver workers for FULL size
  // The receiver will accumulate chunks and mark complete when all received
  thread_pool_->register_pending_recv(reinterpret_cast<uint64_t>(data[0]),
                                      sizes[0], request_id, &handle->completed,
                                      &handle->success);

  // Send RecvReady message ONCE directly on control connection
  RecvReadyMsg msg;
  msg.dest_addr = reinterpret_cast<uint64_t>(data[0]);
  msg.size = sizes[0];
  msg.request_id = request_id;
  msg.reserved = 0;

  if (!send_exact(group->ctrl_fd, &msg, sizeof(msg))) {
    LOG(ERROR) << "uccl_recv_async: failed to send RecvReady";
    delete handle;
    return -1;
  }

  if (ureq) {
    ureq->engine_idx = reinterpret_cast<int64_t>(handle);
    ureq->n = conn_id;
  }

  return 0;
}

int TCPEndpoint::uccl_read_async(uccl::UcclFlow* flow, struct uccl::Mhandle* mh,
                                 void* dst, size_t size,
                                 uccl::FifoItem const& slot_item,
                                 uccl::ucclRequest* ureq) {
  uint64_t conn_id = reinterpret_cast<uint64_t>(flow);
  auto group = get_connection_group(conn_id);
  if (!group) return -1;

  auto handle = new TCPAsyncHandle();
  uint32_t request_id =
      next_request_id_.fetch_add(1, std::memory_order_relaxed);
  handle->request_id = request_id;

  // For READ, we're receiving data - register pending recv for FULL size
  // The receiver will accumulate chunks and mark complete when all received
  thread_pool_->register_pending_recv(reinterpret_cast<uint64_t>(dst), size,
                                      request_id, &handle->completed,
                                      &handle->success);

  // Chunk the read request and distribute across workers
  size_t num_chunks = (size + kChunkSize - 1) / kChunkSize;
  size_t offset = 0;
  uint64_t base_dst = reinterpret_cast<uint64_t>(dst);
  uint64_t base_remote = slot_item.addr;

  for (size_t i = 0; i < num_chunks; ++i) {
    size_t chunk_size = std::min(kChunkSize, size - offset);

    // Select different connection for each chunk (load balance across workers)
    TCPConnection* conn = group->select_data_connection();
    if (!conn) {
      delete handle;
      return -1;
    }

    // Submit READ request to sender worker - it will send the request
    // on a data connection so the remote receiver can see it via epoll
    TCPRequest req;
    req.type = TCPRequestType::READ;
    req.ctrl_fd = group->ctrl_fd;
    req.data = nullptr;  // No data to send for READ request
    req.size = chunk_size;
    req.total_size = size;
    req.dest_addr = base_dst + offset;       // Where to put data locally
    req.remote_addr = base_remote + offset;  // Where to read from remotely
    req.completed = nullptr;  // Completion tracked by pending_recv
    req.success = nullptr;
    req.request_id = request_id;       // For READ response tracking
    req.recv_request_id = request_id;  // Same as request_id for READ
    req.conn_group = group.get();
    req.assigned_conn = conn;

    thread_pool_->submit_request(req);

    conn->inflight_chunks.fetch_add(1, std::memory_order_relaxed);

    offset += chunk_size;
  }

  if (ureq) {
    ureq->engine_idx = reinterpret_cast<int64_t>(handle);
    ureq->n = conn_id;
  }

  return 0;
}

int TCPEndpoint::uccl_write_async(uccl::UcclFlow* flow,
                                  struct uccl::Mhandle* mh, void* src,
                                  size_t size, uccl::FifoItem const& slot_item,
                                  uccl::ucclRequest* ureq) {
  uint64_t conn_id = reinterpret_cast<uint64_t>(flow);
  auto group = get_connection_group(conn_id);
  if (!group) return -1;

  auto handle = new TCPAsyncHandle();
  uint32_t request_id =
      next_request_id_.fetch_add(1, std::memory_order_relaxed);
  handle->request_id = request_id;

  // Register pending send with total size for completion tracking
  thread_pool_->register_pending_send(size, request_id, &handle->completed,
                                      &handle->success);

  // Chunk the message and distribute across workers
  size_t num_chunks = (size + kChunkSize - 1) / kChunkSize;
  size_t offset = 0;
  uint64_t base_dest_addr = slot_item.addr;

  for (size_t i = 0; i < num_chunks; ++i) {
    size_t chunk_size = std::min(kChunkSize, size - offset);
    bool is_last = (i == num_chunks - 1);

    // Select different connection for each chunk (load balance across workers)
    TCPConnection* conn = group->select_data_connection();
    if (!conn) {
      delete handle;
      return -1;
    }

    TCPRequest req;
    req.type = TCPRequestType::WRITE;
    req.ctrl_fd = group->ctrl_fd;
    req.data = static_cast<char*>(src) + offset;
    req.size = chunk_size;
    req.total_size = size;
    req.dest_addr = base_dest_addr + offset;  // Remote address with offset
    req.flags = is_last ? TCPDataHeader::kFlagLastChunk : 0;
    req.request_id = request_id;       // For sender completion tracking
    req.recv_request_id = request_id;  // Not used for WRITE (one-sided)
    req.conn_group = group.get();
    req.assigned_conn = conn;

    thread_pool_->submit_request(req);

    // Track inflight chunks for load balancing
    conn->inflight_chunks.fetch_add(1, std::memory_order_relaxed);

    offset += chunk_size;
  }

  if (ureq) {
    ureq->engine_idx = reinterpret_cast<int64_t>(handle);
    ureq->n = conn_id;
  }

  return 0;
}

bool TCPEndpoint::uccl_poll_ureq_once(struct uccl::ucclRequest* ureq) {
  if (!ureq) return false;

  TCPAsyncHandle* handle = reinterpret_cast<TCPAsyncHandle*>(ureq->engine_idx);
  if (!handle) return true;

  bool completed = handle->completed.load(std::memory_order_acquire);
  if (completed) {
    delete handle;
    ureq->engine_idx = 0;
  }
  return completed;
}

int TCPEndpoint::prepare_fifo_metadata(uccl::UcclFlow* flow,
                                       struct uccl::Mhandle** mhandle,
                                       void const* data, size_t size,
                                       char* out_buf) {
  uccl::FifoItem item;
  item.addr = reinterpret_cast<uint64_t>(data);
  item.size = size;
  std::memset(item.padding, 0, sizeof(item.padding));
  uccl::serialize_fifo_item(item, out_buf);
  return 0;
}

void TCPEndpoint::start_listening() {
  // Create control listen socket on INADDR_ANY
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
  addr.sin_port = htons(0);

  if (bind(listen_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    LOG(ERROR) << "Failed to bind listen socket: " << strerror(errno);
    close(listen_fd_);
    listen_fd_ = -1;
    return;
  }

  socklen_t len = sizeof(addr);
  getsockname(listen_fd_, (struct sockaddr*)&addr, &len);
  listen_port_ = ntohs(addr.sin_port);

  if (listen(listen_fd_, 128) < 0) {
    LOG(ERROR) << "Failed to listen: " << strerror(errno);
    close(listen_fd_);
    listen_fd_ = -1;
    return;
  }

  LOG(INFO) << "TCP control listening on port " << listen_port_;

  // Create per-interface data listen sockets with unique ports
  for (auto const& iface : interfaces_) {
    int data_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (data_fd < 0) {
      LOG(ERROR) << "Failed to create data listen socket for " << iface.ip_addr;
      continue;
    }

    setsockopt(data_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in data_addr;
    std::memset(&data_addr, 0, sizeof(data_addr));
    data_addr.sin_family = AF_INET;
    inet_pton(AF_INET, iface.ip_addr.c_str(), &data_addr.sin_addr);
    data_addr.sin_port = htons(0);  // Let OS assign a unique port

    if (bind(data_fd, (struct sockaddr*)&data_addr, sizeof(data_addr)) < 0) {
      LOG(ERROR) << "Failed to bind data listen socket to " << iface.ip_addr
                 << ": " << strerror(errno);
      close(data_fd);
      continue;
    }

    // Get the assigned port
    socklen_t addr_len = sizeof(data_addr);
    getsockname(data_fd, (struct sockaddr*)&data_addr, &addr_len);
    uint16_t data_port = ntohs(data_addr.sin_port);

    if (listen(data_fd, 128) < 0) {
      LOG(ERROR) << "Failed to listen on " << iface.ip_addr;
      close(data_fd);
      continue;
    }

    data_listen_fds_.push_back(data_fd);
    data_listen_ports_.push_back(data_port);
    LOG(INFO) << "TCP data listening on " << iface.ip_addr << ":" << data_port;
  }
}

int TCPEndpoint::create_tcp_connection(std::string const& remote_ip,
                                       int remote_port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;

  struct sockaddr_in remote_addr;
  std::memset(&remote_addr, 0, sizeof(remote_addr));
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(remote_port);
  inet_pton(AF_INET, remote_ip.c_str(), &remote_addr.sin_addr);

  int retries = 100;
  while (connect(fd, (struct sockaddr*)&remote_addr, sizeof(remote_addr)) < 0) {
    if (--retries <= 0) {
      LOG(ERROR) << "Failed to connect to " << remote_ip << ":" << remote_port;
      close(fd);
      return -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  return fd;
}

int TCPEndpoint::create_tcp_connection_from_interface(
    std::string const& remote_ip, int remote_port,
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
  while (connect(fd, (struct sockaddr*)&remote_addr, sizeof(remote_addr)) < 0) {
    if (--retries <= 0) {
      close(fd);
      return -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  return fd;
}

void TCPEndpoint::setup_tcp_socket_options(int fd) {
  int opt = 1;
  if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0) {
    LOG(ERROR) << "Failed to set TCP_NODELAY: " << strerror(errno);
  }
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    LOG(ERROR) << "Failed to set SO_REUSEADDR: " << strerror(errno);
  }
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
    LOG(ERROR) << "Failed to set SO_REUSEPORT: " << strerror(errno);
  }
}

std::shared_ptr<TCPConnectionGroup> TCPEndpoint::get_connection_group(
    uint64_t conn_id) {
  std::shared_lock<std::shared_mutex> lock(conn_mutex_);
  auto it = connection_groups_.find(conn_id);
  if (it == connection_groups_.end()) return nullptr;
  return it->second;
}

}  // namespace tcp
