#include "tcpx_engine.h"
#include "tcpx_interface.h"
#include <iostream>
#include <socket.h>

thread_local bool inside_python = false;
#define kDescRingSize 2048

struct UnpackDescriptorBlock {
  int todo;
}

struct Desc { // TODO: UnpackDescriptorBlock
  uint64_t transfer_id;
  UnpackDescriptorBlock* desc;
}

namespace tcpx {

/* 
* Helper Functions
*/
static int get_numa_node_from_pci(const char* pci_path) {
  char sysfs_path[256];
  snprintf(sysfs_path, sizeof(sysfs_path),
           "/sys/bus/pci/devices/%s/numa_node", pci_path);
  FILE* f = fopen(sysfs_path, "r");
  if (!f) return -1;
  int numa = -1;
  fscanf(f, "%d", &numa);
  fclose(f);
  return numa;
}

static float get_pci_distance(int local_gpu_idx, const char* nic_pci_path) {
  char gpu_info_path[256];
  snprintf(gpu_info_path, sizeof(gpu_info_path),
           "/proc/driver/nvidia/gpus/%d/information", local_gpu_idx);

  FILE* f = fopen(gpu_info_path, "r");
  if (!f) return 2.5; // unknow

  char line[256];
  std::string gpu_pci;
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "Bus Location", 12) == 0) {
      gpu_pci = std::string(strchr(line, ':') + 2);
      gpu_pci.erase(gpu_pci.find('\n'));
      break;
    }
  }
  fclose(f);

  if (gpu_pci.empty()) return 2.5;

  int gpu_numa = get_numa_node_from_pci(gpu_pci.c_str());
  int nic_numa = get_numa_node_from_pci(nic_pci_path);

  if (gpu_numa == -1 || nic_numa == -1) return 2.5;
  if (gpu_numa == nic_numa) return 1.0;
  return 2.0;
}

int find_best_dev(uint32_t local_gpu_idx) {
  int device_count = tcpx_get_device_count();
  if (device_count <= 0) {
    std::cerr << "No TCPX devices found!" << std::endl;
    return -1;
  }

  int best_dev = -1;
  float best_score = -std::numeric_limits<float>::infinity();

  for (int dev_idx = 0; dev_idx < device_count; ++dev_idx) {
    tcpx_net_properties prop{};
    if (tcpx_get_properties(dev_idx, &prop) != 0) continue;

    float dist = get_pci_distance(local_gpu_idx, prop.pci_path);
    if (dist <= 0.0f) dist = 0.1f;

    float score = static_cast<float>(prop.speed) / dist - prop.latency * 0.05f;

    std::cout << "[DEV " << dev_idx << "] "
              << "name=" << prop.name
              << " pci=" << prop.pci_path
              << " speed=" << prop.speed << "Mbps"
              << " latency=" << prop.latency << "us"
              << " dist=" << dist
              << " score=" << score
              << std::endl;

    if (score > best_score) {
      best_score = score;
      best_dev = dev_idx;
    }
  }

  if (best_dev >= 0)
    std::cout << "Selected best TCPX device: " << best_dev
              << " (score=" << best_score << ")" << std::endl;
  else
    std::cout << "No valid TCPX device selected." << std::endl;

  return best_dev;
}

int get_env_int(char const* name, int def) {
  char const* v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

/* 
* Socket Control Msg
*/
enum CtrlMsgType : uint16_t {
  CTRL_ACK = 0x01,
  CTRL_INT = 0x02,
  CTRL_STRUCT = 0x03,
};

struct CtrlMsgHeader {
  uint16_t type;
  uint16_t flags;
  uint32_t length; // payload lens
};
static_assert(sizeof(CtrlMsgHeader) == 8);

inline bool send_all(int fd, const void* buf, size_t len) {
  const char* p = static_cast<const char*>(buf);
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = ::send(fd, p + sent, len - sent, 0);
    if (n <= 0) return false;
    sent += n;
  }
  return true;
}

inline bool recv_all(int fd, void* buf, size_t len) {
  char* p = static_cast<char*>(buf);
  size_t recvd = 0;
  while (recvd < len) {
    ssize_t n = ::recv(fd, p + recvd, len - recvd, 0);
    if (n <= 0) return false;
    recvd += n;
  }
  return true;
}

// send control msg
inline bool send_ctrl_msg(int fd, CtrlMsgType type, const void* payload, size_t len) {
  CtrlMsgHeader hdr{type, 0, static_cast<uint32_t>(len)};
  if (!send_all(fd, &hdr, sizeof(hdr))) return false;
  if (len > 0 && payload) return send_all(fd, payload, len);
  return true;
}

// recv control msg
inline bool recv_ctrl_msg(int fd, CtrlMsgHeader& hdr, std::vector<uint8_t>& payload) {
  if (!recv_all(fd, &hdr, sizeof(hdr))) return false;
  payload.resize(hdr.length);
  if (hdr.length > 0) return recv_all(fd, payload.data(), hdr.length);
  return true;
}

// send int
inline bool send_ctrl_int(int fd, int value) {
  int32_t net = htonl(value);
  return send_ctrl_msg(fd, CTRL_INT, &net, sizeof(net));
}

// recv int
inline bool recv_ctrl_int(int fd, int& value) {
  CtrlMsgHeader hdr;
  std::vector<uint8_t> payload;
  if (!recv_ctrl_msg(fd, hdr, payload)) return false;
  if (hdr.type != CTRL_INT || payload.size() != sizeof(int32_t)) return false;
  int32_t net;
  std::memcpy(&net, payload.data(), sizeof(net));
  value = ntohl(net);
  return true;
}

// send struct
template <typename T>
bool send_ctrl_struct(int fd, const T& obj) {
  static_assert(std::is_trivially_copyable<T>::value, "T must be POD");
  return send_ctrl_msg(fd, CTRL_STRUCT, &obj, sizeof(T));
}

// recv struct
template <typename T>
bool recv_ctrl_struct(int fd, T& obj) {
  CtrlMsgHeader hdr;
  std::vector<uint8_t> payload;
  if (!recv_ctrl_msg(fd, hdr, payload)) return false;
  if (hdr.type != CTRL_STRUCT || payload.size() != sizeof(T)) return false;
  std::memcpy(&obj, payload.data(), sizeof(T));
  return true;
}

/* 
* Endpoint Class
*/
Endpoint::Endpoint(uint32_t const local_gpu_idx, uint32_t const num_cpus) {
  local_gpu_idx_ = local_gpu_idx;
  dev_id_ = find_best_dev(local_gpu_idx);
  int port = get_env_int("UCCL_TCPX_CTRL_PORT", 9999);

  // TCPX listener
  void* handle = nullptr;
  int ret = tcpx_listen(dev_id_, handle, &listen_comms_);
  if (ret != 0) {
    std::cerr << "Failed to listen at device " << dev_id_ << "!" << std::endl;
  }

  // Control TCP listener
  int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);

  if (::bind(listen_fd, (sockaddr*)&addr, sizeof(addr)) < 0)
    std::cerr << "Failed to bind control TCP listener!" << std::endl;

  ::listen(listen_fd, 128);
  ctrl_listen_fd_ = listen_fd;

  // Start unpacker thread.
  unpacker_thread_ = std::thread(&Endpoint::unpacker_thread_func_, this);
  unpacker_desc_ring_ =
      uccl::create_ring(sizeof(Desc), kDescRingSize);
}

static void Endpoint::free_conn_(Conn* conn) {
  if (!conn) return;
  if (conn->tcpx_comm_) {
      tcpx_close_comm(conn->tcpx_comm_);
      conn->tcpx_comm_ = nullptr;
  }
  if (conn->ctrl_sock_fd_ >= 0) {
      close(conn->ctrl_sock_fd_);
      conn->ctrl_sock_fd_ = -1;
  }
  delete conn;
}

Endpoint::~Endpoint() { 
  int ret = 0;

  unpacker_thread_.join();

  if (listen_comms_)
    ret = tcpx_close_listen(listen_comms_);

  // free conns
  std::vector<Conn*> conns_to_free;
  {
      std::unique_lock<std::shared_mutex> lock(conn_mu_);
      for (auto& pair : conn_id_to_conn_) {
          conns_to_free.push_back(pair.second);
      }
      conn_id_to_conn_.clear();
  }
  for (Conn* conn : conns_to_free) {
      free_conn_(conn);
  }

  if (unpacker_desc_ring_) {
    free(unpacker_desc_ring_);
  }
}

bool Endpoint::connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
                       uint64_t& conn_id) {
  // TCP socket
  int sock_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    std::cerr << "Failed to create socket\n";
    return false;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(remote_port);
  inet_pton(AF_INET, ip_addr.c_str(), &addr.sin_addr);

  if (::connect(sock_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
    std::cerr << "Failed to connect control TCP to " << ip_addr << ":"
              << remote_port << std::endl;
    ::close(sock_fd);
    return false;
  }

  // TODO: danyang 
  // send/recv ncclNetHandle_v7 for TCPX via send_ctrl_struct<TYPE>()

  // TCPX
  void* tcpx_comm = nullptr;
  void* dev_handle = nullptr;
  int ret = tcpx_connect_v5(dev_id_, nullptr, &tcpx_comm, &dev_handle);
  if (ret != 0) {
    std::cerr << "tcpx_connect_v5() failed, ret=" << ret << std::endl;
    ::close(sock_fd);
    return false;
  }

  // Handshake：send GPU ID，wait ACK
  if (!send_ctrl_int(sock_fd, local_gpu_idx_)) {
    std::cerr << "Failed to send GPU ID on control socket" << std::endl;
  }

  CtrlMsgHeader ack_hdr{};
  std::vector<uint8_t> ack_payload;
  if (!recv_ctrl_msg(sock_fd, ack_hdr, ack_payload) || ack_hdr.type != CTRL_ACK) {
    std::cerr << "Handshake ACK failed from " << ip_addr << std::endl;
  }

  // Register
  conn_id = next_conn_id_.fetch_add(1);
  auto* conn = new Conn{
      .conn_id_ = conn_id,
      .ip_addr_ = ip_addr,
      .remote_gpu_idx_ = remote_gpu_idx,
      .remote_port_ = remote_port,
      .tcpx_comm_ = tcpx_comm,
      .ctrl_sock_fd_ = sock_fd,
  };

  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    conn_id_to_conn_[conn_id] = conn;
  }

  std::cout << "[connect] Established connection to " << ip_addr
            << " (dev " << dev_id_ << ", conn_id=" << conn_id
            << ", gpu=" << local_gpu_idx_ << ")" << std::endl;
  return true;
}

bool Endpoint::accept(std::string& ip_addr, int& remote_gpu_idx,
                      uint64_t& conn_id) {
  // TCP
  sockaddr_in client_addr{};
  socklen_t addrlen = sizeof(client_addr);
  int sock_fd = ::accept(ctrl_listen_fd_, (sockaddr*)&client_addr, &addrlen);
  if (sock_fd < 0) {
    std::cerr << "Failed to accept control TCP connection!" << std::endl;
    return false;
  }

  char ip_buf[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &client_addr.sin_addr, ip_buf, sizeof(ip_buf));
  ip_addr = ip_buf;

  // TODO send/recv ncclNetHandle_v7 for TCPX via send_ctrl_struct<TYPE>()

  // TCPX
  void* tcpx_comm = nullptr;
  void* dev_handle = nullptr;
  int ret = tcpx_accept_v5(listen_comms_, &tcpx_comm, &dev_handle);
  if (ret != 0) {
    std::cerr << "tcpx_accept_v5() failed, ret=" << ret << std::endl;
    ::close(sock_fd);
    return false;
  }

  // Handshake：recv GPU ID，send ACK
  int peer_gpu = -1;
  if (!recv_ctrl_int(sock_fd, peer_gpu)) {
    std::cerr << "Failed to receive GPU ID from client " << ip_addr << std::endl;
  }
  remote_gpu_idx = peer_gpu;

  if (!send_ctrl_msg(sock_fd, CTRL_ACK, nullptr, 0)) {
    std::cerr << "Failed to send ACK to " << ip_addr << std::endl;
  }

  // Register
  conn_id = next_conn_id_.fetch_add(1);
  auto* conn = new Conn{
      .conn_id_ = conn_id,
      .ip_addr_ = ip_addr,
      .remote_gpu_idx_ = remote_gpu_idx,
      .remote_port_ = ntohs(client_addr.sin_port),
      .tcpx_comm_ = tcpx_comm,
      .ctrl_sock_fd_ = sock_fd,
  };

  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    conn_id_to_conn_[conn_id] = conn;
  }

  std::cout << "[accept] Accepted connection from " << ip_addr
            << " (dev " << dev_id_ << ", conn_id=" << conn_id
            << ", peer_gpu=" << remote_gpu_idx << ")" << std::endl;
  return true;
}

std::vector<uint8_t> Endpoint::get_metadata() {
  return std::vector<uint8_t>{0};
}

bool Endpoint::advertise(uint64_t conn_id, uint64_t mr_id, void* addr,
                         size_t len, char* out_buf) {
  return true;
}

bool Endpoint::reg(void const* data, size_t size, uint64_t& mr_id) {
  return true;
}

bool Endpoint::dereg(uint64_t mr_id) { return true; }

bool Endpoint::read_async(uint64_t conn_id, uint64_t mr_id, void* dst,
                          size_t size, FifoItem const& slot_item,
                          uint64_t* transfer_id) {
  return true;
}

bool Endpoint::send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                          size_t size, uint64_t* transfer_id) {
  // TODO: danyang 
  // tcpx_isend
  // transfer_id = task_handle;
  return true;
}

void Endpoint::unpacker_thread_func_() {
  struct Desc task; // TODO: struct desc
  
  while (!stop_.load(std::memory_order_acquire)) {
    if (jring_sc_dequeue_bulk(unpacker_desc_ring_, &task, 1, nullptr) == 1) {
      
      // Danyang: get Desc from deque, launch kernel for it


      // Jacelau:
      // Persisitent kernel: launch the kernel if have not been launched,
      // Submit desc to the kernel.

      {
        std::unique_lock<std::shared_mutex> lock(recv_transfer_status_mu_);
        recv_transfer_status_[transfer_id] = true;
      }
    }
  }
  return;
}

bool Endpoint::recv_async(uint64_t conn_id, uint64_t mr_id, void* data,
                          size_t size, uint64_t* transfer_id) {
  // TODO: danyang:
  // tcpx_irecv(), build Desc, then push it/them to the deque
  // transfer_id = task_handle;
  
  struct Desc task;
  task.transfer_id = *transfer_id;
  task.desc = UnpackDescriptorBlock{};
  // save desc to a deque
  while (jring_mp_enqueue_bulk(unpacker_desc_ring_, task, 1, nullptr) !=
         1) {
  }
  {
    std::unique_lock<std::shared_mutex> lock(recv_transfer_status_mu_);
    recv_transfer_status_[transfer_id] = false;
  }
  return true;
}

bool Endpoint::poll_async(uint64_t transfer_id, bool* is_done) {
  *is_done = false;
  int done;
  // wait recv/send task finished TODO: danyang
  // tcpx_test(transfer_id); // transfer_id is the request
  int ret = tcpx_test((void*)transfer_id, int* done, int* size);
  if (ret == 0) {
    // if recv, also wait unpacker finished?
    if (done) {
      std::shared_lock<std::shared_mutex> lock(recv_transfer_status_mu_);
      if (recv_transfer_status_.find(transfer_id) != recv_transfer_status_.end()) {
          if (recv_transfer_status_.at(transfer_id)) { // true
              *is_done = true;
          }
      }
    }
  } else if (ret == -1) {
    return false;
  }
  return true;
}

}  // namespace tcpx