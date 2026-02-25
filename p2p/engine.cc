#include "engine.h"
#include "endpoint_wrapper.h"
#include "util/pause.h"
#include "util/util.h"
#include <arpa/inet.h>
#include <glog/logging.h>
#include <netinet/in.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <optional>
#include <sstream>
#include <thread>
#include <sys/socket.h>
#include <unistd.h>

thread_local bool inside_python = false;

std::size_t VectorUint8Hash::operator()(std::vector<uint8_t> const& vec) const {
  std::size_t hash = vec.size();
  for (uint8_t byte : vec) {
    hash = hash * 31 + static_cast<std::size_t>(byte);
  }
  return hash;
}

ShmMsg::ShmMsg() : src_gpu(0), type(ShmMsgType::COMPLETION), completion(0) {}

// TaskBatch (declaration order: after ShmMsg, before Endpoint)
TaskBatch::TaskBatch() : num_iovs(0) {}

TaskBatch::TaskBatch(TaskBatch&& other) noexcept
    : num_iovs(other.num_iovs),
      const_data_ptr(std::move(other.const_data_ptr)),
      data_ptr(std::move(other.data_ptr)),
      size_ptr(std::move(other.size_ptr)),
      mr_id_ptr(std::move(other.mr_id_ptr)),
      slot_item_ptr(std::move(other.slot_item_ptr)) {}

TaskBatch& TaskBatch::operator=(TaskBatch&& other) noexcept {
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

void const** TaskBatch::const_data_v() const {
  if (!const_data_ptr) return nullptr;
  return const_data_ptr->data();
}
void** TaskBatch::data_v() const {
  if (!data_ptr) return nullptr;
  return data_ptr->data();
}
size_t* TaskBatch::size_v() const {
  if (!size_ptr) return nullptr;
  return size_ptr->data();
}
uint64_t* TaskBatch::mr_id_v() const {
  if (!mr_id_ptr) return nullptr;
  return mr_id_ptr->data();
}
FifoItem* TaskBatch::slot_item_v() const {
  if (!slot_item_ptr) return nullptr;
  return slot_item_ptr->data();
}

// UnifiedTask (declaration order: after TaskBatch, before Endpoint)
UnifiedTask::UnifiedTask()
    : type(TaskType::SEND_NET),
      data(nullptr),
      size(0),
      conn_id(0),
      mr_id(0),
      status_ptr(nullptr),
      specific() {}

UnifiedTask::~UnifiedTask() {
  if (is_batch_task()) {
    specific.batch.task_batch.~TaskBatch();
  }
}

FifoItem& UnifiedTask::slot_item() { return specific.net.slot_item; }

FifoItem const& UnifiedTask::slot_item() const {
  return specific.net.slot_item;
}

IpcTransferInfo& UnifiedTask::ipc_info() { return specific.ipc.ipc_info; }

IpcTransferInfo const& UnifiedTask::ipc_info() const {
  return specific.ipc.ipc_info;
}

TaskBatch& UnifiedTask::task_batch() { return specific.batch.task_batch; }

TaskBatch const& UnifiedTask::task_batch() const {
  return specific.batch.task_batch;
}

bool UnifiedTask::is_batch_task() const {
  return type == TaskType::SENDV || type == TaskType::RECVV ||
         type == TaskType::WRITEV || type == TaskType::READV;
}

UnifiedTask::SpecificData::SpecificData() : base{} {}

UnifiedTask::SpecificData::~SpecificData() {}

static inline void check_python_signals() {
  PyGILState_STATE gstate = PyGILState_Ensure();
  if (PyErr_CheckSignals() != 0) {
    std::cerr << "Python signal caught, exiting..." << std::endl;
    std::abort();
  }
  PyGILState_Release(gstate);
}

// ShmChannel helper function
static inline std::string shm_ring_name(int from_idx, int to_idx) {
  return "/uccl_gpu_jring_" + std::to_string(from_idx) + "_" +
         std::to_string(to_idx);
}

static inline void shm_ring_send(jring_t* ring, ShmMsg const& msg) {
  // jring bulk copy requires 16B aligned buffer
  alignas(16) ShmMsg tmp = msg;

  while (jring_mp_enqueue_bulk(ring, (void*)&tmp, 1, nullptr) != 1) {
    machnet_pause();
  }
}

static inline void shm_ring_recv(jring_t* ring, ShmMsg& msg) {
  alignas(16) ShmMsg tmp;

  while (jring_sc_dequeue_bulk(ring, (void*)&tmp, 1, nullptr) != 1) {
    machnet_pause();
  }
  msg = tmp;
}

std::once_flag Endpoint::glog_init_once;

int Endpoint::parse_log_level_from_env() {
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

// -----------------------------------------------------------------------------
// The main P2P Endpoint class implementation
// -----------------------------------------------------------------------------

Endpoint::Endpoint(uint32_t const local_gpu_idx, uint32_t const num_cpus)
    : local_gpu_idx_(local_gpu_idx),
      num_cpus_(num_cpus),
      passive_accept_(false) {
  std::cout << "Creating Engine with GPU index: " << local_gpu_idx
            << ", CPUs: " << num_cpus << std::endl;
  int n_streams = std::max(1, (int)kNumGpuRtStreams);

  int ngpus = 0;
  GPU_RT_CHECK(gpuGetDeviceCount(&ngpus));
  ipc_streams_.resize(ngpus);
  for (int i = 0; i < ngpus; ++i) {
    GPU_RT_CHECK(gpuSetDevice(i));
    ipc_streams_[i].resize(n_streams);
    for (int j = 0; j < n_streams; ++j) {
      GPU_RT_CHECK(
          gpuStreamCreateWithFlags(&ipc_streams_[i][j], gpuStreamNonBlocking));
    }
  }
  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

  std::call_once(Endpoint::glog_init_once,
                 []() { google::InitGoogleLogging("uccl_p2p"); });
  FLAGS_minloglevel = Endpoint::parse_log_level_from_env();
  FLAGS_logtostderr = true;
  google::InstallFailureSignalHandler();

#ifdef UCCL_P2P_USE_NCCL
  ep_ = std::make_shared<tcp::TCPEndpoint>(local_gpu_idx_, 0);
  numa_node_ = tcp::get_tcp_numa_node_from_iface();
#else
  // Initialize the RDMA endpoint.
  ep_ = std::shared_ptr<NICEndpoint>(
      new NICEndpoint(local_gpu_idx_, INVALID_RANK_ID, 0, false));
#endif

  std::cout << "Engine initialized for GPU " << local_gpu_idx_ << std::endl;
  engine_initialized_ = true;

  send_unified_task_ring_ =
      uccl::create_ring(sizeof(UnifiedTask*), kTaskRingSize);
  recv_unified_task_ring_ =
      uccl::create_ring(sizeof(UnifiedTask*), kTaskRingSize);

#ifndef UCCL_P2P_USE_NCCL
  numa_node_ = RdmaDeviceManager::instance().get_numa_node(local_gpu_idx_);
#endif

  send_proxy_thread_ = std::thread(&Endpoint::send_proxy_thread_func, this);
  recv_proxy_thread_ = std::thread(&Endpoint::recv_proxy_thread_func, this);

  ipc_inflight_ring_ = uccl::create_ring(sizeof(IpcInflightOp*), kTaskRingSize);
  ipc_poller_thread_ = std::thread(&Endpoint::ipc_poller_thread_func, this);

  // Initialize ShmChnnel for local connections
  size_t elem_sz = sizeof(ShmMsg);
  size_t elem_cnt = ShmRingDefaultElemCnt;
  for (int i = 0; i < kMaxNumGPUs; i++) {
    inbox_rings_[i].shm_name = shm_ring_name(i, local_gpu_idx_);
    inbox_rings_[i].ring = uccl::create_shared_ring(
        inbox_rings_[i].shm_name.c_str(), elem_sz, elem_cnt,
        inbox_rings_[i].shm_fd, inbox_rings_[i].shm_size, &inbox_creators_[i]);
  }

  std::cout << "Endpoint initialized successfully" << std::endl;
}

Endpoint::Endpoint(uint32_t const num_cpus)
    : local_gpu_idx_(INVALID_GPU), num_cpus_(num_cpus), passive_accept_(false) {
  std::cout << "Creating Engine with CPUs: " << num_cpus << std::endl;
  int n_streams = std::max(1, (int)kNumGpuRtStreams);

  int ngpus = 0;
  GPU_RT_CHECK(gpuGetDeviceCount(&ngpus));
  ipc_streams_.resize(ngpus);
  for (int i = 0; i < ngpus; ++i) {
    GPU_RT_CHECK(gpuSetDevice(i));
    ipc_streams_[i].resize(n_streams);
    for (int j = 0; j < n_streams; ++j) {
      GPU_RT_CHECK(
          gpuStreamCreateWithFlags(&ipc_streams_[i][j], gpuStreamNonBlocking));
    }
  }

  std::call_once(glog_init_once,
                 []() { google::InitGoogleLogging("uccl_p2p"); });

  google::InstallFailureSignalHandler();
#ifdef UCCL_P2P_USE_NCCL
  ep_ = std::make_shared<tcp::TCPEndpoint>(local_gpu_idx_, 0);
#else
  // Initialize the RDMA endpoint with lazy creation.
  ep_ = std::shared_ptr<NICEndpoint>(
      new NICEndpoint(INVALID_GPU, INVALID_RANK_ID, 0, false));
#endif

  std::cout << "Endpoint initialized successfully" << std::endl;
}

Endpoint::~Endpoint() {
  std::cout << "Destroying Engine..." << std::endl;

  stop_.store(true, std::memory_order_release);

  if (passive_accept_) {
    passive_accept_stop_.store(true, std::memory_order_release);
    if (passive_accept_thread_.joinable()) {
      passive_accept_thread_.join();
    }
  }

  if (send_proxy_thread_.joinable()) {
    send_proxy_thread_.join();
  }
  if (recv_proxy_thread_.joinable()) {
    recv_proxy_thread_.join();
  }
  if (ipc_poller_thread_.joinable()) {
    ipc_poller_thread_.join();
  }

  if (send_unified_task_ring_ != nullptr) {
    free(send_unified_task_ring_);
  }
  if (recv_unified_task_ring_ != nullptr) {
    free(recv_unified_task_ring_);
  }
  if (ipc_inflight_ring_ != nullptr) {
    free(ipc_inflight_ring_);
  }

  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    for (auto& [conn_id, conn] : conn_id_to_conn_) {
      auto& shm_channel_ = conn->remote_inbox_;

      if (conn->shm_attached_) {
        uccl::detach_shared_ring(shm_channel_.ring, shm_channel_.shm_fd,
                                 shm_channel_.shm_size);
      }
      delete conn;
    }
  }

  for (int i = 0; i < kMaxNumGPUs; i++) {
    if (inbox_creators_[i])
      uccl::destroy_shared_ring(inbox_rings_[i].shm_name.c_str(),
                                inbox_rings_[i].ring, inbox_rings_[i].shm_fd,
                                inbox_rings_[i].shm_size);
    else
      uccl::detach_shared_ring(inbox_rings_[i].ring, inbox_rings_[i].shm_fd,
                               inbox_rings_[i].shm_size);
  }

  {
    std::shared_lock<std::shared_mutex> lock(mr_mu_);
    for (auto& [mr_id, mr] : mr_id_to_mr_) {
      delete mr;
    }
  }

  std::cout << "Engine destroyed" << std::endl;
}

bool Endpoint::connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
                       uint64_t& conn_id) {
  std::cout << "Attempting to connect to " << ip_addr << ":" << remote_gpu_idx
            << " via port " << remote_port << std::endl;
  // Create a new connection ID
  conn_id = next_conn_id_.fetch_add(1);

  assert(local_gpu_idx_ != INVALID_GPU);

  std::future<ConnID> uccl_conn_id_future = std::async(
      std::launch::async, [this, remote_gpu_idx, &ip_addr, remote_port]() {
        return uccl_connect(ep_, remote_gpu_idx, ip_addr, remote_port);
      });

  // Check for Python signals (eg, ctrl+c) while waiting for connection
  while (uccl_conn_id_future.wait_for(std::chrono::seconds(0)) !=
         std::future_status::ready) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  ConnID uccl_conn_id = uccl_conn_id_future.get();

  // Store the connection ID.
  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    conn_id_to_conn_[conn_id] =
        new Conn{conn_id, uccl_conn_id, ip_addr, remote_gpu_idx};
  }
  return true;
}

std::vector<uint8_t> Endpoint::get_metadata() {
  std::string ip_str = uccl::get_oob_ip();
  uint16_t port = get_p2p_listen_port(ep_);

  bool is_ipv6 = ip_str.find(':') != std::string::npos;
  size_t ip_len = is_ipv6 ? 16 : 4;

  // Additional 2 bytes for port and 4 bytes for local_gpu_idx_
  size_t total_len = ip_len + 2 + sizeof(int);
  std::vector<uint8_t> metadata(total_len);

  // Copy IP
  if (is_ipv6) {
    struct in6_addr ip6_bin;
    if (inet_pton(AF_INET6, ip_str.c_str(), &ip6_bin) != 1)
      throw std::runtime_error("Invalid IPv6 address: " + ip_str);
    std::memcpy(metadata.data(), &ip6_bin, 16);
  } else {
    struct in_addr ip4_bin;
    if (inet_pton(AF_INET, ip_str.c_str(), &ip4_bin) != 1)
      throw std::runtime_error("Invalid IPv4 address: " + ip_str);
    std::memcpy(metadata.data(), &ip4_bin, 4);
  }

  // Copy port in network byte order
  uint16_t net_port = htons(port);
  std::memcpy(metadata.data() + ip_len, &net_port, 2);

  // Copy local_gpu_idx_ in host byte order
  std::memcpy(metadata.data() + ip_len + 2, &local_gpu_idx_, sizeof(int));

  return metadata;
}

std::vector<uint8_t> Endpoint::get_unified_metadata() {
  int idx = 0;
  std::string ip_str = uccl::get_oob_ip();
  uint16_t port = get_p2p_listen_port(ep_);

  bool is_ipv6 = ip_str.find(':') != std::string::npos;
  size_t ip_len = is_ipv6 ? 16 : 4;

  // Additional 2 bytes for port and 4 bytes for local_gpu_idx_
  size_t total_len = ip_len + 2 + sizeof(int);
  std::vector<uint8_t> metadata(total_len);

  // Copy IP
  if (is_ipv6) {
    struct in6_addr ip6_bin;
    if (inet_pton(AF_INET6, ip_str.c_str(), &ip6_bin) != 1)
      throw std::runtime_error("Invalid IPv6 address: " + ip_str);
    std::memcpy(metadata.data(), &ip6_bin, 16);
  } else {
    struct in_addr ip4_bin;
    if (inet_pton(AF_INET, ip_str.c_str(), &ip4_bin) != 1)
      throw std::runtime_error("Invalid IPv4 address: " + ip_str);
    std::memcpy(metadata.data(), &ip4_bin, 4);
  }

  // Copy port in network byte order
  uint16_t net_port = htons(port);
  std::memcpy(metadata.data() + ip_len, &net_port, 2);

  // Copy local_gpu_idx_ in host byte order
  std::memcpy(metadata.data() + ip_len + 2, &idx, sizeof(int));

  return metadata;
}
std::tuple<std::string, uint16_t, int> Endpoint::parse_metadata(
    std::vector<uint8_t> const& metadata) {
  if (metadata.size() == 10) {
    // IPv4: 4 bytes IP, 2 bytes port, 4 bytes GPU idx
    char ip_str[INET_ADDRSTRLEN];
    if (inet_ntop(AF_INET, metadata.data(), ip_str, sizeof(ip_str)) ==
        nullptr) {
      throw std::runtime_error("Failed to parse IPv4 address from metadata");
    }

    uint16_t net_port;
    std::memcpy(&net_port, metadata.data() + 4, 2);
    uint16_t port = ntohs(net_port);

    int gpu_idx;
    std::memcpy(&gpu_idx, metadata.data() + 6, 4);

    return std::make_tuple(std::string(ip_str), port, gpu_idx);
  } else if (metadata.size() == 22) {
    // IPv6: 16 bytes IP, 2 bytes port, 4 bytes GPU idx
    char ip_str[INET6_ADDRSTRLEN];
    if (inet_ntop(AF_INET6, metadata.data(), ip_str, sizeof(ip_str)) ==
        nullptr) {
      throw std::runtime_error("Failed to parse IPv6 address from metadata");
    }

    uint16_t net_port;
    std::memcpy(&net_port, metadata.data() + 16, 2);
    uint16_t port = ntohs(net_port);

    int gpu_idx;
    std::memcpy(&gpu_idx, metadata.data() + 18, 4);

    return std::make_tuple(std::string(ip_str), port, gpu_idx);
  } else {
    throw std::runtime_error("Unexpected metadata length: " +
                             std::to_string(metadata.size()));
  }
}

bool Endpoint::accept(std::string& ip_addr, int& remote_gpu_idx,
                      uint64_t& conn_id) {
  std::cout << "Waiting to accept incoming connection..." << std::endl;

  // For demo purposes, simulate accepted connection
  conn_id = next_conn_id_.fetch_add(1);

  // Wait until engine is intialized to get the correct local_gpu_idx_
  while (!engine_initialized_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::future<ConnID> uccl_conn_id_future =
      std::async(std::launch::async, [this, &ip_addr, &remote_gpu_idx]() {
        return uccl_accept(ep_, ip_addr, &remote_gpu_idx);
      });

  // Check for Python signals (eg, ctrl+c) while waiting for connection
  while (uccl_conn_id_future.wait_for(std::chrono::seconds(0)) !=
         std::future_status::ready) {
    if (passive_accept_ &&
        passive_accept_stop_.load(std::memory_order_acquire)) {
      std::cout << "Stop background accept..." << std::endl;
      stop_accept(ep_);
    }
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  ConnID uccl_conn_id = uccl_conn_id_future.get();

  // Return if Conn ID is invalid
  if (uccl_conn_id.sock_fd < 0 || uccl_conn_id.flow_id == UINT64_MAX) {
    return false;
  }

  // Store the connection ID.
  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    conn_id_to_conn_[conn_id] =
        new Conn{conn_id, uccl_conn_id, ip_addr, remote_gpu_idx};
  }

  return true;
}

bool Endpoint::reg(void const* data, size_t size, uint64_t& mr_id) {
  mr_id = next_mr_id_.fetch_add(1);

  if (!engine_initialized_) {
    int idx = uccl::get_dev_idx((void*)data);
    if (idx != -1) {
      // Pointer is on device idx
      local_gpu_idx_ = idx;
    } else {
      // Host memory/unknown memory type - fallback to dev 0
      local_gpu_idx_ = 0;
    }
    initialize_engine();
    engine_initialized_ = true;
  }

  P2PMhandle* mhandle = new P2PMhandle();
  if (!uccl_regmr(ep_, const_cast<void*>(data), size, mhandle)) {
    return false;
  }
  {
    std::unique_lock<std::shared_mutex> lock(mr_mu_);
    mr_id_to_mr_[mr_id] = new MR{mr_id, mhandle};
  }

  return true;
}

bool Endpoint::regv(std::vector<void const*> const& data_v,
                    std::vector<size_t> const& size_v,
                    std::vector<uint64_t>& mr_id_v) {
  if (data_v.size() != size_v.size())
    throw std::invalid_argument(
        "[Endpoint::regv] data_v/size_v length mismatch");

  size_t const n = data_v.size();

  // Early return if empty
  if (n == 0) {
    mr_id_v.clear();
    return true;
  }

  if (!engine_initialized_) {
    int idx = uccl::get_dev_idx((void*)data_v[0]);
    if (idx != -1) {
      // Pointer is on device idx
      local_gpu_idx_ = idx;
    } else {
      // Host memory/unknown memory type - fallback to dev 0
      local_gpu_idx_ = 0;
    }
    initialize_engine();
    engine_initialized_ = true;
  }

  mr_id_v.resize(n);

  {
    std::unique_lock<std::shared_mutex> lock(mr_mu_);
    mr_id_to_mr_.reserve(mr_id_to_mr_.size() + n);
  }

  for (size_t i = 0; i < n; ++i) {
    uint64_t id = next_mr_id_.fetch_add(1);
    P2PMhandle* mhandle = new P2PMhandle();

    if (!uccl_regmr(ep_, const_cast<void*>(data_v[i]), size_v[i], mhandle)) {
      std::cerr << "[Endpoint::regv] registration failed at i=" << i << '\n';
      return false;
    }

    {
      std::unique_lock<std::shared_mutex> lock(mr_mu_);
      mr_id_to_mr_[id] = new MR{id, mhandle};
    }
    mr_id_v[i] = id;
  }
  return true;
}

bool Endpoint::dereg(uint64_t mr_id) {
  {
    std::unique_lock<std::shared_mutex> lock(mr_mu_);
    auto it = mr_id_to_mr_.find(mr_id);
    if (it == mr_id_to_mr_.end()) {
      std::cerr << "[dereg] Error: Invalid mr_id " << mr_id << std::endl;
      return false;
    }
    auto mr = it->second;
    uccl_deregmr(ep_, mr->mhandle_);
    delete mr;
    mr_id_to_mr_.erase(mr_id);
  }
  return true;
}

bool Endpoint::send(uint64_t conn_id, uint64_t mr_id, void const* data,
                    size_t size) {
  DCHECK(size <= 0xffffffff) << "size must be less than 4GB";

  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[send] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  P2PMhandle* mhandle = get_mhandle(mr_id);
  if (unlikely(mhandle == nullptr)) {
    std::cerr << "[send] Error: Invalid mr_id " << mr_id << std::endl;
    return false;
  }

  ucclRequest ureq;

  auto* cur_data = const_cast<void*>(data);
  auto to_send = size;
  while (uccl_send_async(ep_, conn, mhandle, cur_data, to_send, &ureq) == -1)
    ;
  while (!uccl_poll_ureq_once(ep_, &ureq)) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
  }

  return true;
}

bool Endpoint::recv(uint64_t conn_id, uint64_t mr_id, void* data, size_t size) {
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[recv] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  P2PMhandle* mhandle = get_mhandle(mr_id);
  if (unlikely(mhandle == nullptr)) {
    std::cerr << "[recv] Error: Invalid mr_id " << mr_id << std::endl;
    return false;
  }
  int size_int = static_cast<int>(size);

  ucclRequest ureq;

  void* cur_data = data;
  while (uccl_recv_async(ep_, conn, mhandle, &cur_data, &size_int, 1, &ureq) ==
         -1)
    ;
  while (!uccl_poll_ureq_once(ep_, &ureq)) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
  }

  return true;
}

bool Endpoint::send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                          size_t size, uint64_t* transfer_id) {
  auto task_ptr = create_task(conn_id, mr_id, TaskType::SEND_NET,
                              const_cast<void*>(data), size);
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();
  while (jring_mp_enqueue_bulk(send_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::recv_async(uint64_t conn_id, uint64_t mr_id, void* data,
                          size_t size, uint64_t* transfer_id) {
  auto task_ptr = create_task(conn_id, mr_id, TaskType::RECV_NET, data, size);
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();
  while (jring_mp_enqueue_bulk(recv_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::sendv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                     std::vector<void const*> data_v,
                     std::vector<size_t> size_v, size_t num_iovs) {
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[sendv] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  std::vector<ucclRequest> ureq(num_iovs);
  std::vector<bool> sent(num_iovs, false);
  std::vector<bool> done(num_iovs, false);
  std::vector<P2PMhandle*> mhandles(num_iovs);

  // Check if mhandles are all valid
  for (int i = 0; i < num_iovs; i++) {
    mhandles[i] = get_mhandle(mr_id_v[i]);
    if (unlikely(mhandles[i] == nullptr)) {
      std::cerr << "[sendv] Error: Invalid mr_id " << mr_id_v[i] << std::endl;
      return false;
    }
  }

  while (1) {
    for (size_t i = 0; i < num_iovs; i++) {
      if (done[i]) continue;
      if (!sent[i]) {
        void* cur_data = (void*)data_v[i];
        size_t cur_size = size_v[i];

        auto mhandle = mhandles[i];

        auto rc =
            uccl_send_async(ep_, conn, mhandle, cur_data, cur_size, &ureq[i]);
        if (rc != -1) {
          sent[i] = true;
        }
      }

      if (sent[i] && !done[i]) {
        if (uccl_poll_ureq_once(ep_, &ureq[i])) {
          done[i] = true;
        }
      }
    }

    if (std::all_of(done.begin(), done.end(), [](bool b) { return b; })) {
      break;
    }

    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
  }

  return true;
}

bool Endpoint::sendv_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                           std::vector<void const*> data_v,
                           std::vector<size_t> size_v, size_t num_iovs,
                           uint64_t* transfer_id) {
  auto const_data_ptr =
      std::make_shared<std::vector<void const*>>(std::move(data_v));
  auto size_ptr = std::make_shared<std::vector<size_t>>(std::move(size_v));
  auto mr_id_ptr = std::make_shared<std::vector<uint64_t>>(std::move(mr_id_v));

  auto task_ptr = create_sendv_task(conn_id, std::move(const_data_ptr),
                                    std::move(size_ptr), std::move(mr_id_ptr));
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  while (jring_mp_enqueue_bulk(send_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::recvv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                     std::vector<void*> data_v, std::vector<size_t> size_v,
                     size_t num_iovs) {
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[recvv] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  std::vector<ucclRequest> ureq(num_iovs);
  std::vector<bool> done(num_iovs, false);
  std::vector<bool> received(num_iovs, false);
  std::vector<P2PMhandle*> mhandles(num_iovs);

  // Check if mhandles are all valid
  for (int i = 0; i < num_iovs; i++) {
    mhandles[i] = get_mhandle(mr_id_v[i]);
    if (unlikely(mhandles[i] == nullptr)) {
      std::cerr << "[recvv] Error: Invalid mr_id " << mr_id_v[i] << std::endl;
      return false;
    }
  }
  while (1) {
    for (size_t i = 0; i < num_iovs; i++) {
      if (done[i]) continue;

      if (!received[i]) {
        void* cur_data = data_v[i];
        size_t cur_size = size_v[i];

        auto mhandle = mhandles[i];

        int size_int = static_cast<int>(cur_size);

        auto rc = uccl_recv_async(ep_, conn, mhandle, &cur_data, &size_int, 1,
                                  &ureq[i]);
        if (rc != -1) {
          received[i] = true;
        }
      }

      if (received[i] && !done[i]) {
        if (uccl_poll_ureq_once(ep_, &ureq[i])) {
          done[i] = true;
        }
      }
    }

    if (std::all_of(done.begin(), done.end(), [](bool b) { return b; })) {
      break;
    }

    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
  }

  return true;
}

bool Endpoint::recvv_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                           std::vector<void*> data_v,
                           std::vector<size_t> size_v, size_t num_iovs,
                           uint64_t* transfer_id) {
  // Use move semantics to reduce memory copies
  auto task_ptr = create_recvv_task(conn_id, std::move(data_v),
                                    std::move(size_v), std::move(mr_id_v));
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  while (jring_mp_enqueue_bulk(recv_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::read(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
                    FifoItem const& slot_item) {
  DCHECK(size <= 0xffffffff) << "size must be < 4 GB";
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[read] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  P2PMhandle* mhandle = get_mhandle(mr_id);
  if (unlikely(mhandle == nullptr)) {
    std::cerr << "[read] Error: Invalid mr_id " << mr_id << std::endl;
    return false;
  }

  ucclRequest ureq = {};
  FifoItem curr_slot_item = slot_item;
  curr_slot_item.size = size;

  while (uccl_read_async(ep_, conn, mhandle, dst, size, curr_slot_item,
                         &ureq) == -1)
    ;

  bool done = false;
  while (!done) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
    if (uccl_poll_ureq_once(ep_, &ureq)) {
      done = true;
    }
  }

  return true;
}

bool Endpoint::read_async(uint64_t conn_id, uint64_t mr_id, void* dst,
                          size_t size, FifoItem const& slot_item,
                          uint64_t* transfer_id) {
  auto task_ptr =
      create_net_task(conn_id, mr_id, TaskType::READ_NET, dst, size, slot_item);
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  while (jring_mp_enqueue_bulk(recv_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::readv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                     std::vector<void*> dst_v, std::vector<size_t> size_v,
                     std::vector<FifoItem> slot_item_v, size_t num_iovs) {
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[readv] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  ucclRequest ureq[kMaxInflightOps] = {};
  bool done[kMaxInflightOps] = {false};

  size_t iov_issued = 0, iov_finished = 0;

  while (iov_finished < num_iovs) {
    // Issue up to kMaxInflightOps IOVs
    while (iov_issued < num_iovs &&
           iov_issued - iov_finished < kMaxInflightOps) {
      P2PMhandle* mhandle = get_mhandle(mr_id_v[iov_issued]);
      if (unlikely(mhandle == nullptr)) {
        std::cerr << "[readv] Error: Invalid mr_id " << mr_id_v[iov_issued]
                  << std::endl;
        return false;
      }

      auto rc = uccl_read_async(ep_, conn, mhandle, dst_v[iov_issued],
                                size_v[iov_issued], slot_item_v[iov_issued],
                                &ureq[iov_issued % kMaxInflightOps]);
      if (rc == -1) break;
      done[iov_issued % kMaxInflightOps] = false;
      iov_issued++;
    }

    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;

    for (size_t i = iov_finished; i < iov_issued; i++) {
      if (done[i % kMaxInflightOps]) continue;
      if (uccl_poll_ureq_once(ep_, &ureq[i % kMaxInflightOps])) {
        done[i % kMaxInflightOps] = true;
      }
    }

    while (iov_finished < iov_issued && done[iov_finished % kMaxInflightOps]) {
      iov_finished++;
    }
  }

  return true;
}

bool Endpoint::readv_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                           std::vector<void*> dst_v, std::vector<size_t> size_v,
                           std::vector<FifoItem> slot_item_v, size_t num_iovs,
                           uint64_t* transfer_id) {
  // Use move semantics to reduce memory copies
  auto task_ptr =
      create_readv_task(conn_id, std::move(dst_v), std::move(size_v),
                        std::move(mr_id_v), std::move(slot_item_v));
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  while (jring_mp_enqueue_bulk(recv_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::write(uint64_t conn_id, uint64_t mr_id, void* src, size_t size,
                     FifoItem const& slot_item) {
  DCHECK(size <= 0xffffffff) << "size must be < 4 GB";
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[write] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }
  P2PMhandle* mhandle = get_mhandle(mr_id);
  if (unlikely(mhandle == nullptr)) {
    std::cerr << "[write] Error: Invalid mr_id " << mr_id << std::endl;
    return false;
  }
  ucclRequest ureq = {};
  FifoItem curr_slot_item = slot_item;
  curr_slot_item.size = size;

  while (uccl_write_async(ep_, conn, mhandle, src, size, curr_slot_item,
                          &ureq) == -1)
    ;

  bool done = false;
  while (!done) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
    if (uccl_poll_ureq_once(ep_, &ureq)) {
      done = true;
    }
  }

  return true;
}

bool Endpoint::write_async(uint64_t conn_id, uint64_t mr_id, void* src,
                           size_t size, FifoItem const& slot_item,
                           uint64_t* transfer_id) {
  auto task_ptr = create_net_task(conn_id, mr_id, TaskType::WRITE_NET, src,
                                  size, slot_item);
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  while (jring_mp_enqueue_bulk(send_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::writev(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                      std::vector<void*> src_v, std::vector<size_t> size_v,
                      std::vector<FifoItem> slot_item_v, size_t num_iovs) {
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[writev] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  ucclRequest ureq[kMaxInflightOps] = {};
  bool done[kMaxInflightOps] = {false};

  size_t iov_issued = 0, iov_finished = 0;

  while (iov_finished < num_iovs) {
    while (iov_issued < num_iovs &&
           iov_issued - iov_finished < kMaxInflightOps) {
      P2PMhandle* mhandle = get_mhandle(mr_id_v[iov_issued]);
      if (unlikely(mhandle == nullptr)) {
        std::cerr << "[writev] Error: Invalid mr_id " << mr_id_v[iov_issued]
                  << std::endl;
        return false;
      }

      auto rc = uccl_write_async(ep_, conn, mhandle, src_v[iov_issued],
                                 size_v[iov_issued], slot_item_v[iov_issued],
                                 &ureq[iov_issued % kMaxInflightOps]);
      if (rc == -1) break;
      done[iov_issued % kMaxInflightOps] = false;
      iov_issued++;
    }

    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;

    for (size_t i = iov_finished; i < iov_issued; i++) {
      if (done[i % kMaxInflightOps]) continue;
      if (uccl_poll_ureq_once(ep_, &ureq[i % kMaxInflightOps])) {
        done[i % kMaxInflightOps] = true;
      }
    }

    while (iov_finished < iov_issued && done[iov_finished % kMaxInflightOps]) {
      iov_finished++;
    }
  }

  return true;
}

bool Endpoint::writev_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                            std::vector<void*> src_v,
                            std::vector<size_t> size_v,
                            std::vector<FifoItem> slot_item_v, size_t num_iovs,
                            uint64_t* transfer_id) {
  // Use move semantics to reduce memory copies
  auto task_ptr =
      create_writev_task(conn_id, std::move(src_v), std::move(size_v),
                         std::move(mr_id_v), std::move(slot_item_v));
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  while (jring_mp_enqueue_bulk(send_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::advertise(uint64_t conn_id, uint64_t mr_id, void* addr,
                         size_t len, char* out_buf) {
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[advertise] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }
  auto mhandle = get_mhandle(mr_id);
  if (unlikely(mhandle == nullptr)) {
    std::cerr << "[advertise] Error: Invalid mr_id " << mr_id << std::endl;
    return false;
  }
  if (prepare_fifo_metadata(ep_, conn, mhandle, addr, len, out_buf) == -1)
    return false;
  return true;
}

bool Endpoint::prepare_fifo(uint64_t mr_id, void* addr, size_t len,
                            char* out_buf) {
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;
  // prepare_fifo_metadata doesn't actually use the endpoint or conn parameters
  if (prepare_fifo_metadata(ep_, nullptr, mhandle, addr, len, out_buf) == -1)
    return false;
  return true;
}

bool Endpoint::advertisev(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                          std::vector<void*> addr_v, std::vector<size_t> len_v,
                          std::vector<char*> out_buf_v, size_t num_iovs) {
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[advertisev] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  std::vector<P2PMhandle*> mhandles(num_iovs);

  // Check if mhandles are all valid
  for (int i = 0; i < num_iovs; i++) {
    mhandles[i] = get_mhandle(mr_id_v[i]);
    if (unlikely(mhandles[i] == nullptr)) {
      std::cerr << "[advertisev] Error: Invalid mr_id " << mr_id_v[i]
                << std::endl;
      return false;
    }
  }

  for (size_t i = 0; i < num_iovs; ++i) {
    auto mhandle = mhandles[i];
    if (prepare_fifo_metadata(ep_, conn, mhandle, addr_v[i], len_v[i],
                              out_buf_v[i]) == -1) {
      return false;
    }
  }
  return true;
}

bool Endpoint::connect_local(int remote_gpu_idx, uint64_t& conn_id) {
  constexpr int kMaxRetry = 20;
  constexpr int kRetrySleepMs = 10;

  std::cout << "Connecting to GPU " << remote_gpu_idx << std::endl;

  Conn* conn = new Conn;
  conn->remote_gpu_idx_ = remote_gpu_idx;

  if (conn->remote_gpu_idx_ == local_gpu_idx_) {
    // same GPU: no attach, bind inbox_rings_[remote_gpu_idx] directly
    conn->remote_inbox_ = inbox_rings_[remote_gpu_idx];
    conn->shm_attached_ = false;
  } else {
    // cross GPU: attach remote inbox
    conn->remote_inbox_.shm_name =
        shm_ring_name(local_gpu_idx_, remote_gpu_idx);
    conn->remote_inbox_.shm_size = inbox_rings_[remote_gpu_idx].shm_size;

    jring_t* ring = nullptr;
    for (int attempt = 0; attempt < kMaxRetry; attempt++) {
      ring = uccl::attach_shared_ring(conn->remote_inbox_.shm_name.c_str(),
                                      conn->remote_inbox_.shm_fd,
                                      conn->remote_inbox_.shm_size);
      if (ring != nullptr) {
        break;
      }
      // retry
      std::this_thread::sleep_for(std::chrono::milliseconds(kRetrySleepMs));
    }

    if (ring == nullptr) {
      std::cout << "connect_local failed: cannot attach remote inbox ring "
                << conn->remote_inbox_.shm_name << " after " << kMaxRetry
                << " retries" << std::endl;
      delete conn;
      return false;
    }

    conn->remote_inbox_.ring = ring;
    conn->shm_attached_ = true;
  }

  // allocate conn_id
  conn_id = next_conn_id_.fetch_add(1);
  conn->conn_id_ = conn_id;

  ShmMsg msg;
  msg.src_gpu = local_gpu_idx_;
  msg.type = ShmMsgType::CONNECT;
  msg.completion = 0;
  shm_ring_send(conn->remote_inbox_.ring, msg);

  {
    std::unique_lock lock(conn_mu_);
    ConnID dummy{nullptr, 0, 0, 0};
    conn->uccl_conn_id_ = dummy;
    conn_id_to_conn_[conn_id] = conn;
  }

  return true;
}

bool Endpoint::accept_local(int& remote_gpu_idx, uint64_t& conn_id) {
  std::cout << "Waiting for local CONNECT..." << std::endl;

  // recv CONNECT
  ShmMsg msg;
  while (true) {
    for (int peer = 0; peer < kMaxNumGPUs; peer++) {
      if (peer == local_gpu_idx_) continue;

      if (!jring_empty(inbox_rings_[peer].ring)) {
        shm_ring_recv(inbox_rings_[peer].ring, msg);
        remote_gpu_idx = peer;
        goto got;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
got:
  CHECK(msg.type == ShmMsgType::CONNECT);

  remote_gpu_idx = msg.src_gpu;
  conn_id = next_conn_id_.fetch_add(1);

  Conn* conn = new Conn;
  conn->conn_id_ = conn_id;
  conn->remote_gpu_idx_ = remote_gpu_idx;

  if (conn->remote_gpu_idx_ == local_gpu_idx_) {
    // same GPU: no attach, bind inbox_rings_[remote_gpu_idx] directly
    conn->remote_inbox_ = inbox_rings_[conn->remote_gpu_idx_];
    conn->shm_attached_ = false;
  } else {
    // cross GPU: attach client inbox
    conn->remote_inbox_.shm_name =
        shm_ring_name(local_gpu_idx_, conn->remote_gpu_idx_);
    conn->remote_inbox_.shm_size = inbox_rings_[conn->remote_gpu_idx_].shm_size;
    conn->remote_inbox_.ring = uccl::attach_shared_ring(
        conn->remote_inbox_.shm_name.c_str(), conn->remote_inbox_.shm_fd,
        conn->remote_inbox_.shm_size);
    conn->shm_attached_ = true;
  }

  {
    std::unique_lock lock(conn_mu_);
    ConnID dummy{nullptr, 0, 0, 0};
    conn->uccl_conn_id_ = dummy;
    conn_id_to_conn_[conn_id] = conn;
  }

  return true;
}

bool Endpoint::send_ipc(uint64_t conn_id, void* data, size_t size) {
  CHECK(data != nullptr) << "send_ipc: data pointer is null!";

  // Get connection info
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[send_ipc] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  bool sender_is_host = (uccl::get_dev_idx(data) == -1);

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  // Receive receiver's IPC_HANDLE message
  ShmMsg msg;
  shm_ring_recv(inbox_rings_[conn->remote_gpu_idx_].ring, msg);
  CHECK(msg.type == ShmMsgType::IPC_HANDLE);
  auto info = msg.info;

  if (!info.is_host) {
    // GPU receiver: open remote handle and copy into it
    void* base = nullptr;
    GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));
    GPU_RT_CHECK(
        gpuIpcOpenMemHandle(&base, info.handle, gpuIpcMemLazyEnablePeerAccess));
    void* dst_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) +
                                            info.offset);

    auto copy_kind =
        sender_is_host ? gpuMemcpyHostToDevice : gpuMemcpyDeviceToDevice;

    std::vector<gpuStream_t>& dst_streams = ipc_streams_[conn->remote_gpu_idx_];
    int num_streams = std::min(
        dst_streams.size(),
        size < kIpcSizePerEngine ? 1 : (size_t)size / kIpcSizePerEngine);
    size_t chunk_size = size / num_streams;

    for (int i = 0; i < num_streams; ++i) {
      void* chunk_data = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(data) + i * chunk_size);
      void* chunk_dst_ptr = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(dst_ptr) + i * chunk_size);
      auto copy_size =
          i == num_streams - 1 ? size - i * chunk_size : chunk_size;
      GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst_ptr, chunk_data, copy_size,
                                  copy_kind, dst_streams[i]));
    }

    for (auto& stream : dst_streams) {
      GPU_RT_CHECK(gpuStreamSynchronize(stream));
    }

    // Notify receiver completion
    ShmMsg ack;
    ack.src_gpu = local_gpu_idx_;
    ack.type = ShmMsgType::COMPLETION;
    ack.completion = 1;
    shm_ring_send(conn->remote_inbox_.ring, ack);
  } else {
    // CPU receiver, GPU sender: role reversal - sender shares its GPU handle,
    // receiver does the copy
    CHECK(!sender_is_host)
        << "send_ipc: both sender and receiver are CPU - not supported";

    // Create IPC handle for our GPU buffer
    IpcTransferInfo sender_info = {};
    sender_info.size = size;
    sender_info.operation = 0;
    sender_info.is_host = false;
    GPU_RT_CHECK(gpuIpcGetMemHandle(&sender_info.handle, data));

    void* base = nullptr;
    size_t base_size;
    GPU_RT_CHECK(gpuMemGetAddressRange(&base, &base_size, data));
    sender_info.offset =
        reinterpret_cast<uintptr_t>(data) - reinterpret_cast<uintptr_t>(base);

    // Send our handle to the receiver
    ShmMsg handle_msg;
    handle_msg.type = ShmMsgType::IPC_HANDLE;
    handle_msg.src_gpu = local_gpu_idx_;
    handle_msg.info = sender_info;
    shm_ring_send(conn->remote_inbox_.ring, handle_msg);

    // Wait for receiver to complete the copy
    ShmMsg ack;
    shm_ring_recv(inbox_rings_[conn->remote_gpu_idx_].ring, ack);
    CHECK(ack.type == ShmMsgType::COMPLETION)
        << "Failed to receive completion notification, unexpected ack.type="
        << static_cast<uint32_t>(ack.type);
    CHECK(ack.src_gpu == conn->remote_gpu_idx_)
        << "Failed to receive completion notification, unexpected ack.src_gpu="
        << static_cast<uint32_t>(ack.src_gpu);
    CHECK_EQ(ack.completion, 1) << "Receiver reported failure";
  }

  return true;
}

bool Endpoint::recv_ipc(uint64_t conn_id, void* data, size_t size) {
  CHECK(data != nullptr) << "recv_ipc: data pointer is null!";

  // Get connection info
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[recv_ipc] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  bool is_host = (uccl::get_dev_idx(data) == -1);

  if (!is_host) {
    // GPU receiver: share our GPU buffer handle, sender copies into it
    IpcTransferInfo info = {};
    info.size = size;
    info.operation = 1;
    info.is_host = false;
    GPU_RT_CHECK(gpuIpcGetMemHandle(&info.handle, data));

    void* base = nullptr;
    size_t base_size;
    GPU_RT_CHECK(gpuMemGetAddressRange(&base, &base_size, data));
    info.offset =
        reinterpret_cast<uintptr_t>(data) - reinterpret_cast<uintptr_t>(base);

    ShmMsg msg;
    msg.type = ShmMsgType::IPC_HANDLE;
    msg.src_gpu = local_gpu_idx_;
    msg.info = info;
    shm_ring_send(conn->remote_inbox_.ring, msg);

    // Wait completion on local inbox
    ShmMsg ack;
    shm_ring_recv(inbox_rings_[conn->remote_gpu_idx_].ring, ack);
    CHECK(ack.type == ShmMsgType::COMPLETION)
        << "Failed to receive completion notification, unexpected ack.type="
        << static_cast<uint32_t>(ack.type);
    CHECK(ack.src_gpu == conn->remote_gpu_idx_)
        << "Failed to receive completion notification, unexpected ack.src_gpu="
        << static_cast<uint32_t>(ack.src_gpu);
    CHECK_EQ(ack.completion, 1) << "Sender reported failure";
  } else {
    // CPU receiver: tell sender we are host, then receive sender's GPU handle
    // and copy D2H ourselves (zero-copy role reversal)
    IpcTransferInfo info = {};
    info.size = size;
    info.operation = 1;
    info.is_host = true;

    ShmMsg msg;
    msg.type = ShmMsgType::IPC_HANDLE;
    msg.src_gpu = local_gpu_idx_;
    msg.info = info;
    shm_ring_send(conn->remote_inbox_.ring, msg);

    // Wait for sender's GPU handle
    ShmMsg sender_msg;
    shm_ring_recv(inbox_rings_[conn->remote_gpu_idx_].ring, sender_msg);
    CHECK(sender_msg.type == ShmMsgType::IPC_HANDLE);
    auto sender_info = sender_msg.info;
    CHECK(!sender_info.is_host)
        << "recv_ipc: both sender and receiver are CPU - not supported";

    int orig_device;
    GPU_RT_CHECK(gpuGetDevice(&orig_device));
    auto dev_reset =
        uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

    // Open sender's GPU buffer
    void* base = nullptr;
    GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));
    GPU_RT_CHECK(gpuIpcOpenMemHandle(&base, sender_info.handle,
                                     gpuIpcMemLazyEnablePeerAccess));
    void* src_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) +
                                            sender_info.offset);

    // Copy from sender's GPU to our CPU buffer using multiple streams
    std::vector<gpuStream_t>& streams = ipc_streams_[conn->remote_gpu_idx_];
    int num_streams = std::min(
        streams.size(),
        size < kIpcSizePerEngine ? 1 : (size_t)size / kIpcSizePerEngine);
    size_t chunk_size = size / num_streams;

    for (int i = 0; i < num_streams; ++i) {
      void* chunk_src = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(src_ptr) + i * chunk_size);
      void* chunk_dst = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(data) + i * chunk_size);
      auto copy_size =
          i == num_streams - 1 ? size - i * chunk_size : chunk_size;
      GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst, chunk_src, copy_size,
                                  gpuMemcpyDeviceToHost, streams[i]));
    }

    for (auto& stream : streams) {
      GPU_RT_CHECK(gpuStreamSynchronize(stream));
    }

    // Send completion to sender
    ShmMsg ack;
    ack.src_gpu = local_gpu_idx_;
    ack.type = ShmMsgType::COMPLETION;
    ack.completion = 1;
    shm_ring_send(conn->remote_inbox_.ring, ack);
  }

  return true;
}

bool Endpoint::send_ipc_async(uint64_t conn_id, void const* data, size_t size,
                              uint64_t* transfer_id) {
  // Create a task for IPC send operation
  auto task_ptr = create_task(conn_id, 0, TaskType::SEND_IPC,
                              const_cast<void*>(data), size);
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  // For now, we'll use the existing async infrastructure but call our IPC
  // function In a real implementation, you might want a separate IPC task ring
  while (jring_mp_enqueue_bulk(send_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::recv_ipc_async(uint64_t conn_id, void* data, size_t size,
                              uint64_t* transfer_id) {
  // Create a task for IPC receive operation
  auto task_ptr = create_task(conn_id, 0, TaskType::RECV_IPC, data, size);
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  // For now, we'll use the existing async infrastructure but call our IPC
  // function In a real implementation, you might want a separate IPC task ring
  while (jring_mp_enqueue_bulk(recv_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::write_ipc(uint64_t conn_id, void const* data, size_t size,
                         IpcTransferInfo const& info) {
  CHECK(data != nullptr) << "write_ipc: data pointer is null!";

  // Get connection info
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[write_ipc] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  bool is_host = (uccl::get_dev_idx(const_cast<void*>(data)) == -1);
  auto memcpy_kind = is_host ? gpuMemcpyHostToDevice : gpuMemcpyDeviceToDevice;

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  // Open the remote IPC memory handle
  void* raw_dst_ptr = nullptr;
  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));
  GPU_RT_CHECK(gpuIpcOpenMemHandle(&raw_dst_ptr, info.handle,
                                   gpuIpcMemLazyEnablePeerAccess));

  // Calculate destination pointer with offset
  void* dst_ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(raw_dst_ptr) + info.offset);

  // Perform the memory copy using multiple streams for better performance
  std::vector<gpuStream_t>& dst_streams = ipc_streams_[conn->remote_gpu_idx_];
  int num_streams =
      std::min(dst_streams.size(),
               size < kIpcSizePerEngine ? 1 : (size_t)size / kIpcSizePerEngine);
  size_t chunk_size = size / num_streams;

  for (int i = 0; i < num_streams; ++i) {
    // Split data and dst_ptr into n_streams chunks
    void* chunk_data = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(data) + i * chunk_size);
    void* chunk_dst_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(dst_ptr) + i * chunk_size);
    auto copy_size = i == num_streams - 1 ? size - i * chunk_size : chunk_size;

    GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst_ptr, chunk_data, copy_size,
                                memcpy_kind, dst_streams[i]));
  }

  // Wait for all streams to complete
  for (auto& stream : dst_streams) {
    GPU_RT_CHECK(gpuStreamSynchronize(stream));
  }

  // Close the IPC memory handle
  GPU_RT_CHECK(gpuIpcCloseMemHandle(raw_dst_ptr));

  return true;
}

bool Endpoint::read_ipc(uint64_t conn_id, void* data, size_t size,
                        IpcTransferInfo const& info) {
  CHECK(data != nullptr) << "read_ipc: data pointer is null!";

  // Get connection info
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[read_ipc] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  bool is_host = (uccl::get_dev_idx(data) == -1);
  auto memcpy_kind = is_host ? gpuMemcpyDeviceToHost : gpuMemcpyDeviceToDevice;

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  // Open the remote IPC memory handle
  void* raw_src_ptr = nullptr;
  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));
  GPU_RT_CHECK(gpuIpcOpenMemHandle(&raw_src_ptr, info.handle,
                                   gpuIpcMemLazyEnablePeerAccess));

  // Calculate source pointer with offset
  void* src_ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(raw_src_ptr) + info.offset);

  // Perform the memory copy using multiple streams for better performance
  std::vector<gpuStream_t>& src_streams = ipc_streams_[conn->remote_gpu_idx_];
  int num_streams =
      std::min(src_streams.size(),
               size < kIpcSizePerEngine ? 1 : (size_t)size / kIpcSizePerEngine);
  size_t chunk_size = size / num_streams;

  for (int i = 0; i < num_streams; ++i) {
    // Split src_ptr and data into n_streams chunks
    void* chunk_src_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(src_ptr) + i * chunk_size);
    void* chunk_data = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(data) + i * chunk_size);
    auto copy_size = i == num_streams - 1 ? size - i * chunk_size : chunk_size;

    GPU_RT_CHECK(gpuMemcpyAsync(chunk_data, chunk_src_ptr, copy_size,
                                memcpy_kind, src_streams[i]));
  }

  // Wait for all streams to complete
  for (auto& stream : src_streams) {
    GPU_RT_CHECK(gpuStreamSynchronize(stream));
  }

  // Close the IPC memory handle
  GPU_RT_CHECK(gpuIpcCloseMemHandle(raw_src_ptr));

  return true;
}

bool Endpoint::writev_ipc(uint64_t conn_id, std::vector<void const*> data_v,
                          std::vector<size_t> size_v,
                          std::vector<IpcTransferInfo> info_v,
                          size_t num_iovs) {
  CHECK_EQ(data_v.size(), num_iovs) << "writev_ipc: data_v size mismatch";
  CHECK_EQ(size_v.size(), num_iovs) << "writev_ipc: size_v size mismatch";
  CHECK_EQ(info_v.size(), num_iovs) << "writev_ipc: info_v size mismatch";

  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[writev_ipc] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));
  std::vector<gpuStream_t>& streams = ipc_streams_[conn->remote_gpu_idx_];

  // Open all handles and issue all memcpys before syncing any stream.
  std::vector<void*> raw_ptrs(num_iovs, nullptr);
  for (size_t iov = 0; iov < num_iovs; ++iov) {
    CHECK(data_v[iov] != nullptr)
        << "writev_ipc: data_v[" << iov << "] is null";
    GPU_RT_CHECK(gpuIpcOpenMemHandle(&raw_ptrs[iov], info_v[iov].handle,
                                     gpuIpcMemLazyEnablePeerAccess));
    void* dst_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(raw_ptrs[iov]) + info_v[iov].offset);

    bool is_host = (uccl::get_dev_idx(const_cast<void*>(data_v[iov])) == -1);
    auto memcpy_kind =
        is_host ? gpuMemcpyHostToDevice : gpuMemcpyDeviceToDevice;

    size_t sz = size_v[iov];
    int num_streams = std::min(
        streams.size(), sz < kIpcSizePerEngine ? 1 : sz / kIpcSizePerEngine);
    size_t chunk_size = sz / num_streams;

    for (int i = 0; i < num_streams; ++i) {
      void* chunk_src = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(data_v[iov]) + i * chunk_size);
      void* chunk_dst = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(dst_ptr) + i * chunk_size);
      auto copy_size = i == num_streams - 1 ? sz - i * chunk_size : chunk_size;
      GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst, chunk_src, copy_size, memcpy_kind,
                                  streams[i]));
    }
  }

  // Sync all streams, then close all handles.
  for (auto& stream : streams) {
    GPU_RT_CHECK(gpuStreamSynchronize(stream));
  }
  for (size_t iov = 0; iov < num_iovs; ++iov) {
    GPU_RT_CHECK(gpuIpcCloseMemHandle(raw_ptrs[iov]));
  }

  return true;
}

bool Endpoint::readv_ipc(uint64_t conn_id, std::vector<void*> data_v,
                         std::vector<size_t> size_v,
                         std::vector<IpcTransferInfo> info_v, size_t num_iovs) {
  CHECK_EQ(data_v.size(), num_iovs) << "readv_ipc: data_v size mismatch";
  CHECK_EQ(size_v.size(), num_iovs) << "readv_ipc: size_v size mismatch";
  CHECK_EQ(info_v.size(), num_iovs) << "readv_ipc: info_v size mismatch";

  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[readv_ipc] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));
  std::vector<gpuStream_t>& streams = ipc_streams_[conn->remote_gpu_idx_];

  // Open all handles and issue all memcpys before syncing any stream.
  std::vector<void*> raw_ptrs(num_iovs, nullptr);
  for (size_t iov = 0; iov < num_iovs; ++iov) {
    CHECK(data_v[iov] != nullptr) << "readv_ipc: data_v[" << iov << "] is null";
    GPU_RT_CHECK(gpuIpcOpenMemHandle(&raw_ptrs[iov], info_v[iov].handle,
                                     gpuIpcMemLazyEnablePeerAccess));
    void* src_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(raw_ptrs[iov]) + info_v[iov].offset);

    bool is_host = (uccl::get_dev_idx(data_v[iov]) == -1);
    auto memcpy_kind =
        is_host ? gpuMemcpyDeviceToHost : gpuMemcpyDeviceToDevice;

    size_t sz = size_v[iov];
    int num_streams = std::min(
        streams.size(), sz < kIpcSizePerEngine ? 1 : sz / kIpcSizePerEngine);
    size_t chunk_size = sz / num_streams;

    for (int i = 0; i < num_streams; ++i) {
      void* chunk_src = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(src_ptr) + i * chunk_size);
      void* chunk_dst = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(data_v[iov]) + i * chunk_size);
      auto copy_size = i == num_streams - 1 ? sz - i * chunk_size : chunk_size;
      GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst, chunk_src, copy_size, memcpy_kind,
                                  streams[i]));
    }
  }

  // Sync all streams, then close all handles.
  for (auto& stream : streams) {
    GPU_RT_CHECK(gpuStreamSynchronize(stream));
  }
  for (size_t iov = 0; iov < num_iovs; ++iov) {
    GPU_RT_CHECK(gpuIpcCloseMemHandle(raw_ptrs[iov]));
  }

  return true;
}

bool Endpoint::write_ipc_async(uint64_t conn_id, void const* data, size_t size,
                               IpcTransferInfo const& info,
                               uint64_t* transfer_id) {
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[write_ipc_async] Error: Invalid conn_id " << conn_id
              << std::endl;
    return false;
  }

  bool is_host = (uccl::get_dev_idx(const_cast<void*>(data)) == -1);
  auto memcpy_kind = is_host ? gpuMemcpyHostToDevice : gpuMemcpyDeviceToDevice;

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));
  void* raw_dst_ptr = nullptr;
  GPU_RT_CHECK(gpuIpcOpenMemHandle(&raw_dst_ptr, info.handle,
                                   gpuIpcMemLazyEnablePeerAccess));
  void* dst_ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(raw_dst_ptr) + info.offset);

  std::vector<gpuStream_t>& streams = ipc_streams_[conn->remote_gpu_idx_];
  int num_streams =
      std::min(streams.size(),
               size < kIpcSizePerEngine ? 1 : (size_t)size / kIpcSizePerEngine);
  size_t chunk_size = size / num_streams;

  auto* op = new IpcInflightOp{{}, raw_dst_ptr, nullptr, conn->remote_gpu_idx_};
  op->events.resize(num_streams);
  for (int i = 0; i < num_streams; ++i) {
    void* chunk_data = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(data) + i * chunk_size);
    void* chunk_dst = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(dst_ptr) + i * chunk_size);
    auto copy_size = i == num_streams - 1 ? size - i * chunk_size : chunk_size;
    GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst, chunk_data, copy_size, memcpy_kind,
                                streams[i]));
    GPU_RT_CHECK(
        gpuEventCreateWithFlags(&op->events[i], gpuEventDisableTiming));
    GPU_RT_CHECK(gpuEventRecord(op->events[i], streams[i]));
  }

  auto* status = new TransferStatus();
  op->status = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  while (jring_mp_enqueue_bulk(ipc_inflight_ring_, &op, 1, nullptr) != 1) {
  }

  return true;
}

bool Endpoint::read_ipc_async(uint64_t conn_id, void* data, size_t size,
                              IpcTransferInfo const& info,
                              uint64_t* transfer_id) {
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[read_ipc_async] Error: Invalid conn_id " << conn_id
              << std::endl;
    return false;
  }

  bool is_host = (uccl::get_dev_idx(data) == -1);
  auto memcpy_kind = is_host ? gpuMemcpyDeviceToHost : gpuMemcpyDeviceToDevice;

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));
  void* raw_src_ptr = nullptr;
  GPU_RT_CHECK(gpuIpcOpenMemHandle(&raw_src_ptr, info.handle,
                                   gpuIpcMemLazyEnablePeerAccess));
  void* src_ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(raw_src_ptr) + info.offset);

  std::vector<gpuStream_t>& streams = ipc_streams_[conn->remote_gpu_idx_];
  int num_streams =
      std::min(streams.size(),
               size < kIpcSizePerEngine ? 1 : (size_t)size / kIpcSizePerEngine);
  size_t chunk_size = size / num_streams;

  auto* op = new IpcInflightOp{{}, raw_src_ptr, nullptr, conn->remote_gpu_idx_};
  op->events.resize(num_streams);
  for (int i = 0; i < num_streams; ++i) {
    void* chunk_src = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(src_ptr) + i * chunk_size);
    void* chunk_data = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(data) + i * chunk_size);
    auto copy_size = i == num_streams - 1 ? size - i * chunk_size : chunk_size;
    GPU_RT_CHECK(gpuMemcpyAsync(chunk_data, chunk_src, copy_size, memcpy_kind,
                                streams[i]));
    GPU_RT_CHECK(
        gpuEventCreateWithFlags(&op->events[i], gpuEventDisableTiming));
    GPU_RT_CHECK(gpuEventRecord(op->events[i], streams[i]));
  }

  auto* status = new TransferStatus();
  op->status = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  while (jring_mp_enqueue_bulk(ipc_inflight_ring_, &op, 1, nullptr) != 1) {
  }

  return true;
}

bool Endpoint::writev_ipc_async(uint64_t conn_id,
                                std::vector<void const*> data_v,
                                std::vector<size_t> size_v,
                                std::vector<IpcTransferInfo> info_v,
                                size_t num_iovs, uint64_t* transfer_id) {
  CHECK_EQ(data_v.size(), num_iovs) << "writev_ipc_async: data_v size mismatch";
  CHECK_EQ(size_v.size(), num_iovs) << "writev_ipc_async: size_v size mismatch";
  CHECK_EQ(info_v.size(), num_iovs) << "writev_ipc_async: info_v size mismatch";

  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[writev_ipc_async] Error: Invalid conn_id " << conn_id
              << std::endl;
    return false;
  }

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));
  std::vector<gpuStream_t>& streams = ipc_streams_[conn->remote_gpu_idx_];

  // Use raw_ptr=nullptr to signal vectorized op to the poller thread.
  auto* op = new IpcInflightOp{{}, nullptr, nullptr, -1};
  op->raw_ptrs_v.resize(num_iovs);
  op->gpu_idxs_v.assign(num_iovs, conn->remote_gpu_idx_);

  for (size_t iov = 0; iov < num_iovs; ++iov) {
    GPU_RT_CHECK(gpuIpcOpenMemHandle(&op->raw_ptrs_v[iov], info_v[iov].handle,
                                     gpuIpcMemLazyEnablePeerAccess));
    void* dst_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(op->raw_ptrs_v[iov]) + info_v[iov].offset);

    bool is_host = (uccl::get_dev_idx(const_cast<void*>(data_v[iov])) == -1);
    auto memcpy_kind =
        is_host ? gpuMemcpyHostToDevice : gpuMemcpyDeviceToDevice;

    size_t sz = size_v[iov];
    int num_streams = std::min(
        streams.size(), sz < kIpcSizePerEngine ? 1 : sz / kIpcSizePerEngine);
    size_t chunk_size = sz / num_streams;

    for (int i = 0; i < num_streams; ++i) {
      void* chunk_src = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(data_v[iov]) + i * chunk_size);
      void* chunk_dst = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(dst_ptr) + i * chunk_size);
      auto copy_size = i == num_streams - 1 ? sz - i * chunk_size : chunk_size;
      GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst, chunk_src, copy_size, memcpy_kind,
                                  streams[i]));
      gpuEvent_t ev;
      GPU_RT_CHECK(gpuEventCreateWithFlags(&ev, gpuEventDisableTiming));
      GPU_RT_CHECK(gpuEventRecord(ev, streams[i]));
      op->events.push_back(ev);
    }
  }

  auto* status = new TransferStatus();
  op->status = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  while (jring_mp_enqueue_bulk(ipc_inflight_ring_, &op, 1, nullptr) != 1) {
  }

  return true;
}

bool Endpoint::readv_ipc_async(uint64_t conn_id, std::vector<void*> data_v,
                               std::vector<size_t> size_v,
                               std::vector<IpcTransferInfo> info_v,
                               size_t num_iovs, uint64_t* transfer_id) {
  CHECK_EQ(data_v.size(), num_iovs) << "readv_ipc_async: data_v size mismatch";
  CHECK_EQ(size_v.size(), num_iovs) << "readv_ipc_async: size_v size mismatch";
  CHECK_EQ(info_v.size(), num_iovs) << "readv_ipc_async: info_v size mismatch";

  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[readv_ipc_async] Error: Invalid conn_id " << conn_id
              << std::endl;
    return false;
  }

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));
  std::vector<gpuStream_t>& streams = ipc_streams_[conn->remote_gpu_idx_];

  // Use raw_ptr=nullptr to signal vectorized op to the poller thread.
  auto* op = new IpcInflightOp{{}, nullptr, nullptr, -1};
  op->raw_ptrs_v.resize(num_iovs);
  op->gpu_idxs_v.assign(num_iovs, conn->remote_gpu_idx_);

  for (size_t iov = 0; iov < num_iovs; ++iov) {
    GPU_RT_CHECK(gpuIpcOpenMemHandle(&op->raw_ptrs_v[iov], info_v[iov].handle,
                                     gpuIpcMemLazyEnablePeerAccess));
    void* src_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(op->raw_ptrs_v[iov]) + info_v[iov].offset);

    bool is_host = (uccl::get_dev_idx(data_v[iov]) == -1);
    auto memcpy_kind =
        is_host ? gpuMemcpyDeviceToHost : gpuMemcpyDeviceToDevice;

    size_t sz = size_v[iov];
    int num_streams = std::min(
        streams.size(), sz < kIpcSizePerEngine ? 1 : sz / kIpcSizePerEngine);
    size_t chunk_size = sz / num_streams;

    for (int i = 0; i < num_streams; ++i) {
      void* chunk_src = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(src_ptr) + i * chunk_size);
      void* chunk_dst = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(data_v[iov]) + i * chunk_size);
      auto copy_size = i == num_streams - 1 ? sz - i * chunk_size : chunk_size;
      GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst, chunk_src, copy_size, memcpy_kind,
                                  streams[i]));
      gpuEvent_t ev;
      GPU_RT_CHECK(gpuEventCreateWithFlags(&ev, gpuEventDisableTiming));
      GPU_RT_CHECK(gpuEventRecord(ev, streams[i]));
      op->events.push_back(ev);
    }
  }

  auto* status = new TransferStatus();
  op->status = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  while (jring_mp_enqueue_bulk(ipc_inflight_ring_, &op, 1, nullptr) != 1) {
  }

  return true;
}

bool Endpoint::advertise_ipc(uint64_t conn_id, void* addr, size_t len,
                             char* out_buf) {
  CHECK(addr != nullptr) << "advertise_ipc: addr pointer is null!";
  CHECK(out_buf != nullptr) << "advertise_ipc: out_buf pointer is null!";

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

  // Generate IPC memory handle for the address
  IpcTransferInfo transfer_info = {};  // Initialize to zero
  transfer_info.size = len;
  transfer_info.operation = 1;  // response

  // Calculate aligned address and offset
  auto addr_aligned = reinterpret_cast<uintptr_t>(addr) & ~(kIpcAlignment - 1);
  auto addr_offset = reinterpret_cast<uintptr_t>(addr) - addr_aligned;
  transfer_info.offset = addr_offset;

  GPU_RT_CHECK(gpuIpcGetMemHandle(&transfer_info.handle,
                                  reinterpret_cast<void*>(addr_aligned)));

  // Copy the transfer info to output buffer
  std::memcpy(out_buf, &transfer_info, sizeof(transfer_info));

  return true;
}

bool Endpoint::advertisev_ipc(uint64_t conn_id, std::vector<void*> addr_v,
                              std::vector<size_t> len_v,
                              std::vector<char*> out_buf_v, size_t num_iovs) {
  CHECK_EQ(addr_v.size(), num_iovs) << "addr_v size mismatch";
  CHECK_EQ(len_v.size(), num_iovs) << "len_v size mismatch";
  CHECK_EQ(out_buf_v.size(), num_iovs) << "out_buf_v size mismatch";

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

  for (size_t i = 0; i < num_iovs; ++i) {
    CHECK(addr_v[i] != nullptr)
        << "advertisev_ipc: addr_v[" << i << "] is null!";
    CHECK(out_buf_v[i] != nullptr)
        << "advertisev_ipc: out_buf_v[" << i << "] is null!";

    // Generate IPC memory handle for each address
    IpcTransferInfo transfer_info = {};  // Initialize to zero
    transfer_info.size = len_v[i];
    transfer_info.operation = 1;  // response

    // Calculate aligned address and offset
    auto addr_aligned =
        reinterpret_cast<uintptr_t>(addr_v[i]) & ~(kIpcAlignment - 1);
    auto addr_offset = reinterpret_cast<uintptr_t>(addr_v[i]) - addr_aligned;
    transfer_info.offset = addr_offset;

    GPU_RT_CHECK(gpuIpcGetMemHandle(&transfer_info.handle,
                                    reinterpret_cast<void*>(addr_aligned)));

    // Copy the transfer info to output buffer
    std::memcpy(out_buf_v[i], &transfer_info, sizeof(transfer_info));
  }

  return true;
}

bool Endpoint::add_remote_endpoint(std::vector<uint8_t> const& metadata,
                                   uint64_t& conn_id) {
  {
    // Check if we have connected to the remote endpoint before
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = remote_endpoint_to_conn_id_.find(metadata);
    if (it != remote_endpoint_to_conn_id_.end()) {
      conn_id = it->second;
      return true;
    }
  }
  auto [remote_ip, remote_port, remote_gpu_idx] = parse_metadata(metadata);
  bool success = connect(remote_ip, remote_gpu_idx, remote_port, conn_id);
  if (success) {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    remote_endpoint_to_conn_id_[metadata] = conn_id;
    return true;
  }
  return false;
}

bool Endpoint::start_passive_accept() {
  if (!passive_accept_) {
    passive_accept_stop_.store(false, std::memory_order_release);
    passive_accept_thread_ =
        std::thread(&Endpoint::passive_accept_thread_func, this);
    passive_accept_ = true;
  }
  return true;
}

bool Endpoint::poll_async(uint64_t transfer_id, bool* is_done) {
  auto* status = reinterpret_cast<TransferStatus*>(transfer_id);
  *is_done = status->done.load(std::memory_order_acquire);
  if (*is_done) {
    delete status;
  }
  return true;
}

int Endpoint::get_sock_fd(uint64_t conn_id) const {
  auto it = conn_id_to_conn_.find(conn_id);
  if (it == conn_id_to_conn_.end()) {
    return -1;
  }
  return it->second->uccl_conn_id_.sock_fd;
}

uint64_t Endpoint::conn_id_of_rank(int rank) const {
  auto it = rank2conn_.find(rank);
  return it != rank2conn_.end() ? it->second : UINT64_MAX;
}

std::shared_ptr<EpollClient> Endpoint::get_oob_client() const {
  return ep_->get_oob_client();
}

std::string Endpoint::get_oob_conn_key(uint64_t conn_id) const {
  auto it = conn_id_to_conn_.find(conn_id);
  if (it == conn_id_to_conn_.end()) {
    return "";
  }
  uint64_t rank_id = it->second->uccl_conn_id_.flow_id;

  return ep_->get_oob_conn_key(rank_id);
}

MR* Endpoint::get_mr(uint64_t mr_id) const {
  std::shared_lock<std::shared_mutex> lock(mr_mu_);
  auto it = mr_id_to_mr_.find(mr_id);
  if (it == mr_id_to_mr_.end()) {
    return nullptr;
  }
  return it->second;
}

P2PMhandle* Endpoint::get_mhandle(uint64_t mr_id) const {
  auto mr = get_mr(mr_id);
  if (unlikely(mr == nullptr)) {
    return nullptr;
  }
  return mr->mhandle_;
}

Conn* Endpoint::get_conn(uint64_t conn_id) const {
  std::shared_lock<std::shared_mutex> lock(conn_mu_);
  auto it = conn_id_to_conn_.find(conn_id);
  if (it == conn_id_to_conn_.end()) {
    return nullptr;
  }
  return it->second;
}

RDMAEndPoint Endpoint::get_endpoint() const { return ep_; }

void Endpoint::initialize_engine() {
  int n_streams = std::max(1, (int)kNumGpuRtStreams);
  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

#ifdef UCCL_P2P_USE_NCCL
  numa_node_ = tcp::get_tcp_numa_node_from_iface();
#else
  numa_node_ = RdmaDeviceManager::instance().get_numa_node(local_gpu_idx_);
#endif

  // Initialize rdma contexts for devices used by the GPU
  initialize_rdma_ctx_for_gpu(ep_, local_gpu_idx_);
  std::cout << "Lazy creation of engine for GPU " << local_gpu_idx_
            << std::endl;

  // Initialize task rings
  send_unified_task_ring_ =
      uccl::create_ring(sizeof(UnifiedTask*), kTaskRingSize);
  recv_unified_task_ring_ =
      uccl::create_ring(sizeof(UnifiedTask*), kTaskRingSize);

  send_proxy_thread_ = std::thread(&Endpoint::send_proxy_thread_func, this);
  recv_proxy_thread_ = std::thread(&Endpoint::recv_proxy_thread_func, this);

  ipc_inflight_ring_ = uccl::create_ring(sizeof(IpcInflightOp*), kTaskRingSize);
  ipc_poller_thread_ = std::thread(&Endpoint::ipc_poller_thread_func, this);

  // Initialize ShmChnnel for local connections
  size_t elem_sz = sizeof(ShmMsg);
  size_t elem_cnt = ShmRingDefaultElemCnt;
  for (int i = 0; i < kMaxNumGPUs; i++) {
    inbox_rings_[i].shm_name = shm_ring_name(i, local_gpu_idx_);
    inbox_rings_[i].ring = uccl::create_shared_ring(
        inbox_rings_[i].shm_name.c_str(), elem_sz, elem_cnt,
        inbox_rings_[i].shm_fd, inbox_rings_[i].shm_size, &inbox_creators_[i]);
  }
}

void Endpoint::send_proxy_thread_func() {
  uccl::pin_thread_to_numa(numa_node_);
  // Use 16-byte buffer to avoid stringop-overflow warning from jring's 16-byte
  // bulk copy
  alignas(16) char task_buffer[16];
  UnifiedTask* task;

  while (!stop_.load(std::memory_order_acquire)) {
    if (jring_sc_dequeue_bulk(send_unified_task_ring_, task_buffer, 1,
                              nullptr) == 1) {
      task = *reinterpret_cast<UnifiedTask**>(task_buffer);
      switch (task->type) {
        case TaskType::SEND_IPC:
          send_ipc(task->conn_id, task->data, task->size);
          break;
        case TaskType::WRITE_NET:
          write(task->conn_id, task->mr_id, task->data, task->size,
                task->slot_item());
          break;
        case TaskType::SEND_NET:
          send(task->conn_id, task->mr_id, task->data, task->size);
          break;
        case TaskType::SENDV: {
          TaskBatch const& batch = task->task_batch();
          std::vector<void const*> const_data_v(
              batch.const_data_v(), batch.const_data_v() + batch.num_iovs);
          std::vector<size_t> size_v(batch.size_v(),
                                     batch.size_v() + batch.num_iovs);
          std::vector<uint64_t> mr_id_v(batch.mr_id_v(),
                                        batch.mr_id_v() + batch.num_iovs);

          sendv(task->conn_id, mr_id_v, const_data_v, size_v, batch.num_iovs);
          break;
        }
        case TaskType::WRITEV: {
          TaskBatch const& batch = task->task_batch();
          std::vector<void*> data_v(batch.data_v(),
                                    batch.data_v() + batch.num_iovs);
          std::vector<size_t> size_v(batch.size_v(),
                                     batch.size_v() + batch.num_iovs);
          std::vector<uint64_t> mr_id_v(batch.mr_id_v(),
                                        batch.mr_id_v() + batch.num_iovs);
          std::vector<FifoItem> slot_item_v(
              batch.slot_item_v(), batch.slot_item_v() + batch.num_iovs);

          writev(task->conn_id, mr_id_v, data_v, size_v, slot_item_v,
                 batch.num_iovs);
          break;
        }
        default:
          LOG(ERROR) << "Unexpected task type in send processing: "
                     << static_cast<int>(task->type);
          break;
      }
      auto* status = task->status_ptr;
      status->task_ptr.reset();
      status->done.store(true, std::memory_order_release);
    }
  }
}

void Endpoint::recv_proxy_thread_func() {
  uccl::pin_thread_to_numa(numa_node_);
  // Use 16-byte buffer to avoid stringop-overflow warning from jring's 16-byte
  // bulk copy
  alignas(16) char task_buffer[16];
  UnifiedTask* task;

  while (!stop_.load(std::memory_order_acquire)) {
    if (jring_sc_dequeue_bulk(recv_unified_task_ring_, task_buffer, 1,
                              nullptr) == 1) {
      task = *reinterpret_cast<UnifiedTask**>(task_buffer);
      switch (task->type) {
        case TaskType::RECV_IPC:
          recv_ipc(task->conn_id, task->data, task->size);
          break;
        case TaskType::READ_NET:
          read(task->conn_id, task->mr_id, task->data, task->size,
               task->slot_item());
          break;
        case TaskType::RECV_NET:
          recv(task->conn_id, task->mr_id, task->data, task->size);
          break;
        case TaskType::RECVV: {
          TaskBatch const& batch = task->task_batch();
          std::vector<void*> data_v(batch.data_v(),
                                    batch.data_v() + batch.num_iovs);
          std::vector<size_t> size_v(batch.size_v(),
                                     batch.size_v() + batch.num_iovs);
          std::vector<uint64_t> mr_id_v(batch.mr_id_v(),
                                        batch.mr_id_v() + batch.num_iovs);

          recvv(task->conn_id, mr_id_v, data_v, size_v, batch.num_iovs);
          break;
        }
        case TaskType::READV: {
          TaskBatch const& batch = task->task_batch();
          std::vector<void*> data_v(batch.data_v(),
                                    batch.data_v() + batch.num_iovs);
          std::vector<size_t> size_v(batch.size_v(),
                                     batch.size_v() + batch.num_iovs);
          std::vector<uint64_t> mr_id_v(batch.mr_id_v(),
                                        batch.mr_id_v() + batch.num_iovs);
          std::vector<FifoItem> slot_item_v(
              batch.slot_item_v(), batch.slot_item_v() + batch.num_iovs);

          readv(task->conn_id, mr_id_v, data_v, size_v, slot_item_v,
                batch.num_iovs);
          break;
        }
        case TaskType::SEND_NET:
        case TaskType::SEND_IPC:
        case TaskType::WRITE_NET:
        default:
          LOG(ERROR) << "Unexpected task type in receive processing: "
                     << static_cast<int>(task->type);
          break;
      }
      auto* status = task->status_ptr;
      status->task_ptr.reset();
      status->done.store(true, std::memory_order_release);
    }
  }
}

void Endpoint::passive_accept_thread_func() {
  std::string ip_addr;
  int remote_gpu_idx;
  uint64_t conn_id;
  while (!stop_.load(std::memory_order_acquire)) {
    (void)accept(ip_addr, remote_gpu_idx, conn_id);
  }
}

void Endpoint::ipc_poller_thread_func() {
  uccl::pin_thread_to_numa(numa_node_);
  std::deque<IpcInflightOp*> active_ops;
  alignas(16) char buf[16];
  int cur_device = -1;

  while (!stop_.load(std::memory_order_acquire) || !active_ops.empty()) {
    // Drain newly submitted ops into the local active list.
    while (jring_sc_dequeue_bulk(ipc_inflight_ring_, buf, 1, nullptr) == 1) {
      active_ops.push_back(*reinterpret_cast<IpcInflightOp**>(buf));
    }

    for (auto it = active_ops.begin(); it != active_ops.end();) {
      IpcInflightOp* op = *it;
      bool all_done = true;
      for (auto& event : op->events) {
        auto result = gpuEventQuery(event);
        if (result == gpuErrorNotReady) {
          all_done = false;
          break;
        }
        GPU_RT_CHECK(result);
      }
      if (all_done) {
        for (auto& event : op->events) {
          GPU_RT_CHECK(gpuEventDestroy(event));
        }
        if (op->raw_ptr != nullptr) {
          // Scalar op: single handle to close.
          if (cur_device != op->gpu_idx) {
            GPU_RT_CHECK(gpuSetDevice(op->gpu_idx));
            cur_device = op->gpu_idx;
          }
          GPU_RT_CHECK(gpuIpcCloseMemHandle(op->raw_ptr));
        } else {
          // Vectorized op: close all handles.
          for (size_t i = 0; i < op->raw_ptrs_v.size(); ++i) {
            if (cur_device != op->gpu_idxs_v[i]) {
              GPU_RT_CHECK(gpuSetDevice(op->gpu_idxs_v[i]));
              cur_device = op->gpu_idxs_v[i];
            }
            GPU_RT_CHECK(gpuIpcCloseMemHandle(op->raw_ptrs_v[i]));
          }
        }
        op->status->done.store(true, std::memory_order_release);
        delete op;
        it = active_ops.erase(it);
      } else {
        ++it;
      }
    }
  }
}

std::shared_ptr<UnifiedTask> Endpoint::create_task(uint64_t conn_id,
                                                   uint64_t mr_id,
                                                   TaskType type, void* data,
                                                   size_t size) {
  auto task = std::make_shared<UnifiedTask>();
  task->type = type;
  task->data = data;
  task->size = size;
  task->conn_id = conn_id;
  task->mr_id = mr_id;
  return task;
}

std::shared_ptr<UnifiedTask> Endpoint::create_batch_task(uint64_t conn_id,
                                                         TaskType type,
                                                         TaskBatch&& batch) {
  auto task = std::make_shared<UnifiedTask>();
  task->type = type;
  task->conn_id = conn_id;
  task->mr_id = 0;
  task->data = nullptr;
  task->size = 0;
  new (&task->specific.batch.task_batch) TaskBatch(std::move(batch));
  return task;
}

std::shared_ptr<UnifiedTask> Endpoint::create_sendv_task(
    uint64_t conn_id, std::shared_ptr<std::vector<void const*>> const_data_ptr,
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
  batch.const_data_ptr = std::move(const_data_ptr);
  batch.size_ptr = std::move(size_ptr);
  batch.mr_id_ptr = std::move(mr_id_ptr);
  return create_batch_task(conn_id, TaskType::SENDV, std::move(batch));
}

std::shared_ptr<UnifiedTask> Endpoint::create_recvv_task(
    uint64_t conn_id, std::vector<void*>&& data_v, std::vector<size_t>&& size_v,
    std::vector<uint64_t>&& mr_id_v) {
  if (data_v.size() != size_v.size() || size_v.size() != mr_id_v.size()) {
    return nullptr;
  }
  size_t num_iovs = data_v.size();
  auto data_ptr = std::make_shared<std::vector<void*>>(std::move(data_v));
  auto size_ptr = std::make_shared<std::vector<size_t>>(std::move(size_v));
  auto mr_id_ptr = std::make_shared<std::vector<uint64_t>>(std::move(mr_id_v));
  TaskBatch batch;
  batch.num_iovs = num_iovs;
  batch.data_ptr = std::move(data_ptr);
  batch.size_ptr = std::move(size_ptr);
  batch.mr_id_ptr = std::move(mr_id_ptr);
  return create_batch_task(conn_id, TaskType::RECVV, std::move(batch));
}

std::shared_ptr<UnifiedTask> Endpoint::create_writev_task(
    uint64_t conn_id, std::vector<void*>&& data_v, std::vector<size_t>&& size_v,
    std::vector<uint64_t>&& mr_id_v, std::vector<FifoItem>&& slot_item_v) {
  if (data_v.size() != size_v.size() || size_v.size() != mr_id_v.size() ||
      mr_id_v.size() != slot_item_v.size()) {
    return nullptr;
  }
  size_t num_iovs = data_v.size();
  auto data_ptr = std::make_shared<std::vector<void*>>(std::move(data_v));
  auto size_ptr = std::make_shared<std::vector<size_t>>(std::move(size_v));
  auto mr_id_ptr = std::make_shared<std::vector<uint64_t>>(std::move(mr_id_v));
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

std::shared_ptr<UnifiedTask> Endpoint::create_readv_task(
    uint64_t conn_id, std::vector<void*>&& data_v, std::vector<size_t>&& size_v,
    std::vector<uint64_t>&& mr_id_v, std::vector<FifoItem>&& slot_item_v) {
  if (data_v.size() != size_v.size() || size_v.size() != mr_id_v.size() ||
      mr_id_v.size() != slot_item_v.size()) {
    return nullptr;
  }
  size_t num_iovs = data_v.size();
  auto data_ptr = std::make_shared<std::vector<void*>>(std::move(data_v));
  auto size_ptr = std::make_shared<std::vector<size_t>>(std::move(size_v));
  auto mr_id_ptr = std::make_shared<std::vector<uint64_t>>(std::move(mr_id_v));
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

std::shared_ptr<UnifiedTask> Endpoint::create_net_task(
    uint64_t conn_id, uint64_t mr_id, TaskType type, void* data, size_t size,
    FifoItem const& slot_item) {
  auto task = create_task(conn_id, mr_id, type, data, size);
  task->slot_item() = slot_item;
  return task;
}

std::shared_ptr<UnifiedTask> Endpoint::create_ipc_task(
    uint64_t conn_id, uint64_t mr_id, TaskType type, void* data, size_t size,
    IpcTransferInfo const& ipc_info) {
  auto task = create_task(conn_id, mr_id, type, data, size);
  task->ipc_info() = ipc_info;
  return task;
}