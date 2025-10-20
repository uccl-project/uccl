#include "engine.h"
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
#include <sys/un.h>
#include <unistd.h>

int const kMaxNumGPUs = 8;
// Assume the local and remote GPUs have the same GPU-NIC mapping.
uint8_t gpu_to_dev[kMaxNumGPUs] = {0};
std::once_flag glog_init_once;
constexpr uint32_t kGpuStreamId = 0;
thread_local bool inside_python = false;

inline void check_python_signals() {
  PyGILState_STATE gstate = PyGILState_Ensure();
  if (PyErr_CheckSignals() != 0) {
    std::cerr << "Python signal caught, exiting..." << std::endl;
    std::abort();
  }
  PyGILState_Release(gstate);
}

Endpoint::Endpoint(uint32_t const local_gpu_idx, uint32_t const num_cpus)
    : local_gpu_idx_(local_gpu_idx), num_cpus_(num_cpus) {
  std::cout << "Creating Engine with GPU index: " << local_gpu_idx
            << ", CPUs: " << num_cpus << std::endl;
  int n_streams = std::max(1, (int)ucclParamNumGpuRtStreams());

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
  streams_.resize(n_streams);
  for (int i = 0; i < n_streams; ++i) {
    GPU_RT_CHECK(gpuStreamCreateWithFlags(&streams_[i], gpuStreamNonBlocking));
  }

  std::call_once(glog_init_once,
                 []() { google::InitGoogleLogging("uccl_p2p"); });

  google::InstallFailureSignalHandler();

  // Initialize the RDMA endpoint with lazy creation.
  ep_ = new uccl::RDMAEndpoint(num_cpus_);

  // Only initialize mapping for detected GPUs
  int ngpus_detected = 0;
  GPU_RT_CHECK(gpuGetDeviceCount(&ngpus_detected));
  for (int i = 0; i < std::min(ngpus_detected, kMaxNumGPUs); i++) {
    gpu_to_dev[i] = ep_->get_best_dev_idx(i);
  }
  // Initialize remaining slots to 0 (fallback to first device)
  for (int i = ngpus_detected; i < kMaxNumGPUs; i++) {
    gpu_to_dev[i] = 0;
  }
  numa_node_ =
      uccl::RDMAFactory::get_factory_dev(gpu_to_dev[local_gpu_idx_])->numa_node;

  // Initialize the engine based on the GPU index.
  std::cout << "Lazy creation of engine, GPU index: " << local_gpu_idx_
            << std::endl;
  ep_->initialize_engine_by_dev(gpu_to_dev[local_gpu_idx_], true);
  std::cout << "Engine initialized for GPU " << local_gpu_idx_ << std::endl;

  send_unified_task_ring_ =
      uccl::create_ring(sizeof(UnifiedTask), kTaskRingSize);
  recv_unified_task_ring_ =
      uccl::create_ring(sizeof(UnifiedTask), kTaskRingSize);

  send_proxy_thread_ = std::thread(&Endpoint::send_proxy_thread_func, this);
  recv_proxy_thread_ = std::thread(&Endpoint::recv_proxy_thread_func, this);

  // Initialize UDS socket for local connections
  init_uds_socket();

  std::cout << "Endpoint initialized successfully" << std::endl;
}

Endpoint::~Endpoint() {
  std::cout << "Destroying Engine..." << std::endl;

  stop_.store(true, std::memory_order_release);

  send_proxy_thread_.join();
  recv_proxy_thread_.join();

  free(send_unified_task_ring_);
  free(recv_unified_task_ring_);

  delete ep_;

  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    for (auto& [conn_id, conn] : conn_id_to_conn_) {
      // Close UDS socket if it exists
      if (conn->uds_sockfd_ >= 0) {
        close(conn->uds_sockfd_);
      }
      delete conn;
    }
  }
  {
    std::shared_lock<std::shared_mutex> lock(mr_mu_);
    for (auto& [mr_id, mr] : mr_id_to_mr_) {
      delete mr;
    }
  }

  if (!streams_.empty()) {
    GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
    for (auto s : streams_)
      if (s) GPU_RT_CHECK(gpuStreamDestroy(s));
  }

  // Cleanup UDS socket
  cleanup_uds_socket();

  std::cout << "Engine destroyed" << std::endl;
}

bool Endpoint::connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
                       uint64_t& conn_id) {
  std::cout << "Attempting to connect to " << ip_addr << ":" << remote_gpu_idx
            << " via port " << remote_port << std::endl;

  // Create a new connection ID
  conn_id = next_conn_id_.fetch_add(1);

  std::future<uccl::ConnID> uccl_conn_id_future = std::async(
      std::launch::async, [this, remote_gpu_idx, &ip_addr, remote_port]() {
        return ep_->uccl_connect(gpu_to_dev[local_gpu_idx_], local_gpu_idx_,
                                 gpu_to_dev[remote_gpu_idx], remote_gpu_idx,
                                 ip_addr, remote_port);
      });

  // Check for Python signals (eg, ctrl+c) while waiting for connection
  while (uccl_conn_id_future.wait_for(std::chrono::seconds(0)) !=
         std::future_status::ready) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  uccl::ConnID uccl_conn_id = uccl_conn_id_future.get();

  // Store the connection ID.
  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    conn_id_to_conn_[conn_id] =
        new Conn{conn_id, uccl_conn_id, ip_addr, remote_gpu_idx};
  }
  return true;
}

std::vector<uint8_t> Endpoint::get_metadata() {
  std::string ip_str = get_oob_ip();
  uint16_t port = ep_->get_p2p_listen_port(gpu_to_dev[local_gpu_idx_]);

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

  std::future<uccl::ConnID> uccl_conn_id_future =
      std::async(std::launch::async, [this, &ip_addr, &remote_gpu_idx]() {
        auto dev_idx = gpu_to_dev[local_gpu_idx_];
        auto p2p_listen_fd = ep_->get_p2p_listen_fd(dev_idx);
        int remote_dev_idx;
        return ep_->uccl_accept(dev_idx, p2p_listen_fd, local_gpu_idx_, ip_addr,
                                &remote_dev_idx, &remote_gpu_idx);
      });

  // Check for Python signals (eg, ctrl+c) while waiting for connection
  while (uccl_conn_id_future.wait_for(std::chrono::seconds(0)) !=
         std::future_status::ready) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  uccl::ConnID uccl_conn_id = uccl_conn_id_future.get();

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

  uccl::Mhandle* mhandle;
  ep_->uccl_regmr(gpu_to_dev[local_gpu_idx_], const_cast<void*>(data), size, 0,
                  &mhandle);
  if (mhandle->mr == nullptr) {
    std::cerr << "[Endpoint::reg] Failed to register memory region, "
              << "mhandle->mr is null\n";
    std::abort();
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
  mr_id_v.resize(n);

  {
    std::unique_lock<std::shared_mutex> lock(mr_mu_);
    mr_id_to_mr_.reserve(mr_id_to_mr_.size() + n);
  }

  for (size_t i = 0; i < n; ++i) {
    uint64_t id = next_mr_id_.fetch_add(1);
    uccl::Mhandle* mhandle = nullptr;

    ep_->uccl_regmr(gpu_to_dev[local_gpu_idx_], const_cast<void*>(data_v[i]),
                    size_v[i], 0, &mhandle);

    if (mhandle == nullptr || mhandle->mr == nullptr) {
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
    MR* mr = mr_id_to_mr_[mr_id];
    ep_->uccl_deregmr(mr->mhandle_);
    delete mr;
    mr_id_to_mr_.erase(mr_id);
  }
  return true;
}

bool Endpoint::send(uint64_t conn_id, uint64_t mr_id, void const* data,
                    size_t size) {
  DCHECK(size <= 0xffffffff) << "size must be less than 4GB";

  Conn* conn;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    conn = conn_id_to_conn_[conn_id];
  }

  uccl::Mhandle* mhandle;
  {
    std::shared_lock<std::shared_mutex> lock(mr_mu_);
    mhandle = mr_id_to_mr_[mr_id]->mhandle_;
  }

  uccl::ucclRequest ureq[kMaxInflightChunks] = {};
  bool done[kMaxInflightChunks] = {false};

  void* cur_data = const_cast<void*>(data);
  size_t size_sent = 0;
  int ureq_max = (size + kChunkSize - 1) / kChunkSize;
  int ureq_issued = 0, ureq_finished = 0;

  while (ureq_finished < ureq_max) {
    while (ureq_issued - ureq_finished < kMaxInflightChunks &&
           size_sent < size) {
      size_t chunk_size = std::min(size - size_sent, kChunkSize);
      auto rc = ep_->uccl_send_async(
          static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), mhandle,
          cur_data, chunk_size, &ureq[ureq_issued % kMaxInflightChunks]);
      if (rc == -1) break;
      cur_data += chunk_size;
      size_sent += chunk_size;
      done[ureq_issued % kMaxInflightChunks] = false;
      ureq_issued++;
    }
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;

    // First, poll all outstanding requests and mark which ones are done.
    for (int i = ureq_finished; i < ureq_issued; i++) {
      if (done[i % kMaxInflightChunks]) {
        continue;
      }
      if (ep_->uccl_poll_ureq_once(&ureq[i % kMaxInflightChunks])) {
        // Just mark it as completed, DO NOT increment ureq_finished here.
        done[i % kMaxInflightChunks] = true;
      }
    }

    // Now, advance the ureq_finished counter as far as possible.
    while (ureq_finished < ureq_issued &&
           done[ureq_finished % kMaxInflightChunks]) {
      ureq_finished++;
    }
  }

  return true;
}

bool Endpoint::recv(uint64_t conn_id, uint64_t mr_id, void* data, size_t size) {
  Conn* conn;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    conn = conn_id_to_conn_[conn_id];
  }

  uccl::Mhandle* mhandle;
  {
    std::shared_lock<std::shared_mutex> lock(mr_mu_);
    mhandle = mr_id_to_mr_[mr_id]->mhandle_;
  }
  int size_int = static_cast<int>(size);

  uccl::ucclRequest ureq[kMaxInflightChunks] = {};
  bool done[kMaxInflightChunks] = {false};

  void* cur_data = data;
  size_t size_post_recv = 0;
  int ureq_max = (size + kChunkSize - 1) / kChunkSize;
  int ureq_issued = 0, ureq_finished = 0;

  while (ureq_finished < ureq_max) {
    while (ureq_issued - ureq_finished < kMaxInflightChunks &&
           size_post_recv < size) {
      int chunk_size = std::min(size - size_post_recv, kChunkSize);
      auto rc = ep_->uccl_recv_async(
          static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), &mhandle,
          &cur_data, &chunk_size, 1, &ureq[ureq_issued % kMaxInflightChunks]);
      if (rc == -1) break;
      cur_data += chunk_size;
      size_post_recv += chunk_size;
      done[ureq_issued % kMaxInflightChunks] = false;
      ureq_issued++;
    }
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;

    // First, poll all outstanding requests and mark which ones are done.
    for (int i = ureq_finished; i < ureq_issued; i++) {
      if (done[i % kMaxInflightChunks]) {
        continue;
      }
      if (ep_->uccl_poll_ureq_once(&ureq[i % kMaxInflightChunks])) {
        // Just mark it as completed, DO NOT increment ureq_finished here.
        done[i % kMaxInflightChunks] = true;
      }
    }

    // Now, advance the ureq_finished counter as far as possible.
    while (ureq_finished < ureq_issued &&
           done[ureq_finished % kMaxInflightChunks]) {
      ureq_finished++;
    }
  }

  return true;
}

bool Endpoint::send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                          size_t size, uint64_t* transfer_id) {
  UnifiedTask* task = create_task(conn_id, mr_id, TaskType::SEND_NET,
                                  const_cast<void*>(data), size);
  if (unlikely(task == nullptr)) {
    return false;
  }

  *transfer_id = reinterpret_cast<uint64_t>(task);

  while (jring_mp_enqueue_bulk(send_unified_task_ring_, task, 1, nullptr) !=
         1) {
  }

  return true;
}

bool Endpoint::recv_async(uint64_t conn_id, uint64_t mr_id, void* data,
                          size_t size, uint64_t* transfer_id) {
  UnifiedTask* task =
      create_task(conn_id, mr_id, TaskType::RECV_NET, data, size);
  if (unlikely(task == nullptr)) {
    return false;
  }

  *transfer_id = reinterpret_cast<uint64_t>(task);

  while (jring_mp_enqueue_bulk(recv_unified_task_ring_, task, 1, nullptr) !=
         1) {
  }

  return true;
}

bool Endpoint::sendv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                     std::vector<void const*> data_v,
                     std::vector<size_t> size_v, size_t num_iovs) {
  Conn* conn;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_id_to_conn_.find(conn_id);
    if (it == conn_id_to_conn_.end()) {
      std::cerr << "[sendv] Error: Invalid conn_id " << conn_id << std::endl;
      return false;
    }
    conn = it->second;
  }
  auto uccl_flow = static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context);

  uccl::ucclRequest ureq[kMaxInflightChunks] = {};
  bool done[kMaxInflightChunks] = {false};

  int estimated_ureq_max = 0;
  for (int i = 0; i < num_iovs; i++) {
    estimated_ureq_max += (size_v[i] + kChunkSize - 1) / kChunkSize;
  }

  std::vector<void*> data_send_vec;
  std::vector<size_t> size_send_vec;
  std::vector<uccl::Mhandle*> mhandle_send_vec;
  // Avoid reallocations.
  data_send_vec.reserve(estimated_ureq_max);
  size_send_vec.reserve(estimated_ureq_max);
  mhandle_send_vec.reserve(estimated_ureq_max);

  for (int i = 0; i < num_iovs; i++) {
    void* cur_data = (void*)data_v[i];
    size_t cur_size_expected = size_v[i];

    size_t cur_size_post_send = 0;
    while (cur_size_post_send < cur_size_expected) {
      int chunk_size =
          std::min(cur_size_expected - cur_size_post_send, kChunkSize);
      data_send_vec.push_back(cur_data);
      size_send_vec.push_back(chunk_size);
      {
        std::shared_lock<std::shared_mutex> lock(mr_mu_);
        auto it = mr_id_to_mr_.find(mr_id_v[i]);
        if (it == mr_id_to_mr_.end()) {
          std::cerr << "[sendv] Error: Invalid mr_id " << mr_id_v[i]
                    << std::endl;
          return false;
        }
        mhandle_send_vec.push_back(it->second->mhandle_);
      }
      cur_data += chunk_size;
      cur_size_post_send += chunk_size;
    }
  }

  int ureq_max = data_send_vec.size();
  int ureq_issued = 0, ureq_finished = 0;

  while (ureq_finished < ureq_max) {
    while (ureq_issued < ureq_max &&
           ureq_issued - ureq_finished < kMaxInflightChunks &&
           size_send_vec[ureq_issued] > 0) {
      auto rc = ep_->uccl_send_async(
          uccl_flow, mhandle_send_vec[ureq_issued], data_send_vec[ureq_issued],
          size_send_vec[ureq_issued], &ureq[ureq_issued % kMaxInflightChunks]);
      if (rc == -1) break;
      done[ureq_issued % kMaxInflightChunks] = false;
      ureq_issued++;
    }
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;

    for (int i = ureq_finished; i < ureq_issued; i++) {
      if (done[i % kMaxInflightChunks]) {
        continue;
      }
      if (ep_->uccl_poll_ureq_once(&ureq[i % kMaxInflightChunks])) {
        done[i % kMaxInflightChunks] = true;
      }
    }

    while (ureq_finished < ureq_issued &&
           done[ureq_finished % kMaxInflightChunks]) {
      ureq_finished++;
    }
  }

  return true;
}

bool Endpoint::recvv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                     std::vector<void*> data_v, std::vector<size_t> size_v,
                     size_t num_iovs) {
  Conn* conn;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_id_to_conn_.find(conn_id);
    if (it == conn_id_to_conn_.end()) {
      std::cerr << "[recvv] Error: Invalid conn_id " << conn_id << std::endl;
      return false;
    }
    conn = it->second;
  }

  auto uccl_flow = static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context);

  uccl::ucclRequest ureq[kMaxInflightChunks] = {};
  bool done[kMaxInflightChunks] = {false};

  // Prepare the data, size, and mhandle vectors for the rest of the chunks.
  std::vector<void*> data_recv_vec;
  std::vector<void**> data_recv_ptr_vec;
  std::vector<int> size_recv_vec;
  std::vector<uccl::Mhandle*> mhandle_recv_vec;
  std::vector<uccl::Mhandle**> mhandle_recv_ptr_vec;

  int estimated_ureq_max = 0;
  for (int i = 0; i < num_iovs; i++) {
    estimated_ureq_max += (size_v[i] + kChunkSize - 1) / kChunkSize;
  }

  data_recv_vec.reserve(estimated_ureq_max);
  data_recv_ptr_vec.reserve(estimated_ureq_max);
  size_recv_vec.reserve(estimated_ureq_max);
  mhandle_recv_vec.reserve(estimated_ureq_max);
  mhandle_recv_ptr_vec.reserve(estimated_ureq_max);

  for (int i = 0; i < num_iovs; i++) {
    void* cur_data = data_v[i];
    size_t cur_size_expected = size_v[i];

    size_t cur_size_post_recv = 0;
    while (cur_size_post_recv < cur_size_expected) {
      int chunk_size =
          std::min(cur_size_expected - cur_size_post_recv, kChunkSize);
      data_recv_vec.push_back(cur_data);
      data_recv_ptr_vec.push_back(&data_recv_vec.back());
      size_recv_vec.push_back(chunk_size);
      {
        std::shared_lock<std::shared_mutex> lock(mr_mu_);
        auto it = mr_id_to_mr_.find(mr_id_v[i]);
        if (it == mr_id_to_mr_.end()) {
          std::cerr << "[recvv] Error: Invalid mr_id " << mr_id_v[i]
                    << std::endl;
          return false;
        }
        mhandle_recv_vec.push_back(it->second->mhandle_);
      }
      mhandle_recv_ptr_vec.push_back(&mhandle_recv_vec.back());
      cur_data += chunk_size;
      cur_size_post_recv += chunk_size;
    }
  }

  // Handle receiving the rest of the sub-chunks.
  int ureq_max = data_recv_vec.size();
  int ureq_issued = 0, ureq_finished = 0;

  while (ureq_finished < ureq_max) {
    while (ureq_issued < ureq_max &&
           ureq_issued - ureq_finished < kMaxInflightChunks &&
           size_recv_vec[ureq_issued] > 0) {
      auto rc = ep_->uccl_recv_async(
          uccl_flow, mhandle_recv_ptr_vec[ureq_issued],
          data_recv_ptr_vec[ureq_issued], &size_recv_vec[ureq_issued], 1,
          &ureq[ureq_issued % kMaxInflightChunks]);
      if (rc == -1) break;
      done[ureq_issued % kMaxInflightChunks] = false;
      ureq_issued++;
    }
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;

    for (int i = ureq_finished; i < ureq_issued; i++) {
      if (done[i % kMaxInflightChunks]) {
        continue;
      }
      if (ep_->uccl_poll_ureq_once(&ureq[i % kMaxInflightChunks])) {
        done[i % kMaxInflightChunks] = true;
      }
    }

    while (ureq_finished < ureq_issued &&
           done[ureq_finished % kMaxInflightChunks]) {
      ureq_finished++;
    }
  }

  return true;
}

bool Endpoint::read(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
                    uccl::FifoItem const& slot_item) {
  if (!ucclParamRCMode()) {
    DCHECK(false) << "RDMA READ is only supported in RC mode, toggle RCMODE to "
                     "be True in transport_config.h";
    std::abort();
  }

  DCHECK(size <= 0xffffffff) << "size must be < 4 GB";
  auto* conn = conn_id_to_conn_[conn_id];
  auto* mhandle = mr_id_to_mr_[mr_id]->mhandle_;

  uccl::ucclRequest ureq[kMaxInflightChunks] = {};
  uccl::FifoItem curr_slot_item[kMaxInflightChunks] = {};
  bool done[kMaxInflightChunks] = {false};

  void* cur_data = dst;
  size_t size_read = 0;
  int ureq_max = (size + kChunkSize - 1) / kChunkSize;
  int ureq_issued = 0, ureq_finished = 0;

  auto num_engines = ucclParamNUM_ENGINES();

  while (ureq_finished < ureq_max) {
    while (ureq_issued - ureq_finished < kMaxInflightChunks &&
           size_read < size) {
      size_t chunk_size = std::min(size - size_read, kChunkSize);
      curr_slot_item[ureq_issued % kMaxInflightChunks] = slot_item;
      curr_slot_item[ureq_issued % kMaxInflightChunks].addr += size_read;
      curr_slot_item[ureq_issued % kMaxInflightChunks].size = chunk_size;
      curr_slot_item[ureq_issued % kMaxInflightChunks].engine_offset =
          ureq_issued % num_engines;
      memset(&ureq[ureq_issued % kMaxInflightChunks], 0,
             sizeof(uccl::ucclRequest));
      auto rc = ep_->uccl_read_async(
          static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), mhandle,
          cur_data, chunk_size,
          curr_slot_item[ureq_issued % kMaxInflightChunks],
          &ureq[ureq_issued % kMaxInflightChunks]);
      if (rc == -1) break;
      cur_data += chunk_size;
      size_read += chunk_size;
      done[ureq_issued % kMaxInflightChunks] = false;
      ureq_issued++;
    }
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;

    // First, poll all outstanding requests and mark which ones are done.
    for (int i = ureq_finished; i < ureq_issued; i++) {
      if (done[i % kMaxInflightChunks]) {
        continue;
      }
      if (ep_->uccl_poll_ureq_once(&ureq[i % kMaxInflightChunks])) {
        // Just mark it as completed, DO NOT increment ureq_finished here.
        done[i % kMaxInflightChunks] = true;
      }
    }

    // Now, advance the ureq_finished counter as far as possible.
    while (ureq_finished < ureq_issued &&
           done[ureq_finished % kMaxInflightChunks]) {
      ureq_finished++;
    }
  }

  return true;
}

bool Endpoint::read_async(uint64_t conn_id, uint64_t mr_id, void* dst,
                          size_t size, uccl::FifoItem const& slot_item,
                          uint64_t* transfer_id) {
  UnifiedTask* rw_task =
      create_net_task(conn_id, mr_id, TaskType::READ_NET, dst, size, slot_item);
  if (unlikely(rw_task == nullptr)) {
    return false;
  }

  *transfer_id = reinterpret_cast<uint64_t>(rw_task);

  while (jring_mp_enqueue_bulk(recv_unified_task_ring_, rw_task, 1, nullptr) !=
         1) {
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

  UnifiedTask* task =
      create_sendv_task(conn_id, std::move(const_data_ptr), std::move(size_ptr),
                        std::move(mr_id_ptr));
  if (unlikely(task == nullptr)) {
    return false;
  }

  *transfer_id = reinterpret_cast<uint64_t>(task);

  while (jring_mp_enqueue_bulk(send_unified_task_ring_, task, 1, nullptr) !=
         1) {
  }

  return true;
}

bool Endpoint::recvv_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                           std::vector<void*> data_v,
                           std::vector<size_t> size_v, size_t num_iovs,
                           uint64_t* transfer_id) {
  // Use move semantics to reduce memory copies
  UnifiedTask* task = create_recvv_task(conn_id, std::move(data_v),
                                        std::move(size_v), std::move(mr_id_v));
  if (unlikely(task == nullptr)) {
    return false;
  }

  *transfer_id = reinterpret_cast<uint64_t>(task);

  while (jring_mp_enqueue_bulk(recv_unified_task_ring_, task, 1, nullptr) !=
         1) {
  }

  return true;
}

bool Endpoint::readv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                     std::vector<void*> dst_v, std::vector<size_t> size_v,
                     std::vector<uccl::FifoItem> slot_item_v, size_t num_iovs) {
  auto conn = conn_id_to_conn_[conn_id];
  auto uccl_flow = static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context);

  uccl::ucclRequest ureq[kMaxInflightChunks] = {};
  uccl::FifoItem curr_slot_item[kMaxInflightChunks] = {};
  bool done[kMaxInflightChunks] = {false};

  int estimated_ureq_max = 0;
  for (int i = 0; i < num_iovs; i++) {
    estimated_ureq_max += (size_v[i] + kChunkSize - 1) / kChunkSize;
  }

  std::vector<void*> data_read_vec;
  std::vector<size_t> size_read_vec;
  std::vector<uccl::Mhandle*> mhandle_read_vec;
  std::vector<uccl::FifoItem> slot_item_vec;
  // Avoid reallocations.
  data_read_vec.reserve(estimated_ureq_max);
  size_read_vec.reserve(estimated_ureq_max);
  mhandle_read_vec.reserve(estimated_ureq_max);
  slot_item_vec.reserve(estimated_ureq_max);

  for (int i = 0; i < num_iovs; i++) {
    void* cur_data = dst_v[i];
    size_t cur_size_expected = size_v[i];
    size_t cur_size_post_read = 0;
    uccl::FifoItem base_slot_item = slot_item_v[i];
    auto mhandle = mr_id_to_mr_[mr_id_v[i]]->mhandle_;

    while (cur_size_post_read < cur_size_expected) {
      size_t chunk_size =
          std::min(cur_size_expected - cur_size_post_read, (size_t)kChunkSize);
      uccl::FifoItem chunk_slot_item = base_slot_item;
      chunk_slot_item.addr += cur_size_post_read;
      chunk_slot_item.size = chunk_size;
      // engine_offset will be set later
      data_read_vec.push_back(cur_data);
      size_read_vec.push_back(chunk_size);
      mhandle_read_vec.push_back(mhandle);
      slot_item_vec.push_back(chunk_slot_item);
      cur_data = (void*)((char*)cur_data + chunk_size);
      cur_size_post_read += chunk_size;
    }
  }

  int ureq_max = data_read_vec.size();
  int ureq_issued = 0, ureq_finished = 0;
  auto num_engines = ucclParamNUM_ENGINES();

  while (ureq_finished < ureq_max) {
    while (ureq_issued < ureq_max &&
           ureq_issued - ureq_finished < kMaxInflightChunks &&
           size_read_vec[ureq_issued] > 0) {
      slot_item_vec[ureq_issued].engine_offset = ureq_issued % num_engines;
      curr_slot_item[ureq_issued % kMaxInflightChunks] =
          slot_item_vec[ureq_issued];
      memset(&ureq[ureq_issued % kMaxInflightChunks], 0,
             sizeof(uccl::ucclRequest));
      auto rc = ep_->uccl_read_async(
          uccl_flow, mhandle_read_vec[ureq_issued], data_read_vec[ureq_issued],
          size_read_vec[ureq_issued],
          curr_slot_item[ureq_issued % kMaxInflightChunks],
          &ureq[ureq_issued % kMaxInflightChunks]);
      if (rc == -1) break;
      done[ureq_issued % kMaxInflightChunks] = false;
      ureq_issued++;
    }
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;

    for (int i = ureq_finished; i < ureq_issued; i++) {
      if (done[i % kMaxInflightChunks]) {
        continue;
      }
      if (ep_->uccl_poll_ureq_once(&ureq[i % kMaxInflightChunks])) {
        done[i % kMaxInflightChunks] = true;
      }
    }

    while (ureq_finished < ureq_issued &&
           done[ureq_finished % kMaxInflightChunks]) {
      ureq_finished++;
    }
  }

  return true;
}

bool Endpoint::write_async(uint64_t conn_id, uint64_t mr_id, void* src,
                           size_t size, uccl::FifoItem const& slot_item,
                           uint64_t* transfer_id) {
  UnifiedTask* rw_task = create_net_task(conn_id, mr_id, TaskType::WRITE_NET,
                                         src, size, slot_item);
  if (unlikely(rw_task == nullptr)) {
    return false;
  }
  *transfer_id = reinterpret_cast<uint64_t>(rw_task);

  while (jring_mp_enqueue_bulk(send_unified_task_ring_, rw_task, 1, nullptr) !=
         1) {
  }

  return true;
}

bool Endpoint::writev(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                      std::vector<void*> src_v, std::vector<size_t> size_v,
                      std::vector<uccl::FifoItem> slot_item_v,
                      size_t num_iovs) {
  auto conn = conn_id_to_conn_[conn_id];
  auto uccl_flow = static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context);

  uccl::ucclRequest ureq[kMaxInflightChunks] = {};
  uccl::FifoItem curr_slot_item[kMaxInflightChunks] = {};
  bool done[kMaxInflightChunks] = {false};

  int estimated_ureq_max = 0;
  for (int i = 0; i < num_iovs; i++) {
    estimated_ureq_max += (size_v[i] + kChunkSize - 1) / kChunkSize;
  }

  std::vector<void*> data_write_vec;
  std::vector<size_t> size_write_vec;
  std::vector<uccl::Mhandle*> mhandle_write_vec;
  std::vector<uccl::FifoItem> slot_item_vec;
  // Avoid reallocations.
  data_write_vec.reserve(estimated_ureq_max);
  size_write_vec.reserve(estimated_ureq_max);
  mhandle_write_vec.reserve(estimated_ureq_max);
  slot_item_vec.reserve(estimated_ureq_max);

  for (int i = 0; i < num_iovs; i++) {
    void* cur_data = src_v[i];
    size_t cur_size_expected = size_v[i];
    size_t cur_size_post_write = 0;
    uccl::FifoItem base_slot_item = slot_item_v[i];
    auto mhandle = mr_id_to_mr_[mr_id_v[i]]->mhandle_;

    while (cur_size_post_write < cur_size_expected) {
      size_t chunk_size =
          std::min(cur_size_expected - cur_size_post_write, (size_t)kChunkSize);
      uccl::FifoItem chunk_slot_item = base_slot_item;
      chunk_slot_item.addr += cur_size_post_write;
      chunk_slot_item.size = chunk_size;
      // engine_offset will be set later
      data_write_vec.push_back(cur_data);
      size_write_vec.push_back(chunk_size);
      mhandle_write_vec.push_back(mhandle);
      slot_item_vec.push_back(chunk_slot_item);
      cur_data = (void*)((char*)cur_data + chunk_size);
      cur_size_post_write += chunk_size;
    }
  }

  int ureq_max = data_write_vec.size();
  int ureq_issued = 0, ureq_finished = 0;
  auto num_engines = ucclParamNUM_ENGINES();

  while (ureq_finished < ureq_max) {
    while (ureq_issued < ureq_max &&
           ureq_issued - ureq_finished < kMaxInflightChunks &&
           size_write_vec[ureq_issued] > 0) {
      slot_item_vec[ureq_issued].engine_offset = ureq_issued % num_engines;
      curr_slot_item[ureq_issued % kMaxInflightChunks] =
          slot_item_vec[ureq_issued];
      memset(&ureq[ureq_issued % kMaxInflightChunks], 0,
             sizeof(uccl::ucclRequest));
      auto rc = ep_->uccl_write_async(
          uccl_flow, mhandle_write_vec[ureq_issued],
          data_write_vec[ureq_issued], size_write_vec[ureq_issued],
          curr_slot_item[ureq_issued % kMaxInflightChunks],
          &ureq[ureq_issued % kMaxInflightChunks]);
      if (rc == -1) break;
      done[ureq_issued % kMaxInflightChunks] = false;
      ureq_issued++;
    }
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;

    for (int i = ureq_finished; i < ureq_issued; i++) {
      if (done[i % kMaxInflightChunks]) {
        continue;
      }
      if (ep_->uccl_poll_ureq_once(&ureq[i % kMaxInflightChunks])) {
        done[i % kMaxInflightChunks] = true;
      }
    }

    while (ureq_finished < ureq_issued &&
           done[ureq_finished % kMaxInflightChunks]) {
      ureq_finished++;
    }
  }

  return true;
}

bool Endpoint::write(uint64_t conn_id, uint64_t mr_id, void* src, size_t size,
                     uccl::FifoItem const& slot_item) {
  if (!ucclParamRCMode()) {
    DCHECK(false) << "We only support RC mode for now.";
    std::abort();
  }

  DCHECK(size <= 0xffffffff) << "size must be < 4 GB";
  auto* conn = conn_id_to_conn_[conn_id];
  auto* mhandle = mr_id_to_mr_[mr_id]->mhandle_;

  uccl::ucclRequest ureq[kMaxInflightChunks] = {};
  uccl::FifoItem curr_slot_item[kMaxInflightChunks] = {};
  bool done[kMaxInflightChunks] = {false};

  void* cur_data = src;
  size_t size_write = 0;
  int ureq_max = (size + kChunkSize - 1) / kChunkSize;
  int ureq_issued = 0, ureq_finished = 0;

  auto num_engines = ucclParamNUM_ENGINES();

  while (ureq_finished < ureq_max) {
    while (ureq_issued - ureq_finished < kMaxInflightChunks &&
           size_write < size) {
      size_t chunk_size = std::min(size - size_write, kChunkSize);
      curr_slot_item[ureq_issued % kMaxInflightChunks] = slot_item;
      curr_slot_item[ureq_issued % kMaxInflightChunks].addr += size_write;
      curr_slot_item[ureq_issued % kMaxInflightChunks].size = chunk_size;
      curr_slot_item[ureq_issued % kMaxInflightChunks].engine_offset =
          ureq_issued % num_engines;
      memset(&ureq[ureq_issued % kMaxInflightChunks], 0,
             sizeof(uccl::ucclRequest));
      auto rc = ep_->uccl_write_async(
          static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), mhandle,
          cur_data, chunk_size,
          curr_slot_item[ureq_issued % kMaxInflightChunks],
          &ureq[ureq_issued % kMaxInflightChunks]);
      if (rc == -1) break;
      cur_data += chunk_size;
      size_write += chunk_size;
      done[ureq_issued % kMaxInflightChunks] = false;
      ureq_issued++;
    }
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;

    // First, poll all outstanding requests and mark which ones are done.
    for (int i = ureq_finished; i < ureq_issued; i++) {
      if (done[i % kMaxInflightChunks]) {
        continue;
      }
      if (ep_->uccl_poll_ureq_once(&ureq[i % kMaxInflightChunks])) {
        // Just mark it as completed, DO NOT increment ureq_finished here.
        done[i % kMaxInflightChunks] = true;
      }
    }

    // Now, advance the ureq_finished counter as far as possible.
    while (ureq_finished < ureq_issued &&
           done[ureq_finished % kMaxInflightChunks]) {
      ureq_finished++;
    }
  }
  return true;
}

bool Endpoint::advertise(uint64_t conn_id, uint64_t mr_id, void* addr,
                         size_t len, char* out_buf) {
  auto* conn = conn_id_to_conn_[conn_id];
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;
  uccl::ucclRequest req_data;
  if (ep_->prepare_fifo_metadata(
          static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), &mhandle,
          addr, len, out_buf) == -1)
    return false;
  return true;
}

bool Endpoint::advertisev(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                          std::vector<void*> addr_v, std::vector<size_t> len_v,
                          std::vector<char*> out_buf_v, size_t num_iovs) {
  auto* conn = conn_id_to_conn_[conn_id];
  for (size_t i = 0; i < num_iovs; ++i) {
    auto mhandle = mr_id_to_mr_[mr_id_v[i]]->mhandle_;
    if (ep_->prepare_fifo_metadata(
            static_cast<uccl::UcclFlow*>(conn->uccl_conn_id_.context), &mhandle,
            addr_v[i], len_v[i], out_buf_v[i]) == -1) {
      return false;
    }
  }
  return true;
}

bool Endpoint::connect_local(int remote_gpu_idx, uint64_t& conn_id) {
  int retries = 5;
  int ret = -1;
  std::cout << "Connecting to remote GPU " << remote_gpu_idx << std::endl;

  std::string remote_socket_path = get_uds_socket_path(remote_gpu_idx);

  // Create socket for connection
  int sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
  CHECK_GE(sockfd, 0) << "Failed to create UDS socket for connection: "
                      << strerror(errno);
  fcntl(sockfd, F_SETFL, fcntl(sockfd, F_GETFL) | O_NONBLOCK);

  // Set up socket address
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, remote_socket_path.c_str(), sizeof(addr.sun_path) - 1);

  // Connect to remote socket
  for (int i = 0; i < retries; ++i) {
    ret = ::connect(sockfd, (struct sockaddr*)&addr, sizeof(addr));
    if (ret == 0) break;

    if (errno == ECONNREFUSED || errno == EAGAIN) {
      std::cerr << "Connect failed: " << strerror(errno) << ", retry "
                << (i + 1) << "/" << retries << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(200 * (i + 1)));
      continue;
    }
    break;
  }

  // Send our GPU index to the remote endpoint
  ret = uccl::send_message_nonblock(sockfd,
                                    static_cast<void const*>(&local_gpu_idx_),
                                    sizeof(local_gpu_idx_));
  CHECK_EQ(ret, sizeof(local_gpu_idx_)) << "Failed to send local GPU index";

  // Create a new connection ID for this local connection
  conn_id = next_conn_id_.fetch_add(1);

  // Create a special connection entry for local UDS connection with persistent
  // socket
  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    uccl::ConnID dummy_conn_id{nullptr, 0, 0, 0};
    conn_id_to_conn_[conn_id] =
        new Conn{conn_id, dummy_conn_id, "localhost", remote_gpu_idx, sockfd};
  }

  return true;
}

bool Endpoint::accept_local(int& remote_gpu_idx, uint64_t& conn_id) {
  std::cout << "Waiting to accept UDS connection" << std::endl;

  CHECK(uds_listen_fd_ >= 0) << "UDS socket not initialized";

  // Accept incoming connection
  struct sockaddr_un client_addr;
  socklen_t client_len = sizeof(client_addr);
  int client_fd =
      ::accept(uds_listen_fd_, (struct sockaddr*)&client_addr, &client_len);
  CHECK_GE(client_fd, 0) << "Failed to accept UDS connection: "
                         << strerror(errno);

  fcntl(client_fd, F_SETFL, fcntl(client_fd, F_GETFL) | O_NONBLOCK);

  // Receive remote GPU index
  auto ret = uccl::receive_message_nonblock(
      client_fd, static_cast<void*>(&remote_gpu_idx), sizeof(remote_gpu_idx));
  CHECK_EQ(ret, sizeof(remote_gpu_idx)) << "Failed to receive remote GPU index";

  // Create connection ID
  conn_id = next_conn_id_.fetch_add(1);

  // Store the connection with persistent socket
  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    uccl::ConnID dummy_conn_id{nullptr, 0, 0, 0};
    conn_id_to_conn_[conn_id] = new Conn{conn_id, dummy_conn_id, "localhost",
                                         remote_gpu_idx, client_fd};
  }

  return true;
}

bool Endpoint::send_ipc(uint64_t conn_id, void* data, size_t size) {
  CHECK(data != nullptr) << "send_ipc: data pointer is null!";

  // Get connection info
  Conn* conn;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_id_to_conn_.find(conn_id);
    CHECK(it != conn_id_to_conn_.end())
        << "Connection not found for conn_id: " << conn_id;
    conn = it->second;
  }

  // Check if we have a valid persistent UDS socket (faster than string
  // comparison)
  CHECK_GE(conn->uds_sockfd_, 0)
      << "send_ipc only supports local connections with valid UDS socket";

  // Use the persistent UDS connection
  int sockfd = conn->uds_sockfd_;

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  IpcTransferInfo transfer_info = {};  // Initialize to zero
  // Wait for receiver's IPC handle (receiver will send this proactively)
  auto ret = uccl::receive_message_nonblock(
      sockfd, static_cast<void*>(&transfer_info), sizeof(transfer_info));
  CHECK_EQ(ret, sizeof(transfer_info))
      << "Failed to receive IPC handle from receiver";
  CHECK_EQ(transfer_info.operation, 1) << "Invalid response from receiver";
  CHECK_EQ(transfer_info.size, size)
      << "Size mismatch: expected " << size << ", got " << transfer_info.size;

  void* raw_dst_ptr = nullptr;
  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));
  GPU_RT_CHECK(gpuIpcOpenMemHandle(&raw_dst_ptr, transfer_info.handle,
                                   gpuIpcMemLazyEnablePeerAccess));

  void* dst_ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(raw_dst_ptr) + transfer_info.offset);

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
    // Works for both intra-GPU and inter-GPU copy
    GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst_ptr, chunk_data, copy_size,
                                gpuMemcpyDeviceToDevice, dst_streams[i]));
  }

  for (auto& stream : dst_streams) {
    GPU_RT_CHECK(gpuStreamSynchronize(stream));
  }

  // Notify receiver of completion
  uint32_t completion = 1;
  ret = uccl::send_message_nonblock(
      sockfd, static_cast<void const*>(&completion), sizeof(completion));
  CHECK_EQ(ret, sizeof(completion)) << "Failed to send completion ack";

  // Okay, this is the slowest part, 46GB/s -> 28GB/s for 100MB, so moving it
  // async. Update: moving async does not help, as gpuIpcOpenMemHandle will be
  // slower.
  // std::thread close_mem_handle(
  //     [raw_dst_ptr]() { GPU_RT_CHECK(gpuIpcCloseMemHandle(raw_dst_ptr)); });
  // close_mem_handle.detach();

  // We close all IPC memory handles when releasing this endpoint.

  return true;
}

bool Endpoint::recv_ipc(uint64_t conn_id, void* data, size_t size) {
  CHECK(data != nullptr) << "recv_ipc: data pointer is null!";

  // Get connection info
  Conn* conn;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_id_to_conn_.find(conn_id);
    CHECK(it != conn_id_to_conn_.end())
        << "Connection not found for conn_id: " << conn_id;
    conn = it->second;
  }

  // Check if we have a valid persistent UDS socket (faster than string
  // comparison)
  CHECK_GE(conn->uds_sockfd_, 0)
      << "recv_ipc only supports local connections with valid UDS socket";

  // Use the persistent UDS connection
  int client_fd = conn->uds_sockfd_;

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
  // Generate IPC memory handle for our receive buffer
  IpcTransferInfo transfer_info = {};  // Initialize to zero
  transfer_info.size = size;
  transfer_info.operation = 1;  // response

  auto data_aligned = reinterpret_cast<uintptr_t>(data) & ~(kIpcAlignment - 1);
  auto data_offset = reinterpret_cast<uintptr_t>(data) - data_aligned;
  transfer_info.offset = data_offset;
  GPU_RT_CHECK(gpuIpcGetMemHandle(&transfer_info.handle,
                                  reinterpret_cast<void*>(data_aligned)));

  auto ret = uccl::send_message_nonblock(
      client_fd, static_cast<void const*>(&transfer_info),
      sizeof(transfer_info));
  CHECK_EQ(ret, sizeof(transfer_info)) << "Failed to send IPC handle to sender";

  // Notify sender of completion
  uint32_t completion = 0;
  ret = uccl::receive_message_nonblock(
      client_fd, static_cast<void*>(&completion), sizeof(completion));
  CHECK_EQ(ret, sizeof(completion))
      << "Failed to receive completion notification";
  CHECK_EQ(completion, 1) << "Sender reported failure";

  return true;
}

bool Endpoint::send_ipc_async(uint64_t conn_id, void const* data, size_t size,
                              uint64_t* transfer_id) {
  // Create a task for IPC send operation
  UnifiedTask* task = create_task(conn_id, 0, TaskType::SEND_IPC,
                                  const_cast<void*>(data), size);
  if (unlikely(task == nullptr)) {
    return false;
  }

  *transfer_id = reinterpret_cast<uint64_t>(task);

  // For now, we'll use the existing async infrastructure but call our IPC
  // function In a real implementation, you might want a separate IPC task ring
  while (jring_mp_enqueue_bulk(send_unified_task_ring_, task, 1, nullptr) !=
         1) {
  }

  return true;
}

bool Endpoint::recv_ipc_async(uint64_t conn_id, void* data, size_t size,
                              uint64_t* transfer_id) {
  // Create a task for IPC receive operation
  UnifiedTask* task = create_task(conn_id, 0, TaskType::RECV_IPC, data, size);
  if (unlikely(task == nullptr)) {
    return false;
  }

  *transfer_id = reinterpret_cast<uint64_t>(task);

  // For now, we'll use the existing async infrastructure but call our IPC
  // function In a real implementation, you might want a separate IPC task ring
  while (jring_mp_enqueue_bulk(recv_unified_task_ring_, task, 1, nullptr) !=
         1) {
  }

  return true;
}

bool Endpoint::write_ipc(uint64_t conn_id, void const* data, size_t size,
                         IpcTransferInfo const& info) {
  CHECK(data != nullptr) << "write_ipc: data pointer is null!";

  // Get connection info
  Conn* conn;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_id_to_conn_.find(conn_id);
    CHECK(it != conn_id_to_conn_.end())
        << "Connection not found for conn_id: " << conn_id;
    conn = it->second;
  }

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

    // Works for both intra-GPU and inter-GPU copy
    GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst_ptr, chunk_data, copy_size,
                                gpuMemcpyDeviceToDevice, dst_streams[i]));
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
  Conn* conn;
  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_id_to_conn_.find(conn_id);
    CHECK(it != conn_id_to_conn_.end())
        << "Connection not found for conn_id: " << conn_id;
    conn = it->second;
  }

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

    // Works for both intra-GPU and inter-GPU copy
    GPU_RT_CHECK(gpuMemcpyAsync(chunk_data, chunk_src_ptr, copy_size,
                                gpuMemcpyDeviceToDevice, src_streams[i]));
  }

  // Wait for all streams to complete
  for (auto& stream : src_streams) {
    GPU_RT_CHECK(gpuStreamSynchronize(stream));
  }

  // Close the IPC memory handle
  GPU_RT_CHECK(gpuIpcCloseMemHandle(raw_src_ptr));

  return true;
}

bool Endpoint::write_ipc_async(uint64_t conn_id, void const* data, size_t size,
                               IpcTransferInfo const& info,
                               uint64_t* transfer_id) {
  // Create an IPC task for IPC write operation
  UnifiedTask* task = create_ipc_task(conn_id, 0, TaskType::WRITE_IPC,
                                      const_cast<void*>(data), size, info);
  if (unlikely(task == nullptr)) {
    return false;
  }

  *transfer_id = reinterpret_cast<uint64_t>(task);

  // Enqueue the task for processing by proxy thread
  while (jring_mp_enqueue_bulk(send_unified_task_ring_, task, 1, nullptr) !=
         1) {
  }

  return true;
}

bool Endpoint::read_ipc_async(uint64_t conn_id, void* data, size_t size,
                              IpcTransferInfo const& info,
                              uint64_t* transfer_id) {
  // Create an IPC task for IPC read operation
  UnifiedTask* task =
      create_ipc_task(conn_id, 0, TaskType::READ_IPC, data, size, info);
  if (unlikely(task == nullptr)) {
    return false;
  }

  *transfer_id = reinterpret_cast<uint64_t>(task);

  // Enqueue the task for processing by proxy thread
  while (jring_mp_enqueue_bulk(recv_unified_task_ring_, task, 1, nullptr) !=
         1) {
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

bool Endpoint::poll_async(uint64_t transfer_id, bool* is_done) {
  auto task = reinterpret_cast<UnifiedTask*>(transfer_id);
  *is_done = task->done.load(std::memory_order_acquire);
  if (*is_done) {
    delete task;
  }
  return true;
}

void Endpoint::init_uds_socket() {
  // Create UDS socket path based on local GPU index
  uds_socket_path_ = get_uds_socket_path(local_gpu_idx_);

  // Remove existing socket file if it exists
  unlink(uds_socket_path_.c_str());

  // Create socket
  uds_listen_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
  if (uds_listen_fd_ < 0) {
    std::cerr << "Failed to create UDS socket: " << strerror(errno)
              << std::endl;
    return;
  }

  // Set up socket address
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, uds_socket_path_.c_str(), sizeof(addr.sun_path) - 1);

  // Bind socket
  if (bind(uds_listen_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    std::cerr << "Failed to bind UDS socket to " << uds_socket_path_ << ": "
              << strerror(errno) << std::endl;
    close(uds_listen_fd_);
    uds_listen_fd_ = -1;
    return;
  }

  // Start listening
  if (listen(uds_listen_fd_, 5) < 0) {
    std::cerr << "Failed to listen on UDS socket: " << strerror(errno)
              << std::endl;
    close(uds_listen_fd_);
    uds_listen_fd_ = -1;
    unlink(uds_socket_path_.c_str());
    return;
  }

  std::cout << "UDS socket initialized at " << uds_socket_path_ << std::endl;
}

void Endpoint::cleanup_uds_socket() {
  if (uds_listen_fd_ >= 0) {
    close(uds_listen_fd_);
    uds_listen_fd_ = -1;
  }

  if (!uds_socket_path_.empty()) {
    unlink(uds_socket_path_.c_str());
    uds_socket_path_.clear();
  }
}

void Endpoint::send_proxy_thread_func() {
  uccl::pin_thread_to_numa(numa_node_);
  UnifiedTask task;

  while (!stop_.load(std::memory_order_acquire)) {
    if (jring_sc_dequeue_bulk(send_unified_task_ring_, &task, 1, nullptr) ==
        1) {
      switch (task.type) {
        case TaskType::SEND_IPC:
          send_ipc(task.conn_id, task.data, task.size);
          break;
        case TaskType::WRITE_NET:
          write(task.conn_id, task.mr_id, task.data, task.size,
                task.slot_item());
          break;
        case TaskType::WRITE_IPC:
          write_ipc(task.conn_id, task.data, task.size, task.ipc_info());
          break;
        case TaskType::SEND_NET:
          send(task.conn_id, task.mr_id, task.data, task.size);
          break;
        case TaskType::SENDV: {
          TaskBatch const& batch = task.task_batch();
          std::vector<void const*> const_data_v(
              batch.const_data_v(), batch.const_data_v() + batch.num_iovs);
          std::vector<size_t> size_v(batch.size_v(),
                                     batch.size_v() + batch.num_iovs);
          std::vector<uint64_t> mr_id_v(batch.mr_id_v(),
                                        batch.mr_id_v() + batch.num_iovs);

          sendv(task.conn_id, mr_id_v, const_data_v, size_v, batch.num_iovs);
          break;
        }
        default:
          LOG(ERROR) << "Unexpected task type in send processing: "
                     << static_cast<int>(task.type);
          break;
      }
      task.self_ptr->done.store(true, std::memory_order_release);
    }
  }
}

void Endpoint::recv_proxy_thread_func() {
  uccl::pin_thread_to_numa(numa_node_);
  UnifiedTask task;

  while (!stop_.load(std::memory_order_acquire)) {
    if (jring_sc_dequeue_bulk(recv_unified_task_ring_, &task, 1, nullptr) ==
        1) {
      switch (task.type) {
        case TaskType::RECV_IPC:
          recv_ipc(task.conn_id, task.data, task.size);
          break;
        case TaskType::READ_NET:
          read(task.conn_id, task.mr_id, task.data, task.size,
               task.slot_item());
          break;
        case TaskType::READ_IPC:
          read_ipc(task.conn_id, task.data, task.size, task.ipc_info());
          break;
        case TaskType::RECV_NET:
          recv(task.conn_id, task.mr_id, task.data, task.size);
          break;
        case TaskType::RECVV: {
          TaskBatch const& batch = task.task_batch();
          std::vector<void*> data_v(batch.data_v(),
                                    batch.data_v() + batch.num_iovs);
          std::vector<size_t> size_v(batch.size_v(),
                                     batch.size_v() + batch.num_iovs);
          std::vector<uint64_t> mr_id_v(batch.mr_id_v(),
                                        batch.mr_id_v() + batch.num_iovs);

          recvv(task.conn_id, mr_id_v, data_v, size_v, batch.num_iovs);
          break;
        }
        case TaskType::SEND_NET:
        case TaskType::SEND_IPC:
        case TaskType::WRITE_NET:
        case TaskType::WRITE_IPC:
        default:
          LOG(ERROR) << "Unexpected task type in receive processing: "
                     << static_cast<int>(task.type);
          break;
      }
      task.self_ptr->done.store(true, std::memory_order_release);
    }
  }
}
