#include "engine.h"
#include "compression.h"
#include "device/ipc.h"
#include "endpoint_wrapper.h"
#include "transport_factory.h"
#include "util/debug.h"
#include "util/gpu_rt.h"
#include "util/pause.h"
#include "util/proxy_scheduler.h"
#include "util/python_signal.h"
#include "util/util.h"
#include <arpa/inet.h>
#include <cc/cc_state.h>
#include <netinet/in.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <sys/socket.h>
#include <unistd.h>

thread_local bool inside_python = false;

namespace {

// Keys borrow immutable handles from this call's IpcTransferInfo vector.
struct IpcMappings {
  explicit IpcMappings(size_t num_iovs) : num_iovs(num_iovs) {
    raw_ptrs.reserve(num_iovs);
  }

  void prepare(size_t iov, gpuIpcMemHandle_t const& handle) {
    std::string_view key(reinterpret_cast<char const*>(&handle),
                         sizeof(handle));
    if (raw_ptrs.empty()) {
      first_handle = key;
      raw_ptrs.push_back(nullptr);
      return;
    }
    if (key == first_handle) return;
    if (iov_slots.empty()) iov_slots.resize(num_iovs);
    auto [it, inserted] = other_handles.try_emplace(key, raw_ptrs.size());
    if (inserted) raw_ptrs.push_back(nullptr);
    iov_slots[iov] = it->second;
  }

  void* open(size_t iov, gpuIpcMemHandle_t const& handle) {
    auto& raw_ptr = raw_ptrs[iov_slots.empty() ? 0 : iov_slots[iov]];
    if (raw_ptr == nullptr) {
      GPU_RT_CHECK(
          gpuIpcOpenMemHandle(&raw_ptr, handle, gpuIpcMemLazyEnablePeerAccess));
    }
    return raw_ptr;
  }

  size_t num_iovs;
  std::string_view first_handle;
  std::vector<void*> raw_ptrs;
  std::vector<size_t> iov_slots;
  std::unordered_map<std::string_view, size_t> other_handles;
};

// Ordinal in this process of the GPU with PCI bus id `bdf`, or -1.
int visible_ordinal_of_bdf(std::string const& bdf) {
  int count = 0;
  if (gpuGetDeviceCount(&count) != gpuSuccess) return -1;
  char buf[64];
  for (int i = 0; i < count; ++i) {
    if (gpuDeviceGetPCIBusId(buf, sizeof(buf), i) != gpuSuccess) continue;
    if (uccl::normalize_pci_bus_id(buf) == bdf) return i;
  }
  return -1;
}

// Device whose context imports an IPC handle from `conn`. The exporter reports
// an owner only on runtimes that need it (`advertised`); otherwise CUDA/HIP
// import into the local context, which needs peer access to the owner GPU.
// Without it, fall back to the owner's context when that GPU is visible here.
int ipc_owner_device(int local, Conn const* conn, int advertised) {
  if (advertised >= 0 || conn == nullptr) return advertised;
  int owner = conn->remote_gpu_ordinal_;
  if (owner < 0 || uccl::device::peer_accessible(local, owner)) return -1;
  return owner;
}

}  // namespace

std::size_t VectorUint8Hash::operator()(std::vector<uint8_t> const& vec) const {
  std::size_t hash = vec.size();
  for (uint8_t byte : vec) {
    hash = hash * 31 + static_cast<std::size_t>(byte);
  }
  return hash;
}

ShmMsg::ShmMsg() : type(ShmMsgType::COMPLETION), completion(0) {
  std::memset(src_bdf, 0, sizeof(src_bdf));
}

// TaskBatch (declaration order: after ShmMsg, before Endpoint)
TaskBatch::TaskBatch() : num_iovs(0) {}

TaskBatch::TaskBatch(TaskBatch&& other) noexcept
    : num_iovs(other.num_iovs),
      data_ptr(std::move(other.data_ptr)),
      size_ptr(std::move(other.size_ptr)),
      mr_id_ptr(std::move(other.mr_id_ptr)),
      slot_item_ptr(std::move(other.slot_item_ptr)) {}

TaskBatch& TaskBatch::operator=(TaskBatch&& other) noexcept {
  if (this != &other) {
    num_iovs = other.num_iovs;
    data_ptr = std::move(other.data_ptr);
    size_ptr = std::move(other.size_ptr);
    mr_id_ptr = std::move(other.mr_id_ptr);
    slot_item_ptr = std::move(other.slot_item_ptr);
  }
  return *this;
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
    : type(TaskType::WRITE_NET),
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
  return type == TaskType::WRITEV || type == TaskType::READV;
}

UnifiedTask::SpecificData::SpecificData() : base{} {}

UnifiedTask::SpecificData::~SpecificData() {}

static inline bool is_retryable_post_failure(int rc) {
  if (is_cxi_transport()) return rc == UCCL_POST_TRANSIENT;
  return rc == -1;
}

static inline bool is_terminal_post_failure(int rc) {
  return rc == SendConnection::kPostError || (is_cxi_transport() && rc < 0);
}

static inline bool raw_one_sided_batch_eligible(
    std::vector<size_t> const& size_v, size_t num_iovs) {
  for (size_t i = 0; i < num_iovs; ++i) {
    if (ChunkSplitStrategy::get_message_chunk_count(size_v[i]) != 1)
      return false;
  }
  return true;
}

static inline size_t max_iov_bytes(std::vector<size_t> const& size_v,
                                   size_t num_iovs) {
  size_t max_bytes = 0;
  for (size_t i = 0; i < num_iovs; ++i) {
    max_bytes = std::max(max_bytes, size_v[i]);
  }
  return max_bytes;
}

size_t Endpoint::max_one_sided_inflight_ops() {
  size_t limit = kMaxInflightOps;
  if (is_cxi_transport()) limit = 32;

  char const* env = std::getenv("UCCL_P2P_MAX_INFLIGHT_OPS");
  if (env && env[0] != '\0') {
    char* end = nullptr;
    unsigned long requested = std::strtoul(env, &end, 10);
    if (end != env && requested > 0) {
      limit = std::min<size_t>(requested, kMaxInflightOps);
    }
  }
  return std::max<size_t>(1, limit);
}

// ShmChannel helper function
// The effective uid is part of the name so that rings of different accounts on
// a shared host cannot clash.
static inline std::string shm_ring_name(std::string const& from_bdf,
                                        std::string const& to_bdf) {
  return "/uccl_gpu_jring_u" + std::to_string(geteuid()) + "_" +
         uccl::sanitize_bdf(from_bdf) + "_" + uccl::sanitize_bdf(to_bdf);
}

// A null inbox ring is skipped while polling, so say which sender is cut off.
static inline void warn_missing_inbox_ring(std::string const& name,
                                           std::string const& sender_bdf) {
  std::cerr << "[UCCL P2P] cannot create or attach local inbox ring " << name
            << "; local connections from GPU " << sender_bdf
            << " will not reach this endpoint" << std::endl;
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

uccl::UCCLLogLevel Endpoint::parse_log_level_from_env() {
  char const* env = std::getenv("UCCL_P2P_LOG_LEVEL");
  if (!env) {
    return uccl::WARN;
  }

  if (!strcasecmp(env, "INFO")) return uccl::INFO;
  if (!strcasecmp(env, "WARN") || !strcasecmp(env, "WARNING"))
    return uccl::WARN;
  if (!strcasecmp(env, "ERROR")) return uccl::ERROR;
  if (!strcasecmp(env, "FATAL")) return uccl::FATAL;

  char* end = nullptr;
  long val = std::strtol(env, &end, 10);
  if (end != env && val >= 0 && val <= 3) {
    return static_cast<uccl::UCCLLogLevel>(val);
  }

  return uccl::WARN;
}

// -----------------------------------------------------------------------------
// The main P2P Endpoint class implementation
// -----------------------------------------------------------------------------

Endpoint::Endpoint(uint32_t const gpu_idx) : passive_accept_(false) {
  (void)get_transport_type();  // Validate before GPU setup.
  // gpu_idx is always a local CUDA device ordinal.
  int ngpus = 0;
  GPU_RT_CHECK(gpuGetDeviceCount(&ngpus));
  UCCL_CHECK(static_cast<int>(gpu_idx) < ngpus)
      << "GPU index " << gpu_idx << " out of range (visible devices: " << ngpus
      << ")";

  local_gpu_idx_ = gpu_idx;

  // Get PCI Bus ID as cross-process identity
  char bdf_buf[64];
  GPU_RT_CHECK(gpuDeviceGetPCIBusId(bdf_buf, sizeof(bdf_buf), gpu_idx));
  gpu_bus_id_ = uccl::normalize_pci_bus_id(bdf_buf);

  std::cout << "Creating Engine with GPU index: " << gpu_idx
            << " (bus_id: " << gpu_bus_id_ << ")" << std::endl;
  int n_streams = std::max(1, (int)kNumGpuRtStreams);

  // IPC streams are created per device on first use (get_ipc_streams):
  // creating them eagerly here initialized a ~0.5 GiB CUDA context on every
  // visible GPU from every process (7 foreign contexts per GPU on an 8-rank
  // node).
  ipc_streams_.resize(ngpus);
  (void)n_streams;
  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

  uccl::ucclLogger.setLogLevel(Endpoint::parse_log_level_from_env());

  ep_ = uccl_create_endpoint(local_gpu_idx_);

  std::cout << "Engine initialized for GPU " << local_gpu_idx_ << std::endl;
  engine_initialized_ = true;

  send_unified_task_ring_ =
      uccl::create_ring(sizeof(UnifiedTask*), kTaskRingSize);
  recv_unified_task_ring_ =
      uccl::create_ring(sizeof(UnifiedTask*), kTaskRingSize);

  numa_node_ = uccl_get_numa_node(ep_, local_gpu_idx_);

  send_proxy_thread_ = std::thread(&Endpoint::send_proxy_thread_func, this);
  recv_proxy_thread_ = std::thread(&Endpoint::recv_proxy_thread_func, this);

  ipc_inflight_ring_ = uccl::create_ring(sizeof(IpcInflightOp*), kTaskRingSize);
  ipc_poller_thread_ = std::thread(&Endpoint::ipc_poller_thread_func, this);

  // Initialize ShmChannel for local connections (use BDF for ring names)
  size_t elem_sz = sizeof(ShmMsg);
  size_t elem_cnt = ShmRingDefaultElemCnt;
  auto all_bdfs = uccl::enumerate_all_gpu_bdfs();
  for (auto const& sender_bdf : all_bdfs) {
    auto name = shm_ring_name(sender_bdf, gpu_bus_id_);
    ShmRingHandle handle;
    handle.shm_name = name;
    bool is_creator = false;
    handle.ring =
        uccl::create_shared_ring(name.c_str(), elem_sz, elem_cnt, handle.shm_fd,
                                 handle.shm_size, &is_creator);
    if (handle.ring == nullptr) warn_missing_inbox_ring(name, sender_bdf);
    inbox_rings_[sender_bdf] = handle;
    inbox_creators_[sender_bdf] = is_creator;
  }

  std::cout << "Endpoint initialized successfully" << std::endl;
}

std::vector<gpuStream_t>& Endpoint::get_ipc_streams(int dev) {
  std::lock_guard<std::mutex> lock(ipc_streams_mu_);
  auto& streams = ipc_streams_[dev];
  if (streams.empty()) {
    int cur_dev = 0;
    GPU_RT_CHECK(gpuGetDevice(&cur_dev));
    GPU_RT_CHECK(gpuSetDevice(dev));
    streams.resize(std::max(1, (int)kNumGpuRtStreams));
    for (auto& st : streams) {
      GPU_RT_CHECK(gpuStreamCreateWithFlags(&st, gpuStreamNonBlocking));
    }
    GPU_RT_CHECK(gpuSetDevice(cur_dev));
  }
  return streams;
}

Endpoint::Endpoint() : local_gpu_idx_(INVALID_GPU), passive_accept_(false) {
  (void)get_transport_type();  // Validate before GPU setup.
  std::cout << "Creating Engine" << std::endl;
  int n_streams = std::max(1, (int)kNumGpuRtStreams);

  int ngpus = 0;
  GPU_RT_CHECK(gpuGetDeviceCount(&ngpus));

  // Save the current device before the loop so we can restore it afterwards.
  // The loop calls gpuSetDevice(i) for each GPU, leaving the active device as
  // GPU ngpus-1, which would corrupt gpu_bus_id_ and the advertised BDF.
  int cur_dev = 0;
  GPU_RT_CHECK(gpuGetDevice(&cur_dev));

  // IPC streams are created per device on first use (get_ipc_streams):
  // creating them eagerly here initialized a ~0.5 GiB CUDA context on every
  // visible GPU from every process (7 foreign contexts per GPU on an 8-rank
  // node).
  ipc_streams_.resize(ngpus);
  (void)n_streams;
  GPU_RT_CHECK(gpuSetDevice(cur_dev));

  // Get the current GPU's BDF for metadata (needed before lazy engine init).
  char bdf_buf[64];
  GPU_RT_CHECK(gpuDeviceGetPCIBusId(bdf_buf, sizeof(bdf_buf), cur_dev));
  gpu_bus_id_ = uccl::normalize_pci_bus_id(bdf_buf);

  ep_ = uccl_create_endpoint(INVALID_GPU);

  std::cout << "Endpoint initialized successfully" << std::endl;
}

Endpoint::~Endpoint() {
  std::cout << "Destroying Engine..." << std::endl;

  stop_.store(true, std::memory_order_release);
  staging_.request_stop();

  if (passive_accept_) {
    passive_accept_stop_.store(true, std::memory_order_release);
    if (passive_accept_thread_.joinable()) {
      passive_accept_thread_.join();
    }
    if (passive_accept_local_thread_.joinable()) {
      passive_accept_local_thread_.join();
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
  staging_.shutdown();

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

  for (auto& [bdf, handle] : inbox_rings_) {
    auto it = inbox_creators_.find(bdf);
    if (it != inbox_creators_.end() && it->second)
      uccl::destroy_shared_ring(handle.shm_name.c_str(), handle.ring,
                                handle.shm_fd, handle.shm_size);
    else
      uccl::detach_shared_ring(handle.ring, handle.shm_fd, handle.shm_size);
  }

  {
    std::unique_lock<std::shared_mutex> lock(mr_mu_);
    // Deregister MRs the caller did not destroy: an MR left open keeps the
    // transport's domain busy, so it could not be closed at teardown. The proxy
    // threads were joined above, so the registry holds the last reference and
    // clearing it releases each registration.
    mr_id_to_mr_.clear();
  }

  std::cout << "Engine destroyed" << std::endl;
}

void Endpoint::stop_accept() { uccl_stop_accept(ep_); }

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

  // RDMA connect does not populate sock_fd; only peer_id indicates rejection
  // across transports (including NCCL protocol-version mismatches).
  if (uccl_conn_id.peer_id == UINT64_MAX) {
    return false;
  }

  // Store the connection ID.
  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    conn_id_to_conn_[conn_id] = new Conn{
        conn_id, uccl_conn_id, ip_addr, static_cast<uint16_t>(remote_port), ""};
  }
  return true;
}

std::vector<uint8_t> Endpoint::get_metadata() {
  std::string ip_str = uccl::get_oob_ip();
  uint16_t port = get_p2p_listen_port(ep_);

  bool is_ipv6 = ip_str.find(':') != std::string::npos;
  size_t ip_len = is_ipv6 ? 16 : 4;

  // 2 bytes port + 1 byte BDF length + N bytes BDF string
  uint8_t bdf_len = static_cast<uint8_t>(gpu_bus_id_.size());
  size_t total_len = ip_len + 2 + 1 + bdf_len;
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

  // Copy BDF string (1 byte length + string data)
  metadata[ip_len + 2] = bdf_len;
  std::memcpy(metadata.data() + ip_len + 3, gpu_bus_id_.data(), bdf_len);

  return metadata;
}
std::tuple<std::string, uint16_t, std::string> Endpoint::parse_metadata(
    std::vector<uint8_t> const& metadata) {
  // Minimum: 4 (IPv4) + 2 (port) + 1 (bdf_len) = 7
  if (metadata.size() < 7) {
    throw std::runtime_error("Metadata too short: " +
                             std::to_string(metadata.size()));
  }

  std::string ip_str;
  size_t ip_len;

  // Determine IP version from metadata length and BDF field
  // Try IPv4 first: 4 bytes IP + 2 bytes port + 1 byte bdf_len + N bytes bdf
  uint8_t bdf_len_v4 = metadata[6];
  if (metadata.size() == static_cast<size_t>(4 + 2 + 1 + bdf_len_v4)) {
    // IPv4
    ip_len = 4;
    char buf[INET_ADDRSTRLEN];
    if (inet_ntop(AF_INET, metadata.data(), buf, sizeof(buf)) == nullptr) {
      throw std::runtime_error("Failed to parse IPv4 address from metadata");
    }
    ip_str = buf;
  } else {
    // IPv6: 16 bytes IP + 2 bytes port + 1 byte bdf_len + N bytes bdf
    ip_len = 16;
    if (metadata.size() < 19) {
      throw std::runtime_error("Metadata too short for IPv6: " +
                               std::to_string(metadata.size()));
    }
    uint8_t bdf_len_v6 = metadata[18];
    if (metadata.size() != static_cast<size_t>(19 + bdf_len_v6)) {
      throw std::runtime_error("Metadata size mismatch for IPv6: got " +
                               std::to_string(metadata.size()) + ", expected " +
                               std::to_string(19 + bdf_len_v6));
    }
    char buf[INET6_ADDRSTRLEN];
    if (inet_ntop(AF_INET6, metadata.data(), buf, sizeof(buf)) == nullptr) {
      throw std::runtime_error("Failed to parse IPv6 address from metadata");
    }
    ip_str = buf;
  }

  uint16_t net_port;
  std::memcpy(&net_port, metadata.data() + ip_len, 2);
  uint16_t port = ntohs(net_port);

  uint8_t bdf_len = metadata[ip_len + 2];
  std::string gpu_bdf;
  if (bdf_len > 0) {
    gpu_bdf = std::string(
        reinterpret_cast<char const*>(metadata.data() + ip_len + 3), bdf_len);
  }

  return std::make_tuple(ip_str, port, gpu_bdf);
}

bool Endpoint::accept(std::string& ip_addr, int& remote_gpu_idx,
                      uint64_t& conn_id) {
  std::cout << "Waiting to accept incoming connection..." << std::endl;

  // For demo purposes, simulate accepted connection
  conn_id = next_conn_id_.fetch_add(1);

  // initialize_engine() runs from reg(), so accept may legitimately arrive
  // first. The wait must also observe stop_accept(): an endpoint that never
  // registers memory would otherwise spin here forever while ~Endpoint()
  // blocks in join().
  while (!engine_initialized_) {
    if (stop_.load(std::memory_order_acquire) ||
        passive_accept_stop_.load(std::memory_order_acquire) ||
        uccl_accept_stopped(ep_)) {
      return false;
    }
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
      uccl_stop_accept(ep_);
    }
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  ConnID uccl_conn_id = uccl_conn_id_future.get();

  // Return if Conn ID is invalid
  if (uccl_conn_id.sock_fd < 0 || uccl_conn_id.peer_id == UINT64_MAX) {
    return false;
  }

  // Store the connection ID.
  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    conn_id_to_conn_[conn_id] = new Conn{conn_id, uccl_conn_id, ip_addr, 0, ""};
  }

  return true;
}

bool Endpoint::reg(void const* data, size_t size, uint64_t& mr_id,
                   FloatType float_type) {
  mr_id = next_mr_id_.fetch_add(1);

  if (!engine_initialized_) {
    int idx = uccl::get_dev_idx((void*)data);
    if (idx != -1) {
      local_gpu_idx_ = idx;
    } else {
      local_gpu_idx_ = 0;
    }
    // Get PCI Bus ID for cross-process identity
    char bdf_buf[64];
    GPU_RT_CHECK(
        gpuDeviceGetPCIBusId(bdf_buf, sizeof(bdf_buf), local_gpu_idx_));
    gpu_bus_id_ = uccl::normalize_pci_bus_id(bdf_buf);
    initialize_engine();
    engine_initialized_ = true;
  }

  P2PMhandle* mhandle = new P2PMhandle();
  mhandle->compress_ctx = make_compress_ctx(float_type);
  if (!uccl_regmr(ep_, const_cast<void*>(data), size, mhandle)) {
    delete mhandle;
    return false;
  }
  {
    std::unique_lock<std::shared_mutex> lock(mr_mu_);
    mr_id_to_mr_[mr_id] = make_mr(mr_id, mhandle);
  }

  return true;
}

bool Endpoint::reg_compress_only(void const* data, size_t size, uint64_t& mr_id,
                                 FloatType float_type) {
  mr_id = next_mr_id_.fetch_add(1);

  if (!engine_initialized_) {
    int idx = uccl::get_dev_idx((void*)data);
    if (idx != -1) {
      local_gpu_idx_ = idx;
    } else {
      local_gpu_idx_ = 0;
    }
    // Get PCI Bus ID for cross-process identity
    char bdf_buf[64];
    GPU_RT_CHECK(
        gpuDeviceGetPCIBusId(bdf_buf, sizeof(bdf_buf), local_gpu_idx_));
    gpu_bus_id_ = uccl::normalize_pci_bus_id(bdf_buf);
    initialize_engine();
    engine_initialized_ = true;
  }

  P2PMhandle* mhandle = new P2PMhandle();
  mhandle->compress_ctx = make_compress_ctx(float_type);
  mhandle->compress_only = true;
  mhandle->registered_addr = data;
  mhandle->registered_len = size;
  if (!uccl_prepare_compress_ctx(ep_, const_cast<void*>(data), size,
                                 mhandle->compress_ctx)) {
    delete mhandle;
    return false;
  }
  {
    std::unique_lock<std::shared_mutex> lock(mr_mu_);
    mr_id_to_mr_[mr_id] = make_mr(mr_id, mhandle);
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
      local_gpu_idx_ = idx;
    } else {
      local_gpu_idx_ = 0;
    }
    char bdf_buf[64];
    GPU_RT_CHECK(
        gpuDeviceGetPCIBusId(bdf_buf, sizeof(bdf_buf), local_gpu_idx_));
    gpu_bus_id_ = uccl::normalize_pci_bus_id(bdf_buf);
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
      mr_id_to_mr_[id] = make_mr(id, mhandle);
    }
    mr_id_v[i] = id;
  }
  return true;
}

std::shared_ptr<MR> Endpoint::make_mr(uint64_t id, P2PMhandle* handle) {
  // A cooperative request pins the registration until its last DMA completes.
  // Capture the transport by value so a deferred release never accesses a
  // destroyed Endpoint or transport. No registry lock is held during DMA.
  return std::shared_ptr<MR>(new MR{id, handle}, [transport = ep_](MR* mr) {
    // Endpoint teardown frees the handle itself and leaves this null.
    if (mr->mhandle_ != nullptr) {
      mr->mhandle_->compress_ctx.reset();
      uccl_deregmr(transport, mr->mhandle_);
      delete mr->mhandle_;
    }
    delete mr;
  });
}

bool Endpoint::dereg(uint64_t mr_id) {
  std::shared_ptr<MR> mr;
  {
    std::unique_lock<std::shared_mutex> lock(mr_mu_);
    auto it = mr_id_to_mr_.find(mr_id);
    if (it == mr_id_to_mr_.end()) {
      std::cerr << "[dereg] Error: Invalid mr_id " << mr_id << std::endl;
      return false;
    }
    mr = std::move(it->second);
    mr_id_to_mr_.erase(mr_id);
  }
  // Immediate for unpinned registrations, otherwise the last active request
  // performs the release. The caller still owns the underlying data buffer.
  mr.reset();
  return true;
}

bool Endpoint::read(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
                    FifoItem const& slot_item) {
  UCCL_DCHECK(size <= 0xffffffff) << "size must be < 4 GB";
  if constexpr (uccl::device::Backend::staged_network)
    return staging_.read(conn_id, dst, size, slot_item);
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
  if (unlikely(mhandle->compress_only)) {
    std::cerr << "[read] Error: compress_only mr_id " << mr_id
              << " cannot be a read destination" << std::endl;
    return false;
  }

  UcclRequest ureq = {};
  FifoItem curr_slot_item = slot_item;
  curr_slot_item.size = size;

  int rc;
  do {
    rc = uccl_read_async(ep_, conn, mhandle, dst, size, curr_slot_item, &ureq);
  } while (is_retryable_post_failure(rc));
  if (is_terminal_post_failure(rc)) {
    UCCL_LOG(ERROR) << "read failed to post: rc=" << rc;
    return false;
  }

  bool done = false;
  while (!done) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
    uccl_drive_recv(ep_);
    if (uccl_poll_ureq_once(ep_, &ureq)) {
      done = true;
    }
  }

  return true;
}

bool Endpoint::read_async(uint64_t conn_id, uint64_t mr_id, void* dst,
                          size_t size, FifoItem const& slot_item,
                          uint64_t* transfer_id) {
  *transfer_id = 0;
  if (size <= kDirectAsyncNetThreshold) {
    auto* conn = get_conn(conn_id);
    if (unlikely(conn == nullptr)) {
      std::cerr << "[read_async] Error: Invalid conn_id " << conn_id
                << std::endl;
      return false;
    }
    P2PMhandle* mhandle = get_mhandle(mr_id);
    if (unlikely(mhandle == nullptr)) {
      std::cerr << "[read_async] Error: Invalid mr_id " << mr_id << std::endl;
      return false;
    }
    if (unlikely(mhandle->compress_only)) {
      std::cerr << "[read_async] Error: compress_only mr_id " << mr_id
                << " cannot be a read destination" << std::endl;
      return false;
    }

    UcclRequest ureq = {};
    FifoItem curr_slot_item = slot_item;
    curr_slot_item.size = size;
    int rc;
    do {
      rc =
          uccl_read_async(ep_, conn, mhandle, dst, size, curr_slot_item, &ureq);
    } while (is_retryable_post_failure(rc));
    if (is_terminal_post_failure(rc)) {
      UCCL_LOG(ERROR) << "read_async failed to post: rc=" << rc;
      return false;
    }

    auto* status = new TransferStatus();
    status->poll_net_ureq = true;
    status->ureq = ureq;
    *transfer_id = reinterpret_cast<uint64_t>(status);
    return true;
  }

  {
    P2PMhandle* mhandle = get_mhandle(mr_id);
    if (unlikely(mhandle != nullptr && mhandle->compress_only)) {
      std::cerr << "[read_async] Error: compress_only mr_id " << mr_id
                << " cannot be a read destination" << std::endl;
      return false;
    }
  }
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
  recv_proxy_adaptive_sleeper_.maybe_wake_proxy_thread();

  return true;
}

bool Endpoint::readv(uint64_t conn_id, std::vector<uint64_t> const& mr_id_v,
                     std::vector<void*> const& dst_v,
                     std::vector<size_t> const& size_v,
                     std::vector<FifoItem> const& slot_item_v,
                     size_t num_iovs) {
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[readv] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  // Pre-resolve all mhandles once to avoid per-iov mutex + map lookup.
  std::vector<P2PMhandle*> mhandle_v(num_iovs);
  for (size_t i = 0; i < num_iovs; ++i) {
    mhandle_v[i] = get_mhandle(mr_id_v[i]);
    if (unlikely(mhandle_v[i] == nullptr)) {
      std::cerr << "[readv] Error: Invalid mr_id " << mr_id_v[i] << std::endl;
      return false;
    }
    if (unlikely(mhandle_v[i]->compress_only)) {
      std::cerr << "[readv] Error: compress_only mr_id " << mr_id_v[i]
                << " cannot be a read destination" << std::endl;
      return false;
    }
  }

  // Enable doorbell-batched posting for the duration of this call.
  struct BatchGuard {
    BatchGuard() { g_uccl_batch_post = true; }
    ~BatchGuard() { g_uccl_batch_post = false; }
  } batch_guard;

  // Resolve the send group once so per-slot completion checks avoid the
  // per-call shared_lock + map.find().
  SendConnection* send_group =
      uccl_resolve_send_group(ep_, conn->uccl_conn_id_.peer_id);

  if (send_group != nullptr && raw_one_sided_batch_eligible(size_v, num_iovs) &&
      send_group->can_use_raw_one_sided_batch(
          SendType::Read, max_iov_bytes(size_v, num_iovs))) {
    return run_raw_one_sided_pipeline(send_group, SendType::Read, mhandle_v,
                                      dst_v, size_v, slot_item_v, num_iovs);
  }

  UcclRequest ureq[kMaxInflightOps] = {};
  bool active[kMaxInflightOps] = {false};
  size_t const max_inflight_ops = max_one_sided_inflight_ops();

  size_t next_iov = 0;
  size_t num_completed = 0;
  size_t num_inflight = 0;
  bool post_failed = false;

  while ((!post_failed && num_completed < num_iovs) || num_inflight > 0) {
    while (!post_failed && next_iov < num_iovs &&
           num_inflight < max_inflight_ops) {
      size_t slot = 0;
      while (slot < kMaxInflightOps && active[slot]) {
        slot++;
      }
      UCCL_DCHECK(slot < kMaxInflightOps);

      auto rc =
          send_group != nullptr
              ? uccl_read_async_on_group(send_group, conn, mhandle_v[next_iov],
                                         dst_v[next_iov], size_v[next_iov],
                                         slot_item_v[next_iov], &ureq[slot])
              : uccl_read_async(ep_, conn, mhandle_v[next_iov], dst_v[next_iov],
                                size_v[next_iov], slot_item_v[next_iov],
                                &ureq[slot]);
      if (is_retryable_post_failure(rc)) {
        if (is_cxi_transport() && num_inflight == 0) {
          uccl_drive_send(ep_);
          uccl_drive_recv(ep_);
          std::this_thread::yield();
          continue;
        }
        break;
      }
      if (is_terminal_post_failure(rc)) {
        UCCL_LOG(ERROR) << "readv failed to post iov " << next_iov
                        << ": rc=" << rc;
        // Drain earlier iovs before callers can reuse their buffers.
        post_failed = true;
        break;
      }
      active[slot] = true;
      next_iov++;
      num_inflight++;
    }

    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;

    // Flush batched WRs in one doorbell per channel, then drive send once.
    // (readv issues RDMA reads via the send path, so drive send here.)
    if (num_inflight > 0) {
      uccl_flush_send(ep_);
      uccl_drive_send(ep_);
      uccl_drive_recv(ep_);
    }

    if (send_group != nullptr) {
      for (size_t slot = 0; slot < kMaxInflightOps; slot++) {
        if (!active[slot]) continue;
        if (uccl_check_wr_fast(send_group, ureq[slot].engine_idx)) {
          active[slot] = false;
          num_completed++;
          num_inflight--;
        }
      }
    } else {
      for (size_t slot = 0; slot < kMaxInflightOps; slot++) {
        if (!active[slot]) continue;
        if (uccl_check_ureq_once(ep_, &ureq[slot])) {
          active[slot] = false;
          num_completed++;
          num_inflight--;
        }
      }
    }
  }

  return !post_failed;
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
  recv_proxy_adaptive_sleeper_.maybe_wake_proxy_thread();

  return true;
}

bool Endpoint::run_raw_one_sided_pipeline(
    SendConnection* send_group, SendType send_type,
    std::vector<P2PMhandle*> const& mhandle_v,
    std::vector<void*> const& local_addr_v, std::vector<size_t> const& size_v,
    std::vector<FifoItem> const& slot_item_v, size_t num_iovs) {
  std::array<SendConnection::RawBatchWait, kMaxInflightOps> pending_waits{};
  size_t const max_inflight_ops = max_one_sided_inflight_ops();
  size_t next_iov = 0;
  size_t num_inflight = 0;
  size_t num_pending = 0;
  bool post_ok = true;

  for (;;) {
    size_t const batch_iovs =
        std::min(max_inflight_ops - num_inflight, num_iovs - next_iov);
    if (post_ok && batch_iovs != 0) {
      std::array<SendConnection::OneSidedBatchOp, kMaxInflightOps> ops{};
      int64_t wr_ids[kMaxInflightOps];
      for (size_t i = 0; i < batch_iovs; ++i) {
        size_t const iov = next_iov + i;
        ops[i].local_addr = local_addr_v[iov];
        ops[i].size = size_v[iov];
        ops[i].local_mr_array = &mhandle_v[iov]->mr_array;
        ops[i].slot_item = &slot_item_v[iov];
      }

      // Each wait covers at least one IOV, so the tail can hold this window.
      size_t num_waits = 0;
      post_ok = send_group->post_write_or_read_batch(
                    send_type, ops.data(), batch_iovs, wr_ids,
                    pending_waits.data() + num_pending, &num_waits) &&
                num_waits != 0;
      if (post_ok) {
        num_pending += num_waits;
        next_iov += batch_iovs;
        num_inflight += batch_iovs;
      }
    }

    // A rejected window leaves earlier DMA in flight until its waits retire.
    if (num_pending == 0) return post_ok;

    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
    uccl_drive_send(ep_);
    auto end = std::remove_if(pending_waits.begin(),
                              pending_waits.begin() + num_pending,
                              [&](SendConnection::RawBatchWait const& wait) {
                                if (!uccl_check_wr_fast(send_group, wait.wr_id))
                                  return false;
                                UCCL_DCHECK(num_inflight >= wait.iov_count);
                                num_inflight -= wait.iov_count;
                                return true;
                              });
    num_pending = end - pending_waits.begin();
  }
}

bool Endpoint::write(uint64_t conn_id, uint64_t mr_id, void* src, size_t size,
                     FifoItem const& slot_item) {
  UCCL_DCHECK(size <= 0xffffffff) << "size must be < 4 GB";
  if constexpr (uccl::device::Backend::staged_network)
    return staging_.write(conn_id, src, size, slot_item);
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
  SendConnection* send_group =
      uccl_resolve_send_group(ep_, conn->uccl_conn_id_.peer_id);
  UcclRequest ureq = {};
  FifoItem curr_slot_item = slot_item;
  curr_slot_item.size = size;

  int rc;
  do {
    rc = send_group != nullptr
             ? uccl_write_async_on_group(send_group, conn, mhandle, src, size,
                                         curr_slot_item, &ureq)
             : uccl_write_async(ep_, conn, mhandle, src, size, curr_slot_item,
                                &ureq);
  } while (is_retryable_post_failure(rc));
  if (is_terminal_post_failure(rc)) {
    UCCL_LOG(ERROR) << "write failed to post: rc=" << rc;
    return false;
  }

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
  *transfer_id = 0;
  if (size <= kDirectAsyncNetThreshold) {
    auto* conn = get_conn(conn_id);
    if (unlikely(conn == nullptr)) {
      std::cerr << "[write_async] Error: Invalid conn_id " << conn_id
                << std::endl;
      return false;
    }
    P2PMhandle* mhandle = get_mhandle(mr_id);
    if (unlikely(mhandle == nullptr)) {
      std::cerr << "[write_async] Error: Invalid mr_id " << mr_id << std::endl;
      return false;
    }

    UcclRequest ureq = {};
    FifoItem curr_slot_item = slot_item;
    curr_slot_item.size = size;
    int rc;
    do {
      rc = uccl_write_async(ep_, conn, mhandle, src, size, curr_slot_item,
                            &ureq);
    } while (is_retryable_post_failure(rc));
    if (is_terminal_post_failure(rc)) {
      UCCL_LOG(ERROR) << "write_async failed to post: rc=" << rc;
      return false;
    }

    auto* status = new TransferStatus();
    status->poll_net_ureq = true;
    status->ureq = ureq;
    *transfer_id = reinterpret_cast<uint64_t>(status);
    return true;
  }

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
  send_proxy_adaptive_sleeper_.maybe_wake_proxy_thread();

  return true;
}

bool Endpoint::writev(uint64_t conn_id, std::vector<uint64_t> const& mr_id_v,
                      std::vector<void*> const& src_v,
                      std::vector<size_t> const& size_v,
                      std::vector<FifoItem> const& slot_item_v,
                      size_t num_iovs) {
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[writev] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  // Pre-resolve all mhandles once to avoid per-iov mutex + map lookup.
  std::vector<P2PMhandle*> mhandle_v(num_iovs);
  for (size_t i = 0; i < num_iovs; ++i) {
    mhandle_v[i] = get_mhandle(mr_id_v[i]);
    if (unlikely(mhandle_v[i] == nullptr)) {
      std::cerr << "[writev] Error: Invalid mr_id " << mr_id_v[i] << std::endl;
      return false;
    }
  }

  // Enable doorbell-batched posting for the duration of this call. All
  // submit_request() calls from this thread accumulate WRs per channel; we
  // flush once per outer pass before polling.
  struct BatchGuard {
    BatchGuard() { g_uccl_batch_post = true; }
    ~BatchGuard() { g_uccl_batch_post = false; }
  } batch_guard;

  // Resolve the send group once so per-slot completion checks avoid the
  // per-call shared_lock + map.find().
  SendConnection* send_group =
      uccl_resolve_send_group(ep_, conn->uccl_conn_id_.peer_id);

  // Validate every compress_only iov before posting anything: it must send
  // exactly its registered buffer and be guaranteed to take the compression
  // path (it has no user MR to fall back on).
  for (size_t i = 0; i < num_iovs; ++i) {
    P2PMhandle const* mh = mhandle_v[i];
    if (likely(!mh->compress_only)) continue;
    if (src_v[i] != mh->registered_addr || size_v[i] != mh->registered_len) {
      std::cerr << "[writev] Error: compress_only mr_id " << mr_id_v[i]
                << " must send exactly its registered buffer" << std::endl;
      return false;
    }
    if (send_group == nullptr ||
        !send_group->will_compress_write(SendType::Write, size_v[i],
                                         mh->compress_ctx)) {
      std::cerr << "[writev] Error: compress_only mr_id " << mr_id_v[i]
                << " would not take the compression path" << std::endl;
      return false;
    }
  }

  if (send_group != nullptr && raw_one_sided_batch_eligible(size_v, num_iovs) &&
      send_group->can_use_raw_one_sided_batch(
          SendType::Write, max_iov_bytes(size_v, num_iovs))) {
    return run_raw_one_sided_pipeline(send_group, SendType::Write, mhandle_v,
                                      src_v, size_v, slot_item_v, num_iovs);
  }

  UcclRequest ureq[kMaxInflightOps] = {};
  bool active[kMaxInflightOps] = {false};
  size_t const max_inflight_ops = max_one_sided_inflight_ops();

  size_t next_iov = 0;
  size_t num_completed = 0;
  size_t num_inflight = 0;
  bool post_failed = false;

  while ((!post_failed && num_completed < num_iovs) || num_inflight > 0) {
    while (!post_failed && next_iov < num_iovs &&
           num_inflight < max_inflight_ops) {
      size_t slot = 0;
      while (slot < kMaxInflightOps && active[slot]) {
        slot++;
      }
      UCCL_DCHECK(slot < kMaxInflightOps);

      auto rc =
          send_group != nullptr
              ? uccl_write_async_on_group(send_group, conn, mhandle_v[next_iov],
                                          src_v[next_iov], size_v[next_iov],
                                          slot_item_v[next_iov], &ureq[slot])
              : uccl_write_async(ep_, conn, mhandle_v[next_iov],
                                 src_v[next_iov], size_v[next_iov],
                                 slot_item_v[next_iov], &ureq[slot]);
      if (is_retryable_post_failure(rc)) {
        if (is_cxi_transport() && num_inflight == 0) {
          uccl_drive_send(ep_);
          uccl_drive_recv(ep_);
          std::this_thread::yield();
          continue;
        }
        break;
      }
      if (is_terminal_post_failure(rc)) {
        UCCL_LOG(ERROR) << "writev failed to post iov " << next_iov
                        << ": rc=" << rc;
        // Drain earlier iovs before callers can reuse their buffers.
        post_failed = true;
        break;
      }
      active[slot] = true;
      next_iov++;
      num_inflight++;
    }

    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;

    // Flush all batched WRs in one doorbell per channel, then drive the send
    // poller once for this pass.
    if (num_inflight > 0) {
      uccl_flush_send(ep_);
      uccl_drive_send(ep_);
    }

    if (send_group != nullptr) {
      // Fast path: avoid per-slot mutex + map lookup.
      for (size_t slot = 0; slot < kMaxInflightOps; slot++) {
        if (!active[slot]) continue;
        if (uccl_check_wr_fast(send_group, ureq[slot].engine_idx)) {
          active[slot] = false;
          num_completed++;
          num_inflight--;
        }
      }
    } else {
      for (size_t slot = 0; slot < kMaxInflightOps; slot++) {
        if (!active[slot]) continue;
        if (uccl_check_ureq_once(ep_, &ureq[slot])) {
          active[slot] = false;
          num_completed++;
          num_inflight--;
        }
      }
    }
  }

  return !post_failed;
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
  send_proxy_adaptive_sleeper_.maybe_wake_proxy_thread();

  return true;
}

bool Endpoint::advertise(uint64_t mr_id, void* addr, size_t len,
                         char* out_buf) {
  if constexpr (uccl::device::Backend::staged_network)
    return staging_.advertise(mr_id, addr, len, out_buf);
  auto mhandle = get_mhandle(mr_id);
  if (unlikely(mhandle == nullptr)) {
    std::cerr << "[advertise] Error: Invalid mr_id " << mr_id << std::endl;
    return false;
  }
  if (unlikely(mhandle->compress_only)) {
    std::cerr << "[advertise] Error: compress_only mr_id " << mr_id
              << " cannot be advertised as a write/read destination"
              << std::endl;
    return false;
  }
  if (prepare_fifo_metadata(ep_, mhandle, addr, len, out_buf) == -1)
    return false;
  return true;
}

bool Endpoint::advertisev(std::vector<uint64_t> mr_id_v,
                          std::vector<void*> addr_v, std::vector<size_t> len_v,
                          std::vector<char*> out_buf_v, size_t num_iovs) {
  std::vector<P2PMhandle*> mhandles(num_iovs);

  for (int i = 0; i < num_iovs; i++) {
    mhandles[i] = get_mhandle(mr_id_v[i]);
    if (unlikely(mhandles[i] == nullptr)) {
      std::cerr << "[advertisev] Error: Invalid mr_id " << mr_id_v[i]
                << std::endl;
      return false;
    }
    if (unlikely(mhandles[i]->compress_only)) {
      std::cerr << "[advertisev] Error: compress_only mr_id " << mr_id_v[i]
                << " cannot be advertised as a write/read destination"
                << std::endl;
      return false;
    }
  }

  for (size_t i = 0; i < num_iovs; ++i) {
    if (prepare_fifo_metadata(ep_, mhandles[i], addr_v[i], len_v[i],
                              out_buf_v[i]) == -1) {
      return false;
    }
  }
  return true;
}

bool Endpoint::connect_local(std::string const& remote_gpu_bdf,
                             uint64_t& conn_id, bool same_process) {
  constexpr int kMaxRetry = 20;
  constexpr int kRetrySleepMs = 10;

  Conn* conn = new Conn;
  conn->remote_gpu_bdf_ = remote_gpu_bdf;
  conn->remote_gpu_ordinal_ = visible_ordinal_of_bdf(remote_gpu_bdf);

  if (same_process) {
    // Same process: use inbox_rings_ directly, no shm attach needed.
    auto it = inbox_rings_.find(remote_gpu_bdf);
    if (it != inbox_rings_.end()) {
      conn->remote_inbox_ = it->second;
    }
    conn->shm_attached_ = false;
  } else {
    // cross process: attach remote inbox via shared memory
    conn->remote_inbox_.shm_name = shm_ring_name(gpu_bus_id_, remote_gpu_bdf);

    jring_t* ring = nullptr;
    int attempts = 0;
    int attach_errno = 0;
    while (attempts < kMaxRetry) {
      errno = 0;
      ring = uccl::attach_shared_ring(conn->remote_inbox_.shm_name.c_str(),
                                      conn->remote_inbox_.shm_fd,
                                      conn->remote_inbox_.shm_size);
      attempts++;
      if (ring != nullptr) {
        break;
      }
      attach_errno = errno;
      // A permission error does not go away; only retry while the peer may
      // still be creating its ring.
      if (attach_errno == EACCES) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(kRetrySleepMs));
    }

    if (ring == nullptr) {
      std::cout << "connect_local failed: cannot attach remote inbox ring "
                << conn->remote_inbox_.shm_name << " after " << attempts
                << (attempts == 1 ? " attempt: " : " attempts: ")
                << (attach_errno != 0 ? strerror(attach_errno)
                                      : "ring not ready")
                << std::endl;
      delete conn;
      return false;
    }

    conn->remote_inbox_.ring = ring;
    conn->shm_attached_ = true;
  }

  // allocate conn_id
  conn_id = next_conn_id_.fetch_add(1);
  conn->conn_id_ = conn_id;

  if (!same_process) {
    // Cross-process: send CONNECT handshake via shm ring.
    ShmMsg msg;
    std::strncpy(msg.src_bdf, gpu_bus_id_.c_str(), sizeof(msg.src_bdf) - 1);
    msg.src_bdf[sizeof(msg.src_bdf) - 1] = '\0';
    msg.type = ShmMsgType::CONNECT;
    msg.completion = 0;
    shm_ring_send(conn->remote_inbox_.ring, msg);
  }

  {
    std::unique_lock lock(conn_mu_);
    ConnID dummy{nullptr, 0, 0, 0};
    conn->uccl_conn_id_ = dummy;
    conn->is_local_ = true;
    conn_id_to_conn_[conn_id] = conn;
  }

  return true;
}

bool Endpoint::accept_local(std::string& remote_gpu_bdf, uint64_t& conn_id) {
  std::cout << "Waiting for local CONNECT..." << std::endl;

  // recv CONNECT
  ShmMsg msg;
  while (true) {
    for (auto& [bdf, handle] : inbox_rings_) {
      if (bdf == gpu_bus_id_) continue;
      if (handle.ring == nullptr) continue;

      if (!jring_empty(handle.ring)) {
        shm_ring_recv(handle.ring, msg);
        goto got;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
got:
  UCCL_CHECK(msg.type == ShmMsgType::CONNECT);

  remote_gpu_bdf = std::string(msg.src_bdf);
  conn_id = next_conn_id_.fetch_add(1);

  Conn* conn = new Conn;
  conn->conn_id_ = conn_id;
  conn->remote_gpu_bdf_ = remote_gpu_bdf;
  conn->remote_gpu_ordinal_ = visible_ordinal_of_bdf(remote_gpu_bdf);

  if (remote_gpu_bdf == gpu_bus_id_) {
    // same GPU: no attach, bind inbox directly
    auto it = inbox_rings_.find(remote_gpu_bdf);
    if (it != inbox_rings_.end()) {
      conn->remote_inbox_ = it->second;
    }
    conn->shm_attached_ = false;
  } else {
    // cross GPU: attach client inbox
    conn->remote_inbox_.shm_name = shm_ring_name(gpu_bus_id_, remote_gpu_bdf);
    conn->remote_inbox_.ring = uccl::attach_shared_ring(
        conn->remote_inbox_.shm_name.c_str(), conn->remote_inbox_.shm_fd,
        conn->remote_inbox_.shm_size);
    conn->shm_attached_ = true;
  }

  {
    std::unique_lock lock(conn_mu_);
    ConnID dummy{nullptr, 0, 0, 0};
    conn->uccl_conn_id_ = dummy;
    conn->is_local_ = true;
    conn_id_to_conn_[conn_id] = conn;
  }

  return true;
}

bool Endpoint::write_ipc(uint64_t conn_id, void const* data, size_t size,
                         IpcTransferInfo const& info) {
  UCCL_CHECK(data != nullptr) << "write_ipc: data pointer is null!";

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

  int ipc_dev = uccl::device::ipc_open_device(
      local_gpu_idx_, ipc_owner_device(local_gpu_idx_, conn, info.gpu_idx));
  void* raw_dst_ptr = nullptr;
  GPU_RT_CHECK(gpuSetDevice(ipc_dev));
  GPU_RT_CHECK(gpuIpcOpenMemHandle(&raw_dst_ptr, info.handle,
                                   gpuIpcMemLazyEnablePeerAccess));

  // Calculate destination pointer with offset
  void* dst_ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(raw_dst_ptr) + info.offset);

  // Perform the memory copy using multiple streams for better performance
  std::vector<gpuStream_t>& dst_streams = get_ipc_streams(ipc_dev);
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

    GPU_RT_CHECK(uccl::device::ipc_copy_async(
        chunk_dst_ptr, ipc_dev, chunk_data, local_gpu_idx_, copy_size, is_host,
        memcpy_kind, dst_streams[i]));
  }

  for (int i = 0; i < num_streams; ++i) {
    GPU_RT_CHECK(gpuStreamSynchronize(dst_streams[i]));
  }

  // Close the IPC memory handle
  GPU_RT_CHECK(gpuIpcCloseMemHandle(raw_dst_ptr));

  return true;
}

bool Endpoint::read_ipc(uint64_t conn_id, void* data, size_t size,
                        IpcTransferInfo const& info) {
  UCCL_CHECK(data != nullptr) << "read_ipc: data pointer is null!";

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

  int ipc_dev = uccl::device::ipc_open_device(
      local_gpu_idx_, ipc_owner_device(local_gpu_idx_, conn, info.gpu_idx));
  void* raw_src_ptr = nullptr;
  GPU_RT_CHECK(gpuSetDevice(ipc_dev));
  GPU_RT_CHECK(gpuIpcOpenMemHandle(&raw_src_ptr, info.handle,
                                   gpuIpcMemLazyEnablePeerAccess));

  // Calculate source pointer with offset
  void* src_ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(raw_src_ptr) + info.offset);

  // Perform the memory copy using multiple streams for better performance
  std::vector<gpuStream_t>& src_streams = get_ipc_streams(ipc_dev);
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

    GPU_RT_CHECK(uccl::device::ipc_copy_async(
        chunk_data, local_gpu_idx_, chunk_src_ptr, ipc_dev, copy_size, is_host,
        memcpy_kind, src_streams[i]));
  }

  for (int i = 0; i < num_streams; ++i) {
    GPU_RT_CHECK(gpuStreamSynchronize(src_streams[i]));
  }

  // Close the IPC memory handle
  GPU_RT_CHECK(gpuIpcCloseMemHandle(raw_src_ptr));

  return true;
}

bool Endpoint::writev_ipc(uint64_t conn_id, std::vector<void const*> data_v,
                          std::vector<size_t> size_v,
                          std::vector<IpcTransferInfo> info_v,
                          size_t num_iovs) {
  UCCL_CHECK_EQ(data_v.size(), num_iovs) << "writev_ipc: data_v size mismatch";
  UCCL_CHECK_EQ(size_v.size(), num_iovs) << "writev_ipc: size_v size mismatch";
  UCCL_CHECK_EQ(info_v.size(), num_iovs) << "writev_ipc: info_v size mismatch";

  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[writev_ipc] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  if (num_iovs == 0) return true;

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  int ipc_dev_w = uccl::device::ipc_open_device(
      local_gpu_idx_,
      ipc_owner_device(local_gpu_idx_, conn, info_v[0].gpu_idx));
  GPU_RT_CHECK(gpuSetDevice(ipc_dev_w));
  std::vector<gpuStream_t>& streams = get_ipc_streams(ipc_dev_w);

  // Finish host allocations before submitting any copies.
  IpcMappings mappings(num_iovs);
  for (size_t iov = 0; iov < num_iovs; ++iov) {
    mappings.prepare(iov, info_v[iov].handle);
  }
  int active_streams = 0;
  for (size_t iov = 0; iov < num_iovs; ++iov) {
    UCCL_CHECK(data_v[iov] != nullptr)
        << "writev_ipc: data_v[" << iov << "] is null";
    void* raw_ptr = mappings.open(iov, info_v[iov].handle);
    void* dst_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(raw_ptr) + info_v[iov].offset);

    bool is_host = (uccl::get_dev_idx(const_cast<void*>(data_v[iov])) == -1);
    auto memcpy_kind =
        is_host ? gpuMemcpyHostToDevice : gpuMemcpyDeviceToDevice;

    size_t sz = size_v[iov];
    int num_streams = std::min(
        streams.size(), sz < kIpcSizePerEngine ? 1 : sz / kIpcSizePerEngine);
    active_streams = std::max(active_streams, num_streams);
    size_t chunk_size = sz / num_streams;

    for (int i = 0; i < num_streams; ++i) {
      void* chunk_src = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(data_v[iov]) + i * chunk_size);
      void* chunk_dst = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(dst_ptr) + i * chunk_size);
      auto copy_size = i == num_streams - 1 ? sz - i * chunk_size : chunk_size;
      GPU_RT_CHECK(uccl::device::ipc_copy_async(
          chunk_dst, ipc_dev_w, chunk_src, local_gpu_idx_, copy_size, is_host,
          memcpy_kind, streams[i]));
    }
  }

  for (int i = 0; i < active_streams; ++i) {
    GPU_RT_CHECK(gpuStreamSynchronize(streams[i]));
  }
  for (void* raw_ptr : mappings.raw_ptrs) {
    GPU_RT_CHECK(gpuIpcCloseMemHandle(raw_ptr));
  }

  return true;
}

bool Endpoint::readv_ipc(uint64_t conn_id, std::vector<void*> data_v,
                         std::vector<size_t> size_v,
                         std::vector<IpcTransferInfo> info_v, size_t num_iovs) {
  UCCL_CHECK_EQ(data_v.size(), num_iovs) << "readv_ipc: data_v size mismatch";
  UCCL_CHECK_EQ(size_v.size(), num_iovs) << "readv_ipc: size_v size mismatch";
  UCCL_CHECK_EQ(info_v.size(), num_iovs) << "readv_ipc: info_v size mismatch";

  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[readv_ipc] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  if (num_iovs == 0) return true;

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  int ipc_dev_r = uccl::device::ipc_open_device(
      local_gpu_idx_,
      ipc_owner_device(local_gpu_idx_, conn, info_v[0].gpu_idx));
  GPU_RT_CHECK(gpuSetDevice(ipc_dev_r));
  std::vector<gpuStream_t>& streams = get_ipc_streams(ipc_dev_r);

  // Finish host allocations before submitting any copies.
  IpcMappings mappings(num_iovs);
  for (size_t iov = 0; iov < num_iovs; ++iov) {
    mappings.prepare(iov, info_v[iov].handle);
  }
  int active_streams = 0;
  for (size_t iov = 0; iov < num_iovs; ++iov) {
    UCCL_CHECK(data_v[iov] != nullptr)
        << "readv_ipc: data_v[" << iov << "] is null";
    void* raw_ptr = mappings.open(iov, info_v[iov].handle);
    void* src_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(raw_ptr) + info_v[iov].offset);

    bool is_host = (uccl::get_dev_idx(data_v[iov]) == -1);
    auto memcpy_kind =
        is_host ? gpuMemcpyDeviceToHost : gpuMemcpyDeviceToDevice;

    size_t sz = size_v[iov];
    int num_streams = std::min(
        streams.size(), sz < kIpcSizePerEngine ? 1 : sz / kIpcSizePerEngine);
    active_streams = std::max(active_streams, num_streams);
    size_t chunk_size = sz / num_streams;

    for (int i = 0; i < num_streams; ++i) {
      void* chunk_src = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(src_ptr) + i * chunk_size);
      void* chunk_dst = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(data_v[iov]) + i * chunk_size);
      auto copy_size = i == num_streams - 1 ? sz - i * chunk_size : chunk_size;
      GPU_RT_CHECK(uccl::device::ipc_copy_async(
          chunk_dst, local_gpu_idx_, chunk_src, ipc_dev_r, copy_size, is_host,
          memcpy_kind, streams[i]));
    }
  }

  for (int i = 0; i < active_streams; ++i) {
    GPU_RT_CHECK(gpuStreamSynchronize(streams[i]));
  }
  for (void* raw_ptr : mappings.raw_ptrs) {
    GPU_RT_CHECK(gpuIpcCloseMemHandle(raw_ptr));
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

  int target_gpu = uccl::device::ipc_async_device(
      local_gpu_idx_, ipc_owner_device(local_gpu_idx_, conn, info.gpu_idx),
      info.direct_addr != 0);
  GPU_RT_CHECK(gpuSetDevice(target_gpu));

  std::vector<gpuStream_t>& streams = get_ipc_streams(target_gpu);
  int num_streams =
      std::min(streams.size(),
               size < kIpcSizePerEngine ? 1 : (size_t)size / kIpcSizePerEngine);
  size_t chunk_size = size / num_streams;

  // Finish host allocations before opening a mapping or submitting copies.
  auto op = std::make_unique<IpcInflightOp>();
  op->gpu_idx = target_gpu;
  op->events.resize(num_streams);
  auto status = std::make_unique<TransferStatus>();

  void* raw_dst_ptr = nullptr;
  void* dst_ptr = nullptr;
  if (info.direct_addr != 0) {
    // Same-process: use virtual address directly, no IPC handle needed.
    dst_ptr = reinterpret_cast<void*>(info.direct_addr);
  } else {
    GPU_RT_CHECK(gpuIpcOpenMemHandle(&raw_dst_ptr, info.handle,
                                     gpuIpcMemLazyEnablePeerAccess));
    dst_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(raw_dst_ptr) +
                                      info.offset);
  }

  op->raw_ptr = raw_dst_ptr;
  for (int i = 0; i < num_streams; ++i) {
    void* chunk_data = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(data) + i * chunk_size);
    void* chunk_dst = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(dst_ptr) + i * chunk_size);
    auto copy_size = i == num_streams - 1 ? size - i * chunk_size : chunk_size;
    GPU_RT_CHECK(uccl::device::ipc_copy_async(
        chunk_dst, target_gpu, chunk_data, local_gpu_idx_, copy_size, is_host,
        memcpy_kind, streams[i]));
    GPU_RT_CHECK(
        gpuEventCreateWithFlags(&op->events[i], gpuEventDisableTiming));
    GPU_RT_CHECK(gpuEventRecord(op->events[i], streams[i]));
  }

  op->status = status.release();
  *transfer_id = reinterpret_cast<uint64_t>(op->status);
  auto* inflight = op.release();

  while (jring_mp_enqueue_bulk(ipc_inflight_ring_, &inflight, 1, nullptr) !=
         1) {
  }
  ipc_proxy_adaptive_sleeper_.maybe_wake_proxy_thread();

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

  int target_gpu = uccl::device::ipc_async_device(
      local_gpu_idx_, ipc_owner_device(local_gpu_idx_, conn, info.gpu_idx),
      info.direct_addr != 0);
  GPU_RT_CHECK(gpuSetDevice(target_gpu));

  std::vector<gpuStream_t>& streams = get_ipc_streams(target_gpu);
  int num_streams =
      std::min(streams.size(),
               size < kIpcSizePerEngine ? 1 : (size_t)size / kIpcSizePerEngine);
  size_t chunk_size = size / num_streams;

  // Finish host allocations before opening a mapping or submitting copies.
  auto op = std::make_unique<IpcInflightOp>();
  op->gpu_idx = target_gpu;
  op->events.resize(num_streams);
  auto status = std::make_unique<TransferStatus>();

  void* raw_src_ptr = nullptr;
  void* src_ptr = nullptr;
  if (info.direct_addr != 0) {
    src_ptr = reinterpret_cast<void*>(info.direct_addr);
  } else {
    GPU_RT_CHECK(gpuIpcOpenMemHandle(&raw_src_ptr, info.handle,
                                     gpuIpcMemLazyEnablePeerAccess));
    src_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(raw_src_ptr) +
                                      info.offset);
  }

  op->raw_ptr = raw_src_ptr;
  for (int i = 0; i < num_streams; ++i) {
    void* chunk_src = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(src_ptr) + i * chunk_size);
    void* chunk_data = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(data) + i * chunk_size);
    auto copy_size = i == num_streams - 1 ? size - i * chunk_size : chunk_size;
    GPU_RT_CHECK(uccl::device::ipc_copy_async(
        chunk_data, local_gpu_idx_, chunk_src, target_gpu, copy_size, is_host,
        memcpy_kind, streams[i]));
    GPU_RT_CHECK(
        gpuEventCreateWithFlags(&op->events[i], gpuEventDisableTiming));
    GPU_RT_CHECK(gpuEventRecord(op->events[i], streams[i]));
  }

  op->status = status.release();
  *transfer_id = reinterpret_cast<uint64_t>(op->status);
  auto* inflight = op.release();

  while (jring_mp_enqueue_bulk(ipc_inflight_ring_, &inflight, 1, nullptr) !=
         1) {
  }
  ipc_proxy_adaptive_sleeper_.maybe_wake_proxy_thread();

  return true;
}

bool Endpoint::writev_ipc_async(uint64_t conn_id,
                                std::vector<void const*> data_v,
                                std::vector<size_t> size_v,
                                std::vector<IpcTransferInfo> info_v,
                                size_t num_iovs, uint64_t* transfer_id) {
  UCCL_CHECK_EQ(data_v.size(), num_iovs)
      << "writev_ipc_async: data_v size mismatch";
  UCCL_CHECK_EQ(size_v.size(), num_iovs)
      << "writev_ipc_async: size_v size mismatch";
  UCCL_CHECK_EQ(info_v.size(), num_iovs)
      << "writev_ipc_async: info_v size mismatch";

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

  int target_gpu = uccl::device::ipc_async_device(
      local_gpu_idx_,
      ipc_owner_device(local_gpu_idx_, conn,
                       num_iovs > 0 ? info_v[0].gpu_idx : -1),
      num_iovs > 0 && info_v[0].direct_addr != 0);
  GPU_RT_CHECK(gpuSetDevice(target_gpu));
  std::vector<gpuStream_t>& streams = get_ipc_streams(target_gpu);

  std::array<size_t, kNumGpuRtStreams> last_iov{};
  size_t active_streams = 0;
  for (size_t iov = num_iovs; iov > 0 && active_streams < streams.size();) {
    --iov;
    size_t sz = size_v[iov];
    auto num_streams = std::min(
        streams.size(), sz < kIpcSizePerEngine ? 1 : sz / kIpcSizePerEngine);
    if (num_streams > active_streams) {
      std::fill(last_iov.begin() + active_streams,
                last_iov.begin() + num_streams, iov);
      active_streams = num_streams;
    }
  }

  // Finish host allocations before submitting any copies.
  IpcMappings mappings(num_iovs);
  for (size_t iov = 0; iov < num_iovs; ++iov) {
    if (info_v[iov].direct_addr == 0) {
      mappings.prepare(iov, info_v[iov].handle);
    }
  }
  // Use raw_ptr=nullptr to signal vectorized op to the poller thread.
  auto op = std::make_unique<IpcInflightOp>();
  op->gpu_idx = -1;
  op->gpu_idxs_v.assign(mappings.raw_ptrs.size(), target_gpu);
  op->events.reserve(active_streams);
  auto status = std::make_unique<TransferStatus>();

  for (size_t iov = 0; iov < num_iovs; ++iov) {
    void* dst_ptr = nullptr;
    if (info_v[iov].direct_addr != 0) {
      dst_ptr = reinterpret_cast<void*>(info_v[iov].direct_addr);
    } else {
      void* raw_ptr = mappings.open(iov, info_v[iov].handle);
      dst_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(raw_ptr) +
                                        info_v[iov].offset);
    }

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
      GPU_RT_CHECK(uccl::device::ipc_copy_async(
          chunk_dst, target_gpu, chunk_src, local_gpu_idx_, copy_size, is_host,
          memcpy_kind, streams[i]));
      // Keep each shared stream's completion point at its last copy.
      if (last_iov[i] == iov) {
        gpuEvent_t ev;
        GPU_RT_CHECK(gpuEventCreateWithFlags(&ev, gpuEventDisableTiming));
        GPU_RT_CHECK(gpuEventRecord(ev, streams[i]));
        op->events.push_back(ev);
      }
    }
  }

  op->raw_ptrs_v = std::move(mappings.raw_ptrs);
  op->status = status.release();
  *transfer_id = reinterpret_cast<uint64_t>(op->status);
  auto* inflight = op.release();

  while (jring_mp_enqueue_bulk(ipc_inflight_ring_, &inflight, 1, nullptr) !=
         1) {
  }
  ipc_proxy_adaptive_sleeper_.maybe_wake_proxy_thread();

  return true;
}

bool Endpoint::readv_ipc_async(uint64_t conn_id, std::vector<void*> data_v,
                               std::vector<size_t> size_v,
                               std::vector<IpcTransferInfo> info_v,
                               size_t num_iovs, uint64_t* transfer_id) {
  UCCL_CHECK_EQ(data_v.size(), num_iovs)
      << "readv_ipc_async: data_v size mismatch";
  UCCL_CHECK_EQ(size_v.size(), num_iovs)
      << "readv_ipc_async: size_v size mismatch";
  UCCL_CHECK_EQ(info_v.size(), num_iovs)
      << "readv_ipc_async: info_v size mismatch";

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

  int target_gpu = uccl::device::ipc_async_device(
      local_gpu_idx_,
      ipc_owner_device(local_gpu_idx_, conn,
                       num_iovs > 0 ? info_v[0].gpu_idx : -1),
      num_iovs > 0 && info_v[0].direct_addr != 0);
  GPU_RT_CHECK(gpuSetDevice(target_gpu));
  std::vector<gpuStream_t>& streams = get_ipc_streams(target_gpu);

  std::array<size_t, kNumGpuRtStreams> last_iov{};
  size_t active_streams = 0;
  for (size_t iov = num_iovs; iov > 0 && active_streams < streams.size();) {
    --iov;
    size_t sz = size_v[iov];
    auto num_streams = std::min(
        streams.size(), sz < kIpcSizePerEngine ? 1 : sz / kIpcSizePerEngine);
    if (num_streams > active_streams) {
      std::fill(last_iov.begin() + active_streams,
                last_iov.begin() + num_streams, iov);
      active_streams = num_streams;
    }
  }

  // Finish host allocations before submitting any copies.
  IpcMappings mappings(num_iovs);
  for (size_t iov = 0; iov < num_iovs; ++iov) {
    if (info_v[iov].direct_addr == 0) {
      mappings.prepare(iov, info_v[iov].handle);
    }
  }
  // Use raw_ptr=nullptr to signal vectorized op to the poller thread.
  auto op = std::make_unique<IpcInflightOp>();
  op->gpu_idx = -1;
  op->gpu_idxs_v.assign(mappings.raw_ptrs.size(), target_gpu);
  op->events.reserve(active_streams);
  auto status = std::make_unique<TransferStatus>();

  for (size_t iov = 0; iov < num_iovs; ++iov) {
    void* src_ptr = nullptr;
    if (info_v[iov].direct_addr != 0) {
      src_ptr = reinterpret_cast<void*>(info_v[iov].direct_addr);
    } else {
      void* raw_ptr = mappings.open(iov, info_v[iov].handle);
      src_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(raw_ptr) +
                                        info_v[iov].offset);
    }

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
      GPU_RT_CHECK(uccl::device::ipc_copy_async(
          chunk_dst, local_gpu_idx_, chunk_src, target_gpu, copy_size, is_host,
          memcpy_kind, streams[i]));
      // Keep each shared stream's completion point at its last copy.
      if (last_iov[i] == iov) {
        gpuEvent_t ev;
        GPU_RT_CHECK(gpuEventCreateWithFlags(&ev, gpuEventDisableTiming));
        GPU_RT_CHECK(gpuEventRecord(ev, streams[i]));
        op->events.push_back(ev);
      }
    }
  }

  op->raw_ptrs_v = std::move(mappings.raw_ptrs);
  op->status = status.release();
  *transfer_id = reinterpret_cast<uint64_t>(op->status);
  auto* inflight = op.release();

  while (jring_mp_enqueue_bulk(ipc_inflight_ring_, &inflight, 1, nullptr) !=
         1) {
  }
  ipc_proxy_adaptive_sleeper_.maybe_wake_proxy_thread();

  return true;
}

bool Endpoint::advertise_ipc(uint64_t conn_id, void* addr, size_t len,
                             char* out_buf) {
  UCCL_CHECK(addr != nullptr) << "advertise_ipc: addr pointer is null!";
  UCCL_CHECK(out_buf != nullptr) << "advertise_ipc: out_buf pointer is null!";

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

  // Generate IPC memory handle for the address
  IpcTransferInfo transfer_info = {};  // Initialize to zero
  transfer_info.size = len;
  transfer_info.operation = 1;  // response

  GpuIpcExport exported;
  auto err = export_ipc_range(addr, len, &exported);
  if (err != gpuSuccess) {
    UCCL_LOG(ERROR) << "IPC export failed: " << gpuGetErrorString(err);
    return false;
  }
  transfer_info.handle = exported.handle;
  transfer_info.offset = exported.offset;
  transfer_info.gpu_idx = exported.owner_device;

  // Copy the transfer info to output buffer
  std::memcpy(out_buf, &transfer_info, sizeof(transfer_info));

  return true;
}

bool Endpoint::advertisev_ipc(uint64_t conn_id, std::vector<void*> addr_v,
                              std::vector<size_t> len_v,
                              std::vector<char*> out_buf_v, size_t num_iovs) {
  UCCL_CHECK_EQ(addr_v.size(), num_iovs) << "addr_v size mismatch";
  UCCL_CHECK_EQ(len_v.size(), num_iovs) << "len_v size mismatch";
  UCCL_CHECK_EQ(out_buf_v.size(), num_iovs) << "out_buf_v size mismatch";

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

  for (size_t i = 0; i < num_iovs; ++i) {
    UCCL_CHECK(addr_v[i] != nullptr)
        << "advertisev_ipc: addr_v[" << i << "] is null!";
    UCCL_CHECK(out_buf_v[i] != nullptr)
        << "advertisev_ipc: out_buf_v[" << i << "] is null!";

    // Generate IPC memory handle for each address
    IpcTransferInfo transfer_info = {};  // Initialize to zero
    transfer_info.size = len_v[i];
    transfer_info.operation = 1;  // response

    GpuIpcExport exported;
    auto err = export_ipc_range(addr_v[i], len_v[i], &exported);
    if (err != gpuSuccess) {
      UCCL_LOG(ERROR) << "IPC vector export failed: " << gpuGetErrorString(err);
      return false;
    }
    transfer_info.handle = exported.handle;
    transfer_info.offset = exported.offset;
    transfer_info.gpu_idx = exported.owner_device;

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
  auto [remote_ip, remote_port, remote_gpu_bdf] = parse_metadata(metadata);
  std::string local_ip = uccl::get_oob_ip();
  bool is_local = (remote_ip == local_ip);
  bool success;
  if (is_local && !remote_gpu_bdf.empty()) {
    success = connect_local(remote_gpu_bdf, conn_id);
  } else {
    success = connect(remote_ip, 0, remote_port, conn_id);
  }
  if (success) {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    auto conn_it = conn_id_to_conn_.find(conn_id);
    if (conn_it != conn_id_to_conn_.end()) {
      conn_it->second->ip_addr_ = remote_ip;
      conn_it->second->remote_port_ = remote_port;
      conn_it->second->remote_gpu_bdf_ = remote_gpu_bdf;
    }
    remote_endpoint_to_conn_id_[metadata] = conn_id;
    return true;
  }
  return false;
}

bool Endpoint::remove_remote_endpoint(uint64_t conn_id) {
  std::unique_lock<std::shared_mutex> lock(conn_mu_);

  auto it = conn_id_to_conn_.find(conn_id);
  if (it == conn_id_to_conn_.end()) {
    std::cerr << "[remove_remote_endpoint] Error: Invalid conn_id " << conn_id
              << std::endl;
    return false;
  }

  Conn* conn = it->second;
  uint64_t loopback_conn_id = conn->rdma_loopback_conn_id_;

  // Detach shared memory if this was a local connection
  if (conn->shm_attached_) {
    auto& shm = conn->remote_inbox_;
    uccl::detach_shared_ring(shm.ring, shm.shm_fd, shm.shm_size);
  }

  // Remove from the metadata memoization map
  for (auto mit = remote_endpoint_to_conn_id_.begin();
       mit != remote_endpoint_to_conn_id_.end(); ++mit) {
    if (mit->second == conn_id) {
      remote_endpoint_to_conn_id_.erase(mit);
      break;
    }
  }

  delete conn;
  conn_id_to_conn_.erase(it);

  if (loopback_conn_id != UINT64_MAX) {
    auto loopback_it = conn_id_to_conn_.find(loopback_conn_id);
    if (loopback_it != conn_id_to_conn_.end()) {
      Conn* loopback_conn = loopback_it->second;
      if (loopback_conn->shm_attached_) {
        auto& shm = loopback_conn->remote_inbox_;
        uccl::detach_shared_ring(shm.ring, shm.shm_fd, shm.shm_size);
      }
      delete loopback_conn;
      conn_id_to_conn_.erase(loopback_it);
    }
  }

  return true;
}

bool Endpoint::start_passive_accept() {
  if (!passive_accept_) {
    passive_accept_stop_.store(false, std::memory_order_release);
    passive_accept_thread_ =
        std::thread(&Endpoint::passive_accept_thread_func, this);
    passive_accept_local_thread_ =
        std::thread(&Endpoint::passive_accept_local_thread_func, this);
    passive_accept_ = true;
  }
  return true;
}

bool Endpoint::poll_async(uint64_t transfer_id, bool* is_done) {
  auto* status = reinterpret_cast<TransferStatus*>(transfer_id);
  if (status->poll_net_ureq && !status->done.load(std::memory_order_acquire)) {
    if (uccl_poll_ureq_once(ep_, &status->ureq)) {
      status->done.store(true, std::memory_order_release);
    }
  }
  *is_done = status->done.load(std::memory_order_acquire);
  if (*is_done) {
    bool success = status->success;
    delete status;
    return success;
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
  return std::visit(
      [](auto const& s) -> std::shared_ptr<EpollClient> {
        return s->get_oob_client();
      },
      ep_);
}

std::string Endpoint::get_oob_conn_key(uint64_t conn_id) const {
  auto it = conn_id_to_conn_.find(conn_id);
  if (it == conn_id_to_conn_.end()) {
    return "";
  }
  uint64_t peer_id = it->second->uccl_conn_id_.peer_id;

  return std::visit(
      [&](auto const& s) -> std::string {
        return s->get_oob_conn_key(peer_id);
      },
      ep_);
}

int Endpoint::send_notification(uint64_t conn_id,
                                NotifyMsg const& notification) const {
  Conn* conn = get_conn(conn_id);
  if (conn == nullptr) {
    return -1;
  }
  uint64_t peer_id = conn->uccl_conn_id_.peer_id;
  if (peer_id == UINT64_MAX) {
    return -1;
  }
  return uccl_send_notification(ep_, peer_id, notification);
}

MR* Endpoint::get_mr(uint64_t mr_id) const {
  std::shared_lock<std::shared_mutex> lock(mr_mu_);
  auto it = mr_id_to_mr_.find(mr_id);
  if (it == mr_id_to_mr_.end()) {
    return nullptr;
  }
  return it->second.get();
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

GenericEndpoint Endpoint::get_endpoint() const { return ep_; }

gpuError_t Endpoint::export_ipc_range(void* ptr, size_t size,
                                      GpuIpcExport* result) {
  return gpuExportIpcRange(&ipc_export_context_, ptr, size, result);
}

void Endpoint::initialize_engine() {
  // Host-memory registration can select a different device from the caller's
  // current device. Keep lazy engine setup from changing subsequent GPU work
  // on this thread, including when setup throws.
  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });
  int n_streams = std::max(1, (int)kNumGpuRtStreams);
  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

  numa_node_ = uccl_get_numa_node(ep_, local_gpu_idx_);

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

  // Initialize ShmChannel for local connections (use BDF for ring names)
  size_t elem_sz = sizeof(ShmMsg);
  size_t elem_cnt = ShmRingDefaultElemCnt;
  auto all_bdfs = uccl::enumerate_all_gpu_bdfs();
  for (auto const& sender_bdf : all_bdfs) {
    auto name = shm_ring_name(sender_bdf, gpu_bus_id_);
    ShmRingHandle handle;
    handle.shm_name = name;
    bool is_creator = false;
    handle.ring =
        uccl::create_shared_ring(name.c_str(), elem_sz, elem_cnt, handle.shm_fd,
                                 handle.shm_size, &is_creator);
    if (handle.ring == nullptr) warn_missing_inbox_ring(name, sender_bdf);
    inbox_rings_[sender_bdf] = handle;
    inbox_creators_[sender_bdf] = is_creator;
  }
}

struct Endpoint::ProxyTaskState {
  struct Range {
    uint64_t begin;
    uint64_t end;
  };

  std::shared_ptr<UnifiedTask> task;
  uint64_t conn_id = 0;
  SendConnection* group = nullptr;
  SendType send_type = SendType::Write;
  size_t num_iovs = 1;
  bool raw = false;
  // This proxy writes remote ranges for WRITEV, and local ranges for READV.
  // Numeric remote addresses are compared conservatively across connections.
  std::vector<Range> written_ranges;
  std::vector<std::shared_ptr<MR>> registrations;
};

struct Endpoint::CooperativeProxyBackend {
  using Task = std::shared_ptr<ProxyTaskState>;

  Endpoint& endpoint;
  bool sending;

  Task prepare(UnifiedTask* task) {
    auto state = std::make_shared<ProxyTaskState>();
    state->task = task->status_ptr->task_ptr;
    state->conn_id = task->conn_id;
    state->send_type = sending ? SendType::Write : SendType::Read;
    if (task->type != (sending ? TaskType::WRITEV : TaskType::READV))
      return state;

    auto const& batch = task->task_batch();
    if (batch.num_iovs == 0 || !batch.data_ptr || !batch.size_ptr ||
        !batch.mr_id_ptr || !batch.slot_item_ptr ||
        batch.data_ptr->size() != batch.num_iovs ||
        batch.size_ptr->size() != batch.num_iovs ||
        batch.mr_id_ptr->size() != batch.num_iovs ||
        batch.slot_item_ptr->size() != batch.num_iovs)
      return state;

    {
      std::shared_lock<std::shared_mutex> lock(endpoint.conn_mu_);
      auto it = endpoint.conn_id_to_conn_.find(task->conn_id);
      if (it == endpoint.conn_id_to_conn_.end()) return state;
      // Do not retain a Conn pointer after dropping conn_mu_. SendConnection
      // is owned by the transport endpoint for the lifetime of the proxy.
      state->group = uccl_resolve_send_group(endpoint.ep_,
                                             it->second->uccl_conn_id_.peer_id);
    }
    if (!state->group ||
        !raw_one_sided_batch_eligible(*batch.size_ptr, batch.num_iovs) ||
        !state->group->can_use_raw_one_sided_batch(
            state->send_type, max_iov_bytes(*batch.size_ptr, batch.num_iovs)))
      return state;

    {
      std::shared_lock<std::shared_mutex> lock(endpoint.mr_mu_);
      for (size_t i = 0; i < batch.num_iovs; ++i) {
        auto it = endpoint.mr_id_to_mr_.find((*batch.mr_id_ptr)[i]);
        if (it == endpoint.mr_id_to_mr_.end() ||
            it->second->mhandle_ == nullptr ||
            it->second->mhandle_->compress_only ||
            (*batch.data_ptr)[i] == nullptr)
          return state;

        uint64_t local = reinterpret_cast<uintptr_t>((*batch.data_ptr)[i]);
        uint64_t remote = (*batch.slot_item_ptr)[i].addr;
        size_t size = (*batch.size_ptr)[i];
        if (size > std::numeric_limits<uint64_t>::max() - local ||
            size > std::numeric_limits<uint64_t>::max() - remote)
          return state;
        uint64_t begin = sending ? remote : local;
        state->written_ranges.push_back({begin, begin + size});
      }
    }

    auto& ranges = state->written_ranges;
    std::sort(ranges.begin(), ranges.end(),
              [](auto const& a, auto const& b) { return a.begin < b.begin; });
    size_t merged = 0;
    for (auto const range : ranges) {
      if (merged && range.begin <= ranges[merged - 1].end) {
        ranges[merged - 1].end = std::max(ranges[merged - 1].end, range.end);
      } else {
        ranges[merged++] = range;
      }
    }
    ranges.resize(merged);
    state->num_iovs = batch.num_iovs;
    state->raw = true;
    return state;
  }

  void drive() {
    uccl_drive_send(endpoint.ep_);
    if (!sending) uccl_drive_recv(endpoint.ep_);
  }

  uccl::ProxyPostResult post(Task& state, size_t offset, size_t budget) {
    auto const& batch = state->task->task_batch();
    if (state->registrations.empty()) {
      state->registrations.reserve(batch.num_iovs);
      std::shared_lock<std::shared_mutex> lock(endpoint.mr_mu_);
      for (auto id : *batch.mr_id_ptr) {
        auto it = endpoint.mr_id_to_mr_.find(id);
        // Registration may have been removed while this request was queued.
        if (it == endpoint.mr_id_to_mr_.end() ||
            it->second->mhandle_ == nullptr ||
            it->second->mhandle_->compress_only)
          return {uccl::ProxyPostStatus::Failed, 0, {}};
        state->registrations.push_back(it->second);
      }
    }

    constexpr size_t kQuantumBytes = 1024 * 1024;
    size_t limit = std::min(budget, batch.num_iovs - offset);
    size_t count = 0;
    size_t bytes = 0;
    while (count < limit) {
      size_t next_size = (*batch.size_ptr)[offset + count];
      if (count &&
          (bytes >= kQuantumBytes || next_size > kQuantumBytes - bytes))
        break;
      bytes += next_size;
      ++count;
    }
    if (count == 0) return {uccl::ProxyPostStatus::Retry, 0, {}};

    // One turn submits at most the scheduler quantum; size the batch for it.
    constexpr size_t kMaxPostIovs = uccl::ProxySchedulerOptions{}.quantum;
    std::array<SendConnection::OneSidedBatchOp, kMaxPostIovs> ops{};
    std::array<SendConnection::RawBatchWait, kMaxPostIovs> waits{};
    std::array<int64_t, kMaxPostIovs> wr_ids{};
    UCCL_CHECK(count <= ops.size());
    for (size_t i = 0; i < count; ++i) {
      size_t iov = offset + i;
      ops[i].local_addr = (*batch.data_ptr)[iov];
      ops[i].size = (*batch.size_ptr)[iov];
      ops[i].local_mr_array = &state->registrations[iov]->mhandle_->mr_array;
      ops[i].slot_item = &(*batch.slot_item_ptr)[iov];
    }

    size_t num_waits = 0;
    // Keep the transport's existing failure behavior, including its fatal
    // handling of provider failures after raw tracker IDs are allocated.
    if (!state->group->post_write_or_read_batch(state->send_type, ops.data(),
                                                count, wr_ids.data(),
                                                waits.data(), &num_waits))
      return {uccl::ProxyPostStatus::Failed, 0, {}};

    UCCL_CHECK(num_waits > 0 && num_waits <= count);
    uccl::ProxyPostResult result{uccl::ProxyPostStatus::Posted, count, {}};
    result.waits.reserve(num_waits);
    size_t represented = 0;
    for (size_t i = 0; i < num_waits; ++i) {
      result.waits.push_back({waits[i].wr_id, waits[i].iov_count});
      represented += waits[i].iov_count;
    }
    UCCL_CHECK(represented == count);
    return result;
  }

  uccl::ProxyCompletion poll(Task& state, uccl::ProxyWait const& wait) {
    return uccl_check_wr_fast(state->group, wait.wr_id)
               ? uccl::ProxyCompletion::Complete
               : uccl::ProxyCompletion::Pending;
  }

  bool conflicts(Task const& a, Task const& b) {
    if (a->conn_id == b->conn_id) return true;
    auto const& lhs = a->written_ranges;
    auto const& rhs = b->written_ranges;
    size_t i = 0, j = 0;
    while (i < lhs.size() && j < rhs.size()) {
      if (lhs[i].begin < rhs[j].end && rhs[j].begin < lhs[i].end) return true;
      if (lhs[i].end <= rhs[j].end)
        ++i;
      else
        ++j;
    }
    return false;
  }

  bool run_exclusive(Task& state) {
    auto const& task = *state->task;
    switch (task.type) {
      case TaskType::WRITE_NET:
        return endpoint.write(task.conn_id, task.mr_id, task.data, task.size,
                              task.slot_item());
      case TaskType::READ_NET:
        return endpoint.read(task.conn_id, task.mr_id, task.data, task.size,
                             task.slot_item());
      case TaskType::WRITEV:
      case TaskType::READV: {
        auto const& batch = task.task_batch();
        if (task.type == TaskType::WRITEV)
          return endpoint.writev(task.conn_id, *batch.mr_id_ptr,
                                 *batch.data_ptr, *batch.size_ptr,
                                 *batch.slot_item_ptr, batch.num_iovs);
        return endpoint.readv(task.conn_id, *batch.mr_id_ptr, *batch.data_ptr,
                              *batch.size_ptr, *batch.slot_item_ptr,
                              batch.num_iovs);
      }
      default:
        UCCL_LOG(ERROR) << "Unexpected task type in cooperative proxy: "
                        << static_cast<int>(task.type);
        return false;
    }
  }

  void finish(Task& state, bool success) {
    auto* status = state->task->status_ptr;
    status->success = success;
    state->registrations.clear();
    state->task.reset();
    status->task_ptr.reset();
    // poll_async may delete status immediately after this publication.
    status->done.store(true, std::memory_order_release);
  }
};

void Endpoint::cooperative_proxy_thread_func(bool sending) {
  uccl::pin_thread_to_numa(numa_node_);
  auto* ring = sending ? send_unified_task_ring_ : recv_unified_task_ring_;
  auto& sleeper =
      sending ? send_proxy_adaptive_sleeper_ : recv_proxy_adaptive_sleeper_;
  uccl::ProxySchedulerOptions options;
  options.max_inflight = max_one_sided_inflight_ops();
  uccl::ProxyScheduler<CooperativeProxyBackend::Task> scheduler(options);
  CooperativeProxyBackend backend{*this, sending};
  alignas(16) char task_buffer[16];
  bool fallback_logged = false;
  sleeper.update_timer();

  for (;;) {
    if (stop_.load(std::memory_order_acquire)) scheduler.stop_admission();
    while (scheduler.accepting() && scheduler.size() < options.max_requests) {
      if (stop_.load(std::memory_order_acquire)) {
        scheduler.stop_admission();
        break;
      }
      if (jring_sc_dequeue_bulk(ring, task_buffer, 1, nullptr) != 1) break;
      auto state =
          backend.prepare(*reinterpret_cast<UnifiedTask**>(task_buffer));
      // A vector request that is not raw-eligible runs as an exclusive
      // barrier. Say so once, or a misconfigured opt-in looks like no effect.
      if (!state->raw && !fallback_logged &&
          state->task->type == (sending ? TaskType::WRITEV : TaskType::READV)) {
        fallback_logged = true;
        UCCL_LOG(WARN) << "cooperative proxy: a "
                       << (sending ? "WRITEV" : "READV")
                       << " request is not raw-eligible (congestion control, "
                          "compression, chunked IOVs or connection state) and "
                          "runs as an exclusive barrier; further cases are not "
                          "logged";
      }
      // The loop condition already guarantees admission; enqueue cannot fail.
      UCCL_CHECK(scheduler.enqueue(state, state->num_iovs, !state->raw));
    }

    if (!scheduler.empty()) {
      scheduler.progress(backend);
      sleeper.update_timer();
      continue;
    }
    // Stop admitting work on shutdown, but drain every accepted request before
    // allowing the endpoint destructor to release connections or registrations.
    if (!scheduler.accepting()) break;
    if (!sending) uccl_drive_recv(ep_);
    // CQ completions do not signal the task eventfd; never sleep with work
    // live.
    sleeper.maybe_sleep();
  }
}

static bool cooperative_proxy_enabled() {
  char const* value = std::getenv("UCCL_P2P_COOPERATIVE_PROXY");
  return value && std::strcmp(value, "1") == 0;
}

void Endpoint::send_proxy_thread_func() {
  if constexpr (!uccl::device::Backend::staged_network) {
    if (cooperative_proxy_enabled()) {
      cooperative_proxy_thread_func(true);
      return;
    }
  }
  uccl::pin_thread_to_numa(numa_node_);
  // Use 16-byte buffer to avoid stringop-overflow warning from jring's 16-byte
  // bulk copy
  alignas(16) char task_buffer[16];
  UnifiedTask* task;
  send_proxy_adaptive_sleeper_.update_timer();

  while (!stop_.load(std::memory_order_acquire)) {
    send_proxy_adaptive_sleeper_.maybe_sleep();

    if (jring_sc_dequeue_bulk(send_unified_task_ring_, task_buffer, 1,
                              nullptr) == 1) {
      task = *reinterpret_cast<UnifiedTask**>(task_buffer);
      bool success = false;
      switch (task->type) {
        case TaskType::WRITE_NET:
          success = write(task->conn_id, task->mr_id, task->data, task->size,
                          task->slot_item());
          break;
        case TaskType::WRITEV: {
          TaskBatch const& batch = task->task_batch();
          success =
              writev(task->conn_id, *batch.mr_id_ptr, *batch.data_ptr,
                     *batch.size_ptr, *batch.slot_item_ptr, batch.num_iovs);
          break;
        }
        default:
          UCCL_LOG(ERROR) << "Unexpected task type in send processing: "
                          << static_cast<int>(task->type);
          break;
      }
      auto* status = task->status_ptr;
      status->success = success;
      status->task_ptr.reset();
      status->done.store(true, std::memory_order_release);
      send_proxy_adaptive_sleeper_.update_timer();
    }
  }
}

void Endpoint::recv_proxy_thread_func() {
  if constexpr (!uccl::device::Backend::staged_network) {
    if (cooperative_proxy_enabled()) {
      cooperative_proxy_thread_func(false);
      return;
    }
  }
  uccl::pin_thread_to_numa(numa_node_);
  // Use 16-byte buffer to avoid stringop-overflow warning from jring's 16-byte
  // bulk copy
  alignas(16) char task_buffer[16];
  UnifiedTask* task;
  recv_proxy_adaptive_sleeper_.update_timer();

  while (!stop_.load(std::memory_order_acquire)) {
    recv_proxy_adaptive_sleeper_.maybe_sleep();

    if (jring_sc_dequeue_bulk(recv_unified_task_ring_, task_buffer, 1,
                              nullptr) == 1) {
      task = *reinterpret_cast<UnifiedTask**>(task_buffer);
      bool success = false;
      switch (task->type) {
        case TaskType::READ_NET:
          success = read(task->conn_id, task->mr_id, task->data, task->size,
                         task->slot_item());
          break;
        case TaskType::READV: {
          TaskBatch const& batch = task->task_batch();
          success =
              readv(task->conn_id, *batch.mr_id_ptr, *batch.data_ptr,
                    *batch.size_ptr, *batch.slot_item_ptr, batch.num_iovs);
          break;
        }
        case TaskType::WRITE_NET:
        default:
          UCCL_LOG(ERROR) << "Unexpected task type in receive processing: "
                          << static_cast<int>(task->type);
          break;
      }
      auto* status = task->status_ptr;
      status->success = success;
      status->task_ptr.reset();
      status->done.store(true, std::memory_order_release);
      recv_proxy_adaptive_sleeper_.update_timer();
    } else {
      uccl_drive_recv(ep_);
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

void Endpoint::passive_accept_local_thread_func() {
  while (!passive_accept_stop_.load(std::memory_order_acquire)) {
    bool found = false;
    for (auto& [bdf, handle] : inbox_rings_) {
      if (bdf == gpu_bus_id_) continue;
      if (handle.ring == nullptr) continue;
      if (jring_empty(handle.ring)) continue;

      std::string remote_gpu_bdf;
      uint64_t conn_id;
      (void)accept_local(remote_gpu_bdf, conn_id);
      found = true;
      break;
    }
    if (!found) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void Endpoint::ipc_poller_thread_func() {
  uccl::pin_thread_to_numa(numa_node_);
  std::deque<IpcInflightOp*> active_ops;
  alignas(16) char buf[16];
  int cur_device = -1;
  ipc_proxy_adaptive_sleeper_.update_timer();

  while (!stop_.load(std::memory_order_acquire) || !active_ops.empty()) {
    // only try to sleep if there are no more inflight ops
    if (!active_ops.empty()) {
      ipc_proxy_adaptive_sleeper_.update_timer();
    }
    ipc_proxy_adaptive_sleeper_.maybe_sleep();

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
            if (op->raw_ptrs_v[i] == nullptr) continue;  // direct_addr path
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
