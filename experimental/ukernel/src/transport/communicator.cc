#include "communicator.h"
#include "adapter/ipc_adapter.h"
#include "adapter/uccl_adapter.h"
#include "adapter/tcp_adapter.h"
#include "adapter/transport_adapter.h"
#include "util/utils.h"
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <ifaddrs.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

namespace UKernel {
namespace Transport {

namespace {

constexpr int kDefaultBootstrapTimeoutMs = 30000;

std::string get_local_ip() {
  if (char const* env_ip = std::getenv("UHM_LOCAL_IP")) {
    if (std::strlen(env_ip) > 0) return env_ip;
  }

  int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (sock < 0) return "127.0.0.1";

  sockaddr_in remote{};
  remote.sin_family = AF_INET;
  remote.sin_port = htons(80);
  ::inet_pton(AF_INET, "8.8.8.8", &remote.sin_addr);

  ::connect(sock, (sockaddr*)&remote, sizeof(remote));

  sockaddr_in local{};
  socklen_t len = sizeof(local);
  ::getsockname(sock, (sockaddr*)&local, &len);
  ::close(sock);

  char buf[INET_ADDRSTRLEN];
  ::inet_ntop(AF_INET, &local.sin_addr, buf, sizeof(buf));
  return buf;
}

bool has_env_value(char const* name) {
  char const* value = std::getenv(name);
  return value != nullptr && value[0] != '\0';
}

bool is_unspecified_ip(std::string const& ip) {
  return ip.empty() || ip == "0.0.0.0" || ip == "127.0.0.1" ||
         ip == "localhost";
}

std::string find_ifname_for_local_ip(std::string const& ip) {
  if (is_unspecified_ip(ip)) return {};

  ifaddrs* ifs = nullptr;
  if (::getifaddrs(&ifs) != 0) return {};

  std::string ifname;
  for (auto* it = ifs; it != nullptr; it = it->ifa_next) {
    if (!it->ifa_addr || it->ifa_addr->sa_family != AF_INET) continue;

    char buf[INET_ADDRSTRLEN] = {};
    auto* addr = reinterpret_cast<sockaddr_in*>(it->ifa_addr);
    if (!::inet_ntop(AF_INET, &addr->sin_addr, buf, sizeof(buf))) continue;
    if (ip == buf) {
      ifname = it->ifa_name;
      break;
    }
  }

  ::freeifaddrs(ifs);
  return ifname;
}

void maybe_configure_uccl_socket_ifname(std::string const& local_hint_ip) {
  if (has_env_value("UCCL_SOCKET_IFNAME") ||
      has_env_value("NCCL_SOCKET_IFNAME")) {
    return;
  }

  std::string ifname = find_ifname_for_local_ip(local_hint_ip);
  if (ifname.empty()) return;

  ::setenv("UCCL_SOCKET_IFNAME", ifname.c_str(), 0);
  std::cout << "[INFO] Auto-selected UCCL_SOCKET_IFNAME=" << ifname
            << std::endl;
}

int get_timeout_ms(char const* env_name, int default_ms) {
  char const* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0') return default_ms;
  try {
    return std::stoi(value);
  } catch (...) {
    return default_ms;
  }
}

int bootstrap_timeout_ms() {
  return get_timeout_ms("UHM_BOOTSTRAP_TIMEOUT_MS", kDefaultBootstrapTimeoutMs);
}

std::string uccl_p2p_key(int src_rank, int dst_rank) {
  return "uccl_p2p_info_" + std::to_string(src_rank) + "_to_" +
         std::to_string(dst_rank);
}

std::string tcp_p2p_key(int src_rank, int dst_rank) {
  return "tcp_p2p_info_" + std::to_string(src_rank) + "_to_" +
         std::to_string(dst_rank);
}

std::string ipc_global_buffer_key(int owner_rank, uint32_t buffer_id) {
  return "ipc:rank:" + std::to_string(owner_rank) + ":buf:" +
         std::to_string(buffer_id);
}

std::string mr_global_buffer_key(int owner_rank, uint32_t buffer_id) {
  return "mr:rank:" + std::to_string(owner_rank) + ":buf:" +
         std::to_string(buffer_id);
}

std::string oob_scoped_key(std::string const& ns, std::string const& key) {
  if (ns.empty()) return key;
  return ns + "/" + key;
}

template <typename Info>
bool oob_put(Exchanger& ex, std::string const& ns, std::string const& key,
             Info const& value) {
  return ex.put(oob_scoped_key(ns, key), value);
}

template <typename Info>
bool oob_get(Exchanger& ex, std::string const& ns, std::string const& key,
             Info& out, int timeout_ms = 0) {
  std::string const full_key = oob_scoped_key(ns, key);
  if (timeout_ms == 0) return ex.get(full_key, out);
  constexpr int kPollDelayMs = 10;
  int const max_retries =
      timeout_ms < 0
          ? -1
          : std::max(1, (timeout_ms + kPollDelayMs - 1) / kPollDelayMs);
  return ex.wait(full_key, out,
                 Exchanger::WaitOptions(max_retries, kPollDelayMs));
}

void validate_dst_hint_for_transport(PeerTransportKind kind,
                                     std::optional<RemoteSlice> const& dst_hint,
                                     size_t src_bytes) {
  if (!dst_hint.has_value()) return;
  auto const& hint = *dst_hint;
  if (hint.buffer_id == 0) {
    throw std::invalid_argument("dst_hint.buffer_id must be non-zero");
  }
  // IPC/TCP use common (`buffer_id`, `offset`) hint and ignore `write`.
  // UCCL validates write-capacity hints when provided.
  if (kind == PeerTransportKind::Uccl &&
      hint.has_write_hint() &&
      hint.write.capacity != 0 && hint.write.capacity < src_bytes) {
    throw std::invalid_argument(
        "dst_hint.write.capacity is smaller than send size");
  }
}

bool detect_local_rdma_capable() {
  if (char const* env = std::getenv("UHM_RDMA_CAPABLE")) {
    return std::strcmp(env, "0") != 0;
  }

  int count = 0;
  ibv_device** devices = ibv_get_device_list(&count);
  if (!devices) return false;
  ibv_free_device_list(devices);
  return count > 0;
}

void signal_eventfd_if_needed(int event_fd) {
  if (event_fd < 0) return;
  uint64_t one = 1;
  while (true) {
    ssize_t n = ::write(event_fd, &one, sizeof(one));
    if (n == static_cast<ssize_t>(sizeof(one))) return;
    if (n < 0 && errno == EINTR) continue;
    // EAGAIN means counter is saturated; wakeup is already pending.
    if (n < 0 && errno == EAGAIN) return;
    return;
  }
}

}  // namespace

Communicator::Communicator(int gpu_id, int rank, int world_size,
                           std::shared_ptr<CommunicatorConfig> config)
    : local_gpu_idx_(gpu_id),
      global_rank_(rank),
      world_size_(world_size),
      peer_states_(static_cast<size_t>(world_size)),
      config_(config) {
  if (!config_) {
    config_ = std::make_shared<CommunicatorConfig>(CommunicatorConfig::from_env());
  }
  if (config_->oob_namespace.empty()) {
    config_->oob_namespace = "default";
  }
  request_slots_ = std::make_unique<TrackedRequest[]>(kRequestSlotCount);
  active_ring_ = std::make_unique<std::atomic<unsigned>[]>(kActiveRingSize);
  for (uint32_t i = 0; i < kActiveRingSize; ++i) {
    active_ring_[i].store(0, std::memory_order_relaxed);
  }
  completion_event_fd_ = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (completion_event_fd_ < 0) {
    throw std::runtime_error("failed to create completion eventfd");
  }
  ipc_adapter_ = std::make_shared<IpcAdapter>(
      this, generate_host_id() + "_p" + std::to_string(config_->exchanger_port),
      config_->local_id >= 0 ? config_->local_id : global_rank_,
      local_gpu_idx_);
  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
  GPU_RT_CHECK(
      gpuStreamCreateWithFlags(&host_copy_stream_, gpuStreamNonBlocking));

  bool is_server = (global_rank_ == 0);
  if (!is_server && config_->exchanger_ip == "0.0.0.0")
    config_->exchanger_ip = "127.0.0.1";
  std::cout << "[INFO] Using hierarchical bootstrap exchanger as "
            << (is_server ? "server" : "client") << " " << config_->exchanger_ip
            << std::endl;
  exchanger_client_ = std::make_shared<HierarchicalExchanger>(
      (global_rank_ == 0), config_->exchanger_ip, config_->exchanger_port,
      /*timeout_ms=*/3000, /*max_line_bytes=*/1 * 1024 * 1024,
      /*local_id=*/config_->local_id);
  if (!exchanger_client_->valid()) {
    fprintf(stderr, "[ERROR] Failed to connect to Exchanger\n");
    return;
  }

  shm_manager_.emplace();

  // Start background progress immediately so post-transport work, such as
  // host-bounce host->device copies, is not delayed until an external
  // poll()/wait_finish() call.
  progress_running_.store(true);
  progress_started_.store(true);
  progress_thread_ = std::thread(&Communicator::progress_loop, this);

  exchange_peer_metas();
  std::cout << "[INFO] Communicator " << global_rank_
            << " initialized: peer meta exchange success" << std::endl;
}

void Communicator::set_oob_namespace(std::string ns) {
  if (ns.empty()) ns = "default";
  std::lock_guard<std::mutex> lk(config_mu_);
  config_->oob_namespace = std::move(ns);
}

std::string Communicator::oob_namespace() const {
  std::lock_guard<std::mutex> lk(config_mu_);
  if (config_->oob_namespace.empty()) return "default";
  return config_->oob_namespace;
}

bool Communicator::barrier(std::string const& barrier_namespace,
                           int timeout_ms) {
  if (!exchanger_client_ || !exchanger_client_->valid()) return false;
  std::string ns = barrier_namespace.empty() ? "default" : barrier_namespace;
  uint64_t const seq = barrier_seq_.fetch_add(1, std::memory_order_relaxed);
  std::string const barrier_prefix =
      oob_namespace() + "/barrier/" + ns + "/seq/" + std::to_string(seq);
  std::string const arrive_key =
      barrier_prefix + "/rank/" + std::to_string(global_rank_);
  if (!exchanger_client_->put(arrive_key, int32_t{1})) return false;

  constexpr int kPollDelayMs = 10;
  int const max_retries =
      timeout_ms < 0
          ? -1
          : std::max(1, (timeout_ms + kPollDelayMs - 1) / kPollDelayMs);
  Exchanger::WaitOptions const wait_opt(max_retries, kPollDelayMs);
  int32_t arrived = 0;
  for (int rank = 0; rank < world_size_; ++rank) {
    std::string const key = barrier_prefix + "/rank/" + std::to_string(rank);
    if (!exchanger_client_->wait(key, arrived, wait_opt)) return false;
  }
  return true;
}

void Communicator::exchange_peer_metas() {
  CommunicatorMeta local;
  local.host_id = generate_host_id();
  local.local_id = config_->local_id >= 0 ? config_->local_id : global_rank_;
  local.rdma_capable = detect_local_rdma_capable();
  local.ip = get_local_ip();
  {
    std::lock_guard<std::mutex> lk(peer_mu_);
    auto& self = peer_states_.at(static_cast<size_t>(global_rank_));
    self.meta = local;
    self.has_meta = true;
    self.connected = true;
  }

  std::string meta_key = "meta:" + std::to_string(global_rank_);
  if (!oob_put(*exchanger_client_, oob_namespace(), meta_key, local)) {
    throw std::runtime_error(
        "failed to publish local communicator meta to exchanger");
  }

  CommunicatorMeta remote;
  std::vector<int> missing_ranks;
  for (int i = 0; i < world_size_; i++) {
    if (i == global_rank_) continue;
    std::string key = "meta:" + std::to_string(i);
    if (oob_get(*exchanger_client_, oob_namespace(), key, remote,
                bootstrap_timeout_ms())) {
      std::lock_guard<std::mutex> lk(peer_mu_);
      auto& peer = peer_states_.at(static_cast<size_t>(i));
      peer.meta = remote;
      peer.has_meta = true;
      ipc_adapter_->set_peer_local_id(i, remote.local_id);
    } else {
      missing_ranks.push_back(i);
    }
  }

  if (!missing_ranks.empty()) {
    std::ostringstream oss;
    oss << "timeout waiting for remote CommunicatorMeta from ranks ";
    for (size_t i = 0; i < missing_ranks.size(); ++i) {
      if (i != 0) oss << ",";
      oss << missing_ranks[i];
    }
    throw std::runtime_error(oss.str());
  }
}

Communicator::~Communicator() {
  if (progress_started_.load()) {
    progress_running_.store(false);
    progress_cv_.notify_all();
    signal_eventfd_if_needed(completion_event_fd_);
    if (progress_thread_.joinable()) {
      progress_thread_.join();
    }
  }

  if (ipc_adapter_) {
    ipc_adapter_->shutdown();
  }

  for (uint32_t i = 0; i < kRequestSlotCount; ++i) {
    auto* slot = &request_slots_[i];
    auto state = slot->state.load(std::memory_order_acquire);
    if (state == TrackedRequest::SlotState::Free) continue;
    if (state == TrackedRequest::SlotState::InFlight) {
      (void)poll_request_completion(slot->request_id, true);
    }
    TrackedRequest snapshot{};
    if (try_release_request_slot(slot->request_id, &snapshot)) {
      cleanup_tracked_request(snapshot);
      slot->state.store(TrackedRequest::SlotState::Free,
                        std::memory_order_release);
    } else {
      slot->state.store(TrackedRequest::SlotState::Free,
                        std::memory_order_release);
    }
  }

  if (host_copy_stream_ != nullptr) {
    int orig_device = -1;
    GPU_RT_CHECK(gpuGetDevice(&orig_device));
    GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
    GPU_RT_CHECK(gpuStreamDestroy(host_copy_stream_));
    GPU_RT_CHECK(gpuSetDevice(orig_device));
    host_copy_stream_ = nullptr;
  }

  for (auto const& [buffer_id, item] : mr_manager_.list_local_mrs()) {
    uint64_t const registered_id = buffer_id;
    if (uccl_adapter_ && uccl_adapter_->is_initialized()) {
      std::lock_guard<std::mutex> lk(uccl_reg_mu_);
      if (uccl_registered_mrs_.find(registered_id) !=
          uccl_registered_mrs_.end()) {
        uccl_adapter_->deregister_memory(registered_id);
        uccl_registered_mrs_.erase(registered_id);
      }
      uccl_direct_reg_failed_mrs_.erase(registered_id);
    }
    (void)mr_manager_.delete_mr(static_cast<uint32_t>(buffer_id));
  }

  std::vector<uint32_t> local_ipc_buffer_ids;
  {
    std::lock_guard<std::mutex> lk(resource_mu_);
    local_ipc_buffer_ids.reserve(local_buffer_to_ipc_.size());
    for (auto const& kv : local_buffer_to_ipc_) {
      local_ipc_buffer_ids.push_back(kv.first);
    }
  }
  for (uint32_t buffer_id : local_ipc_buffer_ids) {
    (void)dereg_ipc(buffer_id);
  }

  for (int i = 0; i < world_size_; ++i) {
    if (i == global_rank_) continue;
    ipc_manager_.delete_ipc(i);
  }

  // Destroy bounce pool before transport adapters to avoid dangling references
  // in the deregister callback during pool teardown.
  shm_manager_.reset();
  uccl_adapter_.reset();
  tcp_adapter_.reset();
  ipc_adapter_.reset();
  if (completion_event_fd_ >= 0) {
    ::close(completion_event_fd_);
    completion_event_fd_ = -1;
  }

  std::cout << "[INFO] Communicator " << global_rank_ << " resources released"
            << std::endl;
}

unsigned Communicator::make_request_id(uint32_t slot_idx, uint32_t generation) {
  uint32_t gen = generation & kRequestGenerationMask;
  if (gen == 0) gen = 1;
  return (gen << kRequestSlotBits) | (slot_idx & (kRequestSlotCount - 1u));
}

uint32_t Communicator::request_slot_index(unsigned req_id) {
  return req_id & (kRequestSlotCount - 1u);
}

uint32_t Communicator::request_generation(unsigned req_id) {
  return (req_id >> kRequestSlotBits) & kRequestGenerationMask;
}

Communicator::TrackedRequest* Communicator::resolve_request_slot(
    unsigned req_id) const {
  if (req_id == 0) return nullptr;
  uint32_t idx = request_slot_index(req_id);
  if (idx >= kRequestSlotCount) return nullptr;
  TrackedRequest* slot = &request_slots_[idx];
  uint32_t gen =
      slot->generation.load(std::memory_order_acquire) & kRequestGenerationMask;
  if (gen == 0 || gen != request_generation(req_id)) return nullptr;
  auto state = slot->state.load(std::memory_order_acquire);
  if (state == TrackedRequest::SlotState::Free) return nullptr;
  return slot;
}

Communicator::TrackedRequest* Communicator::allocate_request_slot(
    unsigned* out_req_id) {
  if (out_req_id == nullptr) return nullptr;
  uint32_t start =
      request_alloc_cursor_.fetch_add(1, std::memory_order_relaxed) &
      (kRequestSlotCount - 1u);
  for (uint32_t i = 0; i < kRequestSlotCount; ++i) {
    uint32_t idx = (start + i) & (kRequestSlotCount - 1u);
    TrackedRequest* slot = &request_slots_[idx];
    auto expected = TrackedRequest::SlotState::Free;
    if (!slot->state.compare_exchange_strong(
            expected, TrackedRequest::SlotState::Reserved,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      continue;
    }
    uint32_t gen =
        (slot->generation.fetch_add(1, std::memory_order_acq_rel) + 1u) &
        kRequestGenerationMask;
    if (gen == 0) {
      gen = 1;
      slot->generation.store(gen, std::memory_order_release);
    }
    unsigned rid = make_request_id(idx, gen);
    slot->request_id = rid;
    slot->adapter_request_id = 0;
    slot->peer_rank = -1;
    slot->kind = PeerTransportKind::Unknown;
    slot->notified = false;
    slot->needs_host_to_device_copy = false;
    slot->host_copy_submitted = false;
    slot->completion_buffer = nullptr;
    slot->completion_offset = 0;
    slot->completion_bytes = 0;
    slot->host_copy_event = nullptr;
    slot->bounce = {};
    slot->bounce_owner.reset();
    *out_req_id = rid;
    return slot;
  }
  return nullptr;
}

bool Communicator::try_release_request_slot(unsigned req_id,
                                            TrackedRequest* out_snapshot) {
  TrackedRequest* slot = resolve_request_slot(req_id);
  if (slot == nullptr) return false;
  auto state = slot->state.load(std::memory_order_acquire);
  if (state != TrackedRequest::SlotState::Completed &&
      state != TrackedRequest::SlotState::Failed) {
    return false;
  }
  if (!slot->state.compare_exchange_strong(
          state, TrackedRequest::SlotState::Releasing,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return false;
  }
  if (out_snapshot) {
    out_snapshot->state.store(TrackedRequest::SlotState::Releasing,
                              std::memory_order_relaxed);
    out_snapshot->generation.store(
        slot->generation.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    out_snapshot->request_id = slot->request_id;
    out_snapshot->adapter_request_id = slot->adapter_request_id;
    out_snapshot->peer_rank = slot->peer_rank;
    out_snapshot->kind = slot->kind;
    out_snapshot->notified = slot->notified;
    out_snapshot->needs_host_to_device_copy = slot->needs_host_to_device_copy;
    out_snapshot->host_copy_submitted = slot->host_copy_submitted;
    out_snapshot->completion_buffer = slot->completion_buffer;
    out_snapshot->completion_offset = slot->completion_offset;
    out_snapshot->completion_bytes = slot->completion_bytes;
    out_snapshot->host_copy_event = slot->host_copy_event;
    out_snapshot->bounce = slot->bounce;
    out_snapshot->bounce_owner = std::move(slot->bounce_owner);
  }
  slot->state.store(TrackedRequest::SlotState::Free, std::memory_order_release);
  return true;
}

bool Communicator::enqueue_active_request(unsigned req_id) {
  while (true) {
    uint32_t tail = active_tail_.load(std::memory_order_acquire);
    uint32_t head = active_head_.load(std::memory_order_acquire);
    if (tail - head >= kActiveRingSize) {
      return false;
    }
    if (!active_tail_.compare_exchange_weak(tail, tail + 1,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
      continue;
    }
    active_ring_[tail & (kActiveRingSize - 1u)].store(
        req_id, std::memory_order_release);
    return true;
  }
}

bool Communicator::dequeue_active_request(unsigned* out_req_id) {
  if (out_req_id == nullptr) return false;
  uint32_t head = active_head_.load(std::memory_order_acquire);
  uint32_t tail = active_tail_.load(std::memory_order_acquire);
  if (head == tail) return false;
  if (!active_head_.compare_exchange_strong(head, head + 1,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
    return false;
  }
  unsigned req_id = active_ring_[head & (kActiveRingSize - 1u)].exchange(
      0, std::memory_order_acq_rel);
  if (req_id == 0) return false;
  *out_req_id = req_id;
  return true;
}

void Communicator::notify_request_completion() {
  completion_seq_.fetch_add(1, std::memory_order_release);
  signal_eventfd_if_needed(completion_event_fd_);
}

UcclTransportAdapter& Communicator::ensure_uccl_adapter(
    CommunicatorMeta const& local_meta) {
  if (!uccl_adapter_) {
    maybe_configure_uccl_socket_ifname(local_meta.ip);
    UcclTransportConfig uccl_cfg;
    uccl_adapter_ = std::make_unique<UcclTransportAdapter>(
        local_gpu_idx_, world_size_, std::move(uccl_cfg));
  }
  return *uccl_adapter_;
}

bool Communicator::exchange_uccl_peer_info(
    int rank, UcclTransportAdapter& uccl_adapter,
    UCCLP2PInfo* out_remote_p2p_info) {
  if (out_remote_p2p_info == nullptr) return false;

  int dev_idx = uccl_adapter.get_best_dev_idx(local_gpu_idx_);
  if (dev_idx < 0) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " UCCL get_best_dev_idx failed for local gpu "
              << local_gpu_idx_ << std::endl;
    return false;
  }
  uint16_t local_port = uccl_adapter.get_p2p_listen_port(dev_idx);
  if (local_port == 0) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " UCCL local listen port is invalid for dev " << dev_idx
              << std::endl;
    return false;
  }
  std::string local_ip_addr = uccl_adapter.get_p2p_listen_ip(dev_idx);
  if (local_ip_addr.empty()) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " UCCL local listen ip is empty for dev " << dev_idx
              << std::endl;
    return false;
  }

  UCCLP2PInfo local_p2p_info(local_ip_addr, local_port, dev_idx, local_gpu_idx_);
  std::string p2p_key = uccl_p2p_key(global_rank_, rank);
  std::string peer_p2p_key = uccl_p2p_key(rank, global_rank_);

  return oob_put(*exchanger_client_, oob_namespace(), p2p_key,
                 local_p2p_info) &&
         oob_get(*exchanger_client_, oob_namespace(), peer_p2p_key,
                 *out_remote_p2p_info, bootstrap_timeout_ms());
}

TcpTransportAdapter& Communicator::ensure_tcp_adapter(
    CommunicatorMeta const& local_meta) {
  if (!tcp_adapter_) {
    tcp_adapter_ =
        std::make_unique<TcpTransportAdapter>(local_meta.ip, global_rank_);
  }
  return *tcp_adapter_;
}

Communicator::ResolvedPeer Communicator::resolve_peer(int rank) const {
  if (rank == global_rank_) {
    throw std::invalid_argument("transport peer rank cannot be self");
  }
  if (rank < 0 || rank >= world_size_) {
    throw std::invalid_argument("transport peer rank out of range");
  }

  std::lock_guard<std::mutex> lk(peer_mu_);
  auto const& local_peer = peer_states_.at(static_cast<size_t>(global_rank_));
  auto const& remote_peer = peer_states_.at(static_cast<size_t>(rank));
  if (!local_peer.has_meta || !remote_peer.has_meta) {
    throw std::runtime_error("transport peer metadata is not established");
  }

  ResolvedPeer resolved;
  resolved.local_meta = local_peer.meta;
  resolved.remote_meta = remote_peer.meta;
  resolved.kind = resolve_peer_transport_kind(*config_, resolved.local_meta,
                                              resolved.remote_meta);
  return resolved;
}

bool Communicator::try_fallback_tcp_connect(
    int rank, CommunicatorMeta const& local_meta) {
  if (config_->preferred_transport != PreferredTransport::Auto) return false;
  auto& tcp_adapter = ensure_tcp_adapter(local_meta);
  if (tcp_adapter.has_peer(rank)) {
    mark_peer_path_ready(rank, PeerTransportKind::Tcp);
    return true;
  }

  TcpP2PInfo local_p2p_info(tcp_adapter.get_listen_ip(),
                            tcp_adapter.get_listen_port());
  std::string p2p_key = tcp_p2p_key(global_rank_, rank);
  std::string peer_p2p_key = tcp_p2p_key(rank, global_rank_);
  if (!oob_put(*exchanger_client_, oob_namespace(), p2p_key,
               local_p2p_info))
    return false;

  TcpP2PInfo remote_p2p_info;
  if (!oob_get(*exchanger_client_, oob_namespace(), peer_p2p_key,
               remote_p2p_info, bootstrap_timeout_ms())) {
    return false;
  }

  PeerConnectSpec spec{};
  spec.peer_rank = rank;
  spec.type = PeerConnectType::Connect;
  spec.detail = TcpPeerConnectSpec{remote_p2p_info.ip, remote_p2p_info.port};
  if (!tcp_adapter.ensure_peer(spec)) {
    return false;
  }
  {
    mark_peer_path_ready(rank, PeerTransportKind::Tcp);
    std::cout << "[INFO] Communicator " << global_rank_
              << " TCP fallback connect_to succeeded to rank " << rank
              << std::endl;
  }
  return true;
}

bool Communicator::try_fallback_tcp_accept(
    int rank, CommunicatorMeta const& local_meta) {
  if (config_->preferred_transport != PreferredTransport::Auto) return false;
  auto& tcp_adapter = ensure_tcp_adapter(local_meta);
  if (tcp_adapter.has_peer(rank)) {
    mark_peer_path_ready(rank, PeerTransportKind::Tcp);
    return true;
  }

  std::string p2p_key = tcp_p2p_key(global_rank_, rank);
  std::string peer_p2p_key = tcp_p2p_key(rank, global_rank_);
  TcpP2PInfo local_p2p_info(tcp_adapter.get_listen_ip(),
                            tcp_adapter.get_listen_port());
  if (!oob_put(*exchanger_client_, oob_namespace(), p2p_key, local_p2p_info))
    return false;

  TcpP2PInfo remote_p2p_info;
  if (!oob_get(*exchanger_client_, oob_namespace(), peer_p2p_key,
               remote_p2p_info, bootstrap_timeout_ms())) {
    return false;
  }

  PeerConnectSpec spec{};
  spec.peer_rank = rank;
  spec.type = PeerConnectType::Accept;
  spec.detail = TcpPeerConnectSpec{remote_p2p_info.ip, 0};
  if (!tcp_adapter.ensure_peer(spec)) {
    return false;
  }
  {
    mark_peer_path_ready(rank, PeerTransportKind::Tcp);
    std::cout << "[INFO] Communicator " << global_rank_
              << " TCP fallback accept_from succeeded from rank " << rank
              << std::endl;
  }
  return true;
}

bool Communicator::connect(int rank) {
  if (rank == global_rank_) return true;
  if (rank < 0 || rank >= world_size_) {
    std::cerr << "[ERROR] Communicator " << global_rank_ << " invalid rank "
              << rank << ", world_size=" << world_size_ << std::endl;
    return false;
  }
  if (has_peer_path(rank)) return true;

  ResolvedPeer resolved;
  try {
    resolved = resolve_peer(rank);
  } catch (std::exception const& ex) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " failed to resolve transport for rank " << rank << ": "
              << ex.what() << std::endl;
    return false;
  }

  if (resolved.kind == PeerTransportKind::Uccl) {
    auto& uccl_adapter = ensure_uccl_adapter(resolved.local_meta);
    if (!uccl_adapter.has_peer(rank)) {
      int local_dev_idx = uccl_adapter.get_best_dev_idx(local_gpu_idx_);
      if (local_dev_idx < 0) {
        return try_fallback_tcp_connect(rank, resolved.local_meta);
      }
      UCCLP2PInfo remote_p2p_info;
      if (!exchange_uccl_peer_info(rank, uccl_adapter, &remote_p2p_info)) {
        return try_fallback_tcp_connect(rank, resolved.local_meta);
      }
      PeerConnectSpec spec{};
      spec.peer_rank = rank;
      spec.type = PeerConnectType::Connect;
      spec.detail = UcclPeerConnectSpec{
          remote_p2p_info.ip, remote_p2p_info.port,    local_dev_idx,
          local_gpu_idx_,     remote_p2p_info.dev_idx, remote_p2p_info.gpu_idx};
      if (!uccl_adapter.ensure_peer(spec)) {
        return try_fallback_tcp_connect(rank, resolved.local_meta);
      }
    }
    mark_peer_path_ready(rank, PeerTransportKind::Uccl);
    register_existing_local_mrs_with_uccl();
    return true;
  }

  if (resolved.kind == PeerTransportKind::Ipc) {
    PeerConnectSpec spec{};
    spec.peer_rank = rank;
    spec.type = PeerConnectType::Connect;
    spec.detail = IpcPeerConnectSpec{};
    if (!ipc_adapter_->ensure_peer(spec)) {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " IPC connect_to failed to rank " << rank << std::endl;
      ipc_adapter_->close_peer(rank);
      return false;
    }
    mark_peer_path_ready(rank, PeerTransportKind::Ipc);
    return true;
  }

  if (resolved.kind == PeerTransportKind::Tcp) {
    auto& tcp_adapter = ensure_tcp_adapter(resolved.local_meta);
    if (!tcp_adapter.has_peer(rank)) {
      TcpP2PInfo local_p2p_info(tcp_adapter.get_listen_ip(),
                                tcp_adapter.get_listen_port());
      std::string p2p_key = tcp_p2p_key(global_rank_, rank);
      std::string peer_p2p_key = tcp_p2p_key(rank, global_rank_);
      TcpP2PInfo remote_p2p_info;
      if (!oob_put(*exchanger_client_, oob_namespace(), p2p_key,
                   local_p2p_info) ||
          !oob_get(*exchanger_client_, oob_namespace(), peer_p2p_key,
                   remote_p2p_info, bootstrap_timeout_ms())) {
        return false;
      }
      PeerConnectSpec spec{};
      spec.peer_rank = rank;
      spec.type = PeerConnectType::Connect;
      spec.detail =
          TcpPeerConnectSpec{remote_p2p_info.ip, remote_p2p_info.port};
      if (!tcp_adapter.ensure_peer(spec)) {
        std::cerr << "[ERROR] Communicator " << global_rank_
                  << " TCP connect_to failed to rank " << rank << std::endl;
        return false;
      }
    }
    mark_peer_path_ready(rank, PeerTransportKind::Tcp);
    return true;
  }
  return false;
}

bool Communicator::accept(int rank) {
  if (rank == global_rank_) return true;
  if (rank < 0 || rank >= world_size_) {
    std::cerr << "[ERROR] Communicator " << global_rank_ << " invalid rank "
              << rank << ", world_size=" << world_size_ << std::endl;
    return false;
  }
  if (has_peer_path(rank)) return true;

  ResolvedPeer resolved;
  try {
    resolved = resolve_peer(rank);
  } catch (std::exception const& ex) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " failed to resolve transport for rank " << rank << ": "
              << ex.what() << std::endl;
    return false;
  }

  if (resolved.kind == PeerTransportKind::Uccl) {
    auto& uccl_adapter = ensure_uccl_adapter(resolved.local_meta);
    if (!uccl_adapter.has_peer(rank)) {
      int local_dev_idx = uccl_adapter.get_best_dev_idx(local_gpu_idx_);
      if (local_dev_idx < 0) {
        return try_fallback_tcp_accept(rank, resolved.local_meta);
      }
      UCCLP2PInfo remote_p2p_info;
      if (!exchange_uccl_peer_info(rank, uccl_adapter, &remote_p2p_info)) {
        return try_fallback_tcp_accept(rank, resolved.local_meta);
      }
      PeerConnectSpec spec{};
      spec.peer_rank = rank;
      spec.type = PeerConnectType::Accept;
      spec.detail = UcclPeerConnectSpec{
          remote_p2p_info.ip, remote_p2p_info.port,    local_dev_idx,
          local_gpu_idx_,     remote_p2p_info.dev_idx, remote_p2p_info.gpu_idx};
      if (!uccl_adapter.ensure_peer(spec)) {
        return try_fallback_tcp_accept(rank, resolved.local_meta);
      }
    }
    mark_peer_path_ready(rank, PeerTransportKind::Uccl);
    register_existing_local_mrs_with_uccl();
    return true;
  }

  if (resolved.kind == PeerTransportKind::Ipc) {
    PeerConnectSpec spec{};
    spec.peer_rank = rank;
    spec.type = PeerConnectType::Accept;
    spec.detail = IpcPeerConnectSpec{};
    if (!ipc_adapter_->ensure_peer(spec)) {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " IPC accept_from failed from rank " << rank << std::endl;
      return false;
    }
    mark_peer_path_ready(rank, PeerTransportKind::Ipc);
    return true;
  }

  if (resolved.kind == PeerTransportKind::Tcp) {
    auto& tcp_adapter = ensure_tcp_adapter(resolved.local_meta);
    if (!tcp_adapter.has_peer(rank)) {
      std::string p2p_key = tcp_p2p_key(global_rank_, rank);
      std::string peer_p2p_key = tcp_p2p_key(rank, global_rank_);
      TcpP2PInfo remote_p2p_info;
      TcpP2PInfo local_p2p_info(tcp_adapter.get_listen_ip(),
                                tcp_adapter.get_listen_port());
      if (!oob_put(*exchanger_client_, oob_namespace(), p2p_key,
                   local_p2p_info) ||
          !oob_get(*exchanger_client_, oob_namespace(), peer_p2p_key,
                   remote_p2p_info, bootstrap_timeout_ms())) {
        return false;
      }
      PeerConnectSpec spec{};
      spec.peer_rank = rank;
      spec.type = PeerConnectType::Accept;
      spec.detail = TcpPeerConnectSpec{remote_p2p_info.ip, 0};
      if (!tcp_adapter.ensure_peer(spec)) {
        std::cerr << "[ERROR] Communicator " << global_rank_
                  << " TCP accept_from failed from rank " << rank << std::endl;
        return false;
      }
    }
    mark_peer_path_ready(rank, PeerTransportKind::Tcp);
    return true;
  }
  return false;
}

bool Communicator::has_peer_path(int rank) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  if (rank < 0 || rank >= world_size_) return false;
  return peer_states_.at(static_cast<size_t>(rank)).connected;
}

void Communicator::mark_peer_path_ready(int rank, PeerTransportKind kind) {
  std::lock_guard<std::mutex> lk(peer_mu_);
  auto& peer = peer_states_.at(static_cast<size_t>(rank));
  peer.kind = kind;
  peer.connected = true;
}

PeerTransportKind Communicator::get_peer_transport_kind(int rank) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  if (rank < 0 || rank >= world_size_) {
    throw std::runtime_error("transport peer rank out of range");
  }
  auto const& peer = peer_states_.at(static_cast<size_t>(rank));
  if (!peer.has_meta) {
    throw std::runtime_error("transport peer session is not established");
  }
  if (!peer.connected) {
    throw std::runtime_error("transport peer path is not established");
  }
  if (peer.kind == PeerTransportKind::Unknown) {
    throw std::runtime_error("transport peer kind is unknown");
  }
  return peer.kind;
}

PeerTransportKind Communicator::peer_transport_kind(int rank) const {
  return get_peer_transport_kind(rank);
}

TransportAdapter* Communicator::get_adapter(PeerTransportKind kind) {
  switch (kind) {
    case PeerTransportKind::Uccl:
      return uccl_adapter_.get();
    case PeerTransportKind::Tcp:
      return tcp_adapter_.get();
    case PeerTransportKind::Ipc:
      return ipc_adapter_.get();
    default:
      return nullptr;
  }
}

bool Communicator::same_host(int rank) const {
  if (rank == global_rank_) return true;
  std::lock_guard<std::mutex> lk(peer_mu_);
  auto const& local_peer = peer_states_.at(static_cast<size_t>(global_rank_));
  auto const& remote_peer = peer_states_.at(static_cast<size_t>(rank));
  if (!local_peer.has_meta || !remote_peer.has_meta) {
    throw std::runtime_error("transport peer metadata is not established");
  }
  return local_peer.meta.host_id == remote_peer.meta.host_id;
}

void Communicator::register_existing_local_mrs_with_uccl() {
  if (!uccl_adapter_ || !uccl_adapter_->is_initialized()) return;
  for (auto const& [buffer_id, item] : mr_manager_.list_local_mrs()) {
    void* ptr = reinterpret_cast<void*>(item.mr.address);
    size_t len = static_cast<size_t>(item.mr.length);
    (void)ensure_uccl_memory_registered(buffer_id, ptr, len);
  }
}

bool Communicator::ensure_uccl_memory_registered(uint32_t buffer_id, void* ptr,
                                                 size_t len) {
  if (uccl_adapter_ && uccl_adapter_->is_initialized()) {
    if (uccl_adapter_->is_memory_registered(buffer_id)) return true;

    void* base_ptr = ptr;
    size_t mr_len = len;
    bool is_direct_local_mr = false;

    auto item = mr_manager_.get_mr(static_cast<uint32_t>(buffer_id));
    if (item.valid) {
      base_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(item.mr.address));
      mr_len = static_cast<size_t>(item.mr.length);
      is_direct_local_mr = true;
    }

    if (is_direct_local_mr) {
      std::lock_guard<std::mutex> lk(uccl_reg_mu_);
      if (uccl_direct_reg_failed_mrs_.find(buffer_id) !=
          uccl_direct_reg_failed_mrs_.end()) {
        return false;
      }
    }

    if (base_ptr == nullptr || mr_len == 0) {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " has invalid base pointer or length for UCCL registration, "
                << "buffer_id=" << buffer_id << std::endl;
      return false;
    }

    bool ok = uccl_adapter_->register_memory(buffer_id, base_ptr, mr_len);
    if (!ok) {
      if (is_direct_local_mr) {
        std::lock_guard<std::mutex> lk(uccl_reg_mu_);
        uccl_direct_reg_failed_mrs_.insert(buffer_id);
        std::cerr << "[WARN] Communicator " << global_rank_
                  << " failed to register local GPU MR " << buffer_id
                  << " with UCCL, base=" << base_ptr << " len=" << mr_len
                  << "; future requests will fallback to host bounce"
                  << std::endl;
      } else {
        std::cerr << "[ERROR] Communicator " << global_rank_
                  << " failed to register host bounce MR " << buffer_id
                  << " with UCCL, base=" << base_ptr << " len=" << mr_len
                  << std::endl;
      }
    } else {
      std::lock_guard<std::mutex> lk(uccl_reg_mu_);
      uccl_registered_mrs_.insert(buffer_id);
    }
    return ok;
  }

  return true;
}

bool Communicator::complete_host_bounce_recv(TrackedRequest& tracked,
                                             bool blocking) {
  if (!tracked.needs_host_to_device_copy) return true;
  if (!tracked.bounce.ptr || !tracked.completion_buffer) return false;

  int orig_device = -1;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset = UKernel::Transport::finally(
      [&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });
  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

  if (!tracked.host_copy_submitted) {
    if (tracked.host_copy_event == nullptr) {
      GPU_RT_CHECK(gpuEventCreateWithFlags(&tracked.host_copy_event,
                                           gpuEventDisableTiming));
    }
    GPU_RT_CHECK(gpuMemcpyAsync(static_cast<char*>(tracked.completion_buffer) +
                                    tracked.completion_offset,
                                tracked.bounce.ptr, tracked.completion_bytes,
                                gpuMemcpyHostToDevice, host_copy_stream_));
    GPU_RT_CHECK(gpuEventRecord(tracked.host_copy_event, host_copy_stream_));
    tracked.host_copy_submitted = true;
    if (!blocking) return false;
  }

  if (blocking) {
    GPU_RT_CHECK(gpuEventSynchronize(tracked.host_copy_event));
  } else {
    gpuError_t query = gpuEventQuery(tracked.host_copy_event);
    if (query == gpuErrorNotReady) return false;
    if (query != gpuSuccess) {
      GPU_RT_CHECK(query);
    }
  }
  GPU_RT_CHECK(gpuEventDestroy(tracked.host_copy_event));
  tracked.host_copy_event = nullptr;
  tracked.host_copy_submitted = false;
  tracked.needs_host_to_device_copy = false;
  return true;
}

void Communicator::cleanup_tracked_request(TrackedRequest& tracked) {
  unsigned adapter_id = tracked.adapter_request_id;
  if (tracked.kind == PeerTransportKind::Uccl && uccl_adapter_) {
    uccl_adapter_->release_request(adapter_id);
  }
  if (tracked.kind == PeerTransportKind::Tcp && tcp_adapter_) {
    tcp_adapter_->release_request(adapter_id);
  }
  if (tracked.kind == PeerTransportKind::Ipc && ipc_adapter_) {
    ipc_adapter_->release_request(adapter_id);
  }
  if (tracked.host_copy_event != nullptr) {
    GPU_RT_CHECK(gpuEventDestroy(tracked.host_copy_event));
    tracked.host_copy_event = nullptr;
  }
  tracked.host_copy_submitted = false;
  tracked.needs_host_to_device_copy = false;
  // NOTE: bounce memory/MR is request-scoped today. Cleanup always tears down
  // the request-owned bounce resources. A future performance PR can replace
  // this with a longer-lived pool and defer teardown outside request cleanup.
  tracked.bounce_owner.reset();
  if (shm_manager_ && tracked.bounce.valid) {
    (void)mr_manager_.delete_mr(tracked.bounce.ptr);
    shm_manager_->delete_local_shm(tracked.bounce.shm_id);
    tracked.bounce = {};
  }
}

SHMManager& Communicator::require_shm_manager(char const* caller) {
  if (shm_manager_.has_value()) return *shm_manager_;
  std::ostringstream oss;
  oss << caller << " called after SHMManager teardown";
  throw std::runtime_error(oss.str());
}

unsigned Communicator::isend(int rank, LocalSlice src,
                             std::optional<RemoteSlice> dst_hint) {
  if (src.buffer_id == 0 || src.bytes == 0) {
    throw std::invalid_argument("isend requires non-empty local slice");
  }
  if (!has_peer_path(rank) && !connect(rank)) {
    throw std::runtime_error("transport peer path is not established");
  }
  MR local_mr = get_mr(src.buffer_id);
  if (src.offset > static_cast<size_t>(local_mr.length) ||
      src.bytes > static_cast<size_t>(local_mr.length) - src.offset) {
    throw std::invalid_argument("isend local slice out of range");
  }

  auto peer_kind = get_peer_transport_kind(rank);
  auto* adapter = get_adapter(peer_kind);
  if (!adapter) {
    throw std::runtime_error("failed to get adapter for peer");
  }
  std::optional<RemoteSlice> effective_dst_hint = dst_hint;
  // Auto-enrich one-sided write hint from exchanged remote MR metadata when the
  // caller only provides {remote buffer_id, offset}. This keeps Python/CCL APIs
  // simple while still enabling UCCL fast paths.
  if (peer_kind == PeerTransportKind::Uccl &&
      effective_dst_hint.has_value() && !effective_dst_hint->has_write_hint() &&
      effective_dst_hint->buffer_id != 0) {
    try {
      MR remote_mr = get_mr(rank, effective_dst_hint->buffer_id);
      if (remote_mr.address != 0 && remote_mr.key != 0) {
        RemoteSlice enriched = *effective_dst_hint;
        uint64_t addr = remote_mr.address;
        if (enriched.offset <= std::numeric_limits<uint64_t>::max() - addr) {
          addr += static_cast<uint64_t>(enriched.offset);
        }
        uint64_t remaining = 0;
        if (remote_mr.length > enriched.offset) {
          remaining = remote_mr.length - static_cast<uint64_t>(enriched.offset);
        }
        enriched.write.addr = addr;
        enriched.write.key = remote_mr.key;
        enriched.write.capacity = static_cast<uint32_t>(
            std::min<uint64_t>(remaining, std::numeric_limits<uint32_t>::max()));
        effective_dst_hint = enriched;
      }
    } catch (std::exception const&) {
      // Remote MR may be unavailable for this buffer_id; keep the original hint
      // and let adapter fallback logic handle it.
    }
  }
  validate_dst_hint_for_transport(peer_kind, effective_dst_hint, src.bytes);

  void* local_ptr = reinterpret_cast<void*>(
      static_cast<uintptr_t>(local_mr.address) + src.offset);

  bool needs_uccl_registration = (peer_kind == PeerTransportKind::Uccl);
  bool use_shareable = (peer_kind == PeerTransportKind::Ipc);
  bool needs_bounce =
      (peer_kind == PeerTransportKind::Tcp) ||
      (peer_kind == PeerTransportKind::Ipc) ||
      ((peer_kind == PeerTransportKind::Uccl &&
        !ensure_uccl_memory_registered(src.buffer_id, local_ptr, src.bytes)));
  unsigned rid = 0;
  TrackedRequest* slot = allocate_request_slot(&rid);
  if (slot == nullptr) return 0;
  slot->peer_rank = rank;
  slot->kind = peer_kind;
  std::shared_ptr<SHMItem> bounce_owner;
  if (needs_bounce) {
    SHMManager& shm_manager = require_shm_manager("Communicator::isend");
    // Request-scoped lease for send bounce memory. Current design allocates
    // and frees per request; this is intentionally simple but not optimal for
    // steady-state throughput. Poolization can be added in a follow-up PR.
    bounce_owner =
        std::shared_ptr<SHMItem>(new SHMItem{}, [this](SHMItem* lease) {
          if (lease != nullptr) {
            if (shm_manager_.has_value() && lease->valid) {
              (void)mr_manager_.delete_mr(lease->ptr);
              shm_manager_->delete_local_shm(lease->shm_id);
            }
            delete lease;
          }
        });
    slot->bounce_owner = bounce_owner;
    (void)shm_manager;
  }
  auto bounce_provider = [this, needs_bounce, needs_uccl_registration,
                          use_shareable,
                          bounce_owner](size_t bytes) -> BounceBufferInfo {
    if (!needs_bounce || !bounce_owner) return {};
    SHMManager& shm_manager =
        require_shm_manager("Communicator::isend::bounce_provider");
    if (!bounce_owner->valid) {
      // Lazy allocate on first use within this request.
      *bounce_owner = shm_manager.create_local_shm(bytes, use_shareable);
    }
    BounceBufferInfo info;
    info.ptr = bounce_owner->ptr;
    if (needs_uccl_registration) {
      // Request-scoped bounce buffer_id. The MR is deleted in request cleanup.
      uint32_t temp_buffer_id =
          next_ephemeral_buffer_id_.fetch_add(1, std::memory_order_relaxed);
      if (temp_buffer_id == 0) {
        temp_buffer_id =
            next_ephemeral_buffer_id_.fetch_add(1, std::memory_order_relaxed);
      }
      (void)mr_manager_.create_local_mr(temp_buffer_id, bounce_owner->ptr,
                                        bytes);
      info.buffer_id = temp_buffer_id;
    }
    if (bounce_owner->shareable) {
      info.shm_name = shm_manager.get_local_shm(bounce_owner->shm_id).shm_name;
    }
    return info;
  };

  unsigned result =
      adapter->send_async(rank, local_ptr, src.bytes, src.buffer_id,
                          effective_dst_hint, bounce_provider);
  if (result == 0) {
    slot->state.store(TrackedRequest::SlotState::Releasing,
                      std::memory_order_release);
    cleanup_tracked_request(*slot);
    slot->state.store(TrackedRequest::SlotState::Free,
                      std::memory_order_release);
    return 0;
  }
  slot->adapter_request_id = result;
  slot->state.store(TrackedRequest::SlotState::InFlight,
                    std::memory_order_release);
  inflight_request_count_.fetch_add(1, std::memory_order_release);
  (void)enqueue_active_request(rid);
  progress_cv_.notify_all();
  return rid;
}

unsigned Communicator::irecv(int rank, LocalSlice dst) {
  if (dst.buffer_id == 0 || dst.bytes == 0) {
    throw std::invalid_argument("irecv requires non-empty local slice");
  }
  if (!has_peer_path(rank) && !connect(rank)) {
    throw std::runtime_error("transport peer path is not established");
  }
  MR local_mr = get_mr(dst.buffer_id);
  if (dst.offset > static_cast<size_t>(local_mr.length) ||
      dst.bytes > static_cast<size_t>(local_mr.length) - dst.offset) {
    throw std::invalid_argument("irecv local slice out of range");
  }

  auto peer_kind = get_peer_transport_kind(rank);
  auto* adapter = get_adapter(peer_kind);
  if (!adapter) {
    throw std::runtime_error("failed to get adapter for peer");
  }

  void* local_ptr = reinterpret_cast<void*>(
      static_cast<uintptr_t>(local_mr.address) + dst.offset);

  bool needs_uccl_registration = (peer_kind == PeerTransportKind::Uccl);
  bool use_shareable = (peer_kind == PeerTransportKind::Ipc);
  unsigned rid = 0;
  TrackedRequest* slot = allocate_request_slot(&rid);
  if (slot == nullptr) return 0;
  slot->peer_rank = rank;
  slot->kind = peer_kind;

  auto needs_bounce =
      (peer_kind == PeerTransportKind::Tcp) ||
      ((peer_kind == PeerTransportKind::Uccl &&
        !ensure_uccl_memory_registered(dst.buffer_id, local_ptr, dst.bytes)));

  // IPC fast path metadata: let sender resolve remote pointer by dst.buffer_id
  // and skip per-request ipc_cache handshake when possible.
  if (peer_kind == PeerTransportKind::Ipc) {
    // Publish MR base mapping (not slice pointer) so sender-side
    // get_ipc(remote_id) + remote_offset applies offset exactly once.
    void* ipc_base_ptr =
        reinterpret_cast<void*>(static_cast<uintptr_t>(local_mr.address));
    size_t ipc_bytes = static_cast<size_t>(local_mr.length);
    if (!reg_ipc(dst.buffer_id, ipc_base_ptr, ipc_bytes, true)) {
      std::cerr << "[WARN] Communicator " << global_rank_
                << " failed to publish IPC buffer metadata for rank " << rank
                << ", buffer_id=" << dst.buffer_id
                << "; sender may fallback to ipc_cache handshake" << std::endl;
    }
  }

  if (needs_bounce) {
    SHMManager& shm_manager = require_shm_manager("Communicator::irecv");
    // Request-scoped recv bounce. It is released when this request completes.
    slot->bounce = shm_manager.create_local_shm(dst.bytes, use_shareable);
    slot->needs_host_to_device_copy = true;
    slot->completion_buffer = local_ptr;
    slot->completion_bytes = dst.bytes;
  }

  auto bounce_provider = [this, slot, needs_bounce, needs_uccl_registration,
                          use_shareable](size_t bytes) -> BounceBufferInfo {
    if (!needs_bounce) return {};
    SHMManager& shm_manager =
        require_shm_manager("Communicator::irecv::bounce_provider");
    if (!slot->bounce.valid) {
      // Lazy allocate on first use within this request.
      slot->bounce = shm_manager.create_local_shm(bytes, use_shareable);
    }
    BounceBufferInfo info;
    info.ptr = slot->bounce.ptr;
    if (needs_uccl_registration) {
      // Request-scoped bounce buffer_id. The MR is deleted in request cleanup.
      uint32_t temp_buffer_id =
          next_ephemeral_buffer_id_.fetch_add(1, std::memory_order_relaxed);
      if (temp_buffer_id == 0) {
        temp_buffer_id =
            next_ephemeral_buffer_id_.fetch_add(1, std::memory_order_relaxed);
      }
      (void)mr_manager_.create_local_mr(temp_buffer_id, slot->bounce.ptr,
                                        bytes);
      info.buffer_id = temp_buffer_id;
    }
    if (slot->bounce.shareable) {
      info.shm_name = shm_manager.get_local_shm(slot->bounce.shm_id).shm_name;
    }
    return info;
  };

  unsigned result =
      adapter->recv_async(rank, local_ptr, dst.bytes, dst.buffer_id,
                          bounce_provider);
  if (result == 0) {
    slot->state.store(TrackedRequest::SlotState::Releasing,
                      std::memory_order_release);
    cleanup_tracked_request(*slot);
    slot->state.store(TrackedRequest::SlotState::Free,
                      std::memory_order_release);
    return 0;
  }
  slot->adapter_request_id = result;
  slot->state.store(TrackedRequest::SlotState::InFlight,
                    std::memory_order_release);
  inflight_request_count_.fetch_add(1, std::memory_order_release);
  (void)enqueue_active_request(rid);
  progress_cv_.notify_all();
  return rid;
}

bool Communicator::poll_request_completion(unsigned id, bool blocking) {
  TrackedRequest* slot = resolve_request_slot(id);
  if (slot == nullptr) return true;
  auto state = slot->state.load(std::memory_order_acquire);
  if (state == TrackedRequest::SlotState::Completed ||
      state == TrackedRequest::SlotState::Failed) {
    return true;
  }
  if (state != TrackedRequest::SlotState::InFlight) return false;

  bool done = false;
  bool failed = false;
  unsigned adapter_id = slot->adapter_request_id;
  if (slot->kind == PeerTransportKind::Uccl) {
    if (!uccl_adapter_) return false;
    done = blocking ? uccl_adapter_->wait_completion(adapter_id)
                    : uccl_adapter_->poll_completion(adapter_id);
    failed = done && uccl_adapter_->request_failed(adapter_id);
  } else if (slot->kind == PeerTransportKind::Tcp) {
    done = blocking ? tcp_adapter_->wait_completion(adapter_id)
                    : tcp_adapter_->poll_completion(adapter_id);
    failed = done && tcp_adapter_->request_failed(adapter_id);
  } else if (slot->kind == PeerTransportKind::Ipc) {
    done = blocking ? ipc_adapter_->wait_completion(adapter_id)
                    : ipc_adapter_->poll_completion(adapter_id);
    failed = done && ipc_adapter_->request_failed(adapter_id);
  }

  if (!done) return false;

  if (!failed) {
    bool copy_done = complete_host_bounce_recv(*slot, blocking);
    if (!copy_done) return false;
  }
  slot->state.store(failed ? TrackedRequest::SlotState::Failed
                           : TrackedRequest::SlotState::Completed,
                    std::memory_order_release);
  inflight_request_count_.fetch_sub(1, std::memory_order_acq_rel);
  notify_request_completion();
  return true;
}

bool Communicator::wait_finish(std::vector<unsigned> const& reqs) {
  std::unordered_set<unsigned> remaining;
  bool any_failed = false;
  if (reqs.empty()) {
    for (uint32_t i = 0; i < kRequestSlotCount; ++i) {
      auto* slot = &request_slots_[i];
      auto state = slot->state.load(std::memory_order_acquire);
      if (state == TrackedRequest::SlotState::InFlight ||
          state == TrackedRequest::SlotState::Completed ||
          state == TrackedRequest::SlotState::Failed) {
        remaining.insert(slot->request_id);
      }
    }
  } else {
    remaining.insert(reqs.begin(), reqs.end());
  }

  uint64_t seen_seq = completion_seq_.load(std::memory_order_acquire);
  bool should_scan = true;

  while (!remaining.empty()) {
    if (should_scan) {
      std::vector<unsigned> finished;
      for (auto id : remaining) {
        if (id == 0) return false;
        if (!progress_started_.load(std::memory_order_acquire)) {
          (void)poll_request_completion(id, false);
        }
        TrackedRequest* slot = resolve_request_slot(id);
        if (slot == nullptr) {
          finished.push_back(id);
          continue;
        }
        auto state = slot->state.load(std::memory_order_acquire);
        if (state == TrackedRequest::SlotState::Completed ||
            state == TrackedRequest::SlotState::Failed) {
          TrackedRequest snapshot{};
          if (try_release_request_slot(id, &snapshot)) {
            any_failed =
                any_failed || (state == TrackedRequest::SlotState::Failed);
            cleanup_tracked_request(snapshot);
            finished.push_back(id);
          }
        }
      }

      for (auto id : finished) remaining.erase(id);
      should_scan = false;
    }

    if (remaining.empty()) return !any_failed;

    if (completion_event_fd_ >= 0) {
      pollfd pfd{};
      pfd.fd = completion_event_fd_;
      pfd.events = POLLIN;
      int rc = ::poll(&pfd, 1, -1);
      if (rc > 0 && (pfd.revents & POLLIN)) {
        uint64_t cnt = 0;
        // Drain accumulated wakeups.
        while (::read(completion_event_fd_, &cnt, sizeof(cnt)) ==
               static_cast<ssize_t>(sizeof(cnt))) {
        }
      }
      uint64_t now_seq = completion_seq_.load(std::memory_order_acquire);
      if (now_seq != seen_seq) {
        seen_seq = now_seq;
        should_scan = true;
      }
    } else {
      std::this_thread::yield();
      should_scan = true;
    }
  }

  return !any_failed;
}

bool Communicator::poll(unsigned const req) {
  if (req == 0) return false;
  auto* slot = resolve_request_slot(req);
  if (slot == nullptr) return true;
  auto state = slot->state.load(std::memory_order_acquire);
  if (state == TrackedRequest::SlotState::InFlight) {
    // Progress thread is the single driver in steady state to avoid
    // duplicate adapter polling contention on hot paths.
    if (!progress_started_.load(std::memory_order_acquire)) {
      if (!poll_request_completion(req, false)) return false;
      state = slot->state.load(std::memory_order_acquire);
    } else {
      return false;
    }
  }
  if (state == TrackedRequest::SlotState::Failed) {
    throw std::runtime_error("transport request failed");
  }
  return state == TrackedRequest::SlotState::Completed;
}

void Communicator::release(unsigned const req) {
  TrackedRequest* slot = resolve_request_slot(req);
  if (slot == nullptr) return;
  auto state = slot->state.load(std::memory_order_acquire);
  if (state != TrackedRequest::SlotState::Completed &&
      state != TrackedRequest::SlotState::Failed) {
    throw std::runtime_error("cannot release an in-flight transport request");
  }
  TrackedRequest snapshot{};
  if (try_release_request_slot(req, &snapshot)) {
    cleanup_tracked_request(snapshot);
  }
}

bool Communicator::wait_finish(unsigned const req) {
  return wait_finish(std::vector<unsigned>{req});
}

bool Communicator::reg_mr(uint32_t buffer_id, void* local_buf, size_t len,
                          bool publish) {
  if (buffer_id == 0 || local_buf == nullptr || len == 0) return false;
  MR mr = mr_manager_.create_local_mr(buffer_id, local_buf, len).mr;
  if (mr.address == 0 || mr.length == 0) return false;

  {
    std::lock_guard<std::mutex> lk(resource_mu_);
    local_buffer_to_mr_[buffer_id] = mr;
  }

  if (!publish || !exchanger_client_ || !exchanger_client_->valid()) {
    return true;
  }

  NamedMRInfos payload{};
  payload.entries.push_back(NamedMR{buffer_id, mr});
  return oob_put(*exchanger_client_, oob_namespace(),
                 mr_global_buffer_key(global_rank_, buffer_id), payload);
}

bool Communicator::dereg_mr(uint32_t buffer_id) {
  MR local_mr{};
  bool found = false;
  {
    std::lock_guard<std::mutex> lk(resource_mu_);
    auto it = local_buffer_to_mr_.find(buffer_id);
    if (it != local_buffer_to_mr_.end()) {
      local_mr = it->second;
      local_buffer_to_mr_.erase(it);
      found = true;
    }
  }
  uint32_t const registered_id = buffer_id;

  if (registered_id != 0 && uccl_adapter_ && uccl_adapter_->is_initialized()) {
    std::lock_guard<std::mutex> lk(uccl_reg_mu_);
    if (uccl_registered_mrs_.erase(registered_id) > 0) {
      uccl_adapter_->deregister_memory(registered_id);
    }
    uccl_direct_reg_failed_mrs_.erase(registered_id);
  }
  if (found) (void)mr_manager_.delete_mr(buffer_id);
  return true;
}

bool Communicator::wait_mr(int owner_rank, uint32_t buffer_id, int timeout_ms) {
  if (buffer_id == 0) return false;
  if (owner_rank == global_rank_) {
    std::lock_guard<std::mutex> lk(resource_mu_);
    return local_buffer_to_mr_.find(buffer_id) != local_buffer_to_mr_.end();
  }
  if (!exchanger_client_ || !exchanger_client_->valid()) return false;

  NamedMRInfos payload{};
  if (!oob_get(*exchanger_client_, oob_namespace(),
               mr_global_buffer_key(owner_rank, buffer_id), payload,
               timeout_ms)) {
    return false;
  }
  if (payload.entries.empty()) return false;

  bool found = false;
  MR mr{};
  for (auto const& entry : payload.entries) {
    if (entry.buffer_id != buffer_id || entry.mr.address == 0 ||
        entry.mr.length == 0) {
      continue;
    }
    mr = entry.mr;
    found = true;
    break;
  }
  if (!found) return false;

  MRItem item{};
  item.buffer_id = buffer_id;
  item.mr = mr;
  item.is_local = false;
  item.rank = owner_rank;
  item.valid = true;
  mr_manager_.register_remote_mr(owner_rank, item);

  {
    std::lock_guard<std::mutex> lk(resource_mu_);
    remote_buffer_to_mr_[owner_rank][buffer_id] = mr;
  }
  return true;
}

MR Communicator::get_mr(uint32_t buffer_id) const {
  std::lock_guard<std::mutex> lk(resource_mu_);
  auto it = local_buffer_to_mr_.find(buffer_id);
  if (it == local_buffer_to_mr_.end()) {
    throw std::runtime_error("local MR not found for buffer_id");
  }
  return it->second;
}

MR Communicator::get_mr(int owner_rank, uint32_t buffer_id) const {
  if (owner_rank == global_rank_) return get_mr(buffer_id);
  std::lock_guard<std::mutex> lk(resource_mu_);
  auto rank_it = remote_buffer_to_mr_.find(owner_rank);
  if (rank_it == remote_buffer_to_mr_.end()) {
    throw std::runtime_error("remote MR rank cache not found");
  }
  auto id_it = rank_it->second.find(buffer_id);
  if (id_it == rank_it->second.end()) {
    throw std::runtime_error("remote MR not found for buffer_id");
  }
  return id_it->second;
}

bool Communicator::reg_ipc(uint32_t buffer_id, void* local_buf, size_t len,
                           bool publish) {
  if (buffer_id == 0) return false;

  IPCItem local{};
  if (local_buf != nullptr && len != 0) {
    int original_device = -1;
    GPU_RT_CHECK(gpuGetDevice(&original_device));
    auto restore = UKernel::Transport::finally(
        [&]() { GPU_RT_CHECK(gpuSetDevice(original_device)); });
    GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
    local = ipc_manager_.create_local_ipc(local_buf, len, local_gpu_idx_);
    if (!local.valid) return false;
  } else {
    local.valid = false;
  }

  {
    std::lock_guard<std::mutex> lk(resource_mu_);
    local_buffer_to_ipc_[buffer_id] = local;
  }

  if (!publish || !exchanger_client_ || !exchanger_client_->valid()) {
    return true;
  }

  IpcBufferInfo info{};
  info.handle = local.handle;
  info.base_offset = static_cast<uint64_t>(local.base_offset);
  info.bytes = static_cast<uint64_t>(local.bytes);
  info.device_idx = local.device_idx;
  info.valid = local.valid;
  return oob_put(*exchanger_client_, oob_namespace(),
                 ipc_global_buffer_key(global_rank_, buffer_id), info);
}

bool Communicator::dereg_ipc(uint32_t buffer_id) {
  IPCItem local{};
  bool found = false;
  {
    std::lock_guard<std::mutex> lk(resource_mu_);
    auto it = local_buffer_to_ipc_.find(buffer_id);
    if (it != local_buffer_to_ipc_.end()) {
      local = it->second;
      local_buffer_to_ipc_.erase(it);
      found = true;
    }
  }
  if (found && local.base_addr != 0) {
    (void)ipc_manager_.delete_ipc(reinterpret_cast<void*>(local.base_addr));
  }
  return true;
}

bool Communicator::wait_ipc(int owner_rank, uint32_t buffer_id, int timeout_ms) {
  if (buffer_id == 0) return false;
  if (owner_rank == global_rank_) {
    std::lock_guard<std::mutex> lk(resource_mu_);
    return local_buffer_to_ipc_.find(buffer_id) != local_buffer_to_ipc_.end();
  }
  if (!exchanger_client_ || !exchanger_client_->valid()) return false;

  IpcBufferInfo info{};
  if (!oob_get(*exchanger_client_, oob_namespace(),
               ipc_global_buffer_key(owner_rank, buffer_id), info,
               timeout_ms)) {
    return false;
  }

  IPCItem state{};
  state.handle = info.handle;
  state.base_offset = static_cast<uintptr_t>(info.base_offset);
  state.bytes = static_cast<size_t>(info.bytes);
  state.device_idx = info.device_idx;
  state.valid = info.valid;
  return ipc_manager_.register_remote_ipc(owner_rank, buffer_id, state);
}

IPCItem Communicator::get_ipc(uint32_t buffer_id) {
  std::lock_guard<std::mutex> lk(resource_mu_);
  auto it = local_buffer_to_ipc_.find(buffer_id);
  if (it == local_buffer_to_ipc_.end()) {
    throw std::runtime_error("local IPC not found for buffer_id");
  }
  return it->second;
}

IPCItem Communicator::get_ipc(int owner_rank, uint32_t buffer_id) {
  if (owner_rank == global_rank_) return get_ipc(buffer_id);
  IPCItem item = ipc_manager_.get_ipc(owner_rank, buffer_id);
  if (!item.valid) {
    throw std::runtime_error("remote IPC not found for buffer_id");
  }
  if (item.direct_ptr == nullptr) {
    int original_device = -1;
    GPU_RT_CHECK(gpuGetDevice(&original_device));
    auto restore = UKernel::Transport::finally(
        [&]() { GPU_RT_CHECK(gpuSetDevice(original_device)); });
    GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

    gpuError_t open_err = gpuIpcOpenMemHandle(&item.direct_ptr, item.handle,
                                              gpuIpcMemLazyEnablePeerAccess);
    if (open_err != gpuSuccess) {
      throw std::runtime_error("failed to open remote IPC mem handle");
    }
    ipc_manager_.register_remote_ipc(owner_rank, buffer_id, item);
  }
  return item;
}

bool Communicator::try_resolve_remote_ipc_pointer(int remote_rank,
                                                  uint32_t remote_buffer_id,
                                                  size_t offset, size_t bytes,
                                                  void** out_ptr,
                                                  int* out_device_idx) {
  if (out_ptr == nullptr || remote_buffer_id == 0) return false;
  *out_ptr = nullptr;

  IPCItem item{};
  if (remote_rank == global_rank_) {
    std::lock_guard<std::mutex> lk(resource_mu_);
    auto it = local_buffer_to_ipc_.find(remote_buffer_id);
    if (it != local_buffer_to_ipc_.end()) {
      item = it->second;
    }
  } else {
    item = ipc_manager_.get_ipc(remote_rank, remote_buffer_id);
  }
  if (!item.valid) return false;

  if (remote_rank == global_rank_) {
    if (item.base_addr == 0) return false;
    if (offset > item.bytes || bytes > item.bytes - offset) return false;
    uintptr_t const resolved = item.base_addr + item.base_offset + offset;
    *out_ptr = reinterpret_cast<void*>(resolved);
    if (out_device_idx != nullptr) {
      *out_device_idx = item.device_idx;
    }
    return true;
  }

  if (item.direct_ptr == nullptr) {
    int original_device = -1;
    if (gpuGetDevice(&original_device) != gpuSuccess) return false;
    if (gpuSetDevice(local_gpu_idx_) != gpuSuccess) return false;
    gpuError_t open_err = gpuIpcOpenMemHandle(&item.direct_ptr, item.handle,
                                              gpuIpcMemLazyEnablePeerAccess);
    gpuError_t restore_err = gpuSetDevice(original_device);
    if (restore_err != gpuSuccess) return false;
    if (open_err != gpuSuccess || item.direct_ptr == nullptr) return false;
    if (!ipc_manager_.register_remote_ipc(remote_rank, remote_buffer_id, item)) {
      return false;
    }
  }

  if (offset > item.bytes || bytes > item.bytes - offset) return false;
  uintptr_t const base = reinterpret_cast<uintptr_t>(item.direct_ptr);
  uintptr_t const resolved = base + item.base_offset + offset;
  *out_ptr = reinterpret_cast<void*>(resolved);
  if (out_device_idx != nullptr) {
    *out_device_idx = item.device_idx;
  }
  return true;
}

void* Communicator::get_or_open_bounce_shm(std::string const& shm_name) {
  return require_shm_manager("Communicator::get_or_open_bounce_shm")
      .open_remote_shm(shm_name)
      .ptr;
}

void Communicator::clear_bounce_shm_cache() {
  require_shm_manager("Communicator::clear_bounce_shm_cache")
      .clear_remote_shm_cache();
}

std::shared_ptr<void> Communicator::register_completion_notifier(
    std::function<void(unsigned, std::chrono::steady_clock::time_point)> cb) {
  auto target = std::make_shared<NotifyTarget>();
  target->emit = std::move(cb);

  {
    std::lock_guard<std::mutex> lk(progress_mu_);
    notify_targets_.push_back(target);
  }

  // The progress loop is already running; registering a notifier only changes
  // which callbacks should observe completed requests.
  progress_cv_.notify_all();

  return std::static_pointer_cast<void>(target);
}

void Communicator::progress_loop() {
  constexpr size_t kProgressBatchSize = 128;
  std::array<unsigned, kProgressBatchSize> batch{};

  while (progress_running_.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lk(progress_mu_);
      progress_cv_.wait(lk, [&] {
        if (!progress_running_.load()) return true;
        return inflight_request_count_.load(std::memory_order_acquire) > 0;
      });
    }

    if (!progress_running_.load()) break;

    bool progress = false;
    bool dequeued_any = false;
    std::vector<std::shared_ptr<NotifyTarget>> targets_snapshot;
    {
      std::lock_guard<std::mutex> nlk(progress_mu_);
      notify_targets_.erase(
          std::remove_if(notify_targets_.begin(), notify_targets_.end(),
                         [](std::weak_ptr<NotifyTarget> const& target) {
                           return target.expired();
                         }),
          notify_targets_.end());
      targets_snapshot.reserve(notify_targets_.size());
      for (auto const& weak_target : notify_targets_) {
        if (auto target = weak_target.lock()) {
          targets_snapshot.push_back(std::move(target));
        }
      }
    }

    size_t batch_count = 0;
    unsigned id = 0;
    while (batch_count < batch.size() && dequeue_active_request(&id)) {
      dequeued_any = true;
      batch[batch_count++] = id;
    }

    auto now = std::chrono::steady_clock::now();
    for (size_t i = 0; i < batch_count; ++i) {
      id = batch[i];
      TrackedRequest* slot = resolve_request_slot(id);
      if (slot == nullptr) continue;
      auto state = slot->state.load(std::memory_order_acquire);
      if (state != TrackedRequest::SlotState::InFlight) continue;
      if (!poll_request_completion(id, false)) {
        (void)enqueue_active_request(id);
        continue;
      }

      bool should_emit = false;
      state = slot->state.load(std::memory_order_acquire);
      if ((state == TrackedRequest::SlotState::Completed ||
           state == TrackedRequest::SlotState::Failed) &&
          !slot->notified) {
        slot->notified = true;
        should_emit = true;
      }
      if (!should_emit) continue;
      for (auto& tgt : targets_snapshot) {
        if (!tgt) continue;
        tgt->emit(id, now);
      }
      progress = true;
    }

    if (!progress && !dequeued_any) {
      if (inflight_request_count_.load(std::memory_order_acquire) > 0) {
        for (uint32_t i = 0; i < kRequestSlotCount; ++i) {
          TrackedRequest* slot = &request_slots_[i];
          auto state = slot->state.load(std::memory_order_acquire);
          if (state != TrackedRequest::SlotState::InFlight) continue;
          (void)enqueue_active_request(slot->request_id);
        }
      } else {
        std::this_thread::yield();
      }
    }
  }
}

}  // namespace Transport
}  // namespace UKernel
