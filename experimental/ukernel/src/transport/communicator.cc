#include "communicator.h"
#include "ipc_channel.h"
#include "transport_engine.h"
#include "utils.h"
#include <arpa/inet.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <netinet/in.h>
#include <stdexcept>
#include <sys/socket.h>
#include <unordered_set>
#include <unistd.h>

namespace UKernel {
namespace Transport {

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

Communicator::Communicator(int gpu_id, int rank, int world_size,
                           std::shared_ptr<CommunicatorConfig> config)
    : local_gpu_idx_(gpu_id),
      global_rank_(rank),
      world_size_(world_size),
      peers_(world_size),
      config_(config),
      uccl_engine_(std::make_unique<UcclTransportEngine>(gpu_id, world_size)) {
  uds_ = std::make_shared<UdsExchanger>(global_rank_);

  // Initialize communicator meta
  CommunicatorMeta local{};
  local.host_id = generate_host_id();
  local.is_ready = true;
  local.ip = get_local_ip();
  set_peer_meta(global_rank_, local);

  // Initialize Redis client
#ifdef USE_REDIS_OOB
  exchanger_client_ = std::make_shared<RedisExchanger>(config_->exchanger_ip,
                                                       config_->exchanger_port);
#else
  bool is_server = (global_rank_ == 0);
  if (!is_server && config_->exchanger_ip == "0.0.0.0")
    config_->exchanger_ip = "127.0.0.1";
  std::cout << "[INFO] Using socket-based exchanger as "
            << (is_server ? "server" : "client") << " " << config_->exchanger_ip
            << std::endl;
  exchanger_client_ = std::make_shared<SockExchanger>(
      (global_rank_ == 0), config_->exchanger_ip, config_->exchanger_port);
#endif
  if (!exchanger_client_->valid()) {
    fprintf(stderr, "[ERROR] Failed to connect to Exchanger\n");
    return;
  }

  // Exchange communicator meta
  std::string meta_key = "meta:" + std::to_string(global_rank_);
  if (!exchanger_client_->publish(meta_key, local)) {
    fprintf(stderr, "[ERROR] Failed to publish local CommunicatorMeta \n");
  }

  // Get all others meta
  CommunicatorMeta remote{};
  for (int i = 0; i < world_size_; i++) {
    if (i == global_rank_) continue;
    std::string key = "meta:" + std::to_string(i);
    if (exchanger_client_->wait_and_fetch(key, remote, -1)) {
      set_peer_meta(i, remote);
    } else {
      fprintf(stderr, "[WARN] Timeout waiting for remote CommunicatorMeta \n");
    }
  }
  std::cout << "[INFO] Communicator " << global_rank_
            << " initialized: peer meta exchange success" << std::endl;
}

Communicator::~Communicator() {
  // Release endpoints
  {
    std::lock_guard<std::mutex> lk(peer_mu_);
    for (auto& peer : peers_) {
      peer.ipc_channel.reset();
    }
  }

  // Deregister local memory regions
  for (auto* p : memory_registry_.local_buffers()) {
    dereg_mr(p);
  }

  // Clear IPC caches
  memory_registry_.clear_remote_ipc_cache();

  // Stop notifier
  if (notifier_started_.load()) {
    notifier_running_.store(false);
    notifier_cv_.notify_all();
    if (notifier_thread_.joinable()) {
      notifier_thread_.join();
    }
  }

  std::cout << "[INFO] Communicator " << global_rank_ << " resources released"
            << std::endl;
}

bool Communicator::connect_to(int rank) {
  if (!check_ready()) {
    std::cerr << "[WARN] Communicator " << global_rank_
              << " not ready, cannot connect to rank " << rank << std::endl;
    return false;
  }

  if (rank == global_rank_) {
    return true;
  }

  if (rank < 0 || rank >= world_size_) {
    std::cerr << "[ERROR] Communicator " << global_rank_ << " invalid rank "
              << rank << ", world_size=" << world_size_ << std::endl;
    return false;
  }

  CommunicatorMeta meta{};
  CommunicatorMeta local_meta{};
  if (!try_get_peer_meta(rank, meta) || !try_get_peer_meta(global_rank_, local_meta)) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " CommunicatorMeta not found for rank " << rank << std::endl;
    return false;
  }

  auto peer_kind = resolve_peer_transport_kind(*config_, local_meta, meta);
  if (peer_kind == PeerTransportKind::Uccl && has_peer_send_path(rank)) {
    return true;
  }
  if (peer_kind == PeerTransportKind::Ipc && has_peer_send_path(rank)) {
    return true;
  }

  if (peer_kind == PeerTransportKind::Uccl) {
    bool ret = uccl_engine_->connect_to_peer(global_rank_, rank, config_,
                                             exchanger_client_, local_meta,
                                             meta);
    if (ret) {
      cache_peer_session(rank, PeerTransportKind::Uccl, nullptr, true, false);
      register_existing_local_mrs_with_uccl();
      std::cout << "[INFO] Communicator " << global_rank_
                << " UCCL connect_to succeeded to rank " << rank << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " UCCL connect_to failed to rank " << rank << std::endl;
    }
    return ret;
  }

  if (peer_kind == PeerTransportKind::Ipc) {
    auto ipc_channel = std::make_shared<IpcChannel>(this);
    bool ret = ipc_channel->connect_to(rank);
    if (ret) {
      std::cout << "[INFO] Communicator " << global_rank_
                << " IPC connect_to succeeded to rank " << rank << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " IPC connect_to failed to rank " << rank << std::endl;
      return false;
    }
    cache_peer_session(rank, PeerTransportKind::Ipc, ipc_channel, true, true);
    return true;
  }
  return false;
}

bool Communicator::accept_from(int rank) {
  if (!check_ready()) return false;
  if (rank == global_rank_) return true;

  CommunicatorMeta meta{};
  CommunicatorMeta local_meta{};
  if (!try_get_peer_meta(rank, meta) || !try_get_peer_meta(global_rank_, local_meta)) {
    return false;
  }

  auto peer_kind = resolve_peer_transport_kind(*config_, local_meta, meta);
  if (peer_kind == PeerTransportKind::Uccl && has_peer_recv_path(rank)) {
    return true;
  }
  if (peer_kind == PeerTransportKind::Ipc && has_peer_recv_path(rank)) {
    return true;
  }

  if (peer_kind == PeerTransportKind::Uccl) {
    bool ret = uccl_engine_->accept_from_peer(global_rank_, rank, config_,
                                              exchanger_client_, local_meta,
                                              meta);
    if (ret) {
      cache_peer_session(rank, PeerTransportKind::Uccl, nullptr, false, true);
      register_existing_local_mrs_with_uccl();
      std::cout << "[INFO] Communicator " << global_rank_
                << " UCCL accept_from succeeded from rank " << rank
                << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " UCCL accept_from failed from rank " << rank << std::endl;
    }
    return ret;
  }

  if (peer_kind == PeerTransportKind::Ipc) {
    auto ipc_channel = std::make_shared<IpcChannel>(this);
    bool ret = ipc_channel->accept_from(rank);
    if (ret) {
      std::cout << "[INFO] Communicator " << global_rank_
                << " IPC accept_from succeeded from rank " << rank << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " IPC accept_from failed from rank " << rank << std::endl;
    }
    if (!ret) return false;
    cache_peer_session(rank, PeerTransportKind::Ipc, ipc_channel, true, true);
    return true;
  }
  return false;
}

std::shared_ptr<IpcChannel> Communicator::get_ipc_channel_by_rank(int rank) {
  std::lock_guard<std::mutex> lock(peer_mu_);
  if (rank < 0 || rank >= world_size_) return nullptr;
  return peers_[rank].ipc_channel;
}

void Communicator::cache_peer_session(int rank, PeerTransportKind kind,
                                      std::shared_ptr<IpcChannel> ipc_channel,
                                      bool mark_send_ready,
                                      bool mark_recv_ready) {
  std::lock_guard<std::mutex> lk(peer_mu_);
  auto& session = peers_[rank];
  session.kind = kind;
  if (ipc_channel) session.ipc_channel = std::move(ipc_channel);
  session.send_ready = session.send_ready || mark_send_ready;
  session.recv_ready = session.recv_ready || mark_recv_ready;
}

bool Communicator::has_peer_send_path(int rank) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  return rank >= 0 && rank < world_size_ && peers_[rank].send_ready;
}

bool Communicator::has_peer_recv_path(int rank) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  return rank >= 0 && rank < world_size_ && peers_[rank].recv_ready;
}

PeerTransportKind Communicator::get_peer_transport_kind(int rank) const {
  std::lock_guard<std::mutex> lk(peer_mu_);
  if (rank < 0 || rank >= world_size_) {
    throw std::runtime_error("transport peer session is not established");
  }
  return peers_[rank].kind;
}

void Communicator::register_existing_local_mrs_with_uccl() {
  if (!uccl_engine_->is_initialized()) return;
  for (auto* p : memory_registry_.local_buffers()) {
    MR mr = memory_registry_.get_local_mr(p);
    if (!uccl_engine_->register_memory(mr.id, p, mr.length)) {
      throw std::runtime_error("UCCL register_memory failed for existing MR");
    }
  }
}

unsigned Communicator::isend(int rank, void* ptr, size_t offset, size_t len,
                             uint32_t local_mr_id, uint32_t remote_mr_id,
                             bool on_gpu) {
  if (!has_peer_send_path(rank)) {
    throw std::runtime_error("transport send path is not established");
  }
  auto peer_kind = get_peer_transport_kind(rank);
  unsigned rid = next_request_id_.fetch_add(1, std::memory_order_relaxed);

  if (peer_kind == PeerTransportKind::Uccl) {
    void* actual_ptr = static_cast<char*>(ptr) + offset;
    int ret = uccl_engine_->send_async(rank, actual_ptr, len, local_mr_id,
                                       remote_mr_id, rid);
    if (ret != 0) return 0;
    {
      std::lock_guard<std::mutex> lk(req_mu_);
      requests_map_[rid] = TrackedRequest{rank, PeerTransportKind::Uccl,
                                          nullptr, false, false, false};
    }
    notifier_cv_.notify_all();
    return rid;
  }

  auto ipc_channel = get_ipc_channel_by_rank(rank);
  if (!ipc_channel) return 0;

  auto req = std::make_shared<Request>(rid, ptr, offset, len, local_mr_id,
                                       remote_mr_id, on_gpu, RequestType::SEND);

  {
    std::lock_guard<std::mutex> lk(req_mu_);
    requests_map_[rid] = TrackedRequest{rank, PeerTransportKind::Ipc, req,
                                        false, false, false};
  }

  if (!ipc_channel->send_async(rank, req)) {
    std::lock_guard<std::mutex> lk(req_mu_);
    requests_map_.erase(rid);
    return 0;
  }
  notifier_cv_.notify_all();

  return rid;
}

unsigned Communicator::irecv(int rank, void* ptr, size_t offset, size_t len,
                             bool on_gpu) {
  if (!has_peer_recv_path(rank)) {
    throw std::runtime_error("transport recv path is not established");
  }
  auto peer_kind = get_peer_transport_kind(rank);
  unsigned rid = next_request_id_.fetch_add(1, std::memory_order_relaxed);

  if (peer_kind == PeerTransportKind::Uccl) {
    void* actual_ptr = static_cast<char*>(ptr) + offset;
    auto local_mr = get_local_mr(actual_ptr);
    int ret = uccl_engine_->recv_async(rank, actual_ptr, len, local_mr.id, rid);
    if (ret != 0) return 0;
    {
      std::lock_guard<std::mutex> lk(req_mu_);
      requests_map_[rid] = TrackedRequest{rank, PeerTransportKind::Uccl,
                                          nullptr, false, false, false};
    }
    notifier_cv_.notify_all();
    return rid;
  }

  auto ipc_channel = get_ipc_channel_by_rank(rank);
  if (!ipc_channel) return 0;

  auto local_mr = get_local_mr(static_cast<char*>(ptr) + offset);

  auto req = std::make_shared<Request>(rid, ptr, offset, len, -1, -1, on_gpu,
                                       RequestType::RECV);

  {
    std::lock_guard<std::mutex> lk(req_mu_);
    requests_map_[rid] = TrackedRequest{rank, PeerTransportKind::Ipc, req,
                                        false, false, false};
  }

  if (!ipc_channel->recv_async(rank, req)) {
    std::lock_guard<std::mutex> lk(req_mu_);
    requests_map_.erase(rid);
    return 0;
  }
  notifier_cv_.notify_all();

  return rid;
}

bool Communicator::poll_request_completion(unsigned id, bool blocking) {
  TrackedRequest snapshot;
  {
    std::lock_guard<std::mutex> lk(req_mu_);
    auto it = requests_map_.find(id);
    if (it == requests_map_.end()) return true;
    if (it->second.completed) return true;
    snapshot = it->second;
  }

  bool done = false;
  bool failed = false;
  if (snapshot.kind == PeerTransportKind::Uccl) {
    done = blocking ? uccl_engine_->wait_completion(id)
                    : uccl_engine_->poll_completion(id);
    if (blocking && !done) failed = true;
  } else if (snapshot.ipc_request) {
    while (true) {
      done = snapshot.ipc_request->finished.load(std::memory_order_acquire);
      if (done || !blocking) break;
      std::this_thread::yield();
    }
    failed = done &&
             snapshot.ipc_request->failed.load(std::memory_order_acquire);
  }

  if (!done) return false;

  std::lock_guard<std::mutex> lk(req_mu_);
  auto it = requests_map_.find(id);
  if (it == requests_map_.end()) return true;
  it->second.completed = true;
  it->second.failed = failed;
  return true;
}

bool Communicator::wait_finish(std::vector<unsigned> const& reqs) {
  std::unordered_set<unsigned> remaining;
  if (reqs.empty()) {
    std::lock_guard<std::mutex> lk(req_mu_);
    for (auto const& [id, _] : requests_map_) {
      remaining.insert(id);
    }
  } else {
    remaining.insert(reqs.begin(), reqs.end());
  }

  while (true) {
    std::vector<unsigned> finished;
    for (auto id : remaining) {
      if (id == 0) return false;
      if (!poll_request_completion(id, true)) {
        return false;
      }
      std::lock_guard<std::mutex> lk(req_mu_);
      auto it = requests_map_.find(id);
      if (it != requests_map_.end() && it->second.failed) {
        requests_map_.erase(it);
        return false;
      }
      if (it != requests_map_.end()) requests_map_.erase(it);
      finished.push_back(id);
    }
    for (auto id : finished) remaining.erase(id);
    if (remaining.empty()) return true;
  }
}

bool Communicator::poll(unsigned const req) {
  if (req == 0) return false;
  if (!poll_request_completion(req, false)) return false;

  std::lock_guard<std::mutex> lk(req_mu_);
  auto it = requests_map_.find(req);
  if (it == requests_map_.end()) return true;
  if (it->second.failed) {
    throw std::runtime_error("transport request failed");
  }
  return it->second.completed;
}

void Communicator::release(unsigned const req) {
  std::lock_guard<std::mutex> lk(req_mu_);
  auto it = requests_map_.find(req);
  if (it == requests_map_.end()) return;
  if (!it->second.completed) {
    throw std::runtime_error("cannot release an in-flight transport request");
  }
  requests_map_.erase(it);
}

bool Communicator::wait_finish(unsigned const req) {
  return wait_finish(std::vector<unsigned>{req});
}

void Communicator::set_peer_meta(int rank, CommunicatorMeta const& meta) {
  if (rank < 0 || rank >= world_size_) {
    throw std::out_of_range("peer rank is out of range");
  }
  std::lock_guard<std::mutex> lock(peer_mu_);
  peers_[rank].meta = meta;
  peers_[rank].has_meta = true;
}

bool Communicator::try_get_peer_meta(int rank, CommunicatorMeta& out) const {
  if (rank < 0 || rank >= world_size_) return false;
  std::lock_guard<std::mutex> lock(peer_mu_);
  if (!peers_[rank].has_meta) return false;
  out = peers_[rank].meta;
  return true;
}

bool Communicator::check_ready() const {
  std::lock_guard<std::mutex> lock(peer_mu_);
  for (int i = 0; i < world_size_; i++) {
    if (!peers_[i].has_meta) {
      std::cerr << "[WARN] Communicator " << global_rank_
                << " check_ready: missing CommunicatorMeta for rank " << i
                << std::endl;
      return false;
    }
    if (!peers_[i].meta.is_ready) {
      std::cerr << "[WARN] Communicator " << global_rank_
                << " check_ready: CommunicatorMeta for rank " << i
                << " is not ready" << std::endl;
      return false;
    }
  }

  std::cerr << "[INFO] Communicator " << global_rank_ << " is ready"
            << std::endl;
  return true;
}

MR Communicator::reg_mr(void* local_buf, size_t len) {
  MR info = memory_registry_.track_local_buffer(local_buf, len);

  if (uccl_engine_->is_initialized()) {
    if (!uccl_engine_->register_memory(info.id, local_buf, len)) {
      throw std::runtime_error("UCCL register_memory failed");
    }
  }

  return info;
}

bool Communicator::dereg_mr(void* local_buf) {
  auto released = memory_registry_.release_local_buffer(local_buf);
  if (uccl_engine_->is_initialized() && released.has_local_mr_id) {
    uccl_engine_->deregister_memory(released.local_mr_id);
  }
  return true;
}

bool Communicator::notify_mr(int remote_rank, MR& mr) {
  if (!exchanger_client_ || !exchanger_client_->valid()) return false;

  std::string key =
      "mr:" + std::to_string(global_rank_) + "->" + std::to_string(remote_rank);

  MRInfos wrapper;
  exchanger_client_->fetch(key, wrapper);
  bool replaced = false;
  for (auto& existing : wrapper.mrs) {
    if (existing.id == mr.id) {
      existing = mr;
      replaced = true;
      break;
    }
  }
  if (!replaced) {
    wrapper.mrs.push_back(mr);
  }

  std::cout << "[notify MR to rank " << remote_rank << "] addr=" << mr.address
            << " length=" << mr.length << " key=" << mr.key << std::endl;

  return exchanger_client_->publish(key, wrapper);
}

bool Communicator::wait_mr_notify(int remote_rank, MR& mr) {
  if (!exchanger_client_ || !exchanger_client_->valid()) {
    throw std::runtime_error("Exchanger client is not valid");
  }

  std::string key =
      "mr:" + std::to_string(remote_rank) + "->" + std::to_string(global_rank_);

  if (memory_registry_.take_pending_remote_mr(remote_rank, mr)) return true;

  while (true) {
    MRInfos wrapper;
    bool fetched = exchanger_client_->fetch(key, wrapper);
    if (fetched && !wrapper.mrs.empty()) {
      memory_registry_.cache_remote_mrs(remote_rank, wrapper.mrs);
      if (memory_registry_.take_pending_remote_mr(remote_rank, mr)) {
        std::cout << "[recv MR from rank " << remote_rank
                  << "] addr=" << mr.address << " length=" << mr.length
                  << " key=" << mr.key << std::endl;
        return true;
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

MR Communicator::get_local_mr(void* local_buf) {
  return memory_registry_.get_local_mr(local_buf);
}

MR Communicator::get_local_mr(uint32_t mr_id) {
  return memory_registry_.get_local_mr(mr_id);
}

MR Communicator::get_remote_mr(int remote_rank, uint32_t mr_id) {
  return memory_registry_.get_remote_mr(remote_rank, mr_id);
}

// Register a remote IPC cache for a given rank and buffer
bool Communicator::register_remote_ipc_cache(int remote_rank,
                                             gpuIpcMemHandle_t handle,
                                             IpcCache const& cache) {
  return memory_registry_.register_remote_ipc_cache(remote_rank, handle, cache);
}

// Get the remote IPC cache of a buffer from a given rank
IpcCache Communicator::get_remote_ipc_cache(int remote_rank,
                                            gpuIpcMemHandle_t handle) {
  return memory_registry_.get_remote_ipc_cache(remote_rank, handle);
}

std::shared_ptr<void> Communicator::register_completion_notifier(
    std::function<void(unsigned, std::chrono::steady_clock::time_point)> cb) {
  auto target = std::make_shared<NotifyTarget>();
  target->emit = std::move(cb);

  {
    std::lock_guard<std::mutex> lk(notifier_mu_);
    notify_targets_.push_back(target);
  }

  bool expected = false;
  if (notifier_started_.compare_exchange_strong(expected, true)) {
    notifier_running_.store(true);
    notifier_thread_ =
        std::thread(&Communicator::completion_notifier_loop, this);
  }

  notifier_cv_.notify_all();

  return std::shared_ptr<void>(nullptr, [this, target](void*) {
    std::lock_guard<std::mutex> lk(notifier_mu_);
    notify_targets_.erase(
        std::remove(notify_targets_.begin(), notify_targets_.end(), target),
        notify_targets_.end());
  });
}

void Communicator::completion_notifier_loop() {
  while (notifier_running_.load(std::memory_order_acquire)) {
    // no req, sleep
    {
      std::unique_lock<std::mutex> lk(notifier_mu_);
      notifier_cv_.wait(lk, [&] {
        if (!notifier_running_.load()) return true;

        std::lock_guard<std::mutex> rlk(req_mu_);
        return !requests_map_.empty();
      });
    }

    if (!notifier_running_.load()) break;

    bool progress = false;
    std::vector<unsigned> ids;
    {
      std::lock_guard<std::mutex> rlk(req_mu_);
      ids.reserve(requests_map_.size());
      for (auto const& [id, _] : requests_map_) ids.push_back(id);
    }

    auto now = std::chrono::steady_clock::now();
    for (auto id : ids) {
      if (!poll_request_completion(id, false)) continue;

      bool should_emit = false;
      {
        std::lock_guard<std::mutex> rlk(req_mu_);
        auto it = requests_map_.find(id);
        if (it != requests_map_.end() && !it->second.notified) {
          it->second.notified = true;
          should_emit = true;
        }
      }
      if (!should_emit) continue;

      std::vector<std::shared_ptr<NotifyTarget>> targets;
      {
        std::lock_guard<std::mutex> nlk(notifier_mu_);
        targets = notify_targets_;
      }
      for (auto& tgt : targets) {
        if (!tgt) continue;
        tgt->emit(id, now);
      }
      progress = true;
    }

    if (!progress) {
      std::this_thread::yield();
    }
  }
}

}  // namespace Transport
}  // namespace UKernel
