#include "cq_poller.h"
#include "transport.h"
#include "util/util.h"
#include "utils.h"
#include <unordered_set>

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
    : local_rank_(gpu_id),
      global_rank_(rank),
      world_size_(world_size),
      config_(config) {
  // Find best NIC for current gpu
  auto [nic_id, nic_name] = find_best_rdma_for_gpu(gpu_id);
  std::cout << "[INFO] Using RDMA NIC " << nic_name << std::endl;
  if (nic_id != -1) {  // Support RDMA
    struct ibv_device** dev_list = ibv_get_device_list(nullptr);
    if (!dev_list) {
      std::cerr << "Failed to get IB devices" << std::endl;
      std::abort();
    }
    struct ibv_device* nic_dev = nullptr;
    for (int i = 0; dev_list[i] != nullptr; ++i) {
      if (nic_name == ibv_get_device_name(dev_list[i])) {
        nic_dev = dev_list[i];
        break;
      }
    }
    if (!nic_dev) {
      std::cerr << "Could not match RDMA NIC: " << nic_name << std::endl;
      ibv_free_device_list(dev_list);
      std::abort();
    }
    nic_ibv_ctx_ = ibv_open_device(nic_dev);
    if (!nic_ibv_ctx_) {
      std::cerr << "Failed to open ibv context for NIC: " << nic_name
                << std::endl;
      ibv_free_device_list(dev_list);
      std::abort();
    }
    ibv_free_device_list(dev_list);

    std::cout << "[INFO] Communicator " << global_rank_ << " initialized: GPU "
              << gpu_id << " map to RDMA NIC " << nic_name << std::endl;

    support_rdma = true;
    // Init RAMD resource
    // Create pd
    pd_ = ibv_alloc_pd(nic_ibv_ctx_);
    if (!pd_) {
      perror("Failed to allocate PD");
      std::abort();
    }
    // Create cq
    for (int i = 0; i < config_->cq_poller_threads; i++) {
      ibv_cq* cq =
          ibv_create_cq(nic_ibv_ctx_, config_->cq_depth, nullptr, nullptr, 0);
      if (!cq) {
        perror("ibv_create_cq failed");
        std::abort();
      }
      cq_list_.push_back(cq);
    }
    // Query port
    struct ibv_port_attr port_attr;
    if (ibv_query_port(nic_ibv_ctx_, 1, &port_attr)) {
      perror("Failed to query port");
      exit(1);
    }
    lid = port_attr.lid;
    active_mtu = port_attr.active_mtu;
    if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
      // RoCE (Ethernet), fill the GID
      union ibv_gid local_gid;
      int gid_index = 1;
      if (ibv_query_gid(nic_ibv_ctx_, 1, gid_index, &local_gid)) {
        perror("Failed to query GID");
        exit(1);
      }
      memcpy(gid, &local_gid, 16);
      support_rdma_roce = true;
    } else {
      support_rdma_roce = false;
    }

    // Start cq poller
    for (int i = 0; i < config_->cq_poller_threads; i++) {
      CQPoller* cq_poller = new CQPoller(this, cq_list_[i]);
      cq_poller->start();
      cq_poller_list_.push_back(cq_poller);
    }

    pending_req_id_to_deal_ =
        uccl::create_ring(sizeof(unsigned), 16);  // change num later

  } else {  // Does not support RDMA
    // If we can't find any rdma nic, we still can do ipc comm on local
    // host.
    support_rdma = false;
  }

  uds_ = std::make_shared<UdsExchanger>(global_rank_);

  // Initialize communicator meta
  CommunicatorMeta local{};
  local.host_id = generate_host_id();
  local.is_ready = true;
  local.ip = get_local_ip();
  set_communicator_meta_with_rank(global_rank_, local);

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
      set_communicator_meta_with_rank(i, remote);
    } else {
      fprintf(stderr, "[WARN] Timeout waiting for remote CommunicatorMeta \n");
    }
  }
  std::cout << "[INFO] Communicator " << global_rank_
            << " initialized: rank_to_comm_meta_ success" << std::endl;
}

Communicator::~Communicator() {
  // Stop cq poller first
  if (!cq_poller_list_.empty()) {
    for (auto& cq_poller : cq_poller_list_) {
      if (cq_poller) {
        cq_poller->stop();
      }
    }
    cq_poller_list_.clear();
  }

  // Release endpoints
  {
    std::lock_guard<std::mutex> lk(ep_mu_);
    for (auto& [rank, ep] : rank_to_endpoint_) {
      if (ep) {
        ep.reset();
      }
    }
    rank_to_endpoint_.clear();
  }

  // Deregister local memory regions
  std::vector<void*> bufs;
  {
    std::lock_guard<std::mutex> lk(local_mr_mu_);
    bufs.reserve(ptr_to_local_ibv_mr_.size());
    for (auto& kv : ptr_to_local_ibv_mr_) {
      bufs.push_back(kv.first);  // ptr
    }
  }
  for (auto* p : bufs) {
    dereg_mr(p);
  }
  {
    std::lock_guard<std::mutex> lk(local_mr_mu_);
    mr_id_to_local_mr_.clear();
  }

  // Destory all CQs
  if (!cq_list_.empty()) {
    for (auto& cq : cq_list_) {
      if (cq) {
        if (ibv_destroy_cq(cq)) {
          std::cerr << "[WARN] Communicator " << global_rank_
                    << " Failed to destroy CQ" << std::endl;
        }
      }
    }
    cq_list_.clear();
  }

  // Deallocate PD
  if (pd_) {
    if (ibv_dealloc_pd(pd_)) {
      std::cerr << "[WARN] Communicator " << global_rank_
                << " Failed to deallocate PD" << std::endl;
    }
    pd_ = nullptr;
  }

  // Close device
  if (nic_ibv_ctx_) {
    if (ibv_close_device(nic_ibv_ctx_)) {
      std::cerr << "[WARN] Communicator " << global_rank_
                << " Failed to close IBV device context" << std::endl;
    }
    nic_ibv_ctx_ = nullptr;
  }

  // Clear remote MRs
  {
    std::lock_guard<std::mutex> lk(remote_mr_mu_);
    rank_mr_id_to_remote_mr_.clear();
  }

  // Clear IPC caches
  {
    std::lock_guard<std::mutex> lk(remote_ipc_cache_mu_);
    rank_handle_to_ipc_cache_.clear();
  }

  // Free pending_req_id_to_deal_ buffer
  if (pending_req_id_to_deal_) {
    free(pending_req_id_to_deal_);
    pending_req_id_to_deal_ = nullptr;
  }

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

  auto [existing_ep, ok] = get_endpoint_by_rank(rank);
  if (ok && existing_ep) return true;  // already

  if (rank == global_rank_) {
    return true;
  }

  if (rank < 0 || rank >= world_size_) {
    std::cerr << "[ERROR] Communicator " << global_rank_ << " invalid rank "
              << rank << ", world_size=" << world_size_ << std::endl;
    return false;
  }

  auto meta = get_communicator_meta_by_rank(rank);
  auto local_meta = get_communicator_meta_by_rank(global_rank_);

  if (!meta) {
    std::cerr << "[ERROR] Communicator " << global_rank_
              << " CommunicatorMeta not found for rank " << rank << std::endl;
    return false;
  }

  bool same_host = meta->host_id == local_meta->host_id;
  // same_host = false;  // force RDMA

  std::shared_ptr<EndpointBase> ep;
  bool ret = false;

  if (same_host) {
    // std::cout << "[INFO] Communicator " << global_rank_
    //           << " same host detected, using IPC endpoint" << std::endl;
    ep = std::make_shared<IPCEndpoint>(config_, this);
    ret = ep->connect_to(rank);
    if (ret) {
      std::cout << "[INFO] Communicator " << global_rank_
                << " IPC connect_to succeeded to rank " << rank << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " IPC connect_to failed to rank " << rank << std::endl;
      return false;
    }
    ep->type = EndpointType::IPC;
  } else {
    // std::cout << "[INFO] Communicator " << global_rank_
    //           << " different host detected, using RDMA endpoint" <<
    //           std::endl;
    ep = std::make_shared<RDMAEndpoint>(config_, this);
    ret = ep->connect_to(rank);
    if (ret) {
      std::cout << "[INFO] Communicator " << global_rank_
                << " RDMA connect_to succeeded to rank " << rank << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " RDMA connect_to failed to rank " << rank << std::endl;
      return false;
    }
    ep->type = EndpointType::RDMA;
  }

  {
    std::lock_guard<std::mutex> lk(ep_mu_);
    rank_to_endpoint_[rank] = ep;
  }
  return ret;
}

bool Communicator::accept_from(int rank) {
  if (!check_ready()) return false;
  if (rank == global_rank_) return true;

  auto [existing_ep, ok] = get_endpoint_by_rank(rank);
  if (ok && existing_ep) return true;

  auto meta = get_communicator_meta_by_rank(rank);
  auto local_meta = get_communicator_meta_by_rank(global_rank_);
  if (!meta || !local_meta) return false;

  bool same_host = meta->host_id == local_meta->host_id;
  // same_host = false; // force RDMA

  std::shared_ptr<EndpointBase> ep;
  bool ret = false;

  if (same_host) {
    ep = std::make_shared<IPCEndpoint>(config_, this);
    ret = ep->accept_from(rank);
    if (ret) {
      std::cout << "[INFO] Communicator " << global_rank_
                << " IPC accept_from succeeded from rank " << rank << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " IPC accept_from failed from rank " << rank << std::endl;
    }
  } else {
    // RDMA: accept == connect
    ep = std::make_shared<RDMAEndpoint>(config_, this);
    ret = ep->connect_to(rank);
    if (ret) {
      std::cout << "[INFO] Communicator " << global_rank_
                << " RDMA accept succeeded from rank " << rank << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << global_rank_
                << " RDMA accept failed from rank " << rank << std::endl;
    }
  }

  {
    std::lock_guard<std::mutex> lk(ep_mu_);
    rank_to_endpoint_[rank] = ep;
  }
  return ret;
}

std::tuple<std::shared_ptr<EndpointBase>, bool>
Communicator::get_endpoint_by_rank(int rank) {
  std::lock_guard<std::mutex> lock(ep_mu_);
  auto it = rank_to_endpoint_.find(rank);
  if (it != rank_to_endpoint_.end()) {
    return {it->second, true};
  }
  return {nullptr, false};
}

unsigned Communicator::isend(int rank, void* ptr, size_t offset, size_t len,
                             uint16_t local_mr_id, uint16_t remote_mr_id,
                             bool on_gpu) {
  auto [ep, ok] = get_endpoint_by_rank(rank);
  if (!ok || !ep) return 0;

  // make sure rid never eq 0
  uint16_t seq_val =
      ep->next_send_seq_.fetch_add(1, std::memory_order_relaxed) % 4095;
  uint16_t safe_seq = seq_val + 1;  // [1, 4095]
  unsigned rid = make_request_id(rank, remote_mr_id, safe_seq);

  auto req = std::make_shared<Request>(rid, ptr, offset, len, local_mr_id,
                                       remote_mr_id, on_gpu, RequestType::SEND);

  {
    std::lock_guard<std::mutex> lk(req_mu_);
    requests_map_[rid] = req;
  }

  if (!ep->send_async(rank, req)) {
    std::lock_guard<std::mutex> lk(req_mu_);
    requests_map_.erase(rid);
    return 0;
  }
  notifier_cv_.notify_all();

  return rid;
}

unsigned Communicator::irecv(int rank, void* ptr, size_t offset, size_t len,
                             bool on_gpu) {
  auto [ep, ok] = get_endpoint_by_rank(rank);
  if (!ok || !ep) return 0;

  auto local_mr = get_local_mr(ptr);
  // make sure rid never eq 0
  uint16_t seq_val =
      ep->next_recv_seq_.fetch_add(1, std::memory_order_relaxed) % 4095;
  uint16_t safe_seq = seq_val + 1;  // [1, 4095]
  unsigned rid = make_request_id(global_rank_, local_mr.id, safe_seq);

  auto req = std::make_shared<Request>(rid, ptr, offset, len, -1, -1, on_gpu,
                                       RequestType::RECV);

  {
    std::lock_guard<std::mutex> lk(req_mu_);
    requests_map_[rid] = req;
  }

  if (!ep->recv_async(rank, req)) {
    std::lock_guard<std::mutex> lk(req_mu_);
    requests_map_.erase(rid);
    return 0;
  }
  notifier_cv_.notify_all();

  // Add a pending queue. RECV work requests (WRs) may have already
  // generated CQEs before irecv is called. For such CQEs that are not yet
  // processed, add them to a pending queue. If the queue is not empty, process
  // these CQEs here and release the corresponding requests.
  if (!cq_poller_list_.empty()) cq_poller_list_[0]->process_pending();

  return rid;
}

bool Communicator::_is_finished_locked(unsigned id) {
  auto it = requests_map_.find(id);
  if (it == requests_map_.end()) {
    // not found → already finished
    return true;
  }
  if (!it->second) {
    return true;
  }
  return it->second->finished.load(std::memory_order_acquire);
}

bool Communicator::wait_finish(std::vector<unsigned> const& reqs) {
  bool const wait_all = reqs.empty();
  std::unordered_set<unsigned> remaining;

  if (!wait_all) {
    remaining.insert(reqs.begin(), reqs.end());
  }

  while (true) {
    std::vector<unsigned> finished;

    {
      std::lock_guard<std::mutex> lk(req_mu_);

      if (wait_all) {
        for (auto const& [id, req] : requests_map_) {
          if (_is_finished_locked(id)) {
            finished.push_back(id);
          }
        }
      } else {
        for (auto id : remaining) {
          if (_is_finished_locked(id)) {
            finished.push_back(id);
          }
        }
      }

      for (auto id : finished) {
        requests_map_.erase(id);
        if (!wait_all) {
          remaining.erase(id);
        }
      }

      if ((wait_all && requests_map_.empty()) ||
          (!wait_all && remaining.empty())) {
        return true;
      }
    }

    std::this_thread::yield();
  }
}

bool Communicator::wait_finish(unsigned const req) {
  return wait_finish(std::vector<unsigned>{req});
}

void Communicator::set_communicator_meta_with_rank(
    int rank, CommunicatorMeta const& meta) {
  std::lock_guard<std::mutex> lock(meta_mu_);
  rank_to_comm_meta_[rank] = std::make_shared<CommunicatorMeta>(meta);
}

std::shared_ptr<CommunicatorMeta> Communicator::get_communicator_meta_by_rank(
    int rank) {
  std::lock_guard<std::mutex> lock(meta_mu_);
  auto it = rank_to_comm_meta_.find(rank);
  return (it != rank_to_comm_meta_.end()) ? it->second : nullptr;
}

bool Communicator::check_ready() {
  std::lock_guard<std::mutex> lock(meta_mu_);

  // Check meta map size
  if (static_cast<int>(rank_to_comm_meta_.size()) < world_size_) {
    std::cerr << "[WARN] Communicator " << global_rank_
              << " check_ready: rank_to_comm_meta_ size "
              << rank_to_comm_meta_.size() << " < world_size " << world_size_
              << std::endl;
    return false;
  }

  // Check each rank meta
  for (int i = 0; i < world_size_; i++) {
    auto it = rank_to_comm_meta_.find(i);
    if (it == rank_to_comm_meta_.end()) {
      std::cerr << "[WARN] Communicator " << global_rank_
                << " check_ready: missing CommunicatorMeta for rank " << i
                << std::endl;
      return false;
    }
    if (!it->second->is_ready) {
      std::cerr << "[WARN] Communicator " << global_rank_
                << " check_ready: CommunicatorMeta for rank " << i
                << " is not ready" << std::endl;
      return false;
    }
  }

  // Check RDMA NIC context if supported
  if (support_rdma) {
    if (!nic_ibv_ctx_) {
      std::cerr << "[WARN] Communicator " << global_rank_
                << " check_ready: nic_ibv_ctx_ is nullptr" << std::endl;
      return false;
    }
    if (!pd_) {
      std::cerr << "[WARN] Communicator " << global_rank_
                << " check_ready: pd_ is nullptr" << std::endl;
      return false;
    }
    if ((cq_list_.size()) != static_cast<size_t>(config_->cq_poller_threads)) {
      return false;
    }
  }

  std::cerr << "[INFO] Communicator " << global_rank_ << " is ready"
            << std::endl;
  return true;
}

MR Communicator::reg_mr(void* local_buf, size_t len) {
  if (!pd_) throw std::runtime_error("PD not initialized");

  ibv_mr* mr = ibv_reg_mr(pd_, local_buf, len,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                              IBV_ACCESS_REMOTE_WRITE);
  if (!mr) {
    perror("ibv_reg_mr failed");
    throw std::runtime_error("ibv_reg_mr failed");
  }

  uint16_t id = next_mr_id.fetch_add(1, std::memory_order_relaxed);

  MR info;
  info.id = id;
  info.address = reinterpret_cast<uint64_t>(local_buf);
  info.length = static_cast<uint32_t>(len);
  info.lkey = mr->lkey;
  info.key = mr->rkey;

  {
    std::lock_guard<std::mutex> lk(local_mr_mu_);
    ptr_to_local_ibv_mr_[local_buf] = mr;
    mr_id_to_local_mr_[id] = info;
  }

  return info;
}

bool Communicator::dereg_mr(void* local_buf) {
  std::lock_guard<std::mutex> lk(local_mr_mu_);
  auto it = ptr_to_local_ibv_mr_.find(local_buf);
  if (it == ptr_to_local_ibv_mr_.end()) {
    return true;
  }

  ibv_mr* mr = it->second;
  ptr_to_local_ibv_mr_.erase(it);

  if (mr) {
    if (ibv_dereg_mr(mr) != 0) {
      std::cerr << "[WARN] Communicator " << global_rank_
                << " Failed to deregister local MR" << std::endl;
      return false;
    } else {
      return true;
    }
  }

  return true;
}

bool Communicator::notify_mr(int remote_rank, MR& mr) {
  if (!exchanger_client_ || !exchanger_client_->valid()) return false;

  // we assume that user will connect to remote before notify MR.
  auto [ep, ok] = get_endpoint_by_rank(remote_rank);
  if (!ok || !ep) {
    throw std::runtime_error("Endpoint is not valid");
    return false;
  }
  if (ep->type != EndpointType::RDMA) {
    std::cout << "MR only support for EndpointRDMA, skip notify mr"
              << std::endl;
    return true;
  }

  MRInfos wrapper;
  wrapper.mrs.push_back(mr);
  std::cout << "[notify MR to rank " << remote_rank << "] addr=" << mr.address
            << " length=" << mr.length << " key=" << mr.key << std::endl;

  std::string key =
      "mr:" + std::to_string(global_rank_) + "->" + std::to_string(remote_rank);

  return exchanger_client_->publish(key, wrapper);
}

bool Communicator::wait_mr_notify(int remote_rank, MR& mr) {
  if (!exchanger_client_ || !exchanger_client_->valid()) {
    throw std::runtime_error("Exchanger client is not valid");
  }

  auto [ep, ok] = get_endpoint_by_rank(remote_rank);
  if (!ok || !ep) {
    throw std::runtime_error("Endpoint is not valid");
  }
  if (ep->type != EndpointType::RDMA) {
    std::cout << "MR only support for EndpointRDMA, skip wait_mr_notify"
              << std::endl;
    return true;
  }

  std::string key =
      "mr:" + std::to_string(remote_rank) + "->" + std::to_string(global_rank_);

  MRInfos wrapper;
  ok = exchanger_client_->wait_and_fetch(key, wrapper);
  if (!ok || wrapper.mrs.empty()) {
    throw std::runtime_error("Failed to fetch MR from remote rank=" +
                             std::to_string(remote_rank));
  }

  mr = wrapper.mrs[0];  // only support one mr now

  {
    std::lock_guard<std::mutex> lk(remote_mr_mu_);
    rank_mr_id_to_remote_mr_[remote_rank][mr.id] = mr;
  }

  std::cout << "[recv MR from rank " << remote_rank << "] addr=" << mr.address
            << " length=" << mr.length << " key=" << mr.key << std::endl;

  return true;
}

MR Communicator::get_local_mr(void* local_buf) {
  uint64_t buf_addr = reinterpret_cast<uint64_t>(local_buf);

  std::lock_guard<std::mutex> lk(local_mr_mu_);

  auto it = ptr_to_local_ibv_mr_.find(local_buf);
  if (it == ptr_to_local_ibv_mr_.end()) {
    throw std::runtime_error("Local MR not found for buffer");
  }

  for (auto& kv : mr_id_to_local_mr_) {
    const MR& mr = kv.second;
    if (buf_addr >= mr.address && buf_addr < mr.address + mr.length) {
      return mr;
    }
  }

  throw std::runtime_error("Local MR info not found");
}

MR Communicator::get_local_mr(uint16_t mr_id) {
  std::lock_guard<std::mutex> lk(local_mr_mu_);
  auto it = mr_id_to_local_mr_.find(mr_id);
  if (it == mr_id_to_local_mr_.end()) {
    throw std::runtime_error("Local MR not found for buffer");
  }
  return it->second;
}

MR Communicator::get_remote_mr(int remote_rank, uint16_t mr_id) {
  std::lock_guard<std::mutex> lk(remote_mr_mu_);
  auto it_rank = rank_mr_id_to_remote_mr_.find(remote_rank);
  if (it_rank == rank_mr_id_to_remote_mr_.end()) {
    throw std::runtime_error("No MR cached for remote rank");
  }
  auto it_mr = it_rank->second.find(mr_id);
  if (it_mr == it_rank->second.end()) {
    throw std::runtime_error("Remote MR not found for id=" +
                             std::to_string(mr_id));
  }
  return it_mr->second;
}

// Register a remote IPC cache for a given rank and buffer
bool Communicator::register_remote_ipc_cache(int remote_rank,
                                             gpuIpcMemHandle_t handle,
                                             IpcCache const& cache) {
  std::lock_guard<std::mutex> lock(remote_ipc_cache_mu_);
  rank_handle_to_ipc_cache_[remote_rank][MakeHandleKey(handle)] = cache;
  return true;
}

// Get the remote IPC cache of a buffer from a given rank
IpcCache Communicator::get_remote_ipc_cache(int remote_rank,
                                            gpuIpcMemHandle_t handle) {
  std::lock_guard<std::mutex> lock(remote_ipc_cache_mu_);
  auto it_rank = rank_handle_to_ipc_cache_.find(remote_rank);
  if (it_rank == rank_handle_to_ipc_cache_.end()) return IpcCache{};

  auto it = it_rank->second.find(MakeHandleKey(handle));
  if (it == it_rank->second.end()) return IpcCache{};
  return it->second;
}

ibv_cq* Communicator::get_cq_by_index(int index) {
  if (index < 0 || index >= static_cast<int>(cq_list_.size())) {
    return nullptr;
  }
  return cq_list_[index];
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
    auto now = std::chrono::steady_clock::now();

    {
      std::lock_guard<std::mutex> rlk(req_mu_);
      std::lock_guard<std::mutex> nlk(notifier_mu_);

      for (auto& [id, req] : requests_map_) {
        if (!req) continue;

        if (!req->finished.load(std::memory_order_acquire)) {
          continue;
        }

        if (req->notified.exchange(true, std::memory_order_acq_rel)) {
          continue;
        }

        for (auto& tgt : notify_targets_) {
          if (!tgt) continue;
          tgt->emit(id, now);
        }

        progress = true;
      }
    }

    if (!progress) {
      std::this_thread::yield();
    }
  }
}

}  // namespace Transport
}  // namespace UKernel