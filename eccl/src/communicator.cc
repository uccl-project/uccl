#include "transport.h"
#include "util/util.h"
#include "utils.h"

Communicator::Communicator(int gpu_id, int rank, int world_size,
                           std::shared_ptr<Config> config)
    : gpu_id_(gpu_id),
      local_rank_(rank),
      world_size_(world_size),
      config_(config) {
  // find best NIC for cur gpu
  auto [nic_id, nic_name] = find_best_rdma_for_gpu(gpu_id_);
  if (nic_id != -1) {  // support RDMA
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

    std::cout << "[INFO] Communicator " << local_rank_ << " initialized: GPU "
              << gpu_id_ << " map to RDMA NIC " << nic_name << std::endl;

    support_rdma = true;
    // Init RAMD resource
    // create pd
    pd_ = ibv_alloc_pd(nic_ibv_ctx_);
    if (!pd_) {
      perror("Failed to allocate PD");
      std::abort();
    }
    // create cq
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

  } else {  // does not support RDMA
    // TODO: if we can't find any rdma nic, we still can do ipc comm on local
    // host.
    support_rdma = false;
  }

  // Initialize communicator meta
  CommunicatorMeta local{};
  local.host_id = generate_host_id();
  local.is_ready = true;
  set_communicator_meta_with_rank(local_rank_, local);

  std::cout << "[INFO] Communicator " << local_rank_
            << " initialized: Meta local_rank: " << local_rank_
            << " world_size: " << world_size_ << " host_id: " << local.host_id
            << std::endl;

  // initialize Redis client
  redis_client_ =
      std::make_shared<RedisExchanger>(config->redis_ip, config->redis_port);
  if (!redis_client_->valid()) {
    fprintf(stderr, "[ERROR] Publisher failed to connect to Redis\n");
    return;
  }

  // exchange communicator meta
  std::string meta_key = "meta:" + std::to_string(local_rank_);
  if (!redis_client_->publish(meta_key, local)) {
    fprintf(stderr, "[ERROR] Failed to publish local CommunicatorMeta \n");
  }

  // get all others meta
  CommunicatorMeta remote{};
  for (int i = 0; i < world_size_; i++) {
    if (i == local_rank_) continue;
    std::string key = "meta:" + std::to_string(i);
    if (redis_client_->wait_and_fetch(key, remote, -1)) {
      set_communicator_meta_with_rank(i, remote);
    } else {
      fprintf(stderr, "[WARN] Timeout waiting for remote CommunicatorMeta \n");
    }
  }
  std::cout << "[INFO] Communicator " << local_rank_
            << " initialized: rank_to_comm_meta_ success" << std::endl;
}

Communicator::~Communicator() {
  // Release endpoints first
  {
    std::lock_guard<std::mutex> lk(ep_mu_);
    for (auto& [rank, ep] : rank_to_endpoint_) {
      if (ep) {
        ep.reset();
      }
    }
    rank_to_endpoint_.clear();
  }

  // Release RDMA resources
  if (!cq_list_.empty()) {
    for (auto& cq : cq_list_) {
      if (cq) {
        if (ibv_destroy_cq(cq)) {
          std::cerr << "[WARN] Communicator " << local_rank_
                    << " Failed to destroy CQ" << std::endl;
        } else {
          std::cout << "[INFO] Communicator " << local_rank_ << " CQ destroyed"
                    << std::endl;
        }
      }
    }
    cq_list_.clear();
  }

  if (pd_) {
    if (ibv_dealloc_pd(pd_)) {
      std::cerr << "[WARN] Communicator " << local_rank_
                << " Failed to deallocate PD" << std::endl;
    } else {
      std::cout << "[INFO] Communicator " << local_rank_ << " PD deallocated"
                << std::endl;
    }
    pd_ = nullptr;
  }

  if (nic_ibv_ctx_) {
    if (ibv_close_device(nic_ibv_ctx_)) {
      std::cerr << "[WARN] Communicator " << local_rank_
                << " Failed to close IBV device context" << std::endl;
    } else {
      std::cout << "[INFO] Communicator " << local_rank_
                << " IBV device context closed" << std::endl;
    }
    nic_ibv_ctx_ = nullptr;
  }

  // Release local memory regions
  {
    std::lock_guard<std::mutex> lk(local_mr_mu_);
    for (auto& [ptr, mr] : ptr_to_local_mr_) {
      if (mr) {
        if (ibv_dereg_mr(mr)) {
          std::cerr << "[WARN] Communicator " << local_rank_
                    << " Failed to deregister local MR" << std::endl;
        } else {
          std::cout << "[INFO] Communicator " << local_rank_
                    << " Local MR deregistered" << std::endl;
        }
      }
    }
    ptr_to_local_mr_.clear();
  }

  // Release remote memory regions
  {
    std::lock_guard<std::mutex> lk(remote_mr_mu_);
    // for (auto& [rank, mr_map] : rank_ptr_to_remote_mr_) {
    //     for (auto& [ptr, mr] : mr_map) {
    //         // remote MR usually not owned by this process; optionally skip
    //         deregister
    //         // just for safety, you can log
    //         std::cout << "[INFO] Cleaned remote MR cache for rank " << rank
    //         << std::endl;
    //     }
    // }
    rank_ptr_to_remote_mr_.clear();
  }

  // Release IPC caches
  {
    std::lock_guard<std::mutex> lk(local_ipc_cache_mu_);
    ptr_to_local_ipc_cache_.clear();
  }
  {
    std::lock_guard<std::mutex> lk(remote_ipc_cache_mu_);
    rank_ptr_to_ipc_cache_.clear();
  }

  std::cout << "[INFO] Communicator " << local_rank_ << " resources released"
            << std::endl;
}

bool Communicator::connect_to(int rank) {
  if (!check_ready()) {
    std::cerr << "[WARN] Communicator " << local_rank_
              << " not ready, cannot connect to rank " << rank << std::endl;
    return false;
  }

  if (rank == local_rank_) {
    std::cout << "[INFO] Communicator " << local_rank_
              << " connect_to called with own rank " << rank
              << ", nothing to do" << std::endl;
    return true;
  }

  if (rank < 0 || rank >= world_size_) {
    std::cerr << "[ERROR] Communicator " << local_rank_ << " invalid rank "
              << rank << ", world_size=" << world_size_ << std::endl;
    return false;
  }

  auto meta = get_communicator_meta_by_rank(rank);
  auto local_meta = get_communicator_meta_by_rank(local_rank_);

  if (!meta) {
    std::cerr << "[ERROR] Communicator " << local_rank_
              << " CommunicatorMeta not found for rank " << rank << std::endl;
    return false;
  }

  std::cout << "[INFO] Communicator " << local_rank_
            << " connecting to rank=" << rank
            << ", local_host_id=" << local_meta->host_id
            << ", peer_host_id=" << meta->host_id << std::endl;

  // 强制使用 RDMA，即使在同主机
  bool same_host = meta->host_id == local_meta->host_id;
  same_host = false;

  if (same_host) {
    std::cout << "[INFO] Communicator " << local_rank_
              << " same host detected, using IPC endpoint" << std::endl;
    auto ep = std::make_shared<IPCEndpoint>(config_, this);
    // ep->connect_to(rank); // optional if IPCEndpoint needs explicit connect
    return true;
  } else {
    std::cout << "[INFO] Communicator " << local_rank_
              << " different host detected, using RDMA endpoint" << std::endl;
    auto ep = std::make_shared<RDMAEndpoint>(config_, this);
    bool ret = ep->connect_to(rank);
    if (ret) {
      std::cout << "[INFO] Communicator " << local_rank_
                << " RDMA connect_to succeeded to rank " << rank << std::endl;
    } else {
      std::cerr << "[ERROR] Communicator " << local_rank_
                << " RDMA connect_to failed to rank " << rank << std::endl;
    }
    return ret;
  }
}

bool Communicator::accept_from(int rank) { return connect_to(rank); }

std::tuple<std::shared_ptr<EndpointBase>, bool>
Communicator::get_endpoint_by_rank(int rank) {
  std::lock_guard<std::mutex> lock(ep_mu_);
  auto it = rank_to_endpoint_.find(rank);
  if (it != rank_to_endpoint_.end()) {
    return {it->second, true};
  }
  return {nullptr, false};
}

bool Communicator::isend(int rank, std::shared_ptr<Request> req) {
  auto [ep, ok] = get_endpoint_by_rank(rank);
  if (ok) {
    // ep->send_async(req);
    return true;
  } else {
    return false;
  }
}

bool Communicator::irecv(int rank, std::shared_ptr<Request> req) {
  auto [ep, ok] = get_endpoint_by_rank(rank);
  if (ok) {
    // ep->recv_async(req);
    return true;
  } else {
    return false;
  }
}

bool Communicator::irecv_red(int rank, std::shared_ptr<Request> req) {
  auto [ep, ok] = get_endpoint_by_rank(rank);
  if (ok) {
    // ep->recv_async(req);
    return true;
  } else {
    return false;
  }
}

bool Communicator::poll_finish(std::vector<std::shared_ptr<Request>>& reqs) {
  size_t n = reqs.size();
  std::vector<bool> unfinished(n, true);
  size_t remaining = n;

  while (remaining > 0) {
    for (size_t i = 0; i < n; ++i) {
      if (!unfinished[i]) continue;

      if (reqs[i]->finished.load(std::memory_order_acquire)) {
        unfinished[i] = false;
        --remaining;
      }
    }
    std::this_thread::yield();
  }
  return true;
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

  // check meta map size
  if (static_cast<int>(rank_to_comm_meta_.size()) < world_size_) {
    std::cout << "[WARN] Communicator " << local_rank_
              << " check_ready: rank_to_comm_meta_ size "
              << rank_to_comm_meta_.size() << " < world_size " << world_size_
              << std::endl;
    return false;
  }

  // check each rank meta
  for (int i = 0; i < world_size_; i++) {
    auto it = rank_to_comm_meta_.find(i);
    if (it == rank_to_comm_meta_.end()) {
      std::cout << "[WARN] Communicator " << local_rank_
                << " check_ready: missing CommunicatorMeta for rank " << i
                << std::endl;
      return false;
    }
    if (!it->second->is_ready) {
      std::cout << "[WARN] Communicator " << local_rank_
                << " check_ready: CommunicatorMeta for rank " << i
                << " is not ready" << std::endl;
      return false;
    }
  }

  // check RDMA NIC context if supported
  if (support_rdma) {
    if (!nic_ibv_ctx_) {
      std::cout << "[WARN] Communicator " << local_rank_
                << " check_ready: nic_ibv_ctx_ is nullptr" << std::endl;
      return false;
    }
    if (!pd_) {
      std::cout << "[WARN] Communicator " << local_rank_
                << " check_ready: pd_ is nullptr" << std::endl;
      return false;
    }
    if (cq_list_.size() != static_cast<size_t>(config_->cq_poller_threads)) {
      std::cout << "[WARN] Communicator " << local_rank_
                << " check_ready: cq_list_.size()=" << cq_list_.size()
                << " != config_->cq_poller_threads="
                << config_->cq_poller_threads << std::endl;
      return false;
    }
  }

  std::cout << "[INFO] Communicator " << local_rank_ << " is ready"
            << std::endl;
  return true;
}

bool Communicator::register_local_mr(void* local_buf, size_t len) {
  std::lock_guard<std::mutex> lock(local_mr_mu_);
  // TODO: ibv_reg_mr
  // ptr_to_local_mr_[local_buf] = std::make_shared<ibv_mr>(/* init mr */);
  return true;
}

ibv_mr* Communicator::get_local_mr(void* local_buf) {
  std::lock_guard<std::mutex> lock(local_mr_mu_);
  auto it = ptr_to_local_mr_.find(local_buf);
  return (it != ptr_to_local_mr_.end()) ? it->second : nullptr;
}

// Register a remote MR for a given rank
bool Communicator::register_remote_mr(int remote_rank, void* local_buf,
                                      ibv_mr* remote_mr) {
  std::lock_guard<std::mutex> lock(remote_mr_mu_);
  rank_ptr_to_remote_mr_[remote_rank][local_buf] = remote_mr;
  return true;
}

// Get remote MR
ibv_mr* Communicator::get_remote_mr(int remote_rank, void* local_buf) {
  std::lock_guard<std::mutex> lock(remote_mr_mu_);
  auto it_rank = rank_ptr_to_remote_mr_.find(remote_rank);
  if (it_rank != rank_ptr_to_remote_mr_.end()) {
    auto it_buf = it_rank->second.find(local_buf);
    if (it_buf != it_rank->second.end()) return it_buf->second;
  }
  return nullptr;
}

// Register a local IPC cache for a buffer
bool Communicator::register_local_ipc_cache(void* local_buf) {
  std::lock_guard<std::mutex> lock(local_ipc_cache_mu_);
  // TODO: open gpu ipc handle
  // ptr_to_local_ipc_cache_[local_buf] = cache;
  return true;
}

// Get the IPC cache of a local buffer
IpcCache Communicator::get_local_ipc_cache(void* local_buf) {
  std::lock_guard<std::mutex> lock(local_ipc_cache_mu_);
  auto it = ptr_to_local_ipc_cache_.find(local_buf);
  if (it != ptr_to_local_ipc_cache_.end()) return it->second;
  return IpcCache{};
}

// Register a remote IPC cache for a given rank and buffer
bool Communicator::register_remote_ipc_cache(int remote_rank, void* local_buf,
                                             IpcCache const& cache) {
  std::lock_guard<std::mutex> lock(remote_ipc_cache_mu_);
  rank_ptr_to_ipc_cache_[remote_rank][local_buf] = cache;
  return true;
}

// Get the remote IPC cache of a buffer from a given rank
IpcCache Communicator::get_remote_ipc_cache(int remote_rank, void* local_buf) {
  std::lock_guard<std::mutex> lock(remote_ipc_cache_mu_);
  auto it_rank = rank_ptr_to_ipc_cache_.find(remote_rank);
  if (it_rank != rank_ptr_to_ipc_cache_.end()) {
    auto it_buf = it_rank->second.find(local_buf);
    if (it_buf != it_rank->second.end()) return it_buf->second;
  }
  return IpcCache{};
}

ibv_cq* Communicator::get_cq_by_index(int index) {
  if (index < 0 || index >= static_cast<int>(cq_list_.size())) {
    return nullptr;
  }
  return cq_list_[index];
}
