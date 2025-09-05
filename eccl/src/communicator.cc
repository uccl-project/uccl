#include "transport.h"
#include "utils.h"
#include "util/util.h"

Communicator::Communicator(int gpu_id, int rank, int world_size, std::shared_ptr<Config> config):
    gpu_id_(gpu_id), local_rank_(rank), world_size_(world_size), config_(config) {
    // find best NIC for cur gpu
    auto [nic_id, nic_name] = find_best_rdma_for_gpu(gpu_id_);
    if (nic_id != -1) { // support RDMA
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
            std::cerr << "Failed to open ibv context for NIC: " << nic_name << std::endl;
            ibv_free_device_list(dev_list);
            std::abort();
        }
        ibv_free_device_list(dev_list);

        std::cout << "Communicator initialized: GPU " << gpu_id_
                << " map to RDMA NIC " << nic_name << std::endl;

        support_rdma = true;
        // Init RAMD resource
        pd_ = ibv_alloc_pd(nic_ibv_ctx_);
        if (!pd_) {
            perror("Failed to allocate PD");
            std::abort();
        }
    } else { // does not support RDMA
        // TODO: if we can't find any rdma nic, we still can do ipc comm on local host.
        support_rdma = false;
    }

    // Initialize communicator meta
    CommunicatorMeta local{};
    local.host_id = generate_host_id();
    local.is_ready = true;
    set_communicator_meta_with_rank(local_rank_, local);

    std::cout << "Communicator initialized: Meta local_rank: " << local_rank_
              << " world_size: " << world_size_
              << " host_id: " << local.host_id << std::endl;

    // initialize Redis client
    redis_client_ = std::make_shared<RedisExchanger>(config->redis_ip, config->redis_port);
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
    for (int i=0; i<world_size_; i++) {
        if(i == local_rank_) continue;
        std::string key = "meta:" + std::to_string(i);
        if (redis_client_->wait_and_fetch(key, remote, -1)) {
            set_communicator_meta_with_rank(i, remote);
        } else {
            fprintf(stderr, "[WARN] Timeout waiting for remote CommunicatorMeta \n");
        }
    }
    std::cout << "Communicator initialized: rank_to_comm_meta_ success" << std::endl;
}

Communicator::~Communicator() {
}

bool Communicator::connect_to(int rank) {
    if (check_ready()) {
        if (rank == local_rank_) return true;
        if (rank < 0 || rank >= world_size_) {
            return false;
        }

        auto meta = get_communicator_meta_by_rank(rank);
        auto local_meta = get_communicator_meta_by_rank(local_rank_);

        if (meta->host_id == local_meta->host_id) { // same host
            auto ep = std::make_shared<IPCEndpoint>(config_);
            // return ep->connect_to(rank);
            return true;
        } else {
            auto ep = std::make_shared<RDMAEndpoint>(config_);
            // return ep->connect_to(rank);
            return true;
        }
    }
    return false;
}

std::tuple<std::shared_ptr<EndpointBase>, bool> Communicator::get_endpoint_by_rank(int rank) {
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

void Communicator::set_communicator_meta_with_rank(int rank, const CommunicatorMeta& meta) {
    std::lock_guard<std::mutex> lock(meta_mu_);
    rank_to_comm_meta_[rank] = std::make_shared<CommunicatorMeta>(meta);
}

std::shared_ptr<CommunicatorMeta> Communicator::get_communicator_meta_by_rank(int rank) {
    std::lock_guard<std::mutex> lock(meta_mu_);
    auto it = rank_to_comm_meta_.find(rank);
    return (it != rank_to_comm_meta_.end()) ? it->second : nullptr;
}

bool Communicator::check_ready() {
    // check meta map
    std::lock_guard<std::mutex> lock(meta_mu_);
    if (static_cast<int>(rank_to_comm_meta_.size()) < world_size_) return false;

    for (int i = 0; i < world_size_; i++) {
        auto it = rank_to_comm_meta_.find(i);
        if (it == rank_to_comm_meta_.end()) {
            return false;
        }
        if (!it->second->is_ready) {
            return false;
        }
    }
    
    // check nic ibv context
    // if (!nic_ibv_ctx_) return false;

    return true;
}

void Communicator::register_local_mr(void* local_buf, size_t len) {
    std::lock_guard<std::mutex> lock(local_mr_mu_);
    // TODO: ibv_reg_mr
    // ptr_to_local_mr_[local_buf] = std::make_shared<ibv_mr>(/* init mr */);
}

std::shared_ptr<ibv_mr> Communicator::get_local_mr(void* local_buf) {
    std::lock_guard<std::mutex> lock(local_mr_mu_);
    auto it = ptr_to_local_mr_.find(local_buf);
    return (it != ptr_to_local_mr_.end()) ? it->second : nullptr;
}

// Register a remote MR for a given rank
void Communicator::register_remote_mr(int remote_rank, void* local_buf, std::shared_ptr<ibv_mr> remote_mr) {
    std::lock_guard<std::mutex> lock(remote_mr_mu_);
    rank_ptr_to_remote_mr_[remote_rank][local_buf] = remote_mr;
}

// Get remote MR
std::shared_ptr<ibv_mr> Communicator::get_remote_mr(int remote_rank, void* local_buf) {
    std::lock_guard<std::mutex> lock(remote_mr_mu_);
    auto it_rank = rank_ptr_to_remote_mr_.find(remote_rank);
    if (it_rank != rank_ptr_to_remote_mr_.end()) {
        auto it_buf = it_rank->second.find(local_buf);
        if (it_buf != it_rank->second.end()) return it_buf->second;
    }
    return nullptr;
}

// Register a local IPC cache for a buffer
void Communicator::register_local_ipc_cache(void* local_buf) {
    std::lock_guard<std::mutex> lock(local_ipc_cache_mu_);
    // TODO: open gpu ipc handle
    // ptr_to_local_ipc_cache_[local_buf] = cache;
}

// Get the IPC cache of a local buffer
IpcCache Communicator::get_local_ipc_cache(void* local_buf) {
    std::lock_guard<std::mutex> lock(local_ipc_cache_mu_);
    auto it = ptr_to_local_ipc_cache_.find(local_buf);
    if (it != ptr_to_local_ipc_cache_.end()) return it->second;
    return IpcCache{};
}

// Register a remote IPC cache for a given rank and buffer
void Communicator::register_remote_ipc_cache(int remote_rank, void* local_buf, const IpcCache& cache) {
    std::lock_guard<std::mutex> lock(remote_ipc_cache_mu_);
    rank_ptr_to_ipc_cache_[remote_rank][local_buf] = cache;
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

