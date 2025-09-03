#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "util/gpu_rt.h"
#include "config.h"
#include "request.h"
#include "oob.h"

class EndpointBase {
    virtual bool send_async(std::shared_ptr<CommRequest> creq) = 0;
    virtual bool recv_async(std::shared_ptr<CommRequest> creq) = 0;
    virtual bool poll_reqs(std::vector<std::shared_ptr<CommRequest>>& creqs) = 0;
};

struct QPcontext {

};

class RDMAEndpoint: public EndpointBase {
public:
    RDMAEndpoint(std::shared_ptr<Config> config);
    ~RDMAEndpoint();

    bool connect_to(std::string server_ip, int port);

    bool send_async(std::shared_ptr<CommRequest> creq) override;
    bool recv_async(std::shared_ptr<CommRequest> creq) override;
    bool poll_reqs(std::vector<std::shared_ptr<CommRequest>>& creqs) override;

private:
    std::vector<std::shared_ptr<QPcontext>> qp_context_list;
    std::shared_ptr<Config> config_;
};

struct IPCcontext {
  gpuIpcMemHandle_t handle;
  bool is_send;
  void* direct_ptr;
  uintptr_t offset;
  size_t size;
};

class IPCEndpoint: public EndpointBase {
public:
    IPCEndpoint(std::shared_ptr<Config> config);
    ~IPCEndpoint();

    bool connect_to(int uds_fd);

    bool send_async(std::shared_ptr<CommRequest> creq) override;
    bool recv_async(std::shared_ptr<CommRequest> creq) override;
    bool poll_reqs(std::vector<std::shared_ptr<CommRequest>>& creqs) override;

private:
    std::shared_ptr<IPCcontext> ipc_context;
    std::shared_ptr<Config> config_;
};

class Communicator {
    // one gpu + nearest nic
public:
    Communicator(int gpu_id, int rank, int world_size, std::shared_ptr<Config> config = std::make_shared<Config>());
    ~Communicator();

    bool connect_to(int rank); // connect and add ep to endpoint_map_
    std::tuple<std::shared_ptr<EndpointBase>, bool> get_endpoint_by_rank(int rank);

    bool send_async(std::shared_ptr<Request> req);
    bool recv_async(std::shared_ptr<Request> req);
    bool poll_finish(std::vector<std::shared_ptr<Request>>& reqs);

    // set communicator meta with rank, user get these info on their way
    void set_communicator_meta_with_rank(int rank, const CommunicatorMeta& meta);
    std::shared_ptr<CommunicatorMeta> get_communicator_meta_by_rank(int rank);
    // check if all config is ready
    bool check_ready();
private:
    std::unordered_map<int, std::shared_ptr<EndpointBase>> endpoint_map_; // rank->ep
    mutable std::mutex ep_mu_;

    std::unordered_map<int, std::shared_ptr<CommunicatorMeta>> comm_meta_map_; // rank->mCommunicatorMeta
    mutable std::mutex meta_mu_;

    int gpu_id_;
    int local_rank_;
    int world_size_;
    struct ibv_context* nic_ibv_ctx_{nullptr};

    std::shared_ptr<Config> config_;

    std::shared_ptr<RedisExchanger> redis_client_;
    mutable std::mutex redis_client_mu_;
};
