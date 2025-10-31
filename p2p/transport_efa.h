#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <infiniband/verbs.h>
#include "util/util.h"

namespace transport_efa {

// Check if EFA NIC is available by looking for /opt/amazon/efa folder
inline bool is_efa_available() {
    return std::filesystem::exists("/opt/amazon/efa");
}

class RDMAEndpoint {
    // RDMA devices.
    int num_devices_;
    struct ibv_device** dev_list_;

    // TCP listening port for p2p communication.
    uint16_t p2p_listen_port_;
    int p2p_listen_fd_;

public:
    RDMAEndpoint(int gid_index);

    ~RDMAEndpoint();

    std::vector<int> get_best_dev_idx(int gpu_idx);

    bool initialize_engine_by_dev(std::vector<int> dev_indices);

    uccl::ConnID uccl_connect(
        const std::vector<int>& devs, int local_gpuidx,
        const std::vector<int>& remote_devs, int remote_gpuidx,
        std::string remote_ip, uint16_t remote_port);
    
    uccl::ConnID uccl_accept(
        const std::vector<int>& devs, int listen_fd,
        int local_gpuidx, std::string remote_ip,
        const std::vector<int>& remote_devs, int remote_gpuidx);
    
    inline uint16_t get_p2p_listen_port() {
        CHECK(p2p_listen_port_ != 0) << "Error: p2p_listen_port_ is not set.";
        return p2p_listen_port_;
    }

    inline int get_p2p_listen_fd() {
        CHECK(p2p_listen_fds_ >= 0) << "Error: p2p_listen_fds_ is not set.";
        return p2p_listen_fds_;
      }

    inline const char* get_dev_name(int dev_idx) {
        CHECK(dev_list_ != nullptr) << "Device list not initialized";
        CHECK(dev_idx >= 0 && dev_idx < num_devices_) << "Invalid device index";
        return ibv_get_device_name(dev_list_[dev_idx]);
    }

private:
    std::vector<struct ibv_context*> ctx_list_;
    std::vector<struct ibv_pd*> pd_list_;
    std::vector<struct ibv_cq_ex*> cq_ex_list_;
    std::unordered_map<uint64_t, struct ibv_mr*> mr_map_;
    std::vector<struct ibv_qp*> qp_list_;
    std::vector<uint32_t> remote_rkey_list_;
    std::vector<uint64_t> remote_addr_list_;
};

}
