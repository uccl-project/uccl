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
    
    inline uint16_t get_p2p_listen_port() {
        CHECK(p2p_listen_port_ != 0) << "Error: p2p_listen_port_ is not set.";
        return p2p_listen_port_;
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
