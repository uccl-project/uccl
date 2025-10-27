#include "transport_efa.h"
#include <iostream>
#include <arpa/inet.h>
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>
#include "util/util.h"

namespace transport_efa {

RDMAEndpoint::RDMAEndpoint(int gpu_index) 
    : p2p_listen_fd_(-1), p2p_listen_port_(0){
    // Get the number of available EFA devices
    dev_list_ = ibv_get_device_list(&num_devices_);
    if (!dev_list_ || num_devices_ <= 0) {
        perror("Failed to initialize EFA devices.\n");
        exit(1);
    }

    // Lazy initialization of the RDMA endpoint
    ctx_list_.clear();
    pd_list_.clear();
    cq_ex_list_.clear();
    mr_map_.clear();
    qp_list_.clear();
    remote_rkey_list_.clear();
    remote_addr_list_.clear();
}

RDMAEndpoint::~RDMAEndpoint() {
    // Clean up all memory regions in the map
    for (auto& pair : mr_map_) {
        if (pair.second) {
            ibv_dereg_mr(pair.second);
        }
    }
    mr_map_.clear();
    // Clean up all QPs in the vector
    for (auto qp : qp_list_) {
        if (qp) {
            ibv_destroy_qp(qp);
        }
    }
    qp_list_.clear();
    // Clean up all CQ_EXs in the vector
    for (auto cq_ex : cq_ex_list_) {
        if (cq_ex) {
            ibv_destroy_cq(ibv_cq_ex_to_cq(cq_ex));
        }
    }
    cq_ex_list_.clear();
    // Clean up all PDs in the vector
    for (auto pd : pd_list_) {
        if (pd) {
            ibv_dealloc_pd(pd);
        }
    }
    pd_list_.clear();
    // Clean up all contexts in the vector
    for (auto ctx : ctx_list_) {
        if (ctx) {
            ibv_close_device(ctx);
        }
    }
    ctx_list_.clear();
    remote_rkey_list_.clear();
    remote_addr_list_.clear();

    // Clean up the p2p listen fds and ports
    if (p2p_listen_fd_ >= 0) {
        close(p2p_listen_fd_);
    }

    // Clean up device list
    if (dev_list_) {
        ibv_free_device_list(dev_list_);
    }
}

std::vector<int> RDMAEndpoint::get_best_dev_idx(int gpu_idx) {    
    // Ranked by GPU idx
    auto gpu_cards = uccl::get_gpu_cards();
    // Ranked by RDMA NIC name (not the ibv_get_device_list order)
    auto ib_nics = uccl::get_rdma_nics();
    // Get GPU pcie path
    auto gpu_device_path = gpu_cards[gpu_idx];
    // Find the RDMA NIC that is closest to the GPU.

    std::vector<std::pair<std::string, uint32_t>> dist;
    dist.reserve(ib_nics.size());

    std::vector<std::string> selected_nic_names;
    for (auto& nic : ib_nics) {
        uint32_t d = uccl::safe_pcie_distance(gpu_device_path, nic.second);
        dist.emplace_back(nic.first, d);
    }

    if (dist.empty()) {
        fprintf(stderr, "[WARN] no NIC found, defaulting to empty\n");
        selected_nic_names.clear();
    } else {
        // Find the minimum distance
        auto min_it = std::min_element(
            dist.begin(), dist.end(),
            [](auto const& a, auto const& b) { return a.second < b.second; });
        auto min_d = min_it->second;

        // Collect all NICs with equal minimum distance
        std::vector<std::string> candidates;
        for (auto& p : dist) {
            if (p.second == min_d) candidates.push_back(p.first);
        }

        if (candidates.empty()) {
            fprintf(stderr, 
                    "[WARN] no candidate NIC found, defaulting to first\n");
            selected_nic_names.push_back(dist.front().first);
        } else {
            // NOTE(xzhiying): This is a temporary hack.
            // On p5en, there are 4 NICs with the same distance.
            // GPU0 uses candidates[0/1], GPU1 uses candidates[2/3], etc.
            assert(candidates.size() == 4);
            int half_size = candidates.size() / 2;
            int start_idx = (gpu_idx % 2 == 0) ? 0 : half_size;
            int end_idx = start_idx + half_size;
            for (int i = start_idx; i < end_idx; i++) {
                selected_nic_names.push_back(candidates[i]);
            }
        }
    }

    std::vector<int> selected_dev_indices;
    for (const auto& nic_name : selected_nic_names) {
        int dev_idx = -1;
        for (int i = 0; i < num_devices_; i++) {
            if (strcmp(ibv_get_device_name(dev_list_[i]), 
                       nic_name.c_str()) == 0) {
                dev_idx = i;
                break;
            }
        }
        if (dev_idx < 0) {
            fprintf(stderr,
                    "[FATAL] Selected RDMA NIC '%s' not found in verbs "
                    "device list\n",
                    nic_name.c_str());
            std::abort();
        }
        selected_dev_indices.push_back(dev_idx);
    }
    
    printf("[RDMA] GPU %d selected NICs: ", gpu_idx);
    for (size_t i = 0; i < selected_nic_names.size(); i++) {
        printf("%s (idx %d)", 
               selected_nic_names[i].c_str(), selected_dev_indices[i]);
        if (i < selected_nic_names.size() - 1) 
            printf(", ");
    }
    printf("\n");
    
    return selected_dev_indices;
}

bool RDMAEndpoint::initialize_engine_by_dev(std::vector<int> dev_indices){
    for (int dev_idx : dev_indices) {
        struct ibv_context* ctx_ = ibv_open_device(dev_list_[dev_idx]);
        printf("Using device: %s\n", ibv_get_device_name(dev_list_[dev_idx]));
        if (!ctx_) {
            std::cerr << "Failed to open device" << std::endl;
            exit(1);
        }
        ctx_list_.push_back(ctx_);

        struct ibv_pd* pd_ = ibv_alloc_pd(ctx_);
        pd_list_.push_back(pd_);

        struct ibv_cq_init_attr_ex init_attr_ex = {
            .cqe = 1024,
            .cq_context = NULL,
            .channel = NULL,
            .comp_vector = 0,
            /* EFA requires these values for wc_flags and comp_mask.
            * See `efa_create_cq_ex` in rdma-core.
            */
            .wc_flags = IBV_WC_STANDARD_FLAGS,
            .comp_mask = 0,
        };

        struct ibv_cq_ex* cq_ex_ = ibv_create_cq_ex(ctx_, &init_attr_ex);
        if (!pd_ || !cq_ex_) {
            perror("Failed to allocate PD or CQ");
            exit(1);
        }
        cq_ex_list_.push_back(cq_ex_);
    }
    p2p_listen_port_ = uccl::create_listen_socket(&p2p_listen_fd_);
    DCHECK(p2p_listen_port_ >= 0) << "Failed to bind after trying many ports!";
    printf("P2P listening on port %d\n", p2p_listen_port_);
    
    return true;
}

}
