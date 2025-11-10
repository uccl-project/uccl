#include "transport_efa.h"
#include <iostream>
#include <arpa/inet.h>
#include <algorithm>
#include <cstdint>
#include <cinttypes>
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
#include <ctime>
#include <chrono>
#include <thread>
#include "util/util.h"
#include "../collective/rdma/rdma_io.h"

#define GID_INDEX 0
#define PORT_NUM 1
#define QKEY 0x12345

namespace transport_efa {

struct metadata {
  uint32_t qpn;
  union ibv_gid gid;
};

RDMAEndpoint::RDMAEndpoint(int gpu_index) 
    : p2p_listen_fd_(-1), p2p_listen_port_(0) {
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
    qp_list_.clear();
    ah_map_.clear();
    qpn_map_.clear();
    mr_map_.clear();
    completed_req_set_list_.clear();
}

RDMAEndpoint::~RDMAEndpoint() {
    // Clean up all memory regions in the map
    for (auto& pair : mr_map_) {
        for (auto mr : pair.second) {
            if (mr) {
                ibv_dereg_mr(mr);
            }
        }
    }
    mr_map_.clear();
    // Clean up all address handles in the map
    for (auto& [conn_id, ah_list] : ah_map_) {
        for (auto ah : ah_list) {
            if (ah) {
                ibv_destroy_ah(ah);
            }
        }
    }
    ah_map_.clear();
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
    qpn_map_.clear();
    completed_req_set_list_.clear();

    // Clean up the p2p listen fd and port
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
        printf("%s (device idx %d)", 
               selected_nic_names[i].c_str(), selected_dev_indices[i]);
        if (i < selected_nic_names.size() - 1) 
            printf(", ");
    }
    printf("\n");
    
    return selected_dev_indices;
}

// Create and configure a SRD QP
struct ibv_qp* RDMAEndpoint::create_qp(struct ibv_context* ctx,
                                        struct ibv_pd* pd,
                                        struct ibv_cq_ex* cq_ex) {
    struct ibv_qp_init_attr_ex qp_attr_ex = {0};
    struct efadv_qp_init_attr efa_attr = {0};
  
    qp_attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
    qp_attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE |
                                IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM |
                                IBV_QP_EX_WITH_RDMA_READ;
  
    qp_attr_ex.cap.max_send_wr = 256;
    qp_attr_ex.cap.max_recv_wr = 256;
    qp_attr_ex.cap.max_send_sge = 1;
    qp_attr_ex.cap.max_recv_sge = 1;
    qp_attr_ex.cap.max_inline_data = 0;
  
    qp_attr_ex.pd = pd;
    qp_attr_ex.qp_context = ctx;
    qp_attr_ex.sq_sig_all = 1;
  
    qp_attr_ex.send_cq = ibv_cq_ex_to_cq(cq_ex);
    qp_attr_ex.recv_cq = ibv_cq_ex_to_cq(cq_ex);
  
    qp_attr_ex.qp_type = IBV_QPT_DRIVER;
  
    efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
  #define EFA_QP_LOW_LATENCY_SERVICE_LEVEL 8
    efa_attr.sl = EFA_QP_LOW_LATENCY_SERVICE_LEVEL;
  
    struct ibv_qp* qp = efadv_create_qp_ex(ctx, &qp_attr_ex, &efa_attr,
                                           sizeof(struct efadv_qp_init_attr));
  
    if (!qp) {
      perror("Failed to create QP");
      exit(1);
    }
  
    struct ibv_qp_attr attr = {};
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = PORT_NUM;
    attr.qkey = QKEY;
    if (ibv_modify_qp(
            qp, &attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
      perror("Failed to modify QP to INIT");
      exit(1);
    }
  
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) {
      perror("Failed to modify QP to RTR");
      exit(1);
    }
  
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
  #define EFA_RDM_DEFAULT_RNR_RETRY (3)
    attr.rnr_retry = EFA_RDM_DEFAULT_RNR_RETRY;  // Set RNR retry count
    if (ibv_modify_qp(qp, &attr,
                      IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_RNR_RETRY)) {
      perror("Failed to modify QP to RTS");
      exit(1);
    }
  
    return qp;
}

// Retrieve GID based on gid_index
void RDMAEndpoint::get_gid(struct ibv_context* ctx, int gid_index, 
                           union ibv_gid* gid) {
  if (ibv_query_gid(ctx, PORT_NUM, gid_index, gid)) {
    perror("Failed to query GID");
    exit(1);
  }
  printf("GID[%d]: %s\n", gid_index, inet_ntoa(*(struct in_addr*)&gid->raw[8]));
}

// Create AH using specific GID index
struct ibv_ah* RDMAEndpoint::create_ah(struct ibv_pd* pd,
                                        union ibv_gid remote_gid) {
    struct ibv_ah_attr ah_attr = {0};

    ah_attr.port_num = PORT_NUM;
    ah_attr.is_global = 1;          // Enable Global Routing Header (GRH)
    ah_attr.grh.dgid = remote_gid;  // Destination GID

    struct ibv_ah* ah = ibv_create_ah(pd, &ah_attr);
    if (!ah) {
        perror("Failed to create AH");
        exit(1);
    }
    return ah;
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
        completed_req_set_list_.push_back(std::unordered_set<uint64_t>());

        struct ibv_qp* qp = create_qp(ctx_, pd_, cq_ex_);
        qp_list_.push_back(qp);
    }
    p2p_listen_port_ = uccl::create_listen_socket(&p2p_listen_fd_);
    DCHECK(p2p_listen_port_ >= 0) << "Failed to bind after trying many ports!";
    printf("P2P listening on port %d\n", p2p_listen_port_);
    
    return true;
}

bool RDMAEndpoint::uccl_connect(
    uint64_t conn_id,
    const std::vector<int>& devs, int local_gpuidx,
    const std::vector<int>& remote_devs, int remote_gpuidx,
    std::string remote_ip, uint16_t remote_port) {

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(remote_port);
    addr.sin_addr.s_addr = inet_addr(remote_ip.c_str());
    
    printf("Client attempting connection...\n");
    int attempts = 5;
    while (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0 &&
            attempts--) {
        perror("Connect failed, retrying...");
        sleep(1);
    }
    if (attempts == 0) {
        perror("Failed to connect after retries");
        exit(1);
    }
    printf("Client connected\n");
    
    // Set receive timeout to avoid blocking
    struct timeval timeout = {5, 0};  // 5 seconds timeout
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    
    auto& ah_list = ah_map_[conn_id];
    auto& qpn_list = qpn_map_[conn_id];
    for (int i = 0; i < devs.size(); i++) {
        metadata local_meta, remote_meta;
        local_meta.qpn = qp_list_[i]->qp_num;
        get_gid(ctx_list_[i], GID_INDEX, &local_meta.gid);

        // Send local metadata
        if (send(sock, &local_meta, sizeof(local_meta), 0) <= 0)
            perror("send() failed");

        // Receive remote metadata
        if (recv(sock, &remote_meta, sizeof(remote_meta), 0) <= 0)
            perror("recv() timeout");
        
        printf("QPN and GID exchanged for device idx %d\n", devs[i]);

        // This is the key to enabling RDMA read/write over SRD.
        ah_list.push_back(create_ah(pd_list_[i], remote_meta.gid));
        qpn_list.push_back(remote_meta.qpn);
    }
    
    close(sock);
    
    return true;
}

bool RDMAEndpoint::uccl_accept(
    uint64_t conn_id,
    const std::vector<int>& devs, int listen_fd,
    int local_gpuidx, std::string& remote_ip,
    const std::vector<int>& remote_devs, int* remote_gpuidx) {

    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    printf("Server waiting for connection...\n");
    int sock = accept(listen_fd, (struct sockaddr*)&addr, &addr_len);
    printf("Server accepted connection\n");
    
    // Set receive timeout to avoid blocking
    struct timeval timeout = {5, 0};  // 5 seconds timeout
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    
    auto& ah_list = ah_map_[conn_id];
    auto& qpn_list = qpn_map_[conn_id];
    for (int i = 0; i < devs.size(); i++) {
        metadata local_meta, remote_meta;
        local_meta.qpn = qp_list_[i]->qp_num;
        get_gid(ctx_list_[i], GID_INDEX, &local_meta.gid);

        // Receive remote metadata first (client sends first)
        if (recv(sock, &remote_meta, sizeof(remote_meta), 0) <= 0)
            perror("recv() timeout");

        // Send local metadata
        if (send(sock, &local_meta, sizeof(local_meta), 0) <= 0)
            perror("send() failed");

        printf("QPN and GID exchanged for device idx %d\n", devs[i]);

        // This is the key to enabling RDMA read/write over SRD.
        ah_list.push_back(create_ah(pd_list_[i], remote_meta.gid));
        qpn_list.push_back(remote_meta.qpn);
    }

    close(sock);
    
    return true;
}

int RDMAEndpoint::uccl_regmr(std::vector<int> devs, void* addr, size_t len,
                            int type, uint64_t mr_id, 
                            struct Mhandle** mhandle) {
    std::vector<struct ibv_mr*> mr_list;
    for (int i = 0; i < devs.size(); i++) {
        struct ibv_mr* mr = ibv_reg_mr(
            pd_list_[i],
            addr,
            len,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                IBV_ACCESS_REMOTE_WRITE);

        if (!mr) {
            fprintf(stderr, "Failed to register memory region\n");
            for (auto m : mr_list) {
                ibv_dereg_mr(m);
            }
            return -1;
        }
        mr_list.push_back(mr);
    }
    mr_map_[mr_id] = mr_list;
    
    // Create a single mhandle with the mr_id
    *mhandle = new Mhandle();
    (*mhandle)->mr_id = mr_id;
    return 0;
}

void RDMAEndpoint::uccl_deregmr(struct Mhandle* mhandle) {
    if (mhandle) {
        // Find and deregister all MRs associated with this mr_id
        auto it = mr_map_.find(mhandle->mr_id);
        if (it != mr_map_.end()) {
            for (auto mr : it->second) {
                if (mr) {
                    ibv_dereg_mr(mr);
                }
            }
            mr_map_.erase(it);
        }
        delete mhandle;
    }
}

int RDMAEndpoint::prepare_fifo_metadata(uint64_t conn_id,
                                        uint64_t mr_id, void const* data,
                                        size_t size, char* out_buf) {
    uccl::FifoItem item{};

    // store multiple rkeys in the padding
    const size_t padding_slots = sizeof(item.padding) / sizeof(uint32_t);
    if (mr_map_[mr_id].size() > padding_slots + 1) {
        printf("prepare_fifo_metadata error: too many MRs for mr_id=%llu\n",
               static_cast<unsigned long long>(mr_id));
        return -1;
    }
    memset(item.padding, 0, sizeof(item.padding));
    auto* padding_rkeys = reinterpret_cast<uint32_t*>(item.padding);

    for (size_t i = 0; i < mr_map_[mr_id].size(); ++i) {
        if (i == 0) {
            item.addr = reinterpret_cast<uint64_t>(data);
            item.size = size;
            item.rkey = mr_map_[mr_id][i]->rkey;
            item.nmsgs = 1;
            item.rid = 0;  // TODO
            item.idx = 0;  // TODO
            item.engine_offset = 0;
        } else {
            padding_rkeys[i - 1] = mr_map_[mr_id][i]->rkey;
        }
    }
    uccl::serialize_fifo_item(item, out_buf);
    return 0;
}

int RDMAEndpoint::uccl_write_async(uint64_t conn_id, Mhandle* local_mh,
                                   int num_devs_per_gpu, void* src, size_t size,
                                   uccl::FifoItem const& slot_item,
                                   ucclRequest* ureq) {
    uint64_t dev_local_idx =
        next_dev_local_idx_.fetch_add(1) % num_devs_per_gpu;

    struct ibv_qp* qp = qp_list_[dev_local_idx];
    struct ibv_mr* mr = mr_map_[local_mh->mr_id][dev_local_idx];
    struct ibv_ah* ah = ah_map_[conn_id][dev_local_idx];
    uint32_t remote_qpn = qpn_map_[conn_id][dev_local_idx];
    uint32_t remote_rkey = slot_item.rkey;
    if (dev_local_idx > 0) {
        auto const* padding_rkeys =
            reinterpret_cast<uint32_t const*>(slot_item.padding);
        remote_rkey = padding_rkeys[dev_local_idx - 1];
    }

    struct ibv_qp_ex* qpx = ibv_qp_to_qp_ex(qp);
    ibv_wr_start(qpx);
    qpx->wr_id = reinterpret_cast<uint64_t>(ureq);
    qpx->comp_mask = 0;
    qpx->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_rdma_write(qpx, remote_rkey, slot_item.addr);

    uint32_t data_len =
        std::min<uint32_t>(static_cast<uint32_t>(size), slot_item.size);
    struct ibv_sge sge {
        reinterpret_cast<uint64_t>(src), data_len, mr->lkey
    };
    ibv_wr_set_sge_list(qpx, 1, &sge);
    ibv_wr_set_ud_addr(qpx, ah, remote_qpn, QKEY);

    if (ibv_wr_complete(qpx)) {
        printf("ibv_wr_complete failed\n");
    }

    ureq->type = ReqTx;
    ureq->n = 1;
    ureq->send_len = static_cast<int>(data_len);
    ureq->dev_local_idx = dev_local_idx;

    return 0;
}

bool RDMAEndpoint::uccl_poll_ureq_once(struct ucclRequest* ureq) {
    uint64_t dev_local_idx = ureq->dev_local_idx;

    auto* cq = ibv_cq_ex_to_cq(cq_ex_list_[dev_local_idx]);
    struct ibv_wc wc{};
    while (ibv_poll_cq(cq, 1, &wc) > 0) {
        if (wc.status != IBV_WC_SUCCESS) {
            fprintf(
                stderr, "uccl_poll_ureq_once: wc.status != IBV_WC_SUCCESS\n");
        }
        completed_req_set_list_[dev_local_idx].insert(wc.wr_id);
    }

    uint64_t wr_id = reinterpret_cast<uint64_t>(ureq);
    if (completed_req_set_list_[dev_local_idx].find(wr_id) != 
        completed_req_set_list_[dev_local_idx].end()) {
        completed_req_set_list_[dev_local_idx].erase(wr_id);
        return true;
    }
    return false;
}

}
