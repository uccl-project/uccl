#ifndef RDMA_HPP
#define RDMA_HPP

#include <infiniband/verbs.h>
#include "unistd.h"

// Global RDMA resources
extern struct ibv_context* context;
extern struct ibv_pd* pd;
extern struct ibv_qp* qp;
extern struct ibv_mr* mr;
extern uint32_t rkey;
extern uintptr_t remote_addr;
extern uint32_t remote_rkey;

struct RDMAConnectionInfo {
    uint32_t qp_num;         // Queue pair number
    uint32_t psn;            // Packet sequence number
    uint32_t rkey;           // Memory region key
    uintptr_t addr;          // Buffer address
    uint16_t lid;            // Local ID
    uint8_t gid[16];         // Global ID for RoCE (optional)
};

// Setup RDMA resources (register GPU memory, create QP, etc.)
void setup_rdma(void* gpu_buffer, size_t size);

// Post an RDMA write
void rdma_write_stub(int dst_rank, void* local_dev_ptr, size_t bytes);

bool GdrSupportInitOnce();

void exchange_connection_info(int rank, const char* peer_ip, RDMAConnectionInfo* local, RDMAConnectionInfo* remote);

#endif // RDMA_HPP