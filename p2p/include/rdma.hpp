#ifndef RDMA_HPP
#define RDMA_HPP

#include "unistd.h"
#include <infiniband/verbs.h>
#include <atomic>

// Global RDMA resources
extern struct ibv_context* context;
extern struct ibv_pd* pd;
extern struct ibv_qp* qp;
extern struct ibv_mr* mr;
extern uint32_t rkey;
extern uintptr_t remote_addr;
extern uint32_t remote_rkey;
extern std::atomic<bool> g_progress_run;

struct RDMAConnectionInfo {
  uint32_t qp_num;  // Queue pair number
  uint32_t psn;     // Packet sequence number
  uint32_t rkey;    // Memory region key
  uintptr_t addr;   // Buffer address
  uint16_t lid;     // Local ID
  uint8_t gid[16];  // Global ID for RoCE (optional)
};

// Setup RDMA resources (register GPU memory, create QP, etc.)
void setup_rdma(void* gpu_buffer, size_t size, RDMAConnectionInfo* local_info,
                int rank);

// Post an RDMA write
void rdma_write_stub(void* local_dev_ptr, size_t bytes);
void post_rdma_async(void* buf, size_t bytes);

bool GdrSupportInitOnce();

void exchange_connection_info(int rank, char const* peer_ip,
                              RDMAConnectionInfo* local,
                              RDMAConnectionInfo* remote);

void modify_qp_to_rtr(RDMAConnectionInfo* remote);

void modify_qp_to_rts(RDMAConnectionInfo* local_info);

void poll_completion();

void modify_qp_to_init();
void progress_thread();
void drain_cq();

#endif  // RDMA_HPP