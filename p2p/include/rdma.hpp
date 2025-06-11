#ifndef RDMA_HPP
#define RDMA_HPP

#include "unistd.h"
#include <infiniband/verbs.h>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <vector>

// Global RDMA resources
extern struct ibv_context* context;
extern struct ibv_pd* pd;
extern thread_local struct ibv_qp* qp;
// extern thread_local struct ibv_cq* cq;
extern struct ibv_mr* mr;
extern uint32_t rkey;
extern thread_local uintptr_t remote_addr;
extern thread_local uint32_t remote_rkey;
extern std::atomic<bool> g_progress_run;
// extern thread_local std::unordered_set<uint64_t> finished_wrs;
// extern thread_local std::mutex finished_wrs_mutex;
// extern thread_local std::unordered_set<uint64_t> finished_wrs;

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
void post_rdma_async(void* buf, size_t bytes, uint64_t wr_id, ibv_cq* cq,
                     std::unordered_set<uint64_t>& finished_wrs,
                     std::mutex& finished_wrs_mutex);
void post_rdma_async_chained(void* buf, size_t bytes, size_t num_wrs,
                             std::vector<uint64_t> wrs_to_post, ibv_cq* cq,
                             std::unordered_set<uint64_t>& finished_wrs,
                             std::mutex& finished_wrs_mutex);

bool GdrSupportInitOnce();

void exchange_connection_info(int rank, char const* peer_ip, int tid,
                              RDMAConnectionInfo* local,
                              RDMAConnectionInfo* remote);

void modify_qp_to_rtr(RDMAConnectionInfo* remote);

void modify_qp_to_rts(RDMAConnectionInfo* local_info);

void modify_qp_to_init();
bool check_cq_completion();
void poll_completions(ibv_cq* cq, std::unordered_set<uint64_t>& finished_wrs,
                      std::mutex& finished_wrs_mutex);
void global_rdma_init(void* gpu_buf, size_t bytes, RDMAConnectionInfo* local,
                      int rank);
void create_per_thread_qp(void* gpu_buffer, size_t size,
                          RDMAConnectionInfo* local_info, int rank, ibv_cq* cq);
ibv_cq* create_per_thread_cq();
void per_thread_polling(int thread_idx, struct ibv_cq* per_thread_cq,
                        std::unordered_set<uint64_t>* per_thread_finished_wrs,
                        std::mutex* per_thread_finished_wrs_mutex);
#endif  // RDMA_HPP