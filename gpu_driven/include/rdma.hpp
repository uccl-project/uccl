#ifndef RDMA_HPP
#define RDMA_HPP
#include "common.hpp"
#include "proxy_ctx.hpp"
#include "ring_buffer.cuh"
#include "unistd.h"
#include <infiniband/verbs.h>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <vector>

struct RDMAConnectionInfo {
  uint32_t qp_num;  // Queue pair number
  uint32_t psn;     // Packet sequence number
  uint32_t ack_qp_num;
  uint32_t recv_ack_qp_num;
  uint32_t ack_psn;
  uint32_t rkey;    // Memory region key
  uintptr_t addr;   // Buffer address
  uint16_t lid;     // Local ID
  uint8_t gid[16];  // Global ID for RoCE (optional)
};

// Setup RDMA resources (register GPU memory, create QP, etc.)
void setup_rdma(void* gpu_buffer, size_t size, RDMAConnectionInfo* local_info,
                int rank);

// Post an RDMA write
void post_receive_buffer_for_imm(ProxyCtx& S);

void exchange_connection_info(int rank, char const* peer_ip, int tid,
                              RDMAConnectionInfo* local,
                              RDMAConnectionInfo* remote);

void modify_qp_to_rtr(ProxyCtx& S, RDMAConnectionInfo* remote);

void modify_qp_to_rts(ProxyCtx& S, RDMAConnectionInfo* local_info);

void modify_qp_to_init(ProxyCtx& S);
void local_poll_completions(ProxyCtx& S,
                            std::unordered_set<uint64_t>& finished_wrs,
                            std::mutex& finished_wrs_mutex, int thread_idx);
void remote_process_completions(ProxyCtx& S, int idx, CopyRingBuffer& ring,
                                int ne, ibv_wc* wc);
void create_per_thread_qp(ProxyCtx& S, void* gpu_buffer, size_t size,
                          RDMAConnectionInfo* local_info, int rank);
ibv_cq* create_per_thread_cq(ProxyCtx& S);
void remote_poll_completions(ProxyCtx& S, int idx, CopyRingBuffer& g_ring);
void per_thread_rdma_init(ProxyCtx& S, void* gpu_buf, size_t bytes, int rank,
                          int block_idx);
void remote_send_ack(struct ibv_qp* ack_qp, uint64_t& wr_id,
                     ibv_mr* local_ack_mr, uint64_t* ack_buf, int worker_idx);
void local_post_ack_buf(ProxyCtx& S, int depth);
void remote_reg_ack_buf(ibv_pd* pd, uint64_t* ack_buf, ibv_mr*& ack_mr);

void post_rdma_async_batched(ProxyCtx& S, void* buf, size_t bytes,
                             size_t num_wrs, std::vector<uint64_t> wrs_to_post,
                             std::unordered_set<uint64_t>& finished_wrs,
                             std::mutex& finished_wrs_mutex);

void local_process_completions(ProxyCtx& S,
                               std::unordered_set<uint64_t>& finished_wrs,
                               std::mutex& finished_wrs_mutex, int thread_idx,
                               ibv_wc* wc, int ne);
void poll_cq_dual(ProxyCtx& S, std::unordered_set<uint64_t>& finished_wrs,
                  std::mutex& finished_wrs_mutex, int thread_idx,
                  CopyRingBuffer& g_ring);
#endif  // RDMA_HPP