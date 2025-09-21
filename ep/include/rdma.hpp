#ifndef RDMA_HPP
#define RDMA_HPP
#include "common.hpp"
#include "proxy_ctx.hpp"
#include "ring_buffer.cuh"
#include "unistd.h"
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <atomic>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_set>
#include <vector>

struct RDMAConnectionInfo {
  uint32_t qp_num;  // Queue pair number
  uint32_t psn;     // Packet sequence number
  uint32_t ack_qp_num;
  uint32_t recv_ack_qp_num;
  uint32_t ack_psn;
  uint32_t rkey;   // Memory region key
  uintptr_t addr;  // Buffer address
  uint64_t len;
  uint16_t lid;     // Local ID
  uint8_t gid[16];  // Global ID for RoCE (optional)
};

struct PendingUpdate {
  std::atomic<int>* addr;
  int value;
  uint32_t imm;
  int low_latency_buffer_idx;
  int expert_idx;
  bool is_combine;
  int src_rank;

  // Needed for std::set ordering
  bool operator<(PendingUpdate const& other) const {
    if (addr != other.addr) return addr < other.addr;
    if (value != other.value) return value < other.value;
    return imm < other.imm;
  }
};

class AtomicsImm {
 public:
  // Bit layout:
  // [31]     is_atomics (1 bit)
  // [30]     is_combine (1 bit)
  // [29]     buffer_idx (1 bit)
  // [28:15]  v14 (14 bits, signed, range [-8192, 8191])
  // [14:0]   off15 (15 bits, unsigned, < 32768)
  constexpr static int kOFF = 0;
  constexpr static int kV14 = 15;
  constexpr static int kBUFFER_IDX = 29;
  constexpr static int kIS_COMBINE = 30;
  constexpr static int kIS_ATOMICS = 31;

  constexpr static uint32_t kOFF_MASK = 0x7FFF;  // 15 bits
  constexpr static uint32_t kV14_MASK = 0x3FFF;  // 14 bits

  AtomicsImm(uint32_t imm_data = 0) : imm_data_(imm_data) {}

  static AtomicsImm Pack(bool is_atomics, bool is_combine, int v14,
                         uint16_t off15, int buffer_idx) {
    uint32_t vfield = static_cast<uint32_t>(v14) & kV14_MASK;
    uint32_t imm = (static_cast<uint32_t>(is_atomics) << kIS_ATOMICS) |
                   (static_cast<uint32_t>(is_combine) << kIS_COMBINE) |
                   ((buffer_idx & 0x1u) << kBUFFER_IDX) | (vfield << kV14) |
                   (off15 & kOFF_MASK);
    return AtomicsImm(imm);
  }
  inline bool IsAtomics() const { return (imm_data_ >> kIS_ATOMICS) & 0x1u; }
  inline bool IsCombine() const { return (imm_data_ >> kIS_COMBINE) & 0x1u; }
  inline int GetBufferIdx() const { return (imm_data_ >> kBUFFER_IDX) & 0x1u; }
  inline uint16_t GetOff() const { return imm_data_ & kOFF_MASK; }
  inline int GetValue() const {
    return (static_cast<int32_t>(imm_data_) << 3) >> 18;
  }

  inline void SetAtomics(bool is_atomics) {
    imm_data_ |= (static_cast<uint32_t>(is_atomics) & 0x1u) << kIS_ATOMICS;
  }
  inline void SetCombine(bool is_combine) {
    imm_data_ |= (static_cast<uint32_t>(is_combine) & 0x1u) << kIS_COMBINE;
  }
  inline void SetBufferIdx(uint32_t idx) {
    imm_data_ |= (idx & 0x1u) << kBUFFER_IDX;
  }
  inline void SetOff(uint16_t off) { imm_data_ |= (off & kOFF_MASK); }
  inline void SetV14(int v14) {
    uint32_t vfield = static_cast<uint32_t>(v14) & kV14_MASK;
    imm_data_ |= (vfield << kV14);
  }

  inline uint32_t GetImmData() const { return imm_data_; }

 private:
  uint32_t imm_data_;
};

class WriteImm {
 public:
  // Bit layout:
  // [31]     reserved (0 for now)
  // [30]     is_combine (1 bit)
  // [29]     low_latency_buffer_idx (1 bit)
  // [28:19]  expert_idx (10 bits)
  // [18:12]  num_tokens (7 bits)
  // [11:0]   my_rank (12 bits)
  constexpr static int kRANK = 0;
  constexpr static int kNUM_TOKENS = 12;
  constexpr static int kEXPERT_IDX = 19;
  constexpr static int kBUFFER_IDX = 29;
  constexpr static int kIS_COMBINE = 30;

  constexpr static uint32_t kRANK_MASK = 0xFFF;    // 12 bits
  constexpr static uint32_t kTOKENS_MASK = 0x7F;   // 7 bits
  constexpr static uint32_t kEXPERT_MASK = 0x3FF;  // 10 bits

  WriteImm(uint32_t imm_data = 0) : imm_data_(imm_data) {}

  static WriteImm Pack(bool is_combine, uint32_t buffer_idx,
                       uint32_t expert_idx, uint32_t num_tokens,
                       uint32_t my_rank) {
    uint32_t imm = ((is_combine & 0x1u) << kIS_COMBINE) |
                   ((buffer_idx & 0x1u) << kBUFFER_IDX) |
                   ((expert_idx & kEXPERT_MASK) << kEXPERT_IDX) |
                   ((num_tokens & kTOKENS_MASK) << kNUM_TOKENS) |
                   (my_rank & kRANK_MASK);
    return WriteImm(imm);
  }

  inline bool IsCombine() const { return (imm_data_ >> kIS_COMBINE) & 0x1u; }
  inline uint32_t GetBufferIdx() const {
    return (imm_data_ >> kBUFFER_IDX) & 0x1u;
  }
  inline uint32_t GetExpertIdx() const {
    return (imm_data_ >> kEXPERT_IDX) & kEXPERT_MASK;
  }
  inline uint32_t GetNumTokens() const {
    return (imm_data_ >> kNUM_TOKENS) & kTOKENS_MASK;
  }
  inline uint32_t GetRank() const { return imm_data_ & kRANK_MASK; }

  inline void SetCombine(bool c) {
    imm_data_ =
        (imm_data_ & ~(1u << kIS_COMBINE)) | ((c & 0x1u) << kIS_COMBINE);
  }
  inline void SetBufferIdx(uint32_t idx) {
    imm_data_ =
        (imm_data_ & ~(1u << kBUFFER_IDX)) | ((idx & 0x1u) << kBUFFER_IDX);
  }
  inline void SetExpertIdx(uint32_t expert) {
    imm_data_ = (imm_data_ & ~(kEXPERT_MASK << kEXPERT_IDX)) |
                ((expert & kEXPERT_MASK) << kEXPERT_IDX);
  }
  inline void SetNumTokens(uint32_t tokens) {
    imm_data_ = (imm_data_ & ~(kTOKENS_MASK << kNUM_TOKENS)) |
                ((tokens & kTOKENS_MASK) << kNUM_TOKENS);
  }
  inline void SetRank(uint32_t rank) {
    imm_data_ = (imm_data_ & ~kRANK_MASK) | (rank & kRANK_MASK);
  }

  inline uint32_t GetImmData() const { return imm_data_; }

 private:
  uint32_t imm_data_;
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
                            std::unordered_set<uint64_t>& acked_wrs,
                            std::mutex& finished_wrs_mutex, int thread_idx,
                            std::vector<ProxyCtx*>& ctx_by_tag);
void remote_process_completions(
    ProxyCtx& S, int idx, CopyRingBuffer& ring, int ne, ibv_wc* wc,
    std::vector<ProxyCtx*>& ctx_by_tag, void* atomic_buffer_ptr, int num_ranks,
    int num_experts, std::set<PendingUpdate>& pending_atomic_updates);
void create_per_thread_qp(ProxyCtx& S, void* gpu_buffer, size_t size,
                          RDMAConnectionInfo* local_info, int rank);
ibv_cq* create_per_thread_cq(ProxyCtx& S);
void remote_poll_completions(ProxyCtx& S, int idx, CopyRingBuffer& g_ring,
                             std::vector<ProxyCtx*>& ctx_by_tag,
                             void* atomic_buffer_ptr, int num_ranks,
                             int num_experts,
                             std::set<PendingUpdate>& pending_atomic_updates);
void per_thread_rdma_init(ProxyCtx& S, void* gpu_buf, size_t bytes, int rank,
                          int thread_idx, int local_rank);
void remote_send_ack(ProxyCtx* ctx, struct ibv_qp* ack_qp, uint64_t& wr_id,
                     ibv_mr* local_ack_mr, uint64_t* ack_buf, int worker_idx);
void local_post_ack_buf(ProxyCtx& S, int depth);
void remote_reg_ack_buf(ibv_pd* pd, uint64_t* ack_buf, ibv_mr*& ack_mr);
void post_rdma_async_batched(ProxyCtx& S, void* buf, size_t num_wrs,
                             std::vector<uint64_t> const& wrs_to_post,
                             std::vector<TransferCmd> const& cmds_to_post,
                             std::vector<std::unique_ptr<ProxyCtx>>& ctxs,
                             int my_rank, int thread_idx,
                             std::unordered_set<uint64_t>& finished_wrs,
                             std::mutex& finished_wrs_mutex);
void local_process_completions(ProxyCtx& S,
                               std::unordered_set<uint64_t>& finished_wrs,
                               std::unordered_set<uint64_t>& acked_wrs,
                               std::mutex& finished_wrs_mutex, int thread_idx,
                               ibv_wc* wc, int ne,
                               std::vector<ProxyCtx*>& ctx_by_tag);
void poll_cq_dual(ProxyCtx& S, std::unordered_set<uint64_t>& finished_wrs,
                  std::unordered_set<uint64_t>& acked_wrs,
                  std::mutex& finished_wrs_mutex, int thread_idx,
                  CopyRingBuffer& g_ring, std::vector<ProxyCtx*>& ctx_by_tag,
                  void* atomic_buffer_ptr, int num_ranks, int num_experts,
                  std::set<PendingUpdate>& pending_atomic_updates);
void post_atomic_operations(ProxyCtx& S,
                            std::vector<uint64_t> const& wrs_to_post,
                            std::vector<TransferCmd> const& cmds_to_post,
                            std::vector<std::unique_ptr<ProxyCtx>>& ctxs,
                            int my_rank, int thread_idx,
                            std::unordered_set<uint64_t>& finished_wrs,
                            std::mutex& finished_wrs_mutex,
                            std::unordered_set<uint64_t>& acked_wrs);

void apply_pending_updates(ProxyCtx& ctx,
                           std::set<PendingUpdate>& pending_atomic_updates,
                           void* atomic_buffer_ptr, int num_experts,
                           int num_ranks);
#endif  // RDMA_HPP