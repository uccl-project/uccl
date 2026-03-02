#ifndef RDMA_HPP
#define RDMA_HPP
#include "common.hpp"
#include "proxy_ctx.hpp"
// clang-format off
// prevent clang-format reordering net.h before util.h
#include "util/util.h"
#include "util/net.h"
// clang-format on
#include "ring_buffer.cuh"
#include "unistd.h"
#ifdef EFA
#include <infiniband/efadv.h>
#endif
#include <infiniband/verbs.h>
#include <atomic>
#include <cassert>
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

  // Atomic buffer info (separate from main GPU buffer)
  uint32_t atomic_buffer_rkey = 0;   // Atomic buffer memory region key
  uintptr_t atomic_buffer_addr = 0;  // Atomic buffer address
  uint64_t atomic_buffer_len = 0;    // Atomic buffer length

  // #ifdef EFA
  uint32_t num_rings;
  uint32_t data_qp_num[kChannelPerProxy];
  // #endif

#ifdef USE_DMABUF
  // Chunked MR info — exchanged when the GPU buffer is split across
  // multiple MRs (with IOMMU DMA-BUF 2 GiB limit).  num_mr_chunks == 0 means
  // a single MR whose rkey is in the |rkey| field above.
  uint32_t num_mr_chunks = 0;
  struct {
    uint32_t rkey;
    uintptr_t addr;
    uint64_t len;
  } mr_chunk_info[kMaxMRChunks];
#endif
};

struct PendingUpdate {
  std::atomic<int64_t>* addr;
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

struct ImmType {
  // Top-bit definitions
  static constexpr uint32_t kAtomicsBit = 1u << 31;
  static constexpr uint32_t kCtrlBit = 1u << 30;

  static inline bool IsAtomics(uint32_t imm) {
    return (imm & kAtomicsBit) != 0u;
  }
  static inline bool IsBarrier(uint32_t imm) {
    return ((imm & kAtomicsBit) == 0u) && ((imm & kCtrlBit) != 0u);
  }
  static inline bool IsWrite(uint32_t imm) {
    return ((imm & kAtomicsBit) == 0u) && ((imm & kCtrlBit) == 0u);
  }
};

class AtomicsImm {
 public:
  // Bit layout (final):
  // [31]     is_atomics  (1)
  // [30]     is_combine  (1)     // reused as seq[3] in PackAtomicWithSeq
  // [29]     buffer_idx  (1)     // reused as seq[2] in PackAtomicWithSeq
  // [28]     reorderable (1)     // NEW: dedicated bit
  // [27:13]  v15         (15 bits, signed, [-16384, 16383])
  // [12:0]   off13       (13 bits, must be < 8192; low 2 bits carry seq[1:0]
  // when used)
  constexpr static int kOFF = 0;
  constexpr static int kV = 13;  // shift for value field
  constexpr static int kREORDERABLE = 28;
  constexpr static int kBUFFER_IDX = 29;
  constexpr static int kIS_COMBINE = 30;
  constexpr static int kIS_ATOMICS = 31;

  constexpr static uint32_t kOFF_MASK = 0x1FFF;  // 13 bits
  constexpr static int kV_BITS = 15;
  constexpr static uint32_t kV_MASK = (1u << kV_BITS) - 1;  // 0x7FFF

  AtomicsImm(uint32_t imm_data = 0) : imm_data_(imm_data) {}

  inline bool IsAtomics() const { return (imm_data_ >> kIS_ATOMICS) & 0x1u; }
  inline bool IsCombine() const { return (imm_data_ >> kIS_COMBINE) & 0x1u; }
  inline int GetBufferIdx() const { return (imm_data_ >> kBUFFER_IDX) & 0x1u; }
  inline bool IsReorderable() const {
    return (imm_data_ >> kREORDERABLE) & 0x1u;
  }
  inline uint16_t GetOff() const { return imm_data_ & kOFF_MASK; }

  inline int GetValue() const {
    // sign-extend 15 bits located at [27:13]
    int32_t x = static_cast<int32_t>(imm_data_);
    return (x << 4) >> (4 + kV);  // == (x << 4) >> 17
  }

  static AtomicsImm Pack(bool is_atomics, bool is_combine, int v15,
                         uint16_t off13, int buffer_idx,
                         bool reorderable = false) {
    // range checks
    if (v15 < -(1 << (kV_BITS - 1)) || v15 > ((1 << (kV_BITS - 1)) - 1)) {
      fprintf(stderr,
              "[AtomicsImm::Pack] v15 overflow: value=%d (expected in "
              "[-16384,16383])\n",
              v15);
      assert(false && "v15 overflow 15 bits");
    }
    if (off13 > kOFF_MASK) {
      fprintf(
          stderr,
          "[AtomicsImm::Pack] off13 overflow: value=%u (expected <= 8191)\n",
          off13);
      assert(false && "off13 overflow 13 bits");
    }
    if ((buffer_idx & ~0x1) != 0) {
      fprintf(stderr,
              "[AtomicsImm::Pack] buffer_idx overflow: value=%d (mask=0x1)\n",
              buffer_idx);
      assert(false && "buffer_idx overflow");
    }

    uint32_t vfield = static_cast<uint32_t>(v15) & kV_MASK;
    uint32_t imm = 0;
    imm |= (static_cast<uint32_t>(is_atomics) & 0x1u) << kIS_ATOMICS;
    imm |= (static_cast<uint32_t>(is_combine) & 0x1u) << kIS_COMBINE;
    imm |= (static_cast<uint32_t>(buffer_idx) & 0x1u) << kBUFFER_IDX;
    imm |= (static_cast<uint32_t>(reorderable) & 0x1u) << kREORDERABLE;
    imm |= (vfield << kV);
    imm |= (off13 & kOFF_MASK);
    return AtomicsImm(imm);
  }

  static AtomicsImm PackAtomic(int v15, uint16_t off_aligned_bytes) {
    assert((off_aligned_bytes & 0x3u) == 0);
    return Pack(/*is_atomics=*/true, /*is_combine=*/false, v15,
                static_cast<uint16_t>(off_aligned_bytes & kOFF_MASK),
                /*buffer_idx=*/0, /*reorderable=*/false);
  }

  inline uint8_t GetSeq() const {
    // low 2 bits of off13 + bits 29–30 -> 4-bit seq
    uint8_t seq = static_cast<uint8_t>(imm_data_ & 0x3u);
    seq |= static_cast<uint8_t>(((imm_data_ >> kBUFFER_IDX) & 0x1u) << 2);
    seq |= static_cast<uint8_t>(((imm_data_ >> kIS_COMBINE) & 0x1u) << 3);
    return seq;
  }

  static AtomicsImm PackAtomicWithSeq(int v15, uint16_t off_aligned_bytes,
                                      uint8_t seq, bool reorderable = true) {
    assert((off_aligned_bytes & 0x3u) == 0);
    assert((seq % kReorderingBufferSize) == seq);
    uint16_t off13 =
        static_cast<uint16_t>((off_aligned_bytes & 0x1FFFu) | (seq & 0x3u));
    int is_combine_as_seq3 = (seq >> 3) & 0x1;
    int bufidx_as_seq2 = (seq >> 2) & 0x1;
    return Pack(/*is_atomics=*/true, /*is_combine=*/is_combine_as_seq3, v15,
                off13, bufidx_as_seq2, reorderable);
  }

  // (Optional) safer setters that clear target bits before setting:
  inline void SetReorderable(bool r) {
    imm_data_ = (imm_data_ & ~(1u << kREORDERABLE)) |
                (static_cast<uint32_t>(r) << kREORDERABLE);
  }

  inline uint32_t GetImmData() const { return imm_data_; }

 private:
  uint32_t imm_data_;
};

class WriteImm {
 public:
  // Bit positions (adjusted for 128 tokens, 512 experts max)
  static constexpr int kRANK = 0;         // 10 bits (0-1023 ranks)
  static constexpr int kNUM_TOKENS = 10;  // 8 bits (0-255 tokens)
  static constexpr int kEXPERT_IDX = 18;  // 9 bits (0-511 experts)
  static constexpr int kBUFFER_IDX = 27;  // 1 bit
  static constexpr int kIS_COMBINE = 28;  // 1 bit
  // bits 29-31 unused (can use for future expansion)

  // Masks
  static constexpr uint32_t kRANK_MASK = 0x3FF;    // 10 bits
  static constexpr uint32_t kTOKENS_MASK = 0xFF;   // 8 bits
  static constexpr uint32_t kEXPERT_MASK = 0x1FF;  // 9 bits

  explicit WriteImm(uint32_t imm_data = 0) : imm_data_(imm_data) {}

  static WriteImm Pack(bool is_combine,
                       uint32_t buffer_idx,  // 0/1
                       uint32_t expert_idx,  // 0..511 (9 bits)
                       uint32_t num_tokens,  // 0..255 (8 bits)
                       uint32_t my_rank) {   // 0..1023 (10 bits)
    constexpr uint32_t kIS_COMBINE_MASK = 0x1u;
    constexpr uint32_t kBUFFER_IDX_MASK = 0x1u;
    constexpr uint32_t kEXPERT_MASK = 0x1FFu;  // 9 bits
    constexpr uint32_t kTOKENS_MASK = 0xFFu;   // 8 bits
    constexpr uint32_t kRANK_MASK = 0x3FFu;    // 10 bits

    // Runtime validation
    assert((is_combine & ~kIS_COMBINE_MASK) == 0 &&
           "is_combine overflow (1 bit)");
    assert((buffer_idx & ~kBUFFER_IDX_MASK) == 0 &&
           "buffer_idx overflow (1 bit)");
    if ((expert_idx & ~kEXPERT_MASK) != 0) {
      fprintf(stderr,
              "[RDMA ERROR] expert_idx=%u exceeds 9-bit limit (max 511)\n",
              expert_idx);
    }
    assert((expert_idx & ~kEXPERT_MASK) == 0 && "expert_idx overflow (9 bits)");
    if ((num_tokens & ~kTOKENS_MASK) != 0) {
      fprintf(stderr,
              "[RDMA ERROR] num_tokens=%u exceeds 8-bit limit (max 255)\n",
              num_tokens);
    }
    assert((num_tokens & ~kTOKENS_MASK) == 0 && "num_tokens overflow (8 bits)");
    if ((my_rank & ~kRANK_MASK) != 0) {
      fprintf(stderr,
              "[RDMA ERROR] my_rank=%u exceeds 10-bit limit (max 1023)\n",
              my_rank);
    }
    assert((my_rank & ~kRANK_MASK) == 0 && "my_rank overflow (10 bits)");

    uint32_t imm = ((is_combine & 0x1u) << kIS_COMBINE) |
                   ((buffer_idx & 0x1u) << kBUFFER_IDX) |
                   ((expert_idx & kEXPERT_MASK) << kEXPERT_IDX) |
                   ((num_tokens & kTOKENS_MASK) << kNUM_TOKENS) |
                   (my_rank & kRANK_MASK);
    // Top bits [31:30] remain 0 → write type
    return WriteImm(imm);
  }

  // Getters
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

  // Setters
  inline void SetCombine(bool c) {
    imm_data_ = (imm_data_ & ~(1u << kIS_COMBINE)) |
                ((uint32_t(c) & 0x1u) << kIS_COMBINE);
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

struct BarrierImm {
  // [31]=0 (non-atomic), [30]=1 (control), [29]=ACK,
  // [28:8]=SEQ (21 bits), [7:0]=SRC_RANK
  static constexpr uint32_t kCtrlBit = 1u << 30;
  static constexpr uint32_t kAckBit = 1u << 29;
  static constexpr uint32_t kSeqMask = 0x1FFFFFu;
  static inline bool IsAck(uint32_t imm) { return (imm & kAckBit) != 0u; }
  static inline uint32_t Pack(bool ack, uint32_t seq, uint8_t src_rank) {
    return kCtrlBit | (ack ? kAckBit : 0u) |
           ((seq & kSeqMask) << 8)  // 21 bits for seq
           | uint32_t(src_rank);
  }
  static inline uint32_t Seq(uint32_t imm) { return (imm >> 8) & kSeqMask; }
  static inline uint8_t Rank(uint32_t imm) { return imm & 0xFFu; }
  explicit BarrierImm(uint32_t imm = 0) : value(imm) {}
  bool GetIsAck() const { return IsAck(value); }
  uint32_t GetSeq() const { return Seq(value); }
  uint8_t GetRank() const { return Rank(value); }

  uint32_t value;
};

// Setup RDMA resources (register GPU memory, create QP, etc.)
void setup_rdma(void* gpu_buffer, size_t size, RDMAConnectionInfo* local_info,
                int rank);

// Post an RDMA write
void post_receive_buffer_for_imm(ProxyCtx& S);

void recv_connection_info_as_server(int my_rank, int* actual_peer,
                                    int listen_fd,
                                    RDMAConnectionInfo* remote_array);
void send_connection_info_as_client(int my_rank, int peer, char const* peer_ip,
                                    int peer_listen_port,
                                    RDMAConnectionInfo* local);

void modify_qp_to_rtr(ProxyCtx& S, RDMAConnectionInfo* remote,
                      bool use_normal_mode);

void modify_qp_to_rts(ProxyCtx& S, RDMAConnectionInfo* local_info);

void modify_qp_to_init(ProxyCtx& S);
void local_poll_completions(ProxyCtx& S,
                            std::unordered_set<uint64_t>& acked_wrs,
                            int thread_idx, std::vector<ProxyCtx*>& ctx_by_tag);
void remote_process_completions(
    ProxyCtx& S, int idx, CopyRingBuffer& ring, int ne, ibv_wc* wc,
    std::vector<ProxyCtx*>& ctx_by_tag, void* atomic_buffer_ptr, int num_ranks,
    int num_experts, std::set<PendingUpdate>& pending_atomic_updates,
    int my_rank, int num_nodes, bool use_normal_mode = false);
void create_per_thread_qp(ProxyCtx& S, void* gpu_buffer, size_t size,
                          RDMAConnectionInfo* local_info, int rank,
                          size_t num_rings, bool use_normal_mode,
                          void* atomic_buffer_ptr = nullptr);
ibv_cq* create_per_thread_cq(ProxyCtx& S);
void remote_poll_completions(ProxyCtx& S, int idx, CopyRingBuffer& g_ring,
                             std::vector<ProxyCtx*>& ctx_by_tag,
                             void* atomic_buffer_ptr, int num_ranks,
                             int num_experts,
                             std::set<PendingUpdate>& pending_atomic_updates,
                             int my_rank, int num_nodes,
                             bool use_normal_mode = false);
void per_thread_rdma_init(ProxyCtx& S, void* gpu_buf, size_t bytes, int rank,
                          int thread_idx, int local_rank);

// Returns true if a cudaMalloc'd buffer can be registered with ibv_reg_mr on
// this node (e.g. with nvidia_peermem). If false, use host memory for the
// atomic buffer. Result is cached per thread. gpu_idx is the local GPU index.
bool can_register_gpu_memory_for_atomics(int gpu_idx);

#ifdef USE_DMABUF
// Release shared RDMA resources (context/pd/mr) for a given NIC + gpu_buf.
// Must be called before Proxy::destroy() frees ctx.mr / ctx.pd / ctx.context.
// When resources are shared with other threads, nulls the pointers so the
// caller skips the actual ibv_dereg / dealloc / close.
void release_shared_rdma_resources(ProxyCtx& ctx, void* gpu_buf);
// Register GPU memory using DMA-BUF (no nvidia_peermem needed).
// Falls back to ibv_reg_mr_iova2 if DMA-BUF is unsupported.
ibv_mr* reg_mr_gpu_dmabuf(ibv_pd* pd, void* gpu_buf, size_t bytes,
                          uint64_t iova, int access);

// Register GPU memory in multiple DMA-BUF chunks when a single registration
// exceeds the driver limit (with IOMMU ~2 GiB per-MR limit).
std::vector<MRChunk> reg_mr_gpu_dmabuf_chunked(ibv_pd* pd, void* gpu_buf,
                                               size_t bytes, uint64_t iova,
                                               int access,
                                               size_t max_chunk_size);
#endif

void remote_send_ack(ProxyCtx* ctx, struct ibv_qp* ack_qp, uint64_t& wr_id,
                     ibv_mr* local_ack_mr, uint64_t* ack_buf, int worker_idx);
void local_post_ack_buf(ProxyCtx& S, int depth);
void remote_reg_ack_buf(ibv_pd* pd, uint64_t* ack_buf, ibv_mr*& ack_mr);
void post_rdma_async_batched(ProxyCtx& S, void* buf, size_t num_wrs,
                             std::vector<uint64_t> const& wrs_to_post,
                             std::vector<TransferCmd> const& cmds_to_post,
                             std::vector<std::unique_ptr<ProxyCtx>>& ctxs,
                             int my_rank, int thread_idx, bool use_normal_mode);
void local_process_completions(ProxyCtx& S,
                               std::unordered_set<uint64_t>& acked_wrs,
                               int thread_idx, ibv_wc* wc, int ne,
                               std::vector<ProxyCtx*>& ctx_by_tag);
void poll_cq_dual(ProxyCtx& S, std::unordered_set<uint64_t>& acked_wrs,
                  int thread_idx, CopyRingBuffer& g_ring,
                  std::vector<ProxyCtx*>& ctx_by_tag, void* atomic_buffer_ptr,
                  int num_ranks, int num_experts,
                  std::set<PendingUpdate>& pending_atomic_updates, int my_rank,
                  int num_nodes, bool use_normal_mode = false);
void post_atomic_operations(ProxyCtx& S,
                            std::vector<uint64_t> const& wrs_to_post,
                            std::vector<TransferCmd> const& cmds_to_post,
                            std::vector<std::unique_ptr<ProxyCtx>>& ctxs,
                            int my_rank, int thread_idx,
                            std::unordered_set<uint64_t>& acked_wrs,
                            bool use_normal_mode);
void apply_pending_updates(ProxyCtx& ctx,
                           std::set<PendingUpdate>& pending_atomic_updates,
                           void* atomic_buffer_ptr, int num_experts,
                           int num_ranks);
int poll_cq_once(ibv_cq* cq, ibv_wc* wc, int max_cqes);

#endif  // RDMA_HPP
