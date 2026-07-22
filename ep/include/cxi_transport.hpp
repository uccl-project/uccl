#pragma once

#include "transport.hpp"

#ifdef USE_LIBFABRIC_CXI
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#endif

#include <list>
#include <stdexcept>
#include <vector>

class CxiTransport final : public EpTransport {
 public:
  CxiTransport() = default;
  ~CxiTransport() override;

  void init(ProxyCtx& ctx) override;
  void register_main_buffer(void* ptr, size_t len, int cuda_device) override;
  void register_atomic_buffer(void* ptr, size_t len, int cuda_device) override;
  LocalConnInfo local_info() const override;
  void connect_peer(int peer, RemoteConnInfo const& remote) override;

  void post_write(int dst_rank, uint64_t wr_id, uint64_t local_offset,
                  uint64_t remote_offset, uint32_t bytes,
                  bool low_latency) override;

  void post_atomic_add(int dst_rank, uint64_t wr_id,
                       uint64_t remote_atomic_offset, int64_t value,
                       bool fence) override;
  void post_barrier_atomic_add(int dst_rank, uint64_t wr_id, size_t slot,
                               uint64_t value);
  uint64_t load_barrier_word(size_t slot) const;

  int poll(TransportCompletion* out, int max) override;
  void destroy() override;

 private:
  [[noreturn]] static void unavailable();

#ifdef USE_LIBFABRIC_CXI
  ProxyCtx* ctx_ = nullptr;
  void* main_buffer_ = nullptr;
  void* atomic_buffer_ = nullptr;
  size_t main_buffer_len_ = 0;
  size_t atomic_buffer_len_ = 0;
  fid_fabric* fabric_ = nullptr;
  fid_domain* domain_ = nullptr;
  fid_ep* ep_ = nullptr;
  fid_cq* cq_ = nullptr;
  fid_av* av_ = nullptr;
  fid_mr* main_mr_ = nullptr;
  fid_mr* atomic_mr_ = nullptr;
  fid_mr* atomic_operand_mr_ = nullptr;
  fid_mr* barrier_mr_ = nullptr;
  LocalConnInfo local_info_{};
  std::vector<fi_addr_t> peer_addrs_;
  std::vector<RemoteConnInfo> peer_infos_;
  std::vector<uint64_t> atomic_operands_;
  std::vector<uint8_t> atomic_operand_used_;
  std::vector<uint64_t> barrier_words_;

  struct OpContext {
    fi_context context = {};
    uint64_t wr_id = 0;
    bool is_write = false;
    bool is_atomic = false;
    size_t atomic_operand_slot = static_cast<size_t>(-1);
  };
  std::list<OpContext> op_contexts_;
#endif
};
