#pragma once

#include "rdma.hpp"
#include <cstddef>
#include <cstdint>

enum class TransportKind { Verbs, Cxi };

using LocalConnInfo = RDMAConnectionInfo;
using RemoteConnInfo = RDMAConnectionInfo;

struct TransportCompletion {
  uint64_t wr_id = 0;
  uint64_t data = 0;
  bool is_write = false;
  bool is_atomic = false;
  bool is_recv_data = false;
  int status = 0;
};

class EpTransport {
 public:
  virtual ~EpTransport() = default;

  virtual void init(ProxyCtx& ctx) = 0;
  virtual void register_main_buffer(void* ptr, size_t len, int cuda_device) = 0;
  virtual void register_atomic_buffer(void* ptr, size_t len,
                                      int cuda_device) = 0;
  virtual LocalConnInfo local_info() const = 0;
  virtual void connect_peer(int peer, RemoteConnInfo const& remote) = 0;

  virtual void post_write(int dst_rank, uint64_t wr_id, uint64_t local_offset,
                          uint64_t remote_offset, uint32_t bytes,
                          bool low_latency) = 0;

  virtual void post_atomic_add(int dst_rank, uint64_t wr_id,
                               uint64_t remote_atomic_offset, int64_t value,
                               bool fence) = 0;

  virtual int poll(TransportCompletion* out, int max) = 0;
  virtual void destroy() = 0;
};
