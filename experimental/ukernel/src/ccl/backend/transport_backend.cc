#include "transport_backend.h"
#include "../../../include/transport.h"
#include "../../transport/oob/oob.h"
#include <cstddef>
#include <cstdio>
#include <stdexcept>

namespace UKernel {
namespace CCL {

static Transport::PeerTransportKind to_peer_transport(PutPath p) {
  switch (p) {
    case PutPath::Ipc:
      return Transport::PeerTransportKind::Ipc;
    case PutPath::Rdma:
      return Transport::PeerTransportKind::Rdma;
    default:
      return Transport::PeerTransportKind::Unknown;
  }
}

TransportBackend::TransportBackend(UKernel::Transport::Communicator* comm) {
  if (!comm) throw std::invalid_argument("TransportBackend: null communicator");
  comm_ = comm;
}

bool TransportBackend::supports(LogicalOpKind kind) const {
  return kind == LogicalOpKind::Put || kind == LogicalOpKind::PutSignal;
}

uint32_t TransportBackend::reserve_slot() {
  return cmd_next_.fetch_add(1, std::memory_order_relaxed) &
         Transport::Communicator::kRidBeIdxMask;
}

bool TransportBackend::do_enqueue_reserved(Cmd const& c, uint32_t be_idx) {
  // Tagged rid: completion paths decode be_idx directly (see
  // Communicator::consume_user_ctx), so submission takes no lock and
  // touches no map. On failure the be_idx is simply skipped — a harmless
  // gap; the executor retries the op through its slot table.
  if (c.kind != LogicalOpKind::Put && c.kind != LogicalOpKind::PutSignal)
    return false;
  unsigned rid = Transport::Communicator::kRidTagTransport | be_idx;
  if (c.kind == LogicalOpKind::PutSignal) {
    // No silent fallback: the executor suppresses the partner Signal
    // only for puts accepted with this flag, so a failed fused
    // submission must fail the op (it is retried next cycle).
    // RDMA fused puts pin to one fixed QP per peer so that immediates
    // arrive in issue order — the receiver matches them per-peer FIFO
    // (kCmdFlagImmWait), which cross-QP reordering would break. IPC
    // ignores the affinity.
    uint32_t const qp_affinity =
        (c.put_path == PutPath::Rdma) ? c.dst_peer : ~0u;
    return comm_->send_put_signal_async_with_rid(
        static_cast<int>(c.dst_peer), c.src_buf, c.src_off, c.dst_buf,
        c.dst_off, c.bytes, to_peer_transport(c.put_path), c.tag, rid,
        qp_affinity);
  }
  return comm_->send_put_async_with_rid(
      static_cast<int>(c.dst_peer), c.src_buf, c.src_off, c.dst_buf, c.dst_off,
      c.bytes, to_peer_transport(c.put_path), rid);
}

size_t TransportBackend::do_enqueue(Cmd const* cmds, size_t n,
                                    uint32_t* out_indices) {
  size_t accepted = 0;
  for (size_t i = 0; i < n; ++i) {
    uint32_t idx = reserve_slot();
    if (!do_enqueue_reserved(cmds[i], idx)) break;
    if (out_indices) out_indices[accepted] = idx;
    ++accepted;
  }
  return accepted;
}

size_t TransportBackend::do_drain(uint32_t* completed, size_t max) {
  UKernel::Transport::CompletionResult results[256];
  size_t n = comm_->try_complete_put(results, std::min(max, (size_t)256));
  size_t out = 0;
  for (size_t i = 0; i < n; ++i) {
    if (results[i].failed) {
      std::fprintf(stderr, "[transport] do_drain: rid %u failed, user_ctx %u\n",
                   results[i].rid, results[i].user_ctx);
    }
    completed[out++] = results[i].user_ctx;
  }
  return out;
}

}  // namespace CCL
}  // namespace UKernel
