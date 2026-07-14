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
    case PutPath::Ipc:   return Transport::PeerTransportKind::Ipc;
    case PutPath::Rdma:  return Transport::PeerTransportKind::Rdma;
    default:             return Transport::PeerTransportKind::Unknown;
  }
}

TransportBackend::TransportBackend(UKernel::Transport::Communicator* comm) {
  if (!comm) throw std::invalid_argument("TransportBackend: null communicator");
  comm_ = comm;
}

bool TransportBackend::supports(ExecOpKind kind) const {
  return kind == ExecOpKind::Put;
}

size_t TransportBackend::do_enqueue(Cmd const* cmds, size_t n,
                                    uint32_t* out_indices) {
  size_t accepted = 0;
  for (size_t i = 0; i < n; ++i) {
    Cmd const& c = cmds[i];

    uint32_t idx;
    unsigned rid;
    {
      std::lock_guard<std::mutex> lk(mu_);
      idx = cmd_next_++;
      rid = comm_->alloc_rid();
      comm_->record_user_ctx(rid, idx);
    }

    bool ok = false;
    if (c.kind == ExecOpKind::Put) {
      ok = comm_->send_put_async_with_rid(
          static_cast<int>(c.dst_peer), c.src_buf,
          c.src_off, c.dst_buf, c.dst_off, c.bytes,
          to_peer_transport(c.put_path), rid);
    }

    if (!ok) {
      std::lock_guard<std::mutex> lk(mu_);
      comm_->consume_user_ctx(rid);
      --cmd_next_;
      break;
    }
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
      std::fprintf(stderr,
                   "[transport] do_drain: rid %u failed, user_ctx %u\n",
                   results[i].rid, results[i].user_ctx);
    }
    completed[out++] = results[i].user_ctx;
  }
  return out;
}

}  // namespace CCL
}  // namespace UKernel
