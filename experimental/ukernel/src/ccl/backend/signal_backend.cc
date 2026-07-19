#include "signal_backend.h"
#include "../../../include/transport.h"
#include <algorithm>

namespace UKernel {
namespace CCL {

bool SignalBackend::supports(ExecOpKind kind) const {
  return kind == ExecOpKind::Signal || kind == ExecOpKind::WaitSignal;
}

uint32_t SignalBackend::reserve_slot() {
  std::lock_guard<std::mutex> lk(mu_);
  uint32_t idx = cmd_next_++;
  unsigned rid = comm_->alloc_rid();
  comm_->record_user_ctx(rid, idx);
  reserved_[idx] = rid;
  return idx;
}

bool SignalBackend::do_enqueue_reserved(Cmd const& c, uint32_t be_idx) {
  unsigned rid = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = reserved_.find(be_idx);
    if (it == reserved_.end()) return false;
    rid = it->second;
    reserved_.erase(it);
  }

  bool ok = false;
  if (c.kind == ExecOpKind::Signal) {
    auto tpt = comm_->same_host(static_cast<int>(c.dst_peer))
                   ? Transport::PeerTransportKind::Ipc
                   : Transport::PeerTransportKind::Rdma;
    ok = comm_->send_signal_async_with_rid(
        static_cast<int>(c.dst_peer), c.tag, tpt, rid);
  } else if (c.kind == ExecOpKind::WaitSignal) {
    auto tpt = comm_->same_host(static_cast<int>(c.src_peer))
                   ? Transport::PeerTransportKind::Ipc
                   : Transport::PeerTransportKind::Rdma;
    ok = comm_->wait_signal_async_with_rid(
        static_cast<int>(c.src_peer), c.tag, tpt, rid);
  }

  if (!ok) {
    std::lock_guard<std::mutex> lk(mu_);
    comm_->consume_user_ctx(rid);
    --cmd_next_;
  }
  return ok;
}

size_t SignalBackend::do_enqueue(Cmd const* cmds, size_t n,
                                 uint32_t* out_indices) {
  size_t accepted = 0;
  for (size_t i = 0; i < n; ++i) {
    uint32_t be_idx = reserve_slot();
    if (!do_enqueue_reserved(cmds[i], be_idx)) break;
    if (out_indices) out_indices[accepted] = be_idx;
    ++accepted;
  }
  return accepted;
}

size_t SignalBackend::do_drain(uint32_t* completed, size_t max) {
  // Losers report no work this round instead of blocking — the winner's
  // drain already advances shared state.
  std::unique_lock<std::mutex> lk(drain_mu_, std::try_to_lock);
  if (!lk.owns_lock()) return 0;

  size_t out = 0;

  {
    UKernel::Transport::SignalCompletion events[256];
    size_t ns =
        comm_->try_complete_sig_wait(events, std::min(max - out, (size_t)256));
    for (size_t i = 0; i < ns; ++i)
      completed[out++] = events[i].user_ctx;
  }

  if (out < max) {
    UKernel::Transport::CompletionResult results[256];
    size_t nd =
        comm_->try_complete_sig_send(results, std::min(max - out, (size_t)256));
    for (size_t i = 0; i < nd; ++i) {
      if (results[i].failed) continue;
      completed[out++] = results[i].user_ctx;
    }
  }

  return out;
}

}  // namespace CCL
}  // namespace UKernel
