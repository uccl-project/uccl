#include "signal_backend.h"
#include "../../../include/transport.h"
#include <algorithm>

namespace UKernel {
namespace CCL {

bool SignalBackend::supports(ExecOpKind kind) const {
  return kind == ExecOpKind::Signal || kind == ExecOpKind::WaitSignal;
}

size_t SignalBackend::do_enqueue(Cmd const* cmds, size_t n,
                                 uint32_t* out_indices) {
  size_t accepted = 0;
  for (size_t i = 0; i < n; ++i) {
    Cmd const& c = cmds[i];

    uint32_t idx = cmd_next_++;

    unsigned rid = 0;
    {
      std::lock_guard<std::mutex> lk(mu_);
      switch (c.kind) {
        case ExecOpKind::Signal: {
          auto tpt = comm_->same_host(static_cast<int>(c.dst_peer))
                         ? Transport::PeerTransportKind::Ipc
                         : Transport::PeerTransportKind::Rdma;
          rid = comm_->send_signal_async(static_cast<int>(c.dst_peer), c.tag,
                                         tpt);
          if (rid) {
            signal_send_rid_to_cmd_[rid] = idx;
          }
          break;
        }
        case ExecOpKind::WaitSignal: {
          auto tpt = comm_->same_host(static_cast<int>(c.src_peer))
                         ? Transport::PeerTransportKind::Ipc
                         : Transport::PeerTransportKind::Rdma;
          rid = comm_->wait_signal_async(static_cast<int>(c.src_peer), c.tag,
                                         tpt);
          if (rid) {
            signal_wait_rid_to_cmd_[rid] = idx;
          }
          break;
        }
        default:
          break;
      }
    }

    if (rid == 0) {
      --cmd_next_;
      break;
    }
    if (out_indices) out_indices[accepted] = idx;
    ++accepted;
  }
  return accepted;
}

size_t SignalBackend::do_drain(uint32_t* completed, size_t max) {
  size_t out = 0;
  std::lock_guard<std::mutex> lk(mu_);

  // Channel 1: Drain SignalWait completions from signal_ring_
  {
    UKernel::Transport::SignalCompletion events[256];
    size_t ns =
        comm_->try_complete_signals(events, std::min(max - out, (size_t)256));
    for (size_t i = 0; i < ns; ++i) {
      auto it = signal_wait_rid_to_cmd_.find(events[i].rid);
      if (it != signal_wait_rid_to_cmd_.end()) {
        completed[out++] = it->second;  // cmd_idx
        signal_wait_rid_to_cmd_.erase(it);
      }
    }
  }

  // Channel 2: Drain Signal (send) completions from completion_ring_
  if (out < max) {
    UKernel::Transport::CompletionResult results[256];
    size_t nd = comm_->try_complete_signal_send(results, std::min(max - out, (size_t)256));
    for (size_t i = 0; i < nd; ++i) {
      if (results[i].failed) {
        auto it = signal_send_rid_to_cmd_.find(results[i].rid);
        if (it != signal_send_rid_to_cmd_.end()) {
          signal_send_rid_to_cmd_.erase(it);
        }
        continue;
      }
      auto it = signal_send_rid_to_cmd_.find(results[i].rid);
      if (it != signal_send_rid_to_cmd_.end()) {
        completed[out++] = it->second;  // cmd_idx
        signal_send_rid_to_cmd_.erase(it);
      }
    }
  }

  return out;
}

void SignalBackend::release(uint32_t cmd_idx) {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto it = signal_send_rid_to_cmd_.begin();
       it != signal_send_rid_to_cmd_.end(); ++it) {
    if (it->second == cmd_idx) {
      signal_send_rid_to_cmd_.erase(it);
      break;
    }
  }
  for (auto it = signal_wait_rid_to_cmd_.begin();
       it != signal_wait_rid_to_cmd_.end(); ++it) {
    if (it->second == cmd_idx) {
      signal_wait_rid_to_cmd_.erase(it);
      break;
    }
  }
}

}  // namespace CCL
}  // namespace UKernel
