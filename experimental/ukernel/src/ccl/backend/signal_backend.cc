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

    unsigned rid = 0;
    uint32_t idx = 0;

    if (c.kind == ExecOpKind::Signal) {
      auto tpt = comm_->same_host(static_cast<int>(c.dst_peer))
                     ? Transport::PeerTransportKind::Ipc
                     : Transport::PeerTransportKind::Rdma;

      // Allocate cmd index under lock, but call send_signal_async
      // outside — it may block on a full IPC signal ring, and holding
      // mu_ across that block would deadlock the drain thread.
      {
        std::lock_guard<std::mutex> lk(mu_);
        idx = cmd_next_++;
      }

      rid = comm_->send_signal_async(static_cast<int>(c.dst_peer), c.tag, tpt);
      if (rid) {
        std::lock_guard<std::mutex> lk(mu_);
        signal_send_rid_to_cmd_[rid] = idx;
      } else {
        std::lock_guard<std::mutex> lk(mu_);
        --cmd_next_;
      }

    } else if (c.kind == ExecOpKind::WaitSignal) {
      auto tpt = comm_->same_host(static_cast<int>(c.src_peer))
                     ? Transport::PeerTransportKind::Ipc
                     : Transport::PeerTransportKind::Rdma;
      {
        std::lock_guard<std::mutex> lk(mu_);
        idx = cmd_next_++;
      }

      rid = comm_->wait_signal_async(static_cast<int>(c.src_peer), c.tag, tpt);
      if (rid) {
        std::lock_guard<std::mutex> lk(mu_);
        signal_wait_rid_to_cmd_[rid] = idx;
      } else {
        std::lock_guard<std::mutex> lk(mu_);
        --cmd_next_;
      }
    }

    if (rid == 0) break;
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
