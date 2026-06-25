#include "signal_backend.h"
#include "../../../include/transport.h"
#include <algorithm>
#include <cstddef>

namespace UKernel {
namespace CCL {

static_assert(offsetof(CmdWithId, cmd) == 0,
              "Cmd must be first field of CmdWithId for caller_id extraction");

bool SignalBackend::supports(OpKind kind) const {
  return kind == OpKind::Signal || kind == OpKind::WaitSignal;
}

void SignalBackend::init(BufSpec[3]) {
  // signals don't need buffer registration
}

size_t SignalBackend::enqueue(Cmd const* cmds, size_t n,
                               uint32_t* out_indices) {
  size_t accepted = 0;
  for (size_t i = 0; i < n; ++i) {
    Cmd const& c = cmds[i];

    // Cmd is embedded as first field in CmdWithId; extract caller_id
    uint32_t caller_id = reinterpret_cast<CmdWithId const*>(&c)->caller_id;

    unsigned rid = 0;
    switch (c.kind) {
      case OpKind::Signal: {
        auto tpt = static_cast<Transport::PeerTransportKind>(c.transport);
        rid =
            comm_->send_signal_async(static_cast<int>(c.dst_peer), c.tag, tpt);
        break;
      }
      case OpKind::WaitSignal: {
        rid = comm_->wait_signal_async(static_cast<int>(c.src_peer), c.tag,
                                       Transport::PeerTransportKind::Unknown);
        break;
      }
      default:
        ++accepted;
        continue;
    }

    if (rid == 0) break;
    uint32_t idx = cmd_next_++;
    if (out_indices) out_indices[accepted] = idx;
    if (c.kind == OpKind::WaitSignal) {
      signal_wait_rid_to_cmd_[rid] = idx;
      signal_wait_rid_to_caller_[rid] = caller_id;
    } else {
      signal_send_rid_to_cmd_[rid] = idx;
      signal_send_rid_to_caller_[rid] = caller_id;
    }
    ++accepted;
  }
  return accepted;
}

size_t SignalBackend::drain(uint32_t* completed, size_t max) {
  size_t out = 0;

  // Channel 1: Drain SignalWait completions from signal_ring_
  {
    UKernel::Transport::SignalCompletion events[256];
    size_t ns =
        comm_->try_complete_signals(events, std::min(max - out, (size_t)256));
    for (size_t i = 0; i < ns; ++i) {
      auto it = signal_wait_rid_to_caller_.find(events[i].rid);
      if (it != signal_wait_rid_to_caller_.end()) {
        completed[out++] = it->second;  // caller_id
        signal_wait_rid_to_cmd_.erase(events[i].rid);
        signal_wait_rid_to_caller_.erase(it);
      }
    }
  }

  // Channel 2: Drain Signal (send) completions from completion_ring_
  if (out < max) {
    UKernel::Transport::CompletionResult results[256];
    size_t nd =
        comm_->try_complete(results, std::min(max - out, (size_t)256));
    for (size_t i = 0; i < nd; ++i) {
      if (results[i].failed) {
        auto it = signal_send_rid_to_cmd_.find(results[i].rid);
        if (it != signal_send_rid_to_cmd_.end()) {
          signal_send_rid_to_caller_.erase(results[i].rid);
          signal_send_rid_to_cmd_.erase(it);
        }
        continue;
      }
      auto it = signal_send_rid_to_caller_.find(results[i].rid);
      if (it != signal_send_rid_to_caller_.end()) {
        completed[out++] = it->second;  // caller_id
        signal_send_rid_to_cmd_.erase(results[i].rid);
        signal_send_rid_to_caller_.erase(it);
      }
    }
  }

  return out;
}

void SignalBackend::release(uint32_t cmd_idx) {
  for (auto it = signal_send_rid_to_cmd_.begin();
       it != signal_send_rid_to_cmd_.end(); ++it) {
    if (it->second == cmd_idx) {
      signal_send_rid_to_caller_.erase(it->first);
      signal_send_rid_to_cmd_.erase(it);
      break;
    }
  }
  for (auto it = signal_wait_rid_to_cmd_.begin();
       it != signal_wait_rid_to_cmd_.end(); ++it) {
    if (it->second == cmd_idx) {
      signal_wait_rid_to_caller_.erase(it->first);
      signal_wait_rid_to_cmd_.erase(it);
      break;
    }
  }
}

}  // namespace CCL
}  // namespace UKernel
