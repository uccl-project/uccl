#include "transport_backend.h"
#include "../../../include/transport.h"
#include "../../transport/oob/oob.h"
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>

namespace UKernel {
namespace CCL {

static_assert(offsetof(CmdWithId, cmd) == 0,
              "Cmd must be first field of CmdWithId for caller_id extraction");

TransportBackend::TransportBackend(UKernel::Transport::Communicator* comm)
    : comm_(comm) {
  if (!comm_) throw std::invalid_argument("TransportBackend: null communicator");
}

bool TransportBackend::supports(OpKind kind) const {
  return kind == OpKind::Send || kind == OpKind::Recv ||
         kind == OpKind::Signal || kind == OpKind::SignalWait;
}

void TransportBackend::init(BufSpec bufs[3]) {
  for (int i = 0; i < 3; ++i) {
    if (bufs[i].ptr != nullptr && bufs[i].bytes > 0) {
      uint32_t id = static_cast<uint32_t>(i + 1);
      if (!comm_->reg_mr(id, bufs[i].ptr, bufs[i].bytes, true)) {
        std::fprintf(stderr, "[tpt-be] reg_mr id=%u failed\n", id);
      }
      if (!comm_->reg_ipc(id, bufs[i].ptr, bufs[i].bytes, true)) {
        std::fprintf(stderr, "[tpt-be] reg_ipc id=%u failed\n", id);
      }
    }
  }
}

size_t TransportBackend::enqueue(Cmd const* cmds, size_t n,
                                  uint32_t* out_indices) {
  size_t accepted = 0;
  for (size_t i = 0; i < n; ++i) {
    Cmd const& c = cmds[i];

    // Cmd is embedded as first field in CmdWithId; extract caller_id
    uint32_t caller_id = reinterpret_cast<CmdWithId const*>(&c)->caller_id;

    unsigned rid = 0;
    switch (c.kind) {
      case OpKind::Send: {
        auto tpt = static_cast<Transport::PeerTransportKind>(c.transport);
        rid = comm_->send_put_async(static_cast<int>(c.dst_peer),
                                     c.src_buf, c.src_off,
                                     c.dst_buf, c.dst_off, c.bytes, tpt);
        break;
      }
      case OpKind::Recv: {
        rid = comm_->wait_signal_async(static_cast<int>(c.src_peer), c.tag,
                                       Transport::PeerTransportKind::Unknown);
        break;
      }
      case OpKind::Signal: {
        auto tpt = static_cast<Transport::PeerTransportKind>(c.transport);
        rid = comm_->send_signal_async(static_cast<int>(c.dst_peer), c.tag, tpt);
        break;
      }
      case OpKind::SignalWait: {
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
    if (c.kind == OpKind::SignalWait) {
      signal_rid_to_cmd_[rid] = idx;
      signal_rid_to_caller_[rid] = caller_id;
    } else {
      rid_to_cmd_[rid] = idx;
    }
    cmd_transport_[idx] = c.transport;
    ++accepted;
  }
  return accepted;
}

size_t TransportBackend::drain(uint32_t* completed, size_t max) {
  UKernel::Transport::CompletionResult results[256];
  size_t n = comm_->try_complete(results, std::min(max, (size_t)256));
  for (size_t i = 0; i < n; ++i) {
    auto it = rid_to_cmd_.find(results[i].rid);
    if (it != rid_to_cmd_.end()) {
      completed[i] = it->second;
      rid_to_cmd_.erase(it);
    }
  }
  return n;
}

size_t TransportBackend::drain_signals(uint32_t* completed, size_t max) {
  size_t total = 0;
  UKernel::Transport::SignalCompletion events[256];
  size_t ns = comm_->try_complete_signals(events, std::min(max, (size_t)256));
  for (size_t i = 0; i < ns; ++i) {
    auto it = signal_rid_to_caller_.find(events[i].rid);
    if (it != signal_rid_to_caller_.end()) {
      completed[total++] = it->second;
      signal_rid_to_caller_.erase(it);
      signal_rid_to_cmd_.erase(events[i].rid);
    }
  }
  return total;
}

void TransportBackend::release(uint32_t cmd_idx) {
  for (auto it = rid_to_cmd_.begin(); it != rid_to_cmd_.end(); ++it) {
    if (it->second == cmd_idx) {
      rid_to_cmd_.erase(it);
      break;
    }
  }
  for (auto it = signal_rid_to_cmd_.begin();
       it != signal_rid_to_cmd_.end(); ++it) {
    if (it->second == cmd_idx) {
      signal_rid_to_caller_.erase(it->first);
      signal_rid_to_cmd_.erase(it);
      break;
    }
  }
  cmd_transport_.erase(cmd_idx);
}

}  // namespace CCL
}  // namespace UKernel
