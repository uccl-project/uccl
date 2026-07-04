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

TransportBackend::TransportBackend(UKernel::Transport::Communicator* comm) {
  if (!comm)
    throw std::invalid_argument("TransportBackend: null communicator");
  comm_ = comm;  // base class member
}

bool TransportBackend::supports(ExecOpKind kind) const {
  return kind == ExecOpKind::Put;
}

size_t TransportBackend::do_enqueue(Cmd const* cmds, size_t n,
                                 uint32_t* out_indices) {
  size_t accepted = 0;
  for (size_t i = 0; i < n; ++i) {
    Cmd const& c = cmds[i];
    unsigned rid = 0;
    switch (c.kind) {
      case ExecOpKind::Put: {
        auto tpt = static_cast<Transport::PeerTransportKind>(c.transport);
        rid = comm_->send_put_async(static_cast<int>(c.dst_peer), c.src_buf,
                                    c.src_off, c.dst_buf, c.dst_off, c.bytes,
                                    tpt);
        break;
      }
      default:
        ++accepted;
        continue;
    }
    if (rid == 0) break;
    uint32_t idx = cmd_next_++;
    if (out_indices) out_indices[accepted] = idx;
    rid_to_cmd_[rid] = idx;
    ++accepted;
  }
  return accepted;
}

size_t TransportBackend::do_drain(uint32_t* completed, size_t max) {
  UKernel::Transport::CompletionResult results[256];
  size_t n = comm_->try_complete(results, std::min(max, (size_t)256));
  size_t out = 0;
  for (size_t i = 0; i < n; ++i) {
    // Skip failed completions — they carry no useful data and would
    // otherwise be reported as artificially fast (sub‑μs) latencies.
    if (results[i].failed) {
      auto it = rid_to_cmd_.find(results[i].rid);
      if (it != rid_to_cmd_.end()) rid_to_cmd_.erase(it);
      continue;
    }
    auto it = rid_to_cmd_.find(results[i].rid);
    if (it != rid_to_cmd_.end()) {
      completed[out++] = it->second;
      rid_to_cmd_.erase(it);
    }
  }
  return out;
}

void TransportBackend::release(uint32_t cmd_idx) {
  for (auto it = rid_to_cmd_.begin(); it != rid_to_cmd_.end(); ++it) {
    if (it->second == cmd_idx) {
      rid_to_cmd_.erase(it);
      break;
    }
  }
}

}  // namespace CCL
}  // namespace UKernel
