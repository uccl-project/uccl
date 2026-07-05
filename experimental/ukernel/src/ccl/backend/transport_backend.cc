#include "transport_backend.h"
#include "../../../include/transport.h"
#include "../../transport/oob/oob.h"
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>

namespace UKernel {
namespace CCL {

TransportBackend::TransportBackend(UKernel::Transport::Communicator* comm) {
  if (!comm) throw std::invalid_argument("TransportBackend: null communicator");
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

    uint32_t idx = cmd_next_++;

    unsigned rid = 0;
    {
      std::lock_guard<std::mutex> lk(mu_);
      switch (c.kind) {
        case ExecOpKind::Put: {
          auto tpt = static_cast<Transport::PeerTransportKind>(c.transport);
          rid = comm_->send_put_async(static_cast<int>(c.dst_peer), c.src_buf,
                                      c.src_off, c.dst_buf, c.dst_off, c.bytes,
                                      tpt);
          if (rid) rid_to_cmd_[rid] = idx;
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

size_t TransportBackend::do_drain(uint32_t* completed, size_t max) {
  UKernel::Transport::CompletionResult results[256];
  size_t n = comm_->try_complete(results, std::min(max, (size_t)256));
  size_t out = 0;
  std::lock_guard<std::mutex> lk(mu_);
  for (size_t i = 0; i < n; ++i) {
    auto it = rid_to_cmd_.find(results[i].rid);
    if (it != rid_to_cmd_.end()) {
      // Return the cmd_idx regardless of success/failure so the caller
      // always sees a completion and can handle the error gracefully
      // instead of spinning forever waiting for a completion that was
      // already consumed.
      completed[out++] = it->second;
      rid_to_cmd_.erase(it);
      if (results[i].failed) {
        std::fprintf(stderr,
                     "[transport] do_drain: rid %u failed, cmd_idx %u\n",
                     results[i].rid, it->second);
      }
    }
  }
  return out;
}

void TransportBackend::release(uint32_t cmd_idx) {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto it = rid_to_cmd_.begin(); it != rid_to_cmd_.end(); ++it) {
    if (it->second == cmd_idx) {
      rid_to_cmd_.erase(it);
      break;
    }
  }
}

}  // namespace CCL
}  // namespace UKernel
