#include "transport_backend.h"
#include "../../../include/transport.h"
#include <cstdio>
#include <stdexcept>
#include <unordered_map>

namespace UKernel {
namespace CCL {

TransportBackend::TransportBackend(UKernel::Transport::Communicator* comm)
    : comm_(comm) {
  if (!comm_) throw std::invalid_argument("TransportBackend: null communicator");
}

bool TransportBackend::supports(OpKind kind) const {
  return kind == OpKind::Send || kind == OpKind::Recv;
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

    unsigned rid = 0;
    if (c.kind == OpKind::Send) {
      rid = comm_->put_async(static_cast<int>(c.dst_peer),
                             c.src_buf, c.src_off,
                             c.dst_buf, c.dst_off, c.bytes);
    } else if (c.kind == OpKind::Recv) {
      rid = comm_->wait_async(static_cast<int>(c.src_peer), 0);
    } else {
      ++accepted; continue;
    }

    if (rid == 0) break;
    uint32_t idx = cmd_next_++;
    if (out_indices) out_indices[accepted] = idx;
    rid_to_cmd_[rid] = idx;
    ++accepted;
  }
  return accepted;
}

size_t TransportBackend::drain(uint32_t* completed, size_t max) {
  unsigned rids[256];
  size_t n = comm_->try_complete(rids, std::min(max, (size_t)256));
  for (size_t i = 0; i < n; ++i) {
    auto it = rid_to_cmd_.find(rids[i]);
    if (it != rid_to_cmd_.end()) {
      completed[i] = it->second;
      rid_to_cmd_.erase(it);
    }
  }
  return n;
}

void TransportBackend::release(uint32_t cmd_idx) {
  for (auto it = rid_to_cmd_.begin(); it != rid_to_cmd_.end(); ++it) {
    if (it->second == cmd_idx) {
      rid_to_cmd_.erase(it);
      return;
    }
  }
}

}  // namespace CCL
}  // namespace UKernel
