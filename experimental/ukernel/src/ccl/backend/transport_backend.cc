#include "transport_backend.h"
#include "../../../include/transport.h"
#include <cstdio>
#include <stdexcept>

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

size_t TransportBackend::enqueue(Cmd const* cmds, size_t n) {
  size_t accepted = 0;
  while (accepted < n && pending_.size() < capacity()) {
    Cmd const& c = cmds[accepted];

    unsigned req = 0;
    if (c.kind == OpKind::Send) {
      req = comm_->isend(static_cast<int>(c.dst_peer),
                         c.src_buf, c.src_off, c.bytes,
                         c.dst_buf, c.dst_off);
    } else if (c.kind == OpKind::Recv) {
      req = comm_->irecv(static_cast<int>(c.src_peer),
                         c.dst_buf, c.dst_off, c.bytes);
    } else {
      ++accepted; continue;
    }

    if (req == 0) break;  // backpressure from communicator
    pending_.push_back({req, cmd_next_++});
    ++accepted;
  }
  return accepted;
}

size_t TransportBackend::drain(uint32_t* completed, size_t max) {
  auto done = comm_->progress();
  size_t count = 0;
  for (auto& [req_id, failed] : done) {
    for (size_t i = 0; i < pending_.size(); ++i) {
      if (pending_[i].req_id == req_id) {
        if (!failed) {
          completed[count++] = pending_[i].cmd_idx;
          if (count >= max) goto finish;
        }
        comm_->release(req_id);
        pending_[i] = pending_.back();
        pending_.pop_back();
        --i;
      }
    }
  }
finish:
  return count;
}

void TransportBackend::release(uint32_t cmd_idx) {
  for (size_t i = 0; i < pending_.size(); ++i) {
    if (pending_[i].cmd_idx == cmd_idx) {
      comm_->release(pending_[i].req_id);
      pending_[i] = pending_.back();
      pending_.pop_back();
      return;
    }
  }
}

}  // namespace CCL
}  // namespace UKernel
