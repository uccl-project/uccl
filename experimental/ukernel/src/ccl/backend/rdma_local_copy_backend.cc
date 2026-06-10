#include "rdma_local_copy_backend.h"
#include "../../transport/adapter/rdma_adapter.h"
#include "../../transport/adapter/transport_adapter.h"
#include "../coll_types.h"
#include "../lower.h"
#include "../utils.h"
#include <cstring>
#include <stdexcept>
#include <thread>

namespace UKernel {
namespace CCL {

RdmaLocalCopyBackend::RdmaLocalCopyBackend(
    RdmaLocalCopyBackendConfig const& config)
    : config_(config) {
  try {
    UKernel::Transport::RdmaTransportConfig acfg;

    adapter_ = std::make_unique<UKernel::Transport::RdmaTransportAdapter>(
        config_.gpu_id, acfg);

    // Fallback: if the given GPU has no nearby active NIC, try GPU 0
    if (!adapter_->is_initialized()) {
      adapter_.reset();
      adapter_ = std::make_unique<UKernel::Transport::RdmaTransportAdapter>(
          0, acfg);
    }

    if (!adapter_->is_initialized()) {
      adapter_.reset();
      degraded_ = true;
      return;
    }
  } catch (...) {
    adapter_.reset();
    degraded_ = true;
    return;
  }
}

RdmaLocalCopyBackend::~RdmaLocalCopyBackend() = default;

char const* RdmaLocalCopyBackend::name() const {
  return degraded_ ? "degraded" : "rdma-local-copy";
}

bool RdmaLocalCopyBackend::is_degraded() const {
  return degraded_ || adapter_ == nullptr;
}

bool RdmaLocalCopyBackend::supports(OpKind kind) const {
  return !is_degraded() && kind == OpKind::Copy;
}

void RdmaLocalCopyBackend::register_remote_buffer(uint32_t buf_id,
                                                   void const* addr,
                                                   uint32_t rkey) {
  remote_bufs_[buf_id] = {rkey, reinterpret_cast<uint64_t>(addr)};
}

UKernel::Transport::RdmaPeerConnectSpec
RdmaLocalCopyBackend::get_connect_spec() {
  return adapter_->get_connect_init(0);
}

void RdmaLocalCopyBackend::setup_external_peer(
    UKernel::Transport::RdmaPeerConnectSpec const& remote) {
  UKernel::Transport::PeerConnectSpec pcs;
  pcs.peer_rank = 0;
  pcs.type = UKernel::Transport::PeerConnectType::Connect;
  pcs.detail = remote;
  if (!adapter_->ensure_put_path(pcs)) {
    degraded_ = true;
    return;
  }
  active_peer_ = 0;
  external_peer_ = true;
}

void RdmaLocalCopyBackend::setup_external_peer_for_client(
    UKernel::Transport::PeerConnectSpec const& spec) {
  if (!adapter_->ensure_wait_path(spec)) {
    degraded_ = true;
    return;
  }
  active_peer_ = 0;
  external_peer_ = true;
}

bool RdmaLocalCopyBackend::ensure_self_peer() {
  if (self_peer_ready_) return true;

  auto sender_spec = adapter_->get_connect_init(kSelfPeer);
  auto recv_spec = adapter_->get_connect_init(kSelfPeerRecv);

  UKernel::Transport::PeerConnectSpec stc;
  stc.peer_rank = kSelfPeer;
  stc.type = UKernel::Transport::PeerConnectType::Connect;
  stc.detail = recv_spec;

  UKernel::Transport::PeerConnectSpec rts;
  rts.peer_rank = kSelfPeerRecv;
  rts.type = UKernel::Transport::PeerConnectType::Accept;
  rts.detail = sender_spec;

  if (!adapter_->ensure_put_path(stc)) return false;
  if (!adapter_->ensure_wait_path(rts)) return false;

  self_peer_ready_ = true;
  return true;
}

void RdmaLocalCopyBackend::validate(TiledResult const& tiled,
                                    void* input_ptr, void* output_ptr,
                                    void* scratch_ptr) {
  if (is_degraded()) return;

  auto reg = [&](uint32_t id, void* ptr, size_t bytes) {
    if (ptr == nullptr || bytes == 0) return;
    adapter_->register_memory(id, ptr, bytes);
  };

  reg(1, input_ptr, tiled.input_bytes);
  reg(2, output_ptr, tiled.output_bytes);
  reg(3, scratch_ptr, tiled.staging_bytes_required);

  if (!external_peer_ && !ensure_self_peer()) {
    degraded_ = true;
    return;
  }

  for (auto const& [id, info] : remote_bufs_) {
    adapter_->register_remote_buffer(active_peer_, id, info.addr, info.rkey);
  }

  buf_id_cache_.clear();
  buf_id_cache_.reserve(tiled.ops.size());
  for (auto const& op : tiled.ops) {
    OpBufInfo info = {0, 0};
    if (op.kind == OpKind::Copy) {
      info.src_buf_id = static_cast<uint32_t>(
          buf_role(OpKind::Copy, true, op.copy_from_staging)) + 1;
      info.dst_buf_id = static_cast<uint32_t>(
          buf_role(OpKind::Copy, false, op.copy_from_staging)) + 1;
    }
    buf_id_cache_.push_back(info);
  }

  validated_ = true;
}

BackendToken RdmaLocalCopyBackend::submit(Op const& op, OpBindings const& bind,
                                          void* input_ptr, void* output_ptr,
                                          void* scratch_ptr) {
  (void)input_ptr; (void)output_ptr; (void)scratch_ptr;
  if (is_degraded() || !validated_) return BackendToken{0};
  if (op.kind != OpKind::Copy) return BackendToken{0};

  OpBufInfo const& info = buf_id_cache_[bind.stream_index];

  unsigned req_id = adapter_->put_async(
      active_peer_, const_cast<void*>(bind.resolved_src), info.src_buf_id,
      bind.resolved_dst, info.dst_buf_id, op.bytes);

  if (req_id == 0) return BackendToken{0};

  BackendToken token{next_token_++};
  {
    std::lock_guard<std::mutex> lk(mu_);
    token_to_req_[token.value] = req_id;
    req_to_token_[req_id] = token.value;
  }
  return token;
}

size_t RdmaLocalCopyBackend::drain(BackendToken* out, size_t max_count) {
  if (is_degraded()) return 0;

  std::lock_guard<std::mutex> lk(mu_);
  if (token_to_req_.empty()) return 0;

  size_t count = 0;
  auto it = token_to_req_.begin();
  while (it != token_to_req_.end() && count < max_count) {
    unsigned req_id = it->second;
    if (adapter_->poll_completion(req_id)) {
      bool failed = adapter_->request_failed(req_id);
      adapter_->release_request(req_id);
      out[count].value = it->first;
      out[count].failed = failed;
      ++count;
      req_to_token_.erase(req_id);
      it = token_to_req_.erase(it);
    } else {
      ++it;
    }
  }
  return count;
}

}  // namespace CCL
}  // namespace UKernel
