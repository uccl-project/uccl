#include "rdma_local_copy_backend.h"
#include "../../transport/adapter/rdma_adapter.h"
#include "../../transport/adapter/transport_adapter.h"
#include "../../transport/oob/oob.h"
#include "../../transport/util/utils.h"
#include "../coll_types.h"
#include "../lower.h"
#include "../utils.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace UKernel {
namespace CCL {
template <class F>
inline void visit_fields(RemoteBufInfo& v, F&& f) {
  f("addr", v.addr); f("rkey", v.rkey);
}
template <class F>
inline void visit_fields(RemoteBufInfo const& v, F&& f) {
  f("addr", v.addr); f("rkey", v.rkey);
}
}  // namespace CCL
}  // namespace UKernel

namespace UKernel {
namespace Transport {
template <class F>
inline void visit_fields(RdmaPeerConnectSpec& v, F&& f) {
  f("qpn0", v.remote_data_qpns[0]); f("qpn1", v.remote_data_qpns[1]);
  f("qpn2", v.remote_data_qpns[2]); f("qpn3", v.remote_data_qpns[3]);
  f("sig_qpn", v.remote_signal_qpn); f("num_qps", v.num_qps);
  f("lid", v.remote_lid); f("ldev", v.local_dev_idx); f("lgpu", v.local_gpu_idx);
  f("rdev", v.remote_dev_idx); f("rgpu", v.remote_gpu_idx);
  for (int i = 0; i < 16; ++i) {
    char n[8]; snprintf(n, sizeof(n), "g%d", i); f(n, v.remote_gid_raw[i]);
  }
}
template <class F>
inline void visit_fields(RdmaPeerConnectSpec const& v, F&& f) {
  f("qpn0", v.remote_data_qpns[0]); f("qpn1", v.remote_data_qpns[1]);
  f("qpn2", v.remote_data_qpns[2]); f("qpn3", v.remote_data_qpns[3]);
  f("sig_qpn", v.remote_signal_qpn); f("num_qps", v.num_qps);
  f("lid", v.remote_lid); f("ldev", v.local_dev_idx); f("lgpu", v.local_gpu_idx);
  f("rdev", v.remote_dev_idx); f("rgpu", v.remote_gpu_idx);
  for (int i = 0; i < 16; ++i) {
    char n[8]; snprintf(n, sizeof(n), "g%d", i); f(n, v.remote_gid_raw[i]);
  }
}
}  // namespace Transport
}  // namespace UKernel

namespace UKernel {
namespace CCL {

static constexpr int kOobPort = 19987;
static constexpr int kSelfPeer = 0x40000000;
static constexpr int kSelfPeerRecv = 0x40000001;

RdmaLocalCopyBackend::RdmaLocalCopyBackend(
    RdmaLocalCopyBackendConfig const& config)
    : config_(config) {
  try {
    UKernel::Transport::RdmaTransportConfig acfg;
    adapter_ = std::make_unique<UKernel::Transport::RdmaTransportAdapter>(
        config_.gpu_id, acfg);
    if (!adapter_->is_initialized()) { adapter_.reset(); degraded_ = true; return; }
  } catch (...) { adapter_.reset(); degraded_ = true; return; }

  std::string ns =
      UKernel::Transport::generate_host_id() + ":" + std::to_string(kOobPort);
  try {
    oob_ = std::make_unique<UKernel::Transport::ShmExchanger>(ns, true, 30000);
  } catch (...) { /* OOB optional — only needed for P2P mode */ }
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

void RdmaLocalCopyBackend::validate(TiledResult const& tiled,
                                    void* input_ptr, void* output_ptr,
                                    void* scratch_ptr) {
  if (is_degraded()) return;

  auto reg = [&](uint32_t id, void* ptr, size_t bytes) {
    if (ptr == nullptr || bytes == 0) return;
    if (!adapter_->register_memory(id, ptr, bytes))
      std::fprintf(stderr, "[rdma-be] reg_mr id=%u failed\n", id);
  };
  reg(1, input_ptr, tiled.input_bytes);
  reg(2, output_ptr, tiled.output_bytes);
  reg(3, scratch_ptr, tiled.staging_bytes_required);

  bool is_self = (config_.rank == config_.peer_rank);

  if (is_self) {
    // Self-peer: dst is local, register as remote on sender peer
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

    if (!adapter_->ensure_put_path(stc) || !adapter_->ensure_wait_path(rts)) {
      std::fprintf(stderr, "[rdma-be] self-peer setup failed\n");
      degraded_ = true; return;
    }
    uint32_t rkey = adapter_->get_memory_rkey(2);
    adapter_->register_remote_buffer(kSelfPeer, 2,
        reinterpret_cast<uint64_t>(output_ptr), rkey);
    active_peer_ = kSelfPeer;
  } else {
    // External peer via OOB
    if (!oob_) { degraded_ = true; return; }

    // Export MR
    uint32_t dst_rkey = adapter_->get_memory_rkey(2);
    RemoteBufInfo my_mr = {dst_rkey, reinterpret_cast<uint64_t>(output_ptr)};
    if (!oob_->put("mr:" + std::to_string(config_.rank) + ":2", my_mr)) {
      degraded_ = true; return;
    }

    // Export QP spec
    UKernel::Transport::RdmaPeerConnectSpec my_qp =
        adapter_->get_connect_init(0);
    if (!oob_->put("qp:" + std::to_string(config_.rank), my_qp)) {
      degraded_ = true; return;
    }

    // Wait peer QP
    UKernel::Transport::RdmaPeerConnectSpec peer_qp;
    if (!oob_->wait("qp:" + std::to_string(config_.peer_rank), peer_qp,
                    UKernel::Transport::Exchanger::WaitOptions{-1, 100})) {
      std::fprintf(stderr, "[rdma-be] wait peer QP failed\n");
      degraded_ = true; return;
    }

    // Wait peer MR
    RemoteBufInfo peer_mr;
    if (!oob_->wait("mr:" + std::to_string(config_.peer_rank) + ":2",
                    peer_mr,
                    UKernel::Transport::Exchanger::WaitOptions{-1, 100})) {
      std::fprintf(stderr, "[rdma-be] wait peer MR failed\n");
      degraded_ = true; return;
    }
    adapter_->register_remote_buffer(0, 2, peer_mr.addr, peer_mr.rkey);

    // Connect to peer QP
    UKernel::Transport::PeerConnectSpec pcs;
    pcs.peer_rank = 0;
    pcs.type = UKernel::Transport::PeerConnectType::Connect;
    pcs.detail = peer_qp;
    if (!adapter_->ensure_put_path(pcs)) {
      std::fprintf(stderr, "[rdma-be] ensure_put_path failed\n");
      degraded_ = true; return;
    }
    active_peer_ = 0;
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
