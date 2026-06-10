#pragma once

#include "backend.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {
class RdmaTransportAdapter;
struct RdmaPeerConnectSpec;
struct PeerConnectSpec;
}
namespace CCL {

struct RdmaLocalCopyBackendConfig {
  int gpu_id = 0;
};

class RdmaLocalCopyBackend final : public Backend {
 public:
  explicit RdmaLocalCopyBackend(RdmaLocalCopyBackendConfig const& config);
  ~RdmaLocalCopyBackend() override;

  char const* name() const override;
  void validate(TiledResult const& tiled,
                void* input_ptr, void* output_ptr,
                void* scratch_ptr) override;
  bool supports(OpKind kind) const override;
  BackendToken submit(Op const& op, OpBindings const& bind,
                      void* input_ptr, void* output_ptr,
                      void* scratch_ptr) override;
  size_t drain(BackendToken* out, size_t max_count) override;

  void register_remote_buffer(uint32_t buf_id, void const* addr, uint32_t rkey);

  UKernel::Transport::RdmaPeerConnectSpec get_connect_spec();
  void setup_external_peer(
      UKernel::Transport::RdmaPeerConnectSpec const& remote);
  void setup_external_peer_for_client(
      UKernel::Transport::PeerConnectSpec const& spec);

 private:
  struct OpBufInfo {
    uint32_t src_buf_id;
    uint32_t dst_buf_id;
  };

  struct RemoteBufInfo {
    uint32_t rkey;
    uint64_t addr;
  };

  bool is_degraded() const;
  bool ensure_self_peer();

  RdmaLocalCopyBackendConfig config_;
  bool degraded_ = false;
  bool validated_ = false;
  bool self_peer_ready_ = false;

  static constexpr int kSelfPeer = 0x40000000;
  static constexpr int kSelfPeerRecv = 0x40000001;
  int active_peer_ = kSelfPeer;
  bool external_peer_ = false;

  std::unique_ptr<UKernel::Transport::RdmaTransportAdapter> adapter_;

  std::unordered_map<uint32_t, RemoteBufInfo> remote_bufs_;

  uint64_t next_token_ = 1;
  std::mutex mu_;
  std::unordered_map<uint64_t, unsigned> token_to_req_;
  std::unordered_map<unsigned, uint64_t> req_to_token_;

  std::vector<OpBufInfo> buf_id_cache_;
};

}  // namespace CCL
}  // namespace UKernel
