#pragma once

#include "backend.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace UKernel {
namespace Transport {
class RdmaTransportAdapter;
struct ShmExchanger;
}
namespace CCL {

struct RdmaLocalCopyBackendConfig {
  int gpu_id = 0;
  int rank = 0;
  int peer_rank = 1;
};

struct RemoteBufInfo {
  uint32_t rkey;
  uint64_t addr;
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

 private:
  struct OpBufInfo {
    uint32_t src_buf_id;
    uint32_t dst_buf_id;
  };

  bool is_degraded() const;
  bool init_oob();
  bool exchange_mr_and_qp();

  RdmaLocalCopyBackendConfig config_;
  bool degraded_ = false;
  bool validated_ = false;

  int active_peer_ = 0;
  static constexpr int kOobRank = 0;

  std::unique_ptr<UKernel::Transport::RdmaTransportAdapter> adapter_;
  std::unique_ptr<UKernel::Transport::ShmExchanger> oob_;

  std::unordered_map<uint32_t, RemoteBufInfo> remote_bufs_;

  uint64_t next_token_ = 1;
  std::mutex mu_;
  std::unordered_map<uint64_t, unsigned> token_to_req_;
  std::unordered_map<unsigned, uint64_t> req_to_token_;

  std::vector<OpBufInfo> buf_id_cache_;

  static bool oob_pid_leader_;
};

}  // namespace CCL
}  // namespace UKernel
