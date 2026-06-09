#pragma once

#include "backend.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <infiniband/verbs.h>

namespace UKernel {
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

 private:
  struct BufInfo {
    uint32_t buffer_id;
    void* ptr;
  };

  struct OpBufInfo {
    uint32_t src_buf_id;
    uint32_t dst_buf_id;
  };

  struct Request {
    uint64_t token;
    uint64_t wr_id;
    std::atomic<bool> done{true};
    std::atomic<bool> failed{false};
  };

  bool is_degraded() const;
  bool init_device();
  bool init_qps();
  void poll_loop();

  RdmaLocalCopyBackendConfig config_;
  bool degraded_ = false;
  bool validated_ = false;

  ibv_context* ctx_ = nullptr;
  ibv_pd* pd_ = nullptr;
  ibv_device_attr dev_attr_ = {};
  uint8_t gid_index_ = 0;
  union ibv_gid gid_ = {};

  ibv_cq* cq_send_ = nullptr;
  ibv_qp* qp_send_ = nullptr;
  ibv_cq* cq_recv_ = nullptr;
  ibv_qp* qp_recv_ = nullptr;

  std::unordered_map<uint32_t, ibv_mr*> mr_map_;

  static constexpr int kMaxReq = 4096;
  Request reqs_[kMaxReq];
  std::atomic<uint32_t> req_head_{0};
  std::atomic<uint32_t> req_tail_{0};

  std::atomic<bool> poll_stop_{false};
  std::unique_ptr<std::thread> poll_thread_;

  uint64_t next_token_ = 1;
  std::mutex mu_;
  std::unordered_map<uint64_t, unsigned> token_to_req_;
  std::unordered_map<unsigned, uint64_t> req_to_token_;

  std::vector<OpBufInfo> buf_id_cache_;

  BufInfo registered_bufs_[3] = {};
  int buf_count_ = 0;
};

}  // namespace CCL
}  // namespace UKernel
