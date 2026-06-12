#pragma once
//
// UcclGinContext — host-side adapter that creates UCCL transport using an
// externally-allocated GPU buffer (e.g., from ncclMemAlloc via NCCL-EP).
//
// Usage (inside ncclEpCreateGroup, after NCCL comm/window are created):
//   UcclGinContext ctx(nccl_buffer_ptr, nccl_buffer_bytes,
//                      rank, world_size, local_world_size, ifname);
//   auto resources = ctx.resources();  // pass to kernel alongside dcomms
//

#include "uccl_gin/resources.cuh"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

class UcclProxy;

namespace nccl_ep_adapter {

class UcclGinContext {
 public:
  struct Config {
    uintptr_t gpu_buffer_addr = 0;
    std::size_t gpu_buffer_bytes = 0;
    int rank = 0;
    int world_size = 1;
    int local_world_size = 8;
    const char* ifname = "enp71s0";
  };

  explicit UcclGinContext(Config cfg);
  ~UcclGinContext();

  UcclGinContext(UcclGinContext const&) = delete;
  UcclGinContext& operator=(UcclGinContext const&) = delete;

  uccl_gin::UCCLGinResources const& resources() const noexcept { return resources_; }
  int num_queues() const noexcept { return num_queues_; }

 private:
  void setup(Config cfg);
  void teardown();

  std::vector<std::unique_ptr<UcclProxy>> proxies_;
  d2hq::D2HHandle** d_handles_ = nullptr;
  int num_queues_ = 0;
  uccl_gin::UCCLGinResources resources_{};
};

}  // namespace nccl_ep_adapter
