#pragma once
//
// UCCLGinResources — the stable resource bundle injected into `UCCLGin` once at
// construction, so the kernel/JIT signature does not churn as the backend grows.
//
// NCCLGin gets everything from nccl_dev_comm/window/qp_idx/sharing_mode; the
// UCCL Rail backend additionally needs the D2H rings, the registered-window base
// (offset origin for `put`), and the atomic buffer base (offset origin for the
// ordered `red_add_rel`). See ep/docs/uccl_gin_plan.md (UCCLGinResources).

#include "../transport/d2h_queue_device.cuh"
#include <cstdint>

namespace uccl_gin {

struct UCCLGinResources {
  // Rail (EFA) transport.
  d2hq::D2HHandle** d2h_queues = nullptr;  // device array of D2H handle pointers
  uint32_t num_queues = 0;

  // Offset origins (single registered symmetric window + atomic buffer).
  uint64_t window_base = 0;        // `put` payload offsets are relative to this
  uint64_t atomic_tail_base = 0;   // `red_add_rel` counter offsets are relative to this
  uint64_t value_staging = 0;      // per-lane staging for put_value (one uint64 per lane)

  // Topology / lane mapping (mirrors NCCLGin's rank info).
  int num_scaleout_ranks = 1;
  int num_scaleup_ranks = 1;
  int scaleout_rank = 0;
  int scaleup_rank = 0;
  uint32_t num_lanes = 1;
};

#if defined(__CUDACC__)
__device__ __forceinline__ uint32_t queue_index_from_hint(
    const UCCLGinResources& resources, int hint) {
  if (resources.num_queues == 0 || resources.num_lanes == 0 ||
      resources.num_queues % resources.num_lanes != 0) {
    __trap();
  }

  // Preserve the original UCCL/EP mapping: logical channels first round-robin
  // across proxy threads, then select a queue local to that proxy. The host
  // resource array is proxy-major, so a direct hint % num_queues would overload
  // the first proxies whenever num_channels is not divisible by num_queues.
  const auto logical_idx =
      static_cast<uint32_t>(hint) % resources.num_queues;
  const auto queues_per_proxy = resources.num_queues / resources.num_lanes;
  const auto proxy_idx = logical_idx % resources.num_lanes;
  const auto queue_in_proxy = logical_idx / resources.num_lanes;
  return proxy_idx * queues_per_proxy + queue_in_proxy;
}
#endif

}  // namespace uccl_gin
