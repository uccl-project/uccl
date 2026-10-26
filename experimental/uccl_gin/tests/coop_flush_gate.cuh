#pragma once

#include "../../../thirdparty/nccl-ep/adapter/uccl_gin_net.cuh"

// Network source-reuse gate: eight complete warp groups publish disjoint
// payloads with independent receiver completion slots. Run with a real Context.
inline constexpr int kCoopFlushGateThreads = 8 * 32;
inline constexpr uint32_t kCoopFlushGateTail = 256;

__global__ void uccl_gin_coop_flush_gate(uccl_gin::UCCLGinResources resources,
                                         int peer, int* send, int* recv,
                                         uint32_t bytes) {
  int const producer = threadIdx.x;
  uint32_t const chunk = bytes / kCoopFlushGateThreads;
  uint32_t const offset = producer * chunk / sizeof(int);
  uccl_gin::UCCLGin gin(resources);
  __nanosleep((producer % 7) * 64);
  gin.put_tail_add<ncclTeamTagRail>(
      recv + offset, send + offset, chunk, peer, 1,
      kCoopFlushGateTail + producer * sizeof(int64_t), producer);
  nccl_ep_adapter::UcclGinNet(resources, producer)
      .flush(ncclCoopWarp(), cuda::memory_order_acquire);
  for (uint32_t i = 0; i < chunk / sizeof(int); ++i)
    send[offset + i] = 0xa5a5a5a5u;
}
