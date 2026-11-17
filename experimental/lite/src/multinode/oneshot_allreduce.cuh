#pragma once

namespace uccl_lite {
namespace multinode_experiment {

// Experimental FP32, two-node/four-GPU one-shot reduction. Each rank sends
// its complete input once to its remote partner. Every GPU then pulls all
// eight inputs from its node's four GPU allocations and reduces directly to
// its full output. No reduction precedes IB and no result gather follows it.
//
// Two banks protect receive buffers across back-to-back calls: sending i+2
// requires receiving i+1, which requires the other node to finish reading i.
// The final local barrier protects all local readers, not only the owner.
__global__ void oneShotAllReduce(LocalDirectChans local, PortChan ib,
                                 char* base, size_t count, Layout layout,
                                 int localRank, size_t bankOffset) {
  size_t sendOffset = layout.ibSend + bankOffset;
  size_t receiveOffset = layout.ibScratch + bankOffset;
  auto const* input = reinterpret_cast<float4 const*>(base + layout.data);
  auto* snapshot = reinterpret_cast<float4*>(base + sendOffset);
  for (size_t i = threadIdx.x; i < count / 4; i += blockDim.x)
    snapshot[i] = input[i];
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    ib.putWithSignal(receiveOffset, sendOffset, count * sizeof(float));
    ib.wait();
    ib.flush();
    for (int p = 0; p < kLocalRanks; ++p)
      if (p != localRank) local.c[p].signal();
    for (int p = 0; p < kLocalRanks; ++p)
      if (p != localRank) local.c[p].wait();
  }
  __syncthreads();

  auto* output = reinterpret_cast<float4*>(base + layout.data);
  for (size_t i = threadIdx.x; i < count / 4; i += blockDim.x) {
    float4 sum = make_float4(0, 0, 0, 0);
    for (int half = 0; half < 2; ++half) {
      size_t offset = half ? receiveOffset : sendOffset;
      for (int p = 0; p < kLocalRanks; ++p) {
        char const* source =
            p == localRank ? base
                           : reinterpret_cast<char const*>(local.c[p].dst_);
        float4 value = reinterpret_cast<float4 const*>(source + offset)[i];
        sum.x += value.x;
        sum.y += value.y;
        sum.z += value.z;
        sum.w += value.w;
      }
    }
    output[i] = sum;
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int p = 0; p < kLocalRanks; ++p)
      if (p != localRank) local.c[p].signal();
    for (int p = 0; p < kLocalRanks; ++p)
      if (p != localRank) local.c[p].wait();
  }
}

}  // namespace multinode_experiment
}  // namespace uccl_lite
