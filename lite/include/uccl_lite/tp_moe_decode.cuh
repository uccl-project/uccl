#pragma once

#include "tp_moe.cuh"

// Sparse decode specialization. Route-major activations remove the grouped
// problem plan. FP32 dot products round at the same BF16 boundaries as the
// tensor-core path; top-k combination remains ordered FP32 accumulation.
namespace tpmoe {
__device__ inline float warpSum(float x) {
#pragma unroll
  for (int d = 16; d; d /= 2) x += __shfl_down_sync(0xffffffff, x, d);
  return x;
}

__global__ void decodeUp(Bf const* x, Bf const* w, int const* ids, Bf* a,
                         uint32_t* done, unsigned long long* gemmTime, int T,
                         int H, int I, int K) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    done[0] = 0;
    done[1] = 0;
    if (gemmTime) *gemmTime = 0;
  }
  int tiles = (I + 7) / 8, r = blockIdx.x / tiles,
      c = (blockIdx.x % tiles) * 8 + threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  if (c >= I) return;
  int expert = ids[r], token = r / K;
  Bf const *gate = w + ((size_t)expert * 2 * I + c) * H,
           *up = gate + (size_t)I * H;
  float g = 0, u = 0;
  for (int j = lane * 8; j < H; j += 32 * 8) {
    uint4 xv = ((uint4 const*)(x + (size_t)token * H))[j / 8];
    uint4 gv = ((uint4 const*)gate)[j / 8], uv = ((uint4 const*)up)[j / 8];
#pragma unroll
    for (int z = 0; z < 8; ++z) {
      float v = __bfloat162float(((Bf*)&xv)[z]);
      g = fmaf(v, __bfloat162float(((Bf*)&gv)[z]), g);
      u = fmaf(v, __bfloat162float(((Bf*)&uv)[z]), u);
    }
  }
  g = warpSum(g);
  u = warpSum(u);
  if (lane == 0) {
    g = __bfloat162float(__float2bfloat16(g));
    u = __bfloat162float(__float2bfloat16(u));
    a[(size_t)r * I + c] = __float2bfloat16(g / (1.f + expf(-g)) * u);
  }
}

struct DecodeParams {
  Bf const *a, *w, *shared;
  int const* ids;
  float const* weights;
  Bf *buffer, *result;
  uint32_t* done;
  mscclpp::MemoryChannelDeviceHandle intra, cross;
  int rank, T, H, I, K;
  size_t scratch;
  unsigned long long *ticks, *gemmTime;
};

__device__ inline void decodeAwait(uint32_t* ptr, int target) {
  if (threadIdx.x == 0) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> done(*ptr);
    while (done.load(cuda::memory_order_acquire) != (uint32_t)target)
      __nanosleep(64);
  }
  __syncthreads();
}

// One resident communication CTA, plus persistent output-tile workers. The
// three-phase NUMA hierarchy moves only half the output across the socket.
// There is one kernel and one complete collective, with no CPU proxy work.
__global__ void decodeDown(DecodeParams p) {
  int tid = threadIdx.x, tiles = p.T * (p.H / 32);
  if (blockIdx.x == 0) {
    if (tid == 0) {
      if (p.ticks) p.ticks[0] = timestamp();
    }
    size_t half = (size_t)p.T * p.H / 2, bytes = half * sizeof(Bf);
    int own = p.rank & 1;
    Bf* mine = p.buffer + own * half;
    Bf* temp = (Bf*)((char*)p.buffer + p.scratch);
    decodeAwait(p.done + (own ^ 1), tiles / 2);
    if (tid == 0 && p.ticks) p.ticks[1] = timestamp();
    p.intra.put<16, true>(p.scratch, (own ^ 1) * bytes, bytes, tid, blockDim.x);
    __syncthreads();
    if (tid == 0) {
      p.intra.signal();
      p.intra.wait();
      if (p.ticks) p.ticks[2] = timestamp();
    }
    __syncthreads();
    decodeAwait(p.done + own, tiles / 2);
    addVector(mine, temp, half, tid, blockDim.x);
    __syncthreads();
    if (tid == 0 && p.ticks) p.ticks[3] = timestamp();
    p.cross.put<16, true>(p.scratch + bytes, own * bytes, bytes, tid,
                          blockDim.x);
    __syncthreads();
    if (tid == 0) {
      p.cross.signal();
      p.cross.wait();
    }
    __syncthreads();
    addVector(mine, temp + half, half, tid, blockDim.x);
    __syncthreads();
    p.intra.put<16, true>(own * bytes, own * bytes, bytes, tid, blockDim.x);
    __syncthreads();
    if (tid == 0) {
      p.intra.signal();
      p.intra.wait();
      if (p.ticks) p.ticks[4] = timestamp();
    }
    __syncthreads();
    for (int j = tid; j < p.T * (p.H / 8); j += blockDim.x)
      ((uint4*)p.result)[j] = ((uint4*)p.buffer)[j];
    __syncthreads();
    if (tid == 0) {
      p.cross.signal();
      p.cross.wait();
      if (p.ticks) p.ticks[5] = timestamp();
    }
    return;
  }
  int lane = tid % 32, warp = tid / 32;
  for (int task = blockIdx.x - 1; task < tiles; task += gridDim.x - 1) {
    // First produce the half sent to the local peer. Its transfer may overlap
    // computation of the retained half, without splitting expert M tiles.
    int tile = (task + ((p.rank & 1) ^ 1) * (tiles / 2)) % tiles;
    int token = tile / (p.H / 32), col = (tile % (p.H / 32)) * 32 + warp * 4;
    float combined[4] = {};
    for (int k = 0; k < p.K; ++k) {
      int r = token * p.K + k, expert = p.ids[r];
      float dot[4] = {};
      for (int i = lane * 4; i < p.I; i += 32 * 4) {
        uint2 raw = ((uint2 const*)(p.a + (size_t)r * p.I))[i / 4];
#pragma unroll
        for (int z = 0; z < 4; ++z) {
          uint2 w = ((uint2 const*)(p.w + ((size_t)expert * p.H + col + z) *
                                              p.I))[i / 4];
#pragma unroll
          for (int j = 0; j < 4; ++j)
            dot[z] = fmaf(__bfloat162float(((Bf*)&raw)[j]),
                          __bfloat162float(((Bf*)&w)[j]), dot[z]);
        }
      }
#pragma unroll
      for (int z = 0; z < 4; ++z) {
        float sum = warpSum(dot[z]);
        if (lane == 0)
          combined[z] += p.weights[r] * __bfloat162float(__float2bfloat16(sum));
      }
    }
    if (lane == 0) {
      uint2 out;
#pragma unroll
      for (int z = 0; z < 4; ++z)
        ((Bf*)&out)[z] = __float2bfloat16(combined[z]);
      if (p.shared) {
        uint2 sh = ((uint2 const*)(p.shared + (size_t)token * p.H))[col / 4];
#pragma unroll
        for (int z = 0; z < 2; ++z)
          ((__nv_bfloat162*)&out)[z] =
              __hadd2(((__nv_bfloat162*)&out)[z], ((__nv_bfloat162*)&sh)[z]);
      }
      ((uint2*)(p.buffer + (size_t)token * p.H))[col / 4] = out;
      __threadfence();
    }
    __syncthreads();
    if (tid == 0) {
      auto old = atomicAdd(p.done + tile / (tiles / 2), 1u);
      if (p.gemmTime && old + 1 == (uint32_t)(tiles / 2))
        atomicMax(p.gemmTime, timestamp());
    }
  }
}
}  // namespace tpmoe
