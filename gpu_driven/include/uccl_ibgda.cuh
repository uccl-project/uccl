#pragma once
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// Strip CUDA attrs when not compiling with nvcc
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#endif

namespace uccl {

// TODO(MaoZiming): Fix. This should go through the proxy.
template <bool kAlwaysDoPostSend = false>
__device__ __forceinline__ void nvshmemi_ibgda_put_nbi_warp(
    uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_pe, int qp_id,
    int lane_id, int message_idx) {
  // no-op
}

// NOTE(MaoZiming): Remove this. We don't need nvshmem and igbda.
#ifdef false
__device__ __forceinline__ nvshmemi_ibgda_device_state_t* ibgda_get_state() {
  return &nvshmemi_ibgda_device_state_d;
}

__device__ nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d;

struct nvshmemi_ibgda_device_state_t {
  int _unused{0};
  uint32_t num_rc_per_pe{0};
};
#endif

// TODO(MaoZiming): Fix. This should be a non-fetch add operation. This could be
// implemented with CPU proxy.
__device__ __forceinline__ void nvshmemi_ibgda_amo_nonfetch_add(
    void* rptr, int const& value, int pe, int qp_id,
    bool is_local_copy = false) {
  (void)rptr;
  (void)value;
  (void)is_local_copy;
}

#ifdef false
__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(uint64_t const& ptr,
                                                         int const& rank,
                                                         int const& dst_rank) {
  return ptr;
}
#endif
}  // namespace uccl