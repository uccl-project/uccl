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
  unsigned long long rptr_val = static_cast<unsigned long long>(req_rptr);
  unsigned long long lptr_val = static_cast<unsigned long long>(req_lptr);
  unsigned long long bytes_val = static_cast<unsigned long long>(bytes);

  printf(
      "[ibgda_put_nbi_warp] req_rptr: %llu, req_lptr: %llu, bytes: %llu, "
      "dst_pe: %d, qp_id: %d, lane_id: %d, message_idx: %d\n",
      rptr_val, lptr_val, bytes_val, dst_pe, qp_id, lane_id, message_idx);
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
  printf(
      "[ibgda_amo_nonfetch_add] rptr: %p, value: %d, pe: %d, qp_id: %d, "
      "is_local_copy: %d\n",
      rptr, value, pe, qp_id);
  // TODO(MaoZiming): Implement it with a remote atomic operation.
  atomicAdd(reinterpret_cast<int*>(rptr), value);
  __threadfence_system();
}

#ifdef false
__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(uint64_t const& ptr,
                                                         int const& rank,
                                                         int const& dst_rank) {
  return ptr;
}
#endif
}  // namespace uccl