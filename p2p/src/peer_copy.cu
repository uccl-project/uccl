// peer_copy.cu
#include "peer_copy.cuh"
#include <cstdio>
#include <cuda_runtime.h>

__global__ void peer_copy_kernel(char const* __restrict__ src,
                                 char* __restrict__ dst, size_t num_bytes) {
  size_t idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total_threads = (gridDim.x * gridDim.y) * blockDim.x;

  for (size_t i = idx; i < num_bytes; i += total_threads) {
    dst[i] = src[i];
  }
}

static inline cudaError_t enable_peer_access(int from_dev, int to_dev) {
  int can_access = 0;
  cudaDeviceCanAccessPeer(&can_access, from_dev, to_dev);
  if (!can_access) return cudaErrorPeerAccessUnsupported;

  cudaSetDevice(from_dev);
  cudaError_t err = cudaDeviceEnablePeerAccess(to_dev, 0);
  if (err == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();  // Clear sticky error
    return cudaSuccess;
  }
  return err;
}

cudaError_t launch_peer_bulk_copy(void* dst_ptr, int dst_dev, void* src_ptr,
                                  int src_dev, size_t bytes,
                                  cudaStream_t stream) {
  // Enable peer access in both directions
  cudaError_t err;
  if ((err = enable_peer_access(src_dev, dst_dev)) != cudaSuccess) return err;
  if ((err = enable_peer_access(dst_dev, src_dev)) != cudaSuccess) return err;

  // Launch kernel on the source device
  cudaSetDevice(src_dev);

  constexpr int threads_per_block = 256;
  size_t total_threads = (bytes + threads_per_block - 1) / threads_per_block;
  dim3 blocks;
  blocks.x = (total_threads > 65535) ? 65535
                                     : static_cast<unsigned int>(total_threads);
  blocks.y = (total_threads + 65534) / 65535;

  peer_copy_kernel<<<blocks, threads_per_block, 0, stream>>>(
      static_cast<char const*>(src_ptr), static_cast<char*>(dst_ptr), bytes);

  return cudaGetLastError();
}
