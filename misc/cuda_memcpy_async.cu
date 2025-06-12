#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

// Average async cudaMemcpy submission overhead: 2407.53 ns

int main() {
  const size_t size = 4096;  // 4 KB buffer size.
  int const iterations = 10000;

  // Allocate device memory.
  void* d_buf = nullptr;
  cudaError_t err = cudaMalloc(&d_buf, size);
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc error: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  // Allocate pinned (page-locked) host memory.
  void* h_buf = nullptr;
  err = cudaMallocHost(&h_buf, size);
  if (err != cudaSuccess) {
    std::cerr << "cudaMallocHost error: " << cudaGetErrorString(err)
              << std::endl;
    cudaFree(d_buf);
    return 1;
  }

  // Create a stream for asynchronous copies.
  cudaStream_t stream;
  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) {
    std::cerr << "cudaStreamCreate error: " << cudaGetErrorString(err)
              << std::endl;
    cudaFree(d_buf);
    cudaFreeHost(h_buf);
    return 1;
  }

  // Warm up the GPU/driver to avoid one-time initialization overhead.
  cudaMemcpyAsync(h_buf, d_buf, size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Use std::chrono to measure the CPU overhead of submitting async copies.
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    // Submit an asynchronous copy from device to host.
    cudaMemcpyAsync(h_buf, d_buf, size, cudaMemcpyDeviceToHost, stream);
    // Note: We do not call cudaStreamSynchronize() here so that we only
    // measure the submission cost.
  }
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the total elapsed time in nanoseconds.
  auto total_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  double avg_submission_time_ns = static_cast<double>(total_ns) / iterations;
  std::cout << "Average async cudaMemcpy submission overhead: "
            << avg_submission_time_ns << " ns" << std::endl;

  // Synchronize the stream to complete all pending operations (not included
  // in the timing).
  cudaStreamSynchronize(stream);

  // Cleanup.
  cudaStreamDestroy(stream);
  cudaFree(d_buf);
  cudaFreeHost(h_buf);

  // Measure cudaMemcpyPeerAsync overhead between two GPUs
  int device_count;
  err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count < 2) {
    std::cout << "Need at least 2 GPUs to measure peer copy overhead. Found "
              << device_count << " GPUs." << std::endl;
    return 0;
  }

  // Allocate memory on two different GPUs
  void *buf0 = nullptr, *buf1 = nullptr;
  cudaSetDevice(0);
  err = cudaMalloc(&buf0, size);
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc error on GPU 0: " << cudaGetErrorString(err)
              << std::endl;
    return 1;
  }

  cudaSetDevice(1);
  err = cudaMalloc(&buf1, size);
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc error on GPU 1: " << cudaGetErrorString(err)
              << std::endl;
    cudaFree(buf0);
    return 1;
  }

  // Enable peer access between the GPUs
  cudaSetDevice(0);
  err = cudaDeviceEnablePeerAccess(1, 0);
  if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
    std::cerr << "Failed to enable peer access 0->1: "
              << cudaGetErrorString(err) << std::endl;
    cudaFree(buf0);
    cudaFree(buf1);
    return 1;
  }

  // Create a new stream for peer copies
  cudaStream_t peer_stream;
  err = cudaStreamCreate(&peer_stream);
  if (err != cudaSuccess) {
    std::cerr << "cudaStreamCreate error: " << cudaGetErrorString(err)
              << std::endl;
    cudaFree(buf0);
    cudaFree(buf1);
    return 1;
  }

  // Warm up
  cudaMemcpyPeerAsync(buf1, 1, buf0, 0, size, peer_stream);
  cudaStreamSynchronize(peer_stream);

  // Measure peer copy submission overhead
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    cudaMemcpyPeerAsync(buf1, 1, buf0, 0, size, peer_stream);
  }
  end = std::chrono::high_resolution_clock::now();

  total_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  double avg_peer_submission_time_ns =
      static_cast<double>(total_ns) / iterations;
  std::cout << "Average async cudaMemcpyPeer submission overhead: "
            << avg_peer_submission_time_ns << " ns" << std::endl;

  // Cleanup
  cudaStreamSynchronize(peer_stream);
  cudaStreamDestroy(peer_stream);
  cudaDeviceDisablePeerAccess(1);
  cudaFree(buf0);
  cudaFree(buf1);

  return 0;
}
