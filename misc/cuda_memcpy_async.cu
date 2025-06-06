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

  return 0;
}
