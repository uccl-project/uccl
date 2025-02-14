#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

int main() {
    const size_t size = 4096;  // 4KB copy
    const int iterations = 1000;

    // Allocate memory on the device.
    void* d_buf;
    cudaError_t err = cudaMalloc(&d_buf, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc error: " << cudaGetErrorString(err)
                  << std::endl;
        return 1;
    }

    // Allocate pinned host memory.
    void* h_buf;
    err = cudaMallocHost(&h_buf, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocHost error: " << cudaGetErrorString(err)
                  << std::endl;
        cudaFree(d_buf);
        return 1;
    }

    // Warm-up: perform one copy to eliminate any one-time overhead.
    cudaMemcpy(h_buf, d_buf, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Measure the cudaMemcpy time over many iterations.
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        // Copy from device to host.
        cudaMemcpy(h_buf, d_buf, size, cudaMemcpyDeviceToHost);
        // Synchronize to ensure the copy is complete before measuring time.
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time.
    auto total_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    double average_time_ns = static_cast<double>(total_ns) / iterations;
    std::cout << "Average cudaMemcpy time: " << average_time_ns << " ns"
              << std::endl;

    // Clean up.
    cudaFree(d_buf);
    cudaFreeHost(h_buf);

    return 0;
}
