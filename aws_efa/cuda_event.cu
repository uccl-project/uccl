#include <cuda_runtime.h>

#include <iostream>

// An empty kernel that does nothing.
__global__ void emptyKernel() {}

int main() {
    const int iterations = 1000;
    float totalTimeMs = 0.0f;
    cudaError_t err;

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        std::cerr << "Error creating start event: " << cudaGetErrorString(err)
                  << std::endl;
        return -1;
    }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        std::cerr << "Error creating stop event: " << cudaGetErrorString(err)
                  << std::endl;
        return -1;
    }

    // Warm up: Launch the kernel once and synchronize.
    emptyKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Run the measurement loop.
    for (int i = 0; i < iterations; i++) {
        // Record the start event.
        cudaEventRecord(start, 0);
        // Launch the empty kernel.
        emptyKernel<<<1, 1>>>();
        // Record the stop event.
        cudaEventRecord(stop, 0);
        // Wait for the stop event to complete.
        cudaEventSynchronize(stop);

        // Measure elapsed time in milliseconds.
        float elapsedMs = 0.0f;
        cudaEventElapsedTime(&elapsedMs, start, stop);
        totalTimeMs += elapsedMs;
    }

    // Calculate average time in milliseconds.
    double averageTimeMs = totalTimeMs / iterations;
    // Convert average time to microseconds.
    double averageTimeUs = averageTimeMs * 1000.0;
    std::cout << "Average empty kernel launch time over " << iterations
              << " iterations: " << averageTimeUs << " microseconds"
              << std::endl;

    // Clean up.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
