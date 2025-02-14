#include <cuda_runtime.h>
#include <stdio.h>

#include <chrono>
#include <iostream>
#include <thread>

// Using volatile __managed__ variables so both CPU and GPU see the latest
// values.
volatile __managed__ int param = 0;   // Parameter written by the CPU.
volatile __managed__ int result = 0;  // Result computed by the GPU.
volatile __managed__ int flag =
    0;  // 0 = idle, 1 = new work available, 2 = work done.
volatile __managed__ int stop =
    0;  // Signal to terminate the persistent kernel.

// Persistent kernel: continuously polls for work until "stop" is set.
__global__ void persistentKernel() {
    // Print a one-time message to indicate the kernel has started.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Persistent kernel started on GPU.\n");
    }
    // The kernel loop: continuously check for new work.
    while (!stop) {
        if (flag == 1) {
            // Compute a simple operation: square the input.
            result = param * param;
            // Ensure that the write to 'result' is visible before updating the
            // flag.
            __threadfence_system();
            flag = 2;
            __threadfence_system();
        }
// Yield a bit to avoid a super-tight spin loop.
// Use __nanosleep if available (supported on compute capability >= 6.0).
#if __CUDA_ARCH__ >= 600
        __nanosleep(100);
#else
        for (volatile int i = 0; i < 1000; i++) {
        }
#endif
    }
}

int main() {
    // Set to GPU 0.
    cudaSetDevice(0);

    persistentKernel<<<1, 1>>>();

    // Launch the persistent kernel with 1 block and 1 thread.
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        persistentKernel<<<1, 1>>>();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double kernel_launch_time_ns =
        std::chrono::duration<double, std::nano>(end - start).count() /
        1000;
    std::cout << "Kernel launch time: " << kernel_launch_time_ns << " ns"
              << std::endl;

    // Check for launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err)
                  << std::endl;
        return 1;
    }

    // Synchronize to allow the kernel to start.
    cudaDeviceSynchronize();

    // Note: The device-side printf from the kernel may not be flushed until the
    // kernel ends. We will rely on host-side prints for progress.

    const int iterations = 1;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        // Write a new parameter.
        param = i;
        // Signal the GPU that new work is available.
        flag = 1;
        // Prefetch the updated flag to device 0 (GPU)
        cudaMemPrefetchAsync((const void*)&flag, sizeof(flag), 0, /*stream*/ 0);

        // Wait until the GPU signals that it has computed the result.
        while (flag != 2) {
            // Sleep a bit on the CPU to reduce busy-wait spinning.
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        // Read back the computed result.
        int res = result;
        // Print every 10 iterations.
        if (i % 10 == 0) {
            printf("Iteration %d: param = %d, result = %d\n", i, i, res);
            fflush(stdout);
        }
        // Reset the flag for the next iteration.
        flag = 0;
        cudaMemPrefetchAsync((const void*)&flag, sizeof(flag), 0, /*stream*/ 0);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time_ms =
        std::chrono::duration<double, std::milli>(total_end - total_start)
            .count();
    std::cout << "Total round-trip time for " << iterations
              << " iterations: " << total_time_ms << " ms" << std::endl;
    std::cout << "Average round-trip latency: " << (total_time_ms / iterations)
              << " ms" << std::endl;

    // Signal the persistent kernel to exit.
    stop = 1;
    cudaDeviceSynchronize();

    return 0;
}
