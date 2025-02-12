#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <vector>

#define NUM_4KB_BLOCKS 32  // Adjust this for different block counts
#define BLOCK_SIZE 4096    // 4KB
#define NUM_STREAMS 4      // Number of CUDA streams for async copy

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void initDataKernel(char **scattered_data, int num_blocks) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_blocks) {
        for (int i = 0; i < BLOCK_SIZE; i++) {
            scattered_data[idx][i] = idx % 256;  // Fill with some pattern
        }
    }
}

__global__ void copyKernel(char **scattered_data, char *continuous_data,
                           int num_blocks) {
    // Each thread copies exactly one byte from scattered_data to
    // continuous_data.
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = num_blocks * BLOCK_SIZE;  // total bytes

    if (globalId < totalSize) {
        // Determine which 4KB block this byte belongs to
        int blockIndex = globalId / BLOCK_SIZE;  // which block
        int offset = globalId % BLOCK_SIZE;      // offset in block

        continuous_data[globalId] = scattered_data[blockIndex][offset];
    }
}

void benchmarkMemcpy() {
    // Allocate scattered 4KB memory blocks on GPU
    char **d_scattered_ptrs;
    checkCudaError(
        cudaMalloc(&d_scattered_ptrs, NUM_4KB_BLOCKS * sizeof(char *)),
        "Allocating scattered pointer array");

    std::vector<char *> d_scattered(NUM_4KB_BLOCKS);
    for (int i = 0; i < NUM_4KB_BLOCKS; i++) {
        checkCudaError(cudaMalloc(&d_scattered[i], BLOCK_SIZE),
                       "Allocating scattered data block");
    }
    checkCudaError(
        cudaMemcpy(d_scattered_ptrs, d_scattered.data(),
                   NUM_4KB_BLOCKS * sizeof(char *), cudaMemcpyHostToDevice),
        "Copying pointer list to device");

    // Allocate continuous GPU buffer
    char *d_continuous;
    checkCudaError(cudaMalloc(&d_continuous, NUM_4KB_BLOCKS * BLOCK_SIZE),
                   "Allocating continuous buffer");

    // Initialize scattered data
    initDataKernel<<<(NUM_4KB_BLOCKS + 255) / 256, 256>>>(d_scattered_ptrs,
                                                          NUM_4KB_BLOCKS);
    cudaDeviceSynchronize();

    // Benchmark cudaMemcpy
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_4KB_BLOCKS; i++) {
        checkCudaError(cudaMemcpy(d_continuous + i * BLOCK_SIZE, d_scattered[i],
                                  BLOCK_SIZE, cudaMemcpyDeviceToDevice),
                       "Memcpy scattered to continuous");
    }

    auto stop_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(
        stop_time - start_time);

    std::cout << "cudaMemcpy performance: "
              << (NUM_4KB_BLOCKS * BLOCK_SIZE) / (elapsed_time.count() * 1e3)
              << " GB/s" << std::endl;

    // Benchmark cudaMemcpyAsync with multiple streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaError(cudaStreamCreate(&streams[i]), "Creating stream");
    }

    start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_4KB_BLOCKS; i++) {
        int stream_id =
            i % NUM_STREAMS;  // Assign each copy operation to a stream
        checkCudaError(
            cudaMemcpyAsync(d_continuous + i * BLOCK_SIZE, d_scattered[i],
                            BLOCK_SIZE, cudaMemcpyDeviceToDevice,
                            streams[stream_id]),
            "MemcpyAsync scattered to continuous");
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    stop_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(
        stop_time - start_time);

    std::cout << "cudaMemcpyAsync performance: "
              << (NUM_4KB_BLOCKS * BLOCK_SIZE) / (elapsed_time.count() * 1e3)
              << " GB/s" << std::endl;

    // --------------------------------------------------
    // Benchmark the GPU kernel copy
    // --------------------------------------------------
    start_time = std::chrono::high_resolution_clock::now();

    // 1D grid to cover all bytes: (NUM_4KB_BLOCKS * BLOCK_SIZE) total
    int totalBytes = NUM_4KB_BLOCKS * BLOCK_SIZE;
    int threadsPerBlock = 256;
    int gridSize = (totalBytes + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel to copy each byte in parallel
    copyKernel<<<gridSize, threadsPerBlock>>>(d_scattered_ptrs, d_continuous,
                                              NUM_4KB_BLOCKS);

    // Wait for the kernel to finish so timing is accurate
    checkCudaError(cudaDeviceSynchronize(), "Synchronize after copyKernel");

    stop_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time_us =
        std::chrono::duration_cast<std::chrono::microseconds>(stop_time -
                                                              start_time);

    double elapsed_time_s =
        static_cast<double>(elapsed_time_us.count()) * 1e-6;  // seconds
    double total_bytes =
        static_cast<double>(NUM_4KB_BLOCKS) * BLOCK_SIZE;  // total bytes
    double bandwidth_gb_s =
        (total_bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed_time_s;  // in GB/s

    std::cout << "Kernel copy performance (" << NUM_4KB_BLOCKS
              << " 4KB pages): " << elapsed_time_us.count() << " us, or "
              << bandwidth_gb_s << " GB/s" << std::endl;

    // Cleanup
    for (int i = 0; i < NUM_4KB_BLOCKS; i++) {
        cudaFree(d_scattered[i]);
    }
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_scattered_ptrs);
    cudaFree(d_continuous);
}

int main() {
    cudaSetDevice(0);
    benchmarkMemcpy();
    return 0;
}
