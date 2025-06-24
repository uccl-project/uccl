#include <chrono>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define NUM_4KB_BLOCKS 32      // Adjust this for different block counts
#define BLOCK_SIZE (8 * 1024)  // 8KB
// #define BLOCK_SIZE (1 * 1024 * 1024)  // 1MB
// Number of CUDA streams for async copy: only one is needed for peer copy
#define NUM_STREAMS 1

void checkCudaError(cudaError_t err, char const* msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

__global__ void initDataKernel(char** scattered_data, int num_blocks) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_blocks) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
      scattered_data[idx][i] = idx % 256;  // Fill with some pattern
    }
  }
}

__global__ void copyKernel(char** scattered_data, char* continuous_data,
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
  // --------------------------------------------------
  // Benchmark cudaMemcpy
  // --------------------------------------------------
  // Allocate scattered 4KB memory blocks on GPU
  char** d_scattered_ptrs;
  checkCudaError(cudaMalloc(&d_scattered_ptrs, NUM_4KB_BLOCKS * sizeof(char*)),
                 "Allocating scattered pointer array");

  std::vector<char*> d_scattered(NUM_4KB_BLOCKS);
  for (int i = 0; i < NUM_4KB_BLOCKS; i++) {
    checkCudaError(cudaMalloc(&d_scattered[i], BLOCK_SIZE),
                   "Allocating scattered data block");
  }
  checkCudaError(
      cudaMemcpy(d_scattered_ptrs, d_scattered.data(),
                 NUM_4KB_BLOCKS * sizeof(char*), cudaMemcpyHostToDevice),
      "Copying pointer list to device");

  // Allocate continuous GPU buffer
  char* d_continuous;
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

  // --------------------------------------------------
  // Benchmark cudaMemcpyAsync with multiple streams
  // --------------------------------------------------
  cudaStream_t streams[NUM_STREAMS];
  for (int i = 0; i < NUM_STREAMS; i++) {
    checkCudaError(cudaStreamCreate(&streams[i]), "Creating stream");
  }

  start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < NUM_4KB_BLOCKS; i++) {
    int stream_id = i % NUM_STREAMS;  // Assign each copy operation to a stream
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
  // Benchmark cudaMemcpyPeerAsync between two GPUs
  // --------------------------------------------------
  int srcDevice = 0;
  int dstDevice = 1;
  void* d_dst;
  cudaStream_t peerStreams[NUM_STREAMS];

  // Check if peer access is possible between the GPUs
  int canAccessPeer;
  cudaDeviceCanAccessPeer(&canAccessPeer, dstDevice, srcDevice);
  if (!canAccessPeer) {
    std::cout << "Peer access not available between GPU " << srcDevice
              << " and GPU " << dstDevice << std::endl;
    return;
  }

  // Enable peer access
  checkCudaError(cudaSetDevice(dstDevice), "Set dst device");
  checkCudaError(cudaDeviceEnablePeerAccess(srcDevice, 0),
                 "Enable peer access");
  checkCudaError(cudaMalloc(&d_dst, NUM_4KB_BLOCKS * BLOCK_SIZE),
                 "Allocate dst buffer");

  // Create multiple streams
  for (int i = 0; i < NUM_STREAMS; i++) {
    checkCudaError(cudaStreamCreate(&peerStreams[i]), "Create peer stream");
  }

  start_time = std::chrono::high_resolution_clock::now();

  // Copy data between GPUs using cudaMemcpyPeerAsync across multiple streams
  char* dst_ptr = static_cast<char*>(d_dst);
  for (int i = 0; i < NUM_4KB_BLOCKS; i++) {
    int stream_id = i % NUM_STREAMS;
    checkCudaError(
        cudaMemcpyPeerAsync(dst_ptr + i * BLOCK_SIZE, dstDevice, d_scattered[i],
                            srcDevice, BLOCK_SIZE, peerStreams[stream_id]),
        "MemcpyPeerAsync scattered to continuous");
  }

  // Synchronize all streams
  for (int i = 0; i < NUM_STREAMS; i++) {
    checkCudaError(cudaStreamSynchronize(peerStreams[i]),
                   "Synchronize peer stream");
  }

  stop_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(
      stop_time - start_time);

  std::cout << "cudaMemcpyPeerAsync performance: "
            << (NUM_4KB_BLOCKS * BLOCK_SIZE) / (elapsed_time.count() * 1e3)
            << " GB/s" << std::endl;

  // Cleanup peer resources
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamDestroy(peerStreams[i]);
  }
  cudaFree(d_dst);
  checkCudaError(cudaDeviceDisablePeerAccess(srcDevice), "Disable peer access");
  checkCudaError(cudaSetDevice(srcDevice), "Reset to src device");

  // --------------------------------------------------
  // Benchmark cudaMemcpyAsync + cudaMemcpyPeerAsync between two GPUs
  // --------------------------------------------------

  cudaSetDevice(dstDevice);
  cudaDeviceEnablePeerAccess(srcDevice, 0);
  checkCudaError(cudaMalloc(&d_dst, NUM_4KB_BLOCKS * BLOCK_SIZE),
                 "re-alloc d_dst");
  dst_ptr = static_cast<char*>(d_dst);

  for (int i = 0; i < NUM_STREAMS; i++) {
    checkCudaError(cudaStreamCreate(&peerStreams[i]), "Create peer stream");
  }
  start_time = std::chrono::high_resolution_clock::now();

  cudaSetDevice(srcDevice);
  for (int i = 0; i < NUM_4KB_BLOCKS; i++) {
    int stream_id = i % NUM_STREAMS;
    checkCudaError(
        cudaMemcpyAsync(d_continuous + i * BLOCK_SIZE, d_scattered[i],
                        BLOCK_SIZE, cudaMemcpyDeviceToDevice,
                        peerStreams[stream_id]),
        "Memcpy scattered to continuous");
  }
  cudaSetDevice(dstDevice);
  checkCudaError(
      cudaMemcpyPeerAsync(dst_ptr, dstDevice, d_continuous, srcDevice,
                          BLOCK_SIZE * NUM_4KB_BLOCKS, peerStreams[0]),
      "MemcpyPeerAsync scattered to continuous");

  // Synchronize all streams
  for (int i = 0; i < NUM_STREAMS; i++) {
    checkCudaError(cudaStreamSynchronize(peerStreams[i]),
                   "Synchronize peer stream");
  }

  stop_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(
      stop_time - start_time);

  std::cout << "cudaMemcpyAsync + cudaMemcpyPeerAsync performance: "
            << (NUM_4KB_BLOCKS * BLOCK_SIZE) / (elapsed_time.count() * 1e3)
            << " GB/s" << std::endl;

  // Cleanup peer resources
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamDestroy(peerStreams[i]);
  }
  cudaFree(d_dst);
  checkCudaError(cudaDeviceDisablePeerAccess(srcDevice), "Disable peer access");
  checkCudaError(cudaSetDevice(srcDevice), "Reset to src device");

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
  auto elapsed_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
      stop_time - start_time);

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

// 1MB:
// cudaMemcpy performance: 266.305 GB/s
// cudaMemcpyAsync performance: 92.9486 GB/s
// cudaMemcpyPeerAsync performance: 154.629 GB/s
// Kernel copy performance (32 pages): 300 us, or 104.167 GB/s

// 8KB:
// cudaMemcpy performance: 2.18453 GB/s
// cudaMemcpyAsync performance: 0.470636 GB/s
// cudaMemcpyPeerAsync performance: 2.29951 GB/s
// Kernel copy performance (32 4KB pages): 185 us, or 1.31968 GB/s

int main() {
  cudaSetDevice(0);
  benchmarkMemcpy();
  return 0;
}
