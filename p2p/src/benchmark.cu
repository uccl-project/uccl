#include "ring_buffer.hpp"
#include "gpu_kernel.cuh"
#include "proxy.hpp"
#include <thread>
#include <vector>
#include <iostream>

static constexpr size_t DEFAULT_SIZE = 1 << 20; // 1 MiB
static constexpr int    ITERS        = 1000;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./benchmark <rank> <peer_ip> [bytes]\n";
        return 0;
    }
    int   rank     = std::atoi(argv[1]);
    size_t bytes   = (argc >= 4) ? std::stoul(argv[3]) : DEFAULT_SIZE;

    std::cout << "Rank " << rank << " bytes " << bytes << "\n";

    // Allocate buffer on GPU 0 (can parametrize)
    CHECK_CUDA(cudaSetDevice(0));
    void* dev_buf;
    CHECK_CUDA(cudaMalloc(&dev_buf, bytes));

    // Create ring buffer
    cudaStream_t stream{}; CHECK_CUDA(cudaStreamCreate(&stream));
    RingBuffer* rb_dev = create_ring_buffer(stream); // device pointer

    // Spawn proxy thread with CPU‑visible ptr
    void* host_ptr = nullptr;
    CHECK_CUDA(cudaHostGetDevicePointer(&host_ptr, rb_dev, 0)); // Actually returns same dev_ptr, get host via inverse mapping
    RingBuffer* rb_host = reinterpret_cast<RingBuffer*>(rb_dev); // unified mapping → same

    ProxyCtx ctx{rb_host, rank};
    std::thread proxy(proxy_loop, ctx);

    // Launch kernel to push commands
    auto start = std::chrono::high_resolution_clock::now();
    push_kernel<<<1,1,0,stream>>>(rb_dev, dev_buf, bytes, /*dst_rank*/1-rank, /*dst_gpu*/0, ITERS);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double,std::micro>(end-start).count();
    std::cout << "GPU push time " << us/ITERS << " µs per cmd\n";

    proxy.join(); // never returns in this demo
    return 0;
}