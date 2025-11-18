#include "memory_allocator.h"
#include <iostream>

// Example usage of MemoryAllocator
int main() {
    try {
        auto& mgr = RdmaDeviceManager::instance();

        auto dev = mgr.getDevice(0);
        if (!dev) {
            std::cerr << "Failed to get device 0" << std::endl;
            return 1;
        }

        auto ctx = std::make_shared<RdmaContext>(dev);

        // Query and display GID
        union ibv_gid gid = ctx->queryGid(0);
        // Create allocator without RDMA support
        MemoryAllocator allocator;

        // Allocate host memory (1MB)
        size_t host_size = 1024 * 1024;
        auto host_mem = allocator.allocate(host_size, MemoryType::HOST, ctx);
        std::cout << "Allocated " << host_mem->size << " bytes of HOST memory at "
                  << host_mem->addr << std::endl;

        // Allocate GPU memory (1MB)
        size_t gpu_size = 1024 * 1024;
        auto gpu_mem = allocator.allocate(gpu_size, MemoryType::GPU, ctx);
        std::cout << "Allocated " << gpu_mem->size << " bytes of GPU memory at "
                  << gpu_mem->addr << std::endl;

        // Memory will be automatically freed when shared_ptr goes out of scope
        std::cout << "Memory will be automatically freed" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Example with RDMA support (requires valid ibv_pd)
    // struct ibv_pd* pd = ...; // Get protection domain from RDMA context
    // MemoryAllocator rdma_allocator(pd);
    // auto rdma_mem = rdma_allocator.allocate(1024 * 1024, MemoryType::HOST);
    // Memory will be registered with RDMA and can be used for RDMA operations

    return 0;
}
