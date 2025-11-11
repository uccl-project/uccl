// memory_pool.h
// Memory pool for efficient memory allocation and reuse with mmap and GPU support

#pragma once
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif



// Memory block metadata
struct MemoryBlock {
    void* addr;
    size_t size;
    MemoryType type;
    bool in_use;

    MemoryBlock(void* a, size_t s, MemoryType t)
        : addr(a), size(s), type(t), in_use(false) {}
};

class MemoryPool {
public:
    // Page size for alignment (typically 4KB for RDMA)
    static constexpr size_t PAGE_SIZE = 4096;

    MemoryPool() = default;

    ~MemoryPool() {
        std::lock_guard<std::mutex> lock(mtx_);
        // Free all allocated blocks
        for (auto& block : blocks_) {
            free_memory(block->addr, block->size, block->type);
        }
        blocks_.clear();
        free_blocks_.clear();
    }

    // Allocate memory from pool or create new block
    // Size will be rounded up to page size for proper alignment
    void* allocate(size_t size, MemoryType type = MemoryType::HOST) {
        std::lock_guard<std::mutex> lock(mtx_);

        // Round up to page size for RDMA alignment
        size_t aligned_size = align_to_page(size);

        // Try to find a free block of sufficient size
        auto key = std::make_pair(aligned_size, type);
        auto it = free_blocks_.find(key);

        if (it != free_blocks_.end() && !it->second.empty()) {
            // Reuse existing block
            auto block = it->second.back();
            it->second.pop_back();
            block->in_use = true;
            std::cout << "MemoryPool: Reusing block at " << block->addr
                      << " size=" << aligned_size << " (requested=" << size << ")\n";
            return block->addr;
        }

        // Allocate new block
        void* addr = allocate_memory(aligned_size, type);
        if (!addr) {
            throw std::bad_alloc();
        }

        auto block = std::make_shared<MemoryBlock>(addr, aligned_size, type);
        block->in_use = true;
        blocks_.push_back(block);

        std::cout << "MemoryPool: Allocated new block at " << addr
                  << " size=" << aligned_size << " (requested=" << size << ")\n";
        return addr;
    }

    // Return memory to pool for reuse
    void deallocate(void* addr) {
        std::lock_guard<std::mutex> lock(mtx_);

        // Find the block
        for (auto& block : blocks_) {
            if (block->addr == addr) {
                if (!block->in_use) {
                    std::cerr << "Warning: Double free detected at " << addr << "\n";
                    return;
                }
                block->in_use = false;
                auto key = std::make_pair(block->size, block->type);
                free_blocks_[key].push_back(block);
                std::cout << "MemoryPool: Returned block at " << addr
                          << " to pool\n";
                return;
            }
        }

        std::cerr << "Warning: Attempt to deallocate unknown address " << addr << "\n";
    }

    // Get pool statistics
    size_t get_total_blocks() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return blocks_.size();
    }

    size_t get_free_blocks() const {
        std::lock_guard<std::mutex> lock(mtx_);
        size_t count = 0;
        for (const auto& kv : free_blocks_) {
            count += kv.second.size();
        }
        return count;
    }

    size_t get_used_blocks() const {
        return get_total_blocks() - get_free_blocks();
    }

private:
    // Align size to page boundary
    static size_t align_to_page(size_t size) {
        return ((size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
    }

    // Allocate memory using mmap (for HOST) or cudaMalloc (for GPU)
    static void* allocate_memory(size_t size, MemoryType type) {
        if (type == MemoryType::HOST) {
            // Use mmap for page-aligned memory suitable for RDMA
            void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (addr == MAP_FAILED) {
                perror("mmap failed");
                return nullptr;
            }
            return addr;
        } else {
            // GPU memory allocation
#ifdef HAVE_CUDA
            void* addr = nullptr;
            cudaError_t err = cudaMalloc(&addr, size);
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
                return nullptr;
            }
            return addr;
#else
            std::cerr << "GPU memory requested but CUDA not available\n";
            return nullptr;
#endif
        }
    }

    // Free memory using munmap (for HOST) or cudaFree (for GPU)
    static void free_memory(void* addr, size_t size, MemoryType type) {
        if (type == MemoryType::HOST) {
            if (munmap(addr, size) != 0) {
                perror("munmap failed");
            }
        } else {
            // GPU memory deallocation
#ifdef HAVE_CUDA
            cudaError_t err = cudaFree(addr);
            if (err != cudaSuccess) {
                std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << "\n";
            }
#else
            std::cerr << "Warning: Attempted to free GPU memory but CUDA not available\n";
#endif
        }
    }

    // Hash function for pair
    struct PairHash {
        template <typename T1, typename T2>
        size_t operator()(const std::pair<T1, T2>& p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };

    mutable std::mutex mtx_;
    std::vector<std::shared_ptr<MemoryBlock>> blocks_;
    // Map: (size, type) -> list of free blocks
    std::unordered_map<std::pair<size_t, MemoryType>,
                       std::vector<std::shared_ptr<MemoryBlock>>,
                       PairHash> free_blocks_;
};
