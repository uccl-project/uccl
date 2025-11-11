// rdma_memory_manager.h
// RDMA memory management with memory pool support

#pragma once
#include "memory_pool.h"
#include "define.h"
#include <infiniband/verbs.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <iostream>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <infiniband/verbs_exp.h>  // For GPU Direct RDMA if available
#endif



class RdmaMemoryManager {
public:
    // Constructor with optional memory pool
    // If pool is nullptr, will not use pooling
    explicit RdmaMemoryManager(struct ibv_pd* pd, std::shared_ptr<MemoryPool> pool = nullptr)
        : pd_(pd), pool_(pool) {
        if (!pd_) {
            throw std::invalid_argument("Protection domain cannot be null");
        }
    }

    ~RdmaMemoryManager() {
        std::lock_guard<std::mutex> lock(mtx_);

        // Deregister all memory regions
        for (auto& kv : registered_memory_) {
            auto& reg_mem = kv.second;

            // Deregister MR
            if (reg_mem->mr) {
                if (ibv_dereg_mr(reg_mem->mr) != 0) {
                    perror("ibv_dereg_mr failed");
                }
            }

            // Free memory if it was allocated by pool
            if (reg_mem->pool_allocated && pool_) {
                pool_->deallocate(reg_mem->addr);
            }
        }
        registered_memory_.clear();
    }

    // Function 1: Allocate memory of given size and type, then register it
    // Returns: pointer to allocated and registered memory
    void* allocate_and_register(size_t size, MemoryType type = MemoryType::HOST,
                                int access_flags = IBV_ACCESS_LOCAL_WRITE |
                                                  IBV_ACCESS_REMOTE_WRITE |
                                                  IBV_ACCESS_REMOTE_READ) {
        std::lock_guard<std::mutex> lock(mtx_);

        // Allocate memory from pool
        void* addr = nullptr;
        if (pool_) {
            addr = pool_->allocate(size, type);
        } else {
            // Fallback to direct allocation if no pool
            addr = allocate_memory_direct(size, type);
        }

        if (!addr) {
            std::cerr << "Failed to allocate memory of size " << size << "\n";
            return nullptr;
        }

        // Register memory with RDMA
        struct ibv_mr* mr = register_memory_internal(addr, size, type, access_flags);
        if (!mr) {
            std::cerr << "Failed to register memory at " << addr << "\n";
            if (pool_) {
                pool_->deallocate(addr);
            } else {
                free_memory_direct(addr, size, type);
            }
            return nullptr;
        }

        // Track registered memory
        auto reg_mem = std::make_shared<RegisteredMemory>(addr, size, type, mr, true);
        registered_memory_[addr] = reg_mem;

        std::cout << "RdmaMemoryManager: Allocated and registered " << size << " bytes at "
                  << addr << " (rkey=" << mr->rkey << ", lkey=" << mr->lkey << ")\n";
        return addr;
    }

    // Function 2: Register already allocated memory
    // Returns: MR pointer on success, nullptr on failure
    struct ibv_mr* register_memory(void* addr, size_t size, MemoryType type = MemoryType::HOST,
                                   int access_flags = IBV_ACCESS_LOCAL_WRITE |
                                                     IBV_ACCESS_REMOTE_WRITE |
                                                     IBV_ACCESS_REMOTE_READ) {
        std::lock_guard<std::mutex> lock(mtx_);

        // Check if already registered
        if (registered_memory_.find(addr) != registered_memory_.end()) {
            std::cerr << "Warning: Memory at " << addr << " already registered\n";
            return registered_memory_[addr]->mr;
        }

        // Register memory
        struct ibv_mr* mr = register_memory_internal(addr, size, type, access_flags);
        if (!mr) {
            std::cerr << "Failed to register memory at " << addr << "\n";
            return nullptr;
        }

        // Track registered memory (not pool-allocated)
        auto reg_mem = std::make_shared<RegisteredMemory>(addr, size, type, mr, false);
        registered_memory_[addr] = reg_mem;

        std::cout << "RdmaMemoryManager: Registered " << size << " bytes at "
                  << addr << " (rkey=" << mr->rkey << ", lkey=" << mr->lkey << ")\n";
        return mr;
    }

    // Deregister and optionally free memory
    bool deregister_memory(void* addr, bool free_memory = true) {
        std::lock_guard<std::mutex> lock(mtx_);

        auto it = registered_memory_.find(addr);
        if (it == registered_memory_.end()) {
            std::cerr << "Warning: Memory at " << addr << " not registered\n";
            return false;
        }

        auto& reg_mem = it->second;

        // Deregister MR
        if (reg_mem->mr) {
            if (ibv_dereg_mr(reg_mem->mr) != 0) {
                perror("ibv_dereg_mr failed");
                return false;
            }
        }

        // Free memory if requested and it was allocated by pool
        if (free_memory && reg_mem->pool_allocated && pool_) {
            pool_->deallocate(reg_mem->addr);
        }

        registered_memory_.erase(it);
        std::cout << "RdmaMemoryManager: Deregistered memory at " << addr << "\n";
        return true;
    }

    // Get MR for a registered address
    struct ibv_mr* get_mr(void* addr) const {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = registered_memory_.find(addr);
        if (it != registered_memory_.end()) {
            return it->second->mr;
        }
        return nullptr;
    }

    // Get statistics
    size_t get_registered_count() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return registered_memory_.size();
    }

private:
    struct ibv_pd* pd_;
    std::shared_ptr<MemoryPool> pool_;
    mutable std::mutex mtx_;
    std::unordered_map<void*, std::shared_ptr<RegisteredMemory>> registered_memory_;

    // Internal memory registration
    struct ibv_mr* register_memory_internal(void* addr, size_t size, MemoryType type, int access_flags) {
        if (type == MemoryType::HOST) {
            // Standard host memory registration
            struct ibv_mr* mr = ibv_reg_mr(pd_, addr, size, access_flags);
            if (!mr) {
                perror("ibv_reg_mr failed");
                return nullptr;
            }
            return mr;
        } else {
#ifdef HAVE_CUDA
            // GPU memory registration (GPU Direct RDMA)
            // Note: This requires special RDMA adapters and drivers that support GPU Direct
            struct ibv_mr* mr = ibv_reg_mr(pd_, addr, size, access_flags);
            if (!mr) {
                perror("ibv_reg_mr (GPU) failed");
                std::cerr << "Note: GPU Direct RDMA may not be supported on this hardware\n";
                return nullptr;
            }
            return mr;
#else
            std::cerr << "GPU memory registration requested but CUDA not available\n";
            return nullptr;
#endif
        }
    }

    // Direct allocation without pool (fallback)
    static void* allocate_memory_direct(size_t size, MemoryType type) {
        if (type == MemoryType::HOST) {
            // Align to page size
            size_t aligned_size = ((size + MemoryPool::PAGE_SIZE - 1) / MemoryPool::PAGE_SIZE) * MemoryPool::PAGE_SIZE;
            void* addr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (addr == MAP_FAILED) {
                perror("mmap failed");
                return nullptr;
            }
            return addr;
        } else {
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

    // Direct deallocation without pool (fallback)
    static void free_memory_direct(void* addr, size_t size, MemoryType type) {
        if (type == MemoryType::HOST) {
            size_t aligned_size = ((size + MemoryPool::PAGE_SIZE - 1) / MemoryPool::PAGE_SIZE) * MemoryPool::PAGE_SIZE;
            if (munmap(addr, aligned_size) != 0) {
                perror("munmap failed");
            }
        } else {
#ifdef HAVE_CUDA
            cudaError_t err = cudaFree(addr);
            if (err != cudaSuccess) {
                std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << "\n";
            }
#endif
        }
    }
};
