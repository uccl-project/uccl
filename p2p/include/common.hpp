#ifndef COMMON_HPP
#define COMMON_HPP
#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(call)                                                         \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_e));                                    \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

struct TransferCmd {
    uint32_t dst_rank;      // remote node id (MPI‑style)
    uint32_t dst_gpu;       // gpu id on remote node
    void*    src_ptr;       // device pointer to data
    uint64_t bytes;         // transfer size
};

constexpr uint32_t QUEUE_SIZE = 1024;
constexpr uint32_t QUEUE_MASK = QUEUE_SIZE - 1;

#endif // COMMON_HPP