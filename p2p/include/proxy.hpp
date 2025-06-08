#ifndef PROXY_HPP
#define PROXY_HPP

#include "ring_buffer.cuh"
#include <chrono>
#include <thread>
#include <vector>
#include <immintrin.h>

#ifdef NO_RDMA
#include <cuda_runtime.h>
void rdma_write_stub(int dst_rank, void* local_dev_ptr, size_t bytes);
#else
#include <infiniband/verbs.h>
// A minimal wrapper around an RDMA WRITE. Production code should create
// protection domain, queue pair, registered MR, CQ, etc.
void rdma_write_stub(int dst_rank, void* local_dev_ptr, size_t bytes);
#endif

struct ProxyCtx {
  RingBuffer* rb_host;  // host pointer (CPU visible address of RingBuffer)
  int my_rank;          // rank id for this proxy (if simulating multiple)
};

void cpu_consume(RingBuffer* rb, int block_idx);

#endif  // PROXY_HPP