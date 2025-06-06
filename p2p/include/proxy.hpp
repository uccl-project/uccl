#ifndef PROXY_HPP
#define PROXY_HPP

#include "ring_buffer.cuh"
#include <thread>
#include <vector>
#include <chrono>

#ifdef NO_RDMA
#include <cuda_runtime.h>
void rdma_write_stub(int dst_rank, void* local_dev_ptr, size_t bytes);
#else
#include <infiniband/verbs.h>
// A minimal wrapper around an RDMA WRITE. Production code should create
// protection domain, queue pair, registered MR, CQ, etc. For brevity we show
// only a conceptual stub; fill with your own i;
#endif

struct ProxyCtx {
    RingBuffer* rb_host;   // host pointer (CPU visible)
    int          my_rank;
};

void proxy_loop(ProxyCtx ctx);

#endif